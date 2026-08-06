"""Prediction scoring (issue 2.4: nothing ever filled actual_price)."""

from contextlib import asynccontextmanager
from datetime import datetime

import pytest
from sqlalchemy.dialects import postgresql

from src.monitoring import evaluation


class RecordingSession:
    def __init__(self, rows=None, rowcount=0):
        self.statements = []
        self._rows = rows or []
        self._rowcount = rowcount

    async def execute(self, statement):
        self.statements.append(statement)
        return _Result(self._rows, self._rowcount)

    async def commit(self):
        pass


class _Result:
    def __init__(self, rows, rowcount):
        self._rows = rows
        self.rowcount = rowcount

    def all(self):
        return self._rows


@pytest.fixture
def session_factory(monkeypatch):
    holder = {}

    def install(session):
        holder["session"] = session

        @asynccontextmanager
        async def fake_session():
            yield session

        monkeypatch.setattr(evaluation, "get_session", fake_session)
        return session

    return install


def compile_sql(statement):
    return str(statement.compile(dialect=postgresql.dialect()))


async def test_backfill_updates_predictions_from_market_history(session_factory):
    session = session_factory(RecordingSession(rowcount=7))

    scored = await evaluation.backfill_actual_prices()

    assert scored == 7
    sql = compile_sql(session.statements[0])
    assert sql.startswith("UPDATE predictions SET actual_price")
    assert "market_history" in sql
    assert "EXISTS" in sql, "must not null out rows that have no realized price yet"


async def test_backfill_only_touches_unscored_elapsed_predictions(session_factory):
    session = session_factory(RecordingSession(rowcount=0))
    await evaluation.backfill_actual_prices()

    sql = compile_sql(session.statements[0])
    assert "actual_price IS NULL" in sql
    assert "target_time <=" in sql


async def test_performance_is_grouped_by_model_and_horizon(session_factory, monkeypatch):
    rows = [
        # type_id, region_id, version, horizon, predicted, actual
        (44992, 10000002, "v1", 1, 100.0, 110.0),
        (44992, 10000002, "v1", 1, 200.0, 180.0),
        (44992, 10000002, "v1", 7, 100.0, 150.0),
        (34, 10000002, "v2", 1, 10.0, 10.0),
    ]
    session_factory(RecordingSession(rows=rows))

    written = []

    async def fake_upsert(session, model, records, **kwargs):
        written.extend(records)
        return len(records)

    monkeypatch.setattr(evaluation, "bulk_upsert", fake_upsert)

    summaries = await evaluation.compute_model_performance()

    assert len(summaries) == 3
    keys = {(r["type_id"], r["model_version"], r["horizon_days"]) for r in written}
    assert keys == {(44992, "v1", 1), (44992, "v1", 7), (34, "v2", 1)}

    plex_1d = next(r for r in written if r["type_id"] == 44992 and r["horizon_days"] == 1)
    assert plex_1d["n_samples"] == 2
    assert plex_1d["mae"] == pytest.approx(15.0)
    assert plex_1d["mape"] == pytest.approx(((10 / 110) + (20 / 180)) / 2 * 100)

    perfect = next(r for r in written if r["type_id"] == 34)
    assert perfect["mae"] == 0.0


async def test_performance_with_nothing_scored_yet(session_factory):
    session_factory(RecordingSession(rows=[]))
    assert await evaluation.compute_model_performance() == []


async def test_recent_prediction_error_returns_mape(session_factory):
    session_factory(RecordingSession(rows=[(100.0, 110.0), (200.0, 180.0)]))

    mape = await evaluation.recent_prediction_error(44992, 10000002)

    assert mape == pytest.approx(((10 / 110) + (20 / 180)) / 2 * 100)


async def test_recent_prediction_error_is_none_without_data(session_factory):
    session_factory(RecordingSession(rows=[]))
    assert await evaluation.recent_prediction_error(44992, 10000002) is None


def test_nan_is_stored_as_null():
    assert evaluation._nan_to_none(float("nan")) is None
    assert evaluation._nan_to_none(1.5) == 1.5
    assert evaluation._nan_to_none(None) is None


async def test_model_version_is_truncated_to_the_column_width(session_factory, monkeypatch):
    long_version = "x" * 80
    session_factory(RecordingSession(rows=[(1, 2, long_version, 1, 1.0, 1.0)]))

    written = []

    async def fake_upsert(session, model, records, **kwargs):
        written.extend(records)
        return len(records)

    monkeypatch.setattr(evaluation, "bulk_upsert", fake_upsert)
    await evaluation.compute_model_performance()

    assert len(written[0]["model_version"]) == 50


async def test_null_model_version_becomes_unknown(session_factory, monkeypatch):
    session_factory(RecordingSession(rows=[(1, 2, None, 1, 1.0, 1.0)]))

    written = []

    async def fake_upsert(session, model, records, **kwargs):
        written.extend(records)
        return len(records)

    monkeypatch.setattr(evaluation, "bulk_upsert", fake_upsert)
    await evaluation.compute_model_performance()

    assert written[0]["model_version"] == "unknown"


def test_evaluation_uses_naive_utc():
    """Windows must be UTC, not server-local (issue 3.1)."""
    import inspect

    source = inspect.getsource(evaluation)
    assert "datetime.now()" not in source
    assert "utcnow()" in source


def test_prediction_rows_carry_a_target_time():
    from src.database import Prediction

    assert isinstance(
        Prediction.__table__.c.target_time.type, type(Prediction.__table__.c.prediction_time.type)
    )
    assert datetime is not None  # imported for the type reference above
