"""Bulk upsert helpers (issues 6.1-6.3: one merge()/SELECT per row).

No live database: statements are captured and compiled against the PostgreSQL
dialect so the generated SQL is asserted directly.
"""

from datetime import datetime

import pytest
from sqlalchemy.dialects import postgresql

from src.database.models import MarketHistory, OrderSnapshot
from src.database.upsert import _dedupe_by_key, bulk_insert_ignore, bulk_upsert


class RecordingSession:
    """Captures executed statements instead of talking to a database."""

    def __init__(self):
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        return None

    async def commit(self):
        pass

    def compiled(self):
        return [
            str(s.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": False}))
            for s in self.statements
        ]


def history_row(day, type_id=44992, average=100.0):
    return {
        "time": datetime(2025, 1, day),
        "region_id": 10000002,
        "type_id": type_id,
        "average": average,
        "highest": average * 1.1,
        "lowest": average * 0.9,
        "volume": 1000,
        "order_count": 10,
    }


async def test_upsert_emits_on_conflict_do_update():
    session = RecordingSession()
    written = await bulk_upsert(session, MarketHistory, [history_row(1), history_row(2)])

    assert written == 2
    assert len(session.statements) == 1
    sql = session.compiled()[0]
    assert "INSERT INTO market_history" in sql
    assert "ON CONFLICT" in sql
    assert "DO UPDATE" in sql


async def test_upsert_conflict_target_is_the_primary_key():
    session = RecordingSession()
    await bulk_upsert(session, MarketHistory, [history_row(1)])
    sql = session.compiled()[0]
    for column in ("time", "region_id", "type_id"):
        assert column in sql.split("ON CONFLICT")[1].split("DO UPDATE")[0]


async def test_upsert_does_not_overwrite_primary_key_columns():
    session = RecordingSession()
    await bulk_upsert(session, MarketHistory, [history_row(1)])
    update_clause = session.compiled()[0].split("DO UPDATE SET")[1]
    assert "average" in update_clause
    assert "excluded" in update_clause.lower()


async def test_upsert_chunks_large_batches():
    session = RecordingSession()
    rows = [history_row(1, type_id=i) for i in range(2500)]
    written = await bulk_upsert(session, MarketHistory, rows, chunk_size=1000)

    assert written == 2500
    assert len(session.statements) == 3


async def test_duplicate_keys_in_one_batch_are_collapsed():
    """PostgreSQL rejects ON CONFLICT DO UPDATE touching a row twice."""
    session = RecordingSession()
    rows = [history_row(1, average=100.0), history_row(1, average=200.0)]
    written = await bulk_upsert(session, MarketHistory, rows)
    assert written == 1


def test_dedupe_keeps_the_last_occurrence():
    rows = [history_row(1, average=100.0), history_row(1, average=200.0)]
    deduped = _dedupe_by_key(rows, ["time", "region_id", "type_id"])
    assert len(deduped) == 1
    assert deduped[0]["average"] == 200.0


async def test_empty_input_issues_no_statements():
    session = RecordingSession()
    assert await bulk_upsert(session, MarketHistory, []) == 0
    assert session.statements == []


async def test_insert_ignore_uses_do_nothing():
    session = RecordingSession()
    await bulk_insert_ignore(session, MarketHistory, [history_row(1)])
    assert "DO NOTHING" in session.compiled()[0]


async def test_order_snapshot_upsert_is_retry_safe():
    """A retried collection run must update, not blow up on the PK."""
    session = RecordingSession()
    row = {
        "snapshot_time": datetime(2025, 1, 1, 12, 0),
        "order_id": 123456789012,
        "type_id": 34,
        "region_id": 10000002,
        "location_id": 60003760,
        "price": 5.5,
        "volume_remain": 3_000_000_000,  # > 2^31, issue 3.4
        "volume_total": 4_000_000_000,
        "is_buy_order": False,
        "duration": 90,
        "issued": datetime(2025, 1, 1),
        "min_volume": 1,
        "range": "region",
    }
    await bulk_upsert(session, OrderSnapshot, [row, dict(row, price=6.0)])
    sql = session.compiled()[0]
    assert "ON CONFLICT" in sql and "DO UPDATE" in sql


@pytest.mark.parametrize(
    "model,expected",
    [
        (MarketHistory, {"time", "region_id", "type_id"}),
        (OrderSnapshot, {"snapshot_time", "order_id"}),
    ],
)
def test_primary_keys_are_what_we_think(model, expected):
    assert {c.name for c in model.__table__.primary_key.columns} == expected


def test_volume_columns_are_64_bit():
    """Mineral orders routinely exceed 2^31 units (issue 3.4)."""
    for column in ("volume_remain", "volume_total", "min_volume"):
        assert OrderSnapshot.__table__.c[column].type.python_type is int
        compiled = OrderSnapshot.__table__.c[column].type.compile(postgresql.dialect())
        assert compiled == "BIGINT"
