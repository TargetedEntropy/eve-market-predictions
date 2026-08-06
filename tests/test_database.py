"""Database layer contracts (issues 1.1, 3.4, 6.4, 8.2)."""

import inspect

import pytest
from sqlalchemy import text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import ArgumentError, ObjectNotExecutableError

from src.database import connection
from src.database.models import Base, Item, ModelPerformance, Prediction

# --------------------------------------------------------------------------- #
# Raw SQL must be text() wrapped (1.1)
# --------------------------------------------------------------------------- #

def test_plain_strings_really_are_not_executable():
    """Why 1.1 mattered: SQLAlchemy 2.0 rejects bare strings.

    ``Connection.execute("...")`` raises ObjectNotExecutableError; the
    coercion layer underneath raises ArgumentError. Either way the statement
    never reaches the database -- and the old code swallowed it as a warning.
    """
    from sqlalchemy.sql import coercions, roles

    with pytest.raises((ObjectNotExecutableError, ArgumentError)):
        coercions.expect(roles.StatementRole, "CREATE EXTENSION IF NOT EXISTS timescaledb;")

    # text() makes the same string executable.
    assert coercions.expect(
        roles.StatementRole, text("CREATE EXTENSION IF NOT EXISTS timescaledb;")
    ) is not None


def test_init_db_never_executes_a_bare_string():
    source = inspect.getsource(connection)
    assert 'conn.execute("' not in source
    assert "conn.execute('" not in source
    assert "conn.execute(text(" in source


@pytest.mark.parametrize("statement", connection.MIGRATIONS)
def test_every_migration_statement_is_valid_sql(statement):
    compiled = text(statement).compile(dialect=postgresql.dialect())
    assert str(compiled)


def test_migrations_widen_the_volume_columns():
    """Issue 3.4 needs a migration, not just a model change."""
    joined = " ".join(connection.MIGRATIONS)
    for column in ("volume_remain", "volume_total", "min_volume"):
        assert f"ALTER COLUMN {column} TYPE BIGINT" in joined


def test_migrations_are_idempotent_in_shape():
    for statement in connection.MIGRATIONS:
        if "ADD COLUMN" in statement:
            assert "IF NOT EXISTS" in statement


def test_all_time_series_tables_are_registered_as_hypertables():
    names = {name for name, _ in connection.HYPERTABLES}
    assert names == {
        "market_history", "order_snapshots", "features", "predictions", "market_prices"
    }
    for table, time_column in connection.HYPERTABLES:
        assert time_column in Base.metadata.tables[table].columns


def test_missing_timescaledb_raises_by_default():
    signature = inspect.signature(connection.init_db)
    assert signature.parameters["require_timescaledb"].default is True


def test_retention_policy_is_configured_for_order_snapshots():
    """Issue 4.4: the highest-volume table grew unbounded."""
    source = inspect.getsource(connection._apply_policies)
    assert "add_retention_policy" in source
    assert "order_snapshots" in source
    assert "add_compression_policy" in source


# --------------------------------------------------------------------------- #
# Session convention (6.4)
# --------------------------------------------------------------------------- #

async def test_get_session_does_not_auto_commit(monkeypatch):
    """One convention: callers commit explicitly."""
    events = []

    class FakeSession:
        async def commit(self):
            events.append("commit")

        async def rollback(self):
            events.append("rollback")

        async def close(self):
            events.append("close")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(connection, "get_session_maker", lambda: FakeSession)

    async with connection.get_session() as session:
        assert session is not None

    assert events == ["close"], "the context manager must not commit for the caller"


async def test_get_session_rolls_back_on_error(monkeypatch):
    events = []

    class FakeSession:
        async def commit(self):
            events.append("commit")

        async def rollback(self):
            events.append("rollback")

        async def close(self):
            events.append("close")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(connection, "get_session_maker", lambda: FakeSession)

    with pytest.raises(ValueError):
        async with connection.get_session():
            raise ValueError("boom")

    assert events == ["rollback", "close"]


# --------------------------------------------------------------------------- #
# Sync bridge for the dashboard (1.7)
# --------------------------------------------------------------------------- #

def test_run_async_reuses_one_event_loop():
    """Issue 1.7: asyncio.run() per callback broke the pooled async engine."""
    import asyncio

    async def which_loop():
        return id(asyncio.get_running_loop())

    assert connection.run_async(which_loop()) == connection.run_async(which_loop())


def test_run_async_propagates_exceptions():
    async def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        connection.run_async(boom())


# --------------------------------------------------------------------------- #
# Models (8.2, 9.2)
# --------------------------------------------------------------------------- #

def test_declarative_base_comes_from_sqlalchemy_orm():
    source = inspect.getsource(__import__("src.database.models", fromlist=["x"]))
    assert "from sqlalchemy.orm import declarative_base" in source
    assert "sqlalchemy.ext.declarative" not in source


def test_model_performance_table_exists():
    """Documented in README/PRICE_PREDICTION_ENGINE but previously unmodelled."""
    assert ModelPerformance.__tablename__ in Base.metadata.tables


def test_prediction_has_a_target_time_for_scoring():
    assert "target_time" in Prediction.__table__.columns
    assert "actual_price" in Prediction.__table__.columns


def test_item_keeps_group_id_separately_from_category():
    assert "group_id" in Item.__table__.columns
    assert "group_name" in Item.__table__.columns
    assert "category" in Item.__table__.columns


def test_no_null_pool_import_left_behind():
    assert "NullPool" not in inspect.getsource(connection)
