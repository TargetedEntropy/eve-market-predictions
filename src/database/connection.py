"""Database connection management.

Session convention: :func:`get_session` does **not** commit for you. Callers
own their transactions and must call ``await session.commit()`` explicitly;
the context manager only rolls back on exceptions and always closes.
"""

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any, Coroutine, List, TypeVar

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import settings
from src.database.models import Base

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Global engine instance
_engine = None
_async_session = None

# Dedicated background loop used by sync callers (the Dash dashboard)
_bg_loop: asyncio.AbstractEventLoop = None
_bg_thread: threading.Thread = None
_bg_lock = threading.Lock()


class TimescaleDBUnavailable(RuntimeError):
    """Raised when TimescaleDB is required but the extension is not usable."""


def get_engine():
    """Get or create the async database engine"""
    global _engine

    if _engine is None:
        logger.info(f"Creating database engine: {settings.database_url.split('@')[-1]}")

        _engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,   # Recycle connections every hour
        )

    return _engine


def get_session_maker():
    """Get or create the async session maker"""
    global _async_session

    if _async_session is None:
        engine = get_engine()
        _async_session = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    return _async_session


@asynccontextmanager
async def get_session():
    """
    Get an async database session.

    The caller is responsible for committing::

        async with get_session() as session:
            session.add(obj)
            await session.commit()
    """
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def run_async(coro: "Coroutine[Any, Any, T]") -> T:
    """Run a coroutine from synchronous code on one long-lived event loop.

    Dash callbacks are synchronous. Calling ``asyncio.run()`` per callback
    creates (and destroys) a fresh event loop each time while the module-global
    async engine keeps pooled asyncpg connections bound to the *first* loop,
    producing "attached to a different loop" / "Event loop is closed" errors.
    Routing every call through a single dedicated loop keeps the pool valid.
    """
    global _bg_loop, _bg_thread

    with _bg_lock:
        if _bg_loop is None or _bg_loop.is_closed():
            _bg_loop = asyncio.new_event_loop()
            _bg_thread = threading.Thread(
                target=_bg_loop.run_forever,
                name="db-event-loop",
                daemon=True,
            )
            _bg_thread.start()

    return asyncio.run_coroutine_threadsafe(coro, _bg_loop).result()


# --------------------------------------------------------------------------- #
# Schema initialization
# --------------------------------------------------------------------------- #

HYPERTABLES = [
    ("market_history", "time"),
    ("order_snapshots", "snapshot_time"),
    ("features", "time"),
    ("predictions", "prediction_time"),
    ("market_prices", "time"),
]

# Idempotent schema migrations for databases created by an earlier version.
# ``Base.metadata.create_all`` only creates missing *tables*, never columns or
# widened types, so these run on every init.
MIGRATIONS = [
    # 32-bit volume columns overflow on mineral orders (issue 3.4)
    "ALTER TABLE order_snapshots ALTER COLUMN volume_remain TYPE BIGINT",
    "ALTER TABLE order_snapshots ALTER COLUMN volume_total TYPE BIGINT",
    "ALTER TABLE order_snapshots ALTER COLUMN min_volume TYPE BIGINT",
    # Columns added alongside prediction persistence and item metadata refresh
    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS target_time TIMESTAMP",
    "ALTER TABLE items ADD COLUMN IF NOT EXISTS group_id INTEGER",
    "ALTER TABLE items ADD COLUMN IF NOT EXISTS group_name VARCHAR(100)",
    "ALTER TABLE items ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP",
]


async def _timescaledb_available(conn) -> bool:
    """Enable the TimescaleDB extension, returning whether it is usable."""
    try:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
        return True
    except Exception as e:  # extension not installed on this server
        logger.error(f"TimescaleDB extension unavailable: {e}")
        return False


async def _existing_hypertables(conn) -> List[str]:
    result = await conn.execute(
        text("SELECT hypertable_name FROM timescaledb_information.hypertables")
    )
    return [row[0] for row in result]


async def init_db(require_timescaledb: bool = True):
    """Create tables, hypertables, and retention/compression policies.

    Args:
        require_timescaledb: Raise :class:`TimescaleDBUnavailable` if the
            extension or any hypertable is missing. The docker-compose stack
            ships TimescaleDB, so silently degrading to plain Postgres (as the
            previous implementation did) hid a broken deployment.
    """
    logger.info("Initializing database tables...")

    engine = get_engine()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        for statement in MIGRATIONS:
            try:
                await conn.execute(text(statement))
            except Exception as e:
                logger.warning(f"Migration skipped ({statement}): {e}")

        has_timescale = await _timescaledb_available(conn)

    if not has_timescale:
        message = (
            "TimescaleDB is not available; hypertables, compression and retention "
            "policies were not created."
        )
        if require_timescaledb:
            raise TimescaleDBUnavailable(message)
        logger.warning(message)
        logger.info("Database initialization complete (plain PostgreSQL)")
        return

    # Hypertables need their own transaction each: a failed create_hypertable
    # aborts the surrounding transaction block in PostgreSQL.
    for table_name, time_column in HYPERTABLES:
        async with engine.begin() as conn:
            try:
                await conn.execute(
                    text(
                        "SELECT create_hypertable(:table, :column, if_not_exists => TRUE, "
                        "migrate_data => TRUE)"
                    ).bindparams(table=table_name, column=time_column)
                )
                logger.info(f"Hypertable ready: {table_name}")
            except Exception as e:
                logger.error(f"Could not create hypertable {table_name}: {e}")

    async with engine.begin() as conn:
        present = set(await _existing_hypertables(conn))

    missing = [name for name, _ in HYPERTABLES if name not in present]
    if missing:
        message = f"Expected hypertables are missing: {', '.join(missing)}"
        if require_timescaledb:
            raise TimescaleDBUnavailable(message)
        logger.warning(message)

    await _apply_policies(engine)

    logger.info("Database initialization complete")


async def _apply_policies(engine):
    """Compression and retention policies (issue 4.4)."""
    policies = [
        # market_history: compress anything older than a week
        (
            "ALTER TABLE market_history SET (timescaledb.compress, "
            "timescaledb.compress_segmentby = 'region_id, type_id')",
            "SELECT add_compression_policy('market_history', INTERVAL '7 days', "
            "if_not_exists => TRUE)",
            None,
        ),
        # order_snapshots is by far the highest-volume table: compress quickly
        # and drop raw chunks once they age out.
        (
            "ALTER TABLE order_snapshots SET (timescaledb.compress, "
            "timescaledb.compress_segmentby = 'region_id, type_id')",
            f"SELECT add_compression_policy('order_snapshots', "
            f"INTERVAL '{settings.order_snapshot_compress_after_days} days', "
            f"if_not_exists => TRUE)",
            f"SELECT add_retention_policy('order_snapshots', "
            f"INTERVAL '{settings.order_snapshot_retention_days} days', "
            f"if_not_exists => TRUE)",
        ),
    ]

    for statements in policies:
        for statement in statements:
            if statement is None:
                continue
            async with engine.begin() as conn:
                try:
                    await conn.execute(text(statement))
                except Exception as e:
                    logger.warning(f"Policy statement failed ({statement.split('(')[0]}): {e}")

    logger.info("Compression and retention policies applied")


async def close_db():
    """Close database connections"""
    global _engine, _async_session

    if _engine:
        await _engine.dispose()
        _engine = None
        _async_session = None
        logger.info("Database connections closed")
