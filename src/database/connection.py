"""Database connection management"""

import logging
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

from src.config import settings
from src.database.models import Base

logger = logging.getLogger(__name__)

# Global engine instance
_engine = None
_async_session = None


def get_engine():
    """Get or create the async database engine"""
    global _engine

    if _engine is None:
        logger.info(f"Creating database engine: {settings.database_url.split('@')[1]}")

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
    Get an async database session

    Usage:
        async with get_session() as session:
            result = await session.execute(query)
    """
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize database tables"""
    logger.info("Initializing database tables...")

    engine = get_engine()

    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

        # Enable TimescaleDB extension (if not already enabled)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
            logger.info("TimescaleDB extension enabled")
        except Exception as e:
            logger.warning(f"Could not enable TimescaleDB: {e}")

        # Create hypertables for time-series data
        hypertables = [
            ("market_history", "time"),
            ("order_snapshots", "snapshot_time"),
            ("features", "time"),
            ("predictions", "prediction_time"),
        ]

        for table_name, time_column in hypertables:
            try:
                await conn.execute(
                    f"SELECT create_hypertable('{table_name}', '{time_column}', "
                    f"if_not_exists => TRUE);"
                )
                logger.info(f"Created hypertable: {table_name}")
            except Exception as e:
                logger.warning(f"Could not create hypertable {table_name}: {e}")

        # Enable compression for market_history (data older than 7 days)
        try:
            await conn.execute(
                "ALTER TABLE market_history SET (timescaledb.compress, "
                "timescaledb.compress_segmentby = 'region_id, type_id');"
            )
            await conn.execute(
                "SELECT add_compression_policy('market_history', INTERVAL '7 days');"
            )
            logger.info("Compression enabled for market_history")
        except Exception as e:
            logger.warning(f"Could not enable compression: {e}")

    logger.info("Database initialization complete")


async def close_db():
    """Close database connections"""
    global _engine, _async_session

    if _engine:
        await _engine.dispose()
        _engine = None
        _async_session = None
        logger.info("Database connections closed")
