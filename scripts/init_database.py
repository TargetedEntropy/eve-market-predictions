#!/usr/bin/env python3
"""Initialize database tables, TimescaleDB hypertables and policies."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import TimescaleDBUnavailable, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-missing-timescaledb",
        action="store_true",
        help=(
            "Continue on plain PostgreSQL instead of failing. Hypertables, "
            "compression and retention policies will NOT exist."
        ),
    )
    return parser.parse_args(argv)


async def main(argv=None):
    args = parse_args(argv)
    logger.info("Initializing database...")

    try:
        await init_db(require_timescaledb=not args.allow_missing_timescaledb)
    except TimescaleDBUnavailable as e:
        logger.error(f"{e}")
        logger.error(
            "The docker-compose stack ships TimescaleDB. If you really mean to run on "
            "plain PostgreSQL, re-run with --allow-missing-timescaledb."
        )
        return 1
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return 1

    logger.info("Database initialization complete!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
