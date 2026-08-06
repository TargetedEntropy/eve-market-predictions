#!/usr/bin/env python3
"""Import historical market data from EVE Ref's daily CSV dumps.

https://data.everef.net/market-history/ is a community-run mirror, so downloads
are paced, retried with backoff, and a missing file (e.g. a future date) is
skipped rather than treated as an error.

Region filtering is explicit: by default only ``COLLECTION_REGION_IDS`` is
imported. Pass ``--all-regions`` to load every region present in the dumps --
useful if you want the dashboard's Amarr/Dodixie options to have data, but it
is a lot more rows.

Examples::

    python scripts/import_everef_data.py --days 90
    python scripts/import_everef_data.py --start 2025-01-01 --end 2025-03-31
    python scripts/import_everef_data.py --days 30 --all-regions
    python scripts/import_everef_data.py --days 30 --all-items
"""

import argparse
import asyncio
import bz2
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import pandas as pd

from src.config import settings
from src.constants import TRACKED_TYPE_IDS
from src.database import MarketHistory, get_session
from src.database.upsert import bulk_upsert
from src.utils.timeutils import utcnow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://data.everef.net/market-history"

CSV_COLUMNS = ["date", "region_id", "type_id", "average", "highest", "lowest",
               "volume", "order_count"]


async def download_file(
    url: str,
    output_path: Path,
    retries: int = 3,
    delay: float = 1.0,
) -> bool:
    """Download a file, retrying transient failures. 404 means "not published"."""
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                response = await client.get(url)

                if response.status_code == 404:
                    logger.info(f"Not published yet: {url}")
                    return False

                response.raise_for_status()
                output_path.write_bytes(response.content)
                logger.info(f"Downloaded {url} -> {output_path.name}")
                return True

        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500 or attempt >= retries:
                logger.error(f"Failed to download {url}: {e}")
                return False
        except Exception as e:
            if attempt >= retries:
                logger.error(f"Failed to download {url}: {e}")
                return False

        backoff = delay * (2 ** attempt)
        logger.warning(f"Retrying {url} in {backoff:.1f}s")
        await asyncio.sleep(backoff)

    return False


def read_and_filter_csv(
    csv_path: Path,
    type_ids: Optional[List[int]],
    region_ids: Optional[List[int]],
) -> pd.DataFrame:
    """Read a daily dump, filtering before any objects are built.

    The dumps hold every region and every item; filtering during the read keeps
    memory bounded and avoids constructing rows that are about to be dropped.
    """
    frames = []
    for chunk in pd.read_csv(csv_path, chunksize=200_000):
        if type_ids is not None:
            chunk = chunk[chunk["type_id"].isin(type_ids)]
        if region_ids is not None:
            chunk = chunk[chunk["region_id"].isin(region_ids)]
        if not chunk.empty:
            frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=CSV_COLUMNS)

    return pd.concat(frames, ignore_index=True)


async def import_csv_file(
    csv_path: Path,
    type_ids: Optional[List[int]] = None,
    region_ids: Optional[List[int]] = None,
) -> int:
    """Bulk-upsert one daily CSV into ``market_history``."""
    logger.info(f"Importing {csv_path.name}...")

    df = read_and_filter_csv(csv_path, type_ids, region_ids)
    if df.empty:
        logger.info("  nothing to import after filtering")
        return 0

    df["time"] = pd.to_datetime(df["date"])

    rows = [
        {
            "time": record["time"].to_pydatetime(),
            "region_id": int(record["region_id"]),
            "type_id": int(record["type_id"]),
            "average": record["average"],
            "highest": record["highest"],
            "lowest": record["lowest"],
            "volume": int(record["volume"]),
            "order_count": int(record["order_count"]),
        }
        for record in df.to_dict("records")
    ]

    # One statement per 1000 rows, instead of a SELECT per row.
    async with get_session() as session:
        written = await bulk_upsert(session, MarketHistory, rows)
        await session.commit()

    logger.info(f"  upserted {written} rows")
    return written


async def import_date_range(
    start_date: date,
    end_date: date,
    type_ids: Optional[List[int]] = None,
    region_ids: Optional[List[int]] = None,
    download_dir: Optional[Path] = None,
    pause_seconds: float = 0.5,
    keep_files: bool = False,
) -> int:
    """Download and import every daily dump in ``[start_date, end_date]``."""
    download_dir = download_dir or Path("data/raw")
    download_dir.mkdir(parents=True, exist_ok=True)

    current = start_date
    total = 0

    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        bz2_path = download_dir / f"market-history-{date_str}.csv.bz2"
        csv_path = download_dir / f"market-history-{date_str}.csv"
        url = f"{BASE_URL}/{current.year}/{bz2_path.name}"

        if not bz2_path.exists() and not csv_path.exists():
            if not await download_file(url, bz2_path):
                current += timedelta(days=1)
                # Be polite to a community mirror even when skipping.
                await asyncio.sleep(pause_seconds)
                continue

        if not csv_path.exists() and bz2_path.exists():
            logger.info(f"Decompressing {bz2_path.name}...")
            with bz2.open(bz2_path, "rb") as source, open(csv_path, "wb") as target:
                for block in iter(lambda: source.read(1 << 20), b""):
                    target.write(block)

        if csv_path.exists():
            total += await import_csv_file(csv_path, type_ids, region_ids)
            if not keep_files:
                csv_path.unlink(missing_ok=True)
                bz2_path.unlink(missing_ok=True)

        current += timedelta(days=1)
        await asyncio.sleep(pause_seconds)

    logger.info(f"Total rows imported: {total}")
    return total


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    window = parser.add_mutually_exclusive_group()
    window.add_argument("--days", type=int, default=90,
                        help="Import the last N days (default: 90)")
    window.add_argument("--start", type=str, help="Start date YYYY-MM-DD")

    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--type-id", type=int, action="append", dest="type_ids",
                        help="Item to import (repeatable; default: tracked items)")
    parser.add_argument("--all-items", action="store_true",
                        help="Import every item (very large)")
    parser.add_argument("--region-id", type=int, action="append", dest="region_ids",
                        help="Region to import (repeatable; default: COLLECTION_REGION_IDS)")
    parser.add_argument("--all-regions", action="store_true",
                        help="Import every region present in the dumps")
    parser.add_argument("--download-dir", default="data/raw")
    parser.add_argument("--keep-files", action="store_true",
                        help="Keep downloaded .csv/.bz2 files instead of deleting them")
    parser.add_argument("--pause", type=float, default=0.5,
                        help="Seconds to wait between daily files")
    return parser.parse_args(argv)


async def main(argv=None):
    args = parse_args(argv)

    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end \
        else utcnow().date()
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start_date = end_date - timedelta(days=args.days)

    if start_date > end_date:
        logger.error("--start must be on or before --end")
        return 1

    type_ids = None if args.all_items else (args.type_ids or TRACKED_TYPE_IDS)
    region_ids = None if args.all_regions else (args.region_ids or settings.region_ids)

    logger.info("EVE Ref Market Data Importer")
    logger.info("=" * 50)
    logger.info(f"Date range: {start_date} .. {end_date}")
    logger.info(f"Items:      {'ALL' if type_ids is None else len(type_ids)}")
    logger.info(f"Regions:    {'ALL' if region_ids is None else region_ids}")

    await import_date_range(
        start_date,
        end_date,
        type_ids=type_ids,
        region_ids=region_ids,
        download_dir=Path(args.download_dir),
        pause_seconds=args.pause,
        keep_files=args.keep_files,
    )

    logger.info("Import complete!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
