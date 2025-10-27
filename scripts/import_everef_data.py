#!/usr/bin/env python3
"""Import market history from EVE Ref CSV files"""

import asyncio
import sys
import logging
import bz2
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import httpx
from sqlalchemy import select

from src.config import settings
from src.database import get_session, MarketHistory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://data.everef.net/market-history"


async def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            output_path.write_bytes(response.content)
            logger.info(f"Downloaded {url} -> {output_path}")
            return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


async def import_csv_file(csv_path: Path, target_type_ids: List[int] = None):
    """Import market history from CSV file"""
    logger.info(f"Importing {csv_path}...")

    # Read CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} records from CSV")

    # Filter by type_ids if specified
    if target_type_ids:
        df = df[df['type_id'].isin(target_type_ids)]
        logger.info(f"Filtered to {len(df)} records for target items")

    if df.empty:
        logger.warning("No records to import after filtering")
        return 0

    # Convert date strings to datetime
    df['time'] = pd.to_datetime(df['date'])

    # Import to database
    async with get_session() as session:
        records_added = 0
        records_updated = 0

        for _, row in df.iterrows():
            # Check if record exists
            result = await session.execute(
                select(MarketHistory).where(
                    MarketHistory.time == row['time'],
                    MarketHistory.region_id == row['region_id'],
                    MarketHistory.type_id == row['type_id']
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing record
                existing.average = row['average']
                existing.highest = row['highest']
                existing.lowest = row['lowest']
                existing.volume = row['volume']
                existing.order_count = row['order_count']
                records_updated += 1
            else:
                # Create new record
                record = MarketHistory(
                    time=row['time'],
                    region_id=row['region_id'],
                    type_id=row['type_id'],
                    average=row['average'],
                    highest=row['highest'],
                    lowest=row['lowest'],
                    volume=row['volume'],
                    order_count=row['order_count']
                )
                session.add(record)
                records_added += 1

            # Commit in batches
            if (records_added + records_updated) % 1000 == 0:
                await session.commit()
                logger.info(f"Progress: {records_added} added, {records_updated} updated")

        await session.commit()

    logger.info(f"Import complete: {records_added} added, {records_updated} updated")
    return records_added + records_updated


async def import_date_range(
    start_date: datetime,
    end_date: datetime,
    target_type_ids: List[int] = None,
    download_dir: Path = None
):
    """Download and import market history for a date range"""
    if download_dir is None:
        download_dir = Path("data/raw")

    download_dir.mkdir(parents=True, exist_ok=True)

    current_date = start_date
    total_records = 0

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        year = current_date.year

        # Construct URLs
        bz2_filename = f"market-history-{date_str}.csv.bz2"
        csv_filename = f"market-history-{date_str}.csv"

        bz2_path = download_dir / bz2_filename
        csv_path = download_dir / csv_filename

        url = f"{BASE_URL}/{year}/{bz2_filename}"

        # Download if not exists
        if not bz2_path.exists() and not csv_path.exists():
            logger.info(f"Downloading {date_str}...")
            success = await download_file(url, bz2_path)
            if not success:
                logger.warning(f"Skipping {date_str}")
                current_date += timedelta(days=1)
                continue

        # Decompress if needed
        if not csv_path.exists() and bz2_path.exists():
            logger.info(f"Decompressing {bz2_filename}...")
            with bz2.open(bz2_path, 'rb') as f_in:
                with open(csv_path, 'wb') as f_out:
                    f_out.write(f_in.read())

        # Import
        if csv_path.exists():
            count = await import_csv_file(csv_path, target_type_ids)
            total_records += count

        current_date += timedelta(days=1)

    logger.info(f"Total records imported: {total_records}")
    return total_records


async def main():
    """Main import function"""
    logger.info("EVE Ref Market Data Importer")
    logger.info("=" * 50)

    # Target items (same as our tracked items)
    target_type_ids = [
        44992,  # PLEX
        40519,  # Skill Extractor
        40520,  # Large Skill Injector
        34,     # Tritanium
        35,     # Pyerite
        36,     # Mexallon
        37,     # Isogen
        38,     # Nocxium
        39,     # Zydrine
        40,     # Megacyte
        11399,  # Morphite
    ]

    # Import last 90 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    logger.info(f"Importing data from {start_date.date()} to {end_date.date()}")
    logger.info(f"Target items: {len(target_type_ids)}")

    await import_date_range(start_date, end_date, target_type_ids)

    logger.info("Import complete!")


if __name__ == "__main__":
    asyncio.run(main())
