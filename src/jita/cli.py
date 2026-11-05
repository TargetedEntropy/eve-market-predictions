"""Nomad batch entrypoint: python -m src.jita.cli batch --require-cuda."""
import argparse
import asyncio
import json
import os
from pathlib import Path

from .engine import EngineConfig, ForecastEngine
from .source import SourceClient
from .storage import SnapshotStore


async def run_batch(args):
    # GPU requirement checked before network requests, even on short histories.
    engine = ForecastEngine(EngineConfig(entry_fee_pct=args.entry_fee_pct,
        exit_fee_pct=args.exit_fee_pct,slippage_pct=args.slippage_pct),
        require_cuda=args.require_cuda)
    url = os.environ.get("JITA_SOURCE_URL", "")
    token = os.environ.get("JITA_SOURCE_TOKEN", "")
    if not url or not token:
        raise ValueError("JITA_SOURCE_URL and JITA_SOURCE_TOKEN are required")
    store = SnapshotStore(Path(args.output_dir))
    async with SourceClient(url, token, concurrency=args.concurrency) as source:
        rows = await source.fetch_all(limit=args.limit, days=365)
        skipped = sorted(source.skipped_type_ids)
    if not rows:
        raise ValueError("empty station universe; previous snapshot retained")
    merged = [store.merge_history(row) for row in rows]
    snapshot = engine.snapshot(merged)
    snapshot.quality["skipped_no_station_book_type_ids"] = skipped
    if all(not item.data_quality["fresh_book"] for item in snapshot.items):
        raise ValueError("all station books stale; previous snapshot retained")
    artifact = store.publish(snapshot)
    return {"run_id":snapshot.run_id,"items":len(snapshot.items),"artifact":str(artifact),
            "device":engine.device,"quality":snapshot.quality}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Jita station-only chronological batch forecasts")
    sub = parser.add_subparsers(dest="command", required=True)
    batch = sub.add_parser("batch", help="fetch, accumulate, train, validate, forecast and publish atomically")
    batch.add_argument("--output-dir", default="artifacts")
    batch.add_argument("--require-cuda", action="store_true")
    batch.add_argument("--limit", type=int, default=200)
    batch.add_argument("--concurrency", type=int, default=4)
    batch.add_argument("--entry-fee-pct", type=float, default=0.0)
    batch.add_argument("--exit-fee-pct", type=float, default=3.6)
    batch.add_argument("--slippage-pct", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(run_batch(args))
    except Exception as exc:
        # Do not expose remote bodies, credentials or environment via exceptions.
        parser.exit(1, f"Jita batch failed ({type(exc).__name__}); previous snapshot retained\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
