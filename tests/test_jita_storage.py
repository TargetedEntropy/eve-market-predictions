from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.jita.schemas import Book, HistoryRow, ItemForecast, Signals, Snapshot, TypeSource
from src.jita.storage import SnapshotStore


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def snapshot(run_id="one", when=NOW):
    book = Book(observed_at=when, best_bid=1, best_ask=2)
    item = ItemForecast(type_id=34, item_name="Tritanium", current_price=1.5, book=book,
                        history=[], data_quality={}, forecasts=[],
                        signals=Signals(fee_entry_pct=0, fee_exit_pct=0, slippage_pct=0))
    return Snapshot(run_id=run_id, generated_at=when, data_as_of=when,
                    model={}, quality={}, items=[item])


def source(rows):
    return TypeSource(type_id=34, item_name="Tritanium", station_id=60003760,
                      region_id=10000002, book=Book(observed_at=NOW), history=rows)


def row(day, price=1):
    return HistoryRow(date=day, mid_price=price, snapshot_count=1)


def test_snapshot_round_trip_immutable_and_monotonic(tmp_path):
    store = SnapshotStore(tmp_path)
    assert store.load() is None
    original = snapshot()
    store.publish(original)
    assert store.load() == original
    assert (tmp_path / "runs/one.json").exists()
    with pytest.raises(FileExistsError):
        store.publish(original)
    with pytest.raises(ValueError, match="newer"):
        store.publish(snapshot("older", NOW - timedelta(days=1)))
    assert store.load() == original


def test_replace_failure_preserves_snapshot(tmp_path):
    store = SnapshotStore(tmp_path)
    store.publish(snapshot())
    before = (tmp_path / "snapshot.json").read_bytes()
    with patch("src.jita.storage.os.replace", side_effect=OSError("failure")):
        with pytest.raises(OSError):
            store.publish(snapshot("two", NOW + timedelta(seconds=1)))
    assert (tmp_path / "snapshot.json").read_bytes() == before
    assert store.load() == snapshot()
    assert (tmp_path / "runs/two.json").exists()


def test_invalid_snapshot_and_mixed_station_rejected(tmp_path):
    store = SnapshotStore(tmp_path)
    store.publish(snapshot())
    invalid = snapshot("bad").model_copy(update={"station_id": 123})
    with pytest.raises(ValidationError):
        store.publish(invalid)
    with pytest.raises(ValidationError):
        store.merge_history(source([]).model_copy(update={"station_id": 123}))
    assert store.load() == snapshot()
    (tmp_path / "snapshot.json").write_text('{"garbage":true}')
    with pytest.raises(ValidationError):
        store.load()


def test_history_merge_overwrite_limit_duplicates_and_corrupt_archive(tmp_path):
    store = SnapshotStore(tmp_path)
    start = date(2024, 1, 1)
    store.merge_history(source([row(start + timedelta(days=i)) for i in range(730)]))
    result = store.merge_history(source([row(start + timedelta(days=5), 3), row(start + timedelta(days=730), 2)]))
    assert len(result.history) == 730
    assert result.history[0].date == start + timedelta(days=1)
    assert next(r for r in result.history if r.date == start + timedelta(days=5)).mid_price == 3
    path = tmp_path / "history/34.json"
    previous = path.read_bytes()
    with pytest.raises(ValueError, match="duplicate"):
        store.merge_history(source([row(start), row(start)]))
    assert path.read_bytes() == previous
    path.write_text('{"station_id": 99}')
    corrupt = path.read_bytes()
    with pytest.raises(ValidationError):
        store.merge_history(source([row(start)]))
    assert path.read_bytes() == corrupt


def test_concurrent_history_updates_are_not_lost(tmp_path):
    start = date(2025, 1, 1)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: SnapshotStore(tmp_path).merge_history(source([row(start + timedelta(days=i))])), range(30)))
    merged = SnapshotStore(tmp_path).merge_history(source([]))
    assert [r.date for r in merged.history] == [start + timedelta(days=i) for i in range(30)]


def test_archive_replace_failure_retains_prior_data(tmp_path):
    store = SnapshotStore(tmp_path)
    store.merge_history(source([row(date(2025, 1, 1))]))
    path = tmp_path / "history/34.json"
    before = path.read_bytes()
    with patch("src.jita.storage.os.replace", side_effect=OSError("failure")):
        with pytest.raises(OSError):
            store.merge_history(source([row(date(2025, 1, 2))]))
    assert path.read_bytes() == before


def test_concurrent_snapshots_never_replace_newer_with_older(tmp_path):
    def publish(i):
        try:
            SnapshotStore(tmp_path).publish(snapshot(str(i), NOW + timedelta(seconds=i)))
        except ValueError as exc:
            assert "newer" in str(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(publish, reversed(range(20))))
    assert SnapshotStore(tmp_path).load().run_id == "19"


def test_run_and_churned_history_retention_is_bounded(tmp_path,monkeypatch):
    from src.jita import storage
    monkeypatch.setattr(storage,"RUN_RETENTION_COUNT",3)
    monkeypatch.setattr(storage,"HISTORY_RETENTION_COUNT",3)
    store = SnapshotStore(tmp_path)
    for i in range(8):
        store.publish(snapshot(str(i),NOW+timedelta(seconds=i)))
    assert len(list((tmp_path/"runs").glob("*.json"))) <= 3
    assert (tmp_path/"runs/7.json").exists()
    store.merge_history(source([]))
    for i in range(8):
        store.merge_history(source([]).model_copy(update={"type_id":100+i}))
    assert len(list((tmp_path/"history").glob("*.json"))) <= 3
    assert (tmp_path/"history/34.json").exists()
    assert (tmp_path/"history/107.json").exists()
