"""Durable, station-scoped JSON artifacts and daily history archives."""
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .schemas import Snapshot, TypeSource

SNAPSHOT_LIMIT = 32 * 1024 * 1024
HISTORY_LIMIT = 8 * 1024 * 1024
RUN_RETENTION_COUNT = 14
RUN_RETENTION_BYTES = 128 * 1024 * 1024
HISTORY_RETENTION_COUNT = 400
HISTORY_RETENTION_BYTES = 128 * 1024 * 1024


def _read(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{path.name} exceeds {limit} bytes")
    return data


def _encode(model, schema, limit: int) -> bytes:
    # Revalidate even model_construct/model_copy or a mutated nested model.
    data = json.dumps(model.model_dump(mode="json"), allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(data) > limit:
        raise ValueError(f"artifact exceeds {limit} bytes")
    schema.model_validate_json(data)
    return data


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        _sync_dir(path.parent)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


class SnapshotStore:
    def __init__(self, directory):
        self.directory = Path(directory)

    def _mkdir(self, path: Path) -> None:
        # Persist directory creation as well as files within it.
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            _sync_dir(path)
            _sync_dir(path.parent)

    @contextmanager
    def _lock(self):
        self._mkdir(self.directory)
        with (self.directory / ".lock").open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def load(self) -> Snapshot | None:
        try:
            data = _read(self.directory / "snapshot.json", SNAPSHOT_LIMIT)
        except FileNotFoundError:
            return None
        return Snapshot.model_validate_json(data)

    @staticmethod
    def _prune(directory: Path, protected: set[str], count: int, byte_limit: int):
        files = sorted(directory.glob("*.json"), key=lambda p:p.stat().st_mtime_ns, reverse=True)
        retained = [p for p in files if p.name in protected]
        used = sum(p.stat().st_size for p in retained)
        removed = False
        for path in files:
            if path.name in protected:
                continue
            size = path.stat().st_size
            if len(retained) < count and used+size <= byte_limit:
                retained.append(path)
                used += size
            else:
                path.unlink()
                removed = True
        if removed:
            _sync_dir(directory)

    def publish(self, snapshot: Snapshot) -> Path:
        data = _encode(snapshot, Snapshot, SNAPSHOT_LIMIT)
        validated = Snapshot.model_validate_json(data)
        with self._lock():
            current = self.load()
            if current is not None and validated.generated_at < current.generated_at:
                raise ValueError("cannot replace a newer snapshot")
            runs = self.directory / "runs"
            self._mkdir(runs)
            run_path = runs / f"{validated.run_id}.json"
            # Exclusive creation preserves immutable run IDs, including failed publishes.
            fd = os.open(run_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                _sync_dir(runs)
            except BaseException:
                run_path.unlink(missing_ok=True)
                _sync_dir(runs)
                raise
            destination = self.directory / "snapshot.json"
            # Prune BEFORE the commit point so an I/O failure cannot be reported
            # as a failed publish after already replacing the active snapshot.
            protected_runs = {run_path.name}
            if current is not None:
                protected_runs.add(f"{current.run_id}.json")
            self._prune(runs, protected_runs, RUN_RETENTION_COUNT, RUN_RETENTION_BYTES)
            history_dir = self.directory / "history"
            if history_dir.exists():
                protected_types = {f"{item.type_id}.json" for item in validated.items}
                self._prune(history_dir, protected_types, HISTORY_RETENTION_COUNT, HISTORY_RETENTION_BYTES)
            _atomic_write(destination, data)
            return destination

    def merge_history(self, source: TypeSource) -> TypeSource:
        data = _encode(source, TypeSource, HISTORY_LIMIT)
        incoming = TypeSource.model_validate_json(data)
        if incoming.book.observed_at > datetime.now(timezone.utc):
            raise ValueError("future book cannot enter archive")
        if any(row.date > incoming.book.observed_at.date() for row in incoming.history):
            raise ValueError("history after book observation cannot enter archive")
        if len({row.date for row in incoming.history}) != len(incoming.history):
            raise ValueError("duplicate incoming history dates")
        with self._lock():
            history_dir = self.directory / "history"
            self._mkdir(history_dir)
            path = history_dir / f"{incoming.type_id}.json"
            try:
                existing = TypeSource.model_validate_json(_read(path, HISTORY_LIMIT))
            except FileNotFoundError:
                existing = None
            rows = {}
            if existing is not None:
                if existing.type_id != incoming.type_id:
                    raise ValueError("archive type does not match")
                for row in existing.history:
                    if row.date in rows:
                        raise ValueError("duplicate archived history dates")
                    rows[row.date] = row
            rows.update({row.date: row for row in incoming.history})
            merged = incoming.model_copy(update={"history": [rows[day] for day in sorted(rows)[-730:]]})
            encoded = _encode(merged, TypeSource, HISTORY_LIMIT)
            _atomic_write(path, encoded)
            # Bound churn even when later forecasting fails, retaining the active
            # snapshot's inputs and the item just archived.
            active = self.load()
            protected = {path.name}
            if active is not None:
                protected.update(f"{item.type_id}.json" for item in active.items)
            self._prune(history_dir, protected, HISTORY_RETENTION_COUNT, HISTORY_RETENTION_BYTES)
            return TypeSource.model_validate_json(encoded)
