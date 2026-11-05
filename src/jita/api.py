"""CPU snapshot API. No DB, training runtime or torch import in request paths."""
from __future__ import annotations

import hmac
import json
import math
import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from .schemas import Snapshot


@dataclass(frozen=True)
class Settings:
    artifact_dir: Path = Path("/artifacts")
    token: str = ""
    stale_seconds: int = 10800
    max_snapshot_bytes: int = 33554432
    requests_per_minute: int = 120
    max_clients: int = 1024

    def __post_init__(self):
        if not 60 <= self.stale_seconds <= 604800:
            raise ValueError("invalid stale threshold")
        if not 1024 <= self.max_snapshot_bytes <= 134217728:
            raise ValueError("invalid snapshot bound")
        if not 1 <= self.requests_per_minute <= 10000 or not 1 <= self.max_clients <= 10000:
            raise ValueError("invalid rate limit")

    @classmethod
    def from_env(cls):
        return cls(artifact_dir=Path(os.environ.get("JITA_ARTIFACT_DIR", "/artifacts")),
                   token=os.environ.get("JITA_API_TOKEN", ""),
                   stale_seconds=int(os.environ.get("JITA_STALE_SECONDS", "10800")),
                   max_snapshot_bytes=int(os.environ.get("JITA_MAX_SNAPSHOT_BYTES", "33554432")),
                   requests_per_minute=int(os.environ.get("JITA_REQUESTS_PER_MINUTE", "120")))


class SnapshotCache:
    """Inode/stat cache including failed versions; invalid replacement fails closed."""
    def __init__(self, settings):
        self.settings = settings
        self.path = settings.artifact_dir / "snapshot.json"
        self.lock = threading.Lock()
        self.signature = None
        self.snapshot = None

    def load(self):
        with self.lock:
            try:
                stat = self.path.stat()
                signature = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
                if signature == self.signature:
                    if self.snapshot is None:
                        raise ValueError()
                    return self.snapshot
                self.signature, self.snapshot = signature, None
                if stat.st_size > self.settings.max_snapshot_bytes:
                    raise ValueError()
                with self.path.open("rb") as source:
                    before = os.fstat(source.fileno())
                    raw = source.read(self.settings.max_snapshot_bytes + 1)
                    after = os.fstat(source.fileno())
                if len(raw) > self.settings.max_snapshot_bytes:
                    raise ValueError()
                if (before.st_ino, before.st_mtime_ns, before.st_size) != (
                    after.st_ino, after.st_mtime_ns, after.st_size
                ) or before.st_ino != stat.st_ino:
                    raise ValueError()
                data = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
                for key, value in {"schema_version": 1, "scope": "station", "station_id": 60003760,
                                   "system_id": 30000142, "region_id": 10000002,
                                   "price_basis": "jita_station_midpoint"}.items():
                    if type(data.get(key)) is not type(value) or data[key] != value:
                        raise ValueError()
                snapshot = Snapshot.model_validate_json(raw).model_dump(mode="json")
                for field in ("generated_at", "data_as_of"):
                    stamp = datetime.fromisoformat(snapshot[field].replace("Z", "+00:00"))
                    if stamp.tzinfo is None or stamp.timestamp() > time.time() + 300:
                        raise ValueError()
                self.snapshot = snapshot
                return snapshot
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError, OverflowError):
                self.snapshot = None
                raise HTTPException(503, "Prediction snapshot unavailable") from None


def create_app(settings: Settings | None = None):
    settings = settings or Settings.from_env()
    cache = SnapshotCache(settings)

    @asynccontextmanager
    async def lifespan(_app):
        if not 24 <= len(settings.token.encode()) <= 256:
            raise RuntimeError("JITA_API_TOKEN must be configured with 24 to 256 bytes")
        yield

    app = FastAPI(title="Jita snapshot API", version="1.0.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    registry = CollectorRegistry()
    requests = Counter("jita_http_requests_total", "Requests", ["route", "status"], registry=registry)
    duration = Histogram("jita_http_duration_seconds", "Request latency", ["route"], registry=registry)
    age = Gauge("jita_snapshot_age_seconds", "Underlying source age", registry=registry)
    stale = Gauge("jita_snapshot_stale", "Source data stale", registry=registry)
    ready = Gauge("jita_snapshot_ready", "Valid snapshot loaded", registry=registry)
    clients = OrderedDict()
    lock = threading.Lock()

    @app.middleware("http")
    async def guard(request: Request, call_next):
        started = time.monotonic()
        if request.url.path == "/v1/health":
            response = await call_next(request)
        elif not 24 <= len(settings.token.encode()) <= 256:
            response = JSONResponse({"detail": "Service not configured"}, status_code=503)
        elif len(request.headers.get("authorization", "")) > 512 or not hmac.compare_digest(
            request.headers.get("authorization", "").encode(), ("Bearer " + settings.token).encode()
        ):
            response = JSONResponse({"detail": "Unauthorized"}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
        else:
            client = request.client.host if request.client else "unknown"
            with lock:
                now = time.monotonic()
                tokens, previous = clients.pop(client, (float(settings.requests_per_minute), now))
                tokens = min(settings.requests_per_minute, tokens + (now - previous) * settings.requests_per_minute / 60)
                allowed = tokens >= 1
                clients[client] = (tokens - 1 if allowed else tokens, now)
                while len(clients) > settings.max_clients:
                    clients.popitem(last=False)
            response = await call_next(request) if allowed else JSONResponse(
                {"detail": "Rate limit exceeded"}, status_code=429, headers={"Retry-After": "1"})
        route = request.scope.get("route")
        label = route.path if route else "unmatched"
        requests.labels(label, str(response.status_code // 100) + "xx").inc()
        duration.labels(label).observe(time.monotonic() - started)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def load():
        try:
            snapshot = cache.load()
        except HTTPException:
            ready.set(0)
            raise
        ready.set(1)
        stamp = datetime.fromisoformat(snapshot["data_as_of"].replace("Z", "+00:00"))
        seconds = max(0.0, (datetime.now(timezone.utc) - stamp).total_seconds())
        age.set(seconds)
        stale.set(int(seconds > settings.stale_seconds))
        meta = {k: snapshot[k] for k in ("schema_version", "run_id", "generated_at", "data_as_of",
                                        "scope", "station_id", "system_id", "region_id", "price_basis")}
        meta.update(stale=seconds > settings.stale_seconds, age_seconds=round(seconds, 3))
        return snapshot, meta

    @app.get("/v1/health")
    def health():
        return {"data": {"status": "alive"}}

    @app.get("/v1/ready")
    def readiness():
        _, meta = load()
        if meta["stale"]:
            raise HTTPException(503, "Prediction snapshot stale")
        return {"data": {"status": "ready"}, "meta": meta}

    @app.get("/v1/overview")
    def overview():
        snapshot, meta = load()
        return {"data": {"tracked_types": len(snapshot["items"]),
                         "forecast_count": sum(len(i["forecasts"]) for i in snapshot["items"]),
                         "horizons_days": snapshot["horizons_days"], "quality": snapshot["quality"]}, "meta": meta}

    @app.get("/v1/opportunities")
    def opportunities(horizon_days: int = Query(7, ge=1, le=30), limit: int = Query(100, ge=1, le=500),
                      type_id: int | None = Query(None, ge=1, le=2147483647),
                      min_return_pct: float | None = Query(None, ge=-100, le=1000000)):
        if horizon_days not in (1, 7, 30) or (min_return_pct is not None and not math.isfinite(min_return_pct)):
            raise HTTPException(422, "Unsupported forecast filter")
        snapshot, meta = load()
        rows = []
        for item in snapshot["items"]:
            if type_id is not None and item["type_id"] != type_id:
                continue
            for forecast in item["forecasts"]:
                if forecast["horizon_days"] != horizon_days:
                    continue
                if min_return_pct is not None and (forecast["net_return_pct"] is None or forecast["net_return_pct"] < min_return_pct):
                    continue
                rows.append({**{k: v for k, v in item.items() if k != "forecasts"}, **forecast})
        rows.sort(key=lambda row: (row["net_return_pct"] is None, -(row["net_return_pct"] or 0), row["type_id"]))
        return {"data": rows[:limit], "meta": meta}

    @app.get("/v1/types/{type_id}")
    def item_detail(type_id: int):
        if not 1 <= type_id <= 2147483647:
            raise HTTPException(422, "Invalid type ID")
        snapshot, meta = load()
        for item in snapshot["items"]:
            if item["type_id"] == type_id:
                return {"data": item, "meta": meta}
        raise HTTPException(404, "Type not found")

    @app.get("/v1/models")
    def models():
        snapshot, meta = load()
        return {"data": snapshot["model"], "meta": meta}

    @app.get("/metrics")
    def metrics():
        try:
            load()
        except HTTPException:
            pass
        return Response(generate_latest(registry), media_type="text/plain; version=0.0.4")

    app.state.snapshot_cache = cache
    return app


app = create_app()
