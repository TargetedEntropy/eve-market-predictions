"""Bounded, station-scoped access to the authenticated prediction source API."""

from __future__ import annotations

import asyncio
import json
import math
from urllib.parse import urlsplit

import httpx

from .schemas import REGION_ID, STATION_ID, SYSTEM_ID, TypeSource, UniverseItem


class SourceError(ValueError):
    """The source request or its validation failed (never contains response content)."""


class SourceUnavailable(SourceError):
    """An explicitly absent/two-sided-unavailable book, not malformed history."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON number")


def _bounded_int(value: int, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _provenance(obj: object, *, require_scope: bool = False, require_ids: bool = False) -> None:
    if not isinstance(obj, dict):
        raise SourceError("Invalid source provenance")
    if require_scope and obj.get("scope") != "station":
        raise SourceError("Source is not station scoped")
    if "scope" in obj and obj["scope"] != "station":
        raise SourceError("Source is not station scoped")
    for key, expected in (("station_id", STATION_ID), ("region_id", REGION_ID),
                          ("system_id", SYSTEM_ID)):
        if key in obj and (type(obj[key]) is not int or obj[key] != expected):
            raise SourceError("Source provenance does not match Jita")
    if require_ids and (obj.get("station_id") != STATION_ID or obj.get("region_id") != REGION_ID):
        raise SourceError("Source station or region missing or invalid")


class SourceClient:
    def __init__(self, base_url: str, token: str, timeout_seconds: float = 20,
                 max_payload_bytes: int = 8_000_000, concurrency: int = 4,
                 transport: httpx.AsyncBaseTransport | None = None):
        try:
            url = urlsplit(base_url)
            valid_port = url.port
        except (ValueError, TypeError):
            raise ValueError("Invalid HTTPS base URL") from None
        if (url.scheme.lower() != "https" or not url.hostname or url.username is not None
                or url.password is not None or url.query or url.fragment
                or (valid_port is not None and not 1 <= valid_port <= 65535)):
            raise ValueError("Invalid HTTPS base URL")
        if not isinstance(token, str) or not token or any(c in token for c in "\r\n"):
            raise ValueError("Invalid bearer token")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 120):
            raise ValueError("timeout_seconds must be positive and at most 120")
        _bounded_int(max_payload_bytes, "max_payload_bytes", 1, 64_000_000)
        _bounded_int(concurrency, "concurrency", 1, 16)
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_seconds
        self._max_payload_bytes = max_payload_bytes
        self._semaphore = asyncio.Semaphore(concurrency)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self.skipped_type_ids: list[int] = []

    async def __aenter__(self) -> SourceClient:
        if self._client is not None:
            raise RuntimeError("SourceClient is already open")
        self._client = httpx.AsyncClient(
            timeout=self._timeout, verify=True, follow_redirects=False, transport=self._transport,
            headers={"Authorization": f"Bearer {self._token}", "Accept": "application/json"},
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, path: str, params: dict[str, int], *, require_meta_ids: bool = False) -> object:
        try:
            async with self._semaphore:
                async with asyncio.timeout(self._timeout):
                    return await self._get_inner(path, params, require_meta_ids=require_meta_ids)
        except TimeoutError:
            raise SourceError("Source request exceeded total timeout") from None

    async def _get_inner(self, path: str, params: dict[str, int], *, require_meta_ids: bool = False) -> object:
        if self._client is None:
            raise RuntimeError("SourceClient must be used as an async context manager")
        try:
            async with asyncio.timeout(self._timeout):
                async with self._client.stream("GET", self._base_url + path, params=params) as response:
                    if response.status_code != 200:
                        raise SourceError(f"Source HTTP status {response.status_code}")
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(data) + len(chunk) > self._max_payload_bytes:
                            raise SourceError("Source response exceeds payload limit")
                        data.extend(chunk)
            envelope = json.loads(data, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
            if not isinstance(envelope, dict) or "data" not in envelope or "meta" not in envelope:
                raise SourceError("Invalid source envelope")
            _provenance(envelope["meta"], require_scope=True, require_ids=require_meta_ids)
            return envelope["data"]
        except SourceError:
            raise
        except (httpx.HTTPError, ValueError, UnicodeError, RecursionError):
            raise SourceError("Source request or response validation failed") from None

    async def universe(self, limit: int = 200) -> list[UniverseItem]:
        _bounded_int(limit, "limit", 1, 200)
        data = await self._get("/api/predictions/source/universe", {"limit": limit}, require_meta_ids=True)
        if not isinstance(data, list) or len(data) > limit:
            raise SourceError("Invalid source universe")
        items = []
        seen = set()
        try:
            for value in data:
                _provenance(value)
                if type(value.get("type_id")) is not int:
                    raise SourceError("Invalid universe type")
                if "observed_at" not in value:
                    raise SourceError("Missing source timestamp field")
                type_id = value["type_id"]
                if type_id <= 0 or type_id in seen:
                    raise SourceError("Invalid or duplicate universe type")
                seen.add(type_id)
                if any(value.get(k) is None for k in ("observed_at", "best_bid", "best_ask")):
                    self.skipped_type_ids.append(type_id)
                    continue
                item = UniverseItem.model_validate(value)
                items.append(item)
        except SourceError:
            raise
        except (ValueError, TypeError, AttributeError):
            raise SourceError("Invalid source universe item") from None
        return items

    async def type_history(self, type_id: int, days: int = 365) -> TypeSource:
        _bounded_int(type_id, "type_id", 1, 2_147_483_647)
        _bounded_int(days, "days", 1, 365)
        data = await self._get(f"/api/predictions/source/types/{type_id}", {"days": days})
        try:
            _provenance(data)
            if not isinstance(data, dict) or type(data.get("type_id")) is not int or data["type_id"] != type_id:
                raise SourceError("Source type does not match request")
            if data.get("station_id") != STATION_ID or data.get("region_id") != REGION_ID:
                raise SourceError("Source type has invalid station or region")
            # Nested records must not contradict the source's station scope either.
            for record in data.get("history", []):
                _provenance(record)
            if data.get("book") is not None:
                _provenance(data["book"])
                if any(data["book"].get(k) is None for k in ("observed_at", "best_bid", "best_ask")):
                    raise SourceUnavailable("Station book unavailable")
            else:
                raise SourceUnavailable("Station book unavailable")
            parsed = TypeSource.model_validate(data)
            dates = [row.date for row in parsed.history]
            if len(set(dates)) != len(dates):
                raise SourceError("Duplicate source history date")
            return parsed
        except SourceError:
            raise
        except (ValueError, TypeError, AttributeError):
            raise SourceError("Invalid source type data") from None

    async def fetch_all(self, limit: int = 200, days: int = 365) -> list[TypeSource]:
        _bounded_int(days, "days", 1, 365)
        self.skipped_type_ids = []
        items = await self.universe(limit=limit)
        async def fetch(item):
            try:
                return await self.type_history(item.type_id, days=days)
            except SourceUnavailable:
                self.skipped_type_ids.append(item.type_id)
                return None
        tasks = [asyncio.create_task(fetch(item)) for item in items]
        try:
            return [row for row in await asyncio.gather(*tasks) if row is not None]
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
