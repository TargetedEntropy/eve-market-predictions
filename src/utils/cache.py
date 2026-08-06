"""Redis-backed cache for ESI responses and ETags.

ESI serves market orders with a 300s cache and history with 3600s. Honouring
``Expires``/``ETag`` means a collection run that lands inside the server-side
cache window costs one 304 instead of a full payload download.

Every operation degrades to a no-op when Redis is unreachable or disabled, so
the collectors keep working without it.
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional, Tuple

from src.config import settings

logger = logging.getLogger(__name__)

_PREFIX = "esi:v1:"


def cache_key(endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
    """Stable key for an (endpoint, params) pair."""
    canonical = json.dumps(params or {}, sort_keys=True, default=str)
    digest = hashlib.sha256(f"{endpoint}|{canonical}".encode()).hexdigest()[:32]
    return f"{_PREFIX}{digest}"


class ESICache:
    """Async ETag + payload cache. Never raises on Redis failures."""

    def __init__(self, url: Optional[str] = None, enabled: Optional[bool] = None):
        self.url = url or settings.redis_url
        self.enabled = settings.esi_cache_enabled if enabled is None else enabled
        self._client = None
        self._unavailable = False

    async def _get_client(self):
        if not self.enabled or self._unavailable:
            return None
        if self._client is None:
            try:
                import redis.asyncio as redis  # imported lazily; optional at runtime

                self._client = redis.from_url(self.url, decode_responses=True)
                await self._client.ping()
                logger.debug("ESI cache connected to Redis")
            except Exception as e:
                logger.warning(f"ESI cache disabled (Redis unavailable): {e}")
                self._client = None
                self._unavailable = True
        return self._client

    async def get(self, key: str) -> Tuple[Optional[str], Optional[Any]]:
        """Return ``(etag, payload)``; either may be None."""
        client = await self._get_client()
        if client is None:
            return None, None
        try:
            raw = await client.get(key)
        except Exception as e:
            logger.debug(f"ESI cache read failed: {e}")
            return None, None
        if not raw:
            return None, None
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            return None, None
        return entry.get("etag"), entry.get("payload")

    async def set(self, key: str, payload: Any, etag: Optional[str], ttl: int) -> None:
        """Store a payload with its ETag for ``ttl`` seconds."""
        client = await self._get_client()
        if client is None:
            return
        # Keep the ETag well past the payload's freshness window so a stale
        # entry can still be revalidated with If-None-Match.
        try:
            await client.set(
                key,
                json.dumps({"etag": etag, "payload": payload}),
                ex=max(ttl, 60) * 4,
            )
        except Exception as e:
            logger.debug(f"ESI cache write failed: {e}")

    async def touch(self, key: str, ttl: int) -> None:
        """Extend an entry's lifetime after a 304 revalidation."""
        client = await self._get_client()
        if client is None:
            return
        try:
            await client.expire(key, max(ttl, 60) * 4)
        except Exception as e:
            logger.debug(f"ESI cache touch failed: {e}")

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # pragma: no cover - best effort
                pass
            self._client = None
