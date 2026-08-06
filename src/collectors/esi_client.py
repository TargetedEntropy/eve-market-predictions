"""EVE Swagger Interface (ESI) API client with rate limiting.

Safety properties this client guarantees (CCP will IP-ban clients that ignore
them):

* **Error budget.** ``X-Esi-Error-Limit-Remain`` is tracked globally. When it
  drops to ``esi_error_limit_floor`` every task blocks on a shared gate until
  ``X-Esi-Error-Limit-Reset`` seconds have elapsed -- not just the task that
  saw the header.
* **HTTP 420.** Closes the same gate for the full reset window, so the ~20
  concurrent requests already in flight stop instead of piling on.
* **5xx / timeouts.** Retried with exponential backoff plus jitter.
* **Caching.** ``ETag``/``If-None-Match`` conditional requests backed by Redis;
  304 responses reuse the cached payload.
* **Pagination.** Driven by the ``X-Pages`` header, not a page-size guess.
"""

import asyncio
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import aiometer
import httpx

from src.config import settings
from src.utils.cache import ESICache, cache_key

logger = logging.getLogger(__name__)

# Fallback cache lifetime when ESI omits Cache-Control/Expires
DEFAULT_TTL_SECONDS = 300


class ESIRateLimiter:
    """Shared gate that pauses *all* requests when the error budget runs low."""

    def __init__(self):
        self._gate = asyncio.Event()
        self._gate.set()
        self._lock = asyncio.Lock()
        self._paused_until = 0.0

    @property
    def is_paused(self) -> bool:
        return not self._gate.is_set()

    async def wait(self) -> None:
        """Block while the gate is closed."""
        await self._gate.wait()

    async def pause(self, seconds: float, reason: str) -> None:
        """Close the gate for ``seconds``.

        Concurrent callers coalesce: the first one owns the sleep loop, later
        ones only extend the deadline and then wait for the reopen.
        """
        loop = asyncio.get_running_loop()

        async with self._lock:
            already_paused = self.is_paused
            deadline = loop.time() + seconds
            if deadline > self._paused_until:
                self._paused_until = deadline
                logger.warning(f"Pausing ESI requests for {seconds:.1f}s: {reason}")
            if not already_paused:
                self._gate.clear()

        if already_paused:
            await self._gate.wait()
            return

        # Owner of the pause: sleep until nobody has extended the deadline.
        while True:
            remaining = self._paused_until - loop.time()
            if remaining <= 0:
                break
            await asyncio.sleep(remaining)

        async with self._lock:
            self._gate.set()
        logger.info("ESI request gate reopened")


class ESIClient:
    """
    Async HTTP client for ESI API with automatic rate limiting.

    Rate limits: 100 requests per 60 seconds (configurable), plus ESI's own
    error budget of 100 errors per 60 seconds.
    """

    def __init__(
        self,
        cache: Optional[ESICache] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        """
        Args:
            cache: Response/ETag cache. Defaults to the Redis-backed one.
            transport: httpx transport override (used by tests to stub ESI).
        """
        self.base_url = settings.esi_base_url
        self.datasource = settings.esi_datasource
        self.rate_limit = settings.esi_rate_limit
        self.rate_period = settings.esi_rate_period
        self.max_retries = settings.esi_max_retries
        self.error_limit_floor = settings.esi_error_limit_floor
        self.client: Optional[httpx.AsyncClient] = None
        self._transport = transport

        self.limiter = ESIRateLimiter()
        self.cache = cache if cache is not None else ESICache()

        # Track rate limiting
        self._request_count = 0
        self._error_count = 0
        self._cache_hits = 0
        self._retry_count = 0

    @property
    def max_per_second(self) -> float:
        return self.rate_limit / self.rate_period

    async def __aenter__(self):
        """Async context manager entry"""
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            headers={"User-Agent": settings.esi_user_agent},
            transport=self._transport,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.aclose()
            self.client = None
        await self.cache.close()

    # ------------------------------------------------------------------ #
    # Request plumbing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ttl_from_headers(headers: httpx.Headers) -> int:
        """Seconds this response stays fresh, from Cache-Control/Expires."""
        cache_control = headers.get("cache-control", "")
        for directive in cache_control.split(","):
            directive = directive.strip()
            if directive.startswith("max-age="):
                try:
                    return max(int(directive.split("=", 1)[1]), 0)
                except ValueError:
                    break
        return DEFAULT_TTL_SECONDS

    async def _handle_error_budget(self, headers: httpx.Headers) -> None:
        """Pause everything if ESI says the error budget is nearly gone."""
        remain_header = headers.get("x-esi-error-limit-remain")
        if remain_header is None:
            return
        try:
            remain = int(remain_header)
        except ValueError:
            return
        if remain > self.error_limit_floor:
            return

        try:
            reset = int(headers.get("x-esi-error-limit-reset", "60"))
        except ValueError:
            reset = 60

        await self.limiter.pause(
            reset + 1,
            f"error budget low ({remain} remaining, resets in {reset}s)",
        )

    async def _send(
        self,
        endpoint: str,
        params: Dict[str, Any],
        method: str,
        etag: Optional[str],
    ) -> httpx.Response:
        await self.limiter.wait()
        headers = {"If-None-Match": etag} if etag else None
        return await self.client.request(method, endpoint, params=params, headers=headers)

    async def request_with_headers(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
    ) -> Tuple[Any, httpx.Headers]:
        """Make a rate-limited, cached, retrying request.

        Returns:
            ``(payload, headers)``. ``payload`` is ``None`` for 404s. On a 304
            the cached payload is returned with the 304 response's headers.

        Raises:
            httpx.HTTPStatusError: for non-404 client errors and for server
                errors that survive every retry.
        """
        if not self.client:
            raise RuntimeError(
                "Client not initialized. Use 'async with ESIClient()' context manager"
            )

        params = dict(params or {})
        params["datasource"] = self.datasource

        key = cache_key(endpoint, params)
        etag, cached_payload = await self.cache.get(key)

        attempt = 0
        while True:
            try:
                response = await self._send(endpoint, params, method, etag)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                self._error_count += 1
                if attempt >= self.max_retries:
                    logger.error(f"Request failed after {attempt} retries: {endpoint} - {e}")
                    raise
                await self._backoff(attempt, f"{type(e).__name__} on {endpoint}")
                attempt += 1
                continue

            await self._handle_error_budget(response.headers)
            status = response.status_code

            if status == 304:
                self._cache_hits += 1
                self._request_count += 1
                await self.cache.touch(key, self._ttl_from_headers(response.headers))
                return cached_payload, response.headers

            if status == 404:
                self._error_count += 1
                logger.debug(f"Resource not found: {endpoint}")
                return None, response.headers

            if status == 420:
                self._error_count += 1
                try:
                    reset = int(response.headers.get("x-esi-error-limit-reset", "60"))
                except ValueError:
                    reset = 60
                await self.limiter.pause(reset + 1, "HTTP 420 error limit exceeded")
                if attempt >= self.max_retries:
                    response.raise_for_status()
                attempt += 1
                self._retry_count += 1
                continue

            if status >= 500:
                self._error_count += 1
                if attempt >= self.max_retries:
                    logger.error(f"HTTP {status} after {attempt} retries: {endpoint}")
                    response.raise_for_status()
                await self._backoff(attempt, f"HTTP {status} on {endpoint}")
                attempt += 1
                continue

            if status >= 400:
                self._error_count += 1
                logger.error(f"HTTP error {status}: {endpoint}")
                response.raise_for_status()

            self._request_count += 1
            payload = response.json()

            response_etag = response.headers.get("etag")
            if response_etag:
                await self.cache.set(
                    key, payload, response_etag, self._ttl_from_headers(response.headers)
                )

            return payload, response.headers

    async def _backoff(self, attempt: int, reason: str) -> None:
        """Exponential backoff with full jitter."""
        self._retry_count += 1
        delay = min(2 ** attempt, 30) * (0.5 + random.random() / 2)
        logger.warning(f"Retrying in {delay:.1f}s ({reason})")
        await asyncio.sleep(delay)

    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
    ) -> Any:
        """Make a request and return just the JSON payload."""
        payload, _ = await self.request_with_headers(endpoint, params, method)
        return payload

    # ------------------------------------------------------------------ #
    # Endpoints
    # ------------------------------------------------------------------ #

    async def get_market_history(
        self, region_id: int, type_id: int
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get historical market data for an item in a region

        Args:
            region_id: Region ID (e.g., 10000002 for The Forge)
            type_id: Item type ID

        Returns:
            List of daily market statistics or None if not found
        """
        endpoint = f"/markets/{region_id}/history/"
        params = {"type_id": type_id}
        return await self._make_request(endpoint, params)

    async def get_market_orders(
        self, region_id: int, type_id: Optional[int] = None, order_type: str = "all"
    ) -> List[Dict[str, Any]]:
        """
        Get current market orders for a region.

        Pagination follows the ``X-Pages`` header. The previous heuristic
        ("stop when a page has fewer than 1000 rows") requested a nonexistent
        page whenever the last page happened to hold exactly 1000 rows, and
        each of those 4xx responses spent error budget.

        Args:
            region_id: Region ID
            type_id: Optional filter for specific item type
            order_type: "all", "buy", or "sell"

        Returns:
            List of market orders
        """
        endpoint = f"/markets/{region_id}/orders/"
        params: Dict[str, Any] = {"order_type": order_type}

        if type_id:
            params["type_id"] = type_id

        first_params = dict(params, page=1)
        first_page, headers = await self.request_with_headers(endpoint, first_params)

        if not first_page:
            return []

        all_orders: List[Dict[str, Any]] = list(first_page)

        try:
            total_pages = int(headers.get("x-pages", "1"))
        except ValueError:
            total_pages = 1

        if total_pages <= 1:
            return all_orders

        async def fetch_page(page: int):
            return await self._make_request(endpoint, dict(params, page=page))

        # Remaining pages go through the same pacing as every other request.
        async with aiometer.amap(
            fetch_page,
            list(range(2, total_pages + 1)),
            max_per_second=self.max_per_second,
            max_at_once=20,
        ) as results:
            async for orders in results:
                if orders:
                    all_orders.extend(orders)

        return all_orders

    async def get_market_prices(self) -> List[Dict[str, Any]]:
        """
        Get average/adjusted reference prices for all items

        Returns:
            List of price data
        """
        endpoint = "/markets/prices/"
        return await self._make_request(endpoint) or []

    async def get_item_info(self, type_id: int) -> Optional[Dict[str, Any]]:
        """
        Get information about an item type

        Args:
            type_id: Item type ID

        Returns:
            Item information or None if not found
        """
        endpoint = f"/universe/types/{type_id}/"
        return await self._make_request(endpoint)

    async def get_group_info(self, group_id: int) -> Optional[Dict[str, Any]]:
        """Get an inventory group (used to resolve an item's category name)"""
        return await self._make_request(f"/universe/groups/{group_id}/")

    async def get_category_info(self, category_id: int) -> Optional[Dict[str, Any]]:
        """Get an inventory category"""
        return await self._make_request(f"/universe/categories/{category_id}/")

    async def bulk_get_market_history(
        self, region_id: int, type_ids: List[int]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get market history for multiple items with rate limiting

        Args:
            region_id: Region ID
            type_ids: List of item type IDs

        Returns:
            Dictionary mapping type_id to market history
        """
        async def fetch_history(type_id: int):
            history = await self.get_market_history(region_id, type_id)
            return (type_id, history)

        async with aiometer.amap(
            fetch_history,
            type_ids,
            max_per_second=self.max_per_second,
            max_at_once=20,  # Concurrent connections
        ) as results:
            result_list = [item async for item in results]

        return {type_id: history for type_id, history in result_list if history}

    async def bulk_get_market_orders(
        self, region_id: int, type_ids: List[int]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get market orders for multiple items with rate limiting

        Args:
            region_id: Region ID
            type_ids: List of item type IDs

        Returns:
            Dictionary mapping type_id to market orders
        """
        async def fetch_orders(type_id: int):
            orders = await self.get_market_orders(region_id, type_id)
            return (type_id, orders)

        async with aiometer.amap(
            fetch_orders,
            type_ids,
            max_per_second=self.max_per_second,
            max_at_once=20,
        ) as results:
            result_list = [item async for item in results]

        return {type_id: orders for type_id, orders in result_list}

    def get_stats(self) -> Dict[str, int]:
        """Get client statistics"""
        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "cache_hits": self._cache_hits,
            "retries": self._retry_count,
        }
