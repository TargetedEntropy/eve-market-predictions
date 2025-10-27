"""EVE Swagger Interface (ESI) API client with rate limiting"""

import logging
from typing import Any, Dict, List, Optional
import httpx
import aiometer

from src.config import settings

logger = logging.getLogger(__name__)


class ESIClient:
    """
    Async HTTP client for ESI API with automatic rate limiting

    Rate limits: 100 requests per 60 seconds (configurable)
    """

    def __init__(self):
        self.base_url = settings.esi_base_url
        self.datasource = settings.esi_datasource
        self.rate_limit = settings.esi_rate_limit
        self.rate_period = settings.esi_rate_period
        self.client: Optional[httpx.AsyncClient] = None

        # Track rate limiting
        self._request_count = 0
        self._error_count = 0

    async def __aenter__(self):
        """Async context manager entry"""
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.aclose()

    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
    ) -> Any:
        """
        Make rate-limited request to ESI API

        Args:
            endpoint: API endpoint (e.g., "/markets/{region_id}/history/")
            params: Query parameters
            method: HTTP method

        Returns:
            JSON response data

        Raises:
            httpx.HTTPError: On HTTP errors
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Use 'async with ESIClient()' context manager")

        # Add datasource to params
        params = params or {}
        params["datasource"] = self.datasource

        try:
            response = await self.client.request(method, endpoint, params=params)

            # Track rate limit headers
            if "X-Esi-Error-Limit-Remain" in response.headers:
                error_limit_remain = int(response.headers["X-Esi-Error-Limit-Remain"])
                if error_limit_remain < 20:
                    logger.warning(f"ESI error limit low: {error_limit_remain} remaining")

            response.raise_for_status()
            self._request_count += 1

            return response.json()

        except httpx.HTTPStatusError as e:
            self._error_count += 1

            if e.response.status_code == 404:
                logger.debug(f"Resource not found: {endpoint}")
                return None
            elif e.response.status_code == 420:
                logger.error("ESI error limit exceeded!")
                raise
            else:
                logger.error(f"HTTP error {e.response.status_code}: {endpoint}")
                raise

        except Exception as e:
            logger.error(f"Request failed: {endpoint} - {e}")
            raise

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
        Get current market orders for a region

        Args:
            region_id: Region ID
            type_id: Optional filter for specific item type
            order_type: "all", "buy", or "sell"

        Returns:
            List of market orders
        """
        endpoint = f"/markets/{region_id}/orders/"
        params = {"order_type": order_type}

        if type_id:
            params["type_id"] = type_id

        # Market orders may be paginated
        all_orders = []
        page = 1

        while True:
            params["page"] = page
            orders = await self._make_request(endpoint, params)

            if not orders:
                break

            all_orders.extend(orders)

            # Check if more pages exist (ESI returns empty array when done)
            if len(orders) < 1000:  # ESI page size is typically 1000
                break

            page += 1

        return all_orders

    async def get_market_prices(self) -> List[Dict[str, Any]]:
        """
        Get average market prices for all items

        Returns:
            List of price data
        """
        endpoint = "/markets/prices/"
        return await self._make_request(endpoint)

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
        # Use aiometer for rate-limited bulk requests
        async def fetch_history(type_id: int):
            history = await self.get_market_history(region_id, type_id)
            return (type_id, history)

        # Rate limit: X requests per second
        max_per_second = self.rate_limit / self.rate_period

        # Collect results from async generator context manager
        async with aiometer.amap(
            fetch_history,
            type_ids,
            max_per_second=max_per_second,
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

        max_per_second = self.rate_limit / self.rate_period

        # Collect results from async generator context manager
        async with aiometer.amap(
            fetch_orders,
            type_ids,
            max_per_second=max_per_second,
            max_at_once=20,
        ) as results:
            result_list = [item async for item in results]

        return {type_id: orders for type_id, orders in result_list}

    def get_stats(self) -> Dict[str, int]:
        """Get client statistics"""
        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
        }
