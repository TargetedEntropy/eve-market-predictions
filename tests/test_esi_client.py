"""ESI client hardening (issues 5.1-5.3).

Everything runs against an httpx MockTransport -- no network, no Redis.
"""

import asyncio

import httpx
import pytest

from src.collectors.esi_client import ESIClient, ESIRateLimiter
from src.utils.cache import ESICache, cache_key


def make_client(handler, cache=None):
    """ESIClient wired to a stub transport and (by default) no cache."""
    return ESIClient(
        cache=cache if cache is not None else ESICache(enabled=False),
        transport=httpx.MockTransport(handler),
    )


# --------------------------------------------------------------------------- #
# Pagination (5.3)
# --------------------------------------------------------------------------- #

async def test_pagination_follows_x_pages_header():
    calls = []

    def handler(request):
        page = int(request.url.params.get("page", 1))
        calls.append(page)
        return httpx.Response(
            200,
            json=[{"order_id": page * 1000 + i} for i in range(1000)],
            headers={"x-pages": "3"},
        )

    async with make_client(handler) as client:
        orders = await client.get_market_orders(10000002, 34)

    assert sorted(calls) == [1, 2, 3]
    assert len(orders) == 3000


async def test_full_last_page_does_not_trigger_an_extra_request():
    """The old heuristic asked for page 2 whenever page 1 held exactly 1000 rows."""
    calls = []

    def handler(request):
        calls.append(int(request.url.params.get("page", 1)))
        return httpx.Response(
            200, json=[{"order_id": i} for i in range(1000)], headers={"x-pages": "1"}
        )

    async with make_client(handler) as client:
        orders = await client.get_market_orders(10000002, 34)

    assert calls == [1]
    assert len(orders) == 1000


async def test_missing_x_pages_header_means_one_page():
    def handler(request):
        return httpx.Response(200, json=[{"order_id": 1}])

    async with make_client(handler) as client:
        assert len(await client.get_market_orders(10000002, 34)) == 1


async def test_empty_first_page_returns_empty_list():
    def handler(request):
        return httpx.Response(200, json=[], headers={"x-pages": "1"})

    async with make_client(handler) as client:
        assert await client.get_market_orders(10000002, 34) == []


# --------------------------------------------------------------------------- #
# Error handling and backoff (5.1)
# --------------------------------------------------------------------------- #

async def test_404_returns_none_without_raising():
    def handler(request):
        return httpx.Response(404, json={"error": "not found"})

    async with make_client(handler) as client:
        assert await client.get_item_info(999_999_999) is None


async def test_server_errors_are_retried_then_succeed(no_sleep):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503, json={"error": "backend down"})
        return httpx.Response(200, json={"name": "PLEX"})

    async with make_client(handler) as client:
        assert await client.get_item_info(44992) == {"name": "PLEX"}

    assert attempts["n"] == 3
    assert len(no_sleep) == 2  # two backoff sleeps
    assert no_sleep == sorted(no_sleep)  # exponential, non-decreasing


async def test_server_errors_raise_after_max_retries(no_sleep):
    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    client = make_client(handler)
    client.max_retries = 2
    async with client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_item_info(44992)


async def test_timeouts_are_retried(no_sleep):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json={"name": "PLEX"})

    async with make_client(handler) as client:
        assert await client.get_item_info(44992) == {"name": "PLEX"}
    assert attempts["n"] == 2


async def test_420_pauses_then_retries(no_sleep):
    """A 420 must pause the shared gate, not just raise into 20 concurrent tasks."""
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(420, json={"error": "limit"},
                                  headers={"x-esi-error-limit-reset": "5"})
        return httpx.Response(200, json={"ok": True})

    async with make_client(handler) as client:
        assert await client.get_item_info(44992) == {"ok": True}

    assert attempts["n"] == 2
    assert any(seconds >= 5 for seconds in no_sleep), "should sleep for the reset window"


async def test_low_error_budget_pauses_requests(no_sleep):
    """Below the floor the client waits for the window instead of firing on."""
    def handler(request):
        return httpx.Response(
            200,
            json={"ok": True},
            headers={"x-esi-error-limit-remain": "3", "x-esi-error-limit-reset": "17"},
        )

    async with make_client(handler) as client:
        await client.get_item_info(44992)

    assert any(seconds >= 17 for seconds in no_sleep)


async def test_healthy_error_budget_does_not_pause(no_sleep):
    def handler(request):
        return httpx.Response(
            200,
            json={"ok": True},
            headers={"x-esi-error-limit-remain": "99", "x-esi-error-limit-reset": "60"},
        )

    async with make_client(handler) as client:
        await client.get_item_info(44992)

    assert no_sleep == []


async def test_user_agent_is_sent():
    """CCP's third-party guidelines require an identifying User-Agent."""
    seen = {}

    def handler(request):
        seen["ua"] = request.headers.get("user-agent")
        return httpx.Response(200, json={})

    async with make_client(handler) as client:
        await client.get_item_info(44992)

    assert seen["ua"] and "eve-discovery" in seen["ua"]


# --------------------------------------------------------------------------- #
# Rate limiter
# --------------------------------------------------------------------------- #

async def test_rate_limiter_blocks_all_waiters_until_reopened():
    limiter = ESIRateLimiter()
    released = []

    async def waiter(name):
        await limiter.wait()
        released.append(name)

    pause = asyncio.create_task(limiter.pause(0.05, "test"))
    await asyncio.sleep(0)  # let pause() close the gate

    waiters = [asyncio.create_task(waiter(i)) for i in range(3)]
    await asyncio.sleep(0.01)
    assert released == [], "waiters must be blocked while paused"

    await asyncio.gather(pause, *waiters)
    assert sorted(released) == [0, 1, 2]


async def test_concurrent_pauses_coalesce():
    limiter = ESIRateLimiter()
    await asyncio.gather(*(limiter.pause(0.02, f"reason-{i}") for i in range(5)))
    assert not limiter.is_paused


# --------------------------------------------------------------------------- #
# Conditional requests / caching (5.2)
# --------------------------------------------------------------------------- #

class FakeCache(ESICache):
    """In-memory stand-in for the Redis cache."""

    def __init__(self):
        super().__init__(enabled=False)
        self.store = {}
        self.touched = []

    async def get(self, key):
        entry = self.store.get(key)
        return (entry["etag"], entry["payload"]) if entry else (None, None)

    async def set(self, key, payload, etag, ttl):
        self.store[key] = {"etag": etag, "payload": payload, "ttl": ttl}

    async def touch(self, key, ttl):
        self.touched.append(key)

    async def close(self):
        pass


async def test_etag_is_stored_then_replayed_on_304():
    cache = FakeCache()
    requests = []

    def handler(request):
        requests.append(request.headers.get("if-none-match"))
        if request.headers.get("if-none-match") == 'W/"abc"':
            return httpx.Response(304, headers={"etag": 'W/"abc"'})
        return httpx.Response(
            200,
            json=[{"date": "2025-01-01", "average": 100}],
            headers={"etag": 'W/"abc"', "cache-control": "max-age=3600"},
        )

    async with make_client(handler, cache=cache) as client:
        first = await client.get_market_history(10000002, 44992)

    async with make_client(handler, cache=cache) as client:
        second = await client.get_market_history(10000002, 44992)
        assert client.get_stats()["cache_hits"] == 1

    assert requests == [None, 'W/"abc"']
    assert first == second
    assert cache.touched, "a 304 should extend the cached entry's lifetime"


async def test_ttl_is_taken_from_cache_control():
    cache = FakeCache()

    def handler(request):
        return httpx.Response(
            200, json=[], headers={"etag": '"x"', "cache-control": "public, max-age=1200"}
        )

    async with make_client(handler, cache=cache) as client:
        await client.get_market_history(10000002, 44992)

    assert next(iter(cache.store.values()))["ttl"] == 1200


async def test_cache_key_is_stable_and_param_sensitive():
    assert cache_key("/x/", {"a": 1, "b": 2}) == cache_key("/x/", {"b": 2, "a": 1})
    assert cache_key("/x/", {"a": 1}) != cache_key("/x/", {"a": 2})
    assert cache_key("/x/", {"a": 1}) != cache_key("/y/", {"a": 1})


async def test_disabled_cache_is_a_no_op():
    cache = ESICache(enabled=False)
    assert await cache.get("anything") == (None, None)
    await cache.set("anything", {"a": 1}, "etag", 60)
    assert await cache.get("anything") == (None, None)
