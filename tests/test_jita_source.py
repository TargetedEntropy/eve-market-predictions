"""Mocked source API tests; never contact a real endpoint or use credentials."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from src.jita import source


META = {"scope": "station", "station_id": source.STATION_ID, "region_id": source.REGION_ID}


def response(data, meta=META):
    return httpx.Response(200, json={"data": data, "meta": meta})


@pytest.mark.parametrize("url", ["http://example.test", "https://user:pass@example.test", "https://example.test/?q=1", "https://example.test/#fragment", "https:///nohost"])
def test_rejects_insecure_or_ambiguous_base_urls(url):
    with pytest.raises(ValueError):
        source.SourceClient(url, "secret")


@pytest.mark.parametrize("kwargs", [{"concurrency": 0}, {"concurrency": 17}, {"concurrency": True}, {"timeout_seconds": float("nan")}, {"max_payload_bytes": 0}])
def test_constructor_bounds(kwargs):
    with pytest.raises(ValueError):
        source.SourceClient("https://example.test", "secret", **kwargs)


async def test_authenticated_requests_and_fetch_all(monkeypatch):
    # Model parsing is tested in schema tests; isolate transport orchestration here.
    monkeypatch.setattr(source, "UniverseItem", SimpleNamespace(model_validate=lambda x: SimpleNamespace(**x)))
    requests = []
    active = 0
    max_active = 0

    async def handler(request):
        nonlocal active, max_active
        requests.append(request)
        if request.url.path.endswith("/universe"):
            return response([{"type_id": i, "best_bid": 4, "best_ask": 6,
                              "observed_at": "2026-01-01T00:00:00Z"} for i in (34, 35)])
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        type_id = int(request.url.path.rsplit("/", 1)[1])
        return response({"type_id": type_id, "item_name": "Test", "station_id": source.STATION_ID,
                         "region_id": source.REGION_ID, "history": [],
                         "book": {"best_bid": 4, "best_ask": 6,
                                  "observed_at": "2026-01-01T00:00:00Z"}}, {"scope": "station"})

    async with source.SourceClient("https://example.test", "fake-secret", concurrency=1,
                                   transport=httpx.MockTransport(handler)) as client:
        result = await client.fetch_all(limit=2, days=30)
    assert [item.type_id for item in result] == [34, 35]
    assert max_active == 1
    assert all(r.headers["authorization"] == "Bearer fake-secret" for r in requests)
    assert requests[0].url.params["limit"] == "2"
    assert all(r.url.params["days"] == "30" for r in requests[1:])


@pytest.mark.parametrize("meta", [None, {}, {"scope": "region"}, {"scope": "station", "station_id": 1, "region_id": source.REGION_ID}, {"scope": "station", "station_id": source.STATION_ID, "region_id": 1}])
async def test_rejects_bad_universe_meta(meta):
    async with source.SourceClient("https://example.test", "secret", transport=httpx.MockTransport(lambda r: response([], meta))) as client:
        with pytest.raises(source.SourceError):
            await client.universe()


async def test_rejects_duplicate_universe_ids(monkeypatch):
    monkeypatch.setattr(source, "UniverseItem", SimpleNamespace(model_validate=lambda x: SimpleNamespace(**x)))
    transport = httpx.MockTransport(lambda r: response([{"type_id": 34}, {"type_id": 34}]))
    async with source.SourceClient("https://example.test", "secret", transport=transport) as client:
        with pytest.raises(source.SourceError):
            await client.universe()


@pytest.mark.parametrize("data", [
    {"type_id": 35, "station_id": source.STATION_ID, "region_id": source.REGION_ID},
    {"type_id": 34, "station_id": 1, "region_id": source.REGION_ID},
    {"type_id": 34, "station_id": source.STATION_ID, "region_id": source.REGION_ID, "history": [{"station_id": 1}]},
])
async def test_rejects_mismatched_type_or_station(data):
    async with source.SourceClient("https://example.test", "secret", transport=httpx.MockTransport(lambda r: response(data))) as client:
        with pytest.raises(source.SourceError):
            await client.type_history(34)


@pytest.mark.parametrize("payload", [b'{"data": [], "data": [], "meta": {}}', b'{"data": NaN, "meta": {}}', b'not-json'])
async def test_bad_json_never_leaks_response(payload):
    async with source.SourceClient("https://example.test", "secret", transport=httpx.MockTransport(lambda r: httpx.Response(200, content=payload))) as client:
        with pytest.raises(source.SourceError) as exc:
            await client.universe()
        assert "secret" not in str(exc.value)
        assert payload.decode() not in str(exc.value)


async def test_oversize_response_and_redirect_are_rejected():
    for result in (httpx.Response(200, content=b"x" * 100), httpx.Response(302, headers={"location": "https://elsewhere.test"})):
        async with source.SourceClient("https://example.test", "secret", max_payload_bytes=10, transport=httpx.MockTransport(lambda r: result)) as client:
            with pytest.raises(source.SourceError):
                await client.universe()


async def test_stream_limit_stops_consuming_chunks():
    consumed = []
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for chunk in (b"12345", b"67890", b"unread"):
                consumed.append(chunk)
                yield chunk
    async with source.SourceClient("https://example.test", "secret", max_payload_bytes=6,
                                   transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=Stream()))) as client:
        with pytest.raises(source.SourceError, match="payload"):
            await client.universe()
    assert consumed == [b"12345", b"67890"]


async def test_transport_exception_does_not_leak_token():
    def handler(request):
        raise httpx.ConnectError("secret body secret", request=request)
    async with source.SourceClient("https://example.test", "secret", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(source.SourceError) as exc:
            await client.universe()
        assert "secret" not in str(exc.value)


async def test_invalid_request_bounds():
    async with source.SourceClient("https://example.test", "secret", transport=httpx.MockTransport(lambda r: response([]))) as client:
        for value in [0, 201, True]:
            with pytest.raises(ValueError):
                await client.universe(value)
        for value in [0, 366, True]:
            with pytest.raises(ValueError):
                await client.type_history(34, value)


async def test_real_models_reject_bad_item_and_duplicate_dates():
    item = {"type_id": 34, "item_name": "Tritanium", "best_bid": 4, "best_ask": 6,
            "observed_at": "2026-01-01T00:00:00Z"}
    row = {"date": "2026-01-01", "mid_price": 5, "snapshot_count": 1}
    typ = {"type_id": 34, "item_name": "Tritanium", "station_id": source.STATION_ID,
           "region_id": source.REGION_ID, "history": [row, row],
           "book": {"best_bid": 4, "best_ask": 6, "observed_at": "2026-01-01T00:00:00Z"}}
    def handler(request):
        return response([item] if request.url.path.endswith("universe") else typ)
    async with source.SourceClient("https://example.test", "secret", transport=httpx.MockTransport(handler)) as client:
        assert (await client.universe())[0].item_name == "Tritanium"
        with pytest.raises(source.SourceError, match="Duplicate"):
            await client.type_history(34)
    item.pop("observed_at")
    async with source.SourceClient("https://example.test", "secret", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(source.SourceError):
            await client.universe()


async def test_explicitly_missing_book_is_skipped_not_retimestamped():
    item = {"type_id":34,"item_name":"Tritanium","best_bid":None,"best_ask":None,
            "observed_at":None}
    async with source.SourceClient("https://example.test", "secret",
            transport=httpx.MockTransport(lambda request:response([item]))) as client:
        assert await client.fetch_all() == []
        assert client.skipped_type_ids == [34]


async def test_total_request_timeout_is_bounded():
    async def handler(request):
        await asyncio.sleep(0.1)
        return response([])
    async with source.SourceClient("https://example.test", "secret", timeout_seconds=0.01,
            transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(source.SourceError,match="timeout"):
            await client.universe()
