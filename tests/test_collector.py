"""Market collector row building (issues 3.1, 3.2, 3.3, 6.1, 6.3).

ESI and the database are both stubbed; what is asserted is the rows the
collector would write.
"""

from contextlib import asynccontextmanager
from datetime import datetime

import pytest

from src.collectors import market_collector as mc
from src.constants import TRACKED_TYPE_IDS


class FakeESIClient:
    """Stands in for ESIClient inside the collector."""

    def __init__(self, history=None, orders=None, items=None, groups=None,
                 categories=None, prices=None):
        self.history = history or {}
        self.orders = orders or {}
        self.items = items or {}
        self.groups = groups or {}
        self.categories = categories or {}
        self.prices = prices or []
        self.group_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def bulk_get_market_history(self, region_id, type_ids):
        return self.history.get(region_id, {})

    async def bulk_get_market_orders(self, region_id, type_ids):
        return self.orders.get(region_id, {})

    async def get_item_info(self, type_id):
        return self.items.get(type_id)

    async def get_group_info(self, group_id):
        self.group_calls.append(group_id)
        return self.groups.get(group_id)

    async def get_category_info(self, category_id):
        return self.categories.get(category_id)

    async def get_market_prices(self):
        return self.prices

    def get_stats(self):
        return {}


@pytest.fixture
def captured(monkeypatch):
    """Capture bulk_upsert calls and stub out the DB session."""
    calls = []

    async def fake_upsert(session, model, rows, **kwargs):
        calls.append((model.__tablename__, rows))
        return len(rows)

    class FakeSession:
        async def commit(self):
            pass

        async def execute(self, *args, **kwargs):  # pragma: no cover - unused here
            raise AssertionError("unexpected query")

    @asynccontextmanager
    async def fake_session():
        yield FakeSession()

    monkeypatch.setattr(mc, "bulk_upsert", fake_upsert)
    monkeypatch.setattr(mc, "get_session", fake_session)
    return calls


def install_client(monkeypatch, client):
    monkeypatch.setattr(mc, "ESIClient", lambda *a, **k: client)


async def test_history_rows_are_bulk_upserted_per_region(monkeypatch, captured):
    client = FakeESIClient(
        history={
            10000002: {44992: [{"date": "2025-01-01", "average": 5_000_000.0,
                                "highest": 5_100_000.0, "lowest": 4_900_000.0,
                                "volume": 1234, "order_count": 42}]},
            10000043: {44992: [{"date": "2025-01-01", "average": 5_200_000.0,
                                "highest": 5_300_000.0, "lowest": 5_100_000.0,
                                "volume": 99, "order_count": 3}]},
        }
    )
    install_client(monkeypatch, client)

    collector = mc.MarketCollector(region_ids=[10000002, 10000043])
    written = await collector.collect_historical_data()

    assert written == 2
    assert [table for table, _ in captured] == ["market_history", "market_history"]
    first_row = captured[0][1][0]
    assert first_row["time"] == datetime(2025, 1, 1)
    assert first_row["region_id"] == 10000002
    assert captured[1][1][0]["region_id"] == 10000043


async def test_order_snapshot_rows(monkeypatch, captured):
    client = FakeESIClient(
        orders={
            10000002: {
                34: [
                    {
                        "order_id": 6_000_000_000,
                        "location_id": 60003760,
                        "price": 5.42,
                        # Mineral volumes exceed 2^31 (issue 3.4)
                        "volume_remain": 3_000_000_000,
                        "volume_total": 4_000_000_000,
                        "is_buy_order": False,
                        "duration": 90,
                        "issued": "2025-10-25T02:02:59Z",
                        "min_volume": 1,
                        "range": "region",
                    }
                ]
            }
        }
    )
    install_client(monkeypatch, client)

    collector = mc.MarketCollector(region_ids=[10000002], type_ids=[34])
    written = await collector.collect_order_snapshots()

    assert written == 1
    table, rows = captured[0]
    assert table == "order_snapshots"
    row = rows[0]
    assert row["volume_remain"] == 3_000_000_000
    assert row["issued"] == datetime(2025, 10, 25, 2, 2, 59)
    assert row["issued"].tzinfo is None
    assert row["snapshot_time"].tzinfo is None, "snapshot_time must be naive UTC"


async def test_item_metadata_resolves_category_names(monkeypatch, captured):
    """Issue 3.3: `category` used to hold a stringified group_id."""
    client = FakeESIClient(
        items={
            44992: {"name": "PLEX", "group_id": 1875, "published": True},
            34: {"name": "Tritanium", "group_id": 18, "published": True},
        },
        groups={
            1875: {"name": "PLEX", "category_id": 5},
            18: {"name": "Mineral", "category_id": 4},
        },
        categories={5: {"name": "Accessories"}, 4: {"name": "Material"}},
    )
    install_client(monkeypatch, client)

    collector = mc.MarketCollector(region_ids=[10000002], type_ids=[44992, 34])
    await collector.update_item_metadata()

    rows = {row["type_id"]: row for row in captured[0][1]}
    assert rows[44992]["category"] == "Accessories"
    assert rows[44992]["group_name"] == "PLEX"
    assert rows[44992]["group_id"] == 1875
    assert rows[34]["category"] == "Material"
    assert not str(rows[34]["category"]).isdigit()
    assert rows[34]["updated_at"] is not None


async def test_group_lookups_are_memoized(monkeypatch, captured):
    client = FakeESIClient(
        items={i: {"name": f"Mineral {i}", "group_id": 18} for i in (34, 35, 36)},
        groups={18: {"name": "Mineral", "category_id": 4}},
        categories={4: {"name": "Material"}},
    )
    install_client(monkeypatch, client)

    await mc.MarketCollector(region_ids=[10000002],
                             type_ids=[34, 35, 36]).update_item_metadata()

    assert client.group_calls == [18], "the same group should only be fetched once"


async def test_metadata_refreshes_rather_than_skipping(monkeypatch, captured):
    """Previously an existing row was skipped, so renames never propagated."""
    client = FakeESIClient(items={44992: {"name": "PLEX (renamed)", "group_id": 1875}},
                           groups={1875: {"name": "PLEX", "category_id": 5}},
                           categories={5: {"name": "Accessories"}})
    install_client(monkeypatch, client)

    await mc.MarketCollector(region_ids=[10000002], type_ids=[44992]).update_item_metadata()

    assert captured[0][1][0]["name"] == "PLEX (renamed)"


async def test_reference_prices_are_filtered_to_tracked_items(monkeypatch, captured):
    client = FakeESIClient(
        prices=[
            {"type_id": 44992, "adjusted_price": 5.0, "average_price": 6.0},
            {"type_id": 99999, "adjusted_price": 1.0, "average_price": 1.0},
        ]
    )
    install_client(monkeypatch, client)

    written = await mc.MarketCollector(region_ids=[10000002]).collect_reference_prices()

    assert written == 1
    assert captured[0][0] == "market_prices"
    assert captured[0][1][0]["type_id"] == 44992


def test_default_type_ids_come_from_shared_constants():
    assert mc.MarketCollector().type_ids == TRACKED_TYPE_IDS
    assert mc.MarketCollector.DEFAULT_TYPE_IDS == TRACKED_TYPE_IDS


def test_default_regions_come_from_settings():
    from src.config import settings

    assert mc.MarketCollector().region_ids == settings.region_ids
