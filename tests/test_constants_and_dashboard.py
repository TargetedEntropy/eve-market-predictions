"""Item identity and dashboard metrics (issues 3.2, 3.5, 4.5)."""

import importlib
import re
from pathlib import Path

import pandas as pd
import pytest

from src.constants import (
    DEFAULT_REGION_ID,
    REGIONS,
    TRACKED_ITEMS,
    TRACKED_TYPE_IDS,
    item_name,
    region_name,
)

dashboard = importlib.import_module("src.dashboard.app")

REPO_ROOT = Path(__file__).parent.parent


# --------------------------------------------------------------------------- #
# Item identity (3.2)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "type_id,name",
    [
        (44992, "PLEX"),
        (40519, "Skill Extractor"),
        (40520, "Large Skill Injector"),
        (11399, "Morphite"),
        (34, "Tritanium"),
    ],
)
def test_tracked_item_names_match_esi(type_id, name):
    """The collector used to label 40519 'Large Skill Injector' and 11399 'Mercoxit'."""
    assert TRACKED_ITEMS[type_id] == name


def test_item_and_region_lookups_have_stable_fallbacks():
    assert item_name(44992) == "PLEX"
    assert "1234" in item_name(1234)
    assert region_name(DEFAULT_REGION_ID) == "The Forge (Jita)"
    assert "999" in region_name(999)


def test_tracked_type_ids_are_unique():
    assert len(TRACKED_TYPE_IDS) == len(set(TRACKED_TYPE_IDS))


def test_no_module_redefines_its_own_item_list():
    """One shared constant, imported everywhere (3.2)."""
    offenders = []
    pattern = re.compile(r"^\s*(ITEMS|DEFAULT_TYPE_IDS|target_type_ids)\s*=\s*\[", re.MULTILINE)

    for path in list((REPO_ROOT / "src").rglob("*.py")) + list(
        (REPO_ROOT / "scripts").rglob("*.py")
    ):
        text = path.read_text()
        for match in pattern.finditer(text):
            # A list comprehension over TRACKED_ITEMS is fine; a literal is not.
            tail = text[match.end():match.end() + 200]
            if "TRACKED_ITEM" not in tail and "for " not in tail:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{match.group(1)}")

    assert offenders == [], f"hardcoded item lists found: {offenders}"


def test_dashboard_items_come_from_the_shared_constant():
    assert [option["value"] for option in dashboard.ITEMS] == TRACKED_TYPE_IDS
    assert [option["label"] for option in dashboard.ITEMS] == list(TRACKED_ITEMS.values())


def test_dashboard_only_offers_regions_that_are_collected():
    """Issue 4.5: Amarr/Dodixie were selectable but never collected."""
    from src.config import settings

    assert [option["value"] for option in dashboard.REGION_OPTIONS] == settings.region_ids
    for option in dashboard.REGION_OPTIONS:
        assert option["value"] in REGIONS


# --------------------------------------------------------------------------- #
# 30-day change (3.5)
# --------------------------------------------------------------------------- #

def frame(days, start=100.0, step=1.0):
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=days, freq="D"),
            "price": [start + step * i for i in range(days)],
        }
    )


def test_thirty_day_change_uses_thirty_days_not_the_whole_range():
    """Issue 3.5: the card used df['price'].iloc[0] of the selected range."""
    df = frame(90)  # 90-day range: first row is 89 days back

    change = dashboard.thirty_day_change(df)

    latest = df["price"].iloc[-1]
    thirty_back = df["price"].iloc[-31]
    assert change == pytest.approx((latest - thirty_back) / thirty_back * 100)

    whole_range = (latest - df["price"].iloc[0]) / df["price"].iloc[0] * 100
    assert change != pytest.approx(whole_range)


def test_thirty_day_change_is_na_without_enough_history():
    assert pd.isna(dashboard.thirty_day_change(frame(10)))


def test_thirty_day_change_on_empty_frame():
    assert pd.isna(dashboard.thirty_day_change(pd.DataFrame()))


def test_thirty_day_change_handles_zero_baseline():
    df = frame(60, start=0.0, step=0.0)
    assert pd.isna(dashboard.thirty_day_change(df))


def test_dashboard_refresh_interval_is_not_sixty_seconds():
    """Issue 6.5: daily data does not need a 60s full re-query for every viewer."""
    from src.config import settings

    assert settings.dashboard_refresh_seconds >= 300


def test_dashboard_exposes_a_wsgi_server_for_gunicorn():
    """Issue 7.1: the container must not run the Werkzeug debug server."""
    assert dashboard.server is not None
    assert not settings_debug()


def settings_debug():
    from src.config import settings

    return settings.dashboard_debug
