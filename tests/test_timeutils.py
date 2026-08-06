"""Timestamp conventions (issue 3.1: three different conventions in one repo)."""

from datetime import datetime, timedelta, timezone

from src.utils.timeutils import parse_esi_datetime, to_naive_utc, utcnow


def test_utcnow_is_naive():
    now = utcnow()
    assert now.tzinfo is None


def test_utcnow_matches_utc_not_local():
    delta = abs(utcnow() - datetime.now(timezone.utc).replace(tzinfo=None))
    assert delta < timedelta(seconds=2)


def test_to_naive_utc_converts_aware_datetimes():
    aware = datetime(2025, 6, 1, 12, 0, tzinfo=timezone(timedelta(hours=5)))
    assert to_naive_utc(aware) == datetime(2025, 6, 1, 7, 0)


def test_to_naive_utc_passes_naive_through():
    naive = datetime(2025, 6, 1, 12, 0)
    assert to_naive_utc(naive) == naive


def test_to_naive_utc_handles_none():
    assert to_naive_utc(None) is None


def test_parse_esi_datetime_handles_z_suffix():
    assert parse_esi_datetime("2025-10-25T02:02:59Z") == datetime(2025, 10, 25, 2, 2, 59)


def test_parse_esi_datetime_handles_offset():
    assert parse_esi_datetime("2025-10-25T04:02:59+02:00") == datetime(2025, 10, 25, 2, 2, 59)


def test_parse_esi_datetime_handles_empty():
    assert parse_esi_datetime(None) is None
    assert parse_esi_datetime("") is None
