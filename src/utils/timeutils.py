"""Timestamp helpers.

The whole project stores **naive UTC** datetimes: every ``DateTime`` column in
``src.database.models`` is ``TIMESTAMP WITHOUT TIME ZONE`` and asyncpg rejects
tz-aware values for those columns.

Rules:

* Never call ``datetime.now()`` (server-local) or the deprecated
  ``datetime.utcnow()`` directly -- use :func:`utcnow`.
* Anything crossing the DB boundary goes through :func:`to_naive_utc`.
"""

from datetime import datetime, timezone
from typing import Optional


def utcnow() -> datetime:
    """Current UTC time as a naive datetime (matches the DB columns)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize any datetime to naive UTC.

    Aware datetimes are converted to UTC then stripped of tzinfo; naive
    datetimes are assumed to already be UTC and returned unchanged.
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def parse_esi_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse an ESI ISO-8601 timestamp (``2025-10-25T02:02:59Z``) to naive UTC."""
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return to_naive_utc(datetime.fromisoformat(text))
