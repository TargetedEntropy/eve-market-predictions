"""Configuration validation (issue 1.6: RETRAINING_DAY=sunday crashed startup)."""

import pytest
from apscheduler.triggers.cron import CronTrigger
from pydantic import ValidationError

from src.config import Settings


@pytest.mark.parametrize(
    "value,expected",
    [
        ("sun", "sun"),
        ("sunday", "sun"),
        ("Sunday", "sun"),
        ("SUNDAY", "sun"),
        ("  monday  ", "mon"),
        ("tues", "tue"),
        ("thurs", "thu"),
        ("0", "0"),
        ("6", "6"),
    ],
)
def test_retraining_day_is_normalized(value, expected):
    assert Settings(retraining_day=value).retraining_day == expected


@pytest.mark.parametrize("value", ["funday", "7", "-1", ""])
def test_invalid_retraining_day_is_rejected(value):
    with pytest.raises(ValidationError):
        Settings(retraining_day=value)


@pytest.mark.parametrize("value", ["sunday", "sun", "3", "wednesday"])
def test_normalized_day_is_accepted_by_apscheduler(value):
    """The whole point: CronTrigger must not raise on the configured value."""
    day = Settings(retraining_day=value).retraining_day
    assert CronTrigger(day_of_week=day, hour=2) is not None


def test_region_ids_parsed_from_csv():
    settings = Settings(collection_region_ids="10000002, 10000043 ,10000032")
    assert settings.region_ids == [10000002, 10000043, 10000032]


def test_region_ids_reject_non_numeric():
    with pytest.raises(ValidationError):
        Settings(collection_region_ids="the-forge")


def test_region_ids_reject_empty():
    with pytest.raises(ValidationError):
        Settings(collection_region_ids="  ")


def test_cors_origins_parsed_from_csv():
    settings = Settings(api_cors_origins="http://a.example, http://b.example")
    assert settings.cors_origins == ["http://a.example", "http://b.example"]


def test_wildcard_cors_with_credentials_is_rejected():
    """Issue 7.2: allow_origins=['*'] plus credentials is invalid and unsafe."""
    with pytest.raises(ValidationError):
        Settings(api_cors_origins="*", api_cors_allow_credentials=True)


def test_wildcard_cors_without_credentials_is_allowed():
    settings = Settings(api_cors_origins="*", api_cors_allow_credentials=False)
    assert settings.cors_origins == ["*"]


def test_unknown_env_keys_are_ignored():
    """Retired keys (e.g. MODEL_PATH) in a stale .env must not break startup."""
    assert Settings(model_path="data/models/best_model.pth") is not None
