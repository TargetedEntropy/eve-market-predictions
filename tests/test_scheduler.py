"""Scheduler wiring (issues 2.1-2.5)."""

from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from scripts import scheduler_daemon
from src.config import settings

EXPECTED_JOBS = {
    "market_collector",
    "feature_computer",
    "history_collector",      # 2.1
    "reference_prices",       # 9.2
    "prediction_evaluator",   # 2.4
    "prediction_generator",   # 2.3
    "drift_checker",          # 2.3
    "model_retrainer",        # 2.2
}


def jobs():
    return {job.id: job for job in scheduler_daemon.build_scheduler().get_jobs()}


def test_every_documented_job_is_scheduled():
    assert set(jobs()) == EXPECTED_JOBS


def test_all_jobs_guard_against_overlap():
    """Issue 2.5: a slow ESI response used to let runs stack up."""
    for job_id, job in jobs().items():
        assert job.max_instances == 1, job_id
        assert job.coalesce is True, job_id
        assert job.misfire_grace_time == scheduler_daemon.MISFIRE_GRACE_SECONDS, job_id


def test_history_collection_is_daily():
    trigger = jobs()["history_collector"].trigger
    assert isinstance(trigger, CronTrigger)
    assert str(trigger.fields[trigger.FIELD_NAMES.index("hour")]) == str(
        settings.history_collection_hour
    )


def test_snapshot_collection_uses_the_configured_interval():
    trigger = jobs()["market_collector"].trigger
    assert isinstance(trigger, IntervalTrigger)
    assert trigger.interval.total_seconds() == settings.collection_interval_minutes * 60


def test_retraining_uses_the_normalized_day():
    trigger = jobs()["model_retrainer"].trigger
    assert isinstance(trigger, CronTrigger)
    assert str(trigger.fields[trigger.FIELD_NAMES.index("day_of_week")]) == \
        settings.retraining_day


def test_scoring_runs_after_history_collection():
    """actual_price can only be filled once the realized day has been collected."""
    history_hour = settings.history_collection_hour
    scoring = jobs()["prediction_evaluator"].trigger
    assert str(scoring.fields[scoring.FIELD_NAMES.index("hour")]) == str(
        (history_hour + 1) % 24
    )


def test_retrain_placeholder_is_gone():
    """Issue 2.2: retrain_model() used to be a bare `pass`."""
    import inspect

    source = inspect.getsource(scheduler_daemon.retrain_models)
    assert "train_item" in source
    assert "promote_only_if_better=True" in source


async def test_job_wrapper_alerts_on_failure(monkeypatch):
    alerts = []

    async def fake_alert(name, error):
        alerts.append((name, type(error).__name__))

    monkeypatch.setattr(scheduler_daemon, "alert_job_failure", fake_alert)

    @scheduler_daemon.job("exploding job")
    async def boom():
        raise ValueError("nope")

    try:
        await boom()
    except ValueError:
        pass

    assert alerts == [("exploding job", "ValueError")]


async def test_job_wrapper_passes_results_through():
    @scheduler_daemon.job("fine job")
    async def fine():
        return 42

    assert await fine() == 42
