"""Minimal alerting: structured logs plus an optional webhook.

The docs promised email/Slack/PagerDuty integration that did not exist. Rather
than pretend, this ships one honest hook: everything is logged, and if
``ALERT_WEBHOOK_URL`` is configured the same message is POSTed as a
Slack-compatible JSON payload. Delivery failures never propagate -- an alert
that cannot be sent must not take down the job that raised it.
"""

import logging
from typing import Any, Dict, Optional

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

SEVERITIES = {"info": logging.INFO, "warning": logging.WARNING, "critical": logging.ERROR}


def _format(title: str, message: str, context: Optional[Dict[str, Any]]) -> str:
    if not context:
        return f"{title}: {message}"
    details = " ".join(f"{k}={v}" for k, v in context.items())
    return f"{title}: {message} ({details})"


async def send_alert(
    title: str,
    message: str,
    severity: str = "warning",
    context: Optional[Dict[str, Any]] = None,
) -> bool:
    """Emit an alert. Returns True if a webhook delivery succeeded."""
    text = _format(title, message, context)
    logger.log(SEVERITIES.get(severity, logging.WARNING), text)

    if not settings.alert_webhook_url:
        return False

    payload = {
        "text": f"[{severity.upper()}] {text}",
        "severity": severity,
        "title": title,
        "message": message,
        "context": context or {},
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(settings.alert_webhook_url, json=payload)
            response.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"Alert webhook delivery failed: {e}")
        return False


async def alert_job_failure(job_name: str, error: BaseException) -> None:
    """Report a scheduled job that raised."""
    await send_alert(
        title="Scheduled job failed",
        message=f"{job_name} raised {type(error).__name__}: {error}",
        severity="critical",
        context={"job": job_name},
    )
