"""Public market context used by hourly HAR alerts."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
GLOBAL_MARKET_URL = "https://api.coingecko.com/api/v3/global"


@dataclass
class MarketContext:
    fear_greed_value: int | None
    fear_greed_label: str | None
    btc_dominance: float | None
    total_mcap_trillion: float | None
    mcap_change_24h: float | None
    fetched_at: str
    fetch_errors: list[str]

    @property
    def is_complete(self) -> bool:
        return (
            self.fear_greed_value is not None
            and self.btc_dominance is not None
            and self.total_mcap_trillion is not None
        )

    @property
    def fear_greed_emoji(self) -> str:
        if self.fear_greed_value is None:
            return ""
        v = self.fear_greed_value
        if v <= 25:
            return "🔴"
        elif v <= 45:
            return "🟠"
        elif v <= 55:
            return "🟡"
        elif v <= 75:
            return "🟢"
        else:
            return "🟢"


def fetch_fear_greed(timeout: float = 8.0) -> tuple[int | None, str | None]:
    """Fetch the current Fear and Greed value, never raising."""
    try:
        response = requests.get(FEAR_GREED_URL, timeout=timeout)
        response.raise_for_status()
        item = response.json()["data"][0]
        return int(item["value"]), str(item["value_classification"])
    except Exception as exc:  # Public context must never interrupt alerting.
        logger.warning("Fear and Greed fetch failed: %s", exc)
        return None, None


def fetch_global_market(
    timeout: float = 8.0,
) -> tuple[float | None, float | None, float | None]:
    """Fetch global crypto market figures, never raising."""
    try:
        response = requests.get(GLOBAL_MARKET_URL, timeout=timeout)
        response.raise_for_status()
        data = response.json()["data"]
        dominance = round(float(data["market_cap_percentage"]["btc"]), 1)
        mcap_trillion = round(float(data["total_market_cap"]["usd"]) / 1e12, 2)
        change = data.get("market_cap_change_percentage_24h_usd")
        change_value = None if change is None else round(float(change), 1)
        return dominance, mcap_trillion, change_value
    except Exception as exc:  # Public context must never interrupt alerting.
        logger.warning("Global market fetch failed: %s", exc)
        return None, None, None


def get_market_context(timeout: float = 8.0) -> MarketContext:
    """Fetch both context sources and always return a context object."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    errors: list[str] = []

    try:
        fear_value, fear_label = fetch_fear_greed(timeout=timeout)
    except Exception as exc:  # Also protects callers if the helper is mocked.
        logger.warning("Fear and Greed context failed: %s", exc)
        fear_value, fear_label = None, None
        errors.append(f"fear_greed: {exc}")
    if fear_value is None or fear_label is None:
        errors.append("fear_greed unavailable")

    try:
        btc, mcap, change = fetch_global_market(timeout=timeout)
    except Exception as exc:
        logger.warning("Global market context failed: %s", exc)
        btc, mcap, change = None, None, None
        errors.append(f"global_market: {exc}")
    if btc is None or mcap is None:
        errors.append("global_market unavailable")

    return MarketContext(
        fear_greed_value=fear_value,
        fear_greed_label=fear_label,
        btc_dominance=btc,
        total_mcap_trillion=mcap,
        mcap_change_24h=change,
        fetched_at=fetched_at,
        fetch_errors=errors,
    )


def format_context_section(context: MarketContext) -> str:
    """Format a complete context for insertion into a Telegram message."""
    if not context.is_complete:
        return ""
    lines = [
        "📊 Market Context",
        f"  Fear & Greed:  {context.fear_greed_value} — "
        f"{context.fear_greed_label} {context.fear_greed_emoji}",
        f"  BTC Dominance: {context.btc_dominance:.1f}%",
        f"  Global MCap:   ${context.total_mcap_trillion:.2f}T",
    ]
    if context.mcap_change_24h is not None:
        lines.append(f"  MCap 24h:      {context.mcap_change_24h:+.1f}%")
    return "\n".join(lines)
