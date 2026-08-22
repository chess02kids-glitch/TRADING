"""Public market context used by hourly HAR alerts."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
GLOBAL_MARKET_URL = "https://api.coingecko.com/api/v3/global"

# Deribit public API (no auth) — BTC DVOL (30-day implied volatility index).
DERIBIT_VOL_INDEX_URL = (
    "https://www.deribit.com/api/v2/public/get_volatility_index_data"
)

# Yahoo Finance tickers used for macro context (no API key required).
DXY_TICKER = "DX-Y.NYB"
VIX_TICKER = "^VIX"


@dataclass
class MacroContext:
    """Optional macro-market enrichment: DXY, VIX and BTC options IV.

    Every field is optional. A missing field only means that data point
    is omitted from the alert — it never blocks alerting.
    """

    dxy: float | None
    dxy_change_1d: float | None
    vix: float | None
    vix_label: str | None
    btc_options_iv: float | None
    fetch_errors: list[str]

    @property
    def dxy_direction(self) -> str:
        """Arrow showing DXY trend."""
        if self.dxy_change_1d is None:
            return ""
        if self.dxy_change_1d > 0.1:
            return "↑ strengthening"
        elif self.dxy_change_1d < -0.1:
            return "↓ weakening"
        else:
            return "→ flat"

    @property
    def vix_label_auto(self) -> str:
        """Auto-classify VIX level."""
        if self.vix is None:
            return ""
        if self.vix < 15:
            return "low fear"
        elif self.vix < 25:
            return "moderate"
        elif self.vix < 35:
            return "elevated"
        else:
            return "extreme fear"


@dataclass
class MarketContext:
    fear_greed_value: int | None
    fear_greed_label: str | None
    btc_dominance: float | None
    total_mcap_trillion: float | None
    mcap_change_24h: float | None
    macro: MacroContext | None = None  # Optional enrichment; never required.
    fetched_at: str = ""
    fetch_errors: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        # Note: macro is optional — is_complete does NOT require macro.
        # Macro failure never blocks the alert.
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


def fetch_dxy_vix(
    timeout: float = 10.0,
) -> MacroContext:
    """
    Fetch DXY and VIX from Yahoo Finance via yfinance.

    Returns MacroContext with whatever succeeded. Never raises.

    DXY ticker: DX-Y.NYB
    VIX ticker: ^VIX

    For each ticker:
    - Download last days of daily data
    - Latest close = current value
    - 1d change = latest close - previous close
    """
    dxy = None
    dxy_change = None
    vix = None
    errors = []

    try:
        import yfinance as yf

        # Fetch DXY
        try:
            dxy_data = yf.download(
                DXY_TICKER,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                timeout=timeout,
            )
            if dxy_data is not None and len(dxy_data) >= 2:
                closes = dxy_data["Close"].dropna()
                if len(closes) >= 2:
                    dxy = round(float(closes.iloc[-1]), 2)
                    dxy_change = round(
                        float(closes.iloc[-1]) - float(closes.iloc[-2]), 2
                    )
        except Exception as e:
            errors.append(f"DXY: {e}")
            logger.warning("DXY fetch failed: %s", e)

        # Fetch VIX
        try:
            vix_data = yf.download(
                VIX_TICKER,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                timeout=timeout,
            )
            if vix_data is not None and len(vix_data) >= 1:
                closes = vix_data["Close"].dropna()
                if len(closes) >= 1:
                    vix = round(float(closes.iloc[-1]), 2)
        except Exception as e:
            errors.append(f"VIX: {e}")
            logger.warning("VIX fetch failed: %s", e)

    except ImportError:
        errors.append("yfinance not installed")
        logger.warning("yfinance not installed — DXY/VIX unavailable")

    return MacroContext(
        dxy=dxy,
        dxy_change_1d=dxy_change,
        vix=vix,
        vix_label=None,
        btc_options_iv=None,
        fetch_errors=errors,
    )


def fetch_btc_options_iv(
    timeout: float = 10.0,
) -> float | None:
    """
    Fetch BTC options implied volatility from Deribit public API.

    Uses the DVOL index (Deribit Volatility). This is the 30-day implied
    vol for BTC options. No authentication required.

    Returns annualized IV as a percentage (e.g. 58.4 means 58.4%
    annualized IV). Returns None on any failure.
    """
    try:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        # Get the last few hours of hourly data.
        start_ms = now_ms - (2 * 3600 * 1000)

        response = requests.get(
            DERIBIT_VOL_INDEX_URL,
            params={
                "currency": "BTC",
                "start_timestamp": start_ms,
                "end_timestamp": now_ms,
                "resolution": "3600",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        # Response: {"result": {"data": [[ts, o, h, l, c], ...]}}
        result = data.get("result", {})
        candles = result.get("data", [])

        if candles:
            # Last candle close value
            latest_close = candles[-1][4]
            return round(float(latest_close), 2)

        return None

    except Exception as e:
        logger.warning("BTC options IV fetch failed: %s", e)
        return None


def get_market_context(timeout: float = 8.0) -> MarketContext:
    """Fetch all context sources and always return a context object.

    Macro data (DXY/VIX/options IV) is optional enrichment only: each
    sub-fetch has its own error handling and a failure never blocks the
    alert or affects `is_complete`.
    """
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

    # Macro context (new) — optional enrichment. Each sub-fetch has its
    # own error handling; macro failures never touch the main errors list
    # and never block the alert.
    try:
        macro = fetch_dxy_vix(timeout=timeout)
    except Exception as exc:  # Also protects callers if the helper is mocked.
        logger.warning("DXY/VIX context failed: %s", exc)
        macro = MacroContext(
            dxy=None,
            dxy_change_1d=None,
            vix=None,
            vix_label=None,
            btc_options_iv=None,
            fetch_errors=[f"dxy_vix: {exc}"],
        )

    try:
        btc_iv = fetch_btc_options_iv(timeout=timeout)
    except Exception as exc:  # Also protects callers if the helper is mocked.
        logger.warning("BTC options IV context failed: %s", exc)
        btc_iv = None
        macro.fetch_errors.append(f"btc_options_iv: {exc}")

    macro = MacroContext(
        dxy=macro.dxy,
        dxy_change_1d=macro.dxy_change_1d,
        vix=macro.vix,
        vix_label=macro.vix_label,
        btc_options_iv=btc_iv,
        fetch_errors=macro.fetch_errors,
    )

    return MarketContext(
        fear_greed_value=fear_value,
        fear_greed_label=fear_label,
        btc_dominance=btc,
        total_mcap_trillion=mcap,
        mcap_change_24h=change,
        macro=macro,
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

    # Macro lines are optional enrichment: show each one only when the
    # data point is available, and never crash when macro is None.
    macro = context.macro
    if macro is not None:
        if macro.dxy is not None:
            dxy_line = f"  DXY:           {macro.dxy:.1f}"
            if macro.dxy_change_1d is not None:
                dxy_line += f" ({macro.dxy_direction})"
            lines.append(dxy_line)
        if macro.vix is not None:
            lines.append(
                f"  VIX:           {macro.vix:.1f}  "
                f"({macro.vix_label_auto})"
            )
        if macro.btc_options_iv is not None:
            lines.append(
                f"  BTC Options IV: {macro.btc_options_iv:.1f}%"
            )
    return "\n".join(lines)
