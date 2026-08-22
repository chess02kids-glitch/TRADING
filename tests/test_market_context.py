import builtins
from unittest.mock import Mock, patch

import pandas as pd
import requests

from kronos_trading.alerts.market_context import (
    MacroContext,
    MarketContext,
    fetch_btc_options_iv,
    fetch_dxy_vix,
    fetch_fear_greed,
    fetch_global_market,
    format_context_section,
    get_market_context,
)


def response(payload):
    r = Mock()
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


def macro_ctx(
    dxy=None,
    dxy_change_1d=None,
    vix=None,
    btc_options_iv=None,
    fetch_errors=None,
):
    return MacroContext(
        dxy=dxy,
        dxy_change_1d=dxy_change_1d,
        vix=vix,
        vix_label=None,
        btc_options_iv=btc_options_iv,
        fetch_errors=fetch_errors if fetch_errors is not None else [],
    )


def macro_ok():
    return macro_ctx(dxy=104.3, dxy_change_1d=0.3, vix=14.2)


def test_fetch_fear_greed_success():
    with patch("kronos_trading.alerts.market_context.requests.get", return_value=response({"data": [{"value": "71", "value_classification": "Greed"}]})):
        assert fetch_fear_greed() == (71, "Greed")


def test_fetch_fear_greed_network_error():
    with patch("kronos_trading.alerts.market_context.requests.get", side_effect=requests.ConnectionError):
        assert fetch_fear_greed() == (None, None)


def test_fetch_fear_greed_bad_json():
    r = response({"data": "bad"})
    with patch("kronos_trading.alerts.market_context.requests.get", return_value=r):
        assert fetch_fear_greed() == (None, None)


def test_fetch_fear_greed_missing_key():
    with patch("kronos_trading.alerts.market_context.requests.get", return_value=response({})):
        assert fetch_fear_greed() == (None, None)


def test_fetch_fear_greed_timeout():
    with patch("kronos_trading.alerts.market_context.requests.get", side_effect=requests.Timeout):
        assert fetch_fear_greed() == (None, None)


def global_payload():
    return {"data": {"market_cap_percentage": {"btc": 54.23}, "total_market_cap": {"usd": 2410000000000}, "market_cap_change_percentage_24h_usd": 2.3}}


def test_fetch_global_market_success():
    with patch("kronos_trading.alerts.market_context.requests.get", return_value=response(global_payload())):
        assert fetch_global_market() == (54.2, 2.41, 2.3)


def test_fetch_global_market_network_error():
    with patch("kronos_trading.alerts.market_context.requests.get", side_effect=requests.ConnectionError):
        assert fetch_global_market() == (None, None, None)


def test_fetch_global_market_missing_btc():
    payload = global_payload(); del payload["data"]["market_cap_percentage"]["btc"]
    with patch("kronos_trading.alerts.market_context.requests.get", return_value=response(payload)):
        assert fetch_global_market() == (None, None, None)


def test_fetch_global_market_timeout():
    with patch("kronos_trading.alerts.market_context.requests.get", side_effect=requests.Timeout):
        assert fetch_global_market() == (None, None, None)


def test_get_market_context_complete():
    with patch("kronos_trading.alerts.market_context.fetch_fear_greed", return_value=(71, "Greed")), patch("kronos_trading.alerts.market_context.fetch_global_market", return_value=(54.2, 2.41, 2.3)), patch("kronos_trading.alerts.market_context.fetch_dxy_vix", return_value=macro_ok()), patch("kronos_trading.alerts.market_context.fetch_btc_options_iv", return_value=58.4):
        c = get_market_context()
    assert c.is_complete and c.fetch_errors == []
    assert (c.fear_greed_value, c.btc_dominance, c.total_mcap_trillion) == (71, 54.2, 2.41)


def test_get_market_context_fear_greed_fails():
    with patch("kronos_trading.alerts.market_context.fetch_fear_greed", return_value=(None, None)), patch("kronos_trading.alerts.market_context.fetch_global_market", return_value=(54.2, 2.41, None)), patch("kronos_trading.alerts.market_context.fetch_dxy_vix", return_value=macro_ok()), patch("kronos_trading.alerts.market_context.fetch_btc_options_iv", return_value=58.4):
        c = get_market_context()
    assert c.fear_greed_value is None and not c.is_complete and c.fetch_errors


def test_get_market_context_global_fails():
    with patch("kronos_trading.alerts.market_context.fetch_fear_greed", return_value=(71, "Greed")), patch("kronos_trading.alerts.market_context.fetch_global_market", return_value=(None, None, None)), patch("kronos_trading.alerts.market_context.fetch_dxy_vix", return_value=macro_ok()), patch("kronos_trading.alerts.market_context.fetch_btc_options_iv", return_value=58.4):
        c = get_market_context()
    assert c.btc_dominance is None and not c.is_complete


def test_get_market_context_both_fail():
    with patch("kronos_trading.alerts.market_context.fetch_fear_greed", return_value=(None, None)), patch("kronos_trading.alerts.market_context.fetch_global_market", return_value=(None, None, None)), patch("kronos_trading.alerts.market_context.fetch_dxy_vix", return_value=macro_ok()), patch("kronos_trading.alerts.market_context.fetch_btc_options_iv", return_value=58.4):
        c = get_market_context()
    assert not c.is_complete and len(c.fetch_errors) == 2


def test_get_market_context_never_raises():
    with patch("kronos_trading.alerts.market_context.fetch_fear_greed", side_effect=RuntimeError("x")), patch("kronos_trading.alerts.market_context.fetch_global_market", side_effect=RuntimeError("y")), patch("kronos_trading.alerts.market_context.fetch_dxy_vix", side_effect=RuntimeError("z")), patch("kronos_trading.alerts.market_context.fetch_btc_options_iv", side_effect=RuntimeError("w")):
        assert isinstance(get_market_context(), MarketContext)


def test_get_market_context_fetched_at_is_utc():
    with patch("kronos_trading.alerts.market_context.fetch_fear_greed", return_value=(None, None)), patch("kronos_trading.alerts.market_context.fetch_global_market", return_value=(None, None, None)), patch("kronos_trading.alerts.market_context.fetch_dxy_vix", return_value=macro_ok()), patch("kronos_trading.alerts.market_context.fetch_btc_options_iv", return_value=58.4):
        assert "+00:00" in get_market_context().fetched_at


def complete(**kwargs):
    values = dict(fear_greed_value=71, fear_greed_label="Greed", btc_dominance=54.2, total_mcap_trillion=2.41, mcap_change_24h=None, macro=None, fetched_at="now", fetch_errors=[])
    values.update(kwargs)
    return MarketContext(**values)


def test_format_context_section_complete():
    text = format_context_section(complete())
    assert all(x in text for x in ("Fear & Greed", "BTC Dominance", "Global MCap", "📊 Market Context"))


def test_format_context_section_incomplete():
    assert format_context_section(complete(btc_dominance=None)) == ""


def test_format_context_section_extreme_fear():
    assert "🔴" in format_context_section(complete(fear_greed_value=10))


def test_format_context_section_extreme_greed():
    assert "🟢" in format_context_section(complete(fear_greed_value=90))


def test_format_context_section_with_mcap_change():
    assert "+2.3%" in format_context_section(complete(mcap_change_24h=2.3))


def test_format_context_section_negative_change():
    assert "-1.5%" in format_context_section(complete(mcap_change_24h=-1.5))


def test_is_complete_true():
    assert complete().is_complete


def test_is_complete_false_missing_one():
    for field in ("fear_greed_value", "btc_dominance", "total_mcap_trillion"):
        assert not complete(**{field: None}).is_complete


# ---------------------------------------------------------------------------
# New macro tests: fetch_dxy_vix
# ---------------------------------------------------------------------------

def test_fetch_dxy_vix_success():
    dxy_df = pd.DataFrame(
        {"Close": [103.5, 104.3]},
        index=pd.to_datetime(["2026-08-20", "2026-08-21"]),
    )
    vix_df = pd.DataFrame(
        {"Close": [15.1, 14.2]},
        index=pd.to_datetime(["2026-08-20", "2026-08-21"]),
    )
    with patch("yfinance.download", side_effect=[dxy_df, vix_df]) as mock_dl:
        ctx = fetch_dxy_vix()
    assert mock_dl.call_count == 2
    assert isinstance(ctx, MacroContext)
    assert isinstance(ctx.dxy, float) and ctx.dxy == 104.3
    assert isinstance(ctx.dxy_change_1d, float) and ctx.dxy_change_1d == 0.8
    assert isinstance(ctx.vix, float) and ctx.vix == 14.2


def test_fetch_dxy_vix_yfinance_not_installed():
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yfinance":
            raise ImportError("No module named 'yfinance'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        ctx = fetch_dxy_vix()
    assert isinstance(ctx, MacroContext)
    assert ctx.dxy is None
    assert ctx.dxy_change_1d is None
    assert ctx.vix is None
    assert any("yfinance" in e for e in ctx.fetch_errors)


def test_fetch_dxy_vix_empty_dataframe():
    with patch("yfinance.download", return_value=pd.DataFrame()):
        ctx = fetch_dxy_vix()
    assert isinstance(ctx, MacroContext)
    assert ctx.dxy is None
    assert ctx.dxy_change_1d is None
    assert ctx.vix is None


def test_fetch_dxy_vix_network_error():
    with patch("yfinance.download", side_effect=RuntimeError("network down")):
        ctx = fetch_dxy_vix()
    assert isinstance(ctx, MacroContext)
    assert ctx.dxy is None
    assert ctx.dxy_change_1d is None
    assert ctx.vix is None
    assert len(ctx.fetch_errors) == 2  # one for DXY, one for VIX


# ---------------------------------------------------------------------------
# New macro tests: fetch_btc_options_iv
# ---------------------------------------------------------------------------

def test_fetch_btc_options_iv_success():
    payload = {
        "result": {
            "data": [[1699999999000, 55.1, 56.0, 54.5, 58.4]]
        }
    }
    with patch("kronos_trading.alerts.market_context.requests.get", return_value=response(payload)):
        assert fetch_btc_options_iv() == 58.4


def test_fetch_btc_options_iv_empty_candles():
    payload = {"result": {"data": []}}
    with patch("kronos_trading.alerts.market_context.requests.get", return_value=response(payload)):
        assert fetch_btc_options_iv() is None


def test_fetch_btc_options_iv_network_error():
    with patch("kronos_trading.alerts.market_context.requests.get", side_effect=requests.ConnectionError):
        assert fetch_btc_options_iv() is None


def test_fetch_btc_options_iv_bad_response():
    r = Mock()
    r.raise_for_status.return_value = None
    r.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
    with patch("kronos_trading.alerts.market_context.requests.get", return_value=r):
        assert fetch_btc_options_iv() is None


# ---------------------------------------------------------------------------
# New macro tests: get_market_context integration
# ---------------------------------------------------------------------------

def test_get_market_context_includes_macro():
    with patch("kronos_trading.alerts.market_context.fetch_fear_greed", return_value=(71, "Greed")), patch("kronos_trading.alerts.market_context.fetch_global_market", return_value=(54.2, 2.41, 2.3)), patch("kronos_trading.alerts.market_context.fetch_dxy_vix", return_value=macro_ok()), patch("kronos_trading.alerts.market_context.fetch_btc_options_iv", return_value=58.4):
        c = get_market_context()
    assert c.macro is not None
    assert c.macro.dxy is not None and c.macro.dxy == 104.3
    assert c.macro.vix is not None and c.macro.vix == 14.2
    assert c.macro.btc_options_iv is not None and c.macro.btc_options_iv == 58.4


def test_get_market_context_macro_failure_ok():
    failed = macro_ctx(fetch_errors=["DXY: boom", "VIX: boom"])
    with patch("kronos_trading.alerts.market_context.fetch_fear_greed", return_value=(71, "Greed")), patch("kronos_trading.alerts.market_context.fetch_global_market", return_value=(54.2, 2.41, 2.3)), patch("kronos_trading.alerts.market_context.fetch_dxy_vix", return_value=failed), patch("kronos_trading.alerts.market_context.fetch_btc_options_iv", return_value=None):
        c = get_market_context()
    assert isinstance(c, MarketContext)
    assert c.is_complete
    assert c.fear_greed_value == 71
    assert c.btc_dominance == 54.2
    assert c.total_mcap_trillion == 2.41
    assert c.macro is not None
    assert c.macro.dxy is None
    assert c.macro.vix is None
    assert c.macro.btc_options_iv is None


def test_get_market_context_partial_macro():
    partial = macro_ctx(dxy=104.3, dxy_change_1d=0.2)  # VIX fails
    with patch("kronos_trading.alerts.market_context.fetch_fear_greed", return_value=(71, "Greed")), patch("kronos_trading.alerts.market_context.fetch_global_market", return_value=(54.2, 2.41, 2.3)), patch("kronos_trading.alerts.market_context.fetch_dxy_vix", return_value=partial), patch("kronos_trading.alerts.market_context.fetch_btc_options_iv", return_value=58.4):
        c = get_market_context()
    assert c.macro is not None
    assert c.macro.dxy is not None
    assert c.macro.vix is None
    assert c.macro.btc_options_iv is not None


# ---------------------------------------------------------------------------
# New macro tests: format_context_section
# ---------------------------------------------------------------------------

def test_format_includes_dxy_when_available():
    text = format_context_section(complete(macro=macro_ctx(dxy=104.3, dxy_change_1d=0.3)))
    assert "104.3" in text
    assert "strengthening" in text


def test_format_includes_vix_when_available():
    text = format_context_section(complete(macro=macro_ctx(vix=14.2)))
    assert "14.2" in text
    assert "low fear" in text


def test_format_includes_btc_iv_when_available():
    text = format_context_section(complete(macro=macro_ctx(btc_options_iv=58.4)))
    assert "58.4%" in text
    assert "Options IV" in text


def test_format_omits_dxy_when_none():
    text = format_context_section(complete(macro=macro_ctx(dxy=None, vix=14.2, btc_options_iv=58.4)))
    assert "DXY" not in text
    assert "VIX" in text and "58.4%" in text


def test_format_omits_vix_when_none():
    text = format_context_section(complete(macro=macro_ctx(dxy=104.3, dxy_change_1d=0.3, btc_options_iv=58.4)))
    assert "VIX" not in text
    assert "104.3" in text and "58.4%" in text


def test_format_omits_iv_when_none():
    text = format_context_section(complete(macro=macro_ctx(dxy=104.3, dxy_change_1d=0.3, vix=14.2)))
    assert "Options IV" not in text
    assert "104.3" in text and "14.2" in text


def test_format_no_macro_field():
    text = format_context_section(complete(macro=None))
    assert "DXY" not in text
    assert "VIX" not in text
    assert "Options IV" not in text
    assert "📊 Market Context" in text


# ---------------------------------------------------------------------------
# New macro tests: MacroContext properties
# ---------------------------------------------------------------------------

def test_dxy_direction_strengthening():
    assert macro_ctx(dxy_change_1d=0.5).dxy_direction == "↑ strengthening"


def test_dxy_direction_weakening():
    assert macro_ctx(dxy_change_1d=-0.5).dxy_direction == "↓ weakening"


def test_dxy_direction_flat():
    assert macro_ctx(dxy_change_1d=0.05).dxy_direction == "→ flat"


def test_vix_label_low_fear():
    assert macro_ctx(vix=12.0).vix_label_auto == "low fear"


def test_vix_label_moderate():
    assert macro_ctx(vix=20.0).vix_label_auto == "moderate"


def test_vix_label_elevated():
    assert macro_ctx(vix=30.0).vix_label_auto == "elevated"


def test_vix_label_extreme():
    assert macro_ctx(vix=40.0).vix_label_auto == "extreme fear"
