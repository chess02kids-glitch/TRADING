from unittest.mock import Mock, patch

import requests

from kronos_trading.alerts.market_context import (
    MarketContext,
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
    with patch("kronos_trading.alerts.market_context.fetch_fear_greed", return_value=(71, "Greed")), patch("kronos_trading.alerts.market_context.fetch_global_market", return_value=(54.2, 2.41, 2.3)):
        c = get_market_context()
    assert c.is_complete and c.fetch_errors == []
    assert (c.fear_greed_value, c.btc_dominance, c.total_mcap_trillion) == (71, 54.2, 2.41)


def test_get_market_context_fear_greed_fails():
    with patch("kronos_trading.alerts.market_context.fetch_fear_greed", return_value=(None, None)), patch("kronos_trading.alerts.market_context.fetch_global_market", return_value=(54.2, 2.41, None)):
        c = get_market_context()
    assert c.fear_greed_value is None and not c.is_complete and c.fetch_errors


def test_get_market_context_global_fails():
    with patch("kronos_trading.alerts.market_context.fetch_fear_greed", return_value=(71, "Greed")), patch("kronos_trading.alerts.market_context.fetch_global_market", return_value=(None, None, None)):
        c = get_market_context()
    assert c.btc_dominance is None and not c.is_complete


def test_get_market_context_both_fail():
    with patch("kronos_trading.alerts.market_context.fetch_fear_greed", return_value=(None, None)), patch("kronos_trading.alerts.market_context.fetch_global_market", return_value=(None, None, None)):
        c = get_market_context()
    assert not c.is_complete and len(c.fetch_errors) == 2


def test_get_market_context_never_raises():
    with patch("kronos_trading.alerts.market_context.fetch_fear_greed", side_effect=RuntimeError("x")), patch("kronos_trading.alerts.market_context.fetch_global_market", side_effect=RuntimeError("y")):
        assert isinstance(get_market_context(), MarketContext)


def test_get_market_context_fetched_at_is_utc():
    with patch("kronos_trading.alerts.market_context.fetch_fear_greed", return_value=(None, None)), patch("kronos_trading.alerts.market_context.fetch_global_market", return_value=(None, None, None)):
        assert "+00:00" in get_market_context().fetched_at


def complete(**kwargs):
    values = dict(fear_greed_value=71, fear_greed_label="Greed", btc_dominance=54.2, total_mcap_trillion=2.41, mcap_change_24h=None, fetched_at="now", fetch_errors=[])
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
