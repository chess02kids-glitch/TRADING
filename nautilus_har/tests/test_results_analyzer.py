import pytest
from nautilus_har.backtest.results_analyzer import evaluate_gates

def test_gates_b_beats_a():
    res_a = {"sharpe_ratio": 1.0, "max_drawdown_pct": -20.0, "volatility_pct": 15.0}
    res_b = {"sharpe_ratio": 2.0, "max_drawdown_pct": -10.0, "volatility_pct": 10.0, "p_value": 0.05}
    res_c = {"sharpe_ratio": 0.5, "max_drawdown_pct": -30.0, "volatility_pct": 20.0}
    gates = evaluate_gates(res_a, res_b, res_c)
    assert gates["G1"] == True

def test_gates_b_worse_than_a():
    res_a = {"sharpe_ratio": 1.0, "max_drawdown_pct": -20.0, "volatility_pct": 15.0}
    res_b = {"sharpe_ratio": 0.5, "max_drawdown_pct": -30.0, "volatility_pct": 20.0, "p_value": 0.20}
    res_c = {"sharpe_ratio": 0.5, "max_drawdown_pct": -30.0, "volatility_pct": 20.0}
    gates = evaluate_gates(res_a, res_b, res_c)
    assert gates["G1"] == False

def test_gates_c_loses_to_a():
    res_a = {"sharpe_ratio": 1.0, "max_drawdown_pct": -20.0, "volatility_pct": 15.0}
    res_b = {"sharpe_ratio": 2.0, "max_drawdown_pct": -10.0, "volatility_pct": 10.0, "p_value": 0.05}
    res_c = {"sharpe_ratio": 0.5, "max_drawdown_pct": -30.0, "volatility_pct": 20.0}
    gates = evaluate_gates(res_a, res_b, res_c)
    assert gates["G4"] == True

def test_evaluate_gates_all_pass():
    res_a = {"sharpe_ratio": 1.0, "max_drawdown_pct": -20.0, "volatility_pct": 15.0}
    res_b = {"sharpe_ratio": 2.0, "max_drawdown_pct": -10.0, "volatility_pct": 10.0, "p_value": 0.05}
    res_c = {"sharpe_ratio": 0.5, "max_drawdown_pct": -30.0, "volatility_pct": 20.0}
    stability = [{"total_return_pct": 5.0}, {"total_return_pct": 2.0}, {"total_return_pct": 10.0}]
    gates = evaluate_gates(res_a, res_b, res_c, stability)
    assert gates["overall"] == "PASS"

def test_evaluate_gates_one_fail():
    res_a = {"sharpe_ratio": 1.0, "max_drawdown_pct": -20.0, "volatility_pct": 15.0}
    # fail G1 (B < A)
    res_b = {"sharpe_ratio": 0.5, "max_drawdown_pct": -10.0, "volatility_pct": 10.0, "p_value": 0.05}
    res_c = {"sharpe_ratio": 0.5, "max_drawdown_pct": -30.0, "volatility_pct": 20.0}
    stability = [{"total_return_pct": 5.0}, {"total_return_pct": 2.0}, {"total_return_pct": 10.0}]
    gates = evaluate_gates(res_a, res_b, res_c, stability)
    assert gates["overall"] == "FAIL"
