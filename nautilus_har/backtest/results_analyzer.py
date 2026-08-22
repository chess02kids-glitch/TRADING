"""
Results analysis and reporting.
"""

import pandas as pd
from datetime import datetime, timezone


def build_comparison_table(
    results_a: dict,
    results_b: dict,
    results_c: dict,
) -> str:
    """
    Build markdown comparison table.
    """
    metrics = [
        ("total_return_pct", "Total Return %"),
        ("annualized_return_pct",
         "Annualized Return %"),
        ("sharpe_ratio", "Sharpe Ratio"),
        ("sortino_ratio", "Sortino Ratio"),
        ("max_drawdown_pct", "Max Drawdown %"),
        ("volatility_pct", "Annual Vol %"),
        ("calmar_ratio", "Calmar Ratio"),
        ("p_value", "p-value"),
    ]

    header = (
        "| Metric | A (Equal Weight) |"
        " B (HAR Targeting) |"
        " C (Inverse HAR) |\n"
        "|---|---|---|---|\n")

    rows = []
    for key, label in metrics:
        a_val = results_a.get(key, "N/A")
        b_val = results_b.get(key, "N/A")
        c_val = results_c.get(key, "N/A")
        rows.append(
            f"| {label} | {a_val} |"
            f" {b_val} | {c_val} |")

    return header + "\n".join(rows)


def build_stability_table(stability_results: list[dict]) -> str:
    from nautilus_har.config import STABILITY_PERIODS
    
    header = (
        "| Period | Start | End | Total Return % | Sharpe | Max DD % |\n"
        "|---|---|---|---|---|---|\n"
    )
    
    rows = []
    for i, (start, end) in enumerate(STABILITY_PERIODS):
        res = stability_results[i] if i < len(stability_results) else {}
        ret = res.get("total_return_pct", "N/A")
        sharpe = res.get("sharpe_ratio", "N/A")
        dd = res.get("max_drawdown_pct", "N/A")
        
        rows.append(f"| P{i+1} | {start} | {end} | {ret} | {sharpe} | {dd} |")
        
    return header + "\n".join(rows)

def evaluate_gates(
    results_a: dict,
    results_b: dict,
    results_c: dict,
    stability_results: list[dict] = None,
) -> dict:
    """
    Evaluate all gate criteria.
    """
    gates = {}

    # G1: B Sharpe > A Sharpe
    gates["G1"] = (
        results_b.get("sharpe_ratio", -999)
        > results_a.get("sharpe_ratio", -999))

    # G2: B max DD < A max DD
    # (less negative = better)
    gates["G2"] = (
        results_b.get("max_drawdown_pct", -999)
        > results_a.get("max_drawdown_pct", -999))

    # G3: B vol < A vol
    gates["G3"] = (
        results_b.get("volatility_pct", 999)
        < results_a.get("volatility_pct", 999))

    # G4: C Sharpe < A Sharpe
    gates["G4"] = (
        results_c.get("sharpe_ratio", 999)
        < results_a.get("sharpe_ratio", 999))

    # G5: Time stability
    if stability_results:
        positive_periods = sum(
            1 for r in stability_results
            if r.get("total_return_pct", -1) > 0)
        gates["G5"] = (
            positive_periods
            >= len(stability_results) * 0.67)
    else:
        gates["G5"] = None

    # G6: p-value < 0.10
    gates["G6"] = (
        results_b.get("p_value", 1.0) < 0.10)

    all_pass = all(
        v is True
        for v in gates.values()
        if v is not None)

    return {
        **gates,
        "overall": "PASS" if all_pass else "FAIL",
    }


def write_results_md(
    results_a: dict,
    results_b: dict,
    results_c: dict,
    stability_b: list[dict],
    gates: dict,
    output_path: str = (
        "nautilus_har/NAUTILUS_RESULTS.md"),
) -> None:
    """Write complete results document."""
    now = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC")

    from nautilus_har.config import (
        TARGET_VOL_PER_BAR, MIN_ALLOCATION,
        MAX_ALLOCATION, REBALANCE_THRESHOLD,
        FEES_BPS)

    content = f"""# NautilusTrader HAR Volatility Targeting Results

Generated: {now}
Research context: HAR validated (p<1e-26),
testing volatility targeting as portfolio
risk management technique.

## Research Question

Does using HAR predicted range for portfolio
position sizing (volatility targeting) improve
risk-adjusted returns vs equal-weight?

## Strategy Specifications

Strategy A: Equal Weight
  50% BTC, 50% ETH
  Rebalance daily (>5% drift)
  Baseline comparison

Strategy B: HAR Volatility Targeting
  allocation = target_vol / vol_estimate
  vol_estimate = HAR_range / price
  target_vol = {TARGET_VOL_PER_BAR}
  Clip: [{MIN_ALLOCATION}, {MAX_ALLOCATION}]

Strategy C: Inverse HAR (Control)
  allocation = vol_estimate / target_vol
  Opposite of Strategy B

Pre-registered parameters (not tuned):
  target_vol = {TARGET_VOL_PER_BAR}
  min_allocation = {MIN_ALLOCATION}
  max_allocation = {MAX_ALLOCATION}
  rebalance_threshold = {REBALANCE_THRESHOLD}
  fees = {FEES_BPS} bps

## Results

{build_comparison_table(
    results_a, results_b, results_c)}

## Time Stability (Strategy B)

{build_stability_table(stability_b)}

## Gate Criteria

G1 (B Sharpe > A): {'PASS' if gates.get('G1') else 'FAIL'}
G2 (B DD < A):     {'PASS' if gates.get('G2') else 'FAIL'}
G3 (B Vol < A):    {'PASS' if gates.get('G3') else 'FAIL'}
G4 (C < A):        {'PASS' if gates.get('G4') else 'FAIL'}
G5 (Stability):    {'PASS' if gates.get('G5') else 'FAIL' if gates.get('G5') is not None else 'PENDING'}
G6 (p<0.10):       {'PASS' if gates.get('G6') else 'FAIL'}

Overall: {gates.get('overall', 'UNKNOWN')}

## Recommendation

{'Paper trade Strategy B after day 30 if HAR passes calibration.' if gates.get('overall') == 'PASS' else 'Volatility targeting did not demonstrate significant improvement. Consider alternative approaches.'}

## Statistical Integrity

Parameters pre-registered.
Single timerange used.
No parameters changed after results.
Paper trading only. No real orders.
"""

    with open(output_path, "w") as f:
        f.write(content)

    print(f"Results written to {output_path}")
