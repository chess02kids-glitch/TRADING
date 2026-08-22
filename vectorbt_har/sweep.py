"""Sweep logic for finding optimal HAR stop-loss multipliers."""

import pandas as pd
from vectorbt_har.data_loader import load_ohlcv_from_feather, filter_timerange
from vectorbt_har.har_computer import compute_har_predictions, validate_predictions
from vectorbt_har.backtester import run_baseline_backtest, run_single_backtest, compute_trade_pvalue

MULTIPLIERS = [
    0.50, 0.75, 1.00, 1.25, 1.50,
    1.75, 2.00, 2.25, 2.50, 2.75, 3.00
]

def run_full_sweep(
    asset: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: str = "2024-01-01",
    end_date: str = "2026-01-01",
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Run complete multiplier sweep for one asset.
    Returns:
      - DataFrame with results
      - HAR predictions Series
      - Prediction validation report dict
    """
    df = load_ohlcv_from_feather(asset, timeframe)
    # The specification says: "Compute HAR predictions walk-forward for the ENTIRE historical dataset"
    # This must be done BEFORE filtering the timerange to avoid NaN issues at start of timerange!
    print(f"Computing HAR for {asset} on {len(df)} total bars...")
    har_preds = compute_har_predictions(df)
    
    # Filter both OHLCV and HAR to the target range
    df = filter_timerange(df, start_date, end_date)
    har_preds = har_preds.loc[df.index]
    
    report = validate_predictions(har_preds)
    
    results = []
    
    # Baseline
    print(f"Running baseline backtest for {asset}...")
    baseline_res = run_baseline_backtest(df)
    if baseline_res:
        results.append(baseline_res)
        
    # Sweep
    for m in MULTIPLIERS:
        print(f"Running backtest for {asset} multiplier {m}...")
        res = run_single_backtest(df, har_preds, m)
        if res:
            results.append(res)
            
    res_df = pd.DataFrame(results)
    return res_df, har_preds, report


def run_stability_check(
    asset: str,
    best_multiplier: float,
    timeframe: str = "1h",
) -> pd.DataFrame:
    """
    Run best multiplier on three periods.
    """
    periods = [
        ("P1 2024-01→09", "2024-01-01", "2024-09-01"),
        ("P2 2024-09→25-05", "2024-09-01", "2025-05-01"),
        ("P3 2025-05→26-01", "2025-05-01", "2026-01-01")
    ]
    
    df = load_ohlcv_from_feather(asset, timeframe)
    har_preds = compute_har_predictions(df)
    
    results = []
    for label, start, end in periods:
        period_df = filter_timerange(df, start, end)
        period_preds = har_preds.loc[period_df.index]
        
        res = run_single_backtest(period_df, period_preds, best_multiplier)
        if res:
            results.append({
                "Period": label,
                "Trades": res["total_trades"],
                "Total%": res["total_return_pct"],
                "Sharpe": res["sharpe_ratio"]
            })
            
    return pd.DataFrame(results)


def select_best_multiplier(
    results_df: pd.DataFrame,
) -> float:
    """
    Select best multiplier from sweep.
    Criteria (in order):
    1. Sharpe > 0 (positive risk-adjusted)
    2. Highest Sharpe among qualifying
    3. Minimum 30 trades
    """
    if results_df.empty:
        return 0.0
        
    # Filter multipliers only (exclude baseline)
    sweep_df = results_df[results_df["multiplier"] > 0.0]
    
    # Apply criteria
    qualifying = sweep_df[(sweep_df["sharpe_ratio"] > 0) & (sweep_df["total_trades"] >= 30)]
    
    if qualifying.empty:
        return 0.0
        
    best_row = qualifying.loc[qualifying["sharpe_ratio"].idxmax()]
    return float(best_row["multiplier"])


def evaluate_gates(
    btc_results: pd.DataFrame,
    eth_results: pd.DataFrame,
    best_mult: float,
    btc_stability: pd.DataFrame,
    eth_stability: pd.DataFrame,
    p_value: float | None,
) -> dict:
    """
    Evaluate all gate criteria.
    """
    # G1: Best Sharpe > baseline Sharpe (for both? Let's check BTC as primary or both)
    # The prompt says: "G1: Best multiplier Sharpe > baseline Sharpe"
    # We will check if the best mult Sharpe is better than baseline Sharpe for both assets.
    
    if best_mult == 0.0:
        return {
            "G1": False, "G2": False, "G3": False, "G4": False, "G5": False, "G6": False,
            "overall": "OPTION_C", "recommendation": "No qualifying multiplier."
        }
        
    btc_base = btc_results[btc_results["multiplier"] == 0.0].iloc[0]
    btc_best = btc_results[btc_results["multiplier"] == best_mult].iloc[0]
    
    eth_base = eth_results[eth_results["multiplier"] == 0.0].iloc[0]
    eth_best = eth_results[eth_results["multiplier"] == best_mult].iloc[0]
    
    g1 = (btc_best["sharpe_ratio"] > btc_base["sharpe_ratio"]) and (eth_best["sharpe_ratio"] > eth_base["sharpe_ratio"])
    g2 = (btc_best["max_drawdown_pct"] < btc_base["max_drawdown_pct"]) and (eth_best["max_drawdown_pct"] < eth_base["max_drawdown_pct"])
    g3 = (btc_best["total_trades"] >= 30) and (eth_best["total_trades"] >= 30)
    
    # G4: Time stability positive in >= 2 of 3 periods
    btc_stab_pos = sum(btc_stability["Total%"] > 0) if not btc_stability.empty else 0
    eth_stab_pos = sum(eth_stability["Total%"] > 0) if not eth_stability.empty else 0
    g4 = (btc_stab_pos >= 2) and (eth_stab_pos >= 2)
    
    # G5: works on both assets
    g5 = (btc_best["sharpe_ratio"] > 0) and (eth_best["sharpe_ratio"] > 0)
    
    # G6: p_value < 0.10
    g6 = (p_value is not None) and (p_value < 0.10)
    
    all_pass = g1 and g2 and g3 and g4 and g5 and g6
    
    overall = "PASS" if all_pass else "OPTION_C"
    
    return {
        "G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6,
        "overall": overall,
        "recommendation": "Paper trading recommended" if all_pass else "Abandon HAR stop-loss approach"
    }
