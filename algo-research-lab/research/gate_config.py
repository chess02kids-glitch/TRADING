"""
Pre-registered validation gate parameters for the algo-research-lab
Gen 1 v2 reset. DO NOT change these values without documenting the
reason in the generation report (Gen 3 may loosen exactly ONE gate,
and the change must be logged).

All gate implementations read their thresholds from this dict only
(no hardcoded parameters anywhere else).
"""

GATE_CONFIG = {
    # Gate 1 - Screening
    "screening": {
        "min_total_trades": 50,
        "min_profit_factor": 1.05,
        "max_drawdown_pct": -50.0,      # catastrophic drawdown guard
        "base_fees": 0.001,             # per side, applied to Gate 1/3 sims
        "base_slippage": 0.0005,
    },
    # Gate 2 - Walk-Forward OOS consistency
    "walk_forward": {
        "n_splits": 3,
        "train_pct": 0.60,
        "oos_sharpe_threshold": 0.0,    # mean OOS Sharpe must exceed this
        "min_positive_splits": 2,       # of n_splits
    },
    # Gate 3 - Concentration
    "concentration": {
        "max_single_trade_pct": 0.20,
        "max_top5_pct": 0.60,
        "min_total_return": 0.0,        # total trade PnL must be positive
    },
    # Gate 4 - Robustness (cost stress)
    "robustness": {
        "fee_scenarios": [0.001, 0.002, 0.003],       # per side
        "slippage_scenarios": [0.0005, 0.001, 0.002],  # per side
        "must_pass_all_scenarios": False,
        "must_pass_majority": True,
    },
    # Gate 5 - Parameter stability
    "stability": {
        "perturbation_levels": [0.10, 0.20],
        "n_perturbations_per_level": 5,
        "sharpe_degradation_threshold": 0.30,  # max allowed Sharpe drop
        "must_pass_majority": True,
    },
    # Simulation defaults
    "sim": {
        "init_cash": 10000.0,
        "freq": "1h",
    },
}
