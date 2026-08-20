"""Phase 4 robustness - multi-window generalization and consolidated report.

This module runs the *same* evaluator (same model revision, tokenizer
revision, context length, deterministic argmax recipe, direction threshold,
no-lookahead rules and baseline definitions) over fixed chronological windows
(recent / middle / older) and assembles a consolidated report.

Nothing here is tuned on results: window placement is a fixed function of the
available data, and the same configuration drives every series and window.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Tuple

from .evaluation import EvaluationConfig, EvaluationResult, PredictionEvaluator

COMPARED_METRICS = [
    'mae_close', 'rmse_close', 'mape_close', 'directional_accuracy',
    'return_mae', 'return_rmse', 'return_correlation',
]


def _winner_labels(windows: Dict[str, EvaluationResult], metric: str) -> Dict[str, Dict[str, str]]:
    """Collect the per-window winner label for ``metric`` vs each baseline."""
    out = {'vs_persistence': {}, 'vs_previous_direction': {}}
    for name, res in windows.items():
        mc = res.report.get('model_comparison', {})
        out['vs_persistence'][name] = mc.get('kronos_vs_persistence', {}).get(
            metric + '_winner')
        out['vs_previous_direction'][name] = mc.get('kronos_vs_previous_direction', {}).get(
            metric + '_winner')
    return out


def _counts(labels: Dict[str, str]) -> Dict[str, Any]:
    values = list(labels.values())
    defined = [v for v in values if v is not None]
    return {
        'kronos': sum(1 for v in values if v == 'kronos'),
        'baseline': sum(1 for v in values if v == 'baseline'),
        'tie': sum(1 for v in values if v == 'tie'),
        'undefined': sum(1 for v in values if v is None),
        'consistent_across_windows': len(set(defined)) <= 1,
    }


def summarize_windows(windows: Dict[str, EvaluationResult]) -> Dict[str, Any]:
    """Summarize where Kronos beats / loses to each baseline across windows."""
    summary: Dict[str, Any] = {}
    for metric in COMPARED_METRICS:
        labels = _winner_labels(windows, metric)
        summary[metric] = {
            'vs_persistence_by_window': labels['vs_persistence'],
            'vs_previous_direction_by_window': labels['vs_previous_direction'],
            'vs_persistence': _counts(labels['vs_persistence']),
            'vs_previous_direction': _counts(labels['vs_previous_direction']),
        }
    return summary


def run_series_robustness(predictor, config: EvaluationConfig, symbol: str,
                          timeframe: str, candles: List[Any]) -> Dict[str, Any]:
    """Evaluate one series over recent/middle/older windows."""
    evaluator = PredictionEvaluator(predictor, config, symbol, timeframe)
    windows, window_info = evaluator.evaluate_windows(candles)
    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'window_info': window_info,
        'windows': {name: res for name, res in windows.items()},
        'summary': summarize_windows(windows),
    }


def build_consolidated_report(config: EvaluationConfig,
                              series_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble the consolidated Phase 4 robustness report."""
    overall = {'series': []}
    for series in series_reports:
        overall['series'].append({
            'symbol': series['symbol'],
            'timeframe': series['timeframe'],
            'window_info': series['window_info'],
            'summary': series['summary'],
            'windows': {
                name: res.report for name, res in series['windows'].items()
            },
        })

    # Cross-series summary: fraction of (series, window) pairs where Kronos wins.
    pair_wins = {metric: {'vs_persistence': 0, 'vs_previous_direction': 0}
                 for metric in COMPARED_METRICS}
    total_pairs = 0
    for series in series_reports:
        for metric in COMPARED_METRICS:
            s = series['summary'][metric]
            pair_wins[metric]['vs_persistence'] += s['vs_persistence']['kronos']
            pair_wins[metric]['vs_previous_direction'] += s['vs_previous_direction']['kronos']
        total_pairs += len(series['windows'])

    return {
        'kind': 'phase4_robustness',
        'generated_at_ms': int(time.time() * 1000),
        'configuration': config.asdict(),
        'compared_metrics': COMPARED_METRICS,
        'series': overall['series'],
        'across_all_series': {
            'total_series': len(series_reports),
            'total_window_evaluations': total_pairs,
            'kronos_wins_by_metric': pair_wins,
        },
        'notes': [
            'same model/tokenizer revision, context, deterministic recipe and '
            'threshold across every series and window',
            'windows are fixed and chronological (older/middle/recent); they '
            'never overlap and were not selected on performance',
            'statistical significance is NOT trading profitability',
            'no tuning, no cherry-picking, no window selection based on results',
        ],
    }


def run_robustness(predictor, config: EvaluationConfig,
                   series: List[Tuple[str, str]],
                   load_candles: Callable[[str, str], List[Any]]) -> Dict[str, Any]:
    """Run the robustness matrix over ``series`` (symbol, timeframe) pairs."""
    series_reports = []
    for symbol, timeframe in series:
        candles = load_candles(symbol, timeframe)
        series_reports.append(run_series_robustness(predictor, config, symbol,
                                                    timeframe, candles))
    return build_consolidated_report(config, series_reports)
