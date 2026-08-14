"""Paper-only CLI.

``predict`` runs the *real* Kronos model by default. The offline mock predictor
is used only when the explicit ``--mock`` flag is passed - it is never selected
automatically, and a missing model is never silently replaced with mock data.
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

from .types import Candle
from .model import (ModelManager, ModelUnavailableError,
                    DeterministicMockPredictor, KronosRealPredictor,
                    REFERENCE_MODEL_REVISION, REFERENCE_TOKENIZER_REVISION)
from .pipeline import PredictionPipeline
from .backtest import Backtester
from .benchmark import measure_model_load, run_benchmark
from .evaluation import (EvaluationConfig, PredictionEvaluator, parse_timestamp)
from .robustness import run_robustness
from .research_targets import run_research_experiment
from .reference_validation import build_validation_report
from .volatility_research import run_volatility_research
from .classical_volatility import (run_classical_volatility_benchmark,
                                   recompute_classical_gate,
                                   recompute_classical_summary)
from .ml_volatility import run_ml_vs_har
from .cross_asset import run_cross_asset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / 'data' / 'db' / 'kronos_trading_verified.db'


def load_candles(db, symbol, timeframe, exchange='binance'):
    path = Path(db)
    if not path.exists():
        raise FileNotFoundError(
            'database not found: %s (verified DB is expected at %s)'
            % (path, DEFAULT_DB))
    conn = sqlite3.connect('file:%s?mode=ro' % path, uri=True)
    try:
        rows = conn.execute(
            'SELECT timestamp_ms, open, high, low, close, volume '
            'FROM ohlcv_raw WHERE exchange=? AND symbol=? AND timeframe=? '
            'ORDER BY timestamp_ms', (exchange, symbol, timeframe)).fetchall()
    finally:
        conn.close()
    if not rows:
        raise ValueError('no candles found for %s %s in %s' % (symbol, timeframe, path))
    return [Candle(*r) for r in rows]


def _build_pipeline(args):
    if args.mock:
        return PredictionPipeline(DeterministicMockPredictor())
    manager = ModelManager(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        device=args.device or None,
        max_context=args.max_context,
        cache_dir=args.cache_dir or None,
    ).load()
    if not manager.available:
        raise ModelUnavailableError(
            'real Kronos model is unavailable: %s\n'
            '(use --mock only for offline unit testing; real inference '
            'requires the Kronos-small weights and tokenizer)' % manager.error)
    return PredictionPipeline(KronosRealPredictor(manager))


def _predict(args):
    candles = load_candles(args.db, args.symbol, args.timeframe)
    pipe = _build_pipeline(args)
    pred = pipe.predict(
        args.symbol, args.timeframe, candles,
        args.context, args.horizon, int(time.time() * 1000),
        seed=args.seed, deterministic=args.deterministic,
    )
    print(json.dumps(pred.asdict(), indent=2, default=str))


def _backtest(args):
    candles = load_candles(args.db, args.symbol, args.timeframe)
    pipe = _build_pipeline(args)
    out = Backtester(pipe).run(args.symbol, args.timeframe, candles, args.context)
    print(json.dumps(out, indent=2, default=str))


def main(argv=None):
    p = argparse.ArgumentParser(prog='kronos-trading')
    sub = p.add_subparsers(dest='cmd', required=True)

    for name in ('predict', 'backtest'):
        q = sub.add_parser(name)
        q.add_argument('--db', default=str(DEFAULT_DB))
        q.add_argument('--symbol', default='BTC/USDT')
        q.add_argument('--timeframe', default='1h')
        q.add_argument('--context', type=int, default=512)
        q.add_argument('--mock', action='store_true',
                       help='use the offline deterministic mock (never automatic)')

    pred = sub.choices['predict']
    pred.add_argument('--horizon', type=int, default=1)
    pred.add_argument('--model', default='NeoQuasar/Kronos-small')
    pred.add_argument('--tokenizer', default='NeoQuasar/Kronos-Tokenizer-base')
    pred.add_argument('--model-revision', default=REFERENCE_MODEL_REVISION)
    pred.add_argument('--tokenizer-revision', default=REFERENCE_TOKENIZER_REVISION)
    pred.add_argument('--device', default=None,
                      help="'cpu' to force CPU, 'cuda:0' for GPU, omit for auto")
    pred.add_argument('--max-context', type=int, default=512)
    pred.add_argument('--cache-dir', default=None,
                      help='HF cache dir / local models directory (default: HF default)')
    pred.add_argument('--seed', type=int, default=None)
    pred.add_argument('--deterministic', action='store_true',
                      help='seeded argmax sampling for reproducible output')

    bench = sub.add_parser('benchmark')
    bench.add_argument('--db', default=str(DEFAULT_DB))
    bench.add_argument('--symbol', default='BTC/USDT')
    bench.add_argument('--timeframe', default='1h')
    bench.add_argument('--context', type=int, default=512)
    bench.add_argument('--horizon', type=int, default=1)
    bench.add_argument('--model', default='NeoQuasar/Kronos-small')
    bench.add_argument('--tokenizer', default='NeoQuasar/Kronos-Tokenizer-base')
    bench.add_argument('--model-revision', default=REFERENCE_MODEL_REVISION)
    bench.add_argument('--tokenizer-revision', default=REFERENCE_TOKENIZER_REVISION)
    bench.add_argument('--device', default=None)
    bench.add_argument('--max-context', type=int, default=512)
    bench.add_argument('--cache-dir', default=None)
    bench.add_argument('--warmed-runs', type=int, default=10)
    bench.add_argument('--seed', type=int, default=123)
    bench.add_argument('--no-deterministic', action='store_true',
                       help='allow stochastic sampling during the benchmark')

    ev = sub.add_parser('evaluate',
                        help='chronological no-lookahead evaluation of real Kronos')
    ev.add_argument('--db', default=str(DEFAULT_DB))
    ev.add_argument('--symbol', default='BTC/USDT')
    ev.add_argument('--timeframe', default='1h')
    ev.add_argument('--context', type=int, default=512)
    ev.add_argument('--horizon', type=int, default=1)
    ev.add_argument('--start', default=None,
                    help='evaluation window start (ISO-8601 UTC or epoch ms)')
    ev.add_argument('--end', default=None,
                    help='evaluation window end (ISO-8601 UTC or epoch ms)')
    ev.add_argument('--max-predictions', type=int, default=1000,
                    help='cap for the default (recent) window; ignored with --start/--end')
    ev.add_argument('--direction-threshold', type=float, default=0.0005,
                    help='|return| below this is treated as a flat move')
    ev.add_argument('--seed', type=int, default=0)
    ev.add_argument('--no-deterministic', action='store_true',
                    help='DISABLE the deterministic argmax recipe (loud opt-out)')
    ev.add_argument('--model', default='NeoQuasar/Kronos-small')
    ev.add_argument('--tokenizer', default='NeoQuasar/Kronos-Tokenizer-base')
    ev.add_argument('--model-revision', default=REFERENCE_MODEL_REVISION)
    ev.add_argument('--tokenizer-revision', default=REFERENCE_TOKENIZER_REVISION)
    ev.add_argument('--device', default=None)
    ev.add_argument('--max-context', type=int, default=512)
    ev.add_argument('--cache-dir', default=None)
    ev.add_argument('--output', default=None,
                    help='JSON output path (default: data/eval/<symbol>_<tf>_<start>.json)')
    ev.add_argument('--include-rows', action='store_true',
                    help='also print per-prediction rows to stdout')

    rb = sub.add_parser('robustness',
                        help='multi-window generalization evaluation across series')
    rb.add_argument('--db', default=str(DEFAULT_DB))
    rb.add_argument('--assets', nargs='+', default=['BTC/USDT', 'ETH/USDT'])
    rb.add_argument('--timeframes', nargs='+', default=['1h', '4h', '1d'])
    rb.add_argument('--context', type=int, default=512)
    rb.add_argument('--horizon', type=int, default=1)
    rb.add_argument('--window-size', type=int, default=1000,
                    help='targets per chronological window (recent/middle/older)')
    rb.add_argument('--direction-threshold', type=float, default=0.0005)
    rb.add_argument('--seed', type=int, default=0)
    rb.add_argument('--no-deterministic', action='store_true')
    rb.add_argument('--model', default='NeoQuasar/Kronos-small')
    rb.add_argument('--tokenizer', default='NeoQuasar/Kronos-Tokenizer-base')
    rb.add_argument('--model-revision', default=REFERENCE_MODEL_REVISION)
    rb.add_argument('--tokenizer-revision', default=REFERENCE_TOKENIZER_REVISION)
    rb.add_argument('--device', default=None)
    rb.add_argument('--max-context', type=int, default=512)
    rb.add_argument('--cache-dir', default=None)
    rb.add_argument('--output', default=None,
                    help='JSON output path (default: data/eval/robustness_report.json)')

    rt = sub.add_parser('research-targets',
                        help='Phase 5 target-formulation research (same model, '
                             'different derived targets)')
    rt.add_argument('--db', default=str(DEFAULT_DB))
    rt.add_argument('--assets', nargs='+', default=['BTC/USDT', 'ETH/USDT'])
    rt.add_argument('--timeframes', nargs='+', default=['1h', '4h', '1d'])
    rt.add_argument('--context', type=int, default=512)
    rt.add_argument('--window-size', type=int, default=1000)
    rt.add_argument('--direction-threshold', type=float, default=0.0005)
    rt.add_argument('--seed', type=int, default=0)
    rt.add_argument('--no-deterministic', action='store_true')
    rt.add_argument('--model', default='NeoQuasar/Kronos-small')
    rt.add_argument('--tokenizer', default='NeoQuasar/Kronos-Tokenizer-base')
    rt.add_argument('--model-revision', default=REFERENCE_MODEL_REVISION)
    rt.add_argument('--tokenizer-revision', default=REFERENCE_TOKENIZER_REVISION)
    rt.add_argument('--device', default=None)
    rt.add_argument('--max-context', type=int, default=512)
    rt.add_argument('--cache-dir', default=None)
    rt.add_argument('--output', default=None,
                    help='JSON output path (default: data/eval/research_targets_report.json)')

    rv = sub.add_parser('validate-reference',
                        help='validate our pipeline against the upstream Kronos '
                             'reference (regression test + fixtures)')
    rv.add_argument('--context', type=int, default=512)
    rv.add_argument('--pred-len', type=int, default=8)
    rv.add_argument('--model', default='NeoQuasar/Kronos-small')
    rv.add_argument('--tokenizer', default='NeoQuasar/Kronos-Tokenizer-base')
    rv.add_argument('--model-revision', default=REFERENCE_MODEL_REVISION)
    rv.add_argument('--tokenizer-revision', default=REFERENCE_TOKENIZER_REVISION)
    rv.add_argument('--device', default=None)
    rv.add_argument('--cache-dir', default=None)
    rv.add_argument('--output', default=None,
                    help='JSON output path (default: data/eval/reference_validation_report.json)')

    vol = sub.add_parser('volatility-research',
                         help='Phase 5b: does Kronos have real volatility skill '
                              'vs strong baselines?')
    vol.add_argument('--db', default=str(DEFAULT_DB))
    vol.add_argument('--assets', nargs='+', default=['BTC/USDT', 'ETH/USDT'])
    vol.add_argument('--timeframes', nargs='+', default=['1h', '4h', '1d'])
    vol.add_argument('--context', type=int, default=512)
    vol.add_argument('--window-size', type=int, default=1000)
    vol.add_argument('--seed', type=int, default=0)
    vol.add_argument('--no-deterministic', action='store_true')
    vol.add_argument('--model', default='NeoQuasar/Kronos-small')
    vol.add_argument('--tokenizer', default='NeoQuasar/Kronos-Tokenizer-base')
    vol.add_argument('--model-revision', default=REFERENCE_MODEL_REVISION)
    vol.add_argument('--tokenizer-revision', default=REFERENCE_TOKENIZER_REVISION)
    vol.add_argument('--device', default=None)
    vol.add_argument('--max-context', type=int, default=512)
    vol.add_argument('--cache-dir', default=None)
    vol.add_argument('--output', default=None,
                    help='JSON output path (default: data/eval/volatility_research_report.json)')

    cv = sub.add_parser('classical-volatility',
                        help='Phase 5c: classical volatility benchmark '
                             '(HAR as primary; Kronos as challenger)')
    cv.add_argument('--db', default=str(DEFAULT_DB))
    cv.add_argument('--assets', nargs='+', default=['BTC/USDT', 'ETH/USDT'])
    cv.add_argument('--timeframes', nargs='+', default=['1h', '4h', '1d'])
    cv.add_argument('--context', type=int, default=512)
    cv.add_argument('--window-size', type=int, default=1000)
    cv.add_argument('--seed', type=int, default=0)
    cv.add_argument('--no-deterministic', action='store_true')
    cv.add_argument('--model', default='NeoQuasar/Kronos-small')
    cv.add_argument('--tokenizer', default='NeoQuasar/Kronos-Tokenizer-base')
    cv.add_argument('--model-revision', default=REFERENCE_MODEL_REVISION)
    cv.add_argument('--tokenizer-revision', default=REFERENCE_TOKENIZER_REVISION)
    cv.add_argument('--device', default=None)
    cv.add_argument('--max-context', type=int, default=512)
    cv.add_argument('--cache-dir', default=None)
    cv.add_argument('--output', default=None,
                    help='JSON output path (default: '
                         'data/eval/classical_volatility_benchmark_report.json)')

    rc = sub.add_parser('recompute-classical-gate',
                        help='re-derive the classical A/B/C gate from a saved '
                             'benchmark report (corrected c6/c8, no re-inference)')
    rc.add_argument('--report', required=True,
                    help='path to a classical_volatility_benchmark_report.json')
    rc.add_argument('--summary', action='store_true',
                    help='print the pooled DM + Kronos-vs-HAR summary too')

    ml = sub.add_parser('ml-vs-har',
                        help='Phase 6: supervised ML (LightGBM/XGBoost) vs HAR '
                             'volatility benchmark, strict walk-forward')
    ml.add_argument('--db', default=str(DEFAULT_DB))
    ml.add_argument('--assets', nargs='+', default=['BTC/USDT', 'ETH/USDT'])
    ml.add_argument('--timeframes', nargs='+', default=['1h', '4h', '1d'])
    ml.add_argument('--context', type=int, default=512,
                    help='window-boundary warm-up (matches the frozen classical '
                         'benchmark; ML features use their own 64-bar warm-up)')
    ml.add_argument('--window-size', type=int, default=1000)
    ml.add_argument('--output', default=None,
                    help='JSON output path (default: '
                         'data/eval/ml_vs_har_volatility_report.json)')

    ca = sub.add_parser('cross-asset',
                        help='Phase 7: cross-asset information vs frozen HAR '
                             '(HAR + linear cross-asset extension, expanding OLS)')
    ca.add_argument('--db', default=str(DEFAULT_DB))
    ca.add_argument('--assets', nargs='+', default=['BTC/USDT', 'ETH/USDT'])
    ca.add_argument('--timeframes', nargs='+', default=['1h', '4h', '1d'])
    ca.add_argument('--context', type=int, default=512,
                    help='window-boundary warm-up (matches the frozen classical '
                         'benchmark)')
    ca.add_argument('--window-size', type=int, default=1000)
    ca.add_argument('--output', default=None,
                    help='JSON output path (default: '
                         'data/eval/cross_asset_volatility_report.json)')

    args = p.parse_args(argv)
    try:
        if args.cmd == 'predict':
            _predict(args)
        elif args.cmd == 'backtest':
            _backtest(args)
        elif args.cmd == 'benchmark':
            _benchmark(args)
        elif args.cmd == 'robustness':
            _robustness(args)
        elif args.cmd == 'research-targets':
            _research_targets(args)
        elif args.cmd == 'validate-reference':
            _validate_reference(args)
        elif args.cmd == 'volatility-research':
            _volatility_research(args)
        elif args.cmd == 'classical-volatility':
            _classical_volatility(args)
        elif args.cmd == 'recompute-classical-gate':
            _recompute_classical_gate(args)
        elif args.cmd == 'ml-vs-har':
            _ml_vs_har(args)
        elif args.cmd == 'cross-asset':
            _cross_asset(args)
        else:
            _evaluate(args)
        return 0
    except (ModelUnavailableError, FileNotFoundError, ValueError) as exc:
        print('error: %s' % exc, file=sys.stderr)
        return 2


def _benchmark(args):
    import time as _time
    candles = load_candles(args.db, args.symbol, args.timeframe)
    manager = ModelManager(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        device=args.device or None,
        max_context=args.max_context,
        cache_dir=args.cache_dir or None,
    )
    load = measure_model_load(manager)
    if not manager.available:
        raise ModelUnavailableError('real Kronos model unavailable: %s' % manager.error)
    result = run_benchmark(
        manager, candles, timeframe=args.timeframe,
        context_length=args.context, horizon=args.horizon,
        now_ms=int(_time.time() * 1000), warmed_runs=args.warmed_runs,
        seed=args.seed, deterministic=not args.no_deterministic)
    print(json.dumps({"model_load": load, "benchmark": result}, indent=2,
                     default=str))


def _evaluate(args):
    import time as _time
    candles = load_candles(args.db, args.symbol, args.timeframe)

    manager = ModelManager(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        device=args.device or None,
        max_context=args.max_context,
        cache_dir=args.cache_dir or None,
    ).load()
    if not manager.available:
        raise ModelUnavailableError(
            'real Kronos model is unavailable for evaluation: %s' % manager.error)

    deterministic = not args.no_deterministic
    if not deterministic:
        print('warning: --no-deterministic disables the argmax recipe; '
              'sampling is stochastic and results are NOT reproducible.',
              file=sys.stderr)

    config = EvaluationConfig(
        context_length=args.context,
        horizon=args.horizon,
        deterministic=deterministic,
        seed=args.seed,
        direction_threshold=args.direction_threshold,
        start_ms=parse_timestamp(args.start),
        end_ms=parse_timestamp(args.end),
        max_predictions=args.max_predictions,
    )
    if not deterministic:
        # Switch from the deterministic argmax recipe (top_k=1/top_p=1.0) to the
        # upstream stochastic sampling defaults, so --no-deterministic is real.
        config.top_k = 0
        config.top_p = 0.9
        config.sample_count = 1
    evaluator = PredictionEvaluator(KronosRealPredictor(manager), config,
                                    args.symbol, args.timeframe)
    result = evaluator.evaluate(candles)

    print(json.dumps(result.report, indent=2, default=str))
    if args.include_rows:
        print(json.dumps([r.asdict() for r in result.rows], indent=2, default=str))

    output = Path(args.output) if args.output else (
        ROOT / 'data' / 'eval' /
        ('%s_%s_%s.json' % (args.symbol.replace('/', '_'), args.timeframe,
                            result.report.get('evaluation_start_ms')
                            or int(_time.time() * 1000))))
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump({
            'report': result.report,
            'rows': [r.asdict() for r in result.rows],
            'baseline_rows': {
                'persistence': [r.asdict() for r in result.baseline_rows.get('persistence', [])],
                'previous_direction': [r.asdict() for r in result.baseline_rows.get('previous_direction', [])],
            },
        }, f, indent=2, default=str)
    print('saved evaluation results to %s' % output, file=sys.stderr)


def _robustness(args):
    import time as _time
    if args.no_deterministic:
        print('warning: --no-deterministic disables the argmax recipe; '
              'sampling is stochastic and results are NOT reproducible.',
              file=sys.stderr)

    manager = ModelManager(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        device=args.device or None,
        max_context=args.max_context,
        cache_dir=args.cache_dir or None,
    ).load()
    if not manager.available:
        raise ModelUnavailableError(
            'real Kronos model is unavailable for robustness evaluation: %s'
            % manager.error)

    config = EvaluationConfig(
        context_length=args.context,
        horizon=args.horizon,
        deterministic=not args.no_deterministic,
        seed=args.seed,
        direction_threshold=args.direction_threshold,
        window_size=args.window_size,
    )
    if not config.deterministic:
        config.top_k = 0
        config.top_p = 0.9
        config.sample_count = 1

    series = [(s, tf) for s in args.assets for tf in args.timeframes]

    def loader(symbol, timeframe):
        return load_candles(args.db, symbol, timeframe)

    report = run_robustness(KronosRealPredictor(manager), config, series, loader)

    output = Path(args.output) if args.output else (
        ROOT / 'data' / 'eval' / 'robustness_report.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Human-readable summary to stdout.
    print(json.dumps({'configuration': report['configuration'],
                      'across_all_series': report['across_all_series'],
                      'series': [{
                          'symbol': s['symbol'], 'timeframe': s['timeframe'],
                          'window_info': s['window_info'],
                          'summary': s['summary'],
                      } for s in report['series']]}, indent=2, default=str))
    print('saved consolidated robustness report to %s' % output, file=sys.stderr)


def _research_targets(args):
    if args.no_deterministic:
        print('warning: --no-deterministic disables the argmax recipe; '
              'sampling is stochastic and results are NOT reproducible.',
              file=sys.stderr)

    manager = ModelManager(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        device=args.device or None,
        max_context=args.max_context,
        cache_dir=args.cache_dir or None,
    ).load()
    if not manager.available:
        raise ModelUnavailableError(
            'real Kronos model is unavailable for target research: %s' % manager.error)

    config = EvaluationConfig(
        context_length=args.context,
        horizon=1,  # base target horizon; multi-period target sets its own
        deterministic=not args.no_deterministic,
        seed=args.seed,
        direction_threshold=args.direction_threshold,
        window_size=args.window_size,
    )
    if not config.deterministic:
        config.top_k = 0
        config.top_p = 0.9
        config.sample_count = 1

    series = [(s, tf) for s in args.assets for tf in args.timeframes]

    def loader(symbol, timeframe):
        return load_candles(args.db, symbol, timeframe)

    report = run_research_experiment(KronosRealPredictor(manager), config,
                                     series, loader)

    output = Path(args.output) if args.output else (
        ROOT / 'data' / 'eval' / 'research_targets_report.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps({
        'kind': report['kind'],
        'frozen_baseline_verified': report['frozen_baseline_verified'],
        'architecture_check': report['architecture_check'],
        'configuration': report['configuration'],
        'targets': {
            tid: {'spec': t['spec'], 'series': t['series']}
            for tid, t in report['targets'].items()
        },
    }, indent=2, default=str))
    print('saved research-targets report to %s' % output, file=sys.stderr)


def _validate_reference(args):
    # Load the model if possible; the contract-level comparison runs regardless.
    manager = ModelManager(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        device=args.device or None,
        max_context=512,
        cache_dir=args.cache_dir or None,
    ).load()
    if not manager.available:
        print('note: model weights unavailable (%s); running the contract-level '
              'comparison only' % (manager.error or 'unknown'), file=sys.stderr)

    report = build_validation_report(manager, context_len=args.context,
                                     pred_len=args.pred_len)
    print(json.dumps(report, indent=2, default=str))

    output = Path(args.output) if args.output else (
        ROOT / 'data' / 'eval' / 'reference_validation_report.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print('saved reference validation report to %s' % output, file=sys.stderr)
    return 0


def _volatility_research(args):
    if args.no_deterministic:
        print('warning: --no-deterministic disables the argmax recipe; '
              'sampling is stochastic and results are NOT reproducible.',
              file=sys.stderr)

    manager = ModelManager(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        device=args.device or None,
        max_context=args.max_context,
        cache_dir=args.cache_dir or None,
    ).load()
    if not manager.available:
        raise ModelUnavailableError(
            'real Kronos model is unavailable for volatility research: %s'
            % manager.error)

    config = EvaluationConfig(
        context_length=args.context,
        horizon=1,  # the range target is next-candle high/low
        deterministic=not args.no_deterministic,
        seed=args.seed,
        direction_threshold=0.0005,
        window_size=args.window_size,
    )
    if not config.deterministic:
        config.top_k = 0
        config.top_p = 0.9
        config.sample_count = 1

    series = [(s, tf) for s in args.assets for tf in args.timeframes]

    def loader(symbol, timeframe):
        return load_candles(args.db, symbol, timeframe)

    report = run_volatility_research(KronosRealPredictor(manager), config,
                                     series, loader)

    output = Path(args.output) if args.output else (
        ROOT / 'data' / 'eval' / 'volatility_research_report.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps({
        'kind': report['kind'],
        'configuration': report['configuration'],
        'baseline_definitions': report['baseline_definitions'],
        'target_definitions': report['target_definitions'],
        'success_gate': report['success_gate'],
        'pooled_statistics': report['pooled_statistics'],
        'window_records': report['window_records'],
    }, indent=2, default=str))
    print('saved volatility research report to %s' % output, file=sys.stderr)
    return 0


def _classical_volatility(args):
    if args.no_deterministic:
        print('warning: --no-deterministic disables the argmax recipe; '
              'sampling is stochastic and results are NOT reproducible.',
              file=sys.stderr)

    manager = ModelManager(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        device=args.device or None,
        max_context=args.max_context,
        cache_dir=args.cache_dir or None,
    ).load()
    if not manager.available:
        raise ModelUnavailableError(
            'real Kronos model is unavailable for the classical volatility '
            'benchmark: %s' % manager.error)

    config = EvaluationConfig(
        context_length=args.context,
        horizon=1,  # next-candle high/low range
        deterministic=not args.no_deterministic,
        seed=args.seed,
        direction_threshold=0.0005,
        window_size=args.window_size,
    )
    if not config.deterministic:
        config.top_k = 0
        config.top_p = 0.9
        config.sample_count = 1

    series = [(s, tf) for s in args.assets for tf in args.timeframes]

    def loader(symbol, timeframe):
        return load_candles(args.db, symbol, timeframe)

    report = run_classical_volatility_benchmark(
        KronosRealPredictor(manager), config, series, loader)

    output = Path(args.output) if args.output else (
        ROOT / 'data' / 'eval' / 'classical_volatility_benchmark_report.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps({
        'kind': report['kind'],
        'configuration': report['configuration'],
        'classical_baselines': report['classical_baselines'],
        'targets': report['targets'],
        'success_gate': report['success_gate'],
        'pooled_statistics': report['pooled_statistics'],
        'regime_pooled': report['regime_pooled'],
        'har_adequacy': report['har_adequacy'],
        'kronos_vs_best_classical': report['kronos_vs_best_classical'],
        'window_records': report['window_records'],
    }, indent=2, default=str))
    print('saved classical volatility benchmark report to %s' % output,
          file=sys.stderr)
    return 0


def _recompute_classical_gate(args):
    path = Path(args.report)
    if not path.exists():
        print('error: report not found: %s' % path, file=sys.stderr)
        return 2
    with open(path) as f:
        report = json.load(f)
    if args.summary:
        print(json.dumps(recompute_classical_summary(report), indent=2, default=str))
    else:
        print(json.dumps(recompute_classical_gate(report), indent=2, default=str))
    return 0


def _ml_vs_har(args):
    config = EvaluationConfig(
        context_length=args.context,
        horizon=1,
        deterministic=True,
        seed=0,
        direction_threshold=0.0005,
        window_size=args.window_size,
    )
    series = [(s, tf) for s in args.assets for tf in args.timeframes]

    def loader(symbol, timeframe):
        return load_candles(args.db, symbol, timeframe)

    report = run_ml_vs_har(loader, config, series)

    output = Path(args.output) if args.output else (
        ROOT / 'data' / 'eval' / 'ml_vs_har_volatility_report.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps({
        'kind': report['kind'],
        'configuration': report['configuration'],
        'ml_configuration': report['ml_configuration'],
        'targets': report['targets'],
        'baselines': report['baselines'],
        'walk_forward': report['walk_forward'],
        'statistical_methodology': report['statistical_methodology'],
        'window_records': report['window_records'],
        'pooled_statistics': report['pooled_statistics'],
        'regime_pooled': report['regime_pooled'],
        'ml_adequacy': report['ml_adequacy'],
        'success_gate': report['success_gate'],
    }, indent=2, default=str))
    print('saved ML-vs-HAR report to %s' % output, file=sys.stderr)
    return 0


def _cross_asset(args):
    config = EvaluationConfig(
        context_length=args.context,
        horizon=1,
        deterministic=True,
        seed=0,
        direction_threshold=0.0005,
        window_size=args.window_size,
    )
    assets = list(dict.fromkeys(args.assets))  # dedupe, preserve order
    pairs = []
    for tf in args.timeframes:
        for i, a in enumerate(assets):
            for b in assets:
                if a != b:
                    pairs.append((a, b, tf))

    def loader(symbol, timeframe):
        return load_candles(args.db, symbol, timeframe)

    report = run_cross_asset(loader, config, pairs)

    output = Path(args.output) if args.output else (
        ROOT / 'data' / 'eval' / 'cross_asset_volatility_report.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps({
        'kind': report['kind'],
        'configuration': report['configuration'],
        'model': report['model'],
        'target': report['target'],
        'frozen_baseline': report['frozen_baseline'],
        'statistical_methodology': report['statistical_methodology'],
        'window_records': report['window_records'],
        'pooled_statistics': report['pooled_statistics'],
        'regime_pooled': report['regime_pooled'],
        'cross_adequacy': report['cross_adequacy'],
        'success_gate': report['success_gate'],
    }, indent=2, default=str))
    print('saved cross-asset report to %s' % output, file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
