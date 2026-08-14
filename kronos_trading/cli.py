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
                    DeterministicMockPredictor, KronosRealPredictor)
from .pipeline import PredictionPipeline
from .backtest import Backtester
from .benchmark import measure_model_load, run_benchmark

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
        model_revision=args.model_revision or None,
        tokenizer_revision=args.tokenizer_revision or None,
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
    pred.add_argument('--model-revision', default=None)
    pred.add_argument('--tokenizer-revision', default=None)
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
    bench.add_argument('--model-revision', default=None)
    bench.add_argument('--tokenizer-revision', default=None)
    bench.add_argument('--device', default=None)
    bench.add_argument('--max-context', type=int, default=512)
    bench.add_argument('--cache-dir', default=None)
    bench.add_argument('--warmed-runs', type=int, default=10)
    bench.add_argument('--seed', type=int, default=123)
    bench.add_argument('--no-deterministic', action='store_true',
                       help='allow stochastic sampling during the benchmark')

    args = p.parse_args(argv)
    try:
        if args.cmd == 'predict':
            _predict(args)
        elif args.cmd == 'backtest':
            _backtest(args)
        else:
            _benchmark(args)
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
        model_revision=args.model_revision or None,
        tokenizer_revision=args.tokenizer_revision or None,
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


if __name__ == '__main__':
    raise SystemExit(main())
