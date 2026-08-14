"""Strict validation of our inference pipeline against the upstream Kronos reference.

The reference is the upstream Kronos regression test and fixture generator at
the pinned commit (67b630e67f6a):

* ``Kronos/tests/test_kronos_regression.py``
* ``Kronos/tests/data/generate_regression_output.py``
* fixtures ``Kronos/tests/data/regression_input.csv`` and
  ``regression_output_{512,256}.csv``

Reference recipe (exact):

* model revision ``901c26c1332695a2a8f243eb2f37243a37bea320``
* tokenizer revision ``0e0117387f39004a9016484a186a908917e22426``
* feature columns ``[open, high, low, close, volume, amount]`` (real turnover)
* context 512, pred_len 8, max_context 512, device cpu, seed 123
* ``tokenizer.eval(); model.eval(); torch.no_grad()``
* ``predict(df, x_timestamp, y_timestamp, pred_len=8, T=1.0, top_k=1,
  top_p=1.0, verbose=False, sample_count=1)``

The validation has two levels:

1. **Contract level (runs offline).** The *real* upstream ``KronosPredictor``
   is instantiated with stub model/tokenizer modules and a capturing
   ``generate()``, so 100% of upstream preprocessing executes (column
   selection, amount derivation, NaN check, ``calc_time_stamps``, per-sequence
   z-score, clip, reshape, API kwargs). Our input construction
   (``to_kronos_frame`` + timestamps) is fed through the SAME upstream
   ``predict()`` and the resulting tensors are compared element-wise.
2. **Output level (requires model weights).** Both paths run real inference on
   the reference input and are compared against the upstream fixtures.

No model is trained, tuned, or downloaded here; the verified dataset and the
upstream submodule are untouched.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .model import (REFERENCE_MODEL_REVISION, REFERENCE_TOKENIZER_REVISION,
                    KronosRealPredictor, ModelManager)
from .preprocess import to_kronos_frame
from .types import Candle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KRONOS_DIR = PROJECT_ROOT / "Kronos"
REFERENCE_INPUT = KRONOS_DIR / "tests" / "data" / "regression_input.csv"
REFERENCE_TEST = KRONOS_DIR / "tests" / "test_kronos_regression.py"

FEATURE_NAMES = ["open", "high", "low", "close", "volume", "amount"]

REFERENCE = {
    "commit": "67b630e67f6a18c9e9be918d9b4337c960db1e9a",
    "source": [
        "Kronos/tests/test_kronos_regression.py",
        "Kronos/tests/data/generate_regression_output.py",
    ],
    "model_name": "NeoQuasar/Kronos-small",
    "tokenizer_name": "NeoQuasar/Kronos-Tokenizer-base",
    "model_revision": REFERENCE_MODEL_REVISION,
    "tokenizer_revision": REFERENCE_TOKENIZER_REVISION,
    "feature_names": FEATURE_NAMES,
    "max_context": 512,
    "pred_len": 8,
    "seed": 123,
    "device": "cpu",
    "temperature": 1.0,
    "top_k": 1,
    "top_p": 1.0,
    "sample_count": 1,
    "verbose": False,
    "eval_mode": True,
    "no_grad": True,
}


def _import_upstream_predictor():
    """Import the upstream KronosPredictor class (read-only)."""
    if str(KRONOS_DIR) not in sys.path:
        sys.path.insert(0, str(KRONOS_DIR))
    from model import KronosPredictor  # noqa: F401  (upstream)
    return KronosPredictor


def upstream_reference_constants() -> Dict[str, Any]:
    """Extract the pinned constants from the upstream test (guard vs drift)."""
    text = REFERENCE_TEST.read_text(encoding="utf-8")
    def grab(name: str) -> str:
        m = re.search(r'^%s\s*=\s*"([^"]+)"' % name, text, re.MULTILINE)
        return m.group(1) if m else None
    return {
        "MODEL_REVISION": grab("MODEL_REVISION"),
        "TOKENIZER_REVISION": grab("TOKENIZER_REVISION"),
        "MAX_CTX_LEN": re.search(r'MAX_CTX_LEN\s*=\s*(\d+)', text).group(1),
        "PRED_LEN": re.search(r'PRED_LEN\s*=\s*(\d+)', text).group(1),
        "SEED": re.search(r'SEED\s*=\s*(\d+)', text).group(1),
    }


def constants_in_sync() -> Dict[str, bool]:
    """True for each constant that matches the upstream test file."""
    up = upstream_reference_constants()
    return {
        "model_revision": up.get("MODEL_REVISION") == REFERENCE_MODEL_REVISION,
        "tokenizer_revision": up.get("TOKENIZER_REVISION") == REFERENCE_TOKENIZER_REVISION,
        "max_context": up.get("MAX_CTX_LEN") == str(REFERENCE["max_context"]),
        "pred_len": up.get("PRED_LEN") == str(REFERENCE["pred_len"]),
        "seed": up.get("SEED") == str(REFERENCE["seed"]),
    }


def load_reference_data() -> Optional[pd.DataFrame]:
    """Load the upstream regression input (official example dataset)."""
    if not REFERENCE_INPUT.exists():
        return None
    return pd.read_csv(REFERENCE_INPUT, parse_dates=["timestamps"])


def load_reference_fixture(context_len: int = 512) -> Optional[pd.DataFrame]:
    path = KRONOS_DIR / "tests" / "data" / ("regression_output_%d.csv" % context_len)
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["timestamps"])


# --------------------------------------------------------------------------- #
# Input construction: reference vs ours
# --------------------------------------------------------------------------- #
def _naive_to_utc_ms(s) -> int:
    """Interpret a naive timestamp string as UTC (Binance/upstream convention)."""
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp() * 1000)


def _row_to_candle(row) -> Candle:
    return Candle(
        timestamp_ms=_naive_to_utc_ms(row.timestamps),
        open=float(row.open), high=float(row.high),
        low=float(row.low), close=float(row.close),
        volume=float(row.volume),
    )


def reference_inputs(df: pd.DataFrame, context_len: int = 512, pred_len: int = 8):
    """Build the reference inputs exactly as the upstream regression test does."""
    context = df.iloc[:context_len]
    ref_df = context[FEATURE_NAMES].reset_index(drop=True)
    ref_x_ts = context["timestamps"].reset_index(drop=True)
    ref_y_ts = df["timestamps"].iloc[context_len:context_len + pred_len].reset_index(drop=True)
    return ref_df, ref_x_ts, ref_y_ts


def our_inputs(df: pd.DataFrame, context_len: int = 512, pred_len: int = 8):
    """Build our pipeline's inputs from the SAME reference rows.

    Our data source (Binance public klines) carries volume but not turnover, so
    ``to_kronos_frame`` emits only OHLCV and upstream derives ``amount``. The
    reference dataset is 5-minute bars (outside our supported TF map), so the
    y_timestamp is taken from the dataset's own future rows rather than
    ``future_timestamps`` (which is the same ms arithmetic for supported
    timeframes).
    """
    context = df.iloc[:context_len]
    candles = [_row_to_candle(row) for row in context.itertuples(index=False)]
    our_df, our_x_ts = to_kronos_frame(candles)
    future = df["timestamps"].iloc[context_len:context_len + pred_len]
    our_y_ts = pd.Series([pd.Timestamp(_naive_to_utc_ms(s), unit="ms", tz="UTC")
                          for s in future])
    return our_df, our_x_ts, our_y_ts


# --------------------------------------------------------------------------- #
# Capturing predictor: real upstream preprocessing, stubbed model forward pass
# --------------------------------------------------------------------------- #
class _CapturingPredictor:
    def __init__(self, device: str = "cpu", max_context: int = 512, clip: float = 5.0):
        import torch
        KronosPredictor = _import_upstream_predictor()

        class _Stub(torch.nn.Module):
            pass

        self.predictor = KronosPredictor(_Stub(), _Stub(), device=device,
                                         max_context=max_context, clip=clip)
        self.capture: Dict[str, Any] = {}

        def generate(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose):
            self.capture["x"] = np.asarray(x, dtype=np.float32)
            self.capture["x_stamp"] = np.asarray(x_stamp, dtype=np.float32)
            self.capture["y_stamp"] = np.asarray(y_stamp, dtype=np.float32)
            self.capture["kwargs"] = {
                "pred_len": pred_len, "T": T, "top_k": top_k,
                "top_p": top_p, "sample_count": sample_count, "verbose": verbose,
            }
            # Return zeros so predict()'s denormalization yields x_mean per row;
            # the returned DataFrame is a preprocessing artifact, not model output.
            return np.zeros((x.shape[0], pred_len, 6), dtype=np.float32)

        self.predictor.generate = generate

    def run(self, df, x_timestamp, y_timestamp, pred_len, **kwargs):
        return self.predictor.predict(
            df=df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=pred_len, **kwargs)


def _max_abs(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.shape != b.shape:
        return None
    return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def _max_abs_per_channel(a: np.ndarray, b: np.ndarray, names) -> Dict[str, Optional[float]]:
    if a.shape != b.shape or a.ndim != 3:
        return {name: None for name in names}
    out = {}
    for i, name in enumerate(names):
        out[name] = float(np.max(np.abs(a[:, :, i] - b[:, :, i])))
    return out


def _amount_summary(ref_df: pd.DataFrame, our_df: pd.DataFrame) -> Dict[str, Any]:
    """Quantify the difference between real turnover and upstream's derived amount."""
    derived = our_df["volume"] * ref_df[["open", "high", "low", "close"]].mean(axis=1)
    real = ref_df["amount"].astype(float)
    diff = (derived - real).abs()
    return {
        "reference_amount": "real turnover (from dataset)",
        "our_amount": "volume * mean(open, high, low, close) [upstream fallback]",
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "mean_rel_diff": float((diff / (real.abs() + 1e-9)).mean()),
    }


# --------------------------------------------------------------------------- #
# Contract comparison (offline)
# --------------------------------------------------------------------------- #
def run_contract_comparison(context_len: int = 512, pred_len: int = 8) -> Dict[str, Any]:
    """Compare reference vs our input construction through the real upstream
    ``predict()`` preprocessing (model forward pass stubbed)."""
    df = load_reference_data()
    if df is None:
        return {"status": "unavailable", "reason": "regression_input.csv not present"}

    ref_df, ref_x_ts, ref_y_ts = reference_inputs(df, context_len, pred_len)
    our_df, our_x_ts, our_y_ts = our_inputs(df, context_len, pred_len)

    cap_ref = _CapturingPredictor()
    cap_our = _CapturingPredictor()
    pred_len_arg = pred_len
    cap_ref.run(ref_df, ref_x_ts, ref_y_ts, pred_len_arg,
                T=REFERENCE["temperature"], top_k=REFERENCE["top_k"],
                top_p=REFERENCE["top_p"], sample_count=REFERENCE["sample_count"],
                verbose=REFERENCE["verbose"])
    cap_our.run(our_df, our_x_ts, our_y_ts, pred_len_arg,
                T=REFERENCE["temperature"], top_k=REFERENCE["top_k"],
                top_p=REFERENCE["top_p"], sample_count=REFERENCE["sample_count"],
                verbose=REFERENCE["verbose"])

    xr = cap_ref.capture["x"]
    xo = cap_our.capture["x"]
    sr = cap_ref.capture["x_stamp"]
    so = cap_our.capture["x_stamp"]
    yr = cap_ref.capture["y_stamp"]
    yo = cap_our.capture["y_stamp"]

    per_channel = _max_abs_per_channel(xr, xo, FEATURE_NAMES)
    stamps_equal = (_max_abs(sr, so) == 0.0) and (_max_abs(yr, yo) == 0.0)
    kwargs_equal = cap_ref.capture["kwargs"] == cap_our.capture["kwargs"]

    # OHLCV channels should be bit-identical (same raw values, same z-score).
    ohlcv_identical = all(per_channel[n] == 0.0 for n in FEATURE_NAMES[:5])
    amount_identical = per_channel["amount"] == 0.0

    contract_matches = ohlcv_identical and stamps_equal and kwargs_equal

    return {
        "status": "ok",
        "reference": dict(REFERENCE),
        "constants_in_sync": constants_in_sync(),
        "input_dataframe_shape": {
            "reference": list(ref_df.shape),
            "ours": list(our_df.shape),
        },
        "exact_columns": {
            "reference": list(ref_df.columns),
            "ours": list(our_df.columns),
            "note": "ours omits 'amount'; upstream derives it as volume * mean(price)",
        },
        "context_length": context_len,
        "x_timestamp": {
            "count": int(len(ref_x_ts)),
            "reference_dtype": str(ref_x_ts.dtype),
            "ours_dtype": str(our_x_ts.dtype),
            "reference_head": [str(x) for x in ref_x_ts.head(2)],
            "ours_head": [str(x) for x in our_x_ts.head(2)],
        },
        "y_timestamp": {
            "count": int(len(ref_y_ts)),
            "reference_head": [str(x) for x in ref_y_ts.head(2)],
            "ours_head": [str(x) for x in our_y_ts.head(2)],
        },
        "pred_len": pred_len,
        "temperature": REFERENCE["temperature"],
        "top_k": REFERENCE["top_k"],
        "top_p": REFERENCE["top_p"],
        "sample_count": REFERENCE["sample_count"],
        "model_revision": REFERENCE["model_revision"],
        "tokenizer_revision": REFERENCE["tokenizer_revision"],
        "preprocessing": {
            "normalization": "per-sequence z-score + clip to [-5, 5] (upstream predict())",
            "timestamp_features": "minute/hour/weekday/day/month (upstream calc_time_stamps())",
            "amount": "reference: real turnover; ours: volume * mean(price) fallback",
        },
        "tensor_comparison": {
            "x_shape": {"reference": list(xr.shape), "ours": list(xo.shape),
                        "match": xr.shape == xo.shape},
            "x_stamp_shape": {"reference": list(sr.shape), "ours": list(so.shape),
                              "match": sr.shape == so.shape},
            "y_stamp_shape": {"reference": list(yr.shape), "ours": list(yo.shape),
                              "match": yr.shape == yo.shape},
            "max_abs_diff_per_channel": per_channel,
            "ohlcv_channels_identical": bool(ohlcv_identical),
            "amount_channel_identical": bool(amount_identical),
            "x_stamp_identical": bool(_max_abs(sr, so) == 0.0),
            "y_stamp_identical": bool(_max_abs(yr, yo) == 0.0),
            "kwargs_identical": bool(kwargs_equal),
            "contract_matches_except_amount": bool(contract_matches),
        },
        "amount_channel": _amount_summary(ref_df, our_df),
        "decoding": {
            "upstream_default": "probabilistic nucleus sampling (top_k=0, top_p=0.9, "
                                "sample_logits=True -> torch.multinomial)",
            "reference_regression_recipe": "deterministic argmax via top_k=1, top_p=1.0 "
                                            "(multinomial over a one-hot distribution) + "
                                            "eval() + seed",
            "our_evaluation_recipe": "deterministic argmax (top_k=1, top_p=1.0, seed) + "
                                     "eval(); matches the regression recipe, not the "
                                     "probabilistic example default",
            "conclusion": "intended decoding is probabilistic sampling; the deterministic "
                          "argmax is a supported variant used by the upstream regression "
                          "test. No production default was changed.",
        },
    }


# --------------------------------------------------------------------------- #
# Output comparison (requires model weights)
# --------------------------------------------------------------------------- #
def run_output_comparison(manager: ModelManager, context_len: int = 512,
                          pred_len: int = 8, seed: int = 123) -> Dict[str, Any]:
    """Run real inference on the reference input via both paths and compare to
    the upstream fixture. Requires the pinned model weights to be present."""
    if manager is None or not manager.available:
        return {"status": "unavailable",
                "reason": "model weights not available in this environment"}
    df = load_reference_data()
    fixture = load_reference_fixture(context_len)
    if df is None or fixture is None:
        return {"status": "unavailable", "reason": "reference data missing"}

    import torch
    from .model import set_seed
    set_seed(seed)

    ref_df, ref_x_ts, ref_y_ts = reference_inputs(df, context_len, pred_len)
    our_df, our_x_ts, our_y_ts = our_inputs(df, context_len, pred_len)
    expected = fixture[FEATURE_NAMES].to_numpy(dtype=np.float32)

    def _predict(features, x_ts, y_ts):
        with torch.no_grad():
            return manager.predictor.predict(
                df=features, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=pred_len, T=REFERENCE["temperature"],
                top_k=REFERENCE["top_k"], top_p=REFERENCE["top_p"],
                sample_count=REFERENCE["sample_count"], verbose=False)

    ref_out = _predict(ref_df, ref_x_ts, ref_y_ts)[FEATURE_NAMES].to_numpy(np.float32)
    our_out = _predict(our_df, our_x_ts, our_y_ts)[FEATURE_NAMES].to_numpy(np.float32)

    def rel(a, b):
        return np.abs(a - b) / (np.abs(b) + 1e-9)

    return {
        "status": "ok",
        "device": manager.device,
        "reference_vs_fixture": {
            "max_abs_diff": float(np.max(np.abs(ref_out - expected))),
            "max_rel_diff": float(np.max(rel(ref_out, expected))),
            "within_upstream_tolerance_1e-5": bool(
                float(np.max(rel(ref_out, expected))) <= 1e-5),
        },
        "ours_vs_fixture": {
            "max_abs_diff": float(np.max(np.abs(our_out - expected))),
            "max_rel_diff": float(np.max(rel(our_out, expected))),
        },
        "ours_vs_reference": {
            "max_abs_diff": float(np.max(np.abs(our_out - ref_out))),
            "max_rel_diff": float(np.max(rel(our_out, ref_out))),
        },
        "predicted_ohlcv_reference": ref_out.tolist(),
        "predicted_ohlcv_ours": our_out.tolist(),
        "expected_ohlcv": expected.tolist(),
    }


# --------------------------------------------------------------------------- #
# Report + verdict
# --------------------------------------------------------------------------- #
def _verdict(contract: Dict[str, Any], output: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if output is not None and output.get("status") == "ok":
        ref_ok = output["reference_vs_fixture"]["within_upstream_tolerance_1e-5"]
        ours_close = output["ours_vs_reference"]["max_rel_diff"] <= 1e-3
        if ref_ok and ours_close:
            return {"verdict": "A", "meaning": "Pipeline matches upstream/reference behavior"}
        return {"verdict": "B", "meaning": "Pipeline mismatch found (output-level)"}
    if contract.get("status") != "ok":
        return {"verdict": "C", "meaning": "Reference experiment unavailable/inconclusive"}
    # Contract-level result only (weights unavailable here).
    if contract["tensor_comparison"]["contract_matches_except_amount"]:
        return {
            "verdict": "B",  # mismatch found (amount channel + revisions) but FIXED
            "meaning": (
                "Contract matches upstream on OHLCV/timestamps/normalization/API; "
                "the 'amount' feature is an upstream-derived proxy (data limitation) "
                "and the model/tokenizer revision defaults were unpinned (now fixed). "
                "Output-level equality is not yet demonstrated in this environment."
            ),
        }
    return {"verdict": "B", "meaning": "Pipeline mismatch found (contract-level)"}


def build_validation_report(manager: Optional[ModelManager] = None,
                            context_len: int = 512, pred_len: int = 8) -> Dict[str, Any]:
    contract = run_contract_comparison(context_len, pred_len)
    output = None
    if manager is not None and manager.available:
        output = run_output_comparison(manager, context_len, pred_len)
    verdict = _verdict(contract, output)
    return {
        "kind": "reference_pipeline_validation",
        "reference": dict(REFERENCE),
        "contract": contract,
        "output": output,
        **verdict,
        "notes": [
            "revision pinning mismatch was fixed in ModelManager/CLI defaults",
            "'amount' is derived (volume * mean price) because the verified "
            "Binance dataset has no turnover column; this is upstream's own "
            "fallback and cannot be fixed without changing the verified data",
            "intended upstream decoding is probabilistic nucleus sampling; the "
            "deterministic argmax is the upstream regression-test variant and "
            "was NOT changed",
            "statistical / evaluation results are unaffected by this validation; "
            "they must be revalidated with pinned revisions per task 7",
        ],
    }
