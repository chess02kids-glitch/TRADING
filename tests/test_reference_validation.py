"""Reference-pipeline validation tests (offline; no model weights required).

These prove that our input construction matches the upstream Kronos reference
on the OHLCV/timestamp/normalization/API contract, that the amount channel is
the one documented difference, and that the pinned revisions are in sync with
the upstream test file.
"""
import json

import numpy as np
import pytest

from kronos_trading.model import (
    REFERENCE_MODEL_REVISION,
    REFERENCE_TOKENIZER_REVISION,
    ModelManager,
)
from kronos_trading.reference_validation import (
    FEATURE_NAMES,
    REFERENCE,
    _amount_summary,
    build_validation_report,
    constants_in_sync,
    load_reference_data,
    load_reference_fixture,
    our_inputs,
    reference_inputs,
    run_contract_comparison,
    upstream_reference_constants,
)


def test_upstream_constants_in_sync_with_test_file():
    sync = constants_in_sync()
    assert sync == {"model_revision": True, "tokenizer_revision": True,
                    "max_context": True, "pred_len": True, "seed": True}


def test_model_manager_defaults_pin_reference_revisions():
    m = ModelManager()
    assert m.model_revision == REFERENCE_MODEL_REVISION
    assert m.tokenizer_revision == REFERENCE_TOKENIZER_REVISION
    assert m.model_name == "NeoQuasar/Kronos-small"
    assert m.tokenizer_name == "NeoQuasar/Kronos-Tokenizer-base"


def test_reference_data_and_fixture_present():
    df = load_reference_data()
    assert df is not None and df.shape == (2500, 7)
    assert list(df.columns) == ["timestamps"] + FEATURE_NAMES
    fixture = load_reference_fixture(512)
    assert fixture is not None and fixture.shape == (8, 7)
    assert fixture is None or list(fixture.columns) == ["timestamps"] + FEATURE_NAMES


def test_inputs_have_expected_shapes_and_columns():
    df = load_reference_data()
    ref_df, ref_x, ref_y = reference_inputs(df, 512, 8)
    our_df, our_x, our_y = our_inputs(df, 512, 8)
    assert ref_df.shape == (512, 6) and list(ref_df.columns) == FEATURE_NAMES
    assert our_df.shape == (512, 5)
    assert list(our_df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(ref_x) == len(our_x) == 512
    assert len(ref_y) == len(our_y) == 8
    # same wall-clock timestamps (reference naive vs ours tz-aware UTC)
    assert str(ref_x.iloc[0]) == str(our_x.iloc[0].tz_localize(None))
    assert str(ref_y.iloc[0]) == str(our_y.iloc[0].tz_localize(None))


def test_contract_comparison_matches_except_amount():
    report = run_contract_comparison(512, 8)
    assert report["status"] == "ok"
    tc = report["tensor_comparison"]
    assert tc["x_shape"]["match"] is True
    assert tc["x_stamp_shape"]["match"] is True
    assert tc["y_stamp_shape"]["match"] is True
    assert tc["ohlcv_channels_identical"] is True
    assert tc["amount_channel_identical"] is False
    assert tc["x_stamp_identical"] is True
    assert tc["y_stamp_identical"] is True
    assert tc["kwargs_identical"] is True
    assert tc["contract_matches_except_amount"] is True
    for ch in FEATURE_NAMES[:5]:
        assert tc["max_abs_diff_per_channel"][ch] == 0.0
    assert tc["max_abs_diff_per_channel"]["amount"] > 0.0


def test_amount_channel_is_derived_proxy():
    df = load_reference_data()
    ref_df, _, _ = reference_inputs(df, 512, 8)
    our_df, _, _ = our_inputs(df, 512, 8)
    summary = _amount_summary(ref_df, our_df)
    assert summary["our_amount"].startswith("volume * mean")
    assert summary["mean_abs_diff"] > 0
    assert summary["mean_rel_diff"] > 0


def test_reference_kwargs_match_regression_recipe():
    report = run_contract_comparison(512, 8)
    assert report["temperature"] == 1.0
    assert report["top_k"] == 1
    assert report["top_p"] == 1.0
    assert report["sample_count"] == 1
    assert report["pred_len"] == 8
    assert report["model_revision"] == REFERENCE_MODEL_REVISION
    assert report["tokenizer_revision"] == REFERENCE_TOKENIZER_REVISION


def test_validation_report_structure_and_verdict_without_weights():
    report = build_validation_report(None, 512, 8)
    assert report["kind"] == "reference_pipeline_validation"
    assert report["contract"]["status"] == "ok"
    assert report["output"] is None
    # no weights here -> verdict B (mismatch found and fixed; output unavailable)
    assert report["verdict"] in ("B", "C")
    assert report["meaning"]


def test_contract_unavailable_without_reference_data(tmp_path, monkeypatch):
    import kronos_trading.reference_validation as rv
    monkeypatch.setattr(rv, "REFERENCE_INPUT", tmp_path / "missing.csv")
    report = rv.run_contract_comparison(512, 8)
    assert report["status"] == "unavailable"
