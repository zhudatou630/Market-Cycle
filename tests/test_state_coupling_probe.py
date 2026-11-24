"""Rule, synthesis, and causal replay tests for state-coupling probe v0a."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from market_cycle.prototypes.state_coupling_probe import (
    _SOURCE_COLUMNS,
    _classify_row,
    build_state_coupling_probe,
)


def _threshold_columns() -> dict[str, dict[str, str]]:
    return {
        source: {label: f"threshold_{source}_{label}" for label in ("p20", "p30", "p70", "p90")}
        for source in _SOURCE_COLUMNS
    }


def _row(**overrides: object) -> pd.Series:
    values: dict[str, object] = {
        "efficiency_ohlc_min_midlong_status": "ok",
        "direction_drift_midlong_raw_status": "ok",
        "direction_scale_agreement_midlong_status": "ok",
        "two_sidedness_v1_20_status": "ok",
        "activity_level_atr_pct_prev_status": "ok",
        "expansion_range_atr_prev_status": "ok",
        "expansion_close_atr_prev_status": "ok",
        "expansion_close_share_status": "ok",
        "prior_range_clearance_up_20_status": "ok",
        "prior_range_clearance_down_20_status": "ok",
        "prior_range_clearance_up_55_status": "ok",
        "prior_range_clearance_down_55_status": "ok",
        "direction_drift_5": 0.8,
        "direction_drift_55": 0.8,
        "direction_drift_equal_raw": 0.8,
        "direction_drift_midlong_raw": 0.8,
        "direction_scale_agreement_midlong": 0.8,
        "efficiency_ohlc_min_midlong": 0.5,
        "two_sidedness_entropy_20": 0.5,
        "two_sidedness_overlap_20": 0.5,
        "two_sidedness_equal_v1": 0.5,
        "two_sidedness_midlong_v1": 0.5,
        "activity_level_atr_pct_prev": 0.5,
        "expansion_range_atr_prev": 0.5,
        "expansion_close_atr_prev": 0.1,
        "expansion_close_share": 0.5,
        "abs_expansion_close_share": 0.5,
        "historical_coverage_20": 0.5,
        "expansion_gap_prev_range_atr": 0.0,
    }
    for source in _SOURCE_COLUMNS:
        values[f"threshold_{source}_ready"] = True
        values[f"threshold_{source}_p20"] = 0.2
        values[f"threshold_{source}_p30"] = 0.3
        values[f"threshold_{source}_p70"] = 0.7
        values[f"threshold_{source}_p90"] = 0.9
    values.update(overrides)
    return pd.Series(values)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, "DIRECTIONAL_UP"),
        ({"direction_drift_midlong_raw": -0.8, "direction_drift_equal_raw": -0.8, "direction_drift_5": -0.8, "direction_drift_55": -0.8}, "DIRECTIONAL_DOWN"),
        ({"efficiency_ohlc_min_midlong": 0.2}, "DIRECTIONAL_UP"),
        ({"two_sidedness_entropy_20": 0.9, "two_sidedness_overlap_20": 0.9}, "DIRECTIONAL_UP"),
        ({"direction_drift_midlong_raw": 0.1, "direction_drift_equal_raw": 0.1, "direction_drift_5": 0.1, "direction_drift_55": 0.1, "efficiency_ohlc_min_midlong": 0.2, "two_sidedness_entropy_20": 0.8, "historical_coverage_20": 0.8}, "BALANCE_LIKE"),
        ({"direction_drift_midlong_raw": 0.4, "direction_drift_equal_raw": 0.4, "direction_drift_5": 0.4, "direction_drift_55": 0.4}, "UNRESOLVED"),
    ],
)
def test_synthetic_background_scenarios(changes: dict[str, object], expected: str):
    result = _classify_row(_row(**changes), _threshold_columns())
    assert result["regime_candidate"] == expected


def test_entropy_and_overlap_count_as_one_balance_evidence():
    result = _classify_row(
        _row(
            direction_drift_midlong_raw=0.1,
            direction_drift_equal_raw=0.1,
            direction_drift_5=0.1,
            direction_drift_55=0.1,
            two_sidedness_entropy_20=0.8,
            two_sidedness_overlap_20=0.8,
        ),
        _threshold_columns(),
    )
    assert result["regime_evidence_hits"] == 2
    assert result["regime_candidate"] == "UNRESOLVED"


def test_scale_conflict_near_zero_does_not_count_as_weak_direction():
    result = _classify_row(
        _row(
            direction_drift_midlong_raw=0.05,
            direction_drift_equal_raw=-0.05,
            direction_drift_5=-0.8,
            direction_drift_55=0.8,
            direction_scale_agreement_midlong=0.1,
            efficiency_ohlc_min_midlong=0.2,
            two_sidedness_entropy_20=0.8,
            historical_coverage_20=0.8,
        ),
        _threshold_columns(),
    )
    assert result["regime_candidate"] == "BALANCE_LIKE"
    assert result["regime_evidence_hits"] == 3
    assert result["scale_conflict"]
    assert "D5_D55_SIGN" in json.loads(result["scale_conflict_type"])


@pytest.mark.parametrize(
    ("close", "share", "expected"),
    [(1.2, 0.8, ["EXPANSION_UP"]), (-1.2, -0.8, ["EXPANSION_DOWN"]), (0.02, 0.1, ["TURBULENT_SHOCK"])],
)
def test_synthetic_expansion_events(close: float, share: float, expected: list[str]):
    result = _classify_row(
        _row(
            expansion_range_atr_prev=1.2,
            expansion_close_atr_prev=close,
            expansion_close_share=share,
            abs_expansion_close_share=abs(share),
        ),
        _threshold_columns(),
    )
    assert json.loads(result["event_candidates"]) == expected
    assert result["regime_candidate"] == "DIRECTIONAL_UP"


def test_gap_is_an_attribute_and_does_not_create_an_event():
    result = _classify_row(
        _row(expansion_gap_prev_range_atr=-1.4, expansion_range_atr_prev=0.5),
        _threshold_columns(),
    )
    assert result["gap_direction"] == "DOWN"
    assert result["gap_size_atr"] == 1.4
    assert json.loads(result["event_candidates"]) == []


def test_non_ok_and_threshold_not_ready_are_not_unresolved():
    non_ok = _classify_row(
        _row(efficiency_ohlc_min_midlong_status="zero_path"), _threshold_columns()
    )
    not_ready_row = _row()
    not_ready_row["threshold_efficiency_ready"] = False
    not_ready = _classify_row(not_ready_row, _threshold_columns())
    assert non_ok["status"] == "non_ok" and non_ok["regime_candidate"] is None
    assert not_ready["status"] == "insufficient_threshold_history"
    assert not_ready["regime_candidate"] is None


def _bars(rows: int = 1700) -> pd.DataFrame:
    dates = pd.bdate_range("2000-01-03", periods=rows)
    index = np.arange(rows, dtype=float)
    close = 100.0 * np.exp(0.0003 * index + 0.015 * np.sin(index / 13.0))
    previous = np.r_[close[0], close[:-1]]
    open_ = previous * (1.0 + 0.001 * np.sin(index / 7.0))
    high = np.maximum(open_, close) * 1.006
    low = np.minimum(open_, close) * 0.994
    true_range = np.maximum.reduce((high - low, np.abs(high - previous), np.abs(low - previous)))
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tr_pct": true_range / close,
            "atr_pct_14": np.full(rows, 0.012),
        }
    )


def test_probe_prefix_replay_and_future_extension_are_identical():
    bars = _bars()
    full, _ = build_state_coupling_probe(bars)
    prefix, _ = build_state_coupling_probe(bars.iloc[:1650])
    columns = [
        "date",
        "status",
        "regime_candidate",
        "event_candidates",
        "evidence_json",
        "threshold_abs_direction_p70",
        "threshold_range_p90",
    ]
    pd.testing.assert_frame_equal(
        full.loc[:1649, columns].reset_index(drop=True),
        prefix.loc[:, columns].reset_index(drop=True),
    )


def test_probe_derived_position_ranges_and_single_regime():
    result, _ = build_state_coupling_probe(_bars())
    ok = result[result["status"] == "ok"]
    assert ok["regime_candidate"].notna().all()
    assert ok["position_bias_20"].between(-1.0, 1.0).all()
    assert ok["position_bias_55"].between(-1.0, 1.0).all()
    assert ok["historical_coverage_20"].between(0.0, 1.0).all()
    assert ok["historical_coverage_55"].between(0.0, 1.0).all()