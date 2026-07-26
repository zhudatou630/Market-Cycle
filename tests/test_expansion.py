"""Tests for Phase 1A daily expansion impulse and clearance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_cycle.data import get_research_bars
from market_cycle.measurements.expansion import (
    CLEARANCE_ID,
    EXPANSION_ID,
    build_expansion_impulse,
)


def _bars(rows: list[tuple[str, float, float, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "tr_pct", "atr_pct_14"],
    )


def test_impulse_matches_hand_calculation_with_strict_gap():
    bars = _bars(
        [
            ("2024-01-02", 100.0, 101.0, 99.0, 100.0, 2.0 / 100.0, 0.02),
            # TR = max(104 - 102, 106 - 100, 100 - 102) = 6.
            ("2024-01-03", 103.0, 106.0, 102.0, 105.0, 6.0 / 105.0, 0.02),
        ]
    )
    frame, meta = build_expansion_impulse(bars, clearance_windows=(1,))

    assert meta.expansion_id == EXPANSION_ID
    assert meta.clearance_id == CLEARANCE_ID
    assert meta.analysis_start == pd.Timestamp("2024-01-03")
    row = frame.iloc[0]
    assert row["activity_level_atr_pct_prev"] == pytest.approx(0.02)
    assert row["expansion_range_atr_prev"] == pytest.approx(3.0)
    assert row["expansion_close_atr_prev"] == pytest.approx(2.5)
    assert row["expansion_gap_prev_range_atr"] == pytest.approx(1.0)
    assert row["expansion_close_share"] == pytest.approx(5.0 / 6.0)
    assert row["expansion_range_atr_prev_status"] == "ok"
    assert row["expansion_close_share_status"] == "ok"


def test_gap_requires_open_outside_previous_complete_range():
    bars = _bars(
        [
            ("2024-01-02", 100.0, 102.0, 98.0, 100.0, 4.0 / 100.0, 0.02),
            # The open is above the prior close but remains inside [98, 102].
            ("2024-01-03", 101.0, 103.0, 100.0, 102.0, 3.0 / 102.0, 0.02),
        ]
    )
    frame, _ = build_expansion_impulse(bars, clearance_windows=(1,))
    assert frame["expansion_gap_prev_range_atr"].iloc[0] == pytest.approx(0.0)


def test_share_is_missing_for_zero_true_range_and_scale_fields_remain_defined():
    bars = _bars(
        [
            ("2024-01-02", 100.0, 100.0, 100.0, 100.0, 0.0, 0.02),
            ("2024-01-03", 100.0, 100.0, 100.0, 100.0, 0.0, 0.02),
        ]
    )
    frame, _ = build_expansion_impulse(bars, clearance_windows=(1,))
    row = frame.iloc[0]
    assert row["expansion_range_atr_prev"] == pytest.approx(0.0)
    assert row["expansion_close_atr_prev"] == pytest.approx(0.0)
    assert row["expansion_gap_prev_range_atr"] == pytest.approx(0.0)
    assert np.isnan(row["expansion_close_share"])
    assert row["expansion_close_share_status"] == "zero_range"


def test_zero_previous_atr_marks_only_scale_dependent_fields_missing():
    bars = _bars(
        [
            ("2024-01-02", 100.0, 100.0, 100.0, 100.0, 0.0, 0.0),
            ("2024-01-03", 101.0, 102.0, 100.0, 101.0, 2.0 / 101.0, 0.02),
        ]
    )
    frame, _ = build_expansion_impulse(bars, clearance_windows=(1,))
    row = frame.iloc[0]
    assert row["activity_level_atr_pct_prev"] == pytest.approx(0.0)
    assert row["activity_level_atr_pct_prev_status"] == "ok"
    for column in (
        "expansion_range_atr_prev",
        "expansion_close_atr_prev",
        "expansion_gap_prev_range_atr",
    ):
        assert np.isnan(row[column])
        assert row[f"{column}_status"] == "zero_scale"
    assert row["expansion_close_share"] == pytest.approx(0.5)
    assert row["expansion_close_share_status"] == "ok"


def test_clearance_uses_strict_current_close_against_prior_complete_ranges():
    bars = _bars(
        [
            ("2024-01-02", 9.0, 10.0, 8.0, 9.0, 2.0 / 9.0, 0.02),
            ("2024-01-03", 10.0, 12.0, 9.0, 10.0, 3.0 / 10.0, 0.02),
            # Current close 11 clears the first bar but not the second bar high.
            ("2024-01-04", 10.0, 11.5, 9.5, 11.0, 2.0 / 11.0, 0.02),
            # Current close equals the first prior low and clears only the second down range.
            ("2024-01-05", 9.0, 9.5, 7.5, 9.0, 3.5 / 9.0, 0.02),
        ]
    )
    frame, meta = build_expansion_impulse(bars, clearance_windows=(2,))

    assert meta.clearance_start == pd.Timestamp("2024-01-04")
    row_up = frame.loc[frame["date"] == pd.Timestamp("2024-01-04")].iloc[0]
    assert row_up["prior_range_clearance_up_2"] == pytest.approx(0.5)
    assert row_up["prior_range_clearance_down_2"] == pytest.approx(0.0)

    row_down = frame.loc[frame["date"] == pd.Timestamp("2024-01-05")].iloc[0]
    assert row_down["prior_range_clearance_up_2"] == pytest.approx(0.0)
    assert row_down["prior_range_clearance_down_2"] == pytest.approx(0.5)


def test_atr_normalized_fields_are_invariant_to_price_scale():
    bars = _bars(
        [
            ("2024-01-02", 100.0, 101.0, 99.0, 100.0, 2.0 / 100.0, 0.02),
            ("2024-01-03", 103.0, 106.0, 102.0, 105.0, 6.0 / 105.0, 0.02),
            ("2024-01-04", 105.0, 106.0, 101.0, 102.0, 5.0 / 102.0, 0.02),
        ]
    )
    scaled = bars.copy()
    for column in ("open", "high", "low", "close"):
        scaled[column] *= 10.0

    baseline, _ = build_expansion_impulse(bars, clearance_windows=(1,))
    rescaled, _ = build_expansion_impulse(scaled, clearance_windows=(1,))
    columns = (
        "activity_level_atr_pct_prev",
        "expansion_range_atr_prev",
        "expansion_close_atr_prev",
        "expansion_gap_prev_range_atr",
        "expansion_close_share",
        "prior_range_clearance_up_1",
        "prior_range_clearance_down_1",
    )
    pd.testing.assert_frame_equal(baseline.loc[:, columns], rescaled.loc[:, columns])


def test_prefix_replay_matches_batch():
    rows = []
    previous_close = 100.0
    for i in range(12):
        open_ = previous_close * (1.006 if i % 4 == 0 else 0.998)
        high = max(open_, previous_close) * 1.012
        low = min(open_, previous_close) * 0.989
        close = open_ * (1.008 if i % 3 else 0.994)
        tr = max(high - low, abs(high - previous_close), abs(low - previous_close))
        rows.append(
            (
                (pd.Timestamp("2024-01-02") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                open_,
                high,
                low,
                close,
                tr / close,
                0.012,
            )
        )
        previous_close = close
    bars = _bars(rows)
    batch, _ = build_expansion_impulse(bars, clearance_windows=(3, 5))

    for end in range(2, len(bars) + 1):
        prefix, _ = build_expansion_impulse(bars.iloc[:end], clearance_windows=(3, 5))
        expected = batch.loc[batch["date"] == prefix["date"].iloc[-1]].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix.tail(1).reset_index(drop=True), expected)


def test_frozen_research_sample_starts_early_and_clearance_aligns_with_edb_family():
    bars, _ = get_research_bars()
    frame, meta = build_expansion_impulse(bars)

    assert meta.analysis_start == pd.Timestamp("1984-02-13")
    assert meta.clearance_start == pd.Timestamp("1984-05-01")
    assert frame["date"].iloc[0] == meta.analysis_start
    assert frame["date"].iloc[-1] == pd.Timestamp("2026-07-21")
    assert (frame["expansion_range_atr_prev"] >= 0.0).all()
    assert (
        frame["expansion_close_atr_prev"].abs()
        <= frame["expansion_range_atr_prev"] + 1e-12
    ).all()
    for direction in ("up", "down"):
        for window in (20, 55):
            values = frame[f"prior_range_clearance_{direction}_{window}"].dropna()
            assert ((values >= 0.0) & (values <= 1.0)).all()