"""Tests for multi-scale two-sidedness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_cycle.data import get_research_bars
from market_cycle.measurements.two_sidedness import (
    TWO_SIDEDNESS_ID,
    build_two_sidedness,
)


def _bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])


def test_alternating_closes_have_high_entropy_and_overlap():
    bars = _bars(
        [
            ("2024-01-02", 100.0, 101.0, 99.0, 100.0),
            ("2024-01-03", 100.0, 101.0, 99.0, 101.0),
            ("2024-01-04", 101.0, 102.0, 100.0, 100.0),
            ("2024-01-05", 100.0, 101.0, 99.0, 101.0),
            ("2024-01-08", 101.0, 102.0, 100.0, 100.0),
        ]
    )
    frame, meta = build_two_sidedness(bars, windows=(4,))
    assert meta.two_sidedness_id == TWO_SIDEDNESS_ID
    row = frame.iloc[-1]
    assert row["two_sidedness_entropy_4"] == pytest.approx(1.0)
    assert row["two_sidedness_overlap_4"] == pytest.approx(0.5)
    assert 0.0 < row["two_sidedness_v1_4"] <= 1.0
    assert row["two_sidedness_v1_4_status"] == "ok"


def test_one_sided_trend_has_low_two_sidedness():
    # Non-overlapping rising ranges: no re-trading, all up closes.
    bars = _bars(
        [
            ("2024-01-02", 100.0, 101.0, 100.0, 101.0),
            ("2024-01-03", 102.0, 103.0, 102.0, 103.0),
            ("2024-01-04", 104.0, 105.0, 104.0, 105.0),
            ("2024-01-05", 106.0, 107.0, 106.0, 107.0),
            ("2024-01-08", 108.0, 109.0, 108.0, 109.0),
        ]
    )
    frame, _ = build_two_sidedness(bars, windows=(4,))
    row = frame.iloc[-1]
    assert row["two_sidedness_entropy_4"] == pytest.approx(0.0)
    assert row["two_sidedness_overlap_4"] == pytest.approx(0.0)
    assert row["two_sidedness_v1_4"] == pytest.approx(0.0)


def test_zero_union_overlap_is_missing():
    bars = _bars(
        [
            ("2024-01-02", 100.0, 100.0, 100.0, 100.0),
            ("2024-01-03", 100.0, 100.0, 100.0, 100.0),
            ("2024-01-04", 100.0, 100.0, 100.0, 100.0),
        ]
    )
    frame, _ = build_two_sidedness(bars, windows=(2,))
    assert np.isnan(frame["two_sidedness_overlap_2"].iloc[0])
    assert np.isnan(frame["two_sidedness_v1_2"].iloc[0])
    assert frame["two_sidedness_v1_2_status"].iloc[0] == "zero_range"
    assert frame["two_sidedness_entropy_2"].iloc[0] == pytest.approx(0.0)


def test_prefix_replay_matches_batch():
    bars = _bars(
        [
            ("2024-01-02", 100.0, 101.0, 99.0, 100.0),
            ("2024-01-03", 100.5, 102.0, 99.5, 101.5),
            ("2024-01-04", 101.0, 103.0, 100.0, 100.5),
            ("2024-01-05", 100.0, 101.5, 99.0, 101.0),
            ("2024-01-08", 101.0, 102.5, 100.0, 100.2),
            ("2024-01-09", 100.0, 101.0, 98.5, 99.5),
            ("2024-01-10", 99.0, 101.0, 98.0, 100.5),
        ]
    )
    batch, _ = build_two_sidedness(bars, windows=(2, 3))
    for end in range(4, len(bars) + 1):
        prefix, _ = build_two_sidedness(bars.iloc[:end], windows=(2, 3))
        expected = batch.loc[batch["date"] == prefix["date"].iloc[-1]].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix.tail(1).reset_index(drop=True), expected)


def test_frozen_research_sample_two_sidedness_range():
    bars, _ = get_research_bars()
    frame, meta = build_two_sidedness(bars)
    assert meta.analysis_start == pd.Timestamp("1984-05-01")
    value_columns = [
        column
        for column in frame.columns
        if column.startswith("two_sidedness_v1_") and not column.endswith("_status")
    ]
    assert ((frame[value_columns] >= 0.0) & (frame[value_columns] <= 1.0)).all().all()
    assert (frame["two_sidedness_equal_v1_status"] == "ok").all()
