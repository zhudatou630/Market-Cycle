"""Tests for multi-scale Theil-Sen direction drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_cycle.data import get_research_bars
from market_cycle.measurements.direction import DIRECTION_ID, build_direction_drift


def _bars(rows: list[tuple[str, float, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "atr_pct_14"],
    )


def test_direction_matches_hand_calculated_theil_sen():
    # Steady 1% geometric rise with constant 1% ATR.
    closes = [100.0 * (1.01**i) for i in range(6)]
    rows = []
    for i, close in enumerate(closes):
        rows.append(
            (
                f"2024-01-{i + 2:02d}",
                close,
                close,
                close,
                close,
                0.01,
            )
        )
    frame, meta = build_direction_drift(_bars(rows), windows=(5,))

    assert meta.direction_id == DIRECTION_ID
    assert len(frame) == 1
    expected = np.log(1.01) / 0.01
    assert frame["direction_drift_5"].iloc[0] == pytest.approx(expected)
    assert frame["direction_drift_5_status"].iloc[0] == "ok"


def test_zero_atr_scale_is_missing():
    bars = _bars(
        [
            ("2024-01-02", 100.0, 100.0, 100.0, 100.0, 0.0),
            ("2024-01-03", 101.0, 101.0, 101.0, 101.0, 0.0),
            ("2024-01-04", 102.0, 102.0, 102.0, 102.0, 0.0),
        ]
    )
    frame, _ = build_direction_drift(bars, windows=(2,))
    assert np.isnan(frame["direction_drift_2"].iloc[0])
    assert frame["direction_drift_2_status"].iloc[0] == "zero_scale"


def test_composites_and_agreement_on_default_windows():
    # 56 rows so the 55-day window is ready.
    rows = []
    price = 100.0
    for i in range(56):
        price *= 1.001
        rows.append(
            (
                (pd.Timestamp("2024-01-02") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                price,
                price,
                price,
                price,
                0.01,
            )
        )
    frame, meta = build_direction_drift(_bars(rows))
    assert meta.windows == (5, 10, 20, 55)
    assert "direction_drift_equal_raw" in frame
    assert "direction_drift_midlong_raw" in frame
    assert "direction_scale_agreement_equal" in frame
    assert frame["direction_scale_agreement_equal"].iloc[-1] == pytest.approx(1.0)
    assert frame["direction_drift_equal_raw"].iloc[-1] > 0


def test_prefix_replay_matches_batch():
    rows = []
    price = 100.0
    for i in range(20):
        price *= 1.002 if i % 3 else 0.999
        rows.append(
            (
                (pd.Timestamp("2024-01-02") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                price,
                price * 1.001,
                price * 0.999,
                price,
                0.012,
            )
        )
    bars = _bars(rows)
    batch, _ = build_direction_drift(bars, windows=(3, 5))
    for end in range(6, len(bars) + 1):
        prefix, _ = build_direction_drift(bars.iloc[:end], windows=(3, 5))
        expected = batch.loc[batch["date"] == prefix["date"].iloc[-1]].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix.tail(1).reset_index(drop=True), expected)


def test_frozen_research_sample_direction_starts_with_efficiency():
    bars, _ = get_research_bars()
    frame, meta = build_direction_drift(bars)
    assert meta.analysis_start == pd.Timestamp("1984-05-01")
    assert frame["date"].iloc[0] == pd.Timestamp("1984-05-01")
    assert (frame["direction_drift_equal_raw_status"] == "ok").all()
    assert ((frame["direction_scale_agreement_equal"] >= 0.0) & (frame["direction_scale_agreement_equal"] <= 1.0)).all()
