"""Tests for Phase 1A ruler (D-010)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_cycle.data.ruler import (
    ATR_N,
    ATR_WARMUP_TR_BARS,
    SAMPLE_ID,
    build_ruler,
    classic_true_range,
    get_research_bars,
    load_research_bars,
    research_bars_path,
    wilder_atr,
    write_research_bars,
)
from market_cycle.data.snapshot import DEFAULT_SNAPSHOT_ID, load_snapshot


def test_classic_true_range_matches_hand_example():
    high = np.array([10.0, 12.0, 11.0])
    low = np.array([9.0, 10.0, 9.5])
    close = np.array([9.5, 11.0, 10.0])
    tr = classic_true_range(high, low, close)
    assert np.isnan(tr[0])
    # day1: max(2, |12-9.5|, |10-9.5|) = max(2, 2.5, 0.5) = 2.5
    assert tr[1] == pytest.approx(2.5)
    # day2: max(1.5, |11-11|, |9.5-11|) = max(1.5, 0, 1.5) = 1.5
    assert tr[2] == pytest.approx(1.5)


def test_wilder_atr_seed_and_step():
    tr = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    # n=3 → seed at 3rd finite TR (value positions 1,2,3) mean=2.0 at index 3
    atr = wilder_atr(tr, n=3)
    assert np.isnan(atr[0]) and np.isnan(atr[1]) and np.isnan(atr[2])
    assert atr[3] == pytest.approx(2.0)
    assert atr[4] == pytest.approx(((3 - 1) * 2.0 + 4.0) / 3)
    assert atr[5] == pytest.approx(((3 - 1) * atr[4] + 5.0) / 3)


def test_build_ruler_research_start_and_no_nulls():
    frame, meta = build_ruler(DEFAULT_SNAPSHOT_ID)
    assert meta.snapshot_id == DEFAULT_SNAPSHOT_ID
    assert meta.sample_id == SAMPLE_ID
    assert meta.atr_n == ATR_N
    assert meta.atr_warmup_tr_bars == ATR_WARMUP_TR_BARS
    assert meta.research_start == pd.Timestamp("1984-02-10")
    assert meta.research_end == pd.Timestamp("2026-07-21")
    assert list(frame.columns) == ["date", "tr_pct", "atr_pct_14"]
    assert len(frame) == meta.rows == 10692
    assert frame["date"].iloc[0] == meta.research_start
    assert not frame[["tr_pct", "atr_pct_14"]].isna().any().any()
    assert (frame["tr_pct"] > 0).all()
    assert (frame["atr_pct_14"] > 0).all()
    # Public frame must not expose absolute intermediates
    assert "tr" not in frame.columns and "atr" not in frame.columns


def test_ruler_relative_to_same_day_close():
    """tr_pct = TR/Close_t on research rows (spot-check first research day)."""
    raw = load_snapshot(DEFAULT_SNAPSHOT_ID)
    work = raw.loc[raw["date"] >= "1984-01-03"].reset_index(drop=True)
    high = work["high"].to_numpy(float)
    low = work["low"].to_numpy(float)
    close = work["close"].to_numpy(float)
    tr = classic_true_range(high, low, close)
    finite = np.flatnonzero(np.isfinite(tr))
    i = int(finite[ATR_WARMUP_TR_BARS - 1])
    frame, _ = build_ruler(DEFAULT_SNAPSHOT_ID)
    assert frame["date"].iloc[0] == work["date"].iloc[i]
    assert frame["tr_pct"].iloc[0] == pytest.approx(tr[i] / close[i])


def test_get_research_bars_materialized_same_level_as_ohlc():
    path = research_bars_path(DEFAULT_SNAPSHOT_ID)
    bars, meta = get_research_bars(DEFAULT_SNAPSHOT_ID, refresh=True)
    assert path.exists()
    assert meta.research_start == pd.Timestamp("1984-02-10")
    assert list(bars.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "tr_pct",
        "atr_pct_14",
    ]
    assert len(bars) == 10692
    assert not bars.isna().any().any()

    loaded, loaded_meta = load_research_bars(DEFAULT_SNAPSHOT_ID)
    assert loaded_meta.rows == meta.rows
    pd.testing.assert_frame_equal(bars, loaded)

    # default get reads file, matches recompute
    again, _ = get_research_bars(DEFAULT_SNAPSHOT_ID, refresh=False)
    pd.testing.assert_frame_equal(bars, again)

    # recompute path still matches file content
    recomputed, _, _ = write_research_bars(DEFAULT_SNAPSHOT_ID)
    pd.testing.assert_frame_equal(bars, recomputed)
