"""Tests for the deferred ATR-threshold causal swing candidate."""

from __future__ import annotations

import pandas as pd
import pytest

from market_cycle.data import get_research_bars
from market_cycle.structures.atr_reversal import (
    PATH_ID,
    build_atr_reversal_path,
    build_atr_reversal_paths,
)


def _bars(rows: list[tuple[str, float, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "atr_pct_14"],
    )


def test_candidate_replacement_confirmation_and_anchor_efficiency():
    bars = _bars(
        [
            ("2024-01-02", 100.0, 100.0, 100.0, 100.0, 0.01),
            ("2024-01-03", 100.0, 103.0, 100.0, 103.0, 0.01),
            ("2024-01-04", 103.0, 106.0, 103.0, 105.0, 0.01),
            ("2024-01-05", 105.0, 105.0, 101.0, 103.0, 0.01),
            ("2024-01-08", 103.0, 104.0, 100.0, 101.0, 0.01),
        ]
    )
    result = build_atr_reversal_path(bars, multiplier=2.0)

    assert result.meta.path_id == PATH_ID
    assert result.meta.path_ready_at == pd.Timestamp("2024-01-05")
    first = result.snapshots.iloc[0]
    assert first["direction"] == "down"
    assert first["anchor_at"] == pd.Timestamp("2024-01-04")
    assert first["anchor_price"] == pytest.approx(106.0)
    assert first["confirmed_at"] == pd.Timestamp("2024-01-05")
    assert first["current_candidate_id"] != 2
    # Anchor partial = |105 - 106| = 1; day 3 L_min = 6; displacement = -3.
    assert first["since_anchor_path_length_min"] == pytest.approx(7.0)
    assert first["since_anchor_path_efficiency"] == pytest.approx(3.0 / 7.0)

    ledger = result.candidates.set_index("candidate_seq")
    assert ledger.loc[1, "status"] == "invalidated"
    assert ledger.loc[1, "invalidated_at"] == pd.Timestamp("2024-01-04")
    assert ledger.loc[2, "status"] == "confirmed"
    assert ledger.loc[2, "confirmed_at"] == pd.Timestamp("2024-01-05")
    # The new down-path candidate starts from confirmation Close=103, then is
    # replaced by the following day's Low=100.
    assert ledger.loc[3, "anchor_type"] == "close"
    assert ledger.loc[3, "status"] == "invalidated"
    assert ledger.loc[4, "anchor_type"] == "low"


def test_same_day_high_and_close_reversal_can_be_first_confirmation():
    bars = _bars(
        [
            ("2024-01-02", 100.0, 100.0, 100.0, 100.0, 0.01),
            ("2024-01-03", 100.0, 106.0, 100.0, 103.0, 0.01),
        ]
    )
    result = build_atr_reversal_path(bars, multiplier=2.0)

    assert result.meta.path_ready_at == pd.Timestamp("2024-01-03")
    snapshot = result.snapshots.iloc[0]
    assert snapshot["direction"] == "down"
    assert snapshot["anchor_at"] == pd.Timestamp("2024-01-03")
    assert snapshot["confirmed_at"] == pd.Timestamp("2024-01-03")
    assert snapshot["current_candidate_id"] == 2


def test_prefix_replay_matches_batch_snapshots():
    bars = _bars(
        [
            ("2024-01-02", 100.0, 100.0, 100.0, 100.0, 0.01),
            ("2024-01-03", 100.0, 103.0, 100.0, 103.0, 0.01),
            ("2024-01-04", 103.0, 106.0, 103.0, 105.0, 0.01),
            ("2024-01-05", 105.0, 105.0, 101.0, 103.0, 0.01),
            ("2024-01-08", 103.0, 104.0, 100.0, 101.0, 0.01),
            ("2024-01-09", 101.0, 104.0, 101.0, 103.0, 0.01),
            ("2024-01-10", 103.0, 106.0, 102.0, 105.0, 0.01),
        ]
    )
    batch = build_atr_reversal_path(bars, multiplier=2.0)

    for end in range(4, len(bars) + 1):
        prefix = build_atr_reversal_path(bars.iloc[:end], multiplier=2.0)
        if prefix.snapshots.empty:
            continue
        expected = batch.snapshots.loc[
            batch.snapshots["date"] == prefix.snapshots["date"].iloc[-1]
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix.snapshots.tail(1).reset_index(drop=True), expected)


def test_frozen_sample_path_ready_dates_and_common_start():
    bars, _ = get_research_bars()
    results = build_atr_reversal_paths(bars)

    assert results[1.0].meta.path_ready_at == pd.Timestamp("1984-02-23")
    assert results[2.0].meta.path_ready_at == pd.Timestamp("1984-04-05")
    assert results[3.0].meta.path_ready_at == pd.Timestamp("1984-05-14")
    assert all(result.snapshots["as_of"].eq(result.snapshots["date"]).all() for result in results.values())
    assert all(
        ((result.snapshots["since_anchor_path_efficiency"] >= 0.0)
         & (result.snapshots["since_anchor_path_efficiency"] <= 1.0)).all()
        for result in results.values()
    )