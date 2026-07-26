"""Tests for the in-memory Phase 1A continuous-behavior replay bundle."""

from __future__ import annotations

import pandas as pd

from market_cycle.audit.replay import REPLAY_SCHEMA_VERSION, build_phase1a_continuous_replay


def _bars() -> pd.DataFrame:
    rows = []
    price = 100.0
    for i in range(12):
        open_ = price
        high = price + 1.5
        low = price - 1.0
        close = price + (0.8 if i % 2 == 0 else -0.5)
        rows.append(
            (
                (pd.Timestamp("2024-01-02") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                open_,
                high,
                low,
                close,
                2.5 / close,
                0.012,
            )
        )
        price = close
    return pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "tr_pct", "atr_pct_14"],
    )


def test_continuous_replay_bundle_contains_only_base_layer_data():
    replay = build_phase1a_continuous_replay(
        _bars(),
        snapshot_id="test_snapshot",
        sample_id="test_sample",
        windows=(2, 3),
    )

    assert replay["schemaVersion"] == REPLAY_SCHEMA_VERSION
    assert replay["meta"]["snapshotId"] == "test_snapshot"
    assert replay["meta"]["sampleId"] == "test_sample"
    assert replay["meta"]["windows"] == [2, 3]
    assert replay["bars"][-1]["time"] is not None
    assert replay["efficiency"][0]["time"] is not None
    assert replay["direction"][0]["time"] is not None
    assert replay["twoSidedness"][0]["time"] is not None
    assert replay["expansion"][0]["time"] is not None
    assert replay["meta"]["expansionId"] is not None
    assert "paths" not in replay


def test_default_windows_include_composites():
    # Enough history for the default 55-day family.
    rows = []
    price = 100.0
    for i in range(60):
        price *= 1.001
        rows.append(
            (
                (pd.Timestamp("2024-01-02") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                price,
                price + 0.5,
                price - 0.5,
                price,
                1.0 / price,
                0.01,
            )
        )
    bars = pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "tr_pct", "atr_pct_14"],
    )
    replay = build_phase1a_continuous_replay(
        bars,
        snapshot_id="test_snapshot",
        sample_id="test_sample",
    )
    last_efficiency = replay["efficiency"][-1]
    last_direction = replay["direction"][-1]
    last_two = replay["twoSidedness"][-1]
    last_expansion = replay["expansion"][-1]
    assert last_efficiency["efficiency_equal"] is not None
    assert last_efficiency["efficiency_midlong"] is not None
    assert last_direction["direction_equal"] is not None
    assert last_direction["direction_midlong"] is not None
    assert last_two["two_sidedness_equal"] is not None
    assert last_two["two_sidedness_midlong"] is not None
    assert last_expansion["range"] is not None
    assert last_expansion["clearance_up_55"] is not None
