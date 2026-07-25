"""Tests for the in-memory Phase 1A continuous-behavior replay bundle."""

from __future__ import annotations

import pandas as pd

from market_cycle.audit.replay import REPLAY_SCHEMA_VERSION, build_phase1a_continuous_replay


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("2024-01-02", 100.0, 100.0, 100.0, 100.0),
            ("2024-01-03", 100.0, 103.0, 100.0, 103.0),
            ("2024-01-04", 103.0, 106.0, 103.0, 105.0),
            ("2024-01-05", 105.0, 105.0, 101.0, 103.0),
            ("2024-01-08", 103.0, 104.0, 100.0, 101.0),
            ("2024-01-09", 101.0, 104.0, 101.0, 103.0),
            ("2024-01-10", 103.0, 106.0, 102.0, 105.0),
        ],
        columns=["date", "open", "high", "low", "close"],
    )


def test_continuous_replay_bundle_contains_only_base_layer_data():
    replay = build_phase1a_continuous_replay(
        _bars(),
        snapshot_id="test_snapshot",
        sample_id="test_sample",
        windows=(2,),
    )

    assert replay["schemaVersion"] == REPLAY_SCHEMA_VERSION
    assert replay["meta"] == {
        "snapshotId": "test_snapshot",
        "sampleId": "test_sample",
        "researchStart": "2024-01-02",
        "researchEnd": "2024-01-10",
        "efficiencyId": "bm_e_01_ohlc_min_v1",
        "efficiencyWindows": [2],
    }
    assert replay["bars"][-1]["time"] == "2024-01-10"
    assert replay["efficiency"][0]["time"] == "2024-01-04"
    assert replay["efficiency"][0]["efficiency_2"] is not None
    assert "paths" not in replay
