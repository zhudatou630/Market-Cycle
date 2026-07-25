"""Tests for the in-memory Phase 1A chart replay bundle."""

from __future__ import annotations

import pandas as pd

from market_cycle.audit.replay import REPLAY_SCHEMA_VERSION, build_phase1a_replay


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("2024-01-02", 100.0, 100.0, 100.0, 100.0, 0.01),
            ("2024-01-03", 100.0, 103.0, 100.0, 103.0, 0.01),
            ("2024-01-04", 103.0, 106.0, 103.0, 105.0, 0.01),
            ("2024-01-05", 105.0, 105.0, 101.0, 103.0, 0.01),
            ("2024-01-08", 103.0, 104.0, 100.0, 101.0, 0.01),
            ("2024-01-09", 101.0, 104.0, 101.0, 103.0, 0.01),
            ("2024-01-10", 103.0, 106.0, 102.0, 105.0, 0.01),
        ],
        columns=["date", "open", "high", "low", "close", "atr_pct_14"],
    )


def test_replay_bundle_uses_time_stamped_events_without_terminal_ledger_fields():
    replay = build_phase1a_replay(
        _bars(),
        snapshot_id="test_snapshot",
        sample_id="test_sample",
        windows=(2,),
        multipliers=(2.0,),
    )

    assert replay["schemaVersion"] == REPLAY_SCHEMA_VERSION
    assert replay["meta"]["researchStart"] == "2024-01-02"
    assert replay["bars"][-1]["time"] == "2024-01-10"
    assert "status" not in replay

    path = replay["paths"]["2"]
    assert path["meta"]["pathReadyAt"] == "2024-01-05"
    assert path["snapshots"][0]["time"] == "2024-01-05"

    events = path["events"]
    first_confirmation = next(event for event in events if event["kind"] == "candidate_confirmed")
    assert first_confirmation["time"] == "2024-01-05"
    assert first_confirmation["anchorAt"] == "2024-01-04"
    assert all("confirmedAt" not in event for event in events)
    assert all("invalidatedAt" not in event for event in events)
    assert all("replacementCandidateSeq" not in event for event in events)

    created = [event for event in events if event["kind"] == "candidate_created"]
    assert created[0]["time"] == "2024-01-05"