"""Tests for state-coupling audit statistics."""

from __future__ import annotations

import json

import pandas as pd

from market_cycle.prototypes.state_coupling_analysis import summarize_probe


def test_background_switch_proximity_ignores_event_only_thresholds():
    background_rule = {
        "raw_value": 1.0,
        "threshold": 2.0,
        "operator": ">=",
        "hit": False,
        "margin": -1.0,
    }
    event_rule = {
        "raw_value": 0.99,
        "threshold": 1.0,
        "operator": ">=",
        "hit": False,
        "margin": -0.01,
    }
    evidence = {
        "direction_strength_high": background_rule,
        "direction_strength_low": background_rule,
        "scale_consistent": background_rule,
        "path_directness_low": background_rule,
        "entropy_contest_high": background_rule,
        "overlap_contest_high": background_rule,
        "coverage_20_high": background_rule,
        "range_extreme": event_rule,
        "share_high": event_rule,
        "share_low": event_rule,
    }
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2000-01-03", "2000-01-04"]),
            "status": ["ok", "ok"],
            "regime_candidate": ["UNRESOLVED", "BALANCE_LIKE"],
            "event_candidates": ["[]", "[]"],
            "scale_conflict": [False, False],
            "evidence_json": [json.dumps(evidence), json.dumps(evidence)],
        }
    )

    summary = summarize_probe(frame, "2000-01-01", "2000-01-31")

    assert summary["switches"] == 1
    assert summary["near_threshold_switches"] == 0