"""Audit statistics and artifact writing for state-coupling probe v0a."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_cycle.prototypes.state_coupling_probe import StateCouplingProbeMeta


def _run_lengths(values: pd.Series) -> list[int]:
    if values.empty:
        return []
    groups = values.ne(values.shift()).cumsum()
    return values.groupby(groups).size().astype(int).tolist()


def summarize_probe(frame: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    segment = frame.loc[
        frame["date"].between(pd.Timestamp(start), pd.Timestamp(end)) & frame["status"].eq("ok")
    ].copy()
    regimes = segment["regime_candidate"]
    counts = regimes.value_counts().to_dict()
    run_lengths = _run_lengths(regimes)
    switches = regimes.ne(regimes.shift())
    if not switches.empty:
        switches.iloc[0] = False
    rapid = 0
    opposite = {"DIRECTIONAL_UP": "DIRECTIONAL_DOWN", "DIRECTIONAL_DOWN": "DIRECTIONAL_UP"}
    for left, middle, right in zip(regimes.iloc[:-2], regimes.iloc[1:-1], regimes.iloc[2:]):
        if left == right and opposite.get(left) == middle:
            rapid += 1
    event_counter: Counter[str] = Counter()
    for value in segment["event_candidates"]:
        event_counter.update(json.loads(value))

    near_switches = 0
    switch_rows = segment.loc[switches]
    background_rules = {
        "direction_strength_high",
        "direction_strength_low",
        "scale_consistent",
        "path_directness_low",
        "entropy_contest_high",
        "overlap_contest_high",
        "coverage_20_high",
    }
    for evidence_value in switch_rows["evidence_json"]:
        evidence = json.loads(evidence_value)
        near = any(
            item["margin"] is not None
            and item["threshold"] not in (None, 0)
            and abs(item["margin"] / item["threshold"]) <= 0.05
            for key, item in evidence.items()
            if key in background_rules
        )
        near_switches += int(near)

    annual_switches = (
        segment.assign(switch=switches, year=segment["date"].dt.year)
        .groupby("year")["switch"]
        .sum()
        .astype(int)
        .to_dict()
    )
    rows = len(segment)
    return {
        "start": start,
        "end": end,
        "rows": rows,
        "regime_counts": counts,
        "regime_shares": {key: value / rows for key, value in counts.items()} if rows else {},
        "event_counts": dict(event_counter),
        "event_shares": {key: value / rows for key, value in event_counter.items()} if rows else {},
        "run_length_mean": float(np.mean(run_lengths)) if run_lengths else None,
        "run_length_median": float(np.median(run_lengths)) if run_lengths else None,
        "one_day_run_share": float(np.mean(np.asarray(run_lengths) == 1)) if run_lengths else None,
        "switches": int(switches.sum()),
        "rapid_direction_reversals": rapid,
        "near_threshold_switches": near_switches,
        "near_threshold_switch_share": near_switches / int(switches.sum()) if switches.sum() else None,
        "annual_switches": annual_switches,
        "scale_conflict_share": float(segment["scale_conflict"].mean()) if rows else None,
    }


def compare_labels(primary: pd.DataFrame, variant: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    mask = (
        primary["date"].between(pd.Timestamp(start), pd.Timestamp(end))
        & primary["status"].eq("ok")
        & variant["status"].eq("ok")
    )
    changed_regime = primary.loc[mask, "regime_candidate"] != variant.loc[mask, "regime_candidate"]
    changed_event = primary.loc[mask, "event_candidates"] != variant.loc[mask, "event_candidates"]
    return {
        "rows": int(mask.sum()),
        "regime_changed": int(changed_regime.sum()),
        "regime_changed_share": float(changed_regime.mean()),
        "event_changed": int(changed_event.sum()),
        "event_changed_share": float(changed_event.mean()),
    }


def summarize_information_steps(frame: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    segment = frame.loc[
        frame["date"].between(pd.Timestamp(start), pd.Timestamp(end)) & frame["status"].eq("ok")
    ]
    evidence = segment["evidence_json"].map(json.loads)
    direction = evidence.map(
        lambda value: value["direction_strength_high"]["hit"] and value["scale_consistent"]["hit"]
    )
    weak = evidence.map(
        lambda value: value["direction_strength_low"]["hit"] and value["scale_consistent"]["hit"]
    )
    low_efficiency = evidence.map(lambda value: value["path_directness_low"]["hit"])
    contest = evidence.map(
        lambda value: value["entropy_contest_high"]["hit"]
        or value["overlap_contest_high"]["hit"]
    )
    coverage = evidence.map(lambda value: value["coverage_20_high"]["hit"])
    pre_clearance_balance = ~direction & weak & low_efficiency & contest
    final_balance = segment["regime_candidate"].eq("BALANCE_LIKE")
    events = segment["event_candidates"].ne("[]")
    return {
        "step1_directional_dates": int(direction.sum()),
        "step1_scale_conflict_dates": int(segment["scale_conflict"].sum()),
        "step2_low_directness_dates": int(low_efficiency.sum()),
        "step3_balance_explainable_without_clearance": int(pre_clearance_balance.sum()),
        "step4_event_dates": int(events.sum()),
        "step4_activity_level_counts": segment["activity_attribute"].value_counts().to_dict(),
        "step5_balance_dates_added_or_recovered_by_coverage": int((final_balance & ~pre_clearance_balance).sum()),
        "step5_final_balance_dates": int(final_balance.sum()),
    }


def diagnostic_dates(frame: pd.DataFrame, start: str, end: str, limit: int = 12) -> dict[str, list[str]]:
    segment = frame.loc[
        frame["date"].between(pd.Timestamp(start), pd.Timestamp(end)) & frame["status"].eq("ok")
    ].copy()
    changed = segment["regime_candidate"].ne(segment["regime_candidate"].shift())
    changed.iloc[0] = False
    singleton = changed & segment["regime_candidate"].ne(segment["regime_candidate"].shift(-1))
    rapid_mask = pd.Series(False, index=segment.index)
    values = segment["regime_candidate"].tolist()
    for offset in range(1, len(values) - 1):
        opposite = {
            "DIRECTIONAL_UP": "DIRECTIONAL_DOWN",
            "DIRECTIONAL_DOWN": "DIRECTIONAL_UP",
        }
        if values[offset - 1] == values[offset + 1] and opposite.get(values[offset - 1]) == values[offset]:
            rapid_mask.iloc[offset] = True
    return {
        "one_day_regime_dates": segment.loc[singleton, "date"].dt.date.astype(str).head(limit).tolist(),
        "rapid_direction_reversal_middle_dates": segment.loc[rapid_mask, "date"].dt.date.astype(str).head(limit).tolist(),
        "event_dates": segment.loc[segment["event_candidates"].ne("[]"), "date"].dt.date.astype(str).head(limit).tolist(),
    }


def write_probe_artifacts(
    frame: pd.DataFrame,
    meta: StateCouplingProbeMeta,
    *,
    output_dir: Path,
    snapshot_id: str,
    sample_id: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{snapshot_id}__{meta.probe_version}"
    parquet_path = output_dir / f"{stem}.parquet"
    metadata_path = output_dir / f"{stem}.json"
    frame.to_parquet(parquet_path, index=False)
    digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    payload = {
        **asdict(meta),
        "input_start": meta.input_start.date().isoformat(),
        "output_start": meta.output_start.date().isoformat(),
        "output_end": meta.output_end.date().isoformat(),
        "snapshot_id": snapshot_id,
        "sample_id": sample_id,
        "parquet_path": str(parquet_path),
        "parquet_sha256": digest,
        "columns": list(frame.columns),
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return parquet_path, metadata_path