"""Build browser replay data without creating Phase 1A research artifacts."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from market_cycle.data import DEFAULT_SNAPSHOT_ID, get_research_bars
from market_cycle.measurements import (
    EFFICIENCY_WINDOWS,
    PATH_MULTIPLIERS,
    build_atr_reversal_paths,
    build_ohlc_min_efficiency,
)

REPLAY_SCHEMA_VERSION = "phase1a_replay_v1"


def _date(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _number(value: object) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return float(value)


def _multiplier_key(multiplier: float) -> str:
    return f"{multiplier:g}"


def _bars_records(bars: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "time": _date(row.date),
            "open": _number(row.open),
            "high": _number(row.high),
            "low": _number(row.low),
            "close": _number(row.close),
        }
        for row in bars.loc[:, ["date", "open", "high", "low", "close"]].itertuples(index=False)
    ]


def _efficiency_records(efficiency: pd.DataFrame, windows: Iterable[int]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in efficiency.itertuples(index=False):
        record: dict[str, object] = {"time": _date(row.date)}
        for window in windows:
            record[f"efficiency_{window}"] = _number(
                getattr(row, f"efficiency_ohlc_min_{window}")
            )
        records.append(record)
    return records


def _snapshot_records(snapshots: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in snapshots.itertuples(index=False):
        records.append(
            {
                "time": _date(row.date),
                "asOf": _date(row.as_of),
                "direction": row.direction,
                "anchorAt": _date(row.anchor_at),
                "anchorPrice": _number(row.anchor_price),
                "confirmedAt": _date(row.confirmed_at),
                "effectiveAt": _date(row.effective_at),
                "currentCandidateId": _number(row.current_candidate_id),
                "displacement": _number(row.since_anchor_displacement),
                "pathLengthMin": _number(row.since_anchor_path_length_min),
                "pathEfficiency": _number(row.since_anchor_path_efficiency),
                "pathEfficiencyStatus": row.since_anchor_path_efficiency_status,
                "maximumCounterMove": _number(row.maximum_counter_move),
                "candidateAgeBars": _number(row.candidate_age_bars),
            }
        )
    return records


_EVENT_ORDER = {
    "candidate_confirmed": 0,
    "candidate_invalidated": 1,
    "candidate_created": 2,
}


def _candidate_events(candidates: pd.DataFrame) -> list[dict[str, object]]:
    """Return causal candidate events, excluding unconfirmed warmup activity.

    The first confirmation can anchor the first formal path even though that
    candidate originated during warmup. Its confirmation is therefore exposed,
    while warmup candidate creation and invalidation stay internal.
    """
    events: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        candidate_id = _number(row.candidate_seq)
        details = {
            "candidateId": candidate_id,
            "direction": row.candidate_direction,
            "anchorType": row.anchor_type,
            "anchorAt": _date(row.anchor_at),
            "anchorPrice": _number(row.anchor_price),
            "threshold": _number(row.threshold),
        }
        if not row.is_warmup:
            events.append({"time": _date(row.candidate_at), "kind": "candidate_created", **details})
            if pd.notna(row.invalidated_at):
                events.append(
                    {
                        "time": _date(row.invalidated_at),
                        "kind": "candidate_invalidated",
                        "replacementCandidateId": _number(row.replacement_candidate_seq),
                        **details,
                    }
                )
        if pd.notna(row.confirmed_at):
            events.append(
                {
                    "time": _date(row.confirmed_at),
                    "kind": "candidate_confirmed",
                    **details,
                }
            )

    return sorted(
        events,
        key=lambda event: (str(event["time"]), _EVENT_ORDER[str(event["kind"])]),
    )


def _path_meta_record(meta: object) -> dict[str, object]:
    return {
        "pathId": meta.path_id,
        "multiplier": _number(meta.multiplier),
        "pathReadyAt": _date(meta.path_ready_at),
        "outputStart": _date(meta.output_start),
        "outputEnd": _date(meta.output_end),
        "snapshotRows": _number(meta.snapshot_rows),
        "candidateRows": _number(meta.candidate_rows),
    }


def build_phase1a_replay(
    bars: pd.DataFrame,
    *,
    snapshot_id: str,
    sample_id: str,
    windows: Sequence[int] = EFFICIENCY_WINDOWS,
    multipliers: Sequence[float] = PATH_MULTIPLIERS,
) -> dict[str, object]:
    """Build a complete causal replay bundle for the local audit page.

    This is an in-memory presentation projection. It does not persist daily
    measurement tables or candidate ledgers. The browser must only reduce rows
    and events with ``time <= as_of`` when rendering a replay frame.
    """
    efficiency, efficiency_meta = build_ohlc_min_efficiency(bars, windows=windows)
    path_results = build_atr_reversal_paths(bars, multipliers=multipliers)

    paths: dict[str, object] = {}
    for multiplier, result in path_results.items():
        paths[_multiplier_key(multiplier)] = {
            "meta": _path_meta_record(result.meta),
            "snapshots": _snapshot_records(result.snapshots),
            "events": _candidate_events(result.candidates),
        }

    return {
        "schemaVersion": REPLAY_SCHEMA_VERSION,
        "meta": {
            "snapshotId": snapshot_id,
            "sampleId": sample_id,
            "researchStart": _date(bars["date"].iloc[0]),
            "researchEnd": _date(bars["date"].iloc[-1]),
            "efficiencyId": efficiency_meta.efficiency_id,
            "efficiencyWindows": list(efficiency_meta.windows),
            "pathMultipliers": [float(multiplier) for multiplier in multipliers],
        },
        "bars": _bars_records(bars),
        "efficiency": _efficiency_records(efficiency, efficiency_meta.windows),
        "paths": paths,
    }


def load_phase1a_replay(snapshot_id: str = DEFAULT_SNAPSHOT_ID) -> dict[str, object]:
    """Load the pinned research table and construct the local replay bundle."""
    bars, ruler_meta = get_research_bars(snapshot_id)
    return build_phase1a_replay(
        bars,
        snapshot_id=ruler_meta.snapshot_id,
        sample_id=ruler_meta.sample_id,
    )