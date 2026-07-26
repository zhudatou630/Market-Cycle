"""Build an in-memory Phase 1A continuous-behavior replay bundle."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from market_cycle.data import DEFAULT_SNAPSHOT_ID, get_research_bars
from market_cycle.measurements import (
    CONTINUOUS_WINDOWS,
    build_expansion_impulse,
    build_direction_drift,
    build_ohlc_min_efficiency,
    build_two_sidedness,
)

REPLAY_SCHEMA_VERSION = "phase1a_continuous_replay_v3"


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


def _feature_records(
    frame: pd.DataFrame,
    *,
    windows: Iterable[int],
    scale_prefix: str,
    equal_column: str,
    midlong_column: str,
    extra_columns: Sequence[str] = (),
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        record: dict[str, object] = {"time": _date(row.date)}
        for window in windows:
            record[f"{scale_prefix}_{window}"] = _number(getattr(row, f"{scale_prefix}_{window}"))
        record[f"{scale_prefix}_equal"] = _number(getattr(row, equal_column))
        record[f"{scale_prefix}_midlong"] = _number(getattr(row, midlong_column))
        for column in extra_columns:
            record[column] = _number(getattr(row, column))
        records.append(record)
    return records


def _expansion_records(frame: pd.DataFrame, *, clearance_windows: Iterable[int]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        record: dict[str, object] = {
            "time": _date(row.date),
            "activity_level": _number(row.activity_level_atr_pct_prev),
            "range": _number(row.expansion_range_atr_prev),
            "close": _number(row.expansion_close_atr_prev),
            "gap": _number(row.expansion_gap_prev_range_atr),
            "share": _number(row.expansion_close_share),
        }
        for window in clearance_windows:
            record[f"clearance_up_{window}"] = _number(
                getattr(row, f"prior_range_clearance_up_{window}")
            )
            record[f"clearance_down_{window}"] = _number(
                getattr(row, f"prior_range_clearance_down_{window}")
            )
        records.append(record)
    return records


def build_phase1a_continuous_replay(
    bars: pd.DataFrame,
    *,
    snapshot_id: str,
    sample_id: str,
    windows: Sequence[int] = CONTINUOUS_WINDOWS,
) -> dict[str, object]:
    """Build the local replay bundle for Phase 1A continuous behavior only.

    The bundle contains daily OHLC plus multi-scale efficiency, direction,
    two-sidedness, and daily expansion values. Causal swing candidates live in
    ``market_cycle.structures`` and are intentionally excluded from the Phase 1A
    base-layer audit surface.
    """
    efficiency, efficiency_meta = build_ohlc_min_efficiency(bars, windows=windows)
    direction, direction_meta = build_direction_drift(bars, windows=windows)
    two_sidedness, two_sidedness_meta = build_two_sidedness(bars, windows=windows)
    expansion, expansion_meta = build_expansion_impulse(bars)

    if list(windows) != list(CONTINUOUS_WINDOWS):
        # Custom window families are supported for unit tests, but they do not
        # carry the frozen equal/midlong compression fields used by the audit UI.
        efficiency_records = [
            {
                "time": _date(row.date),
                **{
                    f"efficiency_{window}": _number(getattr(row, f"efficiency_ohlc_min_{window}"))
                    for window in windows
                },
            }
            for row in efficiency.itertuples(index=False)
        ]
        direction_records = [
            {
                "time": _date(row.date),
                **{
                    f"direction_{window}": _number(getattr(row, f"direction_drift_{window}"))
                    for window in windows
                },
            }
            for row in direction.itertuples(index=False)
        ]
        two_sidedness_records = [
            {
                "time": _date(row.date),
                **{
                    f"two_sidedness_{window}": _number(getattr(row, f"two_sidedness_v1_{window}"))
                    for window in windows
                },
            }
            for row in two_sidedness.itertuples(index=False)
        ]
    else:
        efficiency_records = _feature_records(
            efficiency,
            windows=windows,
            scale_prefix="efficiency_ohlc_min",
            equal_column="efficiency_ohlc_min_equal",
            midlong_column="efficiency_ohlc_min_midlong",
        )
        # Rewrite scale keys to the shorter audit names used by the page.
        efficiency_records = [
            {
                "time": row["time"],
                **{f"efficiency_{window}": row[f"efficiency_ohlc_min_{window}"] for window in windows},
                "efficiency_equal": row["efficiency_ohlc_min_equal"],
                "efficiency_midlong": row["efficiency_ohlc_min_midlong"],
            }
            for row in efficiency_records
        ]
        direction_records = _feature_records(
            direction,
            windows=windows,
            scale_prefix="direction_drift",
            equal_column="direction_drift_equal_raw",
            midlong_column="direction_drift_midlong_raw",
            extra_columns=(
                "direction_scale_agreement_equal",
                "direction_scale_agreement_midlong",
            ),
        )
        direction_records = [
            {
                "time": row["time"],
                **{f"direction_{window}": row[f"direction_drift_{window}"] for window in windows},
                "direction_equal": row["direction_drift_equal"],
                "direction_midlong": row["direction_drift_midlong"],
                "direction_agreement_equal": row["direction_scale_agreement_equal"],
                "direction_agreement_midlong": row["direction_scale_agreement_midlong"],
            }
            for row in direction_records
        ]
        two_sidedness_records = _feature_records(
            two_sidedness,
            windows=windows,
            scale_prefix="two_sidedness_v1",
            equal_column="two_sidedness_equal_v1",
            midlong_column="two_sidedness_midlong_v1",
        )
        two_sidedness_records = [
            {
                "time": row["time"],
                **{f"two_sidedness_{window}": row[f"two_sidedness_v1_{window}"] for window in windows},
                "two_sidedness_equal": row["two_sidedness_v1_equal"],
                "two_sidedness_midlong": row["two_sidedness_v1_midlong"],
            }
            for row in two_sidedness_records
        ]

    return {
        "schemaVersion": REPLAY_SCHEMA_VERSION,
        "meta": {
            "snapshotId": snapshot_id,
            "sampleId": sample_id,
            "researchStart": _date(bars["date"].iloc[0]),
            "researchEnd": _date(bars["date"].iloc[-1]),
            "efficiencyId": efficiency_meta.efficiency_id,
            "directionId": direction_meta.direction_id,
            "twoSidednessId": two_sidedness_meta.two_sidedness_id,
            "expansionId": expansion_meta.expansion_id,
            "clearanceId": expansion_meta.clearance_id,
            "clearanceWindows": list(expansion_meta.clearance_windows),
            "windows": list(windows),
        },
        "bars": _bars_records(bars),
        "efficiency": efficiency_records,
        "direction": direction_records,
        "twoSidedness": two_sidedness_records,
        "expansion": _expansion_records(
            expansion,
            clearance_windows=expansion_meta.clearance_windows,
        ),
    }


def load_phase1a_continuous_replay(snapshot_id: str = DEFAULT_SNAPSHOT_ID) -> dict[str, object]:
    """Load the pinned research table and construct the continuous replay bundle."""
    bars, ruler_meta = get_research_bars(snapshot_id)
    return build_phase1a_continuous_replay(
        bars,
        snapshot_id=ruler_meta.snapshot_id,
        sample_id=ruler_meta.sample_id,
    )
