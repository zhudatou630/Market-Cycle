"""Build an in-memory Phase 1A continuous-behavior replay bundle."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from market_cycle.data import DEFAULT_SNAPSHOT_ID, get_research_bars
from market_cycle.measurements import EFFICIENCY_WINDOWS, build_ohlc_min_efficiency

REPLAY_SCHEMA_VERSION = "phase1a_continuous_replay_v1"


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


def build_phase1a_continuous_replay(
    bars: pd.DataFrame,
    *,
    snapshot_id: str,
    sample_id: str,
    windows: Sequence[int] = EFFICIENCY_WINDOWS,
) -> dict[str, object]:
    """Build the local replay bundle for Phase 1A continuous behavior only.

    The bundle contains daily OHLC and fixed-window efficiency values. Causal
    swing candidates live in ``market_cycle.structures`` and are intentionally
    excluded from the Phase 1A base-layer audit surface.
    """
    efficiency, efficiency_meta = build_ohlc_min_efficiency(bars, windows=windows)
    return {
        "schemaVersion": REPLAY_SCHEMA_VERSION,
        "meta": {
            "snapshotId": snapshot_id,
            "sampleId": sample_id,
            "researchStart": _date(bars["date"].iloc[0]),
            "researchEnd": _date(bars["date"].iloc[-1]),
            "efficiencyId": efficiency_meta.efficiency_id,
            "efficiencyWindows": list(efficiency_meta.windows),
        },
        "bars": _bars_records(bars),
        "efficiency": _efficiency_records(efficiency, efficiency_meta.windows),
    }


def load_phase1a_continuous_replay(snapshot_id: str = DEFAULT_SNAPSHOT_ID) -> dict[str, object]:
    """Load the pinned research table and construct the continuous replay bundle."""
    bars, ruler_meta = get_research_bars(snapshot_id)
    return build_phase1a_continuous_replay(
        bars,
        snapshot_id=ruler_meta.snapshot_id,
        sample_id=ruler_meta.sample_id,
    )
