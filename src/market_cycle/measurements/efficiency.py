"""Phase 1A fixed-window OHLC-min continuous behavior efficiency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from market_cycle.data.geometry import ohlc_min_path_length, validate_ohlc_bars
from market_cycle.measurements.scale_policy import (
    CONTINUOUS_WINDOWS,
    combine_scale_values,
    validated_windows,
    weights_for_windows,
)

EFFICIENCY_ID = "bm_e_01_ohlc_min_v1"
EFFICIENCY_WINDOWS: tuple[int, ...] = CONTINUOUS_WINDOWS

_STATUS_OK = "ok"
_STATUS_INSUFFICIENT_HISTORY = "insufficient_history"
_STATUS_ZERO_PATH = "zero_path"


@dataclass(frozen=True)
class EfficiencyMeta:
    """Resolved metadata for one fixed-window efficiency calculation."""

    efficiency_id: str
    windows: tuple[int, ...]
    input_start: pd.Timestamp
    analysis_start: pd.Timestamp
    analysis_end: pd.Timestamp
    rows: int


def _validated_bars(bars: pd.DataFrame) -> pd.DataFrame:
    return validate_ohlc_bars(bars)


def build_ohlc_min_efficiency(
    bars: pd.DataFrame,
    *,
    windows: Sequence[int] = EFFICIENCY_WINDOWS,
) -> tuple[pd.DataFrame, EfficiencyMeta]:
    """Build fixed-window OHLC-min efficiency without persisting it.

    The returned frame starts only when every requested window has sufficient
    research-calendar history. A zero path denominator yields ``NaN`` and a
    ``zero_path`` status rather than a manufactured efficiency value. When the
    default continuous windows are used, equal and midlong composites are also
    attached without re-normalizing missing scales.
    """
    resolved_windows = validated_windows(windows)
    work = _validated_bars(bars)
    if len(work) <= max(resolved_windows):
        raise ValueError(
            "bars do not contain enough rows for the largest window: "
            f"need > {max(resolved_windows)}, got {len(work)}"
        )

    close = work["close"]
    path_length = ohlc_min_path_length(work)
    out = pd.DataFrame({"date": work["date"]})
    status_columns: list[str] = []
    value_columns: list[str] = []
    scale_values: list[pd.Series] = []
    scale_statuses: list[pd.Series] = []

    for window in resolved_windows:
        value_column = f"efficiency_ohlc_min_{window}"
        status_column = f"{value_column}_status"
        denominator = path_length.rolling(window, min_periods=window).sum()
        numerator = (close - close.shift(window)).abs()
        ready = numerator.notna() & denominator.notna()
        zero_path = ready & denominator.eq(0.0)
        valid = ready & ~zero_path

        values = pd.Series(np.nan, index=work.index, dtype=float)
        values.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
        status = pd.Series(_STATUS_INSUFFICIENT_HISTORY, index=work.index, dtype=object)
        status.loc[zero_path] = _STATUS_ZERO_PATH
        status.loc[valid] = _STATUS_OK

        out[value_column] = values
        out[status_column] = status
        value_columns.append(value_column)
        status_columns.append(status_column)
        scale_values.append(values)
        scale_statuses.append(status)

    if resolved_windows == CONTINUOUS_WINDOWS:
        for policy, label in (("equal", "equal"), ("midlong", "midlong")):
            weights = weights_for_windows(resolved_windows, policy=policy)
            composite, composite_status = combine_scale_values(
                scale_values, scale_statuses, weights
            )
            out[f"efficiency_ohlc_min_{label}"] = composite
            out[f"efficiency_ohlc_min_{label}_status"] = composite_status

    valid_values = out[value_columns].to_numpy(dtype=float)
    valid_values = valid_values[np.isfinite(valid_values)]
    tolerance = 1e-12
    if ((valid_values < -tolerance) | (valid_values > 1.0 + tolerance)).any():
        raise RuntimeError("OHLC-min efficiency fell outside [0, 1]")

    fully_ready = (out[status_columns] != _STATUS_INSUFFICIENT_HISTORY).all(axis=1)
    if not fully_ready.any():
        raise RuntimeError("no row has sufficient history for every requested window")
    analysis_start_index = int(np.flatnonzero(fully_ready.to_numpy())[0])
    out = out.iloc[analysis_start_index:].reset_index(drop=True)

    meta = EfficiencyMeta(
        efficiency_id=EFFICIENCY_ID,
        windows=resolved_windows,
        input_start=pd.Timestamp(work["date"].iloc[0]),
        analysis_start=pd.Timestamp(out["date"].iloc[0]),
        analysis_end=pd.Timestamp(out["date"].iloc[-1]),
        rows=len(out),
    )
    return out, meta