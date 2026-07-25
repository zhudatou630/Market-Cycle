"""Phase 1A multi-scale direction drift from log-close Theil-Sen slopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from market_cycle.data.geometry import validate_ohlc_bars
from market_cycle.measurements.scale_policy import (
    CONTINUOUS_WINDOWS,
    combine_scale_values,
    scale_agreement,
    validated_windows,
    weights_for_windows,
)

DIRECTION_ID = "bm_d_01_theilsen_atr_v1"
DIRECTION_WINDOWS: tuple[int, ...] = CONTINUOUS_WINDOWS

_STATUS_OK = "ok"
_STATUS_INSUFFICIENT_HISTORY = "insufficient_history"
_STATUS_ZERO_SCALE = "zero_scale"
_REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "atr_pct_14")


@dataclass(frozen=True)
class DirectionMeta:
    """Resolved metadata for one multi-scale direction calculation."""

    direction_id: str
    windows: tuple[int, ...]
    input_start: pd.Timestamp
    analysis_start: pd.Timestamp
    analysis_end: pd.Timestamp
    rows: int


def _validated_direction_bars(bars: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _REQUIRED_COLUMNS if column not in bars.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")
    work = validate_ohlc_bars(bars.loc[:, ["date", "open", "high", "low", "close"]])
    atr = pd.to_numeric(bars["atr_pct_14"], errors="coerce")
    if atr.isna().any():
        raise ValueError("bars contain null atr_pct_14 values")
    if (atr < 0).any():
        raise ValueError("bars contain negative atr_pct_14 values")
    work = work.copy()
    work["atr_pct_14"] = atr.to_numpy(dtype=float)
    return work.reset_index(drop=True)


def _theil_sen_slopes(log_close: np.ndarray, points: int) -> np.ndarray:
    """Return Theil-Sen slopes for every trailing window of ``points`` closes."""
    if points < 2:
        raise ValueError("Theil-Sen needs at least 2 close points")
    if len(log_close) < points:
        return np.array([], dtype=float)

    windows = sliding_window_view(log_close, points)
    pair_slopes = [
        (windows[:, right] - windows[:, left]) / (right - left)
        for left in range(points)
        for right in range(left + 1, points)
    ]
    return np.median(np.stack(pair_slopes, axis=0), axis=0)


def build_direction_drift(
    bars: pd.DataFrame,
    *,
    windows: Sequence[int] = DIRECTION_WINDOWS,
) -> tuple[pd.DataFrame, DirectionMeta]:
    """Build multi-scale ATR-normalized Theil-Sen direction drift.

    For window ``n`` the slope uses closes ``[t-n, t]`` (``n`` completed daily
    moves) and divides by the median ``atr_pct_14`` over the same ``n`` process
    days ``[t-n+1, t]``. Composite equal / midlong scores never re-normalize
    when a scale is missing.
    """
    resolved_windows = validated_windows(windows)
    work = _validated_direction_bars(bars)
    # Largest window needs n process days plus one boundary close.
    if len(work) <= max(resolved_windows):
        raise ValueError(
            "bars do not contain enough rows for the largest window: "
            f"need > {max(resolved_windows)}, got {len(work)}"
        )

    log_close = np.log(work["close"].to_numpy(dtype=float))
    atr = work["atr_pct_14"].to_numpy(dtype=float)
    out = pd.DataFrame({"date": work["date"]})
    value_columns: list[str] = []
    status_columns: list[str] = []
    scale_values: list[pd.Series] = []
    scale_statuses: list[pd.Series] = []

    for window in resolved_windows:
        value_column = f"direction_drift_{window}"
        status_column = f"{value_column}_status"
        values = pd.Series(np.nan, index=work.index, dtype=float)
        status = pd.Series(_STATUS_INSUFFICIENT_HISTORY, index=work.index, dtype=object)

        # n completed day intervals need n+1 closes ending at t.
        slopes = _theil_sen_slopes(log_close, window + 1)
        ready_index = np.arange(window, len(work))
        atr_medians = (
            pd.Series(atr)
            .rolling(window, min_periods=window)
            .median()
            .to_numpy(dtype=float)[window:]
        )
        zero_scale = atr_medians == 0.0
        valid = ~zero_scale
        slope_values = slopes
        scaled = np.full(len(slopes), np.nan, dtype=float)
        scaled[valid] = slope_values[valid] / atr_medians[valid]

        values.iloc[ready_index] = scaled
        status.iloc[ready_index] = np.where(zero_scale, _STATUS_ZERO_SCALE, _STATUS_OK)

        out[value_column] = values
        out[status_column] = status
        value_columns.append(value_column)
        status_columns.append(status_column)
        scale_values.append(values)
        scale_statuses.append(status)

    if resolved_windows == CONTINUOUS_WINDOWS:
        for policy, label in (("equal", "equal_raw"), ("midlong", "midlong_raw")):
            weights = weights_for_windows(resolved_windows, policy=policy)
            composite, composite_status = combine_scale_values(
                scale_values, scale_statuses, weights
            )
            agreement, agreement_status = scale_agreement(
                scale_values, scale_statuses, weights
            )
            out[f"direction_drift_{label}"] = composite
            out[f"direction_drift_{label}_status"] = composite_status
            out[f"direction_scale_agreement_{policy}"] = agreement
            out[f"direction_scale_agreement_{policy}_status"] = agreement_status

    fully_ready = (out[status_columns] != _STATUS_INSUFFICIENT_HISTORY).all(axis=1)
    if not fully_ready.any():
        raise RuntimeError("no row has sufficient history for every requested window")
    analysis_start_index = int(np.flatnonzero(fully_ready.to_numpy())[0])
    out = out.iloc[analysis_start_index:].reset_index(drop=True)

    meta = DirectionMeta(
        direction_id=DIRECTION_ID,
        windows=resolved_windows,
        input_start=pd.Timestamp(work["date"].iloc[0]),
        analysis_start=pd.Timestamp(out["date"].iloc[0]),
        analysis_end=pd.Timestamp(out["date"].iloc[-1]),
        rows=len(out),
    )
    return out, meta
