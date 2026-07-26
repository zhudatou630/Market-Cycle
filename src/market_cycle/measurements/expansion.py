"""Phase 1A daily expansion impulse and prior-range clearance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from market_cycle.data.geometry import validate_ohlc_bars

EXPANSION_ID = "bm_x_01_ohlc_impulse_v1"
CLEARANCE_ID = "bm_x_dep_01_prior_range_clearance_v1"
CLEARANCE_WINDOWS: tuple[int, ...] = (20, 55)

_STATUS_OK = "ok"
_STATUS_INSUFFICIENT_HISTORY = "insufficient_history"
_STATUS_ZERO_SCALE = "zero_scale"
_STATUS_ZERO_RANGE = "zero_range"
_REQUIRED_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "tr_pct",
    "atr_pct_14",
)


@dataclass(frozen=True)
class ExpansionMeta:
    """Resolved metadata for daily impulse and clearance calculations."""

    expansion_id: str
    clearance_id: str
    clearance_windows: tuple[int, ...]
    input_start: pd.Timestamp
    analysis_start: pd.Timestamp
    clearance_start: pd.Timestamp | None
    analysis_end: pd.Timestamp
    rows: int


def _validated_clearance_windows(windows: Sequence[int]) -> tuple[int, ...]:
    resolved = tuple(int(window) for window in windows)
    if not resolved:
        raise ValueError("clearance_windows must not be empty")
    if any(window < 1 for window in resolved):
        raise ValueError("clearance_windows must contain positive integers")
    if len(set(resolved)) != len(resolved):
        raise ValueError("clearance_windows must not contain duplicates")
    return resolved


def _validated_expansion_bars(bars: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _REQUIRED_COLUMNS if column not in bars.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")

    work = validate_ohlc_bars(bars.loc[:, ["date", "open", "high", "low", "close"]])
    for column in ("tr_pct", "atr_pct_14"):
        values = pd.to_numeric(bars[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"bars contain non-finite {column} values")
        if (values < 0.0).any():
            raise ValueError(f"bars contain negative {column} values")
        work[column] = values.to_numpy(dtype=float)
    return work.reset_index(drop=True)


def _clearance_values(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return current-close clearance over prior complete OHLC ranges."""
    up = np.full(len(close), np.nan, dtype=float)
    down = np.full(len(close), np.nan, dtype=float)
    if len(close) <= window:
        return up, down

    prior_highs = sliding_window_view(high, window)[: len(close) - window]
    prior_lows = sliding_window_view(low, window)[: len(close) - window]
    current_close = close[window:]
    up[window:] = (current_close[:, None] > prior_highs).mean(axis=1)
    down[window:] = (current_close[:, None] < prior_lows).mean(axis=1)
    return up, down


def build_expansion_impulse(
    bars: pd.DataFrame,
    *,
    clearance_windows: Sequence[int] = CLEARANCE_WINDOWS,
) -> tuple[pd.DataFrame, ExpansionMeta]:
    """Build Phase 1A daily expansion fields from OHLC plus the shared ruler.

    The daily impulse compares today's completed true range and close movement
    with the ATR point scale frozen at ``t-1``. ``tr_pct`` and ``atr_pct_14``
    are read from the shared research table; this module never recomputes a
    second TR or ATR series. Prior-range clearance is a separate continuous
    position reference, not a breakout event or a composite component.
    """
    resolved_windows = _validated_clearance_windows(clearance_windows)
    work = _validated_expansion_bars(bars)
    if len(work) < 2:
        raise ValueError("expansion impulse needs at least two daily research rows")

    close = work["close"].to_numpy(dtype=float)
    high = work["high"].to_numpy(dtype=float)
    low = work["low"].to_numpy(dtype=float)
    open_ = work["open"].to_numpy(dtype=float)
    tr_points = close * work["tr_pct"].to_numpy(dtype=float)
    atr_points = close * work["atr_pct_14"].to_numpy(dtype=float)

    out = pd.DataFrame({"date": work["date"]})
    activity = np.full(len(work), np.nan, dtype=float)
    activity_status = np.full(len(work), _STATUS_INSUFFICIENT_HISTORY, dtype=object)
    activity[1:] = work["atr_pct_14"].to_numpy(dtype=float)[:-1]
    activity_status[1:] = _STATUS_OK
    out["activity_level_atr_pct_prev"] = activity
    out["activity_level_atr_pct_prev_status"] = activity_status

    range_value = np.full(len(work), np.nan, dtype=float)
    close_value = np.full(len(work), np.nan, dtype=float)
    gap_value = np.full(len(work), np.nan, dtype=float)
    scale_status = np.full(len(work), _STATUS_INSUFFICIENT_HISTORY, dtype=object)
    previous_scale = atr_points[:-1]
    scale_ready = previous_scale > 0.0
    scale_status[1:] = np.where(scale_ready, _STATUS_OK, _STATUS_ZERO_SCALE)
    ready_indices = np.arange(1, len(work))[scale_ready]
    range_value[ready_indices] = tr_points[ready_indices] / previous_scale[scale_ready]
    close_value[ready_indices] = (
        close[ready_indices] - close[ready_indices - 1]
    ) / previous_scale[scale_ready]
    gap_up = np.maximum(0.0, open_[ready_indices] - high[ready_indices - 1])
    gap_down = np.maximum(0.0, low[ready_indices - 1] - open_[ready_indices])
    gap_value[ready_indices] = (gap_up - gap_down) / previous_scale[scale_ready]

    for column, values in (
        ("expansion_range_atr_prev", range_value),
        ("expansion_close_atr_prev", close_value),
        ("expansion_gap_prev_range_atr", gap_value),
    ):
        out[column] = values
        out[f"{column}_status"] = scale_status

    share = np.full(len(work), np.nan, dtype=float)
    share_status = np.full(len(work), _STATUS_INSUFFICIENT_HISTORY, dtype=object)
    usable_range = tr_points[1:] > 0.0
    share_status[1:] = np.where(usable_range, _STATUS_OK, _STATUS_ZERO_RANGE)
    share_indices = np.arange(1, len(work))[usable_range]
    share[share_indices] = (
        close[share_indices] - close[share_indices - 1]
    ) / tr_points[share_indices]
    out["expansion_close_share"] = share
    out["expansion_close_share_status"] = share_status

    clearance_status_columns: list[str] = []
    for window in resolved_windows:
        up, down = _clearance_values(close, high, low, window)
        for direction, values in (("up", up), ("down", down)):
            column = f"prior_range_clearance_{direction}_{window}"
            status_column = f"{column}_status"
            status = np.full(len(work), _STATUS_INSUFFICIENT_HISTORY, dtype=object)
            status[window:] = _STATUS_OK
            out[column] = values
            out[status_column] = status
            clearance_status_columns.append(status_column)

    clearance_start: pd.Timestamp | None = None
    clearance_ready = (out[clearance_status_columns] == _STATUS_OK).all(axis=1)
    if clearance_ready.any():
        clearance_start = pd.Timestamp(out.loc[clearance_ready, "date"].iloc[0])

    # Day zero has no prior research-row close or frozen scale for X_now.
    out = out.iloc[1:].reset_index(drop=True)
    meta = ExpansionMeta(
        expansion_id=EXPANSION_ID,
        clearance_id=CLEARANCE_ID,
        clearance_windows=resolved_windows,
        input_start=pd.Timestamp(work["date"].iloc[0]),
        analysis_start=pd.Timestamp(out["date"].iloc[0]),
        clearance_start=clearance_start,
        analysis_end=pd.Timestamp(out["date"].iloc[-1]),
        rows=len(out),
    )
    return out, meta