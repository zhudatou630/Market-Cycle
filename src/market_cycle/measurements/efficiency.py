"""Track A fixed-window OHLC-min path efficiency (Phase 1A v0.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

EFFICIENCY_ID = "bm_e_01_ohlc_min_v1"
EFFICIENCY_WINDOWS: tuple[int, ...] = (5, 10, 20, 55)

_OHLC_COLUMNS = ("date", "open", "high", "low", "close")
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


def _validated_windows(windows: Sequence[int]) -> tuple[int, ...]:
    resolved = tuple(int(window) for window in windows)
    if not resolved:
        raise ValueError("windows must not be empty")
    if any(window < 1 for window in resolved):
        raise ValueError("every window must be >= 1")
    if resolved != tuple(sorted(set(resolved))):
        raise ValueError("windows must be unique and strictly ascending")
    return resolved


def _validated_bars(bars: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _OHLC_COLUMNS if column not in bars.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")

    out = bars.loc[:, _OHLC_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    for column in _OHLC_COLUMNS[1:]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    if out.isna().any().any():
        raise ValueError("bars contain null OHLC values")
    if not out["date"].is_monotonic_increasing or out["date"].duplicated().any():
        raise ValueError("bars must have unique dates in ascending order")
    if (out["high"] < out["low"]).any():
        raise ValueError("bars contain high < low")
    if ((out["open"] < out["low"]) | (out["open"] > out["high"])).any():
        raise ValueError("bars contain open outside [low, high]")
    if ((out["close"] < out["low"]) | (out["close"] > out["high"])).any():
        raise ValueError("bars contain close outside [low, high]")
    return out.reset_index(drop=True)


def ohlc_min_path_length(bars: pd.DataFrame) -> pd.Series:
    """Return the minimum path compatible with each daily OHLC observation.

    The first row is ``NaN`` because it lacks a previous close. Each later row
    counts the observed previous-close-to-open displacement plus the shorter of
    the two high/low visit orders. This is a lower bound on unknown intraday
    path length, not a reconstructed intraday path.
    """
    work = _validated_bars(bars)
    close = work["close"].to_numpy(dtype=float)
    open_ = work["open"].to_numpy(dtype=float)
    high = work["high"].to_numpy(dtype=float)
    low = work["low"].to_numpy(dtype=float)

    path = np.full(len(work), np.nan, dtype=float)
    if len(work) > 1:
        gap = np.abs(open_[1:] - close[:-1])
        high_then_low = (
            np.abs(open_[1:] - high[1:])
            + np.abs(high[1:] - low[1:])
            + np.abs(low[1:] - close[1:])
        )
        low_then_high = (
            np.abs(open_[1:] - low[1:])
            + np.abs(low[1:] - high[1:])
            + np.abs(high[1:] - close[1:])
        )
        path[1:] = gap + np.minimum(high_then_low, low_then_high)

    return pd.Series(path, index=work.index, name="ohlc_min_path_length")


def build_ohlc_min_efficiency(
    bars: pd.DataFrame,
    *,
    windows: Sequence[int] = EFFICIENCY_WINDOWS,
) -> tuple[pd.DataFrame, EfficiencyMeta]:
    """Build Track A fixed-window OHLC-min efficiency without persisting it.

    The returned frame starts only when every requested window has sufficient
    research-calendar history. A zero path denominator yields ``NaN`` and a
    ``zero_path`` status rather than a manufactured efficiency value.
    """
    resolved_windows = _validated_windows(windows)
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