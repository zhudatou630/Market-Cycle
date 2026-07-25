"""Shared daily OHLC geometry available to continuous and structural modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

_OHLC_COLUMNS = ("date", "open", "high", "low", "close")


def validate_ohlc_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Return normalized, validated OHLC bars without inventing intraday order."""
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
    """Return the lower-bound path length compatible with each daily OHLC bar."""
    work = validate_ohlc_bars(bars)
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