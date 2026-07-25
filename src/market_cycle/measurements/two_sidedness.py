"""Phase 1A two-sidedness from active sign entropy and OHLC range overlap."""

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
    validated_windows,
    weights_for_windows,
)

TWO_SIDEDNESS_ID = "bm_b_01_entropy_overlap_v1"
TWO_SIDEDNESS_WINDOWS: tuple[int, ...] = CONTINUOUS_WINDOWS

_STATUS_OK = "ok"
_STATUS_INSUFFICIENT_HISTORY = "insufficient_history"
_STATUS_ZERO_RANGE = "zero_range"


@dataclass(frozen=True)
class TwoSidednessMeta:
    """Resolved metadata for one multi-scale two-sidedness calculation."""

    two_sidedness_id: str
    windows: tuple[int, ...]
    input_start: pd.Timestamp
    analysis_start: pd.Timestamp
    analysis_end: pd.Timestamp
    rows: int


def _binary_entropy_from_counts(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    active = positive + negative
    entropy = np.zeros(len(active), dtype=float)
    usable = active > 0
    if not usable.any():
        return entropy

    p_plus = np.zeros(len(active), dtype=float)
    p_minus = np.zeros(len(active), dtype=float)
    p_plus[usable] = positive[usable] / active[usable]
    p_minus[usable] = negative[usable] / active[usable]

    for probability in (p_plus, p_minus):
        mask = usable & (probability > 0.0)
        entropy[mask] -= probability[mask] * np.log2(probability[mask])
    return entropy


def _active_sign_entropy_series(close_delta: np.ndarray, window: int) -> np.ndarray:
    values = np.full(len(close_delta), np.nan, dtype=float)
    if len(close_delta) <= window:
        return values

    # First close-to-close change lives at index 1; windows of n changes end at >= n.
    delta_views = sliding_window_view(close_delta[1:], window)
    positive = np.sum(delta_views > 0.0, axis=1)
    negative = np.sum(delta_views < 0.0, axis=1)
    active = positive + negative
    activity = active / float(window)
    binary = _binary_entropy_from_counts(positive.astype(float), negative.astype(float))
    values[window:] = activity * binary
    return values


def _union_overlap(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    values = np.full(len(high), np.nan, dtype=float)
    if len(high) < 2:
        return values

    prev_high = high[:-1]
    prev_low = low[:-1]
    curr_high = high[1:]
    curr_low = low[1:]
    intersection = np.maximum(
        0.0,
        np.minimum(curr_high, prev_high) - np.maximum(curr_low, prev_low),
    )
    union = (curr_high - curr_low) + (prev_high - prev_low) - intersection
    ready = union > 0.0
    values[1:][ready] = intersection[ready] / union[ready]
    return values


def _mean_overlap_series(overlap: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.full(len(overlap), np.nan, dtype=float)
    status_ready = np.zeros(len(overlap), dtype=bool)
    zero_range = np.zeros(len(overlap), dtype=bool)
    if len(overlap) <= window:
        return values, zero_range

    # Adjacent overlaps begin at index 1; windows of n pairs end at >= n.
    views = sliding_window_view(overlap[1:], window)
    finite = np.isfinite(views)
    counts = finite.sum(axis=1)
    filled = np.where(finite, views, 0.0)
    totals = filled.sum(axis=1)
    means = np.full(len(counts), np.nan, dtype=float)
    has_valid = counts > 0
    means[has_valid] = totals[has_valid] / counts[has_valid]
    values[window:] = means
    status_ready[window:] = True
    zero_range[window:] = ~has_valid
    return values, zero_range


def build_two_sidedness(
    bars: pd.DataFrame,
    *,
    windows: Sequence[int] = TWO_SIDEDNESS_WINDOWS,
) -> tuple[pd.DataFrame, TwoSidednessMeta]:
    """Build multi-scale two-sidedness from entropy and range re-trading.

    ``B_v0.1 = 0.5 * H_active + 0.5 * mean(O_union)`` on each fixed window.
    Overlap uses the union-normalized adjacent OHLC range ratio. Zero-union
    pairs are omitted from the mean; if a window has no valid pairs the
    overlap and composite are missing.
    """
    resolved_windows = validated_windows(windows)
    work = validate_ohlc_bars(bars)
    if len(work) <= max(resolved_windows):
        raise ValueError(
            "bars do not contain enough rows for the largest window: "
            f"need > {max(resolved_windows)}, got {len(work)}"
        )

    close = work["close"].to_numpy(dtype=float)
    high = work["high"].to_numpy(dtype=float)
    low = work["low"].to_numpy(dtype=float)
    close_delta = np.full(len(work), np.nan, dtype=float)
    close_delta[1:] = close[1:] - close[:-1]
    overlap = _union_overlap(high, low)

    out = pd.DataFrame({"date": work["date"]})
    value_columns: list[str] = []
    status_columns: list[str] = []
    scale_values: list[pd.Series] = []
    scale_statuses: list[pd.Series] = []

    for window in resolved_windows:
        entropy_column = f"two_sidedness_entropy_{window}"
        overlap_column = f"two_sidedness_overlap_{window}"
        value_column = f"two_sidedness_v1_{window}"
        status_column = f"{value_column}_status"

        entropy = _active_sign_entropy_series(close_delta, window)
        mean_overlap, zero_range = _mean_overlap_series(overlap, window)

        values = pd.Series(np.nan, index=work.index, dtype=float)
        status = pd.Series(_STATUS_INSUFFICIENT_HISTORY, index=work.index, dtype=object)
        ready = np.isfinite(entropy)
        valid = ready & ~zero_range
        values.iloc[valid] = 0.5 * entropy[valid] + 0.5 * mean_overlap[valid]
        status.iloc[ready & zero_range] = _STATUS_ZERO_RANGE
        status.iloc[valid] = _STATUS_OK

        out[entropy_column] = entropy
        out[overlap_column] = mean_overlap
        out[value_column] = values
        out[status_column] = status
        value_columns.append(value_column)
        status_columns.append(status_column)
        scale_values.append(values)
        scale_statuses.append(status)

    if resolved_windows == CONTINUOUS_WINDOWS:
        for policy, label in (("equal", "equal_v1"), ("midlong", "midlong_v1")):
            weights = weights_for_windows(resolved_windows, policy=policy)
            composite, composite_status = combine_scale_values(
                scale_values, scale_statuses, weights
            )
            out[f"two_sidedness_{label}"] = composite
            out[f"two_sidedness_{label}_status"] = composite_status

    value_array = out[value_columns].to_numpy(dtype=float)
    finite = value_array[np.isfinite(value_array)]
    tolerance = 1e-12
    if ((finite < -tolerance) | (finite > 1.0 + tolerance)).any():
        raise RuntimeError("two-sidedness fell outside [0, 1]")

    fully_ready = (out[status_columns] != _STATUS_INSUFFICIENT_HISTORY).all(axis=1)
    if not fully_ready.any():
        raise RuntimeError("no row has sufficient history for every requested window")
    analysis_start_index = int(np.flatnonzero(fully_ready.to_numpy())[0])
    out = out.iloc[analysis_start_index:].reset_index(drop=True)

    meta = TwoSidednessMeta(
        two_sidedness_id=TWO_SIDEDNESS_ID,
        windows=resolved_windows,
        input_start=pd.Timestamp(work["date"].iloc[0]),
        analysis_start=pd.Timestamp(out["date"].iloc[0]),
        analysis_end=pd.Timestamp(out["date"].iloc[-1]),
        rows=len(out),
    )
    return out, meta
