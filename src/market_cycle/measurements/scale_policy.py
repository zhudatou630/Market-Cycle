"""Shared multi-scale windows and compression policies for Phase 1A."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

CONTINUOUS_WINDOWS: tuple[int, ...] = (5, 10, 20, 55)
EQUAL_WEIGHTS: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
MIDLONG_WEIGHTS: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)

_STATUS_OK = "ok"
_STATUS_INSUFFICIENT_HISTORY = "insufficient_history"
_STATUS_INCOMPLETE = "incomplete"


def validated_windows(windows: Sequence[int]) -> tuple[int, ...]:
    resolved = tuple(int(window) for window in windows)
    if not resolved:
        raise ValueError("windows must not be empty")
    if any(window < 1 for window in resolved):
        raise ValueError("every window must be >= 1")
    if resolved != tuple(sorted(set(resolved))):
        raise ValueError("windows must be unique and strictly ascending")
    return resolved


def weights_for_windows(
    windows: Sequence[int],
    *,
    policy: str,
) -> tuple[float, ...]:
    """Return compression weights for a validated window family.

    Only the default continuous window family has frozen equal / midlong weights.
    Custom window lists must match that family exactly so the research protocol
    remains comparable.
    """
    resolved = validated_windows(windows)
    if resolved != CONTINUOUS_WINDOWS:
        raise ValueError(
            "equal/midlong compression is only defined for windows "
            f"{CONTINUOUS_WINDOWS}; got {resolved}"
        )
    if policy == "equal":
        return EQUAL_WEIGHTS
    if policy == "midlong":
        return MIDLONG_WEIGHTS
    raise ValueError(f"unknown scale policy: {policy}")


def combine_scale_values(
    values: Sequence[pd.Series],
    statuses: Sequence[pd.Series],
    weights: Sequence[float],
) -> tuple[pd.Series, pd.Series]:
    """Weighted sum without re-normalizing when any scale is missing."""
    if len(values) != len(statuses) or len(values) != len(weights):
        raise ValueError("values, statuses, and weights must have equal length")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("weights must sum to 1")

    index = values[0].index
    total = pd.Series(0.0, index=index, dtype=float)
    ready = pd.Series(True, index=index)
    any_insufficient = pd.Series(False, index=index)

    for value, status, weight in zip(values, statuses, weights, strict=True):
        ok = status.eq(_STATUS_OK) & value.notna()
        ready &= ok
        any_insufficient |= status.eq(_STATUS_INSUFFICIENT_HISTORY)
        total = total + value.fillna(0.0) * float(weight)

    out = pd.Series(np.nan, index=index, dtype=float)
    out.loc[ready] = total.loc[ready]

    out_status = pd.Series(_STATUS_INCOMPLETE, index=index, dtype=object)
    out_status.loc[any_insufficient] = _STATUS_INSUFFICIENT_HISTORY
    out_status.loc[ready] = _STATUS_OK
    return out, out_status


def scale_agreement(
    values: Sequence[pd.Series],
    statuses: Sequence[pd.Series],
    weights: Sequence[float],
) -> tuple[pd.Series, pd.Series]:
    """Signed multi-scale agreement in [0, 1]."""
    if len(values) != len(statuses) or len(values) != len(weights):
        raise ValueError("values, statuses, and weights must have equal length")

    index = values[0].index
    weighted = pd.Series(0.0, index=index, dtype=float)
    weighted_abs = pd.Series(0.0, index=index, dtype=float)
    ready = pd.Series(True, index=index)
    any_insufficient = pd.Series(False, index=index)

    for value, status, weight in zip(values, statuses, weights, strict=True):
        ok = status.eq(_STATUS_OK) & value.notna()
        ready &= ok
        any_insufficient |= status.eq(_STATUS_INSUFFICIENT_HISTORY)
        weighted = weighted + value.fillna(0.0) * float(weight)
        weighted_abs = weighted_abs + value.fillna(0.0).abs() * float(weight)

    out = pd.Series(np.nan, index=index, dtype=float)
    out_status = pd.Series(_STATUS_INCOMPLETE, index=index, dtype=object)
    out_status.loc[any_insufficient] = _STATUS_INSUFFICIENT_HISTORY

    zero_direction = ready & weighted_abs.eq(0.0)
    valid = ready & ~zero_direction
    out.loc[valid] = weighted.loc[valid].abs() / weighted_abs.loc[valid]
    out_status.loc[valid] = _STATUS_OK
    out_status.loc[zero_direction] = "zero_direction"
    return out, out_status
