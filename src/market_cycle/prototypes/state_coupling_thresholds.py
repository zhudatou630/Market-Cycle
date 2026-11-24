"""Causal expanding quantiles for the Phase 1A state-coupling probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

THRESHOLD_METHOD_ID = "causal_expanding_all_history_p20_p30_p70_p90_v1"
QUANTILE_METHOD = "linear"
DEFAULT_QUANTILES: tuple[float, ...] = (0.20, 0.30, 0.70, 0.90)


@dataclass(frozen=True)
class CausalQuantileMeta:
    threshold_method_id: str
    quantile_method: str
    minimum_calendar_years: int
    quantiles: tuple[float, ...]


def causal_expanding_quantiles(
    dates: pd.Series,
    values: pd.Series,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    minimum_calendar_years: int = 5,
) -> tuple[pd.DataFrame, CausalQuantileMeta]:
    """Return thresholds at t from finite observations strictly before t.

    Calendar readiness is measured from the field's first finite observation.
    Observation counts and history end dates likewise exclude the current row.
    """
    resolved = tuple(float(value) for value in quantiles)
    if not resolved or any(value < 0.0 or value > 1.0 for value in resolved):
        raise ValueError("quantiles must be a non-empty sequence inside [0, 1]")
    if minimum_calendar_years < 1:
        raise ValueError("minimum_calendar_years must be positive")

    date_values = pd.to_datetime(dates, errors="raise")
    if not date_values.is_monotonic_increasing or date_values.duplicated().any():
        raise ValueError("dates must be strictly increasing")
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    numeric = numeric.where(np.isfinite(numeric))
    prior = numeric.shift(1)
    valid_date = date_values.where(numeric.notna())
    first_valid = valid_date.dropna().iloc[0] if numeric.notna().any() else pd.NaT

    out = pd.DataFrame(index=values.index)
    out["history_start"] = first_valid
    out["history_end"] = valid_date.shift(1).ffill()
    out["observation_count"] = prior.expanding(min_periods=1).count().astype("int64")
    ready_date = (
        pd.Series(False, index=values.index)
        if pd.isna(first_valid)
        else date_values >= first_valid + pd.DateOffset(years=minimum_calendar_years)
    )
    out["ready"] = ready_date & out["observation_count"].gt(0)

    for quantile in resolved:
        label = f"p{round(quantile * 100):02d}"
        threshold = prior.expanding(min_periods=1).quantile(
            quantile, interpolation=QUANTILE_METHOD
        )
        out[label] = threshold.where(out["ready"])

    meta = CausalQuantileMeta(
        threshold_method_id=THRESHOLD_METHOD_ID,
        quantile_method=QUANTILE_METHOD,
        minimum_calendar_years=minimum_calendar_years,
        quantiles=resolved,
    )
    return out, meta


def causal_trailing_years_quantiles(
    dates: pd.Series,
    values: pd.Series,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    trailing_years: int = 5,
) -> tuple[pd.DataFrame, CausalQuantileMeta]:
    """Return exact calendar-window quantiles from [t-years, t)."""
    resolved = tuple(float(value) for value in quantiles)
    if not resolved or any(value < 0.0 or value > 1.0 for value in resolved):
        raise ValueError("quantiles must be a non-empty sequence inside [0, 1]")
    if trailing_years < 1:
        raise ValueError("trailing_years must be positive")
    date_values = pd.to_datetime(dates, errors="raise")
    if not date_values.is_monotonic_increasing or date_values.duplicated().any():
        raise ValueError("dates must be strictly increasing")
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(numeric))
    first_finite_date = date_values.iloc[int(finite_indices[0])] if len(finite_indices) else pd.NaT

    result = pd.DataFrame(index=values.index)
    starts: list[pd.Timestamp | pd.NaT] = []
    ends: list[pd.Timestamp | pd.NaT] = []
    counts: list[int] = []
    ready: list[bool] = []
    quantile_values = {value: [] for value in resolved}
    dates_array = date_values.to_numpy(dtype="datetime64[ns]")
    for index, current in enumerate(date_values):
        lower = current - pd.DateOffset(years=trailing_years)
        start = int(np.searchsorted(dates_array, np.datetime64(lower), side="left"))
        history = numeric[start:index]
        history = history[np.isfinite(history)]
        is_ready = not pd.isna(first_finite_date) and first_finite_date <= lower and len(history) > 0
        starts.append(date_values.iloc[start] if is_ready else pd.NaT)
        ends.append(date_values.iloc[index - 1] if is_ready else pd.NaT)
        counts.append(int(len(history)))
        ready.append(bool(is_ready))
        for quantile in resolved:
            quantile_values[quantile].append(
                float(np.quantile(history, quantile, method=QUANTILE_METHOD))
                if is_ready
                else np.nan
            )
    result["history_start"] = starts
    result["history_end"] = ends
    result["observation_count"] = counts
    result["ready"] = ready
    for quantile in resolved:
        result[f"p{round(quantile * 100):02d}"] = quantile_values[quantile]
    meta = CausalQuantileMeta(
        threshold_method_id=f"causal_trailing_{trailing_years}y_quantiles_v1",
        quantile_method=QUANTILE_METHOD,
        minimum_calendar_years=trailing_years,
        quantiles=resolved,
    )
    return result, meta