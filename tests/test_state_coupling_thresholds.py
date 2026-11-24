"""Tests for causal state-coupling thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_cycle.prototypes.state_coupling_thresholds import causal_expanding_quantiles


def test_threshold_excludes_current_value_and_waits_five_calendar_years():
    dates = pd.Series(pd.to_datetime(["2000-01-03", "2004-12-31", "2005-01-03"]))
    values = pd.Series([1.0, 3.0, 1000.0])

    result, meta = causal_expanding_quantiles(dates, values)

    assert not result.loc[1, "ready"]
    assert result.loc[2, "ready"]
    assert result.loc[2, "history_end"] == pd.Timestamp("2004-12-31")
    assert result.loc[2, "observation_count"] == 2
    assert result.loc[2, "p90"] == 2.8
    assert meta.quantile_method == "linear"


def test_future_extension_does_not_change_existing_thresholds():
    dates = pd.Series(pd.date_range("2000-01-03", periods=2300, freq="D"))
    values = pd.Series(np.sin(np.arange(len(dates)) / 17.0))
    base, _ = causal_expanding_quantiles(dates.iloc[:2200], values.iloc[:2200])
    extended, _ = causal_expanding_quantiles(dates, values)

    pd.testing.assert_frame_equal(base, extended.iloc[:2200])


def test_history_starts_at_first_finite_field_value():
    dates = pd.Series(pd.date_range("2000-01-03", periods=2200, freq="D"))
    values = pd.Series(np.arange(len(dates), dtype=float))
    values.iloc[:100] = np.nan

    result, _ = causal_expanding_quantiles(dates, values)

    assert result["history_start"].iloc[-1] == dates.iloc[100]
    first_ready = result.index[result["ready"]][0]
    assert dates.iloc[first_ready] >= dates.iloc[100] + pd.DateOffset(years=5)