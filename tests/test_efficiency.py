"""Tests for Track A fixed-window OHLC-min efficiency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_cycle.data import get_research_bars
from market_cycle.measurements.efficiency import (
    EFFICIENCY_ID,
    build_ohlc_min_efficiency,
    ohlc_min_path_length,
)


def _bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])


def test_ohlc_min_path_length_matches_shorter_intrabar_order():
    bars = _bars(
        [
            ("2024-01-02", 100.0, 100.0, 100.0, 100.0),
            ("2024-01-03", 102.0, 110.0, 101.0, 105.0),
        ]
    )
    path = ohlc_min_path_length(bars)
    assert np.isnan(path.iloc[0])
    # Gap = 2. O->H->L->C = 21; O->L->H->C = 15. Total = 17.
    assert path.iloc[1] == pytest.approx(17.0)


def test_efficiency_handles_monotone_and_retracing_paths():
    bars = _bars(
        [
            ("2024-01-02", 100.0, 100.0, 100.0, 100.0),
            ("2024-01-03", 100.0, 105.0, 100.0, 105.0),
            ("2024-01-04", 105.0, 110.0, 105.0, 110.0),
            ("2024-01-05", 110.0, 110.0, 100.0, 100.0),
        ]
    )
    frame, meta = build_ohlc_min_efficiency(bars, windows=(2,))

    assert meta.efficiency_id == EFFICIENCY_ID
    assert frame["date"].tolist() == [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")]
    assert frame["efficiency_ohlc_min_2"].iloc[0] == pytest.approx(1.0)
    assert frame["efficiency_ohlc_min_2"].iloc[1] == pytest.approx(1.0 / 3.0)
    assert (frame["efficiency_ohlc_min_2_status"] == "ok").all()


def test_zero_path_is_missing_and_marked():
    bars = _bars(
        [
            ("2024-01-02", 100.0, 100.0, 100.0, 100.0),
            ("2024-01-03", 100.0, 100.0, 100.0, 100.0),
            ("2024-01-04", 100.0, 100.0, 100.0, 100.0),
        ]
    )
    frame, _ = build_ohlc_min_efficiency(bars, windows=(2,))

    assert len(frame) == 1
    assert np.isnan(frame["efficiency_ohlc_min_2"].iloc[0])
    assert frame["efficiency_ohlc_min_2_status"].iloc[0] == "zero_path"


def test_prefix_replay_matches_batch_at_each_available_date():
    bars = _bars(
        [
            ("2024-01-02", 100.0, 101.0, 99.0, 100.0),
            ("2024-01-03", 100.0, 103.0, 100.0, 102.0),
            ("2024-01-04", 102.0, 104.0, 101.0, 103.0),
            ("2024-01-05", 103.0, 103.0, 99.0, 100.0),
            ("2024-01-08", 100.0, 102.0, 98.0, 101.0),
            ("2024-01-09", 101.0, 105.0, 101.0, 104.0),
        ]
    )
    batch, _ = build_ohlc_min_efficiency(bars, windows=(2, 3))

    for end in range(4, len(bars) + 1):
        prefix, _ = build_ohlc_min_efficiency(bars.iloc[:end], windows=(2, 3))
        expected = batch.loc[batch["date"] == prefix["date"].iloc[-1]].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix.tail(1).reset_index(drop=True), expected)


def test_frozen_research_sample_has_expected_start_and_valid_range():
    bars, _ = get_research_bars()
    frame, meta = build_ohlc_min_efficiency(bars)

    assert meta.windows == (5, 10, 20, 55)
    assert meta.analysis_start == pd.Timestamp("1984-05-01")
    assert frame["date"].iloc[0] == pd.Timestamp("1984-05-01")
    value_columns = [column for column in frame if column.startswith("efficiency_ohlc_min_") and not column.endswith("_status")]
    status_columns = [f"{column}_status" for column in value_columns]
    assert (frame[status_columns] == "ok").all().all()
    assert ((frame[value_columns] >= 0.0) & (frame[value_columns] <= 1.0)).all().all()