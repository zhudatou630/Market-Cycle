"""Memoryless Phase 1A state-coupling probe v0a."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pandas as pd

from market_cycle.measurements import (
    build_direction_drift,
    build_expansion_impulse,
    build_ohlc_min_efficiency,
    build_two_sidedness,
)
from market_cycle.prototypes.state_coupling_thresholds import (
    QUANTILE_METHOD,
    THRESHOLD_METHOD_ID,
    causal_expanding_quantiles,
    causal_trailing_years_quantiles,
)

PROBE_VERSION = "phase1a_state_coupling_probe_v0a"
DERIVED_POSITION_VERSION = "phase1a_probe_position_coverage_v1"
REGIME_RULE_ORDER = "directional_then_balance_then_unresolved"
INPUT_FEATURE_VERSIONS = (
    "bm_e_01_ohlc_min_v1",
    "bm_d_01_theilsen_atr_v1",
    "bm_b_01_entropy_overlap_v1",
    "bm_x_01_ohlc_impulse_v1",
    "bm_x_dep_01_prior_range_clearance_v1",
)

_SOURCE_COLUMNS = {
    "abs_direction": "abs_direction_midlong",
    "agreement": "direction_scale_agreement_midlong",
    "efficiency": "efficiency_ohlc_min_midlong",
    "entropy": "two_sidedness_entropy_20",
    "overlap": "two_sidedness_overlap_20",
    "coverage20": "historical_coverage_20",
    "activity": "activity_level_atr_pct_prev",
    "range": "expansion_range_atr_prev",
    "abs_share": "abs_expansion_close_share",
}


@dataclass(frozen=True)
class StateCouplingProbeMeta:
    probe_version: str
    threshold_method_id: str
    quantile_method: str
    input_feature_versions: tuple[str, ...]
    input_start: pd.Timestamp
    output_start: pd.Timestamp
    output_end: pd.Timestamp
    rows: int


def _merge_features(bars: pd.DataFrame) -> pd.DataFrame:
    efficiency, _ = build_ohlc_min_efficiency(bars)
    direction, _ = build_direction_drift(bars)
    two_sidedness, _ = build_two_sidedness(bars)
    expansion, _ = build_expansion_impulse(bars)

    out = bars.loc[:, ["date", "open", "high", "low", "close"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    for frame in (efficiency, direction, two_sidedness, expansion):
        out = out.merge(frame, on="date", how="left", validate="one_to_one")

    out["abs_direction_midlong"] = out["direction_drift_midlong_raw"].abs()
    out["abs_expansion_close_share"] = out["expansion_close_share"].abs()
    for window in (20, 55):
        up = out[f"prior_range_clearance_up_{window}"]
        down = out[f"prior_range_clearance_down_{window}"]
        out[f"position_bias_{window}"] = up - down
        out[f"historical_coverage_{window}"] = 1.0 - up - down
    return out


def _rule(raw: float, threshold: float, operator: str) -> dict[str, Any]:
    if not np.isfinite(raw) or not np.isfinite(threshold):
        return {"raw_value": None, "threshold": None, "operator": operator, "hit": False, "margin": None}
    if operator in (">=", ">"):
        hit = raw >= threshold if operator == ">=" else raw > threshold
        margin = raw - threshold
    elif operator in ("<=", "<"):
        hit = raw <= threshold if operator == "<=" else raw < threshold
        margin = threshold - raw
    else:
        raise ValueError(f"unsupported operator: {operator}")
    return {
        "raw_value": float(raw),
        "threshold": float(threshold),
        "operator": operator,
        "hit": bool(hit),
        "margin": float(margin),
    }


def _attribute(raw: float, p30: float, p70: float) -> str | None:
    if not all(np.isfinite(value) for value in (raw, p30, p70)):
        return None
    if raw <= p30:
        return "low"
    if raw >= p70:
        return "high"
    return "mid"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _status_is_ok(row: pd.Series, columns: tuple[str, ...]) -> bool:
    return all(row.get(column) == "ok" for column in columns)


def _classify_row(row: pd.Series, threshold_columns: dict[str, dict[str, str]]) -> dict[str, Any]:
    required_statuses = (
        "efficiency_ohlc_min_midlong_status",
        "direction_drift_midlong_raw_status",
        "direction_scale_agreement_midlong_status",
        "two_sidedness_v1_20_status",
        "activity_level_atr_pct_prev_status",
        "expansion_range_atr_prev_status",
        "expansion_close_atr_prev_status",
        "expansion_close_share_status",
        "prior_range_clearance_up_20_status",
        "prior_range_clearance_down_20_status",
        "prior_range_clearance_up_55_status",
        "prior_range_clearance_down_55_status",
    )
    raw_snapshot = {
        "D_5": row.get("direction_drift_5"),
        "D_55": row.get("direction_drift_55"),
        "D_equal": row.get("direction_drift_equal_raw"),
        "D_midlong": row.get("direction_drift_midlong_raw"),
        "A_midlong": row.get("direction_scale_agreement_midlong"),
        "E_midlong": row.get("efficiency_ohlc_min_midlong"),
        "entropy_20": row.get("two_sidedness_entropy_20"),
        "overlap_20": row.get("two_sidedness_overlap_20"),
        "B_equal": row.get("two_sidedness_equal_v1"),
        "B_midlong": row.get("two_sidedness_midlong_v1"),
        "Activity": row.get("activity_level_atr_pct_prev"),
        "X_range": row.get("expansion_range_atr_prev"),
        "X_close": row.get("expansion_close_atr_prev"),
        "X_share": row.get("expansion_close_share"),
    }
    raw_snapshot = {
        key: None if pd.isna(value) else float(value) for key, value in raw_snapshot.items()
    }
    base = {
        "regime_candidate": None,
        "regime_evidence_hits": 0,
        "regime_evidence_total": 0,
        "event_candidates": _json([]),
        "direction_sign": 0,
        "scale_conflict": False,
        "scale_conflict_type": _json([]),
        "path_directness_attribute": None,
        "entropy_contest_attribute": None,
        "overlap_contest_attribute": None,
        "activity_attribute": None,
        "gap_direction": "NONE",
        "gap_size_atr": np.nan,
        "evidence_json": _json({}),
        "raw_feature_snapshot": _json(raw_snapshot),
    }
    if not _status_is_ok(row, required_statuses):
        base["status"] = "non_ok"
        return base
    if not all(bool(row[f"threshold_{name}_ready"]) for name in _SOURCE_COLUMNS):
        base["status"] = "insufficient_threshold_history"
        return base

    def q(source: str, label: str) -> float:
        return float(row[threshold_columns[source][label]])

    d = float(row["direction_drift_midlong_raw"])
    abs_d = abs(d)
    agreement = float(row["direction_scale_agreement_midlong"])
    evidence = {
        "direction_strength_high": _rule(abs_d, q("abs_direction", "p70"), ">="),
        "direction_strength_low": _rule(abs_d, q("abs_direction", "p30"), "<="),
        "scale_consistent": _rule(agreement, q("agreement", "p30"), ">"),
        "path_directness_low": _rule(float(row["efficiency_ohlc_min_midlong"]), q("efficiency", "p30"), "<="),
        "entropy_contest_high": _rule(float(row["two_sidedness_entropy_20"]), q("entropy", "p70"), ">="),
        "overlap_contest_high": _rule(float(row["two_sidedness_overlap_20"]), q("overlap", "p70"), ">="),
        "coverage_20_high": _rule(float(row["historical_coverage_20"]), q("coverage20", "p70"), ">="),
        "range_extreme": _rule(float(row["expansion_range_atr_prev"]), q("range", "p90"), ">="),
        "share_high": _rule(float(row["abs_expansion_close_share"]), q("abs_share", "p70"), ">="),
        "share_low": _rule(float(row["abs_expansion_close_share"]), q("abs_share", "p30"), "<="),
    }

    direction_ok = evidence["direction_strength_high"]["hit"] and evidence["scale_consistent"]["hit"]
    balance_hits = [
        evidence["direction_strength_low"]["hit"] and evidence["scale_consistent"]["hit"],
        evidence["path_directness_low"]["hit"],
        evidence["entropy_contest_high"]["hit"] or evidence["overlap_contest_high"]["hit"],
        evidence["coverage_20_high"]["hit"],
    ]
    if direction_ok and d > 0.0:
        regime = "DIRECTIONAL_UP"
        regime_hits, regime_total = 3, 3
    elif direction_ok and d < 0.0:
        regime = "DIRECTIONAL_DOWN"
        regime_hits, regime_total = 3, 3
    elif sum(balance_hits) >= 3:
        regime = "BALANCE_LIKE"
        regime_hits, regime_total = sum(balance_hits), 4
    else:
        regime = "UNRESOLVED"
        regime_hits, regime_total = sum(balance_hits), 4

    events: list[str] = []
    if evidence["range_extreme"]["hit"] and evidence["share_high"]["hit"]:
        x_close = float(row["expansion_close_atr_prev"])
        if x_close > 0.0:
            events.append("EXPANSION_UP")
        elif x_close < 0.0:
            events.append("EXPANSION_DOWN")
    if evidence["range_extreme"]["hit"] and evidence["share_low"]["hit"]:
        events.append("TURBULENT_SHOCK")

    conflict_types: list[str] = []
    if agreement <= q("agreement", "p20"):
        conflict_types.append("LOW_AGREEMENT")
    d5 = float(row["direction_drift_5"])
    d55 = float(row["direction_drift_55"])
    deq = float(row["direction_drift_equal_raw"])
    if np.sign(d5) != np.sign(d55):
        conflict_types.append("D5_D55_SIGN")
    if np.sign(deq) != np.sign(d):
        conflict_types.append("DEQUAL_DMIDLONG_SIGN")

    gap = float(row["expansion_gap_prev_range_atr"])
    base.update(
        {
            "regime_candidate": regime,
            "regime_evidence_hits": int(regime_hits),
            "regime_evidence_total": regime_total,
            "event_candidates": _json(events),
            "direction_sign": int(np.sign(d)),
            "scale_conflict": bool(conflict_types),
            "scale_conflict_type": _json(conflict_types),
            "path_directness_attribute": _attribute(float(row["efficiency_ohlc_min_midlong"]), q("efficiency", "p30"), q("efficiency", "p70")),
            "entropy_contest_attribute": _attribute(float(row["two_sidedness_entropy_20"]), q("entropy", "p30"), q("entropy", "p70")),
            "overlap_contest_attribute": _attribute(float(row["two_sidedness_overlap_20"]), q("overlap", "p30"), q("overlap", "p70")),
            "activity_attribute": _attribute(float(row["activity_level_atr_pct_prev"]), q("activity", "p30"), q("activity", "p70")),
            "gap_direction": "UP" if gap > 0.0 else "DOWN" if gap < 0.0 else "NONE",
            "gap_size_atr": abs(gap),
            "evidence_json": _json(evidence),
            "status": "ok",
        }
    )
    return base


def build_state_coupling_probe(
    bars: pd.DataFrame,
    *,
    low_quantile: float = 0.30,
    high_quantile: float = 0.70,
    threshold_history: str = "expanding",
) -> tuple[pd.DataFrame, StateCouplingProbeMeta]:
    """Build a one-row-per-day causal v0a probe table from frozen v1 features."""
    if not 0.20 < low_quantile < high_quantile < 0.90:
        raise ValueError("quantile roles must satisfy p20 < low < high < p90")
    if threshold_history not in ("expanding", "trailing_5y"):
        raise ValueError("threshold_history must be 'expanding' or 'trailing_5y'")
    work = _merge_features(bars)
    threshold_columns: dict[str, dict[str, str]] = {}
    threshold_frames: list[pd.DataFrame] = []
    quantiles = (0.20, low_quantile, high_quantile, 0.90)
    threshold_method_id = (
        THRESHOLD_METHOD_ID
        if threshold_history == "expanding" and (low_quantile, high_quantile) == (0.30, 0.70)
        else (
            f"causal_{threshold_history}_p20_p{round(low_quantile * 100)}_"
            f"p{round(high_quantile * 100)}_p90_v1"
        )
    )
    for name, source_column in _SOURCE_COLUMNS.items():
        if threshold_history == "expanding":
            thresholds, _ = causal_expanding_quantiles(
                work["date"], work[source_column], quantiles=quantiles
            )
        else:
            thresholds, _ = causal_trailing_years_quantiles(
                work["date"], work[source_column], quantiles=quantiles
            )
        threshold_columns[name] = {}
        renamed: dict[str, str] = {}
        roles = {
            "history_start": "history_start",
            "history_end": "history_end",
            "observation_count": "observation_count",
            "ready": "ready",
            "p20": "p20",
            f"p{round(low_quantile * 100):02d}": "p30",
            f"p{round(high_quantile * 100):02d}": "p70",
            "p90": "p90",
        }
        for source_name, role in roles.items():
            target = f"threshold_{name}_{role}"
            threshold_columns[name][role] = target
            renamed[source_name] = target
        threshold_frames.append(thresholds.rename(columns=renamed))
    work = pd.concat([work, *threshold_frames], axis=1).copy()

    classified = pd.DataFrame(
        [_classify_row(row, threshold_columns) for _, row in work.iterrows()],
        index=work.index,
    )
    out = pd.concat([work, classified], axis=1).copy()
    out.insert(1, "as_of", out["date"])
    out.insert(2, "probe_version", PROBE_VERSION)
    out.insert(3, "input_feature_versions", _json(list(INPUT_FEATURE_VERSIONS)))
    out.insert(4, "threshold_method_id", threshold_method_id)
    out.insert(5, "quantile_method", QUANTILE_METHOD)
    out.insert(6, "regime_rule_order", REGIME_RULE_ORDER)
    out.insert(7, "derived_position_version", DERIVED_POSITION_VERSION)
    out["threshold_history_start"] = out["threshold_abs_direction_history_start"]
    out["threshold_history_end"] = out["date"].shift(1)
    out["threshold_observation_count"] = out[
        [f"threshold_{name}_observation_count" for name in _SOURCE_COLUMNS]
    ].min(axis=1).astype("int64")

    ok = out["status"].eq("ok")
    tolerance = 1e-12
    for column, lower, upper in (
        ("position_bias_20", -1.0, 1.0),
        ("position_bias_55", -1.0, 1.0),
        ("historical_coverage_20", 0.0, 1.0),
        ("historical_coverage_55", 0.0, 1.0),
    ):
        values = out.loc[ok, column]
        if ((values < lower - tolerance) | (values > upper + tolerance)).any():
            raise RuntimeError(f"{column} fell outside [{lower}, {upper}]")

    normal = out.loc[ok]
    output_start = pd.Timestamp(normal["date"].iloc[0]) if not normal.empty else pd.NaT
    meta = StateCouplingProbeMeta(
        probe_version=PROBE_VERSION,
        threshold_method_id=threshold_method_id,
        quantile_method=QUANTILE_METHOD,
        input_feature_versions=INPUT_FEATURE_VERSIONS,
        input_start=pd.Timestamp(out["date"].iloc[0]),
        output_start=output_start,
        output_end=pd.Timestamp(out["date"].iloc[-1]),
        rows=len(out),
    )
    return out, meta