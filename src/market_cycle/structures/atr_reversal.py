"""Single-scale ATR-threshold causal swing candidate for future structure research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from market_cycle.data.geometry import ohlc_min_path_length

PATH_ID = "bm_g_01_atr_reversal_v1"
PATH_MULTIPLIERS: tuple[float, ...] = (1.0, 2.0, 3.0)

_REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "atr_pct_14")
_SNAPSHOT_COLUMNS = (
    "date",
    "as_of",
    "path_version",
    "k",
    "direction",
    "anchor_at",
    "anchor_price",
    "confirmed_at",
    "effective_at",
    "current_candidate_id",
    "since_anchor_displacement",
    "since_anchor_path_length_min",
    "since_anchor_path_efficiency",
    "since_anchor_path_efficiency_status",
    "maximum_counter_move",
    "candidate_age_bars",
)
_LEDGER_COLUMNS = (
    "candidate_seq",
    "path_version",
    "k",
    "candidate_direction",
    "anchor_type",
    "anchor_at",
    "candidate_at",
    "anchor_price",
    "threshold",
    "is_warmup",
    "status",
    "confirmed_at",
    "invalidated_at",
    "effective_at",
    "replacement_candidate_seq",
)


@dataclass(frozen=True)
class PathMeta:
    """Metadata for a single multiplier's causal path replay."""

    path_id: str
    multiplier: float
    seed_at: pd.Timestamp
    seed_price: float
    seed_threshold: float
    path_ready_at: pd.Timestamp | None
    output_start: pd.Timestamp | None
    output_end: pd.Timestamp | None
    snapshot_rows: int
    candidate_rows: int


@dataclass(frozen=True)
class PathResult:
    """Daily as-of snapshots and candidate lifecycle ledger for one $k$."""

    snapshots: pd.DataFrame
    candidates: pd.DataFrame
    meta: PathMeta


@dataclass
class _Candidate:
    seq: int
    direction: str
    anchor_type: str
    index: int
    at: pd.Timestamp
    price: float
    threshold: float
    warmup: bool
    record: dict[str, object]


def _validated_bars(bars: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _REQUIRED_COLUMNS if column not in bars.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")

    out = bars.loc[:, _REQUIRED_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    for column in _REQUIRED_COLUMNS[1:]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    if out.isna().any().any():
        raise ValueError("bars contain null OHLC/atr_pct_14 values")
    if not out["date"].is_monotonic_increasing or out["date"].duplicated().any():
        raise ValueError("bars must have unique dates in ascending order")
    if (out["high"] < out["low"]).any():
        raise ValueError("bars contain high < low")
    if ((out["open"] < out["low"]) | (out["open"] > out["high"])).any():
        raise ValueError("bars contain open outside [low, high]")
    if ((out["close"] < out["low"]) | (out["close"] > out["high"])).any():
        raise ValueError("bars contain close outside [low, high]")
    if (out["atr_pct_14"] <= 0).any():
        raise ValueError("bars contain non-positive atr_pct_14")
    return out.reset_index(drop=True)


def _empty_snapshots() -> pd.DataFrame:
    return pd.DataFrame(columns=_SNAPSHOT_COLUMNS)


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=_LEDGER_COLUMNS)


def build_atr_reversal_path(
    bars: pd.DataFrame,
    *,
    multiplier: float = 2.0,
) -> PathResult:
    """Replay one ATR-threshold causal path reference through daily bars.

    The initial seed is internal warmup only. Once a direction has been seeded,
    each new extreme becomes a replacement candidate with a frozen threshold.
    A close crossing the candidate's threshold confirms one reversal at the
    close. The returned snapshots start at the first such confirmation.
    """
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("multiplier must be finite and > 0")

    work = _validated_bars(bars)
    if len(work) < 2:
        raise ValueError("at least two bars are required for causal path replay")

    path_length = ohlc_min_path_length(work).to_numpy(dtype=float)
    path_length_prefix = np.cumsum(np.nan_to_num(path_length, nan=0.0))
    dates = work["date"].to_numpy()
    close = work["close"].to_numpy(dtype=float)
    high = work["high"].to_numpy(dtype=float)
    low = work["low"].to_numpy(dtype=float)
    atr_pct = work["atr_pct_14"].to_numpy(dtype=float)

    seed_at = pd.Timestamp(dates[0])
    seed_price = float(close[0])
    seed_threshold = float(multiplier * seed_price * atr_pct[0])
    direction: str | None = None
    current_candidate: _Candidate | None = None
    active_anchor: _Candidate | None = None
    active_confirmed_at: pd.Timestamp | None = None
    path_ready_at: pd.Timestamp | None = None
    maximum_counter_move = 0.0
    next_seq = 1
    ledger_records: list[dict[str, object]] = []
    snapshot_records: list[dict[str, object]] = []

    def new_candidate(
        *,
        index: int,
        candidate_direction: str,
        anchor_type: str,
        price: float,
        warmup: bool,
    ) -> _Candidate:
        nonlocal next_seq
        at = pd.Timestamp(dates[index])
        threshold = float(multiplier * close[index] * atr_pct[index])
        record: dict[str, object] = {
            "candidate_seq": next_seq,
            "path_version": PATH_ID,
            "k": float(multiplier),
            "candidate_direction": candidate_direction,
            "anchor_type": anchor_type,
            "anchor_at": at,
            "candidate_at": at,
            "anchor_price": float(price),
            "threshold": threshold,
            "is_warmup": warmup,
            "status": "active",
            "confirmed_at": pd.NaT,
            "invalidated_at": pd.NaT,
            "effective_at": pd.NaT,
            "replacement_candidate_seq": pd.NA,
        }
        ledger_records.append(record)
        candidate = _Candidate(
            seq=next_seq,
            direction=candidate_direction,
            anchor_type=anchor_type,
            index=index,
            at=at,
            price=float(price),
            threshold=threshold,
            warmup=warmup,
            record=record,
        )
        next_seq += 1
        return candidate

    def invalidate(candidate: _Candidate, *, at: pd.Timestamp, replacement_seq: int) -> None:
        candidate.record["status"] = "invalidated"
        candidate.record["invalidated_at"] = at
        candidate.record["replacement_candidate_seq"] = replacement_seq

    def confirm(candidate: _Candidate, *, at: pd.Timestamp) -> None:
        candidate.record["status"] = "confirmed"
        candidate.record["confirmed_at"] = at
        candidate.record["effective_at"] = at

    def path_snapshot(index: int) -> dict[str, object]:
        if active_anchor is None or active_confirmed_at is None or current_candidate is None or direction is None:
            raise RuntimeError("cannot snapshot without an active confirmed path")

        anchor_index = active_anchor.index
        anchor_price = active_anchor.price
        anchor_close = close[anchor_index]
        anchor_partial_path = abs(anchor_close - anchor_price)
        later_path = path_length_prefix[index] - path_length_prefix[anchor_index]
        path_total = float(anchor_partial_path + later_path)
        displacement = float(close[index] - anchor_price)
        if path_total == 0.0:
            efficiency = np.nan
            efficiency_status = "zero_path"
        else:
            efficiency = abs(displacement) / path_total
            efficiency_status = "ok"
            if efficiency < -1e-12 or efficiency > 1.0 + 1e-12:
                raise RuntimeError("anchored path efficiency fell outside [0, 1]")

        return {
            "date": pd.Timestamp(dates[index]),
            "as_of": pd.Timestamp(dates[index]),
            "path_version": PATH_ID,
            "k": float(multiplier),
            "direction": direction,
            "anchor_at": active_anchor.at,
            "anchor_price": anchor_price,
            "confirmed_at": active_confirmed_at,
            "effective_at": active_confirmed_at,
            "current_candidate_id": current_candidate.seq,
            "since_anchor_displacement": displacement,
            "since_anchor_path_length_min": path_total,
            "since_anchor_path_efficiency": efficiency,
            "since_anchor_path_efficiency_status": efficiency_status,
            "maximum_counter_move": maximum_counter_move,
            "candidate_age_bars": index - current_candidate.index,
        }

    def begin_confirmed_path(candidate: _Candidate, *, index: int, next_direction: str) -> None:
        nonlocal active_anchor, active_confirmed_at, current_candidate, direction
        nonlocal path_ready_at, maximum_counter_move
        at = pd.Timestamp(dates[index])
        confirm(candidate, at=at)
        active_anchor = candidate
        active_confirmed_at = at
        direction = next_direction
        if path_ready_at is None:
            path_ready_at = at
        maximum_counter_move = 0.0
        next_candidate_direction = "up" if next_direction == "down" else "down"
        current_candidate = new_candidate(
            index=index,
            candidate_direction=next_candidate_direction,
            anchor_type="close",
            price=float(close[index]),
            warmup=False,
        )

    for index in range(1, len(work)):
        at = pd.Timestamp(dates[index])

        if direction is None:
            if close[index] >= seed_price + seed_threshold:
                direction = "up"
                current_candidate = new_candidate(
                    index=index,
                    candidate_direction="down",
                    anchor_type="high",
                    price=float(high[index]),
                    warmup=True,
                )
                # A high necessarily occurs before this day's close, so an
                # immediate close-based reversal can be confirmed causally.
                if close[index] <= current_candidate.price - current_candidate.threshold:
                    begin_confirmed_path(current_candidate, index=index, next_direction="down")
            elif close[index] <= seed_price - seed_threshold:
                direction = "down"
                current_candidate = new_candidate(
                    index=index,
                    candidate_direction="up",
                    anchor_type="low",
                    price=float(low[index]),
                    warmup=True,
                )
                if close[index] >= current_candidate.price + current_candidate.threshold:
                    begin_confirmed_path(current_candidate, index=index, next_direction="up")
            if active_anchor is not None:
                snapshot_records.append(path_snapshot(index))
            continue

        if current_candidate is None:
            raise RuntimeError("path direction exists without a reversal candidate")

        if direction == "up":
            if high[index] > current_candidate.price:
                replacement = new_candidate(
                    index=index,
                    candidate_direction="down",
                    anchor_type="high",
                    price=float(high[index]),
                    warmup=path_ready_at is None,
                )
                invalidate(current_candidate, at=at, replacement_seq=replacement.seq)
                current_candidate = replacement

            if close[index] <= current_candidate.price - current_candidate.threshold:
                begin_confirmed_path(current_candidate, index=index, next_direction="down")
        else:
            if low[index] < current_candidate.price:
                replacement = new_candidate(
                    index=index,
                    candidate_direction="up",
                    anchor_type="low",
                    price=float(low[index]),
                    warmup=path_ready_at is None,
                )
                invalidate(current_candidate, at=at, replacement_seq=replacement.seq)
                current_candidate = replacement

            if close[index] >= current_candidate.price + current_candidate.threshold:
                begin_confirmed_path(current_candidate, index=index, next_direction="up")

        if active_anchor is not None:
            # A same-day confirmation has reset the new path's counter move to 0.
            if current_candidate.anchor_type != "close" or current_candidate.index != index:
                if direction == "up":
                    maximum_counter_move = max(maximum_counter_move, current_candidate.price - close[index])
                else:
                    maximum_counter_move = max(maximum_counter_move, close[index] - current_candidate.price)
            snapshot_records.append(path_snapshot(index))

    snapshots = pd.DataFrame(snapshot_records, columns=_SNAPSHOT_COLUMNS)
    candidates = pd.DataFrame(ledger_records, columns=_LEDGER_COLUMNS)
    output_start = pd.Timestamp(snapshots["date"].iloc[0]) if not snapshots.empty else None
    output_end = pd.Timestamp(snapshots["date"].iloc[-1]) if not snapshots.empty else None
    meta = PathMeta(
        path_id=PATH_ID,
        multiplier=float(multiplier),
        seed_at=seed_at,
        seed_price=seed_price,
        seed_threshold=seed_threshold,
        path_ready_at=path_ready_at,
        output_start=output_start,
        output_end=output_end,
        snapshot_rows=len(snapshots),
        candidate_rows=len(candidates),
    )
    return PathResult(snapshots=snapshots, candidates=candidates, meta=meta)


def build_atr_reversal_paths(
    bars: pd.DataFrame,
    *,
    multipliers: Sequence[float] = PATH_MULTIPLIERS,
) -> dict[float, PathResult]:
    """Build the same causal path reference for a finite multiplier family."""
    resolved = tuple(float(multiplier) for multiplier in multipliers)
    if not resolved:
        raise ValueError("multipliers must not be empty")
    if len(set(resolved)) != len(resolved):
        raise ValueError("multipliers must be unique")
    return {multiplier: build_atr_reversal_path(bars, multiplier=multiplier) for multiplier in resolved}