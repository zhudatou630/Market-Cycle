"""Phase 1A shared ruler: relative TR/ATR scale on the research calendar (D-010).

Public research frame columns:
  date, tr_pct, atr_pct_14

Absolute TR/ATR are computation intermediates only and are never returned.
Measurement modules must call ``build_ruler`` / ``get_research_bars`` instead of
re-implementing TR or ATR.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from market_cycle.data.snapshot import (
    DEFAULT_SNAPSHOT_ID,
    load_snapshot,
    load_snapshot_meta,
    snapshot_dir,
)

# OHLC-clean floor for this project era (warmup may load from here; not research rows).
OHLC_CLEAN_START = pd.Timestamp("1984-01-03")
SAMPLE_ID = "spx_ohlc_main_1984"
RULER_ID = "ruler_v1_atr14w28"

ATR_N = 14
ATR_WARMUP_TR_BARS = 2 * ATR_N  # first research row = the 28th complete TR bar

RulerColumn = Literal["date", "tr_pct", "atr_pct_14"]
RESEARCH_COLUMNS: tuple[str, ...] = ("date", "tr_pct", "atr_pct_14")


@dataclass(frozen=True)
class RulerMeta:
    snapshot_id: str
    sample_id: str
    ruler_id: str
    ohlc_clean_start: pd.Timestamp
    research_start: pd.Timestamp
    research_end: pd.Timestamp
    atr_n: int
    atr_warmup_tr_bars: int
    rows: int


def classic_true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Classic TR; index 0 is NaN (no previous close)."""
    prev_close = np.roll(close, 1)
    tr = np.empty(len(close), dtype=float)
    tr[0] = np.nan
    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - prev_close[1:])
    lc = np.abs(low[1:] - prev_close[1:])
    tr[1:] = np.maximum(hl, np.maximum(hc, lc))
    return tr


def wilder_atr(tr: np.ndarray, n: int = ATR_N) -> np.ndarray:
    """Wilder ATR on a TR series that may start with NaNs.

    Seed = SMA of the first ``n`` finite TR values at the index of the n-th
    finite TR; then recursive Wilder smoothing. Positions without a defined
    ATR remain NaN.
    """
    atr = np.full(len(tr), np.nan, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(tr))
    if len(finite_idx) < n:
        return atr

    seed_positions = finite_idx[:n]
    seed_last = int(seed_positions[-1])
    atr[seed_last] = float(np.mean(tr[seed_positions]))

    # Walk forward day by day so calendar alignment is preserved.
    for i in range(seed_last + 1, len(tr)):
        if not np.isfinite(tr[i]) or not np.isfinite(atr[i - 1]):
            continue
        atr[i] = ((n - 1) * atr[i - 1] + tr[i]) / n
    return atr


def _research_start_index(finite_tr_indices: np.ndarray, warmup_tr_bars: int) -> int:
    if len(finite_tr_indices) < warmup_tr_bars:
        raise ValueError(
            f"Need at least {warmup_tr_bars} complete TR bars for ruler warmup; "
            f"got {len(finite_tr_indices)}"
        )
    # Include the warmup_tr_bars-th finite TR as the first research row.
    return int(finite_tr_indices[warmup_tr_bars - 1])


def build_ruler(
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    *,
    atr_n: int = ATR_N,
    warmup_tr_bars: int = ATR_WARMUP_TR_BARS,
    ohlc_clean_start: pd.Timestamp | str = OHLC_CLEAN_START,
) -> tuple[pd.DataFrame, RulerMeta]:
    """Build the public research ruler frame (no absolute TR/ATR columns).

    Returns
    -------
    frame:
        Columns ``date, tr_pct, atr_pct_14`` from the first fully-warmed day
        through snapshot end. No missing ruler values.
    meta:
        Snapshot/sample/ruler identifiers and the resolved research_start.
    """
    if atr_n < 1:
        raise ValueError("atr_n must be >= 1")
    if warmup_tr_bars < atr_n:
        raise ValueError("warmup_tr_bars must be >= atr_n")

    ohlc_clean_start = pd.Timestamp(ohlc_clean_start).normalize()
    raw = load_snapshot(snapshot_id)
    work = raw.loc[raw["date"] >= ohlc_clean_start].copy().reset_index(drop=True)
    if work.empty:
        raise RuntimeError(f"No bars on/after ohlc_clean_start={ohlc_clean_start.date()}")

    high = work["high"].to_numpy(dtype=float)
    low = work["low"].to_numpy(dtype=float)
    close = work["close"].to_numpy(dtype=float)

    tr = classic_true_range(high, low, close)
    atr = wilder_atr(tr, n=atr_n)

    with np.errstate(divide="ignore", invalid="ignore"):
        tr_pct = tr / close
        atr_pct = atr / close

    finite_tr = np.flatnonzero(np.isfinite(tr))
    start_i = _research_start_index(finite_tr, warmup_tr_bars)

    # Research rows require both relative measures.
    if not np.isfinite(tr_pct[start_i]) or not np.isfinite(atr_pct[start_i]):
        raise RuntimeError("Ruler research_start lacks finite tr_pct/atr_pct_14")

    research = pd.DataFrame(
        {
            "date": work["date"].iloc[start_i:].to_numpy(),
            "tr_pct": tr_pct[start_i:],
            "atr_pct_14": atr_pct[start_i:],
        }
    )
    research = research.reset_index(drop=True)

    if research[["tr_pct", "atr_pct_14"]].isna().any().any():
        raise RuntimeError("Research ruler frame contains NaNs after warmup cut")

    meta = RulerMeta(
        snapshot_id=snapshot_id,
        sample_id=SAMPLE_ID,
        ruler_id=RULER_ID,
        ohlc_clean_start=ohlc_clean_start,
        research_start=pd.Timestamp(research["date"].iloc[0]),
        research_end=pd.Timestamp(research["date"].iloc[-1]),
        atr_n=atr_n,
        atr_warmup_tr_bars=warmup_tr_bars,
        rows=len(research),
    )
    return research, meta


def research_artifact_stem(
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    ruler_id: str = RULER_ID,
) -> str:
    """Filename stem for the materialized daily research table."""
    return f"{snapshot_id}__{ruler_id}__research"


def research_bars_path(
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    ruler_id: str = RULER_ID,
) -> Path:
    return snapshot_dir() / f"{research_artifact_stem(snapshot_id, ruler_id)}.parquet"


def research_bars_meta_path(
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    ruler_id: str = RULER_ID,
) -> Path:
    return snapshot_dir() / f"{research_artifact_stem(snapshot_id, ruler_id)}.json"


def _compose_research_bars(
    snapshot_id: str,
    **ruler_kwargs,
) -> tuple[pd.DataFrame, RulerMeta]:
    ruler, meta = build_ruler(snapshot_id, **ruler_kwargs)
    ohlc = load_snapshot(snapshot_id)
    out = ruler.merge(ohlc, on="date", how="left", validate="one_to_one")
    if out[["open", "high", "low", "close"]].isna().any().any():
        raise RuntimeError("OHLC join to ruler failed")
    out = out[["date", "open", "high", "low", "close", "tr_pct", "atr_pct_14"]]
    return out.reset_index(drop=True), meta


def _meta_from_frame(frame: pd.DataFrame, snapshot_id: str, **ruler_kwargs) -> RulerMeta:
    atr_n = int(ruler_kwargs.get("atr_n", ATR_N))
    warmup = int(ruler_kwargs.get("warmup_tr_bars", ATR_WARMUP_TR_BARS))
    ohlc_clean = pd.Timestamp(ruler_kwargs.get("ohlc_clean_start", OHLC_CLEAN_START)).normalize()
    return RulerMeta(
        snapshot_id=snapshot_id,
        sample_id=SAMPLE_ID,
        ruler_id=RULER_ID,
        ohlc_clean_start=ohlc_clean,
        research_start=pd.Timestamp(frame["date"].iloc[0]),
        research_end=pd.Timestamp(frame["date"].iloc[-1]),
        atr_n=atr_n,
        atr_warmup_tr_bars=warmup,
        rows=len(frame),
    )


def write_research_bars(
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    **ruler_kwargs,
) -> tuple[pd.DataFrame, RulerMeta, Path]:
    """Compute research bars and write parquet + json next to the OHLC snapshot."""
    frame, meta = _compose_research_bars(snapshot_id, **ruler_kwargs)
    path = research_bars_path(snapshot_id, meta.ruler_id)
    meta_path = research_bars_meta_path(snapshot_id, meta.ruler_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    to_save = frame.copy()
    to_save["date"] = pd.to_datetime(to_save["date"])
    to_save.to_parquet(path, index=False)

    snap_meta = load_snapshot_meta(snapshot_id)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "artifact_id": research_artifact_stem(snapshot_id, meta.ruler_id),
        "kind": "research_bars",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "snapshot_id": meta.snapshot_id,
        "source_snapshot_sha256": snap_meta["sha256"],
        "sample_id": meta.sample_id,
        "ruler_id": meta.ruler_id,
        "ohlc_clean_start": meta.ohlc_clean_start.date().isoformat(),
        "research_start": meta.research_start.date().isoformat(),
        "research_end": meta.research_end.date().isoformat(),
        "atr_n": meta.atr_n,
        "atr_warmup_tr_bars": meta.atr_warmup_tr_bars,
        "rows": meta.rows,
        "columns": list(frame.columns),
        "path": str(path).replace("\\", "/"),
        "sha256": sha,
        "byte_size": path.stat().st_size,
        "notes": (
            "Daily research table: OHLC + ruler columns at the same row level. "
            "Regenerate with write_research_bars(); do not hand-edit."
        ),
    }
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return frame, meta, path


def load_research_bars(
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    ruler_id: str = RULER_ID,
) -> tuple[pd.DataFrame, RulerMeta]:
    """Load the materialized daily research table (OHLC + ruler)."""
    path = research_bars_path(snapshot_id, ruler_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Research bars not found at {path}. Run write_research_bars() once to materialize."
        )
    df = pd.read_parquet(path)
    expected = ["date", "open", "high", "low", "close", "tr_pct", "atr_pct_14"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise RuntimeError(f"Research bars schema invalid, missing: {missing}")
    out = df[expected].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values("date").reset_index(drop=True)
    if out.isna().any().any():
        raise RuntimeError("Research bars contain nulls")
    meta = _meta_from_frame(out, snapshot_id)
    return out, meta


def get_research_bars(
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    *,
    refresh: bool = False,
    **ruler_kwargs,
) -> tuple[pd.DataFrame, RulerMeta]:
    """Daily research table: OHLC + ruler on the research calendar.

    Columns: date, open, high, low, close, tr_pct, atr_pct_14.

    By default reads the materialized parquet next to the snapshot (same level as
    OHLC for downstream modules). Pass refresh=True to recompute from the pinned
    OHLC snapshot and overwrite the research file.
    """
    path = research_bars_path(snapshot_id)
    if refresh or not path.exists():
        frame, meta, _ = write_research_bars(snapshot_id, **ruler_kwargs)
        return frame, meta
    return load_research_bars(snapshot_id)
