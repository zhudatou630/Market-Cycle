"""Load research-pinned SPX daily snapshots (D-008)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SNAPSHOT_DIR = _PROJECT_ROOT / "data" / "snapshots"

DEFAULT_SNAPSHOT_ID = "spx_daily_2026-07-21_d828fbc8"


def snapshot_dir() -> Path:
    return _SNAPSHOT_DIR


def load_snapshot_meta(snapshot_id: str = DEFAULT_SNAPSHOT_ID) -> dict:
    path = _SNAPSHOT_DIR / f"{snapshot_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Snapshot metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_snapshot(snapshot_id: str = DEFAULT_SNAPSHOT_ID) -> pd.DataFrame:
    """Return full-history daily OHLC for a pinned snapshot.

    Columns: date (datetime64, midnight), open, high, low, close.
    Does not apply research sample filters.
    """
    path = _SNAPSHOT_DIR / f"{snapshot_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Snapshot parquet not found: {path}")

    df = pd.read_parquet(path)
    missing = [c for c in ("date", "open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise RuntimeError(f"Snapshot schema invalid, missing columns: {missing}")

    out = df[["date", "open", "high", "low", "close"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    out = out.reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[["open", "high", "low", "close"]].isna().any().any():
        raise RuntimeError(f"Snapshot {snapshot_id} contains null OHLC")
    return out
