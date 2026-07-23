"""SPX (^GSPC) price layer via yfinance.

Daily OHLC is the sole stored authority; weekly bars are resampled from daily.
Decisions: .pi/grill/2026-07-22-0314——SPX数据层.md
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import pandas as pd
import yfinance as yf

# Project root: 04-Market Cycle/  (file is src/market_cycle/data/bars.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CACHE_PATH = _PROJECT_ROOT / "data" / "raw" / "spx_daily.parquet"

# Canonical Yahoo symbol; aliases for research convenience.
_SYMBOL_ALIASES: dict[str, str] = {
    "^GSPC": "^GSPC",
    "GSPC": "^GSPC",
    "SPX": "^GSPC",
    "spx": "^GSPC",
    "gspc": "^GSPC",
}

_OHLC_COLS = ("open", "high", "low", "close")
Freq = Literal["D", "W", "d", "w", "1d", "1w", "daily", "weekly"]


def _canonicalize(symbol: str) -> str:
    key = symbol.strip()
    if key not in _SYMBOL_ALIASES:
        supported = ", ".join(sorted(set(_SYMBOL_ALIASES)))
        raise ValueError(
            f"Unsupported symbol {symbol!r} in v1. "
            f"Only SPX/^GSPC is implemented. Supported aliases: {supported}"
        )
    return _SYMBOL_ALIASES[key]


def _normalize_history(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned empty history for ^GSPC")

    df = raw.copy()
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
        }
    )
    # Index is DatetimeIndex from yfinance
    dates = pd.to_datetime(df.index)
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_convert(None)
    dates = dates.normalize()

    out = pd.DataFrame(
        {
            "date": dates,
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
        }
    )
    out = out.dropna(subset=list(_OHLC_COLS))
    out = out.drop_duplicates(subset=["date"], keep="last")
    out = out.sort_values("date").reset_index(drop=True)
    # Plain calendar dates (no time component)
    out["date"] = out["date"].dt.date
    return out[["date", *_OHLC_COLS]]


def _fetch_spx_full() -> pd.DataFrame:
    """Full-history daily OHLC for ^GSPC. auto_adjust=False per decision."""
    ticker = yf.Ticker("^GSPC")
    raw = ticker.history(period="max", auto_adjust=False, actions=False)
    return _normalize_history(raw)


def _read_cache(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in ("date", *_OHLC_COLS) if c not in df.columns]
    if missing:
        raise RuntimeError(f"Cache schema invalid, missing columns: {missing}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["date", *_OHLC_COLS]].sort_values("date").reset_index(drop=True)


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Store date as datetime64 for parquet friendliness; read back as date
    to_save = df.copy()
    to_save["date"] = pd.to_datetime(to_save["date"])
    to_save.to_parquet(path, index=False)


def daily_to_weekly(
    daily: pd.DataFrame,
    *,
    week_ending: str = "W-FRI",
) -> pd.DataFrame:
    """Resample daily OHLC to weekly bars (not fetched from the network).

    Rules (standard OHLC aggregation):
      open  = first daily open in the week
      high  = max daily high
      low   = min daily low
      close = last daily close

    ``date`` is the week label (default: week ending Friday, ``W-FRI``).
    The current incomplete week is kept as a partial bar.
    """
    required = {"date", *_OHLC_COLS}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily frame missing columns: {sorted(missing)}")

    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").set_index("date")

    weekly = frame.resample(week_ending).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    )
    weekly = weekly.dropna(subset=list(_OHLC_COLS))
    weekly = weekly.reset_index()
    weekly["date"] = weekly["date"].dt.date
    return weekly[["date", *_OHLC_COLS]].reset_index(drop=True)


def _normalize_freq(freq: str) -> Literal["D", "W"]:
    key = freq.strip().lower()
    if key in {"d", "1d", "day", "daily"}:
        return "D"
    if key in {"w", "1w", "week", "weekly"}:
        return "W"
    raise ValueError(
        f"Unsupported freq {freq!r}. Use 'D'/'daily' or 'W'/'weekly'."
    )


def get_bars(
    symbol: str = "^GSPC",
    refresh: bool = False,
    *,
    freq: Freq = "D",
) -> pd.DataFrame:
    """Return OHLC bars for the requested symbol.

    Parameters
    ----------
    symbol:
        ``^GSPC`` or aliases ``SPX`` / ``GSPC``.
    refresh:
        If True, full re-download daily from yfinance and overwrite cache.
        On failure, raise (no stale fallback).
        If False and cache exists, read cache only.
        If False and cache missing, download once with a warning.
    freq:
        ``D`` (default) daily bars from cache/yfinance.
        ``W`` weekly bars resampled from daily (not stored separately).

    Returns
    -------
    DataFrame with columns: date, open, high, low, close (ascending by date).
    """
    _canonicalize(symbol)  # validate; v1 only ^GSPC
    bar_freq = _normalize_freq(freq)
    cache_file = _CACHE_PATH

    if not refresh and cache_file.exists():
        daily = _read_cache(cache_file)
    else:
        if not refresh and not cache_file.exists():
            warnings.warn(
                f"No local cache at {cache_file}; downloading full ^GSPC history "
                "from yfinance and writing cache. Pass refresh=True to force refresh.",
                UserWarning,
                stacklevel=2,
            )

        # refresh=True or cold cache: network required; hard-fail on error
        try:
            daily = _fetch_spx_full()
        except Exception:
            # Explicit re-raise: do not fall back to stale cache
            raise

        _write_cache(daily, cache_file)
        daily = _read_cache(cache_file)

    if bar_freq == "D":
        return daily
    return daily_to_weekly(daily)


def cache_path() -> Path:
    """Absolute path to the SPX daily parquet cache."""
    return _CACHE_PATH
