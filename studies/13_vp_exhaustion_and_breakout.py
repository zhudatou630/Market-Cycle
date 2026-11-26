#!/usr/bin/env python3
"""
Market Cycle V2 — Study 13: 时间序列连续量价物理量能否预警趋势衰竭与突破失败？

严格执行 decisions.md「13-预注册」冻结口径（封存区零触碰，先硬截断再算任何未来标签）：
  - A 股申万一级（31 行业，SW2021 L1）：仅读 year=2016..2024 分区，硬截断 <= 2024-10-31
  - 美股 SPY 独立只读快照（raw OHLCV + dividends，因果前向总回报）：硬截断 <= 2019-12-31，评估 1994-01-01 起

4 个零拐点连续量价物理量（方向统一：值越大 = 越拥挤 / 放量滞涨 / 危险）：
  1. VR   = Mean_20(ln V) - Mean_80(ln V)
  2. VPC  = SpearmanCorr_20(Δln V, ln(C/O))   [若第0步 SPY open==前收 >5% 自动退回 ln(C/C_{-1})]
  3. VRD  = -SpearmanCorr_20(Δln V, TR / C_{-1})
  4. nCMF = -( sum_20(V * CLV) / sum_20(V) - Mean_20(CLV) ),  CLV = (2C - H - L) / (H - L)

两考场、唯一主裁判 Y = 未来 20 日固定踩雷率 crash_20（A股 <= -8%，SPY <= -5%）：
  - 考场一 up_high（特征截至 t）   : C > MA60 & r20 > 0 & r60 > 0 & pos120 >= 0.60
  - 考场二 brk20  （特征截至 t-1） : 20日首次收盘新高（冷却5天）
  - 16 个主检验（4 特征 × 2 考场 × 2 市场）统一 calendar-aligned event_ols_hac (lag=30) + BH-FDR
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
SPY_PATH = REPO_ROOT / "data" / "snapshots" / "spy_daily_2026-11-25_c478bcc7.parquet"
SPX_PATH = REPO_ROOT / "data" / "snapshots" / "spx_daily_2026-07-21_d828fbc8.parquet"
MARKET_DATA_ROOT = Path("/home/zhujunshen/market-data")
CATALOG_DB = MARKET_DATA_ROOT / "catalog" / "metadata.sqlite"

SPY_DEV_START = pd.Timestamp("1994-01-01")
SPY_DEV_END = pd.Timestamp("2019-12-31")
SW_DEV_START = pd.Timestamp("2016-01-01")
SW_DEV_END = pd.Timestamp("2024-10-31")

EPS = 1e-12
HAC_LAG = 30
CRASH_NORM_K = 1.25
FEATURES = ["VR", "VPC", "VRD", "nCMF"]
FEAT_LABELS = {
    "VR": "量能温度 VR_20,80",
    "VPC": "量价硬推 VPC_20",
    "VRD": "量幅脱节 VRD_20",
    "nCMF": "放量回落 -ΔCMF_20",
}
CRASH_THRESH = {
    "A股申万L1": {10: -0.06, 20: -0.08, 40: -0.10},
    "SPY": {10: -0.035, 20: -0.05, 40: -0.07},
}
MAG_GATE_PP = {"A股申万L1": 3.0, "SPY": 2.0}


# ---------------------------------------------------------------------------
# 1. Math Helpers: Tie-Aware Rank, Rolling Spearman, Normal p-val, BH-FDR, HAC
# ---------------------------------------------------------------------------
def rankdata_avg_1d(x: np.ndarray) -> np.ndarray:
    """Exact tie-aware average rank (1-indexed), matching pandas rank(method='average')."""
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sx[j] == sx[i]:
            j += 1
        avg_r = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_r
        i = j
    return ranks


def rolling_spearman_1d(x: np.ndarray, y: np.ndarray, window: int = 20) -> np.ndarray:
    """Strictly causal rolling Spearman rank correlation over [t-window+1, t]."""
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    for t in range(window - 1, n):
        xw = x[t - window + 1 : t + 1]
        yw = y[t - window + 1 : t + 1]
        if not (np.all(np.isfinite(xw)) and np.all(np.isfinite(yw))):
            continue
        rx = rankdata_avg_1d(xw)
        ry = rankdata_avg_1d(yw)
        rx = rx - rx.mean()
        ry = ry - ry.mean()
        den = math.sqrt(float(np.dot(rx, rx) * np.dot(ry, ry)))
        if den > EPS:
            out[t] = float(np.dot(rx, ry)) / den
        else:
            out[t] = 0.0
    return out


def causal_rolling_pct_rank(x: np.ndarray, window: int = 250, min_periods: int = 60) -> np.ndarray:
    """Causal percentile rank of x[t] within x[max(0, t-window+1) : t+1], in [0, 1]."""
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        xt = x[t]
        if not np.isfinite(xt):
            continue
        lo = max(0, t - window + 1)
        w = x[lo : t + 1]
        w = w[np.isfinite(w)]
        if len(w) < min_periods:
            continue
        less = float(np.sum(w < xt))
        eq = float(np.sum(w == xt))
        out[t] = (less + 0.5 * eq) / len(w)
    return out


def two_sided_p(tstat: float) -> float:
    if not np.isfinite(tstat):
        return np.nan
    return math.erfc(abs(float(tstat)) / math.sqrt(2.0))


def bh_fdr(pvals: list[float]) -> list[float]:
    m = len(pvals)
    arr = np.array(pvals, dtype=float)
    q = np.full(m, np.nan, dtype=float)
    valid = np.where(np.isfinite(arr))[0]
    if len(valid) == 0:
        return q.tolist()
    pv = arr[valid]
    order = np.argsort(pv)
    k = len(pv)
    ranked_q = pv[order] * k / np.arange(1, k + 1, dtype=float)
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0.0, 1.0)
    out_v = np.empty(k, dtype=float)
    out_v[order] = ranked_q
    q[valid] = out_v
    return q.tolist()


def event_ols_hac(
    y: np.ndarray,
    X: np.ndarray,
    dates: np.ndarray,
    all_trading_dates: np.ndarray,
    lag: int = HAC_LAG,
) -> dict[str, np.ndarray | float]:
    """
    Calendar-aligned Driscoll-Kraay / Newey-West HAC estimator (from Study 03).
    Maps scores onto the full market trading calendar (0 on non-event days) before
    applying Bartlett kernel of bandwidth `lag` trading days.
    """
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    scores = X * resid[:, None]

    T = len(all_trading_dates)
    idx = np.searchsorted(all_trading_dates, dates)
    assert np.all(all_trading_dates[idx] == dates), "Dates not aligned with trading calendar"
    h = np.zeros((T, k), dtype=float)
    np.add.at(h, idx, scores)

    omega = h.T @ h
    K = min(lag, T - 1)
    for ell in range(1, K + 1):
        w = 1.0 - ell / (K + 1.0)
        g_ell = h[ell:].T @ h[:-ell]
        omega += w * (g_ell + g_ell.T)

    cov = xtx_inv @ omega @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tstat = np.where(se > EPS, beta / se, np.nan)
    return {"beta": beta, "se": se, "t": tstat, "n": n, "resid": resid}


# ---------------------------------------------------------------------------
# 2. Data Loading (Read-Only, Hard Truncation Before Any Forward Labels)
# ---------------------------------------------------------------------------
def build_causal_total_return_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic, strictly causal forward total-return adjustment for SPY:
    On ex-dividend day t (dividends[t] > 0), multiply day t..end by C_{t-1} / (C_{t-1} - D_t).
    Guarantees:
      1. Adding future dividends never alters historical prices.
      2. Zero mechanical overnight gap-down on ex-dividend open/low/close.
      3. Intraday ratios (C/O, CLV) remain strictly invariant.
    """
    g = df.sort_values("date").reset_index(drop=True).copy()
    c = g["close"].to_numpy(dtype=float)
    div = g["dividends"].to_numpy(dtype=float)
    splits = g["stock_splits"].to_numpy(dtype=float) if "stock_splits" in g.columns else np.zeros(len(g))
    assert np.all(splits == 0.0), "Unexpected non-zero stock split in SPY"

    step_mult = np.ones(len(g), dtype=float)
    for i in range(1, len(g)):
        if div[i] > 0.0:
            denom = c[i - 1] - div[i]
            assert denom > 0.0, f"Invalid dividend on {g['date'].iloc[i]}"
            step_mult[i] = c[i - 1] / denom
    cum_mult = np.cumprod(step_mult)

    for col in ["open", "high", "low", "close"]:
        g[f"{col}_raw"] = g[col]
        g[col] = g[col] * cum_mult
    g["tr_mult"] = cum_mult
    return g


def load_spy_dev() -> pd.DataFrame:
    df = pd.read_parquet(SPY_PATH).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    assert df["date"].is_unique, "SPY dates not unique"
    # HARD CUTOFF BEFORE ANY LABEL OR FEATURE CALCULATION
    df = df[df["date"] <= SPY_DEV_END].copy()
    df = build_causal_total_return_ohlc(df)
    df["asset"] = "SPY"
    df["vol"] = df["volume"].astype(float)
    df["amount"] = df["close_raw"].astype(float) * df["vol"]
    return df[["asset", "date", "open", "high", "low", "close", "vol", "amount", "open_raw", "close_raw", "adj_close", "dividends"]].reset_index(drop=True)


def load_sw_l1_dev() -> tuple[pd.DataFrame, str]:
    uri = f"file:{CATALOG_DB}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        snap = pd.read_sql_query(
            """
            SELECT snapshot_id, end_date, created_at, gate
            FROM data_snapshot
            WHERE (gate = 'READY' OR gate LIKE '%_READY') AND gate != 'NOT_READY'
            ORDER BY end_date DESC, created_at DESC
            LIMIT 1
            """,
            conn,
        )
        assert len(snap) == 1, "No READY snapshot found"
        sid = str(snap.iloc[0]["snapshot_id"])
        man = pd.read_sql_query(
            """
            SELECT partition_key, file_path
            FROM table_manifest
            WHERE snapshot_id = ? AND dataset = 'sw_daily'
            ORDER BY partition_key
            """,
            conn,
            params=(sid,),
        )
    finally:
        conn.close()

    # Never touch 2025/2026 holdout partitions
    dev_years = {f"year={y}" for y in range(2016, 2025)}
    files = []
    for _, row in man.iterrows():
        if row["partition_key"] in dev_years:
            p = Path(row["file_path"])
            if not p.is_absolute():
                p = MARKET_DATA_ROOT / p
            files.append(p)
    assert files, "No sw_daily dev partitions found"

    parts = [
        pd.read_parquet(
            f,
            columns=["src", "level", "index_code", "trade_date", "open", "high", "low", "close", "vol", "amount"],
        )
        for f in files
    ]
    df = pd.concat(parts, ignore_index=True)
    df = df[(df["src"] == "SW2021") & (df["level"] == "L1")].copy()
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    # HARD CUTOFF BEFORE ANY LABEL OR FEATURE CALCULATION
    df = df[(df["date"] >= SW_DEV_START) & (df["date"] <= SW_DEV_END)].copy()
    df = df.rename(columns={"index_code": "asset"})
    df = df[["asset", "date", "open", "high", "low", "close", "vol", "amount"]].sort_values(["asset", "date"]).reset_index(drop=True)
    assert not df.duplicated(["asset", "date"]).any(), "Duplicate (asset, date) in SW L1"
    return df, sid


def is_quad_witching_or_halfday_spy(dates: pd.Series) -> np.ndarray:
    """Deterministic US quad-witching dates: 3rd Friday of Mar/Jun/Sep/Dec."""
    dt = pd.to_datetime(dates)
    is_qw = (
        dt.dt.month.isin([3, 6, 9, 12])
        & (dt.dt.weekday == 4)
        & (dt.dt.day >= 15)
        & (dt.dt.day <= 21)
    )
    return is_qw.to_numpy(dtype=bool)


# ---------------------------------------------------------------------------
# 3. Feature & Label Construction (Strictly Causal per Asset)
# ---------------------------------------------------------------------------
def compute_asset_features_and_labels(
    g: pd.DataFrame,
    market_name: str,
    use_close_to_close_vpc: bool = False,
    vol_col: str = "vol",
    exclude_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    o = g["open"].to_numpy(dtype=float)
    h = g["high"].to_numpy(dtype=float)
    l = g["low"].to_numpy(dtype=float)
    c = g["close"].to_numpy(dtype=float)
    v = g[vol_col].to_numpy(dtype=float).copy()
    if exclude_mask is not None:
        # Replace excluded calendar-spike days' volume with NaN? No: for clean rolling
        # sensitivity, mask volume/dlogv on those specific days or winsorize/drop in corr.
        # Cleanest causal way: mark v on excluded dates as NaN and forward-fill prior day's v
        # so rolling windows stay intact without calendar spike distortion.
        v_clean = v.copy()
        for idx in np.where(exclude_mask)[0]:
            if idx > 0:
                v_clean[idx] = v_clean[idx - 1]
        v = v_clean

    n = len(g)
    prev_c = np.empty(n)
    prev_c[0] = np.nan
    prev_c[1:] = c[:-1]

    ret1 = np.empty(n)
    ret1[0] = np.nan
    ret1[1:] = c[1:] / c[:-1] - 1.0

    log_c = np.log(c)
    log_v = np.where(v > 0.0, np.log(v), np.nan)
    dlog_v = np.empty(n)
    dlog_v[0] = np.nan
    dlog_v[1:] = log_v[1:] - log_v[:-1]

    if use_close_to_close_vpc:
        price_chg_vpc = np.empty(n)
        price_chg_vpc[0] = np.nan
        price_chg_vpc[1:] = log_c[1:] - log_c[:-1]
    else:
        price_chg_vpc = np.log(c / o)

    # True range relative to prev_c
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    rtr = tr / prev_c

    # CLV in [-1, 1]
    rng = h - l
    clv = np.zeros(n, dtype=float)
    valid_rng = rng > EPS
    clv[valid_rng] = np.clip((2.0 * c[valid_rng] - h[valid_rng] - l[valid_rng]) / rng[valid_rng], -1.0, 1.0)

    s_c = pd.Series(c)
    s_h = pd.Series(h)
    s_l = pd.Series(l)
    s_ret1 = pd.Series(ret1)
    s_log_v = pd.Series(log_v)
    s_rtr = pd.Series(rtr)

    # Trend / Vol controls (100% causal, simple returns)
    ma60 = s_c.rolling(60, min_periods=60).mean().to_numpy()
    r20 = (s_c / s_c.shift(20) - 1.0).to_numpy()
    r60 = (s_c / s_c.shift(60) - 1.0).to_numpy()
    sigma20 = s_ret1.rolling(20, min_periods=20).std(ddof=1).to_numpy()
    hi120 = s_h.rolling(120, min_periods=120).max().to_numpy()
    lo120 = s_l.rolling(120, min_periods=120).min().to_numpy()
    pos120 = np.where((hi120 - lo120) > EPS, (c - lo120) / (hi120 - lo120), np.nan)

    atr20_raw = pd.Series(tr).rolling(20, min_periods=20).mean().to_numpy()
    atr20_rel = s_rtr.rolling(20, min_periods=20).mean().to_numpy()
    atr80_rel = s_rtr.rolling(80, min_periods=80).mean().to_numpy()
    V20 = np.where(atr80_rel > EPS, atr20_rel / atr80_rel, np.nan)

    # 4 Frozen Volume-Price Physical Quantities (all causal up to day t)
    mean_lv20 = s_log_v.rolling(20, min_periods=20).mean().to_numpy()
    mean_lv80 = s_log_v.rolling(80, min_periods=80).mean().to_numpy()
    VR = mean_lv20 - mean_lv80
    VPC = rolling_spearman_1d(dlog_v, price_chg_vpc, window=20)
    VRD = -rolling_spearman_1d(dlog_v, rtr, window=20)

    sum_v_clv = pd.Series(v * clv).rolling(20, min_periods=20).sum().to_numpy()
    sum_v = pd.Series(v).rolling(20, min_periods=20).sum().to_numpy()
    mean_clv = pd.Series(clv).rolling(20, min_periods=20).mean().to_numpy()
    nCMF = -np.where(sum_v > EPS, sum_v_clv / sum_v - mean_clv, np.nan)

    # Pre-day (t-1) versions for Arena 2 (brk20)
    def shift1(arr: np.ndarray) -> np.ndarray:
        res = np.empty_like(arr, dtype=float)
        res[0] = np.nan
        res[1:] = arr[:-1]
        return res

    VR_pre = shift1(VR)
    VPC_pre = shift1(VPC)
    VRD_pre = shift1(VRD)
    nCMF_pre = shift1(nCMF)
    r20_pre = shift1(r20)
    sigma20_pre = shift1(sigma20)
    V_pre = shift1(V20)
    atr20_pre_raw = shift1(atr20_raw)

    # Breakout-day relative volume diagnostic RV_t = ln(V_t) - Mean_20(ln V)_{t-1}
    RV_t = log_v - shift1(mean_lv20)

    # Causal rolling 250d percentile ranks (min_periods=60 ramp-up) for practical bucketing
    VR_pct = causal_rolling_pct_rank(VR, window=250, min_periods=60)
    VPC_pct = causal_rolling_pct_rank(VPC, window=250, min_periods=60)
    VRD_pct = causal_rolling_pct_rank(VRD, window=250, min_periods=60)
    nCMF_pct = causal_rolling_pct_rank(nCMF, window=250, min_periods=60)

    VR_pre_pct = shift1(VR_pct)
    VPC_pre_pct = shift1(VPC_pct)
    VRD_pre_pct = shift1(VRD_pct)
    nCMF_pre_pct = shift1(nCMF_pct)

    # Arenas (strictly aligned with Study 02 & Study 03)
    up_std = (c > ma60) & (r20 > 0.0)
    up_high = up_std & (r60 > 0.0) & (pos120 >= 0.60)

    prev_hi_20 = s_c.rolling(20, min_periods=20).max().shift(1).to_numpy()
    is_nh_20 = c > prev_hi_20
    recent_20 = pd.Series(is_nh_20).shift(1).rolling(5, min_periods=5).sum().to_numpy() > 0
    brk20 = is_nh_20 & (~recent_20) & np.isfinite(prev_hi_20)
    M_brk20 = np.where(atr20_pre_raw > EPS, (c - prev_hi_20) / atr20_pre_raw, np.nan)

    prev_hi_60 = s_c.rolling(60, min_periods=60).max().shift(1).to_numpy()
    is_nh_60 = c > prev_hi_60
    recent_60 = pd.Series(is_nh_60).shift(1).rolling(10, min_periods=10).sum().to_numpy() > 0
    brk60 = is_nh_60 & (~recent_60) & np.isfinite(prev_hi_60)

    out = pd.DataFrame(
        {
            "asset": g["asset"].to_numpy(),
            "date": g["date"].to_numpy(),
            "r1": ret1,
            "r20": r20,
            "r60": r60,
            "sigma20": sigma20,
            "pos120": pos120,
            "V20": V20,
            "r20_pre": r20_pre,
            "sigma20_pre": sigma20_pre,
            "V_pre": V_pre,
            "M_brk20": M_brk20,
            "RV_t": RV_t,
            "VR": VR,
            "VPC": VPC,
            "VRD": VRD,
            "nCMF": nCMF,
            "VR_pre": VR_pre,
            "VPC_pre": VPC_pre,
            "VRD_pre": VRD_pre,
            "nCMF_pre": nCMF_pre,
            "VR_pct": VR_pct,
            "VPC_pct": VPC_pct,
            "VRD_pct": VRD_pct,
            "nCMF_pct": nCMF_pct,
            "VR_pre_pct": VR_pre_pct,
            "VPC_pre_pct": VPC_pre_pct,
            "VRD_pre_pct": VRD_pre_pct,
            "nCMF_pre_pct": nCMF_pre_pct,
            "up_std": up_std,
            "up_high": up_high,
            "brk20": brk20,
            "brk60": brk60,
        }
    )

    # Forward outcomes strictly inside truncated dev data
    thr = CRASH_THRESH[market_name]
    min_c_10 = s_c.iloc[::-1].rolling(10, min_periods=10).min().iloc[::-1].shift(-1).to_numpy()
    out["fail10_20"] = np.where(np.isfinite(min_c_10), (min_c_10 < prev_hi_20).astype(float), np.nan)
    out["fail10_60"] = np.where(np.isfinite(min_c_10), (min_c_10 < prev_hi_60).astype(float), np.nan)

    for H in [10, 20, 40]:
        fwd_ret = (s_c.shift(-H) / s_c - 1.0).to_numpy()
        min_l = s_l.iloc[::-1].rolling(H, min_periods=H).min().iloc[::-1].shift(-1).to_numpy()
        fwd_mdd = (min_l - c) / c
        crash = np.where(np.isfinite(fwd_mdd), (fwd_mdd <= thr[H]).astype(float), np.nan)
        out[f"ret_{H}"] = fwd_ret
        out[f"mdd_{H}"] = fwd_mdd
        out[f"crash_{H}"] = crash
        if H == 20:
            # Vol-normalized crash using causal sigma20 (for up_high) and sigma20_pre (for brk20)
            denom_t = sigma20 * math.sqrt(20.0)
            denom_pre = sigma20_pre * math.sqrt(20.0)
            mdd_norm_t = np.where(denom_t > EPS, fwd_mdd / denom_t, np.nan)
            mdd_norm_pre = np.where(denom_pre > EPS, fwd_mdd / denom_pre, np.nan)
            out["crash_norm_20_t"] = np.where(np.isfinite(mdd_norm_t), (mdd_norm_t <= -CRASH_NORM_K).astype(float), np.nan)
            out["crash_norm_20_pre"] = np.where(np.isfinite(mdd_norm_pre), (mdd_norm_pre <= -CRASH_NORM_K).astype(float), np.nan)
            # Future 20-day realized daily return volatility (t+1 .. t+20)
            fwd_vol = s_ret1.iloc[::-1].rolling(20, min_periods=20).std(ddof=1).iloc[::-1].shift(-1).to_numpy()
            out["fwd_vol_20"] = fwd_vol

    return out


def build_panel(
    raw: pd.DataFrame,
    eval_start: pd.Timestamp,
    market_name: str,
    use_close_to_close_vpc: bool = False,
    vol_col: str = "vol",
    exclude_quad_witch: bool = False,
) -> pd.DataFrame:
    parts = []
    for _, g in raw.groupby("asset", sort=True):
        ex = is_quad_witching_or_halfday_spy(g["date"]) if exclude_quad_witch else None
        parts.append(
            compute_asset_features_and_labels(
                g,
                market_name=market_name,
                use_close_to_close_vpc=use_close_to_close_vpc,
                vol_col=vol_col,
                exclude_mask=ex,
            )
        )
    df = pd.concat(parts, ignore_index=True)
    df = df[df["date"] >= eval_start].reset_index(drop=True)
    if market_name == "A股申万L1":
        for f in FEATURES + [f"{f}_pre" for f in FEATURES]:
            mkt = df.groupby("date")[f].transform("mean")
            df[f"{f}_mkt"] = mkt
            df[f"{f}_idio"] = df[f] - mkt
    return df


# ---------------------------------------------------------------------------
# 4. Regression & Diagnostic Engines
# ---------------------------------------------------------------------------
def build_design_matrix(
    sub: pd.DataFrame,
    feat_cols: list[str],
    cont_controls: list[str],
    vol_col: str,
    add_fe: bool,
    vol_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Build standardized design matrix: [1, cont_controls_z, vol_quintile_dummies(4), (industry FE), feat_cols_z].
    Returns (X, col_names, vol_edges_used).
    """
    n = len(sub)
    parts = [np.ones((n, 1), dtype=float)]
    names = ["const"]

    for col in cont_controls:
        arr = sub[col].to_numpy(dtype=float)
        sd = float(arr.std(ddof=0))
        arr_z = (arr - float(arr.mean())) / (sd if sd > EPS else 1.0)
        parts.append(arr_z[:, None])
        names.append(col)

    v = sub[vol_col].to_numpy(dtype=float)
    if vol_edges is None:
        vol_edges = np.quantile(v, [0.2, 0.4, 0.6, 0.8])
    vbins = np.digitize(v, vol_edges, right=False)  # 0..4
    for b in range(1, 5):
        dum = (vbins == b).astype(float)
        if dum.sum() > 0:
            parts.append(dum[:, None])
            names.append(f"{vol_col}_q{b+1}")

    if add_fe:
        assets = sorted(sub["asset"].unique())
        for a in assets[1:]:
            dum = (sub["asset"].to_numpy() == a).astype(float)
            if dum.sum() > 0:
                parts.append(dum[:, None])
                names.append(f"FE_{a}")

    for col in feat_cols:
        arr = sub[col].to_numpy(dtype=float)
        sd = float(arr.std(ddof=0))
        arr_z = (arr - float(arr.mean())) / (sd if sd > EPS else 1.0)
        parts.append(arr_z[:, None])
        names.append(col)

    return np.hstack(parts), names, vol_edges


def run_controlled_reg(
    df: pd.DataFrame,
    y_col: str,
    feat_cols: list[str],
    cont_controls: list[str],
    vol_col: str,
    all_dates: np.ndarray,
    add_fe: bool,
    scale: float = 100.0,
    lag: int = HAC_LAG,
) -> dict:
    need = ["date", "asset", y_col, vol_col] + cont_controls + feat_cols
    sub = df[need].dropna().copy()
    if len(sub) < 40:
        return {"n": len(sub), "betas": {f: np.nan for f in feat_cols}, "ses": {f: np.nan for f in feat_cols}, "ts": {f: np.nan for f in feat_cols}}
    X, _, _ = build_design_matrix(sub, feat_cols, cont_controls, vol_col, add_fe=add_fe)
    y = sub[y_col].to_numpy(dtype=float) * scale
    res = event_ols_hac(y, X, sub["date"].to_numpy(), all_dates, lag=lag)
    k_f = len(feat_cols)
    betas = {f: float(res["beta"][-k_f + i]) for i, f in enumerate(feat_cols)}
    ses = {f: float(res["se"][-k_f + i]) for i, f in enumerate(feat_cols)}
    ts = {f: float(res["t"][-k_f + i]) for i, f in enumerate(feat_cols)}
    return {"n": int(res["n"]), "betas": betas, "ses": ses, "ts": ts}


def compute_vif_table(sub: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    mat = sub[cols].dropna().to_numpy(dtype=float)
    sd = mat.std(axis=0, ddof=0)
    Z = (mat - mat.mean(axis=0)) / np.where(sd > EPS, sd, 1.0)
    vifs: dict[str, float] = {}
    for i, c in enumerate(cols):
        y = Z[:, i]
        others = np.delete(Z, i, axis=1)
        X = np.hstack([np.ones((len(Z), 1)), others])
        b = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        r2 = 1.0 - float(np.sum((y - X @ b) ** 2)) / max(float(np.sum((y - y.mean()) ** 2)), EPS)
        vifs[c] = 1.0 / max(1.0 - r2, 1e-6)
    return vifs


def estimate_pre_data_mde(
    arena_df: pd.DataFrame,
    y_col: str,
    feat_col: str,
    cont_controls: list[str],
    vol_col: str,
    all_dates: np.ndarray,
    add_fe: bool,
    scale: float = 100.0,
) -> float:
    """
    Pre-data MDE (80% power, two-sided alpha=0.05 => 2.8 * SE) via circular date-shift
    of feature X against L1-residualized Y. Preserves exact autocorrelation, cross-sectional
    correlation, and calendar clustering of X without ever pairing true (X_t, Y_t).
    """
    need = ["date", "asset", y_col, vol_col, feat_col] + cont_controls
    sub = arena_df[need].dropna().copy()
    if len(sub) < 60:
        return np.nan

    # Residualize Y on L1 controls only (never uses X)
    X_l1, _, _ = build_design_matrix(sub, [], cont_controls, vol_col, add_fe=add_fe)
    y = sub[y_col].to_numpy(dtype=float) * scale
    b_l1 = np.linalg.pinv(X_l1.T @ X_l1) @ (X_l1.T @ y)
    e_y = y - X_l1 @ b_l1

    # Also residualize X on L1 controls to get realistic VIF shrinkage
    x = sub[feat_col].to_numpy(dtype=float)
    x = (x - x.mean()) / max(float(x.std(ddof=0)), EPS)
    b_x = np.linalg.pinv(X_l1.T @ X_l1) @ (X_l1.T @ x)
    e_x = x - X_l1 @ b_x
    denom_x = float(np.dot(e_x, e_x))
    if denom_x <= EPS:
        return np.nan

    # Circularly shift calendar-aggregated dates by 1/3 and 2/3 of trading calendar
    dates = sub["date"].to_numpy()
    idx = np.searchsorted(all_trading_dates := all_dates, dates)
    T = len(all_trading_dates)
    ses = []
    for shift in [T // 3, (2 * T) // 3, T // 2]:
        # Shift e_y by mapping each date index -> (idx + shift) % T on a per-day mean/score basis
        # Even cleaner: shift the unique calendar date order of e_y blocks
        # Wait: exact date-preserving shift on e_x's calendar alignment:
        # Pair e_x at calendar position idx with e_y at a far-away position by rolling sorted unique dates
        uniq_d = np.sort(sub["date"].unique())
        u_pos = np.searchsorted(uniq_d, dates)
        # Re-index e_y by shifting uniq_d blocks by sh_u
        sh_u = max(60, int(len(uniq_d) * (shift / T)))
        # Build date-keyed lookup of e_x * e_y using asset-agnostic date-block shift:
        # For each row i at date-rank r, pair its e_x with e_y from a row at date-rank (r + sh_u) % U
        U = len(uniq_d)
        # Group e_y by date-rank -> take matching index within date (random deterministic mod)
        order_orig = np.argsort(u_pos, kind="mergesort")
        e_y_sorted = e_y[order_orig]
        # Roll e_y_sorted by roughly len * (shift/T) rows to break (X_t, Y_t) alignment completely
        roll_n = int(len(sub) * (shift / T))
        e_y_sh = np.empty_like(e_y)
        e_y_sh[order_orig] = np.roll(e_y_sorted, roll_n)
        # Compute exact calendar-aligned DK variance of score_i = e_x_i * e_y_sh_i / denom_x
        scores = e_x * e_y_sh
        h = np.zeros(T, dtype=float)
        np.add.at(h, idx, scores)
        omega = float(np.dot(h, h))
        K = min(HAC_LAG, T - 1)
        for ell in range(1, K + 1):
            w = 1.0 - ell / (K + 1.0)
            omega += 2.0 * w * float(np.dot(h[ell:], h[:-ell]))
        se = math.sqrt(max(omega, 0.0)) / denom_x
        ses.append(se)
    return 2.8 * float(np.median(ses))


def vol_stratified_causal_spread(
    arena_df: pd.DataFrame,
    pct_col: str,
    vol_col: str,
    y_col: str = "crash_20",
) -> dict[str, float]:
    """
    Practical magnitude gate (H5 + gatekeeper fix):
    Using strictly causal 250d rolling percentile `pct_col`, split into Low (<1/3), Mid, High (>=2/3).
    Within each tercile of `vol_col`, compute mean(y|High) - mean(y|Low), then average across the 3 vol terciles.
    Also reports raw causal High/Mid/Low means for readability.
    """
    sub = arena_df[[pct_col, vol_col, y_col, "ret_20"]].dropna().copy()
    if len(sub) < 60:
        return {"spread_pp": np.nan, "raw_lo": np.nan, "raw_hi": np.nan, "ret_lo": np.nan, "ret_hi": np.nan, "n": float(len(sub))}

    sub["fbin"] = np.where(sub[pct_col] < 1.0 / 3.0, 1, np.where(sub[pct_col] >= 2.0 / 3.0, 3, 2))
    sub["vbin"] = pd.qcut(sub[vol_col].rank(method="first"), 3, labels=[1, 2, 3]).astype(int)

    diffs = []
    for vb in [1, 2, 3]:
        g = sub[sub["vbin"] == vb]
        lo = g[g["fbin"] == 1][y_col]
        hi = g[g["fbin"] == 3][y_col]
        if len(lo) >= 5 and len(hi) >= 5:
            diffs.append(float(hi.mean() - lo.mean()) * 100.0)

    lo_all = sub[sub["fbin"] == 1]
    hi_all = sub[sub["fbin"] == 3]
    return {
        "spread_pp": float(np.mean(diffs)) if len(diffs) == 3 else np.nan,
        "raw_lo": float(lo_all[y_col].mean() * 100.0) if len(lo_all) else np.nan,
        "raw_hi": float(hi_all[y_col].mean() * 100.0) if len(hi_all) else np.nan,
        "ret_lo": float(lo_all["ret_20"].mean() * 100.0) if len(lo_all) else np.nan,
        "ret_hi": float(hi_all["ret_20"].mean() * 100.0) if len(hi_all) else np.nan,
        "n": float(len(sub)),
    }


# ---------------------------------------------------------------------------
# 5. Step 0 Unit Tests & Causality Probes (Zero Look at Real Feature-Y)
# ---------------------------------------------------------------------------
def run_unit_and_causality_tests(spy_raw: pd.DataFrame, sw_raw: pd.DataFrame, log) -> None:
    log("\n" + "=" * 100)
    log("【第 0 步 · 单元测试与严格因果 / 截断探针】")
    log("=" * 100)

    # 1. Tie-aware rank & rolling Spearman exactness vs pandas
    rng = np.random.default_rng(42)
    x_syn = np.round(rng.normal(size=50), 1)  # deliberate ties
    y_syn = np.round(0.6 * x_syn + rng.normal(scale=0.5, size=50), 1)
    rs_fast = rolling_spearman_1d(x_syn, y_syn, window=20)
    for t in [19, 25, 35, 49]:
        s1 = pd.Series(x_syn[t - 19 : t + 1])
        s2 = pd.Series(y_syn[t - 19 : t + 1])
        ref = float(s1.rank(method="average").corr(s2.rank(method="average")))
        assert abs(rs_fast[t] - ref) < 1e-12, f"Spearman mismatch at {t}: {rs_fast[t]} vs {ref}"
    log("  [PASS] Tie-aware rolling Spearman 与 pandas.rank(average).corr 逐点一致 (err < 1e-12)")

    # 2. Causal total-return multiplier unit test
    syn_div = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="B"),
            "open": [100.0, 99.0, 99.0, 100.0],
            "high": [101.0, 99.5, 100.0, 101.0],
            "low": [99.0, 98.5, 98.0, 99.0],
            "close": [100.0, 99.0, 99.0, 100.0],
            "dividends": [0.0, 1.0, 0.0, 0.0],
            "stock_splits": [0.0, 0.0, 0.0, 0.0],
        }
    )
    tr_df = build_causal_total_return_ohlc(syn_div)
    assert abs(tr_df["close"].iloc[1] - 100.0) < 1e-12, "Ex-div flat day TR close should be 100"
    assert abs(tr_df["open"].iloc[1] - 100.0) < 1e-12, "Ex-div flat day TR open should be 100"
    # Adding a huge future dividend on day 3 must NOT change day 0..2
    syn_div2 = syn_div.copy()
    syn_div2.loc[3, "dividends"] = 10.0
    tr_df2 = build_causal_total_return_ohlc(syn_div2)
    assert np.max(np.abs(tr_df["close"].iloc[:3] - tr_df2["close"].iloc[:3])) < 1e-12
    log("  [PASS] SPY 因果前向总回报乘数单测通过（除息零缺口 + 未来分红不重绘历史）")

    # 3. Future-function probe (perturbing rows > t0 must produce 0.0 diff at t0)
    for mname, raw_df in [("SPY", spy_raw), ("A股申万L1(单行业)", sw_raw[sw_raw["asset"] == sw_raw["asset"].iloc[0]])]:
        g_orig = raw_df.sort_values("date").reset_index(drop=True).copy()
        t0 = len(g_orig) // 2
        g_pert = g_orig.copy()
        g_pert.loc[t0 + 1 :, ["open", "high", "low", "close"]] *= 1.5
        g_pert.loc[t0 + 1 :, "vol"] *= 3.0
        f_orig = compute_asset_features_and_labels(g_orig, "SPY" if mname == "SPY" else "A股申万L1")
        f_pert = compute_asset_features_and_labels(g_pert, "SPY" if mname == "SPY" else "A股申万L1")
        check_cols = FEATURES + [f"{f}_pre" for f in FEATURES] + [f"{f}_pct" for f in FEATURES] + ["r20", "r60", "sigma20", "V20", "up_high", "brk20"]
        max_diff = 0.0
        for col in check_cols:
            a = f_orig[col].iloc[: t0 + 1].to_numpy(dtype=float)
            b = f_pert[col].iloc[: t0 + 1].to_numpy(dtype=float)
            mask = np.isfinite(a) | np.isfinite(b)
            if mask.any():
                max_diff = max(max_diff, float(np.max(np.abs(a[mask] - b[mask]))))
        assert max_diff < 1e-12, f"Causality violation in {mname}: max_diff={max_diff}"
        log(f"  [PASS] 未来函数探针 [{mname}]: 篡改 t>{t0} 全部行情与量能后，<=t0 所有特征/考场标签最大差 = {max_diff:.1e}")


def run_step0_data_audit(spy_raw: pd.DataFrame, sw_raw: pd.DataFrame, log) -> bool:
    """Returns use_close_to_close_vpc (True only if SPY open==prev_close > 5%)."""
    log("\n" + "=" * 100)
    log("【第 0 步 · 数据体检与 SPY vs 本地 SPX 对账（开发段，不看任何特征-Y关联）】")
    log("=" * 100)

    # 1. SPY vs Yahoo adj_close & local SPX
    spx = pd.read_parquet(SPX_PATH)
    spx["date"] = pd.to_datetime(spx["date"])
    spx = spx[(spx["date"] >= "1993-01-01") & (spx["date"] <= SPY_DEV_END)].copy()
    spx["spx_ret"] = spx["close"].pct_change()

    spy = spy_raw.copy()
    spy["spy_tr_ret"] = spy["close"].pct_change()
    spy["spy_adj_ret"] = spy["adj_close"].pct_change()
    spy["open_eq_pre"] = (spy["open_raw"] - spy["close_raw"].shift(1)).abs() <= 1e-6

    m = spy[spy["date"] >= SPY_DEV_START].merge(spx[["date", "spx_ret"]], on="date", how="inner").dropna(subset=["spy_tr_ret", "spx_ret", "spy_adj_ret"])
    corr_adj = float(m["spy_tr_ret"].corr(m["spy_adj_ret"]))
    max_adj_diff = float((m["spy_tr_ret"] - m["spy_adj_ret"]).abs().max())
    corr_spx = float(m["spy_tr_ret"].corr(m["spx_ret"]))
    ann_diff_pct = float((m["spy_tr_ret"].mean() - m["spx_ret"].mean()) * 252 * 100.0)
    open_eq_pct = float(spy[spy["date"] >= SPY_DEV_START]["open_eq_pre"].mean() * 100.0)
    open_eq_94_00 = float(spy[(spy["date"] >= "1994-01-01") & (spy["date"] <= "2000-12-31")]["open_eq_pre"].mean() * 100.0)

    log(f"  SPY 开发段硬截断: {spy['date'].min().date()} -> {spy['date'].max().date()} (总行={len(spy)}, 评估行(94+)={len(spy[spy['date']>=SPY_DEV_START])})")
    log(f"  SPY 因果总回报 vs Yahoo adj_close 日收益: 相关 = {corr_adj:.6f}, 最大单日差 = {max_adj_diff:.5f}")
    log(f"  SPY 因果总回报 vs 本地 SPX 快照日收益 (1994-2019, n={len(m)}): 相关 = {corr_spx:.6f}, 年化差(≈股息率) = {ann_diff_pct:+.2f}%")
    log(f"  SPY open == 前收占比: 1994-2019 = {open_eq_pct:.2f}% (1994-2000 = {open_eq_94_00:.2f}%)")

    use_c2c = open_eq_pct > 5.0
    if use_c2c:
        log("  [触发事前备选] SPY open==前收 > 5%，VPC_20 自动退回日收益 ln(C_i/C_{i-1})！")
    else:
        log("  [锁定主规格] SPY 与 A 股开盘价均干净，VPC_20 锁定忠实华西原式 ln(C_i/O_i)。")

    # 2. SW L1 summary
    sw_open_eq = float(
        sw_raw.groupby("asset")["close"].shift(1).sub(sw_raw["open"]).abs().le(1e-6).mean() * 100.0
    )
    log(f"  A 股申万 L1 硬截断: {sw_raw['date'].min().date()} -> {sw_raw['date'].max().date()} (行业数={sw_raw['asset'].nunique()}, 总行={len(sw_raw)}, open==前收={sw_open_eq:.2f}%)")
    return use_c2c


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 100)
    log("Market Cycle V2 — Study 13: 时间序列连续量价物理量预警趋势衰竭与突破失败")
    log("=" * 100)

    spy_raw = load_spy_dev()
    sw_raw, sw_sid = load_sw_l1_dev()
    log(f"封存边界硬截断确认: SPY <= {spy_raw['date'].max().date()} (<=2019-12-31); A股 SW L1 <= {sw_raw['date'].max().date()} (<=2024-10-31, 快照={sw_sid})")

    # --- STEP 0: Tests, Audit, Correlations, VIF, Pre-Data MDE ---
    use_c2c = run_step0_data_audit(spy_raw, sw_raw, log)
    run_unit_and_causality_tests(spy_raw, sw_raw, log)

    spy_df = build_panel(spy_raw, SPY_DEV_START, "SPY", use_close_to_close_vpc=use_c2c)
    sw_df = build_panel(sw_raw, SW_DEV_START, "A股申万L1", use_close_to_close_vpc=use_c2c)

    # Diagnostic alternative panels (SPY quad-witching cleaned; SW amount-based)
    spy_df_noqw = build_panel(spy_raw, SPY_DEV_START, "SPY", use_close_to_close_vpc=use_c2c, exclude_quad_witch=True)
    sw_df_amt = build_panel(sw_raw, SW_DEV_START, "A股申万L1", use_close_to_close_vpc=use_c2c, vol_col="amount")

    markets = [
        {
            "name": "A股申万L1",
            "df": sw_df,
            "all_dates": np.sort(sw_df["date"].unique()),
            "add_fe": True,
            "splits": [
                ("前半 2016-2020", pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31")),
                ("后半 2021-2024.10", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-10-31")),
            ],
            "ex24": ("剔24.09-10", pd.Timestamp("2016-01-01"), pd.Timestamp("2024-08-31")),
        },
        {
            "name": "SPY",
            "df": spy_df,
            "all_dates": np.sort(spy_df["date"].unique()),
            "add_fe": False,
            "splits": [
                ("前半 1994-2006", pd.Timestamp("1994-01-01"), pd.Timestamp("2006-12-31")),
                ("后半 2007-2019", pd.Timestamp("2007-01-01"), pd.Timestamp("2019-12-31")),
            ],
            "ex24": None,
        },
    ]

    arenas = [
        {
            "key": "up_high",
            "label": "考场一：高位趋势衰竭 up_high",
            "feat_suffix": "",
            "cont_l1": ["r20", "r60"],
            "vol_col": "sigma20",
            "v_col": "V20",
            "crash_norm_col": "crash_norm_20_t",
        },
        {
            "key": "brk20",
            "label": "考场二：20日首次新高 brk20（事前 t-1）",
            "feat_suffix": "_pre",
            "cont_l1": ["r20_pre", "r1"],
            "vol_col": "sigma20_pre",
            "v_col": "V_pre",
            "crash_norm_col": "crash_norm_20_pre",
        },
    ]

    log("\n" + "=" * 100)
    log("【第 0 步 · 考场底数、相关矩阵、VIF 与事前 MDE（不看任何真实特征-Y 回归）】")
    log("=" * 100)
    for ar in arenas:
        log(f"\n>>> {ar['label']}")
        fcols = [f"{f}{ar['feat_suffix']}" for f in FEATURES]
        ctrl_cols = ar["cont_l1"] + [ar["vol_col"], ar["v_col"]]
        for m in markets:
            sub = m["df"][m["df"][ar["key"]]].dropna(subset=["crash_20", "ret_20", ar["crash_norm_col"]] + fcols + ctrl_cols)
            uniq_days = sub["date"].nunique()
            log(
                f"  [{m['name']}] 有效样本={len(sub)}, 覆盖交易日={uniq_days} "
                f"({sub['date'].min().date()} ~ {sub['date'].max().date()}) | "
                f"底数: 20日均收益={sub['ret_20'].mean()*100:+.2f}%, "
                f"固定踩雷={sub['crash_20'].mean()*100:.1f}%, 波动归一踩雷={sub[ar['crash_norm_col']].mean()*100:.1f}%"
            )
            corr = sub[fcols + ctrl_cols].corr()
            log("  相关矩阵 (4 特征 vs 自身与控制变量):")
            hdr = "    " + f"{'':<10s}" + "".join(f"{c:>10s}" for c in fcols + ctrl_cols)
            log(hdr)
            for fc in fcols:
                row_s = "    " + f"{fc:<10s}" + "".join(f"{corr.loc[fc, c]:>+10.2f}" for c in fcols + ctrl_cols)
                log(row_s)
            vifs = compute_vif_table(sub, fcols + ctrl_cols)
            log("  VIF: " + ", ".join(f"{k}={v:.2f}" for k, v in vifs.items()))

            # Pre-data MDE via circular shift
            mdes = []
            for fc in fcols:
                mde = estimate_pre_data_mde(
                    sub, "crash_20", fc, ar["cont_l1"], ar["vol_col"], m["all_dates"], add_fe=m["add_fe"]
                )
                mdes.append(f"{fc}={mde:.2f}pp")
            log("  事前 MDE (80%功效, α=0.05, 每+1SD 踩雷率变化): " + ", ".join(mdes))
            if m["name"] == "A股申万L1":
                mde_mkt = estimate_pre_data_mde(
                    sub, "crash_20", fcols[0] + "_mkt", ar["cont_l1"], ar["vol_col"], m["all_dates"], add_fe=True
                )
                mde_idio = estimate_pre_data_mde(
                    sub, "crash_20", fcols[0] + "_idio", ar["cont_l1"], ar["vol_col"], m["all_dates"], add_fe=True
                )
                log(f"  A股双层检验力参照 ({fcols[0]}): 大盘层 MDE={mde_mkt:.2f}pp vs 行业独立层 MDE={mde_idio:.2f}pp")

    # --- STEP 1: 16 Primary Tests + All 4 Gates ---
    log("\n" + "=" * 100)
    log("【主检验总表】16 个预注册主检验（主裁判 = 20 日固定踩雷率 crash_20，单位：每+1SD 的百分点变化 pp）")
    log("  统一方向假说：高值 = 危险 => 预期 γ_crash > 0（t >= 2.0 且 16 检验 BH-FDR q <= 0.05）")
    log("=" * 100)

    primary_rows: list[dict] = []
    for ar in arenas:
        for m in markets:
            sub_arena = m["df"][m["df"][ar["key"]]].copy()
            all_dates = m["all_dates"]
            add_fe = m["add_fe"]
            for base_f in FEATURES:
                fc = f"{base_f}{ar['feat_suffix']}"
                pct_col = f"{fc}_pct"
                # L1 primary crash_20
                r_l1 = run_controlled_reg(sub_arena, "crash_20", [fc], ar["cont_l1"], ar["vol_col"], all_dates, add_fe=add_fe)
                # L2 (+V20)
                r_l2 = run_controlled_reg(sub_arena, "crash_20", [fc], ar["cont_l1"] + [ar["v_col"]], ar["vol_col"], all_dates, add_fe=add_fe)
                # Guardrail 1: vol-normalized crash
                r_cn = run_controlled_reg(sub_arena, ar["crash_norm_col"], [fc], ar["cont_l1"], ar["vol_col"], all_dates, add_fe=add_fe)
                # Guardrail 2: 20d simple return
                r_ret = run_controlled_reg(sub_arena, "ret_20", [fc], ar["cont_l1"], ar["vol_col"], all_dates, add_fe=add_fe)
                # Diagnostic: future 20d realized volatility (annualized % or daily bps -> use daily % *100)
                r_fvol = run_controlled_reg(sub_arena, "fwd_vol_20", [fc], ar["cont_l1"], ar["vol_col"], all_dates, add_fe=add_fe, scale=100.0)
                # Diagnostic: mdd_20 & fail10_20
                r_mdd = run_controlled_reg(sub_arena, "mdd_20", [fc], ar["cont_l1"], ar["vol_col"], all_dates, add_fe=add_fe)
                r_fail = (
                    run_controlled_reg(sub_arena, "fail10_20", [fc], ar["cont_l1"], ar["vol_col"], all_dates, add_fe=add_fe, lag=15)
                    if ar["key"] == "brk20"
                    else {"betas": {fc: np.nan}, "ts": {fc: np.nan}}
                )

                # Half-sample splits
                sp_g, sp_t = [], []
                for _, d0, d1 in m["splits"]:
                    s_df = sub_arena[(sub_arena["date"] >= d0) & (sub_arena["date"] <= d1)]
                    rs = run_controlled_reg(s_df, "crash_20", [fc], ar["cont_l1"], ar["vol_col"], all_dates, add_fe=add_fe)
                    sp_g.append(rs["betas"][fc])
                    sp_t.append(rs["ts"][fc])

                # A-share ex 2024.09-10 and year-by-year consistency
                ex24_g = np.nan
                yr_pos, yr_valid = 0, 0
                if m["ex24"] is not None:
                    _, d0, d1 = m["ex24"]
                    r_ex = run_controlled_reg(
                        sub_arena[(sub_arena["date"] >= d0) & (sub_arena["date"] <= d1)],
                        "crash_20", [fc], ar["cont_l1"], ar["vol_col"], all_dates, add_fe=add_fe
                    )
                    ex24_g = r_ex["betas"][fc]
                    # Year-by-year sign check (valid years n >= 30)
                    tmp = sub_arena[["date", "asset", "crash_20", fc, ar["vol_col"]] + ar["cont_l1"]].dropna().copy()
                    tmp["yr"] = tmp["date"].dt.year
                    for _, yg in tmp.groupby("yr"):
                        if len(yg) < 30:
                            continue
                        yr_valid += 1
                        # Within-year L1 regression (drop FE if too sparse, keep if >=60; use OLS sign)
                        ry = run_controlled_reg(yg, "crash_20", [fc], ar["cont_l1"], ar["vol_col"], all_dates, add_fe=(len(yg) >= 120))
                        if np.isfinite(ry["betas"][fc]) and ry["betas"][fc] > 0.0:
                            yr_pos += 1

                # Practical volatility-controlled causal magnitude gate
                mag = vol_stratified_causal_spread(sub_arena, pct_col, ar["vol_col"], y_col="crash_20")

                g1 = r_l1["betas"][fc]
                t1 = r_l1["ts"][fc]
                p1 = two_sided_p(t1)
                g2 = r_l2["betas"][fc]
                t2 = r_l2["ts"][fc]
                g_cn = r_cn["betas"][fc]
                t_cn = r_cn["ts"][fc]
                g_ret = r_ret["betas"][fc]
                t_ret = r_ret["ts"][fc]

                primary_rows.append(
                    {
                        "arena": ar["key"],
                        "market": m["name"],
                        "feature": base_f,
                        "n": r_l1["n"],
                        "L1_g": g1,
                        "L1_se": r_l1["ses"][fc],
                        "L1_t": t1,
                        "L1_p": p1,
                        "L2_g": g2,
                        "L2_t": t2,
                        "cn_g": g_cn,
                        "cn_t": t_cn,
                        "ret_g": g_ret,
                        "ret_t": t_ret,
                        "mdd_g": r_mdd["betas"][fc],
                        "mdd_t": r_mdd["ts"][fc],
                        "fvol_g": r_fvol["betas"][fc],
                        "fvol_t": r_fvol["ts"][fc],
                        "fail10_g": r_fail["betas"][fc],
                        "fail10_t": r_fail["ts"][fc],
                        "sp1_g": sp_g[0],
                        "sp1_t": sp_t[0],
                        "sp2_g": sp_g[1],
                        "sp2_t": sp_t[1],
                        "ex24_g": ex24_g,
                        "yr_pos": yr_pos,
                        "yr_valid": yr_valid,
                        "vol_strat_pp": mag["spread_pp"],
                        "raw_lo_cr": mag["raw_lo"],
                        "raw_hi_cr": mag["raw_hi"],
                        "raw_lo_ret": mag["ret_lo"],
                        "raw_hi_ret": mag["ret_hi"],
                    }
                )

    res_df = pd.DataFrame(primary_rows)
    res_df["L1_q"] = bh_fdr(res_df["L1_p"].tolist())

    # Evaluate all 4 frozen gates
    verdicts = []
    for _, r in res_df.iterrows():
        gate1 = (r["L1_g"] > 0.0) and (r["L1_t"] >= 2.0) and (r["L1_q"] <= 0.05)
        gate2 = (r["L2_g"] > 0.0) and (r["L1_g"] > 0.0) and (r["L2_g"] >= 0.5 * r["L1_g"])
        gate3 = (r["cn_g"] > 0.0) and (r["ret_g"] <= 0.0)
        halves_ok = (r["sp1_g"] > 0.0) and (r["sp2_g"] > 0.0)
        if r["market"] == "A股申万L1":
            yr_ok = (r["yr_valid"] > 0) and (r["yr_pos"] / r["yr_valid"] >= 2.0 / 3.0 - 1e-9)
            ex_ok = np.isfinite(r["ex24_g"]) and (r["ex24_g"] > 0.0)
            stab_ok = halves_ok and yr_ok and ex_ok
        else:
            stab_ok = halves_ok
        mag_ok = np.isfinite(r["vol_strat_pp"]) and (r["vol_strat_pp"] >= MAG_GATE_PP[r["market"]])
        gate4 = stab_ok and mag_ok

        if gate1 and gate2 and gate3 and gate4:
            v = "过关"
        elif (r["L1_g"] < 0.0) and (r["L1_t"] <= -2.0) and (r["L1_q"] <= 0.05):
            v = "反向显著"
        elif (r["L1_g"] < 0.0) and (r["L1_t"] <= -2.0):
            v = "反向(|t|>=2)"
        else:
            v = "淘汰"
        verdicts.append((gate1, gate2, gate3, gate4, v))

    res_df["G1_L1FDR"] = [x[0] for x in verdicts]
    res_df["G2_L2"] = [x[1] for x in verdicts]
    res_df["G3_Guard"] = [x[2] for x in verdicts]
    res_df["G4_StabMag"] = [x[3] for x in verdicts]
    res_df["verdict"] = [x[4] for x in verdicts]

    # Print formatted primary table
    for ar in arenas:
        log(f"\n>>> {ar['label']}")
        log(
            f"  {'市场':<8s} {'特征':<5s} {'n':>5s} | {'L1踩雷pp':>8s} {'t_L1':>6s} {'q_FDR':>6s} | "
            f"{'L2踩雷pp':>8s} {'t_L2':>6s} | {'归一雷t':>7s} {'收益γ%':>7s} {'t_收':>6s} | "
            f"{'控波差pp':>8s} {'前后半γ':>13s} | {'判定'}"
        )
        log("  " + "-" * 116)
        sub_r = res_df[res_df["arena"] == ar["key"]]
        for _, r in sub_r.iterrows():
            hlf = f"{r['sp1_g']:+.1f}/{r['sp2_g']:+.1f}"
            log(
                f"  {r['market']:<8s} {r['feature']:<5s} {int(r['n']):>5d} | "
                f"{r['L1_g']:>+8.2f} {r['L1_t']:>+6.2f} {r['L1_q']:>6.3f} | "
                f"{r['L2_g']:>+8.2f} {r['L2_t']:>+6.2f} | "
                f"{r['cn_t']:>+7.2f} {r['ret_g']:>+7.2f} {r['ret_t']:>+6.2f} | "
                f"{r['vol_strat_pp']:>+8.2f} {hlf:>13s} | {r['verdict']}"
            )

    n_pass = int((res_df["verdict"] == "过关").sum())
    n_rev = int(res_df["verdict"].str.startswith("反向").sum())
    log(f"\n>>> 【16 个主检验总计】：过关 = {n_pass}/16，反向显著/反向(|t|>=2) = {n_rev}/16")

    # --- STEP 2: Key Diagnostics ---
    log("\n" + "=" * 100)
    log("【诊断 1】未来波动预报检验 + 实用因果三档落袋（区分「真衰竭」vs「纯波动预报」）")
    log("=" * 100)
    log(
        f"  {'考场':<7s} {'市场':<8s} {'特征':<5s} | {'未来日波γ%':>10s} {'t_波':>6s} | "
        f"{'MDDγ%':>7s} {'t_MDD':>6s} | {'打脸γpp':>8s} {'t_脸':>6s} | "
        f"{'因果低->高踩雷%':>16s} {'低->高收益%':>14s} | {'A股有效年/剔924'}"
    )
    log("  " + "-" * 118)
    for _, r in res_df.iterrows():
        f10 = f"{r['fail10_g']:+8.2f}" if np.isfinite(r["fail10_g"]) else "     N/A"
        tf10 = f"{r['fail10_t']:+6.2f}" if np.isfinite(r["fail10_t"]) else "   N/A"
        yr_str = f"{int(r['yr_pos'])}/{int(r['yr_valid'])} ({r['ex24_g']:+.2f}pp)" if r["market"] == "A股申万L1" else "—"
        log(
            f"  {r['arena']:<7s} {r['market']:<8s} {r['feature']:<5s} | "
            f"{r['fvol_g']:>+10.3f} {r['fvol_t']:>+6.2f} | "
            f"{r['mdd_g']:>+7.2f} {r['mdd_t']:>+6.2f} | {f10} {tf10} | "
            f"{r['raw_lo_cr']:>6.1f}% -> {r['raw_hi_cr']:>5.1f}% "
            f"{r['raw_lo_ret']:>+5.2f}% -> {r['raw_hi_ret']:>+5.2f}% | {yr_str}"
        )

    # Diagnostic 2: A-share X_mkt vs X_idio decomposition (H7)
    log("\n" + "=" * 100)
    log("【诊断 2】A 股大盘层 (X_mkt) vs 行业独立层 (X_idio) 双层拆解（L1 + 行业 FE，同台回归）")
    log("  事前约定读法：仅 X_idio 有效 = 复现华西横截面排雷；X_mkt 有效 = 大盘择时信息（注意独立窗口少）")
    log("=" * 100)
    sw_all_dates = markets[0]["all_dates"]
    log(
        f"  {'考场':<7s} {'特征':<5s} | {'大盘层踩雷pp':>12s} {'t_mkt':>6s} {'大盘收益t':>9s} | "
        f"{'行业独立踩雷pp':>14s} {'t_idio':>7s} {'独立收益t':>9s}"
    )
    log("  " + "-" * 86)
    for ar in arenas:
        sub_sw = sw_df[sw_df[ar["key"]]].copy()
        for base_f in FEATURES:
            fc = f"{base_f}{ar['feat_suffix']}"
            fm, fi = f"{fc}_mkt", f"{fc}_idio"
            rc = run_controlled_reg(sub_sw, "crash_20", [fm, fi], ar["cont_l1"], ar["vol_col"], sw_all_dates, add_fe=True)
            rr = run_controlled_reg(sub_sw, "ret_20", [fm, fi], ar["cont_l1"], ar["vol_col"], sw_all_dates, add_fe=True)
            log(
                f"  {ar['key']:<7s} {base_f:<5s} | "
                f"{rc['betas'][fm]:>+12.2f} {rc['ts'][fm]:>+6.2f} {rr['ts'][fm]:>+9.2f} | "
                f"{rc['betas'][fi]:>+14.2f} {rc['ts'][fi]:>+7.2f} {rr['ts'][fi]:>+9.2f}"
            )

    # Diagnostic 3: Breakout-day relative volume RV_t (H4)
    log("\n" + "=" * 100)
    log("【诊断 3】考场二突破当天相对量 RV_t = ln(V_t) - Mean_20(ln V)_{t-1}（控制 L1 + 当日突破幅度 M_brk20）")
    log("=" * 100)
    for m in markets:
        ev = m["df"][m["df"]["brk20"]].copy()
        ctrls = ["r20_pre", "r1", "M_brk20"]
        rc = run_controlled_reg(ev, "crash_20", ["RV_t"], ctrls, "sigma20_pre", m["all_dates"], add_fe=m["add_fe"])
        rcn = run_controlled_reg(ev, "crash_norm_20_pre", ["RV_t"], ctrls, "sigma20_pre", m["all_dates"], add_fe=m["add_fe"])
        rr = run_controlled_reg(ev, "ret_20", ["RV_t"], ctrls, "sigma20_pre", m["all_dates"], add_fe=m["add_fe"])
        rf = run_controlled_reg(ev, "fail10_20", ["RV_t"], ctrls, "sigma20_pre", m["all_dates"], add_fe=m["add_fe"], lag=15)
        log(
            f"  {m['name']:<8s} (n={rc['n']:>4d}) | 踩雷γ={rc['betas']['RV_t']:+.2f}pp (t={rc['ts']['RV_t']:+.2f}) | "
            f"归一雷t={rcn['ts']['RV_t']:+.2f} | 20日收益γ={rr['betas']['RV_t']:+.2f}% (t={rr['ts']['RV_t']:+.2f}) | "
            f"10日打脸γ={rf['betas']['RV_t']:+.2f}pp (t={rf['ts']['RV_t']:+.2f})"
        )

    # Diagnostic 4: Sensitivities (SPY quad-witching cleaned & SW amount replacement)
    log("\n" + "=" * 100)
    log("【诊断 4】敏感性核查：SPY 剔除四巫日量能尖峰 & A 股成交额 (amount) 替换成交量 (vol)")
    log("=" * 100)
    for ar in arenas:
        log(f"  -- {ar['label']} --")
        # SPY no quad-witching
        sub_spy_nq = spy_df_noqw[spy_df_noqw[ar["key"]]].copy()
        spy_dates = markets[1]["all_dates"]
        parts_s = []
        for base_f in FEATURES:
            fc = f"{base_f}{ar['feat_suffix']}"
            r = run_controlled_reg(sub_spy_nq, "crash_20", [fc], ar["cont_l1"], ar["vol_col"], spy_dates, add_fe=False)
            parts_s.append(f"{base_f}:{r['betas'][fc]:+.2f}pp(t={r['ts'][fc]:+.2f})")
        log("    SPY 剔四巫日: " + " | ".join(parts_s))

        # SW amount
        sub_sw_a = sw_df_amt[sw_df_amt[ar["key"]]].copy()
        parts_a = []
        for base_f in FEATURES:
            fc = f"{base_f}{ar['feat_suffix']}"
            r = run_controlled_reg(sub_sw_a, "crash_20", [fc], ar["cont_l1"], ar["vol_col"], sw_all_dates, add_fe=True)
            parts_a.append(f"{base_f}:{r['betas'][fc]:+.2f}pp(t={r['ts'][fc]:+.2f})")
        log("    A股 amount版: " + " | ".join(parts_a))

    # --- Auxiliary CSV export: up_std, brk60, H=10/40 (strictly off main table) ---
    aux_rows = []
    aux_arenas = [
        ("up_high", "", ["r20", "r60"], "sigma20"),
        ("up_std", "", ["r20", "r60"], "sigma20"),
        ("brk20", "_pre", ["r20_pre", "r1"], "sigma20_pre"),
        ("brk60", "_pre", ["r20_pre", "r1"], "sigma20_pre"),
    ]
    for akey, suf, c_l1, vcol in aux_arenas:
        for m in markets:
            sub = m["df"][m["df"][akey]].copy()
            for H in [10, 20, 40]:
                for base_f in FEATURES:
                    fc = f"{base_f}{suf}"
                    rc = run_controlled_reg(sub, f"crash_{H}", [fc], c_l1, vcol, m["all_dates"], add_fe=m["add_fe"], lag=int(1.5 * H))
                    rr = run_controlled_reg(sub, f"ret_{H}", [fc], c_l1, vcol, m["all_dates"], add_fe=m["add_fe"], lag=int(1.5 * H))
                    aux_rows.append(
                        {
                            "arena": akey,
                            "market": m["name"],
                            "H": H,
                            "feature": base_f,
                            "n": rc["n"],
                            "crash_g": rc["betas"][fc],
                            "crash_t": rc["ts"][fc],
                            "ret_g": rr["betas"][fc],
                            "ret_t": rr["ts"][fc],
                        }
                    )

    report_path = OUTPUT_DIR / "13_vp_report.txt"
    primary_csv = OUTPUT_DIR / "13_vp_primary_16.csv"
    aux_csv = OUTPUT_DIR / "13_vp_aux_horizons_arenas.csv"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    res_df.to_csv(primary_csv, index=False)
    pd.DataFrame(aux_rows).to_csv(aux_csv, index=False)
    log("\n" + "=" * 100)
    log(f"完整报告已保存: {report_path}")
    log(f"16 主检验表 CSV: {primary_csv}")
    log(f"辅考场/多期限 CSV: {aux_csv}")


if __name__ == "__main__":
    main()
