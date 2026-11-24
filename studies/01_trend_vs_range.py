#!/usr/bin/env python3
"""
Market Cycle V2 — Study 01: 趋势 vs 震荡（顺势/逆势开关）极简裸考

遵循 AGENTS.md 五条铁律：
1. 裁判先行：
   - 主裁判 Y1(t, H) = sign(r_{t-L->t}) * r_{t->t+H} / (sigma_L(t) * sqrt(H))  （顺势延续 vs 均值回归）
   - 副裁判 Y2(t, H) = |C_{t+H} - C_t| / sum_{i=t+1}^{t+H} |C_i - C_{i-1}|     （未来单边趋势效率）
2. 打败傻瓜基线：
   - 简单趋势：|r_L|（趋势强弱）与 r_L（带符号涨跌幅）
   - 简单波动率：sigma_L（近 L 日对数收益标准差）
   - 稳健性复核：用 ATR_L / C 替代 sigma_L、用 |C/MA_L - 1| 替代 |r_L|
3. 严守封存边界（先截断后算 Y，绝不碰封存区一根 K 线）：
   - SPX 开发段：1984-01-01 -> 2019-12-31（封存 2020-01-01 起）
   - A 股申万一级行业（31 个，SW2021 L1）开发段：2016-01-01 -> 2024-10-31（封存 2024-11-01 起）
4. 连续优先、单挑过堂、双市场照妖：
   - 4 个零拐点因果连续物理量：E_L（路径效率）、O_L（K 线重叠度）、B_L（实体方向力度）、V_L（ATR_L / ATR_4L）
   - SPX 用 Newey-West (lag=H)；A 股用 Driscoll-Kraay (lag=H) 与 Fama-MacBeth 截面回归 (NW lag=H)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SPX_PATH = REPO_ROOT / "data" / "snapshots" / "spx_daily_2026-07-21_d828fbc8.parquet"
MARKET_DATA_ROOT = Path("/home/zhujunshen/market-data")
CATALOG_DB = MARKET_DATA_ROOT / "catalog" / "metadata.sqlite"
OUTPUT_DIR = REPO_ROOT / "output"

SPX_DEV_START = pd.Timestamp("1984-01-01")
SPX_DEV_END = pd.Timestamp("2019-12-31")
SPX_WARMUP_START = pd.Timestamp("1982-01-01")

SW_DEV_START = pd.Timestamp("2016-01-01")
SW_DEV_END = pd.Timestamp("2024-10-31")

FEATURES = ["E", "O", "B", "V"]
FEATURE_LABELS = {
    "E": "路径效率 E_L",
    "O": "K线重叠度 O_L",
    "B": "实体方向力度 B_L",
    "V": "波动压缩/高潮 V_L",
}
HORIZONS = [10, 20, 60]
LOOKBACKS = [20, 60]
EPS = 1e-12


# ---------------------------------------------------------------------------
# 1. Data Loading (Strictly Read-Only, Hard Truncation Before Forward Labels)
# ---------------------------------------------------------------------------
def load_spx_dev() -> pd.DataFrame:
    df = pd.read_parquet(SPX_PATH).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    assert df["date"].is_unique, "SPX dates not unique"
    # HARD CUTOFF BEFORE ANY LABEL CALCULATION
    df = df[(df["date"] >= SPX_WARMUP_START) & (df["date"] <= SPX_DEV_END)].copy()
    df["asset"] = "SPX"
    return df[["asset", "date", "open", "high", "low", "close"]].reset_index(drop=True)


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
        assert len(snap) == 1, "No READY snapshot found in market-data catalog"
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

    # Only read 2016..2024 partitions (never touch 2025/2026 holdout files)
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
            columns=["src", "level", "index_code", "trade_date", "open", "high", "low", "close"],
        )
        for f in files
    ]
    df = pd.concat(parts, ignore_index=True)
    df = df[(df["src"] == "SW2021") & (df["level"] == "L1")].copy()
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    # HARD CUTOFF BEFORE ANY LABEL CALCULATION
    df = df[(df["date"] >= SW_DEV_START) & (df["date"] <= SW_DEV_END)].copy()
    df = df.rename(columns={"index_code": "asset"})
    df = df[["asset", "date", "open", "high", "low", "close"]].sort_values(["asset", "date"]).reset_index(drop=True)
    assert not df.duplicated(["asset", "date"]).any(), "Duplicate (asset, date) in SW L1"
    return df, sid


def check_ohlc_sanity(df: pd.DataFrame, name: str, eval_start: pd.Timestamp) -> str:
    sub = df[df["date"] >= eval_start]
    n_rows = len(sub)
    n_assets = sub["asset"].nunique()
    zero_range = (sub["high"] <= sub["low"]).mean()
    open_eq_close = (np.isclose(sub["open"], sub["close"])).mean()
    open_eq_prev_close = (
        sub.groupby("asset")["close"].shift(1).sub(sub["open"]).abs().le(1e-9).mean()
    )
    bad_hl = ((sub["high"] < sub[["open", "close"]].max(axis=1) - 1e-6) | (sub["low"] > sub[["open", "close"]].min(axis=1) + 1e-6)).mean()
    return (
        f"[{name}] 标的数={n_assets}, 开发段行数={n_rows} "
        f"({sub['date'].min().date()} ~ {sub['date'].max().date()}), "
        f"H<=L占比={zero_range:.2%}, O==C占比={open_eq_close:.2%}, "
        f"O==前C占比={open_eq_prev_close:.2%}, OHLC越界占比={bad_hl:.2%}"
    )


# ---------------------------------------------------------------------------
# 2. Causal Feature & Forward Label Computation (Per Asset)
# ---------------------------------------------------------------------------
def compute_asset_frame(g: pd.DataFrame, L: int, horizons: list[int]) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    o = g["open"].to_numpy(dtype=float)
    h = g["high"].to_numpy(dtype=float)
    l = g["low"].to_numpy(dtype=float)
    c = g["close"].to_numpy(dtype=float)
    n = len(g)

    prev_c = np.empty(n)
    prev_c[0] = np.nan
    prev_c[1:] = c[:-1]

    prev_h = np.empty(n)
    prev_h[0] = np.nan
    prev_h[1:] = h[:-1]

    prev_l = np.empty(n)
    prev_l[0] = np.nan
    prev_l[1:] = l[:-1]

    log_c = np.log(c)
    ret1 = np.empty(n)
    ret1[0] = np.nan
    ret1[1:] = np.diff(log_c)

    abs_dc = np.empty(n)
    abs_dc[0] = np.nan
    abs_dc[1:] = np.abs(np.diff(c))

    # --- Baselines (100% causal) ---
    ret_L = pd.Series(log_c).diff(L).to_numpy()
    abs_ret_L = np.abs(ret_L)
    vol_L = pd.Series(ret1).rolling(L, min_periods=L).std(ddof=1).to_numpy()
    ma_L = pd.Series(c).rolling(L, min_periods=L).mean().to_numpy()
    ma_dist_L = np.abs(c / ma_L - 1.0)

    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    rtr = tr / prev_c
    atr_rel_L = pd.Series(rtr).rolling(L, min_periods=L).mean().to_numpy()
    atr_rel_4L = pd.Series(rtr).rolling(4 * L, min_periods=4 * L).mean().to_numpy()

    # --- 4 Candidate Physical Quantities (100% causal) ---
    # 1. Path Efficiency E_L
    path_sum_L = pd.Series(abs_dc).rolling(L, min_periods=L).sum().to_numpy()
    net_move_L = np.abs(pd.Series(c).diff(L).to_numpy())
    E_L = np.where(path_sum_L > EPS, net_move_L / path_sum_L, np.nan)

    # 2. Bar Overlap O_L
    inter = np.maximum(0.0, np.minimum(h, prev_h) - np.maximum(l, prev_l))
    union = np.maximum(h, prev_h) - np.minimum(l, prev_l)
    ov = np.where(union > EPS, inter / union, np.nan)
    O_L = pd.Series(ov).rolling(L, min_periods=L).mean().to_numpy()

    # 3. Directional Body Strength B_L
    bar_range = h - l
    signed_body = np.where(bar_range > EPS, (c - o) / bar_range, 0.0)
    signed_body = np.clip(signed_body, -1.0, 1.0)
    B_L = np.abs(pd.Series(signed_body).rolling(L, min_periods=L).mean().to_numpy())

    # 4. Volatility Compression / Climax V_L = ATR_L / ATR_4L
    V_L = np.where(atr_rel_4L > EPS, atr_rel_L / atr_rel_4L, np.nan)

    out = pd.DataFrame(
        {
            "asset": g["asset"].to_numpy(),
            "date": g["date"].to_numpy(),
            "ret_L": ret_L,
            "abs_ret_L": abs_ret_L,
            "vol_L": vol_L,
            "ma_dist_L": ma_dist_L,
            "atr_rel_L": atr_rel_L,
            "E": E_L,
            "O": O_L,
            "B": B_L,
            "V": V_L,
        }
    )

    # Past direction sign (0 -> NaN so ambiguous flat days excluded from Y1)
    past_sign = np.where(ret_L > 0, 1.0, np.where(ret_L < 0, -1.0, np.nan))
    out["past_sign"] = past_sign

    # --- Forward Labels Y1 & Y2 (strictly within truncated dev frame) ---
    s_log_c = pd.Series(log_c)
    s_c = pd.Series(c)
    s_abs_dc = pd.Series(abs_dc)
    for H in horizons:
        fwd_ret = (s_log_c.shift(-H) - s_log_c).to_numpy()
        y1_raw = past_sign * fwd_ret
        denom = vol_L * np.sqrt(H)
        y1_norm = np.where(denom > EPS, y1_raw / denom, np.nan)

        # Future path efficiency Y2(t, H): sum_{i=t+1}^{t+H} |C_i - C_{i-1}|
        fwd_path_sum = s_abs_dc.rolling(H, min_periods=H).sum().shift(-H).to_numpy()
        fwd_net = np.abs((s_c.shift(-H) - s_c).to_numpy())
        y2_eff = np.where(fwd_path_sum > EPS, fwd_net / fwd_path_sum, np.nan)

        out[f"Y1_{H}"] = y1_norm
        out[f"Y1_raw_{H}"] = y1_raw
        out[f"Y2_{H}"] = y2_eff

    return out


def build_dataset(raw: pd.DataFrame, eval_start: pd.Timestamp, L: int) -> pd.DataFrame:
    frames = [compute_asset_frame(g, L=L, horizons=HORIZONS) for _, g in raw.groupby("asset", sort=True)]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["date"] >= eval_start].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. Econometric Engines: OLS + Newey-West / Driscoll-Kraay / Fama-MacBeth
# ---------------------------------------------------------------------------
def ols_hac(
    y: np.ndarray,
    X: np.ndarray,
    dates: np.ndarray | None,
    lag: int,
    panel_dk: bool = False,
) -> dict[str, np.ndarray | float]:
    """
    OLS with either:
      - Newey-West HAC (panel_dk=False, single sorted time series)
      - Driscoll-Kraay HAC (panel_dk=True, aggregates scores by trading date first)
    X must already include intercept column if desired.
    """
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else np.nan

    scores = X * resid[:, None]  # (n, k)
    if panel_dk:
        assert dates is not None
        # Sum scores across assets for each unique trading date (sorted)
        uniq_dates, inv = np.unique(dates, return_inverse=True)
        T = len(uniq_dates)
        h = np.zeros((T, k), dtype=float)
        np.add.at(h, inv, scores)
    else:
        h = scores
        T = n

    omega = h.T @ h
    K = min(lag, T - 1)
    for ell in range(1, K + 1):
        w = 1.0 - ell / (K + 1.0)
        gamma_ell = h[ell:].T @ h[:-ell]
        omega += w * (gamma_ell + gamma_ell.T)

    cov = xtx_inv @ omega @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tstat = np.where(se > EPS, beta / se, np.nan)
    return {"beta": beta, "se": se, "t": tstat, "r2": r2, "n": n, "T": T}


def mean_nw_tstat(series: np.ndarray, lag: int) -> tuple[float, float]:
    """Mean and Newey-West t-stat of a 1D time series."""
    s = series[np.isfinite(series)]
    T = len(s)
    if T < 5:
        return np.nan, np.nan
    m = float(np.mean(s))
    e = s - m
    var_sum = float(np.dot(e, e))
    K = min(lag, T - 1)
    for ell in range(1, K + 1):
        w = 1.0 - ell / (K + 1.0)
        var_sum += 2.0 * w * float(np.dot(e[ell:], e[:-ell]))
    se = np.sqrt(max(var_sum, 0.0)) / T
    return m, (m / se if se > EPS else np.nan)


def run_controlled_regression(
    df: pd.DataFrame,
    y_col: str,
    feat: str,
    baseline_cols: list[str],
    lag: int,
    panel_dk: bool,
) -> dict[str, float]:
    cols = ["date", y_col, feat] + baseline_cols
    sub = df[cols].dropna()
    if len(sub) < 50:
        return {"gamma": np.nan, "t": np.nan, "delta_r2": np.nan, "r2_base": np.nan, "n": len(sub)}

    if not panel_dk:
        sub = sub.sort_values("date")

    y = sub[y_col].to_numpy(dtype=float)
    dates = sub["date"].to_numpy()

    def zscore_mat(frame: pd.DataFrame, col_list: list[str]) -> np.ndarray:
        arr = frame[col_list].to_numpy(dtype=float)
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=0)
        sd = np.where(sd > EPS, sd, 1.0)
        return (arr - mu) / sd

    Z_base = zscore_mat(sub, baseline_cols)
    Z_feat = zscore_mat(sub, [feat])
    ones = np.ones((len(sub), 1), dtype=float)

    res_base = ols_hac(y, np.hstack([ones, Z_base]), dates=dates, lag=lag, panel_dk=panel_dk)
    res_full = ols_hac(y, np.hstack([ones, Z_base, Z_feat]), dates=dates, lag=lag, panel_dk=panel_dk)

    return {
        "gamma": float(res_full["beta"][-1]),
        "t": float(res_full["t"][-1]),
        "r2_base": float(res_base["r2"]),
        "r2_full": float(res_full["r2"]),
        "delta_r2": float(res_full["r2"] - res_base["r2"]),
        "n": int(res_full["n"]),
        "T": int(res_full["T"]),
    }


def run_fama_macbeth(
    df: pd.DataFrame,
    y_col: str,
    feat: str,
    baseline_cols: list[str],
    lag: int,
) -> dict[str, float]:
    """Daily cross-sectional OLS across 31 industries; time-series NW t-stat on gamma_t."""
    cols = ["date", y_col, feat] + baseline_cols
    sub = df[cols].dropna().sort_values("date")
    gammas = []
    for _, g in sub.groupby("date", sort=True):
        if len(g) < 15:
            continue
        x_cols = baseline_cols + [feat]
        X_raw = g[x_cols].to_numpy(dtype=float)
        sd = X_raw.std(axis=0, ddof=0)
        if np.any(sd <= EPS):
            continue
        X_z = (X_raw - X_raw.mean(axis=0)) / sd
        X = np.hstack([np.ones((len(g), 1)), X_z])
        y = g[y_col].to_numpy(dtype=float)
        beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        gammas.append(float(beta[-1]))
    arr = np.array(gammas, dtype=float)
    mean_g, t_g = mean_nw_tstat(arr, lag=lag)
    return {"gamma": mean_g, "t": t_g, "T": len(arr)}


# ---------------------------------------------------------------------------
# 4. Quintile Bare Exam & 3x3 Double Sort
# ---------------------------------------------------------------------------
def quintile_table(
    df: pd.DataFrame,
    feat: str,
    y_col: str,
    raw_col: str | None,
    lag: int,
    panel_dk: bool,
) -> dict[str, float]:
    cols = ["date", feat, y_col] + ([raw_col] if raw_col else [])
    sub = df[cols].dropna().copy()
    if not panel_dk:
        sub = sub.sort_values("date")
    sub["q"] = pd.qcut(sub[feat].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    out: dict[str, float] = {}
    q_means = []
    for q in range(1, 6):
        m = float(sub.loc[sub["q"] == q, y_col].mean())
        out[f"Q{q}"] = m
        q_means.append(m)
        if raw_col:
            out[f"Q{q}_raw"] = float(sub.loc[sub["q"] == q, raw_col].mean())
            out[f"Q{q}_win"] = float((sub.loc[sub["q"] == q, raw_col] > 0).mean())

    # Q5 - Q1 with honest HAC t-stat via dummy regression on Q1 vs Q5
    q15 = sub[sub["q"].isin([1, 5])].copy()
    y15 = q15[y_col].to_numpy(dtype=float)
    d5 = (q15["q"] == 5).to_numpy(dtype=float)
    X15 = np.column_stack([np.ones(len(q15)), d5])
    res = ols_hac(y15, X15, dates=q15["date"].to_numpy(), lag=lag, panel_dk=panel_dk)
    out["Q5_Q1"] = float(res["beta"][1])
    out["t_Q5_Q1"] = float(res["t"][1])
    # Monotonicity check: Spearman rank correlation of [1..5] vs [Q1..Q5]
    ranks = pd.Series(q_means).rank().to_numpy()
    idx = np.arange(1, 6, dtype=float)
    out["mono_rho"] = float(np.corrcoef(idx, ranks)[0, 1])

    if raw_col:
        out["Q5_Q1_raw"] = out["Q5_raw"] - out["Q1_raw"]
        out["Q5_Q1_win"] = out["Q5_win"] - out["Q1_win"]
    return out


def double_sort_table(
    df: pd.DataFrame,
    ctrl_col: str,
    feat: str,
    y_col: str,
    raw_col: str | None,
    lag: int,
    panel_dk: bool,
) -> list[dict[str, float | str]]:
    """First sort into 3 buckets by ctrl_col (Weak/Mid/Strong), then 3 buckets by feat (Low/Mid/High)."""
    cols = ["date", ctrl_col, feat, y_col] + ([raw_col] if raw_col else [])
    sub = df[cols].dropna().copy()
    if not panel_dk:
        sub = sub.sort_values("date")
    sub["c_bin"] = pd.qcut(sub[ctrl_col].rank(method="first"), 3, labels=["弱趋势", "中趋势", "强趋势"])

    rows = []
    for c_label in ["弱趋势", "中趋势", "强趋势"]:
        g = sub[sub["c_bin"] == c_label].copy()
        g["f_bin"] = pd.qcut(g[feat].rank(method="first"), 3, labels=["低", "中", "高"])
        row: dict[str, float | str] = {"趋势组": c_label}
        for fl in ["低", "中", "高"]:
            gg = g[g["f_bin"] == fl]
            row[f"{fl}_Y"] = float(gg[y_col].mean())
            if raw_col:
                row[f"{fl}_raw%"] = float(gg[raw_col].mean() * 100.0)
        # High - Low HAC t-stat within this trend bucket
        ghl = g[g["f_bin"].isin(["低", "高"])].copy()
        d_hi = (ghl["f_bin"] == "高").to_numpy(dtype=float)
        Xhl = np.column_stack([np.ones(len(ghl)), d_hi])
        res = ols_hac(
            ghl[y_col].to_numpy(dtype=float),
            Xhl,
            dates=ghl["date"].to_numpy(),
            lag=lag,
            panel_dk=panel_dk,
        )
        row["高-低_Y"] = float(res["beta"][1])
        row["t(高-低)"] = float(res["t"][1])
        if raw_col:
            row["高-低_raw%"] = float(row["高_raw%"] - row["低_raw%"])
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 5. Main Execution & Reporting
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 92)
    log("Market Cycle V2 — Study 01: 趋势 vs 震荡（顺势/逆势开关）极简裸考")
    log("=" * 92)

    spx_raw = load_spx_dev()
    sw_raw, sw_sid = load_sw_l1_dev()

    log(f"A 股快照来源（只读）: {sw_sid}")
    log(check_ohlc_sanity(spx_raw, "SPX 开发段", SPX_DEV_START))
    log(check_ohlc_sanity(sw_raw, "A股申万L1 开发段", SW_DEV_START))
    log("封存边界确认: SPX 最后一根输入 K 线 = "
        f"{spx_raw['date'].max().date()} (<= 2019-12-31); "
        f"SW L1 最后一根输入 K 线 = {sw_raw['date'].max().date()} (<= 2024-10-31)")
    log()

    summary_records: list[dict] = []

    markets = [
        {
            "name": "SPX",
            "raw": spx_raw,
            "eval_start": SPX_DEV_START,
            "panel_dk": False,
            "splits": [
                ("前半 1984-2001", pd.Timestamp("1984-01-01"), pd.Timestamp("2001-12-31")),
                ("后半 2002-2019", pd.Timestamp("2002-01-01"), pd.Timestamp("2019-12-31")),
            ],
        },
        {
            "name": "A股申万L1",
            "raw": sw_raw,
            "eval_start": SW_DEV_START,
            "panel_dk": True,
            "splits": [
                ("前半 2016-2020", pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31")),
                ("后半 2021-2024.10", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-10-31")),
            ],
        },
    ]

    # --- Baseline reference stats (what does dumb trend-following earn?) ---
    log("-" * 92)
    log("【零号参照】无脑顺势（按过去 L=20 天涨跌方向顺势持有 H 天）的基础表现")
    log("-" * 92)
    for m in markets:
        df20 = build_dataset(m["raw"], m["eval_start"], L=20)
        for H in HORIZONS:
            sub = df20[["date", "past_sign", f"Y1_{H}", f"Y1_raw_{H}", f"Y2_{H}"]].dropna()
            up = sub[sub["past_sign"] > 0]
            dn = sub[sub["past_sign"] < 0]
            log(
                f"{m['name']:<8s} H={H:>2d} | "
                f"全体: Y1={sub[f'Y1_{H}'].mean():+.3f}, 原始={sub[f'Y1_raw_{H}'].mean()*100:+.2f}%, 胜率={(sub[f'Y1_raw_{H}']>0).mean():.1%} | "
                f"过去涨(多): 原始={up[f'Y1_raw_{H}'].mean()*100:+.2f}%, 胜率={(up[f'Y1_raw_{H}']>0).mean():.1%} | "
                f"过去跌(空): 原始={dn[f'Y1_raw_{H}'].mean()*100:+.2f}%, 胜率={(dn[f'Y1_raw_{H}']>0).mean():.1%} | "
                f"未来效率Y2均值={sub[f'Y2_{H}'].mean():.3f}"
            )
    log()

    # --- Main Controlled Regression Table across L in {20, 60} and H in {10, 20, 60} ---
    baseline_cols = ["abs_ret_L", "ret_L", "vol_L"]
    alt_baseline_cols = ["ma_dist_L", "ret_L", "atr_rel_L"]

    for L in LOOKBACKS:
        log("=" * 92)
        log(f"【过堂总表】回看窗口 L={L}：扣掉傻瓜基线 (|r_L|, r_L, sigma_L) 后的净增量 gamma 与抗自相关 t 值")
        log("（注：所有自变量已标准化，gamma 表示特征每+1标准差对应的 Y 变化；SPX 用 NW(H)，A 股用 DK(H) + 截面 FM(H)）")
        log("=" * 92)

        for m in markets:
            df = build_dataset(m["raw"], m["eval_start"], L=L)
            log(f"\n>>> 市场：{m['name']}（L={L}）")

            for ref_name, y_prefix in [("主裁判 Y1（顺势延续）", "Y1"), ("副裁判 Y2（未来趋势效率）", "Y2")]:
                log(f"\n  [{ref_name}]")
                header = (
                    f"  {'特征':<14s} {'H':>2s} | {'裸考Q5-Q1':>9s} {'t_裸':>6s} {'单调ρ':>6s} | "
                    f"{'控基线γ':>8s} {'t_控':>6s} {'ΔR²%':>7s} | {'前半t':>6s} {'后半t':>6s} | {'换基线t':>7s}"
                )
                if m["panel_dk"]:
                    header += f" | {'截面FM_γ':>8s} {'FM_t':>6s}"
                log(header)
                log("  " + "-" * (len(header) - 2))

                for feat in FEATURES:
                    for H in HORIZONS:
                        y_col = f"{y_prefix}_{H}"
                        raw_col = f"Y1_raw_{H}" if y_prefix == "Y1" else None

                        qt = quintile_table(df, feat, y_col, raw_col, lag=H, panel_dk=m["panel_dk"])
                        reg = run_controlled_regression(df, y_col, feat, baseline_cols, lag=H, panel_dk=m["panel_dk"])
                        reg_alt = run_controlled_regression(df, y_col, feat, alt_baseline_cols, lag=H, panel_dk=m["panel_dk"])

                        split_ts = []
                        split_gammas = []
                        for _, s_start, s_end in m["splits"]:
                            sub_s = df[(df["date"] >= s_start) & (df["date"] <= s_end)]
                            r_s = run_controlled_regression(sub_s, y_col, feat, baseline_cols, lag=H, panel_dk=m["panel_dk"])
                            split_ts.append(r_s["t"])
                            split_gammas.append(r_s["gamma"])

                        fm = (
                            run_fama_macbeth(df, y_col, feat, baseline_cols, lag=H)
                            if m["panel_dk"]
                            else {"gamma": np.nan, "t": np.nan}
                        )

                        line = (
                            f"  {FEATURE_LABELS[feat]:<12s} {H:>2d} | "
                            f"{qt['Q5_Q1']:>+9.4f} {qt['t_Q5_Q1']:>+6.2f} {qt['mono_rho']:>+6.2f} | "
                            f"{reg['gamma']:>+8.4f} {reg['t']:>+6.2f} {reg['delta_r2']*100:>6.3f}% | "
                            f"{split_ts[0]:>+6.2f} {split_ts[1]:>+6.2f} | {reg_alt['t']:>+7.2f}"
                        )
                        if m["panel_dk"]:
                            line += f" | {fm['gamma']:>+8.4f} {fm['t']:>+6.2f}"
                        log(line)

                        summary_records.append(
                            {
                                "market": m["name"],
                                "L": L,
                                "referee": y_prefix,
                                "H": H,
                                "feature": feat,
                                "Q5_Q1": qt["Q5_Q1"],
                                "t_Q5_Q1": qt["t_Q5_Q1"],
                                "mono_rho": qt["mono_rho"],
                                "gamma_ctrl": reg["gamma"],
                                "t_ctrl": reg["t"],
                                "delta_r2_pct": reg["delta_r2"] * 100.0,
                                "t_split1": split_ts[0],
                                "t_split2": split_ts[1],
                                "t_alt_base": reg_alt["t"],
                                "fm_gamma": fm["gamma"],
                                "fm_t": fm["t"],
                                "n": reg["n"],
                            }
                        )

    # --- Up vs Down Split on Primary L=20, Y1 (Prevent Bull-Drift Illusion) ---
    log("\n" + "=" * 92)
    log("【防长牛骗局复核】L=20 主裁判 Y1 拆开「过去上涨(多头顺势)」vs「过去下跌(空头顺势)」")
    log("（控制 |r_20| 与 sigma_20 后，看特征在上涨段与下跌段是否方向一致）")
    log("=" * 92)
    for m in markets:
        df20 = build_dataset(m["raw"], m["eval_start"], L=20)
        log(f"\n>>> {m['name']} (L=20)")
        log(f"  {'特征':<14s} {'H':>2s} | {'上涨段γ':>8s} {'上涨段t':>7s} {'上涨ΔR²%':>8s} | {'下跌段γ':>8s} {'下跌段t':>7s} {'下跌ΔR²%':>8s}")
        log("  " + "-" * 74)
        up_df = df20[df20["past_sign"] > 0]
        dn_df = df20[df20["past_sign"] < 0]
        for feat in FEATURES:
            for H in HORIZONS:
                y_col = f"Y1_{H}"
                r_up = run_controlled_regression(up_df, y_col, feat, ["abs_ret_L", "vol_L"], lag=H, panel_dk=m["panel_dk"])
                r_dn = run_controlled_regression(dn_df, y_col, feat, ["abs_ret_L", "vol_L"], lag=H, panel_dk=m["panel_dk"])
                log(
                    f"  {FEATURE_LABELS[feat]:<12s} {H:>2d} | "
                    f"{r_up['gamma']:>+8.4f} {r_up['t']:>+7.2f} {r_up['delta_r2']*100:>7.3f}% | "
                    f"{r_dn['gamma']:>+8.4f} {r_dn['t']:>+7.2f} {r_dn['delta_r2']*100:>7.3f}%"
                )

    # --- Quintile Detail & 3x3 Double Sort for Primary L=20, H=20 (Catch Nonlinearity!) ---
    log("\n" + "=" * 92)
    log("【五分位明细 + 3x3 双排序表（防漏掉非线性/高潮见顶）】主窗口 L=20, 持有期 H=20")
    log("=" * 92)
    for m in markets:
        df20 = build_dataset(m["raw"], m["eval_start"], L=20)
        log(f"\n>>> {m['name']} (L=20, H=20)")

        log("\n  [1] 裸考五分位明细 — 主裁判 Y1（顺势延续：归一值 / 原始收益% / 胜率%）")
        for feat in FEATURES:
            qt = quintile_table(df20, feat, "Y1_20", "Y1_raw_20", lag=20, panel_dk=m["panel_dk"])
            q_str = " | ".join(
                f"Q{q}:{qt[f'Q{q}']:+.2f}({qt[f'Q{q}_raw']*100:+.2f}%,{qt[f'Q{q}_win']:.0%})"
                for q in range(1, 6)
            )
            log(f"  {FEATURE_LABELS[feat]:<12s}: {q_str}  || Q5-Q1={qt['Q5_Q1']:+.3f} (t={qt['t_Q5_Q1']:+.2f})")

        log("\n  [2] 裸考五分位明细 — 副裁判 Y2（未来 20 天路径效率）")
        for feat in FEATURES:
            qt = quintile_table(df20, feat, "Y2_20", None, lag=20, panel_dk=m["panel_dk"])
            q_str = " | ".join(f"Q{q}:{qt[f'Q{q}']:.3f}" for q in range(1, 6))
            log(f"  {FEATURE_LABELS[feat]:<12s}: {q_str}  || Q5-Q1={qt['Q5_Q1']:+.4f} (t={qt['t_Q5_Q1']:+.2f})")

        log("\n  [3] 3x3 双排序表（按 |r_20| 趋势强弱分 3 行，行内按特征分 低/中/高 3 列）— 主裁判 Y1")
        for feat in FEATURES:
            rows = double_sort_table(df20, "abs_ret_L", feat, "Y1_20", "Y1_raw_20", lag=20, panel_dk=m["panel_dk"])
            log(f"  -- {FEATURE_LABELS[feat]} --")
            for r in rows:
                log(
                    f"     {r['趋势组']} | 低:{r['低_Y']:+.3f}({r['低_raw%']:+.2f}%) "
                    f"| 中:{r['中_Y']:+.3f}({r['中_raw%']:+.2f}%) "
                    f"| 高:{r['高_Y']:+.3f}({r['高_raw%']:+.2f}%) "
                    f"|| 高-低={r['高-低_Y']:+.3f}({r['高-低_raw%']:+.2f}%), t={r['t(高-低)']:+.2f}"
                )

        log("\n  [4] 3x3 双排序表（按 |r_20| 趋势强弱分 3 行，行内按特征分 低/中/高 3 列）— 副裁判 Y2")
        for feat in FEATURES:
            rows = double_sort_table(df20, "abs_ret_L", feat, "Y2_20", None, lag=20, panel_dk=m["panel_dk"])
            log(f"  -- {FEATURE_LABELS[feat]} --")
            for r in rows:
                log(
                    f"     {r['趋势组']} | 低:{r['低_Y']:.3f} | 中:{r['中_Y']:.3f} | 高:{r['高_Y']:.3f} "
                    f"|| 高-低={r['高-低_Y']:+.4f}, t={r['t(高-低)']:+.2f}"
                )

    report_path = OUTPUT_DIR / "01_trend_vs_range_report.txt"
    csv_path = OUTPUT_DIR / "01_trend_vs_range_summary.csv"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(summary_records).to_csv(csv_path, index=False)
    log("\n" + "=" * 92)
    log(f"报告已保存: {report_path}")
    log(f"汇总 CSV 已保存: {csv_path}")


if __name__ == "__main__":
    main()
