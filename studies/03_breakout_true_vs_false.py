#!/usr/bin/env python3
"""
Market Cycle V2 — Study 03: 真突破 vs 假突破（刚创 20/60 日新高时的事前蓄势与当日力度）

严格遵循五条铁律，严守封存边界（先截断再算未来标签，绝不碰封存区一根 K 线）：
  - SPX 开发段：1984-01-01 -> 2019-12-31（全程不用 open，零早年数据污染）
  - A 股 31 个申万一级行业开发段：2016-01-01 -> 2024-10-31

两个「刚冒头」的突破考场（避免连涨期每天重复算突破）：
  1. 20日新高首次突破 (brk20): 今日收盘 > 过去20日最高收盘，且过去5个交易日未创20日新高
  2. 60日新高首次突破 (brk60): 今日收盘 > 过去60日最高收盘，且过去10个交易日未创60日新高

4 个真假突破候选特征（全部严格因果、不用 open）：
  事前（截至 t-1）：
    - V_pre : 突破前振幅压缩 = ATR_rel_20(t-1) / ATR_rel_80(t-1)（越低越像缩波蓄势）
    - E_pre : 突破前横盘拧巴度 = E_20(t-1)（越低越像从平台横盘起跳，而非已直拉很远）
  当日（第 t 天）：
    - K_cls : 突破日收盘力度 = (C_t - L_t) / (H_t - L_t)（接近 1 为光头强收，低为冲高回落）
    - M_brk : 突破日净突破幅度 = (C_t - 前高) / ATR_20(t-1)

裁判（突破日收盘买入，观察未来 H in {10, 20, 40} 天）：
  - ret_H     : 未来平均收益 & 胜率
  - fail_10   : 假突破打脸率（未来 10 天内收盘重新跌破前高）
  - mdd_H     : 未来持有期内最深回撤（及踩雷率：A 股 20天<-8%、SPX <-5%）
  - big_win_H : 真突破大肉率（未来 H 天盘中最高涨幅 >= 8% [A股] / >= 4% [SPX]）

傻瓜基线（必须打败）：
  突破前 20 日涨幅 r20_pre + 突破前 20 日绝对波动率 vol20_pre
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"

spec = importlib.util.spec_from_file_location("s01", REPO_ROOT / "studies" / "01_trend_vs_range.py")
s01 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(s01)

EPS = 1e-12
HORIZONS = [10, 20, 40]
FEATURES = ["V_pre", "E_pre", "K_cls", "M_brk"]
FEAT_LABELS = {
    "V_pre": "事前振幅压缩 V_pre",
    "E_pre": "事前路径效率 E_pre",
    "K_cls": "当日收盘力度 K_cls",
    "M_brk": "当日突破幅度 M_brk",
}
THRESH = {
    "A股申万L1": {"crash_20": -0.08, "big_20": 0.08, "big_40": 0.10},
    "SPX": {"crash_20": -0.05, "big_20": 0.04, "big_40": 0.06},
}


def compute_breakout_frame(g: pd.DataFrame, market_name: str) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    h = g["high"].to_numpy(dtype=float)
    l = g["low"].to_numpy(dtype=float)
    c = g["close"].to_numpy(dtype=float)
    n = len(g)

    prev_c = np.empty(n)
    prev_c[0] = np.nan
    prev_c[1:] = c[:-1]

    log_c = np.log(c)
    ret1 = np.empty(n)
    ret1[0] = np.nan
    ret1[1:] = np.diff(log_c)
    abs_dc = np.empty(n)
    abs_dc[0] = np.nan
    abs_dc[1:] = np.abs(np.diff(c))

    s_c = pd.Series(c)
    s_h = pd.Series(h)
    s_l = pd.Series(l)
    s_log_c = pd.Series(log_c)
    s_ret1 = pd.Series(ret1)

    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    rtr = tr / prev_c
    atr20_raw = pd.Series(tr).rolling(20, min_periods=20).mean()
    atr20_rel = pd.Series(rtr).rolling(20, min_periods=20).mean()
    atr80_rel = pd.Series(rtr).rolling(80, min_periods=80).mean()

    # Pre-breakout state (strictly up to t-1)
    V_pre = (atr20_rel / atr80_rel).shift(1).to_numpy()
    path20 = pd.Series(abs_dc).rolling(20, min_periods=20).sum()
    net20 = s_c.diff(20).abs()
    E_pre = np.where(path20.shift(1).to_numpy() > EPS, (net20 / path20).shift(1).to_numpy(), np.nan)

    r20_pre = s_log_c.diff(20).shift(1).to_numpy()
    vol20_pre = s_ret1.rolling(20, min_periods=20).std(ddof=1).shift(1).to_numpy()
    atr20_pre_raw = atr20_raw.shift(1).to_numpy()

    # Breakout-day features
    rng = h - l
    K_cls = np.full(n, 0.5, dtype=float)
    valid_rng = rng > EPS
    K_cls[valid_rng] = np.clip((c[valid_rng] - l[valid_rng]) / rng[valid_rng], 0.0, 1.0)

    # Fresh breakout triggers: N=20 (cool=5) and N=60 (cool=10)
    prev_hi_20 = s_c.rolling(20, min_periods=20).max().shift(1).to_numpy()
    is_nh_20 = c > prev_hi_20
    recent_20 = pd.Series(is_nh_20).shift(1).rolling(5, min_periods=5).sum().to_numpy() > 0
    brk20 = is_nh_20 & (~recent_20) & np.isfinite(prev_hi_20)
    M_brk20 = np.where(atr20_pre_raw > EPS, (c - prev_hi_20) / atr20_pre_raw, np.nan)

    prev_hi_60 = s_c.rolling(60, min_periods=60).max().shift(1).to_numpy()
    is_nh_60 = c > prev_hi_60
    recent_60 = pd.Series(is_nh_60).shift(1).rolling(10, min_periods=10).sum().to_numpy() > 0
    brk60 = is_nh_60 & (~recent_60) & np.isfinite(prev_hi_60)
    M_brk60 = np.where(atr20_pre_raw > EPS, (c - prev_hi_60) / atr20_pre_raw, np.nan)

    out = pd.DataFrame(
        {
            "asset": g["asset"].to_numpy(),
            "date": g["date"].to_numpy(),
            "r20_pre": r20_pre,
            "vol20_pre": vol20_pre,
            "V_pre": V_pre,
            "E_pre": E_pre,
            "K_cls": K_cls,
            "brk20": brk20,
            "prev_hi_20": prev_hi_20,
            "M_brk20": M_brk20,
            "brk60": brk60,
            "prev_hi_60": prev_hi_60,
            "M_brk60": M_brk60,
        }
    )

    # Forward outcomes strictly inside truncated dev frame
    min_c_10 = s_c.iloc[::-1].rolling(10, min_periods=10).min().iloc[::-1].shift(-1).to_numpy()
    out["fail10_20"] = np.where(np.isfinite(min_c_10), (min_c_10 < prev_hi_20).astype(float), np.nan)
    out["fail10_60"] = np.where(np.isfinite(min_c_10), (min_c_10 < prev_hi_60).astype(float), np.nan)

    thr = THRESH[market_name]
    for H in HORIZONS:
        fwd_ret = (s_log_c.shift(-H) - s_log_c).to_numpy()
        min_l = s_l.iloc[::-1].rolling(H, min_periods=H).min().iloc[::-1].shift(-1).to_numpy()
        max_h = s_h.iloc[::-1].rolling(H, min_periods=H).max().iloc[::-1].shift(-1).to_numpy()
        mdd = (min_l - c) / c
        mfe = (max_h - c) / c
        out[f"ret_{H}"] = fwd_ret
        out[f"mdd_{H}"] = mdd
        out[f"mfe_{H}"] = mfe
        if H == 20:
            out["crash_20"] = np.where(np.isfinite(mdd), (mdd <= thr["crash_20"]).astype(float), np.nan)
            out["big_20"] = np.where(np.isfinite(mfe), (mfe >= thr["big_20"]).astype(float), np.nan)
        if H == 40:
            out["big_40"] = np.where(np.isfinite(mfe), (mfe >= thr["big_40"]).astype(float), np.nan)

    return out


def event_ols_hac(
    y: np.ndarray,
    X: np.ndarray,
    dates: np.ndarray,
    all_trading_dates: np.ndarray,
    lag: int,
) -> dict[str, np.ndarray | float]:
    """
    Exact calendar-aligned Driscoll-Kraay / Newey-West for sparse overlapping events.
    Scores are mapped onto the full market trading calendar (0 on non-event days)
    before applying the Bartlett kernel of exact trading-day bandwidth = lag.
    """
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    scores = X * resid[:, None]

    T = len(all_trading_dates)
    idx = np.searchsorted(all_trading_dates, dates)
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
    return {"beta": beta, "se": se, "t": tstat, "n": n}


def run_event_ctrl_reg(
    ev: pd.DataFrame,
    y_col: str,
    feat: str,
    base_cols: list[str],
    all_dates: np.ndarray,
    lag: int,
    scale: float = 100.0,
) -> dict[str, float]:
    cols = ["date", y_col, feat] + base_cols
    sub = ev[cols].dropna()
    if len(sub) < 40:
        return {"gamma": np.nan, "t": np.nan, "n": len(sub)}
    y = sub[y_col].to_numpy(dtype=float) * scale
    X_raw = sub[base_cols + [feat]].to_numpy(dtype=float)
    sd = X_raw.std(axis=0, ddof=0)
    sd = np.where(sd > EPS, sd, 1.0)
    X_z = (X_raw - X_raw.mean(axis=0)) / sd
    X = np.hstack([np.ones((len(sub), 1)), X_z])
    res = event_ols_hac(y, X, sub["date"].to_numpy(), all_dates, lag=lag)
    return {"gamma": float(res["beta"][-1]), "t": float(res["t"][-1]), "n": int(res["n"])}


def tercile_or_quintile_table(
    ev: pd.DataFrame,
    feat: str,
    fail_col: str,
    all_dates: np.ndarray,
    n_bins: int = 3,
) -> dict[str, float]:
    cols = ["date", feat, "ret_20", "ret_40", fail_col, "mdd_20", "crash_20", "big_20", "big_40"]
    sub = ev[cols].dropna().copy()
    if len(sub) < 45:
        return {}
    sub["b"] = pd.qcut(sub[feat].rank(method="first"), n_bins, labels=list(range(1, n_bins + 1))).astype(int)
    out: dict[str, float] = {"n": float(len(sub))}
    for b in range(1, n_bins + 1):
        g = sub[sub["b"] == b]
        out[f"B{b}_ret20"] = float(g["ret_20"].mean() * 100.0)
        out[f"B{b}_win20"] = float((g["ret_20"] > 0).mean() * 100.0)
        out[f"B{b}_fail10"] = float(g[fail_col].mean() * 100.0)
        out[f"B{b}_mdd20"] = float(g["mdd_20"].mean() * 100.0)
        out[f"B{b}_crash20"] = float(g["crash_20"].mean() * 100.0)
        out[f"B{b}_big20"] = float(g["big_20"].mean() * 100.0)
        out[f"B{b}_big40"] = float(g["big_40"].mean() * 100.0)

    # High - Low exact calendar HAC t-stat
    hl = sub[sub["b"].isin([1, n_bins])].copy()
    d_hi = (hl["b"] == n_bins).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(hl)), d_hi])
    dates_hl = hl["date"].to_numpy()
    for name, col, lag in [
        ("ret20", "ret_20", 20),
        ("fail10", fail_col, 10),
        ("crash20", "crash_20", 20),
        ("big20", "big_20", 20),
        ("big40", "big_40", 40),
    ]:
        r = event_ols_hac(hl[col].to_numpy(dtype=float) * 100.0, X, dates_hl, all_dates, lag=lag)
        out[f"d_{name}"] = float(r["beta"][1])
        out[f"t_{name}"] = float(r["t"][1])
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 98)
    log("Market Cycle V2 — Study 03: 真突破 vs 假突破（20/60日首次新高：事前蓄势与当日力度）")
    log("=" * 98)

    spx_raw = s01.load_spx_dev()
    sw_raw, sw_sid = s01.load_sw_l1_dev()
    log(f"封存边界确认: SPX 最后输入日={spx_raw['date'].max().date()} (<=2019-12-31); "
        f"A股最后输入日={sw_raw['date'].max().date()} (<=2024-10-31), 快照={sw_sid}")

    def build_df(raw: pd.DataFrame, start: pd.Timestamp, mname: str) -> pd.DataFrame:
        parts = [compute_breakout_frame(g, mname) for _, g in raw.groupby("asset", sort=True)]
        df = pd.concat(parts, ignore_index=True)
        return df[df["date"] >= start].reset_index(drop=True)

    sw_df = build_df(sw_raw, s01.SW_DEV_START, "A股申万L1")
    spx_df = build_df(spx_raw, s01.SPX_DEV_START, "SPX")

    markets = [
        {
            "name": "A股申万L1",
            "df": sw_df,
            "all_dates": np.sort(sw_df["date"].unique()),
            "splits": [
                ("前半 2016-2020", pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31")),
                ("后半 2021-2024.10", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-10-31")),
            ],
        },
        {
            "name": "SPX",
            "df": spx_df,
            "all_dates": np.sort(spx_df["date"].unique()),
            "splits": [
                ("前半 1984-2001", pd.Timestamp("1984-01-01"), pd.Timestamp("2001-12-31")),
                ("后半 2002-2019", pd.Timestamp("2002-01-01"), pd.Timestamp("2019-12-31")),
            ],
        },
    ]

    base_cols = ["r20_pre", "vol20_pre"]
    summary_rows: list[dict] = []

    # --- 1. Base Rates of Breakouts ---
    log("\n" + "-" * 98)
    log("【突破底数】无脑买「刚冒头的新高突破」的基础表现（未来 20 天）")
    log("-" * 98)
    for m in markets:
        df = m["df"]
        thr = THRESH[m["name"]]
        for brk_col, fail_col, m_col, lbl in [
            ("brk20", "fail10_20", "M_brk20", "20日首次新高(冷却5天)"),
            ("brk60", "fail10_60", "M_brk60", "60日首次新高(冷却10天)"),
        ]:
            ev = df[df[brk_col]].dropna(subset=["ret_20", fail_col, "mdd_20", "big_20"])
            log(
                f"  {m['name']:<8s} | {lbl:<22s} | 次数={len(ev):>4d} | "
                f"20天均收益={ev['ret_20'].mean()*100:+5.2f}%, 胜率={(ev['ret_20']>0).mean()*100:4.1f}% | "
                f"10天跌回前高={ev[fail_col].mean()*100:4.1f}% | "
                f"均最大回撤={ev['mdd_20'].mean()*100:6.2f}%, 踩雷={ev['crash_20'].mean()*100:4.1f}% | "
                f"20天大肉(>={int(thr['big_20']*100)}%)={ev['big_20'].mean()*100:4.1f}%"
            )

    # --- 2. Controlled Regressions across 20d & 60d Breakouts ---
    for brk_col, fail_col, m_col, b_lbl in [
        ("brk20", "fail10_20", "M_brk20", "考场一：20日首次新高突破（短中期平台）"),
        ("brk60", "fail10_60", "M_brk60", "考场二：60日首次新高突破（中期大平台）"),
    ]:
        log("\n" + "=" * 98)
        log(f"【过堂总表】{b_lbl}")
        log("扣掉傻瓜基线 (突破前20日涨幅 r20_pre + 突破前波动率 vol20_pre) 后的净增量（特征每+1标准差）")
        log("=" * 98)

        for m in markets:
            ev = m["df"][m["df"][brk_col]].copy()
            ev["M_brk"] = ev[m_col]
            all_dates = m["all_dates"]
            log(f"\n>>> 市场：{m['name']}（总突破样本数={len(ev)}）")
            log(
                f"  {'特征':<16s} | {'20天收益γ%':>9s} {'t_收':>6s} | {'10天打脸γ%':>9s} {'t_脸':>6s} | "
                f"{'20天踩雷γ%':>9s} {'t_雷':>6s} | {'20天大肉γ%':>9s} {'t_肉':>6s} | {'前半收t':>7s} {'后半收t':>7s}"
            )
            log("  " + "-" * 102)

            for feat in FEATURES:
                r_ret = run_event_ctrl_reg(ev, "ret_20", feat, base_cols, all_dates, lag=20)
                r_ret40 = run_event_ctrl_reg(ev, "ret_40", feat, base_cols, all_dates, lag=40)
                r_fail = run_event_ctrl_reg(ev, fail_col, feat, base_cols, all_dates, lag=10)
                r_cr = run_event_ctrl_reg(ev, "crash_20", feat, base_cols, all_dates, lag=20)
                r_big = run_event_ctrl_reg(ev, "big_20", feat, base_cols, all_dates, lag=20)
                r_big40 = run_event_ctrl_reg(ev, "big_40", feat, base_cols, all_dates, lag=40)

                sp_t = []
                for _, d0, d1 in m["splits"]:
                    sub_s = ev[(ev["date"] >= d0) & (ev["date"] <= d1)]
                    sp_t.append(run_event_ctrl_reg(sub_s, "ret_20", feat, base_cols, all_dates, lag=20)["t"])

                log(
                    f"  {FEAT_LABELS[feat]:<14s} | {r_ret['gamma']:>+9.2f} {r_ret['t']:>+6.2f} | "
                    f"{r_fail['gamma']:>+9.2f} {r_fail['t']:>+6.2f} | "
                    f"{r_cr['gamma']:>+9.2f} {r_cr['t']:>+6.2f} | "
                    f"{r_big['gamma']:>+9.2f} {r_big['t']:>+6.2f} | {sp_t[0]:>+7.2f} {sp_t[1]:>+7.2f}"
                )

                summary_rows.append(
                    {
                        "breakout": brk_col,
                        "market": m["name"],
                        "feature": feat,
                        "n": r_ret["n"],
                        "ret20_g": r_ret["gamma"],
                        "ret20_t": r_ret["t"],
                        "ret40_g": r_ret40["gamma"],
                        "ret40_t": r_ret40["t"],
                        "fail10_g": r_fail["gamma"],
                        "fail10_t": r_fail["t"],
                        "crash20_g": r_cr["gamma"],
                        "crash20_t": r_cr["t"],
                        "big20_g": r_big["gamma"],
                        "big20_t": r_big["t"],
                        "big40_g": r_big40["gamma"],
                        "big40_t": r_big40["t"],
                        "split1_ret20_t": sp_t[0],
                        "split2_ret20_t": sp_t[1],
                    }
                )

    # --- 3. Practical 3-Bucket Table (低 / 中 / 高) for 20d & 60d Breakouts ---
    for brk_col, fail_col, m_col, b_lbl in [
        ("brk20", "fail10_20", "M_brk20", "20日首次新高"),
        ("brk60", "fail10_60", "M_brk60", "60日首次新高"),
    ]:
        log("\n" + "=" * 98)
        log(f"【实战三档落袋表】{b_lbl}：低 / 中 / 高 三档的真实收益、打脸率、踩雷率与大肉率")
        log("=" * 98)
        for m in markets:
            ev = m["df"][m["df"][brk_col]].copy()
            ev["M_brk"] = ev[m_col]
            thr = THRESH[m["name"]]
            log(f"\n>>> {m['name']}（{b_lbl}；踩雷=20天跌超{int(abs(thr['crash_20'])*100)}%，大肉20/40天={int(thr['big_20']*100)}%/{int(thr['big_40']*100)}%）")
            for feat in FEATURES:
                tb = tercile_or_quintile_table(ev, feat, fail_col, m["all_dates"], n_bins=3)
                if not tb:
                    continue
                log(f"\n  -- {FEAT_LABELS[feat]} (突破次数={int(tb['n'])}) --")
                for b, bl in [(1, "低档"), (2, "中档"), (3, "高档")]:
                    log(
                        f"     {bl}: 20天收益={tb[f'B{b}_ret20']:+6.2f}% (胜率{tb[f'B{b}_win20']:4.1f}%) | "
                        f"10天跌回前高={tb[f'B{b}_fail10']:4.1f}% | 均最大回撤={tb[f'B{b}_mdd20']:6.2f}% (踩雷{tb[f'B{b}_crash20']:4.1f}%) | "
                        f"大肉20/40天={tb[f'B{b}_big20']:4.1f}% / {tb[f'B{b}_big40']:4.1f}%"
                    )
                log(
                    f"     高-低: 20天收益={tb['d_ret20']:+6.2f}%(t={tb['t_ret20']:+.2f}), "
                    f"10天打脸={tb['d_fail10']:+5.1f}%(t={tb['t_fail10']:+.2f}), "
                    f"20天踩雷={tb['d_crash20']:+5.1f}%(t={tb['t_crash20']:+.2f}), "
                    f"大肉20天={tb['d_big20']:+5.1f}%(t={tb['t_big20']:+.2f})"
                )

    # --- 4. Direct Spring-Breakout Combo Check: 低V_pre + 低E_pre vs 高V_pre + 高E_pre (20d Breakout) ---
    log("\n" + "=" * 98)
    log("【弹簧突破组合体检】20日首次新高：突破前「安静缩波+横盘拧巴」vs「高波+已直拉」（按中位数2x2切分）")
    log("=" * 98)
    for m in markets:
        ev = m["df"][m["df"]["brk20"]].dropna(subset=["V_pre", "E_pre", "ret_20", "ret_40", "fail10_20", "mdd_20", "crash_20", "big_20", "big_40"]).copy()
        v_med = ev["V_pre"].median()
        e_med = ev["E_pre"].median()
        log(f"\n>>> {m['name']}（20日首次新高，总次数={len(ev)}）")
        for v_hi, vl in [(False, "事前缩波"), (True, "事前放波")]:
            for e_hi, el in [(False, "横盘起跳"), (True, "已直拉高")]:
                g = ev[((ev["V_pre"] >= v_med) == v_hi) & ((ev["E_pre"] >= e_med) == e_hi)]
                log(
                    f"     {vl} + {el} (n={len(g):>4d}) | "
                    f"20天收益={g['ret_20'].mean()*100:+6.2f}% (胜率{(g['ret_20']>0).mean()*100:4.1f}%) | "
                    f"10天跌回前高={g['fail10_20'].mean()*100:4.1f}% | "
                    f"踩雷={g['crash_20'].mean()*100:4.1f}% | 大肉20/40天={g['big_20'].mean()*100:4.1f}%/{g['big_40'].mean()*100:4.1f}%"
                )

    # --- 5. A-share 20d Breakout Year-by-Year for V_pre (check consistency across 2016-2024) ---
    log("\n" + "=" * 98)
    log("【A 股 20 日首次新高逐年体检】事前缩波 V_pre 低一半 vs 高一半（2016..2024）")
    log("=" * 98)
    ev_sw = sw_df[sw_df["brk20"]].dropna(subset=["V_pre", "ret_20", "fail10_20", "crash_20", "big_20"]).copy()
    ev_sw["year"] = ev_sw["date"].dt.year
    log(f"  {'年份':<6s} {'次数':>4s} | {'缩波收益%':>9s} {'放波收益%':>9s} {'差%':>7s} | {'缩波打脸%':>9s} {'放波打脸%':>9s} | {'缩波踩雷%':>9s} {'放波踩雷%':>9s}")
    log("  " + "-" * 86)
    for yr, g in ev_sw.groupby("year"):
        if len(g) < 20:
            continue
        med = g["V_pre"].median()
        lo = g[g["V_pre"] < med]
        hi = g[g["V_pre"] >= med]
        log(
            f"  {yr:<6d} {len(g):>4d} | {lo['ret_20'].mean()*100:>+8.2f}% {hi['ret_20'].mean()*100:>+8.2f}% "
            f"{(hi['ret_20'].mean()-lo['ret_20'].mean())*100:>+6.2f}% | "
            f"{lo['fail10_20'].mean()*100:>8.1f}% {hi['fail10_20'].mean()*100:>8.1f}% | "
            f"{lo['crash_20'].mean()*100:>8.1f}% {hi['crash_20'].mean()*100:>8.1f}%"
        )

    report_path = OUTPUT_DIR / "03_breakout_true_vs_false_report.txt"
    csv_path = OUTPUT_DIR / "03_breakout_true_vs_false_summary.csv"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    log("\n" + "=" * 98)
    log(f"报告已保存: {report_path}")
    log(f"汇总 CSV 已保存: {csv_path}")


if __name__ == "__main__":
    main()
