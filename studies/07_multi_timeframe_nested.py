#!/usr/bin/env python3
"""
Market Cycle V2 — Study 07 (第二阶段方向 C):
多周期嵌套（周线大背景 Always-In × 日线回调 / 牛旗突破 / 突破回踩 / 第2推新高）

严格遵循 AGENTS.md 五条铁律，严守封存边界与多周期防未来函数纪律：
  - SPX 开发段：1984-01-01 -> 2019-12-31（全程只用 H/L/C，不用 open）
  - A 股 31 个申万一级行业开发段：2016-01-01 -> 2024-10-31
  - 周线严格防未来函数：任意交易日 t 仅引用「截至上一完整交易周收盘 (week_end_date < t)」的周线指标，
    绝不偷看本周未收盘数据！

三道裁判考题：
  考题 1（周线大背景 × 日线经典动作）：
    将日线 5 类动作（全部20日新高 / 横盘牛旗突破 / 顺大逆小回调抬头 / 突破回踩BP_H1 / 第2推新高）
    分别放入「周线多头 / 周线震荡 / 周线空头」三个背景格子里，看加上周线滤网后表现如何分化。
  考题 2（核心照妖镜：日线择时 vs 周线多头闭眼买）：
    在「周线多头」大背景下，严格对比上述日线择时买点 vs 「同在周线多头下的任意一天闭眼买」，
    查清到底是多周期嵌套产生了增量，还是纯粹在沾周线趋势本身的光。
  考题 3（周线早中晚期拆解：刚转强早期多头 vs 高位老多头）：
    把周线多头按「站上20周线时长 (<=12周 vs >12周)」与「过去26周涨幅 (温和 vs 高位)」劈成两半，
    检验中美两个市场在周线不同生命周期下做日线顺势接回调/牛旗的性格差异。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"


def _load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "studies" / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


s01 = _load_mod("s01", "01_trend_vs_range.py")
s03 = _load_mod("s03", "03_breakout_true_vs_false.py")
s05 = _load_mod("s05", "05_breakout_pullback_and_failed.py")
s06 = _load_mod("s06", "06_ordered_structures_channel_wedge.py")

EPS = 1e-12
THRESH = {
    "A股申万L1": {"crash_20": -0.08, "big_20": 0.08, "big_40": 0.10},
    "SPX": {"crash_20": -0.05, "big_20": 0.04, "big_40": 0.06},
}


# ---------------------------------------------------------------------------
# 1. Strictly Causal Weekly Regime Builder (Previous Completed Week Only)
# ---------------------------------------------------------------------------
def build_weekly_regime(g: pd.DataFrame) -> pd.DataFrame:
    """
    Build weekly OHLC (H/L/C) grouped by W-FRI natural calendar week,
    compute causal weekly indicators, and keep week_end_date for strict backward asof join.
    """
    d = g.sort_values("date").copy()
    d["wk_period"] = d["date"].dt.to_period("W-FRI")
    wk = (
        d.groupby("wk_period", sort=True)
        .agg(
            week_end_date=("date", "max"),
            w_high=("high", "max"),
            w_low=("low", "min"),
            w_close=("close", "last"),
        )
        .reset_index(drop=True)
    )
    n = len(wk)
    wc = wk["w_close"].to_numpy(dtype=float)
    wh = wk["w_high"].to_numpy(dtype=float)
    wl = wk["w_low"].to_numpy(dtype=float)

    prev_wc = np.empty(n)
    prev_wc[0] = np.nan
    prev_wc[1:] = wc[:-1]
    w_tr = np.maximum.reduce([wh - wl, np.abs(wh - prev_wc), np.abs(wl - prev_wc)])
    s_wc = pd.Series(wc)
    s_log_wc = pd.Series(np.log(wc))

    w_ma20 = s_wc.rolling(20, min_periods=20).mean().to_numpy()
    w_ma20_lag4 = pd.Series(w_ma20).shift(4).to_numpy()
    w_ma20_slope = np.where(w_ma20_lag4 > EPS, (w_ma20 - w_ma20_lag4) / w_ma20_lag4, np.nan)
    w_ret12 = s_log_wc.diff(12).to_numpy()
    w_ret26 = s_log_wc.diff(26).to_numpy()
    w_atr10 = pd.Series(w_tr).rolling(10, min_periods=10).mean().to_numpy()
    w_dist_ma20 = np.where(w_atr10 > EPS, (wc - w_ma20) / w_atr10, np.nan)

    # Consecutive weeks above w_ma20 (strictly causal)
    w_above_streak = np.zeros(n, dtype=float)
    streak = 0
    for i in range(n):
        if np.isfinite(w_ma20[i]) and wc[i] > w_ma20[i]:
            streak += 1
        else:
            streak = 0
        w_above_streak[i] = float(streak)

    # Clean 3-way weekly regime: W_BULL / W_BEAR / W_RANGE
    w_regime = np.full(n, "", dtype=object)
    for i in range(n):
        if not (np.isfinite(w_ma20[i]) and np.isfinite(w_ma20_slope[i]) and np.isfinite(w_ret12[i])):
            continue
        if (wc[i] > w_ma20[i]) and (w_ma20_slope[i] > 0.0) and (w_ret12[i] > 0.0):
            w_regime[i] = "W_BULL"
        elif (wc[i] < w_ma20[i]) and (w_ma20_slope[i] < 0.0) and (w_ret12[i] < 0.0):
            w_regime[i] = "W_BEAR"
        else:
            w_regime[i] = "W_RANGE"

    wk["w_regime"] = w_regime
    wk["w_ret12"] = w_ret12
    wk["w_ret26"] = w_ret26
    wk["w_above_streak"] = w_above_streak
    wk["w_dist_ma20"] = w_dist_ma20
    return wk[["week_end_date", "w_regime", "w_ret12", "w_ret26", "w_above_streak", "w_dist_ma20"]]


# ---------------------------------------------------------------------------
# 2. Per-Asset Daily Signals + Strict Previous-Week Join
# ---------------------------------------------------------------------------
def build_asset_study07(
    g: pd.DataFrame,
    market_name: str,
    eval_start: pd.Timestamp,
) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    wk = build_weekly_regime(g)

    # Strict backward merge_asof with allow_exact_matches=False -> ONLY completed prior week!
    merged = pd.merge_asof(
        g,
        wk.sort_values("week_end_date"),
        left_on="date",
        right_on="week_end_date",
        direction="backward",
        allow_exact_matches=False,
    )

    h = merged["high"].to_numpy(dtype=float)
    l = merged["low"].to_numpy(dtype=float)
    c = merged["close"].to_numpy(dtype=float)
    n = len(merged)

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

    s_c = pd.Series(c)
    s_h = pd.Series(h)
    s_l = pd.Series(l)
    s_log_c = pd.Series(log_c)
    s_ret1 = pd.Series(ret1)

    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr20 = pd.Series(tr).rolling(20, min_periods=20).mean().to_numpy()
    ma10 = s_c.rolling(10, min_periods=10).mean().to_numpy()
    ma20 = s_c.rolling(20, min_periods=20).mean().to_numpy()
    r20 = s_log_c.diff(20).to_numpy()
    r5 = s_log_c.diff(5).to_numpy()
    vol20 = s_ret1.rolling(20, min_periods=20).std(ddof=1).to_numpy()

    # Pre-breakout path efficiency E_pre (up to t-1) for Bull Flag detection
    path20 = pd.Series(abs_dc).rolling(20, min_periods=20).sum()
    net20 = s_c.diff(20).abs()
    e_pre = np.where(path20.shift(1).to_numpy() > EPS, (net20 / path20).shift(1).to_numpy(), np.nan)

    # 1) Fresh 20d breakout (brk20) & Horizontal Bull Flag Breakout (flag_brk: brk20 & E_pre <= 0.35)
    prev_c_hi20 = s_c.rolling(20, min_periods=20).max().shift(1).to_numpy()
    is_nh20 = c > prev_c_hi20
    recent_nh20 = pd.Series(is_nh20).shift(1).rolling(5, min_periods=5).sum().to_numpy() > 0
    brk20 = is_nh20 & (~recent_nh20) & np.isfinite(prev_c_hi20)
    flag_brk = brk20 & np.isfinite(e_pre) & (e_pre <= 0.35)

    # 2) Daily Pullback Turn-Up ("顺大逆小" 日线回调抬头, cooldown 5 bars)
    d_dip = np.zeros(n, dtype=bool)
    last_dip = -100
    for t in range(25, n):
        if not (np.isfinite(ma10[t]) and np.isfinite(ma20[t]) and np.isfinite(r5[t])):
            continue
        had_dip = ((l[t - 1] < l[t - 2]) or (c[t - 1] < c[t - 2])) and (min(l[t], l[t - 1]) <= ma10[t])
        turn_up = (h[t] > h[t - 1]) and (c[t] > c[t - 1])
        not_crashed = c[t] >= ma20[t] * 0.96
        if had_dip and turn_up and not_crashed and (t - last_dip >= 5):
            d_dip[t] = True
            last_dip = t

    # 3) Study 05 Breakout Pullback BP_H1_20 (re-test inside weekly regimes)
    d_bp_h1 = np.zeros(n, dtype=bool)
    for t0 in np.flatnonzero(brk20):
        p_brk = prev_c_hi20[t0]
        in_pb = False
        peak_c = c[t0]
        for t in range(t0 + 1, min(n, t0 + 13)):
            dt = t - t0
            is_dn = (l[t] < l[t - 1]) or (c[t] < c[t - 1])
            is_up = (h[t] > h[t - 1]) and (c[t] > c[t - 1])
            if not in_pb:
                if is_dn and (c[t] < peak_c or l[t] < l[t - 1]):
                    in_pb = True
            else:
                if is_up:
                    if (2 <= dt <= 12) and (c[t] >= p_brk):
                        d_bp_h1[t] = True
                    break
            if c[t] > peak_c:
                peak_c = c[t]

    # 4) Study 06 Push 2 Breakout & Push 3 Breakout (re-test inside weekly regimes)
    prev_h_hi20 = s_h.rolling(20, min_periods=20).max().shift(1).to_numpy()
    prev_l_lo20 = s_l.rolling(20, min_periods=20).min().shift(1).to_numpy()
    d_push2 = np.zeros(n, dtype=bool)
    d_push3 = np.zeros(n, dtype=bool)
    push_cnt = 0
    leg_lows: list[float] = []
    cur_peak = -np.inf
    cur_peak_idx = -1
    in_pb = False
    pb_low = np.inf
    seq_start = -1
    for t in range(25, n):
        a = atr20[t]
        if not (np.isfinite(a) and a > EPS and np.isfinite(prev_h_hi20[t])):
            continue
        if push_cnt == 0:
            if (h[t] > prev_h_hi20[t]) and (c[t] > ma20[t]):
                push_cnt = 1
                cur_peak = h[t]
                cur_peak_idx = t
                seq_start = t
                leg_lows = []
                in_pb = False
                pb_low = l[t]
        else:
            broke = (t - seq_start > 65) or (l[t] < prev_l_lo20[t]) or (bool(leg_lows) and c[t] < leg_lows[-1])
            if broke:
                push_cnt = 0
                continue
            if not in_pb:
                if h[t] > cur_peak:
                    cur_peak = h[t]
                    cur_peak_idx = t
                    pb_low = l[t]
                else:
                    if l[t] < pb_low:
                        pb_low = l[t]
                    if (cur_peak - pb_low >= 1.0 * a) and (t - cur_peak_idx >= 2):
                        in_pb = True
            else:
                if l[t] < pb_low:
                    pb_low = l[t]
                if h[t] > cur_peak:
                    leg_lows.append(pb_low)
                    push_cnt += 1
                    cur_peak = h[t]
                    cur_peak_idx = t
                    in_pb = False
                    pb_low = l[t]
                    if push_cnt == 2:
                        d_push2[t] = True
                    elif push_cnt == 3:
                        d_push3[t] = True

    # Forward outcomes strictly inside truncated dev frame
    thr = THRESH[market_name]
    ret_10 = (s_log_c.shift(-10) - s_log_c).to_numpy()
    ret_20 = (s_log_c.shift(-20) - s_log_c).to_numpy()
    ret_40 = (s_log_c.shift(-40) - s_log_c).to_numpy()
    min_l_20 = s_l.iloc[::-1].rolling(20, min_periods=20).min().iloc[::-1].shift(-1).to_numpy()
    max_h_20 = s_h.iloc[::-1].rolling(20, min_periods=20).max().iloc[::-1].shift(-1).to_numpy()
    max_h_40 = s_h.iloc[::-1].rolling(40, min_periods=40).max().iloc[::-1].shift(-1).to_numpy()
    mdd_20 = (min_l_20 - c) / c
    mfe_20 = (max_h_20 - c) / c
    mfe_40 = (max_h_40 - c) / c
    crash_20 = np.where(np.isfinite(mdd_20), (mdd_20 <= thr["crash_20"]).astype(float), np.nan)
    big_20 = np.where(np.isfinite(mfe_20), (mfe_20 >= thr["big_20"]).astype(float), np.nan)
    big_40 = np.where(np.isfinite(mfe_40), (mfe_40 >= thr["big_40"]).astype(float), np.nan)

    out = pd.DataFrame(
        {
            "asset": merged["asset"],
            "date": merged["date"],
            "w_regime": merged["w_regime"],
            "w_ret12": merged["w_ret12"],
            "w_ret26": merged["w_ret26"],
            "w_above_streak": merged["w_above_streak"],
            "w_dist_ma20": merged["w_dist_ma20"],
            "r20": r20,
            "r5": r5,
            "vol20": vol20,
            "d_brk20": brk20,
            "d_flag_brk": flag_brk,
            "d_dip": d_dip,
            "d_bp_h1": d_bp_h1,
            "d_push2": d_push2,
            "d_push3": d_push3,
            "ret_10": ret_10,
            "ret_20": ret_20,
            "ret_40": ret_40,
            "mdd_20": mdd_20,
            "mfe_20": mfe_20,
            "crash_20": crash_20,
            "big_20": big_20,
            "big_40": big_40,
        }
    )
    out = out[(out["date"] >= eval_start) & (out["w_regime"].isin(["W_BULL", "W_RANGE", "W_BEAR"]))].reset_index(drop=True)
    return out


def assemble_market(raw: pd.DataFrame, market_name: str, eval_start: pd.Timestamp) -> tuple[pd.DataFrame, np.ndarray]:
    parts = [build_asset_study07(g, market_name, eval_start) for _, g in raw.groupby("asset", sort=True)]
    df = pd.concat(parts, ignore_index=True)
    all_dates = np.sort(df["date"].unique())
    return df, all_dates


def fmt_row(label: str, df: pd.DataFrame) -> str:
    st = s05.event_stats(df)
    if st.get("n", 0) == 0:
        return f"  {label:<24s} | n=    0"
    return (
        f"  {label:<24s} | n={st['n']:>5d} | "
        f"10天={st['ret10']:+5.2f}% | 20天={st['ret20']:+5.2f}% (胜{st['win20']:4.1f}%) | "
        f"40天={st['ret40']:+5.2f}% | 回撤={st['mdd20']:6.2f}% 雷={st['crash20']:4.1f}% | "
        f"大肉20={st['big20']:4.1f}% | 盈亏比={st['rr20']:4.2f}"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 114)
    log("Market Cycle V2 — Study 07 (方向 C): 多周期嵌套（周线大背景 Always-In × 日线回调/牛旗/回踩/二推）")
    log("=" * 114)

    spx_raw = s01.load_spx_dev()
    sw_raw, sw_sid = s01.load_sw_l1_dev()
    log(
        f"封存边界确认: SPX 最后输入日={spx_raw['date'].max().date()} (<=2019-12-31, 全程不用open); "
        f"A股最后输入日={sw_raw['date'].max().date()} (<=2024-10-31), 快照={sw_sid}"
    )
    log("周线严格防未来函数: 每天仅引用「上一完整自然周收盘」的周K线指标 (allow_exact_matches=False)")

    sw_df, sw_dates = assemble_market(sw_raw, "A股申万L1", s01.SW_DEV_START)
    spx_df, spx_dates = assemble_market(spx_raw, "SPX", s01.SPX_DEV_START)

    markets = [
        {
            "name": "A股申万L1",
            "df": sw_df,
            "all_dates": sw_dates,
            "splits": [
                ("前半 2016-2020", pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31")),
                ("后半 2021-2024.10", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-10-31")),
            ],
        },
        {
            "name": "SPX",
            "df": spx_df,
            "all_dates": spx_dates,
            "splits": [
                ("前半 1984-2001", pd.Timestamp("1984-01-01"), pd.Timestamp("2001-12-31")),
                ("后半 2002-2019", pd.Timestamp("2002-01-01"), pd.Timestamp("2019-12-31")),
            ],
        },
    ]

    signals = [
        ("d_dip", "日线回调抬头 (顺大逆小)"),
        ("d_flag_brk", "日线横盘牛旗突破 (低E)"),
        ("d_brk20", "日线全部20日首次新高"),
        ("d_bp_h1", "日线突破回踩 (BP_H1)"),
        ("d_push2", "日线第2推新高 (Push2)"),
        ("d_push3", "日线第3推新高 (Push3)"),
    ]

    # =========================================================================
    # 考题 1：周线大背景底数 & 周线大背景 × 日线经典动作
    # =========================================================================
    log("\n" + "=" * 114)
    log("【考题 1】周线大背景本身底数，以及「周线多头 / 震荡 / 空头 × 日线经典动作」全对照")
    log("=" * 114)

    for m in markets:
        df = m["df"]
        all_dates = m["all_dates"]
        log(f"\n>>> 市场：{m['name']}（总有效日样本={len(df)}）")
        log("  [0. 周线大背景自身底数（不挑日线、随便哪天闭眼买）]")
        for rg, rlbl in [("W_BULL", "周线多头 (W_BULL)"), ("W_RANGE", "周线震荡 (W_RANGE)"), ("W_BEAR", "周线空头 (W_BEAR)")]:
            sub = df[df["w_regime"] == rg]
            pct = len(sub) / len(df) * 100.0
            log(fmt_row(f"{rlbl} [{pct:4.1f}%]", sub))

        c_bb = s05.compare_vs_control(df[df["w_regime"] == "W_BULL"], df[df["w_regime"] == "W_BEAR"], all_dates, ["vol20"], "ret_20", 20)
        c_br = s05.compare_vs_control(df[df["w_regime"] == "W_BULL"], df[df["w_regime"] == "W_RANGE"], all_dates, ["vol20"], "ret_20", 20)
        log(
            f"  [周线背景自身差距(控vol20)] 多头vs空头={c_bb['ctrl_diff']:+5.2f}%(t={c_bb['ctrl_t']:+5.2f}) | "
            f"多头vs震荡={c_br['ctrl_diff']:+5.2f}%(t={c_br['ctrl_t']:+5.2f})"
        )

        for sig_col, sig_lbl in signals:
            log(f"\n  --- {sig_lbl} 在不同周线背景下的表现 ---")
            for rg, rlbl in [("W_BULL", "周线多头"), ("W_RANGE", "周线震荡"), ("W_BEAR", "周线空头")]:
                sub = df[(df["w_regime"] == rg) & df[sig_col]]
                log(fmt_row(f"  {rlbl} + {sig_col}", sub))

    # =========================================================================
    # 考题 2：核心照妖镜——周线多头里的「日线择时」vs「周线多头随便哪天闭眼买」
    # =========================================================================
    log("\n" + "=" * 114)
    log("【考题 2】核心照妖镜：同在「周线多头」里，等日线结构买点 vs 随便哪天闭眼买（检验日线嵌套有无净增量）")
    log("=" * 114)

    for m in markets:
        df = m["df"]
        all_dates = m["all_dates"]
        bull = df[df["w_regime"] == "W_BULL"]
        log(f"\n>>> 市场：{m['name']}（周线多头总天数={len(bull)}）")
        log(fmt_row("基线: 周线多头任意一天买", bull))
        log("  " + "-" * 106)
        for sig_col, sig_lbl in signals:
            ev_s = bull[bull[sig_col]]
            c_ret = s05.compare_vs_control(ev_s, bull, all_dates, ["r20", "vol20"], "ret_20", 20)
            c_cr = s05.compare_vs_control(ev_s, bull, all_dates, ["r20", "vol20"], "crash_20", 20)
            c_big = s05.compare_vs_control(ev_s, bull, all_dates, ["r20", "vol20"], "big_20", 20)
            sp_t = []
            for _, d0, d1 in m["splits"]:
                t_s = ev_s[(ev_s["date"] >= d0) & (ev_s["date"] <= d1)]
                b_s = bull[(bull["date"] >= d0) & (bull["date"] <= d1)]
                sp_t.append(s05.compare_vs_control(t_s, b_s, all_dates, ["r20", "vol20"], "ret_20", 20)["ctrl_t"])
            log(
                f"  {sig_lbl:<20s} (n={c_ret['nt']:>4d}) | 裸差={c_ret['raw_diff']:+5.2f}%(t={c_ret['raw_t']:+5.2f}) | "
                f"控基线收差={c_ret['ctrl_diff']:+5.2f}%(t={c_ret['ctrl_t']:+5.2f}), "
                f"踩雷差={c_cr['ctrl_diff']:+5.1f}%(t={c_cr['ctrl_t']:+5.2f}), "
                f"大肉差={c_big['ctrl_diff']:+5.1f}%(t={c_big['ctrl_t']:+5.2f}) | 前/后半t={sp_t[0]:+.2f}/{sp_t[1]:+.2f}"
            )

    # =========================================================================
    # 考题 3：周线多头的「早中晚期」——刚转强早期多头 vs 涨了很久的老多头
    # =========================================================================
    log("\n" + "=" * 114)
    log("【考题 3】周线多头早中晚期拆解：刚站上20周线不久（早期）vs 已经涨了很久（老多头）")
    log("=" * 114)

    for m in markets:
        df = m["df"]
        all_dates = m["all_dates"]
        bull = df[df["w_regime"] == "W_BULL"].copy()
        log(f"\n>>> 市场：{m['name']}（周线多头总天数={len(bull)}）")

        # Split A: Streak above 20-week MA (<=12 weeks vs >12 weeks)
        early_st = bull[bull["w_above_streak"] <= 12]
        late_st = bull[bull["w_above_streak"] > 12]
        log("  [拆法 A：按连续站上 20 周线时长（<=12周 早期刚转强 vs >12周 老多头）]")
        log(fmt_row("  周线早期多头(<=12周) 闭眼买", early_st))
        log(fmt_row("  周线老多头  (>12周)  闭眼买", late_st))
        c_el = s05.compare_vs_control(early_st, late_st, all_dates, ["vol20"], "ret_20", 20)
        c_el_cr = s05.compare_vs_control(early_st, late_st, all_dates, ["vol20"], "crash_20", 20)
        log(
            f"    早期vs老多头(控vol20): 20天收益差={c_el['ctrl_diff']:+5.2f}%(t={c_el['ctrl_t']:+5.2f}) | "
            f"踩雷差={c_el_cr['ctrl_diff']:+5.1f}%(t={c_el_cr['ctrl_t']:+5.2f})"
        )

        # Split B: 26-week return median split inside W_BULL
        med_w26 = bull["w_ret26"].median()
        early_r = bull[bull["w_ret26"] <= med_w26]
        late_r = bull[bull["w_ret26"] > med_w26]
        log(f"  [拆法 B：按过去 26 周(半年)涨幅中位数 ({med_w26*100:+.1f}%) 劈开（温和多头 vs 高位老多头）]")
        log(fmt_row("  周线温和多头(低半) 闭眼买", early_r))
        log(fmt_row("  周线高位多头(高半) 闭眼买", late_r))
        c_er = s05.compare_vs_control(early_r, late_r, all_dates, ["vol20"], "ret_20", 20)
        c_er_cr = s05.compare_vs_control(early_r, late_r, all_dates, ["vol20"], "crash_20", 20)
        log(
            f"    温和vs高位(控vol20): 20天收益差={c_er['ctrl_diff']:+5.2f}%(t={c_er['ctrl_t']:+5.2f}) | "
            f"踩雷差={c_er_cr['ctrl_diff']:+5.1f}%(t={c_er_cr['ctrl_t']:+5.2f})"
        )

        # Check daily actions inside Early vs Late Weekly Bull (by streak <=12 vs >12)
        log("  [嵌套实测：早期周线多头 vs 老周线多头里做日线经典动作]")
        for sig_col, sig_lbl in [
            ("d_dip", "日线回调抬头"),
            ("d_flag_brk", "日线横盘牛旗突破"),
            ("d_bp_h1", "日线突破回踩"),
            ("d_push2", "日线第2推新高"),
        ]:
            log(fmt_row(f"    早期多头+{sig_lbl}", early_st[early_st[sig_col]]))
            log(fmt_row(f"    老多头  +{sig_lbl}", late_st[late_st[sig_col]]))

    # =========================================================================
    # 逐年体检：A 股周线多头 vs 周线空头 / 早期 vs 老多头（防单年主导）
    # =========================================================================
    log("\n" + "=" * 114)
    log("【A 股逐年拆解】核查周线大背景（多头 vs 空头、早期多头 vs 老多头）在 2016-2024 每年的真实表现")
    log("=" * 114)
    sw_yr = sw_df.dropna(subset=["ret_20", "crash_20"]).copy()
    sw_yr["year"] = pd.to_datetime(sw_yr["date"]).dt.year
    log(
        f"  {'年份':<6s} | {'周多20天%':>8s} {'周空20天%':>8s} {'多-空%':>7s} | "
        f"{'早期多%':>8s} {'老多头%':>8s} {'早-老%':>7s} | {'早期雷%':>7s} {'老多雷%':>7s}"
    )
    log("  " + "-" * 86)
    for yr, g in sw_yr.groupby("year"):
        gb = g[g["w_regime"] == "W_BULL"]
        gs = g[g["w_regime"] == "W_BEAR"]
        ge = gb[gb["w_above_streak"] <= 12]
        gl = gb[gb["w_above_streak"] > 12]
        r_b = gb["ret_20"].mean() * 100.0 if len(gb) >= 10 else np.nan
        r_s = gs["ret_20"].mean() * 100.0 if len(gs) >= 10 else np.nan
        r_e = ge["ret_20"].mean() * 100.0 if len(ge) >= 10 else np.nan
        r_l = gl["ret_20"].mean() * 100.0 if len(gl) >= 10 else np.nan
        cr_e = ge["crash_20"].mean() * 100.0 if len(ge) >= 10 else np.nan
        cr_l = gl["crash_20"].mean() * 100.0 if len(gl) >= 10 else np.nan
        log(
            f"  {yr:<6d} | {r_b:>+7.2f}% {r_s:>+7.2f}% {r_b-r_s:>+6.2f}% | "
            f"{r_e:>+7.2f}% {r_l:>+7.2f}% {r_e-r_l:>+6.2f}% | {cr_e:>6.1f}% {cr_l:>6.1f}%"
        )

    report_path = OUTPUT_DIR / "07_multi_timeframe_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\n" + "=" * 114)
    log(f"报告已保存: {report_path}")


if __name__ == "__main__":
    main()
