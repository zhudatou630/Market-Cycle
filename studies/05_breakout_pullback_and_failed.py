#!/usr/bin/env python3
"""
Market Cycle V2 — Study 05 (第二阶段方向 A):
突破回踩二次上车 (Breakout Pullback / High 1 & High 2) 与 假突破反向 (Failed Breakout / Spring)

严格遵循 AGENTS.md 五条铁律，严守封存边界（先截断再算未来标签，绝不碰封存区一根 K 线）：
  - SPX 开发段：1984-01-01 -> 2019-12-31（全程只用 H/L/C，不用 open，全段干净）
  - A 股 31 个申万一级行业开发段：2016-01-01 -> 2024-10-31

三道裁判考题：
  考题 1（突破回踩二次上车）：
    刚创 20/60 日首次新高（第一脚 BRK）后：
      - BP_H1      : 2~12 天内出现回调后首次抬头确认（H_t > H_{t-1} & C_t > C_{t-1}）且收盘站稳前高
      - BP_Strict  : 在 BP_H1 基础上，回调低点确实回踩前高附近 [-1.0*ATR, +0.5*ATR]
      - BP_H2      : 4~20 天内经历两段回调后的第二次抬头确认（High 2）且收盘站稳前高（及 MA20 宽版）
    傻瓜基线对比：
      1) 直接盲追第一脚 (BRK)
      2) 普通上涨趋势中的日常回调抬头 (ORD_PB)，并做控 (r20, vol20, r5) 的日历对齐 HAC 回归

  考题 2（回踩体态质量连续单挑）：
    在 BP_H1 样本内，扣掉触发日 (r20, vol20) 后单挑 4 个连续物理量：
      - D_pb   : 回踩深浅 = (min_L_pb - P_brk) / ATR20(t0-1)
      - T_pb   : 回踩耗时 = t1 - t0 (天)
      - V_pb   : 回踩期相对波幅 = mean(TR_pb) / ATR20(t0-1)（低=缩波弱空头牛旗）
      - K_trig : 确认日收盘力度 = (C - L) / (H - L)

  考题 3（假突破失败反向，双向对称）：
    - 3A 向上假突破失败 (FAIL_UP / FAIL_UP_STRONG)：
      创 20/60 日首次新高后 1~5 天内收盘跌回前高下方（强版：连突破日低点也跌破），对比普通高位收跌基线
    - 3B 向下假跌破拉回 (SPRING / SPRING_STRONG)：
      创 20/60 日首次新低后 1~5 天内收盘重新站回前低上方（强版：站回跌破日高点或光头收强），对比普通下跌反弹基线
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

EPS = 1e-12
HORIZONS = [10, 20, 40]
THRESH = {
    "A股申万L1": {"crash_20": -0.08, "big_20": 0.08, "big_40": 0.10},
    "SPX": {"crash_20": -0.05, "big_20": 0.04, "big_40": 0.06},
}

PB_FEATS = ["D_pb", "T_pb", "V_pb", "K_trig"]
PB_FEAT_LABELS = {
    "D_pb": "回踩深浅 D_pb(正=浅/负=深)",
    "T_pb": "回踩耗时 T_pb(天)",
    "V_pb": "回踩缩波度 V_pb(低=缩波)",
    "K_trig": "确认日收盘力度 K_trig",
}


# ---------------------------------------------------------------------------
# 1. Per-asset Causal Feature, Event & Forward Label Builder
# ---------------------------------------------------------------------------
def build_asset_events(
    g: pd.DataFrame,
    market_name: str,
    eval_start: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    g = g.sort_values("date").reset_index(drop=True)
    dates = g["date"].to_numpy()
    asset = str(g["asset"].iloc[0])
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

    s_c = pd.Series(c)
    s_h = pd.Series(h)
    s_l = pd.Series(l)
    s_log_c = pd.Series(log_c)
    s_ret1 = pd.Series(ret1)

    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr20 = pd.Series(tr).rolling(20, min_periods=20).mean().to_numpy()
    ma20 = s_c.rolling(20, min_periods=20).mean().to_numpy()
    r20 = s_log_c.diff(20).to_numpy()
    r5 = s_log_c.diff(5).to_numpy()
    vol20 = s_ret1.rolling(20, min_periods=20).std(ddof=1).to_numpy()

    rng = h - l
    k_cls = np.full(n, 0.5, dtype=float)
    valid_rng = rng > EPS
    k_cls[valid_rng] = np.clip((c[valid_rng] - l[valid_rng]) / rng[valid_rng], 0.0, 1.0)

    # Forward outcomes strictly inside truncated dev frame
    thr = THRESH[market_name]
    fwd: dict[str, np.ndarray] = {}
    for H in HORIZONS:
        fwd[f"ret_{H}"] = (s_log_c.shift(-H) - s_log_c).to_numpy()
        min_l = s_l.iloc[::-1].rolling(H, min_periods=H).min().iloc[::-1].shift(-1).to_numpy()
        max_h = s_h.iloc[::-1].rolling(H, min_periods=H).max().iloc[::-1].shift(-1).to_numpy()
        mdd = (min_l - c) / c
        mfe = (max_h - c) / c
        fwd[f"mdd_{H}"] = mdd
        fwd[f"mfe_{H}"] = mfe
        if H == 20:
            fwd["crash_20"] = np.where(np.isfinite(mdd), (mdd <= thr["crash_20"]).astype(float), np.nan)
            fwd["big_20"] = np.where(np.isfinite(mfe), (mfe >= thr["big_20"]).astype(float), np.nan)
        if H == 40:
            fwd["big_40"] = np.where(np.isfinite(mfe), (mfe >= thr["big_40"]).astype(float), np.nan)

    def make_row(idx: int, extra: dict | None = None) -> dict:
        row = {
            "asset": asset,
            "date": dates[idx],
            "r20": r20[idx],
            "r5": r5[idx],
            "vol20": vol20[idx],
            "K_trig": k_cls[idx],
            "ret_10": fwd["ret_10"][idx],
            "ret_20": fwd["ret_20"][idx],
            "ret_40": fwd["ret_40"][idx],
            "mdd_20": fwd["mdd_20"][idx],
            "mfe_20": fwd["mfe_20"][idx],
            "crash_20": fwd["crash_20"][idx],
            "big_20": fwd["big_20"][idx],
            "big_40": fwd["big_40"][idx],
        }
        if extra:
            row.update(extra)
        return row

    eval_mask = pd.to_datetime(dates) >= eval_start

    # Fresh upward breakouts (N=20 cool=5; N=60 cool=10) & downward breakdowns
    brk_configs = [
        ("20", 20, 5),
        ("60", 60, 10),
    ]

    results: dict[str, list[dict]] = {}
    for tag, _, _ in brk_configs:
        for k in [
            f"brk_{tag}",
            f"bp_touch_{tag}",
            f"bp_h1_{tag}",
            f"bp_strict_{tag}",
            f"bp_h2_{tag}",
            f"bp_h2_ma20_{tag}",
            f"fail_up_{tag}",
            f"fail_up_strong_{tag}",
            f"bdn_{tag}",
            f"bdn_true_{tag}",
            f"spring_{tag}",
            f"spring_strong_{tag}",
        ]:
            results[k] = []

    brk20_days = np.zeros(n, dtype=bool)
    bdn20_days = np.zeros(n, dtype=bool)

    for tag, N, cool in brk_configs:
        prev_hi = s_c.rolling(N, min_periods=N).max().shift(1).to_numpy()
        is_nh = c > prev_hi
        recent_nh = pd.Series(is_nh).shift(1).rolling(cool, min_periods=cool).sum().to_numpy() > 0
        brk = is_nh & (~recent_nh) & np.isfinite(prev_hi)

        prev_lo = s_c.rolling(N, min_periods=N).min().shift(1).to_numpy()
        is_nl = c < prev_lo
        recent_nl = pd.Series(is_nl).shift(1).rolling(cool, min_periods=cool).sum().to_numpy() > 0
        bdn = is_nl & (~recent_nl) & np.isfinite(prev_lo)

        if tag == "20":
            brk20_days = brk
            bdn20_days = bdn

        # --- Upward breakouts -> BP_H1, BP_Strict, BP_H2, FAIL_UP ---
        brk_indices = np.flatnonzero(brk)
        for t0 in brk_indices:
            p_brk = prev_hi[t0]
            atr0 = atr20[t0 - 1] if t0 >= 1 else np.nan
            if not np.isfinite(atr0) or atr0 <= EPS:
                continue

            if eval_mask[t0]:
                results[f"brk_{tag}"].append(make_row(t0, {"p_brk": p_brk}))

            # 1) Pullback & High 1 / High 2 causal forward scan (max window 20 bars)
            in_pb = False
            leg_cnt = 0
            got_touch = False
            got_h1 = False
            got_h2 = False
            got_h2_ma = False
            pb_min_l = np.inf
            pb_tr_sum = 0.0
            pb_tr_cnt = 0
            peak_c = c[t0]

            max_scan = min(n - 1, t0 + 20)
            for t in range(t0 + 1, max_scan + 1):
                dt = t - t0
                # Direct dip-buy touch near breakout level (no wait for green confirmation bar)
                if (not got_touch) and (1 <= dt <= 10):
                    if (l[t] <= p_brk + 0.3 * atr0) and (c[t] >= p_brk - 0.5 * atr0) and (c[t] < c[t - 1] or l[t] < l[t - 1]):
                        got_touch = True
                        if eval_mask[t]:
                            results[f"bp_touch_{tag}"].append(make_row(t, {"T_touch": float(dt)}))

                # Track drawdown/pullback metrics strictly up to t
                if l[t] < pb_min_l:
                    pb_min_l = l[t]

                # Pullback leg entry: lower low or lower close than prior bar (or below post-brk peak)
                is_down_bar = (l[t] < l[t - 1]) or (c[t] < c[t - 1])
                is_up_turn = (h[t] > h[t - 1]) and (c[t] > c[t - 1])

                if not in_pb:
                    if is_down_bar and (c[t] < peak_c or l[t] < l[t - 1]):
                        in_pb = True
                        pb_tr_sum += tr[t]
                        pb_tr_cnt += 1
                else:
                    if is_up_turn:
                        # Completed a pullback leg and turned up!
                        leg_cnt += 1
                        in_pb = False
                        d_pb = (pb_min_l - p_brk) / atr0
                        v_pb = (pb_tr_sum / pb_tr_cnt) / atr0 if pb_tr_cnt > 0 else np.nan

                        if leg_cnt == 1 and (2 <= dt <= 12) and (not got_h1):
                            if c[t] >= p_brk:
                                got_h1 = True
                                if eval_mask[t]:
                                    extra = {
                                        "D_pb": d_pb,
                                        "T_pb": float(dt),
                                        "V_pb": v_pb,
                                        "t0_date": dates[t0],
                                    }
                                    results[f"bp_h1_{tag}"].append(make_row(t, extra))
                                    if -1.0 <= d_pb <= 0.5:
                                        results[f"bp_strict_{tag}"].append(make_row(t, extra))
                        elif leg_cnt == 2 and (4 <= dt <= 20):
                            extra = {
                                "D_pb": d_pb,
                                "T_pb": float(dt),
                                "V_pb": v_pb,
                                "t0_date": dates[t0],
                            }
                            if (not got_h2) and (c[t] >= p_brk):
                                got_h2 = True
                                if eval_mask[t]:
                                    results[f"bp_h2_{tag}"].append(make_row(t, extra))
                            if (not got_h2_ma) and np.isfinite(ma20[t]) and (c[t] >= ma20[t]):
                                got_h2_ma = True
                                if eval_mask[t]:
                                    results[f"bp_h2_ma20_{tag}"].append(make_row(t, extra))
                    else:
                        pb_tr_sum += tr[t]
                        pb_tr_cnt += 1

                if c[t] > peak_c:
                    peak_c = c[t]

            # 2) Failed Upward Breakout (1~5 bars after t0)
            got_fail = False
            got_fail_strong = False
            max_fail = min(n - 1, t0 + 5)
            for t in range(t0 + 1, max_fail + 1):
                if (not got_fail) and (c[t] < p_brk):
                    got_fail = True
                    if eval_mask[t]:
                        results[f"fail_up_{tag}"].append(
                            make_row(t, {"dt_fail": float(t - t0), "drop_atr": (p_brk - c[t]) / atr0})
                        )
                if (not got_fail_strong) and (c[t] < min(p_brk, l[t0])):
                    got_fail_strong = True
                    if eval_mask[t]:
                        results[f"fail_up_strong_{tag}"].append(
                            make_row(t, {"dt_fail": float(t - t0), "drop_atr": (p_brk - c[t]) / atr0})
                        )

        # --- Downward breakdowns -> SPRING (Failed Downward Breakout) ---
        bdn_indices = np.flatnonzero(bdn)
        for t0 in bdn_indices:
            p_bdn = prev_lo[t0]
            atr0 = atr20[t0 - 1] if t0 >= 1 else np.nan
            if not np.isfinite(atr0) or atr0 <= EPS:
                continue

            if eval_mask[t0]:
                results[f"bdn_{tag}"].append(make_row(t0, {"p_bdn": p_bdn}))

            got_sp = False
            got_sp_strong = False
            max_sp = min(n - 1, t0 + 5)
            for t in range(t0 + 1, max_sp + 1):
                if (not got_sp) and (c[t] > p_bdn):
                    got_sp = True
                    if eval_mask[t]:
                        results[f"spring_{tag}"].append(
                            make_row(t, {"dt_spring": float(t - t0), "reclaim_atr": (c[t] - p_bdn) / atr0})
                        )
                if (not got_sp_strong) and (c[t] > p_bdn) and ((c[t] > h[t0]) or (k_cls[t] >= 0.65)):
                    got_sp_strong = True
                    if eval_mask[t]:
                        results[f"spring_strong_{tag}"].append(
                            make_row(t, {"dt_spring": float(t - t0), "reclaim_atr": (c[t] - p_bdn) / atr0})
                        )
            # If never reclaimed within 5 bars, record confirmed true breakdown at t0+5
            if (not got_sp) and (t0 + 5 < n) and eval_mask[t0 + 5]:
                results[f"bdn_true_{tag}"].append(make_row(t0 + 5))

    # --- Control Pools (Fool's Baselines) ---
    # 1) ORD_PB: Ordinary pullback turn-up in uptrend (no fresh 20d breakout in past 15 bars)
    recent_brk15 = pd.Series(brk20_days).shift(1).rolling(15, min_periods=1).sum().fillna(0).to_numpy() > 0
    recent_bdn15 = pd.Series(bdn20_days).shift(1).rolling(15, min_periods=1).sum().fillna(0).to_numpy() > 0

    ord_pb_rows: list[dict] = []
    ord_down_hi_rows: list[dict] = []
    ord_bounce_rows: list[dict] = []

    for t in range(25, n):
        if not eval_mask[t] or not np.isfinite(ma20[t]) or not np.isfinite(r20[t]):
            continue
        # Ordinary uptrend pullback turn-up
        prev_down = (l[t - 1] < l[t - 2]) or (c[t - 1] < c[t - 2])
        turn_up = (h[t] > h[t - 1]) and (c[t] > c[t - 1])
        if (c[t] > ma20[t]) and (r20[t] > 0) and prev_down and turn_up and (not recent_brk15[t]) and (not brk20_days[t]):
            ord_pb_rows.append(make_row(t))

        # Ordinary high-position down bar (for FAIL_UP control: near upper half / above MA20 and closes down)
        if (c[t - 1] > ma20[t - 1]) and (c[t] < c[t - 1]) and (not recent_brk15[t]):
            ord_down_hi_rows.append(make_row(t))

        # Ordinary oversold bounce (for SPRING control: below MA20 or r20<0, today rebounds up, not recent breakdown)
        if (r20[t] < 0) and (c[t - 1] < ma20[t - 1]) and (c[t] > c[t - 1]) and (h[t] > h[t - 1]) and (not recent_bdn15[t]) and (not bdn20_days[t]):
            ord_bounce_rows.append(make_row(t))

    results["ord_pb"] = ord_pb_rows
    results["ord_down_hi"] = ord_down_hi_rows
    results["ord_bounce"] = ord_bounce_rows

    out_dfs: dict[str, pd.DataFrame] = {}
    for k, rows in results.items():
        if rows:
            dfk = pd.DataFrame(rows).drop_duplicates(subset=["asset", "date"]).reset_index(drop=True)
        else:
            dfk = pd.DataFrame(columns=["asset", "date", "r20", "r5", "vol20", "K_trig", "ret_10", "ret_20", "ret_40", "mdd_20", "mfe_20", "crash_20", "big_20", "big_40"])
        out_dfs[k] = dfk
    return out_dfs


def assemble_market(raw: pd.DataFrame, market_name: str, eval_start: pd.Timestamp) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
    buckets: dict[str, list[pd.DataFrame]] = {}
    for _, g in raw.groupby("asset", sort=True):
        res = build_asset_events(g, market_name, eval_start)
        for k, dfk in res.items():
            buckets.setdefault(k, []).append(dfk)
    merged = {k: pd.concat(v, ignore_index=True) for k, v in buckets.items()}
    all_dates = np.sort(raw.loc[raw["date"] >= eval_start, "date"].unique())
    return merged, all_dates


# ---------------------------------------------------------------------------
# 2. Summary & Controlled Comparison Helpers (Exact Calendar HAC)
# ---------------------------------------------------------------------------
def event_stats(df: pd.DataFrame) -> dict[str, float]:
    sub = df.dropna(subset=["ret_20", "mdd_20", "mfe_20", "crash_20", "big_20"])
    if len(sub) == 0:
        return {"n": 0}
    ret10 = sub["ret_10"].dropna()
    ret40 = sub["ret_40"].dropna()
    big40 = sub["big_40"].dropna()
    mfe_m = float(sub["mfe_20"].mean())
    mdd_m = float(sub["mdd_20"].mean())
    rr = mfe_m / abs(mdd_m) if abs(mdd_m) > EPS else np.nan
    return {
        "n": int(len(sub)),
        "ret10": float(ret10.mean() * 100.0) if len(ret10) else np.nan,
        "ret20": float(sub["ret_20"].mean() * 100.0),
        "win20": float((sub["ret_20"] > 0).mean() * 100.0),
        "ret40": float(ret40.mean() * 100.0) if len(ret40) else np.nan,
        "mdd20": float(mdd_m * 100.0),
        "mfe20": float(mfe_m * 100.0),
        "rr20": float(rr),
        "crash20": float(sub["crash_20"].mean() * 100.0),
        "big20": float(sub["big_20"].mean() * 100.0),
        "big40": float(big40.mean() * 100.0) if len(big40) else np.nan,
    }


def fmt_stats(label: str, st: dict[str, float]) -> str:
    if st.get("n", 0) == 0:
        return f"  {label:<24s} | n=   0"
    return (
        f"  {label:<24s} | n={st['n']:>4d} | "
        f"10天={st['ret10']:+5.2f}% | 20天={st['ret20']:+5.2f}% (胜{st['win20']:4.1f}%) | "
        f"40天={st['ret40']:+5.2f}% | 均回撤={st['mdd20']:6.2f}% 雷={st['crash20']:4.1f}% | "
        f"大肉20/40={st['big20']:4.1f}%/{st['big40']:4.1f}% | 盈亏比={st['rr20']:4.2f}"
    )


def compare_vs_control(
    treat_df: pd.DataFrame,
    ctrl_df: pd.DataFrame,
    all_dates: np.ndarray,
    base_cols: list[str],
    y_col: str,
    lag: int,
    scale: float = 100.0,
) -> dict[str, float]:
    """
    Pool treat (D=1) and ctrl (D=0), exclude overlapping (asset, date) from ctrl,
    and run calendar-aligned HAC OLS: Y = a + b*base_z + gamma * D.
    """
    t_sub = treat_df[["asset", "date", y_col] + base_cols].dropna().copy()
    c_sub = ctrl_df[["asset", "date", y_col] + base_cols].dropna().copy()
    if len(t_sub) < 20 or len(c_sub) < 20:
        return {"raw_diff": np.nan, "raw_t": np.nan, "ctrl_diff": np.nan, "ctrl_t": np.nan, "nt": len(t_sub), "nc": len(c_sub)}

    # Remove treat dates from ctrl so the two groups are disjoint
    t_keys = set(zip(t_sub["asset"], t_sub["date"]))
    keep_c = [k not in t_keys for k in zip(c_sub["asset"], c_sub["date"])]
    c_sub = c_sub[keep_c].copy()
    if len(c_sub) < 20:
        return {"raw_diff": np.nan, "raw_t": np.nan, "ctrl_diff": np.nan, "ctrl_t": np.nan, "nt": len(t_sub), "nc": len(c_sub)}

    t_sub["D"] = 1.0
    c_sub["D"] = 0.0
    pool = pd.concat([t_sub, c_sub], ignore_index=True)

    y = pool[y_col].to_numpy(dtype=float) * scale
    d = pool["D"].to_numpy(dtype=float)
    dates = pool["date"].to_numpy()

    # Raw diff
    X_raw = np.column_stack([np.ones(len(pool)), d])
    r0 = s03.event_ols_hac(y, X_raw, dates, all_dates, lag=lag)

    # Controlled diff (standardize baseline covariates only; keep D as 0/1 dummy -> gamma is exact % diff!)
    B = pool[base_cols].to_numpy(dtype=float)
    sd = B.std(axis=0, ddof=0)
    sd = np.where(sd > EPS, sd, 1.0)
    Bz = (B - B.mean(axis=0)) / sd
    X_ctrl = np.column_stack([np.ones(len(pool)), Bz, d])
    r1 = s03.event_ols_hac(y, X_ctrl, dates, all_dates, lag=lag)

    return {
        "raw_diff": float(r0["beta"][-1]),
        "raw_t": float(r0["t"][-1]),
        "ctrl_diff": float(r1["beta"][-1]),
        "ctrl_t": float(r1["t"][-1]),
        "nt": int(len(t_sub)),
        "nc": int(len(c_sub)),
    }


def tercile_pb_table(ev: pd.DataFrame, feat: str, all_dates: np.ndarray) -> dict[str, float]:
    cols = ["date", feat, "ret_20", "ret_40", "mdd_20", "mfe_20", "crash_20", "big_20", "big_40"]
    sub = ev[cols].dropna().copy()
    if len(sub) < 45:
        return {}
    sub["b"] = pd.qcut(sub[feat].rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
    out: dict[str, float] = {"n": float(len(sub))}
    for b in [1, 2, 3]:
        g = sub[sub["b"] == b]
        out[f"B{b}_med"] = float(g[feat].median())
        out[f"B{b}_ret20"] = float(g["ret_20"].mean() * 100.0)
        out[f"B{b}_win20"] = float((g["ret_20"] > 0).mean() * 100.0)
        out[f"B{b}_mdd20"] = float(g["mdd_20"].mean() * 100.0)
        out[f"B{b}_crash20"] = float(g["crash_20"].mean() * 100.0)
        out[f"B{b}_big20"] = float(g["big_20"].mean() * 100.0)
        out[f"B{b}_rr20"] = float(g["mfe_20"].mean() / max(abs(g["mdd_20"].mean()), EPS))

    hl = sub[sub["b"].isin([1, 3])].copy()
    d_hi = (hl["b"] == 3).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(hl)), d_hi])
    dates_hl = hl["date"].to_numpy()
    for name, col, lag in [
        ("ret20", "ret_20", 20),
        ("crash20", "crash_20", 20),
        ("big20", "big_20", 20),
    ]:
        r = s03.event_ols_hac(hl[col].to_numpy(dtype=float) * 100.0, X, dates_hl, all_dates, lag=lag)
        out[f"d_{name}"] = float(r["beta"][1])
        out[f"t_{name}"] = float(r["t"][1])
    return out


# ---------------------------------------------------------------------------
# 3. Main Execution
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 108)
    log("Market Cycle V2 — Study 05: 突破回踩二次上车 (BP/H1/H2) 与 假突破反向 (Failed Breakout / Spring)")
    log("=" * 108)

    spx_raw = s01.load_spx_dev()
    sw_raw, sw_sid = s01.load_sw_l1_dev()
    log(
        f"封存边界确认: SPX 最后输入日={spx_raw['date'].max().date()} (<=2019-12-31, 全程不用open); "
        f"A股最后输入日={sw_raw['date'].max().date()} (<=2024-10-31), 快照={sw_sid}"
    )

    sw_ev, sw_dates = assemble_market(sw_raw, "A股申万L1", s01.SW_DEV_START)
    spx_ev, spx_dates = assemble_market(spx_raw, "SPX", s01.SPX_DEV_START)

    markets = [
        {
            "name": "A股申万L1",
            "ev": sw_ev,
            "all_dates": sw_dates,
            "splits": [
                ("前半 2016-2020", pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31")),
                ("后半 2021-2024.10", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-10-31")),
            ],
        },
        {
            "name": "SPX",
            "ev": spx_ev,
            "all_dates": spx_dates,
            "splits": [
                ("前半 1984-2001", pd.Timestamp("1984-01-01"), pd.Timestamp("2001-12-31")),
                ("后半 2002-2019", pd.Timestamp("2002-01-01"), pd.Timestamp("2019-12-31")),
            ],
        },
    ]

    summary_rows: list[dict] = []

    # =========================================================================
    # 考题 1：突破回踩二次上车 vs 直接追第一脚 vs 普通上涨回调抬头
    # =========================================================================
    log("\n" + "=" * 108)
    log("【考题 1】等回踩再上车（BP/High 1 & High 2）vs 盲追第一脚突破（BRK）vs 普通上升趋势回调（ORD_PB）")
    log("=" * 108)

    for m in markets:
        ev = m["ev"]
        all_dates = m["all_dates"]
        log(f"\n>>> 市场：{m['name']}")
        log("  [基线参考]")
        log(fmt_stats("普通上涨趋势回调抬头 ORD_PB", event_stats(ev["ord_pb"])))

        for tag, tlbl in [("20", "20日首次新高"), ("60", "60日首次新高")]:
            log(f"\n  --- {tlbl} 系列 ---")
            for key, lbl in [
                (f"brk_{tag}", f"第一脚直接追 (BRK{tag})"),
                (f"bp_h1_{tag}", f"首次回踩企稳 (BP_H1_{tag})"),
                (f"bp_strict_{tag}", f"精准回踩前高 (BP_Str_{tag})"),
                (f"bp_h2_{tag}", f"两段回踩站前高 (BP_H2_{tag})"),
                (f"bp_h2_ma20_{tag}", f"两段回踩站MA20 (H2_MA_{tag})"),
            ]:
                log(fmt_stats(lbl, event_stats(ev[key])))

            # Controlled HAC comparison vs BRK (first leg) and vs ORD_PB (ordinary uptrend pullback)
            log(f"  [严格照妖回归（扣掉 r20 + vol20 + r5，日历对齐 HAC）]")
            for key, lbl in [
                (f"bp_h1_{tag}", f"BP_H1_{tag}"),
                (f"bp_strict_{tag}", f"BP_Str_{tag}"),
                (f"bp_h2_{tag}", f"BP_H2_{tag}"),
                (f"bp_h2_ma20_{tag}", f"H2_MA_{tag}"),
            ]:
                # vs BRK
                c_brk_ret = compare_vs_control(ev[key], ev[f"brk_{tag}"], all_dates, ["r20", "vol20", "r5"], "ret_20", 20)
                c_brk_cr = compare_vs_control(ev[key], ev[f"brk_{tag}"], all_dates, ["r20", "vol20", "r5"], "crash_20", 20)
                c_brk_big = compare_vs_control(ev[key], ev[f"brk_{tag}"], all_dates, ["r20", "vol20", "r5"], "big_20", 20)
                # vs ORD_PB
                c_ord_ret = compare_vs_control(ev[key], ev["ord_pb"], all_dates, ["r20", "vol20", "r5"], "ret_20", 20)
                c_ord_cr = compare_vs_control(ev[key], ev["ord_pb"], all_dates, ["r20", "vol20", "r5"], "crash_20", 20)
                c_ord_big = compare_vs_control(ev[key], ev["ord_pb"], all_dates, ["r20", "vol20", "r5"], "big_20", 20)

                # Split stability vs ORD_PB
                sp_t = []
                for _, d0, d1 in m["splits"]:
                    t_s = ev[key][(ev[key]["date"] >= d0) & (ev[key]["date"] <= d1)]
                    o_s = ev["ord_pb"][(ev["ord_pb"]["date"] >= d0) & (ev["ord_pb"]["date"] <= d1)]
                    sp_t.append(compare_vs_control(t_s, o_s, all_dates, ["r20", "vol20", "r5"], "ret_20", 20)["ctrl_t"])

                log(
                    f"    {lbl:<10s} vs 第一脚BRK : 裸收差={c_brk_ret['raw_diff']:+5.2f}%(t={c_brk_ret['raw_t']:+5.2f}) | "
                    f"控基线收差={c_brk_ret['ctrl_diff']:+5.2f}%(t={c_brk_ret['ctrl_t']:+5.2f}), "
                    f"踩雷差={c_brk_cr['ctrl_diff']:+5.1f}%(t={c_brk_cr['ctrl_t']:+5.2f}), "
                    f"大肉差={c_brk_big['ctrl_diff']:+5.1f}%(t={c_brk_big['ctrl_t']:+5.2f})"
                )
                log(
                    f"    {lbl:<10s} vs 普通回调PB : 裸收差={c_ord_ret['raw_diff']:+5.2f}%(t={c_ord_ret['raw_t']:+5.2f}) | "
                    f"控基线收差={c_ord_ret['ctrl_diff']:+5.2f}%(t={c_ord_ret['ctrl_t']:+5.2f}), "
                    f"踩雷差={c_ord_cr['ctrl_diff']:+5.1f}%(t={c_ord_cr['ctrl_t']:+5.2f}), "
                    f"大肉差={c_ord_big['ctrl_diff']:+5.1f}%(t={c_ord_big['ctrl_t']:+5.2f}) | 前/后半t={sp_t[0]:+.2f}/{sp_t[1]:+.2f}"
                )
                summary_rows.append(
                    {
                        "test": "Q1_BP",
                        "market": m["name"],
                        "event": lbl,
                        "n": c_ord_ret["nt"],
                        "vs_brk_raw_ret20": c_brk_ret["raw_diff"],
                        "vs_brk_ctrl_ret20": c_brk_ret["ctrl_diff"],
                        "vs_brk_ctrl_t": c_brk_ret["ctrl_t"],
                        "vs_ord_raw_ret20": c_ord_ret["raw_diff"],
                        "vs_ord_ctrl_ret20": c_ord_ret["ctrl_diff"],
                        "vs_ord_ctrl_t": c_ord_ret["ctrl_t"],
                        "vs_ord_ctrl_crash20": c_ord_cr["ctrl_diff"],
                        "vs_ord_ctrl_crash_t": c_ord_cr["ctrl_t"],
                        "vs_ord_ctrl_big20": c_ord_big["ctrl_diff"],
                        "vs_ord_ctrl_big_t": c_ord_big["ctrl_t"],
                        "split1_t": sp_t[0],
                        "split2_t": sp_t[1],
                    }
                )

    # =========================================================================
    # 考题 2：回踩体态质量连续单挑（D_pb, T_pb, V_pb, K_trig）
    # =========================================================================
    log("\n" + "=" * 108)
    log("【考题 2】回踩体态质量单挑（在 BP_H1 样本内，扣掉触发日 r20 + vol20 后看连续特征增量）")
    log("=" * 108)

    for tag, tlbl in [("20", "20日突破首次回踩 BP_H1_20"), ("60", "60日突破首次回踩 BP_H1_60")]:
        log(f"\n>>> 考场：{tlbl}")
        for m in markets:
            ev_bp = m["ev"][f"bp_h1_{tag}"]
            all_dates = m["all_dates"]
            log(f"\n  [{m['name']}] 样本数={len(ev_bp)}")
            log(
                f"  {'特征':<24s} | {'20天收益γ%':>9s} {'t_收':>6s} | "
                f"{'20天踩雷γ%':>9s} {'t_雷':>6s} | {'20天大肉γ%':>9s} {'t_肉':>6s} | {'前半t':>6s} {'后半t':>6s}"
            )
            log("  " + "-" * 94)
            for feat in PB_FEATS:
                r_ret = s03.run_event_ctrl_reg(ev_bp, "ret_20", feat, ["r20", "vol20"], all_dates, lag=20)
                r_cr = s03.run_event_ctrl_reg(ev_bp, "crash_20", feat, ["r20", "vol20"], all_dates, lag=20)
                r_big = s03.run_event_ctrl_reg(ev_bp, "big_20", feat, ["r20", "vol20"], all_dates, lag=20)
                sp_t = []
                for _, d0, d1 in m["splits"]:
                    sub_s = ev_bp[(ev_bp["date"] >= d0) & (ev_bp["date"] <= d1)]
                    sp_t.append(s03.run_event_ctrl_reg(sub_s, "ret_20", feat, ["r20", "vol20"], all_dates, lag=20)["t"])
                log(
                    f"  {PB_FEAT_LABELS[feat]:<22s} | {r_ret['gamma']:>+9.2f} {r_ret['t']:>+6.2f} | "
                    f"{r_cr['gamma']:>+9.2f} {r_cr['t']:>+6.2f} | "
                    f"{r_big['gamma']:>+9.2f} {r_big['t']:>+6.2f} | {sp_t[0]:>+6.2f} {sp_t[1]:>+6.2f}"
                )

            # Practical 3-bucket view for D_pb and V_pb (check non-monotonicity like middle=exact pullback)
            if tag == "20":
                for feat in ["D_pb", "V_pb", "T_pb"]:
                    tb = tercile_pb_table(ev_bp, feat, all_dates)
                    if not tb:
                        continue
                    log(f"    -- {PB_FEAT_LABELS[feat]} 三档落袋 (n={int(tb['n'])}) --")
                    for b, bl in [(1, "低档"), (2, "中档"), (3, "高档")]:
                        log(
                            f"       {bl}(中位={tb[f'B{b}_med']:+.2f}): 20天={tb[f'B{b}_ret20']:+5.2f}% (胜{tb[f'B{b}_win20']:4.1f}%) | "
                            f"回撤={tb[f'B{b}_mdd20']:6.2f}% (雷{tb[f'B{b}_crash20']:4.1f}%) | "
                            f"大肉={tb[f'B{b}_big20']:4.1f}% | 盈亏比={tb[f'B{b}_rr20']:4.2f}"
                        )
                    log(
                        f"       高-低: 收益={tb['d_ret20']:+5.2f}%(t={tb['t_ret20']:+.2f}), "
                        f"踩雷={tb['d_crash20']:+5.1f}%(t={tb['t_crash20']:+.2f}), "
                        f"大肉={tb['d_big20']:+5.1f}%(t={tb['t_big20']:+.2f})"
                    )

    # =========================================================================
    # 考题 3A：向上假突破失败（Failed Upward Breakout）
    # =========================================================================
    log("\n" + "=" * 108)
    log("【考题 3A】向上假突破失败（刚创新高后 1~5 天跌回前高下方）— 真的会反向大跌吗？")
    log("=" * 108)

    for m in markets:
        ev = m["ev"]
        all_dates = m["all_dates"]
        log(f"\n>>> 市场：{m['name']}")
        log(fmt_stats("普通高位收跌基线 ORD_DOWN", event_stats(ev["ord_down_hi"])))
        for tag, tlbl in [("20", "20日"), ("60", "60日")]:
            for key, lbl in [
                (f"fail_up_{tag}", f"{tlbl}假突破跌回前高"),
                (f"fail_up_strong_{tag}", f"{tlbl}强假突破(破突破低)"),
            ]:
                log(fmt_stats(lbl, event_stats(ev[key])))
                c10 = compare_vs_control(ev[key], ev["ord_down_hi"], all_dates, ["r20", "vol20", "r5"], "ret_10", 10)
                c20 = compare_vs_control(ev[key], ev["ord_down_hi"], all_dates, ["r20", "vol20", "r5"], "ret_20", 20)
                ccr = compare_vs_control(ev[key], ev["ord_down_hi"], all_dates, ["r20", "vol20", "r5"], "crash_20", 20)
                sp_t = []
                for _, d0, d1 in m["splits"]:
                    t_s = ev[key][(ev[key]["date"] >= d0) & (ev[key]["date"] <= d1)]
                    o_s = ev["ord_down_hi"][(ev["ord_down_hi"]["date"] >= d0) & (ev["ord_down_hi"]["date"] <= d1)]
                    sp_t.append(compare_vs_control(t_s, o_s, all_dates, ["r20", "vol20", "r5"], "ret_20", 20)["ctrl_t"])
                log(
                    f"    vs 普通高位收跌(控r20/vol/r5): 10天差={c10['ctrl_diff']:+5.2f}%(t={c10['ctrl_t']:+5.2f}) | "
                    f"20天差={c20['ctrl_diff']:+5.2f}%(t={c20['ctrl_t']:+5.2f}) | "
                    f"踩雷差={ccr['ctrl_diff']:+5.1f}%(t={ccr['ctrl_t']:+5.2f}) | 前/后半t={sp_t[0]:+.2f}/{sp_t[1]:+.2f}"
                )
                summary_rows.append(
                    {
                        "test": "Q3A_FAIL_UP",
                        "market": m["name"],
                        "event": key,
                        "n": c20["nt"],
                        "vs_ord_raw_ret20": c20["raw_diff"],
                        "vs_ord_ctrl_ret20": c20["ctrl_diff"],
                        "vs_ord_ctrl_t": c20["ctrl_t"],
                        "vs_ord_ctrl_crash20": ccr["ctrl_diff"],
                        "vs_ord_ctrl_crash_t": ccr["ctrl_t"],
                        "split1_t": sp_t[0],
                        "split2_t": sp_t[1],
                    }
                )

    # =========================================================================
    # 考题 3B：向下假跌破拉回（Spring 弹簧效应做多抄底）
    # =========================================================================
    log("\n" + "=" * 108)
    log("【考题 3B】向下假跌破拉回（Spring 弹簧：刚创新低后 1~5 天拉回前低上方）vs 普通超跌反弹")
    log("=" * 108)

    for m in markets:
        ev = m["ev"]
        all_dates = m["all_dates"]
        log(f"\n>>> 市场：{m['name']}")
        log(fmt_stats("普通下跌反弹基线 ORD_BOUNCE", event_stats(ev["ord_bounce"])))
        for tag, tlbl in [("20", "20日"), ("60", "60日")]:
            log(fmt_stats(f"  (参考:{tlbl}首次新低当天)", event_stats(ev[f"bdn_{tag}"])))
            for key, lbl in [
                (f"spring_{tag}", f"{tlbl}假跌破拉回 SPRING"),
                (f"spring_strong_{tag}", f"{tlbl}强弹簧拉回 SP_STR"),
            ]:
                log(fmt_stats(lbl, event_stats(ev[key])))
                c10 = compare_vs_control(ev[key], ev["ord_bounce"], all_dates, ["r20", "vol20", "r5"], "ret_10", 10)
                c20 = compare_vs_control(ev[key], ev["ord_bounce"], all_dates, ["r20", "vol20", "r5"], "ret_20", 20)
                ccr = compare_vs_control(ev[key], ev["ord_bounce"], all_dates, ["r20", "vol20", "r5"], "crash_20", 20)
                cbig = compare_vs_control(ev[key], ev["ord_bounce"], all_dates, ["r20", "vol20", "r5"], "big_20", 20)
                sp_t = []
                for _, d0, d1 in m["splits"]:
                    t_s = ev[key][(ev[key]["date"] >= d0) & (ev[key]["date"] <= d1)]
                    o_s = ev["ord_bounce"][(ev["ord_bounce"]["date"] >= d0) & (ev["ord_bounce"]["date"] <= d1)]
                    sp_t.append(compare_vs_control(t_s, o_s, all_dates, ["r20", "vol20", "r5"], "ret_20", 20)["ctrl_t"])
                log(
                    f"    vs 普通超跌反弹(控r20/vol/r5): 10天差={c10['ctrl_diff']:+5.2f}%(t={c10['ctrl_t']:+5.2f}) | "
                    f"20天差={c20['ctrl_diff']:+5.2f}%(t={c20['ctrl_t']:+5.2f}) | "
                    f"踩雷差={ccr['ctrl_diff']:+5.1f}%(t={ccr['ctrl_t']:+5.2f}), "
                    f"大肉差={cbig['ctrl_diff']:+5.1f}%(t={cbig['ctrl_t']:+5.2f}) | 前/后半t={sp_t[0]:+.2f}/{sp_t[1]:+.2f}"
                )
                summary_rows.append(
                    {
                        "test": "Q3B_SPRING",
                        "market": m["name"],
                        "event": key,
                        "n": c20["nt"],
                        "vs_ord_raw_ret20": c20["raw_diff"],
                        "vs_ord_ctrl_ret20": c20["ctrl_diff"],
                        "vs_ord_ctrl_t": c20["ctrl_t"],
                        "vs_ord_ctrl_crash20": ccr["ctrl_diff"],
                        "vs_ord_ctrl_crash_t": ccr["ctrl_t"],
                        "vs_ord_ctrl_big20": cbig["ctrl_diff"],
                        "vs_ord_ctrl_big_t": cbig["ctrl_t"],
                        "split1_t": sp_t[0],
                        "split2_t": sp_t[1],
                    }
                )

    # =========================================================================
    # 逐年拆解：专门看 A 股核心亮点的年份一致性（避免单年行情幻觉）
    # =========================================================================
    log("\n" + "=" * 108)
    log("【A 股逐年拆解体检】核查主要候选事件在 2016-2024 每年的真实表现（防止单年份主导）")
    log("=" * 108)
    for key, lbl in [
        ("bp_h1_20", "20日首次回踩企稳 BP_H1_20 vs 普通回调 ORD_PB"),
        ("spring_20", "20日假跌破拉回 SPRING_20 vs 普通反弹 ORD_BOUNCE"),
        ("fail_up_20", "20日向上假突破 FAIL_UP_20 vs 普通高位收跌 ORD_DOWN"),
    ]:
        ctrl_key = "ord_pb" if key.startswith("bp") else ("ord_bounce" if key.startswith("spring") else "ord_down_hi")
        t_df = sw_ev[key].dropna(subset=["ret_20", "crash_20"]).copy()
        c_df = sw_ev[ctrl_key].dropna(subset=["ret_20", "crash_20"]).copy()
        t_df["year"] = pd.to_datetime(t_df["date"]).dt.year
        c_df["year"] = pd.to_datetime(c_df["date"]).dt.year
        log(f"\n  -- {lbl} --")
        log(f"  {'年份':<6s} {'事件数':>6s} | {'事件20天%':>9s} {'基线20天%':>9s} {'差%':>7s} | {'事件踩雷%':>9s} {'基线踩雷%':>9s}")
        log("  " + "-" * 72)
        for yr in sorted(t_df["year"].unique()):
            gt = t_df[t_df["year"] == yr]
            gc = c_df[c_df["year"] == yr]
            if len(gt) < 5 or len(gc) < 5:
                continue
            log(
                f"  {yr:<6d} {len(gt):>6d} | {gt['ret_20'].mean()*100:>+8.2f}% {gc['ret_20'].mean()*100:>+8.2f}% "
                f"{(gt['ret_20'].mean()-gc['ret_20'].mean())*100:>+6.2f}% | "
                f"{gt['crash_20'].mean()*100:>8.1f}% {gc['crash_20'].mean()*100:>8.1f}%"
            )

    # =========================================================================
    # 深度聚焦诊断：拆清 3 个关键机制问题
    # =========================================================================
    log("\n" + "=" * 108)
    log("【深度聚焦诊断】A. 不等阳线确认直接低吸回踩 vs 等确认 | B. 60日弹簧真伪 | C. SPX假突破分年代")
    log("=" * 108)

    log("\n[诊断 A] 回踩碰前高当天直接低吸 (BP_Touch) vs 等阳线确认 (BP_H1) vs 盲追第一脚 (BRK)")
    for m in markets:
        ev = m["ev"]
        all_dates = m["all_dates"]
        log(f"  >>> {m['name']}")
        for tag in ["20", "60"]:
            log(fmt_stats(f"  低吸回踩 BP_Touch_{tag}", event_stats(ev[f"bp_touch_{tag}"])))
            c_ord = compare_vs_control(ev[f"bp_touch_{tag}"], ev["ord_pb"], all_dates, ["r20", "vol20", "r5"], "ret_20", 20)
            c_brk = compare_vs_control(ev[f"bp_touch_{tag}"], ev[f"brk_{tag}"], all_dates, ["r20", "vol20", "r5"], "ret_20", 20)
            log(
                f"      vs 第一脚BRK(控基线): {c_brk['ctrl_diff']:+5.2f}%(t={c_brk['ctrl_t']:+5.2f}) | "
                f"vs 普通回调(控基线): {c_ord['ctrl_diff']:+5.2f}%(t={c_ord['ctrl_t']:+5.2f})"
            )

    log("\n[诊断 B] 60日假跌破强弹簧 (SP_STR_60) vs 60日真跌破未拉回 (BDN_TRUE_60) & A股逐年拆解")
    for m in markets:
        ev = m["ev"]
        all_dates = m["all_dates"]
        log(f"  >>> {m['name']}")
        log(fmt_stats("  60日假跌破强弹簧 SP_STR_60", event_stats(ev["spring_strong_60"])))
        log(fmt_stats("  60日真跌破(5日未回) BDN_TRUE", event_stats(ev["bdn_true_60"])))
        c_vs_true = compare_vs_control(ev["spring_strong_60"], ev["bdn_true_60"], all_dates, ["r20", "vol20", "r5"], "ret_20", 20)
        c_vs_true_cr = compare_vs_control(ev["spring_strong_60"], ev["bdn_true_60"], all_dates, ["r20", "vol20", "r5"], "crash_20", 20)
        log(
            f"      强弹簧 vs 真跌破(控r20/vol/r5): 20天差={c_vs_true['ctrl_diff']:+5.2f}%(t={c_vs_true['ctrl_t']:+5.2f}), "
            f"踩雷差={c_vs_true_cr['ctrl_diff']:+5.1f}%(t={c_vs_true_cr['ctrl_t']:+5.2f})"
        )

    log("\n  -- A股 60日强弹簧 SP_STR_60 逐年拆解 (vs 普通反弹 ORD_BOUNCE) --")
    t_df = sw_ev["spring_strong_60"].dropna(subset=["ret_20", "crash_20"]).copy()
    c_df = sw_ev["ord_bounce"].dropna(subset=["ret_20", "crash_20"]).copy()
    t_df["year"] = pd.to_datetime(t_df["date"]).dt.year
    c_df["year"] = pd.to_datetime(c_df["date"]).dt.year
    log(f"  {'年份':<6s} {'事件数':>6s} | {'弹簧20天%':>9s} {'基线20天%':>9s} {'差%':>7s} | {'弹簧踩雷%':>9s} {'基线踩雷%':>9s}")
    log("  " + "-" * 72)
    for yr in sorted(t_df["year"].unique()):
        gt = t_df[t_df["year"] == yr]
        gc = c_df[c_df["year"] == yr]
        if len(gt) < 5 or len(gc) < 5:
            continue
        log(
            f"  {yr:<6d} {len(gt):>6d} | {gt['ret_20'].mean()*100:>+8.2f}% {gc['ret_20'].mean()*100:>+8.2f}% "
            f"{(gt['ret_20'].mean()-gc['ret_20'].mean())*100:>+6.2f}% | "
            f"{gt['crash_20'].mean()*100:>8.1f}% {gc['crash_20'].mean()*100:>8.1f}%"
        )

    log("\n[诊断 C] SPX 20日向上假突破失败 (FAIL_UP_20) 分年代稳健性 (vs 普通高位收跌 ORD_DOWN)")
    f_spx = spx_ev["fail_up_20"].dropna(subset=["ret_20", "crash_20"]).copy()
    d_spx = spx_ev["ord_down_hi"].dropna(subset=["ret_20", "crash_20"]).copy()
    for lbl, y0, y1 in [
        ("1984-1993", 1984, 1993),
        ("1994-2001", 1994, 2001),
        ("2002-2009", 2002, 2009),
        ("2010-2019", 2010, 2019),
    ]:
        gf = f_spx[(pd.to_datetime(f_spx["date"]).dt.year >= y0) & (pd.to_datetime(f_spx["date"]).dt.year <= y1)]
        gd = d_spx[(pd.to_datetime(d_spx["date"]).dt.year >= y0) & (pd.to_datetime(d_spx["date"]).dt.year <= y1)]
        log(
            f"  {lbl} | 假突破 n={len(gf):>3d}: 20天={gf['ret_20'].mean()*100:+5.2f}%(胜{(gf['ret_20']>0).mean()*100:4.1f}%) | "
            f"普通收跌 n={len(gd):>3d}: 20天={gd['ret_20'].mean()*100:+5.2f}% | 差={(gf['ret_20'].mean()-gd['ret_20'].mean())*100:+5.2f}%"
        )

    report_path = OUTPUT_DIR / "05_breakout_pullback_and_failed_report.txt"
    csv_path = OUTPUT_DIR / "05_breakout_pullback_and_failed_summary.csv"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    log("\n" + "=" * 108)
    log(f"报告已保存: {report_path}")
    log(f"汇总 CSV 已保存: {csv_path}")


if __name__ == "__main__":
    main()
