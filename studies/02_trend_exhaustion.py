#!/usr/bin/env python3
"""
Market Cycle V2 — Study 02: 高位上涨趋势的「健康中继 vs 高潮衰竭」预警

紧扣 Study 01 + 01b 结论：
  必须把「高位上涨趋势中的高潮赶顶」和「深跌坑底的暴力救市启动」明确拆开！
严守封存边界（先截断再算未来标签，绝不碰封存区一根 K 线）：
  - SPX 开发段：1984-01-01 -> 2019-12-31（封存 2020-01-01 起；涉及 open 的 U_10 仅测 2008-2019 干净段）
  - A 股 31 个申万一级行业开发段：2016-01-01 -> 2024-10-31（封存 2024-11-01 起）

两个上涨趋势考场（严格因果）：
  1. 标准上涨段 (up_std):  C > MA60 且 r_20 > 0
  2. 高位成熟趋势 (up_high): C > MA60 且 r_20 > 0 且 r_60 > 0 且 120日价格位置 >= 60%

4 个衰竭/高潮候选物理量：
  - V_20: 振幅突放 / 情绪发烧 = ATR_rel_20 / ATR_rel_80（A股同步拆 V_mkt 与 V_idio）
  - A_5 : 末端急拉 / 加速冲刺 = r_5 / (ATR_rel_20 * sqrt(5))
  - U_10: 冲高回落 / 上影线抛压 = SMA_10((H - max(O, C)) / (H - L))
  - E_20: 走得太直 / 路径效率 = |C_t - C_{t-20}| / sum|ΔC|

3 个交易员最关心的落袋裁判（未来 H in {10, 20, 40} 天）：
  - fwd_ret_H  : 未来做多收益（及胜率、波动归一收益 fwd_norm_H）
  - fwd_mdd_H  : 未来持有期内最深回撤 (min(future low) - C_t) / C_t
  - crash_H    : 踩雷率（A 股回撤 <= -6% / -8% / -10%；SPX 回撤 <= -3.5% / -5% / -7%）

傻瓜基线（必须打败）：
  过去 20 日涨幅 r_20 + 过去 60 日涨幅 r_60 + 当前 20 日绝对波动率 vol_20
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
FEATURES = ["V20", "A5", "U10", "E20"]
FEAT_LABELS = {
    "V20": "振幅突放 V_20",
    "A5": "末端急拉 A_5",
    "U10": "上影抛压 U_10",
    "E20": "路径效率 E_20",
}
CRASH_THRESH = {
    "SPX": {10: -0.035, 20: -0.050, 40: -0.070},
    "A股申万L1": {10: -0.060, 20: -0.080, 40: -0.100},
}


def compute_exhaustion_frame(g: pd.DataFrame, market_name: str) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    o = g["open"].to_numpy(dtype=float)
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

    # Trend arena indicators (100% causal)
    ma60 = s_c.rolling(60, min_periods=60).mean().to_numpy()
    r5 = s_log_c.diff(5).to_numpy()
    r20 = s_log_c.diff(20).to_numpy()
    r60 = s_log_c.diff(60).to_numpy()
    vol20 = s_ret1.rolling(20, min_periods=20).std(ddof=1).to_numpy()

    hi120 = s_h.rolling(120, min_periods=120).max().to_numpy()
    lo120 = s_l.rolling(120, min_periods=120).min().to_numpy()
    pos120 = np.where((hi120 - lo120) > EPS, (c - lo120) / (hi120 - lo120), np.nan)

    # 1. V20 = ATR_rel_20 / ATR_rel_80
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    rtr = tr / prev_c
    atr20 = pd.Series(rtr).rolling(20, min_periods=20).mean().to_numpy()
    atr80 = pd.Series(rtr).rolling(80, min_periods=80).mean().to_numpy()
    V20 = np.where(atr80 > EPS, atr20 / atr80, np.nan)

    # 2. A5 = 5-day sprint normalized by 20-day ATR
    A5 = np.where(atr20 > EPS, r5 / (atr20 * np.sqrt(5.0)), np.nan)

    # 3. U10 = 10-day average upper wick ratio
    bar_range = h - l
    top_body = np.maximum(o, c)
    u_daily = np.zeros(n, dtype=float)
    valid_rng = bar_range > EPS
    u_daily[valid_rng] = np.clip((h[valid_rng] - top_body[valid_rng]) / bar_range[valid_rng], 0.0, 1.0)
    U10 = pd.Series(u_daily).rolling(10, min_periods=10).mean().to_numpy()

    # 4. E20 = 20-day path efficiency
    path20 = pd.Series(abs_dc).rolling(20, min_periods=20).sum().to_numpy()
    net20 = np.abs(s_c.diff(20).to_numpy())
    E20 = np.where(path20 > EPS, net20 / path20, np.nan)

    out = pd.DataFrame(
        {
            "asset": g["asset"].to_numpy(),
            "date": g["date"].to_numpy(),
            "r20": r20,
            "r60": r60,
            "vol20": vol20,
            "pos120": pos120,
            "V20": V20,
            "A5": A5,
            "U10": U10,
            "E20": E20,
            "up_std": (c > ma60) & (r20 > 0),
            "up_high": (c > ma60) & (r20 > 0) & (r60 > 0) & (pos120 >= 0.60),
        }
    )

    # Forward outcomes strictly within truncated dev period
    thresh_map = CRASH_THRESH[market_name]
    for H in HORIZONS:
        fwd_ret = (s_log_c.shift(-H) - s_log_c).to_numpy()
        denom = vol20 * np.sqrt(H)
        fwd_norm = np.where(denom > EPS, fwd_ret / denom, np.nan)
        min_low = s_l.iloc[::-1].rolling(H, min_periods=H).min().iloc[::-1].shift(-1).to_numpy()
        fwd_mdd = (min_low - c) / c
        crash = np.where(np.isfinite(fwd_mdd), (fwd_mdd <= thresh_map[H]).astype(float), np.nan)

        out[f"ret_{H}"] = fwd_ret
        out[f"norm_{H}"] = fwd_norm
        out[f"mdd_{H}"] = fwd_mdd
        out[f"crash_{H}"] = crash

    return out


def build_market_df(raw: pd.DataFrame, eval_start: pd.Timestamp, market_name: str) -> pd.DataFrame:
    parts = [compute_exhaustion_frame(g, market_name) for _, g in raw.groupby("asset", sort=True)]
    df = pd.concat(parts, ignore_index=True)
    df = df[df["date"] >= eval_start].reset_index(drop=True)
    if market_name == "A股申万L1":
        # Market-wide average V20 across all 31 industries on that day (causal)
        mkt_v = df.groupby("date")["V20"].transform("mean")
        df["V_mkt"] = mkt_v
        df["V_idio"] = df["V20"] - mkt_v
    return df


def quintile_practical(
    df: pd.DataFrame,
    feat: str,
    H: int,
    panel_dk: bool,
) -> dict[str, float]:
    cols = ["date", feat, f"ret_{H}", f"norm_{H}", f"mdd_{H}", f"crash_{H}"]
    sub = df[cols].dropna().copy()
    if len(sub) < 50:
        return {}
    if not panel_dk:
        sub = sub.sort_values("date")
    sub["q"] = pd.qcut(sub[feat].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    res: dict[str, float] = {"n": float(len(sub))}
    for q in range(1, 6):
        g = sub[sub["q"] == q]
        res[f"Q{q}_ret"] = float(g[f"ret_{H}"].mean() * 100.0)
        res[f"Q{q}_win"] = float((g[f"ret_{H}"] > 0).mean() * 100.0)
        res[f"Q{q}_mdd"] = float(g[f"mdd_{H}"].mean() * 100.0)
        res[f"Q{q}_crash"] = float(g[f"crash_{H}"].mean() * 100.0)

    # Q5 - Q1 with HAC t-stat on ret, mdd, crash
    q15 = sub[sub["q"].isin([1, 5])].copy()
    d5 = (q15["q"] == 5).to_numpy(dtype=float)
    X15 = np.column_stack([np.ones(len(q15)), d5])
    dates15 = q15["date"].to_numpy()
    for key, col, scale in [("ret", f"ret_{H}", 100.0), ("mdd", f"mdd_{H}", 100.0), ("crash", f"crash_{H}", 100.0)]:
        r = s01.ols_hac(q15[col].to_numpy(dtype=float) * scale, X15, dates=dates15, lag=H, panel_dk=panel_dk)
        res[f"d_{key}"] = float(r["beta"][1])
        res[f"t_{key}"] = float(r["t"][1])
    return res


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 96)
    log("Market Cycle V2 — Study 02: 高位上涨趋势的「健康中继 vs 高潮衰竭」预警")
    log("=" * 96)

    spx_raw = s01.load_spx_dev()
    sw_raw, sw_sid = s01.load_sw_l1_dev()
    log(f"封存边界确认: SPX 最后输入日={spx_raw['date'].max().date()} (<=2019-12-31); "
        f"A股最后输入日={sw_raw['date'].max().date()} (<=2024-10-31), 快照={sw_sid}")

    spx_df = build_market_df(spx_raw, s01.SPX_DEV_START, "SPX")
    sw_df = build_market_df(sw_raw, s01.SW_DEV_START, "A股申万L1")

    markets = [
        {
            "name": "A股申万L1",
            "df": sw_df,
            "panel_dk": True,
            "splits": [
                ("前半 2016-2020", pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31")),
                ("后半 2021-2024.10", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-10-31")),
            ],
        },
        {
            "name": "SPX",
            "df": spx_df,
            "panel_dk": False,
            "splits": [
                ("前半 1984-2001", pd.Timestamp("1984-01-01"), pd.Timestamp("2001-12-31")),
                ("后半 2002-2019", pd.Timestamp("2002-01-01"), pd.Timestamp("2019-12-31")),
            ],
        },
    ]

    base_cols = ["r20", "r60", "vol20"]
    summary_rows: list[dict] = []

    # --- 1. Arena Base Rates ---
    log("\n" + "-" * 96)
    log("【考场底数】两档上涨趋势考场的样本占比与基础做多表现（未来 H=20 天）")
    log("-" * 96)
    for m in markets:
        df = m["df"]
        c_thr = int(abs(CRASH_THRESH[m["name"]][20]) * 100)
        for arena_col, arena_label in [("up_std", "标准上涨段(C>MA60,r20>0)"), ("up_high", "高位成熟趋势(+r60>0,pos>=60%)")]:
            sub = df[df[arena_col]].dropna(subset=["ret_20", "mdd_20", "crash_20"])
            pct = len(sub) / len(df.dropna(subset=["ret_20"])) * 100
            log(
                f"  {m['name']:<8s} | {arena_label:<30s} | 样本={len(sub):>5d} ({pct:4.1f}%) | "
                f"均收益={sub['ret_20'].mean()*100:+5.2f}%, 胜率={(sub['ret_20']>0).mean()*100:4.1f}%, "
                f"均最大回撤={sub['mdd_20'].mean()*100:6.2f}%, 踩雷率(>{c_thr}%)={sub['crash_20'].mean()*100:4.1f}%"
            )

    # --- 2. Controlled Regression Summary across Both Arenas and H in {10, 20, 40} ---
    for arena_col, arena_label in [("up_std", "考场一：标准上涨段 (C>MA60 & r20>0)"), ("up_high", "考场二：高位成熟趋势 (C>MA60 & r20>0 & r60>0 & pos120>=60%)")]:
        log("\n" + "=" * 96)
        log(f"【过堂总表】{arena_label}")
        log("扣掉傻瓜基线 (r_20, r_60, vol_20) 后的净增量（特征每+1标准差对应的收益%/最大回撤%/踩雷率%变化）")
        log("=" * 96)

        for m in markets:
            df_m = m["df"]
            log(f"\n>>> 市场：{m['name']}（踩雷阈值 H=10/20/40: "
                f"{abs(CRASH_THRESH[m['name']][10])*100:.1f}%/{abs(CRASH_THRESH[m['name']][20])*100:.1f}%/{abs(CRASH_THRESH[m['name']][40])*100:.1f}%）")
            log(
                f"  {'特征':<14s} {'H':>2s} | {'收益γ%':>7s} {'t_收':>6s} | {'回撤γ%':>7s} {'t_撤':>6s} | "
                f"{'踩雷γ%':>7s} {'t_雷':>6s} | {'前半收益t':>8s} {'后半收益t':>8s} | {'前半雷t':>7s} {'后半雷t':>7s}"
            )
            log("  " + "-" * 94)

            for feat in FEATURES:
                # Enforce SPX clean-open rule for U10
                if m["name"] == "SPX" and feat == "U10":
                    df_use = df_m[df_m["date"] >= pd.Timestamp("2008-01-01")]
                    splits = [
                        ("08-13", pd.Timestamp("2008-01-01"), pd.Timestamp("2013-12-31")),
                        ("14-19", pd.Timestamp("2014-01-01"), pd.Timestamp("2019-12-31")),
                    ]
                    tag = "U_10(08+)"
                else:
                    df_use = df_m
                    splits = m["splits"]
                    tag = FEAT_LABELS[feat]

                arena_df = df_use[df_use[arena_col]].copy()

                for H in HORIZONS:
                    r_ret = s01.run_controlled_regression(arena_df, f"ret_{H}", feat, base_cols, lag=H, panel_dk=m["panel_dk"])
                    r_mdd = s01.run_controlled_regression(arena_df, f"mdd_{H}", feat, base_cols, lag=H, panel_dk=m["panel_dk"])
                    r_cr = s01.run_controlled_regression(arena_df, f"crash_{H}", feat, base_cols, lag=H, panel_dk=m["panel_dk"])

                    sp_ret_t, sp_cr_t = [], []
                    for _, d0, d1 in splits:
                        s_df = arena_df[(arena_df["date"] >= d0) & (arena_df["date"] <= d1)]
                        sp_ret_t.append(s01.run_controlled_regression(s_df, f"ret_{H}", feat, base_cols, lag=H, panel_dk=m["panel_dk"])["t"])
                        sp_cr_t.append(s01.run_controlled_regression(s_df, f"crash_{H}", feat, base_cols, lag=H, panel_dk=m["panel_dk"])["t"])

                    log(
                        f"  {tag:<12s} {H:>2d} | {r_ret['gamma']*100:>+7.2f} {r_ret['t']:>+6.2f} | "
                        f"{r_mdd['gamma']*100:>+7.2f} {r_mdd['t']:>+6.2f} | "
                        f"{r_cr['gamma']*100:>+7.2f} {r_cr['t']:>+6.2f} | "
                        f"{sp_ret_t[0]:>+8.2f} {sp_ret_t[1]:>+8.2f} | {sp_cr_t[0]:>+7.2f} {sp_cr_t[1]:>+7.2f}"
                    )

                    summary_rows.append(
                        {
                            "arena": arena_col,
                            "market": m["name"],
                            "feature": feat,
                            "H": H,
                            "ret_gamma_pct": r_ret["gamma"] * 100.0,
                            "ret_t": r_ret["t"],
                            "mdd_gamma_pct": r_mdd["gamma"] * 100.0,
                            "mdd_t": r_mdd["t"],
                            "crash_gamma_pct": r_cr["gamma"] * 100.0,
                            "crash_t": r_cr["t"],
                            "split1_ret_t": sp_ret_t[0],
                            "split2_ret_t": sp_ret_t[1],
                            "split1_crash_t": sp_cr_t[0],
                            "split2_crash_t": sp_cr_t[1],
                            "n": r_ret["n"],
                        }
                    )

    # --- 3. Practical Quintile Experience at H=20 in 高位成熟趋势 (up_high) ---
    log("\n" + "=" * 96)
    log("【实战五分位落袋表】高位成熟趋势 (up_high)，未来 20 天做多持有（Q1 最温和 vs Q5 最亢奋）")
    log("=" * 96)
    for m in markets:
        df_m = m["df"]
        c_thr = int(abs(CRASH_THRESH[m["name"]][20]) * 100)
        log(f"\n>>> {m['name']}（高位成熟趋势，踩雷 = 20 天内回撤超过 {c_thr}%）")
        for feat in FEATURES:
            df_use = df_m[df_m["date"] >= pd.Timestamp("2008-01-01")] if (m["name"] == "SPX" and feat == "U10") else df_m
            arena_df = df_use[df_use["up_high"]]
            qp = quintile_practical(arena_df, feat, H=20, panel_dk=m["panel_dk"])
            if not qp:
                continue
            lbl = "上影抛压 U_10(08+)" if (m["name"] == "SPX" and feat == "U10") else FEAT_LABELS[feat]
            log(f"\n  -- {lbl} (样本={int(qp['n'])}) --")
            for q in range(1, 6):
                log(
                    f"     Q{q}: 平均收益={qp[f'Q{q}_ret']:+6.2f}%, 胜率={qp[f'Q{q}_win']:4.1f}%, "
                    f"均最大回撤={qp[f'Q{q}_mdd']:6.2f}%, 踩雷率={qp[f'Q{q}_crash']:4.1f}%"
                )
            log(
                f"     Q5 - Q1: 收益差={qp['d_ret']:+6.2f}% (t={qp['t_ret']:+.2f}), "
                f"回撤差={qp['d_mdd']:+6.2f}% (t={qp['t_mdd']:+.2f}), "
                f"踩雷率差={qp['d_crash']:+5.1f}% (t={qp['t_crash']:+.2f})"
            )

    # --- 4. Core Mystery Check: A-Share Year-by-Year in 高位成熟趋势 (up_high) ---
    log("\n" + "=" * 96)
    log("【核心悬念揭晓】A 股踢掉「坑底救市暴涨」后（仅看高位成熟趋势 up_high），V_20 逐年体检（2016..2024）")
    log("=" * 96)
    sw_high = sw_df[sw_df["up_high"]].copy()
    sw_high["year"] = sw_high["date"].dt.year
    log(
        f"  {'年份':<6s} {'样本':>5s} | {'低V收益%':>8s} {'高V收益%':>8s} {'高-低%':>7s} | "
        f"{'低V踩雷%':>8s} {'高V踩雷%':>8s} {'高-低雷%':>8s} | {'控基线收益t':>10s}"
    )
    log("  " + "-" * 86)
    for yr, g in sw_high.groupby("year"):
        gg = g[["date", "V20", "ret_20", "mdd_20", "crash_20", "r20", "r60", "vol20"]].dropna().copy()
        if len(gg) < 60:
            log(f"  {yr:<6d} {len(gg):>5d} | (高位成熟趋势样本过少，跳过)")
            continue
        gg["vbin"] = pd.qcut(gg["V20"].rank(method="first"), 3, labels=["低", "中", "高"])
        lo = gg[gg["vbin"] == "低"]
        hi = gg[gg["vbin"] == "高"]
        r_ctrl = s01.run_controlled_regression(gg, "ret_20", "V20", base_cols, lag=20, panel_dk=True)
        log(
            f"  {yr:<6d} {len(gg):>5d} | {lo['ret_20'].mean()*100:>+7.2f}% {hi['ret_20'].mean()*100:>+7.2f}% "
            f"{(hi['ret_20'].mean()-lo['ret_20'].mean())*100:>+6.2f}% | "
            f"{lo['crash_20'].mean()*100:>7.1f}% {hi['crash_20'].mean()*100:>7.1f}% "
            f"{(hi['crash_20'].mean()-lo['crash_20'].mean())*100:>+7.1f}% | {r_ctrl['t']:>+10.2f}"
        )

    # Also check 2024.09-10 exclusion in up_high explicitly
    log("\n  [高位成熟趋势前后半段直对（A 股 V_20，未来 20 天）]")
    for lbl, d0, d1 in [
        ("全开发段 2016-2024.10", pd.Timestamp("2016-01-01"), pd.Timestamp("2024-10-31")),
        ("前半段 2016-2020", pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31")),
        ("后半段 2021-2024.10", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-10-31")),
        ("后半段 2021-2024.08(剔9-10月)", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-08-31")),
    ]:
        s_df = sw_high[(sw_high["date"] >= d0) & (sw_high["date"] <= d1)]
        rr = s01.run_controlled_regression(s_df, "ret_20", "V20", base_cols, lag=20, panel_dk=True)
        rm = s01.run_controlled_regression(s_df, "mdd_20", "V20", base_cols, lag=20, panel_dk=True)
        rc = s01.run_controlled_regression(s_df, "crash_20", "V20", base_cols, lag=20, panel_dk=True)
        log(
            f"  {lbl:<28s} (n={rr['n']:>5d}) | 收益γ={rr['gamma']*100:+.2f}%(t={rr['t']:+.2f}) | "
            f"回撤γ={rm['gamma']*100:+.2f}%(t={rm['t']:+.2f}) | 踩雷γ={rc['gamma']*100:+.2f}%(t={rc['t']:+.2f})"
        )

    # --- 5. A-Share Two-Layer Fever Matrix: 大盘发烧 V_mkt x 行业自己发烧 V_idio in up_high ---
    log("\n" + "=" * 96)
    log("【双层发烧拆解】A 股高位成熟趋势 (up_high)：大盘发烧 V_mkt × 行业自己发烧 V_idio（未来 20 天）")
    log("=" * 96)
    sub2 = sw_high[["date", "V_mkt", "V_idio", "ret_20", "mdd_20", "crash_20", "r20", "r60", "vol20"]].dropna().copy()
    # Joint controlled regression
    X_raw = sub2[base_cols + ["V_mkt", "V_idio"]].to_numpy(dtype=float)
    X_z = (X_raw - X_raw.mean(axis=0)) / np.where(X_raw.std(axis=0) > EPS, X_raw.std(axis=0), 1.0)
    X = np.hstack([np.ones((len(sub2), 1)), X_z])
    for yname, col in [("未来20天收益%", "ret_20"), ("未来20天最大回撤%", "mdd_20"), ("未来20天踩雷率%", "crash_20")]:
        res = s01.ols_hac(sub2[col].to_numpy(dtype=float) * 100.0, X, dates=sub2["date"].to_numpy(), lag=20, panel_dk=True)
        log(
            f"  {yname:<14s} | 大盘发烧 V_mkt: γ={res['beta'][-2]:>+6.2f}% (t={res['t'][-2]:>+5.2f}) | "
            f"行业自己发烧 V_idio: γ={res['beta'][-1]:>+6.2f}% (t={res['t'][-1]:>+5.2f})"
        )

    # 2x2 simple practical grid (below/above median of V_mkt and V_idio)
    sub2["mkt_hi"] = sub2["V_mkt"] >= sub2["V_mkt"].median()
    sub2["idio_hi"] = sub2["V_idio"] >= sub2["V_idio"].median()
    log("\n  [2x2 实战温度表（按中位数切「安静 / 发烧」）]")
    for mk, mk_lbl in [(False, "大盘安静"), (True, "大盘发烧")]:
        for idio, idio_lbl in [(False, "行业安静"), (True, "行业发烧")]:
            g = sub2[(sub2["mkt_hi"] == mk) & (sub2["idio_hi"] == idio)]
            log(
                f"     {mk_lbl} + {idio_lbl} (n={len(g):>4d}) | "
                f"平均收益={g['ret_20'].mean()*100:+6.2f}%, 胜率={(g['ret_20']>0).mean()*100:4.1f}%, "
                f"均最大回撤={g['mdd_20'].mean()*100:6.2f}%, 踩雷率(>8%)={g['crash_20'].mean()*100:4.1f}%"
            )

    report_path = OUTPUT_DIR / "02_trend_exhaustion_report.txt"
    csv_path = OUTPUT_DIR / "02_trend_exhaustion_summary.csv"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    log("\n" + "=" * 96)
    log(f"报告已保存: {report_path}")
    log(f"汇总 CSV 已保存: {csv_path}")


if __name__ == "__main__":
    main()
