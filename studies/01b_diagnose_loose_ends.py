#!/usr/bin/env python3
"""
Market Cycle V2 — Study 01b: 收口诊断（把 Study 01 的 4 个遗留问号彻底查清）

只用开发段（SPX <= 2019-12-31，A 股 <= 2024-10-31），绝不碰封存区。
查清 4 个问题：
  1. A 股 V_20 后半段（2021-2024.10）熄火之谜：
     拆出「纯上涨段」、剔除 2024.09-10 极端政策月、逐年（2016..2024）摊开看，到底是真失效还是被熊市/极端月冲淡？
  2. A 股 V_20 到底是「全市场大盘情绪择时」还是「行业自身独立周期」？
     把每日行业 V_20 拆成「全市场均值 V_mkt」+「行业偏离 V_idio」，并对照中证全指（000985.CSI）单指数结果。
  3. SPX 早年 open == prev_close（51.6%）是否冤杀了实体柱 B_L 或重叠度 O_L？
     查清 SPX 哪年起 open 变成真实开盘价，仅在干净 OHLC 开发段重跑 B_L 与 O_L。
  4. A 股 E_20 -> Y2（「走越直，未来越震荡」）的实战含义：
     把抽象的 Y2 翻译成未来 20/60 天的绝对涨跌幅、未来波动率、上涨段真实收益与最大回撤。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
MARKET_DATA_ROOT = Path("/home/zhujunshen/market-data")
CATALOG_DB = MARKET_DATA_ROOT / "catalog" / "metadata.sqlite"

# Reuse Study 01 loaders & econometric engines (zero duplication)
spec = importlib.util.spec_from_file_location("s01", REPO_ROOT / "studies" / "01_trend_vs_range.py")
s01 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(s01)

EPS = 1e-12


def load_index_daily_dev(ts_code: str) -> pd.DataFrame | None:
    uri = f"file:{CATALOG_DB}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        snap = pd.read_sql_query(
            """
            SELECT snapshot_id FROM data_snapshot
            WHERE (gate = 'READY' OR gate LIKE '%_READY') AND gate != 'NOT_READY'
            ORDER BY end_date DESC, created_at DESC LIMIT 1
            """,
            conn,
        )
        sid = str(snap.iloc[0]["snapshot_id"])
        man = pd.read_sql_query(
            """
            SELECT partition_key, file_path FROM table_manifest
            WHERE snapshot_id = ? AND dataset = 'index_daily'
            ORDER BY partition_key
            """,
            conn,
            params=(sid,),
        )
    finally:
        conn.close()

    if man.empty:
        return None
    dev_years = {f"year={y}" for y in range(2015, 2025)}
    files = []
    for _, row in man.iterrows():
        if row["partition_key"] in dev_years:
            p = Path(row["file_path"])
            if not p.is_absolute():
                p = MARKET_DATA_ROOT / p
            files.append(p)
    if not files:
        return None
    df = pd.concat(
        [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low", "close"]) for f in files],
        ignore_index=True,
    )
    df = df[df["ts_code"] == ts_code].copy()
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df[(df["date"] >= pd.Timestamp("2015-01-01")) & (df["date"] <= s01.SW_DEV_END)].copy()
    df = df.rename(columns={"ts_code": "asset"})
    return df[["asset", "date", "open", "high", "low", "close"]].sort_values("date").reset_index(drop=True)


def add_practical_forward_metrics(raw: pd.DataFrame, df_feat: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Add practical forward outcomes (strictly inside truncated raw): fwd_ret, abs_fwd_ret, fwd_vol, fwd_mdd (for long)."""
    out_parts = []
    for asset, g_raw in raw.groupby("asset", sort=True):
        g_raw = g_raw.sort_values("date").reset_index(drop=True)
        c = g_raw["close"].to_numpy(dtype=float)
        l = g_raw["low"].to_numpy(dtype=float)
        n = len(g_raw)
        log_c = np.log(c)
        ret1 = np.empty(n)
        ret1[0] = np.nan
        ret1[1:] = np.diff(log_c)
        s_ret1 = pd.Series(ret1)

        rec = pd.DataFrame({"asset": asset, "date": g_raw["date"]})
        for H in horizons:
            fwd_ret = (pd.Series(log_c).shift(-H) - pd.Series(log_c)).to_numpy()
            fwd_vol = s_ret1.rolling(H, min_periods=H).std(ddof=1).shift(-H).to_numpy() * np.sqrt(H)
            # Forward max drawdown from entry close C_t using daily lows over t+1..t+H
            min_future_low = pd.Series(l).iloc[::-1].rolling(H, min_periods=H).min().iloc[::-1].shift(-1).to_numpy()
            fwd_mdd = (min_future_low - c) / c  # negative number, e.g. -0.08 means -8% worst dip
            rec[f"fwd_ret_{H}"] = fwd_ret
            rec[f"abs_fwd_ret_{H}"] = np.abs(fwd_ret)
            rec[f"fwd_vol_{H}"] = fwd_vol
            rec[f"fwd_mdd_{H}"] = fwd_mdd
        out_parts.append(rec)

    extra = pd.concat(out_parts, ignore_index=True)
    return df_feat.merge(extra, on=["asset", "date"], how="left")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 92)
    log("Market Cycle V2 — Study 01b: 4 个遗留问号收口诊断（严格开发段，零触碰封存区）")
    log("=" * 92)

    spx_raw = s01.load_spx_dev()
    sw_raw, sw_sid = s01.load_sw_l1_dev()
    sw20 = s01.build_dataset(sw_raw, s01.SW_DEV_START, L=20)
    sw20 = add_practical_forward_metrics(sw_raw, sw20, [20, 60])

    # =========================================================================
    # 疑案 1：A 股 V_20 后半段（2021-2024.10）到底死没死？
    # =========================================================================
    log("\n" + "=" * 92)
    log("【疑案 1】A 股 V_20 后半段（2021-2024.10）熄火之谜：拆涨跌、剔 2024.9 政策月、逐年摊开")
    log("=" * 92)

    up_all = sw20[sw20["past_sign"] > 0].copy()
    dn_all = sw20[sw20["past_sign"] < 0].copy()

    log(f"A 股开发段上涨日占比: 前半(2016-2020)={(sw20[sw20['date']<='2020-12-31']['past_sign']>0).mean():.1%}, "
        f"后半(2021-2024.10)={(sw20[sw20['date']>='2021-01-01']['past_sign']>0).mean():.1%}")
    log()
    log("  [1A] 分时段 & 剔除 2024.09-10 极端月：V_20 控基线 (|r_20|, sigma_20) 对 Y1_20 的净效应")
    log(f"  {'样本切片':<28s} | {'全体γ':>7s} {'全体t':>6s} | {'纯上涨γ':>7s} {'纯上涨t':>6s} {'ΔR²%':>6s} | {'纯下跌γ':>7s} {'纯下跌t':>6s}")
    log("  " + "-" * 86)

    slices = [
        ("全开发段 2016-2024.10", pd.Timestamp("2016-01-01"), pd.Timestamp("2024-10-31")),
        ("前半段 2016-2020", pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31")),
        ("后半段 2021-2024.10（含9.24）", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-10-31")),
        ("后半段 2021-2024.08（剔9-10月）", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-08-31")),
        ("全段剔9-10月 2016-2024.08", pd.Timestamp("2016-01-01"), pd.Timestamp("2024-08-31")),
    ]
    for label, d0, d1 in slices:
        sub = sw20[(sw20["date"] >= d0) & (sw20["date"] <= d1)]
        r_all = s01.run_controlled_regression(sub, "Y1_20", "V", ["abs_ret_L", "ret_L", "vol_L"], lag=20, panel_dk=True)
        r_up = s01.run_controlled_regression(sub[sub["past_sign"] > 0], "Y1_20", "V", ["abs_ret_L", "vol_L"], lag=20, panel_dk=True)
        r_dn = s01.run_controlled_regression(sub[sub["past_sign"] < 0], "Y1_20", "V", ["abs_ret_L", "vol_L"], lag=20, panel_dk=True)
        log(
            f"  {label:<26s} | {r_all['gamma']:>+7.3f} {r_all['t']:>+6.2f} | "
            f"{r_up['gamma']:>+7.3f} {r_up['t']:>+6.2f} {r_up['delta_r2']*100:>5.2f}% | "
            f"{r_dn['gamma']:>+7.3f} {r_dn['t']:>+6.2f}"
        )

    log("\n  [1B] 逐年体检（2016..2024，纯上涨段 r_20 > 0）：低 V_20 vs 高 V_20 未来 20 天原始收益(%)与控基线 γ")
    log(f"  {'年份':<6s} {'样本数':>6s} | {'低V(前1/3)':>10s} {'中V(中1/3)':>10s} {'高V(后1/3)':>10s} {'高-低%':>8s} | {'控基线γ':>8s} {'DK_t':>6s}")
    log("  " + "-" * 78)
    up_all["year"] = up_all["date"].dt.year
    for yr, g in up_all.groupby("year"):
        gg = g[["date", "V", "Y1_20", "Y1_raw_20", "abs_ret_L", "vol_L"]].dropna().copy()
        if len(gg) < 100:
            continue
        gg["vbin"] = pd.qcut(gg["V"].rank(method="first"), 3, labels=["低", "中", "高"])
        m_lo = gg.loc[gg["vbin"] == "低", "Y1_raw_20"].mean() * 100
        m_md = gg.loc[gg["vbin"] == "中", "Y1_raw_20"].mean() * 100
        m_hi = gg.loc[gg["vbin"] == "高", "Y1_raw_20"].mean() * 100
        reg = s01.run_controlled_regression(gg, "Y1_20", "V", ["abs_ret_L", "vol_L"], lag=20, panel_dk=True)
        log(
            f"  {yr:<6d} {len(gg):>6d} | {m_lo:>+9.2f}% {m_md:>+9.2f}% {m_hi:>+9.2f}% {m_hi-m_lo:>+7.2f}% | "
            f"{reg['gamma']:>+8.3f} {reg['t']:>+6.2f}"
        )

    # =========================================================================
    # 疑案 2：A 股 V_20 到底是「全市场大盘情绪」还是「行业自身周期」？
    # =========================================================================
    log("\n" + "=" * 92)
    log("【疑案 2】A 股 V_20 拆解：全市场均值 V_mkt vs 行业独有偏离 V_idio（外加宽基指数单验）")
    log("=" * 92)

    mkt_v = sw20.groupby("date")["V"].transform("mean")
    sw20["V_mkt"] = mkt_v
    sw20["V_idio"] = sw20["V"] - mkt_v

    log("  [2A] 面板拆解回归（同时放进 V_mkt 和 V_idio，控制 |r_20|, r_20, sigma_20）")
    for label, sub_df, base_cols in [
        ("全体样本 (Y1_20)", sw20, ["abs_ret_L", "ret_L", "vol_L"]),
        ("纯上涨段 (Y1_20)", sw20[sw20["past_sign"] > 0], ["abs_ret_L", "vol_L"]),
        ("全体样本 (Y2_60 未来效率)", sw20, ["abs_ret_L", "ret_L", "vol_L"]),
    ]:
        y_col = "Y2_60" if "Y2_60" in label else "Y1_20"
        lag = 60 if "Y2_60" in label else 20
        cols = ["date", y_col, "V_mkt", "V_idio"] + base_cols
        sub = sub_df[cols].dropna()
        y = sub[y_col].to_numpy(dtype=float)
        X_raw = sub[base_cols + ["V_mkt", "V_idio"]].to_numpy(dtype=float)
        X_z = (X_raw - X_raw.mean(axis=0)) / np.where(X_raw.std(axis=0) > EPS, X_raw.std(axis=0), 1.0)
        X = np.hstack([np.ones((len(sub), 1)), X_z])
        res = s01.ols_hac(y, X, dates=sub["date"].to_numpy(), lag=lag, panel_dk=True)
        log(
            f"  {label:<24s} | 大盘共同 V_mkt: γ={res['beta'][-2]:>+7.4f}, DK_t={res['t'][-2]:>+6.2f} | "
            f"行业独有 V_idio: γ={res['beta'][-1]:>+7.4f}, DK_t={res['t'][-1]:>+6.2f}"
        )

    log("\n  [2B] 宽基单指数直验（中证全指 000985.CSI & 沪深300 000300.SH，开发段 2016-2024.10，NW 标准误）")
    for code, cname in [("000985.CSI", "中证全指"), ("000300.SH", "沪深300")]:
        idx_raw = load_index_daily_dev(code)
        if idx_raw is None or idx_raw.empty:
            log(f"  {cname} ({code}): 无数据")
            continue
        idx20 = s01.build_dataset(idx_raw, s01.SW_DEV_START, L=20)
        r_all20 = s01.run_controlled_regression(idx20, "Y1_20", "V", ["abs_ret_L", "ret_L", "vol_L"], lag=20, panel_dk=False)
        r_up20 = s01.run_controlled_regression(idx20[idx20["past_sign"] > 0], "Y1_20", "V", ["abs_ret_L", "vol_L"], lag=20, panel_dk=False)
        r_y2_60 = s01.run_controlled_regression(idx20, "Y2_60", "V", ["abs_ret_L", "ret_L", "vol_L"], lag=60, panel_dk=False)
        # Also split 2016-2020 vs 2021-2024.08 on pure up days
        up_1 = idx20[(idx20["past_sign"] > 0) & (idx20["date"] <= "2020-12-31")]
        up_2 = idx20[(idx20["past_sign"] > 0) & (idx20["date"] >= "2021-01-01") & (idx20["date"] <= "2024-08-31")]
        r_u1 = s01.run_controlled_regression(up_1, "Y1_20", "V", ["abs_ret_L", "vol_L"], lag=20, panel_dk=False)
        r_u2 = s01.run_controlled_regression(up_2, "Y1_20", "V", ["abs_ret_L", "vol_L"], lag=20, panel_dk=False)
        log(
            f"  {cname:<6s} | 全体Y1_20: γ={r_all20['gamma']:+.3f}(t={r_all20['t']:+.2f}) | "
            f"上涨Y1_20: γ={r_up20['gamma']:+.3f}(t={r_up20['t']:+.2f}, 前半t={r_u1['t']:+.2f}, 后半剔9月t={r_u2['t']:+.2f}) | "
            f"Y2_60: γ={r_y2_60['gamma']:+.4f}(t={r_y2_60['t']:+.2f})"
        )

    # =========================================================================
    # 疑案 3：SPX 早年 open == prev_close（51.6%）是否冤杀了 B_L / O_L？
    # =========================================================================
    log("\n" + "=" * 92)
    log("【疑案 3】SPX 历史 open 口径体检：查清哪年起 open 干净，并在干净开发段重验 B_L 与 O_L")
    log("=" * 92)

    spx_chk = spx_raw[spx_raw["date"] >= s01.SPX_DEV_START].copy()
    spx_chk["prev_c"] = spx_chk["close"].shift(1)
    spx_chk["o_eq_pc"] = (spx_chk["open"] - spx_chk["prev_c"]).abs() <= 1e-6
    spx_chk["year"] = spx_chk["date"].dt.year
    yr_stat = spx_chk.groupby("year")["o_eq_pc"].mean()
    # Print compact decade/transition summary
    dirty_yrs = yr_stat[yr_stat > 0.10]
    clean_yrs = yr_stat[yr_stat <= 0.10]
    clean_start_yr = int(clean_yrs.index.min()) if len(clean_yrs) else 2020
    log(f"  O==前C > 10% 的年份: {list(dirty_yrs.index)} (最高占比 {dirty_yrs.max():.1%})")
    log(f"  O==前C <= 10% 的年份: {clean_start_yr} ~ {int(clean_yrs.index.max())} (期间平均占比 {clean_yrs.mean():.2%})")

    # Check exact transition years around clean_start_yr
    trans = yr_stat.loc[max(1984, clean_start_yr - 3): min(2019, clean_start_yr + 3)]
    log("  切换期逐年 O==前C 占比: " + ", ".join(f"{y}:{v:.1%}" for y, v in trans.items()))

    spx20 = s01.build_dataset(spx_raw, s01.SPX_DEV_START, L=20)
    spx_clean = spx20[spx20["date"] >= pd.Timestamp(f"{clean_start_yr}-01-01")].copy()
    log(f"\n  在 SPX 干净开发段 ({clean_start_yr}-01-01 ~ 2019-12-31, n={len(spx_clean)}) 重验 B_20 与 O_20（控基线）：")
    log(f"  {'特征':<14s} {'裁判':<6s} {'H':>2s} | {'全体γ':>8s} {'全体t':>6s} {'ΔR²%':>6s} | {'上涨段t':>7s} {'下跌段t':>7s}")
    log("  " + "-" * 72)
    for feat in ["B", "O", "E", "V"]:
        for ref in ["Y1", "Y2"]:
            for H in [20, 60]:
                y_col = f"{ref}_{H}"
                r_all = s01.run_controlled_regression(spx_clean, y_col, feat, ["abs_ret_L", "ret_L", "vol_L"], lag=H, panel_dk=False)
                if ref == "Y1":
                    r_up = s01.run_controlled_regression(spx_clean[spx_clean["past_sign"] > 0], y_col, feat, ["abs_ret_L", "vol_L"], lag=H, panel_dk=False)
                    r_dn = s01.run_controlled_regression(spx_clean[spx_clean["past_sign"] < 0], y_col, feat, ["abs_ret_L", "vol_L"], lag=H, panel_dk=False)
                    up_s, dn_s = f"{r_up['t']:>+7.2f}", f"{r_dn['t']:>+7.2f}"
                else:
                    up_s, dn_s = "     --", "     --"
                log(
                    f"  {s01.FEATURE_LABELS[feat]:<12s} {ref:<6s} {H:>2d} | "
                    f"{r_all['gamma']:>+8.4f} {r_all['t']:>+6.2f} {r_all['delta_r2']*100:>5.2f}% | {up_s} {dn_s}"
                )

    # =========================================================================
    # 疑案 4：A 股 E_20 -> Y2（「走越直，未来越震荡」）对实战到底意味着什么？
    # =========================================================================
    log("\n" + "=" * 92)
    log("【疑案 4】A 股 E_20（走得直不直）与 V_20（波动高潮）对实战落袋指标的真实影响（控基线后）")
    log("（翻译 Y2：看未来 H 天绝对幅度 |fwd_ret|、未来波动 fwd_vol、多头收益 fwd_ret 与最大回撤 fwd_mdd）")
    log("=" * 92)

    log("  [4A] 全体样本：E_20 与 V_20 控基线 (|r_20|, r_20, sigma_20) 对未来绝对幅度与未来波动率的影响")
    for feat in ["E", "V"]:
        for H in [20, 60]:
            r_abs = s01.run_controlled_regression(sw20, f"abs_fwd_ret_{H}", feat, ["abs_ret_L", "ret_L", "vol_L"], lag=H, panel_dk=True)
            r_vol = s01.run_controlled_regression(sw20, f"fwd_vol_{H}", feat, ["abs_ret_L", "ret_L", "vol_L"], lag=H, panel_dk=True)
            r_y2 = s01.run_controlled_regression(sw20, f"Y2_{H}", feat, ["abs_ret_L", "ret_L", "vol_L"], lag=H, panel_dk=True)
            log(
                f"  {s01.FEATURE_LABELS[feat]:<12s} H={H:>2d} | "
                f"未来效率Y2: γ={r_y2['gamma']:+.4f}(t={r_y2['t']:+.2f}) | "
                f"未来净幅度|r|: γ={r_abs['gamma']*100:+.2f}%(t={r_abs['t']:+.2f}) | "
                f"未来总波动σ: γ={r_vol['gamma']*100:+.2f}%(t={r_vol['t']:+.2f})"
            )

    log("\n  [4B] 纯上涨段 (r_20 > 0)：五分位看 E_20 与 V_20 对应的未来 20 天「真实收益 / 最大回撤 / 收益回撤比」")
    up_sw = sw20[sw20["past_sign"] > 0].copy()
    for feat in ["E", "V"]:
        sub = up_sw[["date", feat, "fwd_ret_20", "fwd_mdd_20", "fwd_vol_20", "abs_ret_L", "vol_L"]].dropna().copy()
        sub["q"] = pd.qcut(sub[feat].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        log(f"\n  -- {s01.FEATURE_LABELS[feat]}（A 股纯上涨段，未来 20 天做多持有）--")
        for q in range(1, 6):
            g = sub[sub["q"] == q]
            ret_m = g["fwd_ret_20"].mean() * 100
            mdd_m = g["fwd_mdd_20"].mean() * 100
            vol_m = g["fwd_vol_20"].mean() * 100
            win = (g["fwd_ret_20"] > 0).mean() * 100
            log(f"     Q{q}: 平均收益={ret_m:+6.2f}%, 胜率={win:4.1f}%, 平均最大回撤={mdd_m:6.2f}%, 未来20日波动={vol_m:5.2f}%")
        # Controlled regressions on fwd_ret_20 and fwd_mdd_20
        r_ret = s01.run_controlled_regression(sub, "fwd_ret_20", feat, ["abs_ret_L", "vol_L"], lag=20, panel_dk=True)
        r_mdd = s01.run_controlled_regression(sub, "fwd_mdd_20", feat, ["abs_ret_L", "vol_L"], lag=20, panel_dk=True)
        log(
            f"     控基线净效应: 未来收益 γ={r_ret['gamma']*100:+.2f}% (t={r_ret['t']:+.2f}), "
            f"未来最大回撤 γ={r_mdd['gamma']*100:+.2f}% (t={r_mdd['t']:+.2f}, 负=回撤更深)"
        )

    out_path = OUTPUT_DIR / "01b_diagnose_loose_ends.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\n" + "=" * 92)
    log(f"诊断报告已保存: {out_path}")


if __name__ == "__main__":
    main()
