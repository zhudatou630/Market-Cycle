#!/usr/bin/env python3
"""
Market Cycle V2 — Study 04: 封存区一次性终审（零调参、冻结开发段阈值）

严守铁律 3：
  - 所有阈值（分位数、中位数）严格只在开发段计算并冻结，原样套用到从未见过的封存区！
  - 美股 SPX 封存区：2020-01-01 -> 2026-07-21（约 6.5 年）
  - A 股 31 个申万一级行业封存区：2024-11-01 -> 最新（约 2 年）

只考 Study 01-03 真正留下来的 2 条极简物理线索（对应 3 项冻结考题）：
  终审 1（双市场）：20 日首次新高「横盘起跳（低 E_pre）」vs「已直拉高（高 E_pre）」
  终审 2（A 股主考，美股对照）：高位成熟趋势「双层情绪发烧（V_mkt × V_idio / V_20）」避坑预警
  终审 3（A 股主考）：20/60 日首次新高「事前安静缩波（低 V_pre）」vs「事前放波躁动（高 V_pre）」
"""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
SPX_PATH = REPO_ROOT / "data" / "snapshots" / "spx_daily_2026-07-21_d828fbc8.parquet"
MARKET_DATA_ROOT = Path("/home/zhujunshen/market-data")
CATALOG_DB = MARKET_DATA_ROOT / "catalog" / "metadata.sqlite"

def _load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "studies" / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

s01 = _load_mod("s01", "01_trend_vs_range.py")
s02 = _load_mod("s02", "02_trend_exhaustion.py")
s03 = _load_mod("s03", "03_breakout_true_vs_false.py")

SPX_DEV_END = pd.Timestamp("2019-12-31")
SPX_HOLD_START = pd.Timestamp("2020-01-01")

SW_DEV_END = pd.Timestamp("2024-10-31")
SW_HOLD_START = pd.Timestamp("2024-11-01")
EPS = 1e-12


def load_spx_full() -> pd.DataFrame:
    df = pd.read_parquet(SPX_PATH).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[df["date"] >= pd.Timestamp("1982-01-01")].copy()
    df["asset"] = "SPX"
    return df[["asset", "date", "open", "high", "low", "close"]].reset_index(drop=True)


def load_sw_l1_full() -> tuple[pd.DataFrame, str]:
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
            WHERE snapshot_id = ? AND dataset = 'sw_daily'
            ORDER BY partition_key
            """,
            conn,
            params=(sid,),
        )
    finally:
        conn.close()

    files = []
    for _, row in man.iterrows():
        p = Path(row["file_path"])
        if not p.is_absolute():
            p = MARKET_DATA_ROOT / p
        files.append(p)
    parts = [
        pd.read_parquet(f, columns=["src", "level", "index_code", "trade_date", "open", "high", "low", "close"])
        for f in files
    ]
    df = pd.concat(parts, ignore_index=True)
    df = df[(df["src"] == "SW2021") & (df["level"] == "L1")].copy()
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df[df["date"] >= pd.Timestamp("2016-01-01")].copy()
    df = df.rename(columns={"index_code": "asset"})
    df = df[["asset", "date", "open", "high", "low", "close"]].sort_values(["asset", "date"]).reset_index(drop=True)
    assert not df.duplicated(["asset", "date"]).any()
    return df, sid


def apply_frozen_terciles(series: pd.Series, q33: float, q67: float) -> pd.Series:
    return pd.Series(
        np.where(series <= q33, 1, np.where(series <= q67, 2, 3)),
        index=series.index,
    )


def summarize_breakout_bins(
    ev: pd.DataFrame,
    feat: str,
    q33: float,
    q67: float,
    fail_col: str,
    all_dates: np.ndarray,
) -> dict[str, float]:
    cols = ["date", feat, "ret_20", "ret_40", fail_col, "mdd_20", "crash_20", "big_20", "big_40", "r20_pre", "vol20_pre"]
    sub = ev[cols].dropna().copy()
    if len(sub) < 15:
        return {"n": float(len(sub))}
    sub["b"] = apply_frozen_terciles(sub[feat], q33, q67)
    out: dict[str, float] = {"n": float(len(sub))}
    for b in [1, 2, 3]:
        g = sub[sub["b"] == b]
        out[f"B{b}_n"] = float(len(g))
        out[f"B{b}_ret20"] = float(g["ret_20"].mean() * 100.0) if len(g) else np.nan
        out[f"B{b}_win20"] = float((g["ret_20"] > 0).mean() * 100.0) if len(g) else np.nan
        out[f"B{b}_fail10"] = float(g[fail_col].mean() * 100.0) if len(g) else np.nan
        out[f"B{b}_mdd20"] = float(g["mdd_20"].mean() * 100.0) if len(g) else np.nan
        out[f"B{b}_crash20"] = float(g["crash_20"].mean() * 100.0) if len(g) else np.nan
        out[f"B{b}_big20"] = float(g["big_20"].mean() * 100.0) if len(g) else np.nan
        out[f"B{b}_big40"] = float(g["big_40"].mean() * 100.0) if len(g) else np.nan

    # High - Low HAC t-stat & controlled regression
    hl = sub[sub["b"].isin([1, 3])].copy()
    if len(hl) >= 10 and (hl["b"] == 1).sum() >= 3 and (hl["b"] == 3).sum() >= 3:
        X = np.column_stack([np.ones(len(hl)), (hl["b"] == 3).to_numpy(dtype=float)])
        dates_hl = hl["date"].to_numpy()
        for nm, col, lag in [("ret20", "ret_20", 20), ("crash20", "crash_20", 20), ("big20", "big_20", 20), ("big40", "big_40", 40)]:
            r = s03.event_ols_hac(hl[col].to_numpy(dtype=float) * 100.0, X, dates_hl, all_dates, lag=lag)
            out[f"d_{nm}"] = float(r["beta"][1])
            out[f"t_{nm}"] = float(r["t"][1])

    r_ctrl_ret = s03.run_event_ctrl_reg(sub, "ret_20", feat, ["r20_pre", "vol20_pre"], all_dates, lag=20)
    r_ctrl_big = s03.run_event_ctrl_reg(sub, "big_20", feat, ["r20_pre", "vol20_pre"], all_dates, lag=20)
    r_ctrl_cr = s03.run_event_ctrl_reg(sub, "crash_20", feat, ["r20_pre", "vol20_pre"], all_dates, lag=20)
    out["ctrl_ret_g"] = r_ctrl_ret["gamma"]
    out["ctrl_ret_t"] = r_ctrl_ret["t"]
    out["ctrl_big_g"] = r_ctrl_big["gamma"]
    out["ctrl_big_t"] = r_ctrl_big["t"]
    out["ctrl_cr_g"] = r_ctrl_cr["gamma"]
    out["ctrl_cr_t"] = r_ctrl_cr["t"]
    return out


def print_brk_block(log, title: str, tb: dict[str, float]) -> None:
    log(f"  [{title}] (总次数={int(tb.get('n', 0))})")
    if tb.get("n", 0) < 15:
        log("     样本不足")
        return
    for b, lbl in [(1, "低档(冻结)"), (2, "中档(冻结)"), (3, "高档(冻结)")]:
        log(
            f"     {lbl} (n={int(tb[f'B{b}_n']):>4d}): "
            f"20天收益={tb[f'B{b}_ret20']:+6.2f}% (胜率{tb[f'B{b}_win20']:4.1f}%) | "
            f"10天跌回前高={tb[f'B{b}_fail10']:4.1f}% | 均回撤={tb[f'B{b}_mdd20']:6.2f}% (踩雷{tb[f'B{b}_crash20']:4.1f}%) | "
            f"大肉20/40天={tb[f'B{b}_big20']:4.1f}%/{tb[f'B{b}_big40']:4.1f}%"
        )
    if "d_ret20" in tb:
        log(
            f"     高-低差: 收益={tb['d_ret20']:+6.2f}%(t={tb['t_ret20']:+.2f}), "
            f"踩雷={tb['d_crash20']:+5.1f}%(t={tb['t_crash20']:+.2f}), "
            f"大肉20天={tb['d_big20']:+5.1f}%(t={tb['t_big20']:+.2f}), "
            f"大肉40天={tb['d_big40']:+5.1f}%(t={tb['t_big40']:+.2f})"
        )
    log(
        f"     控基线净效应: 收益γ={tb['ctrl_ret_g']:+.2f}%(t={tb['ctrl_ret_t']:+.2f}), "
        f"踩雷γ={tb['ctrl_cr_g']:+.2f}%(t={tb['ctrl_cr_t']:+.2f}), "
        f"大肉20天γ={tb['ctrl_big_g']:+.2f}%(t={tb['ctrl_big_t']:+.2f})"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 98)
    log("Market Cycle V2 — Study 04: 封存区一次性终审（开发段阈值冻结 -> 封存区盲测）")
    log("=" * 98)

    # Load full data (dev + holdout), build dev features strictly on dev-truncated raw so dev == Studies 01-03!
    spx_full_raw = load_spx_full()
    sw_full_raw, sw_sid = load_sw_l1_full()

    spx_dev_raw = spx_full_raw[spx_full_raw["date"] <= SPX_DEV_END].copy()
    sw_dev_raw = sw_full_raw[sw_full_raw["date"] <= SW_DEV_END].copy()

    log(f"A 股快照来源（只读）: {sw_sid}")
    log(f"SPX 封存区范围: {SPX_HOLD_START.date()} ~ {spx_full_raw['date'].max().date()} "
        f"(交易日={(spx_full_raw['date'] >= SPX_HOLD_START).sum()})")
    log(f"A 股封存区范围: {SW_HOLD_START.date()} ~ {sw_full_raw['date'].max().date()} "
        f"(交易日={sw_full_raw[sw_full_raw['date'] >= SW_HOLD_START]['date'].nunique()}, "
        f"行业数={sw_full_raw['asset'].nunique()})")

    # Build breakout frames (dev strictly truncated; holdout uses full raw for causal warmup, sliced to >= hold_start)
    def build_brk(raw: pd.DataFrame, start: pd.Timestamp, mname: str) -> pd.DataFrame:
        parts = [s03.compute_breakout_frame(g, mname) for _, g in raw.groupby("asset", sort=True)]
        df = pd.concat(parts, ignore_index=True)
        return df[df["date"] >= start].reset_index(drop=True)

    spx_brk_dev = build_brk(spx_dev_raw, s01.SPX_DEV_START, "SPX")
    spx_brk_hold = build_brk(spx_full_raw, SPX_HOLD_START, "SPX")
    sw_brk_dev = build_brk(sw_dev_raw, s01.SW_DEV_START, "A股申万L1")
    sw_brk_hold = build_brk(sw_full_raw, SW_HOLD_START, "A股申万L1")

    # Build exhaustion frames
    spx_exh_dev = s02.build_market_df(spx_dev_raw, s01.SPX_DEV_START, "SPX")
    spx_exh_hold = s02.build_market_df(spx_full_raw, SPX_HOLD_START, "SPX")
    sw_exh_dev = s02.build_market_df(sw_dev_raw, s01.SW_DEV_START, "A股申万L1")
    sw_exh_hold = s02.build_market_df(sw_full_raw, SW_HOLD_START, "A股申万L1")

    # =========================================================================
    # 终审 1（双市场）：20日首次新高 — 事前横盘起跳 (低 E_pre) vs 已直拉高 (高 E_pre)
    # =========================================================================
    log("\n" + "=" * 98)
    log("【终审 1（双市场）】20日首次新高：横盘起跳（低 E_pre）vs 已直拉高（高 E_pre）")
    log("开发段结论：低 E_pre 在美股（收益+大肉）与 A 股（大肉）均显著更好。看封存区能否复现！")
    log("=" * 98)

    for mname, d_dev, d_hold in [
        ("SPX", spx_brk_dev, spx_brk_hold),
        ("A股申万L1", sw_brk_dev, sw_brk_hold),
    ]:
        ev_dev = d_dev[d_dev["brk20"]].dropna(subset=["E_pre", "ret_20", "fail10_20", "big_20"]).copy()
        ev_hold = d_hold[d_hold["brk20"]].dropna(subset=["E_pre", "ret_20", "fail10_20", "big_20"]).copy()
        q33, q67 = float(ev_dev["E_pre"].quantile(1 / 3)), float(ev_dev["E_pre"].quantile(2 / 3))
        log(f"\n>>> 市场：{mname}（开发段冻结 E_pre 阈值：低档 <= {q33:.3f}，高档 > {q67:.3f}）")
        tb_dev = summarize_breakout_bins(ev_dev, "E_pre", q33, q67, "fail10_20", np.sort(d_dev["date"].unique()))
        tb_hold = summarize_breakout_bins(ev_hold, "E_pre", q33, q67, "fail10_20", np.sort(d_hold["date"].unique()))
        print_brk_block(log, "开发段参照", tb_dev)
        print_brk_block(log, "★ 封存区盲测", tb_hold)

    # =========================================================================
    # 终审 2（A 股主考，美股对照）：高位成熟趋势 — 双层情绪发烧避坑 (V_20 / V_mkt × V_idio)
    # =========================================================================
    log("\n" + "=" * 98)
    log("【终审 2（A 股主考）】高位成熟趋势 (up_high)：双层情绪发烧（大盘发烧 × 行业自己发烧）避坑")
    log("开发段结论：安静+安静踩雷仅 15.0%，发烧+发烧踩雷飙至 39.8%、均亏 -2.33%。看封存区 2024.11-2026.11！")
    log("=" * 98)

    sw_hi_dev = sw_exh_dev[sw_exh_dev["up_high"]].dropna(subset=["V20", "V_mkt", "V_idio", "ret_20", "mdd_20", "crash_20"]).copy()
    sw_hi_hold = sw_exh_hold[sw_exh_hold["up_high"]].dropna(subset=["V20", "V_mkt", "V_idio", "ret_20", "mdd_20", "crash_20"]).copy()

    # Freeze ALL cutoffs strictly from dev!
    v_q33, v_q67 = float(sw_hi_dev["V20"].quantile(1 / 3)), float(sw_hi_dev["V20"].quantile(2 / 3))
    vmkt_med = float(sw_hi_dev["V_mkt"].median())
    vidio_med = float(sw_hi_dev["V_idio"].median())
    log(f"\n>>> A股申万L1 高位成熟趋势（冻结阈值：V20 <= {v_q33:.3f} / > {v_q67:.3f}；"
        f"大盘发烧线 V_mkt >= {vmkt_med:.3f}；行业偏离线 V_idio >= {vidio_med:+.3f}）")

    for tag, df_part in [("开发段 2016-2024.10", sw_hi_dev), ("★ 封存区 2024.11-2026.11", sw_hi_hold)]:
        log(f"\n  [{tag}] (样本={len(df_part)}, 交易日数={df_part['date'].nunique()})")
        b = apply_frozen_terciles(df_part["V20"], v_q33, v_q67)
        df_tmp = df_part.assign(b=b)
        for bi, bl in [(1, "低V(安静)"), (2, "中V(常态)"), (3, "高V(发烧)")]:
            g = df_tmp[df_tmp["b"] == bi]
            if len(g) == 0:
                log(f"     {bl} (n=   0): 无样本")
                continue
            log(
                f"     {bl} (n={len(g):>5d}): 20天收益={g['ret_20'].mean()*100:+6.2f}% (胜率{(g['ret_20']>0).mean()*100:4.1f}%) | "
                f"均最大回撤={g['mdd_20'].mean()*100:6.2f}% | 踩雷率(>8%)={g['crash_20'].mean()*100:4.1f}%"
            )
        # Controlled regressions
        base_cols = ["r20", "r60", "vol20"]
        rr = s01.run_controlled_regression(df_part, "ret_20", "V20", base_cols, lag=20, panel_dk=True)
        rm = s01.run_controlled_regression(df_part, "mdd_20", "V20", base_cols, lag=20, panel_dk=True)
        rc = s01.run_controlled_regression(df_part, "crash_20", "V20", base_cols, lag=20, panel_dk=True)
        log(
            f"     控基线(r20,r60,vol20): 收益γ={rr['gamma']*100:+.2f}%(t={rr['t']:+.2f}), "
            f"回撤γ={rm['gamma']*100:+.2f}%(t={rm['t']:+.2f}), 踩雷γ={rc['gamma']*100:+.2f}%(t={rc['t']:+.2f})"
        )

        # 2x2 frozen fever grid
        log("     -- 2x2 冻结双层发烧表 --")
        mkt_hi = df_part["V_mkt"] >= vmkt_med
        idio_hi = df_part["V_idio"] >= vidio_med
        for mk, mkl in [(False, "大盘安静"), (True, "大盘发烧")]:
            for id_, idl in [(False, "行业安静"), (True, "行业发烧")]:
                g = df_part[(mkt_hi == mk) & (idio_hi == id_)]
                if len(g) == 0:
                    log(f"        {mkl} + {idl} (n=   0): 无样本")
                    continue
                log(
                    f"        {mkl} + {idl} (n={len(g):>4d}): "
                    f"收益={g['ret_20'].mean()*100:+6.2f}%, 胜率={(g['ret_20']>0).mean()*100:4.1f}%, "
                    f"均回撤={g['mdd_20'].mean()*100:6.2f}%, 踩雷={g['crash_20'].mean()*100:4.1f}%"
                )

    # Also print SPX negative control for V20 in up_high
    spx_hi_dev = spx_exh_dev[spx_exh_dev["up_high"]].dropna(subset=["V20", "ret_20", "mdd_20", "crash_20"]).copy()
    spx_hi_hold = spx_exh_hold[spx_exh_hold["up_high"]].dropna(subset=["V20", "ret_20", "mdd_20", "crash_20"]).copy()
    sv33, sv67 = float(spx_hi_dev["V20"].quantile(1 / 3)), float(spx_hi_dev["V20"].quantile(2 / 3))
    log(f"\n>>> SPX 负对照（高位成熟趋势 V_20，冻结阈值 <= {sv33:.3f} / > {sv67:.3f}，踩雷>5%）")
    for tag, df_part in [("开发段 1984-2019", spx_hi_dev), ("★ 封存区 2020-2026", spx_hi_hold)]:
        b = apply_frozen_terciles(df_part["V20"], sv33, sv67)
        df_tmp = df_part.assign(b=b)
        g1, g3 = df_tmp[df_tmp["b"] == 1], df_tmp[df_tmp["b"] == 3]
        rr = s01.run_controlled_regression(df_part, "ret_20", "V20", ["r20", "r60", "vol20"], lag=20, panel_dk=False)
        log(
            f"  [{tag}] (n={len(df_part)}) | 低V: 收益={g1['ret_20'].mean()*100:+.2f}%, 踩雷={g1['crash_20'].mean()*100:.1f}% | "
            f"高V: 收益={g3['ret_20'].mean()*100:+.2f}%, 踩雷={g3['crash_20'].mean()*100:.1f}% | 控基线收益t={rr['t']:+.2f}"
        )

    # =========================================================================
    # 终审 3（A 股主考）：20日 & 60日首次新高 — 事前缩波突破 (低 V_pre) vs 放波躁动突破 (高 V_pre)
    # =========================================================================
    log("\n" + "=" * 98)
    log("【终审 3（A 股主考）】20/60日首次新高：事前安静缩波（低 V_pre）vs 事前放波躁动（高 V_pre）")
    log("开发段结论：20/60日新高里，缩波突破均赚 +1.19%/+1.42%、踩雷仅 15-16%；放波突破反亏 -1.23%/-1.84%、踩雷 30-36%。")
    log("=" * 98)

    for brk_col, fail_col, blbl in [("brk20", "fail10_20", "20日首次新高"), ("brk60", "fail10_60", "60日首次新高")]:
        ev_dev = sw_brk_dev[sw_brk_dev[brk_col]].dropna(subset=["V_pre", "ret_20", fail_col, "big_20"]).copy()
        ev_hold = sw_brk_hold[sw_brk_hold[brk_col]].dropna(subset=["V_pre", "ret_20", fail_col, "big_20"]).copy()
        q33, q67 = float(ev_dev["V_pre"].quantile(1 / 3)), float(ev_dev["V_pre"].quantile(2 / 3))
        log(f"\n>>> A股申万L1 — {blbl}（开发段冻结 V_pre 阈值：低档 <= {q33:.3f}，高档 > {q67:.3f}）")
        tb_dev = summarize_breakout_bins(ev_dev, "V_pre", q33, q67, fail_col, np.sort(sw_brk_dev["date"].unique()))
        tb_hold = summarize_breakout_bins(ev_hold, "V_pre", q33, q67, fail_col, np.sort(sw_brk_hold["date"].unique()))
        print_brk_block(log, "开发段参照", tb_dev)
        print_brk_block(log, "★ 封存区盲测", tb_hold)

    # Also check A-share holdout regime context (so we know what kind of market 2024.11-2026.11 was!)
    log("\n" + "-" * 98)
    log("【封存区市场背景（客观记录，不做任何调参）】")
    log("-" * 98)
    for mname, df_h in [("SPX 封存区 (2020-2026)", spx_full_raw[spx_full_raw["date"] >= SPX_HOLD_START]),
                        ("A股申万L1 封存区 (2024.11-2026.11)", sw_full_raw[sw_full_raw["date"] >= SW_HOLD_START])]:
        per = df_h.groupby("asset").apply(lambda g: (g["close"].iloc[-1] / g["close"].iloc[0] - 1) * 100.0, include_groups=False)
        log(f"  {mname}: 标的期间累计涨幅中位数 = {per.median():+.1f}% (最小 {per.min():+.1f}%, 最大 {per.max():+.1f}%)")

    out_path = OUTPUT_DIR / "04_holdout_final_audit_report.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\n" + "=" * 98)
    log(f"终审报告已保存: {out_path}")


if __name__ == "__main__":
    main()
