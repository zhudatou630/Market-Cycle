#!/usr/bin/env python3
"""
Market Cycle V2 — Study 13b: 封存区一次性盲测终审（零调参）

严格执行 decisions.md「13b-冻结」：
  - 单一冻结规则：brk20（20日首次收盘新高，5日冷却）× 突破前量幅脱节度 VRD_pre
  - 直接复用 studies/13_vp_exhaustion_and_breakout.py 的全部特征、标签与回归函数（一行不改）
  - 同一审计函数并排重算开发段与封存段对照：
      * A 股申万一级（SW2021 L1）：开发 2016-01-01..2024-10-31 vs 封存 2024-11-01..2026-11-16（2023+ 仅预热）
      * 美股 SPY：开发 1994-01-01..2019-12-31 vs 封存 2020-01-01..2026-11-25（1993+ 因果总回报连续算，2020 起评估）
"""

from __future__ import annotations

import hashlib
import importlib.util
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"

S13_PATH = REPO_ROOT / "studies" / "13_vp_exhaustion_and_breakout.py"
spec = importlib.util.spec_from_file_location("s13", S13_PATH)
s13 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(s13)

SPY_HLD_START = pd.Timestamp("2020-01-01")
SW_HLD_START = pd.Timestamp("2024-11-01")
SW_HLD_WARMUP = pd.Timestamp("2023-01-01")


def load_spy_full_for_holdout() -> pd.DataFrame:
    df = pd.read_parquet(s13.SPY_PATH).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    assert df["date"].is_unique
    df = s13.build_causal_total_return_ohlc(df)
    df["asset"] = "SPY"
    df["vol"] = df["volume"].astype(float)
    df["amount"] = df["close_raw"].astype(float) * df["vol"]
    return df[["asset", "date", "open", "high", "low", "close", "vol", "amount"]].reset_index(drop=True)


def load_sw_l1_holdout_with_warmup() -> tuple[pd.DataFrame, str]:
    uri = f"file:{s13.CATALOG_DB}?mode=ro"
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

    need_years = {f"year={y}" for y in range(2023, 2027)}
    files = []
    for _, row in man.iterrows():
        if row["partition_key"] in need_years:
            p = Path(row["file_path"])
            if not p.is_absolute():
                p = s13.MARKET_DATA_ROOT / p
            files.append(p)
    assert files, "No sw_daily 2023..2026 partitions found"

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
    df = df[df["date"] >= SW_HLD_WARMUP].copy()
    df = df.rename(columns={"index_code": "asset"})
    df = df[["asset", "date", "open", "high", "low", "close", "vol", "amount"]].sort_values(["asset", "date"]).reset_index(drop=True)
    assert not df.duplicated(["asset", "date"]).any()
    return df, sid


def classify_verdict(gamma: float, tstat: float) -> str:
    """Frozen 4-tier verdict matrix (identical to 08d-冻结; dev sign is > 0)."""
    if not (np.isfinite(gamma) and np.isfinite(tstat)):
        return "样本不足"
    if gamma > 0.0 and tstat >= 2.0:
        return "强复现"
    if gamma > 0.0:
        return "弱复现"
    if gamma < 0.0 and tstat <= -2.0:
        return "反转"
    return "未复现"


def audit_one_segment(df: pd.DataFrame, market_name: str, seg_label: str, add_fe: bool) -> dict:
    all_dates = np.sort(df["date"].unique())
    ev = df[df["brk20"]].copy()

    fc = "VRD_pre"
    c_l1 = ["r20_pre", "r1"]
    c_l2 = ["r20_pre", "r1", "V_pre"]
    vcol = "sigma20_pre"

    r_l1 = s13.run_controlled_reg(ev, "crash_20", [fc], c_l1, vcol, all_dates, add_fe=add_fe)
    r_l2 = s13.run_controlled_reg(ev, "crash_20", [fc], c_l2, vcol, all_dates, add_fe=add_fe)
    r_cn = s13.run_controlled_reg(ev, "crash_norm_20_pre", [fc], c_l1, vcol, all_dates, add_fe=add_fe)
    r_ret = s13.run_controlled_reg(ev, "ret_20", [fc], c_l1, vcol, all_dates, add_fe=add_fe)
    r_mdd = s13.run_controlled_reg(ev, "mdd_20", [fc], c_l1, vcol, all_dates, add_fe=add_fe)
    r_fail = s13.run_controlled_reg(ev, "fail10_20", [fc], c_l1, vcol, all_dates, add_fe=add_fe, lag=15)
    r_fvol = s13.run_controlled_reg(ev, "fwd_vol_20", [fc], c_l1, vcol, all_dates, add_fe=add_fe)
    r_c40 = s13.run_controlled_reg(ev, "crash_40", [fc], c_l1, vcol, all_dates, add_fe=add_fe, lag=60)
    r_r40 = s13.run_controlled_reg(ev, "ret_40", [fc], c_l1, vcol, all_dates, add_fe=add_fe, lag=60)

    # Causal 250d rolling percentile 3-bin table
    sub_b = ev[["date", "VRD_pre_pct", "sigma20_pre", "crash_20", "ret_20", "mdd_20", "fail10_20"]].dropna().copy()
    sub_b["fbin"] = np.where(sub_b["VRD_pre_pct"] < 1.0 / 3.0, 1, np.where(sub_b["VRD_pre_pct"] >= 2.0 / 3.0, 3, 2))
    bins = {}
    for b in [1, 2, 3]:
        g = sub_b[sub_b["fbin"] == b]
        bins[b] = {
            "n": len(g),
            "crash": float(g["crash_20"].mean() * 100.0) if len(g) else np.nan,
            "ret": float(g["ret_20"].mean() * 100.0) if len(g) else np.nan,
            "mdd": float(g["mdd_20"].mean() * 100.0) if len(g) else np.nan,
            "fail10": float(g["fail10_20"].mean() * 100.0) if len(g) else np.nan,
        }
    vol_strat = s13.vol_stratified_causal_spread(ev, "VRD_pre_pct", vcol, y_col="crash_20")

    # A-share mkt vs idio decomposition
    mkt_idio = None
    if market_name == "A股申万L1":
        rm = s13.run_controlled_reg(ev, "crash_20", ["VRD_pre_mkt", "VRD_pre_idio"], c_l1, vcol, all_dates, add_fe=True)
        rm_ret = s13.run_controlled_reg(ev, "ret_20", ["VRD_pre_mkt", "VRD_pre_idio"], c_l1, vcol, all_dates, add_fe=True)
        mkt_idio = {
            "mkt_g": rm["betas"]["VRD_pre_mkt"],
            "mkt_t": rm["ts"]["VRD_pre_mkt"],
            "mkt_ret_t": rm_ret["ts"]["VRD_pre_mkt"],
            "idio_g": rm["betas"]["VRD_pre_idio"],
            "idio_t": rm["ts"]["VRD_pre_idio"],
            "idio_ret_t": rm_ret["ts"]["VRD_pre_idio"],
        }

    # Base rates
    base_sub = ev.dropna(subset=["crash_20", "ret_20", fc, vcol] + c_l1)
    return {
        "market": market_name,
        "seg": seg_label,
        "date_min": str(base_sub["date"].min().date()) if len(base_sub) else "N/A",
        "date_max": str(base_sub["date"].max().date()) if len(base_sub) else "N/A",
        "n": r_l1["n"],
        "base_crash": float(base_sub["crash_20"].mean() * 100.0) if len(base_sub) else np.nan,
        "base_ret": float(base_sub["ret_20"].mean() * 100.0) if len(base_sub) else np.nan,
        "L1_g": r_l1["betas"][fc],
        "L1_t": r_l1["ts"][fc],
        "L2_g": r_l2["betas"][fc],
        "L2_t": r_l2["ts"][fc],
        "cn_g": r_cn["betas"][fc],
        "cn_t": r_cn["ts"][fc],
        "ret_g": r_ret["betas"][fc],
        "ret_t": r_ret["ts"][fc],
        "mdd_g": r_mdd["betas"][fc],
        "mdd_t": r_mdd["ts"][fc],
        "fail_g": r_fail["betas"][fc],
        "fail_t": r_fail["ts"][fc],
        "fvol_g": r_fvol["betas"][fc],
        "fvol_t": r_fvol["ts"][fc],
        "c40_g": r_c40["betas"][fc],
        "c40_t": r_c40["ts"][fc],
        "r40_g": r_r40["betas"][fc],
        "r40_t": r_r40["ts"][fc],
        "bins": bins,
        "vol_strat_pp": vol_strat["spread_pp"],
        "mkt_idio": mkt_idio,
        "verdict": classify_verdict(r_l1["betas"][fc], r_l1["ts"][fc]),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    s13_hash = hashlib.sha256(S13_PATH.read_bytes()).hexdigest()[:16]
    log("=" * 100)
    log("Market Cycle V2 — Study 13b: 封存区一次性盲测终审（单一冻结规则：brk20 × VRD_pre）")
    log(f"复用未改动脚本 studies/13_vp_exhaustion_and_breakout.py (sha256[:16]={s13_hash})")
    log("=" * 100)

    # 1. Recompute Dev (identical to Study 13)
    sw_dev_raw, sw_sid_dev = s13.load_sw_l1_dev()
    spy_dev_raw = s13.load_spy_dev()
    sw_dev_df = s13.build_panel(sw_dev_raw, s13.SW_DEV_START, "A股申万L1", use_close_to_close_vpc=False)
    spy_dev_df = s13.build_panel(spy_dev_raw, s13.SPY_DEV_START, "SPY", use_close_to_close_vpc=False)

    # 2. Load Holdout (with causal warmup only, eval strictly >= holdout_start)
    sw_hld_raw, sw_sid_hld = load_sw_l1_holdout_with_warmup()
    spy_full_raw = load_spy_full_for_holdout()
    assert sw_sid_dev == sw_sid_hld
    sw_hld_df = s13.build_panel(sw_hld_raw, SW_HLD_START, "A股申万L1", use_close_to_close_vpc=False)
    spy_hld_df = s13.build_panel(spy_full_raw, SPY_HLD_START, "SPY", use_close_to_close_vpc=False)

    # Confirm zero leakage between dev and holdout
    assert sw_dev_df["date"].max() <= s13.SW_DEV_END
    assert sw_hld_df["date"].min() >= SW_HLD_START
    assert spy_dev_df["date"].max() <= s13.SPY_DEV_END
    assert spy_hld_df["date"].min() >= SPY_HLD_START

    # Also run SPY holdout no-quad-witching & A-share holdout amount sensitivity
    spy_hld_noqw = s13.build_panel(spy_full_raw, SPY_HLD_START, "SPY", use_close_to_close_vpc=False, exclude_quad_witch=True)
    sw_hld_amt = s13.build_panel(sw_hld_raw, SW_HLD_START, "A股申万L1", use_close_to_close_vpc=False, vol_col="amount")

    results = [
        audit_one_segment(sw_dev_df, "A股申万L1", "开发段", add_fe=True),
        audit_one_segment(sw_hld_df, "A股申万L1", "封存段", add_fe=True),
        audit_one_segment(spy_dev_df, "SPY", "开发段", add_fe=False),
        audit_one_segment(spy_hld_df, "SPY", "封存段", add_fe=False),
    ]

    log("\n【主终审对照表】唯一主指标 = L1 回归 VRD_pre 对未来 20 日固定踩雷率 crash_20（单位 pp/SD）")
    log(
        f"  {'市场':<8s} {'时段':<4s} {'区间':<23s} {'n':>4s} | {'底数雷%':>6s} {'底数收%':>7s} | "
        f"{'L1踩雷pp':>8s} {'t_L1':>6s} | {'L2踩雷pp':>8s} {'t_L2':>6s} | {'判定'}"
    )
    log("  " + "-" * 102)
    for r in results:
        span = f"{r['date_min']}~{r['date_max']}"
        vd = "(开发锚)" if r["seg"] == "开发段" else r["verdict"]
        log(
            f"  {r['market']:<8s} {r['seg']:<4s} {span:<23s} {r['n']:>4d} | "
            f"{r['base_crash']:>5.1f}% {r['base_ret']:>+6.2f}% | "
            f"{r['L1_g']:>+8.2f} {r['L1_t']:>+6.2f} | "
            f"{r['L2_g']:>+8.2f} {r['L2_t']:>+6.2f} | {vd}"
        )

    log("\n【辅报机制对照表】收益、最大回撤、10日打脸、归一踩雷、未来波动与 40 日参照")
    log(
        f"  {'市场':<8s} {'时段':<4s} | {'20日收γ%':>8s} {'t_收':>6s} | {'MDDγ%':>7s} {'t_MDD':>6s} | "
        f"{'打脸pp':>7s} {'t_脸':>6s} | {'归一雷t':>7s} {'未来波t':>7s} | {'40日雷pp(t)':>13s} {'40日收%(t)':>13s}"
    )
    log("  " + "-" * 112)
    for r in results:
        c40s = f"{r['c40_g']:+.2f}({r['c40_t']:+.2f})"
        r40s = f"{r['r40_g']:+.2f}({r['r40_t']:+.2f})"
        log(
            f"  {r['market']:<8s} {r['seg']:<4s} | {r['ret_g']:>+8.2f} {r['ret_t']:>+6.2f} | "
            f"{r['mdd_g']:>+7.2f} {r['mdd_t']:>+6.2f} | {r['fail_g']:>+7.2f} {r['fail_t']:>+6.2f} | "
            f"{r['cn_t']:>+7.2f} {r['fvol_t']:>+7.2f} | {c40s:>13s} {r40s:>13s}"
        )

    log("\n【因果 250 日分位三档实战落袋表】（低档 = 量幅同步，高档 = 量幅脱节）")
    for r in results:
        log(f"\n  >>> {r['market']} · {r['seg']} (控波组内高-低踩雷差 = {r['vol_strat_pp']:+.2f}pp)")
        for b, lbl in [(1, "低档(同步)"), (2, "中档"), (3, "高档(脱节)")]:
            d = r["bins"][b]
            log(
                f"     {lbl:<10s} (n={d['n']:>3d}): 20日踩雷={d['crash']:>5.1f}%, "
                f"20日均收益={d['ret']:>+6.2f}%, 均最大回撤={d['mdd']:>6.2f}%, 10日打脸={d['fail10']:>5.1f}%"
            )
        d1, d3 = r["bins"][1], r["bins"][3]
        log(
            f"     高 - 低: 踩雷差={d3['crash']-d1['crash']:+.1f}pp, "
            f"收益差={d3['ret']-d1['ret']:+.2f}%, 打脸差={d3['fail10']-d1['fail10']:+.1f}pp"
        )

    log("\n【A 股双层拆解对照（大盘层 VRD_pre_mkt vs 行业独立层 VRD_pre_idio）】")
    for r in results[:2]:
        mi = r["mkt_idio"]
        log(
            f"  A股申万L1 · {r['seg']}: "
            f"大盘层 踩雷={mi['mkt_g']:+.2f}pp (t={mi['mkt_t']:+.2f}, 收益t={mi['mkt_ret_t']:+.2f}) | "
            f"行业独立层 踩雷={mi['idio_g']:+.2f}pp (t={mi['idio_t']:+.2f}, 收益t={mi['idio_ret_t']:+.2f})"
        )

    log("\n【封存段敏感性核查】")
    r_sw_amt = audit_one_segment(sw_hld_amt, "A股申万L1", "封存段(amount版)", add_fe=True)
    r_spy_nq = audit_one_segment(spy_hld_noqw, "SPY", "封存段(剔四巫日)", add_fe=False)
    log(
        f"  A股申万L1 封存段 amount版 (n={r_sw_amt['n']}): "
        f"L1 踩雷={r_sw_amt['L1_g']:+.2f}pp (t={r_sw_amt['L1_t']:+.2f}), L2={r_sw_amt['L2_g']:+.2f}pp (t={r_sw_amt['L2_t']:+.2f}), 20日收益={r_sw_amt['ret_g']:+.2f}% (t={r_sw_amt['ret_t']:+.2f})"
    )
    log(
        f"  SPY 封存段 剔四巫日 (n={r_spy_nq['n']}): "
        f"L1 踩雷={r_spy_nq['L1_g']:+.2f}pp (t={r_spy_nq['L1_t']:+.2f}), L2={r_spy_nq['L2_g']:+.2f}pp (t={r_spy_nq['L2_t']:+.2f}), 20日收益={r_spy_nq['ret_g']:+.2f}% (t={r_spy_nq['ret_t']:+.2f})"
    )

    out_path = OUTPUT_DIR / "13b_vp_holdout_report.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\n" + "=" * 100)
    log(f"封存终审报告已保存: {out_path}")


if __name__ == "__main__":
    main()
