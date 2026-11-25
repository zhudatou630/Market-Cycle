#!/usr/bin/env python3
"""
Market Cycle V2 — Study 08d（第二阶段方向 D · 封存区终审）：5 条冻结规则在全 A 个股封存区的一次性盲测

冻结记录见 decisions.md「08d-冻结」节（开封前写入）。本脚本：
  1. 用 08 的清洗代码（只改日期常量）构建封存区面板：预热 2023-01-01 起，入池/基准只在 2024-11-01 → 2026-11-16；
  2. 用 08b/08c 的事件、裁判、双胞胎代码，分别在开发段面板与封存区面板上跑「同一个审计函数」，并排对照；
  3. 按冻结判定（同号且 |t|≥2 强复现 / 同号弱复现 / 异号未复现 / 异号且 |t|≥2 反转）给出终审。
"""

from __future__ import annotations

import gc
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
HOLD_PANEL = OUTPUT_DIR / "08d_stock_panel_holdout.parquet"
REPORT_PATH = OUTPUT_DIR / "08d_stock_holdout_report.txt"

HOLD_START = pd.Timestamp("2024-11-01")
HOLD_END_STR = "20261116"
DEV_START = pd.Timestamp("2016-01-01")


def _load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "studies" / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


s08 = _load_mod("s08h", "08_stock_clean_panel_and_baselines.py")
c8 = _load_mod("c8", "08c_stock_index_clues_retest.py")
b8 = c8.b8
log, vs_ctrl, vs_twins, sel = b8.log, b8.vs_ctrl, b8.vs_twins, c8.sel


# ---------------------------------------------------------------------------
# 1. Holdout panel (same cleaning code, only date constants changed)
# ---------------------------------------------------------------------------
def build_holdout_panel() -> dict:
    s08.LOAD_YEARS = [2023, 2024, 2025, 2026]
    s08.WARMUP_START = "20230101"
    s08.DEV_START = HOLD_START
    s08.DEV_END_STR = HOLD_END_STR
    sid, man = s08.latest_ready_manifest()
    sd, mdates, info = s08.build_panel(man)
    fw = s08.build_forward(sd, mdates)
    for c in fw.columns:
        sd[c] = fw[c].to_numpy()
    del fw
    s08.add_excess(sd)
    assert sd["trade_date"].max() <= HOLD_END_STR
    assert not sd.loc[sd["eligible"], "trade_date"].lt("20241101").any()
    keep = (
        ["ts_code", "trade_date", "code_id", "mi", "tier", "szq", "l1", "lim_pct", "is_st", "eligible", "entry_ok",
         "up_close", "open_up", "dn_close", "a_open", "a_high", "a_low", "a_close", "close", "amount",
         "turnover_rate_f", "circ_mv", "ret1"]
        + s08.BASE_COLS
        + [f"ret_{H}" for H in s08.HORIZONS]
        + [f"ex_{b}_{H}" for b in ["sz", "mkt", "tier", "ind"] for H in s08.HORIZONS]
        + ["mdd_20", "mfe_20", "mdd_40", "mfe_40", "crash_20", "tail_20", "big_20", "big_40", "exit_frozen_20"]
    )
    sd[keep].to_parquet(HOLD_PANEL, index=False)
    el = sd[sd["eligible"] & sd["ret_20"].notna()]
    d = el.groupby("trade_date")[["ret_20", "crash_20"]].mean()
    return {
        "sid": sid, "days": int(sd.loc[sd["eligible"], "trade_date"].nunique()),
        "y_days": int(len(d)), "y_first": d.index.min(), "y_last": d.index.max(),
        "elig_per_day": float(sd["eligible"].sum() / max(sd.loc[sd["eligible"], "trade_date"].nunique(), 1)),
        "ret20": float(d["ret_20"].mean() * 100), "crash20": float(d["crash_20"].mean() * 100),
        "frozen": float(el["exit_frozen_20"].mean() * 100),
    }


# ---------------------------------------------------------------------------
# 2. One audit function, run identically on dev and holdout
# ---------------------------------------------------------------------------
def audit(st, eval_start: pd.Timestamp, halves: list[tuple[str, str]]) -> dict:
    pn = st.pn
    pos = pn.groupby("code_id").cumcount().to_numpy()
    sev: dict[str, list[np.ndarray]] = {}
    for s, e in st.slices:
        for k, v in c8.detect_08c(st.AH[s:e], st.AL[s:e], st.AC[s:e], st.dates[s:e], eval_start).items():
            sev.setdefault(k, []).append(v + s)
    sev = {k: np.concatenate(v) for k, v in sev.items()}
    wreg = c8.weekly_regime(pn["code_id"].to_numpy(), st.dates, st.AH, st.AL, st.AC)
    rec, pri = c8.vol_windows(pn["ret1"].to_numpy(float), pos)
    cell = np.full(len(pn), np.nan)
    prow = st.base["prow"].to_numpy()
    cell[prow] = c8.cells_by_day(st.bmi, rec[prow], pri[prow])
    td = st.base["trade_date"].to_numpy()
    tier = st.tier

    def con(ycol: str, S: np.ndarray, C: np.ndarray) -> dict:
        out = {}
        for key, (lo, hi) in [("all", ("00000000", "99999999"))] + list(zip(["h1", "h2"], halves)):
            s_ = S[(td[S] >= lo) & (td[S] <= hi)]
            c_ = C[(td[C] >= lo) & (td[C] <= hi)]
            d, t = vs_ctrl(st.R(ycol), s_, c_, st.bmi)
            out[key] = (d, t, len(s_), len(c_))
        return out

    res: dict = {}
    # R1
    Sh, Sb = st.B(st.ev["bp_h1_hold"]), st.B(st.ev["bp_h1_break"])
    res["R1_ret"] = con("L3_ret_20", Sh, Sb)
    res["R1_cr"] = con("L3_crash_20", Sh, Sb)
    # R2
    brk = st.ev["brk_20"]
    S1, S0 = st.B(sel(brk, cell == 1)), st.B(sel(brk, cell == 0))
    res["R2_ret"] = con("L3_ret_20", S1, S0)
    res["R2_cr"] = con("L3_crash_20", S1, S0)
    res["R2_cells"] = []
    for k in range(4):
        Sk = st.B(sel(brk, cell == k))
        res["R2_cells"].append((vs_twins(st.R("L3_ret_20"), Sk, st.bmi), vs_twins(st.R("L3_crash_20"), Sk, st.bmi)))
    # R3
    ind, syn = st.indep_split(st.B(brk))
    res["R3_ret"] = con("L3_ret_20", syn, ind)
    # R4
    tc, bc = st.B(st.ev["tc"]), st.B(st.ev["bc"])
    tcs, bcs = tc[np.isin(tier[tc], [3, 4])], bc[np.isin(tier[bc], [3, 4])]
    res["R4_cr"] = con("L3_crash_20", tcs, bcs)
    res["R4_ret"] = con("L3_ret_20", tcs, bcs)
    # R5
    bull = wreg == 1.0
    bull_all = st.B(np.flatnonzero(bull & st.elig))
    parts = {k: sel(sev[k], bull) for k in ["dip", "brk20", "p2"]}
    sig = st.B(np.unique(np.concatenate(list(parts.values()))))
    res["R5_ret"] = con("L3_ret_20", sig, bull_all)
    res["R5_cr"] = con("L3_crash_20", sig, bull_all)
    res["R5_parts"] = {k: (vs_ctrl(st.R("L3_ret_20"), st.B(v), bull_all, st.bmi),
                           vs_ctrl(st.R("L3_crash_20"), st.B(v), bull_all, st.bmi), len(st.B(v))) for k, v in parts.items()}
    return res


def judge(dev: tuple, hold: tuple) -> str:
    dd, ht, hd = dev[0], hold[1], hold[0]
    if not (np.isfinite(dd) and np.isfinite(hd)):
        return "样本不足"
    same = np.sign(dd) == np.sign(hd)
    if same:
        return "强复现" if abs(ht) >= 2 else "弱复现"
    return "反转" if abs(ht) >= 2 else "未复现"


MAIN = [
    ("R1", "回踩守住 vs 击穿突破位", [("R1_ret", "20天收益差(%)"), ("R1_cr", "踩雷率差(pp)")]),
    ("R2", "突破当天：刚缩波 vs 一直安静", [("R2_ret", "20天收益差(%)"), ("R2_cr", "踩雷率差(pp)")]),
    ("R3", "20日新高：随行业 vs 独立行情", [("R3_ret", "20天收益差(%)")]),
    ("R4", "小盘(中证1000+小微)窄通道 vs 宽通道", [("R4_cr", "踩雷率差(pp)")]),
    ("R5", "周线多头里等日线买点 vs 随便买", [("R5_ret", "20天收益差(%)"), ("R5_cr", "踩雷率差(pp)")]),
]
CELL = ["一直安静", "刚缩波", "刚放波", "一直躁动"]


def main() -> None:
    t0 = time.time()
    L: list[str] = []

    log("dev side (same audit code)")
    b8.EVAL_START = DEV_START
    st = b8.StockSide()
    dev = audit(st, DEV_START, [("20160101", "20201231"), ("20210101", "20241031")])
    n_dev = len(st.base)
    del st
    gc.collect()

    log("build holdout panel")
    info = build_holdout_panel()
    gc.collect()
    log("holdout side")
    b8.EVAL_START = HOLD_START
    st = b8.StockSide(panel_path=HOLD_PANEL, max_date=HOLD_END_STR)
    assert st.base["trade_date"].min() >= "20241101"
    hold = audit(st, HOLD_START, [("20241101", "20251031"), ("20251101", "20261116")])
    n_hold = len(st.base)

    L.append("=" * 118)
    L.append("Study 08d · 方向 D 封存区终审：5 条冻结规则在全 A 个股封存区一次性盲测（规则与判定见 decisions.md「08d-冻结」）")
    L.append("=" * 118)
    L.append(
        f"封存区: 入池日 {info['days']} 天（2024-11-01 → {HOLD_END_STR}），可评估信号日 {info['y_first']} → {info['y_last']}（{info['y_days']} 天）| "
        f"每天入池约 {info['elig_per_day']:.0f} 只 | 可评估股票日 {n_hold:,}（开发段 {n_dev:,}）"
    )
    L.append(
        f"封存区大环境: 入池股 20 天平均收益 {info['ret20']:+.2f}%、踩雷率 {info['crash20']:.1f}% | 出场冻结 {info['frozen']:.2f}% | 快照 {info['sid']}"
    )
    L.append("\n[主指标并排] 口径 L3（扣铁律 + 热度/市值 + 行业）；括号 t；n = 信号/对照样本数")
    L.append(f"  {'规则':<34}{'指标':<14}{'开发段 2016–2024.10':>28}{'封存区 2024.11–2026.11':>30}{'封存前半':>18}{'封存后半':>18}  判定")
    survive = {}
    for rid, title, metrics in MAIN:
        verdicts = []
        for key, lab in metrics:
            d, h = dev[key]["all"], hold[key]["all"]
            h1, h2 = hold[key]["h1"], hold[key]["h2"]
            v = judge(d, h)
            verdicts.append(v)
            L.append(
                f"  {rid} {title:<31}{lab:<14}"
                f"{d[0]:>+10.2f} (t={d[1]:+5.1f}) n={d[2]:>7,}"
                f"{h[0]:>+10.2f} (t={h[1]:+5.1f}) n={h[2]:>6,}"
                f"{h1[0]:>+9.2f} (t={h1[1]:+4.1f})"
                f"{h2[0]:>+9.2f} (t={h2[1]:+4.1f})  {v}"
            )
        ok = ("反转" not in verdicts) and any(v in ("强复现", "弱复现") for v in verdicts)
        survive[rid] = ok
        L.append(f"  {'':<36}→ {rid} {'存活' if ok else '淘汰'}（{' / '.join(verdicts)}）")

    L.append("\n[辅助] R2 突破当天四种波动状态 vs 双胞胎（L3）：20天收益差% (t) / 踩雷差pp (t)")
    for k in range(4):
        (dr, dc), (hr, hc) = dev["R2_cells"][k], hold["R2_cells"][k]
        L.append(
            f"  {CELL[k]:<8} 开发段 {dr[0]:+.2f} ({dr[1]:+.1f}) / {dc[0]:+.1f} ({dc[1]:+.1f}) n={dr[2]:,} | "
            f"封存区 {hr[0]:+.2f} ({hr[1]:+.1f}) / {hc[0]:+.1f} ({hc[1]:+.1f}) n={hr[2]:,}"
        )
    L.append("[辅助] R4 小盘窄通道 vs 宽通道 的 20 天收益差（非主指标）")
    d, h = dev["R4_ret"]["all"], hold["R4_ret"]["all"]
    L.append(f"  开发段 {d[0]:+.2f} (t={d[1]:+.1f}) | 封存区 {h[0]:+.2f} (t={h[1]:+.1f})")
    L.append("[辅助] R5 拆开看：周线多头 + 各日线买点 vs 周线多头随便买（20天收益差% (t) / 踩雷差pp (t)）")
    for k, lab in [("dip", "日线回调抬头"), ("brk20", "日线20日新高"), ("p2", "日线第2推")]:
        (dr, dc, dn), (hr, hc, hn) = dev["R5_parts"][k], hold["R5_parts"][k]
        L.append(
            f"  {lab:<8} 开发段 {dr[0]:+.2f} ({dr[1]:+.1f}) / {dc[0]:+.1f} ({dc[1]:+.1f}) n={dn:,} | "
            f"封存区 {hr[0]:+.2f} ({hr[1]:+.1f}) / {hc[0]:+.1f} ({hc[1]:+.1f}) n={hn:,}"
        )
    L.append("\n[终审] " + " | ".join(f"{rid} {'存活' if ok else '淘汰'}" for rid, ok in survive.items()))

    report = "\n".join(L)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)
    log(f"done in {time.time() - t0:.0f}s -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
