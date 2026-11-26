#!/usr/bin/env python3
"""
Market Cycle V2 — Study 13c: VRD 窗口 × 突破定义 网格体检（查「20 日是不是孤峰」）

严格执行 decisions.md「13c-预注册」：只做稳健性描述，不换冻结规则、不挑最优格、不做 FDR。
  - 突破定义 N ∈ {10,15,20,30,40,60}（N 日首次收盘新高，冷却固定 5 天）
  - 量幅脱节窗口 W ∈ {10,15,20,30,40,60}（VRD_pre^(W) = -SpearmanCorr_W(Δln V, TR/C_{-1})，截至 t-1）
  - 主 Y / 控制变量 / HAC 全部复用 Study 13 未改动函数
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "studies" / fname)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


s13 = _load("s13", "13_vp_exhaustion_and_breakout.py")
s13b = _load("s13b", "13b_vp_holdout_audit.py")

NS = [10, 15, 20, 30, 40, 60]
WS = [10, 15, 20, 30, 40, 60]
COOLDOWN = 5
ANCHOR = (20, 20)
NEIGH_N = [15, 20, 30]
NEIGH_W = [15, 20, 30]


def add_grid_columns(g: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Append VRD_pre_W and brk_N columns; formulas identical to Study 13 (only window/lookback vary)."""
    g = g.sort_values("date").reset_index(drop=True)
    h = g["high"].to_numpy(float)
    l = g["low"].to_numpy(float)
    c = g["close"].to_numpy(float)
    v = g["vol"].to_numpy(float)
    prev_c = np.r_[np.nan, c[:-1]]
    log_v = np.where(v > 0.0, np.log(v), np.nan)
    dlog_v = np.r_[np.nan, np.diff(log_v)]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    rtr = tr / prev_c

    out = base.copy()
    for W in WS:
        vrd = -s13.rolling_spearman_1d(dlog_v, rtr, window=W)
        out[f"VRD_pre_{W}"] = np.r_[np.nan, vrd[:-1]]

    s_c = pd.Series(c)
    for N in NS:
        prev_hi = s_c.rolling(N, min_periods=N).max().shift(1).to_numpy()
        is_nh = c > prev_hi
        recent = pd.Series(is_nh).shift(1).rolling(COOLDOWN, min_periods=COOLDOWN).sum().to_numpy() > 0
        out[f"brk_{N}"] = is_nh & (~recent) & np.isfinite(prev_hi)
    return out


def build_grid_panel(raw: pd.DataFrame, eval_start: pd.Timestamp, market: str) -> pd.DataFrame:
    parts = []
    for _, g in raw.groupby("asset", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        base = s13.compute_asset_features_and_labels(g, market_name=market)
        parts.append(add_grid_columns(g, base))
    df = pd.concat(parts, ignore_index=True)
    df = df[df["date"] >= eval_start].reset_index(drop=True)
    # Self-check: anchor cell must reproduce Study 13 exactly
    assert (df["brk_20"] == df["brk20"]).all(), "brk_20 != Study 13 brk20"
    a, b = df["VRD_pre_20"].to_numpy(), df["VRD_pre"].to_numpy()
    both = np.isfinite(a) & np.isfinite(b)
    assert (np.isfinite(a) == np.isfinite(b)).all() and np.max(np.abs(a[both] - b[both])) < 1e-12, "VRD_pre_20 != Study 13 VRD_pre"
    return df


def run_grid(df: pd.DataFrame, add_fe: bool) -> pd.DataFrame:
    all_dates = np.sort(df["date"].unique())
    rows = []
    for N in NS:
        ev = df[df[f"brk_{N}"]]
        for W in WS:
            fc = f"VRD_pre_{W}"
            rc = s13.run_controlled_reg(ev, "crash_20", [fc], ["r20_pre", "r1"], "sigma20_pre", all_dates, add_fe=add_fe)
            rr = s13.run_controlled_reg(ev, "ret_20", [fc], ["r20_pre", "r1"], "sigma20_pre", all_dates, add_fe=add_fe)
            rows.append({"N": N, "W": W, "n": rc["n"], "g": rc["betas"][fc], "t": rc["ts"][fc],
                         "ret_g": rr["betas"][fc], "ret_t": rr["ts"][fc]})
    return pd.DataFrame(rows)


def plateau_verdict(grid: pd.DataFrame) -> dict:
    anc = grid[(grid["N"] == ANCHOR[0]) & (grid["W"] == ANCHOR[1])].iloc[0]
    nb = grid[grid["N"].isin(NEIGH_N) & grid["W"].isin(NEIGH_W)]
    others = nb[~((nb["N"] == ANCHOR[0]) & (nb["W"] == ANCHOR[1]))]
    n_pos = int((others["g"] > 0).sum())
    med = float(nb["g"].median())
    if n_pos == 8 and med >= 0.5 * anc["g"]:
        v = "平台"
    elif n_pos >= 6:
        v = "弱平台"
    else:
        v = "孤峰"
    rank = int((grid["g"] > anc["g"]).sum()) + 1
    return {"verdict": v, "anchor_g": anc["g"], "anchor_t": anc["t"], "neigh_pos": n_pos, "neigh_median": med,
            "grid_pos": float((grid["g"] > 0).mean()), "grid_t2": float((grid["t"] >= 2.0).mean()), "anchor_rank": rank}


def fmt_matrix(grid: pd.DataFrame, val: str, fmt: str) -> list[str]:
    piv = grid.pivot(index="N", columns="W", values=val)
    lines = ["      W→ " + "".join(f"{w:>9d}" for w in WS)]
    for N in NS:
        lines.append(f"  N={N:<4d} " + "".join(format(piv.loc[N, w], fmt).rjust(9) for w in WS))
    return lines


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    sw_dev_raw, _ = s13.load_sw_l1_dev()
    sw_hld_raw, _ = s13b.load_sw_l1_holdout_with_warmup()
    spy_dev_raw = s13.load_spy_dev()
    spy_full_raw = s13b.load_spy_full_for_holdout()

    segs = [
        ("A股申万L1 · 开发段（主判）", build_grid_panel(sw_dev_raw, s13.SW_DEV_START, "A股申万L1"), True),
        ("A股申万L1 · 封存段（副判，已用过）", build_grid_panel(sw_hld_raw, s13b.SW_HLD_START, "A股申万L1"), True),
        ("SPY · 开发段（参考）", build_grid_panel(spy_dev_raw, s13.SPY_DEV_START, "SPY"), False),
        ("SPY · 封存段（参考，已用过）", build_grid_panel(spy_full_raw, s13b.SPY_HLD_START, "SPY"), False),
    ]

    log("=" * 96)
    log("Study 13c: VRD 窗口 W × 突破定义 N 网格体检（主 Y = 20 日踩雷率，单位 pp/SD；L1 控制同 Study 13）")
    log(f"锚点 (N,W)={ANCHOR}；邻域 N∈{NEIGH_N} × W∈{NEIGH_W}；冷却期固定 {COOLDOWN} 天")
    log("=" * 96)

    all_rows = []
    for label, df, add_fe in segs:
        grid = run_grid(df, add_fe)
        grid.insert(0, "segment", label)
        all_rows.append(grid)
        pv = plateau_verdict(grid)
        log(f"\n>>> {label}")
        log("  踩雷 γ (pp/SD):")
        lines_g = fmt_matrix(grid, "g", "+.2f")
        for s in lines_g:
            log("  " + s)
        log("  t 值:")
        for s in fmt_matrix(grid, "t", "+.2f"):
            log("  " + s)
        log("  20 日收益 γ (%/SD):")
        for s in fmt_matrix(grid, "ret_g", "+.2f"):
            log("  " + s)
        log("  事件数 n:")
        for s in fmt_matrix(grid, "n", ".0f"):
            log("  " + s)
        log(
            f"  判定={pv['verdict']} | 锚点 γ={pv['anchor_g']:+.2f} (t={pv['anchor_t']:+.2f}), 全网格排名 {pv['anchor_rank']}/36 | "
            f"邻格为正 {pv['neigh_pos']}/8, 邻域中位 γ={pv['neigh_median']:+.2f} | "
            f"全网格 γ>0 占 {pv['grid_pos']*100:.0f}%, t≥2 占 {pv['grid_t2']*100:.0f}%"
        )

    out_csv = OUTPUT_DIR / "13c_vrd_grid.csv"
    pd.concat(all_rows, ignore_index=True).to_csv(out_csv, index=False)
    rpt = OUTPUT_DIR / "13c_vrd_grid_report.txt"
    rpt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\n报告: {rpt}\nCSV: {out_csv}")


if __name__ == "__main__":
    main()
