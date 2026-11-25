#!/usr/bin/env python3
"""
Market Cycle V2 — Study 10c（第三阶段专题 2 · 探索性补测，事后、只用开发段，不改 10 / 10b 的主结论）

起因：gemini-4-neon 独立解读（docs/10_trading_range_summary.md 第八节）指出两处需要分开的替代解释，用户确认补测：
  补测 1（Q1 专属对照被截断）：下跌趋势箱体里「收盘落在下三分之一、但不低于下沿」把继续收阴的日子截掉，
    对照组多为反弹日。做法：① 在 Q1 回归里加控第 t 天单日涨跌（五分位，切点取本侧随机池）；
    ② 分层：事件组与对照组都只留「第 t 天朝边沿方向走」的日子（多头 = 当天收跌，空头镜像 = 当天收涨）。
    看「区间 − 趋势」和「相对随机」是否改变。
  补测 2（「买弱 vs 买强」去机械化）：不用括号单，同一天内比较
    A =「第 t+1 天跌破第 t 天最低价 1 跳、按该价（或更低的开盘价）买入」与
    B =「第 t+1 天越过第 t 天最高价 1 跳、按该价（或更高的开盘价）买入，涨停买不进剔除」，
    持有到第 t+1+h 天收盘（h = 0 为当天收盘的理论值，不受 T+1 约束；h = 1/3/5 可执行），毛收益；
    逐日计算两组均值之差（只 A / 只 B，外包线另计），日序列 Newey-West（lag 30）。
    另报：各组相对「当天开盘买入全部候选」的同日差；以及 10 / 10b 的括号单随机池逐日相减（去掉成交日不同的影响）。
用法：python studies/10c_trading_range_explore.py
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
REPORT_PATH = OUTPUT_DIR / "10c_trading_range_explore_report.txt"


def _load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "studies" / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


s10b = _load_mod("s10b", "10b_trading_range_stocks.py")
s10, s09b = s10b.s10, s10b.s09b
s09, b8 = s10b.s09, s10b.b8
log = b8.log
HS = (0, 1, 3, 5)
f_ = s09.fmt


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def day_diff(day: np.ndarray, y: np.ndarray, ma: np.ndarray, mb: np.ndarray, min_n: int) -> tuple[float, float, int]:
    """Mean over days of (mean y in group a − mean y in group b); NW t on the day series."""
    a = pd.DataFrame({"d": day[ma], "y": y[ma]}).groupby("d")["y"].agg(["mean", "count"])
    b = pd.DataFrame({"d": day[mb], "y": y[mb]}).groupby("d")["y"].agg(["mean", "count"])
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    j = j[(j["count_a"] >= min_n) & (j["count_b"] >= min_n)].dropna()
    if len(j) < 30:
        return np.nan, np.nan, len(j)
    d = (j["mean_a"] - j["mean_b"]).to_numpy(float)
    days = j.index.to_numpy().astype(np.int64)
    bb, t = b8.wls_dk(d, np.ones((len(d), 1)), np.ones(len(d)), days - days.min(), 30)
    return float(bb[0]), float(t[0]), len(j)


def ab_returns(P: dict, tt: np.ndarray, nn: np.ndarray) -> dict:
    """Next-day 'break below yesterday's low' (A) vs 'break above yesterday's high' (B) entries; fixed-hold closes."""
    T, N = P["C"].shape
    flat = lambda X, t: X.reshape(-1)[t * N + nn]
    Hs, Ls, tk0 = flat(P["H"], tt), flat(P["L"], tt), flat(P["TICK"], tt)
    e = tt + 1
    o, h, l, c = (flat(P[x], e) for x in ("O", "H", "L", "C"))
    tk1, up1 = flat(P["TICK"], e), flat(P["UP"], e)
    ok = flat(P["OK"], e) & np.isfinite(o)
    trd = np.isfinite(c) & np.isfinite(h) & np.isfinite(l)
    A = trd & (l < Ls - 0.5 * tk1)
    pa = np.where(ok & (o < Ls - 0.5 * tk1), o, Ls)
    lvl = Hs + tk0
    B = trd & (h > lvl - 0.5 * tk0)
    pb = np.where(ok & (o > lvl - 0.5 * tk0), np.maximum(o, lvl), lvl)
    B &= ~(pb >= up1 - 0.5 * tk1)  # limit-up: cannot buy (NaN limit -> comparison False -> kept)
    Op = trd & ok & ~(o >= up1 - 0.5 * tk1)
    Cf = pd.DataFrame(P["C"]).ffill().to_numpy()
    out = {"A": A, "B": B, "OPEN": Op}
    for hh in HS:
        x = Cf.reshape(-1)[np.minimum(e + hh, T - 1) * N + nn]
        out[f"rA{hh}"] = x / pa - 1.0
        out[f"rB{hh}"] = x / pb - 1.0
        out[f"rO{hh}"] = x / o - 1.0
    return out


def _pair(day, ya, ma, yb, mb, min_n):
    n = len(day)
    z = np.zeros(n, bool)
    return day_diff(np.r_[day, day], np.r_[ya, yb], np.r_[ma, z], np.r_[z, mb], min_n)


def ab_lines(tag: str, day: np.ndarray, R: dict, min_n: int) -> list[str]:
    A, B = R["A"], R["B"]
    Ao, Bo, AB = A & ~B, B & ~A, A & B
    L = [f"  —— {tag}：只跌破（A）{Ao.sum():,} 行 | 只越过（B）{Bo.sum():,} 行 | 两头都破（外包线，剔除）{AB.sum():,} 行；同日每组至少 {min_n} 行才计入",
         "     外包线必须剔除：同一只股票同一天，A 按当天最低价买、B 按当天最高价买，A 机械地便宜一整根 K 线的振幅（含外包线时 A−B 会变成 +0.3%~+0.4%）。"]
    L.append(f"     {'持有到':<10s}{'只 A − 只 B（逐日）':>22s}{'只 A − 开盘买全部':>20s}{'只 B − 开盘买全部':>20s}{'只 A − 只 B（按笔裸均值）':>24s}{'天数':>7s}")
    for hh in HS:
        y_a, y_b, y_o = R[f"rA{hh}"], R[f"rB{hh}"], R[f"rO{hh}"]
        d1, t1, n1 = _pair(day, y_a, Ao, y_b, Bo, min_n)
        da, ta, _ = _pair(day, y_a, Ao, y_o, R["OPEN"], min_n)
        db, tb, _ = _pair(day, y_b, Bo, y_o, R["OPEN"], min_n)
        raw = np.nanmean(y_a[Ao]) - np.nanmean(y_b[Bo])
        lab = "当天收盘*" if hh == 0 else f"第 {hh} 天收盘"
        L.append(f"     {lab:<10s}{d1 * 100:>+12.3f}%（t {t1:+5.1f}）{da * 100:>+10.3f}%（t {ta:+5.1f}）"
                 f"{db * 100:>+10.3f}%（t {tb:+5.1f}）{raw * 100:>+18.3f}%{n1:>9,}")
    L.append("     * 当天收盘 = 不受 T+1 约束的理论值；A 股买入当天卖不掉，可执行的是第 1 天起。毛收益，未扣成本（两组成本相同）。"
             "「开盘买全部」= 同一天所有候选按开盘价买入、同一天卖出；A 的买价本来就低于开盘、B 的高于开盘，这两列含「买价位置」效应，只作参考。")
    return L


def bracket_sameday(tag: str, day: np.ndarray, pools: dict, min_n: int) -> list[str]:
    """Bracket random pools (limit fade vs stop follow), per-trade simple return; raw means vs same-day difference,
    with and without outside bars (where both orders fill on the same stock-day)."""
    L = [f"  —— {tag}：括号单随机池（每笔扣成本简单收益率）：裸均值之差 | 逐日相减（含外包线）| 逐日相减（剔除外包线）| 剔除后的裸均值之差"]
    for side, nm in [(1, "买"), (-1, "卖")]:
        yl, fl = pools[("L", side)]
        ys, fs = pools[("S", side)]
        both = fl & fs
        d, t, n = _pair(day, yl, fl, ys, fs, min_n)
        d2, t2, n2 = _pair(day, yl, fl & ~both, ys, fs & ~both, min_n)
        raw = np.nanmean(yl[fl]) - np.nanmean(ys[fs])
        raw2 = np.nanmean(yl[fl & ~both]) - np.nanmean(ys[fs & ~both])
        L.append(f"     限价反向{nm} − 止损顺势{nm}：{raw * 100:+.3f}% | {d * 100:+.3f}%（t {t:+.1f}）| "
                 f"{d2 * 100:+.3f}%（t {t2:+.1f}，{n2:,} 天）| {raw2 * 100:+.3f}%")
    return L


# ---------------------------------------------------------------------------
# 补测 1：indices
# ---------------------------------------------------------------------------
def excess_x(rx, side, run, E, C, *, extra=None, sub=None, contrast=False):
    y_all = run["net"]
    f = run["filled"] & np.isfinite(y_all)
    if sub is not None:
        f &= sub
    rows = np.flatnonzero(f)
    De, Dc = E[rows], C[rows] & ~E[rows]
    if De.sum() < s10.MIN_EVENTS or Dc.sum() < s10.MIN_EVENTS:
        return np.nan, np.nan
    ref = ~De & ~Dc
    X = s10.controls(rx, side, rows, ref, s10.MAIN)
    if extra is not None:
        v = extra[rows]
        X = np.hstack([X, s09._qdum(v, v[ref])])
    y, mi = y_all[rows].astype(float), rx.tt[rows]
    D = [De, De | Dc] if contrast else [De, Dc]
    b, t = s10.dfit(y, D, X, mi)
    return float(b[0]), float(t[0])


def explore1_index(rx) -> list[str]:
    L = [f"  —— {rx.name}"]
    L.append(f"     {'方向':<14s}{'当天涨跌 事件/对照':>20s}{'主规格 超额 | 区间−趋势':>26s}{'+控当天涨跌':>24s}{'只留朝边沿走的日子':>26s}{'分层后 +控当天涨跌':>24s}")
    for q, side, led, ek, ck, s, lab in s10.QUESTIONS:
        if q != "Q1":
            continue
        run = rx.run(led, side, "main")
        E, C = rx.at(side, ek), rx.at(side, ck) & ~rx.at(side, ek)
        r1 = rx.at(side, "ret1")
        toward = r1 < 0
        cells = []
        for extra, sub in [(None, None), (r1, None), (None, toward), (r1, toward)]:
            a, ta = excess_x(rx, side, run, E, C, extra=extra, sub=sub)
            c, tc = excess_x(rx, side, run, E, C, extra=extra, sub=sub, contrast=True)
            cells.append(f"{f_(a)}({f_(ta, 1)}) | {f_(c)}({f_(tc, 1)})")
        fl = run["filled"]
        L.append(f"     {lab:<12s}{np.nanmean(r1[E & fl]) * 100:>+8.2f}% / {np.nanmean(r1[C & fl]) * 100:+.2f}%  "
                 + "  ".join(f"{x:>24s}" for x in cells))
        n_t = (E & fl & toward).sum(), (C & fl & toward).sum()
        L.append(f"     {'':<12s}分层后样本：事件 {n_t[0]:,}、对照 {n_t[1]:,}（成交笔数）")
    return L


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    L = ["=" * 120, "Study 10c · 专题 2 探索性补测（事后诊断，只用开发段；不改 10 / 10b 的主结论）", "=" * 120]
    raw = s09.load_raw()
    idx = {name: s10.RangeIndex(name, df, ev0) for name, (df, ev0) in raw.items()}

    # ---------------- 补测 1 ----------------
    L += ["", "[补测 1] Q1 专属对照被截断：加控第 t 天涨跌 / 只留「第 t 天朝边沿方向走」的日子",
          "  表格：超额（t）| 区间 − 趋势（t），单位 R；第 1 列为预注册主规格（应与 10 / 10b 报告一致）"]
    for name in ("SPX", "SW"):
        L += explore1_index(idx[name])

    log("stocks: caches + panel controls")
    c = s10b.load_cache()
    c09 = dict(np.load(s09b.CACHE_PATH, allow_pickle=False))
    rows = c["rows"]
    M = len(rows)
    pn = pd.read_parquet(s09b.PANEL_PATH, columns=["code_id", "trade_date", "mi", "l1", "lmv"] + [x for x in s09b.L2_COLS if x != "lmv"])
    pn = pn.sort_values(["code_id", "trade_date"]).reset_index(drop=True)
    mi = pn["mi"].to_numpy()[rows]
    code = pn["code_id"].to_numpy()[rows]
    td = pn["trade_date"].to_numpy()[rows]
    l1 = pd.factorize(pn["l1"].to_numpy()[rows])[0]
    Z = pn[s09b.L2_COLS].to_numpy(float)[rows]
    lmv = pn["lmv"].to_numpy(float)[rows]
    szq = np.ceil((pd.Series(lmv).groupby(mi).rank(pct=True, method="first").to_numpy() * 5).clip(1e-9, 5)).astype(int)
    del pn
    order = np.argsort(mi, kind="stable")
    geo = {x: c09[x][rows] for x in ["g_atr", "g_px", "kcls"]}

    def resid(y, filled, fit, G):
        o = order[filled[order]]
        r, _ = s09b.resid_day(mi[o], szq[o], Z[o], l1[o], G[o], y[o][:, None], fit[o], s09b.MIN_DAY)
        out = np.full(M, np.nan)
        out[o] = r[:, 0]
        return out

    L += ["  —— 个股（双胞胎残差；超额 = vs_twins，区间 − 趋势 = vs_ctrl）"]
    for q, side, led, ek, ck, s, lab in s10.QUESTIONS:
        if q != "Q1":
            continue
        p = "L" if side > 0 else "S"
        E, C = c[f"f{p}_{ek}"], c[f"f{p}_{ck}"] & ~c[f"f{p}_{ek}"]
        F = c[f"{p}_main_filled"]
        fit = F & ~E & ~C
        net = c[f"{p}_main_net"].astype(float)
        r1 = c[f"f{p}_ret1"].astype(float)
        assert np.isfinite(r1[F]).all()
        G = np.hstack([s09b.qdum(geo[x], geo[x][F]) for x in ["g_atr", "g_px", "kcls"]])
        Gr = np.hstack([G, s09b.qdum(r1, r1[fit])])
        log(f"explore1 stocks {lab}")
        res0, res1 = resid(net, F, fit, G), resid(net, F, fit, Gr)
        toward = r1 < 0
        cells = []
        for res, m in [(res0, None), (res1, None), (res0, toward), (res1, toward)]:
            Ei = np.flatnonzero(E & F & (m if m is not None else True))
            Ci = np.flatnonzero(C & F & (m if m is not None else True))
            a, ta, _ = b8.vs_twins(res, Ei, mi)
            cc, tc = b8.vs_ctrl(res, Ei, Ci, mi)
            cells.append(f"{f_(a)}({f_(ta, 1)}) | {f_(cc)}({f_(tc, 1)})")
        L.append(f"     {lab:<12s}{np.mean(r1[E & F]) * 100:>+8.2f}% / {np.mean(r1[C & F]) * 100:+.2f}%  "
                 + "  ".join(f"{x:>24s}" for x in cells))
        L.append(f"     {'':<12s}分层后样本：事件 {(E & F & toward).sum():,}、对照 {(C & F & toward).sum():,}（成交笔数）")

    # ---------------- 补测 2 ----------------
    L += ["", "[补测 2] 买弱 vs 买强（不用括号单；同一天内比较；毛收益）"]
    sw = idx["SW"]
    R = ab_returns(sw.mk.P, sw.tt, sw.nn)
    L += ab_lines("行业（31 个申万一级）", sw.tt, R, 2)
    pools = {}
    for side in (1, -1):
        for led in ("L", "S"):
            r = sw.run(led, side, "main")
            pools[(led, side)] = (r["net"] * r["R"] / np.abs(r["fill"]), r["filled"])
    L += bracket_sameday("行业", sw.tt, pools, 2)

    log("stocks: wide market for explore 2")
    pnf, info = s09b.load_panel()
    P = s09b.wide_market(pnf, info["T"], info["N"])
    del pnf
    R = ab_returns(P, mi, code)
    del P
    L += ab_lines("个股", mi, R, 20)
    cost = np.where(td < s09b.STAMP_CUT, s09b.COST_PRE, s09b.COST_POST)
    pools = {}
    for side, p in [(1, "L"), (-1, "S")]:
        for led, src in [("L", c), ("S", c09)]:
            F = src[f"{p}_main_filled"]
            g, n0 = src[f"{p}_main_gross"].astype(float), src[f"{p}_main_net"].astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                y = np.where(F, n0 * cost / (g - n0), np.nan)
            pools[(led, side)] = (y, F)
    L += bracket_sameday("个股", mi, pools, 20)
    txt = "\n".join(L)
    REPORT_PATH.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    log(f"done ({time.time() - t0:.0f}s) -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
