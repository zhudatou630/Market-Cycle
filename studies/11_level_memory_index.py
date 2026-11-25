#!/usr/bin/env python3
"""
Market Cycle V2 — Study 11（第三阶段专题 3 · 价位记忆：价位识别 + 自检 + SPX + 31 个申万一级行业）

预注册见 decisions.md「11-预注册」（方案全文 output/11_plan_for_review.md v2）。主规格开跑前写死：
  - 统一语法（本侧视角；压力位 = 价格镜像 p' = K − p）：价位 L 在 t−1 收盘已确定且形成以来无收盘跌破（活着）；
    测试 = 第 t 天最低价 ≤ L + 0.3 × ATR20(t−1)，每个价位只认第一次；守住 = 收盘 ≥ L，失守 = 收盘 < L；
    同日碰到多个价位 → 事件归最高的那个，其余一并作废。识别用的根数按标的自身交易日。
  - A 族：波段低点 = 最低价严格低于前 5 根、不高于后 5 根（第 j+5 天确认）；j+1…t−1 最高价 ≥ L + 2 × ATR20(j) 后才开始监测；
    测试日距 j 10–60 根（< 10 根的测试同样消耗该价位；> 60 根失效）；测试次数 = 120 根内同价位 ±0.3 ATR 已确认低点数。
  - 锚点（不进 FDR）：05/08b 的 20 日首次收盘新高，价位 = 前 20 根最高收盘，窗口 20 根，要求当天回落（低点或收盘低于前一根）。
  - 普通价位（专属对照）：前一根 K 线低点，不给容差：普通守住 = 最低 ≤ 昨低且收盘 ≥ 昨低；普通失守 = 收盘 < 昨低；
    另报普通 20 日首次收盘新低；都要求当天最低价 0.5 ATR 内没有同族活价位（活价位 ≥ 最低价 − 0.5 ATR）。
  - 交易（止损单，信号 K 线 = 测试日，专题 1 引擎原样）：① 支撑守住 → 越过高点买；② 支撑失守 → 跌破低点卖；
    ③ 压力守住（两次冲高失败）→ 跌破低点卖。1R、20 天到期、同根双碰先算止损、扣成本。
  - 指数统计：池化回归，净 R = 常数 + 事件哑变量 + 其他真价位事件哑变量 + 控制（近 20 日涨跌、20 日波动、近 5 日涨跌、
    当天涨跌 线性 + 当天涨跌五分位 + 几何五分位，切点取本侧成交池），DK lag 30；基线 = 非真价位事件的随机单（含普通价位）；
    专属对照 = 同一回归里事件与普通价位系数之差。行业 A 族 ①②③ 进 A 股 FDR（在 11b 合并），SPX 只看方向。

用法：
  python studies/11_level_memory_index.py --step0   # 自检 + 样本量（不看任何结果）
  python studies/11_level_memory_index.py           # 第 1 步完整结果
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
STEP0_PATH = OUTPUT_DIR / "11_step0_report.txt"
REPORT_PATH = OUTPUT_DIR / "11_level_memory_index_report.txt"
SUMMARY_PATH = OUTPUT_DIR / "11_level_memory_index_summary.csv"


def _load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "studies" / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


s10 = _load_mod("s10", "10_trading_range_index.py")
s09 = s10.s09
b8 = s09.b8
s01 = b8.s01
log = b8.log

EPS = 1e-12
TOL = 0.3          # test zone (05/08b touch tolerance)
TIE = 1e-7         # ties on the price grid: view prices are rounded to 1e-6, ATR-scaled thresholds get this slack
ROUND = 6
NEAR = 0.5         # generic level must be ≥ 0.5 ATR away from any live same-family level
PIV = 5            # swing low: strictly below the 5 bars before, not above the 5 bars after
BOUNCE = 2.0       # bounce ≥ 2 × ATR20(j) before a level is monitored
AGE_MIN, AGE_MAX = 10, 60
NT_LOOK = 120
C_COOL, C_WIN = 20, 20
N_WIN = 20
CONC_WIN = 60
LAG = 30
MIN_EVENTS = 10
SIDE_NAME = {1: "支撑（本侧）", -1: "压力（镜像）"}
# primary trades: (code, view side, ledger side, event key, control key, label)
TRADES = [
    ("A1", 1, 1, "A_hold", "G_hold", "① 双底守住买"),
    ("A2", 1, -1, "A_fail", "G_fail", "② 双底失守卖"),
    ("A3", -1, -1, "A_hold", "G_hold", "③ 两次冲高失败卖"),
]
ANCHOR = [
    ("N1", 1, 1, "N_hold", "G_hold", "锚点 突破位守住买"),
    ("N2", 1, -1, "N_fail", "G_fail", "锚点 突破位失守卖"),
]


# ---------------------------------------------------------------------------
# 1. Level detectors (one asset, side view, own trading days; all loops causal)
# ---------------------------------------------------------------------------
def pivots_low(l: np.ndarray) -> np.ndarray:
    n = len(l)
    piv = np.zeros(n, bool)
    if n < 2 * PIV + 1:
        return piv
    W = sliding_window_view(l, 2 * PIV + 1)
    ctr = W[:, PIV]
    piv[PIV:n - PIV] = (ctr < W[:, :PIV].min(1)) & (ctr <= W[:, PIV + 1:].min(1))
    return piv


def _empty(n: int, keys_b: list[str], keys_f: list[str]) -> dict[str, np.ndarray]:
    d = {k: np.zeros(n, bool) for k in keys_b}
    d.update({k: np.full(n, np.nan) for k in keys_f})
    return d


A_B = ["A_test", "A_hold", "A_fail", "A_clean", "A_near"]
A_F = ["A_lvl", "A_dist", "A_age", "A_ntest", "A_ntouch", "A_narm", "A_toparm"]


def detect_A(h, l, c, atr, em) -> dict[str, np.ndarray]:
    n = len(c)
    out = _empty(n, A_B, A_F)
    piv = pivots_low(l)
    pidx = np.flatnonzero(piv)
    H, Lw, C, A, EM = h.tolist(), l.tolist(), c.tolist(), atr.tolist(), em.tolist()
    fin = math.isfinite
    lv_ = []  # [level, j, atr_j, max high since j+1, armed, consumed]
    for t in range(n):
        j = t - PIV - 1
        if j >= PIV and piv[j] and fin(A[j]) and A[j] > EPS:
            lv = Lw[j]
            if not (c[j + 1:t] < lv).any():
                mh = float(h[j + 1:t].max())
                lv_.append([lv, j, A[j], mh, mh >= lv + BOUNCE * A[j] - TIE, False])
        if lv_:
            lv_ = [x for x in lv_ if t - x[1] <= AGE_MAX]
        arm = [x[0] for x in lv_ if x[4] and not x[5]]
        out["A_narm"][t] = len(arm)
        out["A_toparm"][t] = max(arm) if arm else np.nan
        z = A[t - 1] if t >= 1 else float("nan")
        if t >= 1 and fin(z) and z > EPS and lv_:
            lt = Lw[t]
            touched = [x for x in lv_ if x[4] and not x[5] and lt <= x[0] + TOL * z + TIE]
            if touched:
                top = max(touched, key=lambda x: x[0])
                for x in touched:
                    x[5] = True
                age = t - top[1]
                if AGE_MIN <= age <= AGE_MAX and EM[t]:
                    L0 = top[0]
                    out["A_test"][t] = True
                    out["A_hold"][t] = C[t] >= L0
                    out["A_fail"][t] = C[t] < L0
                    out["A_clean"][t] = lt >= L0
                    out["A_lvl"][t] = L0
                    out["A_dist"][t] = (C[t - 1] - L0) / z
                    out["A_age"][t] = age
                    out["A_ntouch"][t] = len(touched)
                    a, b = np.searchsorted(pidx, t - NT_LOOK), np.searchsorted(pidx, t - PIV - 1, side="right")
                    out["A_ntest"][t] = int((np.abs(l[pidx[a:b]] - L0) <= TOL * z + TIE).sum())
            out["A_near"][t] = any(x[0] >= lt - NEAR * z - TIE for x in lv_)
        if lv_:
            ct, ht = C[t], H[t]
            lv_ = [x for x in lv_ if not (ct < x[0])]
            for x in lv_:
                if ht > x[3]:
                    x[3] = ht
                if not x[4] and x[3] >= x[0] + BOUNCE * x[2] - TIE:
                    x[4] = True
    return out


C_B = ["C_test", "C_hold", "C_fail", "C_clean", "C_near", "C_lian", "C_yizi"]
C_F = ["C_lvl", "C_dist", "C_age"]


def detect_C(h, l, c, atr, upc, elig, em) -> dict[str, np.ndarray]:
    """Limit-up launch price: first close-at-limit-up in 20 days (itself eligible), level = previous close, window 20."""
    n = len(c)
    out = _empty(n, C_B, C_F)
    if n < 2:
        return out
    upf = upc.astype(float)
    prior = pd.Series(upf).shift(1).rolling(C_COOL, min_periods=1).sum().fillna(0).to_numpy() > 0
    fresh = upc & ~prior & elig
    fresh[0] = False
    H, Lw, C, A, EM, UP, FR = h.tolist(), l.tolist(), c.tolist(), atr.tolist(), em.tolist(), upc.tolist(), fresh.tolist()
    fin = math.isfinite
    cur = None  # [level, g, consumed, lianban, yizi]
    for t in range(n):
        if cur is not None and t - cur[1] > C_WIN:
            cur = None
        z = A[t - 1] if t >= 1 else float("nan")
        if cur is not None and fin(z) and z > EPS:
            lt = Lw[t]
            if not cur[2] and lt <= cur[0] + TOL * z + TIE:
                cur[2] = True
                if EM[t]:
                    L0 = cur[0]
                    out["C_test"][t] = True
                    out["C_hold"][t] = C[t] >= L0
                    out["C_fail"][t] = C[t] < L0
                    out["C_clean"][t] = lt >= L0
                    out["C_lian"][t] = cur[3]
                    out["C_yizi"][t] = cur[4]
                    out["C_lvl"][t] = L0
                    out["C_dist"][t] = (C[t - 1] - L0) / z
                    out["C_age"][t] = t - cur[1]
            out["C_near"][t] = cur[0] >= lt - NEAR * z - TIE
        if cur is not None:
            if C[t] < cur[0]:
                cur = None
            elif UP[t]:
                cur[3] = True
        if FR[t]:
            cur = [C[t - 1], t, False, False, H[t] - Lw[t] <= EPS * max(1.0, abs(H[t]))]
    return out


N_B = ["N_test", "N_hold", "N_fail", "N_clean"]
N_F = ["N_lvl"]


def detect_anchor(h, l, c, atr, em) -> dict[str, np.ndarray]:
    """08b breakout level: 20-day first closing high (05 `brk_20`), level = prior 20 closes' max, window 20, pullback bar."""
    n = len(c)
    out = _empty(n, N_B, N_F)
    brk = s10.first_break(c, atr, np.ones(n, bool))
    prev_hi = pd.Series(c).rolling(s10.BOX, min_periods=s10.BOX).max().shift(1).to_numpy()
    fin = math.isfinite
    for t0 in np.flatnonzero(brk).tolist():
        L0 = prev_hi[t0]
        for t in range(t0 + 1, min(n, t0 + N_WIN + 1)):
            z = atr[t - 1]
            if not (fin(z) and z > EPS):
                continue
            if l[t] <= L0 + TOL * z + TIE and (l[t] < l[t - 1] or c[t] < c[t - 1]):
                if em[t] and not out["N_test"][t]:
                    out["N_test"][t] = True
                    out["N_hold"][t] = c[t] >= L0
                    out["N_fail"][t] = c[t] < L0
                    out["N_clean"][t] = l[t] >= L0
                    out["N_lvl"][t] = L0
                break
            if c[t] < L0:
                break
    return out


def generic(h, l, c, atr, em, near: np.ndarray, tag: str) -> dict[str, np.ndarray]:
    """Ordinary level = previous bar's low (no tolerance); excluded when a live same-family level is within 0.5 ATR."""
    lp = s09._shift(l, 1)
    z = s09._shift(atr, 1)
    ok = em & np.isfinite(lp) & np.isfinite(z) & (z > EPS) & ~near
    bdn = s10.first_break(-c, atr, em)
    return {f"{tag}_hold": ok & (l <= lp) & (c >= lp), f"{tag}_fail": ok & (c < lp), f"{tag}_bdn": ok & bdn}


def asset_view(h, l, c, em, side, upc=None, dnc=None, elig=None) -> dict[str, np.ndarray]:
    """All topic-3 features of one asset in the side's view (side −1 = mirrored prices: lows ↔ highs)."""
    hv0, lv0, cv0, K = s10.side_view(h, l, c, side)
    hv, lv, cv = (np.round(x, ROUND) for x in (hv0, lv0, cv0))
    atr = s09.base_feats(hv, lv, cv)["atr20"]
    d = detect_A(hv, lv, cv, atr, em)
    d.update(generic(hv, lv, cv, atr, em, d["A_near"], "G"))
    # anchor = the 08b definition on unrounded prices (checked against 08b detect_events, not in the mirror test)
    d.update(detect_anchor(hv0, lv0, cv0, s09.base_feats(hv0, lv0, cv0)["atr20"], em))
    if upc is not None:
        lim = upc if side > 0 else dnc
        dc = detect_C(hv, lv, cv, atr, lim, elig, em)
        d.update(dc)
        d.update(generic(hv, lv, cv, atr, em, dc["C_near"], "GC"))
    for k in ["A_lvl", "A_toparm", "N_lvl", "C_lvl"]:
        if k in d and side < 0:
            d[k] = K - d[k]  # report levels in real prices
    d["atr_v"] = atr
    d["ret1"] = side * (c / s09._shift(c, 1) - 1.0)
    d["r5v"] = side * (c / s09._shift(c, 5) - 1.0)
    d["rng5"] = pd.Series(h - l).rolling(5, min_periods=5).mean().shift(1).to_numpy() / s09._shift(atr, 1)
    return d


BOOL_KEYS = set(A_B + C_B + N_B + ["G_hold", "G_fail", "G_bdn", "GC_hold", "GC_fail", "GC_bdn"])


def concentration(h, l, amt, t_idx, lvl, z) -> np.ndarray:
    """Turnover share of t−60…t−1 traded inside the band L ± 0.3 × ATR20(t−1) (range-overlap weighted)."""
    out = np.full(len(t_idx), np.nan)
    for i, (t, L0, zz) in enumerate(zip(t_idx.tolist(), lvl.tolist(), z.tolist())):
        if t < CONC_WIN or not (math.isfinite(L0) and math.isfinite(zz)):
            continue
        hh, ll, aa = h[t - CONC_WIN:t], l[t - CONC_WIN:t], amt[t - CONC_WIN:t]
        lo, hi = L0 - TOL * zz, L0 + TOL * zz
        rng = hh - ll
        ov = np.clip(np.minimum(hh, hi) - np.maximum(ll, lo), 0, None)
        frac = np.where(rng > EPS, ov / np.where(rng > EPS, rng, 1), ((ll >= lo) & (ll <= hi)).astype(float))
        tot = np.nansum(aa)
        if tot > 0:
            out[i] = float(np.nansum(frac * aa) / tot)
    return out


# ---------------------------------------------------------------------------
# 2. Self-tests (definitions only; no outcomes)
# ---------------------------------------------------------------------------
def _bars(lows, rng=0.4, close_pos=0.5):
    lows = np.asarray(lows, float)
    h = lows + rng
    c = lows + close_pos * rng
    return h, lows.copy(), c


def selftest_defs(raw) -> list[str]:
    out = []

    def chk(name, cond):
        assert cond, f"topic-3 self-test failed: {name}"
        out.append(f"  ✔ {name}")

    # standard double bottom: down to 10 at j=10, rally to ≥ 10 + 2 ATR, back down to test on day ~30
    z = 0.3

    def run(lows, closes=None, hi=None):
        h, l, c = _bars(lows)
        if closes is not None:
            for k, v in closes.items():
                c[k] = v
        if hi is not None:
            for k, v in hi.items():
                h[k] = v
        atr = np.full(len(l), z)
        return detect_A(h, l, c, atr, np.ones(len(l), bool)), (h, l, c)

    dn = list(np.linspace(12, 10.2, 10)) + [10.0]
    up = list(np.linspace(10.3, 11.2, 9))
    back = list(np.linspace(11.0, 10.3, 9))
    seq = dn + up + back + [10.05, 10.4, 10.6, 10.8, 11.0]
    t_test = len(dn) + len(up) + len(back)
    d, (h, l, c) = run(seq)
    chk("标准双底：第二次下探进入 L + 0.3 ATR → 测试；收盘 ≥ L → 守住；测试次数 = 1",
        d["A_test"][t_test] and d["A_hold"][t_test] and d["A_ntest"][t_test] == 1 and abs(d["A_lvl"][t_test] - 10.0) < 1e-12
        and d["A_test"].sum() == 1 and d["A_clean"][t_test])
    sf = list(seq)
    sf[t_test] = 9.85
    tail = list(np.linspace(10.3, 11.2, 9)) + list(np.linspace(11.0, 10.3, 8)) + [10.02, 10.5, 10.6]
    d, _ = run(sf + tail, closes={t_test: 9.9})
    chk("同一序列收盘 < L → 失守（刺穿）；失守后价位死亡，之后再回到 10.0 不再算这个价位的事件",
        d["A_test"][t_test] and d["A_fail"][t_test] and not d["A_clean"][t_test]
        and int((np.abs(d["A_lvl"] - 10.0) < 1e-12).sum()) == 1)
    d, _ = run(dn + [10.1, 10.12, 10.15, 10.12, 10.1, 10.13, 10.15, 10.12, 10.1, 10.14, 10.12] + [10.05, 10.1, 10.12])
    chk("反弹不足 2 ATR → 不监测、无测试", d["A_test"].sum() == 0)
    seq_old = dn + up + list(np.full(60, 11.2)) + list(np.linspace(11.0, 10.05, 6)) + [10.5, 10.6]
    d, _ = run(seq_old)
    chk("年龄 > 60 根 → 价位失效、无测试", d["A_test"].sum() == 0)
    fast = dn + [10.5, 10.8, 11.0, 10.9, 10.7, 10.4, 10.05] + list(np.linspace(10.3, 11.2, 9)) + list(np.linspace(11.0, 10.3, 8)) + [10.05, 10.5, 10.6]
    d, _ = run(fast)
    lv_f = d["A_lvl"][d["A_test"]]
    chk("年龄 < 10 根的测试同样消耗价位：10.0 在第 7 根被碰后作废，之后只有新低点 10.05 的事件",
        not d["A_test"][len(dn) + 6] and len(lv_f) >= 1 and np.all(np.abs(lv_f - 10.0) > 1e-9) and np.any(np.abs(lv_f - 10.05) < 1e-9))
    trip = dn + up + back + [10.05] + list(np.linspace(10.4, 11.3, 9)) + list(np.linspace(11.0, 10.3, 9)) + [10.02, 10.5, 10.6]
    d, _ = run(trip)
    tt = len(trip) - 3
    chk("三重底：第三次下探的测试次数 = 2（两个已确认的同价位低点）", d["A_test"][tt] and d["A_ntest"][tt] == 2)
    two = list(np.linspace(12, 10.2, 10)) + [10.0, 10.4, 10.8, 11.2, 11.0, 10.6, 10.2, 10.5, 10.9, 11.3, 11.1, 10.8,
                                            10.7, 10.6, 10.5, 10.45, 10.4, 10.35, 9.95, 10.3, 10.5]
    d, _ = run(two)
    chk("同日碰到两个价位：事件归最高的 10.2，较低的 10.0 一并作废", d["A_test"][28] and abs(d["A_lvl"][28] - 10.2) < 1e-12
        and d["A_ntouch"][28] == 2 and d["A_test"].sum() == 1)
    # C family
    n = 40
    lo = np.r_[np.full(10, 10.0), 10.0, np.full(5, 11.0), [10.95, 10.7, 10.5, 10.25, 10.05], np.full(n - 21, 10.4)]
    h, l, c = lo + 0.3, lo.copy(), lo + 0.15
    c[10], h[10], l[10] = 11.0, 11.0, 10.1  # limit-up bar, previous close = 10.15
    upc = np.zeros(n, bool)
    upc[10] = True
    d = detect_C(h, l, c, np.full(n, 0.3), upc, np.ones(n, bool), np.ones(n, bool))
    tC = int(np.flatnonzero(d["C_test"])[0]) if d["C_test"].any() else -1
    chk("涨停起涨价：价位 = 涨停前一天收盘，第一次回到 L + 0.3 ATR 才算测试", tC == 20 and abs(d["C_lvl"][tC] - 10.15) < 1e-12
        and (l[11:tC] > 10.15 + 0.09).all() and d["C_test"].sum() == 1 and d["C_hold"][tC])
    upc2 = upc.copy()
    upc2[3] = True
    d2 = detect_C(h, l, c, np.full(n, 0.3), upc2, np.ones(n, bool), np.ones(n, bool))
    chk("20 天内已有收盘涨停 → 不是新鲜起涨，不建价位", not d2["C_test"][tC])
    hy, ly, cy = h.copy(), l.copy(), c.copy()
    hy[10] = ly[10] = cy[10] = 11.0
    d3 = detect_C(hy, ly, cy, np.full(n, 0.3), upc, np.ones(n, bool), np.ones(n, bool))
    chk("一字板标记", d3["C_yizi"][d3["C_test"]].all() and not d["C_yizi"][d["C_test"]].any())
    el = np.ones(n, bool)
    el[10] = False
    chk("涨停日本身不入池 → 不建价位", detect_C(h, l, c, np.full(n, 0.3), upc, el, np.ones(n, bool))["C_test"].sum() == 0)
    # generic
    lg = np.array([10, 9.8, 9.9, 9.7, 9.75])
    hg, cg = lg + 0.4, np.array([10.2, 10.05, 10.1, 9.75, 9.6])
    g = generic(hg, lg, cg, np.full(5, 0.3), np.ones(5, bool), np.zeros(5, bool), "G")
    chk("普通价位：最低 ≤ 昨低且收盘 ≥ 昨低 = 普通守住；收盘 < 昨低 = 普通失守；最低 > 昨低且收盘在上不算",
        g["G_hold"].tolist() == [False, True, False, False, False] and g["G_fail"].tolist() == [False, False, False, True, True])
    g2 = generic(hg, lg, cg, np.full(5, 0.3), np.ones(5, bool), np.array([False, True, False, False, False]), "G")
    chk("普通价位：靠近同族活价位的 K 线排除", not g2["G_hold"][1])

    # real data: mirror symmetry, truncation, t−1 probe, anchor vs 08b
    for mk, (df, ev0) in raw.items():
        bad_m = bad_p = bad_q = bad_b = tot_b = n_ev = 0
        rng = np.random.default_rng(11)
        for asset, gdf in df.groupby("asset", sort=True):
            gdf = gdf.sort_values("date").reset_index(drop=True)
            h, l, c = (gdf[x].to_numpy(float) for x in ("high", "low", "close"))
            dts = gdf["date"].to_numpy()
            em = np.asarray(pd.to_datetime(dts) >= ev0, bool)
            a = asset_view(h, l, c, em, 1)
            K = 2 * np.nanmax(h) + 7.0
            m = asset_view(K - l, K - h, K - c, em, -1)
            n_ev += int(a["A_test"].sum())
            for k in ["A_test", "A_hold", "A_fail", "A_clean", "A_near", "G_hold", "G_fail", "G_bdn"]:
                bad_m += not np.array_equal(a[k], m[k])
            bad_m += not np.allclose(np.nan_to_num(a["A_ntest"], nan=-1), np.nan_to_num(m["A_ntest"], nan=-1))
            cuts = rng.choice(np.arange(80, len(c)), size=min(30, len(c) - 80), replace=False)
            for tc in cuts.tolist():
                p = asset_view(h[:tc + 1], l[:tc + 1], c[:tc + 1], em[:tc + 1], 1)
                for k in ["A_test", "A_hold", "A_lvl", "A_ntest", "A_near", "G_hold", "G_fail", "N_test", "N_hold"]:
                    x, y = p[k][tc], a[k][tc]
                    if not (x == y or (isinstance(x, float) and np.isnan(x) and np.isnan(y))):
                        bad_p += 1
                # t−1 probe: data up to tc−1 plus a dummy copy of bar tc−1 at position tc
                hq, lq, cq = (np.round(np.r_[v[:tc], v[tc - 1]], ROUND) for v in (h, l, c))
                q = detect_A(hq, lq, cq, s09.base_feats(hq, lq, cq)["atr20"], em[:tc + 1])
                if not (q["A_narm"][tc] == a["A_narm"][tc] and
                        (q["A_toparm"][tc] == a["A_toparm"][tc] or (np.isnan(q["A_toparm"][tc]) and np.isnan(a["A_toparm"][tc])))):
                    bad_q += 1
            ev, _ = b8.detect_events(h, l, c, dts, ev0)
            ref = set(ev["brk_20"].tolist())
            mine = set(np.flatnonzero(s10.first_break(c, s09.base_feats(h, l, c)["atr20"], em)).tolist())
            tot_b += len(ref)
            bad_b += ref != mine
        out.append(f"  {mk}：镜像对称（价格取负后支撑 ↔ 压力，A 族事件 / 守住 / 失守 / 测试次数 / 普通价位）"
                   + ("全部一致 ✔" if bad_m == 0 else f"{bad_m} 处不一致 ✘") + f"（A 族支撑测试 {n_ev:,} 条）")
        out.append(f"  {mk}：截断到第 t 天重算与全样本一致" + ("✔" if bad_p == 0 else f"✘ {bad_p} 处")
                   + "；截到 t−1 天（第 t 天换成假 K 线）时第 t 天可用的价位清单不变" + ("✔" if bad_q == 0 else f"✘ {bad_q} 处"))
        out.append(f"  {mk}：锚点所用 20 日首次收盘新高 vs 08b detect_events 的 brk_20（{tot_b:,} 条）" + ("全部一致 ✔" if bad_b == 0 else f"✘ {bad_b}"))
        assert bad_m == 0 and bad_p == 0 and bad_q == 0 and bad_b == 0, f"real-data verification failed on {mk}: {bad_m},{bad_p},{bad_q},{bad_b}"
    return out


# ---------------------------------------------------------------------------
# 3. Index markets
# ---------------------------------------------------------------------------
def load_sw_amount(dates, assets) -> np.ndarray:
    """Read-only: SW L1 amount for the dev years from the latest READY snapshot (same as s01.load_sw_l1_dev)."""
    conn = sqlite3.connect(f"file:{s01.CATALOG_DB}?mode=ro", uri=True)
    try:
        sid = pd.read_sql_query("SELECT snapshot_id FROM data_snapshot WHERE (gate = 'READY' OR gate LIKE '%_READY') "
                                "AND gate != 'NOT_READY' ORDER BY end_date DESC, created_at DESC LIMIT 1", conn).iloc[0, 0]
        man = pd.read_sql_query("SELECT partition_key, file_path FROM table_manifest WHERE snapshot_id = ? AND dataset = 'sw_daily'",
                                conn, params=(sid,))
    finally:
        conn.close()
    files = []
    for _, r in man.iterrows():
        if r["partition_key"] in {f"year={y}" for y in range(2016, 2025)}:
            p = Path(r["file_path"])
            files.append(p if p.is_absolute() else s01.MARKET_DATA_ROOT / p)
    df = pd.concat([pd.read_parquet(f, columns=["src", "level", "index_code", "trade_date", "amount"]) for f in files])
    df = df[(df["src"] == "SW2021") & (df["level"] == "L1")]
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df[df["date"] <= pd.Timestamp("2024-10-31")]
    W = np.full((len(dates), len(assets)), np.nan)
    ti = np.searchsorted(dates, df["date"].to_numpy())
    ok = (ti < len(dates))
    ti = np.minimum(ti, len(dates) - 1)
    ok &= dates[ti] == df["date"].to_numpy()
    ni = pd.Index(assets).get_indexer(df["index_code"])
    ok &= ni >= 0
    W[ti[ok], ni[ok]] = df["amount"].to_numpy(float)[ok]
    return W


class LevelIndex:
    def __init__(self, name: str, df: pd.DataFrame, eval_start: pd.Timestamp) -> None:
        self.mk = mk = s09.IndexMarket(name, df, eval_start)
        self.name = name
        T, N = mk.T, mk.N
        em_all = np.asarray(mk.dates >= np.datetime64(eval_start), bool)
        self.feat: dict[int, dict[str, np.ndarray]] = {}
        for side in (1, -1):
            W: dict[str, np.ndarray] = {}
            for n in range(N):
                rows = np.flatnonzero(np.isfinite(mk.P["C"][:, n]))
                if len(rows) < 30:
                    continue
                d = asset_view(mk.P["H"][rows, n], mk.P["L"][rows, n], mk.P["C"][rows, n], em_all[rows], side)
                for k, v in d.items():
                    if k not in W:
                        W[k] = np.zeros((T, N), bool) if k in BOOL_KEYS else np.full((T, N), np.nan)
                    W[k][rows, n] = v
            self.feat[side] = W
        self.tt, self.nn = np.nonzero(mk.cand)
        self.year = mk.year[self.tt]
        self.runs: dict[tuple, dict] = {}
        self.amount = load_sw_amount(mk.dates, mk.assets) if name == "SW" else None
        self.real = self.at(1, "A_test") | self.at(-1, "A_test") | self.at(1, "N_test")

    def at(self, side: int, key: str) -> np.ndarray:
        return self.feat[side][key][self.tt, self.nn]

    def run(self, side: int, var: str) -> dict:
        key = (side, var)
        if key not in self.runs:
            kw = {"main": {}, "opt": {"both": "opt"}, "t1": {"t1": "forced"}, "nogap": {"use_open": False},
                  "open": {"entry": "open"}, "k2": {"k": 2.0}}[var]
            self.runs[key] = s09.run_ledger(self.mk.P, self.tt, self.nn, side, cost=self.mk.cost, **kw)
        return self.runs[key]


def controls(lx: LevelIndex, rows: np.ndarray, ref: np.ndarray, layers: tuple) -> np.ndarray:
    mk, tt, nn = lx.mk, lx.tt[rows], lx.nn[rows]
    parts = []
    if "geo" in layers:
        for x in ["g_atr", "g_px", "kcls"]:
            v = mk.F[x][tt, nn]
            parts.append(s09._qdum(v, v[ref]))
    lin = []
    if "ret1" in layers:
        r1 = lx.feat[1]["ret1"][tt, nn]
        parts.append(s09._qdum(r1, r1[ref]))
        lin.append(r1)
    if "trend" in layers:
        lin += [mk.F[x][tt, nn] for x in ["r20", "vol20", "r5"]]
    if lin:
        X = np.column_stack(lin)
        sd = np.nanstd(X, 0)
        parts.append((X - np.nanmean(X, 0)) / np.where(sd > 0, sd, 1))
    return np.hstack(parts) if parts else np.zeros((len(rows), 0))


MAIN = ("geo", "ret1", "trend")


def excess(lx: LevelIndex, run: dict, E: np.ndarray, C: np.ndarray | None = None, *, layers=MAIN, sub=None,
           contrast=False, ycol="net", y_override=None, extra: list | None = None):
    """Event dummy vs the non-real-level random pool (other real-level events get a nuisance dummy).
    contrast=True → event − control (control dummy = paired ordinary level)."""
    y_all = run[ycol] if y_override is None else y_override
    f = run["filled"] & np.isfinite(y_all)
    if sub is not None:
        f &= sub
    rows = np.flatnonzero(f)
    De = E[rows]
    if De.sum() < MIN_EVENTS:
        return np.nan, np.nan
    other = lx.real[rows] & ~De
    Ds = [De]
    if contrast:
        if C is None:
            return np.nan, np.nan
        Dc = C[rows] & ~De & ~other
        if Dc.sum() < MIN_EVENTS:
            return np.nan, np.nan
        Ds = [De, De | Dc]
    for x in (extra or []):
        Ds.append(x[rows] & ~np.any(Ds, axis=0) & ~other)
    Ds.append(other)
    ref = ~np.any(Ds, axis=0)
    X = controls(lx, rows, ref, layers)
    y = y_all[rows].astype(float)
    b, t = s10.dfit(y, Ds, X, lx.tt[rows])
    return float(b[0]), float(t[0])


def support_block(f: dict, H, Lo, C, evalm, cand, vs: int, rng) -> dict:
    """Real prices, wide (T, N) arrays on the market calendar; f = feature dict of view vs (levels in real prices)."""
    tm = f["A_test"] & evalm & np.isfinite(f["A_dist"])
    ext = Lo if vs > 0 else H
    lvl = f["A_lvl"]
    hold = vs * (C[tm] - lvl[tm]) >= 0
    pierce = vs * (ext[tm] - lvl[tm]) < 0
    r5, rg5 = f["r5v"], f["rng5"]
    qr5 = np.nanquantile(r5[tm], [1 / 3, 2 / 3])
    qrg = np.nanquantile(rg5[tm], [1 / 3, 2 / 3])
    cell = lambda a, b: np.digitize(a, qr5) * 3 + np.digitize(b, qrg)
    okp = cand & ~f["A_near"] & ~f["A_test"] & np.isfinite(r5) & np.isfinite(rg5)
    cp_w = np.r_[np.full((1, C.shape[1]), np.nan), C[:-1]]
    z_w = np.r_[np.full((1, C.shape[1]), np.nan), f["atr_v"][:-1]]
    tix, nix = np.nonzero(okp)
    pool = {"ext": ext[tix, nix], "c": C[tix, nix], "cp": cp_w[tix, nix], "z": z_w[tix, nix], "cell": cell(r5[tix, nix], rg5[tix, nix])}
    g = np.isfinite(pool["cp"]) & np.isfinite(pool["z"]) & (pool["z"] > EPS)
    pool = {k: v[g] for k, v in pool.items()}
    ss = support_strength(f["A_dist"][tm], hold, pierce, cell(r5[tm], rg5[tm]), pool, vs, rng)
    ss["view"] = vs
    ss["n_real"] = int(tm.sum())
    return ss


def support_strength(d_real, hold_real, pierce_real, cells_real, pool: dict, vs: int, rng) -> dict:
    """Hold rate at real levels vs same-distance pseudo levels (matched on d-decile, and on d × r5 × range5 cells)."""
    out = {}
    phold_real = hold_real
    qd = np.nanquantile(d_real, np.linspace(0, 1, 11)[1:-1])
    dsamp = rng.choice(d_real, size=len(pool["c"]), replace=True)
    z = pool["z"]
    V = pool["cp"] - vs * dsamp * z
    touch = vs * (pool["ext"] - V) <= TOL * z
    hold = vs * (pool["c"] - V) >= 0
    pierce = vs * (pool["ext"] - V) < 0
    bin_r = np.digitize(d_real, qd)
    bin_p = np.digitize(dsamp, qd)
    for tag, br, bp in [("d", bin_r, bin_p), ("d3", bin_r * 9 + cells_real, bin_p * 9 + pool["cell"])]:
        diff, w = [], []
        for b in np.unique(br):
            mr = br == b
            mp = (bp == b) & touch
            if mp.sum() < 30:
                continue
            diff.append(hold_real[mr].mean() - hold[mp].mean())
            w.append(mr.sum())
        out[f"hold_gap_{tag}"] = float(np.average(diff, weights=w)) if w else np.nan
    out["hold_real"] = float(hold_real.mean())
    out["hold_pseudo"] = float(hold[touch].mean())
    pr = pierce_real
    out["phold_real"] = float(phold_real[pr].mean()) if pr.any() else np.nan
    out["phold_pseudo"] = float(hold[touch & pierce].mean())
    # pierced-recovery gap matched on d deciles
    diff, w = [], []
    for b in np.unique(bin_r[pr]):
        mr = pr & (bin_r == b)
        mp = (bin_p == b) & touch & pierce
        if mp.sum() >= 30:
            diff.append(phold_real[mr].mean() - hold[mp].mean())
            w.append(mr.sum())
    out["phold_gap"] = float(np.average(diff, weights=w)) if w else np.nan
    out["rw_hold"] = rw_hold(d_real, rng)
    return out


def rw_hold(d_real: np.ndarray, rng, m: int = 256, n: int = 60000) -> float:
    """Pure random walk geometry: P(close ≥ V | low ≤ V + 0.3 ATR) for V = prev close − d × ATR, d ~ real distribution."""
    steps = rng.normal(0, 1.0 / math.sqrt(m), size=(n, m))
    path = np.cumsum(steps, axis=1)
    lo, hi, cl = np.minimum(path.min(1), 0), np.maximum(path.max(1), 0), path[:, -1]
    atr = float(np.mean(hi - lo))
    d = rng.choice(d_real, size=n, replace=True)
    V = -d * atr
    t = lo <= V + TOL * atr
    return float((cl[t] >= V[t]).mean()) if t.any() else np.nan


# ---------------------------------------------------------------------------
# 4. Step 0
# ---------------------------------------------------------------------------
def load_raw():
    return s09.load_raw()


def step0(raw) -> list[str]:
    L = ["=" * 110, "Study 11 · 第 0 步：自检 + 样本量（不看任何结果）", "=" * 110,
         "", "[1] 专题 1 账本引擎自检（引擎未改，原样重跑）"]
    L += s09.selftest()
    L += ["", "[2] 价位识别自检（手工序列 + 真实数据镜像 / 截断 / t−1 探针 / 锚点比对）"]
    L += selftest_defs(raw)
    L += ["", "[3] 样本量与事前特征（只看计数与事前特征，不看任何账本结果）"]
    for name, (df, ev0) in raw.items():
        lx = LevelIndex(name, df, ev0)
        mk = lx.mk
        yrs = lx.year
        L.append(f"  —— {name}（{mk.N} 个标的；合格信号 K 线 {mk.cand.sum():,}；评估起点 {ev0.date()}）")
        for side in (1, -1):
            f = lx.feat[side]
            em = mk.evalm
            nt = int((f["A_test"] & em).sum())
            nh, nf = int((f["A_hold"] & em).sum()), int((f["A_fail"] & em).sum())
            cand_t = int((f["A_test"] & mk.cand).sum())
            L.append(f"    {SIDE_NAME[side]} A 族测试 {nt:,}（守住 {nh:,} / 失守 {nf:,}，守住占 {nh / max(nt, 1) * 100:.0f}%）；"
                     f"通过最小风险过滤与截尾 {cand_t:,}；干净守住占守住 {f['A_clean'][f['A_hold'] & em].mean() * 100:.0f}%；"
                     f"同日碰到 ≥2 个价位 {np.mean(f['A_ntouch'][f['A_test'] & em] >= 2) * 100:.1f}%")
            nts = f["A_ntest"][f["A_test"] & em]
            L.append(f"      测试次数 1 / 2 / ≥3：{np.mean(nts == 1) * 100:.0f}% / {np.mean(nts == 2) * 100:.0f}% / {np.mean(nts >= 3) * 100:.0f}%；"
                     f"年龄中位 {np.nanmedian(f['A_age'][f['A_test'] & em]):.0f} 根；距离 d 中位 {np.nanmedian(f['A_dist'][f['A_test'] & em]):.2f} ATR")
            gh, gf, gb = (float(lx.at(side, k).mean()) for k in ("G_hold", "G_fail", "G_bdn"))
            L.append(f"      普通价位占合格 K 线：普通守住 {gh * 100:.1f}% / 普通失守 {gf * 100:.1f}% / 普通 20 日首次收盘新低 {gb * 100:.2f}%；"
                     f"靠近同族活价位而被排除的 K 线 {lx.at(side, 'A_near').mean() * 100:.1f}%")
        nN = int((lx.feat[1]["N_test"] & mk.evalm).sum())
        L.append(f"    锚点（突破位）测试 {nN:,}（守住 {int((lx.feat[1]['N_hold'] & mk.evalm).sum()):,}）")
        r_l, r_s = lx.run(1, "main"), lx.run(-1, "main")
        for code, vs, ls, ek, ck, lab in TRADES + ANCHOR:
            run = r_l if ls > 0 else r_s
            fl = run["filled"]
            E, C = lx.at(vs, ek), lx.at(vs, ck) & ~lx.at(vs, ek) & ~lx.real
            line = f"    {lab:<14s} 事件 {int(E.sum()):>6,} → 成交 {int((E & fl).sum()):>6,} | 普通价位对照 {int(C.sum()):>7,} → 成交 {int((C & fl).sum()):>7,}"
            if name == "SPX":
                line += f" | 2010–19 事件成交 {int((E & fl & (yrs >= s09.SPX_REAL_OPEN_FROM)).sum())}"
            L.append(line)
            r1, r5, kc, ga = (lx.at(vs, "ret1"), lx.at(vs, "r5v"), mk.F["kcls"][lx.tt, lx.nn], mk.F["g_atr"][lx.tt, lx.nn])
            if vs < 0:
                kc = 1 - kc
            L.append(f"       事前特征（本侧视角）事件 / 对照：当天涨跌 {np.nanmean(r1[E]) * 100:+.2f}% / {np.nanmean(r1[C]) * 100:+.2f}%；"
                     f"近 5 日 {np.nanmean(r5[E]) * 100:+.2f}% / {np.nanmean(r5[C]) * 100:+.2f}%；收盘位置 {np.nanmean(kc[E]):.2f} / {np.nanmean(kc[C]):.2f}；"
                     f"振幅/ATR {np.nanmean(ga[E]):.2f} / {np.nanmean(ga[C]):.2f}")
            if name == "SW":
                L.append("       逐年事件数：" + " ".join(f"{y}:{int((E & (yrs == y)).sum())}" for y in range(2016, 2025)))
            else:
                L.append("       分段事件数（84-92/93-01/02-10/11-19）：" + " ".join(
                    str(int((E & (yrs >= a) & (yrs <= b)).sum())) for a, b in s10.periods("SPX")[0]))
        # overlaps (counts only)
        A1 = lx.feat[1]["A_test"] & mk.evalm
        p3 = mk.sig.get("p3_bot_turn", np.zeros_like(A1))
        L.append(f"    重叠：A 族支撑测试日同时是锚点测试 {np.mean(lx.feat[1]['N_test'][A1]) * 100:.1f}%、"
                 f"同时是三推底拐头 {np.mean(p3[A1]) * 100:.1f}%；A 族失守日里属于 20 日首次收盘新低（未排除前）"
                 f"{np.mean(new_low_any(lx)[lx.feat[1]['A_fail'] & mk.evalm]) * 100:.1f}%")
        if name == "SW":
            L.append("    2016 年行业数据没有预热期：A 族事件最早出现在 "
                     f"{pd.Timestamp(mk.dates[np.flatnonzero(A1.any(1))[0]]).date() if A1.any() else '—'}；"
                     f"成交额覆盖 {np.isfinite(lx.amount[mk.evalm]).mean() * 100:.1f}%")
    return L


def new_low_any(lx: LevelIndex) -> np.ndarray:
    mk = lx.mk
    T, N = mk.T, mk.N
    W = np.zeros((T, N), bool)
    for n in range(N):
        rows = np.flatnonzero(np.isfinite(mk.P["C"][:, n]))
        if len(rows) < 30:
            continue
        c = mk.P["C"][rows, n]
        W[rows, n] = s10.first_break(-c, mk.F["atr20"][rows, n], np.ones(len(rows), bool))
    return W


# ---------------------------------------------------------------------------
# 5. Main analysis (step 1)
# ---------------------------------------------------------------------------
def f_(x, nd=3, pct=False):
    return s09.fmt(x, nd, pct)


def analyze(lx: LevelIndex, rng) -> tuple[list[dict], list[str]]:
    mk = lx.mk
    per, halves = s10.periods(lx.name)
    yr = lx.year
    rows_out, notes = [], []
    for code, vs, ls, ek, ck, lab in TRADES + ANCHOR:
        run, ro = lx.run(ls, "main"), lx.run(ls, "opt")
        E = lx.at(vs, ek)
        C = lx.at(vs, ck)
        fl = run["filled"]
        Ef = E & fl
        base = fl & ~lx.real
        r = {"market": lx.name, "code": code, "label": lab, "view": vs, "side": ls, "s": 1,
             "n_sig": int(E.sum()), "n_fill": int(Ef.sum()), "n_ctrl": int((C & ~lx.real & fl).sum())}
        for nm, rr in [("", run), ("_opt", ro)]:
            f2 = rr["filled"]
            r["win" + nm] = float(rr["win"][E & f2].mean()) if (E & f2).any() else np.nan
            r["net" + nm] = float(np.nanmean(rr["net"][E & f2])) if (E & f2).any() else np.nan
            r["rwin" + nm] = float(rr["win"][f2 & ~lx.real].mean())
            r["rnet" + nm] = float(np.nanmean(rr["net"][f2 & ~lx.real]))
        r["ex"], r["ex_t"] = excess(lx, run, E)
        r["ex_opt"], r["ex_opt_t"] = excess(lx, ro, E)
        r["ctrl"], r["ctrl_t"] = excess(lx, run, E, C, contrast=True)
        r["ctrl_opt"], _ = excess(lx, ro, E, C, contrast=True)
        r["ex_abs"], _ = excess(lx, run, E, layers=())
        r["ex_geo"], _ = excess(lx, run, E, layers=("geo",))
        r["ex_geor"], _ = excess(lx, run, E, layers=("geo", "ret1"))
        r["ex_trend_only"], _ = excess(lx, run, E, layers=("trend",))
        r["ex_gen"], r["ex_gen_t"] = excess(lx, run, E, layers=MAIN, extra=[C] + ([lx.at(vs, "G_bdn")] if code in ("A2", "N2") else []))
        if code == "A2":
            r["ctrl_bdn"], r["ctrl_bdn_t"] = excess(lx, run, E, lx.at(vs, "G_bdn"), contrast=True)
        r["per"] = [excess(lx, run, E, sub=(yr >= a) & (yr <= b))[0] for a, b in per]
        r["halves"] = [excess(lx, run, E, sub=(yr >= a) & (yr <= b))[0] for a, b in halves]
        r["p"] = b8.p_two(r["ex_t"])
        # value-at-the-level layer: all tests of this family, regardless of outcome
        tkey = ek.split("_")[0] + "_test"
        r["ex_touch"], r["ex_touch_t"] = excess(lx, run, lx.at(vs, tkey))
        if ls > 0 and lx.name == "SW":
            r["ex_t1"], _ = excess(lx, lx.run(ls, "t1"), E)
        if lx.name == "SPX":
            sub = yr >= s09.SPX_REAL_OPEN_FROM
            r["gap_on"], _ = excess(lx, run, E, sub=sub)
            r["gap_off"], _ = excess(lx, lx.run(ls, "nogap"), E, sub=sub)
            r["n_fill_1019"] = int((Ef & sub).sum())
        if lx.name == "SW":
            y = run["net"].astype(float)
            d = pd.DataFrame({"t": lx.tt, "y": np.where(fl, y, np.nan)})
            g = d.groupby("t")["y"]
            ydm = np.where(g.transform("count").to_numpy() >= 10, y - g.transform("mean").to_numpy(), np.nan)
            r["ex_sameday"], r["ex_sameday_t"] = excess(lx, run, E, y_override=ydm)
        op = lx.run(ls, "open")
        r["open_ex"], r["open_ex_t"] = excess(lx, op, E)
        r["k2_ex"], _ = excess(lx, lx.run(ls, "k2"), E)
        keep = s10.thin(lx, run, E) | s10.thin(lx, run, ~E)
        r["ex_thin"], r["ex_thin_t"] = excess(lx, {**run, "filled": run["filled"] & keep}, E)
        r["time_share"] = float((run["kind"][Ef] == s09.K_TIME).mean()) if Ef.any() else np.nan
        fam = ek.split("_")[0]
        if fam == "A":
            hold_e = ek == "A_hold"
            cl = lx.at(vs, "A_clean")
            if hold_e:
                r["clean_ex"], r["clean_ex_t"] = excess(lx, run, E & cl)
                r["pierce_ex"], r["pierce_ex_t"] = excess(lx, run, E & ~cl)
            nt = lx.at(vs, "A_ntest")
            r["dose_nt"] = [excess(lx, run, E & m)[0] for m in (nt == 1, nt == 2, nt >= 3)]
            r["dose_nt_t"] = [excess(lx, run, E & m)[1] for m in (nt == 1, nt == 2, nt >= 3)]
            r["dose_nt_n"] = [int((Ef & m).sum()) for m in (nt == 1, nt == 2, nt >= 3)]
            if lx.amount is not None:
                cc = np.full(len(lx.tt), np.nan)
                ev = np.flatnonzero(E)
                hv = lx.feat[vs]
                for n in np.unique(lx.nn[ev]):
                    rows_a = np.flatnonzero(np.isfinite(mk.P["C"][:, n]))
                    pos = np.searchsorted(rows_a, lx.tt[ev[lx.nn[ev] == n]])
                    sel = ev[lx.nn[ev] == n]
                    z = hv["atr_v"][rows_a, n]
                    zz = np.r_[np.nan, z[:-1]][pos]
                    cc[sel] = concentration(mk.P["H"][rows_a, n], mk.P["L"][rows_a, n], lx.amount[rows_a, n], pos,
                                            lx.at(vs, "A_lvl")[sel], zz)
                qs = np.nanquantile(cc[Ef], [0.2, 0.4, 0.6, 0.8])
                bins = np.digitize(cc, qs)
                ok = np.isfinite(cc)
                r["dose_cc"] = [excess(lx, run, E & ok & (bins == k))[0] for k in range(5)]
                r["dose_cc_t"] = [excess(lx, run, E & ok & (bins == k))[1] for k in range(5)]
        rows_out.append(r)
    # support strength (descriptive): A-family support tests vs same-distance pseudo levels and random walk
    for vs in (1, -1):
        notes.append(support_block(lx.feat[vs], mk.P["H"], mk.P["L"], mk.P["C"], mk.evalm, mk.cand, vs, rng))
    return rows_out, notes


def render(name: str, rows: list[dict], notes: list[dict]) -> list[str]:
    L = []
    per_lab = "2016…2024 逐年" if name == "SW" else "1984-92/93-01/02-10/11-19"
    L.append(f"\n  【{name}】超额 = 比非真价位随机单多出的每笔净 R（控几何 + 当天涨跌 + 趋势 / 波动，DK lag 30）；"
             "专属对照 = 真价位 − 普通价位（前一根 K 线高低点，同一回归）")
    L.append(f"  {'交易':<16s}{'信号':>7s}{'成交':>7s}{'对照成交':>9s}{'胜率 保守~乐观':>16s}{'随机胜率':>9s}"
             f"{'净R 保守~乐观':>18s}{'随机净R':>9s}{'超额':>8s}{'t':>6s}{'乐观超额':>9s}{'真−普通':>9s}{'t':>6s}")
    for r in rows:
        if name == "SPX" and r["n_fill"] < s09.MIN_TRADES_SPX:
            r["note"] = f"成交 {r['n_fill']} < {s09.MIN_TRADES_SPX} 笔：按预注册只描述，t 值不解读"
        L.append(f"  {r['label']:<14s}{r['n_sig']:>7,}{r['n_fill']:>7,}{r['n_ctrl']:>9,}"
                 f"{r['win'] * 100:>9.1f}%~{r['win_opt'] * 100:.1f}%{r['rwin'] * 100:>8.1f}%"
                 f"{f_(r['net']):>10s}~{f_(r['net_opt'])}{f_(r['rnet']):>9s}{f_(r['ex']):>8s}{f_(r['ex_t'], 1):>6s}"
                 f"{f_(r['ex_opt']):>9s}{f_(r['ctrl']):>9s}{f_(r['ctrl_t'], 1):>6s}" + (f"  ※{r['note']}" if "note" in r else ""))
    L.append("  拆解：绝对 → +几何 → +当天涨跌 → +趋势/波动（主规格）→ +普通价位哑变量；只控趋势；价位事件本身（全部测试）；"
             "不重叠；t+1 开盘进场；2R" + ("；同日去均值" if name == "SW" else ""))
    for r in rows:
        sd = f"；同日 {f_(r['ex_sameday'])}（t {f_(r['ex_sameday_t'], 1)}）" if name == "SW" else ""
        L.append(f"  {r['label']:<14s} {f_(r['ex_abs'])} → {f_(r['ex_geo'])} → {f_(r['ex_geor'])} → {f_(r['ex'])} → {f_(r['ex_gen'])}；"
                 f"只控趋势 {f_(r['ex_trend_only'])}；全部测试 {f_(r['ex_touch'])}（t {f_(r['ex_touch_t'], 1)}）；"
                 f"不重叠 {f_(r['ex_thin'])}（t {f_(r['ex_thin_t'], 1)}）；开盘进 {f_(r['open_ex'])}；2R {f_(r['k2_ex'])}{sd}"
                 + (f"；对普通 20 日新低 {f_(r['ctrl_bdn'])}（t {f_(r['ctrl_bdn_t'], 1)}）" if "ctrl_bdn" in r else ""))
    L.append(f"  稳定性（{per_lab} 超额方向 | 前后两段）" + ("；SPX 2010–2019 计跳空 vs 不计跳空" if name == "SPX" else "；T+1（多头）"))
    for r in rows:
        sg = "".join("+" if (np.isfinite(v) and v > 0) else ("−" if np.isfinite(v) else "·") for v in r["per"])
        npos = sum(1 for v in r["per"] if np.isfinite(v) and v > 0)
        nv = sum(1 for v in r["per"] if np.isfinite(v))
        ex = ""
        if name == "SPX":
            ex = f" | 2010–19：计跳空 {f_(r['gap_on'])} 不计跳空 {f_(r['gap_off'])}（成交 {r['n_fill_1019']}）"
        if "ex_t1" in r:
            ex = f" | T+1 {f_(r['ex_t1'])}"
        L.append(f"  {r['label']:<14s} {sg:<10s} 正 {npos}/{nv} | {' / '.join(f_(v) for v in r['halves'])}{ex}")
    L.append("  机制诊断：守住拆分 干净守住 / 刺穿收回 | 测试次数 1 / 2 / ≥3（超额，成交数）" + (" | 成交密集度五档（低→高）" if name == "SW" else ""))
    for r in rows:
        if "dose_nt" not in r:
            continue
        a = (f"{f_(r['clean_ex'])}（t {f_(r['clean_ex_t'], 1)}）/ {f_(r['pierce_ex'])}（t {f_(r['pierce_ex_t'], 1)}）"
             if "clean_ex" in r else "—")
        b = " ".join(f"{f_(v)}({f_(t, 1)},n{n})" for v, t, n in zip(r["dose_nt"], r["dose_nt_t"], r["dose_nt_n"]))
        c = (" | " + " ".join(f"{f_(v)}({f_(t, 1)})" for v, t in zip(r["dose_cc"], r["dose_cc_t"]))) if "dose_cc" in r else ""
        L.append(f"  {r['label']:<14s} {a} | {b}{c}")
    L.append("  支撑力（描述）：真价位守住率 vs 同距离伪价位（按距离十分位配对 / 再配近 5 日涨跌与振幅）vs 随机游走；刺穿后收回率 真 vs 伪")
    for ss in notes:
        L.append(f"  {SIDE_NAME[ss['view']]}：守住率 真 {ss['hold_real'] * 100:.1f}% 伪 {ss['hold_pseudo'] * 100:.1f}% 随机游走 {ss['rw_hold'] * 100:.1f}%；"
                 f"配对后 真 − 伪 {ss['hold_gap_d'] * 100:+.1f}pp / {ss['hold_gap_d3'] * 100:+.1f}pp；"
                 f"刺穿后收回 真 {ss['phold_real'] * 100:.1f}% 伪 {ss['phold_pseudo'] * 100:.1f}%（配对差 {ss['phold_gap'] * 100:+.1f}pp）")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step0", action="store_true")
    args = ap.parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    raw = load_raw()
    if args.step0:
        L = step0(raw)
        txt = "\n".join(L)
        STEP0_PATH.write_text(txt + "\n", encoding="utf-8")
        print(txt)
        log(f"step0 done ({time.time() - t0:.0f}s) -> {STEP0_PATH}")
        return
    L = ["=" * 120, "Study 11 · 第 1 步：价位记忆（SPX + 31 个申万一级行业，开发段）", "=" * 120,
         "① 支撑守住 → 越过高点买；② 支撑失守 → 跌破低点卖；③ 压力守住（两次冲高失败）→ 跌破低点卖。止损单、1R、20 天、同根双碰先算止损、扣成本。",
         "行业 ①②③ 进 A 股 FDR（在 11b 与个股合并，共 10 个）；SPX 只作方向核验（成交 ≥ 50 笔才看 t）；锚点只作校准。"]
    allrows = []
    rng = np.random.default_rng(20261126)
    for name, (df, ev0) in raw.items():
        log(f"build {name}")
        lx = LevelIndex(name, df, ev0)
        rows, notes = analyze(lx, rng)
        L += ["", "#" * 120, f"# 市场：{name}（{lx.mk.N} 个标的，评估起点 {ev0.date()}；来回成本 {lx.mk.cost * 100:.2f}%）", "#" * 120]
        L += render(name, rows, notes)
        allrows += rows
    txt = "\n".join(L)
    REPORT_PATH.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    flat = []
    for r in allrows:
        d = {k: v for k, v in r.items() if not isinstance(v, list)}
        for k in ["per", "halves"]:
            d[k] = ";".join(f"{v:+.4f}" if np.isfinite(v) else "nan" for v in r[k])
        flat.append(d)
    pd.DataFrame(flat).to_csv(SUMMARY_PATH, index=False)
    log(f"done ({time.time() - t0:.0f}s) -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
