#!/usr/bin/env python3
"""
Market Cycle V2 — Study 10（第三阶段专题 2 · 交易区间：区间识别 + 限价单自检 + SPX + 31 个申万一级行业）

预注册见 decisions.md「10-预注册」（方案全文 output/10_plan_for_review.md v2）。主规格开跑前写死：
  - 箱体：第 t 天用 t−20…t−1 的 20 根收盘；上 / 下沿 = 最高 / 最低收盘，中轴 = 正中，三等分。
    区间 = |C(t−1) − C(t−20)| ≤ 高度/3 + 收盘相对中轴符号翻转 ≥ 3 次（先剔除恰好等于中轴的点）+ 高度 ≥ 2×ATR20(t−1)；
    上涨趋势 = 净位移 ≥ 高度×2/3 + 翻转 ≤ 2 次 + 高度 ≥ 2×ATR20(t−1)；下跌趋势镜像；其余不贴标签。
  - Q1（s=+1）：区间里收盘落在下三分之一 → 在 t 日最低价挂买入限价单（t+1 跌破 1 跳成交；开盘已越过按开盘价），
    止损 / 目标 = 挂单价 ∓ 计划风险（1R）。空头镜像 = 上三分之一、在最高价挂卖单。专属对照 = 同位置、箱体是趋势。
  - Q2（s=−1）：05 的 20 日首次收盘新高（= 收盘高于箱体上沿、前 5 天无同类）在区间里 → t+1 越过高点追买（专题 1 引擎原样）；
    镜像 = 首次收盘新低在区间里 → 跌破低点卖出。专属对照 = 同向趋势里的同类突破。
  - 基线 = 同规则随机入场（Q1 新限价单随机池；Q2 专题 1 止损单随机池），剔除本题事件与对照。
    池化回归：净 R = 常数 + 事件哑变量 + 对照哑变量 + 近 20 日涨跌、20 日波动、近 5 日涨跌 + 几何五分位（切点取本侧随机池），
    交易日历对齐 Driscoll-Kraay（lag 30），绝对口径；专属对照 = 同一回归里两个系数之差。
  - SPX 不进 FDR，只作方向核验（成交 ≥ 50 笔报 t）；A 股 8 个主检验的 FDR 在 10b 与个股合并。

用法：
  python studies/10_trading_range_index.py --step0   # 引擎与区间识别自检 + 样本量（不看任何结果）
  python studies/10_trading_range_index.py           # 第 1 步完整结果
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
STEP0_PATH = OUTPUT_DIR / "10_step0_report.txt"
REPORT_PATH = OUTPUT_DIR / "10_trading_range_index_report.txt"
SUMMARY_PATH = OUTPUT_DIR / "10_trading_range_index_summary.csv"


def _load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "studies" / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


s09 = _load_mod("s09", "09_trade_ledger_index.py")
b8 = s09.b8
log = b8.log

EPS = 1e-12
BOX = 20
FLAT = 1.0 / 3.0
TREND = 2.0 / 3.0
MIN_CROSS = 3
MAX_CROSS_TREND = 2
MIN_HGT_ATR = 2.0
REL = 1e-9
ZERO = 1e-12
COOL = 5
BACK_N = 10
LAG = 30
MIN_EVENTS = 10
ST_NONE, ST_TR, ST_UP, ST_DN = 0, 1, 2, 3
SIDE_NAME = {1: "多头", -1: "空头镜像"}
K_TARGET, K_STOP, K_TIME, K_FORCED = s09.K_TARGET, s09.K_STOP, s09.K_TIME, s09.K_FORCED
# (question, side, ledger, event key, control key, Brooks direction s, label)
QUESTIONS = [
    ("Q1", 1, "L", "q1_ev", "q1_ctrl", 1, "区间下沿限价买"),
    ("Q1", -1, "L", "q1_ev", "q1_ctrl", 1, "区间上沿限价卖"),
    ("Q2", 1, "S", "q2_ev", "q2_ctrl", -1, "区间向上突破后追买"),
    ("Q2", -1, "S", "q2_ev", "q2_ctrl", -1, "区间向下跌破后卖出"),
]


# ---------------------------------------------------------------------------
# 1. Range detector (closes only; each day computed independently from its own 20 prior closes)
# ---------------------------------------------------------------------------
def range_features(c: np.ndarray, atr: np.ndarray) -> dict[str, np.ndarray]:
    """Box of day t = closes t−20 … t−1. atr = ATR20 series (the box test uses atr[t−1])."""
    n = len(c)
    f = {k: np.full(n, np.nan) for k in ["top", "bot", "mid", "hgt", "net", "cross", "atrp"]}
    if n > BOX:
        W = sliding_window_view(c, BOX)[:-1]  # row j = closes j … j+19, used on day t = j + 20
        top, bot = W.max(1), W.min(1)
        mid = 0.5 * (top + bot)
        D = W - mid[:, None]
        S = np.sign(D)
        S[np.abs(D) <= ZERO * np.abs(mid)[:, None]] = 0.0
        prev = np.zeros(len(W))
        cnt = np.zeros(len(W))
        for j in range(BOX):
            s = S[:, j]
            nz = (s != 0) & np.isfinite(s)
            cnt += nz & (prev != 0) & (s != prev)
            prev = np.where(nz, s, prev)
        sl = slice(BOX, n)
        f["top"][sl], f["bot"][sl], f["mid"][sl], f["hgt"][sl] = top, bot, mid, top - bot
        f["net"][sl] = W[:, -1] - W[:, 0]
        f["cross"][sl] = np.where(np.isfinite(top - bot), cnt, np.nan)
        f["atrp"][sl] = atr[BOX - 1:n - 1]
    hgt, net, cr = f["hgt"], f["net"], f["cross"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.isfinite(hgt) & np.isfinite(f["atrp"]) & (hgt > EPS)
        wide = ok & (hgt >= MIN_HGT_ATR * f["atrp"] * (1 - REL))
        flat = ok & (np.abs(net) <= hgt * (FLAT + REL))
        up = ok & (net >= hgt * (TREND - REL))
        dn = ok & (net <= -hgt * (TREND - REL))
        two = ok & (cr >= MIN_CROSS)
        one = ok & (cr <= MAX_CROSS_TREND)
        f["pos"] = np.where(ok, (c - f["bot"]) / hgt, np.nan)
        f["nr"] = np.where(ok, np.abs(net) / hgt, np.nan)
        f["hatr"] = np.where(ok, hgt / f["atrp"], np.nan)
    state = np.zeros(n, np.int8)
    state[wide & flat & two] = ST_TR
    state[wide & up & one] = ST_UP
    state[wide & dn & one] = ST_DN
    f["state"] = state
    f["wide"] = wide
    f["tight"] = ok & ~wide & flat & two
    f["c1_excl"] = wide & (up | dn) & two  # directional box that is also two-sided (excluded by C1)
    return f


def first_break(c: np.ndarray, atr: np.ndarray, em: np.ndarray) -> np.ndarray:
    """05 / 08b `brk_20`: close above the prior 20 closes' max, none in the prior 5 days, ATR20(t−1) valid, eval day."""
    n = len(c)
    if n < 30:
        return np.zeros(n, bool)
    prev_hi = pd.Series(c).rolling(BOX, min_periods=BOX).max().shift(1).to_numpy()
    is_nh = c > prev_hi
    recent = pd.Series(is_nh.astype(float)).shift(1).rolling(COOL, min_periods=COOL).sum().to_numpy() > 0
    atr0 = s09._shift(atr, 1)
    return is_nh & ~recent & np.isfinite(prev_hi) & em & np.isfinite(atr0) & (atr0 > EPS)


def side_view(h, l, c, side):
    if side > 0:
        return h, l, c, 0.0
    K = 2.0 * np.nanmax(h) + 1.0
    return K - l, K - h, K - c, K


def asset_side(h: np.ndarray, l: np.ndarray, c: np.ndarray, em: np.ndarray, side: int) -> dict[str, np.ndarray]:
    """All topic-2 features for one asset in the side's view (side −1 = mirrored prices, so 'bottom' = real top)."""
    hv, lv, cv, K = side_view(h, l, c, side)
    atr = s09.base_feats(hv, lv, cv)["atr20"]
    f = range_features(cv, atr)
    pos, st = f["pos"], f["state"]
    with np.errstate(invalid="ignore"):
        bot3 = (pos >= -REL) & (pos <= FLAT + REL)
        mid3 = (pos > FLAT + REL) & (pos < TREND - REL)
    brk = first_break(cv, atr, em)
    n = len(cv)
    back = np.full(n, np.nan)
    if n > BACK_N:
        fut_min = sliding_window_view(cv[1:], BACK_N).min(1)
        back[:len(fut_min)] = (fut_min < f["top"][:len(fut_min)]).astype(float)
    trend = (st == ST_UP) | (st == ST_DN)
    out = {
        "state": st.astype(float), "pos": pos, "nr": f["nr"], "cross": f["cross"], "hatr": f["hatr"],
        "wide": f["wide"], "c1_excl": f["c1_excl"],
        "q1_ev": em & (st == ST_TR) & bot3, "q1_ctrl": em & trend & bot3, "q1_mid": em & (st == ST_TR) & mid3,
        "bot3w": em & f["wide"] & bot3,
        "q2_ev": brk & (st == ST_TR), "q2_ctrl": brk & (st == ST_UP), "q2_tight": brk & f["tight"], "brk": brk,
        "back10": back, "mid_real": (f["mid"] if side > 0 else K - f["mid"]),
        # returns from REAL prices (ratios are not translation-invariant), signed to the side's view
        "ret1": side * (c / s09._shift(c, 1) - 1.0), "r5v": side * (c / s09._shift(c, 5) - 1.0),
    }
    return out


BOOL_KEYS = {"wide", "c1_excl", "q1_ev", "q1_ctrl", "q1_mid", "bot3w", "q2_ev", "q2_ctrl", "q2_tight", "brk"}


def wide_side_feats(H, L, C, dates, eval_start, side, min_len=30) -> dict[str, np.ndarray]:
    T, N = C.shape
    W: dict[str, np.ndarray] = {}
    em_all = np.asarray(dates >= np.datetime64(eval_start), bool)
    for n in range(N):
        rows = np.flatnonzero(np.isfinite(C[:, n]))
        if len(rows) < min_len:
            continue
        d = asset_side(H[rows, n], L[rows, n], C[rows, n], em_all[rows], side)
        for k, v in d.items():
            if k not in W:
                W[k] = np.zeros((T, N), bool) if k in BOOL_KEYS else np.full((T, N), np.nan)
            W[k][rows, n] = v
    return W


# ---------------------------------------------------------------------------
# 2. Self-tests: limit-order engine branch + range detector
# ---------------------------------------------------------------------------
def _paths(rng, T: int, N: int, gap_prob: float, m: int = 256):
    steps = rng.normal(0, 0.004 * math.sqrt(24 / m), size=(T, N, m))
    gaps = rng.normal(0, 0.006, size=(T, N)) * (rng.random((T, N)) < gap_prob)
    paths = np.empty((T, N, m + 1))
    prev = np.full(N, 100.0)
    for t in range(T):
        o = prev * np.exp(gaps[t])
        paths[t, :, 0] = o
        paths[t, :, 1:] = o[:, None] * np.exp(np.cumsum(steps[t], axis=1))
        prev = paths[t, :, -1]
    P = {"O": paths[:, :, 0].copy(), "H": paths.max(2), "L": paths.min(2), "C": paths[:, :, -1].copy(),
         "OK": np.ones((T, N), bool), "UP": np.full((T, N), np.nan), "DN": np.full((T, N), np.nan),
         "TICK": np.zeros((T, N))}
    return P, paths


_TRUTH_OS: list[float] = []


def _truth_limit(pp, s, n, k, t1, max_hold=s09.MAX_HOLD):
    """Exact outcome of the limit bracket on the continuous intraday path pp (long view). Returns (R, kind) or None."""
    ps = pp[s, n]
    level, rp = ps.min(), ps.max() - ps.min()
    stp, tg = level - rp, level + k * rp
    e = s + 1
    p = pp[e, n]
    if not (p < level).any():
        return None
    if p[0] < level:
        fill, seq = p[0], p
        overshoot = 0.0
    else:
        j = int(np.argmax(p < level))
        fill, seq = level, np.r_[level, p[j:]]
        overshoot = (level - p[j]) / rp  # rest of the down step that crossed the limit (path discretisation)
    big = 10 ** 9
    _TRUTH_OS.append(overshoot)
    if seq[0] <= stp:
        return ((pp[e + 1, n, 0] - fill) / rp, "forced") if t1 else (0.0, "stop")
    hs_, ht_ = seq <= stp, seq >= tg
    if t1:
        if hs_.any():
            return (pp[e + 1, n, 0] - fill) / rp, "forced"
    else:
        i_s = int(np.argmax(hs_)) if hs_.any() else big
        i_t = int(np.argmax(ht_)) if ht_.any() else big
        if i_s < i_t:
            return (stp - fill) / rp, "stop"
        if i_t < i_s:
            return (tg - fill) / rp, "target"
    for d in range(1, max_hold + 1):
        q = pp[e + d, n]
        if q[0] <= stp:
            return (q[0] - fill) / rp, "stop"
        if q[0] >= tg:
            return (q[0] - fill) / rp, "target"
        hs_, ht_ = q <= stp, q >= tg
        i_s = int(np.argmax(hs_)) if hs_.any() else big
        i_t = int(np.argmax(ht_)) if ht_.any() else big
        if i_s < i_t:
            return (stp - fill) / rp, "stop"
        if i_t < i_s:
            return (tg - fill) / rp, "target"
        if d == max_hold:
            return (q[-1] - fill) / rp, "time"
    return None


def selftest_limit() -> list[str]:
    out = []

    def chk(name, cond):
        assert cond, f"limit self-test failed: {name}"
        out.append(f"  ✔ {name}")

    toy, one = s09._toy, s09._one
    L = dict(entry="limit")
    sig = (9.5, 10.0, 9.0, 9.2)  # limit 9.0, planned risk 1.0, stop 8.0, target 10.0
    r = one(toy({5: sig, 6: (9.3, 9.5, 8.9, 9.4), 7: (9.5, 10.2, 9.3, 10.1)}), **L)
    chk("盘中成交于挂单价、次日到目标 = +1R", r["filled"] and abs(r["fill"] - 9.0) < 1e-9 and abs(r["gross"] - 1) < 1e-9
        and r["hold"] == 1 and not r["gap_in"])
    chk("指数：只碰到挂单价不成交（严格小于）", one(toy({5: sig, 6: (9.3, 9.5, 9.0, 9.4)}), **L)["why"] == s09.W_NOTRIG)
    chk("个股：最低价 = 挂单价 − 1 跳才成交", one(toy({5: sig, 6: (9.3, 9.5, 8.99, 9.4)}, tick=0.01), **L)["filled"]
        and one(toy({5: sig, 6: (9.3, 9.5, 9.00, 9.4)}, tick=0.01), **L)["why"] == s09.W_NOTRIG)
    r = one(toy({5: sig, 6: (8.6, 8.9, 8.5, 8.8), 7: (8.9, 10.1, 8.7, 10.0)}), **L)
    chk("开盘跳空低于挂单价 → 按开盘价成交，止损 / 目标价与 R 不变：+1.4R", abs(r["fill"] - 8.6) < 1e-9 and abs(r["R"] - 1.0) < 1e-9
        and abs(r["gross"] - 1.4) < 1e-9 and r["gap_in"])
    r = one(toy({5: sig, 6: (9.00, 9.2, 8.95, 9.1)}, tick=0.01), **L)
    chk("个股：开盘恰好等于挂单价不算开盘成交", r["filled"] and abs(r["fill"] - 9.0) < 1e-9 and not r["gap_in"])
    r = one(toy({5: sig, 6: (7.8, 8.2, 7.6, 8.0)}), **L)
    chk("开盘越过止损：开盘成交、开盘止损 = 毛 0R", abs(r["fill"] - 7.8) < 1e-9 and abs(r["xpx"] - 7.8) < 1e-9
        and abs(r["gross"]) < 1e-12 and r["kind"] == K_STOP and r["gstop"] and r["hold"] == 0)
    P = toy({5: sig, 6: (9.4, 9.5, 7.9, 8.1)})
    chk("成交当天只碰止损：保守 = 乐观 = −1R", abs(one(P, **L)["gross"] + 1) < 1e-9 and abs(one(P, both="opt", **L)["gross"] + 1) < 1e-9)
    P = toy({5: sig, 6: (9.4, 10.1, 8.9, 9.6)})
    rc, ro = one(P, **L), one(P, both="opt", **L)
    chk("盘中成交当天只碰目标：保守继续持有、乐观当天止盈", rc["hold"] == 1 and ro["hold"] == 0 and ro["kind"] == K_TARGET)
    P = toy({5: sig, 6: (9.4, 10.1, 7.9, 8.5)})
    chk("盘中成交当天两头都碰：保守 −1R、乐观 +1R", abs(one(P, **L)["gross"] + 1) < 1e-9 and abs(one(P, both="opt", **L)["gross"] - 1) < 1e-9)
    P = toy({5: sig, 6: (8.9, 10.1, 8.8, 10.0)})
    chk("开盘成交当天只碰目标：两种假设都止盈 +1.1R", abs(one(P, **L)["gross"] - 1.1) < 1e-9 and abs(one(P, both="opt", **L)["gross"] - 1.1) < 1e-9)
    P = toy({5: sig, 6: (8.9, 10.1, 7.9, 9.0)})
    chk("开盘成交当天两头都碰：保守 −0.9R、乐观 +1.1R", abs(one(P, **L)["gross"] + 0.9) < 1e-9 and abs(one(P, both="opt", **L)["gross"] - 1.1) < 1e-9)
    P = toy({5: sig, 6: (9.4, 9.5, 7.9, 8.1), 7: (8.3, 8.5, 8.2, 8.4)}, tick=0.01)
    rc, ro = one(P, t1="forced", **L), one(P, t1="forced", both="opt", **L)
    chk("T+1：成交当天跌破止损 → 次日开盘卖（保守乐观一致）", abs(rc["gross"] + 0.7) < 1e-9 and rc["kind"] == K_FORCED
        and abs(ro["gross"] - rc["gross"]) < 1e-12)
    P = toy({5: sig, 6: (9.4, 10.1, 8.9, 9.6)}, tick=0.01)
    chk("T+1：成交当天碰目标不算（次日才止盈）", one(P, t1="forced", **L)["hold"] == 1 and one(P, t1="forced", both="opt", **L)["hold"] == 1)
    chk("T+1 忽略变体：当天跌破止损不算", one(toy({5: sig, 6: (9.4, 9.5, 7.9, 8.1)}, tick=0.01), t1="ignore", **L)["kind"] != K_FORCED)
    P = toy({5: sig, 6: (8.28, 8.28, 8.28, 8.28), 7: (7.45, 7.45, 7.45, 7.45), 8: (7.2, 7.6, 7.1, 7.5)},
            dn={6: 8.28, 7: 7.45}, tick=0.01)
    r = one(P, **L)
    chk("一字跌停开盘成交、次日一字跌停越过止损 → 顺延到第 3 天开盘", abs(r["fill"] - 8.28) < 1e-9 and abs(r["xpx"] - 7.2) < 1e-9
        and r["hold"] == 2 and abs(r["gross"] + 1.08) < 1e-9)
    chk("次日停牌 → 未成交", one(toy({5: sig, 6: None}), **L)["why"] == s09.W_SUSP)
    sg2 = (9.5, 10.0, 9.0, 9.8)  # short: sell limit 10.0, stop 11.0, target 9.0
    r = one(toy({5: sg2, 6: (9.9, 10.2, 9.8, 10.1), 7: (10.0, 10.1, 8.9, 9.0)}), side=-1, **L)
    chk("空头镜像：最高价挂卖单、跌 1R 买回 = +1R", r["filled"] and abs(r["fill"] - 10.0) < 1e-9 and abs(r["gross"] - 1) < 1e-9)
    P = toy({5: (9.7, 10.0, 9.6, 9.8), 6: (10.78, 10.78, 10.78, 10.78), 7: (10.6, 10.9, 10.5, 10.7)}, up={6: 10.78}, tick=0.01)
    r = one(P, side=-1, **L)
    chk("空头镜像：一字涨停开盘越过止损、封死买不回 → 次日开盘买回", abs(r["fill"] - 10.78) < 1e-9 and abs(r["xpx"] - 10.6) < 1e-9
        and r["gstop"] and r["hold"] == 1 and abs(r["gross"] - 0.45) < 1e-9)
    r = one(toy({5: sig, 6: (9.3, 9.4, 8.9, 9.1), 7: (9.2, 9.6, 9.1, 9.5)}), tgt_px=np.array([9.5]), **L)
    chk("显式目标价（中轴 9.5）：+0.5R", abs(r["gross"] - 0.5) < 1e-9 and r["kind"] == K_TARGET)
    r = one(toy({5: sg2, 6: (9.9, 10.2, 9.8, 10.1), 7: (10.0, 10.05, 9.4, 9.6)}), side=-1, tgt_px=np.array([9.5]), **L)
    chk("空头镜像显式目标价同步取负：+0.5R", abs(r["gross"] - 0.5) < 1e-9 and r["kind"] == K_TARGET)
    r = one(toy({5: sig, 6: (9.3, 9.5, 8.9, 9.4), 7: (9.5, 10.2, 9.3, 10.1)}), cost=0.001, **L)
    chk("成本：净 R = 毛 R − 0.1% × 成交价 / 计划风险", abs((r["gross"] - r["net"]) - 0.001 * 9.0 / 1.0) < 1e-12)

    # random-walk truth bracket
    rng = np.random.default_rng(20261125)
    T, N = 260, 40
    tt, nn = np.meshgrid(np.arange(30, T - 25), np.arange(N), indexing="ij")
    tt, nn = tt.ravel(), nn.ravel()
    for gap_prob, tag in [(0.0, "无跳空"), (0.3, "有跳空")]:
        P, paths = _paths(rng, T, N, gap_prob)
        for side, t1 in [(1, None), (1, "forced"), (-1, None)]:
            rc = s09.run_ledger(P, tt, nn, side, entry="limit", both="cons", t1=t1, buffer=22)
            ro = s09.run_ledger(P, tt, nn, side, entry="limit", both="opt", t1=t1, buffer=22)
            pp = paths if side > 0 else -paths
            _TRUTH_OS.clear()
            tru = [_truth_limit(pp, s, n, 1.0, t1 is not None) for s, n in zip(tt, nn)]
            os_mean = float(np.mean(_TRUTH_OS))
            fil = np.array([x is not None for x in tru])
            assert (fil == rc["filled"]).all() and (fil == ro["filled"]).all(), "limit fill mismatch vs truth"
            val = np.array([x[0] if x else np.nan for x in tru])
            knd = np.array([x[1] if x else "" for x in tru])
            bad1 = ((knd == "target") & (ro["kind"] != K_TARGET)).sum()
            bad2 = ((knd == "stop") & (rc["kind"] != K_STOP)).sum()
            bad3 = ((knd == "forced") & ((rc["kind"] != K_FORCED) | (ro["kind"] != K_FORCED))).sum()
            same = fil & (rc["kind"] == ro["kind"]) & (np.abs(rc["xpx"] - ro["xpx"]) < 1e-9)
            bad4 = (same & (np.abs(rc["gross"] - val) > 1e-6)).sum()
            assert bad1 == bad2 == bad3 == bad4 == 0, f"limit truth bracket failed {tag} side={side} t1={t1}: {bad1},{bad2},{bad3},{bad4}"
            win_t = np.mean(knd[fil] == "target")
            mu, se = np.nanmean(val[fil]), np.nanstd(val[fil]) / math.sqrt(fil.sum())
            if gap_prob == 0 and t1 is None:
                assert 0.45 < win_t < 0.55, f"no-gap 1R truth win rate off: {win_t}"
            assert abs(mu) < 4 * se + 0.02, f"truth mean gross R not ~0: {mu}"
            out.append(f"  ✔ 随机游走真值夹逼（{tag}，{SIDE_NAME[side]}{'，T+1' if t1 else ''}）：{fil.sum():,} 笔成交，零违例；"
                       f"真实 1R 胜率 {win_t * 100:.1f}%，引擎 保守 {rc['win'][fil].mean() * 100:.1f}% ~ 乐观 {ro['win'][fil].mean() * 100:.1f}%；"
                       f"两种假设一致 {same.sum():,} 笔逐笔等于真值；真实毛 R 均值 {mu:+.3f}（标准误 {se:.3f}），"
                       f"其中模拟路径穿过挂单价那一步的下冲均值 {os_mean:.3f}R")
    return out


def selftest_range(raw) -> list[str]:
    out = []

    def chk(name, cond):
        assert cond, f"range self-test failed: {name}"
        out.append(f"  ✔ {name}")

    t = np.arange(60, dtype=float)
    atr = np.full(60, 0.2)
    rf = range_features(10 + 0.5 * np.sin(2 * np.pi * t / 9.5), atr)
    chk("标准区间（正弦、净位移 0、来回 4 次）→ 区间", (rf["state"][25:] == ST_TR).all())
    rf = range_features(10 + 0.1 * t, atr)
    chk("直线上涨 → 上涨趋势；镜像 → 下跌趋势", (rf["state"][25:] == ST_UP).all()
        and (range_features(10 - 0.1 * t, atr)["state"][25:] == ST_DN).all())
    v = np.r_[np.linspace(12, 10, 11), np.linspace(10, 12, 11)[1:], np.full(10, 12.0)]
    rf = range_features(v, np.full(len(v), 0.2))
    chk("V 字（20 天净位移 0、只穿中轴 2 次）→ 不贴标签", rf["state"][20] == ST_NONE and rf["cross"][20] == 2)
    rf = range_features(10 + 0.05 * np.sin(2 * np.pi * t / 9.5), atr)
    chk("太窄区间（高度 < 2 ATR）→ 不贴标签、记为太窄", (rf["state"][25:] == ST_NONE).all() and rf["tight"][25:].all())
    ch = np.r_[12.0, np.tile([10.2, 11.8], 9), 10.0, 10.0]
    rf = range_features(ch, np.full(len(ch), 0.2))
    chk("首尾在两端、中间来回震荡（C1）→ 不算下跌趋势也不算区间", rf["state"][20] == ST_NONE and rf["c1_excl"][20])
    z = np.r_[np.full(5, 10.5), 10.0, np.full(5, 9.5), 10.0, np.full(8, 10.5), 10.5]
    rf = range_features(z, np.full(len(z), 0.1))
    chk("恰好落在中轴的收盘不计入翻转（+→中轴→− 只算 1 次）", rf["cross"][20] == 2)
    lines_m, lines_p, lines_b = [], [], []
    for mk, (df, ev0) in raw.items():
        bad_m = bad_p = bad_b = tot_b = 0
        rng = np.random.default_rng(7)
        for asset, g in df.groupby("asset", sort=True):
            g = g.sort_values("date").reset_index(drop=True)
            h, l, c = (g[x].to_numpy(float) for x in ("high", "low", "close"))
            d = g["date"].to_numpy()
            em = np.asarray(pd.to_datetime(d) >= ev0, bool)
            a, b = asset_side(h, l, c, em, 1), asset_side(h, l, c, em, -1)
            sa, sb = a["state"], b["state"]
            mirror_ok = (np.array_equal(sa == ST_TR, sb == ST_TR) and np.array_equal(sa == ST_UP, sb == ST_DN)
                         and np.array_equal(sa == ST_DN, sb == ST_UP)
                         and np.array_equal(np.nan_to_num(a["cross"], nan=-1), np.nan_to_num(b["cross"], nan=-1)))
            bad_m += not mirror_ok
            for tcut in rng.choice(np.arange(60, len(c)), size=min(40, len(c) - 60), replace=False):
                p = asset_side(h[:tcut + 1], l[:tcut + 1], c[:tcut + 1], em[:tcut + 1], 1)
                for k in ["state", "pos", "cross", "q1_ev", "q1_ctrl", "q2_ev", "q2_ctrl"]:
                    x, y = p[k][tcut], a[k][tcut]
                    if not ((x == y) or (isinstance(x, float) and np.isnan(x) and np.isnan(y))):
                        bad_p += 1
            ev, _ = b8.detect_events(h, l, c, d, ev0)
            ref = set(ev["brk_20"].tolist())
            mine = set(np.flatnonzero(a["brk"]).tolist())
            tot_b += len(ref)
            bad_b += ref != mine
        lines_m.append(f"  {mk}：镜像对称（区间↔区间、上涨↔下跌、翻转次数）" + ("全部一致 ✔" if bad_m == 0 else f"{bad_m} 个标的不一致 ✘"))
        lines_p.append(f"  {mk}：截断到第 t 天重算（状态、位置、翻转、事件）与全样本一致" + ("✔" if bad_p == 0 else f"✘ {bad_p} 处"))
        lines_b.append(f"  {mk}：向量化首次收盘新高 vs 08b detect_events 的 brk_20（{tot_b:,} 条）" + ("全部一致 ✔" if bad_b == 0 else f"{bad_b} 个标的不一致 ✘"))
        assert bad_m == 0 and bad_p == 0 and bad_b == 0, f"range verification failed on {mk}"
    return out + lines_m + lines_p + lines_b


# ---------------------------------------------------------------------------
# 3. Index markets: features, ledgers, pooled regressions
# ---------------------------------------------------------------------------
class RangeIndex:
    def __init__(self, name: str, df: pd.DataFrame, eval_start: pd.Timestamp) -> None:
        self.mk = mk = s09.IndexMarket(name, df, eval_start)
        self.name = name
        self.feat = {side: wide_side_feats(mk.P["H"], mk.P["L"], mk.P["C"], mk.dates, eval_start, side) for side in (1, -1)}
        self.tt, self.nn = np.nonzero(mk.cand)
        self.year = mk.year[self.tt]
        self.runs: dict[tuple, dict] = {}

    def at(self, side: int, key: str) -> np.ndarray:
        return self.feat[side][key][self.tt, self.nn]

    def run(self, led: str, side: int, var: str) -> dict:
        key = (led, side, var)
        if key not in self.runs:
            kw = {"main": {}, "opt": {"both": "opt"}, "t1": {"t1": "forced"}, "nogap": {"use_open": False},
                  "open": {"entry": "open"}}[var]
            if led == "L":
                kw = {**kw, "entry": "limit"}
            self.runs[key] = s09.run_ledger(self.mk.P, self.tt, self.nn, side, cost=self.mk.cost, **kw)
        return self.runs[key]


def cap_w(mi: np.ndarray, grp: np.ndarray) -> np.ndarray:
    key = mi.astype(np.int64) * 16 + grp.astype(np.int64)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return np.minimum(1.0, b8.W_CAP / cnt[inv])


def dfit(y, Ds, X, mi, lag=LAG):
    """WLS + DK of y on [1, dummies, controls]; returns (coefs, t) of the dummies."""
    Ds = [np.asarray(d, float) for d in Ds]
    grp = np.zeros(len(y), int)
    for j, d in enumerate(Ds):
        grp[d > 0] = j + 1
    Z = np.column_stack([np.ones(len(y))] + Ds + ([X] if X.shape[1] else []))
    if any((d > 0).sum() < MIN_EVENTS for d in Ds[:1]):
        return np.full(len(Ds), np.nan), np.full(len(Ds), np.nan)
    b, t = b8.wls_dk(y, Z, cap_w(mi, grp), mi, lag)
    return b[1:1 + len(Ds)], t[1:1 + len(Ds)]


def pos_dummies(p: np.ndarray) -> np.ndarray:
    D = np.column_stack([p < -REL, (p >= -REL) & (p <= FLAT + REL), (p >= TREND - REL) & (p <= 1 + REL), p > 1 + REL]).astype(float)
    D[~np.isfinite(p)] = np.nan
    return D


def controls(rx: RangeIndex, side: int, rows: np.ndarray, ref: np.ndarray, layers: tuple) -> np.ndarray:
    mk, tt, nn = rx.mk, rx.tt[rows], rx.nn[rows]
    parts = []
    if "geo" in layers:
        for x in ["g_atr", "g_px", "kcls"]:
            v = mk.F[x][tt, nn]
            parts.append(s09._qdum(v, v[ref]))
    if "trend" in layers:
        lin = np.column_stack([mk.F[x][tt, nn] for x in ["r20", "vol20", "r5"]])
        sd = np.nanstd(lin, 0)
        parts.append((lin - np.nanmean(lin, 0)) / np.where(sd > 0, sd, 1))
    if "pos" in layers:
        parts.append(pos_dummies(rx.feat[side]["pos"][tt, nn]))
    return np.hstack(parts) if parts else np.zeros((len(rows), 0))


MAIN = ("geo", "trend")


def excess(rx: RangeIndex, side: int, run: dict, E: np.ndarray, C: np.ndarray | None, *, layers=MAIN, sub=None,
           ycol="net", contrast=False, y_override=None, extra_nuis: list | None = None):
    """Coefficient of the event dummy vs the baseline (rows that are neither event nor control nor extra nuisance)."""
    y_all = run[ycol] if y_override is None else y_override
    f = run["filled"] & np.isfinite(y_all)
    if sub is not None:
        f &= sub
    rows = np.flatnonzero(f)
    De = E[rows]
    Dc = (C[rows] & ~De) if C is not None else np.zeros(len(rows), bool)
    nuis = [x[rows] & ~De & ~Dc for x in (extra_nuis or [])]
    if De.sum() < MIN_EVENTS:
        return np.nan, np.nan
    ref = ~De & ~Dc
    for x in nuis:
        ref &= ~x
    X = controls(rx, side, rows, ref, layers)
    y = y_all[rows].astype(float)
    mi = rx.tt[rows]
    if contrast:
        if Dc.sum() < MIN_EVENTS:
            return np.nan, np.nan
        b, t = dfit(y, [De, De | Dc] + nuis, X, mi)
        return float(b[0]), float(t[0])
    b, t = dfit(y, [De, Dc] + nuis if C is not None else [De] + nuis, X, mi)
    return float(b[0]), float(t[0])


def abs_mean(rx: RangeIndex, y: np.ndarray, rows: np.ndarray) -> tuple[float, float]:
    rows = rows[np.isfinite(y[rows])]
    if len(rows) < 30:
        return np.nan, np.nan
    mi = rx.tt[rows]
    b, t = b8.wls_dk(y[rows].astype(float), np.ones((len(rows), 1)), cap_w(mi, np.zeros(len(rows), int)), mi, LAG)
    return float(b[0]), float(t[0])


def thin(rx: RangeIndex, run: dict, mask: np.ndarray) -> np.ndarray:
    """Non-overlapping subset within `mask`: per asset, skip new trades while one is open."""
    keep = np.zeros(len(mask), bool)
    idx = np.flatnonzero(mask & run["filled"])
    order = np.lexsort((rx.tt[idx], rx.nn[idx]))
    idx = idx[order]
    last_n, last_x = -1, -1
    for i in idx.tolist():
        n, t = rx.nn[i], rx.tt[i]
        if n != last_n:
            last_n, last_x = n, -1
        if t + 1 > last_x:
            keep[i] = True
            last_x = t + 1 + run["hold"][i]
    return keep


def periods(name):
    if name == "SW":
        return [(y, y) for y in range(2016, 2025)], [(2016, 2020), (2021, 2024)]
    return [(1984, 1992), (1993, 2001), (2002, 2010), (2011, 2019)], [(1984, 2001), (2002, 2019)]


def analyze(rx: RangeIndex) -> tuple[list[dict], list[str]]:
    mk = rx.mk
    rows_out, notes = [], []
    per, halves = periods(rx.name)
    yr = rx.year
    for q, side, led, ek, ck, s, lab in QUESTIONS:
        run = rx.run(led, side, "main")
        ro = rx.run(led, side, "opt")
        E, C = rx.at(side, ek), rx.at(side, ck)
        fl = run["filled"]
        base = fl & ~E & ~C
        r = {"market": rx.name, "q": q, "side": side, "s": s, "label": lab, "ledger": led,
             "n_sig": int(E.sum()), "n_fill": int((E & fl).sum()), "n_ctrl": int((C & ~E & fl).sum()),
             "fill_rate": float(fl[E].mean()) if E.any() else np.nan,
             "rand_fill_rate": float(fl.mean())}
        Ef = E & fl
        for nm, rr in [("", run), ("_opt", ro)]:
            f2 = rr["filled"]
            r["win" + nm] = float(rr["win"][E & f2].mean()) if (E & f2).any() else np.nan
            r["net" + nm] = float(np.nanmean(rr["net"][E & f2])) if (E & f2).any() else np.nan
            r["rwin" + nm] = float(rr["win"][f2 & ~E & ~C].mean())
            r["rnet" + nm] = float(np.nanmean(rr["net"][f2 & ~E & ~C]))
        r["ex"], r["ex_t"] = excess(rx, side, run, E, C)
        r["ex_opt"], r["ex_opt_t"] = excess(rx, side, ro, E, C)
        r["ctrl"], r["ctrl_t"] = excess(rx, side, run, E, C, contrast=True)
        r["ex_abs"], _ = excess(rx, side, run, E, C, layers=())
        r["ex_geo"], _ = excess(rx, side, run, E, C, layers=("geo",))
        if q == "Q1":
            r["ex_pos"], r["ex_pos_t"] = excess(rx, side, run, E, C, layers=MAIN + ("pos",))
            r["ctrl_pos"], _ = excess(rx, side, run, E, C, layers=MAIN + ("pos",), contrast=True)
        r["ex_nok"], _ = excess(rx, side, run, E, C, layers=("trend",))
        r["per"] = [excess(rx, side, run, E, C, sub=(yr >= a) & (yr <= b))[0] for a, b in per]
        r["halves"] = [excess(rx, side, run, E, C, sub=(yr >= a) & (yr <= b))[0] for a, b in halves]
        r["p"] = b8.p_two(r["ex_t"])
        if rx.name == "SW" and side > 0:
            r["ex_t1"], _ = excess(rx, side, rx.run(led, side, "t1"), E, C)
        if rx.name == "SPX":
            sub = yr >= s09.SPX_REAL_OPEN_FROM
            r["gap_on"], _ = excess(rx, side, run, E, C, sub=sub)
            r["gap_off"], _ = excess(rx, side, rx.run(led, side, "nogap"), E, C, sub=sub)
            r["n_fill_1019"] = int((Ef & sub).sum())
        if rx.name == "SW":
            # same-day demeaned ("choose industry") diagnostic
            y = run["net"].astype(float)
            d = pd.DataFrame({"t": rx.tt, "y": np.where(fl, y, np.nan)})
            g = d.groupby("t")["y"]
            cnt = g.transform("count").to_numpy()
            ydm = np.where(cnt >= 10, y - g.transform("mean").to_numpy(), np.nan)
            r["ex_sameday"], r["ex_sameday_t"] = excess(rx, side, run, E, C, y_override=ydm)
        # time exits, gap-through-stop share
        r["time_share"] = float((run["kind"][Ef] == K_TIME).mean()) if Ef.any() else np.nan
        if led == "L":
            r["gstop"] = float(run["gstop"][Ef].mean()) if Ef.any() else np.nan
            r["rgstop"] = float(run["gstop"][base].mean())
            nog = {**run, "filled": run["filled"] & ~run["gstop"]}
            r["ex_cancel"], _ = excess(rx, side, nog, E, C)
        # non-overlapping version
        keep = thin(rx, run, E) | thin(rx, run, C & ~E) | thin(rx, run, ~E & ~C)
        r["ex_thin"], r["ex_thin_t"] = excess(rx, side, {**run, "filled": run["filled"] & keep}, E, C)
        if q == "Q1":
            # tradeable label (long only) + order-type and entry diagnostics
            if side > 0:
                rows_e = np.flatnonzero(Ef)
                r["abs_c"], r["abs_c_t"] = abs_mean(rx, run["net"], rows_e)
                r["abs_o"], _ = abs_mean(rx, ro["net"], np.flatnonzero(E & ro["filled"]))
                net2 = run["net"] - mk.cost * np.abs(run["fill"]) / run["R"]
                r["abs_c2"], r["abs_c2_t"] = abs_mean(rx, net2, rows_e)
                r["label_trade"] = ("日线口径可交易" if (r["abs_c"] > 0 and r["abs_c_t"] >= 2)
                                    else ("不可交易" if r["abs_o"] <= 0 else "日线判断不了"))
            st_run = rx.run("S", side, "main")
            r["stop_ex"], r["stop_ex_t"] = excess(rx, side, st_run, E, C)
            r["stop_ctrl"], _ = excess(rx, side, st_run, E, C, contrast=True)
            op_run = rx.run("S", side, "open")
            r["open_ex"], r["open_ex_t"] = excess(rx, side, op_run, E, C)
            r["open_ctrl"], _ = excess(rx, side, op_run, E, C, contrast=True)
            M3 = rx.at(side, "q1_mid")
            r["mid3_ex"], r["mid3_ex_t"] = excess(rx, side, run, M3, None, extra_nuis=[E, C])
            # middle target: event + control rows only, contrast with trend / geometry controls
            sel = np.flatnonzero(E | C)
            tp = rx.at(side, "mid_real")[sel]
            rm = s09.run_ledger(mk.P, rx.tt[sel], rx.nn[sel], side, entry="limit", cost=mk.cost, tgt_px=tp)
            fm = rm["filled"] & np.isfinite(rm["net"])
            rr_ = np.flatnonzero(fm)
            if E[sel][rr_].sum() >= MIN_EVENTS and (~E[sel][rr_]).sum() >= MIN_EVENTS:
                Xm = controls(rx, side, sel[rr_], ~E[sel][rr_], MAIN)
                b, t = dfit(rm["net"][rr_].astype(float), [E[sel][rr_]], Xm, rx.tt[sel][rr_])
                r["midtgt_ctrl"], r["midtgt_ctrl_t"] = float(b[0]), float(t[0])
                r["midtgt_net_ev"] = float(np.nanmean(rm["net"][rr_][E[sel][rr_]]))
            # dose-response among wide-box bottom-third rows (any context)
            B3 = rx.at(side, "bot3w")
            nr, cr = rx.at(side, "nr"), rx.at(side, "cross")
            bins_nr = [B3 & (nr <= FLAT + REL), B3 & (nr > FLAT + REL) & (nr < TREND - REL), B3 & (nr >= TREND - REL)]
            bins_cr = [B3 & (cr <= 1), B3 & (cr == 2), B3 & (cr == 3), B3 & (cr >= 4)]
            for nm_, bins in [("dose_nr", bins_nr), ("dose_cr", bins_cr)]:
                f = run["filled"] & np.isfinite(run["net"])
                rows = np.flatnonzero(f)
                Ds = [bb[rows] for bb in bins]
                ref = ~np.any(Ds, axis=0)
                X = controls(rx, side, rows, ref, MAIN)
                if min(d.sum() for d in Ds) >= MIN_EVENTS:
                    grp = np.zeros(len(rows), int)
                    for j, d in enumerate(Ds):
                        grp[d] = j + 1
                    Z = np.column_stack([np.ones(len(rows))] + [d.astype(float) for d in Ds] + [X])
                    b, t = b8.wls_dk(run["net"][rows].astype(float), Z, cap_w(rx.tt[rows], grp), rx.tt[rows], LAG)
                    r[nm_] = [float(x) for x in b[1:1 + len(Ds)]]
                    r[nm_ + "_t"] = [float(x) for x in t[1:1 + len(Ds)]]
                    r[nm_ + "_n"] = [int(d.sum()) for d in Ds]
        else:
            # failure rates (Brooks language): quick ledger failure + 10-day close back inside the box
            for nm, rr in [("", run), ("_opt", ro)]:
                f2 = rr["filled"]
                r["fail_ev" + nm] = float((rr["kind"][E & f2] == K_STOP).mean()) if (E & f2).any() else np.nan
                r["fail_ctrl" + nm] = float((rr["kind"][C & ~E & f2] == K_STOP).mean()) if (C & ~E & f2).any() else np.nan
                r["fail_rand" + nm] = float((rr["kind"][f2 & ~E & ~C] == K_STOP).mean())
            bk = rx.at(side, "back10")
            brk = rx.at(side, "brk")
            r["back_ev"], r["back_ctrl"], r["back_all"] = (float(np.nanmean(bk[m])) if m.any() else np.nan
                                                           for m in (E, C & ~E, brk))
            # fade translation: opposite-side limit order at the breakout bar's extreme
            lr = rx.run("L", -side, "main")
            r["fade_ex"], r["fade_ex_t"] = excess(rx, -side, lr, E, None)
            r["fade_n"] = int((E & lr["filled"]).sum())
            TB = rx.at(side, "q2_tight")
            r["tight_ex"], r["tight_ex_t"] = excess(rx, side, run, TB, None, extra_nuis=[E, C])
            r["tight_n"] = int((TB & fl).sum())
        rows_out.append(r)
    return rows_out, notes


def order_type_compare(rx: RangeIndex) -> list[str]:
    """Descriptive: random-pool simple return per trade (net of cost, %) — limit fades vs stop follows (R units differ)."""
    L = [f"\n  【{rx.name} · 限价单 vs 止损单的随机池（描述，不是检验）】每笔扣成本简单收益率（保守~乐观），成交率"]
    for side, nm in [(1, "买"), (-1, "卖")]:
        parts = []
        for led, lab in [("L", "限价反向"), ("S", "止损顺势")]:
            v = []
            for var in ("main", "opt"):
                r = rx.run(led, side, var)
                f = r["filled"]
                v.append(np.nanmean(r["net"][f] * r["R"][f] / np.abs(r["fill"][f])) * 100)
            parts.append(f"{lab}{nm} {v[0]:+.3f}%~{v[1]:+.3f}%（成交率 {rx.run(led, side, 'main')['filled'].mean() * 100:.0f}%）")
        L.append("  " + " | ".join(parts))
    return L


# ---------------------------------------------------------------------------
# 4. Report
# ---------------------------------------------------------------------------
def f_(x, nd=3, pct=False):
    return s09.fmt(x, nd, pct)


def render(name: str, rows: list[dict]) -> list[str]:
    L = []
    per_lab = "2016…2024 逐年" if name == "SW" else "1984-92/93-01/02-10/11-19"
    for q in ("Q1", "Q2"):
        L.append(f"\n  【{name} · {q}】超额 = 比同规则随机多出的每笔净 R（控趋势 / 波动 / 几何，DK lag 30）；专属对照 = 区间 − 趋势（同位置 / 同类事件）；"
                 f"Brooks 方向 s = {'+1' if q == 'Q1' else '−1'}")
        L.append(f"  {'方向':<16s}{'信号':>7s}{'成交':>7s}{'对照成交':>8s}{'成交率':>7s}{'胜率 保守~乐观':>16s}{'随机胜率':>9s}"
                 f"{'净R 保守~乐观':>18s}{'随机净R':>9s}{'超额':>8s}{'t':>6s}{'乐观超额':>9s}{'区间−趋势':>10s}{'t':>6s}")
        for r in [x for x in rows if x["q"] == q]:
            L.append(f"  {r['label']:<14s}{r['n_sig']:>7,}{r['n_fill']:>7,}{r['n_ctrl']:>8,}{r['fill_rate'] * 100:>6.0f}%"
                     f"{r['win'] * 100:>9.1f}%~{r['win_opt'] * 100:.1f}%{r['rwin'] * 100:>8.1f}%"
                     f"{f_(r['net']):>10s}~{f_(r['net_opt'])}{f_(r['rnet']):>9s}{f_(r['ex']):>8s}{f_(r['ex_t'], 1):>6s}"
                     f"{f_(r['ex_opt']):>9s}{f_(r['ctrl']):>10s}{f_(r['ctrl_t'], 1):>6s}")
        L.append("  拆解（超额）：绝对 → +几何 → +趋势 / 波动（主规格）" + (" → +箱体位置" if q == "Q1" else "")
                 + "；只控趋势不控几何；不重叠版本" + ("；同日去均值（选行业）" if name == "SW" else ""))
        for r in [x for x in rows if x["q"] == q]:
            extra = f" → {f_(r['ex_pos'])}（t {f_(r['ex_pos_t'], 1)}；区间−趋势 {f_(r['ctrl_pos'])}）" if q == "Q1" else ""
            sd = f"；同日 {f_(r['ex_sameday'])}（t {f_(r['ex_sameday_t'], 1)}）" if name == "SW" else ""
            L.append(f"  {r['label']:<14s} {f_(r['ex_abs'])} → {f_(r['ex_geo'])} → {f_(r['ex'])}{extra}；只控趋势 {f_(r['ex_nok'])}；"
                     f"不重叠 {f_(r['ex_thin'])}（t {f_(r['ex_thin_t'], 1)}）{sd}")
        L.append(f"  稳定性（{per_lab} 超额方向 | 前后两段）" + ("；SPX 2010–2019 计跳空 vs 不计跳空" if name == "SPX" else "")
                 + ("；行业 T+1" if name == "SW" else ""))
        for r in [x for x in rows if x["q"] == q]:
            sg = "".join("+" if (np.isfinite(v) and v > 0) else ("−" if np.isfinite(v) else "·") for v in r["per"])
            npos = sum(1 for v in r["per"] if np.isfinite(v) and r["s"] * v > 0)
            nv = sum(1 for v in r["per"] if np.isfinite(v))
            ex = ""
            if name == "SPX":
                ex = f" | 2010–19：计跳空 {f_(r['gap_on'])} 不计跳空 {f_(r['gap_off'])}（成交 {r['n_fill_1019']}）"
            if name == "SW" and "ex_t1" in r:
                ex = f" | T+1 {f_(r['ex_t1'])}"
            L.append(f"  {r['label']:<14s} {sg:<10s} Brooks 方向 {npos}/{nv} | {' / '.join(f_(v) for v in r['halves'])}{ex}")
        if q == "Q1":
            L.append("  诊断：止损单版本（超额 / 区间−趋势）| t+1 开盘进场（超额 / 区间−趋势）| 中间三分之一限价 | 中轴目标（区间−趋势，事件净R）"
                     "| 开盘越过止损占比（形态 / 随机）、视为撤单后超额 | 到期占比")
            for r in [x for x in rows if x["q"] == q]:
                L.append(f"  {r['label']:<14s} {f_(r['stop_ex'])}（t {f_(r['stop_ex_t'], 1)}）/ {f_(r['stop_ctrl'])} | "
                         f"{f_(r['open_ex'])}（t {f_(r['open_ex_t'], 1)}）/ {f_(r['open_ctrl'])} | {f_(r['mid3_ex'])}（t {f_(r['mid3_ex_t'], 1)}）| "
                         f"{f_(r.get('midtgt_ctrl', np.nan))}（t {f_(r.get('midtgt_ctrl_t', np.nan), 1)}），{f_(r.get('midtgt_net_ev', np.nan))} | "
                         f"{r['gstop'] * 100:.2f}% / {r['rgstop'] * 100:.2f}%，{f_(r['ex_cancel'])} | {r['time_share'] * 100:.1f}%")
            L.append("  剂量-反应（箱体 ≥ 2 ATR、收盘在下三分之一的全部日子，相对其余随机单）：净位移/高度 [0,1/3] (1/3,2/3) [2/3,1] | 翻转 0–1 / 2 / 3 / 4+")
            for r in [x for x in rows if x["q"] == q]:
                if "dose_nr" in r:
                    a = " ".join(f"{f_(v)}({f_(t, 1)},n{n})" for v, t, n in zip(r["dose_nr"], r["dose_nr_t"], r["dose_nr_n"]))
                    b = " ".join(f"{f_(v)}({f_(t, 1)},n{n})" for v, t, n in zip(r["dose_cr"], r["dose_cr_t"], r["dose_cr_n"]))
                    L.append(f"  {r['label']:<14s} {a} | {b}")
            for r in [x for x in rows if x["q"] == q and x["side"] > 0]:
                L.append(f"  可交易标签（只看多头，绝对每笔净 R）：保守 {f_(r['abs_c'])}（t {f_(r['abs_c_t'], 1)}），乐观 {f_(r['abs_o'])}，"
                         f"2 倍成本 {f_(r['abs_c2'])} → 【{r['label_trade']}】")
        else:
            L.append("  失败率（Brooks 说区间里 80% 突破失败）：账本快速失败 = 先碰止损（保守~乐观）区间突破 / 趋势突破 / 随机单 | "
                     "10 天内收盘跌回箱体：区间 / 趋势 / 全部首次新高")
            for r in [x for x in rows if x["q"] == q]:
                L.append(f"  {r['label']:<14s} {r['fail_ev'] * 100:.1f}~{r['fail_ev_opt'] * 100:.1f}% / "
                         f"{r['fail_ctrl'] * 100:.1f}~{r['fail_ctrl_opt'] * 100:.1f}% / {r['fail_rand'] * 100:.1f}~{r['fail_rand_opt'] * 100:.1f}% | "
                         f"{r['back_ev'] * 100:.1f}% / {r['back_ctrl'] * 100:.1f}% / {r['back_all'] * 100:.1f}%")
            L.append("  诊断：反着做（突破 K 线极值挂反向限价单，超额，成交笔数）| 太窄箱体的突破（超额，成交）| 到期占比")
            for r in [x for x in rows if x["q"] == q]:
                L.append(f"  {r['label']:<14s} {f_(r['fade_ex'])}（t {f_(r['fade_ex_t'], 1)}，{r['fade_n']}）| "
                         f"{f_(r['tight_ex'])}（t {f_(r['tight_ex_t'], 1)}，{r['tight_n']}）| {r['time_share'] * 100:.1f}%")
    return L


# ---------------------------------------------------------------------------
# 5. Step 0 and main
# ---------------------------------------------------------------------------
def load_raw():
    return s09.load_raw()


def step0(raw) -> list[str]:
    L = ["=" * 110, "Study 10 · 第 0 步：引擎与区间识别自检 + 样本量（不看任何结果）", "=" * 110,
         "", "[1] 专题 1 账本引擎自检（原分支）"]
    L += s09.selftest()
    L += ["", "[2] 限价单分支自检"]
    L += selftest_limit()
    L += ["", "[3] 区间识别自检"]
    L += selftest_range(raw)
    L += ["", "[4] 样本量与特征分布（只看计数与事前特征，不看任何账本结果）"]
    for name, (df, ev0) in raw.items():
        rx = RangeIndex(name, df, ev0)
        mk = rx.mk
        st = rx.feat[1]["state"]
        evd = (mk.dates >= np.datetime64(ev0))[:, None] & np.isfinite(st)
        L.append(f"  —— {name}（{mk.N} 个标的；合格 K 线 {mk.cand.sum():,}）")
        yrs = pd.DatetimeIndex(mk.dates).year.to_numpy()
        tab = []
        for y in np.unique(yrs[mk.dates >= np.datetime64(ev0)]):
            m = evd & (yrs == y)[:, None]
            v = st[m]
            tab.append(f"{y}: 区间 {np.mean(v == ST_TR) * 100:4.1f}% 上 {np.mean(v == ST_UP) * 100:4.1f}% 下 {np.mean(v == ST_DN) * 100:4.1f}%")
        v = st[evd]
        L.append(f"    全段：区间 {np.mean(v == ST_TR) * 100:.1f}%  上涨趋势 {np.mean(v == ST_UP) * 100:.1f}%  下跌趋势 {np.mean(v == ST_DN) * 100:.1f}%  "
                 f"不贴标签 {np.mean(v == ST_NONE) * 100:.1f}%")
        for i in range(0, len(tab), 3):
            L.append("    " + " | ".join(tab[i:i + 3]))
        ha = rx.feat[1]["hatr"][evd]
        ha = ha[np.isfinite(ha)]
        L.append(f"    箱体高度 / ATR 分位（10/25/50/75/90%）：{' / '.join(f'{x:.2f}' for x in np.quantile(ha, [.1, .25, .5, .75, .9]))}；"
                 f"高度 < 2 ATR 占 {np.mean(ha < MIN_HGT_ATR) * 100:.1f}%")
        c1 = rx.feat[1]["c1_excl"][evd]
        L.append(f"    「净位移 ≥ 2/3 但翻转 ≥ 3 次」（C1 排除、两边都不贴）占全部日子 {np.mean(c1) * 100:.2f}%")
        for q, side, led, ek, ck, s, lab in QUESTIONS:
            run = rx.run(led, side, "main")
            fl = run["filled"]
            E, C = rx.at(side, ek), rx.at(side, ck)
            Cx = C & ~E
            line = (f"    {q} {lab:<12s} 事件 {int(E.sum()):>6,} → 成交 {int((E & fl).sum()):>6,} | 对照 {int(Cx.sum()):>7,} → 成交 {int((Cx & fl).sum()):>7,}"
                    f" | 随机池成交率 {fl.mean() * 100:.1f}%")
            if led == "L":
                line += f" | 开盘越过止损：事件 {run['gstop'][E & fl].mean() * 100:.2f}% 随机 {run['gstop'][fl].mean() * 100:.2f}%"
            if name == "SPX":
                sub = rx.year >= s09.SPX_REAL_OPEN_FROM
                line += f" | 2010–19 事件成交 {int((E & fl & sub).sum())}"
            L.append(line)
            if q == "Q1":
                r1, r5 = rx.at(side, "ret1"), rx.at(side, "r5v")
                L.append(f"       事前特征（本侧视角）：第 t 天涨跌均值 事件 {np.nanmean(r1[E]) * 100:+.2f}% 对照 {np.nanmean(r1[Cx]) * 100:+.2f}%；"
                         f"近 5 日 事件 {np.nanmean(r5[E]) * 100:+.2f}% 对照 {np.nanmean(r5[Cx]) * 100:+.2f}%；"
                         f"中间三分之一（诊断）{int(rx.at(side, 'q1_mid').sum()):,}")
            else:
                L.append(f"       太窄箱体的突破（诊断）{int(rx.at(side, 'q2_tight').sum()):,}；全部首次突破 {int(rx.at(side, 'brk').sum()):,}")
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
    L = ["=" * 120, "Study 10 · 第 1 步：交易区间（SPX + 31 个申万一级行业，开发段）", "=" * 120,
         "Q1 = 区间贴边限价单反向做（Brooks 方向 +）；Q2 = 区间突破后止损单追（Brooks 方向 −，避坑）。1R / 20 天到期 / 同根双碰先算止损 / 扣成本。",
         "SPX 只作跨市场方向核验（不进 FDR，成交 ≥ 50 笔才看 t）；A 股行业的 BH-FDR 在 10b 与个股合并（共 8 个）。"]
    allrows = []
    for name, (df, ev0) in raw.items():
        log(f"build {name}")
        rx = RangeIndex(name, df, ev0)
        rows, notes = analyze(rx)
        L += ["", "#" * 120, f"# 市场：{name}（{rx.mk.N} 个标的，评估起点 {ev0.date()}；来回成本 {rx.mk.cost * 100:.2f}%）", "#" * 120]
        L += notes + render(name, rows) + order_type_compare(rx)
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
