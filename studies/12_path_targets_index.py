#!/usr/bin/env python3
"""
Market Cycle V2 — Study 12（第三阶段专题 4 · 路径形状与目标位：结构识别 + 零假设 + 自检 + SPX + 31 个申万一级行业）

预注册见 decisions.md「12-预注册」（方案全文 output/12_plan_for_review.md v2 冻结版）。主规格开跑前写死：
  - 统一尺子 z_t = ATR20(t−1)；Q2 / Q3a 直接调用未取整 s10.asset_side，新写 Q1 / Q3b 及全部 Y 判定取 6 位小数 + 1e-7 容差。
  - 观察窗按市场交易日历（10 天窗 ≥ 8 日、20 天窗 ≥ 15 日、条件两段固定 10 天窗 ≥ 8 日、40 天窗 ≥ 30 日）；指数缓冲 31 天。
  - 5 组命题 × 多空两侧（Brooks 方向全为正；行业 10 个进 A 股 20 检验 FDR，在 12b 与个股合并；SPX 10 个只看方向，事件 ≥ 50 报 t）：
      Q1a 高潮后至少歇 10 根（TB）：10 日最高收盘 ≤ C_t + 1.0·z_t；主对照 = 横盘起跳大阳 / 大阴线。
      Q1b 高潮后两段式调整（TL）：20 日收盘在未先超 +1.0·z_t 前走出「跌 ≥ 1.0·z → 弹 ≥ 0.5·z → 破低」；过关附加固定 10 日条件两段率。
      Q2  区间突破确认后等距目标（MM-TR）：s10 区间首次突破后 3 标的日内首次二次收盘站稳箱体外（确认日 t*），目标 = top + hgt，d ≥ 0.5·z；
          主对照 = 趋势箱体同类确认日（共同 k 支撑区间）。
      Q3a 区间中轴磁吸：s10 区间下 / 上三分之一，目标 = 中轴 mid，d ≥ 0.5·z；主对照 = 上涨箱体回落到下三分之一（镜像对称）。
      Q3b 前波段高 / 低点磁吸：收盘后未消耗活价位（复现 s11.detect_A），从 1.5·z 外首次落入 [0.5, 1.5]·z；主对照 = 结构安慰剂。
  - Q2 / Q3 主裁判 Y_net = I(先触目标) − I(先触等距反向) ∈ {+1, −1, 0}，先用开盘价判跳空，同根双碰保守记 −1（另报乐观 +1 与记 0 版）。
  - 主零假设 = 分层链式 5 天分块自助法（真实价格生成再镜像），双样本 t0 ≥ 2；账本仅作止盈诊断，studies/09、10、11 不改。

用法：
  python studies/12_path_targets_index.py --step0   # 哈希 / 手工 / 镜像 / 截断 / 校准 / 假数据端到端 / 样本量与平衡（不看真实事件 Y）
  python studies/12_path_targets_index.py           # 第 1 步完整结果（SPX + 31 申万一级行业）
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
STEP0_PATH = OUTPUT_DIR / "12_step0_report.txt"
REPORT_PATH = OUTPUT_DIR / "12_path_targets_index_report.txt"
SUMMARY_PATH = OUTPUT_DIR / "12_path_targets_index_summary.csv"


def _load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "studies" / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


s11 = _load_mod("s11", "11_level_memory_index.py")
s10 = s11.s10
s09 = s10.s09
b8 = s09.b8
log = b8.log

EPS = 1e-12
ROUND = 6
TIE = 1e-7
WIN_TB, MIN_TB = 10, 8
WIN_MAIN, MIN_MAIN = 20, 15
WIN_COND, MIN_COND = 10, 8
WIN_SENS, MIN_SENS = 40, 30
IDX_BUF = 31
IDX_BUF_40 = 41
LAG = 30
LAG_40 = 40
MIN_EVENTS = 10
MIN_SPX = 50
COOL = 5
CONFIRM_WIN = 3
D_MIN_ATR = 0.5
MAG_LO, MAG_HI = 0.5, 1.5
PLAC_SEP = 0.5
Q2_MULTS = (0.6, 0.8, 1.0, 1.2, 1.4)
SW_DAY_CAP = 5
W_PAST_TGT = 6

# Primary 10 index tests: (code, qgroup, side, label, ev_key, ctrl_key, rdir_rel)
# rdir_rel = direction of target relative to `side` view (+1 means target is ABOVE c_t in side view; -1 means BELOW c_t)
PRIMARY = [
    ("Q1a_L", "Q1a",  1, "Q1a 买盘高潮歇10根(TB)",  "q1_ev",  "q1_ctrl",  1),
    ("Q1a_S", "Q1a", -1, "Q1a 卖盘高潮歇10根(TB)",  "q1_ev",  "q1_ctrl",  1),
    ("Q1b_L", "Q1b",  1, "Q1b 买盘高潮两段(TL)",    "q1_ev",  "q1_ctrl",  1),
    ("Q1b_S", "Q1b", -1, "Q1b 卖盘高潮两段(TL)",    "q1_ev",  "q1_ctrl",  1),
    ("Q2_L",  "Q2",   1, "Q2 区间向上确认等距",     "q2_ev",  "q2_ctrl",  1),
    ("Q2_S",  "Q2",  -1, "Q2 区间向下确认等距",     "q2_ev",  "q2_ctrl",  1),
    ("Q3a_L", "Q3a",  1, "Q3a 区间下沿→中轴",       "q3a_ev", "q3a_ctrl", 1),
    ("Q3a_S", "Q3a", -1, "Q3a 区间上沿→中轴",       "q3a_ev", "q3a_ctrl", 1),
    ("Q3b_L", "Q3b",  1, "Q3b 吸向前波段低点",      "q3b_ev", "q3b_ctrl", -1),
    ("Q3b_S", "Q3b", -1, "Q3b 吸向前波段高点",      "q3b_ev", "q3b_ctrl", -1),
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _cool(raw: np.ndarray, k: int = COOL) -> np.ndarray:
    """Exact same raw-condition cooling as s10.first_break: raw[t] & no raw in t-k..t-1."""
    if len(raw) == 0:
        return raw.copy()
    rec = pd.Series(raw.astype(float)).shift(1).rolling(k, min_periods=1).sum().fillna(0.0).to_numpy() > 0
    return raw & ~rec


# ---------------------------------------------------------------------------
# 1. Structure detectors (one asset, one side view, own trading days; causal)
# ---------------------------------------------------------------------------
def detect_A_magnet(h: np.ndarray, l: np.ndarray, c: np.ndarray, atr: np.ndarray, em: np.ndarray) -> dict[str, np.ndarray]:
    """Per-level causal tracker reproducing s11.detect_A (A_narm/A_toparm/A_test) + Q3b walk-in magnet & structural placebo.
    Prices h, l, c must already be rounded to ROUND decimals in the side view (support side: level L below C_t).
    """
    n = len(c)
    out = {k: np.zeros(n, bool) for k in [
        "A_test", "q3b_ev", "q3b_in", "q3b_any", "q3b_ctrl",
    ]}
    out.update({k: np.full(n, np.nan) for k in [
        "A_narm", "A_toparm", "q3b_d", "q3b_lvl", "q3b_age", "q3b_ntouch", "q3b_ctrl_d", "q3b_ctrl_lvl",
    ]})
    piv = s11.pivots_low(l)
    H, Lw, C, A, EM = h.tolist(), l.tolist(), c.tolist(), atr.tolist(), em.tolist()
    fin = math.isfinite
    # real levels: [L, j, atr_j, mh, armed, consumed, mag_fired]
    lv_ = []
    # structural placebo levels (non-pivot lows): same schema
    pl_ = []
    PIV, BOUNCE, TOL, AGE_MIN, AGE_MAX = s11.PIV, s11.BOUNCE, s11.TOL, s11.AGE_MIN, s11.AGE_MAX

    for t in range(n):
        j = t - PIV - 1
        if j >= PIV and fin(A[j]) and A[j] > EPS:
            lv = Lw[j]
            if not (c[j + 1:t] < lv).any():
                mh = float(h[j + 1:t].max())
                item = [lv, j, A[j], mh, mh >= lv + BOUNCE * A[j] - TIE, False, False]
                if piv[j]:
                    lv_.append(item)
                else:
                    pl_.append(item)
        if lv_:
            lv_ = [x for x in lv_ if t - x[1] <= AGE_MAX]
        if pl_:
            pl_ = [x for x in pl_ if t - x[1] <= AGE_MAX]

        # Pre-bar-t armed summary (exact match to s11.detect_A)
        arm = [x[0] for x in lv_ if x[4] and not x[5]]
        out["A_narm"][t] = len(arm)
        out["A_toparm"][t] = max(arm) if arm else np.nan

        z = A[t - 1] if t >= 1 else float("nan")
        z_ok = t >= 1 and fin(z) and z > EPS
        lt, ct, ht = Lw[t], C[t], H[t]

        # s11 consumption on day t (before post-close removal)
        if z_ok and lv_:
            touched = [x for x in lv_ if x[4] and not x[5] and lt <= x[0] + TOL * z + TIE]
            if touched:
                top = max(touched, key=lambda x: x[0])
                for x in touched:
                    x[5] = True
                if AGE_MIN <= t - top[1] <= AGE_MAX and EM[t]:
                    out["A_test"][t] = True
        if z_ok and pl_:
            for x in pl_:
                if x[4] and not x[5] and lt <= x[0] + TOL * z + TIE:
                    x[5] = True

        # Post-close survival: drop levels closed below on day t (BEFORE updating mh/armed with day t!)
        if lv_:
            lv_ = [x for x in lv_ if not (ct < x[0])]
        if pl_:
            pl_ = [x for x in pl_ if not (ct < x[0])]

        # Q3b magnet check on post-close surviving, unconsumed, pre-armed levels (age 10..60)
        if z_ok:
            cp = C[t - 1]
            # Any live (unbroken, unconsumed, age <= 60, INCLUDING unarmed) real level within [0, 2.0]*z below ct
            out["q3b_any"][t] = any((not x[5]) and (0.0 - TIE <= ct - x[0] <= 2.0 * z + TIE) for x in lv_)

            walk, already = [], []
            for x in lv_:
                if x[4] and (not x[5]) and (not x[6]) and (AGE_MIN <= t - x[1] <= AGE_MAX):
                    dt = ct - x[0]
                    if MAG_LO * z - TIE <= dt <= MAG_HI * z + TIE:
                        if cp - x[0] > MAG_HI * z + TIE:
                            walk.append(x)
                        else:
                            already.append(x)
            if already and not walk and EM[t]:
                out["q3b_in"][t] = True
            if walk:
                best = max(walk, key=lambda x: x[0])  # nearest (highest L below ct)
                for x in walk:
                    x[6] = True
                if EM[t]:
                    out["q3b_ev"][t] = True
                    out["q3b_d"][t] = ct - best[0]
                    out["q3b_lvl"][t] = best[0]
                    out["q3b_age"][t] = t - best[1]
                    out["q3b_ntouch"][t] = len(walk)

            # Structural placebo: same walk-in rule on non-pivot lows, separated >= 0.5*z from any live real level
            if pl_ and not out["q3b_ev"][t]:
                real_lvls = [x[0] for x in lv_ if not x[5]]
                pwalk = []
                for x in pl_:
                    if x[4] and (not x[5]) and (not x[6]) and (AGE_MIN <= t - x[1] <= AGE_MAX):
                        dt = ct - x[0]
                        if (MAG_LO * z - TIE <= dt <= MAG_HI * z + TIE) and (cp - x[0] > MAG_HI * z + TIE):
                            if all(abs(x[0] - rl) >= PLAC_SEP * z - TIE for rl in real_lvls):
                                pwalk.append(x)
                if pwalk:
                    bestp = max(pwalk, key=lambda x: x[0])
                    for x in pwalk:
                        x[6] = True
                    if EM[t]:
                        out["q3b_ctrl"][t] = True
                        out["q3b_ctrl_d"][t] = ct - bestp[0]
                        out["q3b_ctrl_lvl"][t] = bestp[0]

        # Post-close mh / armed update (for day t+1 onwards, exact match to s11.detect_A)
        for pool in (lv_, pl_):
            for x in pool:
                if ht > x[3]:
                    x[3] = ht
                if not x[4] and x[3] >= x[0] + BOUNCE * x[2] - TIE:
                    x[4] = True
    return out


def detect_leg_mm(h: np.ndarray, l: np.ndarray, c: np.ndarray, atr: np.ndarray, em: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagnostic: Leg 1 = Leg 2 (AB=CD) on rounded side-view prices.
    Most recent confirmed swing high B (j_B + 5 <= t - 1); most recent prior confirmed swing low A (j_A < j_B) with
    H_B - L_A >= 2*ATR(j_A) and no close < L_A through t-1; pullback <= H_B - 1*ATR(j_B); first close > H_B at t (t - j_A <= 60).
    """
    n = len(c)
    ev = np.zeros(n, bool)
    dist = np.full(n, np.nan)
    piv_l = np.flatnonzero(s11.pivots_low(l))
    piv_h = np.flatnonzero(s11.pivots_low(-h))
    if len(piv_l) == 0 or len(piv_h) == 0:
        return ev, dist
    PIV = s11.PIV
    for t in range(2 * PIV + 2, n):
        if not em[t]:
            continue
        z = atr[t - 1]
        if not (math.isfinite(z) and z > EPS):
            continue
        # most recent confirmed B with j_B + PIV <= t - 1
        ib = int(np.searchsorted(piv_h, t - PIV - 1, side="right")) - 1
        if ib < 0:
            continue
        jb = int(piv_h[ib])
        if t - jb > 55 or jb + 1 >= t:
            continue
        hb = h[jb]
        # first close above H_B since jb+1
        if not (c[t] > hb and (c[jb + 1:t] <= hb).all()):
            continue
        # pullback >= 1.0 * ATR(jb)
        ab = atr[jb]
        if not (math.isfinite(ab) and ab > EPS):
            continue
        min_l = float(l[jb + 1:t + 1].min())
        if float(l[jb + 1:t].min()) > hb - 1.0 * ab + TIE:
            continue
        # find most recent confirmed A before jb with t - ja <= 60
        ia = int(np.searchsorted(piv_l, jb, side="left")) - 1
        while ia >= 0:
            ja = int(piv_l[ia])
            if t - ja > s11.AGE_MAX:
                break
            aa = atr[ja]
            la = l[ja]
            if math.isfinite(aa) and aa > EPS and (hb - la >= 2.0 * aa - TIE) and not (c[ja + 1:t] < la).any():
                tgt = min_l + (hb - la)
                d = tgt - c[t]
                if d >= D_MIN_ATR * z - TIE:
                    ev[t] = True
                    dist[t] = d
                break
            ia -= 1
    return ev, dist


def detect_asset_side(h: np.ndarray, l: np.ndarray, c: np.ndarray, em: np.ndarray, side: int,
                      *, with_diag: bool = True) -> dict[str, np.ndarray]:
    """All Topic 4 features for one asset in one side view (+1 = long/bull/support, -1 = mirrored)."""
    n = len(c)
    hv0, lv0, cv0, K0 = s10.side_view(h, l, c, side)
    # 1) Unrounded s10 features (exact zero-diff reuse for Q2 and Q3a)
    atr10 = s09.base_feats(hv0, lv0, cv0)["atr20"]
    f10 = s10.range_features(cv0, atr10)
    d10 = s10.asset_side(h, l, c, em, side)
    z10 = f10["atrp"]  # ATR20(t-1) on unrounded prices
    z10_ok = np.isfinite(z10) & (z10 > EPS)

    # Q3a: trading range bottom third -> midpoint (d >= 0.5*z10 primary; full q1_ev diagnostic)
    q3a_d = f10["mid"] - cv0
    q3a_k = np.where(z10_ok, q3a_d / z10, np.nan)
    q3a_d_ok = np.isfinite(q3a_k) & (q3a_k >= D_MIN_ATR - TIE)
    st10 = f10["state"]
    bot3 = (f10["pos"] >= -s10.REL) & (f10["pos"] <= s10.FLAT + s10.REL)
    q3a_all = d10["q1_ev"]
    q3a_ev = q3a_all & q3a_d_ok
    q3a_ctrl = em & (st10 == s10.ST_UP) & bot3 & q3a_d_ok     # main control: uptrend pullback (mirror: downtrend rally)
    q3a_ctrl_dn = em & (st10 == s10.ST_DN) & bot3 & q3a_d_ok  # secondary diagnostic

    # Q2: confirmed breakout day t* (within 3 own trading days after raw first_break, second close > frozen top0)
    brk_raw = s10.first_break(cv0, atr10, np.ones(n, bool))
    q2_ev = np.zeros(n, bool)
    q2_ctrl = np.zeros(n, bool)
    q2_d = np.full(n, np.nan)
    q2_hgt = np.full(n, np.nan)
    q2_top = np.full(n, np.nan)
    q2_bot = np.full(n, np.nan)
    q2_drop_d = 0  # confirmed breakouts dropped for d < 0.5*z
    for t0 in np.flatnonzero(brk_raw & ((st10 == s10.ST_TR) | (st10 == s10.ST_UP))).tolist():
        is_tr = (st10[t0] == s10.ST_TR)
        top0, hgt0, bot0 = float(f10["top"][t0]), float(f10["hgt"][t0]), float(f10["bot"][t0])
        for ts in range(t0 + 1, min(n, t0 + CONFIRM_WIN + 1)):
            if cv0[ts] > top0:
                if em[ts] and z10_ok[ts]:
                    dt = (top0 + hgt0) - float(cv0[ts])
                    if dt >= D_MIN_ATR * float(z10[ts]) - TIE:
                        if is_tr:
                            q2_ev[ts] = True
                        else:
                            q2_ctrl[ts] = True
                        q2_d[ts], q2_hgt[ts], q2_top[ts], q2_bot[ts] = dt, hgt0, top0, bot0
                    elif is_tr:
                        q2_drop_d += 1
                break

    # Q2 raw (unconfirmed first_break) & Measuring Gap diagnostics
    q2_raw_d = (f10["top"] + f10["hgt"]) - cv0
    q2_raw_ev = d10["q2_ev"] & z10_ok & (q2_raw_d >= D_MIN_ATR * z10 - TIE)
    q2_raw_ctrl = d10["q2_ctrl"] & z10_ok & (q2_raw_d >= D_MIN_ATR * z10 - TIE)
    cv0_1 = s09._shift(cv0, 1)
    q2_mg_d = cv0 - f10["top"]
    q2_mg_ev = brk_raw & em & z10_ok & ((cv0 - cv0_1) >= 1.5 * z10 - TIE) & (q2_mg_d >= D_MIN_ATR * z10 - TIE)

    # 2) Rounded prices (ROUND = 6, matching studies/11) for Q1, Q3b, and Leg 1 = Leg 2
    hv, lv, cv = (np.round(x, ROUND) for x in (hv0, lv0, cv0))
    atr = s09.base_feats(hv, lv, cv)["atr20"]
    z = s09._shift(atr, 1)
    z_ok = np.isfinite(z) & (z > EPS)

    c1, c20 = s09._shift(cv, 1), s09._shift(cv, 20)
    prev_hi20 = pd.Series(cv).rolling(20, min_periods=20).max().shift(1).to_numpy()
    ok20 = z_ok & np.isfinite(c20) & np.isfinite(prev_hi20)

    raw_q1 = ok20 & ((c1 - c20) >= 2.0 * z - TIE) & (cv > prev_hi20) & ((cv - c1) >= 2.0 * z - TIE)
    q1_ev = _cool(raw_q1, COOL) & em

    # R4 main control: flat-base breakout big bar (|C(t-1)-C(t-20)| < 1.0*z + 20d high + >=2*z + 5d cool)
    raw_q1c = ok20 & (np.abs(c1 - c20) < 1.0 * z - TIE) & (cv > prev_hi20) & ((cv - c1) >= 2.0 * z - TIE)
    q1_ctrl = _cool(raw_q1c, COOL) & em

    # Mild trend new-high auxiliary control
    q1_mild = ok20 & ((c1 - c20) >= 2.0 * z - TIE) & (cv > prev_hi20) & (cv > c1) & ((cv - c1) <= 1.0 * z + TIE) & ~q1_ev & em

    # 3-bar climax diagnostic
    c2, c3, c22, z3 = s09._shift(cv, 2), s09._shift(cv, 3), s09._shift(cv, 22), s09._shift(atr, 3)
    ok3 = np.isfinite(z3) & (z3 > EPS) & np.isfinite(c22) & np.isfinite(prev_hi20)
    raw_3b = ok3 & ((c3 - c22) >= 2.0 * z3 - TIE) & (cv > c1) & (c1 > c2) & (c2 > c3) & ((cv - c3) >= 3.5 * z3 - TIE) & (cv > prev_hi20)
    q1_3b = _cool(raw_3b, COOL) & em

    # Q3b magnet + structural placebo (exact s11.detect_A match)
    dm = detect_A_magnet(hv, lv, cv, atr, em)
    q2_leg_ev, q2_leg_d = detect_leg_mm(hv, lv, cv, atr, em) if with_diag else (np.zeros(n, bool), np.full(n, np.nan))

    # Real-price levels for ledger / reporting
    q2_tgt_real = np.where(np.isfinite(q2_top + q2_hgt), (q2_top + q2_hgt) if side > 0 else (K0 - (q2_top + q2_hgt)), np.nan)
    q2_raw_tgt_real = np.where(np.isfinite(f10["top"] + f10["hgt"]),
                               (f10["top"] + f10["hgt"]) if side > 0 else (K0 - (f10["top"] + f10["hgt"])), np.nan)
    q3b_lvl_real = dm["q3b_lvl"] if side > 0 else (K0 - dm["q3b_lvl"])

    # Pre-event TR5 / ATR20(t-1) volatility ratio (uses True Range, not H-L, safe for pre-2010 SPX)
    pc_real = s09._shift(c, 1)
    tr_real = np.maximum(h - l, np.maximum(np.abs(h - pc_real), np.abs(l - pc_real)))
    atr_real = s09.base_feats(h, l, c)["atr20"]
    z_real = s09._shift(atr_real, 1)
    tr5 = pd.Series(tr_real).rolling(5, min_periods=5).mean().to_numpy()

    out = {
        "z10": z10, "z": z, "atr_t": atr, "atr10_t": atr10,
        "q1_ev": q1_ev, "q1_ctrl": q1_ctrl, "q1_mild": q1_mild, "q1_3b": q1_3b,
        "q2_ev": q2_ev, "q2_ctrl": q2_ctrl, "q2_d": q2_d, "q2_hgt": q2_hgt, "q2_top": q2_top, "q2_bot": q2_bot,
        "q2_tgt_real": q2_tgt_real, "q2_raw_ev": q2_raw_ev, "q2_raw_ctrl": q2_raw_ctrl, "q2_raw_d": q2_raw_d,
        "q2_raw_tgt_real": q2_raw_tgt_real, "q2_mg_ev": q2_mg_ev, "q2_mg_d": q2_mg_d,
        "q2_leg_ev": q2_leg_ev, "q2_leg_d": q2_leg_d,
        "q3a_ev": q3a_ev, "q3a_all": q3a_all, "q3a_ctrl": q3a_ctrl, "q3a_ctrl_dn": q3a_ctrl_dn,
        "q3a_d": q3a_d, "q3a_mid_real": d10["mid_real"],
        "q3b_ev": dm["q3b_ev"], "q3b_in": dm["q3b_in"], "q3b_any": dm["q3b_any"],
        "q3b_ctrl": dm["q3b_ctrl"], "q3b_d": dm["q3b_d"], "q3b_ctrl_d": dm["q3b_ctrl_d"],
        "q3b_lvl_real": q3b_lvl_real, "q3b_age": dm["q3b_age"], "q3b_ntouch": dm["q3b_ntouch"],
        "A_narm": dm["A_narm"], "A_toparm": dm["A_toparm"],
        "A_toparm_real": dm["A_toparm"] if side > 0 else (K0 - dm["A_toparm"]), "A_test": dm["A_test"],
        # Signed real returns & vol ratio
        "ret1": side * (c / pc_real - 1.0), "r5v": side * (c / s09._shift(c, 5) - 1.0),
        "v_ratio": np.where(z_real > EPS, tr5 / z_real, np.nan),
        "atr_px": np.where(c > EPS, z_real / c, np.nan),
        "q2_drop_d": np.full(n, q2_drop_d, float),
    }
    return out


BOOL_KEYS = {
    "q1_ev", "q1_ctrl", "q1_mild", "q1_3b", "q2_ev", "q2_ctrl", "q2_raw_ev", "q2_raw_ctrl",
    "q2_mg_ev", "q2_leg_ev", "q3a_ev", "q3a_all", "q3a_ctrl", "q3a_ctrl_dn",
    "q3b_ev", "q3b_in", "q3b_any", "q3b_ctrl", "A_test",
}


# ---------------------------------------------------------------------------
# 2. Path outcome engines (wide market-calendar arrays P)
# ---------------------------------------------------------------------------
def eval_q1_outcomes(P: dict, tt: np.ndarray, nn: np.ndarray, side: int, z: np.ndarray) -> dict[str, np.ndarray]:
    """Compute Q1a (TB) and Q1b (TL + fixed-10d conditional two-legs) on wide (T, N) market arrays."""
    C, H, L = P["C"], P["H"], P["L"]
    T = C.shape[0]
    m = len(tt)
    y_tb = np.full(m, np.nan)
    y_tb_strict = np.full(m, np.nan)
    y_tb_c10 = np.full(m, np.nan)
    surv5 = np.full(m, np.nan)
    surv10 = np.full(m, np.nan)
    surv20 = np.full(m, np.nan)
    y_tl = np.full(m, np.nan)
    y_tl_loose = np.full(m, np.nan)
    y_tl_cond = np.full(m, np.nan)
    l1_hit = np.zeros(m, bool)
    l1_lag = np.full(m, np.nan)

    for i in range(m):
        t, n, zi = int(tt[i]), int(nn[i]), float(z[i])
        if not (math.isfinite(zi) and zi > EPS) or t + WIN_MAIN >= T:
            continue
        c0 = round(float(C[t, n]), ROUND)
        if not math.isfinite(c0):
            continue
        ext0 = round(float(H[t, n] if side > 0 else L[t, n]), ROUND)
        # Future closes t+1 .. t+20
        ck = C[t + 1:t + WIN_MAIN + 1, n]
        fin_k = np.isfinite(ck)
        # 5 / 10 / 20 day survival & TB
        for w, min_d, arr in [(5, 4, surv5), (WIN_TB, MIN_TB, surv10), (WIN_MAIN, MIN_MAIN, surv20)]:
            sub = ck[:w]
            f = fin_k[:w]
            if int(f.sum()) >= min_d:
                sc = np.round(sub[f], ROUND)
                diff = side * (sc - c0)
                arr[i] = 1.0 if float(diff.max()) <= 1.0 * zi + TIE else 0.0
                if w == WIN_TB:
                    y_tb[i] = arr[i]
                    y_tb_strict[i] = 1.0 if float((side * (sc - ext0)).max()) <= TIE else 0.0
                    y_tb_c10[i] = 1.0 if float(diff[-1]) <= TIE else 0.0

        # TL over 20 market days (must have >= MIN_MAIN traded days)
        if int(fin_k.sum()) < MIN_MAIN:
            continue
        # Strict TL (aborts if close exceeds c0 + 1.0*z in side direction before completion)
        st = 0
        peak = 0.0  # signed (side * (c - c0))
        low1 = 0.0
        t1_rel = -1
        aborted = False
        for k in range(WIN_MAIN):
            if not fin_k[k]:
                continue
            v = side * (round(float(ck[k]), ROUND) - c0)
            if v > 1.0 * zi + TIE:
                aborted = True
                break
            if st == 0:
                if v > peak:
                    peak = v
                if peak - v >= 1.0 * zi - TIE:
                    st = 1
                    low1 = v
                    t1_rel = k + 1  # market days after t (1..20)
            elif st == 1:
                if v < low1:
                    low1 = v
                if v - low1 >= 0.5 * zi - TIE:
                    st = 2
            elif st == 2:
                if v < low1 - TIE:
                    st = 3
                    break
        y_tl[i] = 1.0 if (st == 3 and not aborted) else 0.0

        # Loose TL (no upper-bound abort)
        stl, peakl, low1l = 0, 0.0, 0.0
        for k in range(WIN_MAIN):
            if not fin_k[k]:
                continue
            v = side * (round(float(ck[k]), ROUND) - c0)
            if stl == 0:
                if v > peakl:
                    peakl = v
                if peakl - v >= 1.0 * zi - TIE:
                    stl = 1
                    low1l = v
            elif stl == 1:
                if v < low1l:
                    low1l = v
                if v - low1l >= 0.5 * zi - TIE:
                    stl = 2
            elif stl == 2:
                if v < low1l - TIE:
                    stl = 3
                    break
        y_tl_loose[i] = 1.0 if stl == 3 else 0.0

        # Fixed 10-market-day conditional two-legs from t1 = t + t1_rel (R5: eliminates remaining-time bias)
        if t1_rel > 0:
            l1_hit[i] = True
            l1_lag[i] = float(t1_rel)
            t1 = t + t1_rel
            if t1 + WIN_COND < T:
                c_post = C[t1 + 1:t1 + WIN_COND + 1, n]
                f_post = np.isfinite(c_post)
                if int(f_post.sum()) >= MIN_COND:
                    stc = 1
                    lowc = side * (round(float(C[t1, n]), ROUND) - c0)
                    done_c = 0.0
                    for j in range(WIN_COND):
                        if not f_post[j]:
                            continue
                        v = side * (round(float(c_post[j]), ROUND) - c0)
                        if v > 1.0 * zi + TIE:
                            done_c = 0.0
                            break
                        if stc == 1:
                            if v < lowc:
                                lowc = v
                            if v - lowc >= 0.5 * zi - TIE:
                                stc = 2
                        elif stc == 2:
                            if v < lowc - TIE:
                                done_c = 1.0
                                break
                    y_tl_cond[i] = done_c

    return {
        "y_tb": y_tb, "y_tb_strict": y_tb_strict, "y_tb_c10": y_tb_c10,
        "surv5": surv5, "surv10": surv10, "surv20": surv20,
        "y_tl": y_tl, "y_tl_loose": y_tl_loose, "y_tl_cond": y_tl_cond,
        "l1_hit": l1_hit, "l1_lag": l1_lag,
    }


def eval_race(P: dict, tt: np.ndarray, nn: np.ndarray, rdir: int, d_tgt: np.ndarray,
              d_rev: np.ndarray | None = None, *, horizon: int = WIN_MAIN, min_days: int = MIN_MAIN,
              post_days: int = 5) -> dict[str, np.ndarray]:
    """Two-sided race on wide market arrays P (real prices). Open-gap checked first when OK[k, n] is True.
    Returns cons (-1 on both-hit), opt (+1), tie0 (0), close-only, hit_day, both_flag, hit_tgt/rev, post5 directional return.
    """
    O, H, L, C, OK = P["O"], P["H"], P["L"], P["C"], P["OK"]
    T = C.shape[0]
    m = len(tt)
    if d_rev is None:
        d_rev = d_tgt
    y_cons = np.full(m, np.nan)
    y_opt = np.full(m, np.nan)
    y_tie0 = np.full(m, np.nan)
    y_cls = np.full(m, np.nan)
    both = np.zeros(m, bool)
    hit_day = np.full(m, np.nan)
    post5 = np.full(m, np.nan)

    for i in range(m):
        t, n = int(tt[i]), int(nn[i])
        dt, dr = float(d_tgt[i]), float(d_rev[i])
        if not (math.isfinite(dt) and dt > EPS and math.isfinite(dr) and dr > EPS) or t + horizon >= T:
            continue
        c0 = round(float(C[t, n]), ROUND)
        if not math.isfinite(c0):
            continue
        ck = C[t + 1:t + horizon + 1, n]
        fin = np.isfinite(ck)
        if int(fin.sum()) < min_days:
            continue
        bt = round(c0 + rdir * dt, ROUND)
        br = round(c0 - rdir * dr, ROUND)

        # High/low + open-gap race
        res_c = res_o = res_0 = 0.0
        bflag = False
        hday = float(horizon + 1)
        for k in range(horizon):
            if not fin[k]:
                continue
            tk = t + 1 + k
            ok_o = bool(OK[tk, n])
            if ok_o:
                ok_px = round(float(O[tk, n]), ROUND)
                if rdir * (ok_px - bt) >= -TIE:
                    res_c = res_o = res_0 = 1.0
                    hday = float(k + 1)
                    break
                if rdir * (br - ok_px) >= -TIE:
                    res_c = res_o = res_0 = -1.0
                    hday = float(k + 1)
                    break
            hk, lk = round(float(H[tk, n]), ROUND), round(float(L[tk, n]), ROUND)
            ht = (hk >= bt - TIE) if rdir > 0 else (lk <= bt + TIE)
            hr = (lk <= br + TIE) if rdir > 0 else (hk >= br - TIE)
            if ht and not hr:
                res_c = res_o = res_0 = 1.0
                hday = float(k + 1)
                break
            if hr and not ht:
                res_c = res_o = res_0 = -1.0
                hday = float(k + 1)
                break
            if ht and hr:
                res_c, res_o, res_0 = -1.0, 1.0, 0.0
                bflag = True
                hday = float(k + 1)
                break

        y_cons[i], y_opt[i], y_tie0[i], both[i], hit_day[i] = res_c, res_o, res_0, bflag, hday
        # Post-hit 5-market-day directional close return (only when target clearly hit first)
        if res_c == 1.0 and post_days > 0:
            th = t + int(hday)
            if th + post_days < T:
                c_hit = float(C[th, n])
                # last traded close up to th + post_days
                sub_p = C[th + 1:th + post_days + 1, n]
                fp = np.isfinite(sub_p)
                if fp.any() and math.isfinite(c_hit) and c_hit > EPS:
                    post5[i] = rdir * (float(sub_p[fp][-1]) / c_hit - 1.0)

        # Pure close-only race
        rc = 0.0
        for k in range(horizon):
            if not fin[k]:
                continue
            c_k = round(float(ck[k]), ROUND)
            if rdir * (c_k - bt) >= -TIE:
                rc = 1.0
                break
            if rdir * (br - c_k) >= -TIE:
                rc = -1.0
                break
        y_cls[i] = rc

    return {
        "cons": y_cons, "opt": y_opt, "tie0": y_tie0, "cls": y_cls,
        "both": both, "hit_day": hit_day, "post5": post5,
    }


def non_overlap_mask(tt: np.ndarray, nn: np.ndarray, mask: np.ndarray, hold_days: np.ndarray | None = None,
                     default_win: int = WIN_MAIN) -> np.ndarray:
    """Path non-overlap: per asset, ignore new events until the prior event's outcome resolves (or `default_win` days)."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return np.zeros(len(tt), bool)
    order = np.lexsort((tt[idx], nn[idx]))
    keep = np.zeros(len(tt), bool)
    last_n = -1
    busy_until = -1
    for p in order.tolist():
        i = int(idx[p])
        ni, ti = int(nn[i]), int(tt[i])
        if ni != last_n or ti > busy_until:
            keep[i] = True
            last_n = ni
            hd = int(hold_days[i]) if (hold_days is not None and math.isfinite(hold_days[i])) else default_win
            busy_until = ti + min(hd, default_win)
    return keep


# ---------------------------------------------------------------------------
# 3. Q2 stop-entry explicit-target ledger diagnostic (exact s09 semantics; s09 untouched)
# ---------------------------------------------------------------------------
def run_stop_tgt_ledger(P: dict, sig_t: np.ndarray, sig_n: np.ndarray, side: int, tgt_px: np.ndarray, *,
                        both: str = "cons", t1: str | None = None, cost: float | np.ndarray = 0.0,
                        max_hold: int = s09.MAX_HOLD, exit_delay: int = s09.EXIT_DELAY,
                        use_open: bool = True, buffer: int = IDX_BUF) -> dict:
    """Stop-entry bracket ledger with fixed explicit target price `tgt_px` (real prices).
    If entry open/fill already >= tgt_px, trade is skipped (W_PAST_TGT). Matches s09.run_ledger(entry='stop') verbatim
    whenever no gap-in occurs and tgt_px = level + k*(level - stop).
    """
    T, N = P["C"].shape
    sig_t, sig_n = np.asarray(sig_t, np.int64), np.asarray(sig_n, np.int64)
    M = len(sig_t)
    Of, Hf, Lf, Cf = (P[x].reshape(-1) for x in ("O", "H", "L", "C"))
    OKf = P["OK"].reshape(-1) if use_open else np.zeros(T * N, bool)
    UPf, DNf, TKf = P["UP"].reshape(-1), P["DN"].reshape(-1), P["TICK"].reshape(-1)

    def g(t, n):
        f = t * N + n
        o, h, l, c, up, dn = Of[f], Hf[f], Lf[f], Cf[f], UPf[f], DNf[f]
        return (o, h, l, c, up, dn) if side > 0 else (-o, -l, -h, -c, -dn, -up)

    why = np.full(M, s09.W_OK, np.int8)
    st = np.zeros(M, np.int8)
    fill, stop, R, tgt, xpx = (np.full(M, np.nan) for _ in range(5))
    kind = np.zeros(M, np.int8)
    xt = np.full(M, -1, np.int64)
    deadline = np.full(M, -1, np.int64)
    last_c = np.full(M, np.nan)
    ht = np.zeros(M)
    e = sig_t + 1
    ok = sig_t + buffer <= T - 1
    why[~ok] = s09.W_TRUNC
    i0 = np.flatnonzero(ok)
    _, Hs, Ls, _, _, _ = g(sig_t[i0], sig_n[i0])
    tick = TKf[sig_t[i0] * N + sig_n[i0]]
    ht[i0] = 0.5 * tick
    o, h, l, c, up, dn = g(e[i0], sig_n[i0])
    tol_e = 0.5 * TKf[e[i0] * N + sig_n[i0]]
    oke = OKf[e[i0] * N + sig_n[i0]]
    trd = np.isfinite(c) & np.isfinite(h) & np.isfinite(l)
    why[i0[~trd]] = s09.W_SUSP
    level = Hs + tick
    trig = trd & (h > level - 0.5 * tick)
    gap = oke & (o > level - 0.5 * tick)
    f_ = np.where(gap, np.maximum(o, level), level)
    stp = Ls - tick
    why[i0[trd & ~trig]] = s09.W_NOTRIG
    blk = trig & (f_ >= up - tol_e)
    if side < 0:
        blk &= c >= up - tol_e
    why[i0[blk]] = s09.W_LIMIT
    tp_all = np.asarray(tgt_px, float)[i0] * (1.0 if side > 0 else -1.0)
    past = trig & ~blk & (~np.isfinite(tp_all) | (f_ >= tp_all - 0.5 * tick))
    why[i0[past]] = W_PAST_TGT
    go = trig & ~blk & ~past
    ii = i0[go]
    f_, stp, gap, tp = f_[go], stp[go], gap[go], tp_all[go]
    o, h, l, c, dn, tol_e = o[go], h[go], l[go], c[go], dn[go], tol_e[go]
    fill[ii], stop[ii], R[ii], tgt[ii] = f_, stp, f_ - stp, tp
    last_c[ii] = c
    st[ii] = 1

    sh = l < stp + ht[ii]
    th = h >= tp
    if t1 is None:
        if both == "cons":
            xs, xg = sh, ~sh & th
        else:
            xg = th
            xs = ~th & sh & gap
        sel = ii[xg]
        xpx[sel], kind[sel], xt[sel], st[sel] = tgt[sel], s09.K_TARGET, e[sel], 3
        px = stp[xs]
        sealed = (px <= dn[xs] + tol_e[xs]) & (c[xs] <= dn[xs] + tol_e[xs])
        sel = ii[xs]
        okx = ~sealed
        xpx[sel[okx]], kind[sel[okx]], xt[sel[okx]], st[sel[okx]] = px[okx], s09.K_STOP, e[sel[okx]], 3
        kind[sel[~okx]], st[sel[~okx]], deadline[sel[~okx]] = s09.K_STOP, 2, e[sel[~okx]] + exit_delay
    elif t1 == "forced":
        cnt = sh if both == "cons" else (sh & gap)
        sel = ii[cnt]
        kind[sel], st[sel], deadline[sel] = s09.K_FORCED, 2, e[sel] + exit_delay
    elif t1 != "ignore":
        raise ValueError(t1)

    for d in range(1, max_hold + exit_delay + 2):
        ip = np.flatnonzero(st == 2)
        if len(ip):
            kk = e[ip] + d
            late = (kk > deadline[ip]) | (kk > T - 1)
            sel = ip[late]
            xpx[sel], xt[sel], st[sel] = last_c[sel], np.minimum(deadline[sel], T - 1), 3
            ip, kk = ip[~late], kk[~late]
            o, h, l, c, _, dn = g(kk, sig_n[ip])
            okk = OKf[kk * N + sig_n[ip]]
            tol = 0.5 * TKf[kk * N + sig_n[ip]]
            trd = np.isfinite(c)
            last_c[ip[trd]] = c[trd]
            free_dn = ~(dn > -np.inf)
            at_open = trd & okk & (free_dn | (o > dn + tol))
            at_lim = trd & ~at_open & (free_dn | (c > dn + tol))
            sel = ip[at_open]
            xpx[sel], xt[sel], st[sel] = o[at_open], kk[at_open], 3
            sel = ip[at_lim]
            xpx[sel] = np.where(free_dn[at_lim], c[at_lim], dn[at_lim])
            xt[sel], st[sel] = kk[at_lim], 3
        if d > max_hold:
            if not (st == 2).any():
                break
            continue
        ia = np.flatnonzero(st == 1)
        if not len(ia):
            continue
        kk = e[ia] + d
        o, h, l, c, _, dn = g(kk, sig_n[ia])
        okk = OKf[kk * N + sig_n[ia]]
        tol = 0.5 * TKf[kk * N + sig_n[ia]]
        trd = np.isfinite(c)
        last_c[ia[trd]] = c[trd]
        sp, tg, hh = stop[ia], tgt[ia], ht[ia]
        gs = trd & okk & (o < sp + hh)
        gt = trd & okk & ~gs & (o >= tg)
        s_hit = trd & (l < sp + hh)
        t_hit = trd & (h >= tg)
        rest = trd & ~gs & ~gt
        if both == "cons":
            xs_in, xg_in = rest & s_hit, rest & ~s_hit & t_hit
        else:
            xg_in, xs_in = rest & t_hit, rest & ~t_hit & s_hit
        xg = gt | xg_in
        px_g = np.where(gt, o, tg)
        sel = ia[xg]
        xpx[sel], kind[sel], xt[sel], st[sel] = px_g[xg], s09.K_TARGET, kk[xg], 3
        xs = gs | xs_in
        px_s = np.where(gs, o, sp)
        sealed = (px_s <= dn + tol) & (c <= dn + tol)
        a = xs & ~sealed
        sel = ia[a]
        xpx[sel], kind[sel], xt[sel], st[sel] = px_s[a], s09.K_STOP, kk[a], 3
        b = xs & sealed
        sel = ia[b]
        kind[sel], st[sel], deadline[sel] = s09.K_STOP, 2, kk[b] + exit_delay
        if d == max_hold:
            rem = ~xg & ~xs
            a = rem & trd & ~(c <= dn + tol)
            sel = ia[a]
            xpx[sel], kind[sel], xt[sel], st[sel] = c[a], s09.K_TIME, kk[a], 3
            b = rem & ~a
            sel = ia[b]
            kind[sel], st[sel], deadline[sel] = s09.K_TIME, 2, kk[b] + exit_delay

    filled = why == s09.W_OK
    c_arr = np.broadcast_to(np.asarray(cost, float), (M,))
    gross = np.full(M, np.nan)
    gross[filled] = (xpx[filled] - fill[filled]) / R[filled]
    net = gross - c_arr * np.abs(fill) / R
    return {
        "filled": filled, "why": why, "kind": kind, "win": filled & (kind == s09.K_TARGET),
        "gross": gross, "net": net, "hold": np.where(filled, xt - e, -1), "R": R, "fill": fill,
    }


# ---------------------------------------------------------------------------
# 4. Null hypothesis generators (Markov 5-day block bootstrap + IID + 256-step Martingale RW)
# ---------------------------------------------------------------------------
def build_block_pool(P: dict, elig: np.ndarray, excl: np.ndarray, *, blk_len: int = 5,
                     n_abins: int = 3, n_vbins: int = 2, split_mag: bool = False,
                     lim_bin: np.ndarray | None = None, max_gap: int = 5) -> dict:
    """Build stratified real-price multiplier blocks from non-event eligible days (no long suspension gaps).
    Event alignment: day t uses t-4..t; block starting after s uses s-4..s as pre-state and s+1..s+blk_len as future.
    """
    O, H, L, C, OK = P["O"], P["H"], P["L"], P["C"], P["OK"]
    T, N = C.shape
    mults, ok_arr = [], []
    a0_l, v0_l, dir_l, mag_l, enda_l, endv_l, limb_l = ([] for _ in range(7))

    for n in range(N):
        rows = np.flatnonzero(np.isfinite(C[:, n]))
        m = len(rows)
        if m < 30 + blk_len:
            continue
        o, h, l, c = O[rows, n], H[rows, n], L[rows, n], C[rows, n]
        ok_n = OK[rows, n]
        el_n = elig[rows, n]
        ex_n = excl[rows, n]
        # exclude event days AND the 5 own trading days after an event day
        ex_pad = pd.Series(ex_n.astype(float)).rolling(6, min_periods=1).sum().to_numpy() > 0
        atr = s09.base_feats(h, l, c)["atr20"]
        pc = s09._shift(c, 1)
        tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
        tr5 = pd.Series(tr).rolling(5, min_periods=5).mean().to_numpy()
        lb_n = lim_bin[rows, n] if lim_bin is not None else np.zeros(m, np.int8)

        for s in range(21, m - blk_len):
            if not el_n[s] or ex_pad[s]:
                continue
            # Check no long suspension gap inside s..s+blk_len
            if int(np.diff(rows[s:s + blk_len + 1]).max()) > max_gap:
                continue
            z0 = atr[s - 1]
            z_end = atr[s + blk_len - 1]
            if not (math.isfinite(z0) and z0 > EPS and math.isfinite(z_end) and z_end > EPS and c[s] > EPS and c[s + blk_len] > EPS):
                continue
            v0, ve = tr5[s] / z0, tr5[s + blk_len] / z_end
            if not (math.isfinite(v0) and math.isfinite(ve)):
                continue
            # Multipliers relative to prior close for bars s+1..s+blk_len
            prev_c = c[s:s + blk_len]
            if not (prev_c > EPS).all():
                continue
            bl = np.column_stack([
                o[s + 1:s + blk_len + 1] / prev_c,
                h[s + 1:s + blk_len + 1] / prev_c,
                l[s + 1:s + blk_len + 1] / prev_c,
                c[s + 1:s + blk_len + 1] / prev_c,
            ])
            if not np.isfinite(bl).all():
                continue
            mults.append(bl)
            ok_arr.append(ok_n[s + 1:s + blk_len + 1])
            a0_l.append(z0 / c[s])
            v0_l.append(v0)
            dir_l.append(1 if c[s] >= c[s - 1] else 0)
            mag_l.append(1 if abs(c[s] - c[s - 1]) >= 1.5 * z0 else 0)
            enda_l.append(z_end / c[s + blk_len])
            endv_l.append(ve)
            limb_l.append(int(lb_n[s]))

    M = np.asarray(mults, np.float64) if mults else np.zeros((0, blk_len, 4))
    OKB = np.asarray(ok_arr, bool) if ok_arr else np.zeros((0, blk_len), bool)
    a0, v0 = np.asarray(a0_l), np.asarray(v0_l)
    enda, endv = np.asarray(enda_l), np.asarray(endv_l)
    dr, mg, lb = np.asarray(dir_l, np.int8), np.asarray(mag_l, np.int8), np.asarray(limb_l, np.int8)

    qa = np.quantile(a0, np.linspace(0, 1, n_abins + 1)[1:-1]) if len(a0) > 10 else np.array([])
    qv = np.quantile(v0, np.linspace(0, 1, n_vbins + 1)[1:-1]) if len(v0) > 10 else np.array([])
    ab = np.digitize(a0, qa)
    vb = np.digitize(v0, qv)
    eab = np.digitize(enda, qa)
    evb = np.digitize(endv, qv)

    # Pre-index buckets for fast sampling
    def _idx_map(keys_tuple):
        d = {}
        for i in range(len(M)):
            k = tuple(int(x[i]) for x in keys_tuple)
            d.setdefault(k, []).append(i)
        return {k: np.asarray(v, np.int64) for k, v in d.items()}

    b_full = _idx_map((lb, ab, vb, dr, mg)) if split_mag else _idx_map((lb, ab, vb, dr))
    b_nodir = _idx_map((lb, ab, vb))
    b_a = _idx_map((lb, ab))
    b_all = np.arange(len(M), dtype=np.int64)

    # Martingale-adjusted multipliers: zero conditional price drift per (lb, ab, vb, dr) bucket (fallback to (lb, ab, vb))
    M_mart = M.copy()
    blk_ret = np.prod(M[:, :, 3], axis=1)
    done_m = np.zeros(len(M), bool)
    for idx_b in b_full.values():
        if len(idx_b) >= 30:
            mr = float(np.mean(blk_ret[idx_b]))
            if mr > EPS:
                M_mart[idx_b] /= (mr ** (1.0 / blk_len))
                done_m[idx_b] = True
    for idx_b in b_nodir.values():
        rem = idx_b[~done_m[idx_b]]
        if len(rem):
            mr = float(np.mean(blk_ret[idx_b]))
            if mr > EPS:
                M_mart[rem] /= (mr ** (1.0 / blk_len))
                done_m[rem] = True
    return {
        "M": M, "M_mart": M_mart, "OK": OKB, "blk_len": blk_len, "n_abins": n_abins, "n_vbins": n_vbins,
        "split_mag": split_mag, "qa": qa, "qv": qv, "a0": a0, "ab": ab, "vb": vb, "eab": eab, "evb": evb, "lb": lb,
        "b_full": b_full, "b_nodir": b_nodir, "b_a": b_a, "b_all": b_all,
    }


def _pick_bucket(pool: dict, key_full: tuple, min_sz: int = 100) -> np.ndarray:
    """Hierarchical fallback ensuring >= min_sz blocks (or largest available)."""
    arr = pool["b_full"].get(key_full)
    if arr is not None and len(arr) >= min_sz:
        return arr
    if pool["split_mag"] and len(key_full) == 5:
        # drop mag -> match (lb, ab, vb, dr)
        lb_, ab_, vb_, dr_ = key_full[:4]
        parts = [pool["b_full"].get((lb_, ab_, vb_, dr_, m), np.empty(0, np.int64)) for m in (0, 1)]
        arr = np.concatenate(parts)
        if len(arr) >= min_sz:
            return arr
    arr = pool["b_nodir"].get(key_full[:3])
    if arr is not None and len(arr) >= min_sz:
        return arr
    arr = pool["b_a"].get(key_full[:2])
    if arr is not None and len(arr) >= min_sz:
        return arr
    return pool["b_all"]


def simulate_markov_paths(pool: dict, ev_keys: list[tuple], ev_k: np.ndarray, *, B: int = 10000,
                          horizon: int = WIN_MAIN, seed: int = 12, martingale: bool = False,
                          pool_sub: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate B simulated price paths (starting at C_0 = 1.0) of length `horizon` via Markov-chained blocks.
    Returns (paths (B, horizon, 4) for O/H/L/C, ok (B, horizon), z_sim (B,), d_sim (B,)).
    """
    rng = np.random.default_rng(seed)
    M = pool["M_mart"] if martingale else pool["M"]
    OKB = pool["OK"]
    bl = pool["blk_len"]
    n_blk = (horizon + bl - 1) // bl
    ne = len(ev_keys)
    pick_e = rng.integers(0, ne, size=B)
    k_sim = ev_k[pick_e] if ev_k is not None else np.ones(B)

    # Optional pool subsampling (for block-pool bootstrap SE)
    if pool_sub is not None:
        sub_set = np.zeros(len(M), bool)
        sub_set[pool_sub] = True
    else:
        sub_set = None

    def _filt(arr):
        if sub_set is None:
            return arr
        f = arr[sub_set[arr]]
        return f if len(f) >= 20 else arr

    # Block 1: group by unique ev_key
    idx_b = np.empty((B, n_blk), np.int64)
    ukeys: dict[tuple, list[int]] = {}
    for bi, ei in enumerate(pick_e.tolist()):
        ukeys.setdefault(ev_keys[ei], []).append(bi)
    for key, pos_list in ukeys.items():
        pos = np.asarray(pos_list, np.int64)
        cand = _filt(_pick_bucket(pool, key, min_sz=min(100, max(20, len(M) // 20))))
        idx_b[pos, 0] = rng.choice(cand, size=len(pos), replace=True)

    # Subsequent blocks: Markov-chained on previous block's (lb, eab, evb)
    lb_all, eab_all, evb_all = pool["lb"], pool["eab"], pool["evb"]
    for b in range(1, n_blk):
        prev = idx_b[:, b - 1]
        cur_lb, cur_a, cur_v = lb_all[prev], eab_all[prev], evb_all[prev]
        # group by (lb, a, v)
        code = cur_lb * 100 + cur_a * 10 + cur_v
        for cd in np.unique(code).tolist():
            pos = np.flatnonzero(code == cd)
            l_v, a_v, v_v = cd // 100, (cd % 100) // 10, cd % 10
            cand = _filt(_pick_bucket(pool, (l_v, a_v, v_v), min_sz=min(100, max(20, len(M) // 20))))
            idx_b[pos, b] = rng.choice(cand, size=len(pos), replace=True)

    # Assemble price paths starting at C_0 = 1.0
    tot_d = n_blk * bl
    paths = np.empty((B, tot_d, 4), np.float64)
    ok_p = np.empty((B, tot_d), bool)
    cur_c = np.ones(B, np.float64)
    for b in range(n_blk):
        bi = idx_b[:, b]
        mb = M[bi]  # (B, bl, 4)
        ok_p[:, b * bl:(b + 1) * bl] = OKB[bi]
        for j in range(bl):
            t_j = b * bl + j
            paths[:, t_j, :] = cur_c[:, None] * mb[:, j, :]
            cur_c = paths[:, t_j, 3]
    z_sim = pool["a0"][idx_b[:, 0]]
    d_sim = k_sim * z_sim
    return paths[:, :horizon, :], ok_p[:, :horizon], z_sim, d_sim


def eval_q1_sim(paths: np.ndarray, z_sim: np.ndarray, side: int) -> dict[str, float]:
    """Vectorized Q1a (TB) & Q1b (TL + fixed-10d cond TL) on simulated paths (B, >=30, 4) starting at C_0 = 1.0."""
    B = len(paths)
    C = paths[:, :, 3]
    # Note: paths are in REAL prices starting at 1.0; signed move = side * (C - 1.0)
    S = side * (C - 1.0)
    # TB over 10 days
    tb = (S[:, :WIN_TB].max(1) <= 1.0 * z_sim + TIE).astype(float)
    # TL over 20 days + fixed-10d cond TL (requires paths length >= 30)
    tl = np.zeros(B, float)
    cond = np.full(B, np.nan)
    for i in range(B):
        zi = float(z_sim[i])
        si = S[i]
        st = 0
        pk = 0.0
        l1 = 0.0
        t1 = -1
        ab = False
        for k in range(WIN_MAIN):
            v = float(si[k])
            if v > 1.0 * zi + TIE:
                ab = True
                break
            if st == 0:
                if v > pk:
                    pk = v
                if pk - v >= 1.0 * zi - TIE:
                    st = 1
                    l1 = v
                    t1 = k
            elif st == 1:
                if v < l1:
                    l1 = v
                if v - l1 >= 0.5 * zi - TIE:
                    st = 2
            elif st == 2:
                if v < l1 - TIE:
                    st = 3
                    break
        if st == 3 and not ab:
            tl[i] = 1.0
        if t1 >= 0 and t1 + 1 + WIN_COND <= C.shape[1]:
            stc, lc = 1, float(si[t1])
            dc = 0.0
            for j in range(t1 + 1, t1 + 1 + WIN_COND):
                v = float(si[j])
                if v > 1.0 * zi + TIE:
                    dc = 0.0
                    break
                if stc == 1:
                    if v < lc:
                        lc = v
                    if v - lc >= 0.5 * zi - TIE:
                        stc = 2
                elif stc == 2:
                    if v < lc - TIE:
                        dc = 1.0
                        break
            cond[i] = dc
    return {"tb": float(tb.mean()), "tl": float(tl.mean()),
            "cond": float(np.nanmean(cond)) if np.isfinite(cond).any() else np.nan}


def eval_race_sim(paths: np.ndarray, ok_p: np.ndarray, d_sim: np.ndarray, rdir: int,
                  d_rev: np.ndarray | None = None) -> dict[str, float]:
    """Vectorized symmetric/asymmetric race on simulated real-price paths (B, H, 4) starting at C_0 = 1.0."""
    B, H, _ = paths.shape
    if d_rev is None:
        d_rev = d_sim
    bt = 1.0 + rdir * d_sim
    br = 1.0 - rdir * d_rev
    O, Hi, Lo = paths[:, :, 0], paths[:, :, 1], paths[:, :, 2]
    rc = np.zeros(B, float)
    ro = np.zeros(B, float)
    r0 = np.zeros(B, float)
    bflag = np.zeros(B, bool)
    d_hit = np.full(B, H + 1, int)
    done = np.zeros(B, bool)
    for k in range(H):
        act = ~done
        if not act.any():
            break
        ok_k = ok_p[act, k]
        o_k, h_k, l_k = O[act, k], Hi[act, k], Lo[act, k]
        bt_a, br_a = bt[act], br[act]
        go_t = ok_k & (rdir * (o_k - bt_a) >= -TIE)
        go_r = ok_k & ~go_t & (rdir * (br_a - o_k) >= -TIE)
        rem = ~(go_t | go_r)
        ht = rem & ((h_k >= bt_a - TIE) if rdir > 0 else (l_k <= bt_a + TIE))
        hr = rem & ((l_k <= br_a + TIE) if rdir > 0 else (h_k >= br_a - TIE))
        win_only = go_t | (ht & ~hr)
        los_only = go_r | (hr & ~ht)
        both_k = ht & hr
        fin_k = win_only | los_only | both_k
        if fin_k.any():
            ia = np.flatnonzero(act)[fin_k]
            w_, l_, b_ = win_only[fin_k], los_only[fin_k], both_k[fin_k]
            rc[ia] = np.where(w_, 1.0, -1.0)
            ro[ia] = np.where(l_, -1.0, 1.0)
            r0[ia] = np.where(w_, 1.0, np.where(l_, -1.0, 0.0))
            bflag[ia] = b_
            d_hit[ia] = k + 1
            done[ia] = True
    return {
        "cons": float(rc.mean()), "opt": float(ro.mean()), "tie0": float(r0.mean()),
        "both": float(bflag.mean()), "blk1_share": float((d_hit <= 5).mean()),
    }


def null_summary(pool: dict, ev_keys: list[tuple], ev_k: np.ndarray | None, mode: str, side_or_rdir: int, *,
                 B_main: int = 20000, n_boot: int = 80, B_boot: int = 1500, seed: int = 120) -> dict:
    """Compute Markov null mean, martingale zero-drift mean, and SE(mu_0) via outer block-pool resampling."""
    H = 30 if mode == "q1" else WIN_MAIN
    p, ok, zs, ds = simulate_markov_paths(pool, ev_keys, ev_k, B=B_main, horizon=H, seed=seed, martingale=False)
    pm, okm, zsm, dsm = simulate_markov_paths(pool, ev_keys, ev_k, B=B_main, horizon=H, seed=seed + 1, martingale=True)
    if mode == "q1":
        m_main = eval_q1_sim(p, zs, side_or_rdir)
        m_mart = eval_q1_sim(pm, zsm, side_or_rdir)
        # Outer block-pool resampling for SE(mu_0)
        rng = np.random.default_rng(seed + 7)
        M_len = len(pool["M"])
        bt_tb, bt_tl, bt_c = [], [], []
        for b in range(n_boot):
            sub = rng.choice(M_len, size=M_len, replace=True)
            pb, _, zb, _ = simulate_markov_paths(pool, ev_keys, ev_k, B=B_boot, horizon=H, seed=seed + 100 + b, pool_sub=sub)
            rb = eval_q1_sim(pb, zb, side_or_rdir)
            bt_tb.append(rb["tb"])
            bt_tl.append(rb["tl"])
            bt_c.append(rb["cond"])
        return {
            "tb": m_main["tb"], "tb_mart": m_mart["tb"], "tb_se": float(np.nanstd(bt_tb, ddof=1)),
            "tl": m_main["tl"], "tl_mart": m_mart["tl"], "tl_se": float(np.nanstd(bt_tl, ddof=1)),
            "cond": m_main["cond"], "cond_mart": m_mart["cond"], "cond_se": float(np.nanstd(bt_c, ddof=1)),
        }
    m_main = eval_race_sim(p, ok, ds, side_or_rdir)
    m_mart = eval_race_sim(pm, okm, dsm, side_or_rdir)
    rng = np.random.default_rng(seed + 7)
    M_len = len(pool["M"])
    bt_c = []
    for b in range(n_boot):
        sub = rng.choice(M_len, size=M_len, replace=True)
        pb, okb, _, db = simulate_markov_paths(pool, ev_keys, ev_k, B=B_boot, horizon=H, seed=seed + 100 + b, pool_sub=sub)
        bt_c.append(eval_race_sim(pb, okb, db, side_or_rdir)["cons"])
    return {
        "cons": m_main["cons"], "opt": m_main["opt"], "tie0": m_main["tie0"], "both": m_main["both"],
        "blk1_share": m_main["blk1_share"], "mart": m_mart["cons"], "se": float(np.nanstd(bt_c, ddof=1)),
    }


def rw256_null(ev_k: np.ndarray | None, mode: str, side_or_rdir: int, sigma_day: float, drift_day: float, *,
               B: int = 8000, seed: int = 999) -> dict:
    """256-step intraday random walk auxiliary null (both price-martingale zero-drift & empirical-drift)."""
    rng = np.random.default_rng(seed)
    H = 30 if mode == "q1" else WIN_MAIN
    m = 64  # 64 intraday steps per day (fast & accurate)
    sig_step = sigma_day / math.sqrt(m)
    out = {}
    for tag, dr in [("mart", -0.5 * sigma_day * sigma_day), ("drift", drift_day - 0.5 * sigma_day * sigma_day)]:
        steps = rng.normal(dr / m, sig_step, size=(B, H, m))
        paths = np.empty((B, H, 4))
        cur = np.ones(B)
        for t in range(H):
            seg = cur[:, None] * np.exp(np.cumsum(steps[:, t, :], axis=1))
            paths[:, t, 0] = cur
            paths[:, t, 1] = np.maximum(cur, seg.max(1))
            paths[:, t, 2] = np.minimum(cur, seg.min(1))
            paths[:, t, 3] = seg[:, -1]
            cur = seg[:, -1]
        # Theoretical ATR20 / C ≈ 1.25 * sigma_day for continuous BM (exact simulated mean of range/C)
        z_sim = np.full(B, float(np.mean(paths[:, :, 1] - paths[:, :, 2])))
        if mode == "q1":
            r = eval_q1_sim(paths, z_sim, side_or_rdir)
            out[tag] = r
        else:
            ks = rng.choice(ev_k, size=B, replace=True)
            out[tag] = eval_race_sim(paths, np.ones((B, H), bool), ks * z_sim, side_or_rdir)
    return out


# ---------------------------------------------------------------------------
# 5. Index container & regression helpers
# ---------------------------------------------------------------------------
class PathIndex:
    def __init__(self, name: str, df: pd.DataFrame, eval_start: pd.Timestamp, *, with_diag: bool = True) -> None:
        self.mk = mk = s09.IndexMarket(name, df, eval_start)
        self.name = name
        T, N = mk.T, mk.N
        em_all = np.asarray(mk.dates >= np.datetime64(eval_start), bool)
        self.em_wide = np.repeat(em_all[:, None], N, axis=1) & np.isfinite(mk.P["C"])
        self.feat: dict[int, dict[str, np.ndarray]] = {}
        self.q2_drop_d = {1: 0, -1: 0}
        for side in (1, -1):
            W: dict[str, np.ndarray] = {}
            for n in range(N):
                rows = np.flatnonzero(np.isfinite(mk.P["C"][:, n]))
                if len(rows) < 30:
                    continue
                d = detect_asset_side(mk.P["H"][rows, n], mk.P["L"][rows, n], mk.P["C"][rows, n],
                                      em_all[rows], side, with_diag=with_diag)
                self.q2_drop_d[side] += int(d["q2_drop_d"][0])
                for k, v in d.items():
                    if k == "q2_drop_d":
                        continue
                    if k not in W:
                        W[k] = np.zeros((T, N), bool) if k in BOOL_KEYS else np.full((T, N), np.nan)
                    W[k][rows, n] = v
            self.feat[side] = W
        # Candidate rows: all eligible bars with finite z10 & z & buffer IDX_BUF
        inb = np.arange(T)[:, None] + IDX_BUF <= T - 1
        ok_z = np.isfinite(self.feat[1]["z"]) & (self.feat[1]["z"] > EPS) & np.isfinite(self.feat[1]["z10"]) & (self.feat[1]["z10"] > EPS)
        ok_ctrl = np.isfinite(mk.F["r20"]) & np.isfinite(mk.F["vol20"]) & np.isfinite(mk.F["r5"]) & np.isfinite(self.feat[1]["ret1"])
        self.cand = self.em_wide & inb & ok_z & ok_ctrl
        self.tt, self.nn = np.nonzero(self.cand)
        self.year = mk.year[self.tt]

    def at(self, side: int, key: str) -> np.ndarray:
        return self.feat[side][key][self.tt, self.nn]

    def event_keys(self, pool: dict, mask: np.ndarray, side: int) -> list[tuple]:
        """Map each masked row to its Block-1 matching tuple (lb=0, ab, vb, dir)."""
        a = self.at(1, "atr_px")[mask]
        v = self.at(1, "v_ratio")[mask]
        r1 = self.at(1, "ret1")[mask]  # real ret1
        ab = np.digitize(a, pool["qa"])
        vb = np.digitize(v, pool["qv"])
        dr = (r1 >= 0).astype(int)
        return [(0, int(ab[i]), int(vb[i]), int(dr[i])) for i in range(len(a))]


def _dk_mean_se(y: np.ndarray, tt: np.ndarray, lag: int = LAG) -> tuple[float, float]:
    """Mean and DK standard error of a single series on calendar index tt."""
    ok = np.isfinite(y)
    n_ok = int(ok.sum())
    if n_ok < 5:
        return np.nan, np.nan
    mu_raw = float(np.mean(y[ok]))
    se_raw = float(np.std(y[ok], ddof=1) / math.sqrt(n_ok))
    b, t = b8.wls_dk(y[ok].astype(float), np.ones((n_ok, 1)), np.ones(n_ok), tt[ok], lag)
    if math.isfinite(b[0]) and math.isfinite(t[0]) and abs(t[0]) > EPS:
        return float(b[0]), abs(float(b[0]) / float(t[0]))
    return mu_raw, se_raw


def _cap_day_w(tt: np.ndarray, grp: np.ndarray, cap: int = SW_DAY_CAP) -> np.ndarray:
    w = np.ones(len(tt), float)
    for g in np.unique(grp):
        m = np.flatnonzero(grp == g)
        if len(m) == 0:
            continue
        _, inv, cnt = np.unique(tt[m], return_inverse=True, return_counts=True)
        w[m] = np.minimum(1.0, cap / cnt[inv])
    return w


def fit_index_reg(px: PathIndex, y: np.ndarray, E: np.ndarray, C: np.ndarray | None, *,
                  k_arr: np.ndarray | None = None, d_px: np.ndarray | None = None,
                  layers: tuple = ("dist", "geo", "ret1", "trend"), contrast: bool = False,
                  sub: np.ndarray | None = None, demean_day: bool = False,
                  day_cap: int | None = None, lag: int = LAG) -> tuple[float, float, int]:
    """Pooled DK regression for index markets. Fitting pool excludes ONLY true events E (never ordinary controls)."""
    f = np.isfinite(y)
    if k_arr is not None:
        f &= np.isfinite(k_arr)
    if d_px is not None:
        f &= np.isfinite(d_px)
    if sub is not None:
        f &= sub
    rows = np.flatnonzero(f)
    if len(rows) < 50:
        return np.nan, np.nan, 0
    De = E[rows]
    ne = int(De.sum())
    if ne < MIN_EVENTS:
        return np.nan, np.nan, ne
    Ds = [De]
    if contrast:
        if C is None:
            return np.nan, np.nan, ne
        Dc = C[rows] & ~De
        if int(Dc.sum()) < MIN_EVENTS:
            return np.nan, np.nan, ne
        Ds = [De, De | Dc]  # coef[0] = E - C!
    ref = ~De  # fitting reference excludes ONLY true events
    mk, tt, nn = px.mk, px.tt[rows], px.nn[rows]
    parts = []
    lin = []
    if "dist" in layers and k_arr is not None:
        kv = k_arr[rows]
        parts.append(s09._qdum(kv, kv[ref]))
        lin.append(kv)
        if d_px is not None:
            lin.append(d_px[rows])
    if "geo" in layers:
        for col in ["g_atr", "g_px", "kcls"]:
            v = mk.F[col][tt, nn]
            parts.append(s09._qdum(v, v[ref]))
    if "ret1" in layers:
        r1 = px.feat[1]["ret1"][tt, nn]
        parts.append(s09._qdum(r1, r1[ref]))
        lin.append(r1)
    if "trend" in layers:
        lin += [mk.F[col][tt, nn] for col in ["r20", "vol20", "r5"]]
    if lin:
        X = np.column_stack(lin)
        sd = np.nanstd(X[ref], 0)
        mu = np.nanmean(X[ref], 0)
        parts.append((X - mu) / np.where(sd > EPS, sd, 1.0))
    Xall = np.hstack(parts) if parts else np.zeros((len(rows), 0))
    yy = y[rows].astype(float)
    if demean_day:
        # Subtract daily mean across industries (requires >= 5 industries on that day)
        u_t, inv, cnt = np.unique(tt, return_inverse=True, return_counts=True)
        keep = cnt[inv] >= 5
        if keep.sum() < 50 or De[keep].sum() < MIN_EVENTS:
            return np.nan, np.nan, int(De[keep].sum())
        yy, tt = yy[keep], tt[keep]
        Ds = [d[keep] for d in Ds]
        Xall = Xall[keep]
        _, inv2 = np.unique(tt, return_inverse=True)
        dmean = np.bincount(inv2, weights=yy) / np.bincount(inv2)
        yy = yy - dmean[inv2]
        if Xall.shape[1] > 0:
            for c_i in range(Xall.shape[1]):
                xm = np.bincount(inv2, weights=Xall[:, c_i]) / np.bincount(inv2)
                Xall[:, c_i] -= xm[inv2]
    if day_cap is not None:
        grp = np.zeros(len(yy), int)
        grp[Ds[0]] = 1
        if len(Ds) > 1:
            grp[Ds[1] & ~Ds[0]] = 2
        w = _cap_day_w(tt, grp, cap=day_cap)
        Z = np.column_stack([np.ones(len(yy))] + [d.astype(float) for d in Ds] + ([Xall] if Xall.shape[1] else []))
        b, t = b8.wls_dk(yy, Z, w, tt, lag)
        return float(b[1]), float(t[1]), ne
    b, t = s10.dfit(yy, Ds, Xall, tt, lag=lag)
    return float(b[0]), float(t[0]), ne


# ---------------------------------------------------------------------------
# 6. Step 0 self-tests (ZERO real event outcomes touched!)
# ---------------------------------------------------------------------------
def selftest_unit() -> list[str]:
    out = []
    # 1) Q1 climax + TB + TL + fixed-10d cond TL hand-crafted sequence
    n = 70
    c = np.full(n, 100.0)
    for i in range(1, 25):
        c[i] = 100.0 + 0.1 * i
    # 20..30: steady uptrend (+2.5 ATR over 20 bars)
    for i in range(15, 35):
        c[i] = 100.0 + 0.3 * (i - 15)  # c[34]=105.7, c[15]=100.0 -> diff 5.7
    h = c + 0.5
    l = c - 0.5
    # bar 35: buy climax jump +3.0 (> 2 ATR since ATR~1.0)
    c[35] = c[34] + 3.0
    h[35], l[35] = c[35] + 0.2, c[34]
    # bars 36..55: Leg 1 drop (-1.5), bounce (+0.8), Leg 2 break below Leg 1 low -> TL=1, cond=1, TB=1
    c0 = c[35]
    traj = [0.2, -0.4, -1.2, -1.5, -1.1, -0.6, -0.8, -1.8, -2.0, -1.6] + [-1.4] * 20
    for k, dv in enumerate(traj):
        c[36 + k] = c0 + dv
        h[36 + k] = c[36 + k] + 0.3
        l[36 + k] = c[36 + k] - 0.3
    em = np.ones(n, bool)
    d = detect_asset_side(h, l, c, em, 1)
    assert d["q1_ev"][35], "unit: buy climax failed to trigger"
    P = {"O": c[:, None].copy(), "H": h[:, None], "L": l[:, None], "C": c[:, None], "OK": np.ones((n, 1), bool)}
    q1r = eval_q1_outcomes(P, np.array([35]), np.array([0]), 1, np.array([d["z"][35]]))
    assert q1r["y_tb"][0] == 1.0 and q1r["y_tl"][0] == 1.0 and q1r["y_tl_cond"][0] == 1.0, f"unit Q1 failed: {q1r}"
    # Upper-bound abort test: bar 37 jumps +2.0 above c0 -> TB=0, TL=0
    c2 = c.copy()
    c2[37] = c0 + 2.0
    P2 = {"O": c2[:, None], "H": (c2 + 0.3)[:, None], "L": (c2 - 0.3)[:, None], "C": c2[:, None], "OK": np.ones((n, 1), bool)}
    q1r2 = eval_q1_outcomes(P2, np.array([35]), np.array([0]), 1, np.array([d["z"][35]]))
    assert q1r2["y_tb"][0] == 0.0 and q1r2["y_tl"][0] == 0.0, f"unit Q1 abort failed: {q1r2}"

    # 2) Open-gap race & same-bar both-hit (cons / opt / tie0)
    Tr = 30
    Or = np.full((Tr, 1), 100.0)
    Hr = np.full((Tr, 1), 100.5)
    Lr = np.full((Tr, 1), 99.5)
    Cr = np.full((Tr, 1), 100.0)
    OKr = np.ones((Tr, 1), bool)
    # Case A: day 1 opens at 103 (past target 102) then drops to 97 (past rev 98) -> open-gap wins (+1 in ALL 3 modes!)
    Or[1, 0], Hr[1, 0], Lr[1, 0], Cr[1, 0] = 103.0, 103.0, 97.0, 98.0
    Pr = {"O": Or, "H": Hr, "L": Lr, "C": Cr, "OK": OKr}
    rr = eval_race(Pr, np.array([0]), np.array([0]), 1, np.array([2.0]))
    assert rr["cons"][0] == 1.0 and rr["opt"][0] == 1.0 and rr["tie0"][0] == 1.0 and not rr["both"][0], f"open-gap failed: {rr}"
    # Case B: open at 100 (between 98 and 102), hits 103 and 97 same day -> cons=-1, opt=+1, tie0=0, both=True
    Or[1, 0] = 100.0
    rr2 = eval_race(Pr, np.array([0]), np.array([0]), 1, np.array([2.0]))
    assert rr2["cons"][0] == -1.0 and rr2["opt"][0] == 1.0 and rr2["tie0"][0] == 0.0 and rr2["both"][0], f"both-hit failed: {rr2}"
    # Case C: mirror check (negate prices, flip rdir)
    Pm = {"O": 200.0 - Or, "H": 200.0 - Lr, "L": 200.0 - Hr, "C": 200.0 - Cr, "OK": OKr}
    rrm = eval_race(Pm, np.array([0]), np.array([0]), -1, np.array([2.0]))
    assert rrm["cons"][0] == rr2["cons"][0] and rrm["opt"][0] == rr2["opt"][0] and rrm["tie0"][0] == rr2["tie0"][0]

    # 3) Q2 stop-entry explicit-target ledger vs s09.run_ledger(entry='stop', k=k) on random paths (no gap)
    rng = np.random.default_rng(1212)
    P_rw, _ = s10._paths(rng, 160, 12, gap_prob=0.0, m=64)
    st_t, st_n = np.meshgrid(np.arange(20, 120), np.arange(12), indexing="ij")
    st_t, st_n = st_t.ravel(), st_n.ravel()
    bad_l = 0
    for side in (1, -1):
        for k_val in (1.0, 2.0):
            for both_m in ("cons", "opt"):
                for t1_m in (None, "forced"):
                    if side < 0 and t1_m is not None:
                        continue
                    r0 = s09.run_ledger(P_rw, st_t, st_n, side, k=k_val, both=both_m, t1=t1_m, cost=0.001, buffer=IDX_BUF)
                    Hs = P_rw["H"][st_t, st_n] if side > 0 else -P_rw["L"][st_t, st_n]
                    Ls = P_rw["L"][st_t, st_n] if side > 0 else -P_rw["H"][st_t, st_n]
                    tpx = (Hs + k_val * (Hs - Ls)) * (1.0 if side > 0 else -1.0)
                    r1 = run_stop_tgt_ledger(P_rw, st_t, st_n, side, tpx, both=both_m, t1=t1_m, cost=0.001, buffer=IDX_BUF)
                    bad_l += int((r0["filled"] != r1["filled"]).sum())
                    f = r0["filled"]
                    bad_l += int((np.abs(r0["net"][f] - r1["net"][f]) > 1e-10).sum())
    assert bad_l == 0, f"Q2 ledger mismatch vs s09: {bad_l}"
    # Gap-past-target test: day 1 opens above tgt_px -> W_PAST_TGT (unfilled)
    Pg = {"O": np.array([[100.0], [106.0]] + [[106.0]] * 35), "H": np.array([[101.0], [107.0]] + [[107.0]] * 35),
          "L": np.array([[99.0], [105.0]] + [[105.0]] * 35), "C": np.array([[100.0], [106.0]] + [[106.0]] * 35),
          "OK": np.ones((37, 1), bool), "UP": np.full((37, 1), np.nan), "DN": np.full((37, 1), np.nan), "TICK": np.zeros((37, 1))}
    rg = run_stop_tgt_ledger(Pg, np.array([0]), np.array([0]), 1, np.array([104.0]), buffer=IDX_BUF)
    assert (not rg["filled"][0]) and rg["why"][0] == W_PAST_TGT, f"gap-past-target failed: {rg}"

    out.append("  手工单元自检（Q1 高潮/TB/TL/中断/条件两段、开盘跳空、同根双碰三口径、镜像、Q2 撮合 vs s09 零误差 & 跳空越过目标）全部通过 ✔")
    return out


def selftest_real(raw: dict) -> tuple[list[str], dict]:
    """Real-data mirror, truncation, t-1 future-leak probe, and exact s10 / s11 consistency (NO event Y!)."""
    out = []
    pxs = {}
    rng = np.random.default_rng(1200)
    for name, (df, ev0) in raw.items():
        px = PathIndex(name, df, ev0)
        pxs[name] = px
        mk = px.mk
        em_all = np.asarray(mk.dates >= np.datetime64(ev0), bool)
        bad_m = bad_s11 = bad_s10 = bad_p = bad_q = 0
        # Check mirror symmetry, s11.detect_A exact match, s10.asset_side exact match on ALL assets
        for n in range(mk.N):
            rows = np.flatnonzero(np.isfinite(mk.P["C"][:, n]))
            h, l, c = mk.P["H"][rows, n], mk.P["L"][rows, n], mk.P["C"][rows, n]
            em = em_all[rows]
            a = detect_asset_side(h, l, c, em, 1)
            K = 2.0 * np.nanmax(h) + 7.0
            m = detect_asset_side(K - l, K - h, K - c, em, -1)
            for k in ["q1_ev", "q1_ctrl", "q1_mild", "q1_3b", "q2_ev", "q2_ctrl", "q2_raw_ev",
                      "q3a_ev", "q3a_ctrl", "q3b_ev", "q3b_ctrl", "q3b_any", "q2_leg_ev"]:
                bad_m += int(not np.array_equal(a[k], m[k]))
            # Exact match vs s11.detect_A on both sides
            for sd in (1, -1):
                hv0, lv0, cv0, _ = s10.side_view(h, l, c, sd)
                hv, lv, cv = (np.round(x, ROUND) for x in (hv0, lv0, cv0))
                ref11 = s11.detect_A(hv, lv, cv, s09.base_feats(hv, lv, cv)["atr20"], em)
                cur = detect_asset_side(h, l, c, em, sd, with_diag=False)
                for k in ["A_narm", "A_toparm", "A_test"]:
                    x, y = cur[k], ref11[k]
                    eq = np.array_equal(x, y) if x.dtype == bool else np.all((x == y) | (np.isnan(x) & np.isnan(y)))
                    bad_s11 += int(not eq)
                # Exact match vs s10.asset_side
                ref10 = s10.asset_side(h, l, c, em, sd)
                bad_s10 += int(not np.array_equal(cur["q3a_all"], ref10["q1_ev"]))
        # Truncation & t-1 future-leak probe (includes mirror side K check for Q1 & Q3b!)
        sample_n = np.arange(mk.N) if mk.N <= 5 else rng.choice(mk.N, size=5, replace=False)
        for n in sample_n.tolist():
            rows = np.flatnonzero(np.isfinite(mk.P["C"][:, n]))
            h, l, c, em = mk.P["H"][rows, n], mk.P["L"][rows, n], mk.P["C"][rows, n], em_all[rows]
            for sd in (1, -1):
                full = detect_asset_side(h, l, c, em, sd, with_diag=False)
                tcs = rng.choice(np.arange(120, len(rows) - 10), size=8, replace=False)
                for tc in tcs.tolist():
                    sub = detect_asset_side(h[:tc + 1], l[:tc + 1], c[:tc + 1], em[:tc + 1], sd, with_diag=False)
                    for k in ["q1_ev", "q1_ctrl", "q2_ev", "q2_ctrl", "q3a_ev", "q3a_ctrl", "q3b_ev", "q3b_ctrl", "q3b_any"]:
                        bad_p += int(sub[k][tc] != full[k][tc])
                    # t-1 probe: replace bar tc with dummy bar (flat at prev close) -> A_narm / A_toparm_real unchanged
                    hd, ld, cd = (np.r_[v[:tc], v[tc - 1]] for v in (h, l, c))
                    dum = detect_asset_side(hd, ld, cd, em[:tc + 1], sd, with_diag=False)
                    for k in ["A_narm", "A_toparm_real"]:
                        x, y = dum[k][tc], full[k][tc]
                        if not (abs(x - y) < 1e-5 or (math.isnan(x) and math.isnan(y))):
                            bad_q += 1
        out.append(f"  {name}：镜像对称 {bad_m} 不一致 | 复现 s11.detect_A {bad_s11} 不一致 | 复现 s10.asset_side {bad_s10} 不一致 | "
                   f"截断重算(含镜像常数K) {bad_p} 不一致 | t−1 探针 {bad_q} 不一致"
                   + (" ✔" if (bad_m + bad_s11 + bad_s10 + bad_p + bad_q) == 0 else " ✘"))
        assert bad_m == 0 and bad_s11 == 0 and bad_s10 == 0 and bad_p == 0 and bad_q == 0, out[-1]
    return out, pxs


def selftest_null_and_synthetic(px: PathIndex) -> list[str]:
    """Step 0.5 null calibration (random pool vs Markov null, NO event Y!) + synthetic end-to-end null pipeline test."""
    out = []
    # 1) Step 0.5 calibration on random non-event pool (SW)
    ex_all = np.zeros_like(px.cand)
    for sd in (1, -1):
        ex_all |= px.feat[sd]["q1_ev"] | px.feat[sd]["q2_ev"] | px.feat[sd]["q3a_ev"] | px.feat[sd]["q3b_ev"]
    pool = build_block_pool(px.mk.P, px.em_wide, ex_all, blk_len=5, n_abins=3, n_vbins=3 if px.name == "SW" else 2)
    # Mirror test on Markov null (price-martingale zero drift: rdir +1 vs -1 should sum to ~0)
    dummy_keys = [(0, 1, 1, 1)] * 500
    dk = np.full(500, 1.0)
    pm, okm, _, dm = simulate_markov_paths(pool, dummy_keys, dk, B=12000, horizon=WIN_MAIN, seed=77, martingale=True)
    rp = eval_race_sim(pm, okm, dm, 1)["cons"]
    rm = eval_race_sim(pm, okm, dm, -1)["cons"]
    # Note: cons is biased negative by both-hit on both sides, so tie0 should be near-exact opposites!
    rp0 = eval_race_sim(pm, okm, dm, 1)["tie0"]
    rm0 = eval_race_sim(pm, okm, dm, -1)["tie0"]
    out.append(f"  {px.name} 零假设镜像校验（鞅零漂移 k=1.0, tie0）：rdir=+1 {rp0:+.4f} vs rdir=−1 {rm0:+.4f}（之和 {rp0 + rm0:+.4f}）")
    assert abs(rp0 + rm0) < 0.015, f"null mirror imbalance: {rp0}, {rm0}"

    # Step 0.5 calibration: random non-event pool vs Markov null on Q1 (TB/TL) & fixed k=1.5 race
    non_ev = ~ex_all[px.tt, px.nn]
    rng = np.random.default_rng(1205)
    pick = np.flatnonzero(non_ev)
    if len(pick) > 8000:
        pick = rng.choice(pick, size=8000, replace=False)
    mk_sub = np.zeros(len(px.tt), bool)
    mk_sub[pick] = True
    keys_r = px.event_keys(pool, mk_sub, 1)
    # Real random pool outcomes
    q1_r = eval_q1_outcomes(px.mk.P, px.tt[pick], px.nn[pick], 1, px.at(1, "z")[pick])
    race_r = eval_race(px.mk.P, px.tt[pick], px.nn[pick], 1, 1.5 * px.at(1, "z")[pick])
    # Markov null on same keys
    n_q1 = null_summary(pool, keys_r, None, "q1", 1, B_main=12000, n_boot=40, B_boot=1000, seed=55)
    n_rc = null_summary(pool, keys_r, np.full(len(keys_r), 1.5), "race", 1, B_main=12000, n_boot=40, B_boot=1000, seed=56)
    out.append(f"  {px.name} 第 0.5 步校准（随机池 vs 链式分块零假设，不看任何事件 Y）："
               f"TB {np.nanmean(q1_r['y_tb'])*100:.1f}% vs {n_q1['tb']*100:.1f}% | "
               f"TL {np.nanmean(q1_r['y_tl'])*100:.1f}% vs {n_q1['tl']*100:.1f}% | "
               f"k=1.5 赛跑 {np.nanmean(race_r['cons']):+.3f} vs {n_rc['cons']:+.3f}")

    # 2) Synthetic end-to-end null pipeline test (only run once on SW for speed: RW panel + Markov 1-day real-bar panel)
    if px.name == "SW":
        pool1_syn = build_block_pool(px.mk.P, px.em_wide, np.zeros_like(px.cand), blk_len=1, n_abins=3, n_vbins=3)
        for syn_nm, use_blk in [("纯随机游走合成面板(20标的×600日)", False), ("真实K线马尔可夫合成面板(30标的×800日)", True)]:
            N_s, T_s = (30, 800) if use_blk else (20, 600)
            if not use_blk:
                P_s, _ = s10._paths(np.random.default_rng(1299), T_s, N_s, gap_prob=0.3, m=32)
            else:
                # 1-day Markov real-bar panel: preserves fat tails, gaps, skewness & vol clustering, zero serial corr
                keys_s = [(0, 1, 1, 1)] * N_s
                ps, oks, _, _ = simulate_markov_paths(pool1_syn, keys_s, np.ones(N_s), B=N_s, horizon=T_s, seed=1298, martingale=True)
                P_s = {"O": (ps[:, :, 0] * 100).T, "H": (ps[:, :, 1] * 100).T, "L": (ps[:, :, 2] * 100).T,
                       "C": (ps[:, :, 3] * 100).T, "OK": oks.T, "UP": np.full((T_s, N_s), np.nan),
                       "DN": np.full((T_s, N_s), np.nan), "TICK": np.zeros((T_s, N_s))}
            # Build synthetic DataFrame and run PathIndex + full Q1/Q2/Q3a/Q3b regression + null
            dts = pd.bdate_range("2017-01-02", periods=T_s)
            dfs = []
            for j in range(N_s):
                dfs.append(pd.DataFrame({"asset": f"S{j:02d}", "date": dts, "open": P_s["O"][:, j],
                                         "high": P_s["H"][:, j], "low": P_s["L"][:, j], "close": P_s["C"][:, j]}))
            spx = PathIndex("SW", pd.concat(dfs, ignore_index=True), dts[80], with_diag=False)
            ex_s = np.zeros_like(spx.cand)
            for sd in (1, -1):
                ex_s |= spx.feat[sd]["q1_ev"] | spx.feat[sd]["q2_ev"] | spx.feat[sd]["q3a_ev"] | spx.feat[sd]["q3b_ev"]
            spool = build_block_pool(spx.mk.P, spx.em_wide, ex_s, blk_len=5, n_abins=3, n_vbins=2)
            n_pass = 0
            diffs = []
            for code, qg, sd, _, ek, ck, rrel in PRIMARY:
                E = spx.at(sd, ek)
                if E.sum() < 15:
                    continue
                if qg in ("Q1a", "Q1b"):
                    res = eval_q1_outcomes(spx.mk.P, spx.tt, spx.nn, sd, spx.at(sd, "z"))
                    y = res["y_tb"] if qg == "Q1a" else res["y_tl"]
                    b, t, _ = fit_index_reg(spx, y, E, None, layers=("geo", "ret1", "trend"))
                    mu_e, se_e = _dk_mean_se(y[E], spx.tt[E])
                    nl = null_summary(spool, spx.event_keys(spool, E, sd), None, "q1", sd, B_main=4000, n_boot=20, B_boot=800, seed=31)
                    mu0, se0 = (nl["tb"], nl["tb_se"]) if qg == "Q1a" else (nl["tl"], nl["tl_se"])
                    t0 = (mu_e - mu0) / math.sqrt(se_e ** 2 + se0 ** 2 + EPS)
                    diffs.append(b)
                    if b >= 0.05 and t >= 2.0 and t0 >= 2.0:
                        n_pass += 1
                else:
                    rdir = sd * rrel
                    z_use = spx.at(sd, "z10") if qg in ("Q2", "Q3a") else spx.at(sd, "z")
                    d_col = {"Q2": "q2_d", "Q3a": "q3a_d", "Q3b": "q3b_d"}[qg]
                    d_e = spx.at(sd, d_col)
                    k_real = d_e[E] / z_use[E]
                    k_all = np.random.default_rng(42).choice(k_real, size=len(spx.tt), replace=True)
                    k_all[E] = k_real
                    d_all = k_all * z_use
                    rc = eval_race(spx.mk.P, spx.tt, spx.nn, rdir, d_all)
                    y = rc["cons"]
                    dpx = d_all / spx.mk.P["C"][spx.tt, spx.nn]
                    b, t, _ = fit_index_reg(spx, y, E, None, k_arr=k_all, d_px=dpx)
                    mu_e, se_e = _dk_mean_se(y[E], spx.tt[E])
                    nl = null_summary(spool, spx.event_keys(spool, E, sd), k_real, "race", rdir, B_main=4000, n_boot=20, B_boot=800, seed=32)
                    t0 = (mu_e - nl["cons"]) / math.sqrt(se_e ** 2 + nl["se"] ** 2 + EPS)
                    diffs.append(b)
                    if b >= 0.10 and t >= 2.0 and t0 >= 2.0:
                        n_pass += 1
            mean_d = float(np.nanmean(diffs)) if diffs else np.nan
            out.append(f"  假数据端到端测试【{syn_nm}】：跨 {len(diffs)} 题平均超额 {mean_d:+.4f}，双门槛假阳性过关数 {n_pass}/10"
                       + (" ✔" if n_pass == 0 else " ⚠"))
    return out


def step0(raw: dict) -> list[str]:
    L = ["=" * 115, "Study 12 · 第 0 步：自检 + 零假设校准 + 假数据端到端 + 样本量 / MDE / 事前平衡（不看任何真实事件 Y）", "=" * 115]
    L.append("\n[1] 旧脚本只读校验（SHA-256 前 16 位）")
    for fn in ["09_trade_ledger_index.py", "10_trading_range_index.py", "11_level_memory_index.py"]:
        L.append(f"  studies/{fn}: {sha256_file(REPO_ROOT / 'studies' / fn)}")
    L += ["", "[2] 手工单元测试 & Q2 显式目标撮合自检"] + selftest_unit()
    L += ["", "[3] 真实数据一致性自检（镜像 / 复现 s10 & s11 / 截断含镜像常数 K / t−1 探针）"]
    r_out, pxs = selftest_real(raw)
    L += r_out
    L += ["", "[4] 第 0.5 步零假设校准 & 假数据端到端测试（不看任何真实事件 Y）"]
    for name in ("SPX", "SW"):
        L += selftest_null_and_synthetic(pxs[name])

    L += ["", "[5] 样本量、最小可检测效应（MDE 估算）、距离重叠与事前特征平衡（严格不看事件 Y）"]
    for name in ("SPX", "SW"):
        px = pxs[name]
        L.append(f"\n  —— {name}（{px.mk.N} 个标的，入池候选行 {len(px.tt):,}；Q2 确认后因 d < 0.5 ATR 剔除 多/空={px.q2_drop_d[1]}/{px.q2_drop_d[-1]}）")
        for code, qg, sd, lab, ek, ck, _ in PRIMARY:
            E, C = px.at(sd, ek), px.at(sd, ck) & ~px.at(sd, ek)
            ne, nc = int(E.sum()), int(C.sum())
            # Effective years (year with >= 10 events)
            yrs = np.unique(px.year)
            yc = {int(y): int((E & (px.year == y)).sum()) for y in yrs}
            n_eff_y = sum(1 for v in yc.values() if v >= 10)
            # Crude MDE estimate (binary sd<=0.5 for Q1, race sd<=0.9 for Q2/Q3; conservative cluster factor ~1.4)
            sd_y = 0.5 if qg in ("Q1a", "Q1b") else 0.9
            mde = 2.0 * 1.4 * sd_y / math.sqrt(max(ne, 1))
            thr = 0.05 if qg in ("Q1a", "Q1b") else 0.10
            pflag = " [检验力不足]" if mde > thr else ""
            # Pre-event balance
            r1 = px.at(sd, "ret1")
            r5 = px.at(sd, "r5v")
            kcls = px.mk.F["kcls"][px.tt, px.nn]
            vr = px.at(1, "v_ratio")
            k_info = ""
            if qg in ("Q2", "Q3a", "Q3b"):
                z_u = px.at(sd, "z10") if qg in ("Q2", "Q3a") else px.at(sd, "z")
                d_col = {"Q2": "q2_d", "Q3a": "q3a_d", "Q3b": "q3b_d"}[qg]
                dc_col = {"Q2": "q2_d", "Q3a": "q3a_d", "Q3b": "q3b_ctrl_d"}[qg]
                ke = px.at(sd, d_col)[E] / z_u[E]
                kc = px.at(sd, dc_col)[C] / z_u[C]
                if len(ke) and len(kc):
                    lo = max(float(np.nanquantile(ke, 0.05)), float(np.nanquantile(kc, 0.05)))
                    hi = min(float(np.nanquantile(ke, 0.95)), float(np.nanquantile(kc, 0.95)))
                    ov = float(np.mean((ke >= lo) & (ke <= hi))) * 100
                    k_info = f" | k中位 事{np.nanmedian(ke):.2f}/对{np.nanmedian(kc):.2f} 共同[5-95%]覆盖{ov:.0f}%"
            ex_info = ""
            if qg == "Q3b":
                ex_info = f" | 本就在射程内(不计)={int(px.at(sd, 'q3b_in').sum()):,}"
            if qg == "Q3a":
                ex_info = f" | 全量q1_ev={int(px.at(sd, 'q3a_all').sum()):,}"
            L.append(f"    {lab:<20s} 事 {ne:>5,} 对 {nc:>5,} | 有效年(≥10) {n_eff_y}/{len(yrs)} | MDE≈{mde:.3f}{pflag}{k_info}{ex_info}")
            L.append(f"      事前特征(事/对)：ret1 {np.nanmean(r1[E])*100:+.2f}%/{np.nanmean(r1[C])*100:+.2f}% | "
                     f"r5 {np.nanmean(r5[E])*100:+.2f}%/{np.nanmean(r5[C])*100:+.2f}% | "
                     f"kcls {np.nanmean(kcls[E]):.2f}/{np.nanmean(kcls[C]):.2f} | "
                     f"vol比 {np.nanmean(vr[E]):.2f}/{np.nanmean(vr[C]):.2f}")
    return L


# ---------------------------------------------------------------------------
# 7. Step 1: Full index analysis (SPX + 31 SW L1 industries)
# ---------------------------------------------------------------------------
def analyze_market(px: PathIndex) -> tuple[list[dict], list[str]]:
    name = px.name
    is_spx = (name == "SPX")
    # Per plan v2 (Opus #19/5.3): SPX full-sample main omits K-line shape controls (pre-2010 H/L stretched)
    main_layers_q1 = ("ret1", "trend") if is_spx else ("geo", "ret1", "trend")
    main_layers_race = ("dist", "ret1", "trend") if is_spx else ("dist", "geo", "ret1", "trend")

    # Per-question-family block pools (exclude ONLY that family's own events + 5 days, per pre-registration)
    pools5, pools1 = {}, {}
    for fam, ekey in [("Q1", "q1_ev"), ("Q2", "q2_ev"), ("Q3a", "q3a_ev"), ("Q3b", "q3b_ev")]:
        ex_f = px.feat[1][ekey] | px.feat[-1][ekey]
        pools5[fam] = build_block_pool(px.mk.P, px.em_wide, ex_f, blk_len=5, n_abins=3, n_vbins=3 if not is_spx else 2)
        pools1[fam] = build_block_pool(px.mk.P, px.em_wide, ex_f, blk_len=1, n_abins=3, n_vbins=3 if not is_spx else 2)

    # Market daily sigma & drift for 256-step RW auxiliary null
    r1_all = px.at(1, "ret1")
    sig_day = float(np.nanstd(r1_all))
    drift_day = float(np.nanmean(r1_all))

    # Cache Q1 outcomes per side
    q1_cache = {sd: eval_q1_outcomes(px.mk.P, px.tt, px.nn, sd, px.at(sd, "z")) for sd in (1, -1)}
    q1_t_cache = {sd: eval_q1_outcomes(px.mk.P, px.tt, px.nn, sd, px.at(sd, "atr_t")) for sd in (1, -1)}

    rows_out = []
    notes = []
    C_px = px.mk.P["C"][px.tt, px.nn]
    yrs_all = np.unique(px.year)
    spx_periods = [(1984, 1992), (1993, 2001), (2002, 2010), (2011, 2019)]
    sw_halves = [(2016, 2020), (2021, 2024)]
    spx_halves = [(1984, 2001), (2002, 2019)]

    for code, qg, sd, lab, ek, ck, rrel in PRIMARY:
        fam = "Q1" if qg in ("Q1a", "Q1b") else qg
        pool5, pool1 = pools5[fam], pools1[fam]
        E = px.at(sd, ek)
        C_ctrl = px.at(sd, ck) & ~E
        ne, nc = int(E.sum()), int(C_ctrl.sum())
        rec: dict = {"market": name, "code": code, "qg": qg, "side": sd, "label": lab, "n_ev": ne, "n_ctrl": nc}

        if qg in ("Q1a", "Q1b"):
            res = q1_cache[sd]
            y = res["y_tb"] if qg == "Q1a" else res["y_tl"]
            y_t = (q1_t_cache[sd]["y_tb"] if qg == "Q1a" else q1_t_cache[sd]["y_tl"])  # ATR(t) sensitivity
            # Non-overlap
            E_no = non_overlap_mask(px.tt, px.nn, E, None, WIN_TB if qg == "Q1a" else WIN_MAIN)
            # Raw event mean & DK SE
            mu_e, se_e = _dk_mean_se(y[E], px.tt[E])
            # Nulls
            ekeys = px.event_keys(pool5, E, sd) if ne >= MIN_EVENTS else []
            if len(ekeys) >= MIN_EVENTS:
                n5 = null_summary(pool5, ekeys, None, "q1", sd, seed=120 + sd)
                n1 = null_summary(pool1, ekeys, None, "q1", sd, B_main=10000, n_boot=20, B_boot=1000, seed=220 + sd)
                rw = rw256_null(None, "q1", sd, sig_day, drift_day, seed=320 + sd)
                key_n = "tb" if qg == "Q1a" else "tl"
                mu0, se0, mu0_m = n5[key_n], n5[key_n + "_se"], n5[key_n + "_mart"]
                t0 = (mu_e - mu0) / math.sqrt(se_e ** 2 + se0 ** 2 + EPS) if math.isfinite(mu_e) else np.nan
                mu_iid = n1[key_n]
                mu_rw_m, mu_rw_d = rw["mart"][key_n], rw["drift"][key_n]
            else:
                mu0 = se0 = mu0_m = t0 = mu_iid = mu_rw_m = mu_rw_d = np.nan

            # Regressions & decomposition
            ex, ex_t, _ = fit_index_reg(px, y, E, None, layers=main_layers_q1)
            ex_abs, _, _ = fit_index_reg(px, y, E, None, layers=())
            ex_no_r1, _, _ = fit_index_reg(px, y, E, None, layers=("trend",))
            ex_geo, _, _ = fit_index_reg(px, y, E, None, layers=("geo", "ret1", "trend"))
            ex_no, ex_no_t, _ = fit_index_reg(px, y, E_no, None, layers=main_layers_q1)
            ex_atrt, _, _ = fit_index_reg(px, y_t, E, None, layers=main_layers_q1)
            ctrl_d, ctrl_t, _ = fit_index_reg(px, y, E, C_ctrl, layers=main_layers_q1, contrast=True)
            # Mild trend new-high auxiliary control (WITHOUT ret1 control, per plan!)
            C_mild = px.at(sd, "q1_mild") & ~E
            mild_d, mild_t, _ = fit_index_reg(px, y, E, C_mild, layers=("trend",), contrast=True)
            # 3-bar climax diagnostic
            E3 = px.at(sd, "q1_3b")
            ex_3b, ex_3b_t, n3b = fit_index_reg(px, y, E3, None, layers=main_layers_q1)

            rec.update(dict(
                mu_e=mu_e, se_e=se_e, mu0=mu0, mu0_mart=mu0_m, t0=t0, mu_iid=mu_iid, mu_rw_m=mu_rw_m, mu_rw_d=mu_rw_d,
                ex=ex, ex_t=ex_t, ex_abs=ex_abs, ex_no_r1=ex_no_r1, ex_geo=ex_geo, ex_no=ex_no, ex_no_t=ex_no_t,
                ex_atrt=ex_atrt, ctrl=ctrl_d, ctrl_t=ctrl_t, mild=mild_d, mild_t=mild_t,
                ex_3b=ex_3b, ex_3b_t=ex_3b_t, n_3b=n3b,
                ex_opt=ex, ex_tie0=ex, both_rate=0.0, atr_adv=np.nan, blk1_share=np.nan,
            ))
            if qg == "Q1a":
                rec["tb_strict"] = float(np.nanmean(res["y_tb_strict"][E])) if ne else np.nan
                rec["tb_c10"] = float(np.nanmean(res["y_tb_c10"][E])) if ne else np.nan
                rec["surv5"] = float(np.nanmean(res["surv5"][E])) if ne else np.nan
                rec["surv10"] = float(np.nanmean(res["surv10"][E])) if ne else np.nan
                rec["surv20"] = float(np.nanmean(res["surv20"][E])) if ne else np.nan
            else:
                # Q1b conditional two-legs (fixed 10 market days after Leg 1)
                yc = res["y_tl_cond"]
                l1m = res["l1_hit"]
                mu_c, se_c = _dk_mean_se(yc[E & l1m], px.tt[E & l1m])
                ex_c, ex_ct, n_c = fit_index_reg(px, yc, E & l1m, None, layers=main_layers_q1, sub=l1m)
                mu0_c = n5["cond"] if len(ekeys) >= MIN_EVENTS else np.nan
                se0_c = n5["cond_se"] if len(ekeys) >= MIN_EVENTS else np.nan
                t0_c = (mu_c - mu0_c) / math.sqrt(se_c ** 2 + se0_c ** 2 + EPS) if (math.isfinite(mu_c) and math.isfinite(mu0_c)) else np.nan
                # Stratified by Leg 1 lag (1-5 / 6-10 / 11-20)
                lag_bins = []
                for a_, b_ in [(1, 5), (6, 10), (11, 20)]:
                    ml = E & l1m & (res["l1_lag"] >= a_) & (res["l1_lag"] <= b_)
                    lag_bins.append(float(np.nanmean(yc[ml])) if ml.sum() >= 5 else np.nan)
                rec.update(dict(
                    cond_e=mu_c, cond_0=mu0_c, cond_t0=t0_c, cond_ex=ex_c, cond_ex_t=ex_ct, cond_n=int((E & l1m).sum()),
                    cond_lags=lag_bins, tl_loose=float(np.nanmean(res["y_tl_loose"][E])) if ne else np.nan,
                    tbtl_joint=float(np.nanmean(((res["y_tb"] == 1) & (res["y_tl"] == 1))[E])) if ne else np.nan,
                ))
            # SW demeaned, top-10-days removed, daily cap
            if not is_spx:
                rec["ex_dm"], rec["ex_dm_t"], _ = fit_index_reg(px, y, E, None, layers=main_layers_q1, demean_day=True)
                rec["ex_cap"], rec["ex_cap_t"], _ = fit_index_reg(px, y, E, None, layers=main_layers_q1, day_cap=SW_DAY_CAP)
                # Top-10 event days sensitivity
                u_t, cnt_t = np.unique(px.tt[E], return_counts=True)
                top10 = u_t[np.argsort(-cnt_t)[:10]]
                top10_share = float(np.isin(px.tt[E], top10).mean()) if ne else np.nan
                ex_no10, ex_no10_t, _ = fit_index_reg(px, y, E, None, layers=main_layers_q1, sub=~np.isin(px.tt, top10))
                rec.update(top10_share=top10_share, ex_no10=ex_no10, ex_no10_t=ex_no10_t)
            else:
                sub10 = px.year >= 2010
                rec["ex_1019"], rec["ex_1019_t"], rec["n_1019"] = fit_index_reg(px, y, E, None, layers=("geo", "ret1", "trend"), sub=sub10)
            # Period stability
            per_list = []
            if not is_spx:
                for yr in yrs_all:
                    sy = px.year == yr
                    if (E & sy).sum() >= 10:
                        b_, _, _ = fit_index_reg(px, y, E, None, layers=main_layers_q1, sub=sy)
                        per_list.append(b_)
                    else:
                        per_list.append(np.nan)
                halves = [fit_index_reg(px, y, E, None, layers=main_layers_q1, sub=(px.year >= a_) & (px.year <= b_))[0] for a_, b_ in sw_halves]
            else:
                for a_, b_ in spx_periods:
                    sy = (px.year >= a_) & (px.year <= b_)
                    per_list.append(fit_index_reg(px, y, E, None, layers=main_layers_q1, sub=sy)[0] if (E & sy).sum() >= MIN_EVENTS else np.nan)
                halves = [fit_index_reg(px, y, E, None, layers=main_layers_q1, sub=(px.year >= a_) & (px.year <= b_))[0] for a_, b_ in spx_halves]
            rec["per"], rec["halves"] = per_list, halves

        else:
            # Q2, Q3a, Q3b: symmetric race Y_net
            rdir = sd * rrel
            z_u = px.at(sd, "z10") if qg in ("Q2", "Q3a") else px.at(sd, "z")
            d_col = {"Q2": "q2_d", "Q3a": "q3a_d", "Q3b": "q3b_d"}[qg]
            dc_col = {"Q2": "q2_d", "Q3a": "q3a_d", "Q3b": "q3b_ctrl_d"}[qg]
            d_ev_arr = px.at(sd, d_col)
            d_ct_arr = px.at(sd, dc_col)
            k_ev = d_ev_arr[E] / z_u[E] if ne else np.array([])

            # Build pseudo-target distance array for all candidate rows
            rng = np.random.default_rng(1250 + hash(code) % 1000)
            k_all = rng.choice(k_ev, size=len(px.tt), replace=True) if ne >= MIN_EVENTS else np.full(len(px.tt), np.nan)
            k_all[E] = k_ev
            if nc >= MIN_EVENTS:
                k_all[C_ctrl] = d_ct_arr[C_ctrl] / z_u[C_ctrl]
            d_all = k_all * z_u
            d_px = d_all / C_px

            # Random pool exclusion for Q3b: exclude any bar with a live real level in [0, 2.0]*z
            sub_base = None
            if qg == "Q3b":
                sub_base = E | C_ctrl | (~px.at(sd, "q3b_any"))

            rc = eval_race(px.mk.P, px.tt, px.nn, rdir, d_all)
            y, yo, y0, ycls = rc["cons"], rc["opt"], rc["tie0"], rc["cls"]
            E_no = non_overlap_mask(px.tt, px.nn, E, rc["hit_day"], WIN_MAIN)

            mu_e, se_e = _dk_mean_se(y[E], px.tt[E])
            mu_e_opt, _ = _dk_mean_se(yo[E], px.tt[E])
            mu_e_t0, _ = _dk_mean_se(y0[E], px.tt[E])

            ekeys = px.event_keys(pool5, E, sd) if ne >= MIN_EVENTS else []
            if len(ekeys) >= MIN_EVENTS:
                n5 = null_summary(pool5, ekeys, k_ev, "race", rdir, seed=420 + hash(code) % 100)
                n1 = null_summary(pool1, ekeys, k_ev, "race", rdir, B_main=10000, n_boot=20, B_boot=1000, seed=520 + hash(code) % 100)
                rw = rw256_null(k_ev, "race", rdir, sig_day, drift_day, seed=620 + hash(code) % 100)
                mu0, se0, mu0_m = n5["cons"], n5["se"], n5["mart"]
                t0 = (mu_e - mu0) / math.sqrt(se_e ** 2 + se0 ** 2 + EPS) if math.isfinite(mu_e) else np.nan
                mu_iid = n1["cons"]
                mu_rw_m, mu_rw_d = rw["mart"]["cons"], rw["drift"]["cons"]
                blk1_sh = n5["blk1_share"]
            else:
                mu0 = se0 = mu0_m = t0 = mu_iid = mu_rw_m = mu_rw_d = blk1_sh = np.nan

            # Main regression & decomposition
            ex, ex_t, _ = fit_index_reg(px, y, E, None, k_arr=k_all, d_px=d_px, layers=main_layers_race, sub=sub_base)
            ex_opt, _, _ = fit_index_reg(px, yo, E, None, k_arr=k_all, d_px=d_px, layers=main_layers_race, sub=sub_base)
            ex_t0, _, _ = fit_index_reg(px, y0, E, None, k_arr=k_all, d_px=d_px, layers=main_layers_race, sub=sub_base)
            ex_cls, _, _ = fit_index_reg(px, ycls, E, None, k_arr=k_all, d_px=d_px, layers=main_layers_race, sub=sub_base)
            ex_abs, _, _ = fit_index_reg(px, y, E, None, k_arr=k_all, d_px=d_px, layers=(), sub=sub_base)
            ex_no_r1, _, _ = fit_index_reg(px, y, E, None, k_arr=k_all, d_px=d_px, layers=("dist", "trend"), sub=sub_base)
            ex_no, ex_no_t, _ = fit_index_reg(px, y, E_no, None, k_arr=k_all, d_px=d_px, layers=main_layers_race, sub=sub_base)
            # ATR-unit advantage = Y_net * k
            y_atr = y * k_all
            atr_adv, _, _ = fit_index_reg(px, y_atr, E, None, k_arr=k_all, d_px=d_px, layers=main_layers_race, sub=sub_base)

            # Exclusive control (for Q2: restrict to common k support [5%, 95%]!)
            sub_ctrl = sub_base
            if qg == "Q2" and ne >= MIN_EVENTS and nc >= MIN_EVENTS:
                kc = k_all[C_ctrl]
                lo = max(float(np.nanquantile(k_ev, 0.05)), float(np.nanquantile(kc, 0.05)))
                hi = min(float(np.nanquantile(k_ev, 0.95)), float(np.nanquantile(kc, 0.95)))
                sub_ctrl = (k_all >= lo) & (k_all <= hi)
            ctrl_d, ctrl_t, _ = fit_index_reg(px, y, E, C_ctrl, k_arr=k_all, d_px=d_px, layers=main_layers_race, contrast=True, sub=sub_ctrl)

            # Survival & breakdown on events
            hd_e = rc["hit_day"][E]
            rec.update(dict(
                mu_e=mu_e, mu_e_opt=mu_e_opt, mu_e_tie0=mu_e_t0, se_e=se_e, mu0=mu0, mu0_mart=mu0_m, t0=t0,
                mu_iid=mu_iid, mu_rw_m=mu_rw_m, mu_rw_d=mu_rw_d, blk1_share=blk1_sh,
                ex=ex, ex_t=ex_t, ex_opt=ex_opt, ex_tie0=ex_t0, ex_cls=ex_cls, ex_abs=ex_abs, ex_no_r1=ex_no_r1,
                ex_no=ex_no, ex_no_t=ex_no_t, ctrl=ctrl_d, ctrl_t=ctrl_t, atr_adv=atr_adv,
                hit_tgt=float(np.nanmean(y[E] == 1.0)) if ne else np.nan,
                hit_rev=float(np.nanmean(y[E] == -1.0)) if ne else np.nan,
                unres=float(np.nanmean(y[E] == 0.0)) if ne else np.nan,
                both_rate=float(np.nanmean(rc["both"][E])) if ne else np.nan,
                surv5=float(np.nanmean(hd_e > 5)) if ne else np.nan,
                surv10=float(np.nanmean(hd_e > 10)) if ne else np.nan,
                surv20=float(np.nanmean(hd_e > 20)) if ne else np.nan,
                post5=float(np.nanmean(rc["post5"][E])) if ne else np.nan,
            ))

            # Question-specific diagnostics
            if qg == "Q2" and ne >= MIN_EVENTS:
                # 1) Unified multiplier grid 0.6/0.8/1.0/1.2/1.4 * hgt + 1.0x stall stat
                hgt_e = px.at(sd, "q2_hgt")[E]
                top_e = px.at(sd, "q2_top")[E]
                cv0_e = C_px[E] if sd > 0 else (2.0 * np.nanmax(px.mk.P["H"]) + 1.0 - C_px[E])
                # Note: easier: d(mult) = d_ev + (mult - 1.0) * hgt_e! Exact regardless of side!
                m_ynet, m_post5 = [], []
                for mult in Q2_MULTS:
                    dm = d_ev_arr[E] + (mult - 1.0) * hgt_e
                    okm = dm >= D_MIN_ATR * z_u[E] - TIE
                    if okm.sum() >= MIN_EVENTS:
                        rm = eval_race(px.mk.P, px.tt[E][okm], px.nn[E][okm], rdir, dm[okm])
                        m_ynet.append(float(np.nanmean(rm["cons"])))
                        m_post5.append(float(np.nanmean(rm["post5"])))
                    else:
                        m_ynet.append(np.nan)
                        m_post5.append(np.nan)
                stall = m_post5[2] - 0.5 * (m_post5[1] + m_post5[3]) if all(math.isfinite(x) for x in m_post5[1:4]) else np.nan
                rec["q2_mult_y"], rec["q2_mult_p5"], rec["q2_stall"] = m_ynet, m_post5, stall
                # 2) Unconfirmed first_break diagnostic
                Er = px.at(sd, "q2_raw_ev")
                if Er.sum() >= MIN_EVENTS:
                    rr_raw = eval_race(px.mk.P, px.tt[Er], px.nn[Er], rdir, px.at(sd, "q2_raw_d")[Er])
                    rec["q2_raw_mu"], rec["q2_raw_n"] = float(np.nanmean(rr_raw["cons"])), int(Er.sum())
                # 3) Leg 1 = Leg 2 & Measuring Gap diagnostics
                for dk_nm, ek_nm, dc_nm in [("leg", "q2_leg_ev", "q2_leg_d"), ("mg", "q2_mg_ev", "q2_mg_d")]:
                    Ed = px.at(sd, ek_nm)
                    if Ed.sum() >= MIN_EVENTS:
                        dd = px.at(sd, dc_nm)[Ed]
                        rd = eval_race(px.mk.P, px.tt[Ed], px.nn[Ed], rdir, dd)
                        rec[f"q2_{dk_nm}_mu"], rec[f"q2_{dk_nm}_n"] = float(np.nanmean(rd["cons"])), int(Ed.sum())
                # 4) Structural stop (target top+hgt vs stop bot)
                d_bot = (px.at(sd, "q2_hgt")[E] * 2.0) - d_ev_arr[E]  # since top+hgt - c = d_ev => c - bot = 2*hgt - d_ev
                rs_str = eval_race(px.mk.P, px.tt[E], px.nn[E], rdir, d_ev_arr[E], d_bot)
                theo_p = float(np.nanmean(d_bot / (d_ev_arr[E] + d_bot)))
                rec["q2_struct_win"], rec["q2_struct_theo"] = float(np.nanmean(rs_str["cons"] == 1.0)), theo_p
                # 5) 40-day sensitivity (DK lag 40)
                inb40 = px.tt + IDX_BUF_40 <= px.mk.T - 1
                rc40 = eval_race(px.mk.P, px.tt, px.nn, rdir, d_all, horizon=WIN_SENS, min_days=MIN_SENS)
                rec["q2_ex40"], rec["q2_ex40_t"], _ = fit_index_reg(
                    px, rc40["cons"], E, None, k_arr=k_all, d_px=d_px, layers=main_layers_race, sub=inb40, lag=LAG_40)
                # 6) Q2 ledger diagnostic: MM target vs fixed 1R on confirmed events
                l_mm = run_stop_tgt_ledger(px.mk.P, px.tt[E], px.nn[E], sd, px.at(sd, "q2_tgt_real")[E], cost=px.mk.cost)
                l_1r = s09.run_ledger(px.mk.P, px.tt[E], px.nn[E], sd, k=1.0, cost=px.mk.cost, buffer=IDX_BUF)
                rec["led_mm_net"] = float(np.nanmean(l_mm["net"][l_mm["filled"]])) if l_mm["filled"].any() else np.nan
                rec["led_1r_net"] = float(np.nanmean(l_1r["net"][l_1r["filled"]])) if l_1r["filled"].any() else np.nan
                rec["led_mm_win"] = float(l_mm["win"][l_mm["filled"]].mean()) if l_mm["filled"].any() else np.nan
                rec["led_1r_win"] = float(l_1r["win"][l_1r["filled"]].mean()) if l_1r["filled"].any() else np.nan
                rec["led_mm_past"] = float((l_mm["why"] == W_PAST_TGT).mean())

            elif qg == "Q3a" and ne >= MIN_EVENTS:
                # 1) Downtrend control & toward-edge slice
                C_dn = px.at(sd, "q3a_ctrl_dn") & ~E
                if C_dn.sum() >= MIN_EVENTS:
                    k_dn = k_all.copy()
                    k_dn[C_dn] = px.at(sd, "q3a_d")[C_dn] / z_u[C_dn]
                    rc_dn = eval_race(px.mk.P, px.tt, px.nn, rdir, k_dn * z_u)
                    rec["q3a_ctrl_dn"], rec["q3a_ctrl_dn_t"], _ = fit_index_reg(
                        px, rc_dn["cons"], E, C_dn, k_arr=k_dn, d_px=k_dn * z_u / C_px, layers=main_layers_race, contrast=True)
                tw = px.at(sd, "ret1") < 0  # moved toward edge on day t
                rec["q3a_tw"], rec["q3a_tw_t"], _ = fit_index_reg(
                    px, y, E, C_ctrl, k_arr=k_all, d_px=d_px, layers=main_layers_race, contrast=True, sub=tw)
                # 2) Full q1_ev (including 0.33..0.5 ATR) diagnostic
                Ea = px.at(sd, "q3a_all")
                ra = eval_race(px.mk.P, px.tt[Ea], px.nn[Ea], rdir, px.at(sd, "q3a_d")[Ea])
                rec["q3a_all_mu"], rec["q3a_all_both"] = float(np.nanmean(ra["cons"])), float(np.nanmean(ra["both"]))
                # 3) Ledger diagnostic via UNTOUCHED s09.run_ledger(entry='limit', tgt_px=mid) vs fixed 1R
                l_mid = s09.run_ledger(px.mk.P, px.tt[E], px.nn[E], sd, entry="limit", tgt_px=px.at(sd, "q3a_mid_real")[E],
                                       cost=px.mk.cost, buffer=IDX_BUF)
                l_1r = s09.run_ledger(px.mk.P, px.tt[E], px.nn[E], sd, entry="limit", k=1.0, cost=px.mk.cost, buffer=IDX_BUF)
                rec["led_mid_net"] = float(np.nanmean(l_mid["net"][l_mid["filled"]])) if l_mid["filled"].any() else np.nan
                rec["led_1r_net"] = float(np.nanmean(l_1r["net"][l_1r["filled"]])) if l_1r["filled"].any() else np.nan
                rec["led_mid_win"] = float(l_mid["win"][l_mid["filled"]].mean()) if l_mid["filled"].any() else np.nan
                rec["led_1r_win"] = float(l_1r["win"][l_1r["filled"]].mean()) if l_1r["filled"].any() else np.nan
                rec["led_mid_hold"] = float(np.nanmedian(l_mid["hold"][l_mid["filled"]])) if l_mid["filled"].any() else np.nan
                rec["led_1r_hold"] = float(np.nanmedian(l_1r["hold"][l_1r["filled"]])) if l_1r["filled"].any() else np.nan

            elif qg == "Q3b" and ne >= MIN_EVENTS:
                # Deterministic cell-matched robustness: k(5) x r5(3) x v_ratio(3)
                r5v = px.at(sd, "r5v")
                vrv = px.at(1, "v_ratio")
                qk = np.nanquantile(k_ev, [0.2, 0.4, 0.6, 0.8])
                qr = np.nanquantile(r5v[E], [1 / 3, 2 / 3])
                qv = np.nanquantile(vrv[E], [1 / 3, 2 / 3])
                bk = np.digitize(k_all, qk)
                br = np.digitize(r5v, qr)
                bv = np.digitize(vrv, qv)
                cell = bk * 9 + br * 3 + bv
                rand_m = (~E) & (~px.at(sd, "q3b_any")) & np.isfinite(y)
                diffs_c, ws_c = [], []
                for c_id in np.unique(cell[E]):
                    me = E & (cell == c_id) & np.isfinite(y)
                    mr = rand_m & (cell == c_id)
                    if me.sum() >= 3 and mr.sum() >= 15:
                        diffs_c.append(float(np.nanmean(y[me]) - np.nanmean(y[mr])))
                        ws_c.append(int(me.sum()))
                rec["q3b_det_match"] = float(np.average(diffs_c, weights=ws_c)) if ws_c else np.nan
                # Distance-only matched version (compare vs 3-way matched)
                diffs_k, ws_k = [], []
                for b_id in range(5):
                    me = E & (bk == b_id) & np.isfinite(y)
                    mr = rand_m & (bk == b_id)
                    if me.sum() >= 3 and mr.sum() >= 15:
                        diffs_k.append(float(np.nanmean(y[me]) - np.nanmean(y[mr])))
                        ws_k.append(int(me.sum()))
                rec["q3b_k_only"] = float(np.average(diffs_k, weights=ws_k)) if ws_k else np.nan
                # Vacuum reversal: post5 < 0 means price bounced back away from magnet after touching!
                rec["q3b_vac_rev"] = float(np.nanmean(rc["post5"][E] < 0)) if np.isfinite(rc["post5"][E]).any() else np.nan

            # SW demeaned, cap, top10 / SPX 2010-2019
            if not is_spx:
                rec["ex_dm"], rec["ex_dm_t"], _ = fit_index_reg(px, y, E, None, k_arr=k_all, d_px=d_px, layers=main_layers_race, sub=sub_base, demean_day=True)
                rec["ex_cap"], rec["ex_cap_t"], _ = fit_index_reg(px, y, E, None, k_arr=k_all, d_px=d_px, layers=main_layers_race, sub=sub_base, day_cap=SW_DAY_CAP)
                u_t, cnt_t = np.unique(px.tt[E], return_counts=True)
                top10 = u_t[np.argsort(-cnt_t)[:10]]
                rec["top10_share"] = float(np.isin(px.tt[E], top10).mean()) if ne else np.nan
                rec["ex_no10"], rec["ex_no10_t"], _ = fit_index_reg(
                    px, y, E, None, k_arr=k_all, d_px=d_px, layers=main_layers_race,
                    sub=(sub_base & ~np.isin(px.tt, top10)) if sub_base is not None else ~np.isin(px.tt, top10))
            else:
                sub10 = (px.year >= 2010) & (sub_base if sub_base is not None else True)
                rec["ex_1019"], rec["ex_1019_t"], rec["n_1019"] = fit_index_reg(
                    px, y, E, None, k_arr=k_all, d_px=d_px, layers=("dist", "geo", "ret1", "trend"), sub=sub10)

            # Period stability
            per_list = []
            if not is_spx:
                for yr in yrs_all:
                    sy = (px.year == yr) & (sub_base if sub_base is not None else True)
                    if (E & sy).sum() >= 10:
                        per_list.append(fit_index_reg(px, y, E, None, k_arr=k_all, d_px=d_px, layers=main_layers_race, sub=sy)[0])
                    else:
                        per_list.append(np.nan)
                halves = [fit_index_reg(px, y, E, None, k_arr=k_all, d_px=d_px, layers=main_layers_race,
                                        sub=(px.year >= a_) & (px.year <= b_) & (sub_base if sub_base is not None else True))[0]
                          for a_, b_ in sw_halves]
            else:
                for a_, b_ in spx_periods:
                    sy = (px.year >= a_) & (px.year <= b_) & (sub_base if sub_base is not None else True)
                    per_list.append(fit_index_reg(px, y, E, None, k_arr=k_all, d_px=d_px, layers=main_layers_race, sub=sy)[0]
                                    if (E & sy).sum() >= MIN_EVENTS else np.nan)
                halves = [fit_index_reg(px, y, E, None, k_arr=k_all, d_px=d_px, layers=main_layers_race,
                                        sub=(px.year >= a_) & (px.year <= b_) & (sub_base if sub_base is not None else True))[0]
                          for a_, b_ in spx_halves]
            rec["per"], rec["halves"] = per_list, halves

        rows_out.append(rec)
    return rows_out, notes


def f_(x, nd=3, pct=False):
    return s09.fmt(x, nd, pct)


def render_market(name: str, rows: list[dict]) -> list[str]:
    L = [f"\n{'#' * 115}\n# 市场：{name}（SPX 仅作方向核验，事件 < {MIN_SPX} 不解读 t；SW 行业 10 项在 12b 合并个股进 20 项 FDR）\n{'#' * 115}"]
    L.append(f"  {'命题':<22s}{'事件':>6s}{'对照':>6s}{'原始Y':>8s}{'零假设':>8s}{'t0':>6s}{'超额':>8s}{'t':>6s}{'专属对照':>9s}{'t':>6s}{'不重叠':>8s}{'乐观/记0':>14s}")
    for r in rows:
        is_q1 = r["qg"] in ("Q1a", "Q1b")
        spx_low = (name == "SPX" and r["n_ev"] < MIN_SPX)
        t_str = f"{f_(r['ex_t'], 1)}*" if spx_low else f_(r["ex_t"], 1)
        t0_str = f"{f_(r['t0'], 1)}*" if spx_low else f_(r["t0"], 1)
        ct_str = f"{f_(r['ctrl_t'], 1)}*" if spx_low else f_(r["ctrl_t"], 1)
        if is_q1:
            L.append(f"  {r['label']:<20s}{r['n_ev']:>6,}{r['n_ctrl']:>6,}{r['mu_e']*100:>7.1f}%{r['mu0']*100:>7.1f}%{t0_str:>6s}"
                     f"{r['ex']*100:>+7.1f}pp{t_str:>6s}{r['ctrl']*100:>+8.1f}pp{ct_str:>6s}{r['ex_no']*100:>+7.1f}pp{'—':>14s}")
        else:
            L.append(f"  {r['label']:<20s}{r['n_ev']:>6,}{r['n_ctrl']:>6,}{f_(r['mu_e']):>8s}{f_(r['mu0']):>8s}{t0_str:>6s}"
                     f"{f_(r['ex']):>8s}{t_str:>6s}{f_(r['ctrl']):>9s}{ct_str:>6s}{f_(r['ex_no']):>8s}"
                     f"{f_(r['ex_opt']):>7s}/{f_(r['ex_tie0']):>6s}")
        # Details per row
        per_s = "".join("+" if (math.isfinite(v) and v > 0) else ("−" if math.isfinite(v) else "·") for v in r["per"])
        nv = sum(1 for v in r["per"] if math.isfinite(v))
        np_ = sum(1 for v in r["per"] if math.isfinite(v) and v > 0)
        h_s = " / ".join(f_(v) for v in r["halves"])
        if is_q1:
            extra = f" | 拆解 绝{r['ex_abs']*100:+.1f}→无日涨{r['ex_no_r1']*100:+.1f}→主{r['ex']*100:+.1f}pp | 零假设(鞅{r['mu0_mart']*100:.1f}% IID{r['mu_iid']*100:.1f}% RW{r['mu_rw_m']*100:.1f}/{r['mu_rw_d']*100:.1f}%)"
            extra += f" | 温和新高对照 {r['mild']*100:+.1f}pp(t{f_(r['mild_t'],1)}) | 3根高潮 {r['ex_3b']*100:+.1f}pp(n{r['n_3b']}) | ATR(t) {r['ex_atrt']*100:+.1f}pp"
            if r["qg"] == "Q1a":
                extra += f"\n      TB辅报：严格版(≤H_t) {r['tb_strict']*100:.1f}% | C10≤C0 {r['tb_c10']*100:.1f}% | 生存5/10/20天 {r['surv5']*100:.0f}/{r['surv10']*100:.0f}/{r['surv20']*100:.0f}%"
            else:
                lb = "/".join(f"{x*100:.0f}%" if math.isfinite(x) else "·" for x in r["cond_lags"])
                extra += (f"\n      Q1b条件两段(固定10日,n={r['cond_n']})：事{r['cond_e']*100:.1f}% 零{r['cond_0']*100:.1f}%(t0={f_(r['cond_t0'],1)}) "
                          f"双胞胎{r['cond_ex']*100:+.1f}pp(t{f_(r['cond_ex_t'],1)}) | 分档(1-5/6-10/11-20d) {lb} | 宽松TL {r['tl_loose']*100:.1f}% | TB∩TL {r['tbtl_joint']*100:.1f}%")
        else:
            extra = (f" | 拆解 绝{f_(r['ex_abs'])}→无日涨{f_(r['ex_no_r1'])}→主{f_(r['ex'])} 纯收{f_(r['ex_cls'])} | ATR优势 {f_(r['atr_adv'])}"
                     f" | 零假设(鞅{f_(r['mu0_mart'])} IID{f_(r['mu_iid'])} RW{f_(r['mu_rw_m'])}/{f_(r['mu_rw_d'])}) 第1块定输赢{r['blk1_share']*100:.0f}%"
                     f"\n      路径：先目标{r['hit_tgt']*100:.1f}% 先反向{r['hit_rev']*100:.1f}% 未碰{r['unres']*100:.1f}% 双碰{r['both_rate']*100:.1f}% | 生存5/10/20d {r['surv5']*100:.0f}/{r['surv10']*100:.0f}/{r['surv20']*100:.0f}% | 触后5日{r['post5']*100:+.2f}%")
            if r["qg"] == "Q2" and "q2_mult_y" in r:
                my = "/".join(f_(x, 2) for x in r["q2_mult_y"])
                mp = "/".join(f"{x*100:+.2f}%" if math.isfinite(x) else "·" for x in r["q2_mult_p5"])
                extra += (f"\n      Q2诊断：倍数(0.6/0.8/1.0/1.2/1.4) Y={my} 触后5d={mp} 1.0x停顿={r['q2_stall']*100:+.2f}% | "
                          f"未确认原版 {f_(r.get('q2_raw_mu', np.nan))}(n{r.get('q2_raw_n', 0)}) | Leg1=Leg2 {f_(r.get('q2_leg_mu', np.nan))}(n{r.get('q2_leg_n', 0)}) | "
                          f"缺口 {f_(r.get('q2_mg_mu', np.nan))}(n{r.get('q2_mg_n', 0)}) | 结构止损胜率{r['q2_struct_win']*100:.1f}%(理论{r['q2_struct_theo']*100:.1f}%) | "
                          f"40d超额 {f_(r['q2_ex40'])}(t{f_(r['q2_ex40_t'],1)}) | 账本 MM净R {f_(r['led_mm_net'])}(胜{r['led_mm_win']*100:.0f}%) vs 1R {f_(r['led_1r_net'])}(胜{r['led_1r_win']*100:.0f}%) 跳过目标{r['led_mm_past']*100:.1f}%")
            elif r["qg"] == "Q3a" and "led_mid_net" in r:
                extra += (f"\n      Q3a诊断：下跌箱体对照 {f_(r.get('q3a_ctrl_dn', np.nan))}(t{f_(r.get('q3a_ctrl_dn_t', np.nan),1)}) | 朝边沿走分层 {f_(r['q3a_tw'])}(t{f_(r['q3a_tw_t'],1)}) | "
                          f"全量q1_ev {f_(r['q3a_all_mu'])}(双碰{r['q3a_all_both']*100:.1f}%) | 账本 中轴净R {f_(r['led_mid_net'])}(胜{r['led_mid_win']*100:.0f}%,持{r['led_mid_hold']:.0f}d) vs 1R {f_(r['led_1r_net'])}(胜{r['led_1r_win']*100:.0f}%,持{r['led_1r_hold']:.0f}d)")
            elif r["qg"] == "Q3b" and "q3b_det_match" in r:
                extra += f"\n      Q3b诊断：确定性3维配对 {f_(r['q3b_det_match'])} vs 仅距离配对 {f_(r['q3b_k_only'])} | 触磁铁后5d反转率 {r['q3b_vac_rev']*100:.1f}%"
        sw_s = (f" | 选行业去均值 {f_(r['ex_dm'])}(t{f_(r['ex_dm_t'],1)}) | 权重上限 {f_(r['ex_cap'])}(t{f_(r['ex_cap_t'],1)}) | "
                f"前10日占{r['top10_share']*100:.0f}% 剔除后 {f_(r['ex_no10'])}(t{f_(r['ex_no10_t'],1)})") if name == "SW" else (
                f" | 2010–19(含形状) {f_(r['ex_1019'])}(t{f_(r['ex_1019_t'],1)},n{r['n_1019']})")
        L.append(f"      稳定 {per_s} ({np_}/{nv}) 两段 {h_s}{sw_s}{extra}")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step0", action="store_true")
    args = ap.parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    raw = s09.load_raw()
    if args.step0:
        L = step0(raw)
        txt = "\n".join(L)
        STEP0_PATH.write_text(txt + "\n", encoding="utf-8")
        print(txt)
        log(f"step0 done ({time.time() - t0:.0f}s) -> {STEP0_PATH}")
        return
    L = ["=" * 115, "Study 12 · 第 1 步：路径形状与目标位（SPX + 31 申万一级行业，开发段）", "=" * 115]
    allrows = []
    for name in ("SPX", "SW"):
        df, ev0 = raw[name]
        log(f"build & analyze {name}")
        px = PathIndex(name, df, ev0)
        rows, _ = analyze_market(px)
        L += render_market(name, rows)
        allrows += rows
    txt = "\n".join(L)
    REPORT_PATH.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    flat = []
    for r in allrows:
        d = {k: v for k, v in r.items() if not isinstance(v, list)}
        d["per"] = ";".join(f"{v:+.4f}" if math.isfinite(v) else "nan" for v in r["per"])
        d["halves"] = ";".join(f"{v:+.4f}" if math.isfinite(v) else "nan" for v in r["halves"])
        flat.append(d)
    pd.DataFrame(flat).to_csv(SUMMARY_PATH, index=False)
    log(f"index done ({time.time() - t0:.0f}s) -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
