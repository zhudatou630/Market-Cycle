#!/usr/bin/env python3
"""
Market Cycle V2 — Study 08b（第二阶段方向 D · 复活赛）：
指数上失效的 4 个 Al Brooks 结构，切到 A 股个股日线后能不能「起死回生」？

  Q1 突破回踩二次上车：BP_H1_20（20 日首次新高后 2~12 天回调抬头确认、收盘站稳前高） vs 普通上涨回调 ORD_PB
  Q2 弹簧（假跌破拉回）：SPRING_20（20 日首次新低后 1~5 天收回前低） vs 真跌破 BDN_TRUE_20（5 天内没收回）
  Q3 向上假突破失败：FAIL_UP_20（20 日首次新高后 1~5 天收盘跌回前高下方） vs 普通高位收跌 ORD_DOWN_HI
  Q4 窄通道 vs 宽通道：上升段里贴着 20 日均线稳推的窄通道日 TC vs 频破均线/深回撤的宽通道日 BC

形态定义原样冻结自 studies/05、06（只修正一处：一字板 H=L 时收盘力度涨停记 1、跌停记 0）；
脚本启动时先在 31 个申万一级行业上校验「本脚本的快速实现」与 05/06 原代码产出的事件完全一致。

裁判与推断（预注册见 decisions.md 第 08 节）：
  - 个股：t+1 开盘买、t+20 收盘卖（简单收益），主裁判 = 相对同日同市值五档；风险 = 固定 -15% 踩雷（控波动），
    波动归一踩雷同向复核。「双胞胎」= 每日截面回归残差，分步剥离：
      L0 只扣同日同市值档 → L1 +铁律（近20/60日涨跌、偏离20日均线、20日波动）
      → L2 +热度/市值（近5日涨跌、20日换手、涨停次数、对数市值） → L3 +申万一级行业固定效应
  - 显著性：交易日历对齐 Driscoll-Kraay（lag=30），同一天同一组最多按 30 只计权重。
  - 指数端：同一套事件定义在申万一级上重算，t 收盘进、t+20 收盘出，绝对收益 + 相对 31 行业等权两套口径。
"""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
PANEL_PATH = OUTPUT_DIR / "08_stock_panel_dev.parquet"
REPORT_PATH = OUTPUT_DIR / "08b_stock_revival_report.txt"
SUMMARY_PATH = OUTPUT_DIR / "08b_stock_revival_summary.csv"


def _load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "studies" / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


s01 = _load_mod("s01", "01_trend_vs_range.py")
s05 = _load_mod("s05", "05_breakout_pullback_and_failed.py")
s06 = _load_mod("s06", "06_ordered_structures_channel_wedge.py")
s08 = _load_mod("s08", "08_stock_clean_panel_and_baselines.py")

EPS = 1e-12
EVAL_START = pd.Timestamp("2016-01-01")
NW_LAG = 30
W_CAP = 30.0
IND_CRASH = -0.08

L1_COLS = ["r20", "r60", "ma20pos", "vol20"]
L2_COLS = L1_COLS + ["r5", "turn20", "n_up20", "lmv"]
IDX_COLS = ["r20", "vol20", "r5"]
LEVELS = ["L0", "L1", "L2", "L3"]
LEVEL_NAMES = {"L0": "裸（只扣同日同市值档）", "L1": "+铁律 趋势/波动", "L2": "+热度/市值", "L3": "+行业"}
TIER_NAMES = {1: "沪深300", 2: "中证500", 3: "中证1000", 4: "其余小微"}

CHECK_KEYS_05 = [
    f"{k}_{tag}" for tag in ["20", "60"]
    for k in ["brk", "bp_touch", "bp_h1", "bp_strict", "bp_h2", "fail_up", "fail_up_strong", "bdn", "bdn_true",
              "spring", "spring_strong"]
] + ["ord_pb", "ord_down_hi", "ord_bounce"]
CHECK_KEYS_06 = ["tc", "mid", "bc", "tc_pb", "bc_pb"]

# Pre-registered primary specs: (key, label, treat, control, Brooks direction of return)
QUESTIONS = [
    ("Q1", "突破回踩二次上车", "bp_h1_20", "ord_pb", +1, "等回踩确认后上车 vs 普通上涨回调抬头"),
    ("Q2", "弹簧（假跌破拉回）", "spring_20", "bdn_true_20", +1, "跌破前低 5 天内收回 vs 5 天内没收回的真跌破"),
    ("Q3", "向上假突破失败", "fail_up_20", "ord_down_hi", -1, "新高后 5 天内跌回前高 vs 普通高位收跌"),
    ("Q4", "窄通道 vs 宽通道", "tc", "bc", +1, "上升段贴均线稳推的窄通道日 vs 颠簸深踩的宽通道日"),
]
DIAG = [
    ("Q1", "brk_20", "第一脚直接追（20日首次新高当天）"),
    ("Q1", "bp_touch_20", "碰前高直接低吸"),
    ("Q1", "bp_strict_20", "严格回踩到前高附近再确认"),
    ("Q1", "bp_h2_20", "两段回踩 High 2"),
    ("Q1", "bp_h1_hold", "回踩低点守住前高（个股加试）"),
    ("Q1", "bp_h1_break", "回踩低点击穿前高后收回"),
    ("Q1", "brk_60", "60日首次新高直接追"),
    ("Q1", "bp_h1_60", "60日突破回踩确认"),
    ("Q2", "spring_strong_20", "强弹簧（收回且强收）"),
    ("Q2", "bdn_20", "20日首次新低当天"),
    ("Q2", "spring_60", "60日弹簧"),
    ("Q2", "spring_strong_60", "60日强弹簧"),
    ("Q2", "bdn_true_60", "60日真跌破"),
    ("Q2", "ord_bounce", "普通超跌反弹日"),
    ("Q3", "fail_up_strong_20", "强假突破（连突破日低点也跌破）"),
    ("Q3", "fail_up_60", "60日假突破失败"),
    ("Q4", "mid", "上升段中间过渡通道"),
    ("Q4", "tc_pb", "窄通道里第一次像样回调"),
    ("Q4", "bc_pb", "宽通道里第一次像样回调"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Fast, frozen re-implementation of the 05/06 event definitions
# ---------------------------------------------------------------------------
def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[: len(x) - k]
    return out


def detect_events(h, l, c, dates, eval_start, up_close=None, dn_close=None):
    """Mirror of s05.build_asset_events (event part) + s06.build_asset_study06 (channel part).
    Returns {key: sorted unique positions}, extras {"bp_h1_20_dpb": {pos: D_pb}}."""
    n = len(c)
    ev: dict[str, list[int]] = {k: [] for k in CHECK_KEYS_05 + CHECK_KEYS_06}
    dpb_h1_20: dict[int, float] = {}
    if n < 30:
        return {k: np.array([], dtype=int) for k in ev}, dpb_h1_20
    eval_mask = pd.to_datetime(dates) >= eval_start
    eval_mask = np.asarray(eval_mask, dtype=bool)
    prev_c = _shift(c, 1)
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    s_c = pd.Series(c)
    atr20 = pd.Series(tr).rolling(20, min_periods=20).mean().to_numpy()
    ma20 = s_c.rolling(20, min_periods=20).mean().to_numpy()
    ma10 = s_c.rolling(10, min_periods=10).mean().to_numpy()
    r20 = pd.Series(np.log(c)).diff(20).to_numpy()
    rng = h - l
    k_cls = np.full(n, 0.5)
    vr = rng > EPS
    k_cls[vr] = np.clip((c[vr] - l[vr]) / rng[vr], 0.0, 1.0)
    if up_close is not None:
        k_cls[(~vr) & up_close] = 1.0
        k_cls[(~vr) & dn_close] = 0.0

    H, Lw, C, TR, ATR, KC, EM = h.tolist(), l.tolist(), c.tolist(), tr.tolist(), atr20.tolist(), k_cls.tolist(), eval_mask.tolist()
    fin = math.isfinite
    brk20 = np.zeros(n, dtype=bool)
    bdn20 = np.zeros(n, dtype=bool)
    for tag, N, cool in [("20", 20, 5), ("60", 60, 10)]:
        prev_hi = s_c.rolling(N, min_periods=N).max().shift(1).to_numpy()
        is_nh = c > prev_hi
        recent_nh = pd.Series(is_nh.astype(float)).shift(1).rolling(cool, min_periods=cool).sum().to_numpy() > 0
        brk = is_nh & (~recent_nh) & np.isfinite(prev_hi)
        prev_lo = s_c.rolling(N, min_periods=N).min().shift(1).to_numpy()
        is_nl = c < prev_lo
        recent_nl = pd.Series(is_nl.astype(float)).shift(1).rolling(cool, min_periods=cool).sum().to_numpy() > 0
        bdn = is_nl & (~recent_nl) & np.isfinite(prev_lo)
        if tag == "20":
            brk20, bdn20 = brk, bdn
        PH, PL = prev_hi.tolist(), prev_lo.tolist()
        e_brk, e_touch, e_h1, e_strict, e_h2 = ev[f"brk_{tag}"], ev[f"bp_touch_{tag}"], ev[f"bp_h1_{tag}"], ev[f"bp_strict_{tag}"], ev[f"bp_h2_{tag}"]
        e_fail, e_fails = ev[f"fail_up_{tag}"], ev[f"fail_up_strong_{tag}"]
        for t0 in np.flatnonzero(brk).tolist():
            p_brk = PH[t0]
            atr0 = ATR[t0 - 1] if t0 >= 1 else float("nan")
            if not fin(atr0) or atr0 <= EPS:
                continue
            if EM[t0]:
                e_brk.append(t0)
            in_pb = False
            leg_cnt = 0
            got_touch = got_h1 = got_h2 = False
            pb_min_l = float("inf")
            peak_c = C[t0]
            for t in range(t0 + 1, min(n - 1, t0 + 20) + 1):
                dt = t - t0
                if (not got_touch) and (1 <= dt <= 10):
                    if (Lw[t] <= p_brk + 0.3 * atr0) and (C[t] >= p_brk - 0.5 * atr0) and (C[t] < C[t - 1] or Lw[t] < Lw[t - 1]):
                        got_touch = True
                        if EM[t]:
                            e_touch.append(t)
                if Lw[t] < pb_min_l:
                    pb_min_l = Lw[t]
                is_down_bar = (Lw[t] < Lw[t - 1]) or (C[t] < C[t - 1])
                is_up_turn = (H[t] > H[t - 1]) and (C[t] > C[t - 1])
                if not in_pb:
                    if is_down_bar and (C[t] < peak_c or Lw[t] < Lw[t - 1]):
                        in_pb = True
                else:
                    if is_up_turn:
                        leg_cnt += 1
                        in_pb = False
                        d_pb = (pb_min_l - p_brk) / atr0
                        if leg_cnt == 1 and (2 <= dt <= 12) and (not got_h1):
                            if C[t] >= p_brk:
                                got_h1 = True
                                if EM[t]:
                                    e_h1.append(t)
                                    if tag == "20" and t not in dpb_h1_20:
                                        dpb_h1_20[t] = d_pb
                                    if -1.0 <= d_pb <= 0.5:
                                        e_strict.append(t)
                        elif leg_cnt == 2 and (4 <= dt <= 20):
                            if (not got_h2) and (C[t] >= p_brk):
                                got_h2 = True
                                if EM[t]:
                                    e_h2.append(t)
                if C[t] > peak_c:
                    peak_c = C[t]
            got_f = got_fs = False
            for t in range(t0 + 1, min(n - 1, t0 + 5) + 1):
                if (not got_f) and (C[t] < p_brk):
                    got_f = True
                    if EM[t]:
                        e_fail.append(t)
                if (not got_fs) and (C[t] < min(p_brk, Lw[t0])):
                    got_fs = True
                    if EM[t]:
                        e_fails.append(t)
        e_bdn, e_true, e_sp, e_sps = ev[f"bdn_{tag}"], ev[f"bdn_true_{tag}"], ev[f"spring_{tag}"], ev[f"spring_strong_{tag}"]
        for t0 in np.flatnonzero(bdn).tolist():
            p_bdn = PL[t0]
            atr0 = ATR[t0 - 1] if t0 >= 1 else float("nan")
            if not fin(atr0) or atr0 <= EPS:
                continue
            if EM[t0]:
                e_bdn.append(t0)
            got_sp = got_sps = False
            for t in range(t0 + 1, min(n - 1, t0 + 5) + 1):
                if (not got_sp) and (C[t] > p_bdn):
                    got_sp = True
                    if EM[t]:
                        e_sp.append(t)
                if (not got_sps) and (C[t] > p_bdn) and ((C[t] > H[t0]) or (KC[t] >= 0.65)):
                    got_sps = True
                    if EM[t]:
                        e_sps.append(t)
            if (not got_sp) and (t0 + 5 < n) and EM[t0 + 5]:
                e_true.append(t0 + 5)

    # Control pools (05)
    idx = np.arange(n)
    c1, c2, l1_, l2_, h1 = _shift(c, 1), _shift(c, 2), _shift(l, 1), _shift(l, 2), _shift(h, 1)
    ma20_1 = _shift(ma20, 1)
    rb15 = pd.Series(brk20.astype(float)).shift(1).rolling(15, min_periods=1).sum().fillna(0).to_numpy() > 0
    rd15 = pd.Series(bdn20.astype(float)).shift(1).rolling(15, min_periods=1).sum().fillna(0).to_numpy() > 0
    valid = (idx >= 25) & eval_mask & np.isfinite(ma20) & np.isfinite(r20)
    prev_down = (l1_ < l2_) | (c1 < c2)
    turn_up = (h > h1) & (c > c1)
    ev["ord_pb"] = np.flatnonzero(valid & (c > ma20) & (r20 > 0) & prev_down & turn_up & ~rb15 & ~brk20).tolist()
    ev["ord_down_hi"] = np.flatnonzero(valid & (c1 > ma20_1) & (c < c1) & ~rb15).tolist()
    ev["ord_bounce"] = np.flatnonzero(valid & (r20 < 0) & (c1 < ma20_1) & (c > c1) & (h > h1) & ~rd15 & ~bdn20).tolist()

    # Channels (06)
    ma20_lag10 = _shift(ma20, 10)
    hold = pd.Series((c >= ma20).astype(float)).rolling(20, min_periods=20).mean().to_numpy()
    acc = np.full(n, -np.inf)
    s_h = pd.Series(h)
    for k in range(20):
        rm = s_h.rolling(20 - k, min_periods=20 - k).max().to_numpy()
        acc = np.fmax(acc, _shift(rm - l, k))
    ok_pb = (idx >= 20) & np.isfinite(atr20) & (atr20 > EPS) & np.isfinite(acc)
    max_pb = np.where(ok_pb, acc / np.where(atr20 > EPS, atr20, np.nan), np.nan)
    valid_u = (idx >= 30) & eval_mask & np.isfinite(r20) & np.isfinite(ma20) & np.isfinite(ma20_lag10) & np.isfinite(max_pb)
    is_up = valid_u & (r20 > 0) & (ma20 > ma20_lag10)
    is_tc = (hold >= 0.90) & (max_pb <= 2.5)
    is_bc = (hold <= 0.75) | (max_pb >= 3.5)
    ev["tc"] = np.flatnonzero(is_up & is_tc).tolist()
    ev["bc"] = np.flatnonzero(is_up & ~is_tc & is_bc).tolist()
    ev["mid"] = np.flatnonzero(is_up & ~is_tc & ~is_bc).tolist()
    hold1, pb1 = _shift(hold, 1), _shift(max_pb, 1)
    prev_up = (_shift(r20, 1) > 0) & (ma20_1 > _shift(ma20_lag10, 1))
    dip = (c < c1) & ((l <= ma10) | (c1 < c2))
    cand = is_up & prev_up & dip & np.isfinite(hold1) & np.isfinite(pb1)
    for key, cond in [("tc_pb", cand & (hold1 >= 0.90) & (pb1 <= 2.5)),
                      ("bc_pb", cand & ((hold1 <= 0.75) | (pb1 >= 3.5)))]:
        last = -100
        for t in np.flatnonzero(cond).tolist():
            if t - last >= 5:
                ev[key].append(t)
                last = t
    return {k: np.unique(np.asarray(v, dtype=int)) for k, v in ev.items()}, dpb_h1_20


def verify_against_originals(sw: pd.DataFrame) -> list[str]:
    lines = []
    bad = 0
    tot = 0
    for asset, g in sw.groupby("asset", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        o5 = s05.build_asset_events(g, "A股申万L1", EVAL_START)
        o6 = s06.build_asset_study06(g, "A股申万L1", EVAL_START)
        mine, _ = detect_events(g["high"].to_numpy(float), g["low"].to_numpy(float), g["close"].to_numpy(float),
                                g["date"].to_numpy(), EVAL_START)
        d = g["date"].to_numpy()
        ref = {k: set(pd.to_datetime(o5[k]["date"])) for k in CHECK_KEYS_05}
        up = o6["uptrend"]
        for k, ct in [("tc", "TC"), ("mid", "MID"), ("bc", "BC")]:
            ref[k] = set(pd.to_datetime(up.loc[up["ch_type"] == ct, "date"]))
        ref["tc_pb"] = set(pd.to_datetime(o6["tc_pb"]["date"]))
        ref["bc_pb"] = set(pd.to_datetime(o6["bc_pb"]["date"]))
        for k in CHECK_KEYS_05 + CHECK_KEYS_06:
            got = set(pd.to_datetime(d[mine[k]]))
            tot += len(ref[k])
            if got != ref[k]:
                bad += 1
                lines.append(f"  不一致: {asset} {k} 原={len(ref[k])} 新={len(got)}")
    lines.insert(0, f"  校验 31 行业 × {len(CHECK_KEYS_05) + len(CHECK_KEYS_06)} 类事件（原代码事件总数 {tot:,}）："
                    + ("全部一致 ✔" if bad == 0 else f"{bad} 处不一致 ✘"))
    assert bad == 0, "re-implementation deviates from frozen 05/06 definitions"
    return lines


# ---------------------------------------------------------------------------
# 2. Statistics: per-day cross-sectional residuals & weighted Driscoll-Kraay
# ---------------------------------------------------------------------------
def day_slices(mi: np.ndarray):
    starts = np.flatnonzero(np.r_[True, mi[1:] != mi[:-1]])
    ends = np.r_[starts[1:], len(mi)]
    return zip(starts.tolist(), ends.tolist())


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std(0)
    return (x - x.mean(0)) / np.where(sd > EPS, sd, 1.0)


def residualize_stocks(base: pd.DataFrame, ycols: list[str]) -> dict[str, np.ndarray]:
    out = {f"{lv}_{y}": np.full(len(base), np.nan) for lv in LEVELS for y in ycols}
    out["L3_ret_40"] = np.full(len(base), np.nan)
    mi = base["mi"].to_numpy()
    Y = base[ycols].to_numpy(float)
    Y40 = base["ret_40"].to_numpy(float)
    X2 = base[L2_COLS].to_numpy(float)
    szq = base["szq"].to_numpy().astype(int) - 1
    l1c = pd.factorize(base["l1"])[0]
    for s, e in day_slices(mi):
        m = e - s
        if m < 300:
            continue
        xz = _z(X2[s:e])
        dsz = np.zeros((m, 5))
        dsz[np.arange(m), szq[s:e]] = 1.0
        codes, inv = np.unique(l1c[s:e], return_inverse=True)
        dl1 = np.zeros((m, len(codes)))
        dl1[np.arange(m), inv] = 1.0
        mats = {
            "L0": dsz,
            "L1": np.hstack([dsz, xz[:, : len(L1_COLS)]]),
            "L2": np.hstack([dsz, xz]),
            "L3": np.hstack([dsz, xz, dl1[:, 1:]]),
        }
        ys = Y[s:e]
        for lv, X in mats.items():
            b, *_ = np.linalg.lstsq(X, ys, rcond=None)
            r = ys - X @ b
            for j, y in enumerate(ycols):
                out[f"{lv}_{y}"][s:e] = r[:, j]
        y40 = Y40[s:e]
        f = np.isfinite(y40)
        if f.sum() >= 300:
            X = mats["L3"][f]
            b, *_ = np.linalg.lstsq(X, y40[f], rcond=None)
            tmp = np.full(m, np.nan)
            tmp[f] = y40[f] - X @ b
            out["L3_ret_40"][s:e] = tmp
    return out


def residualize_index(df: pd.DataFrame, ycols: list[str]) -> dict[str, np.ndarray]:
    out = {f"R_{y}": np.full(len(df), np.nan) for y in ycols}
    mi = df["mi"].to_numpy()
    X = df[IDX_COLS].to_numpy(float)
    Y = df[ycols].to_numpy(float)
    for s, e in day_slices(mi):
        if e - s < 15:
            continue
        Xd = np.column_stack([np.ones(e - s), _z(X[s:e])])
        b, *_ = np.linalg.lstsq(Xd, Y[s:e], rcond=None)
        r = Y[s:e] - Xd @ b
        for j, y in enumerate(ycols):
            out[f"R_{y}"][s:e] = r[:, j]
    return out


def cap_weights(mi: np.ndarray, grp: np.ndarray) -> np.ndarray:
    key = mi.astype(np.int64) * 2 + grp.astype(np.int64)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return np.minimum(1.0, W_CAP / cnt[inv])


def wls_dk(y: np.ndarray, X: np.ndarray, w: np.ndarray, mi: np.ndarray, lag: int = NW_LAG) -> tuple[np.ndarray, np.ndarray]:
    f = np.isfinite(y) & np.isfinite(X).all(1)
    y, X, w, mi = y[f], X[f], w[f], mi[f]
    k = X.shape[1]
    if len(y) < 30:
        return np.full(k, np.nan), np.full(k, np.nan)
    Xw = X * w[:, None]
    inv = np.linalg.pinv(X.T @ Xw)
    beta = inv @ (Xw.T @ y)
    e = y - X @ beta
    sc = Xw * e[:, None]
    T = int(mi.max()) + 1
    hh = np.zeros((T, k))
    np.add.at(hh, mi, sc)
    om = hh.T @ hh
    for ell in range(1, min(lag, T - 1) + 1):
        g = hh[ell:].T @ hh[:-ell]
        om += (1 - ell / (lag + 1.0)) * (g + g.T)
    se = np.sqrt(np.maximum(np.diag(inv @ om @ inv), 0))
    return beta, np.where(se > 0, beta / se, np.nan)


def vs_twins(res: np.ndarray, S: np.ndarray, mi: np.ndarray) -> tuple[float, float, int]:
    s = S[np.isfinite(res[S])]
    if len(s) < 30:
        return np.nan, np.nan, len(s)
    w = cap_weights(mi[s], np.zeros(len(s)))
    b, t = wls_dk(res[s], np.ones((len(s), 1)), w, mi[s])
    return float(b[0]), float(t[0]), len(s)


def vs_ctrl(y: np.ndarray, S: np.ndarray, C: np.ndarray, mi: np.ndarray, Xc: np.ndarray | None = None) -> tuple[float, float]:
    C = np.setdiff1d(C, S, assume_unique=False)
    rows = np.r_[S, C]
    d = np.r_[np.ones(len(S)), np.zeros(len(C))]
    f = np.isfinite(y[rows])
    rows, d = rows[f], d[f]
    if d.sum() < 30 or (1 - d).sum() < 30:
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(rows)), d])
    if Xc is not None:
        xc = Xc[rows]
        g = np.isfinite(xc).all(1)
        rows, d, X, xc = rows[g], d[g], X[g], xc[g]
        X = np.column_stack([X, _z(xc)])
    w = cap_weights(mi[rows], d)
    b, t = wls_dk(y[rows], X, w, mi[rows])
    return float(b[1]), float(t[1])


def p_two(t: float) -> float:
    return float(math.erfc(abs(t) / math.sqrt(2))) if np.isfinite(t) else np.nan


def bh_fdr(p: list[float]) -> list[float]:
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in reversed(list(enumerate(o, start=1))):
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q.tolist()


# ---------------------------------------------------------------------------
# 3. Reusable scoring contexts (shared with 08c)
# ---------------------------------------------------------------------------
class IndexSide:
    """31 SW L1 industries, same event definitions, entry at close t, exit at close t+20."""

    def __init__(self, verify: bool = True) -> None:
        sw, self.sid = s01.load_sw_l1_dev()
        assert sw["date"].max() <= pd.Timestamp("2024-10-31")
        self.ver = verify_against_originals(sw) if verify else []
        sw = sw.sort_values(["asset", "date"]).reset_index(drop=True)
        ev: dict[str, list[np.ndarray]] = {}
        feats = []
        dpb_all: dict[int, float] = {}
        self.slices: list[tuple[str, int, int]] = []
        offset = 0
        for asset, g in sw.groupby("asset", sort=True):
            hh, ll, cc = g["high"].to_numpy(float), g["low"].to_numpy(float), g["close"].to_numpy(float)
            e, dpb_i = detect_events(hh, ll, cc, g["date"].to_numpy(), EVAL_START)
            for k, v in e.items():
                ev.setdefault(k, []).append(v + offset)
            dpb_all.update({p + offset: v for p, v in dpb_i.items()})
            lc = pd.Series(np.log(cc))
            r1 = lc.diff()
            sc = pd.Series(cc)
            lo_f = pd.Series(ll[::-1]).rolling(20, min_periods=20).min().to_numpy()[::-1]
            mn = np.full(len(cc), np.nan)
            mn[:-20] = lo_f[1: len(cc) - 19]
            feats.append(pd.DataFrame({
                "r20": lc.diff(20).to_numpy(), "r5": lc.diff(5).to_numpy(),
                "vol20": r1.rolling(20, min_periods=20).std(ddof=1).to_numpy(),
                "ret1": r1.to_numpy(),
                "ret_20": (sc.shift(-20) / sc - 1).to_numpy(),
                "crash_20": np.where(np.isfinite(mn), (mn / cc - 1 <= IND_CRASH).astype(float), np.nan),
            }))
            self.slices.append((asset, offset, offset + len(g)))
            offset += len(g)
        sw = pd.concat([sw, pd.concat(feats, ignore_index=True)], axis=1)
        ev = {k: np.concatenate(v) for k, v in ev.items()}
        h1 = ev["bp_h1_20"]
        dv = pd.Series(dpb_all).reindex(h1).to_numpy()
        ev["bp_h1_hold"] = h1[dv >= 0]
        ev["bp_h1_break"] = h1[dv < 0]
        dates = np.sort(sw["date"].unique())
        sw["mi"] = np.searchsorted(dates, sw["date"].to_numpy())
        sw["rel_20"] = sw["ret_20"] - sw.groupby("mi")["ret_20"].transform("mean")
        sw["rel_crash"] = sw["crash_20"] - sw.groupby("mi")["crash_20"].transform("mean")
        cols = ["mi"] + IDX_COLS + ["rel_20", "rel_crash"]
        srt = sw.iloc[np.argsort(sw["mi"].to_numpy(), kind="stable")][cols].dropna()
        rr = residualize_index(srt, ["rel_20", "rel_crash"])
        for k, v in rr.items():
            sw[k] = np.nan
            sw.loc[srt.index, k] = v
        self.sw, self.ev = sw, ev
        self.mi = sw["mi"].to_numpy()
        self.Xc = sw[IDX_COLS].to_numpy(float)

    def score(self, S: np.ndarray, C: np.ndarray) -> dict:
        ret = self.sw["ret_20"].to_numpy()
        S = S[np.isfinite(ret[S])]
        C = C[np.isfinite(ret[C])]
        out = {"n": len(S)}
        out["abs_d"], out["abs_t"] = vs_ctrl(ret * 100, S, C, self.mi, self.Xc)
        out["rel_d"], out["rel_t"] = vs_ctrl(self.sw["R_rel_20"].to_numpy() * 100, S, C, self.mi)
        out["tw_d"], out["tw_t"], _ = vs_twins(self.sw["R_rel_20"].to_numpy() * 100, S, self.mi)
        out["cr_d"], out["cr_t"] = vs_ctrl(self.sw["R_rel_crash"].to_numpy() * 100, S, C, self.mi)
        out["S_ret"] = float(np.nanmean(ret[S]) * 100) if len(S) else np.nan
        out["C_ret"] = float(np.nanmean(ret[C]) * 100) if len(C) else np.nan
        return out


class StockSide:
    """Cleaned all-A panel + 08b events + per-day cross-sectional residuals (L0..L3)."""

    COLS = ["ts_code", "trade_date", "code_id", "mi", "tier", "szq", "l1", "eligible", "entry_ok", "up_close",
            "dn_close", "a_high", "a_low", "a_close", "ret1", "ret_20", "ret_40", "mdd_20", "crash_20", "tail_20",
            "big_20", "ex_sz_20"] + L2_COLS

    def __init__(self, panel_path: Path | None = None, max_date: str = "20241031") -> None:
        t0 = time.time()
        pn = pd.read_parquet(panel_path or PANEL_PATH, columns=self.COLS)
        assert pn["trade_date"].max() <= max_date
        pn = pn.sort_values(["code_id", "trade_date"]).reset_index(drop=True)
        self.pn = pn
        self.dates = pd.to_datetime(pn["trade_date"], format="%Y%m%d").to_numpy()
        cid = pn["code_id"].to_numpy()
        st = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
        self.slices = list(zip(st.tolist(), np.r_[st[1:], len(pn)].tolist()))
        self.AH, self.AL, self.AC = (pn[c].to_numpy(float) for c in ["a_high", "a_low", "a_close"])
        self.UPC, self.DNC = pn["up_close"].to_numpy(bool), pn["dn_close"].to_numpy(bool)
        log("detect 05/06 events on stocks")
        ev: dict[str, list[np.ndarray]] = {}
        dpb_rows, dpb_vals = [], []
        for s, e in self.slices:
            ev_, dpb = detect_events(self.AH[s:e], self.AL[s:e], self.AC[s:e], self.dates[s:e], EVAL_START,
                                     self.UPC[s:e], self.DNC[s:e])
            for k, v in ev_.items():
                ev.setdefault(k, []).append(v + s)
            if dpb:
                dpb_rows.extend([p + s for p in dpb])
                dpb_vals.extend(dpb.values())
        ev = {k: np.concatenate(v) for k, v in ev.items()}
        h1 = ev["bp_h1_20"]
        dv = pd.Series(dpb_vals, index=dpb_rows).reindex(h1).to_numpy()
        ev["bp_h1_hold"] = h1[dv >= 0]
        ev["bp_h1_break"] = h1[dv < 0]
        self.ev = ev
        log(f"events done ({time.time() - t0:.0f}s); residualize")
        self.elig = pn["eligible"].to_numpy()
        self.entry_ok = pn["entry_ok"].to_numpy()
        bmask = self.elig & np.isfinite(pn[["ret_20", "mdd_20", "crash_20", "tail_20"] + L2_COLS].to_numpy(float)).all(1)
        base = pn.loc[bmask, ["mi", "szq", "l1", "tier", "trade_date", "ret_20", "ret_40", "crash_20", "tail_20",
                              "big_20", "ex_sz_20"] + L2_COLS].copy()
        base["prow"] = np.flatnonzero(bmask)
        base = base.sort_values(["mi", "prow"], kind="stable").reset_index(drop=True)
        for k, v in residualize_stocks(base, ["ret_20", "crash_20", "tail_20"]).items():
            base[k] = v
        self.base = base
        self.pos_of = np.full(len(pn), -1, dtype=np.int64)
        self.pos_of[base["prow"].to_numpy()] = np.arange(len(base))
        self.bmi = base["mi"].to_numpy()
        self.year = base["trade_date"].str[:4].to_numpy()
        self.tier = base["tier"].to_numpy()
        self.l1v = base["l1"].to_numpy()
        self.Xl1 = base[L1_COLS].to_numpy(float)
        k1 = pd.Series(self.bmi.astype(np.int64) * 1000 + pd.factorize(self.l1v)[0])
        self.k1 = k1.to_numpy()
        self.k1_tot = k1.map(k1.value_counts()).to_numpy()
        log(f"residuals done ({time.time() - t0:.0f}s)")

    def B(self, rows: np.ndarray) -> np.ndarray:
        p = self.pos_of[rows]
        return np.sort(p[p >= 0])

    def R(self, col: str, scale: float = 100.0) -> np.ndarray:
        return self.base[col].to_numpy(float) * scale

    def describe(self, rows: np.ndarray) -> dict:
        e = rows[self.elig[rows]]
        b = self.B(rows)
        d = {
            "n_el": len(e), "n": len(b), "days": len(np.unique(self.bmi[b])) if len(b) else 0,
            "unbuy": float((~self.entry_ok[e]).mean() * 100) if len(e) else np.nan,
            "upc": float(self.UPC[e].mean() * 100) if len(e) else np.nan,
        }
        for c in ["ret_20", "ex_sz_20", "crash_20", "tail_20", "big_20"]:
            d[c] = float(np.nanmean(self.base[c].to_numpy()[b]) * 100) if len(b) else np.nan
        return d

    def indep_split(self, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ks = self.k1[b]
        sig = pd.Series(ks).map(pd.Series(ks).value_counts()).to_numpy()
        share = sig / self.k1_tot[b]
        return b[share < 0.15], b[share >= 0.15]


def score_question(st: StockSide, ix: IndexSide | None, q: str, title: str, S_rows: np.ndarray, C_rows: np.ndarray,
                   S_idx: np.ndarray | None, C_idx: np.ndarray | None, bdir: int, desc: str, tk: str, ck: str
                   ) -> tuple[list[str], dict]:
    L: list[str] = []
    L.append("\n" + "=" * 118)
    L.append(f"【{q} · {title}】{desc}")
    L.append("=" * 118)
    R, bmi, year, tier = st.R, st.bmi, st.year, st.tier
    S, C = st.B(S_rows), st.B(C_rows)
    dS, dC = st.describe(S_rows), st.describe(C_rows)
    L.append("  [底数] 入池信号数 / 可评估数 / 覆盖交易日 | 次日买不进 | 信号日收盘涨停 | 20天绝对收益 | 相对同市值档 | 踩雷(-15%) | 波动归一踩雷 | 大肉(+15%)")
    for lab, d in [(f"信号 {tk}", dS), (f"对照 {ck}", dC)]:
        L.append(
            f"    {lab:<22} {d['n_el']:>8,} / {d['n']:>8,} / {d['days']:>5} | {d['unbuy']:5.1f}% | {d['upc']:5.1f}% | "
            f"{d['ret_20']:+6.2f}% | {d['ex_sz_20']:+6.2f}% | {d['crash_20']:5.1f}% | {d['tail_20']:5.1f}% | {d['big_20']:5.1f}%"
        )
    L.append("  [分步剥离] 20天相对收益差(%)，括号 t；踩雷率差(百分点)")
    L.append(f"    {'口径':<20}{'信号 vs 双胞胎':>22}{'信号 vs 专属对照':>22}{'踩雷 vs 双胞胎':>20}{'踩雷 vs 对照':>20}")
    rec = {}
    for lv in LEVELS:
        tw = vs_twins(R(f"{lv}_ret_20"), S, bmi)
        cc = vs_ctrl(R(f"{lv}_ret_20"), S, C, bmi)
        tcr = vs_twins(R(f"{lv}_crash_20"), S, bmi)
        ccr = vs_ctrl(R(f"{lv}_crash_20"), S, C, bmi)
        rec[lv] = (tw, cc, tcr, ccr)
        L.append(
            f"    {LEVEL_NAMES[lv]:<20}{tw[0]:+9.2f} ({tw[1]:+5.1f}){'':>5}{cc[0]:+9.2f} ({cc[1]:+5.1f}){'':>5}"
            f"{tcr[0]:+8.1f} ({tcr[1]:+5.1f}){'':>3}{ccr[0]:+8.1f} ({ccr[1]:+5.1f})"
        )
    tw3, cc3, tcr3, ccr3 = rec["L3"]
    ttail = vs_twins(R("L3_tail_20"), S, bmi)
    ctail = vs_ctrl(R("L3_tail_20"), S, C, bmi)
    tw40 = vs_twins(R("L3_ret_40"), S, bmi)
    cc40 = vs_ctrl(R("L3_ret_40"), S, C, bmi)
    abs_c = vs_ctrl(st.base["ret_20"].to_numpy() * 100, S, C, bmi, st.Xl1)
    L.append(
        f"    复核：波动归一踩雷 vs 双胞胎 {ttail[0]:+.1f}pp (t={ttail[1]:+.1f}) / vs 对照 {ctail[0]:+.1f}pp (t={ctail[1]:+.1f}) | "
        f"40天(L3) vs 双胞胎 {tw40[0]:+.2f}% (t={tw40[1]:+.1f}) / vs 对照 {cc40[0]:+.2f}% (t={cc40[1]:+.1f}) | "
        f"绝对口径 vs 对照(控铁律) {abs_c[0]:+.2f}% (t={abs_c[1]:+.1f})"
    )
    dfv = pd.DataFrame({"mi": bmi[S], "r": R("L3_ret_20")[S]}).groupby("mi")["r"].agg(["mean", "size"])
    dv_ = dfv.loc[dfv["size"] >= 5, "mean"].to_numpy()
    m_v, t_v = s08.nw_mean_t(dv_, lag=NW_LAG)
    L.append(f"    辅轨：每天一票（当天 ≥5 只，共 {len(dv_)} 天）信号 vs 双胞胎(L3) {m_v:+.2f}% (t={t_v:+.1f})")

    L.append("  [逐年 L3] 年份: 收益 vs 双胞胎 / vs 对照(%) ; 踩雷 vs 双胞胎 / vs 对照(pp)")
    yrs_tw, yrs_cr, line = [], [], []
    for y in sorted(set(year[S])):
        sy, cy = S[year[S] == y], C[year[C] == y]
        a = vs_twins(R("L3_ret_20"), sy, bmi)[0]
        b_ = vs_ctrl(R("L3_ret_20"), sy, cy, bmi)[0]
        crt = vs_twins(R("L3_crash_20"), sy, bmi)[0]
        cr = vs_ctrl(R("L3_crash_20"), sy, cy, bmi)[0]
        yrs_tw.append(a)
        yrs_cr.append(crt)
        line.append(f"{y}:{a:+.2f}/{b_:+.2f};{crt:+.1f}/{cr:+.1f}")
    L.append("    " + "  ".join(line))
    halves = []
    for lo_, hi_ in [("2016", "2020"), ("2021", "2024")]:
        sh = S[(year[S] >= lo_) & (year[S] <= hi_)]
        ch = C[(year[C] >= lo_) & (year[C] <= hi_)]
        halves.append((vs_twins(R("L3_ret_20"), sh, bmi), vs_ctrl(R("L3_ret_20"), sh, ch, bmi)))
    L.append(
        f"    前段 2016-2020: vs 双胞胎 {halves[0][0][0]:+.2f} (t={halves[0][0][1]:+.1f}) / vs 对照 {halves[0][1][0]:+.2f} (t={halves[0][1][1]:+.1f}) | "
        f"后段 2021-2024.10: vs 双胞胎 {halves[1][0][0]:+.2f} (t={halves[1][0][1]:+.1f}) / vs 对照 {halves[1][1][0]:+.2f} (t={halves[1][1][1]:+.1f})"
    )
    L.append("  [分档 L3] 指数成分档: 信号 vs 双胞胎 / vs 对照 收益差(%)（t）；踩雷 vs 对照(pp)")
    tiers_tw, tiers_cr, line = {}, {}, []
    for tv in [1, 2, 3, 4]:
        st_, ct_ = S[tier[S] == tv], C[tier[C] == tv]
        a = vs_twins(R("L3_ret_20"), st_, bmi)
        b_ = vs_ctrl(R("L3_ret_20"), st_, ct_, bmi)
        cr = vs_ctrl(R("L3_crash_20"), st_, ct_, bmi)
        tiers_tw[tv] = (a, b_)
        tiers_cr[tv] = vs_twins(R("L3_crash_20"), st_, bmi)[0]
        line.append(f"{TIER_NAMES[tv]}(n={a[2]:,}): {a[0]:+.2f}({a[1]:+.1f}) / {b_[0]:+.2f}({b_[1]:+.1f}) / {cr[0]:+.1f}")
    L.append("    " + " | ".join(line))
    big, bigc = S[np.isin(tier[S], [1, 2])], C[np.isin(tier[C], [1, 2])]
    big_tw = vs_twins(R("L3_ret_20"), big, bmi)
    big_cc = vs_ctrl(R("L3_ret_20"), big, bigc, bmi)
    L.append(f"    大中盘合并(沪深300+中证500, 已含行业固定效应): vs 双胞胎 {big_tw[0]:+.2f} (t={big_tw[1]:+.1f}) / vs 对照 {big_cc[0]:+.2f} (t={big_cc[1]:+.1f})")
    ind_, syn_ = st.indep_split(S)
    a_i = vs_twins(R("L3_ret_20"), ind_, bmi)
    a_s = vs_twins(R("L3_ret_20"), syn_, bmi)
    L.append(
        f"  [独立 vs 随大流] 当天同行业触发同信号 <15% 的独立行情 n={a_i[2]:,}: {a_i[0]:+.2f}% (t={a_i[1]:+.1f}) | "
        f"随行业一起 n={a_s[2]:,}: {a_s[0]:+.2f}% (t={a_s[1]:+.1f})   （L3 vs 双胞胎）"
    )
    if ix is not None and S_idx is not None:
        isc = ix.score(S_idx, C_idx)
        L.append(
            f"  [指数端·申万一级同定义] n={isc['n']:,} | 20天绝对收益 信号 {isc['S_ret']:+.2f}% vs 对照 {isc['C_ret']:+.2f}% | "
            f"绝对口径差(控 r20/vol20/r5) {isc['abs_d']:+.2f}% (t={isc['abs_t']:+.1f}) | "
            f"相对31行业差 {isc['rel_d']:+.2f}% (t={isc['rel_t']:+.1f}) | 相对31行业 vs 双胞胎 {isc['tw_d']:+.2f}% (t={isc['tw_t']:+.1f}) | "
            f"踩雷(-8%)差 {isc['cr_d']:+.1f}pp (t={isc['cr_t']:+.1f})"
        )
    else:
        isc = {k: np.nan for k in ["rel_d", "rel_t", "abs_d", "abs_t"]}

    sgn = np.sign(tw3[0])
    row = {
        "q": q, "title": title, "treat": tk, "ctrl": ck, "n": len(S),
        "L0_tw": rec["L0"][0][0], "L3_tw": tw3[0], "L3_tw_t": tw3[1], "L3_cc": cc3[0], "L3_cc_t": cc3[1],
        "L3_cr_tw": tcr3[0], "L3_cr_tw_t": tcr3[1], "L3_cr_cc": ccr3[0], "L3_cr_cc_t": ccr3[1],
        "tail_tw": ttail[0],
        "years_same": int(np.sum(np.sign(yrs_tw) == sgn)),
        "halves_ok": bool(np.sign(halves[0][0][0]) == sgn and np.sign(halves[1][0][0]) == sgn),
        "tiers_ok": int(sum(np.sign(tiers_tw[tv][0][0]) == sgn for tv in [1, 2, 3])),
        "big_tw": big_tw[0], "big_tw_t": big_tw[1], "idx_rel": isc["rel_d"], "idx_rel_t": isc["rel_t"],
        "idx_abs": isc["abs_d"], "idx_abs_t": isc["abs_t"],
        "p_ret": max(p_two(tw3[1]), p_two(cc3[1])), "p_cr": max(p_two(tcr3[1]), p_two(ccr3[1])),
        "sig_ret": bool((np.sign(tw3[0]) == np.sign(cc3[0])) and min(abs(tw3[1]), abs(cc3[1])) >= 2),
        "sig_cr": bool((np.sign(tcr3[0]) == np.sign(ccr3[0])) and min(abs(tcr3[1]), abs(ccr3[1])) >= 2),
        "econ": bool(abs(tw3[0]) >= 0.8), "cr_tail_same": bool(np.sign(tcr3[0]) == np.sign(ttail[0])), "bdir": bdir,
        "cr_yrs": int(np.sum(np.sign(yrs_cr) == np.sign(tcr3[0]))),
        "cr_tiers": int(sum(np.sign(tiers_cr[tv]) == np.sign(tcr3[0]) for tv in [1, 2, 3])),
    }
    return L, row


def verdict_lines(rows: list[dict], qvals: list[float]) -> list[str]:
    def label(sig, sign, bdir, fails, econ_ok, econ_txt, risk=False):
        if not sig:
            return "无显著差异（双胞胎与对照未同时 |t|≥2 且同向）"
        if risk:
            d = "信号组更容易踩雷" if sign > 0 else "信号组更少踩雷"
        else:
            d = "Brooks 方向" if sign == bdir else "与 Brooks 相反"
        if fails:
            return f"{d}显著，但不过关：" + "、".join(fails) + ("" if econ_ok else f"；{econ_txt}")
        return f"过关（{d}）" + ("" if econ_ok else f"，但{econ_txt}，交易价值有限")

    L = []
    for i, r in enumerate(rows):
        r["q_ret"], r["q_cr"] = qvals[2 * i], qvals[2 * i + 1]
        f_ret = [m for m, bad in [("FDR 未过", r["q_ret"] > 0.05), (f"同向年仅 {r['years_same']}/9", r["years_same"] < 6),
                                  ("前后两段不同向", not r["halves_ok"]), (f"三档仅 {r['tiers_ok']}/3 同向（主要靠小盘）", r["tiers_ok"] < 2)] if bad]
        f_cr = [m for m, bad in [("FDR 未过", r["q_cr"] > 0.05), ("波动归一复核反向", not r["cr_tail_same"]),
                                 (f"同向年仅 {r['cr_yrs']}/9", r["cr_yrs"] < 6), (f"三档仅 {r['cr_tiers']}/3 同向", r["cr_tiers"] < 2)] if bad]
        r["ret_pass"] = r["sig_ret"] and not f_ret
        r["cr_pass"] = r["sig_cr"] and not f_cr
        v_ret = label(r["sig_ret"], np.sign(r["L3_tw"]), r["bdir"], f_ret, r["econ"], "幅度 <0.8%")
        v_cr = label(r["sig_cr"], np.sign(r["L3_cr_tw"]), r["bdir"], f_cr, abs(r["L3_cr_tw"]) >= 2, "相对双胞胎 <2pp", risk=True)
        L.append(
            f"  {r['q']} {r['title']:<12} 收益: 双胞胎 {r['L3_tw']:+.2f}%(t={r['L3_tw_t']:+.1f}) 对照 {r['L3_cc']:+.2f}%(t={r['L3_cc_t']:+.1f}) "
            f"q={r['q_ret']:.3f} 同向年 {r['years_same']}/9 两段{'同向' if r['halves_ok'] else '不同向'} 三档同向 {r['tiers_ok']}/3"
        )
        L.append(f"     {'':<12} → {v_ret}")
        L.append(
            f"     {'':<12} 踩雷: 双胞胎 {r['L3_cr_tw']:+.1f}pp(t={r['L3_cr_tw_t']:+.1f}) 对照 {r['L3_cr_cc']:+.1f}pp(t={r['L3_cr_cc_t']:+.1f}) "
            f"q={r['q_cr']:.3f} 归一复核{'同向' if r['cr_tail_same'] else '反向'} 同向年 {r['cr_yrs']}/9 三档同向 {r['cr_tiers']}/3"
        )
        L.append(f"     {'':<12} → {v_cr}")
        L.append(f"     {'':<12} 指数端（申万一级）相对31行业 {r['idx_rel']:+.2f}%(t={r['idx_rel_t']:+.1f}) | 绝对口径 {r['idx_abs']:+.2f}%(t={r['idx_abs_t']:+.1f})")
    return L


def diag_line(st: StockSide, q: str, rows: np.ndarray, lab: str) -> str:
    S = st.B(rows)
    d = st.describe(rows)
    a = vs_twins(st.R("L3_ret_20"), S, st.bmi)
    cr = vs_twins(st.R("L3_crash_20"), S, st.bmi)
    a40 = vs_twins(st.R("L3_ret_40"), S, st.bmi)
    return (
        f"  {q} {lab:<26} n={d['n']:>9,} 买不进 {d['unbuy']:4.1f}% | 底数 {d['ret_20']:+6.2f}% / {d['ex_sz_20']:+6.2f}% 雷 {d['crash_20']:4.1f}% | "
        f"L3: {a[0]:+.2f} (t={a[1]:+.1f}) / 雷 {cr[0]:+.1f} (t={cr[1]:+.1f}) / 40天 {a40[0]:+.2f} (t={a40[1]:+.1f})"
    )


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    log("index side (verify frozen definitions)")
    ix = IndexSide(verify=True)
    log("stock side")
    st = StockSide()

    L: list[str] = []
    L.append("=" * 118)
    L.append("Study 08b · 方向 D 复活赛：指数上失效的 Al Brooks 结构，切到 A 股个股后能不能起死回生（开发段 2016-01 → 2024-10，封存区零触碰）")
    L.append(f"个股面板: {PANEL_PATH.name} | 指数快照: {ix.sid} | 入池且可评估的股票日: {len(st.base):,}")
    L.append("=" * 118)
    L.append("\n[0] 形态定义冻结校验（本脚本快速实现 vs 05/06 原代码，在申万一级行业上逐条比对）")
    L.extend(ix.ver)

    rows, pvals = [], []
    for q, title, tk, ck, bdir, desc in QUESTIONS:
        lines, row = score_question(st, ix, q, title, st.ev[tk], st.ev[ck], ix.ev[tk], ix.ev[ck], bdir, desc, tk, ck)
        L.extend(lines)
        rows.append(row)
        pvals += [row["p_ret"], row["p_cr"]]

    L.append("\n" + "=" * 118)
    L.append("【预注册判定】主规格 L3（扣铁律 + 热度/市值 + 行业），收益与踩雷各一检验，共 8 个；FDR 为本脚本 8 个检验内的临时校正（08c 完成后按 14 个合并重算）")
    L.append("=" * 118)
    L.extend(verdict_lines(rows, bh_fdr(pvals)))

    L.append("\n" + "=" * 118)
    L.append("【诊断变体（不参与判定）】L3 口径：vs 双胞胎 20天收益差(%) / 踩雷差(pp) / 40天收益差(%)；底数为 20 天绝对收益与相对同市值档")
    L.append("=" * 118)
    for q, key, lab in DIAG:
        L.append(diag_line(st, q, st.ev[key], lab))

    # ---- stock-specific add-on: does holding the breakout level matter? (stocks vs index)
    L.append("\n  【Q1 个股加试】回踩低点守住前高 vs 击穿前高后再收回（两者直接对比，L3 / 指数端相对31行业）")
    R, bmi, tier, year = st.R, st.bmi, st.tier, st.year
    Sh, Sb = st.B(st.ev["bp_h1_hold"]), st.B(st.ev["bp_h1_break"])
    hb = vs_ctrl(R("L3_ret_20"), Sh, Sb, bmi)
    hb_cr = vs_ctrl(R("L3_crash_20"), Sh, Sb, bmi)
    hb40 = vs_ctrl(R("L3_ret_40"), Sh, Sb, bmi)
    hb_big = vs_ctrl(R("L3_ret_20"), Sh[np.isin(tier[Sh], [1, 2])], Sb[np.isin(tier[Sb], [1, 2])], bmi)
    hb_sm = vs_ctrl(R("L3_ret_20"), Sh[np.isin(tier[Sh], [3, 4])], Sb[np.isin(tier[Sb], [3, 4])], bmi)
    yl = [f"{y}:{vs_ctrl(R('L3_ret_20'), Sh[year[Sh] == y], Sb[year[Sb] == y], bmi)[0]:+.2f}" for y in sorted(set(year[Sh]))]
    Ih, Ib = ix.ev["bp_h1_hold"], ix.ev["bp_h1_break"]
    ihb = vs_ctrl(ix.sw["R_rel_20"].to_numpy() * 100, Ih, Ib, ix.mi)
    L.append(
        f"    个股: 守住 − 击穿 20天 {hb[0]:+.2f}% (t={hb[1]:+.1f}) | 踩雷 {hb_cr[0]:+.1f}pp (t={hb_cr[1]:+.1f}) | 40天 {hb40[0]:+.2f}% (t={hb40[1]:+.1f}) | "
        f"大中盘 {hb_big[0]:+.2f}% (t={hb_big[1]:+.1f}) / 中小微 {hb_sm[0]:+.2f}% (t={hb_sm[1]:+.1f})"
    )
    L.append("    逐年(守住−击穿, %): " + "  ".join(yl))
    L.append(f"    指数端(申万一级): 守住 n={len(Ih)} / 击穿 n={len(Ib)}，守住 − 击穿 相对31行业 {ihb[0]:+.2f}% (t={ihb[1]:+.1f})")

    report = "\n".join(L)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(SUMMARY_PATH, index=False)
    print(report)
    log(f"done in {time.time() - t0:.0f}s -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
