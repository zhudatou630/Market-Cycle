#!/usr/bin/env python3
"""
Market Cycle V2 — Study 09（第三阶段专题 1 · 交易账本重考核心买卖点：引擎 + SPX + 31 个申万一级行业）

预注册见 decisions.md「09-预注册」（方案全文 output/09_plan_for_review.md v2）。主规格开跑前写死：
  - 统一进场：t 日收盘时形态只用 ≤t 数据确认 → t+1 越过信号 K 线高点（多）/ 低点（空）才进场，止损在信号 K 线另一侧；
    计划风险 R_plan = 高 − 低（t 日已知），< 0.3×ATR20 不做；目标 1R（辅 2R）；进场后 20 个市场交易日按收盘平仓；
    同根双碰先算止损（乐观变体：进场当天低点视为发生在进场前、其余日子先算目标）；跳空越过按开盘价；
    来回成本 SPX 0.03% / 行业 0.10%；右端截尾：信号日距开发段末日不足 21 个交易日不入池。
  - 傻瓜基线：每一根合格 K 线都按同样规则当信号 K 线（随机池剔除该形态族自己的信号日）。
    主检验 = 池化回归：净 R = 常数 + 形态哑变量 + 控制（近 20 日涨跌、20 日波动、近 5 日涨跌 线性；
    计划风险/ATR20、计划风险/价格、收盘位置 各五分位哑变量），交易日历对齐 Driscoll-Kraay（lag 20），绝对口径。
  - 5 形态 × 多空（多 = 交易，空 = 风控）：
      H2 趋势中 High 2（对照 High 1） / Low 2（对照 Low 1）           —— 新写，Brooks 官方词典原文
      BP 突破回踩 05 bp_h1_20（对照 ord_pb） / 05 代码严格镜像            —— 沿用 05（经 08b 快速实现）
      FB 假跌破 05 spring_20（对照 bdn_true_20） / fail_up_20（对照真突破）—— 沿用 05
      W3 三推底 06 p3_bot_turn（对照 p2_bot_turn） / p3_top_turn（对照 p2_top_turn）—— 沿用 06
      MC 微型通道第一次回调（对照同背景非微型通道回调） / 镜像          —— 新写，Brooks 官方词典原文
    空头一律由「价格镜像 p' = K − p」复用同一份多头代码得到（ATR、均线、收盘位置均精确镜像，r20 只用符号）。
  - SPX：开盘价不可靠（open == 前收 或 2010 年前）的行按挂单价成交（全段 = 无跳空近似版）；另报 2010–2019 计跳空 vs 不计跳空。
  - SPX 不进 FDR，只作跨市场方向核验（成交 ≥ 50 笔报 t）；A 股 FDR（行业 10 + 个股 10）在 09b 合并。

用法：
  python studies/09_trade_ledger_index.py --step0   # 引擎自检 + 定义比对 + 样本量（不看任何结果）
  python studies/09_trade_ledger_index.py           # 第 1 步完整结果
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
STEP0_PATH = OUTPUT_DIR / "09_step0_report.txt"
REPORT_PATH = OUTPUT_DIR / "09_trade_ledger_index_report.txt"
SUMMARY_PATH = OUTPUT_DIR / "09_trade_ledger_index_summary.csv"


def _load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "studies" / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


b8 = _load_mod("b8", "08b_stock_revival_structures.py")
s01, s06 = b8.s01, b8.s06
log = b8.log

EPS = 1e-12
MAX_HOLD = 20
EXIT_DELAY = 10
R_MIN_ATR = 0.3
IDX_BUFFER = 21
NW_LAG = 20
SPX_EVAL_START = pd.Timestamp("1984-01-01")
SW_EVAL_START = pd.Timestamp("2016-01-01")
SPX_REAL_OPEN_FROM = 2010
COST = {"SPX": 0.0003, "SW": 0.0010}
MIN_TRADES_SPX = 50

# (family, label, long signal, long control, short signal, short control)
FAMILIES = [
    ("H2", "趋势中 High 2 / Low 2", "h2", "h1", "l2", "l1"),
    ("BP", "突破回踩", "bp_h1_20", "ord_pb", "bp_dn", "ord_pb_dn"),
    ("FB", "假突破反向", "spring_20", "bdn_true_20", "fail_up_20", "brk_true_20"),
    ("W3", "三推反转", "p3_bot_turn", "p2_bot_turn", "p3_top_turn", "p2_top_turn"),
    ("MC", "微型通道第一次回调", "mc", "mc_ctrl", "mc_dn", "mc_ctrl_dn"),
]
SIDE_NAME = {1: "多头（交易）", -1: "空头镜像（风控）"}

# exit kinds / not-entered reasons
K_TARGET, K_STOP, K_TIME, K_FORCED = 1, 2, 3, 4
W_OK, W_TRUNC, W_SUSP, W_NOTRIG, W_LIMIT, W_NOOPEN = 0, 1, 2, 3, 4, 5


# ---------------------------------------------------------------------------
# 1. Bracket-order ledger engine (vectorised over trades; wide (T, N) market-calendar arrays)
# ---------------------------------------------------------------------------
def run_ledger(P: dict, sig_t: np.ndarray, sig_n: np.ndarray, side: int, *, k: float = 1.0, both: str = "cons",
               t1: str | None = None, entry: str = "stop", cost: float | np.ndarray = 0.0,
               max_hold: int = MAX_HOLD, exit_delay: int = EXIT_DELAY, use_open: bool = True,
               buffer: int = IDX_BUFFER, tgt_px: np.ndarray | None = None) -> dict:
    """Simulate one bracket trade per signal bar (sig_t, sig_n).

    entry="limit" (added for study 10; the "stop" / "open" paths are unchanged): buy limit at the signal-bar low, filled
    if the next bar trades at least one tick (next bar's tick) below it, at the open if the open is already that far below;
    stop / target are fixed prices limit ∓ planned risk (signal high − low), R unit = planned risk; tgt_px (real prices,
    per trade) overrides the target. No limit-up / limit-down entry block. Entry-day truth table (T+0): stop only → stop;
    target only → target if filled at the open, else cons holds / opt takes it; both → cons stop / opt target; an open
    beyond the stop fills and stops at the open (subject to the sealed-limit check). T+1 ("forced"): stop only, both modes.

    P: wide arrays on the market calendar, NaN where the asset did not trade:
       O, H, L, C (prices), OK (open reliable), UP / DN (limit prices, NaN = no limit), TICK (price tick, 0 = none).
    side +1: buy stop above signal high, stop below signal low. side −1: exact mirror (prices negated, H↔L, UP↔DN).
    Long entries at the limit-up price are unfilled; short (holder-sell) entries at limit-down are unfilled only if the
    close is sealed there. Stop / time exits blocked by a sealed limit (close at limit) go pending and are retried at the
    next traded open for up to `exit_delay` market days, then frozen at the last close.
    """
    T, N = P["C"].shape
    sig_t = np.asarray(sig_t, np.int64)
    sig_n = np.asarray(sig_n, np.int64)
    M = len(sig_t)
    Of, Hf, Lf, Cf = (P[x].reshape(-1) for x in ("O", "H", "L", "C"))
    OKf = P["OK"].reshape(-1) if use_open else np.zeros(T * N, bool)
    UPf, DNf, TKf = P["UP"].reshape(-1), P["DN"].reshape(-1), P["TICK"].reshape(-1)

    def g(t: np.ndarray, n: np.ndarray):
        f = t * N + n
        o, h, l, c, up, dn = Of[f], Hf[f], Lf[f], Cf[f], UPf[f], DNf[f]
        if side > 0:
            return o, h, l, c, up, dn
        return -o, -l, -h, -c, -dn, -up

    why = np.full(M, W_OK, np.int8)
    st = np.zeros(M, np.int8)  # 0 not entered / 1 active / 2 pending exit / 3 done
    fill, stop, R, tgt, xpx = (np.full(M, np.nan) for _ in range(5))
    kind = np.zeros(M, np.int8)
    xt = np.full(M, -1, np.int64)
    deadline = np.full(M, -1, np.int64)
    last_c = np.full(M, np.nan)
    gap_in = np.zeros(M, bool)
    outside = np.zeros(M, bool)
    frozen = np.zeros(M, bool)
    blocked = np.zeros(M, bool)
    gstop = np.zeros(M, bool)
    ht = np.zeros(M)
    e = sig_t + 1

    ok = sig_t + buffer <= T - 1
    why[~ok] = W_TRUNC
    i0 = np.flatnonzero(ok)
    _, Hs, Ls, _, _, _ = g(sig_t[i0], sig_n[i0])
    tick = TKf[sig_t[i0] * N + sig_n[i0]]
    ht[i0] = 0.5 * tick
    o, h, l, c, up, dn = g(e[i0], sig_n[i0])
    tol_e = 0.5 * TKf[e[i0] * N + sig_n[i0]]
    oke = OKf[e[i0] * N + sig_n[i0]]
    trd = np.isfinite(c) & np.isfinite(h) & np.isfinite(l)
    why[i0[~trd]] = W_SUSP
    if entry == "stop":
        level = Hs + tick
        trig = trd & (h > level - 0.5 * tick)
        gap = oke & (o > level - 0.5 * tick)
        f_ = np.where(gap, np.maximum(o, level), level)
        stp = Ls - tick
    elif entry == "open":
        trig = trd & oke & np.isfinite(o)
        gap = np.ones(len(i0), bool)
        f_ = o
        stp = o - (Hs - Ls)
        why[i0[trd & ~trig]] = W_NOOPEN
    elif entry == "limit":
        tick_e = TKf[e[i0] * N + sig_n[i0]]
        level = Ls
        rp = Hs - Ls
        trig = trd & (l < level - 0.5 * tick_e)
        gap = oke & (o < level - 0.5 * tick_e)
        f_ = np.where(gap, o, level)
        stp = level - rp
        why[i0[trd & ~trig]] = W_NOTRIG
    else:
        raise ValueError(entry)
    if tgt_px is not None and entry != "limit":
        raise ValueError("tgt_px is only supported for limit entries")
    if entry == "stop":
        why[i0[trd & ~trig]] = W_NOTRIG
    if entry == "limit":
        blk = np.zeros_like(trig)
    else:
        blk = trig & (f_ >= up - tol_e)
        if side < 0:
            blk &= c >= up - tol_e
    why[i0[blk]] = W_LIMIT
    blocked[i0[blk]] = True
    go = trig & ~blk
    ii = i0[go]
    f_, stp, gap = f_[go], stp[go], gap[go]
    o, h, l, c, dn, tol_e = o[go], h[go], l[go], c[go], dn[go], tol_e[go]
    fill[ii], stop[ii] = f_, stp
    if entry == "limit":
        rp_g, lv_g = rp[go], level[go]
        R[ii] = rp_g
        if tgt_px is None:
            tgt[ii] = lv_g + k * rp_g
        else:
            tp = np.asarray(tgt_px, float)[ii] * (1.0 if side > 0 else -1.0)
            assert (tp > lv_g).all(), "explicit target must lie beyond the limit price"
            tgt[ii] = tp
    else:
        R[ii] = f_ - stp
        tgt[ii] = f_ + k * (f_ - stp)
    gap_in[ii] = gap
    last_c[ii] = c
    st[ii] = 1

    # --- entry day
    sh = l < stp + ht[ii]
    th = h >= tgt[ii]
    outside[ii] = sh
    if entry == "limit":
        thru = gap & (o < stp + ht[ii])  # opened beyond the stop: filled at the open, the attached stop fires at the open
        gstop[ii] = thru
        if t1 is None:
            if both == "cons":
                xs, xg = sh, ~sh & th & gap
            else:
                xs, xg = thru | (sh & ~th), ~thru & th
            sel = ii[xg]
            xpx[sel], kind[sel], xt[sel], st[sel] = tgt[sel], K_TARGET, e[sel], 3
            px = np.where(thru, o, stp)[xs]
            sealed = (px <= dn[xs] + tol_e[xs]) & (c[xs] <= dn[xs] + tol_e[xs])
            sel = ii[xs]
            okx = ~sealed
            xpx[sel[okx]], kind[sel[okx]], xt[sel[okx]], st[sel[okx]] = px[okx], K_STOP, e[sel[okx]], 3
            kind[sel[~okx]], st[sel[~okx]], deadline[sel[~okx]] = K_STOP, 2, e[sel[~okx]] + exit_delay
        elif t1 == "forced":
            sel = ii[sh]
            kind[sel], st[sel], deadline[sel] = K_FORCED, 2, e[sel] + exit_delay
        elif t1 != "ignore":
            raise ValueError(t1)
    elif t1 is None:
        if both == "cons":
            xs, xg = sh, ~sh & th
        else:
            xg = th
            xs = ~th & sh & gap
        sel = ii[xg]
        xpx[sel], kind[sel], xt[sel], st[sel] = tgt[sel], K_TARGET, e[sel], 3
        px = stp[xs]
        sealed = (px <= dn[xs] + tol_e[xs]) & (c[xs] <= dn[xs] + tol_e[xs])
        sel = ii[xs]
        okx = ~sealed
        xpx[sel[okx]], kind[sel[okx]], xt[sel[okx]], st[sel[okx]] = px[okx], K_STOP, e[sel[okx]], 3
        kind[sel[~okx]], st[sel[~okx]], deadline[sel[~okx]] = K_STOP, 2, e[sel[~okx]] + exit_delay
    elif t1 == "forced":
        cnt = sh if both == "cons" else (sh & gap)
        sel = ii[cnt]
        kind[sel], st[sel], deadline[sel] = K_FORCED, 2, e[sel] + exit_delay
    elif t1 != "ignore":
        raise ValueError(t1)

    # --- following days (pending exits first, then active trades)
    for d in range(1, max_hold + exit_delay + 2):
        ip = np.flatnonzero(st == 2)
        if len(ip):
            kk = e[ip] + d
            late = (kk > deadline[ip]) | (kk > T - 1)
            sel = ip[late]
            xpx[sel], xt[sel], st[sel], frozen[sel] = last_c[sel], np.minimum(deadline[sel], T - 1), 3, True
            ip, kk = ip[~late], kk[~late]
            o, h, l, c, up, dn = g(kk, sig_n[ip])
            okk = OKf[kk * N + sig_n[ip]]
            tol = 0.5 * TKf[kk * N + sig_n[ip]]
            trd = np.isfinite(c)
            last_c[ip[trd]] = c[trd]
            free_dn = ~(dn > -np.inf)  # NaN -> no limit
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
        assert (kk <= T - 1).all(), "right-end buffer too small"
        o, h, l, c, up, dn = g(kk, sig_n[ia])
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
        # target exits (selling into strength is never blocked by a limit)
        xg = gt | xg_in
        px_g = np.where(gt, o, tg)
        sel = ia[xg]
        xpx[sel], kind[sel], xt[sel], st[sel] = px_g[xg], K_TARGET, kk[xg], 3
        # stop exits
        xs = gs | xs_in
        px_s = np.where(gs, o, sp)
        sealed = (px_s <= dn + tol) & (c <= dn + tol)
        a = xs & ~sealed
        sel = ia[a]
        xpx[sel], kind[sel], xt[sel], st[sel] = px_s[a], K_STOP, kk[a], 3
        b = xs & sealed
        sel = ia[b]
        kind[sel], st[sel], deadline[sel] = K_STOP, 2, kk[b] + exit_delay
        # time exit
        if d == max_hold:
            rem = ~xg & ~xs
            a = rem & trd & ~(c <= dn + tol)
            sel = ia[a]
            xpx[sel], kind[sel], xt[sel], st[sel] = c[a], K_TIME, kk[a], 3
            b = rem & ~a
            sel = ia[b]
            kind[sel], st[sel], deadline[sel] = K_TIME, 2, kk[b] + exit_delay
    assert not (st == 1).any() and not (st == 2).any(), "unresolved trades"

    filled = why == W_OK
    gross = np.full(M, np.nan)
    gross[filled] = (xpx[filled] - fill[filled]) / R[filled]
    cst = np.broadcast_to(np.asarray(cost, float), (M,))
    net = gross - cst * np.abs(fill) / R
    sg = 1.0 if side > 0 else -1.0  # report real prices (short side was simulated on negated prices)
    return {"why": why, "filled": filled, "fill": sg * fill, "R": R, "xpx": sg * xpx, "gross": gross, "net": net,
            "win": filled & (kind == K_TARGET), "kind": kind, "hold": np.where(filled, xt - e, -1),
            "gap_in": gap_in, "outside": outside, "frozen": frozen, "blocked": blocked, "gstop": gstop}


# ---------------------------------------------------------------------------
# 2. Engine self-tests: hand-made cases + random-walk paths with known intraday truth
# ---------------------------------------------------------------------------
def _toy(bars: dict[int, tuple], T: int = 45, neutral=(10.0, 10.3, 9.7, 10.0), up=None, dn=None, tick=0.0):
    O, H, L, C = (np.full((T, 1), v) for v in neutral)
    for t, b in bars.items():
        if b is None:
            O[t], H[t], L[t], C[t] = np.nan, np.nan, np.nan, np.nan
        else:
            O[t], H[t], L[t], C[t] = b
    P = {"O": O, "H": H, "L": L, "C": C, "OK": np.isfinite(O), "UP": np.full((T, 1), np.nan),
         "DN": np.full((T, 1), np.nan), "TICK": np.full((T, 1), tick)}
    for t, v in (up or {}).items():
        P["UP"][t] = v
    for t, v in (dn or {}).items():
        P["DN"][t] = v
    return P


def _one(P, side=1, **kw):
    r = run_ledger(P, np.array([5]), np.array([0]), side, buffer=25, **kw)
    return {k: (v[0].item() if hasattr(v[0], "item") else v[0]) for k, v in r.items()}


def _truth_paths(rng, T: int, N: int, m: int = 256):
    """Random walk with m intraday steps and occasional opening gaps; returns OHLC and full paths.
    (Coarse paths, e.g. 24 steps/day, overshoot after a stop-entry crossing and inflate the 1R win rate to ~53-55%;
    with fine paths the exact truth converges to the 50% / 33% random-walk values.)"""
    steps = rng.normal(0, 0.004 * math.sqrt(24 / m), size=(T, N, m))
    gaps = rng.normal(0, 0.006, size=(T, N)) * (rng.random((T, N)) < 0.3)
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


def _truth_trade(paths, s, n, k, max_hold=MAX_HOLD):
    """Exact outcome from the continuous (linearly interpolated) intraday path; long only, no limits."""
    hs, ls = paths[s, n].max(), paths[s, n].min()
    e = s + 1
    p = paths[e, n]
    if p.max() <= hs:
        return None
    level, stp = hs, ls
    if p[0] > level:
        fill, seq = p[0], p
    else:
        j = int(np.argmax(p > level))
        fill, seq = level, np.r_[level, p[j:]]
    R = fill - stp
    tg = fill + k * R
    big = 10 ** 9
    for d in range(max_hold + 1):
        q = seq if d == 0 else paths[e + d, n]
        if d > 0 and (q[0] <= stp or q[0] >= tg):
            return (q[0] - fill) / R
        hs_, ht_ = q <= stp, q >= tg
        i_s = int(np.argmax(hs_)) if hs_.any() else big
        i_t = int(np.argmax(ht_)) if ht_.any() else big
        if i_s < i_t:
            return -1.0
        if i_t < i_s:
            return float(k)
        if d == max_hold:
            return (q[-1] - fill) / R
    return np.nan


def selftest() -> list[str]:
    out = []

    def chk(name, cond):
        assert cond, f"self-test failed: {name}"
        out.append(f"  ✔ {name}")

    sig = (9.5, 10.0, 9.0, 9.8)
    e1 = (9.9, 10.5, 9.8, 10.2)
    r = _one(_toy({5: sig, 6: e1, 7: (10.3, 11.2, 10.1, 11.0)}))
    chk("普通止盈：+1R、持有 1 天", r["filled"] and abs(r["gross"] - 1) < 1e-9 and r["hold"] == 1 and r["kind"] == K_TARGET)
    r = _one(_toy({5: sig, 6: (10.4, 10.6, 10.3, 10.5)}))
    chk("跳空高开按开盘价成交、R 变大、到期按收盘", abs(r["fill"] - 10.4) < 1e-9 and abs(r["R"] - 1.4) < 1e-9
        and r["kind"] == K_TIME and abs(r["gross"] - (10.0 - 10.4) / 1.4) < 1e-9 and r["hold"] == MAX_HOLD)
    P = _toy({5: sig, 6: e1, 7: (10.2, 11.5, 8.5, 10.0)})
    chk("同根双碰：保守 = -1R", abs(_one(P)["gross"] + 1) < 1e-9)
    chk("同根双碰：乐观 = +1R", abs(_one(P, both="opt")["gross"] - 1) < 1e-9)
    P = _toy({5: sig, 6: (9.5, 10.2, 8.9, 10.1), 7: (9.6, 10.3, 9.4, 10.0)})
    chk("进场当天外包线：保守 = 当天止损 -1R", abs(_one(P)["gross"] + 1) < 1e-9 and _one(P)["outside"])
    chk("进场当天外包线：乐观 = 低点在进场前、继续持有", _one(P, both="opt")["kind"] == K_TIME)
    chk("T+1 强制：次日开盘卖", abs(_one(P, t1="forced")["gross"] - (9.6 - 10) / 1) < 1e-9
        and _one(P, t1="forced")["kind"] == K_FORCED)
    chk("T+1 忽略：当天碰线不算", _one(P, t1="ignore")["kind"] == K_TIME)
    P = _toy({5: sig, 6: (10.4, 10.6, 8.8, 9.0)})
    chk("跳空高开后盘中跌破：乐观也算止损", _one(P, both="opt")["kind"] == K_STOP)
    r = _one(_toy({5: sig, 6: e1, 7: (8.5, 8.8, 8.2, 8.6)}))
    chk("跳空止损按开盘价：-1.5R", abs(r["gross"] + 1.5) < 1e-9)
    chk("未越过高点不成交（严格大于）", _one(_toy({5: sig, 6: (9.9, 10.0, 9.5, 9.8)}))["why"] == W_NOTRIG)
    chk("一跳：等于高点不成交", _one(_toy({5: sig, 6: (9.9, 10.0, 9.5, 9.8)}, tick=0.01))["why"] == W_NOTRIG)
    r = _one(_toy({5: sig, 6: (9.9, 10.01, 9.5, 9.8)}, tick=0.01))
    chk("一跳：高点 + 0.01 成交于 10.01、止损 8.99", r["filled"] and abs(r["fill"] - 10.01) < 1e-9 and abs(r["R"] - 1.02) < 1e-9)
    chk("开盘涨停买不进", _one(_toy({5: sig, 6: (10.4, 10.4, 10.4, 10.4)}, up={6: 10.4}, tick=0.01))["why"] == W_LIMIT)
    chk("盘中涨停但成交价在涨停价之下可买", _one(_toy({5: sig, 6: (9.9, 10.4, 9.8, 10.4)}, up={6: 10.4}, tick=0.01))["filled"])
    P = _toy({5: sig, 6: (9.9, 10.5, 9.8, 10.2), 7: (8.6, 8.6, 8.6, 8.6), 8: (8.0, 8.5, 7.9, 8.2)},
             dn={7: 8.6, 8: 7.74}, tick=0.01)
    r = _one(P)
    chk("一字跌停卖不出 → 次日开盘卖", abs(r["xpx"] - 8.0) < 1e-9 and r["hold"] == 2 and not r["frozen"])
    bars = {5: sig, 6: (9.9, 10.5, 9.8, 10.2)}
    dns = {}
    for t in range(7, 20):
        p = round(8.6 * 0.9 ** (t - 7), 2)
        bars[t] = (p, p, p, p)
        dns[t] = p
    r = _one(_toy(bars, dn=dns, tick=0.01))
    chk("连续一字跌停超过 10 天 → 冻结在最后收盘价", r["frozen"] and abs(r["xpx"] - bars[17][3]) < 1e-9)
    chk("次日停牌 → 不成交", _one(_toy({5: sig, 6: None}))["why"] == W_SUSP)
    r = _one(_toy({5: (9.5, 10.0, 9.0, 9.2), 6: (9.1, 9.3, 8.8, 8.9), 7: (8.8, 8.9, 7.9, 8.0)}), side=-1)
    chk("空头镜像：跌破低点卖出、跌 1R 买回 = +1R", r["filled"] and abs(r["fill"] - 9.0) < 1e-9 and abs(r["gross"] - 1) < 1e-9)
    P = _toy({5: (9.5, 10.0, 9.0, 9.2), 6: (8.5, 8.5, 8.5, 8.5)}, dn={6: 8.5}, tick=0.01)
    chk("空头：一字跌停卖不出 = 未成交", _one(P, side=-1)["why"] == W_LIMIT)
    P = _toy({5: (9.5, 10.0, 9.0, 9.2), 6: (8.5, 9.0, 8.5, 8.8)}, dn={6: 8.5}, tick=0.01)
    r = _one(P, side=-1)
    chk("空头：开盘跌停但收盘打开 → 按跌停价卖出", r["filled"] and abs(r["fill"] - 8.5) < 1e-9)
    P = _toy({5: (9.3, 9.5, 9.0, 9.1), 6: (9.1, 9.2, 8.8, 8.9), 7: (9.79, 9.79, 9.79, 9.79), 8: (9.6, 9.7, 9.5, 9.6)},
             up={7: 9.79}, tick=0.01)
    r = _one(P, side=-1)
    chk("空头：止损买回遇封死涨停 → 次日开盘买回", abs(r["fill"] - 8.99) < 1e-9 and abs(r["xpx"] - 9.6) < 1e-9
        and r["hold"] == 2 and abs(r["gross"] - (8.99 - 9.6) / 0.52) < 1e-9)
    r = _one(_toy({5: sig, 6: e1, 7: (10.3, 11.2, 10.1, 11.0)}), k=2.0)
    chk("2R 目标：到 11.2 未达 12，到期", r["kind"] == K_TIME)
    r = _one(_toy({5: sig, 6: e1}), entry="open")
    chk("开盘进场诊断：成交于开盘、止损宽度 = 计划风险", abs(r["fill"] - 9.9) < 1e-9 and abs(r["R"] - 1.0) < 1e-9)
    r = _one(_toy({5: sig, 6: e1}), cost=0.001)
    chk("成本换算：净 R = 毛 R − 0.1% × 价格 / R", abs((r["gross"] - r["net"]) - 0.001 * 10 / 1) < 1e-12)

    # random-walk paths: engine bounds must bracket the exact intraday truth
    rng = np.random.default_rng(20261125)
    T, N = 260, 40
    P, paths = _truth_paths(rng, T, N)
    tt, nn = np.meshgrid(np.arange(30, T - 25), np.arange(N), indexing="ij")
    tt, nn = tt.ravel(), nn.ravel()
    for k in (1.0, 2.0):
        rc = run_ledger(P, tt, nn, 1, k=k, both="cons", buffer=22)
        ro = run_ledger(P, tt, nn, 1, k=k, both="opt", buffer=22)
        tr = np.array([np.nan if (v := _truth_trade(paths, s, n, k)) is None else v for s, n in zip(tt, nn)])
        f = rc["filled"]
        assert (np.isfinite(tr) == f).all(), "fill mismatch vs truth"
        twin = f & (tr >= k - 1e-9)
        tloss = f & (tr <= -1 + 1e-9)
        same = f & (rc["kind"] == ro["kind"]) & (np.abs(rc["xpx"] - ro["xpx"]) < 1e-9)
        bad1 = (twin & ~ro["win"]).sum()
        bad2 = (tloss & ~(rc["kind"] == K_STOP)).sum()
        bad3 = (same & (np.abs(rc["gross"] - tr) > 1e-6)).sum()
        assert bad1 == 0 and bad2 == 0 and bad3 == 0, f"truth bracket failed k={k}: {bad1},{bad2},{bad3}"
        out.append(f"  ✔ 随机游走真值夹逼（{k:.0f}R）：{f.sum():,} 笔成交；真实胜率 {np.mean(tr[f] >= k - 1e-9) * 100:.1f}% "
                   f"落在 保守 {rc['win'][f].mean() * 100:.1f}% ~ 乐观 {ro['win'][f].mean() * 100:.1f}% 之间；"
                   f"两种假设一致的 {same.sum():,} 笔与真值逐笔相同（理论值 {100 / (1 + k):.1f}%）")
    return out


# ---------------------------------------------------------------------------
# 3. Signal definitions (long side on real prices; short side = same code on mirrored prices K − p)
# ---------------------------------------------------------------------------
def _shift(x: np.ndarray, k: int) -> np.ndarray:
    return b8._shift(x, k)


def base_feats(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> dict[str, np.ndarray]:
    prev_c = _shift(c, 1)
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    s_c = pd.Series(c)
    ma20 = s_c.rolling(20, min_periods=20).mean().to_numpy()
    lc = pd.Series(np.log(np.abs(c)))
    d20 = c - _shift(c, 20)  # sign of r20 (exact under the mirror p' = K − p)
    rng = h - l
    with np.errstate(invalid="ignore", divide="ignore"):
        kcls = np.where(rng > EPS, (c - l) / rng, np.nan)
    return {
        "atr20": pd.Series(tr).rolling(20, min_periods=20).mean().to_numpy(),
        "ma20": ma20, "ma20_lag10": _shift(ma20, 10),
        "r20": lc.diff(20).to_numpy(), "r5": lc.diff(5).to_numpy(),
        "vol20": lc.diff().rolling(20, min_periods=20).std(ddof=1).to_numpy(),
        "up_ctx": (d20 > 0) & (ma20 > _shift(ma20, 10)),
        "ph20": pd.Series(h).rolling(20, min_periods=20).max().shift(1).to_numpy(),
        "pl20": pd.Series(l).rolling(20, min_periods=20).min().shift(1).to_numpy(),
        "kcls": kcls,
    }


def detect_hl2(h, l, ph, pl, ctx, em):
    """Brooks glossary High 1/2: a High 1 is a bar whose high is above the prior bar's high in a pullback; after a bar
    with a lower high, the next bar whose high is above the prior bar's high is a High 2. Anchor = latest 20-day high;
    a new 20-day high restarts the count; > 20 bars or a break of the 20-day low kills it; once the H2 bar happens the
    search stops (traded or not). Returns signal bars (the bar BEFORE the H1 / H2 bar) for H1 and H2."""
    H, L, PH, PL, CX, EM = (x.tolist() for x in (h, l, ph, pl, ctx, em))
    fin = math.isfinite
    c1: list[int] = []
    c2: list[int] = []
    alive, anchor, got1, lower = False, -1, False, False
    for t in range(1, len(H)):
        if not (fin(PH[t]) and fin(PL[t]) and fin(H[t]) and fin(L[t])):
            alive = False
            continue
        if alive and (t - anchor > 20 or L[t] < PL[t]):
            alive = False
        if H[t] > PH[t]:
            alive, anchor, got1, lower = True, t, False, False
            continue
        if not alive:
            continue
        up = H[t] > H[t - 1]
        if not got1:
            if up:
                got1 = True
        else:
            if lower and up:
                alive = False
                continue
            if H[t] < H[t - 1]:
                lower = True
        if CX[t] and EM[t]:
            if not got1:
                c1.append(t)
            elif lower:
                c2.append(t)
    return np.asarray(c1, dtype=int), np.asarray(c2, dtype=int)


def detect_mc(h, l, c, ctx, em):
    """Brooks micro channel: the 5 bars before the signal bar each have a low not below the prior bar's low and close
    net higher; the signal bar is the first bar whose low is below the prior bar's low (micro trend line false break).
    Control = same uptrend context, a bar with a lower low that is NOT preceded by such a micro channel."""
    nd = (l >= _shift(l, 1)).astype(float)
    run5 = pd.Series(nd).rolling(5, min_periods=5).sum().shift(1).to_numpy() == 5
    micro = run5 & (_shift(c, 1) > _shift(c, 6))
    pb = l < _shift(l, 1)
    base = pb & ctx & em
    return np.flatnonzero(base & micro), np.flatnonzero(base & ~micro)


def detect_push_turns(h, l, c, f, em):
    """Fast mirror of studies/06 three-push TOP machine (turn events). Bottom turns = same code on mirrored prices."""
    H, Lw, C, A, MA, PH, PL, EM = (x.tolist() for x in (h, l, c, f["atr20"], f["ma20"], f["ph20"], f["pl20"], em))
    fin = math.isfinite
    p2, p3 = [], []
    push, peaks, leg_lows = 0, 0, []
    cur_peak, cur_idx, in_pb, pb_low, turn_fired, seq0 = -math.inf, -1, False, math.inf, False, -1
    for t in range(25, len(C)):
        a = A[t]
        if not fin(a) or a <= EPS or not fin(PH[t]):
            continue
        if push == 0:
            if H[t] > PH[t] and C[t] > MA[t]:
                push, cur_peak, cur_idx, seq0, peaks, leg_lows = 1, H[t], t, t, 0, []
                in_pb, pb_low, turn_fired = False, Lw[t], False
            continue
        broke = (t - seq0 > 65) or (Lw[t] < PL[t]) or (bool(leg_lows) and C[t] < leg_lows[-1])
        if broke:
            push = 0
            continue
        if not in_pb:
            if H[t] > cur_peak:
                cur_peak, cur_idx, pb_low = H[t], t, Lw[t]
            else:
                if Lw[t] < pb_low:
                    pb_low = Lw[t]
                if (C[t] < C[t - 1]) and (Lw[t] < Lw[t - 1]) and (t - cur_idx <= 4) and not turn_fired:
                    turn_fired = True
                    if push == 2 and EM[t]:
                        p2.append(t)
                    elif push >= 3 and peaks >= 2 and EM[t]:
                        p3.append(t)
                if (cur_peak - pb_low >= a) and (t - cur_idx >= 2):
                    in_pb = True
        else:
            if Lw[t] < pb_low:
                pb_low = Lw[t]
            if H[t] > cur_peak:
                peaks += 1
                leg_lows.append(pb_low)
                push += 1
                cur_peak, cur_idx, in_pb, pb_low, turn_fired = H[t], t, False, Lw[t], False
    return np.asarray(p2, dtype=int), np.asarray(p3, dtype=int)


def _one_side(h, l, c, dates, eval_start, upc=None, dnc=None):
    f = base_feats(h, l, c)
    em = np.asarray(pd.to_datetime(dates) >= eval_start, dtype=bool)
    ev, dpb = b8.detect_events(h, l, c, dates, eval_start, upc, dnc)
    h1, h2 = detect_hl2(h, l, f["ph20"], f["pl20"], f["up_ctx"], em)
    mc, mcc = detect_mc(h, l, c, f["up_ctx"], em)
    p2t, p3t = detect_push_turns(h, l, c, f, em)
    out = {"h1": h1, "h2": h2, "mc": mc, "mc_ctrl": mcc, "p2_turn": p2t, "p3_turn": p3t}
    for key in ["bp_h1_20", "ord_pb", "spring_20", "bdn_true_20", "fail_up_20", "tc_pb"]:
        out[key] = ev[key]
    h1e = ev["bp_h1_20"]
    dv = pd.Series(dpb, dtype=float).reindex(h1e).to_numpy()
    out["bp_h1_hold"], out["bp_h1_break"] = h1e[dv >= 0], h1e[dv < 0]
    return out, f


def detect_all(h, l, c, dates, eval_start, upc=None, dnc=None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Long-side keys from real prices; short-side keys from the exact mirror p' = K − p (H' = K − L, L' = K − H).
    upc / dnc: one-word limit-up / limit-down close flags (08b frozen rule for close position); swapped on the mirror."""
    up, f = _one_side(h, l, c, dates, eval_start, upc, dnc)
    K = 2.0 * np.nanmax(h) + 1.0
    dn, _ = _one_side(K - l, K - h, K - c, dates, eval_start, dnc, upc)
    sig = {
        "h2": up["h2"], "h1": up["h1"], "l2": dn["h2"], "l1": dn["h1"],
        "bp_h1_20": up["bp_h1_20"], "ord_pb": up["ord_pb"], "bp_dn": dn["bp_h1_20"], "ord_pb_dn": dn["ord_pb"],
        "spring_20": up["spring_20"], "bdn_true_20": up["bdn_true_20"],
        "fail_up_20": up["fail_up_20"], "brk_true_20": dn["bdn_true_20"],
        "p3_bot_turn": dn["p3_turn"], "p2_bot_turn": dn["p2_turn"],
        "p3_top_turn": up["p3_turn"], "p2_top_turn": up["p2_turn"],
        "mc": up["mc"], "mc_ctrl": up["mc_ctrl"], "mc_dn": dn["mc"], "mc_ctrl_dn": dn["mc_ctrl"],
        "tc_pb": up["tc_pb"], "bp_h1_hold": up["bp_h1_hold"], "bp_h1_break": up["bp_h1_break"],
        "_mirror_spring": dn["spring_20"],
    }
    return sig, f


def verify_definitions(raw: dict[str, tuple[pd.DataFrame, pd.Timestamp]]) -> list[str]:
    """06 three-push turns (top & bottom via mirror) vs original code; mirrored spring == fail_up (mirror sanity)."""
    lines = []
    for mk, (df, ev0) in raw.items():
        bad, tot, badm = 0, 0, 0
        for asset, g in df.groupby("asset", sort=True):
            g = g.sort_values("date").reset_index(drop=True)
            h, l, c = (g[x].to_numpy(float) for x in ("high", "low", "close"))
            d = g["date"].to_numpy()
            sig, _ = detect_all(h, l, c, d, ev0)
            o6 = s06.build_asset_study06(g, "A股申万L1", ev0)
            for mine, ref in [("p2_top_turn", "p2_top_turn"), ("p3_top_turn", "p3_top_turn"),
                              ("p2_bot_turn", "p2_bot_turn"), ("p3_bot_turn", "p3_bot_turn")]:
                a = set(pd.to_datetime(d[sig[mine]]))
                b = set(pd.to_datetime(o6[ref]["date"])) if len(o6[ref]) else set()
                tot += len(b)
                if a != b:
                    bad += 1
                    lines.append(f"    不一致 {mk} {asset} {mine}: 原 {len(b)} 新 {len(a)}")
            if not np.array_equal(np.sort(sig["_mirror_spring"]), np.sort(sig["fail_up_20"])):
                badm += 1
        lines.insert(0, f"  {mk}：06 三推拐头（顶 / 底）快速实现 vs 原代码，原记录 {tot:,} 条 → "
                        + ("全部一致 ✔" if bad == 0 else f"{bad} 处不一致 ✘")
                        + f"；镜像自检（镜像弹簧 == 向上假突破）" + ("全部一致 ✔" if badm == 0 else f"{badm} 个标的不一致 ✘"))
        assert bad == 0 and badm == 0, "definition verification failed"
    return lines


# ---------------------------------------------------------------------------
# 4. Index markets on the market calendar
# ---------------------------------------------------------------------------
class IndexMarket:
    def __init__(self, name: str, df: pd.DataFrame, eval_start: pd.Timestamp) -> None:
        self.name, self.eval_start = name, eval_start
        df = df.sort_values(["asset", "date"]).reset_index(drop=True)
        self.dates = np.sort(df["date"].unique())
        self.assets = sorted(df["asset"].unique())
        T, N = len(self.dates), len(self.assets)
        self.T, self.N = T, N
        ti = np.searchsorted(self.dates, df["date"].to_numpy())
        ni = pd.Index(self.assets).get_indexer(df["asset"])

        def wide(v, fill=np.nan, dtype=float):
            W = np.full((T, N), fill, dtype=dtype)
            W[ti, ni] = v
            return W

        O, H, L, C = (wide(df[x].to_numpy(float)) for x in ("open", "high", "low", "close"))
        pc = df.groupby("asset")["close"].shift(1).to_numpy(float)
        fake = np.isclose(df["open"].to_numpy(float), pc)
        if name == "SPX":
            okv = (~fake) & (df["date"].dt.year.to_numpy() >= SPX_REAL_OPEN_FROM)
        else:
            okv = np.ones(len(df), bool)
        self.P = {"O": O, "H": H, "L": L, "C": C, "OK": wide(okv, False, bool),
                  "UP": np.full((T, N), np.nan), "DN": np.full((T, N), np.nan), "TICK": np.zeros((T, N))}
        self.fake_share = float(fake[df["date"].to_numpy() >= np.datetime64(eval_start)].mean())
        feats = {k: np.full((T, N), np.nan) for k in ["atr20", "r20", "r5", "vol20", "kcls"]}
        self.sig: dict[str, np.ndarray] = {}
        for asset, g in df.groupby("asset", sort=True):
            n = self.assets.index(asset)
            t_ = np.searchsorted(self.dates, g["date"].to_numpy())
            h, l, c = (g[x].to_numpy(float) for x in ("high", "low", "close"))
            sig, f = detect_all(h, l, c, g["date"].to_numpy(), eval_start)
            for kf in feats:
                feats[kf][t_, n] = f[kf]
            for ks, pos in sig.items():
                if ks.startswith("_"):
                    continue
                self.sig.setdefault(ks, np.zeros((T, N), bool))[t_[pos], n] = True
        self.F = feats
        evalm = (self.dates >= np.datetime64(eval_start))[:, None] & np.ones((1, N), bool)
        rplan = H - L
        with np.errstate(invalid="ignore", divide="ignore"):
            self.F["g_atr"] = rplan / feats["atr20"]
            self.F["g_px"] = rplan / C
        tix = np.arange(T)[:, None] * np.ones((1, N), int)
        ctrl_ok = np.isfinite(np.stack([feats[x] for x in ["r20", "r5", "vol20", "kcls", "atr20"]])).all(0)
        self.cand = evalm & np.isfinite(C) & (rplan > EPS) & ctrl_ok & (rplan >= R_MIN_ATR * feats["atr20"]) \
            & (tix + IDX_BUFFER <= T - 1)
        self.evalm = evalm & np.isfinite(C)
        fwd = np.full((T, N), np.nan)
        fwd[:-20] = C[20:] / C[:-20] - 1.0
        self.ret20 = fwd
        self.year = pd.DatetimeIndex(self.dates).year.to_numpy()
        self.cost = COST[name]

    def trades(self, side: int, **kw) -> dict:
        tt, nn = np.nonzero(self.cand)
        r = run_ledger(self.P, tt, nn, side, cost=self.cost, **kw)
        r["t"], r["n"] = tt, nn
        return r


# ---------------------------------------------------------------------------
# 5. Statistics
# ---------------------------------------------------------------------------
def _qdum(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    qs = np.nanquantile(ref, [0.2, 0.4, 0.6, 0.8])
    b = np.digitize(x, qs)
    return np.stack([(b == j).astype(float) for j in range(1, 5)], 1)


def design(mk: IndexMarket, tt, nn, ref_rows, with_kcls=True) -> np.ndarray:
    lin = np.column_stack([mk.F[x][tt, nn] for x in ["r20", "vol20", "r5"]])
    lin = (lin - np.nanmean(lin, 0)) / np.where(np.nanstd(lin, 0) > 0, np.nanstd(lin, 0), 1)
    parts = [lin]
    geo = ["g_atr", "g_px"] + (["kcls"] if with_kcls else [])
    for x in geo:
        v = mk.F[x][tt, nn]
        parts.append(_qdum(v, v[ref_rows]))
    return np.hstack(parts)


def coef_D(y, D, X, mi, lag=NW_LAG):
    Z = np.column_stack([np.ones(len(y)), D, X])
    w = b8.cap_weights(mi, D.astype(int))
    b, t = b8.wls_dk(y, Z, w, mi, lag)
    return float(b[1]), float(t[1])


def p_two(t):
    return b8.p_two(t)


def fmt(x, nd=3, pct=False):
    if x is None or not np.isfinite(x):
        return "   —  "
    return f"{x * 100:+.1f}%" if pct else f"{x:+.{nd}f}"


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def load_raw() -> dict[str, tuple[pd.DataFrame, pd.Timestamp]]:
    spx = s01.load_spx_dev()
    assert spx["date"].max() <= pd.Timestamp("2019-12-31")
    sw, _ = s01.load_sw_l1_dev()
    assert sw["date"].max() <= pd.Timestamp("2024-10-31")
    return {"SPX": (spx, SPX_EVAL_START), "SW": (sw, SW_EVAL_START)}


def step0(raw) -> list[str]:
    L = ["=" * 100, "Study 09 · 第 0 步：引擎自检 + 定义比对 + 样本量（不看任何结果）", "=" * 100, "", "[1] 账本引擎自检"]
    L += selftest()
    L += ["", "[2] 旧定义快速实现比对"]
    sw = raw["SW"][0]
    L += ["  SW：" + b8.verify_against_originals(sw)[0].strip()]
    L += verify_definitions(raw)
    L += ["", "[3] 样本量（信号 K 线数 = 通过最小风险过滤与截尾；成交 = 次日越过信号 K 线）"]
    for mk_name, (df, ev0) in raw.items():
        mk = IndexMarket(mk_name, df, ev0)
        L.append(f"  —— {mk_name}（{mk.N} 个标的，{pd.Timestamp(mk.dates[0]).date()} → {pd.Timestamp(mk.dates[-1]).date()}；"
                 f"开发段假开盘占比 {mk.fake_share * 100:.1f}%；合格 K 线 {mk.cand.sum():,}）")
        for side in (1, -1):
            r = mk.trades(side)
            filled = np.zeros_like(mk.cand)
            filled[r["t"][r["filled"]], r["n"][r["filled"]]] = True
            L.append(f"    {SIDE_NAME[side]}：随机池成交率 {r['filled'].mean() * 100:.1f}%")
            for fam, lab, ls, lc, ss, sc in FAMILIES:
                s_, c_ = (ls, lc) if side > 0 else (ss, sc)
                S, C_ = mk.sig[s_], mk.sig[c_]
                L.append(f"      {fam} {lab:<10s} 形态：信号 {int((S & mk.evalm).sum()):>6,} → 合格 {int((S & mk.cand).sum()):>6,}"
                         f" → 成交 {int((S & filled).sum()):>6,}   | 对照：合格 {int((C_ & mk.cand).sum()):>7,} → 成交 {int((C_ & filled).sum()):>7,}")
    return L


def analyze_market(mk: IndexMarket) -> tuple[list[str], list[dict]]:
    L, rows = [], []
    runs = {}
    for side in (1, -1):
        runs[(side, "main")] = mk.trades(side)
        runs[(side, "opt")] = mk.trades(side, both="opt")
        runs[(side, "k2")] = mk.trades(side, k=2.0)
        runs[(side, "open")] = mk.trades(side, entry="open")
        if mk.name == "SW" and side > 0:
            runs[(side, "t1")] = mk.trades(side, t1="forced")
        if mk.name == "SPX":
            runs[(side, "nogap")] = mk.trades(side, use_open=False)
    tt, nn = runs[(1, "main")]["t"], runs[(1, "main")]["n"]
    mi = tt
    yr = mk.year[tt]
    for side in (1, -1):
        base = runs[(side, "main")]
        fl = base["filled"]
        ro = runs[(side, "opt")]
        L.append(f"\n  —— {SIDE_NAME[side]}：随机池（全部合格 K 线）成交 {fl.sum():,} 笔，胜率 {base['win'][fl].mean() * 100:.1f}%"
                 f"（乐观 {ro['win'][ro['filled']].mean() * 100:.1f}%），"
                 f"毛 {np.nanmean(base['gross'][fl]):+.3f}R，净 {np.nanmean(base['net'][fl]):+.3f}R"
                 f"（乐观 {np.nanmean(ro['net'][ro['filled']]):+.3f}R），"
                 f"到期占比 {(base['kind'][fl] == K_TIME).mean() * 100:.1f}%，持有中位 {np.median(base['hold'][fl]):.0f} 天")
        for fam, lab, ls, lc, ss, sc in FAMILIES:
            s_, c_ = (ls, lc) if side > 0 else (ss, sc)
            Sm = mk.sig[s_][tt, nn]
            Cm = mk.sig[c_][tt, nn] & ~Sm
            row = {"market": mk.name, "family": fam, "side": side, "label": lab}

            def excess(run, with_kcls=True, sub=None, ycol="net"):
                f = run["filled"] & np.isfinite(run[ycol])
                if sub is not None:
                    f = f & sub
                rows_ = np.flatnonzero(f)
                if Sm[rows_].sum() < 10:
                    return np.nan, np.nan
                ref = ~Sm[rows_]
                X = design(mk, tt[rows_], nn[rows_], ref, with_kcls)
                y = run[ycol][rows_].astype(float)
                return coef_D(y, Sm[rows_].astype(float), X, mi[rows_])

            Sf = Sm & fl
            row.update({
                "n_sig": int(Sm.sum()), "n_fill": int(Sf.sum()), "fill_rate": float(fl[Sm].mean()) if Sm.any() else np.nan,
                "win": float(base["win"][Sf].mean()) if Sf.any() else np.nan,
                "gross": float(np.nanmean(base["gross"][Sf])) if Sf.any() else np.nan,
                "net": float(np.nanmean(base["net"][Sf])) if Sf.any() else np.nan,
                "rand_win": float(base["win"][fl & ~Sm].mean()), "rand_net": float(np.nanmean(base["net"][fl & ~Sm])),
                "time_share": float((base["kind"][Sf] == K_TIME).mean()) if Sf.any() else np.nan,
                "outside": float(base["outside"][Sf].mean()) if Sf.any() else np.nan,
                "rand_outside": float(base["outside"][fl & ~Sm].mean()),
            })
            ro = runs[(side, "opt")]
            fo = ro["filled"]
            row.update({
                "win_opt": float(ro["win"][Sm & fo].mean()) if (Sm & fo).any() else np.nan,
                "net_opt": float(np.nanmean(ro["net"][Sm & fo])) if (Sm & fo).any() else np.nan,
                "rand_win_opt": float(ro["win"][fo & ~Sm].mean()), "rand_net_opt": float(np.nanmean(ro["net"][fo & ~Sm])),
            })
            row["ex"], row["ex_t"] = excess(base)
            row["ex_nok"], row["ex_nok_t"] = excess(base, with_kcls=False)
            row["ex_win"], row["ex_win_t"] = excess({**base, "w": base["win"].astype(float)}, ycol="w")
            row["ex_gross"], row["ex_gross_t"] = excess(base, ycol="gross")
            for v in ["opt", "k2", "open", "t1"]:
                if (side, v) in runs:
                    row[f"ex_{v}"], row[f"ex_{v}_t"] = excess(runs[(side, v)])
            # vs specialized control
            fc = fl & (Sm | Cm) & np.isfinite(base["net"])
            rr = np.flatnonzero(fc)
            if Sm[rr].sum() >= 10 and Cm[rr].sum() >= 10:
                X = design(mk, tt[rr], nn[rr], ~Sm[rr])
                row["vs_ctrl"], row["vs_ctrl_t"] = coef_D(base["net"][rr], Sm[rr].astype(float), X, mi[rr])
                row["n_ctrl"] = int(Cm[rr].sum())
            # stability by period
            if mk.name == "SW":
                periods = [(y, y) for y in range(2016, 2025)]
                halves = [(2016, 2020), (2021, 2024)]
            else:
                periods = [(1984, 1992), (1993, 2001), (2002, 2010), (2011, 2019)]
                halves = [(1984, 2001), (2002, 2019)]
            sgn = []
            for a, b in periods:
                v, _ = excess(base, sub=(yr >= a) & (yr <= b))
                sgn.append(v)
            row["per"] = sgn
            row["halves"] = [excess(base, sub=(yr >= a) & (yr <= b))[0] for a, b in halves]
            if mk.name == "SPX":
                sub = yr >= SPX_REAL_OPEN_FROM
                row["gap_on"], _ = excess(base, sub=sub)
                row["gap_off"], _ = excess(runs[(side, "nogap")], sub=sub)
            # old judge: hold 20 days from close t, all detected signal days vs all eval days
            ev = mk.evalm & np.isfinite(mk.ret20) & np.isfinite(mk.F["r20"]) & np.isfinite(mk.F["vol20"]) & np.isfinite(mk.F["r5"])
            ot, on = np.nonzero(ev)
            Do = mk.sig[s_][ot, on]
            lin = np.column_stack([mk.F[x][ot, on] for x in ["r20", "vol20", "r5"]])
            lin = (lin - lin.mean(0)) / lin.std(0)
            row["old"], row["old_t"] = coef_D(side * mk.ret20[ot, on] * 100, Do.astype(float), lin, ot)
            row["old_n"] = int(Do.sum())
            row["p"] = p_two(row["ex_t"])
            rows.append(row)
    return L, rows


def render(mk_name: str, rows: list[dict]) -> list[str]:
    L = []
    for side in (1, -1):
        L.append(f"\n  【{mk_name} · {SIDE_NAME[side]}】（R 为单位；超额 = 形态比同规则随机多出的每笔净 R，控制趋势 / 波动 / 几何，DK t）")
        L.append(f"  {'形态':<14s}{'信号':>7s}{'成交':>7s}{'成交率':>7s}{'胜率':>7s}{'随机胜率':>8s}{'毛R':>8s}{'净R':>8s}"
                 f"{'超额净R':>9s}{'t':>6s}{'胜率超额':>9s}{'不控收盘位':>10s}{'vs专属对照':>11s}{'t':>6s}")
        for r in [x for x in rows if x["side"] == side]:
            L.append(f"  {r['family'] + ' ' + r['label']:<14s}{r['n_sig']:>7,}{r['n_fill']:>7,}{r['fill_rate'] * 100:>6.0f}%"
                     f"{r['win'] * 100:>6.1f}%{r['rand_win'] * 100:>7.1f}%{fmt(r['gross']):>8s}{fmt(r['net']):>8s}"
                     f"{fmt(r['ex']):>9s}{fmt(r['ex_t'], 1):>6s}{fmt(r['ex_win'], pct=True):>9s}{fmt(r['ex_nok']):>10s}"
                     f"{fmt(r.get('vs_ctrl', np.nan)):>11s}{fmt(r.get('vs_ctrl_t', np.nan), 1):>6s}")
        L.append(f"  {'诊断':<14s}{'乐观超额':>9s}{'2R目标':>9s}{'开盘进场':>9s}{'T+1':>8s}{'毛R超额':>9s}"
                 f"{'外包线(形态/随机)':>18s}{'到期占比':>9s}{'旧裁判(20天%)':>14s}{'t':>6s}"
                 f"{'绝对胜率 保守~乐观':>20s}{'绝对净R 保守~乐观':>22s}")
        for r in [x for x in rows if x["side"] == side]:
            L.append(f"  {r['family']:<14s}{fmt(r.get('ex_opt', np.nan)):>9s}{fmt(r.get('ex_k2', np.nan)):>9s}"
                     f"{fmt(r.get('ex_open', np.nan)):>9s}{fmt(r.get('ex_t1', np.nan)):>8s}{fmt(r['ex_gross']):>9s}"
                     f"{r['outside'] * 100:>9.1f}%/{r['rand_outside'] * 100:>4.1f}%{r['time_share'] * 100:>8.1f}%"
                     f"{fmt(r['old'], 2):>14s}{fmt(r['old_t'], 1):>6s}"
                     f"{r['win'] * 100:>11.1f}%~{r['win_opt'] * 100:.1f}%{fmt(r['net']):>13s}~{fmt(r['net_opt'])}")
        per_lab = "2016…2024 逐年" if mk_name == "SW" else "1984-92/93-01/02-10/11-19"
        L.append(f"  稳定性（{per_lab} 的超额净 R 方向；前后两段）" + ("；SPX 2010–2019 计跳空 vs 不计跳空" if mk_name == "SPX" else ""))
        for r in [x for x in rows if x["side"] == side]:
            per = "".join("+" if (np.isfinite(v) and v > 0) else ("−" if np.isfinite(v) else "·") for v in r["per"])
            npos = sum(1 for v in r["per"] if np.isfinite(v) and v > 0)
            nvalid = sum(1 for v in r["per"] if np.isfinite(v))
            hv = " / ".join(fmt(v) for v in r["halves"])
            extra = f"   | 2010–19：计跳空 {fmt(r['gap_on'])}  不计跳空 {fmt(r['gap_off'])}" if mk_name == "SPX" else ""
            L.append(f"  {r['family']:<6s} {per:<10s} 同正 {npos}/{nvalid}   两段 {hv}{extra}")
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
    L = ["=" * 120, "Study 09 · 第 1 步：交易账本重考核心买卖点（SPX + 31 个申万一级行业，开发段）", "=" * 120,
         "主规格：t+1 越过信号 K 线进场 / 另一侧止损 / 1R / 20 天到期 / 同根双碰先算止损 / 扣成本。超额 = 比同规则随机入场多出的每笔净 R。",
         "SPX 只作跨市场方向核验（不进 FDR）；A 股行业的 BH-FDR 在 09b 与个股合并（共 20 个）。"]
    all_rows = []
    for mk_name, (df, ev0) in raw.items():
        log(f"build {mk_name}")
        mk = IndexMarket(mk_name, df, ev0)
        Lm, rows = analyze_market(mk)
        L += ["", "#" * 120, f"# 市场：{mk_name}（{mk.N} 个标的，评估起点 {ev0.date()}；来回成本 {mk.cost * 100:.2f}%）", "#" * 120]
        L += Lm
        L += render(mk_name, rows)
        all_rows += rows
    sw = [r for r in all_rows if r["market"] == "SW"]
    q = b8.bh_fdr([r["p"] if np.isfinite(r["p"]) else 1.0 for r in sw])
    for r, qq in zip(sw, q):
        r["q_sw_only"] = qq
    L += ["", "[行业 10 个主检验的临时 BH-FDR（仅行业内部，正式结论待 09b 与个股合并）]"]
    for r in sw:
        L.append(f"  {r['family']} {SIDE_NAME[r['side']]:<10s} 超额 {fmt(r['ex'])}R  t={fmt(r['ex_t'], 2)}  q={r['q_sw_only']:.3f}")
    txt = "\n".join(L)
    REPORT_PATH.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    flat = []
    for r in all_rows:
        d = {k: v for k, v in r.items() if k not in ("per", "halves")}
        d["per"] = ";".join(f"{v:+.3f}" if np.isfinite(v) else "nan" for v in r["per"])
        d["halves"] = ";".join(f"{v:+.3f}" if np.isfinite(v) else "nan" for v in r["halves"])
        flat.append(d)
    pd.DataFrame(flat).to_csv(SUMMARY_PATH, index=False)
    log(f"done ({time.time() - t0:.0f}s) -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
