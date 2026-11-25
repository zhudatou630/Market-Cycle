#!/usr/bin/env python3
"""
Market Cycle V2 — Study 10b（第三阶段专题 2 · 交易区间：全 A 个股 + A 股 8 个主检验收口）

预注册见 decisions.md「10-预注册」。区间识别、事件定义、限价单分支全部调用 studies/10 与 studies/09（已在指数上自检）。
个股主规格（开跑前写死）：
  - 面板 = output/08_stock_panel_dev.parquet（≤ 2024-10-31）+ 只读补读涨跌停价与复权因子（复用 09b.load_panel）；
    候选信号 K 线 = 09b 缓存里的同一批行（入池、2016 起、计划风险 ≥ 0.3 ATR、控制变量齐全、截尾 31 天）。
  - Q1（s=+1）：区间下三分之一 → t 日最低价挂买入限价单（个股 1 跳 = 0.01 元原始价，取 t+1 日复权跳价），T+1 主规则
    （成交当天跌破止损 → 次日开盘卖），变体当天不算；空头镜像 = 上三分之一、最高价挂卖单（持有者卖老仓，无 T+1）。
  - Q2（s=−1）：区间里的 20 日首次收盘新高 / 新低，止损单追（直接复用 09b 缓存账本）。
  - 双胞胎：每日截面回归只用随机单拟合（剔除本题事件与对照），同日市值五档 + L2 控制 + 申万一级行业 + 几何五分位
    （切点取本侧成交池）；超额 = 事件单残差均值（DK lag 30，每天每组最多 30 只，当天成交 < 100 只跳过）。
  - 专属对照 = 事件与对照的残差差（DK）。过关门槛见 decisions.md「10-预注册」。

用法：
  python studies/10b_trading_range_stocks.py --step0   # 建缓存 + 样本量（不看任何结果）
  python studies/10b_trading_range_stocks.py           # 第 2 步完整结果 + A 股 8 个主检验
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
CACHE_PATH = OUTPUT_DIR / "10b_cache.npz"
STEP0_PATH = OUTPUT_DIR / "10b_step0_report.txt"
REPORT_PATH = OUTPUT_DIR / "10b_trading_range_stocks_report.txt"
SUMMARY_PATH = OUTPUT_DIR / "10b_trading_range_stocks_summary.csv"
SUMMARY_10 = OUTPUT_DIR / "10_trading_range_index_summary.csv"


def _load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "studies" / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


s09b = _load_mod("s09b", "09b_trade_ledger_stocks.py")
s10 = _load_mod("s10", "10_trading_range_index.py")
s09 = s09b.s09
b8 = s09b.b8
log = b8.log

EVAL_START = s09b.EVAL_START
MIN_DAY = s09b.MIN_DAY
STOCK_BUFFER = s09b.STOCK_BUFFER
L2_COLS = s09b.L2_COLS
QUESTIONS = s10.QUESTIONS
K_TARGET, K_STOP, K_TIME, K_FORCED = s09.K_TARGET, s09.K_STOP, s09.K_TIME, s09.K_FORCED
FEAT_KEYS = ["q1_ev", "q1_ctrl", "q1_mid", "bot3w", "q2_ev", "q2_ctrl", "q2_tight", "brk",
             "state", "pos", "nr", "cross", "back10", "mid_real", "ret1", "r5v", "c1_excl", "hatr"]
SIDE_NAME = s10.SIDE_NAME


# ---------------------------------------------------------------------------
# 1. Build cache: features (both sides) + limit-order ledgers on the 09b candidate rows
# ---------------------------------------------------------------------------
def build_cache() -> dict:
    t0 = time.time()
    c09 = np.load(s09b.CACHE_PATH, allow_pickle=False)
    rows = c09["rows"]
    log("load panel + read-only limits / adj factors (09b.load_panel)")
    pn, info = s09b.load_panel()
    T, N = info["T"], info["N"]
    assert eval(str(c09["info"][0]))["rows"] == len(pn), "panel changed since the 09b cache"
    H, L, C = (pn[c].to_numpy(float) for c in ("a_high", "a_low", "a_close"))
    td = pn["trade_date"].to_numpy()
    em = td >= "20160101"
    cid = pn["code_id"].to_numpy()
    st = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    en = np.r_[st[1:], len(pn)]
    feats = {side: {} for side in (1, -1)}
    bad_mirror = 0
    log(f"features for {len(st):,} stocks")
    for s, e in zip(st.tolist(), en.tolist()):
        if e - s < 30:
            continue
        d = {side: s10.asset_side(H[s:e], L[s:e], C[s:e], em[s:e], side) for side in (1, -1)}
        a, b = d[1]["state"], d[-1]["state"]
        if not (np.array_equal(a == s10.ST_TR, b == s10.ST_TR) and np.array_equal(a == s10.ST_UP, b == s10.ST_DN)
                and np.array_equal(a == s10.ST_DN, b == s10.ST_UP)):
            bad_mirror += 1
        for side in (1, -1):
            for k in FEAT_KEYS:
                if k not in feats[side]:
                    feats[side][k] = np.zeros(len(pn), bool) if k in s10.BOOL_KEYS else np.full(len(pn), np.nan)
                feats[side][k][s:e] = d[side][k]
    info["bad_mirror_10"] = bad_mirror
    # brk_20 cross-check vs 08b detect_events on a random sample of stocks
    rng = np.random.default_rng(10)
    pick = rng.choice(len(st), size=300, replace=False)
    dates = pd.to_datetime(pn["trade_date"], format="%Y%m%d").to_numpy()
    bad_b, tot_b = 0, 0
    for i in pick.tolist():
        s, e = st[i], en[i]
        if e - s < 30:
            continue
        ev, _ = b8.detect_events(H[s:e], L[s:e], C[s:e], dates[s:e], EVAL_START)
        ref = set(ev["brk_20"].tolist())
        tot_b += len(ref)
        bad_b += ref != set(np.flatnonzero(feats[1]["brk"][s:e]).tolist())
    info["brk_check"] = (300, tot_b, bad_b)
    log(f"features done ({time.time() - t0:.0f}s); limit ledgers")
    P = s09b.wide_market(pn, T, N)
    tt, nn = pn["mi"].to_numpy()[rows], cid[rows]
    cost = np.where(td[rows] < s09b.STAMP_CUT, s09b.COST_PRE, s09b.COST_POST)
    cache: dict[str, np.ndarray] = {"rows": rows}
    specs = {1: {"main": dict(t1="forced"), "opt": dict(t1="forced", both="opt"), "t1i": dict(t1="ignore")},
             -1: {"main": {}, "opt": dict(both="opt")}}
    for side, sp in specs.items():
        for v, kw in sp.items():
            r = s09.run_ledger(P, tt, nn, side, entry="limit", cost=cost, buffer=STOCK_BUFFER, **kw)
            pre = f"{'L' if side > 0 else 'S'}_{v}_"
            cache[pre + "filled"] = r["filled"]
            cache[pre + "net"] = r["net"].astype(np.float32)
            cache[pre + "win"] = r["win"]
            if v == "main":
                for k in ["why", "kind", "gstop", "frozen"]:
                    cache[pre + k] = r[k]
                cache[pre + "hold"] = r["hold"].astype(np.int16)
                cache[pre + "gross"] = r["gross"].astype(np.float32)
            log(f"limit ledger side={side:+d} {v}: filled {r['filled'].mean() * 100:.1f}% ({time.time() - t0:.0f}s)")
        # middle-target diagnostic on Q1 event + control rows
        fs = feats[side]
        sel = np.flatnonzero(fs["q1_ev"][rows] | fs["q1_ctrl"][rows])
        kw = dict(t1="forced") if side > 0 else {}
        r = s09.run_ledger(P, tt[sel], nn[sel], side, entry="limit", cost=cost[sel], buffer=STOCK_BUFFER,
                           tgt_px=fs["mid_real"][rows][sel], **kw)
        p = "L" if side > 0 else "S"
        cache[f"{p}_mid_sel"], cache[f"{p}_mid_filled"], cache[f"{p}_mid_net"] = sel, r["filled"], r["net"].astype(np.float32)
    del P
    for side in (1, -1):
        p = "L" if side > 0 else "S"
        for k in FEAT_KEYS:
            v = feats[side][k][rows]
            cache[f"f{p}_{k}"] = v if v.dtype == bool else v.astype(np.float32)
    # market equal-weight daily return (for the skipped-day check) on the market calendar
    pc = pn.groupby("code_id")["a_close"].shift(1).to_numpy(float)
    dr = pd.Series(C / pc - 1.0).where(pn["eligible"].to_numpy(bool))
    cache["mkt_ret"] = dr.groupby(pn["mi"].to_numpy()).mean().reindex(range(T)).to_numpy()
    cache["info10"] = np.array([repr(info)])
    np.savez(CACHE_PATH, **cache)
    log(f"cache saved ({time.time() - t0:.0f}s)")
    return dict(np.load(CACHE_PATH, allow_pickle=False))


def load_cache() -> dict:
    return dict(np.load(CACHE_PATH, allow_pickle=False)) if CACHE_PATH.exists() else build_cache()


# ---------------------------------------------------------------------------
# 2. Step 0: counts only
# ---------------------------------------------------------------------------
def step0() -> list[str]:
    c = load_cache()
    c09 = np.load(s09b.CACHE_PATH, allow_pickle=False)
    info = eval(str(c["info10"][0]))
    rows = c["rows"]
    pn = pd.read_parquet(s09b.PANEL_PATH, columns=["code_id", "trade_date", "mi"]).sort_values(["code_id", "trade_date"])
    mi = pn["mi"].to_numpy()[rows]
    td = pn["trade_date"].to_numpy()[rows]
    year = np.array([int(x[:4]) for x in td])
    L = ["=" * 110, "Study 10b · 第 0 步：个股样本量与特征分布（不看任何账本结果）", "=" * 110,
         f"面板 {info['rows']:,} 行；市场日 {info['T']:,}；股票 {info['N']:,}；候选信号 K 线（与 09b 同一批）{len(rows):,}",
         f"镜像自检（区间↔区间、上涨↔下跌）不一致股票 {info['bad_mirror_10']}；"
         f"向量化首次收盘新高 vs 08b detect_events（随机 {info['brk_check'][0]} 只股票、{info['brk_check'][1]:,} 条）不一致股票 {info['brk_check'][2]}"]
    assert info["bad_mirror_10"] == 0 and info["brk_check"][2] == 0
    st = c["fL_state"]
    L.append(f"状态占比（候选信号 K 线）：区间 {np.mean(st == s10.ST_TR) * 100:.1f}%  上涨趋势 {np.mean(st == s10.ST_UP) * 100:.1f}%  "
             f"下跌趋势 {np.mean(st == s10.ST_DN) * 100:.1f}%  不贴标签 {np.mean(st == s10.ST_NONE) * 100:.1f}%；"
             f"C1 排除（净位移 ≥ 2/3 且翻转 ≥ 3）{np.mean(c['fL_c1_excl']) * 100:.1f}%")
    L.append("  逐年 区间 / 上 / 下：" + " | ".join(
        f"{y} {np.mean(st[year == y] == 1) * 100:.0f}/{np.mean(st[year == y] == 2) * 100:.0f}/{np.mean(st[year == y] == 3) * 100:.0f}%"
        for y in range(2016, 2025)))
    ha = c["fL_hatr"][np.isfinite(c["fL_hatr"])]
    L.append(f"箱体高度 / ATR 分位（10/25/50/75/90%）：{' / '.join(f'{x:.2f}' for x in np.quantile(ha, [.1, .25, .5, .75, .9]))}；"
             f"< 2 ATR 占 {np.mean(ha < s10.MIN_HGT_ATR) * 100:.1f}%")
    T = info["T"]
    mret = c["mkt_ret"]
    for q, side, led, ek, ck, s, lab in QUESTIONS:
        p = "L" if side > 0 else "S"
        E, C_ = c[f"f{p}_{ek}"], c[f"f{p}_{ck}"] & ~c[f"f{p}_{ek}"]
        F = c[f"{p}_main_filled"] if led == "L" else c09[f"{p}_main_filled"]
        line = (f"  {q} {lab:<12s} 事件 {int(E.sum()):>8,} → 成交 {int((E & F).sum()):>7,} | 对照 {int(C_.sum()):>8,} → 成交 {int((C_ & F).sum()):>7,}"
                f" | 随机池成交率 {F.mean() * 100:.1f}%")
        if led == "L":
            g = c[f"{p}_main_gstop"]
            line += f" | 开盘越过止损：事件 {g[E & F].mean() * 100:.2f}% 随机 {g[F].mean() * 100:.2f}%"
        L.append(line)
        cnt = np.bincount(mi[F & ~E & ~C_], minlength=T)
        days = np.unique(mi)
        cd = cnt[days]
        skip = days[cd < MIN_DAY]
        keep = days[cd >= MIN_DAY]
        ent = lambda d: mret[np.minimum(d + 1, T - 1)]
        L.append(f"     随机池每个信号日的成交数 分位（5/25/50/75/95%）：{' / '.join(str(int(x)) for x in np.quantile(cd, [.05, .25, .5, .75, .95]))}；"
                 f"不足 {MIN_DAY} 只而跳过 {len(skip)} 天（占 {len(skip) / len(days) * 100:.1f}%）；"
                 f"进场日大盘等权涨跌 跳过日 {np.nanmean(ent(skip)) * 100 if len(skip) else np.nan:+.2f}% / 其余 {np.nanmean(ent(keep)) * 100:+.2f}%")
        if q == "Q1":
            r1, r5 = c[f"f{p}_ret1"], c[f"f{p}_r5v"]
            L.append(f"     事前特征（本侧视角）：第 t 天涨跌均值 事件 {np.nanmean(r1[E]) * 100:+.2f}% 对照 {np.nanmean(r1[C_]) * 100:+.2f}%；"
                     f"近 5 日 事件 {np.nanmean(r5[E]) * 100:+.2f}% 对照 {np.nanmean(r5[C_]) * 100:+.2f}%；中间三分之一（诊断）{int(c[f'f{p}_q1_mid'].sum()):,}")
            ms = c[f"{p}_mid_filled"]
            L.append(f"     中轴目标诊断：事件 + 对照 {len(c[f'{p}_mid_sel']):,} 行，成交 {int(ms.sum()):,}")
        else:
            L.append(f"     太窄箱体的突破（诊断）{int(c[f'f{p}_q2_tight'].sum()):,}；全部首次突破 {int(c[f'f{p}_brk'].sum()):,}")
        L.append("     逐年事件成交：" + " ".join(f"{y}:{int((E & F & (year == y)).sum())}" for y in range(2016, 2025)))
    return L


# ---------------------------------------------------------------------------
# 3. Main analysis
# ---------------------------------------------------------------------------
def qdum(x, ref):
    return s09b.qdum(x, ref)


def main_analysis() -> None:
    t0 = time.time()
    c = load_cache()
    c09 = dict(np.load(s09b.CACHE_PATH, allow_pickle=False))
    rows = c["rows"]
    assert np.array_equal(rows, c09["rows"])
    M = len(rows)
    pn = pd.read_parquet(s09b.PANEL_PATH, columns=["code_id", "trade_date", "mi", "tier", "l1", "lmv", "crash_20"]
                         + [x for x in L2_COLS if x != "lmv"])
    pn = pn.sort_values(["code_id", "trade_date"]).reset_index(drop=True)
    mi = pn["mi"].to_numpy()[rows]
    td = pn["trade_date"].to_numpy()[rows]
    year = np.array([int(x[:4]) for x in td])
    tier = pn["tier"].to_numpy()[rows]
    l1 = pd.factorize(pn["l1"].to_numpy()[rows])[0]
    Z = pn[L2_COLS].to_numpy(float)[rows]
    lmv = pn["lmv"].to_numpy(float)[rows]
    szq = np.ceil((pd.Series(lmv).groupby(mi).rank(pct=True, method="first").to_numpy() * 5).clip(1e-9, 5)).astype(int)
    crash = pn["crash_20"].to_numpy(float)[rows]
    del pn
    order = np.argsort(mi, kind="stable")
    geo = {x: c09[x][rows] if c09[x].shape[0] != M else c09[x] for x in ["g_atr", "g_px", "kcls"]}

    def Gside(F):
        return np.hstack([qdum(geo[x], geo[x][F]) for x in ["g_atr", "g_px", "kcls"]])

    def resid(Y, filled, fit, G, l0=False, P=None):
        Y = Y if Y.ndim == 2 else Y[:, None]
        o = order[filled[order]]
        Gx = G if P is None else np.hstack([G, P])
        r, sk = s09b.resid_day(mi[o], szq[o], None if l0 else Z[o], l1[o], None if Gx is None else Gx[o], Y[o], fit[o], MIN_DAY)
        out = np.full(Y.shape, np.nan)
        out[o] = r
        return out, sk

    def tw(res, idx):
        b, t, n = b8.vs_twins(res, idx, mi)
        return b, t

    def thin_mask(F, hold, mask):
        idx = np.flatnonzero(mask & F)
        keep = np.zeros(M, bool)
        nn_ = code[idx]
        o = np.lexsort((mi[idx], nn_))
        last_n, last_x = -1, -1
        for i in idx[o].tolist():
            n, t = code[i], mi[i]
            if n != last_n:
                last_n, last_x = n, -1
            if t + 1 > last_x:
                keep[i] = True
                last_x = t + 1 + hold[i]
        return keep

    pnc = pd.read_parquet(s09b.PANEL_PATH, columns=["code_id", "trade_date"]).sort_values(["code_id", "trade_date"])
    code = pnc["code_id"].to_numpy()[rows]
    del pnc
    out_rows = []
    L = ["=" * 120, "Study 10b · 第 2 步：交易区间（全 A 个股，开发段 2016-01 → 2024-10）", "=" * 120,
         f"候选信号 K 线 {M:,}（与 09b 同一批）；超额 = 事件单相对同日双胞胎随机单的每笔净 R（L3 + 几何，DK lag 30）；"
         "专属对照 = 区间 − 趋势（同位置 / 同类事件）的残差差。"]
    for q, side, led, ek, ck, s, lab in QUESTIONS:
        p = "L" if side > 0 else "S"
        src = c if led == "L" else c09
        E = c[f"f{p}_{ek}"]
        Cm = c[f"f{p}_{ck}"] & ~E
        F = src[f"{p}_main_filled"]
        fit = F & ~E & ~Cm
        G = Gside(F)
        variants = ["main", "opt"] + (["t1i"] if side > 0 else [])
        Y = np.column_stack([src[f"{p}_{v}_net"].astype(float) for v in variants]
                            + [src[f"{p}_main_win"].astype(float), src[f"{p}_main_gross"].astype(float)])
        log(f"{q} side={side:+d}: residualize")
        R_, sk = resid(Y, F, fit, G)
        res = {v: R_[:, j] for j, v in enumerate(variants)}
        res["win"], res["gross"] = R_[:, len(variants)], R_[:, len(variants) + 1]
        net = src[f"{p}_main_net"].astype(float)
        res["l0"] = resid(net, F, fit, None, l0=True)[0][:, 0]
        res["l0g"] = resid(net, F, fit, G, l0=True)[0][:, 0]
        res["nok"] = resid(net, F, fit, G[:, :8])[0][:, 0]
        if q == "Q1":
            Pd = np.nan_to_num(s10.pos_dummies(c[f"f{p}_pos"].astype(float)), nan=0.0)
            res["pos"] = resid(net, F, fit, G, P=Pd)[0][:, 0]
        Fc = F & np.isfinite(crash)
        res["crash"] = resid(crash * 100, Fc, fit & Fc, G)[0][:, 0]
        S = np.flatnonzero(E & F)
        Ci = np.flatnonzero(Cm & F)
        r = {"market": "STK", "q": q, "side": side, "s": s, "label": lab, "ledger": led,
             "n_sig": int(E.sum()), "n_fill": len(S), "n_ctrl": len(Ci), "fill_rate": float(F[E].mean()),
             "rand_fill_rate": float(F.mean()), "skipped": sk}
        win, win_o = src[f"{p}_main_win"], src[f"{p}_opt_win"]
        net_o = src[f"{p}_opt_net"].astype(float)
        r.update({"win": float(win[S].mean()), "win_opt": float(win_o[S].mean()), "rwin": float(win[fit].mean()),
                  "rwin_opt": float(win_o[fit].mean()), "net": float(net[S].mean()), "net_opt": float(net_o[S].mean()),
                  "rnet": float(net[fit].mean()), "rnet_opt": float(net_o[fit].mean())})
        r["ex"], r["ex_t"] = tw(res["main"], S)
        r["ex_opt"], r["ex_opt_t"] = tw(res["opt"], S)
        if side > 0:
            r["ex_t1i"], _ = tw(res["t1i"], S)
        r["ex_win"], _ = tw(res["win"], S)
        r["ex_l0"], r["ex_l0_t"] = tw(res["l0"], S)
        r["ex_l0g"], r["ex_l0g_t"] = tw(res["l0g"], S)
        r["ex_nok"], _ = tw(res["nok"], S)
        if q == "Q1":
            r["ex_pos"], r["ex_pos_t"] = tw(res["pos"], S)
            r["ctrl_pos"], r["ctrl_pos_t"] = b8.vs_ctrl(res["pos"], S, Ci, mi)
        r["ex_crash"], r["ex_crash_t"] = tw(res["crash"], S)
        r["ctrl"], r["ctrl_t"] = b8.vs_ctrl(res["main"], S, Ci, mi)
        r["ctrl_l0"], _ = b8.vs_ctrl(res["l0"], S, Ci, mi)
        r["per"] = [tw(res["main"], S[year[S] == y])[0] for y in range(2016, 2025)]
        r["halves"] = [tw(res["main"], S[year[S] <= 2020])[0], tw(res["main"], S[year[S] >= 2021])[0]]
        r["tiers"] = [tw(res["main"], S[tier[S] == k])[0] for k in (1, 2, 3, 4)]
        r["ctrl_tiers"] = [b8.vs_ctrl(res["main"], S[tier[S] == k], Ci[tier[Ci] == k], mi)[0] for k in (1, 2, 3)]
        r["p"] = b8.p_two(r["ex_t"])
        r["time_share"] = float((src[f"{p}_main_kind"][S] == K_TIME).mean())
        # non-overlapping version
        hold = src[f"{p}_main_hold"].astype(np.int64)
        keep = thin_mask(F, hold, E) | thin_mask(F, hold, Cm) | thin_mask(F, hold, fit)
        Rt, _ = resid(net, F & keep, fit & keep, G)
        r["ex_thin"], r["ex_thin_t"] = tw(Rt[:, 0], np.flatnonzero(E & F & keep))
        if led == "L":
            gs = c[f"{p}_main_gstop"]
            r["gstop"], r["rgstop"] = float(gs[S].mean()), float(gs[fit].mean())
            Rc, _ = resid(net, F & ~gs, fit & ~gs, G)
            r["ex_cancel"], _ = tw(Rc[:, 0], np.flatnonzero(E & F & ~gs))
        if q == "Q1":
            if side > 0:
                y = net
                w = b8.cap_weights(mi[S], np.zeros(len(S)))
                b, t = b8.wls_dk(y[S], np.ones((len(S), 1)), w, mi[S])
                r["abs_c"], r["abs_c_t"] = float(b[0]), float(t[0])
                b, _ = b8.wls_dk(net_o[S], np.ones((len(S), 1)), w, mi[S])  # same day-capped weights as the conservative mean
                r["abs_o"] = float(b[0])
                y2 = 2 * net - src[f"{p}_main_gross"].astype(float)
                b, t = b8.wls_dk(y2[S], np.ones((len(S), 1)), w, mi[S])
                r["abs_c2"], r["abs_c2_t"] = float(b[0]), float(t[0])
                r["label_trade"] = ("日线口径可交易" if (r["abs_c"] > 0 and r["abs_c_t"] >= 2)
                                    else ("不可交易" if r["abs_o"] <= 0 else "日线判断不了"))
            # stop-entry version and t+1 open entry (09b cache), same events
            for tag, pre in [("stop", f"{p}_main"), ("open", f"{p}_open")]:
                Fx = c09[f"{pre}_filled"]
                Rx, _ = resid(c09[f"{pre}_net"].astype(float), Fx, Fx & ~E & ~Cm, Gside(Fx))
                r[f"{tag}_ex"], r[f"{tag}_ex_t"] = tw(Rx[:, 0], np.flatnonzero(E & Fx))
                r[f"{tag}_ctrl"], _ = b8.vs_ctrl(Rx[:, 0], np.flatnonzero(E & Fx), np.flatnonzero(Cm & Fx), mi)
            M3 = c[f"f{p}_q1_mid"]
            Rm, _ = resid(net, F, fit & ~M3, G)
            r["mid3_ex"], r["mid3_ex_t"] = tw(Rm[:, 0], np.flatnonzero(M3 & F))
            sel, fm, nm_ = c[f"{p}_mid_sel"], c[f"{p}_mid_filled"], c[f"{p}_mid_net"].astype(float)
            ym = np.full(M, np.nan)
            ym[sel[fm]] = nm_[fm]
            Xc = np.hstack([Z, G])
            r["midtgt_ctrl"], r["midtgt_ctrl_t"] = b8.vs_ctrl(ym, np.flatnonzero(E & np.isfinite(ym)), np.flatnonzero(Cm & np.isfinite(ym)), mi, Xc)
            r["midtgt_net_ev"] = float(np.nanmean(ym[E]))
            # dose-response among wide-box bottom-third rows
            B3 = c[f"f{p}_bot3w"]
            nr, cr = c[f"f{p}_nr"].astype(float), c[f"f{p}_cross"].astype(float)
            Rd, _ = resid(net, F, fit & ~B3, G)
            bins_nr = [B3 & (nr <= s10.FLAT + 1e-6), B3 & (nr > s10.FLAT + 1e-6) & (nr < s10.TREND - 1e-6), B3 & (nr >= s10.TREND - 1e-6)]
            bins_cr = [B3 & (cr <= 1), B3 & (cr == 2), B3 & (cr == 3), B3 & (cr >= 4)]
            for nm2, bins in [("dose_nr", bins_nr), ("dose_cr", bins_cr)]:
                vals = [tw(Rd[:, 0], np.flatnonzero(bb & F)) for bb in bins]
                r[nm2] = [v[0] for v in vals]
                r[nm2 + "_t"] = [v[1] for v in vals]
                r[nm2 + "_n"] = [int((bb & F).sum()) for bb in bins]
        else:
            kind = src[f"{p}_main_kind"]
            failk = (kind == K_STOP) | (kind == K_FORCED)
            r["fail_ev"], r["fail_ctrl"], r["fail_rand"] = (float(failk[m].mean()) for m in (E & F, Cm & F, fit))
            r["fail_ev_opt"], r["fail_ctrl_opt"], r["fail_rand_opt"] = (float(1 - win_o[m].mean()) for m in (E & F, Cm & F, fit))
            bk = c[f"f{p}_back10"].astype(float)
            brk = c[f"f{p}_brk"]
            r["back_ev"], r["back_ctrl"], r["back_all"] = (float(np.nanmean(bk[m])) for m in (E, Cm, brk))
            # fade translation: opposite-side limit order at the breakout bar's extreme
            q_ = "S" if side > 0 else "L"
            Ff = c[f"{q_}_main_filled"]
            Efit = Ff & ~c[f"f{q_}_q1_ev"] & ~c[f"f{q_}_q1_ctrl"] & ~E
            Rf, _ = resid(c[f"{q_}_main_net"].astype(float), Ff, Efit, Gside(Ff))
            r["fade_ex"], r["fade_ex_t"] = tw(Rf[:, 0], np.flatnonzero(E & Ff))
            r["fade_n"] = int((E & Ff).sum())
            TB = c[f"f{p}_q2_tight"]
            Rt2, _ = resid(net, F, fit & ~TB, G)
            r["tight_ex"], r["tight_ex_t"] = tw(Rt2[:, 0], np.flatnonzero(TB & F))
            r["tight_n"] = int((TB & F).sum())
        out_rows.append(r)
        log(f"{q} side={side:+d} done ({time.time() - t0:.0f}s)")
    L += render_stock(out_rows)
    # descriptive: random-pool simple return per trade (net of cost, %), limit fades vs stop follows
    cost = np.where(td < s09b.STAMP_CUT, s09b.COST_PRE, s09b.COST_POST)
    L.append("\n  【个股 · 限价单 vs 止损单的随机池（描述，不是检验）】每笔扣成本简单收益率（保守~乐观），成交率")
    for p, nm in [("L", "买"), ("S", "卖")]:
        parts = []
        for src, lab in [(c, "限价反向"), (c09, "止损顺势")]:
            F = src[f"{p}_main_filled"]
            g, n0, no = (src[f"{p}_{k}"].astype(float)[F] for k in ("main_gross", "main_net", "opt_net"))
            fac = cost[F] / (g - n0)  # = planned-or-actual R / fill price
            parts.append(f"{lab}{nm} {np.nanmean(n0 * fac) * 100:+.3f}%~{np.nanmean(no * fac) * 100:+.3f}%（成交率 {F.mean() * 100:.0f}%）")
        L.append("  " + " | ".join(parts))
    L += final_verdict(out_rows)
    txt = "\n".join(L)
    REPORT_PATH.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    flat = []
    for r in out_rows:
        d = {k: v for k, v in r.items() if not isinstance(v, list)}
        for k in ["per", "halves", "tiers", "ctrl_tiers"]:
            d[k] = ";".join(f"{v:+.4f}" if np.isfinite(v) else "nan" for v in r[k])
        flat.append(d)
    pd.DataFrame(flat).to_csv(SUMMARY_PATH, index=False)
    log(f"done ({time.time() - t0:.0f}s) -> {REPORT_PATH}")


# ---------------------------------------------------------------------------
# 4. Report + A-share 8 primary tests
# ---------------------------------------------------------------------------
def render_stock(rows: list[dict]) -> list[str]:
    f = s09.fmt
    L = []
    for q in ("Q1", "Q2"):
        L.append(f"\n  【个股 · {q}】Brooks 方向 s = {'+1' if q == 'Q1' else '−1'}")
        L.append(f"  {'方向':<16s}{'信号':>9s}{'成交':>8s}{'对照成交':>9s}{'成交率':>7s}{'胜率 保守~乐观':>16s}{'随机胜率':>9s}"
                 f"{'净R 保守~乐观':>18s}{'随机净R':>9s}{'超额':>8s}{'t':>6s}{'乐观超额':>9s}{'区间−趋势':>10s}{'t':>6s}")
        for r in [x for x in rows if x["q"] == q]:
            L.append(f"  {r['label']:<14s}{r['n_sig']:>9,}{r['n_fill']:>8,}{r['n_ctrl']:>9,}{r['fill_rate'] * 100:>6.0f}%"
                     f"{r['win'] * 100:>9.1f}%~{r['win_opt'] * 100:.1f}%{r['rwin'] * 100:>8.1f}%"
                     f"{f(r['net']):>10s}~{f(r['net_opt'])}{f(r['rnet']):>9s}{f(r['ex']):>8s}{f(r['ex_t'], 1):>6s}"
                     f"{f(r['ex_opt']):>9s}{f(r['ctrl']):>10s}{f(r['ctrl_t'], 1):>6s}")
        L.append("  拆解（超额）：只扣同日同市值 → +几何 → +趋势 / 波动 / 热度 / 行业（主规格）" + (" → +箱体位置" if q == "Q1" else "")
                 + "；不控收盘位置；不重叠版本；T+1 忽略变体；20 天踩雷差（双胞胎）")
        for r in [x for x in rows if x["q"] == q]:
            ex = f" → {f(r['ex_pos'])}（t {f(r['ex_pos_t'], 1)}；区间−趋势 {f(r['ctrl_pos'])}，t {f(r['ctrl_pos_t'], 1)}）" if q == "Q1" else ""
            t1 = f"；T+1 忽略 {f(r['ex_t1i'])}" if "ex_t1i" in r else ""
            L.append(f"  {r['label']:<14s} {f(r['ex_l0'])}（t {f(r['ex_l0_t'], 1)}）→ {f(r['ex_l0g'])} → {f(r['ex'])}{ex}；不控收盘位置 {f(r['ex_nok'])}；"
                     f"不重叠 {f(r['ex_thin'])}（t {f(r['ex_thin_t'], 1)}）{t1}；踩雷差 {f(r['ex_crash'], 2)}pp（t {f(r['ex_crash_t'], 1)}）")
        L.append("  稳定性：2016…2024 逐年方向 | 前后两段 | 市值档 沪深300 / 中证500 / 中证1000 / 其余小微 | 专属对照分档（前三档）| 截面跳过天数")
        for r in [x for x in rows if x["q"] == q]:
            sg = "".join("+" if (np.isfinite(v) and v > 0) else ("−" if np.isfinite(v) else "·") for v in r["per"])
            npos = sum(1 for v in r["per"] if np.isfinite(v) and r["s"] * v > 0)
            L.append(f"  {r['label']:<14s} {sg:<10s} Brooks 方向 {npos}/9 | {' / '.join(f(v) for v in r['halves'])} | "
                     f"{' / '.join(f(v) for v in r['tiers'])} | {' / '.join(f(v) for v in r['ctrl_tiers'])} | {r['skipped']}")
        if q == "Q1":
            L.append("  诊断：止损单版本（超额 / 区间−趋势）| t+1 开盘进场（超额 / 区间−趋势）| 中间三分之一限价 | 中轴目标（区间−趋势，事件净R）"
                     "| 开盘越过止损（形态 / 随机）、视为撤单后超额 | 到期占比")
            for r in [x for x in rows if x["q"] == q]:
                L.append(f"  {r['label']:<14s} {f(r['stop_ex'])}（t {f(r['stop_ex_t'], 1)}）/ {f(r['stop_ctrl'])} | "
                         f"{f(r['open_ex'])}（t {f(r['open_ex_t'], 1)}）/ {f(r['open_ctrl'])} | {f(r['mid3_ex'])}（t {f(r['mid3_ex_t'], 1)}）| "
                         f"{f(r['midtgt_ctrl'])}（t {f(r['midtgt_ctrl_t'], 1)}），{f(r['midtgt_net_ev'])} | "
                         f"{r['gstop'] * 100:.2f}% / {r['rgstop'] * 100:.2f}%，{f(r['ex_cancel'])} | {r['time_share'] * 100:.1f}%")
            L.append("  剂量-反应（箱体 ≥ 2 ATR、收盘在下三分之一的全部日子）：净位移/高度 [0,1/3] (1/3,2/3) [2/3,1] | 翻转 0–1 / 2 / 3 / 4+")
            for r in [x for x in rows if x["q"] == q]:
                a = " ".join(f"{f(v)}({f(t, 1)})" for v, t in zip(r["dose_nr"], r["dose_nr_t"]))
                b = " ".join(f"{f(v)}({f(t, 1)})" for v, t in zip(r["dose_cr"], r["dose_cr_t"]))
                L.append(f"  {r['label']:<14s} {a} | {b}")
            for r in [x for x in rows if x["q"] == q and x["side"] > 0]:
                L.append(f"  可交易标签（多头，绝对每笔净 R，每天最多计 30 只的加权均值）：保守 {f(r['abs_c'])}（t {f(r['abs_c_t'], 1)}），乐观 {f(r['abs_o'])}，"
                         f"2 倍成本 {f(r['abs_c2'])}（t {f(r['abs_c2_t'], 1)}）→ 【{r['label_trade']}】")
        else:
            L.append("  失败率：账本快速失败（先碰止损，保守 ~ 1−乐观胜率）区间突破 / 趋势突破 / 随机单 | 10 天内收盘跌回箱体：区间 / 趋势 / 全部首次新高")
            for r in [x for x in rows if x["q"] == q]:
                L.append(f"  {r['label']:<14s} {r['fail_ev'] * 100:.1f}~{r['fail_ev_opt'] * 100:.1f}% / {r['fail_ctrl'] * 100:.1f}~{r['fail_ctrl_opt'] * 100:.1f}% / "
                         f"{r['fail_rand'] * 100:.1f}~{r['fail_rand_opt'] * 100:.1f}% | {r['back_ev'] * 100:.1f}% / {r['back_ctrl'] * 100:.1f}% / {r['back_all'] * 100:.1f}%")
            L.append("  诊断：反着做（突破 K 线极值挂反向限价单，超额，成交）| 太窄箱体的突破（超额，成交）| 到期占比")
            for r in [x for x in rows if x["q"] == q]:
                L.append(f"  {r['label']:<14s} {f(r['fade_ex'])}（t {f(r['fade_ex_t'], 1)}，{r['fade_n']:,}）| "
                         f"{f(r['tight_ex'])}（t {f(r['tight_ex_t'], 1)}，{r['tight_n']:,}）| {r['time_share'] * 100:.1f}%")
    return L


def _yes(b):
    return "—" if b is None else ("✔" if b else "✘")


def verdict(r: dict, q: float) -> dict:
    s = int(r["s"])
    stock = r["market"] == "STK"
    per = [v for v in r["per"] if np.isfinite(v)]
    out = {
        "stat": s * r["ex_t"] >= 2 and q <= 0.05,
        "size": s * r["ex"] >= 0.10,
        "ctrl": np.isfinite(r.get("ctrl_t", np.nan)) and s * r["ctrl_t"] >= 2 and s * r["ctrl"] >= 0.05,
        "stab": sum(s * v > 0 for v in per) >= 6 and all(np.isfinite(h) and s * h > 0 for h in r["halves"]),
        "tier": (sum(s * v > 0 for v in r["tiers"][:3] if np.isfinite(v)) >= 2) if stock else None,
        "opt": np.isfinite(r.get("ex_opt", np.nan)) and s * r["ex_opt"] > 0,
        "t1": (np.isfinite(r.get("ex_t1i", np.nan)) and s * r["ex_t1i"] > 0) if (stock and int(r["side"]) > 0) else None,
        "crash": None,
    }
    if stock and int(r["side"]) < 0:
        out["crash"] = (r["ex_crash"] > 0) if r["q"] == "Q1" else (r["ex_crash"] <= 0)
    out["pass"] = all(v for v in out.values() if v is not None)
    return out


def final_verdict(stk_rows: list[dict]) -> list[str]:
    sw = pd.read_csv(SUMMARY_10)
    sw = sw[sw["market"] == "SW"]
    prim = []
    for _, x in sw.iterrows():
        d = x.to_dict()
        d["per"] = [float(v) for v in str(d["per"]).split(";")]
        d["halves"] = [float(v) for v in str(d["halves"]).split(";")]
        prim.append(d)
    prim += stk_rows
    q = b8.bh_fdr([r["p"] if np.isfinite(r["p"]) else 1.0 for r in prim])
    f = s09.fmt
    L = ["", "#" * 120, "# A 股 8 个主检验：BH-FDR 与预注册过关判定（行业来自 10，个股来自本脚本；一律乘 Brooks 方向 s）", "#" * 120,
         f"  {'市场':<5s}{'题':<4s}{'方向':<16s}{'s':>3s}{'超额':>8s}{'t':>7s}{'q':>7s}{'区间−趋势':>10s}{'t':>6s}  "
         f"{'①显著':>5s}{'②幅度':>5s}{'③对照':>5s}{'④稳定':>5s}{'⑤市值档':>6s}{'⑥乐观':>5s}{'⑦T+1':>5s}{'⑧踩雷':>5s}  结论"]
    for r, qq in zip(prim, q):
        v = verdict(r, qq)
        r["q8"], r["pass"] = qq, v["pass"]
        mk = "行业" if r["market"] == "SW" else "个股"
        rev = (r["s"] * r["ex_t"] <= -2) and qq <= 0.05
        L.append(f"  {mk:<5s}{r['q']:<4s}{r['label']:<14s}{int(r['s']):>+3d}{f(r['ex']):>8s}{f(r['ex_t'], 2):>7s}{qq:>7.3f}"
                 f"{f(r['ctrl']):>10s}{f(r['ctrl_t'], 1):>6s}  "
                 + "".join(f"{_yes(v[k]):>6s}" for k in ["stat", "size", "ctrl", "stab", "tier", "opt", "t1", "crash"])
                 + f"  {'过关 ✔' if v['pass'] else ('与 Brooks 反向显著 ✘' if rev else '不过关')}")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step0", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if args.step0:
        L = step0()
        txt = "\n".join(L)
        STEP0_PATH.write_text(txt + "\n", encoding="utf-8")
        print(txt)
        log(f"step0 done ({time.time() - t0:.0f}s) -> {STEP0_PATH}")
        return
    main_analysis()


if __name__ == "__main__":
    main()
