#!/usr/bin/env python3
"""
Market Cycle V2 — Study 11b（第三阶段专题 3 · 价位记忆：全 A 个股 + A 股 10 个主检验收口）

预注册见 decisions.md「11-预注册」。价位识别全部调用 studies/11（已在指数上自检）；账本直接取 09b 缓存（引擎不改）。
个股主规格（开跑前写死）：
  - 面板 = output/08_stock_panel_dev.parquet（≤ 2024-10-31）+ 09b.load_panel 的只读补读（涨跌停价、复权因子）。
  - A 族 ①②③ 与 C 族 ①②：信号 K 线 = 测试日，取 09b 缓存里同一行的止损单账本（多头 T+1 主规则，空头 = 持有者卖出）。
  - 双胞胎：每日截面回归，拟合只用非真价位事件的随机单（A / C / 锚点测试日一律剔除，普通价位不剔除），
    同日市值五档 + L2 控制 + 当天涨跌（同日标准化连续值）+ 申万一级行业 + 几何五分位（切点取本侧成交池）+ 当天涨跌同日五分位；
    超额 = 事件单残差均值（DK lag 30，每天每组最多 30 只，当天成交 < 100 只跳过）。专属对照 = 事件与普通价位的残差差。
  - ④ 踩雷：全部入池、次日可买的失守日（不经缓存与次日成交筛选），固定 −15% 踩雷（08 口径）与波动归一踩雷，
    控制 = L3 + 当天涨跌 + 几何（一字板收盘位置涨停记 1、跌停记 0）。
  - A 股 10 个主检验（行业 3 个来自 11 + 个股 7 个）BH-FDR；过关门槛见 decisions.md「11-预注册」。

用法：
  python studies/11b_level_memory_stocks.py --step0   # 建缓存 + 自检 + 样本量（不看任何结果）
  python studies/11b_level_memory_stocks.py           # 第 2 步完整结果 + A 股 10 个主检验
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
CACHE_PATH = OUTPUT_DIR / "11b_cache.npz"
STEP0_PATH = OUTPUT_DIR / "11b_step0_report.txt"
REPORT_PATH = OUTPUT_DIR / "11b_level_memory_stocks_report.txt"
SUMMARY_PATH = OUTPUT_DIR / "11b_level_memory_stocks_summary.csv"
SUMMARY_11 = OUTPUT_DIR / "11_level_memory_index_summary.csv"


def _load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "studies" / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


s11 = _load_mod("s11", "11_level_memory_index.py")
s09b = _load_mod("s09b", "09b_trade_ledger_stocks.py")
s09 = s09b.s09
s10 = s11.s10
b8 = s09b.b8
log = b8.log

EPS = 1e-12
MIN_DAY = s09b.MIN_DAY
L2_COLS = s09b.L2_COLS
KEYS_P = ["A_test", "A_hold", "A_fail", "A_clean", "A_near", "A_lvl", "A_dist", "A_age", "A_ntest", "A_ntouch",
          "G_hold", "G_fail", "G_bdn", "N_test", "N_hold", "N_fail", "N_clean",
          "C_test", "C_hold", "C_fail", "C_clean", "C_near", "C_lian", "C_yizi", "C_lvl", "C_dist", "C_age",
          "GC_hold", "GC_fail", "GC_bdn", "atr_v", "r5v", "rng5"]
KEYS_M = ["A_test", "A_hold", "A_fail", "A_clean", "A_near", "A_lvl", "A_dist", "A_ntest", "A_ntouch",
          "G_hold", "G_fail", "G_bdn", "r5v", "rng5"]
EXTRA = ["entry_ok", "tail_20", "ret1", "amount", "lim_pct"]

_G: dict = {}


def _worker(se):
    s, e = se
    H, L, C, UPC, DNC, EL, EM, AMT = (_G[k] for k in ("H", "L", "C", "UPC", "DNC", "EL", "EM", "AMT"))
    h, l, c = H[s:e], L[s:e], C[s:e]
    p = s11.asset_view(h, l, c, EM[s:e], 1, UPC[s:e], DNC[s:e], EL[s:e])
    m = s11.asset_view(h, l, c, EM[s:e], -1)
    out = {f"p_{k}": p[k] for k in KEYS_P}
    out.update({f"m_{k}": m[k] for k in KEYS_M})
    z1 = s09._shift(p["atr_v"], 1)
    for tag, d, key in [("p_A_cc", p, "A"), ("m_A_cc", m, "A"), ("p_C_cc", p, "C")]:
        ev = np.flatnonzero(d[f"{key}_test"])
        cc = np.full(e - s, np.nan)
        if len(ev):
            cc[ev] = s11.concentration(h, l, AMT[s:e], ev, d[f"{key}_lvl"][ev], z1[ev])
        out[tag] = cc
    return s, e, out


# ---------------------------------------------------------------------------
# 1. Build cache
# ---------------------------------------------------------------------------
def load_all() -> tuple[pd.DataFrame, dict]:
    pn, info = s09b.load_panel()
    ex = pd.read_parquet(s09b.PANEL_PATH, columns=["code_id", "trade_date"] + [x for x in EXTRA if x != "lim_pct"])
    ex = ex.sort_values(["code_id", "trade_date"]).reset_index(drop=True)
    assert np.array_equal(ex["code_id"].to_numpy(), pn["code_id"].to_numpy()) and np.array_equal(
        ex["trade_date"].to_numpy(), pn["trade_date"].to_numpy()), "panel alignment"
    for k in EXTRA:
        if k != "lim_pct":
            pn[k] = ex[k].to_numpy()
    return pn, info


def build_cache() -> dict:
    t0 = time.time()
    c09 = np.load(s09b.CACHE_PATH, allow_pickle=False)
    pn, info = load_all()
    assert eval(str(c09["info"][0]))["rows"] == len(pn), "panel changed since the 09b cache"
    H, L, C = (pn[c].to_numpy(float) for c in ("a_high", "a_low", "a_close"))
    cid = pn["code_id"].to_numpy()
    st = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    en = np.r_[st[1:], len(pn)]
    _G.update(H=H, L=L, C=C, UPC=pn["up_close"].to_numpy(bool), DNC=pn["dn_close"].to_numpy(bool),
              EL=pn["eligible"].to_numpy(bool), EM=pn["trade_date"].to_numpy() >= "20160101",
              AMT=pn["amount"].to_numpy(float))
    jobs = [(int(s), int(e)) for s, e in zip(st, en) if e - s >= 30]
    log(f"features for {len(jobs):,} stocks (8 workers)")
    feats: dict[str, np.ndarray] = {}
    with mp.get_context("fork").Pool(8) as pool:
        for k_, (s, e, out) in enumerate(pool.imap_unordered(_worker, jobs, chunksize=20)):
            for k, v in out.items():
                if k not in feats:
                    feats[k] = np.zeros(len(pn), bool) if v.dtype == bool else np.full(len(pn), np.nan, np.float32)
                feats[k][s:e] = v
            if k_ % 1000 == 0:
                log(f"  {k_:,} stocks ({time.time() - t0:.0f}s)")
    log(f"features done ({time.time() - t0:.0f}s); self-checks on samples")
    rng = np.random.default_rng(1111)
    pick = rng.choice(len(jobs), size=300, replace=False)
    dates = pd.to_datetime(pn["trade_date"], format="%Y%m%d").to_numpy()
    upc, dnc, el = _G["UPC"], _G["DNC"], _G["EL"]
    em = _G["EM"]
    bad_m = bad_b = tot_b = bad_p = bad_q = 0
    for i in pick.tolist():
        s, e = jobs[i]
        h, l, c = H[s:e], L[s:e], C[s:e]
        a = s11.asset_view(h, l, c, em[s:e], 1, upc[s:e], dnc[s:e], el[s:e])
        K = 2 * np.nanmax(h) + 7.0
        m = s11.asset_view(K - l, K - h, K - c, em[s:e], -1, dnc[s:e], upc[s:e], el[s:e])
        for k in ["A_test", "A_hold", "A_fail", "A_near", "G_hold", "G_fail", "G_bdn",
                  "C_test", "C_hold", "C_fail", "C_lian", "GC_hold", "GC_fail"]:
            bad_m += not np.array_equal(a[k], m[k])
        ev, _ = b8.detect_events(h, l, c, dates[s:e], s09b.EVAL_START)
        ref = set(ev["brk_20"].tolist())
        tot_b += len(ref)
        bad_b += ref != set(np.flatnonzero(s10.first_break(c, s09.base_feats(h, l, c)["atr20"], em[s:e])).tolist())
        if i % 6 == 0 and e - s > 100:
            full = s11.detect_A(*(np.round(v, s11.ROUND) for v in (h, l, c)), a["atr_v"], em[s:e])
            for tc in rng.choice(np.arange(80, e - s), size=8, replace=False).tolist():
                p = s11.asset_view(h[:tc + 1], l[:tc + 1], c[:tc + 1], em[s:s + tc + 1], 1, upc[s:s + tc + 1], dnc[s:s + tc + 1], el[s:s + tc + 1])
                for k in ["A_test", "A_hold", "A_lvl", "A_ntest", "A_near", "G_hold", "G_fail", "N_test", "N_hold", "C_test", "C_hold", "C_lvl"]:
                    x, y = p[k][tc], a[k][tc]
                    if not (x == y or (isinstance(x, float) and np.isnan(x) and np.isnan(y))):
                        bad_p += 1
                hq, lq, cq = (np.round(np.r_[v[:tc], v[tc - 1]], s11.ROUND) for v in (h, l, c))
                q = s11.detect_A(hq, lq, cq, s09.base_feats(hq, lq, cq)["atr20"], em[s:s + tc + 1])
                if not (q["A_narm"][tc] == full["A_narm"][tc] and (q["A_toparm"][tc] == full["A_toparm"][tc]
                        or (np.isnan(q["A_toparm"][tc]) and np.isnan(full["A_toparm"][tc])))):
                    bad_q += 1
    info["mirror"] = (300, bad_m)
    info["brk_check"] = (300, tot_b, bad_b)
    info["trunc"] = (bad_p, bad_q)
    # 09b cache consistency: re-run the ledger on 1,000 random event rows
    rows = c09["rows"]
    evrows = np.flatnonzero((feats["p_A_test"] | feats["m_A_test"] | feats["p_C_test"])[rows])
    smp = rows[rng.choice(evrows, size=1000, replace=False)]
    T, N = info["T"], info["N"]
    P = s09b.wide_market(pn, T, N)
    tt, nn = pn["mi"].to_numpy()[smp], cid[smp]
    cost = np.where(pn["trade_date"].to_numpy()[smp] < s09b.STAMP_CUT, s09b.COST_PRE, s09b.COST_POST)
    pos = np.searchsorted(rows, smp)
    bad_c = 0
    for side, pre, kw in [(1, "L_main", dict(t1="forced")), (-1, "S_main", {})]:
        r = s09.run_ledger(P, tt, nn, side, cost=cost, buffer=s09b.STOCK_BUFFER, **kw)
        f0 = c09[f"{pre}_filled"][pos]
        bad_c += int((r["filled"] != f0).sum())
        bad_c += int((np.abs(r["net"][f0] - c09[f"{pre}_net"][pos][f0]) > 1e-4).sum())
    info["ledger_check"] = (1000, bad_c)
    del P
    cache = {k: v for k, v in feats.items()}
    cache["info11"] = np.array([repr(info)])
    np.savez(CACHE_PATH, **cache)
    log(f"cache saved ({time.time() - t0:.0f}s)")
    return dict(np.load(CACHE_PATH, allow_pickle=False))


def load_cache() -> dict:
    return dict(np.load(CACHE_PATH, allow_pickle=False)) if CACHE_PATH.exists() else build_cache()


# ---------------------------------------------------------------------------
# 2. Step 0: counts only
# ---------------------------------------------------------------------------
PRIM = [  # (code, view prefix, ledger side, event, control, label)
    ("A1", "p", 1, "A_hold", "G_hold", "A① 双底守住买"),
    ("A2", "p", -1, "A_fail", "G_fail", "A② 双底失守卖"),
    ("A3", "m", -1, "A_hold", "G_hold", "A③ 两次冲高失败卖"),
    ("C1", "p", 1, "C_hold", "GC_hold", "C① 起涨价守住买"),
    ("C2", "p", -1, "C_fail", "GC_fail", "C② 起涨价失守卖"),
    ("N1", "p", 1, "N_hold", "G_hold", "锚点 突破位守住买"),
    ("N2", "p", -1, "N_fail", "G_fail", "锚点 突破位失守卖"),
]


def step0() -> list[str]:
    c = load_cache()
    c09 = np.load(s09b.CACHE_PATH, allow_pickle=False)
    info = eval(str(c["info11"][0]))
    rows = c09["rows"]
    pn = pd.read_parquet(s09b.PANEL_PATH, columns=["code_id", "trade_date", "mi", "tier", "eligible", "entry_ok", "crash_20",
                                                   "dn_close", "a_high", "a_low", "lim_pct", "ret1", "r5"])
    pn = pn.sort_values(["code_id", "trade_date"]).reset_index(drop=True)
    td = pn["trade_date"].to_numpy()
    year = np.array([int(x[:4]) for x in td])
    tier = pn["tier"].to_numpy()
    real = c["p_A_test"] | c["m_A_test"] | c["p_C_test"] | c["p_N_test"]
    L = ["=" * 110, "Study 11b · 第 0 步：个股自检 + 样本量（不看任何结果）", "=" * 110,
         f"面板 {info['rows']:,} 行；市场日 {info['T']:,}；股票 {info['N']:,}；09b 候选信号 K 线 {len(rows):,}",
         f"镜像自检（随机 {info['mirror'][0]} 只：支撑 ↔ 压力、涨停 ↔ 跌停）不一致 {info['mirror'][1]} 处；"
         f"锚点首次收盘新高 vs 08b detect_events（{info['brk_check'][1]:,} 条）不一致股票 {info['brk_check'][2]}；"
         f"截断重算不一致 {info['trunc'][0]} 处，t−1 探针不一致 {info['trunc'][1]} 处；"
         f"抽 {info['ledger_check'][0]} 个事件行重跑账本 vs 09b 缓存 不一致 {info['ledger_check'][1]} 处"]
    assert info["mirror"][1] == 0 and info["brk_check"][2] == 0 and info["trunc"] == (0, 0) and info["ledger_check"][1] == 0
    inrows = np.zeros(len(pn), bool)
    inrows[rows] = True
    FL, FS = np.zeros(len(pn), bool), np.zeros(len(pn), bool)
    FL[rows], FS[rows] = c09["L_main_filled"], c09["S_main_filled"]
    yizi = pn["a_high"].to_numpy() - pn["a_low"].to_numpy() <= 1e-9
    dnc = pn["dn_close"].to_numpy(bool)
    for pre, fam, nm in [("p", "A", "A 族支撑（双底）"), ("m", "A", "A 族压力（双顶）"), ("p", "C", "C 族涨停起涨价"), ("p", "N", "锚点 突破位")]:
        T_, Hh, Ff = c[f"{pre}_{fam}_test"], c[f"{pre}_{fam}_hold"], c[f"{pre}_{fam}_fail"]
        nt = int(T_.sum())
        L.append(f"\n  —— {nm}：测试 {nt:,}（守住 {int(Hh.sum()):,} / 失守 {int(Ff.sum()):,}，守住占 {Hh.sum() / max(nt, 1) * 100:.0f}%）；"
                 f"落在 09b 候选行（计划风险 ≥ 0.3 ATR 等）{int((T_ & inrows).sum()):,}，被剔除 {int((T_ & ~inrows).sum()):,}"
                 f"（其中一字板 {int((T_ & ~inrows & yizi).sum()):,}）；失守日一字跌停 {np.mean((yizi & dnc)[Ff]) * 100:.2f}%、"
                 f"收盘封跌停 {np.mean(dnc[Ff]) * 100:.1f}%")
        if f"{pre}_{fam}_clean" in c:
            L.append(f"     干净守住占守住 {c[f'{pre}_{fam}_clean'][Hh].mean() * 100:.0f}%" + (
                f"；同日碰到 ≥2 个价位 {np.mean(c[f'{pre}_A_ntouch'][T_] >= 2) * 100:.1f}%；测试次数 1/2/≥3 "
                + " / ".join(f"{np.mean(m) * 100:.0f}%" for m in [c[f'{pre}_A_ntest'][T_] == 1, c[f'{pre}_A_ntest'][T_] == 2, c[f'{pre}_A_ntest'][T_] >= 3])
                if fam == "A" else ""))
        if fam == "C":
            lp = pn["lim_pct"].to_numpy(float)
            L.append(f"     一字板起涨 {c['p_C_yizi'][T_].mean() * 100:.1f}%；连板后回落 {c['p_C_lian'][T_].mean() * 100:.1f}%；"
                     f"20% 板 {np.mean(lp[T_] >= 0.19) * 100:.1f}%；测试日距涨停日中位 {np.nanmedian(c['p_C_age'][T_]):.0f} 根")
        L.append("     逐年测试：" + " ".join(f"{y}:{int((T_ & (year == y)).sum())}" for y in range(2016, 2025))
                 + " | 市值档 沪深300/中证500/中证1000/其余：" + " / ".join(str(int((T_ & (tier == k)).sum())) for k in (1, 2, 3, 4)))
    L.append("\n  —— 普通价位（占 09b 候选行）：")
    for pre, gk, nm in [("p", "G", "A 族支撑侧"), ("m", "G", "A 族压力侧"), ("p", "GC", "C 族")]:
        L.append(f"     {nm}：普通守住 {c[f'{pre}_{gk}_hold'][rows].mean() * 100:.1f}% / 普通失守 {c[f'{pre}_{gk}_fail'][rows].mean() * 100:.1f}% / "
                 f"普通 20 日首次收盘新低 {c[f'{pre}_{gk}_bdn'][rows].mean() * 100:.2f}%")
    L.append(f"     真价位事件（任一族测试日，拟合池剔除）占候选行 {real[rows].mean() * 100:.1f}%")
    L.append("\n  —— 主检验与锚点：信号 → 成交（09b 缓存），对照成交，事前特征（本侧视角）事件 / 对照")
    r1, r5 = pn["ret1"].to_numpy(float), pn["r5"].to_numpy(float)
    c09k = c09["kcls"]
    mi = pn["mi"].to_numpy()
    for code, pre, ls, ek, ck, lab in PRIM:
        E = c[f"{pre}_{ek}"] & inrows
        Cc = c[f"{pre}_{ck}"] & inrows & ~real
        F = FL if ls > 0 else FS
        sg = 1 if pre == "p" else -1
        kk = c09k if sg > 0 else 1 - c09k
        L.append(f"  {lab:<14s} 信号 {int(E.sum()):>8,} → 成交 {int((E & F).sum()):>7,} | 普通价位对照 {int(Cc.sum()):>9,} → 成交 {int((Cc & F).sum()):>8,}"
                 f" | 当天涨跌 {np.nanmean(sg * r1[E]) * 100:+.2f}% / {np.nanmean(sg * r1[Cc]) * 100:+.2f}%；近 5 日 {np.nanmean(sg * r5[E]) * 100:+.2f}% / "
                 f"{np.nanmean(sg * r5[Cc]) * 100:+.2f}%；收盘位置 {np.nanmean(kk[E]):.2f} / {np.nanmean(kk[Cc]):.2f}")
        L.append("     逐年成交：" + " ".join(f"{y}:{int((E & F & (year == y)).sum())}" for y in range(2016, 2025)))
        if True:
            cnt = np.bincount(mi[F & ~real], minlength=info["T"])
            days = np.unique(mi[rows])
            L.append(f"     本侧拟合池每日成交数 5/50/95% 分位 {' / '.join(str(int(x)) for x in np.quantile(cnt[days], [.05, .5, .95]))}；"
                     f"不足 {MIN_DAY} 只的天数 {int((cnt[days] < MIN_DAY).sum())}")
    # ④ sample
    base = pn["eligible"].to_numpy(bool) & pn["entry_ok"].to_numpy(bool) & np.isfinite(pn["crash_20"].to_numpy(float)) & (td >= "20160101")
    for pre, fam, nm in [("p", "A", "A④"), ("p", "C", "C④")]:
        Ff = c[f"{pre}_{fam}_fail"]
        L.append(f"  {nm} 踩雷样本：失守日 {int(Ff.sum()):,} → 入池且次日可买 {int((Ff & base).sum()):,}"
                 f"（不在 09b 候选行的 {int((Ff & base & ~inrows).sum()):,}）；普通失守对照 {int((c[f'{pre}_{('G' if fam == 'A' else 'GC')}_fail'] & base & ~real).sum()):,}")
    # overlaps
    A_ = c["p_A_test"]
    sig_p3 = c09["sig_p3_bot_turn"]
    L.append(f"\n  重叠：A 族支撑测试同时是 C 族测试 {np.mean(c['p_C_test'][A_]) * 100:.1f}%、锚点测试 {np.mean(c['p_N_test'][A_]) * 100:.1f}%、"
             f"三推底拐头 {np.mean(sig_p3[A_]) * 100:.1f}%；C 族测试同时是 A 族测试 {np.mean(A_[c['p_C_test']]) * 100:.1f}%")
    cc = c["p_A_cc"][c["p_A_test"]]
    L.append(f"  成交密集度（A 族支撑测试，60 天窄带成交额占比）分位 10/50/90%：{' / '.join(f'{x:.3f}' for x in np.nanquantile(cc, [.1, .5, .9]))}；"
             f"缺失 {np.mean(~np.isfinite(cc)) * 100:.1f}%")
    return L


# ---------------------------------------------------------------------------
# 3. Main analysis (step 2)
# ---------------------------------------------------------------------------
def qdum(x, ref):
    return s09b.qdum(x, ref)


def day_quint(x: np.ndarray, mi: np.ndarray) -> np.ndarray:
    """Within-day quintile dummies (4 columns) of x."""
    r = pd.Series(x).groupby(mi).rank(pct=True, method="first").to_numpy()
    b = np.ceil(np.nan_to_num(r, nan=0.5) * 5).clip(1, 5).astype(int)
    return np.stack([(b == j).astype(float) for j in range(2, 6)], 1)


def fwd_low_min(a_low: np.ndarray, a_close: np.ndarray, cid: np.ndarray, H: int = 20) -> np.ndarray:
    """min(low[t+1..t+H]) / close[t] − 1 on each stock's own trading days (NaN when fewer than H rows remain)."""
    out = np.full(len(a_low), np.nan)
    st = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    en = np.r_[st[1:], len(cid)]
    for s, e in zip(st.tolist(), en.tolist()):
        lo = a_low[s:e]
        if e - s <= H:
            continue
        mn = pd.Series(lo[::-1]).rolling(H, min_periods=H).min().to_numpy()[::-1]  # mn[i] = min lo[i..i+H-1]
        out[s:e - H] = mn[1:e - s - H + 1] / a_close[s:e - H] - 1.0
    return out


def main_analysis() -> None:
    t0 = time.time()
    s10c = _load_mod("s10c", "10c_trading_range_explore.py")
    c = load_cache()
    c09 = dict(np.load(s09b.CACHE_PATH, allow_pickle=False))
    rows = c09["rows"]
    M = len(rows)
    cols = ["code_id", "trade_date", "mi", "tier", "l1", "eligible", "entry_ok", "crash_20", "tail_20", "ret_20", "ret_10", "ret1",
            "max20", "lim_pct", "up_close", "dn_close", "a_open", "a_high", "a_low", "a_close"] + [x for x in L2_COLS]
    pn = pd.read_parquet(s09b.PANEL_PATH, columns=list(dict.fromkeys(cols)))
    pn = pn.sort_values(["code_id", "trade_date"]).reset_index(drop=True)
    NP = len(pn)
    mi_all = pn["mi"].to_numpy()
    td_all = pn["trade_date"].to_numpy()
    cid_all = pn["code_id"].to_numpy()
    year_all = np.array([int(x[:4]) for x in td_all])
    tier_all = pn["tier"].to_numpy()
    l1_all = pd.factorize(pn["l1"].to_numpy())[0]
    Z_all = np.column_stack([pn[L2_COLS].to_numpy(float), np.nan_to_num(pn["ret1"].to_numpy(float))])
    lmv_all = pn["lmv"].to_numpy(float)
    real_all = c["p_A_test"] | c["m_A_test"] | c["p_C_test"] | c["p_N_test"]
    f_ = s09.fmt
    tw = lambda res, idx, mi: b8.vs_twins(res, idx, mi)[:2]

    # ---------------- ledger questions on the 09b candidate rows
    mi, year, tier, code = mi_all[rows], year_all[rows], tier_all[rows], cid_all[rows]
    l1, Z, lmv = l1_all[rows], Z_all[rows], lmv_all[rows]
    szq = np.ceil((pd.Series(lmv).groupby(mi).rank(pct=True, method="first").to_numpy() * 5).clip(1e-9, 5)).astype(int)
    real = real_all[rows]
    ev = {k: v[rows] for k, v in c.items() if k[:2] in ("p_", "m_")}
    order = np.argsort(mi, kind="stable")
    geo = {x: c09[x][rows] for x in ["g_atr", "g_px", "kcls"]}
    r1q = day_quint(Z[:, -1], mi)
    crash_full_res: dict[str, np.ndarray] = {}

    def Gside(F):
        return np.hstack([qdum(geo[x], geo[x][F]) for x in ["g_atr", "g_px", "kcls"]])

    def resid(Y, filled, fit, G, zcols=None, l0=False):
        Y = Y if Y.ndim == 2 else Y[:, None]
        o = order[filled[order]]
        Zx = None if l0 else (Z if zcols is None else Z[:, zcols])
        r, sk = s09b.resid_day(mi[o], szq[o], None if Zx is None else Zx[o], l1[o], None if G is None else G[o], Y[o], fit[o], MIN_DAY)
        out = np.full(Y.shape, np.nan)
        out[o] = r
        return out, sk

    def thin_mask(F, hold, mask):
        idx = np.flatnonzero(mask & F)
        keep = np.zeros(M, bool)
        o = np.lexsort((mi[idx], code[idx]))
        last_n, last_x = -1, -1
        for i in idx[o].tolist():
            n, t = code[i], mi[i]
            if n != last_n:
                last_n, last_x = n, -1
            if t + 1 > last_x:
                keep[i] = True
                last_x = t + 1 + hold[i]
        return keep

    nL2 = len(L2_COLS)
    out_rows = []
    L = ["=" * 120, "Study 11b · 第 2 步：价位记忆（全 A 个股，开发段 2016-01 → 2024-10）", "=" * 120,
         f"候选信号 K 线 {M:,}（与 09b 同一批）；超额 = 事件单相对同日双胞胎随机单的每笔净 R（L3 + 当天涨跌 + 几何，DK lag 30）；"
         "专属对照 = 真价位 − 普通价位（前一根 K 线高低点）的残差差。拟合池只剔除真价位事件。"]
    res_side: dict[int, dict] = {}
    for side in (1, -1):
        p = "L" if side > 0 else "S"
        F = c09[f"{p}_main_filled"]
        fit = F & ~real
        G = Gside(F)
        Gm = np.hstack([G, r1q])
        variants = ["main", "opt", "k2"] + (["t1i"] if side > 0 else [])
        Y = np.column_stack([c09[f"{p}_{v}_net"].astype(float) for v in variants] + [c09[f"{p}_main_win"].astype(float)])
        log(f"residualize side={side:+d}")
        R_, sk = resid(Y, F, fit, Gm)
        res = {v: R_[:, j] for j, v in enumerate(variants)}
        res["win"] = R_[:, len(variants)]
        net = c09[f"{p}_main_net"].astype(float)
        res["l0"] = resid(net, F, fit, None, l0=True)[0][:, 0]
        res["l3"] = resid(net, F, fit, None, zcols=list(range(nL2)))[0][:, 0]
        res["l3g"] = resid(net, F, fit, G, zcols=list(range(nL2)))[0][:, 0]
        Fo = c09[f"{p}_open_filled"]
        res["open"] = resid(c09[f"{p}_open_net"].astype(float), Fo, Fo & ~real, np.hstack([Gside(Fo), r1q]))[0][:, 0]
        res_side[side] = {"res": res, "F": F, "fit": fit, "Gm": Gm, "sk": sk, "net": net}
        log(f"side={side:+d} residuals done ({time.time() - t0:.0f}s)")

    # ---------------- ④ crash residuals on all eligible next-day-buyable rows (08b style, not conditioned on fills)
    log("crash residuals (full eligible sample)")
    gap_low = fwd_low_min(pn["a_low"].to_numpy(float), pn["a_close"].to_numpy(float), cid_all)
    crash_gap = np.where(np.isfinite(gap_low), (gap_low <= -0.15).astype(float), np.nan)
    kfix = c09["kcls"].astype(float).copy()
    one = np.isnan(kfix)
    kfix[one & pn["up_close"].to_numpy(bool)] = 1.0
    kfix[one & pn["dn_close"].to_numpy(bool)] = 0.0
    kfix[np.isnan(kfix)] = 0.5
    gfull = {"g_atr": np.nan_to_num(c09["g_atr"].astype(float), nan=0.0, posinf=0.0), "g_px": np.nan_to_num(c09["g_px"].astype(float), nan=0.0, posinf=0.0),
             "kcls": kfix}
    base = pn["eligible"].to_numpy(bool) & pn["entry_ok"].to_numpy(bool) & (td_all >= "20160101") \
        & np.isfinite(pn[["crash_20", "tail_20", "ret_20"] + L2_COLS].to_numpy(float)).all(1)
    brow = np.flatnonzero(base)
    bmi = mi_all[brow]
    bo = np.argsort(bmi, kind="stable")
    bszq = np.ceil((pd.Series(lmv_all[brow]).groupby(bmi).rank(pct=True, method="first").to_numpy() * 5).clip(1e-9, 5)).astype(int)
    bfit = ~real_all[brow]
    Gb = np.hstack([qdum(gfull[x][brow], gfull[x][brow]) for x in ["g_atr", "g_px", "kcls"]] + [day_quint(Z_all[brow, -1], bmi)])
    Yb = np.column_stack([pn["crash_20"].to_numpy(float)[brow] * 100, pn["tail_20"].to_numpy(float)[brow] * 100,
                          pn["ret_20"].to_numpy(float)[brow] * 100, np.nan_to_num(crash_gap[brow], nan=0.0) * 100,
                          np.nan_to_num(pn["ret_10"].to_numpy(float)[brow], nan=0.0) * 100])
    rb, _ = s09b.resid_day(bmi[bo], bszq[bo], Z_all[brow][bo], l1_all[brow][bo], Gb[bo], Yb[bo], bfit[bo], 300)
    rb3, _ = s09b.resid_day(bmi[bo], bszq[bo], Z_all[brow][bo][:, :nL2], l1_all[brow][bo], None, Yb[bo][:, [0, 2]], bfit[bo], 300)
    CR = np.full((NP, 7), np.nan)
    CR[brow[bo], :4] = rb[:, :4]
    CR[brow[bo], 4] = rb3[:, 0]
    CR[brow[bo], 5] = rb[:, 4]
    CR[brow[bo], 6] = rb3[:, 1]
    CR[~np.isfinite(pn["ret_10"].to_numpy(float)), 5] = np.nan
    CR[~np.isfinite(crash_gap), 3] = np.nan
    log(f"crash residuals done ({time.time() - t0:.0f}s)")
    inb = np.zeros(NP, bool)
    inb[brow] = True

    def crash_block(Em: np.ndarray, Cm: np.ndarray | None) -> dict:
        """Em / Cm: full-panel boolean masks of signal days."""
        S = np.flatnonzero(Em & inb)
        r = {"n_cr": len(S)}
        r["cr"], r["cr_t"] = tw(CR[:, 0], S, mi_all)
        r["cr_tail"], r["cr_tail_t"] = tw(CR[:, 1], S, mi_all)
        r["cr_ret"], r["cr_ret_t"] = tw(CR[:, 2], S, mi_all)
        r["cr_gap"], _ = tw(CR[:, 3], S, mi_all)
        r["cr_l3"], r["cr_l3_t"] = tw(CR[:, 4], S, mi_all)
        r["cr_raw"] = float(np.nanmean(pn["crash_20"].to_numpy(float)[S]) * 100)
        r["cr_raw_gap"] = float(np.nanmean(crash_gap[S]) * 100)
        if Cm is not None:
            Ci = np.flatnonzero(Cm & inb & ~real_all)
            r["cr_ctrl"], r["cr_ctrl_t"] = b8.vs_ctrl(CR[:, 0], S, Ci, mi_all)
            r["cr_raw_ctrl"] = float(np.nanmean(pn["crash_20"].to_numpy(float)[Ci]) * 100)
        r["cr_per"] = [tw(CR[:, 0], S[year_all[S] == y], mi_all)[0] for y in range(2016, 2025)]
        r["cr_halves"] = [tw(CR[:, 0], S[year_all[S] <= 2020], mi_all)[0], tw(CR[:, 0], S[year_all[S] >= 2021], mi_all)[0]]
        r["cr_tiers"] = [tw(CR[:, 0], S[tier_all[S] == k], mi_all)[0] for k in (1, 2, 3, 4)]
        return r

    for code_, pre, ls, ek, ck, lab in PRIM:
        sd = res_side[ls]
        res, F, fit, Gm, net = sd["res"], sd["F"], sd["fit"], sd["Gm"], sd["net"]
        p = "L" if ls > 0 else "S"
        E = ev[f"{pre}_{ek}"]
        Cm = ev[f"{pre}_{ck}"] & ~real
        S = np.flatnonzero(E & F)
        Ci = np.flatnonzero(Cm & F)
        win, win_o = c09[f"{p}_main_win"], c09[f"{p}_opt_win"]
        net_o = c09[f"{p}_opt_net"].astype(float)
        Fopt = c09[f"{p}_opt_filled"]
        why = c09[f"{p}_main_why"]
        trig = (why == s09.W_OK) | (why == s09.W_LIMIT)
        r = {"market": "STK", "code": code_, "label": lab, "side": ls, "s": 1, "n_sig": int(E.sum()), "n_fill": len(S),
             "n_ctrl": len(Ci), "fill_rate": float(F[E].mean()), "skipped": sd["sk"],
             "win": float(win[S].mean()), "win_opt": float(win_o[E & Fopt].mean()), "rwin": float(win[fit].mean()),
             "net": float(net[S].mean()), "net_opt": float(net_o[E & Fopt].mean()), "rnet": float(net[fit].mean()),
             "blocked": float((why[E & trig] == s09.W_LIMIT).mean()) if (E & trig).any() else np.nan,
             "rblocked": float((why[trig & ~real] == s09.W_LIMIT).mean())}
        r["ex"], r["ex_t"] = tw(res["main"], S, mi)
        r["ex_opt"], _ = tw(res["opt"], np.flatnonzero(E & F), mi)
        r["ex_k2"], _ = tw(res["k2"], S, mi)
        if ls > 0:
            r["ex_t1i"], _ = tw(res["t1i"], S, mi)
        r["ex_win"], _ = tw(res["win"], S, mi)
        for v in ["l0", "l3", "l3g"]:
            r[f"ex_{v}"], r[f"ex_{v}_t"] = tw(res[v], S, mi)
        r["ex_open"], r["ex_open_t"] = tw(res["open"], np.flatnonzero(E & c09[f"{p}_open_filled"]), mi)
        r["ctrl"], r["ctrl_t"] = b8.vs_ctrl(res["main"], S, Ci, mi)
        r["ctrl_opt"], _ = b8.vs_ctrl(res["opt"], S, Ci, mi)
        r["ctrl_day"], r["ctrl_day_t"], _ = s10c.day_diff(mi, res["main"], E & F, Cm & F, 1)
        # + paired ordinary-level dummy (event vs the non-ordinary remainder)
        Pd = Cm.astype(float)[:, None]
        if code_ in ("A2", "N2", "C2"):
            gb = ev[f"{pre}_{'GC' if code_ == 'C2' else 'G'}_bdn"] & ~real
            Pd = np.column_stack([Pd, gb.astype(float)])
            r["ctrl_bdn"], r["ctrl_bdn_t"] = b8.vs_ctrl(res["main"], S, np.flatnonzero(gb & F), mi)
        rg, _ = resid(net, F, fit, np.hstack([Gm, Pd]))
        r["ex_gen"], r["ex_gen_t"] = tw(rg[:, 0], S, mi)
        tkey = f"{pre}_{ek.split('_')[0]}_test"
        r["ex_touch"], r["ex_touch_t"] = tw(res["main"], np.flatnonzero(ev[tkey] & F), mi)
        r["per"] = [tw(res["main"], S[year[S] == y], mi)[0] for y in range(2016, 2025)]
        r["halves"] = [tw(res["main"], S[year[S] <= 2020], mi)[0], tw(res["main"], S[year[S] >= 2021], mi)[0]]
        r["tiers"] = [tw(res["main"], S[tier[S] == k], mi)[0] for k in (1, 2, 3, 4)]
        r["ctrl_tiers"] = [b8.vs_ctrl(res["main"], S[tier[S] == k], Ci[tier[Ci] == k], mi)[0] for k in (1, 2, 3)]
        r["p"] = b8.p_two(r["ex_t"])
        hold = c09[f"{p}_main_hold"].astype(np.int64)
        keep = thin_mask(F, hold, E) | thin_mask(F, hold, fit)
        Rt, _ = resid(net, F & keep, fit & keep, Gm)
        r["ex_thin"], r["ex_thin_t"] = tw(Rt[:, 0], np.flatnonzero(E & F & keep), mi)
        r["time_share"] = float((c09[f"{p}_main_kind"][S] == s09.K_TIME).mean())
        fam = ek.split("_")[0]
        if ek.endswith("hold") and f"{pre}_{fam}_clean" in ev:
            cl = ev[f"{pre}_{fam}_clean"]
            r["clean_ex"], r["clean_ex_t"] = tw(res["main"], np.flatnonzero(E & F & cl), mi)
            r["pierce_ex"], r["pierce_ex_t"] = tw(res["main"], np.flatnonzero(E & F & ~cl), mi)
        if fam == "A":
            nt = ev[f"{pre}_A_ntest"]
            vals = [tw(res["main"], np.flatnonzero(E & F & m), mi) for m in (nt == 1, nt == 2, nt >= 3)]
            r["dose_nt"], r["dose_nt_t"] = [v[0] for v in vals], [v[1] for v in vals]
            r["dose_nt_n"] = [int((E & F & m).sum()) for m in (nt == 1, nt == 2, nt >= 3)]
        if fam in ("A", "C"):
            cc = ev[f"{pre}_{fam}_cc"].astype(float)
            ok = np.isfinite(cc) & E & F
            qs = np.nanquantile(cc[ok], [0.2, 0.4, 0.6, 0.8])
            bins = np.digitize(cc, qs)
            vals = [tw(res["main"], np.flatnonzero(ok & (bins == k)), mi) for k in range(5)]
            r["dose_cc"], r["dose_cc_t"] = [v[0] for v in vals], [v[1] for v in vals]
        if fam == "C":
            lp = pn["lim_pct"].to_numpy(float)[rows]
            for nm, m in [("yizi", ev["p_C_yizi"]), ("nonyizi", ~ev["p_C_yizi"]), ("lian", ev["p_C_lian"]), ("gu", ~ev["p_C_lian"]),
                          ("b20", lp >= 0.19), ("b10", lp < 0.19)]:
                r[f"c_{nm}"], r[f"c_{nm}_t"] = tw(res["main"], np.flatnonzero(E & F & m), mi)
                r[f"c_{nm}_n"] = int((E & F & m).sum())
            # heat robustness: add 20-day turnover quintiles, limit-up-count dummies and max single-day gain
            heat = np.hstack([day_quint(Z[:, L2_COLS.index("turn20")], mi),
                              np.column_stack([(Z[:, L2_COLS.index("n_up20")] == k).astype(float) for k in (1, 2)]
                                              + [(Z[:, L2_COLS.index("n_up20")] >= 3).astype(float)]),
                              day_quint(pn["max20"].to_numpy(float)[rows], mi)])
            rh, _ = resid(net, F, fit, np.hstack([Gm, heat]))
            r["c_heat"], r["c_heat_t"] = tw(rh[:, 0], S, mi)
        if ls > 0:
            w = b8.cap_weights(mi[S], np.zeros(len(S)))
            b, t = b8.wls_dk(net[S], np.ones((len(S), 1)), w, mi[S])
            r["abs_c"], r["abs_c_t"] = float(b[0]), float(t[0])
            So = np.flatnonzero(E & Fopt)
            b, _ = b8.wls_dk(net_o[So], np.ones((len(So), 1)), b8.cap_weights(mi[So], np.zeros(len(So))), mi[So])
            r["abs_o"] = float(b[0])
            y2 = 2 * net - c09[f"{p}_main_gross"].astype(float)
            b, t = b8.wls_dk(y2[S], np.ones((len(S), 1)), w, mi[S])
            r["abs_c2"], r["abs_c2_t"] = float(b[0]), float(t[0])
            r["label_trade"] = ("日线口径可交易" if (r["abs_c"] > 0 and r["abs_c_t"] >= 2)
                                else ("不可交易" if r["abs_o"] <= 0 else "日线判断不了"))
        else:
            # holder-sell crash direction on ALL signal days (full eligible sample)
            Efull = c[f"{pre}_{ek}"]
            cb = crash_block(Efull, c[f"{pre}_{ck}"])
            r["ex_crash"], r["ex_crash_t"] = cb["cr"], cb["cr_t"]
        out_rows.append(r)
        log(f"{code_} done ({time.time() - t0:.0f}s)")

    # ---------------- ④ primary rows
    dn_close = pn["dn_close"].to_numpy(bool)
    cr_rows = []
    for code_, fam, gk, lab in [("A4", "A", "G", "A④ 双底失守后踩雷"), ("C4", "C", "GC", "C④ 起涨价失守后踩雷"), ("N4", "N", "G", "锚点 突破位失守后踩雷")]:
        Ef = c[f"p_{fam}_fail"]
        r = {"market": "STK", "code": code_, "label": lab, "side": -1, "s": 1}
        r.update(crash_block(Ef, c[f"p_{gk}_fail"]))
        r["ex"], r["ex_t"] = r["cr"], r["cr_t"]
        r["ctrl"], r["ctrl_t"] = r["cr_ctrl"], r["cr_ctrl_t"]
        r["per"], r["halves"], r["tiers"] = r["cr_per"], r["cr_halves"], r["cr_tiers"]
        r["p"] = b8.p_two(r["ex_t"])
        S = np.flatnonzero(Ef & inb)
        nx = np.minimum(S + 1, NP - 1)
        same = cid_all[nx] == cid_all[S]
        r["sig_dn"] = float(dn_close[S].mean() * 100)
        yz = (pn["a_high"].to_numpy(float) - pn["a_low"].to_numpy(float)) <= 1e-9
        r["next_dn_open"] = float(np.mean((yz & dn_close)[nx][same]) * 100)
        cr_rows.append(r)
    L += render_stock(out_rows, cr_rows)
    # ---------------- old judge side by side with 08b (all signal days, t+1 open buy, 10 / 20-day returns and crash vs twins)
    f = s09.fmt
    L.append("\n  【个股 · 旧裁判并排（诊断）】全部入池、次日可买的信号日，次日开盘买；比双胞胎（L3 + 当天涨跌 + 几何）；"
             "面板没有 5 天标签，用 10 天代替；守住 − 失守 = 两组残差差（对照 08b：突破位守住 − 击穿 20 天 +0.44%、踩雷 −1.8pp）")
    for pre, fam, nm in [("p", "A", "A 双底（支撑）"), ("m", "A", "A 双顶（压力，守住 = 冲高失败）"), ("p", "C", "C 起涨价"), ("p", "N", "锚点 突破位")]:
        Sh = np.flatnonzero(c[f"{pre}_{fam}_hold"] & inb)
        Sf = np.flatnonzero(c[f"{pre}_{fam}_fail"] & inb)
        vals = []
        for j, unit in [(5, "%"), (2, "%"), (0, "pp")]:
            h_ = tw(CR[:, j], Sh, mi_all)
            f2 = tw(CR[:, j], Sf, mi_all)
            d_ = b8.vs_ctrl(CR[:, j], Sh, Sf, mi_all)
            vals.append(f"{f(h_[0], 2)}{unit} / {f(f2[0], 2)}{unit} / 差 {f(d_[0], 2)}{unit}（t {f(d_[1], 1)}）")
        d3 = b8.vs_ctrl(CR[:, 6], Sh, Sf, mi_all)
        L.append(f"  {nm:<22s} 守住 / 失守 / 差：10 天 {vals[0]}；20 天 {vals[1]}；踩雷 {vals[2]}；只控 L3 的 20 天差 {f(d3[0], 2)}%")
    # ---------------- support strength (descriptive)
    L.append("\n  【个股 · 支撑力（描述）】真价位守住率 vs 同距离伪价位（按距离十分位配对 / 再配近 5 日涨跌与振幅）vs 随机游走；刺穿后收回率")
    rng = np.random.default_rng(20261126)
    Hh, Ll, Cc = (pn[x].to_numpy(float) for x in ("a_high", "a_low", "a_close"))
    first = np.r_[True, cid_all[1:] != cid_all[:-1]]
    cp = np.r_[np.nan, Cc[:-1]]
    cp[first] = np.nan
    zp = np.r_[np.nan, c["p_atr_v"][:-1].astype(float)]
    zp[first] = np.nan
    inrows = np.zeros(NP, bool)
    inrows[rows] = True
    for pre, fam, vs, nm in [("p", "A", 1, "A 双底（支撑）"), ("m", "A", -1, "A 双顶（压力）"), ("p", "C", 1, "C 起涨价")]:
        test, dist = c[f"{pre}_{fam}_test"], c[f"{pre}_{fam}_dist"].astype(float)
        tm = test & np.isfinite(dist)
        r5 = c[f"{pre}_r5v"].astype(float)
        rg5 = c[f"{pre}_rng5"].astype(float)
        qr5, qrg = np.nanquantile(r5[tm], [1 / 3, 2 / 3]), np.nanquantile(rg5[tm], [1 / 3, 2 / 3])
        cell = lambda a, b: np.digitize(a, qr5) * 3 + np.digitize(b, qrg)
        okp = np.flatnonzero(inrows & ~c[f"{pre}_{fam}_near"] & ~test & np.isfinite(r5) & np.isfinite(rg5) & np.isfinite(cp) & np.isfinite(zp) & (zp > EPS))
        if len(okp) > 3_000_000:
            okp = np.sort(rng.choice(okp, 3_000_000, replace=False))
        ext = Ll if vs > 0 else Hh
        pool = {"ext": ext[okp], "c": Cc[okp], "cp": cp[okp], "z": zp[okp], "cell": cell(r5[okp], rg5[okp])}
        hold = c[f"{pre}_{fam}_hold"][tm]
        pierce = ~c[f"{pre}_{fam}_clean"][tm]
        ss = s11.support_strength(dist[tm], hold, pierce, cell(r5[tm], rg5[tm]), pool, vs, rng)
        L.append(f"  {nm:<14s} n={int(tm.sum()):,}：守住率 真 {ss['hold_real'] * 100:.1f}% 伪 {ss['hold_pseudo'] * 100:.1f}% 随机游走 {ss['rw_hold'] * 100:.1f}%；"
                 f"配对后 真 − 伪 {ss['hold_gap_d'] * 100:+.1f}pp / {ss['hold_gap_d3'] * 100:+.1f}pp；"
                 f"刺穿后收回 真 {ss['phold_real'] * 100:.1f}% 伪 {ss['phold_pseudo'] * 100:.1f}%（配对差 {ss['phold_gap'] * 100:+.1f}pp）")
    # ---------------- buy directly at the level vs wait for the hold and chase (descriptive, simple returns)
    L.append("\n  【个股 · 在价位上直接接 vs 等守住后追（描述，简单收益，减去同日全部候选股同期收盘到收盘均值）】"
             "直接接 = 测试日最低价低于 L 的行按 min(开盘, L) 成交（不分守住失守）；等守住后追 = 守住日次日越过高点按 max(开盘, 高点) 成交；都持有到第 t+h 天收盘")
    Oo = pn["a_open"].to_numpy(float)
    for pre, fam, nm in [("p", "A", "A 双底"), ("p", "C", "C 起涨价")]:
        test = c[f"{pre}_{fam}_test"] & inb
        lvl = c[f"{pre}_{fam}_lvl"].astype(float)
        direct = np.flatnonzero(test & ~c[f"{pre}_{fam}_clean"])
        hold_rows = np.flatnonzero(c[f"{pre}_{fam}_hold"] & inb)
        parts = [f"未碰到 L 的测试占 {np.mean(c[f'{pre}_{fam}_clean'][test]) * 100:.0f}%"]
        for h in (1, 5, 20):
            j = np.minimum(np.arange(NP) + h, NP - 1)
            ok_h = cid_all[j] == cid_all
            bench = np.where(ok_h & inrows, Cc[j] / Cc - 1.0, np.nan)
            bm = pd.Series(bench).groupby(mi_all).transform("mean").to_numpy()
            fill_d = np.minimum(Oo, lvl)
            rd = np.where(ok_h, Cc[j] / fill_d - 1.0, np.nan)[direct] - bm[direct]
            n1 = np.minimum(hold_rows + 1, NP - 1)
            trig = (cid_all[n1] == cid_all[hold_rows]) & (Hh[n1] > Hh[hold_rows])
            hr = hold_rows[trig]
            fill_c = np.maximum(Oo[hr + 1], Hh[hr])
            okc = cid_all[np.minimum(hr + h, NP - 1)] == cid_all[hr]
            rc = np.where(okc, Cc[np.minimum(hr + h, NP - 1)] / fill_c - 1.0, np.nan) - bm[hr]
            parts.append(f"h={h}：直接接 {np.nanmean(rd) * 100:+.2f}%（n{np.isfinite(rd).sum():,}）/ 等守住后追 {np.nanmean(rc) * 100:+.2f}%（n{np.isfinite(rc).sum():,}）")
        L.append(f"  {nm}：" + "；".join(parts))
    L += final_verdict(out_rows, cr_rows)
    txt = "\n".join(L)
    REPORT_PATH.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    flat = []
    for r in out_rows + cr_rows:
        d = {k: v for k, v in r.items() if not isinstance(v, list)}
        for k in ["per", "halves", "tiers"]:
            d[k] = ";".join(f"{v:+.4f}" if np.isfinite(v) else "nan" for v in r[k])
        flat.append(d)
    pd.DataFrame(flat).to_csv(SUMMARY_PATH, index=False)
    log(f"done ({time.time() - t0:.0f}s) -> {REPORT_PATH}")


def _sg(per, s=1):
    sg = "".join("+" if (np.isfinite(v) and v > 0) else ("−" if np.isfinite(v) else "·") for v in per)
    return sg, sum(1 for v in per if np.isfinite(v) and s * v > 0)


def render_stock(rows: list[dict], cr_rows: list[dict]) -> list[str]:
    f = s09.fmt
    L = [f"\n  【个股 · 账本主检验与锚点】全部 Brooks 方向 s = +1",
         f"  {'交易':<16s}{'信号':>9s}{'成交':>8s}{'对照成交':>10s}{'成交率':>7s}{'胜率 保守~乐观':>16s}{'随机胜率':>9s}"
         f"{'净R 保守~乐观':>18s}{'随机净R':>9s}{'超额':>8s}{'t':>6s}{'乐观超额':>9s}{'真−普通':>9s}{'t':>6s}"]
    for r in rows:
        L.append(f"  {r['label']:<14s}{r['n_sig']:>9,}{r['n_fill']:>8,}{r['n_ctrl']:>10,}{r['fill_rate'] * 100:>6.0f}%"
                 f"{r['win'] * 100:>9.1f}%~{r['win_opt'] * 100:.1f}%{r['rwin'] * 100:>8.1f}%"
                 f"{f(r['net']):>10s}~{f(r['net_opt'])}{f(r['rnet']):>9s}{f(r['ex']):>8s}{f(r['ex_t'], 1):>6s}"
                 f"{f(r['ex_opt']):>9s}{f(r['ctrl']):>9s}{f(r['ctrl_t'], 1):>6s}")
    L.append("  拆解：只扣同日同市值 → +L3（不控形状和当天涨跌）→ +几何 → +当天涨跌（主规格）→ +普通价位哑变量；价位事件本身（全部测试）；"
             "不重叠；开盘进；2R；T+1 忽略变体；专属对照同日相减")
    for r in rows:
        t1 = f"；T+1 忽略 {f(r['ex_t1i'])}" if "ex_t1i" in r else ""
        bd = f"；对普通 20 日新低 {f(r['ctrl_bdn'])}（t {f(r['ctrl_bdn_t'], 1)}）" if "ctrl_bdn" in r else ""
        L.append(f"  {r['label']:<14s} {f(r['ex_l0'])}（t {f(r['ex_l0_t'], 1)}）→ {f(r['ex_l3'])} → {f(r['ex_l3g'])} → {f(r['ex'])} → {f(r['ex_gen'])}；"
                 f"全部测试 {f(r['ex_touch'])}（t {f(r['ex_touch_t'], 1)}）；不重叠 {f(r['ex_thin'])}（t {f(r['ex_thin_t'], 1)}）；"
                 f"开盘进 {f(r['ex_open'])}；2R {f(r['ex_k2'])}{t1}；同日相减 {f(r['ctrl_day'])}（t {f(r['ctrl_day_t'], 1)}）{bd}")
    L.append("  稳定性：2016…2024 逐年 | 前后两段 | 市值档 沪深300 / 中证500 / 中证1000 / 其余 | 专属对照分档（前三档）| 截面跳过天数 | 卖出信号日 20 天踩雷差")
    for r in rows:
        sg, npos = _sg(r["per"])
        cr = f" | {f(r['ex_crash'], 2)}pp（t {f(r['ex_crash_t'], 1)}）" if "ex_crash" in r else ""
        L.append(f"  {r['label']:<14s} {sg:<10s} 正 {npos}/9 | {' / '.join(f(v) for v in r['halves'])} | "
                 f"{' / '.join(f(v) for v in r['tiers'])} | {' / '.join(f(v) for v in r['ctrl_tiers'])} | {r['skipped']}{cr}")
    L.append("  机制诊断：干净守住 / 刺穿收回 | 测试次数 1 / 2 / ≥3 | 成交密集度五档（低→高）| 触发后涨停买不进 / 跌停逃不掉（事件 / 随机）| 到期占比")
    for r in rows:
        a = f"{f(r['clean_ex'])}（t {f(r['clean_ex_t'], 1)}）/ {f(r['pierce_ex'])}（t {f(r['pierce_ex_t'], 1)}）" if "clean_ex" in r else "—"
        b = " ".join(f"{f(v)}({f(t, 1)},n{n:,})" for v, t, n in zip(r["dose_nt"], r["dose_nt_t"], r["dose_nt_n"])) if "dose_nt" in r else "—"
        cc = " ".join(f"{f(v)}({f(t, 1)})" for v, t in zip(r["dose_cc"], r["dose_cc_t"])) if "dose_cc" in r else "—"
        L.append(f"  {r['label']:<14s} {a} | {b} | {cc} | {r['blocked'] * 100:.2f}% / {r['rblocked'] * 100:.2f}% | {r['time_share'] * 100:.1f}%")
    for r in [x for x in rows if x["code"].startswith("C")]:
        L.append(f"  {r['label']} 拆分：一字板 {f(r['c_yizi'])}（t {f(r['c_yizi_t'], 1)}，n{r['c_yizi_n']:,}）/ 非一字 {f(r['c_nonyizi'])}；"
                 f"连板后回落 {f(r['c_lian'])}（t {f(r['c_lian_t'], 1)}，n{r['c_lian_n']:,}）/ 孤板 {f(r['c_gu'])}（t {f(r['c_gu_t'], 1)}）；"
                 f"20% 板 {f(r['c_b20'])}（t {f(r['c_b20_t'], 1)}，n{r['c_b20_n']:,}）/ 10% 板 {f(r['c_b10'])}；"
                 f"加热度分档后 {f(r['c_heat'])}（t {f(r['c_heat_t'], 1)}）")
    for r in [x for x in rows if x["side"] > 0]:
        L.append(f"  可交易标签（{r['label']}，绝对每笔净 R，每天最多计 30 只）：保守 {f(r['abs_c'])}（t {f(r['abs_c_t'], 1)}），乐观 {f(r['abs_o'])}，"
                 f"2 倍成本 {f(r['abs_c2'])} → 【{r['label_trade']}】")
    L.append("\n  【个股 · ④ 失守后 20 天踩雷】全部入池、次日可买的失守日；比双胞胎（L3 + 当天涨跌 + 几何）多出的百分点")
    for r in cr_rows:
        sg, npos = _sg(r["per"])
        L.append(f"  {r['label']:<14s} n={r['n_cr']:,}：裸踩雷率 {r['cr_raw']:.1f}%（普通失守 {r['cr_raw_ctrl']:.1f}%）；比双胞胎 {f(r['cr'], 2)}pp（t {f(r['cr_t'], 1)}）；"
                 f"比普通失守 {f(r['cr_ctrl'], 2)}pp（t {f(r['cr_ctrl_t'], 1)}）；波动归一 {f(r['cr_tail'], 2)}pp（t {f(r['cr_tail_t'], 1)}）；"
                 f"20 天收益 {f(r['cr_ret'], 2)}%（t {f(r['cr_ret_t'], 1)}）；只控 L3 {f(r['cr_l3'], 2)}pp；含隔夜跳空 {f(r['cr_gap'], 2)}pp（裸 {r['cr_raw_gap']:.1f}%）")
        L.append(f"  {'':<14s} 逐年 {sg} 正 {npos}/9 | 两段 {' / '.join(f(v, 2) for v in r['halves'])} | 市值档 {' / '.join(f(v, 2) for v in r['tiers'])} | "
                 f"信号日收盘封跌停 {r['sig_dn']:.1f}%，次日一字跌停 {r['next_dn_open']:.1f}%")
    return L


def _yes(b):
    return "—" if b is None else ("✔" if b else "✘")


def verdict(r: dict, q: float) -> dict:
    stock = r["market"] == "STK"
    per = [v for v in r["per"] if np.isfinite(v)]
    crash_q = str(r["code"]).endswith("4")
    size_thr, ctrl_thr = (2.0, 1.0) if crash_q else (0.10, 0.05)
    out = {
        "stat": r["ex_t"] >= 2 and q <= 0.05,
        "size": r["ex"] >= size_thr,
        "ctrl": np.isfinite(r.get("ctrl_t", np.nan)) and r["ctrl_t"] >= 2 and r["ctrl"] >= ctrl_thr,
        "stab": sum(v > 0 for v in per) >= 6 and all(np.isfinite(h) and h > 0 for h in r["halves"]),
        "tier": (sum(v > 0 for v in r["tiers"][:3] if np.isfinite(v)) >= 2) if stock else None,
        "opt": None if crash_q else (np.isfinite(r.get("ex_opt", np.nan)) and r["ex_opt"] > 0),
        "t1": (np.isfinite(r.get("ex_t1i", np.nan)) and r["ex_t1i"] > 0) if (stock and int(r["side"]) > 0) else None,
        "crash": (r["ex_crash"] > 0) if (stock and int(r["side"]) < 0 and not crash_q) else None,
        "tail": (r["cr_tail"] > 0) if crash_q else None,
        "ret": (r["cr_ret"] <= 0) if crash_q else None,
    }
    out["pass"] = all(v for v in out.values() if v is not None)
    return out


def final_verdict(stk_rows: list[dict], cr_rows: list[dict]) -> list[str]:
    sw = pd.read_csv(SUMMARY_11)
    sw = sw[(sw["market"] == "SW") & sw["code"].isin(["A1", "A2", "A3"])]
    prim = []
    for _, x in sw.iterrows():
        d = x.to_dict()
        d["per"] = [float(v) for v in str(d["per"]).split(";")]
        d["halves"] = [float(v) for v in str(d["halves"]).split(";")]
        d["label"] = "A" + str(d["label"])
        prim.append(d)
    prim += [r for r in stk_rows if r["code"] in ("A1", "A2", "A3", "C1", "C2")]
    prim += [r for r in cr_rows if r["code"] in ("A4", "C4")]
    assert len(prim) == 10, f"expected 10 primaries, got {len(prim)}"
    q = b8.bh_fdr([r["p"] if np.isfinite(r["p"]) else 1.0 for r in prim])
    f = s09.fmt
    L = ["", "#" * 120, "# A 股 10 个主检验：BH-FDR 与预注册过关判定（行业来自 11，个股来自本脚本；④ 的单位是百分点）", "#" * 120,
         f"  {'市场':<5s}{'主检验':<16s}{'超额':>8s}{'t':>7s}{'q':>7s}{'真−普通':>9s}{'t':>6s}  "
         f"{'①显著':>5s}{'②幅度':>5s}{'③对照':>5s}{'④稳定':>5s}{'⑤市值档':>6s}{'⑥乐观':>5s}{'⑦T+1':>5s}{'⑧踩雷':>5s}{'⑨波动归一':>6s}{'⑩20天收益':>7s}  结论"]
    for r, qq in zip(prim, q):
        v = verdict(r, qq)
        r["q10"], r["pass"] = qq, v["pass"]
        mk = "行业" if r["market"] == "SW" else "个股"
        rev = (r["ex_t"] <= -2) and qq <= 0.05
        L.append(f"  {mk:<5s}{r['label']:<14s}{f(r['ex']):>8s}{f(r['ex_t'], 2):>7s}{qq:>7.3f}{f(r['ctrl']):>9s}{f(r['ctrl_t'], 1):>6s}  "
                 + "".join(f"{_yes(v[k]):>6s}" for k in ["stat", "size", "ctrl", "stab", "tier", "opt", "t1", "crash", "tail", "ret"])
                 + f"  {'过关 ✔' if v['pass'] else ('与 Brooks 反向显著 ✘' if rev else '不过关')}")
    L.append(f"  合计：{sum(r['pass'] for r in prim)} / 10 过关")
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
