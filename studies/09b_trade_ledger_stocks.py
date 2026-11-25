#!/usr/bin/env python3
"""
Market Cycle V2 — Study 09b（第三阶段专题 1 · 交易账本重考核心买卖点：全 A 个股 + A 股 20 个主检验收口）

预注册见 decisions.md「09-预注册」。引擎、形态定义全部直接调用 studies/09（已在 31 个行业上与 05/06 原代码比对一致）。
个股主规格（开跑前写死）：
  - 面板 = output/08_stock_panel_dev.parquet（08 清洗口径，≤ 2024-10-31）；按市场日历宽表推进，停牌日 = 无成交。
    从 ~/market-data 只读补读 stk_limit 与 stock_adj_factor（同一快照），涨跌停价与 1 跳（0.01 元）换到后复权坐标。
  - 信号日须为 08 入池股、2016-01-01 起；计划风险 R_plan = 高 − 低 ≥ 0.3×ATR20；右端截尾 31 个市场交易日。
  - 多头：t+1 越过信号 K 线高点 + 1 跳成交（成交价落在涨停价 = 买不进），止损 = 低点 − 1 跳；T+1 主规则 = 进场当天已跌破止损
    → 次日开盘卖（当天碰目标不算）；止损 / 到期遇收盘封死跌停顺延（最多 10 天，否则冻结最后收盘价）。
  - 空头镜像 = 持有者卖出信号：跌破低点 − 1 跳卖出老仓（不受 T+1 约束），跌停卖不出（收盘封死）= 未成交，另报「跌停逃不掉比例」；
    涨过高点 + 1 跳买回认错，跌 1R 买回兑现；买回遇封死涨停顺延。
  - 成本（来回）0.25%，2023-08-28 起 0.20%；目标 1R；20 个市场交易日到期；同根双碰先算止损。
  - 傻瓜基线 = 同日所有合格股票日按同规则交易；每日截面回归（拟合只用不属于任何 5 个主形态的随机单，当天 ≥ 100 笔）：
    同日市值五档 + 近 20/60 日涨跌、偏离 MA20、20 日波动、近 5 日涨跌、换手、涨停次数、对数市值 + 申万一级行业
    + 信号 K 线几何（R_plan/ATR20、R_plan/价格、收盘位置 各五分位）；形态超额 = 形态单残差均值（DK lag 30，每天每组最多计 30 只）。
  - A 股 20 个主检验（行业 10 来自 09 + 个股 10）合并 BH-FDR；过关门槛见 decisions.md。
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
PANEL_PATH = OUTPUT_DIR / "08_stock_panel_dev.parquet"
CACHE_PATH = OUTPUT_DIR / "09b_cache.npz"
REPORT_PATH = OUTPUT_DIR / "09b_trade_ledger_stocks_report.txt"
SUMMARY_PATH = OUTPUT_DIR / "09b_trade_ledger_stocks_summary.csv"
SUMMARY_09 = OUTPUT_DIR / "09_trade_ledger_index_summary.csv"


def _load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "studies" / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


s09 = _load_mod("s09", "09_trade_ledger_index.py")
b8 = s09.b8
s08 = b8.s08
log = b8.log

SNAPSHOT = "weight_backfill_20261121_092022_raw_lake"
EVAL_START = pd.Timestamp("2016-01-01")
STOCK_BUFFER = 1 + s09.MAX_HOLD + s09.EXIT_DELAY
MIN_DAY = 100
COST_PRE, COST_POST, STAMP_CUT = 0.0025, 0.0020, "20230828"
L2_COLS = b8.L2_COLS
TIER_NAMES = b8.TIER_NAMES
FAMILIES = s09.FAMILIES
SIDE_NAME = s09.SIDE_NAME
MAIN_KEYS = {1: [f[2] for f in FAMILIES], -1: [f[4] for f in FAMILIES]}
COLS = ["ts_code", "trade_date", "code_id", "mi", "tier", "szq", "l1", "lim_pct", "eligible", "up_close", "dn_close",
        "a_open", "a_high", "a_low", "a_close", "close", "ret_20", "crash_20"] + L2_COLS
EPS = 1e-12


# ---------------------------------------------------------------------------
# 1. Data: cleaned panel + read-only limit prices & adjustment factors
# ---------------------------------------------------------------------------
def load_panel() -> tuple[pd.DataFrame, dict]:
    info: dict = {}
    pn = pd.read_parquet(PANEL_PATH, columns=COLS)
    assert pn["trade_date"].max() <= "20241031", "holdout leak"
    pn = pn.sort_values(["code_id", "trade_date"]).reset_index(drop=True)
    sid, man = s08.latest_ready_manifest()
    assert sid == SNAPSHOT, f"snapshot changed: {sid}"
    af = s08.cut_dev(s08.read_ds(man, "stock_adj_factor", ["ts_code", "trade_date", "adj_factor"], s08.LOAD_YEARS))
    lim = s08.cut_dev(s08.read_ds(man, "stk_limit", ["ts_code", "trade_date", "up_limit", "down_limit"], s08.LOAD_YEARS))
    pn = s08.merge_unique(pn, af, "adj_factor")
    pn = s08.merge_unique(pn, lim, "stk_limit")
    del af, lim
    assert pn["adj_factor"].notna().all(), "missing adj_factor"
    miss = pn["up_limit"].isna().to_numpy() | pn["down_limit"].isna().to_numpy()
    prev_c = pn.groupby("code_id")["close"].shift(1).to_numpy(float)
    lp = pn["lim_pct"].to_numpy(float)
    pn.loc[miss, "up_limit"] = np.round(prev_c[miss] * (1 + lp[miss]), 2)
    pn.loc[miss, "down_limit"] = np.round(prev_c[miss] * (1 - lp[miss]), 2)
    info["lim_fallback"] = int(miss.sum())
    adj = pn["adj_factor"].to_numpy(float)
    rel = np.abs(pn["a_close"].to_numpy() - pn["close"].to_numpy(float) * adj) / pn["a_close"].to_numpy()
    info["adj_mismatch"] = float((rel > 1e-4).mean())
    pn["UPa"] = pn["up_limit"].to_numpy(float) * adj
    pn["DNa"] = pn["down_limit"].to_numpy(float) * adj
    pn["TKa"] = 0.01 * adj
    upc_chk = pn["a_close"].to_numpy() >= pn["UPa"].to_numpy() - 0.5 * pn["TKa"].to_numpy()
    info["upc_agree"] = float((upc_chk == pn["up_close"].to_numpy()).mean())
    dates = np.sort(pn["trade_date"].unique())
    assert len(dates) == int(pn["mi"].max()) + 1 and (np.searchsorted(dates, pn["trade_date"].to_numpy()) == pn["mi"].to_numpy()).all()
    info["T"], info["N"], info["rows"] = len(dates), int(pn["code_id"].max()) + 1, len(pn)
    return pn, info


def wide_market(pn: pd.DataFrame, T: int, N: int) -> dict:
    mi, ci = pn["mi"].to_numpy(), pn["code_id"].to_numpy()

    def wide(v, fill=np.nan, dtype=float):
        W = np.full((T, N), fill, dtype=dtype)
        W[mi, ci] = v
        return W

    return {"O": wide(pn["a_open"].to_numpy()), "H": wide(pn["a_high"].to_numpy()), "L": wide(pn["a_low"].to_numpy()),
            "C": wide(pn["a_close"].to_numpy()), "OK": wide(np.ones(len(pn), bool), False, bool),
            "UP": wide(pn["UPa"].to_numpy()), "DN": wide(pn["DNa"].to_numpy()), "TICK": wide(pn["TKa"].to_numpy())}


def detect(pn: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, int]:
    cid = pn["code_id"].to_numpy()
    st = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    en = np.r_[st[1:], len(pn)]
    H, L, C = (pn[c].to_numpy(float) for c in ("a_high", "a_low", "a_close"))
    UPC, DNC = pn["up_close"].to_numpy(bool), pn["dn_close"].to_numpy(bool)
    dates = pd.to_datetime(pn["trade_date"], format="%Y%m%d").to_numpy()
    sig: dict[str, np.ndarray] = {}
    atr = np.full(len(pn), np.nan)
    kcls = np.full(len(pn), np.nan)
    bad_mirror = 0
    for s, e in zip(st.tolist(), en.tolist()):
        if e - s < 30:
            continue
        sg, f = s09.detect_all(H[s:e], L[s:e], C[s:e], dates[s:e], EVAL_START, UPC[s:e], DNC[s:e])
        for k, pos in sg.items():
            if k.startswith("_"):
                continue
            sig.setdefault(k, np.zeros(len(pn), bool))[s + pos] = True
        if not np.array_equal(np.sort(sg["_mirror_spring"]), np.sort(sg["fail_up_20"])):
            bad_mirror += 1
        atr[s:e], kcls[s:e] = f["atr20"], f["kcls"]
    return sig, atr, kcls, bad_mirror


# ---------------------------------------------------------------------------
# 2. Per-day cross-sectional twins (fit on non-pattern random trades, predict all)
# ---------------------------------------------------------------------------
def qdum(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    qs = np.nanquantile(ref, [0.2, 0.4, 0.6, 0.8])
    b = np.digitize(x, qs)
    return np.stack([(b == j).astype(float) for j in range(1, 5)], 1)


def resid_day(mi, szq, Z, ind, G, Y, fit, min_n=MIN_DAY):
    """Rows must be sorted by mi. Returns residuals (NaN on skipped days) and the number of skipped days."""
    out = np.full(Y.shape, np.nan)
    skipped = 0
    for a, b in b8.day_slices(mi):
        f = fit[a:b]
        if f.sum() < min_n:
            skipped += 1
            continue
        m = b - a
        dsz = np.zeros((m, 5))
        dsz[np.arange(m), szq[a:b] - 1] = 1.0
        parts = [dsz]
        if Z is not None:
            codes, inv = np.unique(ind[a:b], return_inverse=True)
            dl = np.zeros((m, len(codes)))
            dl[np.arange(m), inv] = 1.0
            parts += [b8._z(Z[a:b]), dl[:, 1:]]
        if G is not None:
            parts.append(G[a:b])
        X = np.hstack(parts)
        beta, *_ = np.linalg.lstsq(X[f], Y[a:b][f], rcond=None)
        out[a:b] = Y[a:b] - X @ beta
    return out, skipped


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def build_cache() -> dict:
    t0 = time.time()
    log("load panel + read-only limits / adj factors")
    pn, info = load_panel()
    T, N = info["T"], info["N"]
    log(f"panel {len(pn):,} rows, T={T}, N={N} ({time.time() - t0:.0f}s); detect signals")
    sig, atr, kcls, bad_mirror = detect(pn)
    info["bad_mirror"] = bad_mirror
    log(f"signals done ({time.time() - t0:.0f}s); candidates")
    H, L, C = (pn[c].to_numpy(float) for c in ("a_high", "a_low", "a_close"))
    rplan = H - L
    with np.errstate(invalid="ignore", divide="ignore"):
        g_atr, g_px = rplan / atr, rplan / C
    td = pn["trade_date"].to_numpy()
    mi = pn["mi"].to_numpy()
    fin = np.isfinite(pn[L2_COLS].to_numpy(float)).all(1) & np.isfinite(g_atr) & np.isfinite(g_px) & np.isfinite(kcls)
    cand = pn["eligible"].to_numpy(bool) & (td >= "20160101") & (rplan > EPS) & (rplan >= s09.R_MIN_ATR * atr) & fin \
        & (mi + STOCK_BUFFER <= T - 1)
    rows = np.flatnonzero(cand)
    info["cand"] = len(rows)
    P = wide_market(pn, T, N)
    tt, nn = mi[rows], pn["code_id"].to_numpy()[rows]
    cost = np.where(td[rows] < STAMP_CUT, COST_PRE, COST_POST)
    cache = {"rows": rows}
    specs = {1: {"main": dict(t1="forced"), "opt": dict(t1="forced", both="opt"), "k2": dict(t1="forced", k=2.0),
                 "t1i": dict(t1="ignore"), "open": dict(t1="forced", entry="open")},
             -1: {"main": {}, "opt": dict(both="opt"), "k2": dict(k=2.0), "open": dict(entry="open")}}
    for side, sp in specs.items():
        for v, kw in sp.items():
            r = s09.run_ledger(P, tt, nn, side, cost=cost, buffer=STOCK_BUFFER, **kw)
            pre = f"{'L' if side > 0 else 'S'}_{v}_"
            cache[pre + "filled"] = r["filled"]
            cache[pre + "net"] = r["net"].astype(np.float32)
            cache[pre + "win"] = r["win"]
            if v == "main":
                for k in ["why", "kind", "outside", "frozen"]:
                    cache[pre + k] = r[k]
                cache[pre + "hold"] = r["hold"].astype(np.int16)
                cache[pre + "gross"] = r["gross"].astype(np.float32)
            log(f"ledger side={side:+d} {v}: filled {r['filled'].mean() * 100:.1f}% ({time.time() - t0:.0f}s)")
    for k, v in sig.items():
        cache["sig_" + k] = v
    cache["g_atr"], cache["g_px"], cache["kcls"] = g_atr, g_px, kcls
    cache["info"] = np.array([repr(info)])
    np.savez(CACHE_PATH, **cache)
    log(f"cache saved ({time.time() - t0:.0f}s)")
    return dict(np.load(CACHE_PATH, allow_pickle=False))


def main() -> None:
    t0 = time.time()
    cache = dict(np.load(CACHE_PATH, allow_pickle=False)) if CACHE_PATH.exists() else build_cache()
    info = eval(str(cache["info"][0]))
    pn = pd.read_parquet(PANEL_PATH, columns=["code_id", "trade_date", "mi", "tier", "szq", "l1", "eligible", "lmv",
                                              "ret_20", "crash_20"] + [c for c in L2_COLS if c != "lmv"])
    pn = pn.sort_values(["code_id", "trade_date"]).reset_index(drop=True)
    rows = cache["rows"]
    M = len(rows)
    mi = pn["mi"].to_numpy()[rows]
    td = pn["trade_date"].to_numpy()[rows]
    year = np.array([int(x[:4]) for x in td])
    tier = pn["tier"].to_numpy()[rows]
    l1 = pd.factorize(pn["l1"].to_numpy()[rows])[0]
    Z = pn[L2_COLS].to_numpy(float)[rows]
    lmv = pn["lmv"].to_numpy(float)[rows]
    szq = (pd.Series(lmv).groupby(mi).rank(pct=True, method="first").to_numpy() * 5).clip(1e-9, 5)
    szq = np.ceil(szq).astype(int)
    crash = pn["crash_20"].to_numpy(float)[rows]
    sig = {k[4:]: v[rows] for k, v in cache.items() if k.startswith("sig_")}
    sig_all = {k[4:]: v for k, v in cache.items() if k.startswith("sig_")}
    order = np.argsort(mi, kind="stable")
    inv_order = np.empty(M, np.int64)
    inv_order[order] = np.arange(M)
    G_full = np.hstack([qdum(cache[x][rows], cache[x][rows][cache["L_main_filled"]]) for x in ["g_atr", "g_px", "kcls"]])

    def resid(Ycols: np.ndarray, filled: np.ndarray, fit: np.ndarray, G, l0: bool = False) -> tuple[np.ndarray, int]:
        o = order[filled[order]]
        Yo = Ycols[o]
        r, sk = resid_day(mi[o], szq[o], None if l0 else Z[o], l1[o], None if G is None else G[o], Yo, fit[o])
        out = np.full(Ycols.shape, np.nan)
        out[o] = r
        return out, sk

    # old judge: t+1 open buy, hold 20 days (08 labels), L3 twins on all eligible rows (08b style)
    log("old-judge residuals")
    base = pn["eligible"].to_numpy(bool) & (pn["trade_date"].to_numpy() >= "20160101") & (pn["szq"].to_numpy() >= 1) \
        & np.isfinite(pn[["ret_20", "crash_20"] + L2_COLS].to_numpy(float)).all(1)
    brow = np.flatnonzero(base)
    bmi = pn["mi"].to_numpy()[brow]
    bo = np.argsort(bmi, kind="stable")
    Yb = pn[["ret_20"]].to_numpy(float)[brow][bo]
    rb, _ = resid_day(bmi[bo], pn["szq"].to_numpy()[brow][bo], pn[L2_COLS].to_numpy(float)[brow][bo],
                      pd.factorize(pn["l1"].to_numpy()[brow])[0][bo], None, Yb, np.ones(len(brow), bool), 300)
    old_res = np.full(len(brow), np.nan)
    old_res[bo] = rb[:, 0] * 100
    old_sig = {k: v[brow] for k, v in sig_all.items()}

    L = ["=" * 120, "Study 09b · 第 2 步：交易账本重考核心买卖点（全 A 个股，开发段 2016-01 → 2024-10）", "=" * 120,
         f"面板行数 {info['rows']:,}（含 2015 预热）；市场日 {info['T']:,}；股票 {info['N']:,}；只读补读快照 {SNAPSHOT}",
         f"数据体检：涨跌停价缺失按前收推算 {info['lim_fallback']:,} 行；复权坐标核对不一致 {info['adj_mismatch'] * 100:.3f}%；"
         f"收盘涨停标记与补读涨停价一致 {info['upc_agree'] * 100:.2f}%；镜像自检不一致股票 {info['bad_mirror']}",
         f"合格信号 K 线（入池、R_plan ≥ 0.3 ATR、截尾 31 天）{M:,}；超额 = 形态单相对同日双胞胎随机单的每笔净 R（L3 + 几何控制，DK lag 30）"]
    key = mi.astype(np.int64) * 1000 + l1
    cnt_all = pd.Series(key).map(pd.Series(key).value_counts()).to_numpy()
    rows_out = []
    for side in (1, -1):
        p = "L" if side > 0 else "S"
        F = cache[f"{p}_main_filled"]
        union = np.zeros(M, bool)
        for k in MAIN_KEYS[side]:
            union |= sig[k]
        fit = F & ~union
        variants = ["main", "opt", "k2"] + (["t1i"] if side > 0 else [])
        Ycols = np.column_stack([cache[f"{p}_{v}_net"].astype(float) for v in variants]
                                + [cache[f"{p}_main_win"].astype(float), cache[f"{p}_main_gross"].astype(float)])
        log(f"residualize side={side:+d}")
        R_, sk = resid(Ycols, F, fit, G_full)
        res = {v: R_[:, j] for j, v in enumerate(variants)}
        res["win"], res["gross"] = R_[:, len(variants)], R_[:, len(variants) + 1]
        res["nok"], _ = resid(cache[f"{p}_main_net"].astype(float)[:, None], F, fit, G_full[:, :8])
        res["nok"] = res["nok"][:, 0]
        res["l0"], _ = resid(cache[f"{p}_main_net"].astype(float)[:, None], F, fit, None, l0=True)
        res["l0"] = res["l0"][:, 0]
        res["l0g"], _ = resid(cache[f"{p}_main_net"].astype(float)[:, None], F, fit, G_full, l0=True)
        res["l0g"] = res["l0g"][:, 0]
        Fo = cache[f"{p}_open_filled"]
        res["open"], _ = resid(cache[f"{p}_open_net"].astype(float)[:, None], Fo, Fo & ~union, G_full)
        res["open"] = res["open"][:, 0]
        Fc = F & np.isfinite(crash)
        res["crash"], _ = resid(crash[:, None] * 100, Fc, Fc & ~union, G_full)
        res["crash"] = res["crash"][:, 0]
        net, win, gross = (cache[f"{p}_main_{x}"].astype(float) for x in ("net", "win", "gross"))
        net_o, win_o = cache[f"{p}_opt_net"].astype(float), cache[f"{p}_opt_win"].astype(float)
        Fopt = cache[f"{p}_opt_filled"]
        why = cache[f"{p}_main_why"]
        trig = (why == s09.W_OK) | (why == s09.W_LIMIT)
        pool = F & ~union
        L.append(f"\n  —— {SIDE_NAME[side]}：随机池成交 {pool.sum():,} 笔，胜率 {win[pool].mean() * 100:.1f}%（乐观 {win_o[Fopt & ~union].mean() * 100:.1f}%），"
                 f"净 {net[pool].mean():+.3f}R（乐观 {net_o[Fopt & ~union].mean():+.3f}R），持有中位 {np.median(cache[f'{p}_main_hold'][pool]):.0f} 天，"
                 f"到期 {(cache[f'{p}_main_kind'][pool] == s09.K_TIME).mean() * 100:.1f}%，冻结 {cache[f'{p}_main_frozen'][pool].mean() * 100:.2f}%；"
                 f"触发后{'涨停买不进' if side > 0 else '跌停逃不掉'} {(why[trig & ~union] == s09.W_LIMIT).mean() * 100:.2f}%；截面跳过 {sk} 天")
        for fam, lab, ls, lc, ss, sc in FAMILIES:
            s_, c_ = (ls, lc) if side > 0 else (ss, sc)
            Sm = sig[s_]
            Cm = sig[c_] & ~Sm
            S = np.flatnonzero(Sm & F)
            Ci = np.flatnonzero(Cm & F)

            def tw(key, idx=S):
                b, t, n = b8.vs_twins(res[key], idx, mi)
                return b, t

            r = {"market": "STK", "family": fam, "side": side, "label": lab, "n_sig": int(Sm.sum()), "n_fill": len(S),
                 "fill_rate": float(F[Sm].mean()) if Sm.any() else np.nan,
                 "blocked": float((why[Sm & trig] == s09.W_LIMIT).mean()) if (Sm & trig).any() else np.nan,
                 "rand_blocked": float((why[trig & ~union] == s09.W_LIMIT).mean()),
                 "win": float(win[S].mean()), "win_opt": float(win_o[Sm & Fopt].mean()),
                 "gross": float(gross[S].mean()), "net": float(net[S].mean()), "net_opt": float(net_o[Sm & Fopt].mean()),
                 "rand_win": float(win[pool].mean()), "rand_net": float(net[pool].mean()),
                 "outside": float(cache[f"{p}_main_outside"][S].mean()), "rand_outside": float(cache[f"{p}_main_outside"][pool].mean())}
            r["ex"], r["ex_t"] = tw("main")
            for v in ["opt", "k2", "t1i", "nok", "open", "gross", "crash", "l0", "l0g"]:
                if v in res:
                    r[f"ex_{v}"], r[f"ex_{v}_t"] = tw(v, np.flatnonzero(Sm & (Fo if v == "open" else F)))
            r["ex_win"], r["ex_win_t"] = tw("win")
            r["vs_ctrl"], r["vs_ctrl_t"] = b8.vs_ctrl(res["main"], S, Ci, mi)
            r["n_ctrl"] = len(Ci)
            r["per"] = [tw("main", S[year[S] == y])[0] for y in range(2016, 2025)]
            r["halves"] = [tw("main", S[year[S] <= 2020])[0], tw("main", S[year[S] >= 2021])[0]]
            r["tiers"] = [tw("main", S[tier[S] == k])[0] for k in (1, 2, 3, 4)]
            # with-industry vs independent (same-day same-industry share of this signal among candidates >= 15%)
            ks = pd.Series(key[Sm])
            share = np.zeros(M)
            share[Sm] = ks.map(ks.value_counts()).to_numpy() / cnt_all[Sm]
            r["indep"] = tw("main", S[share[S] < 0.15])[0]
            r["withind"] = tw("main", S[share[S] >= 0.15])[0]
            So = np.flatnonzero(old_sig[s_])
            b, t, _ = b8.vs_twins(side * old_res, So, bmi)
            r["old"], r["old_t"], r["old_n"] = b, t, len(So)
            r["p"] = b8.p_two(r["ex_t"])
            rows_out.append(r)
        L += render_stock(side, [x for x in rows_out if x["side"] == side])

    # --- A-share 20 primary tests: SW (09) + stocks (09b)
    sw = pd.read_csv(SUMMARY_09)
    sw = sw[sw["market"] == "SW"].copy()
    prim = []
    for _, x in sw.iterrows():
        d = x.to_dict()
        d["per"] = [float(v) for v in str(d["per"]).split(";")]
        d["halves"] = [float(v) for v in str(d["halves"]).split(";")]
        prim.append(d)
    prim += rows_out
    q = b8.bh_fdr([r["p"] if np.isfinite(r["p"]) else 1.0 for r in prim])
    L += ["", "#" * 120, "# A 股 20 个主检验：BH-FDR 与预注册过关判定（行业来自 09，个股来自本脚本）", "#" * 120,
          f"  {'市场':<5s}{'形态':<6s}{'方向':<12s}{'超额净R':>9s}{'t':>7s}{'q':>7s}  "
          f"{'①显著':>5s}{'②幅度':>5s}{'③对照':>5s}{'④逐年/两段':>9s}{'⑤市值档':>6s}{'⑥乐观':>5s}{'⑦T+1':>5s}{'⑧踩雷':>5s}  结论"]
    for r, qq in zip(prim, q):
        r["q20"] = qq
        v = verdict(r, qq)
        r["verdict"] = v["pass"]
        mk = "行业" if r["market"] == "SW" else "个股"
        L.append(f"  {mk:<5s}{r['family']:<6s}{SIDE_NAME[int(r['side'])]:<12s}{s09.fmt(r['ex']):>9s}{s09.fmt(r['ex_t'], 2):>7s}{qq:>7.3f}  "
                 + "".join(f"{v[k]:>6s}" for k in ["stat", "size", "ctrl", "stab", "tier", "opt", "t1", "crash"])
                 + f"  {'过关 ✔' if v['pass'] else ('反向显著 ✘' if (r['ex_t'] <= -2 and qq <= 0.05) else '不过关')}")
    txt = "\n".join(L)
    REPORT_PATH.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    flat = []
    for r in rows_out:
        d = {k: v for k, v in r.items() if not isinstance(v, list)}
        for k in ["per", "halves", "tiers"]:
            d[k] = ";".join(f"{v:+.3f}" if np.isfinite(v) else "nan" for v in r[k])
        flat.append(d)
    pd.DataFrame(flat).to_csv(SUMMARY_PATH, index=False)
    log(f"done ({time.time() - t0:.0f}s) -> {REPORT_PATH}")


def _yes(b: bool | None) -> str:
    return "—" if b is None else ("✔" if b else "✘")


def verdict(r: dict, q: float) -> dict:
    side = int(r["side"])
    stock = r["market"] == "STK"
    per = [v for v in r["per"] if np.isfinite(v)]
    halves = r["halves"]
    out = {
        "stat": r["ex_t"] >= 2 and q <= 0.05,
        "size": r["ex"] >= 0.10 and (side < 0 or r["net"] > 0),
        "ctrl": np.isfinite(r.get("vs_ctrl", np.nan)) and r["vs_ctrl"] > 0,
        "stab": sum(v > 0 for v in per) >= 6 and all(np.isfinite(h) and h > 0 for h in halves),
        "tier": (sum(v > 0 for v in r["tiers"][:3] if np.isfinite(v)) >= 2) if stock else None,
        "opt": np.isfinite(r.get("ex_opt", np.nan)) and r["ex_opt"] > 0,
        "t1": (np.isfinite(r.get("ex_t1i", np.nan)) and r["ex_t1i"] > 0) if (stock and side > 0) else None,
        "crash": (np.isfinite(r.get("ex_crash", np.nan)) and r["ex_crash"] > 0) if (stock and side < 0) else None,
    }
    out["pass"] = all(v for v in out.values() if v is not None)
    return {k: (_yes(v) if k != "pass" else v) for k, v in out.items()}


def render_stock(side: int, rows: list[dict]) -> list[str]:
    f = s09.fmt
    L = [f"\n  【个股 · {SIDE_NAME[side]}】（R 为单位；超额 = 比同日双胞胎随机单多出的每笔净 R）",
         f"  {'形态':<14s}{'信号':>9s}{'成交':>9s}{'成交率':>7s}{'胜率 保守~乐观':>16s}{'随机胜率':>8s}{'净R 保守~乐观':>18s}"
         f"{'超额净R':>9s}{'t':>6s}{'胜率超额':>9s}{'不控收盘位':>10s}{'vs专属对照':>11s}{'t':>6s}"]
    for r in rows:
        L.append(f"  {r['family'] + ' ' + r['label']:<14s}{r['n_sig']:>9,}{r['n_fill']:>9,}{r['fill_rate'] * 100:>6.0f}%"
                 f"{r['win'] * 100:>9.1f}%~{r['win_opt'] * 100:.1f}%{r['rand_win'] * 100:>7.1f}%"
                 f"{f(r['net']):>10s}~{f(r['net_opt'])}{f(r['ex']):>9s}{f(r['ex_t'], 1):>6s}{f(r['ex_win'], pct=True):>9s}"
                 f"{f(r['ex_nok']):>10s}{f(r['vs_ctrl']):>11s}{f(r['vs_ctrl_t'], 1):>6s}")
    L.append(f"  {'诊断':<8s}{'乐观超额':>9s}{'2R':>8s}{'开盘进场':>9s}{'T+1忽略':>9s}{'毛R超额':>9s}{'20天踩雷差':>10s}"
             f"{'外包线 形态/随机':>17s}{'买不进/逃不掉 形态/随机':>22s}{'独立/随行业':>15s}{'旧裁判(20天%)':>13s}{'t':>6s}")
    for r in rows:
        L.append(f"  {r['family']:<8s}{f(r['ex_opt']):>9s}{f(r['ex_k2']):>8s}{f(r['ex_open']):>9s}{f(r.get('ex_t1i', np.nan)):>9s}"
                 f"{f(r['ex_gross']):>9s}{f(r['ex_crash'], 2):>9s}pp{r['outside'] * 100:>9.1f}%/{r['rand_outside'] * 100:.1f}%"
                 f"{r['blocked'] * 100:>13.2f}%/{r['rand_blocked'] * 100:.2f}%{f(r['indep']):>9s}/{f(r['withind'])}"
                 f"{f(r['old'], 2):>12s}{f(r['old_t'], 1):>6s}")
    L.append("  优势拆解（超额净R）：只扣同日同市值档 → + 信号 K 线几何 → + 趋势 / 波动 / 热度 / 行业（主规格）")
    for r in rows:
        L.append(f"  {r['family']:<6s} 裸 {f(r['ex_l0'])}（t {f(r['ex_l0_t'], 1)}） → +几何 {f(r['ex_l0g'])}（t {f(r['ex_l0g_t'], 1)}）"
                 f" → 主规格 {f(r['ex'])}（t {f(r['ex_t'], 1)}）")
    L.append("  稳定性：2016…2024 逐年方向 | 前后两段 | 市值档 沪深300 / 中证500 / 中证1000 / 其余小微")
    for r in rows:
        per = "".join("+" if (np.isfinite(v) and v > 0) else ("−" if np.isfinite(v) else "·") for v in r["per"])
        npos = sum(1 for v in r["per"] if np.isfinite(v) and v > 0)
        L.append(f"  {r['family']:<6s} {per:<10s} 同正 {npos}/9 | {' / '.join(f(v) for v in r['halves'])} | "
                 + " / ".join(f(v) for v in r["tiers"]))
    return L


if __name__ == "__main__":
    main()
