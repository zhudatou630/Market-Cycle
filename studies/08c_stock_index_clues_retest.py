#!/usr/bin/env python3
"""
Market Cycle V2 — Study 08c（第二阶段方向 D · 保真赛）：
指数上跑出的 3 条 A 股硬线索，切到个股日线后还稳不稳？

主规格（开跑前定死，与 decisions.md 第 08 节预注册一致）：
  Q5 缩波蓄势突破：在 20 日首次新高（05 定义）当天，按「突破前 20 天波动」与「再往前 60 天波动」
     （两段不重叠：[t-20,t-1] 与 [t-80,t-21]，各按当天全市场入池股中位数切高低）分 2×2：
       一直安静（前低·近低） / 刚缩波（前高·近低） / 刚放波（前低·近高） / 一直躁动（前高·近高）
     主检验 = 「刚缩波」突破 vs 「一直安静」突破（两者近期都安静，差别只在之前是否躁动过 → 隔离出 Brooks 的「缩波蓄势」）。
  Q6 第 2 推 vs 第 3 推：06 的实时数推（盘中新高 + ≥1 ATR 且 ≥2 天的回调分隔），第 2 推新高当天 vs 第 3 推新高当天。
  Q7 周线休整 + 日线率先突破：07 的周线三态（上一完整自然周），日线 20 日首次新高（07 定义）
     在「周线非多头（震荡/空头）」 vs 「周线多头」。

裁判、双胞胎（L0~L3 分步剥离）、显著性、过关门槛全部沿用 08b（直接复用 08b 的代码）；
新增事件先在 31 个申万一级行业上与 06/07 原代码逐条比对一致后才上个股。
最后把 08b 的 8 个与本脚本 6 个正式检验合并做 BH-FDR（共 14 个）。
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
REPORT_PATH = OUTPUT_DIR / "08c_stock_index_clues_report.txt"
SUMMARY_PATH = OUTPUT_DIR / "08c_stock_index_clues_summary.csv"
SUMMARY_08B = OUTPUT_DIR / "08b_stock_revival_summary.csv"


def _load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "studies" / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


b8 = _load_mod("b8", "08b_stock_revival_structures.py")
s06, s07 = b8.s06, _load_mod("s07", "07_multi_timeframe_nested.py")
EPS, EVAL_START, log = b8.EPS, b8.EVAL_START, b8.log
vs_twins, vs_ctrl = b8.vs_twins, b8.vs_ctrl

CELL_NAMES = {0: "一直安静(前低·近低)", 1: "刚缩波(前高·近低)", 2: "刚放波(前低·近高)", 3: "一直躁动(前高·近高)"}
REG_NAMES = {1: "周线多头", 0: "周线震荡", -1: "周线空头"}


# ---------------------------------------------------------------------------
# 1. Frozen re-implementations: 06 push counting, 07 daily signals, 07 weekly regime
# ---------------------------------------------------------------------------
def detect_08c(h, l, c, dates, eval_start):
    n = len(c)
    out = {k: [] for k in ["p2", "p3", "p2_close", "p3_close"]}
    empty = {k: np.array([], dtype=int) for k in ["p2", "p3", "p2_close", "p3_close", "brk20", "dip"]}
    if n < 30:
        return empty
    em = np.asarray(pd.to_datetime(dates) >= eval_start, dtype=bool)
    prev_c = b8._shift(c, 1)
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr20 = pd.Series(tr).rolling(20, min_periods=20).mean().to_numpy()
    s_c = pd.Series(c)
    ma20 = s_c.rolling(20, min_periods=20).mean().to_numpy()
    ma10 = s_c.rolling(10, min_periods=10).mean().to_numpy()
    ph = pd.Series(h).rolling(20, min_periods=20).max().shift(1).to_numpy()
    pl = pd.Series(l).rolling(20, min_periods=20).min().shift(1).to_numpy()
    H, Lw, C, A, MA, PH, PL, EM = (x.tolist() for x in (h, l, c, atr20, ma20, ph, pl, em))
    fin = math.isfinite
    push = 0
    leg_lows: list[float] = []
    cur_peak, cur_idx, in_pb, pb_low, seq0 = -math.inf, -1, False, math.inf, -1
    for t in range(25, n):
        a = A[t]
        if not fin(a) or a <= EPS or not fin(PH[t]):
            continue
        if push == 0:
            if H[t] > PH[t] and C[t] > MA[t]:
                push, cur_peak, cur_idx, seq0, leg_lows, in_pb, pb_low = 1, H[t], t, t, [], False, Lw[t]
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
                if (cur_peak - pb_low >= a) and (t - cur_idx >= 2):
                    in_pb = True
        else:
            if Lw[t] < pb_low:
                pb_low = Lw[t]
            if H[t] > cur_peak:
                leg_lows.append(pb_low)
                push += 1
                prev_peak = cur_peak
                cur_peak, cur_idx, in_pb, pb_low = H[t], t, False, Lw[t]
                if EM[t] and push in (2, 3):
                    out[f"p{push}"].append(t)
                    if C[t] > prev_peak:
                        out[f"p{push}_close"].append(t)
    # 07 brk20 (close-based, cool 5) and daily pullback turn-up d_dip (cooldown 5 over all bars)
    pch = s_c.rolling(20, min_periods=20).max().shift(1).to_numpy()
    is_nh = c > pch
    rnh = pd.Series(is_nh.astype(float)).shift(1).rolling(5, min_periods=5).sum().to_numpy() > 0
    brk20 = is_nh & ~rnh & np.isfinite(pch)
    r5 = pd.Series(np.log(c)).diff(5).to_numpy()
    idx = np.arange(n)
    c1, c2, l1_, l2_, h1 = (b8._shift(x, k) for x, k in [(c, 1), (c, 2), (l, 1), (l, 2), (h, 1)])
    had = ((l1_ < l2_) | (c1 < c2)) & (np.fmin(l, l1_) <= ma10)
    cand = (idx >= 25) & np.isfinite(ma10) & np.isfinite(ma20) & np.isfinite(r5) & had & (h > h1) & (c > c1) & (c >= ma20 * 0.96)
    dip, last = [], -100
    for t in np.flatnonzero(cand).tolist():
        if t - last >= 5:
            dip.append(t)
            last = t
    dip = np.asarray(dip, dtype=int)
    res = {k: np.asarray(v, dtype=int) for k, v in out.items()}
    res["brk20"] = np.flatnonzero(brk20 & em)
    res["dip"] = dip[em[dip]] if len(dip) else dip
    return res


def weekly_regime(code: np.ndarray, date: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """07 weekly 3-state regime (W-FRI weeks), attached to each daily row from the previous completed week only.
    Returns float array: 1 bull / 0 range / -1 bear / nan unknown."""
    d = pd.DataFrame({"code": code, "date": pd.to_datetime(date), "h": h, "l": l, "c": c})
    d["wk"] = (d["date"] - pd.Timestamp("1970-01-03")).dt.days // 7  # Sat..Fri weeks == W-FRI
    wk = (d.groupby(["code", "wk"], sort=True)
          .agg(week_end_date=("date", "max"), wc=("c", "last")).reset_index())
    pos = wk.groupby("code").cumcount().to_numpy()
    wc = wk["wc"].to_numpy(float)
    ma20 = pd.Series(wc).rolling(20, min_periods=20).mean().to_numpy().copy()
    ma20[pos < 19] = np.nan
    lag4 = b8._shift(ma20, 4)
    lag4[pos < 4] = np.nan
    slope = np.where(lag4 > EPS, (ma20 - lag4) / np.where(lag4 > EPS, lag4, np.nan), np.nan)
    r12 = b8._shift(np.log(wc), 12)
    r12 = np.log(wc) - r12
    r12[pos < 12] = np.nan
    ok = np.isfinite(ma20) & np.isfinite(slope) & np.isfinite(r12)
    reg = np.where(ok, 0.0, np.nan)
    reg[ok & (wc > ma20) & (slope > 0) & (r12 > 0)] = 1.0
    reg[ok & (wc < ma20) & (slope < 0) & (r12 < 0)] = -1.0
    wk["reg"] = reg
    left = d[["code", "date"]].reset_index().sort_values("date")
    m = pd.merge_asof(left, wk[["code", "week_end_date", "reg"]].sort_values("week_end_date"),
                      left_on="date", right_on="week_end_date", by="code", direction="backward",
                      allow_exact_matches=False).set_index("index").sort_index()
    assert len(m) == len(d)
    return m["reg"].to_numpy(float)


def verify_08c(ix) -> list[str]:
    sw = ix.sw
    reg_all = weekly_regime(pd.factorize(sw["asset"])[0], sw["date"].to_numpy(), sw["high"].to_numpy(float),
                            sw["low"].to_numpy(float), sw["close"].to_numpy(float))
    bad, tot = 0, 0
    lines = []
    rmap = {1.0: "W_BULL", 0.0: "W_RANGE", -1.0: "W_BEAR"}
    for asset, s, e in ix.slices:
        g = sw.iloc[s:e][["asset", "date", "high", "low", "close"]].reset_index(drop=True)
        o6 = s06.build_asset_study06(g, "A股申万L1", EVAL_START)
        o7 = s07.build_asset_study07(g, "A股申万L1", EVAL_START)
        mine = detect_08c(g["high"].to_numpy(float), g["low"].to_numpy(float), g["close"].to_numpy(float), g["date"].to_numpy(), EVAL_START)
        d = pd.to_datetime(g["date"]).to_numpy()
        reg = reg_all[s:e]
        evalm = d >= np.datetime64(EVAL_START)
        validd = set(pd.to_datetime(d[evalm & np.isfinite(reg)]))
        checks = {
            "p2": (set(pd.to_datetime(o6["p2_top_brk"]["date"])), set(pd.to_datetime(d[mine["p2"]]))),
            "p3": (set(pd.to_datetime(o6["p3_top_brk"]["date"])), set(pd.to_datetime(d[mine["p3"]]))),
            "w_regime": (set(zip(pd.to_datetime(o7["date"]), o7["w_regime"])),
                         {(pd.Timestamp(x), rmap[r]) for x, r in zip(d[evalm & np.isfinite(reg)], reg[evalm & np.isfinite(reg)])}),
            "d_brk20": (set(pd.to_datetime(o7.loc[o7["d_brk20"], "date"])), set(pd.to_datetime(d[mine["brk20"]])) & validd),
            "d_dip": (set(pd.to_datetime(o7.loc[o7["d_dip"], "date"])), set(pd.to_datetime(d[mine["dip"]])) & validd),
            "d_push2": (set(pd.to_datetime(o7.loc[o7["d_push2"], "date"])), set(pd.to_datetime(d[mine["p2"]])) & validd),
            "d_push3": (set(pd.to_datetime(o7.loc[o7["d_push3"], "date"])), set(pd.to_datetime(d[mine["p3"]])) & validd),
        }
        for k, (ref, got) in checks.items():
            tot += len(ref)
            if ref != got:
                bad += 1
                lines.append(f"  不一致: {asset} {k} 原={len(ref)} 新={len(got)}")
    lines.insert(0, f"  校验 31 行业 × 7 类（06 数推、07 周线三态/日线突破/回调/二三推；原代码记录总数 {tot:,}）："
                    + ("全部一致 ✔" if bad == 0 else f"{bad} 处不一致 ✘"))
    assert bad == 0, "08c re-implementation deviates from frozen 06/07 definitions"
    return lines


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
def vol_windows(ret1: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(ret1)
    rec = s.rolling(20, min_periods=20).std(ddof=1).shift(1).to_numpy().copy()
    pri = s.rolling(60, min_periods=60).std(ddof=1).shift(21).to_numpy().copy()
    rec[pos < 21] = np.nan
    pri[pos < 81] = np.nan
    return rec, pri


def cells_by_day(mi: np.ndarray, rec: np.ndarray, pri: np.ndarray) -> np.ndarray:
    df = pd.DataFrame({"mi": mi, "rec": rec, "pri": pri})
    mr = df.groupby("mi")["rec"].transform("median").to_numpy()
    mp = df.groupby("mi")["pri"].transform("median").to_numpy()
    ok = np.isfinite(rec) & np.isfinite(pri) & np.isfinite(mr) & np.isfinite(mp)
    cell = (pri > mp).astype(float) + 2 * (rec > mr).astype(float)
    return np.where(ok, cell, np.nan)


def sel(rows: np.ndarray, flag_by_row: np.ndarray) -> np.ndarray:
    return rows[flag_by_row[rows]]


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    log("index side")
    ix = b8.IndexSide(verify=False)
    ver = verify_08c(ix)
    sw = ix.sw
    # index: extra events, weekly regime, vol cells, temperature
    log("index extras")
    sw_reg = weekly_regime(pd.factorize(sw["asset"])[0], sw["date"].to_numpy(), sw["high"].to_numpy(float),
                           sw["low"].to_numpy(float), sw["close"].to_numpy(float))
    iev: dict[str, list[np.ndarray]] = {}
    rec_i = np.full(len(sw), np.nan)
    pri_i = np.full(len(sw), np.nan)
    v20 = np.full(len(sw), np.nan)
    for asset, s, e in ix.slices:
        g = sw.iloc[s:e]
        hh, ll, cc = g["high"].to_numpy(float), g["low"].to_numpy(float), g["close"].to_numpy(float)
        for k, v in detect_08c(hh, ll, cc, g["date"].to_numpy(), EVAL_START).items():
            iev.setdefault(k, []).append(v + s)
        r, p = vol_windows(g["ret1"].to_numpy(float), np.arange(e - s))
        rec_i[s:e], pri_i[s:e] = r, p
        pc = b8._shift(cc, 1)
        tr = np.maximum.reduce([hh - ll, np.abs(hh - pc), np.abs(ll - pc)])
        a20 = pd.Series(tr).rolling(20, min_periods=20).mean()
        a80 = pd.Series(tr).rolling(80, min_periods=80).mean()
        v20[s:e] = (a20 / a80).to_numpy()
    iev = {k: np.concatenate(v) for k, v in iev.items()}
    cell_i = cells_by_day(ix.mi, rec_i, pri_i)
    sw["v20"] = v20
    sw["ind_pct"] = sw.groupby("asset")["v20"].transform(lambda x: x.rolling(250, min_periods=120).rank(pct=True))
    sw["mkt_pct"] = sw.groupby("mi")["ind_pct"].transform("mean")
    sw["wreg"] = sw_reg
    brk_i = ix.ev["brk_20"]

    log("stock side")
    st = b8.StockSide()
    pn = st.pn
    N = len(pn)
    pos = pn.groupby("code_id").cumcount().to_numpy()
    log("stock extras: pushes / 07 daily signals / weekly regime / vol windows")
    sev: dict[str, list[np.ndarray]] = {}
    for s, e in st.slices:
        for k, v in detect_08c(st.AH[s:e], st.AL[s:e], st.AC[s:e], st.dates[s:e], EVAL_START).items():
            sev.setdefault(k, []).append(v + s)
    sev = {k: np.concatenate(v) for k, v in sev.items()}
    wreg = weekly_regime(pn["code_id"].to_numpy(), st.dates, st.AH, st.AL, st.AC)
    rec, pri = vol_windows(pn["ret1"].to_numpy(float), pos)
    cell = np.full(N, np.nan)
    prow = st.base["prow"].to_numpy()
    cell[prow] = cells_by_day(st.bmi, rec[prow], pri[prow])
    # industry temperature & weekly regime mapped by (l1, date)
    sw_key = pd.DataFrame({"l1": sw["asset"].to_numpy(), "trade_date": sw["date"].dt.strftime("%Y%m%d").to_numpy(),
                           "ind_pct": sw["ind_pct"].to_numpy(), "mkt_pct": sw["mkt_pct"].to_numpy(), "ind_reg": sw_reg})
    bk = st.base[["l1", "trade_date"]].merge(sw_key, on=["l1", "trade_date"], how="left")
    assert len(bk) == len(st.base)
    cover = float(bk["ind_pct"].notna().mean() * 100)
    ind_pct_p = np.full(N, np.nan)
    mkt_pct_p = np.full(N, np.nan)
    ind_reg_p = np.full(N, np.nan)
    ind_pct_p[prow], mkt_pct_p[prow], ind_reg_p[prow] = bk["ind_pct"].to_numpy(), bk["mkt_pct"].to_numpy(), bk["ind_reg"].to_numpy()
    log(f"extras done ({time.time() - t0:.0f}s)")

    L: list[str] = []
    L.append("=" * 118)
    L.append("Study 08c · 方向 D 保真赛：指数上的 3 条 A 股硬线索，到个股还稳不稳（开发段 2016-01 → 2024-10，封存区零触碰）")
    L.append(f"入池且可评估的股票日: {len(st.base):,} | 行业温度/行业周线可映射覆盖: {cover:.1f}%")
    L.append("=" * 118)
    L.append("\n[0] 形态定义冻结校验（快速实现 vs 06/07 原代码，在申万一级行业上逐条比对；05 事件沿用 08b 已校验实现）")
    L.extend(ver)

    brk = st.ev["brk_20"]
    nonbull = np.isin(wreg, [0.0, -1.0])
    bull = wreg == 1.0
    nonbull_i = np.isin(sw_reg, [0.0, -1.0])
    bull_i = sw_reg == 1.0
    Q = [
        ("Q5", "缩波蓄势突破", sel(brk, cell == 1), sel(brk, cell == 0), sel(brk_i, cell_i == 1), sel(brk_i, cell_i == 0), +1,
         "20日首次新高当天：之前躁动、最近 20 天才安静下来（刚缩波） vs 一直都安静", "brk_20&刚缩波", "brk_20&一直安静"),
        ("Q6", "第2推 vs 第3推", sev["p2"], sev["p3"], iev["p2"], iev["p3"], +1,
         "洗盘回调后第 2 次创波段新高当天 vs 第 3 次创波段新高当天", "push2", "push3"),
        ("Q7", "周线休整+日线突破", sel(sev["brk20"], nonbull), sel(sev["brk20"], bull), sel(iev["brk20"], nonbull_i),
         sel(iev["brk20"], bull_i), +1, "周线震荡/空头时日线率先创 20 日新高 vs 周线多头时创 20 日新高", "brk20&周线非多头", "brk20&周线多头"),
    ]
    rows, pvals = [], []
    for q, title, S, C, Si, Ci, bdir, desc, tk, ck in Q:
        lines, row = b8.score_question(st, ix, q, title, S, C, Si, Ci, bdir, desc, tk, ck)
        L.extend(lines)
        rows.append(row)
        pvals += [row["p_ret"], row["p_cr"]]

    # ---- combined BH over 14 formal tests (08b 8 + 08c 6)
    prev = pd.read_csv(SUMMARY_08B).to_dict("records")
    allrows = prev + rows
    allp = []
    for r in allrows:
        allp += [r["p_ret"], r["p_cr"]]
    L.append("\n" + "=" * 118)
    L.append("【预注册判定 · 方向 D 全部 14 个正式检验合并 FDR】08b 复活赛 Q1–Q4 + 08c 保真赛 Q5–Q7，主规格 L3")
    L.append("=" * 118)
    L.extend(b8.verdict_lines(allrows, b8.bh_fdr(allp)))

    # ---- diagnostics
    R, bmi = st.R, st.bmi
    L.append("\n" + "=" * 118)
    L.append("【诊断（不参与判定）】L3 口径 vs 双胞胎：20天收益差(%) / 踩雷差(pp) / 40天收益差(%)")
    L.append("=" * 118)
    L.append("  [Q5-a] 20日首次新高当天的四种波动状态")
    for k in range(4):
        L.append(b8.diag_line(st, "Q5", sel(brk, cell == k), "突破·" + CELL_NAMES[k]))
    up = np.unique(np.concatenate([st.ev["tc"], st.ev["mid"], st.ev["bc"]]))
    L.append("  [Q5-b] 上升段（06 定义）日常持有的四种波动状态（「刚放波」= 高位放波）")
    for k in range(4):
        L.append(b8.diag_line(st, "Q5", sel(up, cell == k), "上升段·" + CELL_NAMES[k]))
    L.append("  [Q5-c] 行业层发烧（申万一级 ATR20/ATR80 在自身过去 250 天的分位；扣掉个股特征但不扣行业 = L2 口径）")
    Su = st.B(up)
    hot = Su[ind_pct_p[st.base["prow"].to_numpy()[Su]] >= 2 / 3]
    cold = Su[ind_pct_p[st.base["prow"].to_numpy()[Su]] <= 1 / 3]
    a = vs_ctrl(R("L2_ret_20"), hot, cold, bmi)
    c_ = vs_ctrl(R("L2_crash_20"), hot, cold, bmi)
    L.append(f"    上升段 · 行业发烧(前1/3) vs 行业安静(后1/3): 20天 {a[0]:+.2f}% (t={a[1]:+.1f}) | 踩雷 {c_[0]:+.1f}pp (t={c_[1]:+.1f}) | n={len(hot):,}/{len(cold):,}")
    L.append("  [Q5-d] 大盘层发烧（31 行业温度分位的当日平均；择时口径 = 绝对收益，控铁律）")
    mk = mkt_pct_p[st.base["prow"].to_numpy()[Su]]
    mh, mc = Su[mk >= 2 / 3], Su[mk <= 1 / 3]
    ab = st.base["ret_20"].to_numpy() * 100
    crb = st.base["crash_20"].to_numpy() * 100
    a = vs_ctrl(ab, mh, mc, bmi, st.Xl1)
    c_ = vs_ctrl(crb, mh, mc, bmi, st.Xl1)
    L.append(f"    上升段 · 大盘发烧日 vs 大盘安静日: 20天绝对收益 {a[0]:+.2f}% (t={a[1]:+.1f}) | 踩雷 {c_[0]:+.1f}pp (t={c_[1]:+.1f}) | n={len(mh):,}/{len(mc):,}")
    hh_ = Su[(mk >= 2 / 3) & (ind_pct_p[st.base["prow"].to_numpy()[Su]] >= 2 / 3)]
    qq_ = Su[(mk <= 1 / 3) & (ind_pct_p[st.base["prow"].to_numpy()[Su]] <= 1 / 3)]
    a = vs_ctrl(ab, hh_, qq_, bmi, st.Xl1)
    c_ = vs_ctrl(crb, hh_, qq_, bmi, st.Xl1)
    L.append(f"    上升段 · 双层发烧 vs 双层安静: 20天绝对收益 {a[0]:+.2f}% (t={a[1]:+.1f}) | 踩雷 {c_[0]:+.1f}pp (t={c_[1]:+.1f}) | n={len(hh_):,}/{len(qq_):,}")
    # index-side cells for reference
    line = []
    for k in range(4):
        s_ = sel(brk_i, cell_i == k)
        tw = vs_twins(sw["R_rel_20"].to_numpy() * 100, s_[np.isfinite(sw["ret_20"].to_numpy()[s_])], ix.mi)
        line.append(f"{CELL_NAMES[k]} n={tw[2]} {tw[0]:+.2f}({tw[1]:+.1f})")
    L.append("  [Q5-e] 指数端 20 日突破四种波动状态（相对 31 行业，扣 r20/vol20/r5）: " + " | ".join(line))

    L.append("  [Q6] 数推变体")
    for k, lab in [("p2", "第2推新高当天"), ("p3", "第3推新高当天"), ("p2_close", "第2推且收盘站上前推高点"), ("p3_close", "第3推且收盘站上前推高点")]:
        L.append(b8.diag_line(st, "Q6", sev[k], lab))
    a = vs_ctrl(R("L3_ret_20"), st.B(sev["p2_close"]), st.B(sev["p3_close"]), bmi)
    c_ = vs_ctrl(R("L3_crash_20"), st.B(sev["p2_close"]), st.B(sev["p3_close"]), bmi)
    L.append(f"    收盘确认版 第2推 − 第3推: 20天 {a[0]:+.2f}% (t={a[1]:+.1f}) | 踩雷 {c_[0]:+.1f}pp (t={c_[1]:+.1f})")

    L.append("  [Q7] 周线背景变体")
    for rv in [1.0, 0.0, -1.0]:
        L.append(b8.diag_line(st, "Q7", sel(sev["brk20"], wreg == rv), f"{REG_NAMES[int(rv)]} + 日线20日新高"))
    for rv in [1.0, 0.0, -1.0]:
        allr = np.flatnonzero((wreg == rv) & st.elig)
        L.append(b8.diag_line(st, "Q7", allr, f"{REG_NAMES[int(rv)]} 随便哪天买"))
    bull_all = st.B(np.flatnonzero(bull & st.elig))
    for k, lab in [("dip", "日线回调抬头"), ("brk20", "日线20日新高"), ("p2", "日线第2推")]:
        s_ = st.B(sel(sev[k], bull))
        a = vs_ctrl(R("L3_ret_20"), s_, bull_all, bmi)
        c_ = vs_ctrl(R("L3_crash_20"), s_, bull_all, bmi)
        L.append(f"    周线多头 + {lab} vs 周线多头随便买: 20天 {a[0]:+.2f}% (t={a[1]:+.1f}) | 踩雷 {c_[0]:+.1f}pp (t={c_[1]:+.1f}) | n={len(s_):,}")
    nb_all = st.B(np.flatnonzero(nonbull & st.elig))
    s_ = st.B(sel(sev["brk20"], nonbull))
    a = vs_ctrl(R("L3_ret_20"), s_, nb_all, bmi)
    c_ = vs_ctrl(R("L3_crash_20"), s_, nb_all, bmi)
    L.append(f"    周线非多头 + 日线20日新高 vs 周线非多头随便买: 20天 {a[0]:+.2f}% (t={a[1]:+.1f}) | 踩雷 {c_[0]:+.1f}pp (t={c_[1]:+.1f})")
    a = vs_ctrl(R("L3_ret_20"), st.B(sel(sev["p2"], nonbull)), st.B(sel(sev["p2"], bull)), bmi)
    L.append(f"    第2推：周线非多头 vs 周线多头: 20天 {a[0]:+.2f}% (t={a[1]:+.1f})")
    L.append("  [Q7-b] 个股周线 × 行业周线（日线20日新高当天；不扣行业 = L2 口径，vs 双胞胎）")
    for sr, sl in [(False, "个股周线非多头"), (True, "个股周线多头")]:
        parts = []
        for ir, il in [(False, "行业周线非多头"), (True, "行业周线多头")]:
            m_ = (bull if sr else nonbull) & (ind_reg_p == 1.0 if ir else np.isin(ind_reg_p, [0.0, -1.0]))
            s_ = st.B(sel(sev["brk20"], m_))
            tw = vs_twins(R("L2_ret_20"), s_, bmi)
            cr = vs_twins(R("L2_crash_20"), s_, bmi)
            parts.append(f"{il}: {tw[0]:+.2f}% (t={tw[1]:+.1f}) 雷 {cr[0]:+.1f} n={tw[2]:,}")
        L.append(f"    {sl:<10} | " + " | ".join(parts))

    report = "\n".join(L)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(SUMMARY_PATH, index=False)
    print(report)
    log(f"done in {time.time() - t0:.0f}s -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
