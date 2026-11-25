#!/usr/bin/env python3
"""
Market Cycle V2 — Study 12b（第三阶段专题 4 · 路径形状与目标位：全 A 个股 + A 股 20 个主检验 BH-FDR 收口）

预注册见 decisions.md「12-预注册」（方案全文 output/12_plan_for_review.md v2 冻结版）。
结构识别、路径裁判、零假设与 Q2 显式目标撮合全部直接调用 studies/12；账本调用未修改的 studies/09。
个股主规格（开跑前写死）：
  - 面板 = output/08_stock_panel_dev.parquet（≤ 2024-10-31）+ 09b.load_panel 只读补读涨跌停价与复权因子；
    候选行 = 2016 起、08 入池、z_t / z10_t > 0、L2_COLS & ret1 齐全、右端缓冲 31 个市场日（不经次日成交筛选）。
  - 双胞胎：每日截面回归，拟合只剔除本题真事件（不剔除普通对照），控同日市值五档 + L2 控制 + 当天涨跌（同日标准化 + 五分位）
    + 申万一级行业 + K 线形状（一字板涨停记 1 / 跌停记 0）+（Q2/Q3）目标距离 k（连续 + 五分位）与 d/C；
    DK lag 30，每天每组最多计 30 只，当天拟合池 < 100 只跳过；另报同日相减稳健性。
  - 主零假设：400 只抽样股票构建分层链式 5 天分块池（ATR20/C 五档 × 10%/20% 涨跌停制度；第 1 块匹配起始波动三档 × 涨跌方向 × 幅度）；
    双样本检验 t0 ≥ 2。
  - 合并 12 号行业 10 项 + 本脚本个股 10 项 = A 股 20 个主检验做 BH-FDR 并逐条判定过关。

用法：
  python studies/12b_path_targets_stocks.py --step0   # 建缓存 + 300 股自检 + 第 0.5 步校准 + 样本量 / MDE / 事前平衡（不看事件 Y）
  python studies/12b_path_targets_stocks.py           # 第 2 步全 A 个股结果 + A 股 20 个主检验总表
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
CACHE_PATH = OUTPUT_DIR / "12b_cache.npz"
STEP0_PATH = OUTPUT_DIR / "12b_step0_report.txt"
REPORT_PATH = OUTPUT_DIR / "12b_path_targets_stocks_report.txt"
SUMMARY_PATH = OUTPUT_DIR / "12b_path_targets_stocks_summary.csv"
SUMMARY_12 = OUTPUT_DIR / "12_path_targets_index_summary.csv"


def _load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "studies" / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


s12 = _load_mod("s12", "12_path_targets_index.py")
s11 = s12.s11
s10 = s12.s10
s09 = s12.s09
b8 = s12.b8
s09b = _load_mod("s09b", "09b_trade_ledger_stocks.py")
s10c = _load_mod("s10c", "10c_trading_range_explore.py")
s11b = _load_mod("s11b", "11b_level_memory_stocks.py")
log = b8.log

EPS = s12.EPS
STOCK_BUFFER = s09b.STOCK_BUFFER  # 31 market days
STOCK_BUFFER_40 = 51
MIN_DAY = s09b.MIN_DAY
L2_COLS = s09b.L2_COLS
PRIMARY = s12.PRIMARY
RACE_CODES = [p for p in PRIMARY if p[1] in ("Q2", "Q3a", "Q3b")]

_G: dict = {}


# ---------------------------------------------------------------------------
# 1. Two-pass parallel cache builder (8 workers, fork)
# ---------------------------------------------------------------------------
def _make_P1(T: int, mi: np.ndarray, o, h, l, c) -> dict:
    O, H, L, C = (np.full((T, 1), np.nan) for _ in range(4))
    OK = np.zeros((T, 1), bool)
    O[mi, 0], H[mi, 0], L[mi, 0], C[mi, 0], OK[mi, 0] = o, h, l, c, True
    return {"O": O, "H": H, "L": L, "C": C, "OK": OK}


def _worker_pass1(se):
    s, e = se
    T = _G["T"]
    O, H, L, C = _G["O"][s:e], _G["H"][s:e], _G["L"][s:e], _G["C"][s:e]
    MI, EM, CAND = _G["MI"][s:e], _G["EM"][s:e], _G["CAND"][s:e]
    P1 = _make_P1(T, MI, O, H, L, C)
    c_pos = np.flatnonzero(CAND)
    mi_c = MI[c_pos]
    nn_c = np.zeros(len(c_pos), np.int64)
    out = {}
    q2_drops = [0, 0]
    for sd, pre in [(1, "p"), (-1, "m")]:
        d = s12.detect_asset_side(H, L, C, EM, sd, with_diag=True)
        q2_drops[0 if sd > 0 else 1] = int(d["q2_drop_d"][0])
        for k in ["q1_ev", "q1_ctrl", "q1_mild", "q1_3b", "q2_ev", "q2_ctrl", "q2_raw_ev", "q2_mg_ev", "q2_leg_ev",
                  "q3a_ev", "q3a_all", "q3a_ctrl", "q3a_ctrl_dn", "q3b_ev", "q3b_in", "q3b_any", "q3b_ctrl"]:
            out[f"{pre}_{k}"] = d[k]
        for k in ["z", "z10", "atr_t", "q2_d", "q2_hgt", "q2_tgt_real", "q2_raw_d", "q2_mg_d", "q2_leg_d",
                  "q3a_d", "q3a_mid_real", "q3b_d", "q3b_ctrl_d", "ret1", "r5v"]:
            out[f"{pre}_{k}"] = d[k].astype(np.float32)
        if sd > 0:
            out["v_ratio"] = d["v_ratio"].astype(np.float32)
            out["atr_px"] = d["atr_px"].astype(np.float32)
        # Q1 outcomes on ALL candidate rows of this stock
        m_len = e - s
        for yk in ["y_tb", "y_tb_strict", "y_tb_c10", "surv5", "surv10", "surv20",
                   "y_tl", "y_tl_loose", "y_tl_cond", "l1_lag", "y_tb_atrt", "y_tl_atrt"]:
            out[f"{pre}_{yk}"] = np.full(m_len, np.nan, np.float32)
        out[f"{pre}_l1_hit"] = np.zeros(m_len, bool)
        if len(c_pos):
            q1r = s12.eval_q1_outcomes(P1, mi_c, nn_c, sd, d["z"][c_pos])
            for yk in ["y_tb", "y_tb_strict", "y_tb_c10", "surv5", "surv10", "surv20",
                       "y_tl", "y_tl_loose", "y_tl_cond", "l1_lag"]:
                out[f"{pre}_{yk}"][c_pos] = q1r[yk].astype(np.float32)
            out[f"{pre}_l1_hit"][c_pos] = q1r["l1_hit"]
            # ATR(t) sensitivity on Q1 event rows only (fast)
            ev1 = np.flatnonzero(d["q1_ev"] & CAND)
            if len(ev1):
                q1t = s12.eval_q1_outcomes(P1, MI[ev1], np.zeros(len(ev1), np.int64), sd, d["atr_t"][ev1])
                out[f"{pre}_y_tb_atrt"][ev1] = q1t["y_tb"].astype(np.float32)
                out[f"{pre}_y_tl_atrt"][ev1] = q1t["y_tl"].astype(np.float32)
    return s, e, out, q2_drops


def _worker_pass2(se):
    s, e = se
    T = _G["T"]
    O, H, L, C = _G["O"][s:e], _G["H"][s:e], _G["L"][s:e], _G["C"][s:e]
    MI, CAND = _G["MI"][s:e], _G["CAND"][s:e]
    P1 = _make_P1(T, MI, O, H, L, C)
    m_len = e - s
    c_pos = np.flatnonzero(CAND)
    mi_c = MI[c_pos]
    nn_c = np.zeros(len(c_pos), np.int64)
    rng = np.random.default_rng(12000 + s)
    out = {}

    for code, qg, sd, _, ek, ck, rrel in RACE_CODES:
        pre = "p" if sd > 0 else "m"
        rdir = sd * rrel
        z_key = f"{pre}_z10" if qg in ("Q2", "Q3a") else f"{pre}_z"
        d_col = f"{pre}_{'q2_d' if qg == 'Q2' else ('q3a_d' if qg == 'Q3a' else 'q3b_d')}"
        dc_col = f"{pre}_{'q2_d' if qg == 'Q2' else ('q3a_d' if qg == 'Q3a' else 'q3b_ctrl_d')}"
        z_u = _G[z_key][s:e].astype(np.float64)
        E = _G[f"{pre}_{ek}"][s:e] & CAND
        Cc = _G[f"{pre}_{ck}"][s:e] & CAND & ~E
        d_ev = _G[d_col][s:e].astype(np.float64)
        d_ct = _G[dc_col][s:e].astype(np.float64)
        k_pool = _G["k_ev"][code]

        k_arr = np.full(m_len, np.nan, np.float32)
        yc = np.full(m_len, np.nan, np.float32)
        yo = np.full(m_len, np.nan, np.float32)
        y0 = np.full(m_len, np.nan, np.float32)
        ycls = np.full(m_len, np.nan, np.float32)
        hd = np.full(m_len, np.nan, np.float32)
        p5 = np.full(m_len, np.nan, np.float32)
        bf = np.zeros(m_len, bool)

        if len(c_pos):
            kc = rng.choice(k_pool, size=len(c_pos), replace=True).astype(np.float64)
            em_c = E[c_pos]
            cm_c = Cc[c_pos]
            if em_c.any():
                kc[em_c] = d_ev[c_pos][em_c] / z_u[c_pos][em_c]
            if cm_c.any():
                kc[cm_c] = d_ct[c_pos][cm_c] / z_u[c_pos][cm_c]
            k_arr[c_pos] = kc.astype(np.float32)
            dc_all = kc * z_u[c_pos]
            rc = s12.eval_race(P1, mi_c, nn_c, rdir, dc_all)
            yc[c_pos] = rc["cons"].astype(np.float32)
            yo[c_pos] = rc["opt"].astype(np.float32)
            y0[c_pos] = rc["tie0"].astype(np.float32)
            ycls[c_pos] = rc["cls"].astype(np.float32)
            hd[c_pos] = rc["hit_day"].astype(np.float32)
            p5[c_pos] = rc["post5"].astype(np.float32)
            bf[c_pos] = rc["both"]

        out[f"{code}_k"] = k_arr
        out[f"{code}_cons"] = yc
        out[f"{code}_opt"] = yo
        out[f"{code}_tie0"] = y0
        out[f"{code}_cls"] = ycls
        out[f"{code}_hd"] = hd
        out[f"{code}_p5"] = p5
        out[f"{code}_both"] = bf

        # Question-specific event-only diagnostics
        if qg == "Q2":
            ev_i = np.flatnonzero(E)
            hgt = _G[f"{pre}_q2_hgt"][s:e].astype(np.float64)
            # Multiplier grid (5 values)
            my = np.full((m_len, 5), np.nan, np.float32)
            mp = np.full((m_len, 5), np.nan, np.float32)
            y40 = np.full(m_len, np.nan, np.float32)
            str_w = np.full(m_len, np.nan, np.float32)
            str_th = np.full(m_len, np.nan, np.float32)
            if len(ev_i):
                nn_e = np.zeros(len(ev_i), np.int64)
                mi_e = MI[ev_i]
                for jm, mult in enumerate(s12.Q2_MULTS):
                    dm = d_ev[ev_i] + (mult - 1.0) * hgt[ev_i]
                    okm = dm >= s12.D_MIN_ATR * z_u[ev_i] - s12.TIE
                    if okm.any():
                        rm = s12.eval_race(P1, mi_e[okm], nn_e[okm], rdir, dm[okm])
                        my[ev_i[okm], jm] = rm["cons"].astype(np.float32)
                        mp[ev_i[okm], jm] = rm["post5"].astype(np.float32)
                d_bot = 2.0 * hgt[ev_i] - d_ev[ev_i]
                okb = d_bot > EPS
                if okb.any():
                    rs = s12.eval_race(P1, mi_e[okb], nn_e[okb], rdir, d_ev[ev_i][okb], d_bot[okb])
                    str_w[ev_i[okb]] = (rs["cons"] == 1.0).astype(np.float32)
                    str_th[ev_i[okb]] = (d_bot[okb] / (d_ev[ev_i][okb] + d_bot[okb])).astype(np.float32)
            # 40-day horizon on all candidate rows within 51-day buffer (for proper twin regression)
            c40 = c_pos[mi_c + STOCK_BUFFER_40 <= T - 1]
            if len(c40):
                rc40 = s12.eval_race(P1, MI[c40], np.zeros(len(c40), np.int64), rdir,
                                     k_arr[c40].astype(np.float64) * z_u[c40],
                                     horizon=s12.WIN_SENS, min_days=s12.MIN_SENS, post_days=0)
                y40[c40] = rc40["cons"].astype(np.float32)
            out[f"{code}_my"] = my
            out[f"{code}_mp"] = mp
            out[f"{code}_y40"] = y40
            out[f"{code}_sw"] = str_w
            out[f"{code}_sth"] = str_th
            # Unconfirmed raw / Leg 1=Leg 2 / Measuring Gap on their own event rows
            for dk_nm, ek_nm, dc_nm in [("raw", "q2_raw_ev", "q2_raw_d"),
                                        ("leg", "q2_leg_ev", "q2_leg_d"),
                                        ("mg", "q2_mg_ev", "q2_mg_d")]:
                arr_d = np.full(m_len, np.nan, np.float32)
                di = np.flatnonzero(_G[f"{pre}_{ek_nm}"][s:e] & CAND)
                if len(di):
                    dd = _G[f"{pre}_{dc_nm}"][s:e][di].astype(np.float64)
                    rd = s12.eval_race(P1, MI[di], np.zeros(len(di), np.int64), rdir, dd, post_days=0)
                    arr_d[di] = rd["cons"].astype(np.float32)
                out[f"{code}_{dk_nm}"] = arr_d

        elif qg == "Q3a":
            # q3a_ctrl_dn & q3a_all diagnostics
            ydn = np.full(m_len, np.nan, np.float32)
            yall = np.full(m_len, np.nan, np.float32)
            ball = np.zeros(m_len, bool)
            di = np.flatnonzero(_G[f"{pre}_q3a_ctrl_dn"][s:e] & CAND)
            if len(di):
                rd = s12.eval_race(P1, MI[di], np.zeros(len(di), np.int64), rdir, d_ev[di], post_days=0)
                ydn[di] = rd["cons"].astype(np.float32)
            ai = np.flatnonzero(_G[f"{pre}_q3a_all"][s:e] & CAND)
            if len(ai):
                ra = s12.eval_race(P1, MI[ai], np.zeros(len(ai), np.int64), rdir, d_ev[ai], post_days=0)
                yall[ai] = ra["cons"].astype(np.float32)
                ball[ai] = ra["both"]
            out[f"{code}_ydn"] = ydn
            out[f"{code}_yall"] = yall
            out[f"{code}_ball"] = ball

    return s, e, out


def load_panel_12() -> tuple[pd.DataFrame, dict]:
    pn, info = s09b.load_panel()
    ex = pd.read_parquet(s09b.PANEL_PATH, columns=["code_id", "trade_date", "ret1"])
    ex = ex.sort_values(["code_id", "trade_date"]).reset_index(drop=True)
    assert np.array_equal(ex["code_id"].to_numpy(), pn["code_id"].to_numpy()) and np.array_equal(
        ex["trade_date"].to_numpy(), pn["trade_date"].to_numpy()), "panel alignment"
    pn["ret1_p"] = ex["ret1"].to_numpy(float)
    return pn, info


def build_cache() -> dict:
    t0 = time.time()
    pn, info = load_panel_12()
    NP, T = len(pn), info["T"]
    O, H, L, C = (pn[x].to_numpy(float) for x in ("a_open", "a_high", "a_low", "a_close"))
    MI = pn["mi"].to_numpy(np.int64)
    CID = pn["code_id"].to_numpy(np.int64)
    TD = pn["trade_date"].to_numpy()
    EM = TD >= "20160101"
    ELIG = pn["eligible"].to_numpy(bool)
    ok_l2 = np.isfinite(pn[L2_COLS].to_numpy(float)).all(1) & np.isfinite(pn["ret1_p"].to_numpy(float))
    CAND_BASE = ELIG & EM & (MI + STOCK_BUFFER <= T - 1) & ok_l2

    st = np.flatnonzero(np.r_[True, CID[1:] != CID[:-1]])
    en = np.r_[st[1:], NP]
    jobs = [(int(s), int(e)) for s, e in zip(st, en) if e - s >= 30]

    _G.clear()
    _G.update(T=T, O=O, H=H, L=L, C=C, MI=MI, EM=EM, CAND=CAND_BASE)

    log(f"Pass 1: features & Q1 outcomes for {len(jobs):,} stocks (8 workers)")
    data: dict[str, np.ndarray] = {}
    q2_drops_tot = [0, 0]
    with mp.get_context("fork").Pool(8) as pool:
        for idx_k, (s, e, out, q2d) in enumerate(pool.imap_unordered(_worker_pass1, jobs, chunksize=25)):
            q2_drops_tot[0] += q2d[0]
            q2_drops_tot[1] += q2d[1]
            for k, v in out.items():
                if k not in data:
                    data[k] = np.zeros(NP, bool) if v.dtype == bool else np.full(NP, np.nan, v.dtype)
                data[k][s:e] = v
            if idx_k % 1000 == 0:
                log(f"  Pass 1: {idx_k:,}/{len(jobs):,} stocks ({time.time() - t0:.0f}s)")
    info["q2_drops"] = tuple(q2_drops_tot)

    # Final candidate mask (also requires finite z and z10 > 0)
    CAND = CAND_BASE & np.isfinite(data["p_z"]) & (data["p_z"] > EPS) & np.isfinite(data["p_z10"]) & (data["p_z10"] > EPS)
    data["cand"] = CAND
    info["n_cand"] = int(CAND.sum())
    log(f"Pass 1 done ({time.time() - t0:.0f}s); cand rows {info['n_cand']:,}; running 300-stock self-checks")

    # Real-data self-checks on 300 random stocks (mirror, exact s10, exact s11.detect_A, truncation + K, t-1 probe)
    rng = np.random.default_rng(1212)
    pick = rng.choice(len(jobs), size=min(300, len(jobs)), replace=False)
    bad_m = bad_s11 = bad_s10 = bad_p = bad_q = 0
    for ii, ji in enumerate(pick.tolist()):
        s, e = jobs[ji]
        h, l, c, em = H[s:e], L[s:e], C[s:e], EM[s:e]
        a = s12.detect_asset_side(h, l, c, em, 1, with_diag=False)
        K = 2.0 * np.nanmax(h) + 7.0
        m = s12.detect_asset_side(K - l, K - h, K - c, em, -1, with_diag=False)
        for k in ["q1_ev", "q1_ctrl", "q2_ev", "q2_ctrl", "q3a_ev", "q3a_ctrl", "q3b_ev", "q3b_ctrl", "q3b_any"]:
            bad_m += int(not np.array_equal(a[k], m[k]))
        hv0, lv0, cv0, _ = s10.side_view(h, l, c, 1)
        hv, lv, cv = (np.round(x, s12.ROUND) for x in (hv0, lv0, cv0))
        ref11 = s11.detect_A(hv, lv, cv, s09.base_feats(hv, lv, cv)["atr20"], em)
        for k in ["A_narm", "A_toparm", "A_test"]:
            x, y = a[k], ref11[k]
            eq = np.array_equal(x, y) if x.dtype == bool else np.all((x == y) | (np.isnan(x) & np.isnan(y)))
            bad_s11 += int(not eq)
        ref10 = s10.asset_side(h, l, c, em, 1)
        bad_s10 += int(not np.array_equal(a["q3a_all"], ref10["q1_ev"]))
        if ii < 50 and e - s > 120:
            for sd in (1, -1):
                full = a if sd > 0 else s12.detect_asset_side(h, l, c, em, -1, with_diag=False)
                for tc in rng.choice(np.arange(80, e - s - 5), size=4, replace=False).tolist():
                    sub = s12.detect_asset_side(h[:tc + 1], l[:tc + 1], c[:tc + 1], em[:tc + 1], sd, with_diag=False)
                    for k in ["q1_ev", "q2_ev", "q3a_ev", "q3b_ev", "q3b_ctrl"]:
                        bad_p += int(sub[k][tc] != full[k][tc])
                    hd, ld, cd = (np.r_[v[:tc], v[tc - 1]] for v in (h, l, c))
                    dum = s12.detect_asset_side(hd, ld, cd, em[:tc + 1], sd, with_diag=False)
                    for k in ["A_narm", "A_toparm_real"]:
                        x, y = dum[k][tc], full[k][tc]
                        if not (abs(x - y) < 1e-5 or (math.isnan(x) and math.isnan(y))):
                            bad_q += 1
    info["checks"] = (len(pick), bad_m, bad_s11, bad_s10, bad_p, bad_q)
    assert bad_m == 0 and bad_s11 == 0 and bad_s10 == 0 and bad_p == 0 and bad_q == 0, f"stock self-check failed: {info['checks']}"
    log(f"300-stock self-checks passed ✔ ({time.time() - t0:.0f}s)")

    # Collect global event k distributions for Pass 2
    k_ev_map = {}
    for code, qg, sd, _, ek, _, _ in RACE_CODES:
        pre = "p" if sd > 0 else "m"
        z_u = data[f"{pre}_z10"] if qg in ("Q2", "Q3a") else data[f"{pre}_z"]
        d_col = f"{pre}_{'q2_d' if qg == 'Q2' else ('q3a_d' if qg == 'Q3a' else 'q3b_d')}"
        Em = data[f"{pre}_{ek}"] & CAND
        k_ev_map[code] = (data[d_col][Em].astype(np.float64) / z_u[Em].astype(np.float64))

    # Pass 2: race outcomes across all 5,300 stocks (8 workers)
    _G["CAND"] = CAND
    _G["k_ev"] = k_ev_map
    for pre in ("p", "m"):
        for k in ["z", "z10", "q2_ev", "q2_ctrl", "q2_d", "q2_hgt", "q2_raw_ev", "q2_raw_d",
                  "q2_mg_ev", "q2_mg_d", "q2_leg_ev", "q2_leg_d",
                  "q3a_ev", "q3a_all", "q3a_ctrl", "q3a_ctrl_dn", "q3a_d",
                  "q3b_ev", "q3b_ctrl", "q3b_d", "q3b_ctrl_d"]:
            _G[f"{pre}_{k}"] = data[f"{pre}_{k}"]

    log(f"Pass 2: race outcomes for {len(jobs):,} stocks (8 workers)")
    with mp.get_context("fork").Pool(8) as pool:
        for idx_k, (s, e, out) in enumerate(pool.imap_unordered(_worker_pass2, jobs, chunksize=25)):
            for k, v in out.items():
                if k not in data:
                    if v.ndim == 2:
                        data[k] = np.full((NP, v.shape[1]), np.nan, v.dtype)
                    else:
                        data[k] = np.zeros(NP, bool) if v.dtype == bool else np.full(NP, np.nan, v.dtype)
                data[k][s:e] = v
            if idx_k % 1000 == 0:
                log(f"  Pass 2: {idx_k:,}/{len(jobs):,} stocks ({time.time() - t0:.0f}s)")
    _G.clear()

    # Keep ONLY candidate rows (5.5M instead of 7.6M) + rows index for compact fast cache!
    rows = np.flatnonzero(CAND)
    cache: dict[str, np.ndarray] = {"rows": rows, "info12": np.array([repr(info)])}
    for k, v in data.items():
        if k == "cand":
            continue
        cache[k] = v[rows]
    np.savez(CACHE_PATH, **cache)
    log(f"cache saved ({len(rows):,} cand rows, {time.time() - t0:.0f}s) -> {CACHE_PATH}")
    return dict(np.load(CACHE_PATH, allow_pickle=False))


def load_cache() -> dict:
    return dict(np.load(CACHE_PATH, allow_pickle=False)) if CACHE_PATH.exists() else build_cache()


def build_stock_block_pools(pn: pd.DataFrame, cache: dict, info: dict) -> tuple[dict, dict, float, float]:
    """Build per-question 5-day & 1-day block pools from a deterministic 400-stock sample (5 ATR/C x 10%/20% limit)."""
    T = info["T"]
    rows_c = cache["rows"]
    NP = len(pn)
    cid = pn["code_id"].to_numpy(np.int64)
    ucids = np.unique(cid)
    rng = np.random.default_rng(1288)
    pick_c = np.sort(rng.choice(ucids, size=min(400, len(ucids)), replace=False))
    c_map = {int(c): i for i, c in enumerate(pick_c.tolist())}
    sel = np.isin(cid, pick_c)
    sub = pn.loc[sel]
    Ns = len(pick_c)
    mi = sub["mi"].to_numpy(np.int64)
    ci = np.array([c_map[int(x)] for x in sub["code_id"].to_numpy()], np.int64)

    def _w(v, fill=np.nan, dtype=float):
        W = np.full((T, Ns), fill, dtype=dtype)
        W[mi, ci] = v
        return W

    P_s = {
        "O": _w(sub["a_open"].to_numpy(float)), "H": _w(sub["a_high"].to_numpy(float)),
        "L": _w(sub["a_low"].to_numpy(float)), "C": _w(sub["a_close"].to_numpy(float)),
        "OK": _w(np.ones(len(sub), bool), False, bool),
    }
    elig_s = _w(sub["eligible"].to_numpy(bool) & (sub["trade_date"].to_numpy() >= "20160101"), False, bool)
    lim_s = _w((sub["lim_pct"].to_numpy(float) >= 0.19).astype(np.int8), 0, np.int8)

    # Map candidate-row event flags back onto full panel then into P_s
    pools5, pools1 = {}, {}
    for fam, ek in [("Q1", "q1_ev"), ("Q2", "q2_ev"), ("Q3a", "q3a_ev"), ("Q3b", "q3b_ev")]:
        ef = np.zeros(NP, bool)
        ef[rows_c] = cache[f"p_{ek}"] | cache[f"m_{ek}"]
        ex_s = _w(ef[sel], False, bool)
        pools5[fam] = s12.build_block_pool(P_s, elig_s, ex_s, blk_len=5, n_abins=5, n_vbins=3, split_mag=True, lim_bin=lim_s)
        pools1[fam] = s12.build_block_pool(P_s, elig_s, ex_s, blk_len=1, n_abins=5, n_vbins=3, split_mag=True, lim_bin=lim_s)

    r1_s = sub["ret1_p"].to_numpy(float)
    r1_s = r1_s[sub["eligible"].to_numpy(bool) & np.isfinite(r1_s)]
    return pools5, pools1, float(np.std(r1_s)), float(np.mean(r1_s))


def stock_event_keys(pool: dict, a: np.ndarray, v: np.ndarray, r1: np.ndarray,
                     z: np.ndarray, c: np.ndarray, lim_b: np.ndarray) -> list[tuple]:
    ab = np.digitize(a, pool["qa"])
    vb = np.digitize(v, pool["qv"])
    dr = (r1 >= 0).astype(int)
    # |C_t - C_{t-1}| >= 1.5 * z_t  <=>  |ret1| * C_{t-1} >= 1.5 * z_t
    c_prev = c / np.where(1.0 + r1 > EPS, 1.0 + r1, 1.0)
    mg = (np.abs(c - c_prev) >= 1.5 * z).astype(int)
    return [(int(lim_b[i]), int(ab[i]), int(vb[i]), int(dr[i]), int(mg[i])) for i in range(len(a))]


# ---------------------------------------------------------------------------
# 2. Step 0 on stocks (NO real event Y looked at!)
# ---------------------------------------------------------------------------
def step0() -> list[str]:
    cache = load_cache()
    info = eval(str(cache["info12"][0]))
    rows = cache["rows"]
    pn, _ = load_panel_12()
    pools5, _, _, _ = build_stock_block_pools(pn, cache, info)

    td = pn["trade_date"].to_numpy()[rows]
    year = np.array([int(x[:4]) for x in td])
    tier = pn["tier"].to_numpy()[rows]
    C_px = pn["a_close"].to_numpy(float)[rows]
    lim_b = (pn["lim_pct"].to_numpy(float)[rows] >= 0.19).astype(np.int8)
    r1_real = pn["ret1_p"].to_numpy(float)[rows]
    a_px = cache["atr_px"].astype(float)
    v_rat = cache["v_ratio"].astype(float)
    z_p = cache["p_z"].astype(float)

    chk = info["checks"]
    L = ["=" * 115, "Study 12b · 第 0 步：全 A 个股自检 + 第 0.5 步校准 + 样本量 / MDE / 事前平衡（不看任何真实事件 Y）", "=" * 115,
         f"面板 {info['rows']:,} 行；市场日 {info['T']:,}；股票 {info['N']:,}；入池候选行 {len(rows):,}（Q2 确认因 d<0.5 ATR 剔除 多/空={info['q2_drops'][0]:,}/{info['q2_drops'][1]:,}）",
         f"300 股自检：镜像 {chk[1]} / s11.detect_A {chk[2]} / s10.asset_side {chk[3]} / 截断(含K) {chk[4]} / t−1探针 {chk[5]} 不一致 ✔"]

    # Step 0.5 calibration on stocks (non-event random pool vs Markov null, NO real event Y!)
    non_ev = np.ones(len(rows), bool)
    for pre in ("p", "m"):
        for ek in ("q1_ev", "q2_ev", "q3a_ev", "q3b_ev"):
            non_ev &= ~cache[f"{pre}_{ek}"]
    rng = np.random.default_rng(1205)
    pick = rng.choice(np.flatnonzero(non_ev), size=min(20000, int(non_ev.sum())), replace=False)
    p5 = pools5["Q1"]
    keys_r = stock_event_keys(p5, a_px[pick], v_rat[pick], r1_real[pick], z_p[pick], C_px[pick], lim_b[pick])
    nq1 = s12.null_summary(p5, keys_r, None, "q1", 1, B_main=15000, n_boot=40, B_boot=1000, seed=55)
    # Compare Q3a_L random pool k distribution vs Q3a Markov null
    p5_3a = pools5["Q3a"]
    keys_3a = stock_event_keys(p5_3a, a_px[pick], v_rat[pick], r1_real[pick], z_p[pick], C_px[pick], lim_b[pick])
    k_3a = cache["Q3a_L_k"][pick].astype(float)
    n3a = s12.null_summary(p5_3a, keys_3a, k_3a, "race", 1, B_main=15000, n_boot=40, B_boot=1000, seed=56)
    L.append(f"  个股第 0.5 步校准（2万随机非事件日 vs 400股链式分块零假设，不看任何事件 Y）："
             f"TB {np.nanmean(cache['p_y_tb'][pick])*100:.1f}% vs {nq1['tb']*100:.1f}% | "
             f"TL {np.nanmean(cache['p_y_tl'][pick])*100:.1f}% vs {nq1['tl']*100:.1f}% | "
             f"Q3a_L 随机池赛跑 {np.nanmean(cache['Q3a_L_cons'][pick]):+.3f} vs {n3a['cons']:+.3f}")

    # Sample sizes, missingness, MDE, pre-event balance (NO real event Y!)
    upc = pn["up_close"].to_numpy(bool)[rows]
    dnc = pn["dn_close"].to_numpy(bool)[rows]
    H, Lo = pn["a_high"].to_numpy(float)[rows], pn["a_low"].to_numpy(float)[rows]
    kcls = np.where(H - Lo > EPS, (C_px - Lo) / (H - Lo), np.where(upc, 1.0, np.where(dnc, 0.0, 0.5)))

    for code, qg, sd, lab, ek, ck, _ in PRIMARY:
        pre = "p" if sd > 0 else "m"
        E = cache[f"{pre}_{ek}"]
        Cc = cache[f"{pre}_{ck}"] & ~E
        ne, nc = int(E.sum()), int(Cc.sum())
        # Missingness of Y on E (from suspension rule only, does NOT print Y values!)
        yk_chk = f"{pre}_y_tb" if qg == "Q1a" else (f"{pre}_y_tl" if qg == "Q1b" else f"{code}_cons")
        miss_r = float(np.mean(~np.isfinite(cache[yk_chk][E]))) * 100 if ne else 0.0
        sd_y = 0.5 if qg in ("Q1a", "Q1b") else 0.9
        mde = 2.0 * 1.8 * sd_y / math.sqrt(max(ne, 1))
        yc = " ".join(f"{y}:{int((E & (year == y)).sum())}" for y in range(2016, 2025))
        tc = "/".join(str(int((E & (tier == t_)).sum())) for t_ in (1, 2, 3, 4))
        r1s = cache[f"{pre}_ret1"]
        r5s = cache[f"{pre}_r5v"]
        kk = kcls if sd > 0 else (1.0 - kcls)
        extra = ""
        if qg in ("Q1a", "Q1b"):
            lim_sh = float(np.mean((upc if sd > 0 else dnc)[E])) * 100 if ne else 0.0
            extra = f" | 封涨/跌停占 {lim_sh:.1f}%"
        else:
            ke, kc = cache[f"{code}_k"][E], cache[f"{code}_k"][Cc]
            lo = max(float(np.nanquantile(ke, 0.05)), float(np.nanquantile(kc, 0.05)))
            hi = min(float(np.nanquantile(ke, 0.95)), float(np.nanquantile(kc, 0.95)))
            ov = float(np.mean((ke >= lo) & (ke <= hi))) * 100
            extra = f" | k中位 事{np.nanmedian(ke):.2f}/对{np.nanmedian(kc):.2f} 共同[5-95%]覆盖{ov:.0f}%"
            if qg == "Q3b":
                extra += f" | 本就在射程内(不计)={int(cache[f'{pre}_q3b_in'].sum()):,}"
        L.append(f"\n  {lab:<20s} 事 {ne:>7,} 对 {nc:>7,} | 停牌缺失 {miss_r:.2f}% | MDE≈{mde:.3f}{extra}")
        L.append(f"    逐年：{yc} | 市值300/500/1000/余：{tc}")
        L.append(f"    事前特征(事/对)：ret1 {np.nanmean(r1s[E])*100:+.2f}%/{np.nanmean(r1s[Cc])*100:+.2f}% | "
                 f"r5 {np.nanmean(r5s[E])*100:+.2f}%/{np.nanmean(r5s[Cc])*100:+.2f}% | "
                 f"kcls {np.nanmean(kk[E]):.2f}/{np.nanmean(kk[Cc]):.2f} | vol比 {np.nanmean(v_rat[E]):.2f}/{np.nanmean(v_rat[Cc]):.2f}")
    return L


# ---------------------------------------------------------------------------
# 3. Step 2: Full stock analysis + A-share 20-test BH-FDR收口
# ---------------------------------------------------------------------------
def main_analysis() -> None:
    t0 = time.time()
    cache = load_cache()
    info = eval(str(cache["info12"][0]))
    rows = cache["rows"]
    M = len(rows)
    pn, _ = load_panel_12()
    pools5, pools1, sig_day, drift_day = build_stock_block_pools(pn, cache, info)
    log(f"loaded cache ({M:,} cand rows) & block pools ({time.time() - t0:.0f}s)")

    # Cross-sectional controls (sorted by mi for fast s09b.resid_day)
    mi = pn["mi"].to_numpy(np.int64)[rows]
    cid = pn["code_id"].to_numpy(np.int64)[rows]
    td = pn["trade_date"].to_numpy()[rows]
    year = np.array([int(x[:4]) for x in td])
    tier = pn["tier"].to_numpy()[rows]
    l1 = pd.factorize(pn["l1"].to_numpy()[rows])[0]
    lmv = pn["lmv"].to_numpy(float)[rows]
    szq = np.ceil((pd.Series(lmv).groupby(mi).rank(pct=True, method="first").to_numpy() * 5).clip(1e-9, 5)).astype(int)
    Z_l2 = pn[L2_COLS].to_numpy(float)[rows]
    r1_real = pn["ret1_p"].to_numpy(float)[rows]
    Z_full = np.column_stack([Z_l2, r1_real])  # L2 + continuous ret1 (standardized within day by resid_day)
    r1q = s11b.day_quint(r1_real, mi)

    O_px, H_px, L_px, C_px = (pn[x].to_numpy(float)[rows] for x in ("a_open", "a_high", "a_low", "a_close"))
    upc = pn["up_close"].to_numpy(bool)[rows]
    dnc = pn["dn_close"].to_numpy(bool)[rows]
    lim_b = (pn["lim_pct"].to_numpy(float)[rows] >= 0.19).astype(np.int8)
    rng_bar = H_px - L_px
    z_p = cache["p_z"].astype(float)
    g_atr = np.where(z_p > EPS, rng_bar / z_p, 0.0)
    g_px = np.where(C_px > EPS, rng_bar / C_px, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kcls = np.where(rng_bar > EPS, (C_px - L_px) / rng_bar, np.where(upc, 1.0, np.where(dnc, 0.0, 0.5)))
    G_shape = np.hstack([s09b.qdum(g_atr, g_atr), s09b.qdum(g_px, g_px), s09b.qdum(kcls, kcls)])
    G_q1 = np.hstack([G_shape, r1q])

    a_px = cache["atr_px"].astype(float)
    v_rat = cache["v_ratio"].astype(float)
    order = np.argsort(mi, kind="stable")

    def resid(Y, mask_valid, fit_mask, G, Z_mat=Z_full, l0=False):
        Y = Y if Y.ndim == 2 else Y[:, None]
        o = order[mask_valid[order]]
        Zx = None if l0 else Z_mat[o]
        Gx = None if G is None else G[o]
        r, sk = s09b.resid_day(mi[o], szq[o], Zx, l1[o], Gx, Y[o], fit_mask[o], MIN_DAY)
        out = np.full(Y.shape, np.nan)
        out[o] = r
        return out, sk

    def tw(res, idx):
        m, t, _ = b8.vs_twins(res, np.flatnonzero(idx), mi)
        return float(m), float(t)

    def ctrl_cmp(res, idx_e, idx_c):
        m, t = b8.vs_ctrl(res, np.flatnonzero(idx_e), np.flatnonzero(idx_c), mi)
        return float(m), float(t)

    def day_diff_cmp(y_raw, idx_e, idx_c):
        m, t, _ = s10c.day_diff(mi, y_raw, idx_e, idx_c, 1)
        return float(m), float(t)

    # Wide market arrays ONLY for Q2 & Q3a ledger diagnostics (built once, freed after)
    P_wide = s09b.wide_market(pn, info["T"], info["N"])
    cost_all = np.where(td < s09b.STAMP_CUT, s09b.COST_PRE, s09b.COST_POST)
    del pn

    rows_out = []
    for code, qg, sd, lab, ek, ck, rrel in PRIMARY:
        t_q = time.time()
        pre = "p" if sd > 0 else "m"
        fam = "Q1" if qg in ("Q1a", "Q1b") else qg
        pool5, pool1 = pools5[fam], pools1[fam]
        E = cache[f"{pre}_{ek}"]
        Cc = cache[f"{pre}_{ck}"] & ~E
        ne, nc = int(E.sum()), int(Cc.sum())
        rec: dict = {"market": "STOCK", "code": code, "qg": qg, "side": sd, "label": lab, "n_ev": ne, "n_ctrl": nc}

        if qg in ("Q1a", "Q1b"):
            yk = f"{pre}_y_tb" if qg == "Q1a" else f"{pre}_y_tl"
            y = cache[yk].astype(float)
            val = np.isfinite(y)
            fit = val & ~E  # fitting pool excludes ONLY true events!
            # Main residual + decomposition (L0 -> L3 no shape/ret1 -> L3+shape -> Main)
            R_main, _ = resid(y, val, fit, G_q1)
            r_m = R_main[:, 0]
            r_l0 = resid(y, val, fit, None, l0=True)[0][:, 0]
            r_l3 = resid(y, val, fit, None, Z_mat=Z_l2)[0][:, 0]
            r_l3g = resid(y, val, fit, G_shape, Z_mat=Z_l2)[0][:, 0]

            ex, ex_t = tw(r_m, E)
            ex_l0, _ = tw(r_l0, E)
            ex_l3, _ = tw(r_l3, E)
            ex_l3g, _ = tw(r_l3g, E)
            ctrl_d, ctrl_t = ctrl_cmp(r_m, E, Cc)
            dd_d, dd_t = day_diff_cmp(y, E, Cc)
            # Auxiliary mild new-high control (on r_l3 WITHOUT ret1 control)
            C_mild = cache[f"{pre}_q1_mild"] & ~E
            mild_d, mild_t = ctrl_cmp(r_l3, E, C_mild)
            # Non-overlap
            E_no = s12.non_overlap_mask(mi, cid, E, None, s12.WIN_TB if qg == "Q1a" else s12.WIN_MAIN)
            ex_no, ex_no_t = tw(r_m, E_no)
            # 3-bar climax & limit-up/down split
            E3 = cache[f"{pre}_q1_3b"]
            ex_3b, ex_3b_t = tw(r_m, E3)
            lim_f = upc if sd > 0 else dnc
            ex_lim1, _ = tw(r_m, E & lim_f)
            ex_lim0, _ = tw(r_m, E & ~lim_f)
            # ATR(t) sensitivity on E
            y_atrt = cache[f"{pre}_{'y_tb_atrt' if qg == 'Q1a' else 'y_tl_atrt'}"].astype(float)
            mu_atrt = float(np.nanmean(y_atrt[E]))

            # Raw event mean & DK SE vs Markov 5-day null
            mu_e, se_e = s12._dk_mean_se(y[E], mi[E])
            # Subsample event keys up to 20,000 for speed (exact empirical distribution)
            ei = np.flatnonzero(E & val)
            if len(ei) > 20000:
                ei = np.random.default_rng(121).choice(ei, size=20000, replace=False)
            ekeys = stock_event_keys(pool5, a_px[ei], v_rat[ei], r1_real[ei], z_p[ei], C_px[ei], lim_b[ei])
            n5 = s12.null_summary(pool5, ekeys, None, "q1", sd, B_main=20000, n_boot=60, B_boot=1500, seed=120 + sd)
            n1 = s12.null_summary(pool1, ekeys, None, "q1", sd, B_main=10000, n_boot=20, B_boot=1000, seed=220 + sd)
            rw = s12.rw256_null(None, "q1", sd, sig_day, drift_day, seed=320 + sd)
            kn = "tb" if qg == "Q1a" else "tl"
            mu0, se0, mu0_m = n5[kn], n5[kn + "_se"], n5[kn + "_mart"]
            t0 = (mu_e - mu0) / math.sqrt(se_e ** 2 + se0 ** 2 + EPS)

            # Stability: 9 years, 2 halves, 4 size tiers
            per_y = [tw(r_m, E & (year == yr))[0] if (E & (year == yr)).sum() >= 10 else np.nan for yr in range(2016, 2025)]
            halves = [tw(r_m, E & (year >= a_) & (year <= b_))[0] for a_, b_ in [(2016, 2020), (2021, 2024)]]
            tiers = [tw(r_m, E & (tier == t_))[0] for t_ in (1, 2, 3, 4)]

            rec.update(dict(
                mu_e=mu_e, se_e=se_e, mu0=mu0, mu0_mart=mu0_m, t0=t0, mu_iid=n1[kn],
                mu_rw_m=rw["mart"][kn], mu_rw_d=rw["drift"][kn],
                ex=ex, ex_t=ex_t, ex_l0=ex_l0, ex_l3=ex_l3, ex_l3g=ex_l3g,
                ctrl=ctrl_d, ctrl_t=ctrl_t, dd=dd_d, dd_t=dd_t, mild=mild_d, mild_t=mild_t,
                ex_no=ex_no, ex_no_t=ex_no_t, ex_3b=ex_3b, ex_3b_t=ex_3b_t, n_3b=int(E3.sum()),
                ex_lim1=ex_lim1, ex_lim0=ex_lim0, lim_share=float(lim_f[E].mean()),
                mu_atrt=mu_atrt, ex_opt=ex, ex_tie0=ex, both_rate=0.0, atr_adv=np.nan, blk1_share=np.nan,
                per=per_y, halves=halves, tiers=tiers,
            ))
            if qg == "Q1a":
                rec["tb_strict"] = float(np.nanmean(cache[f"{pre}_y_tb_strict"][E]))
                rec["tb_c10"] = float(np.nanmean(cache[f"{pre}_y_tb_c10"][E]))
                rec["surv5"] = float(np.nanmean(cache[f"{pre}_surv5"][E]))
                rec["surv10"] = float(np.nanmean(cache[f"{pre}_surv10"][E]))
                rec["surv20"] = float(np.nanmean(cache[f"{pre}_surv20"][E]))
            else:
                yc = cache[f"{pre}_y_tl_cond"].astype(float)
                l1m = cache[f"{pre}_l1_hit"] & np.isfinite(yc)
                rc_m, _ = resid(yc, l1m, l1m & ~E, G_q1)
                cond_ex, cond_ex_t = tw(rc_m[:, 0], E & l1m)
                cond_e, cond_se = s12._dk_mean_se(yc[E & l1m], mi[E & l1m])
                cond_0, cond_se0 = n5["cond"], n5["cond_se"]
                cond_t0 = (cond_e - cond_0) / math.sqrt(cond_se ** 2 + cond_se0 ** 2 + EPS)
                lags = cache[f"{pre}_l1_lag"]
                lag_b = [float(np.nanmean(yc[E & l1m & (lags >= a_) & (lags <= b_)])) for a_, b_ in [(1, 5), (6, 10), (11, 20)]]
                rec.update(dict(
                    cond_e=cond_e, cond_0=cond_0, cond_t0=cond_t0, cond_ex=cond_ex, cond_ex_t=cond_ex_t,
                    cond_n=int((E & l1m).sum()), cond_lags=lag_b,
                    tl_loose=float(np.nanmean(cache[f"{pre}_y_tl_loose"][E])),
                    tbtl_joint=float(np.nanmean(((cache[f"{pre}_y_tb"] == 1) & (cache[f"{pre}_y_tl"] == 1))[E])),
                ))

        else:
            rdir = sd * rrel
            z_u = cache[f"{pre}_z10"].astype(float) if qg in ("Q2", "Q3a") else z_p
            k_all = cache[f"{code}_k"].astype(float)
            d_px = (k_all * z_u) / C_px
            y = cache[f"{code}_cons"].astype(float)
            yo = cache[f"{code}_opt"].astype(float)
            y0 = cache[f"{code}_tie0"].astype(float)
            ycls = cache[f"{code}_cls"].astype(float)
            y_atr = y * k_all

            val = np.isfinite(y) & np.isfinite(k_all) & np.isfinite(d_px)
            if qg == "Q3b":
                val &= (E | Cc | (~cache[f"{pre}_q3b_any"]))
            fit = val & ~E  # exclude ONLY true events!

            # Controls: L2 + ret1 + k + d/C (continuous in Z), shape + ret1q + kq (dummies in G)
            Z_race = np.column_stack([Z_full, k_all, d_px])
            kq = s09b.qdum(k_all, k_all[fit])
            G_race = np.hstack([G_shape, r1q, kq])

            Y_stack = np.column_stack([y, yo, y0, ycls, y_atr])
            R_st, _ = resid(Y_stack, val, fit, G_race, Z_mat=Z_race)
            r_m, r_opt, r_t0, r_cls, r_atr = (R_st[:, j] for j in range(5))
            r_l0 = resid(y, val, fit, None, l0=True)[0][:, 0]
            r_l3 = resid(y, val, fit, kq, Z_mat=np.column_stack([Z_l2, k_all, d_px]))[0][:, 0]
            r_l3g = resid(y, val, fit, np.hstack([G_shape, kq]), Z_mat=np.column_stack([Z_l2, k_all, d_px]))[0][:, 0]

            ex, ex_t = tw(r_m, E)
            ex_opt, _ = tw(r_opt, E)
            ex_tie0, _ = tw(r_t0, E)
            ex_cls, _ = tw(r_cls, E)
            atr_adv, _ = tw(r_atr, E)
            ex_l0, _ = tw(r_l0, E)
            ex_l3, _ = tw(r_l3, E)
            ex_l3g, _ = tw(r_l3g, E)

            # Exclusive control (Q2 restricted to common k [5%, 95%])
            E_c, C_c = E, Cc
            if qg == "Q2":
                ke, kc = k_all[E], k_all[Cc]
                lo = max(float(np.nanquantile(ke, 0.05)), float(np.nanquantile(kc, 0.05)))
                hi = min(float(np.nanquantile(ke, 0.95)), float(np.nanquantile(kc, 0.95)))
                csup = (k_all >= lo) & (k_all <= hi)
                E_c, C_c = E & csup, Cc & csup
            ctrl_d, ctrl_t = ctrl_cmp(r_m, E_c, C_c)
            dd_d, dd_t = day_diff_cmp(y, E_c, C_c)

            E_no = s12.non_overlap_mask(mi, cid, E, cache[f"{code}_hd"].astype(float), s12.WIN_MAIN)
            ex_no, ex_no_t = tw(r_m, E_no)

            mu_e, se_e = s12._dk_mean_se(y[E], mi[E])
            ei = np.flatnonzero(E & val)
            if len(ei) > 20000:
                ei = np.random.default_rng(122).choice(ei, size=20000, replace=False)
            ekeys = stock_event_keys(pool5, a_px[ei], v_rat[ei], r1_real[ei], z_p[ei], C_px[ei], lim_b[ei])
            ke_sub = k_all[ei]
            n5 = s12.null_summary(pool5, ekeys, ke_sub, "race", rdir, B_main=20000, n_boot=60, B_boot=1500, seed=420 + hash(code) % 100)
            n1 = s12.null_summary(pool1, ekeys, ke_sub, "race", rdir, B_main=10000, n_boot=20, B_boot=1000, seed=520 + hash(code) % 100)
            rw = s12.rw256_null(ke_sub, "race", rdir, sig_day, drift_day, seed=620 + hash(code) % 100)
            mu0, se0, mu0_m = n5["cons"], n5["se"], n5["mart"]
            t0 = (mu_e - mu0) / math.sqrt(se_e ** 2 + se0 ** 2 + EPS)

            per_y = [tw(r_m, E & (year == yr))[0] if (E & (year == yr)).sum() >= 10 else np.nan for yr in range(2016, 2025)]
            halves = [tw(r_m, E & (year >= a_) & (year <= b_))[0] for a_, b_ in [(2016, 2020), (2021, 2024)]]
            tiers = [tw(r_m, E & (tier == t_))[0] for t_ in (1, 2, 3, 4)]

            hd_e = cache[f"{code}_hd"][E].astype(float)
            p5_e = cache[f"{code}_p5"][E].astype(float)
            rec.update(dict(
                mu_e=mu_e, se_e=se_e, mu0=mu0, mu0_mart=mu0_m, t0=t0, mu_iid=n1["cons"],
                mu_rw_m=rw["mart"]["cons"], mu_rw_d=rw["drift"]["cons"], blk1_share=n5["blk1_share"],
                ex=ex, ex_t=ex_t, ex_opt=ex_opt, ex_tie0=ex_tie0, ex_cls=ex_cls, atr_adv=atr_adv,
                ex_l0=ex_l0, ex_l3=ex_l3, ex_l3g=ex_l3g, ctrl=ctrl_d, ctrl_t=ctrl_t, dd=dd_d, dd_t=dd_t,
                ex_no=ex_no, ex_no_t=ex_no_t,
                hit_tgt=float(np.nanmean(y[E] == 1.0)), hit_rev=float(np.nanmean(y[E] == -1.0)),
                unres=float(np.nanmean(y[E] == 0.0)), both_rate=float(np.nanmean(cache[f"{code}_both"][E])),
                surv5=float(np.nanmean(hd_e > 5)), surv10=float(np.nanmean(hd_e > 10)), surv20=float(np.nanmean(hd_e > 20)),
                post5=float(np.nanmean(p5_e)), per=per_y, halves=halves, tiers=tiers,
            ))

            # Question-specific diagnostics
            if qg == "Q2":
                my = cache[f"{code}_my"][E].astype(float)
                mp = cache[f"{code}_mp"][E].astype(float)
                m_ynet = [float(np.nanmean(my[:, j])) for j in range(5)]
                m_p5 = [float(np.nanmean(mp[:, j])) for j in range(5)]
                stall = m_p5[2] - 0.5 * (m_p5[1] + m_p5[3])
                # 40-day sensitivity (DK lag 40)
                y40 = cache[f"{code}_y40"].astype(float)
                v40 = np.isfinite(y40) & val
                r40, _ = resid(y40, v40, v40 & ~E, G_race, Z_mat=Z_race)
                s40 = np.flatnonzero(E & v40 & np.isfinite(r40[:, 0]))
                if len(s40) >= 30:
                    w40 = b8.cap_weights(mi[s40], np.zeros(len(s40)))
                    b40, t40 = b8.wls_dk(r40[s40, 0], np.ones((len(s40), 1)), w40, mi[s40], lag=s12.LAG_40)
                    ex40, ex40_t = float(b40[0]), float(t40[0])
                else:
                    ex40, ex40_t = np.nan, np.nan
                # Ledger diagnostic on Q2 events (match 09b R_plan >= 0.3*ATR filter; T+1 forced on long)
                t1_kw = "forced" if sd > 0 else None
                El = E & (rng_bar >= s09.R_MIN_ATR * z_p)
                l_mm = s12.run_stop_tgt_ledger(P_wide, mi[El], cid[El], sd, cache[f"{pre}_q2_tgt_real"][El].astype(float),
                                               t1=t1_kw, cost=cost_all[El], buffer=STOCK_BUFFER)
                l_1r = s09.run_ledger(P_wide, mi[El], cid[El], sd, k=1.0, t1=t1_kw, cost=cost_all[El], buffer=STOCK_BUFFER)
                rec.update(dict(
                    q2_mult_y=m_ynet, q2_mult_p5=m_p5, q2_stall=stall,
                    q2_raw_mu=float(np.nanmean(cache[f"{code}_raw"][cache[f"{pre}_q2_raw_ev"]])),
                    q2_raw_n=int(cache[f"{pre}_q2_raw_ev"].sum()),
                    q2_leg_mu=float(np.nanmean(cache[f"{code}_leg"][cache[f"{pre}_q2_leg_ev"]])),
                    q2_leg_n=int(cache[f"{pre}_q2_leg_ev"].sum()),
                    q2_mg_mu=float(np.nanmean(cache[f"{code}_mg"][cache[f"{pre}_q2_mg_ev"]])),
                    q2_mg_n=int(cache[f"{pre}_q2_mg_ev"].sum()),
                    q2_struct_win=float(np.nanmean(cache[f"{code}_sw"][E])),
                    q2_struct_theo=float(np.nanmean(cache[f"{code}_sth"][E])),
                    q2_ex40=float(ex40), q2_ex40_t=float(ex40_t),
                    led_mm_net=float(np.nanmean(l_mm["net"][l_mm["filled"]])) if l_mm["filled"].any() else np.nan,
                    led_1r_net=float(np.nanmean(l_1r["net"][l_1r["filled"]])) if l_1r["filled"].any() else np.nan,
                    led_mm_win=float(l_mm["win"][l_mm["filled"]].mean()) if l_mm["filled"].any() else np.nan,
                    led_1r_win=float(l_1r["win"][l_1r["filled"]].mean()) if l_1r["filled"].any() else np.nan,
                    led_mm_past=float((l_mm["why"] == s12.W_PAST_TGT).mean()),
                ))
            elif qg == "Q3a":
                C_dn = cache[f"{pre}_q3a_ctrl_dn"] & ~E
                rec["q3a_ctrl_dn_raw"] = float(np.nanmean(y[E]) - np.nanmean(cache[f"{code}_ydn"][C_dn])) if C_dn.sum() else np.nan
                tw_m = cache[f"{pre}_ret1"] < 0
                rec["q3a_tw"], rec["q3a_tw_t"] = ctrl_cmp(r_m, E & tw_m, Cc & tw_m)
                Ea = cache[f"{pre}_q3a_all"]
                rec["q3a_all_mu"] = float(np.nanmean(cache[f"{code}_yall"][Ea]))
                rec["q3a_all_both"] = float(np.nanmean(cache[f"{code}_ball"][Ea]))
                # Ledger diagnostic via UNTOUCHED s09.run_ledger(entry='limit', tgt_px=mid) vs fixed 1R (R_plan >= 0.3*ATR)
                t1_kw = "forced" if sd > 0 else None
                El = E & (rng_bar >= s09.R_MIN_ATR * z_p)
                l_mid = s09.run_ledger(P_wide, mi[El], cid[El], sd, entry="limit", tgt_px=cache[f"{pre}_q3a_mid_real"][El].astype(float),
                                       t1=t1_kw, cost=cost_all[El], buffer=STOCK_BUFFER)
                l_1r = s09.run_ledger(P_wide, mi[El], cid[El], sd, entry="limit", k=1.0, t1=t1_kw, cost=cost_all[El], buffer=STOCK_BUFFER)
                rec.update(dict(
                    led_mid_net=float(np.nanmean(l_mid["net"][l_mid["filled"]])) if l_mid["filled"].any() else np.nan,
                    led_1r_net=float(np.nanmean(l_1r["net"][l_1r["filled"]])) if l_1r["filled"].any() else np.nan,
                    led_mid_win=float(l_mid["win"][l_mid["filled"]].mean()) if l_mid["filled"].any() else np.nan,
                    led_1r_win=float(l_1r["win"][l_1r["filled"]].mean()) if l_1r["filled"].any() else np.nan,
                    led_mid_hold=float(np.nanmedian(l_mid["hold"][l_mid["filled"]])) if l_mid["filled"].any() else np.nan,
                    led_1r_hold=float(np.nanmedian(l_1r["hold"][l_1r["filled"]])) if l_1r["filled"].any() else np.nan,
                ))
            elif qg == "Q3b":
                # Deterministic 3-way matched (k 5 x r5 3 x v_ratio 3) vs distance-only matched
                r5s = cache[f"{pre}_r5v"].astype(float)
                qk = np.nanquantile(k_all[E], [0.2, 0.4, 0.6, 0.8])
                qr = np.nanquantile(r5s[E], [1 / 3, 2 / 3])
                qv = np.nanquantile(v_rat[E], [1 / 3, 2 / 3])
                bk = np.digitize(k_all, qk)
                cell = bk * 9 + np.digitize(r5s, qr) * 3 + np.digitize(v_rat, qv)
                rm = (~E) & (~cache[f"{pre}_q3b_any"]) & np.isfinite(y)
                dc_, wc_ = [], []
                for cid_ in np.unique(cell[E]):
                    me, mr = E & (cell == cid_) & np.isfinite(y), rm & (cell == cid_)
                    if me.sum() >= 10 and mr.sum() >= 50:
                        dc_.append(float(np.nanmean(y[me]) - np.nanmean(y[mr])))
                        wc_.append(int(me.sum()))
                dk_, wk_ = [], []
                for bid_ in range(5):
                    me, mr = E & (bk == bid_) & np.isfinite(y), rm & (bk == bid_)
                    if me.sum() >= 10 and mr.sum() >= 50:
                        dk_.append(float(np.nanmean(y[me]) - np.nanmean(y[mr])))
                        wk_.append(int(me.sum()))
                rec["q3b_det_match"] = float(np.average(dc_, weights=wc_)) if wc_ else np.nan
                rec["q3b_k_only"] = float(np.average(dk_, weights=wk_)) if wk_ else np.nan
                rec["q3b_vac_rev"] = float(np.nanmean(p5_e < 0)) if np.isfinite(p5_e).any() else np.nan

        rows_out.append(rec)
        log(f"  done {code} (n={ne:,}, ex={rec['ex']:+.4f}, t={rec['ex_t']:+.2f}, ctrl={rec['ctrl']:+.4f}, t0={rec['t0']:+.2f}, {time.time() - t_q:.0f}s)")

    del P_wide

    # ---------------- Combine SW (10) + STOCK (10) -> A-share 20-test BH-FDR verdict!
    sw_df = pd.read_csv(SUMMARY_12)
    sw_rows = sw_df[sw_df["market"] == "SW"].to_dict("records")

    def _p2(t_val):
        if not math.isfinite(t_val):
            return 1.0
        return float(math.erfc(abs(t_val) / math.sqrt(2.0)))

    comb = []
    for r in sw_rows:
        per = [float(x) if x != "nan" else np.nan for x in str(r["per"]).split(";")]
        hlv = [float(x) if x != "nan" else np.nan for x in str(r["halves"]).split(";")]
        comb.append(dict(market="SW", code=r["code"], qg=r["qg"], label="行业 " + r["label"],
                         ex=float(r["ex"]), ex_t=float(r["ex_t"]), t0=float(r["t0"]),
                         ctrl=float(r["ctrl"]), ctrl_t=float(r["ctrl_t"]),
                         ex_opt=float(r["ex_opt"]), ex_tie0=float(r["ex_tie0"]), ex_no=float(r["ex_no"]),
                         per=per, halves=hlv, tiers=None,
                         cond_t0=float(r.get("cond_t0", np.nan)), cond_ex=float(r.get("cond_ex", np.nan)),
                         cond_ex_t=float(r.get("cond_ex_t", np.nan)),
                         q3b_det=float(r.get("q3b_det_match", np.nan))))
    for r in rows_out:
        comb.append(dict(market="STOCK", code=r["code"], qg=r["qg"], label="个股 " + r["label"],
                         ex=r["ex"], ex_t=r["ex_t"], t0=r["t0"], ctrl=r["ctrl"], ctrl_t=r["ctrl_t"],
                         ex_opt=r["ex_opt"], ex_tie0=r["ex_tie0"], ex_no=r["ex_no"],
                         per=r["per"], halves=r["halves"], tiers=r["tiers"],
                         cond_t0=r.get("cond_t0", np.nan), cond_ex=r.get("cond_ex", np.nan),
                         cond_ex_t=r.get("cond_ex_t", np.nan),
                         q3b_det=r.get("q3b_det_match", np.nan)))

    pvals = np.array([_p2(x["ex_t"]) for x in comb])
    qvals = b8.bh_fdr(pvals)
    for x, q in zip(comb, qvals):
        x["q"] = float(q)
        is_q1 = x["qg"] in ("Q1a", "Q1b")
        thr_ex = 0.05 if is_q1 else 0.10
        thr_ct = 0.025 if is_q1 else 0.05
        nv = sum(1 for v in x["per"] if math.isfinite(v))
        np_ = sum(1 for v in x["per"] if math.isfinite(v) and v > 0)
        ok_yr = (nv >= 5) and (np_ >= math.ceil(nv * 2 / 3))
        ok_hlv = all(math.isfinite(v) and v > 0 for v in x["halves"])
        ok_tier = True if x["tiers"] is None else (sum(1 for v in x["tiers"][:3] if math.isfinite(v) and v > 0) >= 2)
        ok_bnd = (x["ex_opt"] > 0 and x["ex_tie0"] > 0 and x["ex_no"] > 0)
        ok_q1b = True
        if x["qg"] == "Q1b":
            ok_q1b = (math.isfinite(x["cond_ex"]) and x["cond_ex"] > 0 and math.isfinite(x["cond_t0"]) and x["cond_t0"] > 0
                      and (x["cond_ex_t"] >= 2.0 or x["cond_t0"] >= 2.0))
        ok_q3b = True if x["qg"] != "Q3b" else (math.isfinite(x["q3b_det"]) and x["q3b_det"] > 0)
        passed = (x["ex_t"] >= 2.0 and x["q"] <= 0.05 and x["t0"] >= 2.0 and x["ex"] >= thr_ex
                  and x["ctrl"] >= thr_ct and x["ctrl_t"] >= 2.0 and ok_yr and ok_hlv and ok_tier
                  and ok_bnd and ok_q1b and ok_q3b)
        if passed:
            x["verdict"] = "过关"
        elif x["ex_t"] <= -2.0 and x["q"] <= 0.05:
            x["verdict"] = "反向显著"
        else:
            x["verdict"] = "不过关"

    # ---------------- Render full report
    f_ = s09.fmt
    L = ["=" * 120, "Study 12b · 第 2 步：路径形状与目标位（全 A 个股，开发段 2016-01 → 2024-10）+ A 股 20 项主检验收口", "=" * 120,
         f"入池候选行 {M:,}；双胞胎 = 同日市值五档 + L2 + 当天涨跌 + 行业 + 形状 (+ k, d/C)，拟合池仅剔除真事件，DK lag 30（每天每组上限 30 只）。"]
    L.append(f"\n  {'命题':<22s}{'事件':>8s}{'对照':>8s}{'原始Y':>8s}{'零假设':>8s}{'t0':>6s}{'超额':>8s}{'t':>6s}{'专属对照':>9s}{'t':>6s}{'同日差':>8s}{'不重叠':>8s}{'乐观/记0':>14s}")
    for r in rows_out:
        is_q1 = r["qg"] in ("Q1a", "Q1b")
        if is_q1:
            L.append(f"  {r['label']:<20s}{r['n_ev']:>8,}{r['n_ctrl']:>8,}{r['mu_e']*100:>7.1f}%{r['mu0']*100:>7.1f}%{f_(r['t0'],1):>6s}"
                     f"{r['ex']*100:>+7.1f}pp{f_(r['ex_t'],1):>6s}{r['ctrl']*100:>+8.1f}pp{f_(r['ctrl_t'],1):>6s}{r['dd']*100:>+7.1f}pp{r['ex_no']*100:>+7.1f}pp{'—':>14s}")
        else:
            L.append(f"  {r['label']:<20s}{r['n_ev']:>8,}{r['n_ctrl']:>8,}{f_(r['mu_e']):>8s}{f_(r['mu0']):>8s}{f_(r['t0'],1):>6s}"
                     f"{f_(r['ex']):>8s}{f_(r['ex_t'],1):>6s}{f_(r['ctrl']):>9s}{f_(r['ctrl_t'],1):>6s}{f_(r['dd']):>8s}{f_(r['ex_no']):>8s}"
                     f"{f_(r['ex_opt']):>7s}/{f_(r['ex_tie0']):>6s}")
        per_s = "".join("+" if (math.isfinite(v) and v > 0) else ("−" if math.isfinite(v) else "·") for v in r["per"])
        nv = sum(1 for v in r["per"] if math.isfinite(v))
        np_ = sum(1 for v in r["per"] if math.isfinite(v) and v > 0)
        h_s = " / ".join(f_(v) for v in r["halves"])
        t_s = " / ".join(f_(v) for v in r["tiers"])
        if is_q1:
            L.append(f"      稳定 {per_s}({np_}/{nv}) 两段 {h_s} | 市值(300/500/1000/余) {t_s}")
            L.append(f"      拆解 L0 {r['ex_l0']*100:+.1f}→L3(无形状日涨) {r['ex_l3']*100:+.1f}→+形状 {r['ex_l3g']*100:+.1f}→主 {r['ex']*100:+.1f}pp | "
                     f"零假设(鞅{r['mu0_mart']*100:.1f}% IID{r['mu_iid']*100:.1f}% RW{r['mu_rw_m']*100:.1f}/{r['mu_rw_d']*100:.1f}%) | "
                     f"温和新高 {r['mild']*100:+.1f}pp(t{f_(r['mild_t'],1)}) | 3根 {r['ex_3b']*100:+.1f}pp(n{r['n_3b']:,}) | "
                     f"封板({r['lim_share']*100:.0f}%)/未封 {r['ex_lim1']*100:+.1f}/{r['ex_lim0']*100:+.1f}pp | ATR(t)原始 {r['mu_atrt']*100:.1f}%")
            if r["qg"] == "Q1a":
                L.append(f"      TB辅报：严格版(≤H_t) {r['tb_strict']*100:.1f}% | C10≤C0 {r['tb_c10']*100:.1f}% | 生存5/10/20d {r['surv5']*100:.0f}/{r['surv10']*100:.0f}/{r['surv20']*100:.0f}%")
            else:
                lb = "/".join(f"{x*100:.0f}%" if math.isfinite(x) else "·" for x in r["cond_lags"])
                L.append(f"      Q1b条件两段(固定10日,n={r['cond_n']:,})：事{r['cond_e']*100:.1f}% 零{r['cond_0']*100:.1f}%(t0={f_(r['cond_t0'],1)}) "
                         f"双胞胎{r['cond_ex']*100:+.1f}pp(t{f_(r['cond_ex_t'],1)}) | 分档 {lb} | 宽松TL {r['tl_loose']*100:.1f}% | TB∩TL {r['tbtl_joint']*100:.1f}%")
        else:
            L.append(f"      稳定 {per_s}({np_}/{nv}) 两段 {h_s} | 市值(300/500/1000/余) {t_s}")
            L.append(f"      拆解 L0 {f_(r['ex_l0'])}→L3(无形状日涨) {f_(r['ex_l3'])}→+形状 {f_(r['ex_l3g'])}→主 {f_(r['ex'])} 纯收 {f_(r['ex_cls'])} | ATR优势 {f_(r['atr_adv'])} | "
                     f"零假设(鞅{f_(r['mu0_mart'])} IID{f_(r['mu_iid'])} RW{f_(r['mu_rw_m'])}/{f_(r['mu_rw_d'])}) 第1块定输赢{r['blk1_share']*100:.0f}%")
            L.append(f"      路径：先目标{r['hit_tgt']*100:.1f}% 先反向{r['hit_rev']*100:.1f}% 未碰{r['unres']*100:.1f}% 双碰{r['both_rate']*100:.1f}% | "
                     f"生存5/10/20d {r['surv5']*100:.0f}/{r['surv10']*100:.0f}/{r['surv20']*100:.0f}% | 触后5d {r['post5']*100:+.2f}%")
            if r["qg"] == "Q2":
                my = "/".join(f_(x, 2) for x in r["q2_mult_y"])
                mp = "/".join(f"{x*100:+.2f}%" for x in r["q2_mult_p5"])
                L.append(f"      Q2诊断：倍数(0.6/0.8/1.0/1.2/1.4) Y={my} 触后5d={mp} 1.0x停顿={r['q2_stall']*100:+.2f}% | "
                         f"未确认 {f_(r['q2_raw_mu'])}(n{r['q2_raw_n']:,}) | Leg1=Leg2 {f_(r['q2_leg_mu'])}(n{r['q2_leg_n']:,}) | "
                         f"缺口 {f_(r['q2_mg_mu'])}(n{r['q2_mg_n']:,}) | 结构止损胜{r['q2_struct_win']*100:.1f}%(理论{r['q2_struct_theo']*100:.1f}%) | "
                         f"40d超额 {f_(r['q2_ex40'])}(t{f_(r['q2_ex40_t'],1)}) | 账本 MM净R {f_(r['led_mm_net'])}(胜{r['led_mm_win']*100:.0f}%) vs 1R {f_(r['led_1r_net'])}(胜{r['led_1r_win']*100:.0f}%) 跳过{r['led_mm_past']*100:.1f}%")
            elif r["qg"] == "Q3a":
                L.append(f"      Q3a诊断：下跌箱体对照(原始差) {f_(r['q3a_ctrl_dn_raw'])} | 朝边沿走分层 {f_(r['q3a_tw'])}(t{f_(r['q3a_tw_t'],1)}) | "
                         f"全量q1_ev {f_(r['q3a_all_mu'])}(双碰{r['q3a_all_both']*100:.1f}%) | "
                         f"账本 中轴净R {f_(r['led_mid_net'])}(胜{r['led_mid_win']*100:.0f}%,持{r['led_mid_hold']:.0f}d) vs 1R {f_(r['led_1r_net'])}(胜{r['led_1r_win']*100:.0f}%,持{r['led_1r_hold']:.0f}d)")
            elif r["qg"] == "Q3b":
                L.append(f"      Q3b诊断：确定性3维配对 {f_(r['q3b_det_match'])} vs 仅距离配对 {f_(r['q3b_k_only'])} | 触磁铁后5d反转率 {r['q3b_vac_rev']*100:.1f}%")

    L += ["", "=" * 120, "A 股 20 个主检验 BH-FDR 收口总表（行业 10 + 个股 10）", "=" * 120,
          f"  {'命题':<28s}{'超额':>9s}{'t':>6s}{'q(FDR)':>8s}{'t0(零)':>7s}{'专属对照':>9s}{'t':>6s}{'判定':>10s}"]
    n_pass = sum(1 for x in comb if x["verdict"] == "过关")
    n_rev = sum(1 for x in comb if x["verdict"] == "反向显著")
    for x in comb:
        is_q1 = x["qg"] in ("Q1a", "Q1b")
        ex_s = f"{x['ex']*100:>+7.1f}pp" if is_q1 else f"{x['ex']:>+9.3f}"
        ct_s = f"{x['ctrl']*100:>+7.1f}pp" if is_q1 else f"{x['ctrl']:>+9.3f}"
        L.append(f"  {x['label']:<26s}{ex_s:>9s}{f_(x['ex_t'],1):>6s}{x['q']:>8.4f}{f_(x['t0'],1):>7s}{ct_s:>9s}{f_(x['ctrl_t'],1):>6s}{x['verdict']:>10s}")
    L.append(f"\n  合计：过关 {n_pass}/20；反向显著 {n_rev}/20。")

    txt = "\n".join(L)
    REPORT_PATH.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    flat = []
    for r in rows_out:
        d = {k: v for k, v in r.items() if not isinstance(v, list)}
        d["per"] = ";".join(f"{v:+.4f}" if math.isfinite(v) else "nan" for v in r["per"])
        d["halves"] = ";".join(f"{v:+.4f}" if math.isfinite(v) else "nan" for v in r["halves"])
        d["tiers"] = ";".join(f"{v:+.4f}" if math.isfinite(v) else "nan" for v in r["tiers"])
        flat.append(d)
    pd.DataFrame(flat).to_csv(SUMMARY_PATH, index=False)
    log(f"stocks done ({time.time() - t0:.0f}s) -> {REPORT_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step0", action="store_true")
    args = ap.parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    if args.step0:
        L = step0()
        txt = "\n".join(L)
        STEP0_PATH.write_text(txt + "\n", encoding="utf-8")
        print(txt)
    else:
        main_analysis()


if __name__ == "__main__":
    main()
