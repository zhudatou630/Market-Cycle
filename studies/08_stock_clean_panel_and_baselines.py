#!/usr/bin/env python3
"""
Market Cycle V2 — Study 08（第二阶段方向 D · 地基）：全 A 股个股日线清洗面板 + 傻瓜基线画像

本脚本不考任何 Al Brooks 结构，只做三件事：
  1. 按事先定好的防污染口径，把全 A 个股日线清洗成「可研究、可成交」的面板（缓存到 output/，不入库）；
  2. 输出清洗漏斗（每条规则剔掉多少、买不进比例、数据兜底比例）；
  3. 给「傻瓜基线」画像：市值、换手、异常换手、短期/中期涨跌、波动、彩票效应、涨停次数，各自对未来 20 天
     超额收益 / 踩雷的影响，看清个股层面错觉的主要来源，为 08b/08c 的控制口径定标。

纪律：
  - ~/market-data 严格只读（sqlite 以 mode=ro 打开，只读 parquet）；经 catalog 取最新 READY 快照。
  - 开发段 2016-01-01 → 2024-10-31；2015 年只做指标预热；读入后立即硬截断 2024-10-31；
    未来标签窗口（含跌停顺延缓冲）跨过 2024-10-31 一律为空（封存区 2024-11-01 → 最新 零触碰）。

清洗口径（t 日收盘时可知的信息）：
  - 只要沪深 A 股（代码首位 0/3/6，后缀 SH/SZ；剔北交所、B 股）。
  - 后复权：OHLC × adj_factor（因子只反映 t 日及以前的除权，无未来函数）；涨跌停判断用原始价。
  - ST：stock_st（2016-08-09 起）∪ 名称历史中 t 日名称含 ST/PT/退。
  - 上市满 250 个市场交易日（stock_basic.list_date × 交易日历）；近 120 个市场交易日内该股交易 ≥ 108 天；
    停牌 ≥5 天复牌后 10 个交易日内不入池。
  - 流动性：近 20 个自身交易日均成交额处于当日截面最低 10% 剔除；原始收盘价 < 2 元剔除；缺流通市值剔除。
  - 入场：t+1 开盘买；t+1 停牌或开盘即涨停（含一字）= 买不进，标签为空（信号与基准同口径）。
  - 出场：t+H 收盘卖；若当日停牌或收盘封死跌停，顺延到 10 个交易日内第一个可卖收盘价，否则按最后成交价冻结（含退市）。
  - 主基准：同日、入池且买得进的股票按流通市值五等分，同组等权（简单收益）。
  - 辅基准：沪深300/中证500/中证1000/其余（t 日之前最新一期成分）、全市场等权、申万一级行业等权
    （行业库只有现行归属、无剔除日期，按 in_date 近似时点，仅作诊断）。
  - 风险裁判：① 波动归一踩雷 = 20 天内最深回撤超过「自身 20 日波动 × √20 × 1.5」；② 固定 -15%（体感辅助）。
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
MARKET_DATA_ROOT = Path("/home/zhujunshen/market-data")
CATALOG_DB = MARKET_DATA_ROOT / "catalog" / "metadata.sqlite"

PANEL_PATH = OUTPUT_DIR / "08_stock_panel_dev.parquet"
REPORT_PATH = OUTPUT_DIR / "08_stock_clean_panel_report.txt"

LOAD_YEARS = list(range(2015, 2025))
WARMUP_START = "20150101"
DEV_START = pd.Timestamp("2016-01-01")
DEV_END_STR = "20241031"

MIN_LIST_DAYS = 250
SUSP_LOOKBACK = 120
SUSP_MIN_TRADED = 108
RESUME_GAP = 5
RESUME_COOLDOWN = 10
LIQ_PCT = 0.10
MIN_PRICE = 2.0
MV_FFILL = 5
EXIT_DELAY_MAX = 10
PX_TOL = 1e-3

HORIZONS = [10, 20, 40]
CRASH_20 = -0.15
TAIL_K = 1.5
BIG_20 = 0.15
BIG_40 = 0.20
NW_LAG = 30

TIER_INDEX = [("000300.SH", 1), ("000905.SH", 2), ("000852.SH", 3)]
TIER_NAMES = {1: "沪深300", 2: "中证500", 3: "中证1000", 4: "其余小微"}

BASE_COLS = ["r5", "r20", "r60", "ma20pos", "vol20", "atr20p", "max20", "n_up20", "turn20", "abn_turn", "lmv"]
BASE_LABELS = {
    "r5": "近5日涨跌(短期反转)",
    "r20": "近20日涨跌",
    "r60": "近60日涨跌",
    "ma20pos": "相对20日均线位置",
    "vol20": "20日收益波动",
    "atr20p": "20日平均振幅(ATR/价)",
    "max20": "20日最大单日涨幅(彩票)",
    "n_up20": "20日涨停次数",
    "turn20": "20日平均换手",
    "abn_turn": "异常换手(近20/前120日)",
    "lmv": "流通市值(对数)",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Read-only snapshot access
# ---------------------------------------------------------------------------
def latest_ready_manifest() -> tuple[str, pd.DataFrame]:
    conn = sqlite3.connect(f"file:{CATALOG_DB}?mode=ro", uri=True)
    try:
        snap = pd.read_sql_query(
            """
            SELECT snapshot_id, end_date, created_at, gate
            FROM data_snapshot
            WHERE (gate = 'READY' OR gate LIKE '%_READY') AND gate != 'NOT_READY'
            ORDER BY end_date DESC, created_at DESC
            LIMIT 1
            """,
            conn,
        )
        assert len(snap) == 1, "No READY snapshot found in market-data catalog"
        sid = str(snap.iloc[0]["snapshot_id"])
        man = pd.read_sql_query(
            "SELECT dataset, partition_key, file_path FROM table_manifest WHERE snapshot_id = ?",
            conn,
            params=(sid,),
        )
    finally:
        conn.close()
    return sid, man


def read_ds(man: pd.DataFrame, dataset: str, cols: list[str], years: list[int] | None = None) -> pd.DataFrame:
    d = man[man["dataset"] == dataset]
    if years is not None:
        d = d[d["partition_key"].isin({f"year={y}" for y in years})]
    assert len(d), f"No partitions for {dataset}"
    parts = []
    for fp in sorted(d["file_path"]):
        p = Path(fp)
        if not p.is_absolute():
            p = MARKET_DATA_ROOT / p
        parts.append(pd.read_parquet(p, columns=cols))
    return pd.concat(parts, ignore_index=True)


def cut_dev(df: pd.DataFrame, col: str = "trade_date", start: str = WARMUP_START) -> pd.DataFrame:
    out = df[(df[col] >= start) & (df[col] <= DEV_END_STR)].copy()
    assert out[col].max() <= DEV_END_STR, "Holdout leak"
    return out


def merge_unique(left: pd.DataFrame, right: pd.DataFrame, name: str) -> pd.DataFrame:
    assert not right.duplicated(["ts_code", "trade_date"]).any(), f"duplicate keys in {name}"
    n = len(left)
    out = left.merge(right, on=["ts_code", "trade_date"], how="left")
    assert len(out) == n, f"row explosion merging {name}"
    return out


# ---------------------------------------------------------------------------
# 2. Build long panel with cleaning flags & causal covariates
# ---------------------------------------------------------------------------
def is_a_share(codes: pd.Series) -> np.ndarray:
    return (codes.str[-2:].isin(["SH", "SZ"]) & codes.str[0].isin(["0", "3", "6"])).to_numpy()


def board_limit_pct(codes: pd.Series, dates: pd.Series, is_st: np.ndarray) -> np.ndarray:
    star = codes.str.startswith(("688", "689")).to_numpy() & (dates >= "20190722").to_numpy()
    gem = codes.str.startswith(("300", "301")).to_numpy() & (dates >= "20200824").to_numpy()
    lim = np.where(star | gem, 0.20, 0.10)
    lim = np.where(is_st & ~(star | gem), 0.05, lim)
    return lim


def build_panel(man: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict]:
    info: dict = {}
    log("load stock_daily")
    sd = read_ds(
        man, "stock_daily",
        ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"],
        LOAD_YEARS,
    )
    sd = cut_dev(sd)
    sd = sd[is_a_share(sd["ts_code"])]
    sd = sd[(sd["vol"] > 0) & sd[["open", "high", "low", "close"]].notna().all(axis=1)]
    assert not sd.duplicated(["ts_code", "trade_date"]).any()

    log("merge adj_factor / stk_limit / daily_basic")
    af = cut_dev(read_ds(man, "stock_adj_factor", ["ts_code", "trade_date", "adj_factor"], LOAD_YEARS))
    sd = merge_unique(sd, af, "adj_factor")
    info["adj_missing"] = int(sd["adj_factor"].isna().sum())
    sd = sd[sd["adj_factor"].notna()]
    lim = cut_dev(read_ds(man, "stk_limit", ["ts_code", "trade_date", "up_limit", "down_limit"], LOAD_YEARS))
    sd = merge_unique(sd, lim, "stk_limit")
    db = cut_dev(
        read_ds(man, "daily_basic", ["ts_code", "trade_date", "turnover_rate_f", "circ_mv"], LOAD_YEARS)
    )
    sd = merge_unique(sd, db, "daily_basic")
    del af, lim, db

    sd = sd.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    codes = sd["ts_code"]
    tds = sd["trade_date"]
    sd["date"] = pd.to_datetime(tds, format="%Y%m%d")
    code_id = pd.factorize(codes)[0]
    sd["code_id"] = code_id
    pos = sd.groupby("code_id").cumcount().to_numpy()

    # --- ST flag: stock_st ∪ name history (name at t contains ST/PT/退)
    log("ST flags")
    st = cut_dev(read_ds(man, "stock_st", ["ts_code", "trade_date"]))
    st_key = set(zip(st["ts_code"], st["trade_date"]))
    in_st_table = np.fromiter((k in st_key for k in zip(codes, tds)), dtype=bool, count=len(sd))
    nh = read_ds(man, "stock_name_history", ["ts_code", "name", "start_date", "end_date"])
    nh = nh.dropna(subset=["start_date"]).sort_values(["ts_code", "start_date"])
    nh = nh.drop_duplicates(["ts_code", "start_date"], keep="last")
    nh["sdt"] = pd.to_datetime(nh["start_date"], format="%Y%m%d", errors="coerce")
    nh["edt"] = pd.to_datetime(nh["end_date"], format="%Y%m%d", errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    nh["bad_name"] = nh["name"].fillna("").str.contains("ST|PT|退", regex=True)
    left = sd[["ts_code", "date"]].reset_index().sort_values("date")
    nm = pd.merge_asof(
        left, nh[["ts_code", "sdt", "edt", "bad_name"]].dropna(subset=["sdt"]).sort_values("sdt"),
        left_on="date", right_on="sdt", by="ts_code", direction="backward",
    ).set_index("index").sort_index()
    assert len(nm) == len(sd)
    bad_name = (nm["bad_name"].fillna(False).astype(bool) & (nm["date"] <= nm["edt"])).to_numpy()
    is_st = in_st_table | bad_name
    info["st_by_table_only"] = int((in_st_table & ~bad_name).sum())
    info["st_by_name_only"] = int((bad_name & ~in_st_table).sum())
    sd["is_st"] = is_st
    del st, st_key, nh, left, nm

    # --- limit price fallback (raw prices) & limit flags
    lim_pct = board_limit_pct(codes, tds, is_st)
    miss = sd["up_limit"].isna().to_numpy()
    info["limit_fallback_rows"] = int(miss.sum())
    pc = sd["pre_close"].to_numpy(dtype=float)
    sd.loc[miss, "up_limit"] = np.round(pc[miss] * (1 + lim_pct[miss]), 2)
    sd.loc[miss, "down_limit"] = np.round(pc[miss] * (1 - lim_pct[miss]), 2)
    sd["lim_pct"] = lim_pct
    sd["up_close"] = sd["close"] >= sd["up_limit"] - PX_TOL
    sd["open_up"] = sd["open"] >= sd["up_limit"] - PX_TOL
    sd["dn_close"] = sd["close"] <= sd["down_limit"] + PX_TOL

    # --- market calendar (SSE open days) & listing age
    log("calendar / listing age / suspension")
    cal = read_ds(man, "trade_cal", ["cal_date", "is_open", "exchange"])
    cal = cal[(cal["is_open"].astype(int) == 1) & (cal["exchange"].fillna("SSE") == "SSE")]
    cal_days = np.sort(cal["cal_date"].astype(str).unique())
    cal_days = cal_days[cal_days <= DEV_END_STR]
    mdates_str = cal_days[cal_days >= WARMUP_START]
    assert np.isin(np.sort(tds.unique()), mdates_str).all(), "stock dates not on SSE calendar"
    mdates = pd.to_datetime(mdates_str, format="%Y%m%d").to_numpy()
    mi = np.searchsorted(mdates_str, tds.to_numpy())
    sd["mi"] = mi

    sb = read_ds(man, "stock_basic", ["ts_code", "list_date"]).drop_duplicates("ts_code")
    ld = codes.map(sb.set_index("ts_code")["list_date"])
    first_seen = sd.groupby("code_id")["trade_date"].transform("min")
    info["no_list_date_codes"] = int(sd.loc[ld.isna(), "ts_code"].nunique())
    ld = ld.fillna(first_seen).astype(str)
    age = np.searchsorted(cal_days, tds.to_numpy(), side="right") - np.searchsorted(cal_days, ld.to_numpy(), side="left")
    sd["list_age"] = age

    # --- suspension: traded days in last 120 market days; cooldown after long suspension
    key = code_id.astype(np.int64) * 100_000 + mi
    lo = np.searchsorted(key, key - (SUSP_LOOKBACK - 1), side="left")
    sd["traded_lb"] = np.arange(len(sd)) - lo + 1
    new_code = pos == 0
    gap = np.r_[0, np.diff(mi)] - 1
    gap[new_code] = 0
    resume = gap >= RESUME_GAP
    ridx = pd.Series(np.where(resume, np.arange(len(sd)), np.nan)).groupby(code_id).ffill().to_numpy()
    since = np.arange(len(sd)) - ridx
    sd["cooldown"] = np.isfinite(since) & (since < RESUME_COOLDOWN)

    # --- adjusted prices & causal covariates on the stock's own trading series (no fake bars)
    log("adjusted prices & covariates")
    a = sd["adj_factor"].to_numpy(dtype=float)
    for c_ in ["open", "high", "low", "close"]:
        sd[f"a_{c_}"] = sd[c_].to_numpy(dtype=float) * a
    s_lc = pd.Series(np.log(sd["a_close"].to_numpy()))

    def lagdiff(k: int) -> np.ndarray:
        x = (s_lc - s_lc.shift(k)).to_numpy().copy()
        x[pos < k] = np.nan
        return x

    def roll(x: np.ndarray, w: int, fn: str, minp: int, need: int) -> np.ndarray:
        r = getattr(pd.Series(x).rolling(w, min_periods=minp), fn)().to_numpy().copy()
        r[pos < need] = np.nan
        return r

    ret1 = lagdiff(1)
    sd["ret1"] = ret1
    sd["r5"] = lagdiff(5)
    sd["r20"] = lagdiff(20)
    sd["r60"] = lagdiff(60)
    sd["vol20"] = roll(ret1, 20, "std", 20, 20)
    sd["max20"] = roll(ret1, 20, "max", 20, 20)
    prev_c = pd.Series(sd["a_close"].to_numpy()).shift(1).to_numpy()
    h_ = sd["a_high"].to_numpy()
    l_ = sd["a_low"].to_numpy()
    tr = np.maximum.reduce([h_ - l_, np.abs(h_ - prev_c), np.abs(l_ - prev_c)])
    tr[pos < 1] = np.nan
    sd["atr20p"] = roll(tr, 20, "mean", 20, 20) / sd["a_close"].to_numpy()
    sd["ma20pos"] = sd["a_close"].to_numpy() / roll(sd["a_close"].to_numpy(), 20, "mean", 20, 19) - 1.0
    sd["n_up20"] = roll(sd["up_close"].to_numpy(dtype=float), 20, "sum", 20, 19)
    turn = sd["turnover_rate_f"].to_numpy(dtype=float)
    turn20 = roll(turn, 20, "mean", 15, 19)
    turn_prev = pd.Series(turn).rolling(120, min_periods=90).mean().shift(20).to_numpy().copy()
    turn_prev[pos < 139] = np.nan
    sd["turn20"] = turn20
    sd["abn_turn"] = turn20 / np.where(turn_prev > 0, turn_prev, np.nan)
    sd["amt20"] = roll(sd["amount"].to_numpy(dtype=float), 20, "mean", 15, 19)

    mv = sd["circ_mv"].where(sd["circ_mv"] > 0)
    info["mv_missing_raw"] = int(mv.isna().sum())
    mv = mv.groupby(code_id).ffill(limit=MV_FFILL)
    sd["circ_mv"] = mv
    sd["lmv"] = np.log(mv)

    # data sanity: adjusted daily moves beyond board limit (age>=250, no gap)
    abs_simple = np.abs(np.expm1(ret1))
    chk = (age >= MIN_LIST_DAYS) & ~sd["cooldown"].to_numpy() & np.isfinite(abs_simple) & (gap == 0)
    info["adj_jump_rows"] = int((chk & (abs_simple > lim_pct + 0.012)).sum())
    info["adj_jump_checked"] = int(chk.sum())

    # --- liquidity percentile (cross-section of all A-share traded stocks that day)
    sd["amt_pct"] = sd.groupby("trade_date")["amt20"].rank(pct=True)

    # --- index-membership tiers by latest constituents strictly before t
    log("tiers / industry")
    iw = read_ds(man, "index_weight", ["index_code", "con_code", "trade_date"])
    iw = iw[iw["trade_date"] <= DEV_END_STR]
    cnt = iw[iw["trade_date"] >= "20151201"].groupby(["index_code", "trade_date"]).size()
    info["iw_counts"] = {k: (int(v.min()), int(v.max()), int(len(v))) for k, v in cnt.groupby(level=0)}
    tier = np.full(len(sd), 4, dtype=np.int8)
    for idx_code, tval in reversed(TIER_INDEX):
        w = iw[iw["index_code"] == idx_code]
        wd = np.sort(w["trade_date"].unique())
        k = np.searchsorted(wd, tds.to_numpy(), side="left") - 1
        wdate = np.where(k >= 0, wd[np.clip(k, 0, None)], "")
        mem = set(zip(w["con_code"], w["trade_date"]))
        flag = np.fromiter((kk in mem for kk in zip(codes, wdate)), dtype=bool, count=len(sd))
        tier[flag] = tval
    sd["tier"] = tier
    del iw

    sw = read_ds(man, "sw_member_history", ["src", "level", "ts_code", "in_date", "l1_code"])
    sw = sw[(sw["src"] == "SW2021") & (sw["level"] == "L1")].dropna(subset=["in_date"])
    sw["idt"] = pd.to_datetime(sw["in_date"], format="%Y%m%d", errors="coerce")
    sw = sw.dropna(subset=["idt"]).sort_values("idt").drop_duplicates(["ts_code", "idt"], keep="last")
    left = sd[["ts_code", "date"]].reset_index().sort_values("date")
    ind = pd.merge_asof(left, sw[["ts_code", "idt", "l1_code"]], left_on="date", right_on="idt",
                        by="ts_code", direction="backward").set_index("index").sort_index()
    assert len(ind) == len(sd)
    earliest = sw.groupby("ts_code")["l1_code"].first()
    sd["l1"] = ind["l1_code"].fillna(codes.map(earliest)).fillna("NA").to_numpy()
    info["l1_missing_rows"] = int((sd["l1"] == "NA").sum())

    # --- eligibility (sequential funnel flags)
    sd["f_st"] = sd["is_st"]
    sd["f_age"] = sd["list_age"] < MIN_LIST_DAYS
    sd["f_susp"] = sd["traded_lb"] < SUSP_MIN_TRADED
    sd["f_cool"] = sd["cooldown"]
    sd["f_liq"] = ~(sd["amt_pct"] >= LIQ_PCT)
    sd["f_px"] = sd["close"] < MIN_PRICE
    sd["f_mv"] = sd["circ_mv"].isna()
    bad = sd[["f_st", "f_age", "f_susp", "f_cool", "f_liq", "f_px", "f_mv"]].any(axis=1)
    sd["eligible"] = (~bad) & (sd["date"] >= DEV_START)
    return sd, mdates, info


# ---------------------------------------------------------------------------
# 3. Forward labels on the market calendar (entry t+1 open, delayed exit)
# ---------------------------------------------------------------------------
def build_forward(sd: pd.DataFrame, mdates: np.ndarray) -> pd.DataFrame:
    log("forward labels (wide)")
    T = len(mdates)
    N = int(sd["code_id"].max()) + 1
    di = sd["mi"].to_numpy()
    ci = sd["code_id"].to_numpy()

    def wide(vals: np.ndarray, fill=np.nan, dtype=float) -> np.ndarray:
        W = np.full((T, N), fill, dtype=dtype)
        W[di, ci] = vals
        return W

    a_high = wide(sd["a_high"].to_numpy())
    a_low = wide(sd["a_low"].to_numpy())
    a_close = wide(sd["a_close"].to_numpy())
    traded = wide(np.ones(len(sd), dtype=bool), False, bool)
    open_up = wide(sd["open_up"].to_numpy(), False, bool)
    dn_close = wide(sd["dn_close"].to_numpy(), False, bool)

    buy_px = np.where(traded & ~open_up, wide(sd["a_open"].to_numpy()), np.nan)
    entry = np.full((T, N), np.nan)
    entry[:-1] = buy_px[1:]
    del buy_px, open_up

    sell = np.where(traded & ~dn_close, a_close, np.nan)
    exit_bf = pd.DataFrame(sell).bfill(limit=EXIT_DELAY_MAX).to_numpy()
    exit_ff = pd.DataFrame(a_close).ffill().to_numpy()
    frozen = ~np.isfinite(exit_bf)
    exit_px = np.where(frozen, exit_ff, exit_bf)
    del sell, exit_bf, exit_ff, dn_close, traded

    out: dict[str, np.ndarray] = {"entry_ok": np.isfinite(entry)[di, ci]}
    for H in HORIZONS:
        last_ok = T - 1 - H - EXIT_DELAY_MAX  # leave buffer for delayed exit inside dev
        ex = np.full((T, N), np.nan)
        ex[: T - H] = exit_px[H:]
        ex[last_ok + 1:] = np.nan
        ret = ex / entry - 1.0
        out[f"ret_{H}"] = ret[di, ci]
        if H == 20:
            fz = np.zeros((T, N), dtype=bool)
            fz[: T - H] = frozen[H:]
            out["exit_frozen_20"] = (fz & np.isfinite(ret))[di, ci]
        if H in (20, 40):
            lo_ = pd.DataFrame(a_low[::-1]).rolling(H, min_periods=1).min().to_numpy()[::-1]
            hi_ = pd.DataFrame(a_high[::-1]).rolling(H, min_periods=1).max().to_numpy()[::-1]
            mn = np.full((T, N), np.nan)
            mx = np.full((T, N), np.nan)
            mn[: T - H] = lo_[1: T - H + 1]
            mx[: T - H] = hi_[1: T - H + 1]
            mn[last_ok + 1:] = np.nan
            mx[last_ok + 1:] = np.nan
            out[f"mdd_{H}"] = (mn / entry - 1.0)[di, ci]
            out[f"mfe_{H}"] = (mx / entry - 1.0)[di, ci]
            del lo_, hi_, mn, mx
        del ex, ret
    fw = pd.DataFrame(out, index=sd.index)
    for c in ["ret_10", "ret_20", "ret_40", "mdd_20", "mfe_20", "mdd_40", "mfe_40"]:
        fw.loc[~fw["entry_ok"], c] = np.nan

    def flag(cond: np.ndarray, base: pd.Series) -> np.ndarray:
        return np.where(np.isfinite(base), cond.astype(float), np.nan)

    tail_thr = -TAIL_K * sd["vol20"].to_numpy() * np.sqrt(20)
    fw["crash_20"] = flag(fw["mdd_20"].to_numpy() <= CRASH_20, fw["mdd_20"])
    fw["tail_20"] = np.where(np.isfinite(tail_thr), flag(fw["mdd_20"].to_numpy() <= tail_thr, fw["mdd_20"]), np.nan)
    fw["big_20"] = flag(fw["mfe_20"].to_numpy() >= BIG_20, fw["mfe_20"])
    fw["big_40"] = flag(fw["mfe_40"].to_numpy() >= BIG_40, fw["mfe_40"])
    return fw


def add_excess(sd: pd.DataFrame) -> None:
    log("benchmarks & excess")
    base = sd["eligible"].to_numpy() & sd["entry_ok"].to_numpy()
    pct = sd["lmv"].where(base).groupby(sd["trade_date"]).rank(pct=True, method="first")
    sd["szq"] = np.ceil(pct * 5).clip(1, 5).fillna(0).astype(np.int8)
    for H in HORIZONS:
        r = sd[f"ret_{H}"].where(base)
        sd[f"ex_sz_{H}"] = sd[f"ret_{H}"] - r.groupby([sd["trade_date"], sd["szq"]]).transform("mean")
        sd[f"ex_mkt_{H}"] = sd[f"ret_{H}"] - r.groupby(sd["trade_date"]).transform("mean")
        sd[f"ex_tier_{H}"] = sd[f"ret_{H}"] - r.groupby([sd["trade_date"], sd["tier"]]).transform("mean")
        sd[f"ex_ind_{H}"] = sd[f"ret_{H}"] - r.groupby([sd["trade_date"], sd["l1"]]).transform("mean")
        # Non-eligible rows have no size bucket: leave ex_sz empty for them
        sd.loc[sd["szq"] == 0, f"ex_sz_{H}"] = np.nan


# ---------------------------------------------------------------------------
# 4. Statistics helpers (daily series, each trading day = one vote)
# ---------------------------------------------------------------------------
def nw_mean_t(x: np.ndarray, lag: int = NW_LAG) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 60:
        return np.nan, np.nan
    m = x.mean()
    e = x - m
    s = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    se = np.sqrt(max(s, 0) / n)
    return float(m), float(m / se) if se > 0 else np.nan


def quintile_profile(df: pd.DataFrame, feat: str, ycol: str, by_tier: bool = False) -> dict:
    sub = df[["trade_date", "tier", feat, ycol, f"{ycol}_w", "crash_20", "tail_20"]].dropna()
    grp = ["trade_date", "tier"] if by_tier else ["trade_date"]
    pct = sub.groupby(grp)[feat].rank(pct=True, method="first")
    sub = sub.assign(q=np.ceil(pct * 5).clip(1, 5).astype(int))
    daily = sub.groupby(["trade_date", "q"])[[ycol, f"{ycol}_w", "crash_20", "tail_20"]].mean().unstack("q")
    res = {
        "q_ret": [float(daily[ycol][q].mean() * 100) for q in range(1, 6)],
        "q_crash": [float(daily["crash_20"][q].mean() * 100) for q in range(1, 6)],
        "q_tail": [float(daily["tail_20"][q].mean() * 100) for q in range(1, 6)],
    }
    spread = (daily[ycol][5] - daily[ycol][1]) * 100
    res["spread"], res["t"] = nw_mean_t(spread.to_numpy())
    res["spread_w"], res["t_w"] = nw_mean_t(((daily[f"{ycol}_w"][5] - daily[f"{ycol}_w"][1]) * 100).to_numpy())
    res["tail_d"], res["tail_t"] = nw_mean_t(((daily["tail_20"][5] - daily["tail_20"][1]) * 100).to_numpy())
    yr = spread.groupby(spread.index.str[:4]).mean()
    res["years_same"] = int((np.sign(yr) == np.sign(res["spread"])).sum())
    res["years_n"] = int(len(yr))
    return res


def fama_macbeth(df: pd.DataFrame, ycol: str, cols: list[str], ind_fe: bool = True) -> pd.DataFrame:
    sub = df[["trade_date", "l1", ycol] + cols].dropna().copy()
    # standardise each covariate cross-sectionally per day, then (optionally) demean within (day, industry)
    g = sub.groupby("trade_date")[cols]
    sub[cols] = (sub[cols] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    sub = sub.dropna()
    vals = cols + [ycol]
    if ind_fe:
        sub[vals] = sub[vals] - sub.groupby(["trade_date", "l1"])[vals].transform("mean")
    betas = []
    for _, gg in sub.groupby("trade_date", sort=True):
        if len(gg) < 300:
            continue
        X = gg[cols].to_numpy(dtype=float)
        y = gg[ycol].to_numpy(dtype=float) * 100
        X = np.column_stack([np.ones(len(gg)), X])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        betas.append(b[1:])
    B = np.array(betas)
    rows = []
    for j, c in enumerate(cols):
        m, t = nw_mean_t(B[:, j])
        rows.append({"var": c, "coef": m, "t": t})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    sid, man = latest_ready_manifest()
    log(f"snapshot {sid}")
    sd, mdates, info = build_panel(man)
    fw = build_forward(sd, mdates)
    for c in fw.columns:
        sd[c] = fw[c].to_numpy()
    del fw
    add_excess(sd)
    sd["year"] = sd["trade_date"].str[:4]
    assert sd["trade_date"].max() <= DEV_END_STR

    L: list[str] = []
    L.append("=" * 110)
    L.append("Study 08 · 方向 D 地基：全 A 个股清洗面板 + 傻瓜基线画像（开发段 2016-01-01 → 2024-10-31，封存区零触碰）")
    L.append(f"快照: {sid} | 面板行数(含2015预热): {len(sd):,} | 股票数: {sd['ts_code'].nunique():,}")
    L.append("=" * 110)

    dev = sd[sd["date"] >= DEV_START]
    el = dev[dev["eligible"]]
    y_ok = el.dropna(subset=["ret_20"]).copy()

    # ---- data notes
    L.append("\n[0] 数据兜底与体检")
    L.append(f"  复权因子缺失行: {info['adj_missing']} | 涨跌停价缺失→按板块规则兜底: {info['limit_fallback_rows']:,} 行")
    L.append(f"  ST 仅由风险警示表识别: {info['st_by_table_only']:,} 行 | 仅由名称历史识别: {info['st_by_name_only']:,} 行")
    L.append(f"  无上市日期代码: {info['no_list_date_codes']} | 无申万行业归属行: {info['l1_missing_rows']:,}")
    L.append(f"  流通市值原始缺失: {info['mv_missing_raw']:,} 行（向前沿用 ≤{MV_FFILL} 天后仍缺的剔除）")
    mvm = dev[~dev["f_st"] & ~dev["f_age"]].groupby("year")["f_mv"].mean() * 100
    L.append("  非ST、上市满一年股票中缺市值占比(%)：" + " ".join(f"{y}:{v:.2f}" for y, v in mvm.items()))
    L.append("  成分股每期只数 (最少/最多/期数, 2015-12 起): " + " | ".join(
        f"{k}:{v[0]}/{v[1]}/{v[2]}" for k, v in info["iw_counts"].items()))
    L.append(
        f"  后复权日涨跌超板块限幅(+1.2%容差)行: {info['adj_jump_rows']:,} / {info['adj_jump_checked']:,} "
        f"({info['adj_jump_rows'] / max(info['adj_jump_checked'], 1) * 100:.4f}%)"
    )
    L.append(f"  20 天持有期出场价被冻结（停牌/封跌停超 10 天或退市）占比: {y_ok['exit_frozen_20'].mean() * 100:.3f}%")

    # ---- funnel
    L.append("\n[1] 清洗漏斗（每年平均每天的股票数；按规则顺序逐层剔除）")
    steps = [("f_st", "ST/退"), ("f_age", "上市<250天"), ("f_susp", "近120天停>10%"),
             ("f_cool", "长停复牌10天"), ("f_liq", "成交额末10%"), ("f_px", "股价<2元"), ("f_mv", "缺市值")]
    L.append(f"  {'年份':<6}{'交易中':>7}" + "".join(f"{lab:>13}" for _, lab in steps) + f"{'入池':>7}{'入池率':>8}")
    for y, g in dev.groupby("year"):
        nd = g["trade_date"].nunique()
        alive = np.ones(len(g), dtype=bool)
        row = f"  {y:<6}{len(g) / nd:>7.0f}"
        for f, _ in steps:
            fl = g[f].to_numpy()
            row += f"{-(alive & fl).sum() / nd:>13.0f}"
            alive &= ~fl
        row += f"{alive.sum() / nd:>7.0f}{alive.mean() * 100:>7.1f}%"
        L.append(row)

    L.append("\n[2] 入池股票的可成交性与分档")
    L.append(
        f"  入池日中: 当天收盘涨停 {el['up_close'].mean() * 100:.2f}% | 次日买不进(停牌或开盘涨停) "
        f"{(~el['entry_ok']).mean() * 100:.2f}% | 当天收盘涨停者次日买不进 "
        f"{(~el.loc[el['up_close'], 'entry_ok']).mean() * 100:.1f}%"
    )
    nd_y = el.groupby("year")["trade_date"].nunique()
    tc = el.groupby(["year", "tier"]).size().unstack("tier").div(nd_y, axis=0)
    L.append("  每天平均入池数（指数成分档）: " + " | ".join(TIER_NAMES[t] for t in tc.columns))
    for y, r in tc.iterrows():
        L.append(f"    {y}: " + " | ".join(f"{v:7.0f}" for v in r.to_numpy()))
    med = y_ok.groupby(["year", "szq"])["circ_mv"].median().unstack("szq") / 1e4
    L.append("  市值五档（主基准）各档流通市值中位数(亿元): Q1 最小 → Q5 最大")
    for y, r in med.iterrows():
        L.append(f"    {y}: " + " | ".join(f"{v:7.1f}" for v in r.to_numpy()))

    L.append("\n[3] 裁判底数（入池且买得进；每个交易日一票；简单收益）")
    for nm_, key in [("沪深300", 1), ("中证500", 2), ("中证1000", 3), ("其余小微", 4)]:
        g = y_ok[y_ok["tier"] == key]
        d = g.groupby("trade_date")[["ret_20", "crash_20", "tail_20", "big_20", "mdd_20"]].mean()
        L.append(
            f"  {nm_:<6} 20天收益 {d['ret_20'].mean() * 100:+5.2f}% | 固定踩雷(-15%) {d['crash_20'].mean() * 100:4.1f}% | "
            f"波动归一踩雷 {d['tail_20'].mean() * 100:4.1f}% | 大肉(+15%) {d['big_20'].mean() * 100:4.1f}% | "
            f"平均最深回撤 {d['mdd_20'].mean() * 100:6.2f}%"
        )
    d = y_ok.groupby("trade_date")[["ret_20", "crash_20", "tail_20", "big_20"]].mean()
    L.append(
        f"  全部     20天收益 {d['ret_20'].mean() * 100:+5.2f}% | 固定踩雷 {d['crash_20'].mean() * 100:4.1f}% | "
        f"波动归一踩雷 {d['tail_20'].mean() * 100:4.1f}% | 大肉 {d['big_20'].mean() * 100:4.1f}%"
    )
    yrs = y_ok.groupby(["year", "szq"])["ex_mkt_20"].mean().unstack("szq") * 100
    L.append("  市值五档相对全市场等权的 20 天超额(%)：Q1 最小 → Q5 最大")
    for y, r in yrs.iterrows():
        L.append(f"    {y}: " + " | ".join(f"{v:+6.2f}" for v in r.to_numpy()))

    # ---- baseline profile
    for yc in ["ex_sz_20", "ex_mkt_20"]:
        lo_, hi_ = y_ok[yc].quantile([0.005, 0.995])
        y_ok[f"{yc}_w"] = y_ok[yc].clip(lo_, hi_)
    L.append("\n[4] 傻瓜基线画像：每天截面五等分（Q1 最低 → Q5 最高），未来 20 天相对同市值档超额(%)")
    L.append("    价差 = Q5 − Q1（t 值：按交易日序列，重叠持有期 Newey-West lag=30）；缩尾 = 0.5%/99.5% 截尾后价差；")
    L.append("    同向年 = 9 年里与总体方向一致的年数；踩雷列为波动归一踩雷率 Q1→Q5（括号为固定 -15% 的 Q1/Q5）")
    for c in BASE_COLS:
        ycol = "ex_mkt_20" if c == "lmv" else "ex_sz_20"
        r = quintile_profile(y_ok, c, ycol)
        L.append(
            f"  {BASE_LABELS[c]:<20} 超额 " + " ".join(f"{v:+5.2f}" for v in r["q_ret"])
            + f" | 价差 {r['spread']:+5.2f} (t={r['t']:+5.2f}, 缩尾 {r['spread_w']:+5.2f}, {r['years_same']}/{r['years_n']}年)"
            + " | 归一踩雷 " + " ".join(f"{v:4.1f}" for v in r["q_tail"])
            + f" (t={r['tail_t']:+5.1f}; 固定 {r['q_crash'][0]:4.1f}/{r['q_crash'][4]:4.1f})"
            + ("  [相对全市场]" if c == "lmv" else "")
        )
    L.append("\n  分指数成分档的 Q5−Q1 价差（每档内部各自五等分，相对同市值档超额 %；括号为 t）")
    L.append("  " + f"{'':<20}" + "".join(f"{TIER_NAMES[t]:>16}" for t in [1, 2, 3, 4]))
    for c in BASE_COLS:
        row = f"  {BASE_LABELS[c]:<20}"
        ycol = "ex_mkt_20" if c == "lmv" else "ex_sz_20"
        for t in [1, 2, 3, 4]:
            r = quintile_profile(y_ok[y_ok["tier"] == t], c, ycol)
            row += f"{r['spread']:+8.2f}({r['t']:+5.1f})"
        L.append(row)

    L.append("\n[5] 多变量同台（每日截面回归 + 行业固定效应，跨日平均；系数 = 该变量高 1 个截面标准差时的变化）")
    fm = fama_macbeth(y_ok, "ex_sz_20", BASE_COLS)
    fm_t = fama_macbeth(y_ok, "tail_20", BASE_COLS)
    fm_c = fama_macbeth(y_ok, "crash_20", BASE_COLS)
    L.append(f"  {'':<20}{'20天超额(%)':>16}{'归一踩雷(pp)':>18}{'固定踩雷(pp)':>18}")
    for i, c in enumerate(BASE_COLS):
        L.append(
            f"  {BASE_LABELS[c]:<20}{fm.loc[i, 'coef']:+8.3f} (t={fm.loc[i, 't']:+5.1f})"
            f"{fm_t.loc[i, 'coef']:+9.3f} (t={fm_t.loc[i, 't']:+5.1f})"
            f"{fm_c.loc[i, 'coef']:+9.3f} (t={fm_c.loc[i, 't']:+5.1f})"
        )

    report = "\n".join(L)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)

    # ---- cache panel for 08b / 08c (rows <= 2024-10-31 only)
    keep = (
        ["ts_code", "trade_date", "code_id", "mi", "tier", "szq", "l1", "lim_pct", "is_st", "eligible", "entry_ok",
         "up_close", "open_up", "dn_close", "a_open", "a_high", "a_low", "a_close", "close", "amount",
         "turnover_rate_f", "circ_mv", "ret1"]
        + BASE_COLS
        + [f"ret_{H}" for H in HORIZONS]
        + [f"ex_{b}_{H}" for b in ["sz", "mkt", "tier", "ind"] for H in HORIZONS]
        + ["mdd_20", "mfe_20", "mdd_40", "mfe_40", "crash_20", "tail_20", "big_20", "big_40", "exit_frozen_20"]
    )
    out = sd[keep].copy()
    for c in out.columns:
        if out[c].dtype == np.float64 and c not in ("a_open", "a_high", "a_low", "a_close"):
            out[c] = out[c].astype(np.float32)
    assert out["trade_date"].max() <= DEV_END_STR
    out.to_parquet(PANEL_PATH, index=False)
    log(f"panel cached -> {PANEL_PATH} ({PANEL_PATH.stat().st_size / 1e6:.0f} MB); total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
