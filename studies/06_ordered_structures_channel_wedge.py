#!/usr/bin/env python3
"""
Market Cycle V2 — Study 06 (第二阶段方向 B):
有先后顺序的博弈结构：
  1. 窄通道 (Tight Channel) vs 宽通道 (Broad Channel)
  2. 三推楔形 (Wedge Top / Wedge Bottom — 严格因果实时数推、零 ZigZag)
  3. 周期演化顺序 (BO -> TC -> BC -> TR 的真实转移概率矩阵与 Lift)

严格遵循 AGENTS.md 五条铁律，严守封存边界（先截断再算未来标签，绝不碰封存区一根 K 线）：
  - SPX 开发段：1984-01-01 -> 2019-12-31（全程只用 H/L/C，不用 open）
  - A 股 31 个申万一级行业开发段：2016-01-01 -> 2024-10-31
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"


def _load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "studies" / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


s01 = _load_mod("s01", "01_trend_vs_range.py")
s03 = _load_mod("s03", "03_breakout_true_vs_false.py")
s05 = _load_mod("s05", "05_breakout_pullback_and_failed.py")

EPS = 1e-12
THRESH = {
    "A股申万L1": {"crash_20": -0.08, "big_20": 0.08, "big_40": 0.10},
    "SPX": {"crash_20": -0.05, "big_20": 0.04, "big_40": 0.06},
}


# ---------------------------------------------------------------------------
# 1. Per-Asset Causal Indicators, Channels, 3-Push Wedges & Regime Sequence
# ---------------------------------------------------------------------------
def build_asset_study06(
    g: pd.DataFrame,
    market_name: str,
    eval_start: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    g = g.sort_values("date").reset_index(drop=True)
    dates = g["date"].to_numpy()
    asset = str(g["asset"].iloc[0])
    h = g["high"].to_numpy(dtype=float)
    l = g["low"].to_numpy(dtype=float)
    c = g["close"].to_numpy(dtype=float)
    n = len(g)

    prev_c = np.empty(n)
    prev_c[0] = np.nan
    prev_c[1:] = c[:-1]

    log_c = np.log(c)
    ret1 = np.empty(n)
    ret1[0] = np.nan
    ret1[1:] = np.diff(log_c)

    s_c = pd.Series(c)
    s_h = pd.Series(h)
    s_l = pd.Series(l)
    s_log_c = pd.Series(log_c)
    s_ret1 = pd.Series(ret1)

    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr20 = pd.Series(tr).rolling(20, min_periods=20).mean().to_numpy()
    ma10 = s_c.rolling(10, min_periods=10).mean().to_numpy()
    ma20 = s_c.rolling(20, min_periods=20).mean().to_numpy()
    ma20_lag10 = pd.Series(ma20).shift(10).to_numpy()
    r20 = s_log_c.diff(20).to_numpy()
    r5 = s_log_c.diff(5).to_numpy()
    vol20 = s_ret1.rolling(20, min_periods=20).std(ddof=1).to_numpy()

    # Rolling 20-day channel structure metrics (strictly causal)
    above_ma20 = (c >= ma20).astype(float)
    above_ma10 = (c >= ma10).astype(float)
    below_ma20 = (c <= ma20).astype(float)
    ma20_hold = pd.Series(above_ma20).rolling(20, min_periods=20).mean().to_numpy()
    ma10_hold = pd.Series(above_ma10).rolling(20, min_periods=20).mean().to_numpy()
    ma20_hold_dn = pd.Series(below_ma20).rolling(20, min_periods=20).mean().to_numpy()

    # In-window maximum adverse pullback / rally in ATR units (causal loop over 20-bar window)
    max_pb_atr = np.full(n, np.nan)
    max_rally_atr = np.full(n, np.nan)
    for t in range(20, n):
        a = atr20[t]
        if not np.isfinite(a) or a <= EPS:
            continue
        win_h = h[t - 19 : t + 1]
        win_l = l[t - 19 : t + 1]
        run_max = np.maximum.accumulate(win_h)
        max_pb_atr[t] = float(np.max(run_max - win_l) / a)
        run_min = np.minimum.accumulate(win_l)
        max_rally_atr[t] = float(np.max(win_h - run_min) / a)

    # Fresh 20d breakouts / breakdowns for regime & wedge
    prev_hi20 = s_h.rolling(20, min_periods=20).max().shift(1).to_numpy()
    prev_lo20 = s_l.rolling(20, min_periods=20).min().shift(1).to_numpy()
    prev_c_hi20 = s_c.rolling(20, min_periods=20).max().shift(1).to_numpy()
    prev_c_lo20 = s_c.rolling(20, min_periods=20).min().shift(1).to_numpy()

    is_nh_c = c > prev_c_hi20
    recent_nh_c = pd.Series(is_nh_c).shift(1).rolling(5, min_periods=5).sum().to_numpy() > 0
    brk20_up = is_nh_c & (~recent_nh_c) & np.isfinite(prev_c_hi20)

    is_nl_c = c < prev_c_lo20
    recent_nl_c = pd.Series(is_nl_c).shift(1).rolling(5, min_periods=5).sum().to_numpy() > 0
    brk20_dn = is_nl_c & (~recent_nl_c) & np.isfinite(prev_c_lo20)

    recent_brk_up_5 = pd.Series(brk20_up).rolling(5, min_periods=1).sum().to_numpy() > 0
    recent_brk_dn_5 = pd.Series(brk20_dn).rolling(5, min_periods=1).sum().to_numpy() > 0

    # Forward labels strictly inside truncated dev frame
    thr = THRESH[market_name]
    ret_10 = (s_log_c.shift(-10) - s_log_c).to_numpy()
    ret_20 = (s_log_c.shift(-20) - s_log_c).to_numpy()
    ret_40 = (s_log_c.shift(-40) - s_log_c).to_numpy()
    min_l_20 = s_l.iloc[::-1].rolling(20, min_periods=20).min().iloc[::-1].shift(-1).to_numpy()
    max_h_20 = s_h.iloc[::-1].rolling(20, min_periods=20).max().iloc[::-1].shift(-1).to_numpy()
    max_h_40 = s_h.iloc[::-1].rolling(40, min_periods=40).max().iloc[::-1].shift(-1).to_numpy()
    mdd_20 = (min_l_20 - c) / c
    mfe_20 = (max_h_20 - c) / c
    mfe_40 = (max_h_40 - c) / c
    crash_20 = np.where(np.isfinite(mdd_20), (mdd_20 <= thr["crash_20"]).astype(float), np.nan)
    big_20 = np.where(np.isfinite(mfe_20), (mfe_20 >= thr["big_20"]).astype(float), np.nan)
    big_40 = np.where(np.isfinite(mfe_40), (mfe_40 >= thr["big_40"]).astype(float), np.nan)

    # Whether price makes a new high above recent 20d high within next 20 bars
    recent_hi20_incl = s_h.rolling(20, min_periods=20).max().to_numpy()
    make_new_hi_20 = np.where(np.isfinite(max_h_20) & np.isfinite(recent_hi20_incl), (max_h_20 > recent_hi20_incl).astype(float), np.nan)

    eval_mask = pd.to_datetime(dates) >= eval_start

    def make_row(idx: int, extra: dict | None = None) -> dict:
        row = {
            "asset": asset,
            "date": dates[idx],
            "r20": r20[idx],
            "r5": r5[idx],
            "vol20": vol20[idx],
            "ma20_hold": ma20_hold[idx],
            "ma10_hold": ma10_hold[idx],
            "max_pb_atr": max_pb_atr[idx],
            "ret_10": ret_10[idx],
            "ret_20": ret_20[idx],
            "ret_40": ret_40[idx],
            "mdd_20": mdd_20[idx],
            "mfe_20": mfe_20[idx],
            "crash_20": crash_20[idx],
            "big_20": big_20[idx],
            "big_40": big_40[idx],
            "new_hi_20": make_new_hi_20[idx],
        }
        if extra:
            row.update(extra)
        return row

    # -----------------------------------------------------------------------
    # Part 1: Tight Channel vs Broad Channel (Daily Uptrend Panel + 1st Pullback Events)
    # -----------------------------------------------------------------------
    uptrend_rows: list[dict] = []
    tc_pb_rows: list[dict] = []
    bc_pb_rows: list[dict] = []

    # Track cooldown so 1st pullback events are distinct episodes (cooldown 5 bars)
    last_tc_pb = -100
    last_bc_pb = -100

    for t in range(30, n):
        if not eval_mask[t]:
            continue
        if not (np.isfinite(r20[t]) and np.isfinite(ma20[t]) and np.isfinite(ma20_lag10[t]) and np.isfinite(max_pb_atr[t])):
            continue

        # Uptrend arena: net positive past 20d return AND rising 20d MA
        is_up = (r20[t] > 0.0) and (ma20[t] > ma20_lag10[t])
        if not is_up:
            continue

        # Clear structural buckets (also keep continuous vars ma20_hold & max_pb_atr)
        is_tc = (ma20_hold[t] >= 0.90) and (max_pb_atr[t] <= 2.5)
        is_bc = (ma20_hold[t] <= 0.75) or (max_pb_atr[t] >= 3.5)
        ch_type = "TC" if is_tc else ("BC" if is_bc else "MID")
        uptrend_rows.append(make_row(t, {"ch_type": ch_type}))

        # First meaningful dip inside channel (yesterday was TC/BC, today pulls back: down close & touches MA10 or 2-day dip)
        prev_hold = ma20_hold[t - 1]
        prev_pb = max_pb_atr[t - 1]
        prev_up = (r20[t - 1] > 0.0) and (ma20[t - 1] > ma20_lag10[t - 1])
        is_dip_today = (c[t] < c[t - 1]) and ((l[t] <= ma10[t]) or (c[t - 1] < c[t - 2]))
        if prev_up and is_dip_today and np.isfinite(prev_hold) and np.isfinite(prev_pb):
            if (prev_hold >= 0.90) and (prev_pb <= 2.5) and (t - last_tc_pb >= 5):
                last_tc_pb = t
                tc_pb_rows.append(make_row(t, {"ch_type": "TC"}))
            elif ((prev_hold <= 0.75) or (prev_pb >= 3.5)) and (t - last_bc_pb >= 5):
                last_bc_pb = t
                bc_pb_rows.append(make_row(t, {"ch_type": "BC"}))

    # -----------------------------------------------------------------------
    # Part 2: Strictly Causal Real-Time 3-Push Wedge Detector (Top & Bottom)
    # -----------------------------------------------------------------------
    # Upward pushes (Wedge Top)
    p2_top_brk: list[dict] = []
    p3_top_brk: list[dict] = []
    p2_top_turn: list[dict] = []
    p3_top_turn: list[dict] = []
    p3_top_decay: list[dict] = []
    p3_top_strong: list[dict] = []

    push_cnt = 0
    peaks: list[float] = []
    peak_idxs: list[int] = []
    leg_lows: list[float] = []
    cur_peak = -np.inf
    cur_peak_idx = -1
    in_pb = False
    pb_low = np.inf
    turn_fired = False
    seq_start = -1

    for t in range(25, n):
        a = atr20[t]
        if not np.isfinite(a) or a <= EPS or not np.isfinite(prev_hi20[t]):
            continue

        if push_cnt == 0:
            # Start Push 1 on 20d high & above MA20
            if (h[t] > prev_hi20[t]) and (c[t] > ma20[t]):
                push_cnt = 1
                cur_peak = h[t]
                cur_peak_idx = t
                seq_start = t
                peaks = []
                peak_idxs = []
                leg_lows = []
                in_pb = False
                pb_low = l[t]
                turn_fired = False
        else:
            # Check structure invalidation: too old (>65 bars) or broke below prior leg low or 20d low
            broke_struct = (t - seq_start > 65) or (l[t] < prev_lo20[t])
            if leg_lows and (c[t] < leg_lows[-1]):
                broke_struct = True
            if broke_struct:
                push_cnt = 0
                continue

            if not in_pb:
                if h[t] > cur_peak:
                    cur_peak = h[t]
                    cur_peak_idx = t
                    pb_low = l[t]
                else:
                    if l[t] < pb_low:
                        pb_low = l[t]

                    # Right-side reversal turn check (first clear down turn after peak of push 2 or push 3)
                    is_turn_dn = (c[t] < c[t - 1]) and (l[t] < l[t - 1]) and (t - cur_peak_idx <= 4)
                    if is_turn_dn and (not turn_fired):
                        turn_fired = True
                        if push_cnt == 2 and eval_mask[t]:
                            p2_top_turn.append(make_row(t, {"push_cnt": 2}))
                        elif push_cnt >= 3 and len(peaks) >= 2 and eval_mask[t]:
                            g12 = peaks[-1] - peaks[-2]
                            g23 = cur_peak - peaks[-1]
                            decay = g23 / max(g12, 0.25 * a)
                            row = make_row(t, {"push_cnt": push_cnt, "decay": decay, "span": float(t - seq_start)})
                            p3_top_turn.append(row)
                            if decay < 0.75:
                                p3_top_decay.append(row)
                            else:
                                p3_top_strong.append(row)

                    # Meaningful pullback separates pushes (>= 1.0 ATR off peak and >= 2 bars)
                    if (cur_peak - pb_low >= 1.0 * a) and (t - cur_peak_idx >= 2):
                        in_pb = True
            else:
                if l[t] < pb_low:
                    pb_low = l[t]
                # New high after meaningful pullback -> next push!
                if h[t] > cur_peak:
                    peaks.append(cur_peak)
                    peak_idxs.append(cur_peak_idx)
                    leg_lows.append(pb_low)
                    push_cnt += 1
                    cur_peak = h[t]
                    cur_peak_idx = t
                    in_pb = False
                    pb_low = l[t]
                    turn_fired = False
                    if eval_mask[t]:
                        if push_cnt == 2:
                            p2_top_brk.append(make_row(t, {"push_cnt": 2}))
                        elif push_cnt == 3:
                            p3_top_brk.append(make_row(t, {"push_cnt": 3, "span": float(t - seq_start)}))

    # Downward pushes (Wedge Bottom — exact mirror)
    p2_bot_brk: list[dict] = []
    p3_bot_brk: list[dict] = []
    p2_bot_turn: list[dict] = []
    p3_bot_turn: list[dict] = []
    p3_bot_decay: list[dict] = []
    p3_bot_strong: list[dict] = []

    push_cnt = 0
    troughs: list[float] = []
    leg_highs: list[float] = []
    cur_tr = np.inf
    cur_tr_idx = -1
    in_rb = False
    rb_high = -np.inf
    turn_fired = False
    seq_start = -1

    for t in range(25, n):
        a = atr20[t]
        if not np.isfinite(a) or a <= EPS or not np.isfinite(prev_lo20[t]):
            continue

        if push_cnt == 0:
            if (l[t] < prev_lo20[t]) and (c[t] < ma20[t]):
                push_cnt = 1
                cur_tr = l[t]
                cur_tr_idx = t
                seq_start = t
                troughs = []
                leg_highs = []
                in_rb = False
                rb_high = h[t]
                turn_fired = False
        else:
            broke_struct = (t - seq_start > 65) or (h[t] > prev_hi20[t])
            if leg_highs and (c[t] > leg_highs[-1]):
                broke_struct = True
            if broke_struct:
                push_cnt = 0
                continue

            if not in_rb:
                if l[t] < cur_tr:
                    cur_tr = l[t]
                    cur_tr_idx = t
                    rb_high = h[t]
                else:
                    if h[t] > rb_high:
                        rb_high = h[t]

                    is_turn_up = (c[t] > c[t - 1]) and (h[t] > h[t - 1]) and (t - cur_tr_idx <= 4)
                    if is_turn_up and (not turn_fired):
                        turn_fired = True
                        if push_cnt == 2 and eval_mask[t]:
                            p2_bot_turn.append(make_row(t, {"push_cnt": 2}))
                        elif push_cnt >= 3 and len(troughs) >= 2 and eval_mask[t]:
                            d12 = troughs[-2] - troughs[-1]
                            d23 = troughs[-1] - cur_tr
                            decay = d23 / max(d12, 0.25 * a)
                            row = make_row(t, {"push_cnt": push_cnt, "decay": decay, "span": float(t - seq_start)})
                            p3_bot_turn.append(row)
                            if decay < 0.75:
                                p3_bot_decay.append(row)
                            else:
                                p3_bot_strong.append(row)

                    if (rb_high - cur_tr >= 1.0 * a) and (t - cur_tr_idx >= 2):
                        in_rb = True
            else:
                if h[t] > rb_high:
                    rb_high = h[t]
                if l[t] < cur_tr:
                    troughs.append(cur_tr)
                    leg_highs.append(rb_high)
                    push_cnt += 1
                    cur_tr = l[t]
                    cur_tr_idx = t
                    in_rb = False
                    rb_high = h[t]
                    turn_fired = False
                    if eval_mask[t]:
                        if push_cnt == 2:
                            p2_bot_brk.append(make_row(t, {"push_cnt": 2}))
                        elif push_cnt == 3:
                            p3_bot_brk.append(make_row(t, {"push_cnt": 3, "span": float(t - seq_start)}))

    # -----------------------------------------------------------------------
    # Part 3: Market Cycle State Classification for Transition Matrix
    # States (directional so up/down flips aren't masked):
    #   BO+ : recent fresh 20d upside breakout (within 5d) & strong push (c > prev_c_hi20 or r5 > 0)
    #   TC+ : uptrend tight channel (net 20d gain >= 2.0*ATR, ma20_hold >= 0.85, max_pb_atr <= 2.5)
    #   BC+ : uptrend broad channel (net 20d gain >= 1.2*ATR, not TC+)
    #   TR  : trading range (|net 20d move| < 1.2*ATR and not BO)
    #   DOWN: downside breakout / tight or broad down channel
    # -----------------------------------------------------------------------
    states = np.full(n, "", dtype=object)
    for t in range(25, n):
        a = atr20[t]
        if not np.isfinite(a) or a <= EPS or not np.isfinite(ma20_hold[t]) or not np.isfinite(max_pb_atr[t]):
            continue
        net20_atr = (c[t] - c[t - 20]) / a
        net5_atr = (c[t] - c[t - 5]) / a
        if recent_brk_up_5[t] and (net5_atr >= 1.0) and (c[t] > ma20[t]):
            states[t] = "BO+"
        elif recent_brk_dn_5[t] and (net5_atr <= -1.0) and (c[t] < ma20[t]):
            states[t] = "DOWN"
        elif net20_atr >= 2.0 and ma20_hold[t] >= 0.85 and max_pb_atr[t] <= 2.5:
            states[t] = "TC+"
        elif net20_atr >= 1.2:
            states[t] = "BC+"
        elif net20_atr <= -1.2:
            states[t] = "DOWN"
        else:
            states[t] = "TR"

    trans_rows: list[dict] = []
    for t in range(25, n - 20):
        if not eval_mask[t]:
            continue
        s0 = states[t]
        s10 = states[t + 10]
        s20 = states[t + 20]
        if s0 and s10 and s20:
            trans_rows.append({"asset": asset, "date": dates[t], "s0": s0, "s10": s10, "s20": s20})

    def to_df(rows: list[dict], dedup: bool = True) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["asset", "date", "r20", "r5", "vol20", "ret_10", "ret_20", "ret_40", "mdd_20", "mfe_20", "crash_20", "big_20", "big_40", "new_hi_20"])
        df = pd.DataFrame(rows)
        if dedup:
            df = df.drop_duplicates(subset=["asset", "date"]).reset_index(drop=True)
        return df

    return {
        "uptrend": to_df(uptrend_rows),
        "tc_pb": to_df(tc_pb_rows),
        "bc_pb": to_df(bc_pb_rows),
        "p2_top_brk": to_df(p2_top_brk),
        "p3_top_brk": to_df(p3_top_brk),
        "p2_top_turn": to_df(p2_top_turn),
        "p3_top_turn": to_df(p3_top_turn),
        "p3_top_decay": to_df(p3_top_decay),
        "p3_top_strong": to_df(p3_top_strong),
        "p2_bot_brk": to_df(p2_bot_brk),
        "p3_bot_brk": to_df(p3_bot_brk),
        "p2_bot_turn": to_df(p2_bot_turn),
        "p3_bot_turn": to_df(p3_bot_turn),
        "p3_bot_decay": to_df(p3_bot_decay),
        "p3_bot_strong": to_df(p3_bot_strong),
        "trans": pd.DataFrame(trans_rows) if trans_rows else pd.DataFrame(columns=["asset", "date", "s0", "s10", "s20"]),
    }


def assemble_market(raw: pd.DataFrame, market_name: str, eval_start: pd.Timestamp) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
    buckets: dict[str, list[pd.DataFrame]] = {}
    for _, g in raw.groupby("asset", sort=True):
        res = build_asset_study06(g, market_name, eval_start)
        for k, dfk in res.items():
            buckets.setdefault(k, []).append(dfk)
    merged = {k: pd.concat(v, ignore_index=True) for k, v in buckets.items()}
    all_dates = np.sort(raw.loc[raw["date"] >= eval_start, "date"].unique())
    return merged, all_dates


def fmt_line(label: str, df: pd.DataFrame, show_newhi: bool = False) -> str:
    st = s05.event_stats(df)
    if st.get("n", 0) == 0:
        return f"  {label:<24s} | n=   0"
    nh_str = ""
    if show_newhi and "new_hi_20" in df.columns:
        nh = df["new_hi_20"].dropna().mean() * 100.0
        nh_str = f" | 20天再创新高={nh:4.1f}%"
    return (
        f"  {label:<24s} | n={st['n']:>5d} | "
        f"10天={st['ret10']:+5.2f}% | 20天={st['ret20']:+5.2f}% (胜{st['win20']:4.1f}%) | "
        f"40天={st['ret40']:+5.2f}% | 回撤={st['mdd20']:6.2f}% 雷={st['crash20']:4.1f}% | "
        f"盈亏比={st['rr20']:4.2f}{nh_str}"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 110)
    log("Market Cycle V2 — Study 06 (方向 B): 窄通道 vs 宽通道、三推楔形与市场周期演化顺序")
    log("=" * 110)

    spx_raw = s01.load_spx_dev()
    sw_raw, sw_sid = s01.load_sw_l1_dev()
    log(
        f"封存边界确认: SPX 最后输入日={spx_raw['date'].max().date()} (<=2019-12-31, 全程不用open); "
        f"A股最后输入日={sw_raw['date'].max().date()} (<=2024-10-31), 快照={sw_sid}"
    )

    sw_ev, sw_dates = assemble_market(sw_raw, "A股申万L1", s01.SW_DEV_START)
    spx_ev, spx_dates = assemble_market(spx_raw, "SPX", s01.SPX_DEV_START)

    markets = [
        {
            "name": "A股申万L1",
            "ev": sw_ev,
            "all_dates": sw_dates,
            "splits": [
                ("前半 2016-2020", pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31")),
                ("后半 2021-2024.10", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-10-31")),
            ],
        },
        {
            "name": "SPX",
            "ev": spx_ev,
            "all_dates": spx_dates,
            "splits": [
                ("前半 1984-2001", pd.Timestamp("1984-01-01"), pd.Timestamp("2001-12-31")),
                ("后半 2002-2019", pd.Timestamp("2002-01-01"), pd.Timestamp("2019-12-31")),
            ],
        },
    ]

    # =========================================================================
    # 考题 1：窄通道 vs 宽通道（整体状态 + 首次回调节点 + 连续单挑）
    # =========================================================================
    log("\n" + "=" * 110)
    log("【考题 1】窄通道 (Tight Channel) vs 宽通道 (Broad Channel)")
    log("=" * 110)

    for m in markets:
        ev = m["ev"]
        all_dates = m["all_dates"]
        up = ev["uptrend"]
        tc = up[up["ch_type"] == "TC"]
        mid = up[up["ch_type"] == "MID"]
        bc = up[up["ch_type"] == "BC"]

        log(f"\n>>> 市场：{m['name']}（上升趋势总日样本={len(up)}）")
        log("  [1A. 上升段日常持有对比]")
        log(fmt_line("窄通道 TC (稳站MA20且浅踩)", tc, show_newhi=True))
        log(fmt_line("中间过渡通道 MID", mid, show_newhi=True))
        log(fmt_line("宽通道 BC (频破MA20或深踩)", bc, show_newhi=True))

        c_ret = s05.compare_vs_control(tc, bc, all_dates, ["r20", "vol20"], "ret_20", 20)
        c_cr = s05.compare_vs_control(tc, bc, all_dates, ["r20", "vol20"], "crash_20", 20)
        c_nh = s05.compare_vs_control(tc, bc, all_dates, ["r20", "vol20"], "new_hi_20", 20)
        sp_t = []
        for _, d0, d1 in m["splits"]:
            t_s = tc[(tc["date"] >= d0) & (tc["date"] <= d1)]
            b_s = bc[(bc["date"] >= d0) & (bc["date"] <= d1)]
            sp_t.append(s05.compare_vs_control(t_s, b_s, all_dates, ["r20", "vol20"], "ret_20", 20)["ctrl_t"])
        log(
            f"  [TC vs BC 控 r20+vol20] 20天收益差={c_ret['ctrl_diff']:+5.2f}%(t={c_ret['ctrl_t']:+5.2f}) | "
            f"踩雷差={c_cr['ctrl_diff']:+5.1f}%(t={c_cr['ctrl_t']:+5.2f}) | "
            f"再创新高率差={c_nh['ctrl_diff']:+5.1f}%(t={c_nh['ctrl_t']:+5.2f}) | 前/后半t={sp_t[0]:+.2f}/{sp_t[1]:+.2f}"
        )

        # Continuous single-feature checks inside all uptrend days
        log("  [连续特征单挑（扣 r20 + vol20，每+1标准差）]")
        for feat, flbl in [("ma20_hold", "MA20坚守占比(高=窄)"), ("max_pb_atr", "途中最大回撤ATR(高=宽)")]:
            rr = s03.run_event_ctrl_reg(up, "ret_20", feat, ["r20", "vol20"], all_dates, lag=20)
            rc = s03.run_event_ctrl_reg(up, "crash_20", feat, ["r20", "vol20"], all_dates, lag=20)
            rnh = s03.run_event_ctrl_reg(up, "new_hi_20", feat, ["r20", "vol20"], all_dates, lag=20)
            log(
                f"    {flbl:<22s}: 20天收益γ={rr['gamma']:+5.2f}%(t={rr['t']:+5.2f}) | "
                f"踩雷γ={rc['gamma']:+5.1f}%(t={rc['t']:+5.2f}) | 再创新高γ={rnh['gamma']:+5.1f}%(t={rnh['t']:+5.2f})"
            )

        log("  [1B. Al Brooks 关键节点：通道里遇到「第一次像样回调」后谁能再创新高？]")
        log(fmt_line("窄通道首次回调 (TC_PB)", ev["tc_pb"], show_newhi=True))
        log(fmt_line("宽通道首次回调 (BC_PB)", ev["bc_pb"], show_newhi=True))
        cp_ret = s05.compare_vs_control(ev["tc_pb"], ev["bc_pb"], all_dates, ["r20", "vol20", "r5"], "ret_20", 20)
        cp_cr = s05.compare_vs_control(ev["tc_pb"], ev["bc_pb"], all_dates, ["r20", "vol20", "r5"], "crash_20", 20)
        cp_nh = s05.compare_vs_control(ev["tc_pb"], ev["bc_pb"], all_dates, ["r20", "vol20", "r5"], "new_hi_20", 20)
        log(
            f"  [TC回调 vs BC回调 控基线] 20天收益差={cp_ret['ctrl_diff']:+5.2f}%(t={cp_ret['ctrl_t']:+5.2f}) | "
            f"踩雷差={cp_cr['ctrl_diff']:+5.1f}%(t={cp_cr['ctrl_t']:+5.2f}) | "
            f"再创新高率差={cp_nh['ctrl_diff']:+5.1f}%(t={cp_nh['ctrl_t']:+5.2f})"
        )

    # =========================================================================
    # 考题 2：三推楔形（Wedge Top & Wedge Bottom，严格因果实时数推）
    # =========================================================================
    log("\n" + "=" * 110)
    log("【考题 2】三推楔形（一鼓作气、再而衰、三而竭）：实时因果数推 vs 第 2 推对照")
    log("=" * 110)

    for m in markets:
        ev = m["ev"]
        all_dates = m["all_dates"]
        log(f"\n>>> 市场：{m['name']}")

        log("  [2A. 三推顶（Wedge Top）：真的比第 2 推更容易见顶大跌吗？]")
        log(fmt_line("对照：第2推创出新高当天", ev["p2_top_brk"]))
        log(fmt_line("左侧：第3推创出新高当天", ev["p3_top_brk"]))
        c_brk = s05.compare_vs_control(ev["p3_top_brk"], ev["p2_top_brk"], all_dates, ["r20", "vol20"], "ret_20", 20)
        c_brk_cr = s05.compare_vs_control(ev["p3_top_brk"], ev["p2_top_brk"], all_dates, ["r20", "vol20"], "crash_20", 20)
        log(
            f"    3推新高 vs 2推新高(控r20/vol): 20天收益差={c_brk['ctrl_diff']:+5.2f}%(t={c_brk['ctrl_t']:+5.2f}) | "
            f"踩雷差={c_brk_cr['ctrl_diff']:+5.1f}%(t={c_brk_cr['ctrl_t']:+5.2f})"
        )

        log(fmt_line("对照：第2推后首次收阴拐头", ev["p2_top_turn"]))
        log(fmt_line("右侧：全部3推后首次拐头", ev["p3_top_turn"]))
        log(fmt_line("  ├─ 衰减三推楔形(推不动)", ev["p3_top_decay"]))
        log(fmt_line("  └─ 加速三推(依然强势)", ev["p3_top_strong"]))
        for k, klbl in [("p3_top_turn", "全部3推拐头"), ("p3_top_decay", "衰减三推楔形")]:
            c_t = s05.compare_vs_control(ev[k], ev["p2_top_turn"], all_dates, ["r20", "vol20", "r5"], "ret_20", 20)
            c_tcr = s05.compare_vs_control(ev[k], ev["p2_top_turn"], all_dates, ["r20", "vol20", "r5"], "crash_20", 20)
            sp_t = []
            for _, d0, d1 in m["splits"]:
                t_s = ev[k][(ev[k]["date"] >= d0) & (ev[k]["date"] <= d1)]
                b_s = ev["p2_top_turn"][(ev["p2_top_turn"]["date"] >= d0) & (ev["p2_top_turn"]["date"] <= d1)]
                sp_t.append(s05.compare_vs_control(t_s, b_s, all_dates, ["r20", "vol20", "r5"], "ret_20", 20)["ctrl_t"])
            log(
                f"    {klbl:<12s} vs 2推拐头(控基线): 20天收益差={c_t['ctrl_diff']:+5.2f}%(t={c_t['ctrl_t']:+5.2f}) | "
                f"踩雷差={c_tcr['ctrl_diff']:+5.1f}%(t={c_tcr['ctrl_t']:+5.2f}) | 前/后半t={sp_t[0]:+.2f}/{sp_t[1]:+.2f}"
            )

        log("\n  [2B. 三推底（Wedge Bottom）：跌了三波砸不动了，真的比第 2 推超跌更容易反转吗？]")
        log(fmt_line("对照：第2推创出新低当天", ev["p2_bot_brk"]))
        log(fmt_line("左侧：第3推创出新低当天", ev["p3_bot_brk"]))
        cb_brk = s05.compare_vs_control(ev["p3_bot_brk"], ev["p2_bot_brk"], all_dates, ["r20", "vol20"], "ret_20", 20)
        log(
            f"    3推新低 vs 2推新低(控r20/vol): 20天收益差={cb_brk['ctrl_diff']:+5.2f}%(t={cb_brk['ctrl_t']:+5.2f})"
        )

        log(fmt_line("对照：第2推新低后首次抬头", ev["p2_bot_turn"]))
        log(fmt_line("右侧：全部3推新低后抬头", ev["p3_bot_turn"]))
        log(fmt_line("  ├─ 衰减三推底(砸不动了)", ev["p3_bot_decay"]))
        log(fmt_line("  └─ 加速下杀三推底", ev["p3_bot_strong"]))
        for k, klbl in [("p3_bot_turn", "全部3推底抬头"), ("p3_bot_decay", "衰减三推底")]:
            c_b = s05.compare_vs_control(ev[k], ev["p2_bot_turn"], all_dates, ["r20", "vol20", "r5"], "ret_20", 20)
            c_bcr = s05.compare_vs_control(ev[k], ev["p2_bot_turn"], all_dates, ["r20", "vol20", "r5"], "crash_20", 20)
            sp_t = []
            for _, d0, d1 in m["splits"]:
                t_s = ev[k][(ev[k]["date"] >= d0) & (ev[k]["date"] <= d1)]
                b_s = ev["p2_bot_turn"][(ev["p2_bot_turn"]["date"] >= d0) & (ev["p2_bot_turn"]["date"] <= d1)]
                sp_t.append(s05.compare_vs_control(t_s, b_s, all_dates, ["r20", "vol20", "r5"], "ret_20", 20)["ctrl_t"])
            log(
                f"    {klbl:<12s} vs 2推底抬头(控基线): 20天收益差={c_b['ctrl_diff']:+5.2f}%(t={c_b['ctrl_t']:+5.2f}) | "
                f"踩雷差={c_bcr['ctrl_diff']:+5.1f}%(t={c_bcr['ctrl_t']:+5.2f}) | 前/后半t={sp_t[0]:+.2f}/{sp_t[1]:+.2f}"
            )

    # =========================================================================
    # 考题 3：周期演化顺序（BO+ -> TC+ -> BC+ -> TR -> DOWN 转移矩阵与 Lift）
    # =========================================================================
    log("\n" + "=" * 110)
    log("【考题 3】市场周期演化顺序：真实行情按「突破(BO+) -> 窄通道(TC+) -> 宽通道(BC+) -> 区间(TR)」排队走吗？")
    log("=" * 110)

    order = ["BO+", "TC+", "BC+", "TR", "DOWN"]
    for m in markets:
        tr_df = m["ev"]["trans"]
        log(f"\n>>> 市场：{m['name']}（有效转移样本数={len(tr_df)}）")
        base_dist_10 = tr_df["s10"].value_counts(normalize=True).reindex(order).fillna(0.0)
        base_dist_20 = tr_df["s20"].value_counts(normalize=True).reindex(order).fillna(0.0)

        for horizon_lbl, col, base_dist in [("10个交易日后", "s10", base_dist_10), ("20个交易日后", "s20", base_dist_20)]:
            log(f"\n  [{horizon_lbl} 的状态转移概率 %（括号内为相对平时基准的倍数 Lift，>1.2 为明显偏好）]")
            hdr = f"  {'当前状态 (占比)':<18s} | " + " | ".join([f"-> {s:<9s}" for s in order])
            log(hdr)
            log("  " + "-" * (len(hdr) - 2))
            log(
                f"  {'平时基准分布':<18s} | "
                + " | ".join([f"{base_dist[s]*100:5.1f}%       " for s in order])
            )
            log("  " + "-" * (len(hdr) - 2))
            for s0 in order:
                sub = tr_df[tr_df["s0"] == s0]
                frac = len(sub) / max(len(tr_df), 1) * 100.0
                if len(sub) == 0:
                    continue
                vc = sub[col].value_counts(normalize=True).reindex(order).fillna(0.0)
                cells = []
                for s1 in order:
                    p = vc[s1] * 100.0
                    lift = vc[s1] / base_dist[s1] if base_dist[s1] > EPS else np.nan
                    cells.append(f"{p:4.1f}%({lift:4.2f}x)")
                log(f"  {s0:<5s} ({frac:4.1f}%,n={len(sub):>5d}) | " + " | ".join(cells))

    # =========================================================================
    # A 股逐年稳健性拆解（窄通道 vs 宽通道 & 衰减三推楔形）
    # =========================================================================
    log("\n" + "=" * 110)
    log("【A 股逐年拆解】核查窄通道 vs 宽通道在 2016-2024 每年的真实表现")
    log("=" * 110)
    up_sw = sw_ev["uptrend"].dropna(subset=["ret_20", "crash_20", "new_hi_20"]).copy()
    up_sw["year"] = pd.to_datetime(up_sw["date"]).dt.year
    log(f"  {'年份':<6s} {'TC天数':>6s} {'BC天数':>6s} | {'TC收益%':>8s} {'BC收益%':>8s} {'差%':>7s} | {'TC踩雷%':>8s} {'BC踩雷%':>8s} | {'TC新高%':>8s} {'BC新高%':>8s}")
    log("  " + "-" * 94)
    for yr, g in up_sw.groupby("year"):
        tc_y = g[g["ch_type"] == "TC"]
        bc_y = g[g["ch_type"] == "BC"]
        if len(tc_y) < 10 or len(bc_y) < 10:
            continue
        log(
            f"  {yr:<6d} {len(tc_y):>6d} {len(bc_y):>6d} | "
            f"{tc_y['ret_20'].mean()*100:>+7.2f}% {bc_y['ret_20'].mean()*100:>+7.2f}% "
            f"{(tc_y['ret_20'].mean()-bc_y['ret_20'].mean())*100:>+6.2f}% | "
            f"{tc_y['crash_20'].mean()*100:>7.1f}% {bc_y['crash_20'].mean()*100:>7.1f}% | "
            f"{tc_y['new_hi_20'].mean()*100:>7.1f}% {bc_y['new_hi_20'].mean()*100:>7.1f}%"
        )

    report_path = OUTPUT_DIR / "06_ordered_structures_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\n" + "=" * 110)
    log(f"报告已保存: {report_path}")


if __name__ == "__main__":
    main()
