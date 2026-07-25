---
title: Phase 1A 连续行为表示工作协议
status: development-v0.3
updated: 2026-07-25
owner: phase1a-research
---

# Phase 1A 连续行为表示工作协议

## 1. 目的与边界

Phase 1A 建立 SPX 日线的**基础连续行为表示**。当前研究无需显式腿、摆动或
Episode 起点的固定窗口、多尺度量：方向、效率、双向性。

已实现：

- `BM-E-01-OHLC-min v1` 多尺度效率，及等权 / 中长期优先综合；
- `BM-D-01-theilsen-atr v1` 多尺度方向漂移，及等权 / 中长期优先综合与尺度一致性；
- `BM-B-01-entropy-overlap v1` 多尺度双向性，及等权 / 中长期优先综合。

它们是连续行为层的测量，不是完整状态机，也不预设 Brooks 阶段标签。

不做：因果路径分段、当前腿效率、回调比例、路径年龄、Episode、通道、三推、双底、
完整形态分类、交易信号或仓位决策。

这些对象若在后续成为明确研究问题，必须在 Phase 2 建立独立的结构解析协议。当前
`bm_g_01_atr_reversal_v1` 保留为 `structures` 下的因果摆动候选和工程资产，不属于
本协议的基础表示或比较对象。依据 D-013。

设计决策记录：`.pi/grill/2026-07-25-0212——方向与双向性设计.md`。

## 2. 数据与样本

| 项目 | 值 |
|---|---|
| `snapshot_id` | `spx_daily_2026-07-21_d828fbc8` |
| `sample_id` | `spx_ohlc_main_1984` |
| 输入 | `date, open, high, low, close, tr_pct, atr_pct_14` |
| 研究日表起点 | 1984-02-10 |
| 当前特征族齐套日 | 1984-05-01 |
| 当前正式样本 | 1984-05-01 至快照末日 |

`tr_pct` 与 `atr_pct_14` 只从共享研究日表读取。效率与双向性当前不使用 ATR；方向
使用共享 `atr_pct_14` 做标准化，禁止模块内重算 TR 或 ATR。

## 3. 共同时间契约与尺度政策

每个连续输出都有 `as_of`，且只读取截至该日收盘的数据。固定窗口测量必须声明：

```text
reference_type    fixed_calendar_window
window            最近 n 个已完成的日度价格过程
as_of             该窗口结束日收盘后可用的时点
feature_version   公式与实现版本
```

窗口口径：`$n` 表示最近 $n$ 个交易日过程；窗口开始前一日只作已知边界参考，不把
未来信息写入历史。

预声明窗口族：

$$
n \in \{5,10,20,55\}
$$

跨尺度压缩政策（`E/D/B` 共用）：

$$
w^{\mathrm{equal}}=(0.25,0.25,0.25,0.25)
$$

$$
w^{\mathrm{midlong}}=(0.10,0.20,0.30,0.40)
$$

综合版本在任一尺度缺失时也记缺失，**不重归一权重**。

图表契约：

```text
每个特征一个附图
默认显示：等权综合、中长期优先综合
可点开显示：5 / 10 / 20 / 55 底层尺度
```

## 4. 效率 `BM-E-01-OHLC-min v1`

对每根日线，定义与前收和 OHLC 相容的最短可见路径：

$$
L_i^{\min}
=
|O_i-C_{i-1}|
+
\min
\begin{cases}
|O_i-H_i|+|H_i-L_i|+|L_i-C_i|,\\
|O_i-L_i|+|L_i-H_i|+|H_i-C_i|.
\end{cases}
$$

固定窗口效率：

$$
E^{\mathrm{OHLC\text{-}min}}_{t,n}
=
\frac{|C_t-C_{t-n}|}
{\sum_{i=t-n+1}^{t}L_i^{\min}}
$$

综合：

$$
E^{\mathrm{equal}}_t=\sum_n w^{\mathrm{equal}}_n E_{t,n},
\qquad
E^{\mathrm{midlong}}_t=\sum_n w^{\mathrm{midlong}}_n E_{t,n}
$$

分母为零时效率缺失并标记 `zero_path`。

主要字段：

```text
efficiency_ohlc_min_{5,10,20,55}[+_status]
efficiency_ohlc_min_equal[+_status]
efficiency_ohlc_min_midlong[+_status]
```

## 5. 方向 `BM-D-01-theilsen-atr v1`

职责：波动调整后的价格中心迁移速度。不是带符号效率，也不是趋势腿。

对窗口 $n$：

1. 取收盘价 $C_{t-n},\ldots,C_t$ 的对数；
2. 用 Theil-Sen 估计稳健斜率 $\hat\beta_{t,n}$（单位：每日对数收益）；
3. 用同窗口过程日 $i\in[t-n+1,t]$ 的 `atr_pct_14` 中位数标准化：

$$
d_{t,n}
=
\frac{\hat\beta_{t,n}}
{\operatorname{median}_{i=t-n+1}^{t}(atr\_pct_{14,i})}
$$

综合与尺度一致性：

$$
D^{\mathrm{equal}}_t=\sum_n w^{\mathrm{equal}}_n d_{t,n}
$$

$$
D^{\mathrm{midlong}}_t=\sum_n w^{\mathrm{midlong}}_n d_{t,n}
$$

$$
A^{(v)}_t
=
\frac{\left|\sum_n w^{(v)}_n d_{t,n}\right|}
{\sum_n w^{(v)}_n |d_{t,n}|}
$$

正式输出保留带符号 `raw`；当前不使用 `tanh`。ATR 中位数为零时记缺失并标记
`zero_scale`；一致性分母为零时标记 `zero_direction`。

主要字段：

```text
direction_drift_{5,10,20,55}[+_status]
direction_drift_equal_raw[+_status]
direction_drift_midlong_raw[+_status]
direction_scale_agreement_{equal,midlong}[+_status]
```

## 6. 双向性 `BM-B-01-entropy-overlap v1`

职责：固定窗口内的双向竞争与价格反复再交易，不是 $1-E$。

### 6.1 活跃度加权涨跌熵

在最近 $n$ 个收盘变动中：

$$
q=\frac{N_++N_-}{n}
$$

若 $N_++N_->0$：

$$
H^{active}_{t,n}
=
q\cdot
\left[
-\frac{p_+\log p_++p_-\log p_-}{\log 2}
\right]
$$

其中 $p_+$、$p_-$ 只在非零涨跌日上计算。全零涨跌时 $H^{active}=0$。

### 6.2 相邻 K 线交并比重叠

$$
I_i=\max(0,\min(H_i,H_{i-1})-\max(L_i,L_{i-1}))
$$

$$
O^{union}_i
=
\frac{I_i}
{(H_i-L_i)+(H_{i-1}-L_{i-1})-I_i}
$$

并集为零时该日对缺失。窗口重叠为最近 $n$ 个有效日对的算术平均；若无有效日对，
重叠缺失并标记 `zero_range`。

### 6.3 单尺度与综合

$$
B^{v0.1}_{t,n}
=
0.5H^{active}_{t,n}
+
0.5\overline{O}^{union}_{t,n}
$$

$$
B^{\mathrm{equal}}_t=\sum_n w^{\mathrm{equal}}_n B_{t,n},
\qquad
B^{\mathrm{midlong}}_t=\sum_n w^{\mathrm{midlong}}_n B_{t,n}
$$

主要字段：

```text
two_sidedness_entropy_{5,10,20,55}
two_sidedness_overlap_{5,10,20,55}
two_sidedness_v1_{5,10,20,55}[+_status]
two_sidedness_equal_v1[+_status]
two_sidedness_midlong_v1[+_status]
```

中心穿越与摆动对称不在本版本。

## 7. 开发验收

每个连续特征族必须满足：

1. 手算性质测试（单边、往返、零路径/零尺度/零区间等边界）；
2. 有效值落在定义范围内（`E/B` 在 $[0,1]$；`D` 带符号无上界）；
3. 批量计算与任意截至日独立重算完全一致；
4. 默认窗口族从 1984-05-01 齐套；
5. 图表审计时区分短窗口局部波动与中长窗口持续过程。

## 8. 三个研究块

### 开发块（当前）

- 允许改正连续公式实现、补足字段、记录版本变化；
- 允许使用本地图表审计窗口语义和反例；
- 不得把调试图或个别历史片段当作状态表示有效性的结论；
- 不得因路径候选已存在而将其重新引入基础连续特征。

### 选择块（后续）

在预先划定的连续时间块中检查：

- 多尺度窗口的稳定性与尺度关系；
- 等权与中长期优先两种压缩政策的解释差异；
- 连续行为量对方向性、双向性和扩张过程的描述覆盖；
- 新连续特征相对已有尺度场的信息保留；
- 固定窗口连续行为层是否足以作为后续状态压缩的底图。

### 锁定确认块（后续）

在查看确认块前冻结：快照、样本起点、连续特征公式、窗口族、综合权重、正式结果、
比较基线、停止条件和报告格式。

## 9. 当前结论边界

本协议只定义可复现的连续价格过程实验。它尚未证明固定窗口足以构成市场状态，也不
产生 `BALANCE`、`DIRECTIONAL` 或其他市场状态，更不产生交易结论。

扩张 `X` 仍待设计。当前 `bm_g_01_atr_reversal_v1` 不构成本协议连续表示的同层对照。
