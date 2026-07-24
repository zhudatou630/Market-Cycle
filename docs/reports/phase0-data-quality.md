---
title: Phase 0 数据质量报告与样本边界
status: canonical
updated: 2026-07-23
owner: phase0-data
snapshot_id: spx_daily_2026-07-21_d828fbc8
note: 研究行起点已被 D-010 收紧为齐套日 1984-02-10；见第 3 节。
---

# Phase 0 数据质量报告与样本边界

本文是 Phase 0 的正式产物，并已按 D-010 标注齐套研究起点。它固定当前研究使用
的数据快照、质量事实和样本资格规则。概念边界仍以
[foundation.md](../foundation.md) 为准；已确认选择以
[decisions.md](../decisions.md) 的 D-008、D-010 为准（D-009 中研究起点已被
D-010 取代，OHLC 干净下限事实仍有效）。

## 1. 快照标识

| 项 | 值 |
|---|---|
| `snapshot_id` | `spx_daily_2026-07-21_d828fbc8` |
| 不可变文件 | `data/snapshots/spx_daily_2026-07-21_d828fbc8.parquet` |
| 元数据 | `data/snapshots/spx_daily_2026-07-21_d828fbc8.json` |
| SHA-256 | `d828fbc8a85d70a4de240b67de66e88d8b09767622a20743a04236f896373e9d` |
| 字节数 | 771,444 |
| 符号 | Yahoo `^GSPC`（研究别名 SPX/GSPC） |
| 频率 | 日线 |
| 字段 | `date, open, high, low, close` |
| 全量行数 | 24,754 |
| 全量日期区间 | 1927-12-30 → 2026-07-21 |
| 抓取设定 | `yfinance`，`auto_adjust=False`，`actions=False` |

工作缓存 `data/raw/spx_daily.parquet` 仍可按 D-002 刷新覆盖。正式实验、报告和
复现只引用 `snapshot_id` 及其不可变文件，不引用“当前机器上的 raw 碰巧内容”。

若未来需要更新截止日，应生成**新的** snapshot 文件与 id，不得覆盖本快照。

## 2. 全量质量审计

对上述快照实测：

| 检查 | 结果 |
|---|---|
| 重复交易日 | 0 |
| OHLC 空值 | 0 |
| 日期单调递增 | 是 |
| `high >= max(open, close)` | 全部成立 |
| `low <= min(open, close)` | 全部成立 |
| `high >= low` | 全部成立 |
| 零区间 bar（`high == low`） | 8,547 |
| 退化 OHLC（`open=high=low=close`） | 8,547 |
| 零区间但非完全退化 | 0 |

区间结构：

| 区间 | 行数 | 零区间 / 退化 |
|---|---:|---|
| 1927-12-30 至 1961-12-29 | 8,509 | 全部退化 |
| 1962-01-02 至 1983-12-30 | 5,525 | 其中 38 根退化；最后一根 1983-06-30 |
| 1984-01-03 至 2026-07-21 | 10,720 | 0 |

说明：

- 本快照中“零区间”与“退化 OHLC”完全重合；
- 1962 年起才出现可用的非退化日线 OHLC；
- 1984 年起无零区间，适合作为完整 OHLC 几何特征的主样本；
- 原始层保留全量历史，不删除、不伪造、不插值 high/low。

日历缺口（非数据错误，仅记录）：

- 最大缺口 12 个日历日：1933-03-03 → 1933-03-15（银行假相关）；
- 现代样本中较显著者：2001-09-10 → 2001-09-17（7 日）。
- 周末与常规假日造成的 3 日左右间隔属正常交易日历，不单独判 bar 无效。

## 3. 主研究样本（D-010，继承 D-009 的 1984 边界）

Phase 1A 及后续在本快照上的**唯一对外主研究样本**：

```text
sample_id:      spx_ohlc_main_1984
snapshot_id:    spx_daily_2026-07-21_d828fbc8
rule:           date >= 1984-02-10
rows:           10,692
range:          1984-02-10 → 2026-07-21
zero_range:     0
ruler:          tr_pct, atr_pct_14（Wilder-14 / Close_t；28 根 TR 预热后齐套）
research_file:  data/snapshots/spx_daily_2026-07-21_d828fbc8__ruler_v1_atr14w28__research.parquet
```

研究日表与 OHLC **同级落盘**：每行是一天的 `open/high/low/close` 加上尺子列
`tr_pct`、`atr_pct_14`。下游默认读该文件，不现场重算波动尺。

分层说明：

- **质量下限**：1984-01-03 起 OHLC 无零区间（D-009 事实仍成立），可供尺子内部预热；
- **研究表起点**：1984-02-10 起（D-010），为 `tr_pct` 与 `atr_pct_14` 均可用的齐套首日；
- **1984 以前**：不进入当前研究协议；快照仍保留 1927–1983 全量事实供审计。

不设 1962+ 敏感性主样本，不设 pre-1962 close-only 为当前必做项。若将来重开更早
历史或改预热规则，必须新决策，不得静默扩大 `spx_ohlc_main_1984` 含义。

## 4. 研究层样本资格规则

规则版本：`eligibility_v1`，研究行日历按 **D-010** 齐套裁切（`date >= 1984-02-10`）

### 4.1 原始层

- 快照与 raw 缓存保存 Yahoo 提供的全量 OHLC 事实；
- 不因质量问题删除历史 bar；
- 不把退化 bar 改写成虚构的 high/low；
- 不在数据层对价格做前向填充或区间插值。

### 4.2 Bar 级可用性

```text
ohlc_clean_floor(t)   := date_t >= 1984-01-03 且属于 snapshot   # 预热可加载
bar_in_main_sample(t) := date_t >= 1984-02-10 且属于 snapshot   # 研究表行
bar_ohlc_usable(t)    := high_t > low_t
bar_close_usable(t)   := close_t 非空
```

对研究表 `spx_ohlc_main_1984`（≥1984-02-10），全部行同时满足 OHLC/close 可用，
且尺子列 `tr_pct`、`atr_pct_14` 无缺失。

研究表之外的 bar 对测量模块一律视为**不可用**（尺子内部预热除外）。

### 4.3 特征级可用性

特征或事件在交易日 `t` 可写入研究表，当且仅当：

1. `bar_in_main_sample(t)`；
2. 该特征声明的字段需求在 `t` 满足（full OHLC 或 close-only）；
3. 若依赖回顾窗口 `W`（含当日），则窗口内每一个 bar 都在**研究表**内且满足对应
   字段可用性；
4. 窗口不足 `W` 时，该特征记为不可用（缺失），**不得**用研究起点前或 1984 前
   bar 悄悄补窗口。尺子自身的 28 日 ATR 预热只在数据层内部完成，不把预热行
   暴露给测量模块。

禁止：

- 跳过窗口中的坏点继续计算；
- 用相邻日插值修复 high/low；
- 用未来 bar 回填历史可用性。

### 4.4 周线

周线仍由日线重采样（D-001），不另存权威文件。研究用周线必须由**同一
snapshot** 的日线生成；周线 bar 落入主样本的条件是其聚合所用日线均满足主样本
资格。当前 Phase 1A 默认日线；若使用周线，协议中单独声明。

## 5. 对后续阶段的含义

### 5.1 Phase 1A 必须引用

任一测量、事件或验证结果至少记录：

```text
snapshot_id:      spx_daily_2026-07-21_d828fbc8
sample_id:        spx_ohlc_main_1984   # date >= 1984-02-10
ruler:            D-010 / tr_pct + atr_pct_14
eligibility:      eligibility_v1 + D-010 齐套裁切
feature_asof:     计算截至日
```

缺少上述标识的结果不能视为可复现实验。

### 5.2 不再由数据质量解释的问题

在主样本内，OHLC 几何（TR、重叠、实体、影线、区间）的输入完整性已成立。此后
若特征无效或结论不稳，应优先检查公式、时点、参数和验证设计，而不是早期退化
OHLC。

### 5.3 明确非结论

本报告不声称：

- 1984 起点对经济学或 Brooks 语义“正确”；
- Yahoo `^GSPC` 与某官方 SPX 历史逐点一致；
- 主样本长度已足够证明任何行为假说。

它只关闭 Phase 0：输入边界可追溯，主样本干净且规则明确。

## 6. Phase 0 完成判定

| 条件 | 状态 |
|---|---|
| 不可变 snapshot + hash | 已完成 |
| 质量事实与实测一致并成文 | 已完成 |
| 主样本起点决定（关闭 Q-003） | 已完成：研究表 1984-02-10 起（D-010） |
| 资格规则写明且禁止静默修复 | 已完成：`eligibility_v1` |
| 研究锚点可引用 snapshot/sample/rule | 已完成 |

Phase 0 收口。下一工作是 Phase 1A 的工作协议初稿与候选 `BM-*` 选择，而不是继续
扩大数据层。
