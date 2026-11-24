---
title: Phase 1A 状态耦合探针：操作指南与实施协议
status: exploratory-operation-guide
updated: 2026-07-28
owner: phase1a-research
snapshot_id: spx_daily_2026-07-21_d828fbc8
sample_id: spx_ohlc_main_1984
probe_version: phase1a_state_coupling_probe_v0
input_feature_versions:
  - bm_e_01_ohlc_min_v1
  - bm_d_01_theilsen_atr_v1
  - bm_b_01_entropy_overlap_v1
  - bm_x_01_ohlc_impulse_v1
  - bm_x_dep_01_prior_range_clearance_v1
---

# Phase 1A 状态耦合探针：操作指南与实施协议

## 1. 文档定位

本文给后续 Agent 提供一套可直接实施的“状态耦合探针”方案。它的目的不是现在就冻结
最终的 Brooks Market Cycle 定义，也不是跳过 Phase 1A 的研究纪律，而是把当前已经实现的
`E / D / A / B / Activity / X / clearance` 装入一个最小、透明、可抛弃的读取层，提前跑通
一次从测量、耦合、解释、审计到消融的纵向闭环。

本文是探索性操作指南，不是 canonical decision，也不是正式状态协议。未经用户明确确认：

- 不得把本文中的状态名称、阈值、时间分块或文件结构写成已确认的 `D-*` 决策；
- 不得把探针输出接入正式市场状态 API；
- 不得把候选标签解释成交易信号、收益概率或标签正确概率；
- 不得因为探针结果不好而边看结果边修改底层 `E/D/B/X/clearance v1`；
- 不得把固定窗口连续量重新命名为当前腿、真实交易区间或形态结构。

当前相关权威文件：

- [项目决策记录](../decisions.md)，特别是 D-013；
- [Phase 1A 连续行为表示工作协议](../phase1a-working-protocol.md)；
- [连续特征候选开发审计](phase1a-continuous-feature-development-audit.md)；
- [扩张 X 第一性原理设计分析](phase1a-expansion-x-design-analysis.md)；
- [研究计划](../research-plan.md)；
- [研究基础](../foundation.md)。

一句话概括本文：

> 不继续无限打磨孤立变量，也不立即量产最终状态机；先冻结现有传感器，搭一台最小试验车，
> 用状态稳定性、压缩损失、前向行为和消融结果判断哪些变量有用、哪些重复、哪些缺口必须交给
> Phase 2。

---

## 2. 为什么现在需要耦合探针

当前项目已经完成：

- 冻结 SPX 日线 OHLC 研究快照与共享波动尺；
- `E/D/B/X/clearance v1` 的版本化实现；
- 性质测试、边界状态与逐日前缀重放；
- 各变量的分项真实窗口审计；
- 对 `B` 长窗饱和、安静/剧烈盲区的明确记录；
- 对 `X` 中活动背景、当日冲击、收盘结果、Gap 与历史位置的分层。

但仍未回答：

1. `B` 的盲区能否由 `ActivityLevel/X` 在系统层补足，而无需修改 `B`；
2. `E` 与 `|D|` 高相关究竟是有意义的职责互补，还是大部分冗余；
3. clearance 是否提供独立历史位置，还是主要重复长期方向 `D`；
4. `X` 是否真的补充了现有 `E/D/B`，还是只把波动换了一种表达；
5. 当前连续向量能否压缩成稳定、可解释的体制与转换事件；
6. 哪些失败案例确实需要路径、区域、Retention 或 Episode。

这些问题仅靠继续逐个优化特征无法完整回答。只有把变量放入一个下游读取任务，进行组合、
消融和残差检查，才能观察它们的系统价值。

因此，本探针采用循环式研究路线：

```text
冻结一组候选测量
→ 构造最小读取层
→ 检查耦合、稳定、压缩和前向行为
→ 删除冗余或记录缺口
→ 再决定修改特征、进入 Phase 1B，或为明确残差立项 Phase 2
```

这不是提前宣布 Phase 1A 已通过，而是用一个可失败的纵向切片检验当前变量是否值得继续。

---

## 3. 核心架构裁决

### 3.1 耦合解释层，不耦合测量层

推荐架构：

```text
冻结 OHLC 与共享 ruler
        ↓
E / D / A / B / Activity / X / clearance
        ↓
state_coupling_probe_v0
        ↓
候选持续体制 + 候选转换事件 + 解释属性 + 证据明细
        ↓
历史窗口审计 + 稳定性 + 压缩损失 + 前向结果 + 消融
```

底层测量继续保持独立、可复现、可前缀重放。耦合只发生在新的原型层中。

禁止把不同职责的变量直接压成一个不可诊断总分，例如：

```text
MarketStateScore = 0.3 D + 0.2 E - 0.2 B + 0.3 X
```

这类总分失败时无法判断问题来自定义、标准化、权重、阈值还是状态概念。

### 3.2 第一轮不强迫四个互斥 Market Cycle 状态

当前优先比较的是：

```text
A. 原始连续多尺度向量

B. BALANCE_LIKE / DIRECTIONAL 持续体制
   + 独立转换事件

C. BALANCE / EXPANSION / DIRECTIONAL / REBALANCING
   + UNRESOLVED
```

第一轮建议先实施方案 B。原因是：

- `BALANCE_LIKE` 与 `DIRECTIONAL` 更像可以持续的背景体制；
- `EXPANSION` 更像突然发生的转换或冲击；
- `REBALANCING` 需要此前方向背景，天然带上下文；
- 当前还没有真实区域边界、Retention 或路径 Episode，直接把四者做成完全平级状态会产生
  过强语义。

第一轮输出分为四类：

| 层次 | 输出 | 作用 |
|---|---|---|
| 持续体制候选 | `DIRECTIONAL_UP` / `DIRECTIONAL_DOWN` / `BALANCE_LIKE` / `UNRESOLVED` | 描述主要背景 |
| 转换事件候选 | `EXPANSION_UP` / `EXPANSION_DOWN` / `TURBULENT_SHOCK` / `SCALE_CONFLICT` | 描述当日新变化 |
| 上下文事件 | `REBALANCING_CANDIDATE` / `FAILED_EXPANSION_CANDIDATE` 等后续可选项 | 需要历史上下文 |
| 属性 | 活动、直接性、双向争夺、位置、Gap | 保留状态标签未表达的信息 |

使用 `BALANCE_LIKE` 而非 `BALANCE`，因为当前没有识别真实交易区间。

使用 `EXPANSION_CANDIDATE` 或转换事件，而非确认扩张状态，因为当前没有未来接受结果。

---

## 4. 当前输入及职责边界

| 家族 | 主字段 | 人话 | 不允许声称 |
|---|---|---|---|
| `D` | `direction_drift_*` | 价格中心向哪边迁移、速度多快 | 当前腿、趋势已确认 |
| `A` | `direction_scale_agreement_*` | 短中长期方向是否一致 | 概率、标签置信度 |
| `E` | `efficiency_ohlc_min_*` | 可见路径有多少转化成净位移 | 趋势强度、未来延续概率 |
| `B` | entropy / overlap / composite | 涨跌是否混杂、相邻区域是否反复覆盖 | 波动大小、剧烈程度 |
| Activity | `activity_level_atr_pct_prev` | 市场原本有多活跃 | 今天是否出现新冲击 |
| `X_range` | `expansion_range_atr_prev` | 今天相对昨日冻结尺动得多猛 | 方向已解决 |
| `X_close` | `expansion_close_atr_prev` | 今天留下多少带方向收盘位移 | 后续接受 |
| `X_share` | `expansion_close_share` | 当日 TR 有多少保留为收盘净位移 | 订单流占比 |
| Gap | `expansion_gap_prev_range_atr` | 冲击是否从昨日完整范围外开盘 | 额外扩张强度 |
| clearance | up/down 20/55 | 当前收盘相对旧日线分布的位置 | 突破、真实区域、接受 |

探针必须保存原始字段和版本信息，任何派生轴都不得替代原始测量记录。

---

## 5. 建议的派生读取轴

这些读取轴只用于探针，不修改底层特征协议。

### 5.1 方向共识

使用中长期方向与尺度一致性：

\[
DirectionConsensus_t
=
D^{midlong}_t A^{midlong}_t
\]

并保存：

\[
DirectionStrength_t
=
|D^{midlong}_t| A^{midlong}_t
\]

解释：

- `D` 决定方向与迁移速度；
- `A` 低时，短中长期互相抵消，方向判断应降级；
- `DirectionConsensus` 接近零可能是各尺度都弱，也可能是尺度冲突，必须同时读 `A` 和底层尺度。

不得把该轴解释成方向正确概率。

### 5.2 路径直接性

主读：

```text
PathDirectness = efficiency_ohlc_min_midlong
```

辅助读短窗与长窗：

```text
E_5 / E_20 / E_55
```

`E` 主要作为属性和辅助证据，不应成为 `DIRECTIONAL` 的绝对硬门槛。慢牛、阶梯上涨或宽通道
可以持续正方向，但效率只处于中低水平。

### 5.3 双向争夺

第一轮同时保存：

```text
B_equal
B_5 / B_20
entropy_5 / entropy_20
overlap_5 / overlap_20
```

不要只保留 `B` 总分。当前审计显示 entropy 与 overlap 相关很低，等权相加可能掩盖不同机制。

第一轮不要求 `B` 区分安静与剧烈；这由 Activity 和 X 提供上下文。

### 5.4 冲击大小与当日方向结果

```text
ShockSize       = X_range
ShockDirection  = sign(X_close)
CloseDisplacement = X_close
CloseRetention  = abs(X_share)
```

`CloseRetention` 只是当日收盘保留，不是未来 Retention。

### 5.5 历史位置

对 `n ∈ {20,55}`：

\[
PositionBias_{t,n}
=
Clearance^{up}_{t,n}-Clearance^{down}_{t,n}
\]

范围为 `[-1,1]`：

- 接近 `+1`：当前收盘高于大多数旧完整日线；
- 接近 `-1`：当前收盘低于大多数旧完整日线；
- 接近 `0`：上下偏置弱，或两边都有旧日线。

再定义仅供探针阅读的历史覆盖：

\[
HistoricalCoverage_{t,n}
=
1-Clearance^{up}_{t,n}-Clearance^{down}_{t,n}
\]

它表示过去有多少根完整日线高低范围仍覆盖当前收盘。

该式由现有 clearance 直接推导，但不是当前正式测量字段。Agent 若持久化该字段，应明确标为
probe-derived，并记录公式版本。

### 5.6 尺度冲突

连续版本：

\[
ScaleConflict_t=1-A^{midlong}_t
\]

离散诊断：

```text
sign(D_5) != sign(D_55)
sign(D_equal) != sign(D_midlong)
```

尺度冲突可表示回调、熊市反弹、趋势受到挑战或转换过程，但本身不能区分这些具体结构。

---

## 6. 两阶段原型策略

### 6.1 v0a：无记忆日度读取层

第一步只根据当日及以前已经计算的连续特征产生：

- 当日 `regime_candidate`；
- 当日可并存的 `transition_candidates`；
- 解释属性；
- 规则命中与距离阈值的 margin。

无记忆版本不进行：

- 进入/退出滞回；
- Episode 起点；
- 状态年龄；
- 事后回填；
- Retention；
- 当前腿或回调判断。

它的价值是先观察静态读取逻辑本身是否过度抖动、覆盖是否合理、证据是否可解释。

### 6.2 v0b：因果上下文版本

只有 v0a 的基本语义和证据追踪稳定后，才增加：

- 前一日或此前确认体制；
- 进入和退出滞回；
- `REBALANCING_CANDIDATE`；
- `FAILED_EXPANSION_CANDIDATE`；
- 状态持续时间与候选确认时间。

所有上下文只能向前更新，不得根据未来结果回填历史标签。

---

## 7. 阈值政策

### 7.1 第一轮不用未来收益寻找阈值

建议使用开发块分位数作为透明脚手架：

```text
低       ≤ 开发块 p30
高       ≥ 开发块 p70
极端     ≥ 开发块 p90
很低     ≤ 开发块 p20
```

阈值必须：

- 仅从开发块计算；
- 写入独立配置和元数据；
- 在选择块运行前冻结；
- 不因个别历史案例不好看而修改；
- 不被描述成市场天然阈值。

### 7.2 建议初始时间块

建议作为第一轮可执行方案：

```text
开发块：1984-05-01 至 2004-12-31
选择块：2005-01-01 至 2014-12-31
确认块：2015-01-01 至 2026-07-21
```

注意：项目已经审阅过全样本分布和多个著名历史窗口。因此最后一段只能称“冻结确认块”，
不能声称是完全未见的纯盲测。

更强的外部验证以后需要新增未来数据，或使用未参与设计的其他指数。

---

## 8. v0a 持续体制候选规则

以下规则是启动脚手架，不是正式 Market Cycle 定义。

### 8.1 `DIRECTIONAL_UP`

最低要求：

```text
D_midlong > 0
DirectionStrength >= development p70
A_midlong > development p30
```

同时输出属性：

```text
path_directness = low / mid / high
contest = low / mid / high
activity = low / mid / high
position_20 / position_55
```

解释示例：

```text
DIRECTIONAL_UP + high directness + low contest
    干净单边推进

DIRECTIONAL_UP + mid/low directness + high contest
    宽通道或阶梯式方向推进

DIRECTIONAL_UP + high activity
    高风险方向环境
```

### 8.2 `DIRECTIONAL_DOWN`

对称规则：

```text
D_midlong < 0
DirectionStrength >= development p70
A_midlong > development p30
```

### 8.3 `BALANCE_LIKE`

建议使用“证据命中制”，而非加权总分。以下四项中至少满足三项：

```text
1. abs(DirectionConsensus) <= development p30
2. E_midlong <= development p30
3. B_equal >= development p70 或 overlap_20 >= development p70
4. HistoricalCoverage_20 >= development p70
```

额外记录：

```text
quiet_balance_like:
    ActivityLevel 低，且近期 X_range 不高

turbulent_balance_like:
    ActivityLevel 高，或近期出现高 X_range 但方向未保留
```

这一步正面检验：`B` 无法区分的安静/剧烈，是否能由 Activity/X 在组合层补足。

### 8.4 `UNRESOLVED`

其余日期全部进入 `UNRESOLVED`。

`UNRESOLVED` 是必要输出，不是失败兜底。它表示现有证据不足、相互冲突或不适合被压缩成当前
体制标签。

红旗：若 `UNRESOLVED` 几乎为零，通常说明规则在强迫市场进入预设状态。

---

## 9. v0a 转换事件候选

转换事件可以与持续体制同时存在，不要求互斥。

### 9.1 `EXPANSION_UP`

```text
X_range >= development p90
abs(X_share) >= development p70
X_close > 0
```

解释：今天发生极端范围冲击，并且较大比例保留为向上收盘位移。

### 9.2 `EXPANSION_DOWN`

```text
X_range >= development p90
abs(X_share) >= development p70
X_close < 0
```

### 9.3 `TURBULENT_SHOCK`

```text
X_range >= development p90
abs(X_share) <= development p30
```

解释：活动范围极大，但最终未形成明确方向收盘结果。

该事件不能自动叫作平衡、反转、吸收或高潮；这些需要上下文或后续结果。

### 9.4 `GAP_SHOCK` 属性

Gap 首版只作为事件属性：

```text
gap_direction
gap_size_atr
```

不得把 Gap 与 `X_range` 相加，因为 Gap 已经进入 TR。

### 9.5 `SCALE_CONFLICT`

满足任一条件：

```text
A_midlong <= development p20
sign(D_5) != sign(D_55)
sign(D_equal) != sign(D_midlong)
```

输出具体冲突类型，而不是只有布尔值。

---

## 10. v0b 上下文事件与滞回

### 10.1 `REBALANCING_CANDIDATE`

只有此前已处于 `DIRECTIONAL_UP/DOWN` 候选背景时才能产生。

向上方向体制中的候选条件可包括：

```text
前一确认体制为 DIRECTIONAL_UP
且出现 SCALE_CONFLICT
且至少满足一项：
    D_5 转负
    短窗 E 明显低于自身近期基线
    出现显著负 X_close
    B_5 或 overlap_5 上升到高分位
```

向下方向对称。

这只是“方向体制受到挑战”的候选，不是完整回调或再平衡确认。

### 10.2 `FAILED_EXPANSION_CANDIDATE`

第一轮若尚未实现 Retention，只允许做非常保守的规则事件，例如：

```text
此前出现 EXPANSION_UP
随后固定 k 日内 X_close 反向，且 PositionBias 明显回落
```

但该对象本质上已经进入规则事件与前向结果研究。建议在 v0a 完成后单独立项，不要与第一版
无记忆读取层同时实现。

### 10.3 滞回比较

至少比较：

```text
state_probe_stateless_v0
state_probe_hysteresis_v0
```

滞回版本可以要求：

- 进入方向体制连续若干日满足证据；
- 退出方向体制使用较宽松门槛；
- 强反向扩张可覆盖普通惯性；
- `confirmed_at` 与 `effective_at` 明确；
- 不回填进入条件首次出现之前的日期。

滞回参数也必须从开发块冻结，不能为消除某几个难看翻转而事后调整。

---

## 11. 最小候选演变图

第一轮可以用以下图作为审计语言，不作为强制状态机：

```text
BALANCE_LIKE
    │
    ├── EXPANSION_UP / DOWN
    │       ↓
    │   EXPANSION_CANDIDATE
    │       ├── 方向共识建立 → DIRECTIONAL_UP / DOWN
    │       └── 冲击消失     → BALANCE_LIKE / UNRESOLVED
    │
    └── TURBULENT_SHOCK
            ├── 后续形成方向 → DIRECTIONAL
            └── 继续混杂     → BALANCE_LIKE / UNRESOLVED
```

方向体制内部：

```text
DIRECTIONAL
    │
    ├── 同方向 EXPANSION → 再加速候选
    ├── SCALE_CONFLICT   → REBALANCING_CANDIDATE
    │       ├── 原方向恢复 → DIRECTIONAL
    │       ├── 方向消失   → BALANCE_LIKE / UNRESOLVED
    │       └── 反向扩张   → 反方向 DIRECTIONAL 候选
    └── 无重要变化       → 保持原候选体制
```

这已经能表达最小 Brooks-inspired 骨架：

```text
平衡倾向
→ 冲击
→ 方向形成或失败
→ 方向持续
→ 尺度冲突与效率下降
→ 恢复、再平衡或反向扩张
```

它仍不识别真实腿、回调深度、交易区间边界或多次推进。

---

## 12. 输出数据契约

建议每日输出至少包含：

```text
date
as_of
probe_version
input_feature_versions
threshold_config_id

regime_candidate
regime_evidence_hits
regime_evidence_total
regime_margin

transition_candidates
transition_evidence

direction_sign
direction_consensus
direction_strength
scale_conflict

activity_attribute
path_directness_attribute
contest_attribute
position_attribute_20
position_attribute_55
gap_attribute

raw_feature_snapshot
status
```

建议示例：

```json
{
  "date": "2024-11-06",
  "as_of": "2024-11-06",
  "probe_version": "phase1a_state_coupling_probe_v0",
  "regime_candidate": "DIRECTIONAL_UP",
  "transition_candidates": ["EXPANSION_UP"],
  "attributes": {
    "activity": "mid",
    "path_directness": "high",
    "contest": "low",
    "position_55": "upper_extreme",
    "gap": "up"
  },
  "evidence": {
    "direction_strength": 0.31,
    "scale_agreement": 0.94,
    "x_range": 2.73,
    "x_close": 2.61,
    "x_share": 0.95,
    "clearance_up_55": 1.0
  }
}
```

`regime_margin` 只能表示规则证据距离阈值有多远，不能命名为 `confidence`，更不能解释成概率。

---

## 13. 实现位置建议

为避免原型污染正式连续测量层，建议：

```text
docs/reports/
  phase1a-state-coupling-probe-operation-guide.md
  phase1a-state-coupling-probe-audit.md

src/market_cycle/
  prototypes/
    __init__.py
    state_coupling_probe.py
    state_coupling_thresholds.py
    state_coupling_outcomes.py

tests/
  test_state_coupling_probe.py
  test_state_coupling_thresholds.py
  test_state_coupling_outcomes.py
```

探针不应放入 `measurements/`，因为它不是新的基础测量；也不应直接放入正式 `states/`，因为尚未
经过选择与锁定确认。

---

## 14. 推荐实施顺序

### 步骤 1：冻结输入

本轮不修改：

```text
E v1
D v1
A
B v1
X v1
clearance v1
```

记录输入版本、快照、样本和字段哈希。

发现底层问题时先写入审计报告，不在同一轮中根据状态结果直接修改公式。

### 步骤 2：实现阈值配置

从开发块计算并持久化：

```text
p20 / p30 / p70 / p90
```

配置必须包括：

```text
threshold_config_id
source_period
source_snapshot
feature_versions
created_at
quantile_method
```

选择块和确认块只读取冻结配置，不重新拟合。

### 步骤 3：实现 v0a 无记忆读取层

完成：

- 派生轴；
- 候选体制；
- 转换事件；
- 属性；
- 证据命中和 margin；
- 每日前缀重放。

### 步骤 4：接入审计页面

图表至少显示：

```text
K 线
E / D / A / B / Activity / X / clearance
候选体制背景
转换事件标记
当日规则命中
原始字段和阈值距离
```

用户点击任一日期时，应能够回答：

```text
为什么今天是这个候选体制？
哪些证据支持？
哪些证据冲突？
距离进入另一候选还差多少？
```

### 步骤 5：运行联合开发审计

先看开发块，不改阈值；重点记录：

- 覆盖过度或不足；
- 标签抖动；
- 同标签内部语义不一致；
- X 是否补足 B 的安静/剧烈盲区；
- clearance 是否只重复 D；
- `UNRESOLVED` 的主要成因；
- 需要结构才能解释的残差。

### 步骤 6：实现消融与模型阶梯

见第 16 节。

### 步骤 7：冻结后运行选择块

只允许比较预注册模型和有限稳健性版本。

### 步骤 8：可选实现 v0b

只有 v0a 暴露出“静态证据合理但标签抖动”时，才增加滞回和上下文事件。

---

## 15. 必须通过的测试

### 15.1 因果与重放

- 批量计算任一日期的输出必须等于仅提供截至该日期数据的独立计算；
- 开发块阈值不得读取选择块或确认块；
- 任何前向结果不得进入当日标签；
- v0b 不得根据未来确认回填状态起点。

### 15.2 确定性

相同：

```text
snapshot
feature versions
threshold config
probe version
```

必须产生完全相同输出。

### 15.3 规则性质

- 每日最多一个 `regime_candidate`；
- 转换事件可以并存；
- 不满足明确状态时必须允许 `UNRESOLVED`；
- `DIRECTIONAL_UP` 与 `DIRECTIONAL_DOWN` 不得同日并存；
- `PositionBias` 必须位于 `[-1,1]`；
- `HistoricalCoverage` 必须位于 `[0,1]`，允许浮点容差；
- Gap 不得进入 `X_range` 的二次加权；
- 零值、缺失和非 `ok` 输入状态必须传播，不得自动填零。

### 15.4 证据可解释性

每个标签必须能够返回：

```text
命中规则
未命中规则
原始值
阈值
margin
输入状态
```

禁止只返回最终颜色或整数编码。

### 15.5 合成场景

至少覆盖：

1. 安静、低方向、低效率、高重叠；
2. 剧烈往返、低净位移；
3. 干净向上方向推进；
4. 干净向下方向推进；
5. 慢牛、正方向但中低效率；
6. 短空长多的尺度冲突；
7. 高 X_range、低 X_share 的冲击；
8. 高 X_range、高正 X_share 的向上扩张；
9. Gap-and-go；
10. Gap-and-fade；
11. 高 clearance、普通小 bar；
12. 高 clearance、强反向冲击；
13. 证据互相冲突，应输出 `UNRESOLVED`。

---

## 16. 模型阶梯与消融

### 16.1 逐级模型

| 模型 | 输入 | 核心问题 |
|---|---|---|
| `M0` | `D + A` | 方向与尺度一致性单独能解释多少 |
| `M1` | `M0 + E` | 路径直接性是否改善体制解释 |
| `M2` | `M1 + B` | 双向争夺是否增加可区分结构 |
| `M3` | `M2 + Activity + X` | 活动背景与当日冲击是否补充转换过程 |
| `M4` | `M3 + clearance` | 历史位置是否有独立价值 |

所有模型使用同一时间块、同一评估协议和相同输出格式。

### 16.2 Leave-one-family-out

从完整模型出发分别删除：

```text
-E
-B
-Activity/X
-clearance
-A
```

检查删除后：

- 标签改变比例；
- 状态持续时间；
- 一日翻转率；
- 历史案例解释；
- 前向路径分离；
- 压缩残差；
- 复杂度下降是否远大于信息损失。

预期可能结论包括：

```text
E 保留为路径属性，不必参与核心体制门槛
B 总分降级，只保留 entropy/overlap 分量
X 只作为转换事件，不进入持续体制
clearance 只作为位置属性
A 是方向读取不可缺少的冲突信息
```

这些都是允许的结果，不预设哪一个必须成立。

---

## 17. 解释效率与验收指标

### 17.1 覆盖率

统计：

```text
BALANCE_LIKE 占比
DIRECTIONAL_UP / DOWN 占比
UNRESOLVED 占比
各转换事件占比
状态与事件的联合占用
```

红旗：

- 单一状态占据绝大多数样本；
- `UNRESOLVED` 接近零；
- EXPANSION 过于频繁，失去事件含义；
- 某状态只在极少数著名历史日出现。

### 17.2 稳定性

统计：

```text
平均和中位持续时间
一日翻转率
UP → DOWN → UP 抖动频率
每年状态切换次数
小幅阈值扰动后的标签变化率
```

持续体制若每天改变，说明它更像噪声分类而非状态。

### 17.3 内部一致性

检查：

- `DIRECTIONAL_UP` 内是否整体正方向且尺度一致；
- `BALANCE_LIKE` 是否大量包含强方向共识；
- 同一体制能否通过属性区分安静与剧烈、干净与宽通道；
- `UNRESOLVED` 是否主要由可解释冲突产生。

### 17.4 压缩损失

知道候选体制与属性后，原始变量仍有多少未表达差异？

如果同一标签内部同时包含：

- 极端安静与极端剧烈；
- 强单边与剧烈双向；
- 高位稳定与低位恐慌；

而属性也无法拆开，则压缩过度。

### 17.5 前向行为分离

未来只能作为 outcome，不能参与状态定义。

预注册期限：

```text
k = 5 / 20 / 55 日
```

可比较：

- 未来累计对数收益；
- 方向调整后的未来位移；
- 最大有利与最大不利移动；
- 未来真实波幅与活动水平；
- 原体制持续时间；
- 是否快速进入相反方向候选；
- 价格是否回到旧历史覆盖区域。

目的不是最大化收益，而是检验不同状态或事件之后，价格路径分布是否存在稳定差异。

### 17.6 历史残差

在已经知道候选体制、事件和属性后，再加入原始 `E/D/B/X/clearance`，是否仍显著提高对未来
路径或历史语义的描述？

若增量很大，说明状态压缩丢失过多，正确结论可能是：

```text
保留连续向量 + 转换事件，不做硬状态压缩
```

---

## 18. 优先联合审计问题

### 18.1 X 是否补足 B 的盲区

在：

```text
低 E + 高 B
```

的样本中，再按：

```text
低 / 高 ActivityLevel
低 / 高 X_range
低 / 高 abs(X_share)
```

分组，检查是否稳定区分：

- 安静停滞；
- 剧烈往返；
- 大冲击但收平；
- 大冲击并形成方向结果。

### 18.2 同样冲击规模下，X_share 是否重要

在相似 `X_range` 下比较：

```text
高正 X_share
高负 X_share
X_share 接近零
```

观察它们的历史形态、后续方向、波动和状态转换是否不同。

### 18.3 clearance 是否有条件增量

匹配相似的：

```text
D / A / E / B / Activity / X
```

再比较高低 clearance。

若 clearance 只重复长期方向，则降级为审计属性或删除；若它能区分旧区覆盖、后续回归和持续
离区，则保留。

### 18.4 B 总分是否掩盖机制

比较：

```text
B composite
entropy components
overlap components
```

检查同一 B 总分下是否存在完全不同的方向混杂与区域重叠组合。

---

## 19. 建议真实窗口

第一轮审计至少覆盖现有报告中的代表案例：

### E / D / A / B

```text
1987-10-19
1989-10-13
1994-03-14
1998-10-14
2008-07-14
2008-10-10
2009-03-09
2020-03-23
2024-07-01
```

### X / clearance

```text
2011-08-05
2016-09-09
2017-05-17
2020-02-24
2024-11-06
2025-03-18
2025-04-07
2025-04-30
2026-05-29
```

这些案例只用于发现语义冲突和解释失败，不能替代连续时间块统计。

---

## 20. 停止条件与失败模式

### 20.1 停止增加复杂度

出现以下任一情况，应停止向探针继续添加规则：

- 新规则只改善少数著名图表；
- 需要不断增加例外条款才能维持标签；
- 标签对阈值微调极端敏感；
- 同一状态内部仍无法通过属性解释主要差异；
- 复杂模型相对 `D+A` 基线无稳定增量；
- 前向行为、压缩或接口均无收益；
- 规则开始隐式使用腿、区间起点、摆动或未来确认。

### 20.2 触发 Phase 2 的明确条件

若两段拥有相似连续向量，却持续表现出不同技术结构意义，例如：

- 普通回调与趋势反转无法区分；
- 假突破与新趋势起点无法区分；
- 趋势中第几次推进产生稳定差异；
- 真实交易区间边界决定后续行为；
- 突破后的接受/失败必须引用事件边界；

则将该残差写成具体 Phase 2 研究问题。

不得因为“状态图不好看”就泛泛引入 ZigZag、摆动树或 Episode 平台。

---

## 21. 实验后允许的四类结论

### 结论 A：两类体制加事件已经足够

```text
BALANCE_LIKE / DIRECTIONAL
+ EXPANSION / REBALANCING 候选事件
```

若稳定、可解释且压缩损失可接受，则准备正式 Phase 1B 协议。

### 结论 B：连续向量有用，但硬状态压缩损失太大

保留：

```text
连续向量 + 转换事件 + 属性
```

不强迫市场只有一个状态标签。

### 结论 C：部分变量无增量

可能动作：

- 删除或降级 clearance；
- `B` 只保留短窗或分量；
- `E` 仅作属性；
- `X` 仅作事件；
- 合并高度冗余的读取轴。

任何删除或升版都应另写正式决策与协议。

### 结论 D：出现明确结构残差

据残差立项 Phase 2：

```text
路径起点
回调深度
区域边界
事件 Retention
多次推进
```

这样 Phase 2 由实证缺口驱动，而非由工具清单驱动。

---

## 22. Agent 交付清单

实施 PR 至少应交付：

```text
1. 探针协议与版本说明
2. 冻结阈值配置及来源区间
3. 无记忆候选体制与事件实现
4. 派生轴及字段状态传播
5. 证据命中与 margin 输出
6. 性质测试和前缀重放测试
7. 审计页面或可复现审计表
8. 模型 M0-M4 与 leave-one-family-out 消融
9. 覆盖、稳定、持续时间和转移统计
10. 前向 outcome 表，严格与当日输入分离
11. 典型案例与失败案例
12. 明确停止条件和 Phase 2 残差清单
```

---

## 23. Agent 执行纪律

后续 Agent 必须遵守：

1. 本文是探索性操作指南，不是已确认市场状态规范；
2. 开始前读取 D-013、工作协议、连续特征审计和 X 设计文档；
3. 本轮冻结 `E/D/B/X/clearance v1`，不边看探针结果边改底层公式；
4. 阈值只能从开发块计算，并在选择块前落盘冻结；
5. 所有输出必须逐日前缀重放一致；
6. 未来 outcome 只用于评价，不得回填当日状态；
7. `regime_margin` 不得称作概率或置信度；
8. `BALANCE_LIKE` 不得描述成真实交易区间；
9. clearance 不得描述成突破；
10. Gap 不得与 TR 重复计分；
11. `X_share` 不得描述成订单流占比；
12. `REBALANCING_CANDIDATE` 必须有此前方向上下文；
13. 允许并鼓励输出 `UNRESOLVED`；
14. 不以彩色状态图是否好看决定保留规则；
15. 不以未来收益最大化选择状态定义；
16. 必须完成模型阶梯与消融，不能只展示完整模型；
17. 若规则开始依赖路径、区域或 Episode，应停止向 Phase 1A 原型塞规则并记录 Phase 2 缺口；
18. 不得将原型接入正式状态 API，除非另有确认决策；
19. 所有新字段、配置和报告必须记录版本、快照、样本、时间角色与状态处理；
20. 实验失败是允许结果，不得通过增加例外规则掩盖失败。

---

## 24. 最终执行摘要

本探针要完成的不是“立刻定义正确的 Market Cycle”，而是回答下面这些更基础、也更有价值的
问题：

```text
哪些变量真正有用？
哪些变量只是重复？
哪些单变量缺陷可以通过组合补足？
状态标签是否比连续向量更有解释效率？
扩张应该是状态还是事件？
clearance 是核心信息还是位置属性？
B 应保留总分还是拆回分量？
哪些无法解释的差异真正需要 Phase 2？
```

推荐研究闭环：

```text
冻结现有传感器
→ 搭建无记忆读取层
→ 检查覆盖、稳定、压缩和解释
→ 做模型阶梯与消融
→ 冻结后检查前向行为
→ 比较无记忆与滞回版本
→ 决定保留连续向量、进入 Phase 1B，或为明确残差立项 Phase 2
```

最好的结果不一定是一张完整、平滑、颜色漂亮的 Market Cycle 图。

更可靠的成果是形成一份可证伪的判断：

```text
现有连续特征能耦合到什么程度；
哪些信息在压缩中丢失；
哪些变量值得保留；
哪些关系必须由事件或结构层表达；
下一阶段应当解决什么具体缺口。
```

这会把项目从“不断生产局部合理的变量”转向“通过纵向闭环验证整个表示是否能够工作”。
