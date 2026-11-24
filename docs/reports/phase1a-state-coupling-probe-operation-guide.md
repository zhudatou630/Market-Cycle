---
title: Phase 1A 状态耦合探针：操作指南与实施协议
status: exploratory-operation-guide-revised
updated: 2026-07-28
owner: phase1a-research
snapshot_id: spx_daily_2026-07-21_d828fbc8
sample_id: spx_ohlc_main_1984
probe_version: phase1a_state_coupling_probe_v0a
decision_record: .pi/grill/2026-07-28-1636——状态耦合探针.md
archived_original: docs/reports/archive/phase1a-state-coupling-probe-operation-guide-2026-07-28-original.md
archived_original_sha256: 9fe96a4da6a6db5234f2be4f952b05ad665da169edaa155a6c2f24dfb95fa275
input_feature_versions:
  - bm_e_01_ohlc_min_v1
  - bm_d_01_theilsen_atr_v1
  - bm_b_01_entropy_overlap_v1
  - bm_x_01_ohlc_impulse_v1
  - bm_x_dep_01_prior_range_clearance_v1
---

# Phase 1A 状态耦合探针：操作指南与实施协议

## 1. 文档定位

本文是 Phase 1A 现有连续指标第一次组合读取的**现行操作指南**。原始外部方案已原样
归档至
[phase1a-state-coupling-probe-operation-guide-2026-07-28-original.md](archive/phase1a-state-coupling-probe-operation-guide-2026-07-28-original.md)。
归档版本只用于溯源，不再作为执行说明。

本次修订依据用户确认的
`.pi/grill/2026-07-28-1636——状态耦合探针.md`。若本文与归档版本冲突，以本文为准；
若与代码、测试或 `docs/decisions.md` 冲突，仍按项目权威顺序处理。

本探针只回答：

> 现有 `E / D / A / entropy / overlap / Activity / X / clearance` 放在一起后，
> 能否形成一个因果、透明、可逐日复查的初步市场描述？

它不回答：

- 哪个状态之后一定上涨或下跌；
- 当前是否处于真实交易区间、趋势腿或回调；
- 扩张后来是否成功；
- 最终 Brooks Market Cycle 应如何定义；
- 当前原型能否进入正式市场状态 API。

本文不是 canonical decision，不新增 `D-*`。`v0a` 可以失败，也可以得出“连续指标加事件
比硬状态更合适”的结论。

---

## 2. 本轮范围

### 2.1 只做无记忆 `v0a`

每天只使用当日收盘时已经可得的连续指标，以及截至前一交易日的历史分布，输出：

- 一个背景候选；
- 零个或多个当天事件；
- 活动、直接性、争夺、位置、Gap 和尺度冲突等属性；
- 每条规则的原始值、当日阈值、是否命中和阈值距离。

第一轮明确不做：

- 根据前一日标签维持状态；
- 连续若干日才确认进入或退出；
- 状态年龄、Episode 起点或事后回填；
- `REBALANCING_CANDIDATE`；
- `FAILED_EXPANSION_CANDIDATE`；
- 未来 5 / 20 / 55 日收益或路径结果；
- 任何 Retention、接受或失败判断。

状态跳动在 `v0a` 中只展示和统计，不用平滑规则修饰。

### 2.2 冻结底层指标

本轮冻结：

```text
E v1
D v1
A
B v1 的 entropy / overlap 分量及综合输出
X v1
clearance v1
```

若组合结果暴露普通效果问题，只写入审计，不在同一轮修改公式。若发现明确实现错误、未来
信息、错误状态传播或公式与协议不一致，应停止本轮，修复并升版后从头重跑。

---

## 3. 输出分层

### 3.1 背景候选

每天最多一个：

```text
DIRECTIONAL_UP
DIRECTIONAL_DOWN
BALANCE_LIKE
UNRESOLVED
```

人话含义：

| 候选 | 含义 | 不代表 |
|---|---|---|
| `DIRECTIONAL_UP` | 中长期价格中心明显向上，尺度没有严重冲突 | 趋势已确认、未来继续上涨 |
| `DIRECTIONAL_DOWN` | 中长期价格中心明显向下，尺度没有严重冲突 | 趋势已确认、未来继续下跌 |
| `BALANCE_LIKE` | 多项证据呈现偏盘整特征 | 已识别真实交易区间 |
| `UNRESOLVED` | 证据不足、处于中间地带或彼此冲突 | 计算失败或必须消灭的残差 |

允许大量 `UNRESOLVED`。若它几乎为零，应优先怀疑规则在强迫分类。

### 3.2 当天事件

事件与背景可以同时存在：

```text
EXPANSION_UP
EXPANSION_DOWN
TURBULENT_SHOCK
```

例如某天可以同时是：

```text
背景：DIRECTIONAL_UP
事件：EXPANSION_UP
```

Gap 只作为事件属性，不单独提高事件强度。尺度冲突只作为诊断属性，不称为转换事件；无记忆
版本不知道冲突是否当天刚发生。

### 3.3 属性

至少保留：

```text
path_directness
entropy_contest
overlap_contest
activity_level
position_20
position_55
gap_direction
gap_size_atr
scale_conflict
scale_conflict_type
```

不拆分 `QUIET_BALANCE_LIKE` 与 `TURBULENT_BALANCE_LIKE`。安静或剧烈是属性；今天突然
发生的大动作是事件。这样不会不断膨胀背景状态数量。

---

## 4. 输入与派生读取量

### 4.1 输入职责

| 输入 | 本轮用途 | 边界 |
|---|---|---|
| `D_midlong` | 决定背景方向与迁移力度 | 不是当前腿 |
| `A_midlong` | 判断各尺度方向是否严重冲突 | 不是正确概率 |
| `E_midlong` | 描述路径直接性；参与偏盘整证据 | 不是趋势强度 |
| entropy / overlap | 分别描述涨跌混杂和 K 线覆盖 | 不用 `B` 综合分作状态门槛 |
| Activity | 描述今天发生前市场已有多活跃 | 不是今天的新冲击 |
| `X_range` | 描述今天相对昨日冻结尺动得多猛 | 没有方向 |
| `X_close` | 描述今天留下多少带方向收盘位移 | 不表示未来接受 |
| `X_share` | 描述当日真实波幅有多少留成收盘净位移 | 不是订单流占比 |
| clearance | 描述当前收盘相对旧日线分布的位置 | 不是突破 |

原始字段和版本必须保留；派生读取量不能替代底层测量记录。

### 4.2 历史覆盖与位置

对 $n\in\{20,55\}$：

$$
PositionBias_{t,n}
=
Clearance^{up}_{t,n}-Clearance^{down}_{t,n}
$$

$$
HistoricalCoverage_{t,n}
=
1-Clearance^{up}_{t,n}-Clearance^{down}_{t,n}
$$

`HistoricalCoverage` 表示过去有多少根完整日线仍覆盖当前收盘。它是 probe-derived 字段，
必须记录公式版本。20 日覆盖参与偏盘整判断；55 日位置只作较长期属性。

### 4.3 尺度冲突

诊断满足任一条件：

```text
A_midlong <= 当日历史 p20
sign(D_5) != sign(D_55)
sign(D_equal) != sign(D_midlong)
```

输出具体冲突类型，不只保存一个布尔值。尺度冲突可以是回撤、反弹或转换过程，但 `v0a`
不进一步命名。

---

## 5. 因果阈值政策

### 5.1 每天只看过去

对日期 $t$ 的任一分位阈值：

$$
q_{p,t}
=
Quantile_p\left(x_s:s<t\right)
$$

不得包含当天值，更不得包含未来日期。禁止用完整开发块算一套分位数再回填开发块早期标签。

阈值计算至少满足：

```text
history_end = t 的前一交易日
minimum_history = 5 calendar years
primary_history = 从该字段首个 ok 值至 history_end 的全部历史
```

不足五年时，探针输出 `insufficient_threshold_history`，不产生背景或事件候选。正式历史审计
从 1990 年首个交易日开始，保证已有超过五年候选特征历史。

### 5.2 第一轮分位脚手架

```text
很低：<= p20
低：  <= p30
高：  >= p70
极端：>= p90
```

这些只是启动线，不是市场天然标准。量化实现必须记录：

```text
threshold_method_id
history_start
history_end
observation_count
quantile_method
p20 / p30 / p70 / p90
input_feature_version
```

### 5.3 稳定性对照

主版本完成后再做两类对照：

1. 把 p30 / p70 小幅改为 p25 / p75、p35 / p65，观察标签是否大面积改变；
2. 使用截至昨天的过去五年滚动分位，观察全历史分位是否被老环境明显拖住。

五年滚动版本不是主规则，不得边看结果边替换主版本。若两者差异很大，应把“高低标准如何
适应市场变化”列为单独问题，而不是静默选图形更好看的版本。

---

## 6. `v0a` 背景规则

### 6.1 判断顺序

严格按以下顺序：

```text
1. DIRECTIONAL_UP / DIRECTIONAL_DOWN
2. BALANCE_LIKE
3. UNRESOLVED
```

方向优先，因为已经确认 `D + A` 承担背景方向主判。路径曲折、争夺高或仍被旧日线覆盖，
应作为宽通道或阶梯推进的属性，不应抹掉明确且一致的方向。

### 6.2 `DIRECTIONAL_UP`

```text
D_midlong > 0
abs(D_midlong) >= 截至昨天历史 p70
A_midlong > 截至昨天历史 p30
```

### 6.3 `DIRECTIONAL_DOWN`

```text
D_midlong < 0
abs(D_midlong) >= 截至昨天历史 p70
A_midlong > 截至昨天历史 p30
```

不使用 `|D|×A` 作为主门槛。可将乘积保留为审计字段，但不得让 `A` 通过乘积和额外门槛
被重复使用而难以解释。

### 6.4 `BALANCE_LIKE`

不满足方向候选后，再检查以下四项；至少命中三项：

```text
1. 方向整体弱：
   abs(D_midlong) <= 截至昨天历史 p30
   且 A_midlong > 截至昨天历史 p30

2. 路径不直接：
   E_midlong <= 截至昨天历史 p30

3. 双向争夺高：
   entropy_20 >= 截至昨天历史 p70
   或 overlap_20 >= 截至昨天历史 p70

4. 仍被近期旧日线较多覆盖：
   HistoricalCoverage_20 >= 截至昨天历史 p70
```

第 3 项无论两个分量命中几个，在四项制中都只算一项，同时保存具体命中来源。`B_equal`、
`B_midlong` 和各尺度综合仍可展示，但不参与状态门槛。

`A` 很低时，综合方向接近零可能只是短期和长期互相抵消，不得把它计为“方向整体弱”。
若另外三项全部命中，仍可得到 `BALANCE_LIKE + scale_conflict`；证据明细必须让这种组合
可见。

### 6.5 `UNRESOLVED`

其余所有有效日期进入 `UNRESOLVED`。必须区分：

```text
UNRESOLVED             输入有效，但证据不足或冲突
not_ready / non_ok     阈值或底层输入尚不可用
```

不能把缺失、零尺度或历史不足伪装成 `UNRESOLVED`。

---

## 7. `v0a` 当天事件规则

### 7.1 向上扩张

```text
X_range >= 截至昨天历史 p90
abs(X_share) >= 截至昨天历史 p70
X_close > 0
```

输出 `EXPANSION_UP`。

### 7.2 向下扩张

```text
X_range >= 截至昨天历史 p90
abs(X_share) >= 截至昨天历史 p70
X_close < 0
```

输出 `EXPANSION_DOWN`。

### 7.3 剧烈但无明确方向结果

```text
X_range >= 截至昨天历史 p90
abs(X_share) <= 截至昨天历史 p30
```

输出 `TURBULENT_SHOCK`。

处于 p30 和 p70 中间时不强行贴事件，只保留连续值。`X_share` 只描述当日收盘保留，不是
未来 Retention。Gap 已进入 TR，只作为方向和大小属性，不能与 `X_range` 再次加权。

---

## 8. 属性分级

属性分级也使用截至昨天的因果分位线：

```text
low   <= p30
mid   p30 至 p70
high  >= p70
```

第一版建议：

| 属性 | 来源 |
|---|---|
| 路径直接性 | `E_midlong` |
| 涨跌混杂 | `entropy_20` |
| 区域重叠 | `overlap_20` |
| 活动水平 | `ActivityLevel` |
| 20 / 55 日位置 | 连续 `PositionBias` 与 `HistoricalCoverage`，不强制压成单一状态 |
| Gap | 原始方向与 ATR 距离 |

不要把“安静/剧烈”和“偏盘整/偏方向”交叉扩展成大量背景状态。

---

## 9. 日度结果契约

结果表必须是一日一行，并至少包含：

```text
date
as_of
probe_version
input_feature_versions
threshold_method_id
threshold_history_start
threshold_history_end
threshold_observation_count

regime_candidate
regime_rule_order
regime_evidence_hits
regime_evidence_total
event_candidates

direction_sign
scale_conflict
scale_conflict_type
path_directness_attribute
entropy_contest_attribute
overlap_contest_attribute
activity_attribute
position_bias_20
historical_coverage_20
position_bias_55
historical_coverage_55
gap_direction
gap_size_atr

每条规则的 raw_value
每条规则的 threshold
每条规则的 hit
每条规则的 margin

raw_feature_snapshot
status
```

`margin` 只表示原始值离当日阈值多远。不得称为概率、置信度或标签正确率。动态阈值必须
逐日落在结果中，不能只保存一个全局配置 id。

推荐把规则证据做成结构化列或结构化对象，不用拼接字符串作为唯一事实来源。人话解释由
同一份结构化证据生成，避免页面文案和实际规则漂移。

---

## 10. 逐步增加信息

第一轮仍要观察每增加一类指标后多解释了什么，但不做“模型谁更好”的排名：

| 步骤 | 新增信息 | 要观察的变化 |
|---|---|---|
| 1 | `D + A` | 方向和尺度冲突能描述到什么程度 |
| 2 | `+ E` | 是否补充了走得直接还是曲折 |
| 3 | `+ entropy + overlap` | 是否补充了涨跌混杂和区域覆盖 |
| 4 | `+ Activity + X` | 是否拆开安静环境、剧烈环境和当天冲击 |
| 5 | `+ clearance` | 是否补充了相对旧日线分布的位置 |

页面应允许逐步显示这些信息。统计中可以记录加入某类信息后，多少日期获得了新的解释或
改变了背景候选，但不得把标签变化比例直接解释成该指标“有效”。

也可从完整读数中临时隐藏某一类信息，检查失去了什么解释；结论限于当前描述职责，不延伸
到未来预测。

---

## 11. 历史检查顺序

### 11.1 开发段

```text
1990-01-01 至 2004-12-31
```

允许：

- 修正探针代码错误；
- 检查规则覆盖、语义冲突和阈值敏感性；
- 对系统性问题修改探针规则并升 probe 版本；
- 完成页面与日度表。

禁止为了少数著名日期增加例外规则。

### 11.2 冻结检查段

```text
2005-01-01 至 2014-12-31
```

进入前冻结规则顺序、分位政策、字段、状态处理和报告格式。这里冻结的是**计算方法**，不是
每天的阈值数值；每天的阈值仍按截至昨天的历史因果更新。

只观察和记录，不根据个别案例回头修改同一版本。

### 11.3 最后确认段

```text
2015-01-01 至 2026-07-21
```

项目过去已经看过全样本分布和多个著名窗口，因此该段不是纯盲测。它的作用是检查冻结规则
在不同市场年代是否仍能形成可解释描述。

若后段暴露系统性问题，应得出当前版本失败或需要新版本，而不是静默调整原版本。

---

## 12. 验收标准

第一轮只评价当前描述，不评价未来收益。

### 12.1 能解释

点击任一日期，应能回答：

```text
今天是什么背景候选？
为什么？
哪些条件命中？
哪些没有命中？
实际值和当日阈值分别是多少？
是否存在尺度冲突或其他相反证据？
```

### 12.2 不强迫分类

统计四类背景占比和事件占比。红旗包括：

- `UNRESOLVED` 几乎为零；
- 单一背景长期占据绝大多数日期；
- 扩张事件过于频繁，失去“极端当天冲击”的含义；
- 缺失输入被自动填零后进入正常标签。

### 12.3 暴露跳动

统计：

```text
平均和中位连续天数
一日标签占比
UP → DOWN → UP 的快速翻转
每年背景切换次数
临界阈值附近的切换比例
```

这些统计只诊断 `v0a`，不触发自动平滑。

### 12.4 压缩不过度失真

检查同一背景内部的主要差异能否由属性说明。例如：

- 偏方向但路径直接性不同；
- 偏方向但争夺程度不同；
- 偏盘整但活动水平不同；
- 相同背景但旧区位置不同。

若属性仍无法拆开明显不同的市场过程，允许结论为：

```text
保留连续向量 + 当天事件，不保留硬背景标签
```

### 12.5 阈值不应极端敏感

小幅移动分位线后，若大量日期改变标签，应记录当前规则不稳定。不得为降低变化率而增加例外
条款。

---

## 13. 组合观察页面

现有连续指标页面保留，新增独立的“组合观察”模式。它不是简单增加一行最终标签。

### 13.1 价格与状态层

至少显示：

```text
SPX K 线
背景候选时间带
扩张事件标记
背景发生变化的日期
```

时间带展示 `v0a` 的原始逐日结果，不允许为了视觉平滑合并或回填。

### 13.2 证据变化层

与 K 线同步显示：

```text
方向力度及其当日阈值
尺度一致性及其当日阈值
E 及其当日阈值
entropy / overlap 及其当日阈值
20 日历史覆盖及其当日阈值
Activity 与 X
```

用户应能看出一次标签变化由哪个证据跨线推动，而不只看到颜色改变。

### 13.3 逐日解释层

点击某天后，以人话优先显示：

```text
最终背景和当天事件
支持原因
相反或冲突证据
未命中的条件
原始值、阈值、margin 和输入状态
```

### 13.4 历史筛选

至少支持：

- 只看某一背景；
- 只看某一事件；
- 跳到上一个或下一个同类日期；
- 只看发生背景变化的日期；
- 切换逐步增加信息的观察步骤。

页面具体布局、颜色和交互在实现设计中细化，但不得删减“时间过程、证据变化、逐日解释、
同类筛选”这四类能力。

---

## 14. 必须通过的测试

### 14.1 因果阈值与逐日前缀重放

对任一日期 $t$：

```text
全样本批量计算在 t 的结果
=
只提供截至 t 数据独立计算的最后一行结果
```

同时验证：

- $t$ 的阈值最多读取到 $t-1$；
- 不足五年历史时不输出正常候选；
- 扩展未来数据不会修改既有日期的阈值、标签和证据；
- 三段时间检查不改变每日阈值的因果口径。

### 14.2 确定性

相同的：

```text
snapshot
feature versions
threshold method
probe version
```

必须产生完全相同结果。

### 14.3 规则性质

- 每天最多一个背景候选；
- 方向候选优先于偏盘整；
- `DIRECTIONAL_UP` 与 `DIRECTIONAL_DOWN` 不得同日并存；
- 当天事件可以与背景并存；
- entropy 与 overlap 同时命中也只算一项盘整证据；
- 尺度冲突导致的综合接近零不得命中“方向整体弱”；
- `PositionBias` 位于 `[-1,1]`；
- `HistoricalCoverage` 位于 `[0,1]`，允许浮点容差；
- Gap 不得重复进入 `X_range` 强度；
- 缺失、零尺度和非 `ok` 状态不得自动填零；
- `UNRESOLVED` 与输入不可用必须区分。

### 14.4 合成场景

至少覆盖：

1. 干净向上方向推进；
2. 干净向下方向推进；
3. 慢牛，方向明确但效率中低；
4. 宽通道，方向明确但争夺较高；
5. 安静、低方向、低效率、高重叠；
6. 剧烈往返、低净位移；
7. 短空长多或短多长空，综合方向接近零但不能算方向弱；
8. 极端 `X_range`、低 `X_share`；
9. 极端 `X_range`、高正 `X_share`；
10. 极端 `X_range`、高负 `X_share`；
11. Gap-and-go 与 Gap-and-fade；
12. 高 clearance、普通小 bar；
13. 证据不足或冲突，输出 `UNRESOLVED`；
14. 五年历史不足；
15. 当天极端值不得进入自己的分位阈值。

---

## 15. 实现边界

建议代码仍放在原型层：

```text
src/market_cycle/prototypes/
  __init__.py
  state_coupling_thresholds.py
  state_coupling_probe.py

tests/
  test_state_coupling_thresholds.py
  test_state_coupling_probe.py
```

不创建 `state_coupling_outcomes.py`，因为第一轮不研究未来结果。探针不进入
`measurements/`，也不进入正式 `states/`。

结果表的具体落盘路径在实现时按项目现有研究产物惯例确定，但必须记录快照、样本、版本、
阈值历史区间和状态处理。不能只依赖网页运行时临时计算而没有可复现日表。

---

## 16. 实施顺序

1. 固定本文、输入版本和开发段规则；
2. 实现截至昨天、至少五年历史的逐日分位阈值；
3. 实现派生位置、尺度冲突、背景候选、事件和属性；
4. 输出日度结果表和完整证据；
5. 完成因果重放、性质与合成测试；
6. 在现有审计页增加组合观察模式；
7. 只运行 1990-2004 开发段，记录系统性问题；
8. 冻结 probe 版本后运行 2005-2014；
9. 不修改同一版本，再运行 2015-2026；
10. 形成审计报告，决定保留硬标签、只保留连续向量加事件，或另立后续问题。

不得提前实现 `v0b` 或未来 outcome。

---

## 17. 停止条件

出现以下任一情况，应停止增加规则：

- 新规则只改善少数著名日期；
- 需要不断增加例外才能维持标签；
- 小幅移动分位线就导致大面积标签变化；
- 同一标签内部的主要差异连属性也解释不了；
- 规则开始依赖腿、区域起点、摆动、Episode 或未来确认；
- 页面需要人为平滑才能显得合理；
- 当前描述并不比直接阅读连续向量更清楚。

若问题需要真实区间、回调深度、事件边界或多次推进，应把具体残差写成后续结构研究问题，
不能继续向 `v0a` 塞规则。

---

## 18. 允许的实验结论

### 结论 A：两类背景加事件有用

```text
偏方向 / 偏盘整 / 看不清
+ 当天扩张事件
+ 解释属性
```

若它们在三段历史中可解释且不过度失真，再单独讨论下一版。

### 结论 B：连续向量有用，硬背景压缩无益

保留连续指标、属性和当天事件，不要求每天只有一个背景标签。

### 结论 C：部分指标在组合中没有清楚职责

记录其没有补充什么，再另行决定降级、删除或升版。不得在本轮边看边改。

### 结论 D：缺口需要结构信息

把普通回调与反转、真实区域边界、突破接受、多次推进等残差转成明确的后续研究问题。

上述四种都是合法结果。实验失败不允许用例外规则掩盖。

---

## 19. 最小交付清单

```text
1. 因果分位阈值实现及逐日阈值记录
2. v0a 背景、事件和属性实现
3. 一日一行的结果表与完整证据
4. 前缀重放、规则性质和合成测试
5. 组合观察页面
6. 覆盖、连续天数、切换和阈值敏感性统计
7. 逐步增加信息的观察记录
8. 三段历史按顺序形成的审计报告
9. 失败案例与停止条件判断
10. 下一步只保留已由结果支持的问题
```

不交付未来收益表、未来路径比较、`v0b`、正式状态 API 或交易信号。

---

## 20. 执行纪律

后续 Agent 必须遵守：

1. 先读 D-013、Phase 1A 工作协议、连续特征审计、本指南和 grill 决策记录；
2. 归档原指南只溯源，不作为当前执行合同；
3. 每日阈值只能读到前一交易日，不能用完整开发块回填历史；
4. 五年历史不足时必须明确不可用；
5. 第一轮只做 `v0a`；
6. 背景与当天事件分开；
7. 方向由 `D` 和 `A` 分开主判；
8. `B` 综合分不作状态门槛；
9. 尺度冲突是诊断属性，不是转换事件；
10. `BALANCE_LIKE` 不得称为真实交易区间；
11. expansion 不得称为成功突破或未来接受；
12. Gap 不得与 TR 重复计分；
13. `X_share` 不得称为订单流或未来 Retention；
14. 允许并保留 `UNRESOLVED`；
15. 状态跳动只诊断，不在 `v0a` 平滑；
16. 第一轮不计算或展示未来收益和路径；
17. 不用彩色图是否平滑决定规则；
18. 不边看组合结果边修改底层 v1 指标；
19. 不把原型接入正式状态 API；
20. 所有输出必须逐日前缀重放一致。

---

## 21. 一句话执行摘要

```text
冻结现有指标
→ 每天只用截至昨天的历史划分高低
→ 先读方向，再读偏盘整，否则诚实地看不清
→ 把当天扩张和背景分开
→ 用日度表与组合观察页面复查整个历史过程
→ 只评价当前描述，不偷看未来收益
→ 再决定硬状态是否值得保留
```