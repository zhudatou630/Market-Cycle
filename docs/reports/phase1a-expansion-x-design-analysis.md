---
title: Phase 1A 扩张 X：第一性原理设计分析与 Agent 实施说明
status: external-design-proposal
updated: 2026-07-25
owner: phase1a-research
snapshot_id: spx_daily_2026-07-21_d828fbc8
sample_id: spx_ohlc_main_1984
---

# Phase 1A 扩张 X：第一性原理设计分析与 Agent 实施说明

## 1. 文档定位

本文是对 [Phase 1A 扩张 X 外部设计评审备忘](phase1a-expansion-x-external-review.md)
的完整外部分析，目的是把“扩张”从技术分析叙述拆成可计算、可审计、可反驳的对象，
并给后续 Agent 提供清晰的设计与实施边界。

本文不是正式研究协议，也不是已确认决策。除非用户另行确认并写入
[decisions.md](../decisions.md)，本文中的版本名、字段、窗口、结果期限与实施顺序均是
设计建议，不得自动升级为项目事实。

本文服从当前架构边界：Phase 1A 只研究固定窗口或明确滞后尺子上的连续行为量，不使用
当前腿、摆动起点、Episode、交易区间识别或事后结构边界。若需要真正的区间、形态或摆动
突破，必须进入 Phase 2 的独立结构协议。该边界见 D-013 与
[Phase 1A 工作协议](../phase1a-working-protocol.md)。

一句话结论是：

> 扩张 X 不应是一个把 TR、Gap、Escape 和 Retention 加权相加的总分。应把它拆成
> “活动背景、当日冲击、相对位置、规则事件、未来接受结果”五类对象；Phase 1A 的核心
> X_now 只负责记录当日相对既有活动尺度出现了多大的新冲击，以及该冲击最终形成了多少
> 方向位移。

---

## 2. 从第一性原理重新定义“扩张”

### 2.1 扩张必须相对于某个基准

“今天波动很大”没有绝对意义。同样是 2% 的日内波动：

- 对平时只波动 0.5% 的市场，它是正常水平的四倍；
- 对平时每天波动 3% 的市场，它反而比正常水平更安静。

所以扩张的第一条原则是：

```text
扩张 = 今天已经发生的价格运动 / 今天之前已经知道的正常活动尺度
```

没有基准，就只有绝对涨跌，没有“扩张”。

### 2.2 必须把动作、结果、位置和后续分开

一根日线至少包含四种不同信息：

1. **动作有多大**：今天覆盖了多大的真实波幅；
2. **结果向哪里**：今天最终把收盘推离昨日收盘多远；
3. **相对位置在哪里**：当前价格是否离开此前大部分活动范围或越过某个明确边界；
4. **后来是否站住**：未来若干日是否继续停留在边界外。

这四种信息经常同时出现，但逻辑上并不等价：

- 大波动可以最终收平；
- 小波动也可以缓慢创出 20 日新高；
- 当日强突破可以几天后完全失败；
- 趋势中可以持续创新高，却没有新的波动加速。

因此不能用一个数字同时回答四个问题。

### 2.3 最小对象分层

建议使用下面的对象分层：

| 对象 | 要回答的问题 | 可得时间 | 是否属于 Phase 1A 日度 X_now |
|---|---|---|---|
| `activity_level` | 市场原本有多活跃 | `t-1` 已知 | 否，属于上下文尺子 |
| `X_now` | 今天出现了多大的新冲击，形成了多少方向位移 | `t` 收盘后 | 是 |
| `reference_departure` | 当前价格相对旧活动分布处在哪里 | `t` 收盘后 | 可作并列连续候选 |
| `channel_escape` | 是否越过预设通道边界 | `t` 收盘后 | 独立规则事件 |
| `acceptance_outcome` | 越界后是否继续留在外面 | `t+k` 后 | 否，独立前向结果 |
| `structure_escape` | 是否突破真实区间、摆动或形态边界 | 取决于结构协议 | Phase 2 |

可以把整个流程想象成：

```text
昨天冻结的活动尺子
        ↓
今天 OHLC ──→ 当日冲击 X_now
        ↓
相对旧价格分布的位置
        ↓
是否触发某个预设边界事件
        ↓
未来价格 ──→ 接受 / 回归 / 失败结果
```

---

## 3. 时间与因果契约

### 3.1 使用昨日冻结的尺度

当前共享 ruler 提供：

```text
date, open, high, low, close, tr_pct, atr_pct_14
```

其中：

\[
tr\_pct_t = \frac{TR_t}{C_t}
\]

\[
atr\_pct_{14,t} = \frac{ATR_{14,t}}{C_t}
\]

所以可从共享字段恢复价格点数尺度：

\[
TR_t = C_t\cdot tr\_pct_t
\]

\[
A_{t-1}=C_{t-1}\cdot atr\_pct_{14,t-1}
\]

这里的 \(A_{t-1}\) 是今天开始之前已经冻结的正常活动尺度。

所有 X_now 主分量应使用 \(A_{t-1}\)，而不是 \(A_t\)。原因是今天的大波动会进入
今天的 ATR；若用 \(A_t\) 作分母，冲击会同时把自己的尺子拉长，从而自我稀释。

### 3.2 正确的参考类型

X_now 不是标准的“最近 n 日固定窗口总量”，而是“今日观察相对昨日冻结尺子”的日度
冲击。因此建议显式声明：

```text
reference_type    daily_impulse_with_lagged_ruler
observation_at    t
scale_at          t-1
as_of             t close
feature_version   公式与实现版本
```

这仍然完全因果，也不引入路径或状态机，只是比统一写成 `fixed_calendar_window` 更准确。

### 3.3 Retention 的时间角色必须单独保存

若边界事件发生在 \(t\)，前向结果使用未来 \(k\) 日，则：

```text
boundary_at    t-1 或更早：事件边界已经确定
event_at       t：当日收盘触发事件
confirmed_at   t+k：前向结果首次完整可知
```

未来结果不得回填成 \(t\) 日的 X_now，也不得根据未来成功或失败修改当时的边界和事件资格。

---

## 4. 活动背景：市场原本有多活跃

建议把昨日相对 ATR 作为显式上下文：

\[
\boxed{
ActivityLevel_t = atr\_pct_{14,t-1}
}
\]

它回答的是：

> 在今天发生之前，市场日常活动尺度大约占价格的多少？

它不属于 X_now，因为它描述的是当前波动制度的水平，而不是今天相对旧制度出现的新冲击。

这一分层可以直接修复当前 E/D/B 审计暴露出的一个语义缺口：安静低效率和剧烈低效率都
可能具有低 E、高 B，但它们的 ATR 水平完全不同。`ActivityLevel` 负责“当前有多剧烈”，
X_now 负责“今天是否比原来更剧烈”。

后续若确实需要研究活动制度是否抬升，可另设 challenger：

\[
ActivityShift_{t,n}
=
\frac{
atr\_pct_{14,t-1}
}{
\operatorname{median}(atr\_pct_{14,t-n:t-2})
}
\]

但它不建议进入 X_now v1，以免把“当前水平”和“今日冲击”重新混合。

---

## 5. X_now 的最小独立分量

建议的最小独立向量是：

\[
\boxed{
\mathbf X^{now}_t
=
\left(
X^{range}_t,
X^{close}_t,
X^{gap}_t
\right)
}
\]

另保存一个由前两项推导的解释比率 \(X^{share}_t\)。它便于阅读，但不是新的独立证据，
不应在未来综合时再次计权。

### 5.1 当日波幅冲击 `X_range`

经典真实波幅为：

\[
TR_t
=
\max\left(
H_t-L_t,
|H_t-C_{t-1}|,
|L_t-C_{t-1}|
\right)
\]

定义：

\[
\boxed{
X^{range}_t
=
\frac{TR_t}{A_{t-1}}
}
\]

它回答：

> 今天覆盖的真实价格范围，是昨日正常日波幅的多少倍？

直观解释：

| `X_range` | 含义 |
|---:|---|
| `0.5` | 今天只有正常波幅的一半 |
| `1.0` | 今天大致是正常波幅 |
| `2.0` | 今天覆盖了约两倍正常波幅 |
| `4.0` | 极强的日度波动冲击 |

这是 X_now 最核心的量，但它没有方向。大涨、大跌和剧烈往返都可以得到高 `X_range`。

### 5.2 收盘方向冲击 `X_close`

定义：

\[
\boxed{
X^{close}_t
=
\frac{C_t-C_{t-1}}{A_{t-1}}
}
\]

它回答：

> 今天最终把收盘价向上或向下推动了多少个昨日 ATR？

例如：

- `+1.5`：收盘比昨日高约 1.5 个昨日 ATR；
- `-2.0`：收盘比昨日低约 2 个昨日 ATR；
- 接近 `0`：即使盘中很剧烈，最终也没有形成明显净位移。

`X_range` 是“今天用了多大的动作”，`X_close` 是“最终留下了多少方向结果”。

### 5.3 方向转化率 `X_share`

在 \(TR_t>0\) 时定义：

\[
\boxed{
X^{share}_t
=
\frac{X^{close}_t}{X^{range}_t}
=
\frac{C_t-C_{t-1}}{TR_t}
}
\]

由 OHLC 几何可知：

\[
-1\le X^{share}_t\le 1
\]

它回答：

> 今日真实波幅中，有多少最终转化成了一个方向上的收盘净位移？

解释：

| `X_share` | 含义 |
|---:|---|
| 接近 `+1` | 大部分真实波幅转化为向上收盘位移 |
| 接近 `-1` | 大部分真实波幅转化为向下收盘位移 |
| 接近 `0` | 波动很大，但最终大部分被双向运动抵消 |

注意：它不是订单流效率，也不知道日内真实路径。它只是由日线 OHLC 可见事实构成的
“收盘位移 / 真实波幅”。

### 5.4 开盘越过前日完整区间的 Gap `X_gap`

不建议把“开盘高于昨日收盘”一律称为 Gap。首版只记录开盘是否越过昨日完整高低区间：

\[
G^{up}_t
=
\frac{\max(0,O_t-H_{t-1})}{A_{t-1}}
\]

\[
G^{down}_t
=
\frac{\max(0,L_{t-1}-O_t)}{A_{t-1}}
\]

带方向版本：

\[
\boxed{
X^{gap}_t
=
G^{up}_t-G^{down}_t
}
\]

它回答：

> 今天开盘时，市场是否已经出现在昨日完整价格范围之外？若是，距离是多少个昨日 ATR？

解释：

- `+1.0`：开盘在昨日最高价之上约 1 个 ATR；
- `-0.5`：开盘在昨日最低价之下约半个 ATR；
- `0`：开盘仍位于昨日完整范围内。

Gap 已经进入 TR，所以 `X_gap` 不应再与 `X_range` 数学相加。它的作用是说明冲击来源：
冲击是在开盘前发生，还是在盘中形成。

### 5.5 可选的隔夜 / 盘中分解

为区分 gap-and-go 与 gap-and-fade，可保存两个解释字段：

\[
X^{overnight}_t
=
\frac{O_t-C_{t-1}}{A_{t-1}}
\]

\[
X^{intraday}_t
=
\frac{C_t-O_t}{A_{t-1}}
\]

两者满足：

\[
X^{close}_t
=
X^{overnight}_t+X^{intraday}_t
\]

联合阅读：

- `overnight > 0` 且 `intraday > 0`：向上 gap-and-go；
- `overnight > 0` 且 `intraday < 0`：向上 gap-and-fade；
- `overnight < 0` 且 `intraday < 0`：向下 gap-and-go；
- `overnight < 0` 且 `intraday > 0`：向下跳空后回补。

这两个字段建议作为诊断，不是 X_now v1 的必要独立分量。

---

## 6. 为什么不使用滚动 z-score 作为主定义

滚动 z-score 可以描述统计异常，但不建议作为 X_range 主版本：

1. TR 分布重尾，均值和标准差容易被极端日影响；
2. 若窗口包含当日，极端日会同时放大自己的均值和标准差；
3. z-score 的技术分析解释不如“几倍昨日 ATR”直接；
4. 不同波动制度中，窗口长度和极端值清理会显著影响结果；
5. 它容易诱导后续按未来结果挑窗口、截尾和标准化方法。

建议主版本保留原始 `TR / ATR_prev`。若长尾影响图表，可增加仅用于显示的
`log1p(X_range)`，但原始值必须保存。

稳健性 challenger 可以是：

\[
X^{range,median}_{t,n}
=
\frac{TR_t}{\operatorname{median}(TR_{t-n:t-1})}
\]

它应作为明确的替代尺度比较，不应与主版本静默混合。

---

## 7. 技术分析上如何联合阅读 X_now

### 7.1 高波幅、高净位移、高转化率

```text
X_range 高
|X_close| 高
|X_share| 高
```

含义：活动突然放大，而且大部分运动最终保留为一个方向上的收盘位移。这是最接近经典
方向性扩张棒的组合。

### 7.2 高波幅、低净位移、低转化率

```text
X_range 高
|X_close| 低
|X_share| 低
```

含义：市场非常剧烈，但没有哪一方把大部分运动保留到收盘。可能是大型 outside bar、
消息冲击、清算与反向吸收，或区间内部的激烈双向交易。它是波动扩张，不是方向确认。

### 7.3 高 Gap、盘中继续同向

```text
X_gap 与 X_intraday 同号
```

含义：开盘前发生重新定价，开盘后继续沿相同方向推进，即 gap-and-go。

### 7.4 高 Gap、盘中反向

```text
X_gap 与 X_intraday 异号
```

含义：开盘前发生重新定价，但开盘后受到反向交易，即 gap-and-fade 或部分回补。

### 7.5 高波幅但仍在旧区内部

```text
X_range 高
reference_departure 低
```

含义：旧活动区域内部发生了强烈运动，但价格尚未离开此前大部分价格范围。不能直接称为
突破。

### 7.6 离区明显但当日波幅不高

```text
reference_departure 高
channel_escape = true
X_range 低
```

含义：价格缓慢越过旧通道或继续沿趋势延伸，但没有新的日度波动冲击。它是位置变化，
不是速度或活动方式的突然变化。

### 7.7 最完整的突破式扩张证据

```text
X_range 高
|X_close| 高
|X_share| 高
reference_departure 高
channel_escape = true
```

含义：活动放大、方向结果清楚、收盘保留度高，并且当前价格位于大量旧区之外。即便如此，
当天也只能称为强突破式扩张证据，不能声称未来一定成功。是否站住要由独立前向结果判断。

---

## 8. 离开旧活动区域：三种不同强度的定义

“离开旧区”与“波动扩张”必须分开。建议区分软离区、预设通道逃逸和真实结构突破。

### 8.1 软离区：清除了多少旧日线范围

对过去 \(n\) 根已完成日线，定义：

\[
\boxed{
Clearance^{up}_{t,n}
=
\frac1n
\sum_{i=t-n}^{t-1}
\mathbf 1(C_t>H_i)
}
\]

\[
\boxed{
Clearance^{down}_{t,n}
=
\frac1n
\sum_{i=t-n}^{t-1}
\mathbf 1(C_t<L_i)
}
\]

例如 `Clearance_up_20 = 0.8` 表示当前收盘高于过去 20 根日线中 16 根的完整最高价，即
清除了其中 80% 的日线价格范围。

优点：

- 完全因果；
- 不需要先宣布“平衡已经建立”；
- 一个孤立长影线只影响 `1/n`，不会决定整个边界；
- 输出连续、容易手算和前缀重放。

限制：

- 它只是相对旧日线分布的位置；
- 它不证明此前存在真实交易区间；
- 在持续趋势中可以长期偏高；
- 上下两个分量理论上都可能非零，不应未经研究强行压成单一有符号值。

建议把它作为 Phase 1A challenger，名称使用 `prior_range_clearance`，不要称为真实突破。

### 8.2 预设通道逃逸：Donchian 型规则事件

对过去 \(n\) 日：

\[
b^{up}_{t,n}
=
\max_{i=t-n}^{t-1}H_i
\]

\[
b^{down}_{t,n}
=
\min_{i=t-n}^{t-1}L_i
\]

当日收盘越界距离：

\[
Escape^{up}_{t,n}
=
\frac{\max(0,C_t-b^{up}_{t,n})}{A_{t-1}}
\]

\[
Escape^{down}_{t,n}
=
\frac{\max(0,b^{down}_{t,n}-C_t)}{A_{t-1}}
\]

建议命名：

```text
channel_close_escape_20
channel_close_escape_55
```

不建议命名：

```text
balance_breakout
true_breakout
structure_escape
```

20 日可作为主规则事件，55 日作为稳健性版本。5 日和 10 日新高在持续趋势中更像局部
通道延伸，不能机械继承 E/D/B 的完整窗口族。

每日越界条件与事件采样应分开保存：

- 原始日表可以保留每个满足条件的日期；
- 前向事件分析必须预声明如何处理连续多日新高、新低和重叠观察窗口；
- 不得为了得到更漂亮的 Retention 结果而事后去重；
- 若采用冷却期、首次事件或事件簇规则，必须单独版本化，因为这已属于事件采样协议，
  不是通道边界公式本身。

### 8.3 真正的结构突破

技术分析中的“突破交易区间、形态或重要摆动边界”通常暗含：

- 此前有一个有意义的活动区域；
- 边界被多次测试；
- 区域内存在反复占用；
- 中心迁移较弱；
- 区间有起点、年龄和失效条件。

这些对象需要区域识别、摆动或 Episode。它们不是普通固定窗口特征，应该进入 Phase 2。

路线 B 的“先判断活动区域有资格，再判断逃逸”本质上已经是两阶段模型：区域识别模型
加逃逸模型。其软证据可以在 Phase 1A 保留为连续候选，但硬资格、区域成立事件和结构
突破语义应后置，不能作为几个阈值静默塞回 X_now。

---

## 9. Retention 应改写为独立的边界接受结果族

设：

- \(b\) 为事件日已经冻结的边界；
- \(s=+1\) 表示向上事件，\(s=-1\) 表示向下事件；
- \(k\) 为预先指定的未来期限；
- 所有结果始终引用事件时边界 \(b\)，不能在未来改用滚动更新后的通道。

### 9.1 第 k 日边界外进展

\[
\boxed{
EndpointProgress_{t,k}
=
\frac{s(C_{t+k}-b)}{A_{t-1}}
}
\]

解释：

- 大于 `0`：第 k 日仍在事件边界外；
- 接近 `0`：回到边界附近；
- 小于 `0`：重新进入边界内；
- 数值越大：相对事件前活动尺度，后续延伸越远。

### 9.2 边界外收盘占比

\[
\boxed{
TimeOutside_{t,k}
=
\frac1k
\sum_{i=1}^{k}
\mathbf 1\{s(C_{t+i}-b)>0\}
}
\]

它回答：

> 突破后未来 k 日中，有多少比例的收盘仍位于事件边界外？

### 9.3 重新进入与首次重新进入

建议至少保存：

```text
any_close_reentry_k
first_close_reentry_lag_k
```

其中 `any_close_reentry_k` 表示未来 k 日内是否至少一次收盘重新进入事件边界；
`first_close_reentry_lag_k` 表示首次重新进入距离事件日多少根交易日。

### 9.4 为什么不把原始 Retention 比率作为主结果

原式：

\[
\frac{s(C_{t+k}-b)}{s(C_t-b)}
\]

若突破日只越过边界很小距离，分母会接近零，结果可能爆炸。裁剪虽然限制极值，却会丢失
继续延伸信息。

因此建议：

- `EndpointProgress` 与 `TimeOutside` 作为主结果；
- 原比例仅作次要诊断；
- 若输出比例，必须预先规定最小初始越界距离；
- 最小距离不得反向成为事件资格，除非另建明确版本；
- 不得从多个 k 中事后选择表现最好者。

可讨论但尚未确认的期限方案：

```text
k = 5    主结果：短期价格接受
k = 20   次结果：约一个交易月的持续性
```

---

## 10. 四个数值例子

以下统一假设昨日：

```text
C_prev = 100
H_prev = 101
L_prev = 99
ATR_prev = 2
```

### 10.1 强势盘中方向扩张

今天：

```text
O = 100.2
H = 106
L = 99.8
C = 105.5
```

计算：

\[
TR=\max(6.2,6,0.2)=6.2
\]

\[
X^{range}=6.2/2=3.10
\]

\[
X^{close}=(105.5-100)/2=2.75
\]

\[
X^{share}=5.5/6.2\approx0.89
\]

\[
X^{gap}=0
\]

解释：今天波幅约为正常水平的 3.1 倍，最终向上移动 2.75 个 ATR，约 89% 的 TR
转化成向上收盘位移，而且不是开盘跳空造成。这是典型方向性扩张。

### 10.2 向上 Gap 后大幅回落

今天：

```text
O = 103
H = 104
L = 99.5
C = 100.5
```

计算：

\[
TR=4.5
\]

\[
X^{range}=2.25
\]

\[
X^{close}=0.25
\]

\[
X^{share}=0.5/4.5\approx0.11
\]

\[
X^{gap}=(103-101)/2=1.0
\]

\[
X^{overnight}=1.5,
\qquad
X^{intraday}=-1.25
\]

解释：开盘前出现强向上重新定价，但盘中几乎全部被卖回。它是开盘扩张、盘中拒绝，
不能与 gap-and-go 视为同一种行为。

### 10.3 剧烈双向冲击

今天：

```text
O = 100
H = 106
L = 94
C = 100.2
```

计算：

\[
X^{range}=12/2=6.0
\]

\[
X^{close}=0.2/2=0.10
\]

\[
X^{share}=0.2/12\approx0.017
\]

解释：市场活动达到正常水平的六倍，但最终几乎没有净位移。这是极强的波动扩张和双向
竞争，不是方向性扩张。

### 10.4 缓慢越过 20 日高点

假设：

```text
C_prev = 101.8
过去20日最高价 = 102
ATR_prev = 2
O = 101.9
H = 102.4
L = 101.7
C = 102.2
```

计算：

\[
TR=0.7
\]

\[
X^{range}=0.35
\]

\[
X^{close}=0.20
\]

\[
Escape^{up}_{20}=(102.2-102)/2=0.10
\]

解释：当前收盘确实越过 20 日最高价，但当日活动低于正常水平。这是通道位置上的缓慢
延伸，不是新的波动冲击。

---

## 11. X 与 E / D / B 的职责分工

| 量 | 核心问题 | 时间形态 |
|---|---|---|
| `D` | 最近一段价格中心向哪边迁移、速度多快 | 多日窗口行为 |
| `E` | 最近一段路径有多少转化成净位移 | 多日窗口行为 |
| `B` | 最近一段是否涨跌混杂、价格是否反复覆盖 | 多日窗口行为 |
| `ActivityLevel` | 当前波动制度本身有多高 | 滞后活动背景 |
| `X_now` | 今天是否相对原活动尺度出现新冲击 | 单日转换证据 |
| `Departure` | 当前价格相对旧区处在哪里 | 固定窗口位置 |
| `Retention` | 边界事件后来是否被保留 | 前向事件结果 |

可以用河流类比：

- `D`：河水总体流向；
- `E`：河道是否直接；
- `B`：水流是否频繁双向回荡；
- `ActivityLevel`：平时水量有多大；
- `X_now`：今天水量是否突然暴涨；
- `Departure`：水是否已经漫出原河道；
- `Retention`：几天后水是否仍在河道外。

特别注意：`X_share` 与单日效率直觉相近，但它是由当日 TR 和收盘位移推导的诊断量，
不应被解释成新的多日 E，也不应在综合时与 `X_range`、`X_close`重复计权。

---

## 12. 为什么不应该一开始产生总分

原始候选形式类似：

\[
X_t
=
a_1z(TR_t)
+a_2Escape_t
+a_3z(V_t)
+a_4Gap_t
+a_5Retention_t
\]

首版不应采用，原因如下。

### 12.1 混合了不同时间角色

- TR 和 Gap 是当日事实；
- Escape 是相对某个参考边界的位置；
- Retention 是未来结果；
- volume 当前不可用。

它们不是同一种对象。

### 12.2 重复记录同一冲击

Gap 已经进入 TR。再把 Gap 与 TR 相加，会让 Gap 日被重复加分，即使开盘后完全回补。

### 12.3 同分不同义

大幅收平、单边趋势棒、gap-and-fade、缓慢 55 日新高，都可能通过不同分量组合得到相似
总分，但技术分析含义完全不同。

### 12.4 权重没有自然依据

在没有明确统计目标、标准化和独立选择协议前，`a_1...a_5` 只是叙事权重，不是研究结果。

因此本文中的“组成”是并列对象，不是数学加法：

\[
\mathbf X^{now}
=
(X^{range},X^{close},X^{gap})
\]

\[
\mathbf X^{departure}
=
(Clearance,ChannelEscape)
\]

\[
\mathbf Y^{acceptance}
=
(EndpointProgress,TimeOutside,Reentry)
\]

只有在后续选择块证明压缩有明确收益、权重可解释且信息损失可接受时，才讨论综合。

---

## 13. 建议的版本、字段与模块边界

以下是建议，不是已冻结命名。

### 13.1 日度冲击版本

```text
BM-X-01-OHLC-impulse-v1
code id: bm_x_01_ohlc_impulse_v1
```

主字段：

```text
expansion_range_atr_prev
expansion_close_atr_prev
expansion_gap_prev_range_atr
expansion_close_share
```

诊断字段：

```text
expansion_opening_jump_atr_prev
expansion_intraday_followthrough_atr_prev
activity_level_atr_pct_prev
```

建议状态：

```text
ok
insufficient_history
zero_scale
zero_range
invalid_ohlc
```

其中：

- `zero_scale`：昨日 ATR 尺度为零或无效；
- `zero_range`：当日 TR 为零，`X_share` 缺失；
- 不得用零、极小常数或人工值替代未定义结果。

### 13.2 离区候选版本

```text
BM-X-DEP-01-prior-range-clearance-v1
```

字段：

```text
prior_range_clearance_up_20
prior_range_clearance_down_20
prior_range_clearance_up_55
prior_range_clearance_down_55
```

### 13.3 通道事件版本

```text
BM-X-CH-01-donchian-close-escape-v1
```

事件表至少保存：

```text
event_id
event_at
boundary_at
direction
reference_type
window
boundary
initial_excess_atr
feature_version
```

### 13.4 边界接受结果版本

```text
BM-X-ACC-01-boundary-acceptance-v1
```

结果表至少保存：

```text
event_id
event_at
boundary_at
confirmed_at
horizon
endpoint_progress_atr
time_outside
any_close_reentry
first_close_reentry_lag
outcome_version
```

### 13.5 建议代码边界

```text
src/market_cycle/measurements/expansion.py
src/market_cycle/measurements/departure.py
src/market_cycle/events/channel_escape.py
src/market_cycle/outcomes/acceptance.py
```

Phase 1A continuous replay 只应加入 `activity_level`、X_now 和经确认保留的连续 departure；
事件表和前向 outcome 不应回填进日度连续 replay。

X_now 本身只需要前一日和已预热 ruler，理论上可以早于 E/D/B 四尺度齐套日产生；但联合
审计页面若要求全特征齐套，可以在展示层对齐到共同起点。测量模块不应为了展示方便而
丢弃本来有效的更早行。

---

## 14. 最低性质测试与前缀重放要求

### 14.1 数学性质

1. 所有 OHLC 同乘正数常量后，全部 ATR 标准化 X 分量不变；
2. 对称上下镜像后，`X_range` 不变，`X_close`、`X_gap` 反号；
3. 对所有有效行：
   \[
   |X^{close}|\le X^{range}
   \]
4. 对所有有效 `X_share`：
   \[
   -1\le X^{share}\le1
   \]
5. `X_gap != 0` 时，开盘必须位于前一日完整区间之外；
6. `ATR_prev=0` 时，依赖尺度的分量缺失并标记 `zero_scale`；
7. `TR=0` 时，`X_range=0`，但 `X_share` 缺失并标记 `zero_range`；
8. Clearance 必须位于 `[0,1]`；
9. channel boundary 只能读取 `t-1` 及以前；
10. Retention 在 `t+k` 前不可见，且始终引用事件时冻结边界。

### 14.2 合成反例

至少覆盖：

- 安静 inside doji；
- 巨大 outside doji；
- 无 Gap 的大趋势棒；
- gap-and-go；
- gap-and-fade；
- 缓慢连续创新高但当日 TR 很小；
- 巨大 TR 但仍位于旧通道内；
- 一根孤立长影线决定 Donchian 边界；
- 相同 TR、不同收盘净位移；
- 相同净位移、不同 TR；
- 连续多日通道新高，验证事件采样与日度条件没有混淆；
- 突破日初始越界极小，验证 outcome 不发生比率爆炸。

### 14.3 前缀重放

对每个可用日期 \(t\)：

```text
批量全样本计算在 t 的结果
=
只提供截至 t 数据独立重算的最后一行结果
```

对事件和 outcome 还要分别验证：

- 事件表截至 `t` 不包含任何未来 outcome；
- 当样本扩展到 `t+k` 时，只追加 outcome，不修改原事件的边界与资格；
- 同一版本下不删除当时确实发生但后来失败的事件。

---

## 15. 真实窗口开发审计与选择块

### 15.1 开发期真实窗口

至少复核：

- 1987-10-19：极端方向性冲击；
- 1989-10-13：短长尺度冲突；
- 1994 年安静低效率窗口；
- 2008-10-10：高波动急跌但仍有往返；
- 2020-03-23：剧烈低效率与高双向性；
- 2024 年慢牛：方向存在但扩张不一定高。

这些窗口用于发现公式反例，不能替代正式选择块。

### 15.2 增量价值不能只看相关系数

选择块应回答：

1. `X_range` 是否真正区分了当前 B 无法区分的安静低效率与剧烈低效率；
2. 在 E/D/B 与 ActivityLevel 相似的日期中，X_now 是否仍描述不同的当日冲击形态；
3. `X_close` 是否只是 D 的一日重命名，还是能表达转换日的局部冲击；
4. `X_gap` 在控制 `X_range`、`X_close` 后是否仍有解释价值；
5. Clearance 是否能被 D/E 简单重构；
6. 在匹配 X/E/D/B 后，channel escape 是否仍对应不同的接受结果；
7. 新增维度是否值得其复杂度、审计和接口成本。

### 15.3 明确淘汰条件

- Gap 在控制 range shock 与 close impulse 后没有信息：降级为诊断；
- Clearance 可被现有连续量稳定重构：删除；
- channel escape 在匹配连续量后没有接受差异：保留为基准而非核心；
- 复杂综合不优于简单向量：拒绝总分；
- 任何必须依赖“真实平衡区域”才能成立的结论：移交 Phase 2；
- 因果回放、尺度状态或边界冻结失败：实验无效，先修实现。

正式比较应使用预先划定的连续时间块。若分析 k 日前向结果，块边界必须处理未来标签重叠
与事件聚集，不能让同一未来路径在训练、选择和确认块之间泄漏。

---

## 16. 主要失败模式与禁止路径

### 16.1 用今日 ATR 衡量今日冲击

会使大波动抬高自己的分母，低估冲击。主定义应使用 `ATR_{t-1}`。

### 16.2 把 ActivityLevel 当成 X_now

高 ATR 表示市场已经处于高活动制度，不等于今天出现了新的扩张。活动水平和活动创新
必须分开。

### 16.3 把 Gap 与 TR 再次相加

Gap 已进入 TR，数学相加会重复记录同一价格跳跃。

### 16.4 把 Donchian 新高叫成真实结构突破

滚动最高高 / 最低低是规则通道边界，不证明此前存在交易区间、形态或重要摆动。

### 16.5 用未来 Retention 修正当天 X

高 X 后失败是应保留的历史事实，不是公式错误。不得回填、删事件或重画边界。

### 16.6 根据未来成功率挑阈值、窗口和期限

不能先看 Retention 或未来收益，再决定 TR 尺子、Gap 口径、20/55 窗口、k 或最小越界距离。

### 16.7 把派生比率当作独立证据重复计权

`X_share = X_close / X_range`，不是第三份独立信息。它适合解释，不适合再加一次权重。

### 16.8 静默恢复区域状态机

若路线 B 开始定义区域成立、年龄、测试次数、首次逃逸与失效，它已经进入结构或事件状态，
必须独立版本化，不能继续假装是普通 X 分量。

---

## 17. 推荐实施顺序

### 步骤 1：先实现最小 X_now

只实现：

```text
activity_level_atr_pct_prev
expansion_range_atr_prev
expansion_close_atr_prev
expansion_gap_prev_range_atr
expansion_close_share
```

并完成性质测试、边界状态、冻结样本检查和逐日前缀重放。

### 步骤 2：加入连续审计页面

与 E/D/B 并列显示，不生成总分，不设置预测阈值。重点观察：

- 安静低效率 vs 剧烈低效率；
- 大波动收平 vs 大波动单边；
- gap-and-go vs gap-and-fade；
- 慢趋势延伸 vs 新冲击。

### 步骤 3：实现 prior-range clearance challenger

先作为独立连续位置量，不与 X_now 综合，不称为突破。

### 步骤 4：实现 20/55 日 channel escape 基准事件

边界在事件日前冻结；原始日度条件、事件采样和事件簇处理分开版本化。

### 步骤 5：实现独立 acceptance outcome engine

结果表只在 `confirmed_at=t+k` 后产生，不进入连续 replay。主结果使用
`EndpointProgress`、`TimeOutside` 和 reentry 字段。

### 步骤 6：完成联合开发审计后再预注册选择块

先观察分量职责、相关性、反例与重复信息，再决定：

- Gap 是否保留；
- Clearance 是否保留；
- channel event 是否有增量；
- 是否需要任何压缩；
- 是否出现必须进入 Phase 2 的真实结构缺口。

---

## 18. 面向后续 Agent 的执行约束

后续 Agent 读取本文时，应遵守以下规则：

1. 本文是 `external-design-proposal`，不是 canonical decision；
2. 未经用户明确确认，不得新增或修改 D-* 决策来宣告方案已采用；
3. 不得把建议字段、20/55 窗口或 5/20 日 horizon 描述成已冻结参数；
4. 实现 TR/ATR 相关量时必须读取共享 ruler，不得在测量模块内重算另一套尺子；
5. X_now、channel event 和 acceptance outcome 必须分模块、分表、分时间角色；
6. 不得产生首版加权总分，也不得把连续分数称为概率、置信度或趋势确认；
7. 不得将 Donchian 边界描述成真实交易区间或 Brooks 结构突破；
8. 所有边界、事件和 outcome 必须通过逐日前缀重放；
9. 若实现结果与本文公式不同，必须说明原因、升版本并同步文档；
10. 若发现必须识别区域、腿、摆动或 Episode，停止向 Phase 1A 塞规则，转为提出 Phase 2
    结构研究问题；
11. 对已有工作树或仓库做修改前，先检查相关决策、协议、测试和审计页面，避免只改代码；
12. 任何正式选择都应在连续时间块中预注册，不以几个历史图形或未来收益最大化作为依据。

建议 Agent 在实施 PR 中至少交付：

```text
公式与字段定义
输入和时间契约
状态处理
性质测试
前缀重放测试
冻结样本起止检查
审计页面字段
协议与候选目录更新
未决问题与淘汰条件
```

---

## 19. 最终设计摘要

从第一性原理看，完整的扩张逻辑应按下面顺序组织：

```text
市场原本每天正常活动多远？
→ 今天的真实波幅是否相对旧尺度突然放大？
→ 放大的运动最终形成了多少方向收盘位移？
→ 冲击是在开盘前跳跃形成，还是盘中形成？
→ 当前价格相对旧活动分布处在哪里？
→ 是否越过一个明确、预先冻结的规则边界？
→ 越界之后，未来是否继续留在边界外？
```

因此，Phase 1A 的最小扩张表示应是：

\[
\boxed{
\mathbf X^{now}_t
=
\left(
\frac{TR_t}{ATR_{t-1}},
\frac{C_t-C_{t-1}}{ATR_{t-1}},
\frac{GapOutsidePrevRange_t}{ATR_{t-1}}
\right)
}
\]

并附加派生解释量：

\[
\boxed{
X^{share}_t
=
\frac{C_t-C_{t-1}}{TR_t}
}
\]

这里的向量表示并列证据，不是数学相加。

`prior_range_clearance` 是相对旧价格分布的位置；`channel_escape` 是预设通道规则事件；
`acceptance_outcome` 是未来结果；真正的区间、摆动和形态突破属于 Phase 2。

这套分层的核心价值不是让 X 看起来更复杂，而是避免一个数字同时冒充活动水平、当日冲击、
技术突破和未来成功。只有先把这些对象分开，后续的实现、审计、反例、选择和否决才会真正
可复现、可解释和可反驳。
