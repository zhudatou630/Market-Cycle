# Project Agent Guide

本项目研究美股市场周期与市场状态表示。当前实现范围很窄：只有 SPX
（Yahoo `^GSPC`）日线 OHLC 数据层；当前研究优先级是 Brooks Market Cycle，
O'Neil M、个股横截面和交易系统均在后续阶段。是否以完整 Brooks 复刻还是
Brooks-inspired 表示为最终目标仍是 `Q-001`，不要提前宣告已经决定。

## 必读顺序

不要默认通读 `docs/references/ChatGPT-O Neil & Brooks Market Cycle.md`。它是
428KB 的历史对话档案，不是当前规范。

按任务读取最小上下文：

| 任务 | 必读材料 |
|---|---|
| 理解项目整体目标、研究顺序或阶段分叉 | `docs/research-guide.md` |
| 理解共同方法论和术语 | `docs/foundation.md` 第 3 节 |
| 研究 Brooks Market Cycle | `docs/foundation.md` 第 4 节 |
| 设计或选择 Brooks 基础变量公式 | `docs/brooks-measurement-candidates.md`，再读 `docs/research-plan.md` 第 5 节 |
| 研究 Brooks 路径骨架或形态结构 | `docs/foundation.md` 第 4.4 节、`docs/brooks-measurement-candidates.md` 第 5-6 节和 `docs/research-plan.md` 第 7 节 |
| 研究 O'Neil Market Direction | `docs/foundation.md` 第 5 节 |
| 研究两者关系和远期融合 | `docs/foundation.md` 第 6-7 节 |
| 研究当前 SPX/Brooks 阶段、假说或验证方案 | `docs/research-plan.md`，再按需读 `docs/foundation.md` 对应章节 |
| 修改已确认方向、数据边界或项目范围 | `docs/decisions.md` |
| 修改 SPX 数据 API | `src/market_cycle/data/bars.py` 和 `docs/decisions.md` 的 D-001、D-002 |
| 核验历史论述、原始链接或被舍弃方案 | 先在归档对话中用标题/关键词定位，再定点读取相关对话 |

## 权威顺序

1. 代码和测试说明当前实际行为。
2. `docs/decisions.md` 说明已经确认的项目意图。
3. `docs/foundation.md` 说明当前权威概念和边界。
4. `docs/research-plan.md` 中的 `H-*`、`Q-*` 是待验证假说和开放问题。
5. `docs/brooks-measurement-candidates.md` 保存候选操作定义，不代表已经采用。
6. `.pi/grill/` 和 `docs/references/` 只用于历史溯源。

`docs/research-guide.md` 只做面向人的导航，不产生独立权威；其中细节冲突时仍按
以上顺序处理。

代码与确认决策冲突时，不要静默选择一边：指出冲突并修正错误的一方。

## 文档规则

- 一个概念只有一个所有者：概念归 `foundation`，Brooks 候选公式归候选目录，
  研究假说和路线归 `research-plan`，已确认选择及理由归 `decisions`，实际行为归
  代码和测试。
- `research-guide` 只解释顺序、理由和分叉，并链接到所有者，不复制公式、参数和
  开放问题清单。
- 不把 ChatGPT 的建议、示例数值或伪输出升级为事实或决策。
- 不把未经校准的分数称为概率或置信度。
- 不用未来数据重写过去时点的状态；候选、确认和失效时间必须分开。
- 新决策用 `D-*`；新假说用 `H-*`；新开放问题用 `Q-*`。
- 文档以中文为主，代码标识和必要术语保留英文。
