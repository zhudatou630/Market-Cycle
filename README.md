# Market Cycle V2

极简、结果导向的「市场周期（Market Cycle）」量化研究仓库。

## 研究路线

### 第一阶段（已收口）：日线宽基/行业指数上的简单连续物理量裸考

围绕 Al Brooks 的核心命题，先在 SPX 与 A 股 31 个申万一级行业日线上，用零拐点单窗口连续物理量完成三道裁判题与一次性封存区盲测（详见 `decisions.md` 与 `docs/`）：
1. **趋势 vs 震荡（顺势/逆势开关）**（已完成：`studies/01_trend_vs_range.py`、`01b_diagnose_loose_ends.py`，总结见 `docs/01_trend_vs_range_summary.md`）
2. **趋势衰竭（健康中继 vs 高潮赶顶）**（已完成：`studies/02_trend_exhaustion.py`，总结见 `docs/02_trend_exhaustion_summary.md`）
3. **趋势启动（真突破 vs 假突破）**（已完成：`studies/03_breakout_true_vs_false.py`，总结见 `docs/03_breakout_true_vs_false_summary.md`）
4. **第一阶段封存区盲测收口**（已完成：`studies/04_holdout_final_audit.py`，总结见 `docs/04_holdout_final_verdict.md`）

> **第一阶段核心沉淀**：在日线指数上，单靠滚动平均型 K 线几何量扣掉「价格位置 + 当前波动率」后几乎没有独立超额魔法，无需建离散周期状态机；但样本外依然稳健的两条极简准则已沉淀下来——A 股看「位置 + 波动温度（安静低波慢推与 60 日缩波突破更稳，高位发烧易踩雷）」，美股看「横盘箱体起跳的 20 日突破」。

### 第二阶段（待分次推进）：深入 Al Brooks 的结构、序列、多周期与标的粒度

第一阶段只摸到了日线指数上“单窗口均值”的表层；下面 4 个 Al Brooks 核心深水区彼此独立，后续按单文件脚本（`studies/05_...` 起）分 session 逐个拆解验证，全程继续遵守五条铁律：

- **方向 A：突破回踩与二次上车（Breakout Pullback / High 2 / 假突破反向）**
  - 紧扣 Study 03 发现的「65%–70% 首次新高会在 10 天内回踩前高」：检验 Al Brooks 核心理念——不盲追第一脚突破，改考「强突破后第一次回踩测试成功（Pullback / High 2）」的胜率与赔率，以及「箱体边缘假突破失败后做反向回归」是否成立。
- **方向 B：有先后顺序的博弈结构（窄通道 vs 宽通道、三推楔形、周期演化路径）**
  - 摆脱把 20 天拍扁成静态均值的局限，检验真正带时间顺序的结构：
    1. **窄通道（Tight Channel）vs 宽通道（Broad Channel）**：回调深浅与贴均线推进节奏，能否区分“只能顺势接回调”与“可双向博弈”；
    2. **三推楔形（Wedge / 3 Pushes）**：连续三次创阶段新高但每次推进幅度/动能递减，是否比单窗口指标更准地提示衰竭转折；
    3. **周期演化顺序**：突破 $\to$ 窄通道 $\to$ 宽通道 $\to$ 交易区间的转移概率在真实价格里到底成不成立。
- **方向 C：多周期嵌套（大周期背景 Always-In × 小周期周期切换）**
  - 检验周线处于强趋势背景时，日线进入横盘交易区间（牛旗 Bull Flag）或回调，与周线本身也在震荡时的日线结构，在未来收益与赔率上有多大差异。
- **方向 D：切换标的粒度（从行业指数切到 A 股个股日线）**
  - 行业指数是几十只股票的加权平均，天然抹平了个股微观价格行为（微型通道、双底测试、突破回踩、上影派发等）；用 `~/market-data` 全 A 股个股日线 `stock_daily` 重验核心结构，查明第一阶段失效的特征到底是真的没用，还是被指数加权平均抹平了。

> **第二阶段纪律备注**：SPX `2020–2026` 与 A 股申万行业 `2024.11–2026.11` 封存区已在第一阶段开封过一次；第二阶段做新命题时，绝不拿旧封存区倒推参数，优先以全新标的池（如个股）或独立样本外规则做严格检验。

## 目录结构

```text
04-Market Cycle/
├── AGENTS.md        # 五条研究铁律与极简原则
├── README.md        # 本说明与分阶段研究路线
├── decisions.md     # 每次实验的大白话结论台账
├── docs/            # 各次实验的大白话总结报告
├── data/snapshots/  # 冻结的 SPX (1927–2026) 日线 OHLC 快照
├── studies/         # 单文件研究脚本（一个问题一个脚本）
└── output/          # 临时输出与图表（不入 Git）
```

## 数据与环境

- **Python 环境**：沿用本地 `.venv/bin/python`（已装 `pandas`、`numpy`、`pyarrow`）。
- **美股数据**：`data/snapshots/spx_daily_2026-07-21_d828fbc8.parquet`（SPX 日线 OHLC，1927-12-30 → 2026-07-21；研究从 1984 起，2020 起为第一阶段封存区；**注意：`1984–2007` 普遍存在 `open == 前收`，凡用到 `open` 的特征强制仅用 `2008` 年起干净段**）。
- **A 股数据**：严格只读复用 `~/market-data`——读取申万一级行业 `sw_daily`（`src='SW2021', level='L1'`）与 `index_daily`、`stock_daily` 时，通过只读 `~/market-data/catalog/metadata.sqlite` 定位最新 READY 快照的 `table_manifest`（直接读初始 `raw_lake/sw_daily` 只有 L2；`2024-11-01` 起为第一阶段封存区）。

## 关于 V1 归档

旧版 V1（Phase 1A 连续特征审计、因果路径、前端审计页等）已完整封存在 Git 分支 `archive/v1-phase1a-202607` 与标签 `v1-phase1a-archive`，随时可用 `git show` 或切分支查阅；主分支 `main` 从本版本起轻装重来。
