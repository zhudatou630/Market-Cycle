# Market Cycle V2

极简、结果导向的「市场周期（Market Cycle）」量化研究仓库。

## 研究什么

围绕 Al Brooks 的核心命题，把问题收束为三个可客观验证的裁判题：
1. **趋势 vs 震荡（顺势/逆势开关）**：当前价格过程能否区分未来 10–60 天更偏向「顺势延续」还是「均值回归」？
2. **趋势启动（真突破 vs 假突破）**：刚创阶段新高或出现扩张冲击时，事前/当日结构能否拉开真突破与假突破的胜率与赔率？
3. **趋势衰竭（健康中继 vs 见顶转震荡）**：处于上升趋势中时，效率衰减、重叠抬升或高潮加速能否提前提示后续收益风险比恶化？

## 目录结构

```text
04-Market Cycle/
├── AGENTS.md        # 五条研究铁律与极简原则
├── README.md        # 本说明
├── decisions.md     # 每次实验的大白话结论台账
├── data/snapshots/  # 冻结的 SPX (1927–2026) 日线 OHLC 快照
├── studies/         # 单文件研究脚本（一个问题一个脚本）
└── output/          # 临时输出与图表（不入 Git）
```

## 数据与环境

- **Python 环境**：沿用本地 `.venv/bin/python`（已装 `pandas`、`numpy`、`pyarrow`）。
- **美股数据**：`data/snapshots/spx_daily_2026-07-21_d828fbc8.parquet`（SPX 日线 OHLC，1927-12-30 → 2026-07-21；研究从 1984 起，2020 起为封存区）。
- **A 股数据**：只读复用 `~/market-data/raw_lake/`（如 `sw_daily` 申万一级行业日线、`sw_index_meta`、`stock_daily`；2024-11-01 起为封存区）。

## 关于 V1 归档

旧版 V1（Phase 1A 连续特征审计、因果路径、前端审计页等）已完整封存在 Git 分支 `archive/v1-phase1a-202607` 与标签 `v1-phase1a-archive`，随时可用 `git show` 或切分支查阅；主分支 `main` 从本版本起轻装重来。
