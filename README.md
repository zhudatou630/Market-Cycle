# Market Cycle

Market Cycle 是一个美股市场周期与市场状态表示研究项目。当前范围很窄：使用
Yahoo Finance 的 SPX（`^GSPC`）日线 OHLC 数据，优先研究 Brooks Market Cycle；
O'Neil M、个股横截面和交易决策属于后续阶段。

当前目标不是直接生成买卖信号，而是研究一种可复现、时点因果、不会重绘的价格过程
表示：它应能说明当前判断依赖什么证据，以及后续研究是否能证明这些证据有超越简单
趋势和波动基线的增量信息。

## 当前状态

Phase 0 已收口；Phase 1A 尺子已实现。研究快照
`spx_daily_2026-07-21_d828fbc8`，主样本 `spx_ohlc_main_1984`（研究行
1984-02-10 起，含 `tr_pct` / `atr_pct_14`），见质量报告与 D-010。下一步是
工作协议与测量模块（路径效率等）；尚未实现状态引擎或交易系统。

## 快速开始

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

读取或首次下载 SPX 日线数据：

```python
from market_cycle.data import get_bars, get_research_bars

daily = get_bars("^GSPC")
weekly = get_bars("SPX", freq="W")

# Phase 1A 研究入口：快照 + 尺子，齐套日起无空头
bars, meta = get_research_bars()
```

首次运行或执行 `refresh=True` 需要网络访问 Yahoo Finance。工作缓存为
`data/raw/spx_daily.parquet`（可覆盖，默认不提交）。正式研究钉住
`data/snapshots/` 与 `snapshot_id`，波动尺见 D-010。

## 文档入口

- [研究操作指南](docs/research-guide.md)：面向人的整体路线、阶段理由、产物和分叉。
- [研究基础](docs/foundation.md)：共同方法论、Brooks/O'Neil 概念边界。
- [研究计划](docs/research-plan.md)：当前 SPX/Brooks 假说、验证要求和开放问题。
- [Brooks 测量候选](docs/brooks-measurement-candidates.md)：原研究公式及其数据、
  时点和前视风险说明。
- [决策记录](docs/decisions.md)：已经确认的项目选择。
- [Phase 0 数据质量报告](docs/reports/phase0-data-quality.md)：快照、主样本与
  资格规则。
- 原始对话归档：本地私有文件 `docs/references/`，不提交到公开仓库；只用于历史
  溯源，不是当前规范。

## 研究边界

- 当前只有 SPX 日线 `open/high/low/close`，没有可靠的指数自身成交量；
- 不能据此实现完整 O'Neil FTD、派发、市场宽度或领导股分析；
- Brooks 路径和形态结构计划在基础测量之后作为独立阶段研究；
- 状态分析与交易决策分层，项目当前不生成订单或仓位建议。

数据、公式和研究结论都应以当前文档和代码为准，不把历史 AI 对话中的建议自动
视为已确认事实。