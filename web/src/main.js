import "./styles.css";

import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  LineSeries,
  PriceScaleMode,
  createChart,
} from "lightweight-charts";
import { createIcons, icons } from "lucide";

const API_URL = "/api/replay.json";
const DEFAULT_RANGE_BARS = 252;
const PLAYBACK_INTERVAL_MS = 240;
const MIN_FEATURE_PANE_HEIGHT = 72;
const MIN_PRICE_PANE_HEIGHT = 260;
const DEFAULT_PANE_WEIGHTS = {
  price: 5,
  efficiency: 1,
  direction: 1,
  twoSidedness: 1,
  expansion: 1,
  clearance: 1,
};
const FEATURES = [
  {
    key: "efficiency",
    payloadKey: "efficiency",
    elementId: "efficiency",
    shortTitle: "E",
    title: "效率",
    unitBounded: true,
    legendId: "efficiency-legend",
    defaultSeries: ["equal", "midlong"],
    series: [
      { key: "equal", field: "efficiency_equal", label: "等权", color: "#087f6f" },
      { key: "midlong", field: "efficiency_midlong", label: "中长期", color: "#576db5" },
      { key: "5", field: "efficiency_5", label: "5", color: "#c4573d" },
      { key: "10", field: "efficiency_10", label: "10", color: "#9c711f" },
      { key: "20", field: "efficiency_20", label: "20", color: "#6b5b95" },
      { key: "55", field: "efficiency_55", label: "55", color: "#2f6f8f" },
    ],
  },
  {
    key: "direction",
    payloadKey: "direction",
    elementId: "direction",
    shortTitle: "D",
    title: "方向",
    unitBounded: false,
    legendId: "direction-legend",
    defaultSeries: ["equal", "midlong"],
    series: [
      { key: "equal", field: "direction_equal", label: "等权", color: "#087f6f" },
      { key: "midlong", field: "direction_midlong", label: "中长期", color: "#576db5" },
      { key: "5", field: "direction_5", label: "5", color: "#c4573d" },
      { key: "10", field: "direction_10", label: "10", color: "#9c711f" },
      { key: "20", field: "direction_20", label: "20", color: "#6b5b95" },
      { key: "55", field: "direction_55", label: "55", color: "#2f6f8f" },
    ],
  },
  {
    key: "twoSidedness",
    payloadKey: "twoSidedness",
    elementId: "two-sidedness",
    shortTitle: "B",
    title: "双向性",
    unitBounded: true,
    legendId: "two-sidedness-legend",
    defaultSeries: ["equal", "midlong"],
    series: [
      { key: "equal", field: "two_sidedness_equal", label: "等权", color: "#087f6f" },
      { key: "midlong", field: "two_sidedness_midlong", label: "中长期", color: "#576db5" },
      { key: "5", field: "two_sidedness_5", label: "5", color: "#c4573d" },
      { key: "10", field: "two_sidedness_10", label: "10", color: "#9c711f" },
      { key: "20", field: "two_sidedness_20", label: "20", color: "#6b5b95" },
      { key: "55", field: "two_sidedness_55", label: "55", color: "#2f6f8f" },
    ],
  },
  {
    key: "expansion",
    payloadKey: "expansion",
    elementId: "expansion",
    shortTitle: "X",
    title: "当日扩张",
    unitBounded: false,
    legendId: "expansion-legend",
    defaultSeries: ["range", "close", "gap"],
    series: [
      { key: "range", field: "range", label: "范围", color: "#087f6f" },
      { key: "close", field: "close", label: "收盘", color: "#576db5" },
      { key: "gap", field: "gap", label: "Gap", color: "#c4573d" },
    ],
  },
  {
    key: "clearance",
    payloadKey: "expansion",
    elementId: "clearance",
    shortTitle: "CL",
    title: "旧区位置",
    unitBounded: true,
    legendId: "clearance-legend",
    defaultSeries: ["up20", "down20", "up55", "down55"],
    series: [
      { key: "up20", field: "clearance_up_20", label: "上 20", color: "#087f6f" },
      { key: "down20", field: "clearance_down_20", label: "下 20", color: "#c4573d" },
      { key: "up55", field: "clearance_up_55", label: "上 55", color: "#2f6f8f" },
      { key: "down55", field: "clearance_down_55", label: "下 55", color: "#9c711f" },
    ],
  },
];
const PRICE_SCALE_MODES = { log: PriceScaleMode.Logarithmic, linear: PriceScaleMode.Normal };

const dom = {
  asOfInput: document.querySelector("#as-of-input"),
  asOfReadout: document.querySelector("#as-of-readout"),
  chartBoard: document.querySelector(".chart-board"),
  datasetMeta: document.querySelector("#dataset-meta"),
  featureFrames: Object.fromEntries(
    FEATURES.map((feature) => [
      feature.key,
      document.querySelector(`#${feature.elementId}-frame`),
    ]),
  ),
  featureLegends: Object.fromEntries(
    FEATURES.map((feature) => [feature.key, document.querySelector(`#${feature.legendId}`)]),
  ),
  featureTimeMarkers: Object.fromEntries(
    FEATURES.map((feature) => [
      feature.key,
      document.querySelector(`#${feature.elementId}-time-marker`),
    ]),
  ),
  indicatorScaleButtons: [...document.querySelectorAll("[data-indicator-scale]")],
  loading: document.querySelector("#loading-state"),
  nextButton: document.querySelector("#next-button"),
  playButton: document.querySelector("#play-button"),
  previousButton: document.querySelector("#previous-button"),
  priceFrame: document.querySelector("#price-frame"),
  priceLegend: document.querySelector("#price-legend"),
  priceScaleButtons: [...document.querySelectorAll("[data-price-scale]")],
  priceTimeMarker: document.querySelector("#price-time-marker"),
  paneElements: {
    price: document.querySelector('[data-pane-key="price"]'),
    ...Object.fromEntries(
      FEATURES.map((feature) => [
        feature.key,
        document.querySelector(`[data-pane-key="${feature.key}"]`),
      ]),
    ),
  },
  paneResizers: Object.fromEntries(
    FEATURES.map((feature) => [
      feature.key,
      document.querySelector(`[data-pane-resizer="${feature.key}"]`),
    ]),
  ),
  paneToggleButtons: [...document.querySelectorAll("[data-pane-toggle]")],
  rangeButtons: [...document.querySelectorAll("[data-range-bars]")],
  snapshotPanel: document.querySelector("#snapshot-panel"),
  snapshotToggle: document.querySelector("#snapshot-toggle"),
  snapshotValues: document.querySelector("#snapshot-values"),
  workspace: document.querySelector(".workspace"),
};

const state = {
  asOfIndex: -1,
  chartInteraction: false,
  charts: null,
  crosshairSource: null,
  crosshairTime: null,
  currentIndicatorScale: "auto",
  currentPriceScale: "log",
  currentRangeBars: DEFAULT_RANGE_BARS,
  paneWeights: { ...DEFAULT_PANE_WEIGHTS },
  payload: null,
  playbackTimer: null,
  snapshotOpen: true,
  visiblePanes: Object.fromEntries(FEATURES.map((feature) => [feature.key, true])),
  visibleSeries: Object.fromEntries(
    FEATURES.map((feature) => [feature.key, new Set(feature.defaultSeries)]),
  ),
};

function chartOptions({ timeVisible = false, fontSize = 11 } = {}) {
  return {
    autoSize: true,
    layout: {
      background: { type: ColorType.Solid, color: "#fbfcfb" },
      textColor: "#65716f",
      fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
      fontSize,
      attributionLogo: false,
    },
    grid: {
      vertLines: { color: "#eef2f0" },
      horzLines: { color: "#eef2f0" },
    },
    rightPriceScale: {
      borderColor: "#d5ddd9",
      minimumWidth: 58,
      entireTextOnly: true,
    },
    timeScale: {
      borderColor: "#d5ddd9",
      rightOffset: 4,
      visible: timeVisible,
      timeVisible: false,
      secondsVisible: false,
    },
    crosshair: { mode: CrosshairMode.Normal },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
    handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
  };
}

function createUnitBounds(chart) {
  return chart.addSeries(LineSeries, {
    color: "rgba(0, 0, 0, 0)",
    lineWidth: 1,
    lastValueVisible: false,
    priceLineVisible: false,
    crosshairMarkerVisible: false,
  });
}

function createTimeAnchor(chart) {
  const series = chart.addSeries(LineSeries, {
    priceScaleId: "time-anchor",
    color: "rgba(0, 0, 0, 0)",
    lineWidth: 1,
    lastValueVisible: false,
    priceLineVisible: false,
    crosshairMarkerVisible: false,
  });
  chart.priceScale("time-anchor").applyOptions({ visible: false });
  return series;
}

function createFeatureSeries(chart, feature) {
  return Object.fromEntries(
    feature.series.map((seriesConfig) => [
      seriesConfig.key,
      chart.addSeries(LineSeries, {
        color: seriesConfig.color,
        lineWidth: feature.defaultSeries.includes(seriesConfig.key) ? 2 : 1,
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
      }),
    ]),
  );
}

function createCharts() {
  const price = createChart(document.querySelector("#price-chart"), {
    ...chartOptions({ timeVisible: false, fontSize: 11 }),
    rightPriceScale: {
      borderColor: "#d5ddd9",
      minimumWidth: 58,
      entireTextOnly: true,
      mode: PRICE_SCALE_MODES[state.currentPriceScale],
    },
  });
  const featureCharts = Object.fromEntries(
    FEATURES.map((feature, index) => {
      const isBottom = index === FEATURES.length - 1;
      return [
        feature.key,
        createChart(
          document.querySelector(`#${feature.elementId}-chart`),
          chartOptions({ timeVisible: isBottom, fontSize: 10 }),
        ),
      ];
    }),
  );

  const charts = {
    featureBounds: Object.fromEntries(
      FEATURES.map((feature) => [feature.key, createUnitBounds(featureCharts[feature.key])]),
    ),
    featureSeries: Object.fromEntries(
      FEATURES.map((feature) => [
        feature.key,
        createFeatureSeries(featureCharts[feature.key], feature),
      ]),
    ),
    featureTimeAnchors: Object.fromEntries(
      FEATURES.map((feature) => [feature.key, createTimeAnchor(featureCharts[feature.key])]),
    ),
    features: featureCharts,
    price,
    priceSeries: price.addSeries(CandlestickSeries, {
      upColor: "#087f6f",
      downColor: "#c4573d",
      borderUpColor: "#087f6f",
      borderDownColor: "#c4573d",
      wickUpColor: "#087f6f",
      wickDownColor: "#c4573d",
    }),
  };

  syncTimeScales([price, ...Object.values(featureCharts)]);
  subscribeCrosshairs(charts);
  bindChartFrames();
  return charts;
}

function visibleFeatureKeys() {
  return FEATURES.filter((feature) => state.visiblePanes[feature.key]).map((feature) => feature.key);
}

function previousVisiblePaneKey(targetKey) {
  const keys = ["price", ...FEATURES.map((feature) => feature.key)];
  const targetIndex = keys.indexOf(targetKey);
  for (let index = targetIndex - 1; index >= 0; index -= 1) {
    const key = keys[index];
    if (key === "price" || state.visiblePanes[key]) return key;
  }
  return null;
}

function minimumPaneHeight(key) {
  return key === "price" ? MIN_PRICE_PANE_HEIGHT : MIN_FEATURE_PANE_HEIGHT;
}

function scheduleChartResize() {
  window.requestAnimationFrame(() => {
    window.dispatchEvent(new Event("resize"));
    scheduleMarkers();
  });
}

function applyPaneLayout() {
  const visibleKeys = visibleFeatureKeys();
  const lastVisibleKey = visibleKeys.at(-1) ?? "price";

  for (const key of ["price", ...FEATURES.map((feature) => feature.key)]) {
    dom.paneElements[key].classList.toggle("is-last-visible", key === lastVisibleKey);
  }
  for (const feature of FEATURES) {
    const visible = state.visiblePanes[feature.key];
    const pane = dom.paneElements[feature.key];
    pane.hidden = !visible;
    pane.style.setProperty("--pane-weight", String(state.paneWeights[feature.key]));
    dom.paneResizers[feature.key].hidden = !visible;
    const toggle = dom.paneToggleButtons.find((button) => button.dataset.paneToggle === feature.key);
    toggle?.setAttribute("aria-pressed", String(visible));
    state.charts?.features[feature.key].applyOptions({
      timeScale: { visible: feature.key === lastVisibleKey },
    });
  }
  dom.paneElements.price.style.setProperty("--pane-weight", String(state.paneWeights.price));
  state.charts?.price.applyOptions({ timeScale: { visible: visibleKeys.length === 0 } });
  scheduleChartResize();
}

function setPaneVisible(key, visible) {
  state.visiblePanes[key] = visible;
  applyPaneLayout();
}

function resizePane(targetKey, requestedDelta) {
  const previousKey = previousVisiblePaneKey(targetKey);
  if (!previousKey) return false;

  const target = dom.paneElements[targetKey];
  const previous = dom.paneElements[previousKey];
  const targetHeight = target.getBoundingClientRect().height;
  const previousHeight = previous.getBoundingClientRect().height;
  const minDelta = minimumPaneHeight(previousKey) - previousHeight;
  const maxDelta = targetHeight - minimumPaneHeight(targetKey);
  const delta = Math.max(minDelta, Math.min(maxDelta, requestedDelta));
  if (delta === 0) return false;

  const pairWeight = state.paneWeights[previousKey] + state.paneWeights[targetKey];
  const pairHeight = previousHeight + targetHeight;
  state.paneWeights[targetKey] = pairWeight * ((targetHeight - delta) / pairHeight);
  state.paneWeights[previousKey] = pairWeight - state.paneWeights[targetKey];
  target.style.setProperty("--pane-weight", String(state.paneWeights[targetKey]));
  previous.style.setProperty("--pane-weight", String(state.paneWeights[previousKey]));
  dom.paneResizers[targetKey].setAttribute(
    "aria-valuenow",
    String(Math.round(targetHeight - delta)),
  );
  scheduleChartResize();
  return true;
}

function bindPaneResizers() {
  for (const feature of FEATURES) {
    const resizer = dom.paneResizers[feature.key];
    resizer.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || window.matchMedia("(max-width: 980px)").matches) return;
      event.preventDefault();
      let lastY = event.clientY;
      resizer.setPointerCapture(event.pointerId);
      dom.chartBoard.classList.add("is-resizing");
      const onMove = (moveEvent) => {
        resizePane(feature.key, moveEvent.clientY - lastY);
        lastY = moveEvent.clientY;
      };
      const onEnd = () => {
        dom.chartBoard.classList.remove("is-resizing");
        resizer.removeEventListener("pointermove", onMove);
        resizer.removeEventListener("pointerup", onEnd);
        resizer.removeEventListener("pointercancel", onEnd);
      };
      resizer.addEventListener("pointermove", onMove);
      resizer.addEventListener("pointerup", onEnd);
      resizer.addEventListener("pointercancel", onEnd);
    });
    resizer.addEventListener("keydown", (event) => {
      if (window.matchMedia("(max-width: 980px)").matches) return;
      if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
      event.preventDefault();
      resizePane(feature.key, event.key === "ArrowDown" ? 16 : -16);
    });
  }
}

function syncTimeScales(charts) {
  let syncing = false;
  for (const source of charts) {
    source.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (!range || syncing) return;
      state.chartInteraction = true;
      state.crosshairTime = null;
      state.crosshairSource = null;
      syncing = true;
      for (const target of charts) {
        if (target !== source) target.timeScale().setVisibleLogicalRange(range);
      }
      syncing = false;
      scheduleMarkers();
    });
  }
}

function subscribeCrosshairs(charts) {
  const pairs = [
    ["price", charts.price],
    ...FEATURES.map((feature) => [feature.key, charts.features[feature.key]]),
  ];
  for (const [source, chart] of pairs) {
    chart.subscribeCrosshairMove((param) => {
      const time = timeKey(param.time);
      if (time && time <= asOfDate()) {
        state.chartInteraction = false;
        state.crosshairTime = time;
        state.crosshairSource = source;
        renderReadouts(time);
      } else if (state.crosshairSource === source) {
        clearCrosshair();
      }
      scheduleMarkers();
    });
  }
}

function bindChartFrames() {
  const pairs = [
    ["price", dom.priceFrame],
    ...FEATURES.map((feature) => [feature.key, dom.featureFrames[feature.key]]),
  ];
  for (const [source, frame] of pairs) {
    frame.addEventListener("pointerdown", () => {
      state.chartInteraction = true;
      state.crosshairTime = null;
      state.crosshairSource = null;
      scheduleMarkers();
    });
    frame.addEventListener("pointerleave", () => {
      if (state.crosshairSource === source) clearCrosshair();
    });
  }
}

function clearCrosshair() {
  state.crosshairTime = null;
  state.crosshairSource = null;
  renderReadouts(asOfDate());
  scheduleMarkers();
}

function timeKey(time) {
  if (!time) return null;
  if (typeof time === "string") return time;
  if (typeof time === "number") return new Date(time * 1000).toISOString().slice(0, 10);
  return `${time.year}-${String(time.month).padStart(2, "0")}-${String(time.day).padStart(2, "0")}`;
}

function asOfDate() {
  return state.payload.bars[state.asOfIndex].time;
}

function endIndex(records, date) {
  let low = 0;
  let high = records.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (records[middle].time <= date) low = middle + 1;
    else high = middle;
  }
  return low;
}

function recordsThrough(records, date) {
  return records.slice(0, endIndex(records, date));
}

function lastRecordThrough(records, date) {
  const index = endIndex(records, date) - 1;
  return index >= 0 ? records[index] : null;
}

function fieldName(feature, seriesKey) {
  return feature.series.find((seriesConfig) => seriesConfig.key === seriesKey)?.field;
}

function formatDecimal(value, digits = 3) {
  return value === null || value === undefined || Number.isNaN(Number(value))
    ? "—"
    : Number(value).toFixed(digits);
}

function formatPrice(value) {
  return value === null || value === undefined || Number.isNaN(Number(value))
    ? "—"
    : Number(value).toFixed(2);
}

function formatPercent(value, digits = 2) {
  return value === null || value === undefined || Number.isNaN(Number(value))
    ? "—"
    : `${(Number(value) * 100).toFixed(digits)}%`;
}

function percentChange(current, previous) {
  if (
    current === null ||
    current === undefined ||
    previous === null ||
    previous === undefined ||
    Number(previous) === 0
  ) {
    return null;
  }
  return ((Number(current) - Number(previous)) / Number(previous)) * 100;
}

function updateReplay({ resetRange = false } = {}) {
  const date = asOfDate();
  const { charts, payload } = state;
  const wasFollowingEnd = isFollowingAsOf();
  const visibleBars = payload.bars.slice(0, state.asOfIndex + 1);
  charts.priceSeries.setData(visibleBars);

  for (const feature of FEATURES) {
    charts.featureTimeAnchors[feature.key].setData(
      visibleBars.map((bar) => ({ time: bar.time, value: 0 })),
    );
    const records = recordsThrough(payload[feature.payloadKey], date);
    for (const seriesConfig of feature.series) {
      const series = charts.featureSeries[feature.key][seriesConfig.key];
      series.setData(
        state.visibleSeries[feature.key].has(seriesConfig.key)
          ? records
              .map((row) => ({ time: row.time, value: row[seriesConfig.field] }))
              .filter((row) => row.value !== null && row.value !== undefined)
          : [],
      );
    }
    updateIndicatorBounds(charts.featureBounds[feature.key], records, date, feature.unitBounded);
  }

  dom.asOfInput.value = date;
  dom.asOfReadout.textContent = date;
  dom.previousButton.disabled = state.asOfIndex === 0;
  dom.nextButton.disabled = state.asOfIndex === payload.bars.length - 1;
  renderReadouts(state.crosshairTime && state.crosshairTime <= date ? state.crosshairTime : date);
  scheduleMarkers();
  if (resetRange || wasFollowingEnd) setDefaultVisibleRange();
}

function updateIndicatorBounds(series, records, fallbackTime, unitBounded) {
  if (!unitBounded || state.currentIndicatorScale === "auto") {
    series.setData([]);
    return;
  }
  const first = records[0]?.time ?? fallbackTime;
  const last = records.at(-1)?.time ?? fallbackTime;
  series.setData([
    { time: first, value: 0 },
    { time: last, value: 1 },
  ]);
}

function isFollowingAsOf() {
  const range = state.charts?.price.timeScale().getVisibleLogicalRange();
  return !range || range.to >= state.asOfIndex - 0.25;
}

function setDefaultVisibleRange() {
  const range = {
    from: Math.max(0, state.asOfIndex - state.currentRangeBars),
    to: state.asOfIndex,
  };
  for (const chart of [state.charts.price, ...Object.values(state.charts.features)]) {
    chart.timeScale().setVisibleLogicalRange(range);
  }
}

function renderReadouts(time) {
  const bar = lastRecordThrough(state.payload.bars, time);
  const previousBar = bar ? lastRecordThrough(state.payload.bars, previousDate(bar.time)) : null;
  const efficiency = lastRecordThrough(state.payload.efficiency, time);
  const direction = lastRecordThrough(state.payload.direction, time);
  const twoSidedness = lastRecordThrough(state.payload.twoSidedness, time);
  const expansion = lastRecordThrough(state.payload.expansion, time);

  renderPriceLegend(bar, previousBar, time);
  for (const feature of FEATURES) {
    renderFeatureLegend(feature, lastRecordThrough(state.payload[feature.payloadKey], time));
  }
  renderSnapshot(time, bar, previousBar, efficiency, direction, twoSidedness, expansion);
}

function previousDate(date) {
  const index = endIndex(state.payload.bars, date) - 2;
  return index >= 0 ? state.payload.bars[index].time : null;
}

function renderPriceLegend(bar, previousBar, time) {
  const change = bar ? percentChange(bar.close, previousBar?.close) : null;
  const changeText =
    change === null ? "—" : `${change >= 0 ? "+" : ""}${change.toFixed(2)}%`;
  const changeClass =
    change === null ? "" : change >= 0 ? "is-up" : "is-down";

  dom.priceLegend.innerHTML = `
    <span class="legend-title">SPX</span>
    <span class="legend-metric"><span class="legend-label">日</span><span class="legend-value">${time}</span></span>
    <span class="legend-metric"><span class="legend-label">O</span><span class="legend-value">${formatPrice(bar?.open)}</span></span>
    <span class="legend-metric"><span class="legend-label">H</span><span class="legend-value">${formatPrice(bar?.high)}</span></span>
    <span class="legend-metric"><span class="legend-label">L</span><span class="legend-value">${formatPrice(bar?.low)}</span></span>
    <span class="legend-metric"><span class="legend-label">C</span><span class="legend-value">${formatPrice(bar?.close)}</span></span>
    <span class="legend-metric legend-change ${changeClass}">${changeText}</span>
  `;
}

function renderFeatureLegend(feature, row) {
  const root = dom.featureLegends[feature.key];
  const parts = [
    `<span class="legend-title">${feature.shortTitle}</span>`,
    `<span class="legend-label">${feature.title}</span>`,
  ];
  for (const seriesConfig of feature.series) {
    const on = state.visibleSeries[feature.key].has(seriesConfig.key);
    const value = row ? row[seriesConfig.field] : null;
    parts.push(`
      <button
        type="button"
        class="legend-item ${on ? "" : "is-off"}"
        data-feature="${feature.key}"
        data-series="${seriesConfig.key}"
        title="切换 ${seriesConfig.label}"
      >
        <span class="legend-swatch" style="background:${seriesConfig.color}"></span>
        <span class="legend-label">${seriesConfig.label}</span>
        <span class="legend-value">${on ? formatDecimal(value) : "—"}</span>
      </button>
    `);
  }
  root.innerHTML = parts.join("");
}

function renderSnapshot(time, bar, previousBar, efficiency, direction, twoSidedness, expansion) {
  const change = bar ? percentChange(bar.close, previousBar?.close) : null;
  const values = bar
    ? [
        ["观察日", time, true],
        ["O", formatPrice(bar.open)],
        ["H", formatPrice(bar.high)],
        ["L", formatPrice(bar.low)],
        ["C", formatPrice(bar.close), true],
        [
          "涨跌",
          change === null ? "—" : `${change >= 0 ? "+" : ""}${change.toFixed(2)}%`,
        ],
      ]
    : [["观察日", time, true]];

  if (expansion) {
    values.push(
      ["活动背景", formatPercent(expansion.activity_level)],
      ["X 范围", formatDecimal(expansion.range), true],
      ["X 收盘", formatDecimal(expansion.close)],
      ["X Gap", formatDecimal(expansion.gap)],
      ["X 转化", formatDecimal(expansion.share)],
      ["清除 上20", formatDecimal(expansion.clearance_up_20)],
      ["清除 下20", formatDecimal(expansion.clearance_down_20)],
      ["清除 上55", formatDecimal(expansion.clearance_up_55)],
      ["清除 下55", formatDecimal(expansion.clearance_down_55)],
    );
  }

  if (efficiency && direction && twoSidedness) {
    values.push(
      ["E 等权", formatDecimal(efficiency.efficiency_equal), true],
      ["E 中长期", formatDecimal(efficiency.efficiency_midlong)],
      ["D 等权", formatDecimal(direction.direction_equal), true],
      ["D 中长期", formatDecimal(direction.direction_midlong)],
      ["D 一致·等权", formatDecimal(direction.direction_agreement_equal)],
      ["B 等权", formatDecimal(twoSidedness.two_sidedness_equal), true],
      ["B 中长期", formatDecimal(twoSidedness.two_sidedness_midlong)],
    );
  }

  dom.snapshotValues.replaceChildren(
    ...values.map(([label, value, primary]) => {
      const item = document.createElement("div");
      const term = document.createElement("dt");
      const definition = document.createElement("dd");
      term.textContent = label;
      definition.textContent = value;
      if (primary) item.classList.add("primary");
      item.append(term, definition);
      return item;
    }),
  );
}

function scheduleMarkers() {
  window.requestAnimationFrame(renderSynchronizedTimeMarkers);
}

function renderSynchronizedTimeMarkers() {
  const targets = [
    ["price", state.charts?.price, dom.priceTimeMarker],
    ...FEATURES.map((feature) => [
      feature.key,
      state.charts?.features[feature.key],
      dom.featureTimeMarkers[feature.key],
    ]),
  ];
  for (const [name, chart, marker] of targets) {
    const coordinate =
      state.crosshairTime && !state.chartInteraction && state.crosshairSource !== name && chart
        ? chart.timeScale().timeToCoordinate(state.crosshairTime)
        : null;
    marker.classList.toggle("is-visible", coordinate !== null);
    if (coordinate !== null) marker.style.transform = `translateX(${coordinate}px)`;
  }
}

function setAsOfIndex(index, { resetRange = false } = {}) {
  state.asOfIndex = Math.max(0, Math.min(index, state.payload.bars.length - 1));
  updateReplay({ resetRange });
}

function setPriceScale(scale) {
  state.currentPriceScale = scale;
  state.charts.price.applyOptions({ rightPriceScale: { mode: PRICE_SCALE_MODES[scale] } });
  for (const button of dom.priceScaleButtons) {
    button.setAttribute("aria-pressed", String(button.dataset.priceScale === scale));
  }
}

function setIndicatorScale(scale) {
  state.currentIndicatorScale = scale;
  for (const button of dom.indicatorScaleButtons) {
    button.setAttribute("aria-pressed", String(button.dataset.indicatorScale === scale));
  }
  updateReplay();
}

function setSnapshotOpen(open) {
  state.snapshotOpen = open;
  dom.workspace.classList.toggle("snapshot-collapsed", !open);
  dom.snapshotToggle.setAttribute("aria-pressed", String(open));
  dom.snapshotToggle.title = open ? "折叠快照栏" : "展开快照栏";
  dom.snapshotToggle.setAttribute("aria-label", open ? "折叠快照栏" : "展开快照栏");
  // Charts need a resize pass after the grid width changes.
  window.requestAnimationFrame(() => {
    window.dispatchEvent(new Event("resize"));
  });
}

function setPlaying(playing) {
  if (!playing) {
    window.clearInterval(state.playbackTimer);
    state.playbackTimer = null;
    dom.playButton.innerHTML = '<i data-lucide="play" aria-hidden="true"></i>';
    dom.playButton.title = "播放回放（空格键）";
    dom.playButton.setAttribute("aria-label", "播放回放");
    createIcons({ icons });
    return;
  }
  dom.playButton.innerHTML = '<i data-lucide="pause" aria-hidden="true"></i>';
  dom.playButton.title = "暂停回放（空格键）";
  dom.playButton.setAttribute("aria-label", "暂停回放");
  createIcons({ icons });
  state.playbackTimer = window.setInterval(() => {
    if (state.asOfIndex >= state.payload.bars.length - 1) {
      setPlaying(false);
      return;
    }
    setAsOfIndex(state.asOfIndex + 1);
  }, PLAYBACK_INTERVAL_MS);
}

function toggleSeries(featureKey, seriesKey) {
  const visible = state.visibleSeries[featureKey];
  if (visible.has(seriesKey)) {
    if (visible.size === 1) return;
    visible.delete(seriesKey);
  } else {
    visible.add(seriesKey);
  }
  updateReplay();
}

function bindControls() {
  dom.previousButton.addEventListener("click", () => setAsOfIndex(state.asOfIndex - 1));
  dom.nextButton.addEventListener("click", () => setAsOfIndex(state.asOfIndex + 1));
  dom.playButton.addEventListener("click", () => setPlaying(!state.playbackTimer));
  dom.asOfInput.addEventListener("change", () => {
    const index = endIndex(state.payload.bars, dom.asOfInput.value) - 1;
    setAsOfIndex(index < 0 ? 0 : index);
  });
  for (const button of dom.priceScaleButtons) {
    button.addEventListener("click", () => setPriceScale(button.dataset.priceScale));
  }
  for (const button of dom.indicatorScaleButtons) {
    button.addEventListener("click", () => setIndicatorScale(button.dataset.indicatorScale));
  }
  for (const button of dom.paneToggleButtons) {
    button.addEventListener("click", () => {
      const key = button.dataset.paneToggle;
      setPaneVisible(key, !state.visiblePanes[key]);
    });
  }
  for (const button of dom.rangeButtons) {
    button.addEventListener("click", () => {
      state.currentRangeBars = Number(button.dataset.rangeBars);
      for (const candidate of dom.rangeButtons) {
        candidate.setAttribute("aria-pressed", String(candidate === button));
      }
      setDefaultVisibleRange();
    });
  }
  dom.snapshotToggle.addEventListener("click", () => setSnapshotOpen(!state.snapshotOpen));
  for (const feature of FEATURES) {
    dom.featureLegends[feature.key].addEventListener("click", (event) => {
      const button = event.target.closest("[data-series]");
      if (!button) return;
      toggleSeries(button.dataset.feature, button.dataset.series);
    });
  }
  window.addEventListener("keydown", (event) => {
    if (
      event.defaultPrevented ||
      event.target.closest("input, button, select, textarea, [contenteditable='true']")
    ) {
      return;
    }
    if (event.code === "Space") {
      event.preventDefault();
      setPlaying(!state.playbackTimer);
      return;
    }
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      const direction = event.key === "ArrowLeft" ? -1 : 1;
      setAsOfIndex(state.asOfIndex + direction * (event.shiftKey ? 21 : 1));
    }
  });
}

function setLoading(message, error = false) {
  dom.loading.textContent = message;
  dom.loading.classList.toggle("error", error);
  dom.loading.classList.remove("ready");
}

async function boot() {
  try {
    const response = await fetch(API_URL, { cache: "no-store" });
    if (!response.ok) throw new Error(`数据请求失败 (${response.status})`);
    state.payload = await response.json();
    if (state.payload.schemaVersion !== "phase1a_continuous_replay_v3") {
      throw new Error("回放数据版本不受当前页面支持");
    }
    state.charts = createCharts();
    applyPaneLayout();
    state.asOfIndex = state.payload.bars.length - 1;
    dom.asOfInput.min = state.payload.meta.researchStart;
    dom.asOfInput.max = state.payload.meta.researchEnd;
    dom.datasetMeta.textContent = `${state.payload.meta.snapshotId} · ${state.payload.meta.researchStart} - ${state.payload.meta.researchEnd}`;
    bindControls();
    bindPaneResizers();
    createIcons({ icons });
    setSnapshotOpen(true);
    updateReplay({ resetRange: true });
    const observer = new ResizeObserver(scheduleMarkers);
    observer.observe(dom.priceFrame);
    for (const frame of Object.values(dom.featureFrames)) observer.observe(frame);
    dom.loading.classList.add("ready");
  } catch (error) {
    setLoading(error instanceof Error ? error.message : "图表加载失败", true);
    console.error(error);
  }
}

boot();
