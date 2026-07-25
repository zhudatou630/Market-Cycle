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
const SERIES_COLORS = {
  equal: "#087f6f",
  midlong: "#576db5",
  5: "#c4573d",
  10: "#9c711f",
  20: "#6b5b95",
  55: "#2f6f8f",
};
const SERIES_LABELS = {
  equal: "等权",
  midlong: "中长期",
  5: "5",
  10: "10",
  20: "20",
  55: "55",
};
const FEATURES = [
  {
    key: "efficiency",
    payloadKey: "efficiency",
    fieldPrefix: "efficiency",
    shortTitle: "E",
    title: "效率",
    unitBounded: true,
    legendId: "efficiency-legend",
  },
  {
    key: "direction",
    payloadKey: "direction",
    fieldPrefix: "direction",
    shortTitle: "D",
    title: "方向",
    unitBounded: false,
    legendId: "direction-legend",
  },
  {
    key: "twoSidedness",
    payloadKey: "twoSidedness",
    fieldPrefix: "two_sidedness",
    shortTitle: "B",
    title: "双向性",
    unitBounded: true,
    legendId: "two-sidedness-legend",
  },
];
const SERIES_KEYS = ["equal", "midlong", "5", "10", "20", "55"];
const PRICE_SCALE_MODES = { log: PriceScaleMode.Logarithmic, linear: PriceScaleMode.Normal };

const dom = {
  asOfInput: document.querySelector("#as-of-input"),
  asOfReadout: document.querySelector("#as-of-readout"),
  datasetMeta: document.querySelector("#dataset-meta"),
  featureFrames: Object.fromEntries(
    FEATURES.map((feature) => [
      feature.key,
      document.querySelector(
        feature.key === "twoSidedness" ? "#two-sidedness-frame" : `#${feature.key}-frame`,
      ),
    ]),
  ),
  featureLegends: Object.fromEntries(
    FEATURES.map((feature) => [feature.key, document.querySelector(`#${feature.legendId}`)]),
  ),
  featureTimeMarkers: Object.fromEntries(
    FEATURES.map((feature) => [
      feature.key,
      document.querySelector(
        feature.key === "twoSidedness"
          ? "#two-sidedness-time-marker"
          : `#${feature.key}-time-marker`,
      ),
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
  payload: null,
  playbackTimer: null,
  snapshotOpen: true,
  visibleSeries: Object.fromEntries(
    FEATURES.map((feature) => [feature.key, new Set(["equal", "midlong"])]),
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

function createFeatureSeries(chart) {
  return Object.fromEntries(
    SERIES_KEYS.map((key) => [
      key,
      chart.addSeries(LineSeries, {
        color: SERIES_COLORS[key],
        lineWidth: key === "equal" || key === "midlong" ? 2 : 1,
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
      const elementId =
        feature.key === "twoSidedness" ? "two-sidedness-chart" : `${feature.key}-chart`;
      const isBottom = index === FEATURES.length - 1;
      return [
        feature.key,
        createChart(
          document.querySelector(`#${elementId}`),
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
      FEATURES.map((feature) => [feature.key, createFeatureSeries(featureCharts[feature.key])]),
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
  return `${feature.fieldPrefix}_${seriesKey}`;
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
    for (const seriesKey of SERIES_KEYS) {
      const series = charts.featureSeries[feature.key][seriesKey];
      series.setData(
        state.visibleSeries[feature.key].has(seriesKey)
          ? records
              .map((row) => ({ time: row.time, value: row[fieldName(feature, seriesKey)] }))
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

  renderPriceLegend(bar, previousBar, time);
  renderFeatureLegend(FEATURES[0], efficiency);
  renderFeatureLegend(FEATURES[1], direction);
  renderFeatureLegend(FEATURES[2], twoSidedness);
  renderSnapshot(time, bar, previousBar, efficiency, direction, twoSidedness);
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
  for (const seriesKey of SERIES_KEYS) {
    const on = state.visibleSeries[feature.key].has(seriesKey);
    const value = row ? row[fieldName(feature, seriesKey)] : null;
    parts.push(`
      <button
        type="button"
        class="legend-item ${on ? "" : "is-off"}"
        data-feature="${feature.key}"
        data-series="${seriesKey}"
        title="切换 ${SERIES_LABELS[seriesKey]}"
      >
        <span class="legend-swatch series-${seriesKey}"></span>
        <span class="legend-label">${SERIES_LABELS[seriesKey]}</span>
        <span class="legend-value">${on ? formatDecimal(value) : "—"}</span>
      </button>
    `);
  }
  root.innerHTML = parts.join("");
}

function renderSnapshot(time, bar, previousBar, efficiency, direction, twoSidedness) {
  const change = bar ? percentChange(bar.close, previousBar?.close) : null;
  const values =
    bar && efficiency && direction && twoSidedness
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
          ["E 等权", formatDecimal(efficiency.efficiency_equal), true],
          ["E 中长期", formatDecimal(efficiency.efficiency_midlong)],
          ["D 等权", formatDecimal(direction.direction_equal), true],
          ["D 中长期", formatDecimal(direction.direction_midlong)],
          ["D 一致·等权", formatDecimal(direction.direction_agreement_equal)],
          ["B 等权", formatDecimal(twoSidedness.two_sidedness_equal), true],
          ["B 中长期", formatDecimal(twoSidedness.two_sidedness_midlong)],
        ]
      : [
          ["观察日", time, true],
          ["状态", "窗口尚未齐套"],
        ];

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
    if (state.payload.schemaVersion !== "phase1a_continuous_replay_v2") {
      throw new Error("回放数据版本不受当前页面支持");
    }
    state.charts = createCharts();
    state.asOfIndex = state.payload.bars.length - 1;
    dom.asOfInput.min = state.payload.meta.researchStart;
    dom.asOfInput.max = state.payload.meta.researchEnd;
    dom.datasetMeta.textContent = `${state.payload.meta.snapshotId} · ${state.payload.meta.researchStart} - ${state.payload.meta.researchEnd}`;
    bindControls();
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
