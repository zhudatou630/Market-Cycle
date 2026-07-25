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
const EFFICIENCY_COLORS = {
  5: "#c4573d",
  10: "#087f6f",
  20: "#576db5",
  55: "#9c711f",
};
const PRICE_SCALE_MODES = { log: PriceScaleMode.Logarithmic, linear: PriceScaleMode.Normal };

const dom = {
  asOfInput: document.querySelector("#as-of-input"),
  asOfReadout: document.querySelector("#as-of-readout"),
  datasetMeta: document.querySelector("#dataset-meta"),
  efficiencyChecks: [...document.querySelectorAll("[data-window]")],
  efficiencyFrame: document.querySelector("#efficiency-frame"),
  efficiencyTimeMarker: document.querySelector("#efficiency-time-marker"),
  indicatorScaleButtons: [...document.querySelectorAll("[data-indicator-scale]")],
  loading: document.querySelector("#loading-state"),
  nextButton: document.querySelector("#next-button"),
  playButton: document.querySelector("#play-button"),
  previousButton: document.querySelector("#previous-button"),
  priceFrame: document.querySelector("#price-frame"),
  priceScaleButtons: [...document.querySelectorAll("[data-price-scale]")],
  priceTimeMarker: document.querySelector("#price-time-marker"),
  rangeButtons: [...document.querySelectorAll("[data-range-bars]")],
  snapshotValues: document.querySelector("#snapshot-values"),
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
  visibleEfficiency: new Set([5, 10, 20, 55]),
};

function chartOptions() {
  return {
    autoSize: true,
    layout: {
      background: { type: ColorType.Solid, color: "#fbfcfb" },
      textColor: "#65716f",
      fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
      fontSize: 11,
    },
    grid: {
      vertLines: { color: "#edf0ee" },
      horzLines: { color: "#edf0ee" },
    },
    rightPriceScale: { borderColor: "#cfd7d3", minimumWidth: 86 },
    timeScale: { borderColor: "#cfd7d3", rightOffset: 0 },
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

function createCharts() {
  const price = createChart(document.querySelector("#price-chart"), {
    ...chartOptions(),
    rightPriceScale: {
      borderColor: "#cfd7d3",
      minimumWidth: 86,
      mode: PRICE_SCALE_MODES[state.currentPriceScale],
    },
  });
  const efficiency = createChart(document.querySelector("#efficiency-chart"), chartOptions());

  const priceSeries = price.addSeries(CandlestickSeries, {
    upColor: "#087f6f",
    downColor: "#c4573d",
    borderUpColor: "#087f6f",
    borderDownColor: "#c4573d",
    wickUpColor: "#087f6f",
    wickDownColor: "#c4573d",
  });
  const efficiencySeries = Object.fromEntries(
    Object.entries(EFFICIENCY_COLORS).map(([window, color]) => [
      window,
      efficiency.addSeries(LineSeries, {
        color,
        lineWidth: 2,
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
      }),
    ]),
  );

  const charts = {
    efficiency,
    efficiencyBounds: createUnitBounds(efficiency),
    efficiencySeries,
    efficiencyTimeAnchor: createTimeAnchor(efficiency),
    price,
    priceSeries,
  };
  syncTimeScales([price, efficiency]);
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
  for (const [source, chart] of [["price", charts.price], ["efficiency", charts.efficiency]]) {
    chart.subscribeCrosshairMove((param) => {
      const time = timeKey(param.time);
      if (time && time <= asOfDate()) {
        state.chartInteraction = false;
        state.crosshairTime = time;
        state.crosshairSource = source;
        renderSnapshot(time);
      } else if (state.crosshairSource === source) {
        clearCrosshair();
      }
      scheduleMarkers();
    });
  }
}

function bindChartFrames() {
  for (const [source, frame] of [["price", dom.priceFrame], ["efficiency", dom.efficiencyFrame]]) {
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
  renderSnapshot(asOfDate());
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

function updateReplay({ resetRange = false } = {}) {
  const date = asOfDate();
  const { charts, payload } = state;
  const wasFollowingEnd = isFollowingAsOf();
  const visibleBars = payload.bars.slice(0, state.asOfIndex + 1);
  charts.priceSeries.setData(visibleBars);
  charts.efficiencyTimeAnchor.setData(visibleBars.map((bar) => ({ time: bar.time, value: 0 })));

  const efficiency = recordsThrough(payload.efficiency, date);
  for (const [window, series] of Object.entries(charts.efficiencySeries)) {
    const windowNumber = Number(window);
    series.setData(
      state.visibleEfficiency.has(windowNumber)
        ? efficiency
            .map((row) => ({ time: row.time, value: row[`efficiency_${window}`] }))
            .filter((row) => row.value !== null)
        : [],
    );
  }
  updateIndicatorBounds(charts.efficiencyBounds, efficiency, date);

  dom.asOfInput.value = date;
  dom.asOfReadout.textContent = `AS OF ${date}`;
  dom.previousButton.disabled = state.asOfIndex === 0;
  dom.nextButton.disabled = state.asOfIndex === payload.bars.length - 1;
  renderSnapshot(date);
  scheduleMarkers();
  if (resetRange || wasFollowingEnd) setDefaultVisibleRange();
}

function updateIndicatorBounds(series, records, fallbackTime) {
  if (state.currentIndicatorScale === "auto") {
    series.setData([]);
    return;
  }
  const first = records[0]?.time ?? fallbackTime;
  const last = records.at(-1)?.time ?? fallbackTime;
  series.setData([{ time: first, value: 0 }, { time: last, value: 1 }]);
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
  for (const chart of [state.charts.price, state.charts.efficiency]) {
    chart.timeScale().setVisibleLogicalRange(range);
  }
}

function renderSnapshot(time) {
  const row = lastRecordThrough(state.payload.efficiency, time);
  const values = row
    ? [
        ["观察日", time, true],
        ["效率 5", formatDecimal(row.efficiency_5)],
        ["效率 10", formatDecimal(row.efficiency_10)],
        ["效率 20", formatDecimal(row.efficiency_20), true],
        ["效率 55", formatDecimal(row.efficiency_55), true],
      ]
    : [["观察日", time, true], ["效率状态", "窗口尚未齐套"]];
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

function formatDecimal(value) {
  return value === null || value === undefined ? "-" : Number(value).toFixed(3);
}

function scheduleMarkers() {
  window.requestAnimationFrame(renderSynchronizedTimeMarkers);
}

function renderSynchronizedTimeMarkers() {
  const targets = [
    ["price", state.charts?.price, dom.priceTimeMarker],
    ["efficiency", state.charts?.efficiency, dom.efficiencyTimeMarker],
  ];
  for (const [name, chart, marker] of targets) {
    const coordinate = state.crosshairTime && !state.chartInteraction && state.crosshairSource !== name && chart
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
  for (const checkbox of dom.efficiencyChecks) {
    checkbox.addEventListener("change", () => {
      const window = Number(checkbox.dataset.window);
      if (checkbox.checked) state.visibleEfficiency.add(window);
      else state.visibleEfficiency.delete(window);
      updateReplay();
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
  window.addEventListener("keydown", (event) => {
    if (event.defaultPrevented || event.target.closest("input, button, select, textarea, [contenteditable='true']")) {
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
    if (state.payload.schemaVersion !== "phase1a_continuous_replay_v1") {
      throw new Error("回放数据版本不受当前页面支持");
    }
    state.charts = createCharts();
    state.asOfIndex = state.payload.bars.length - 1;
    dom.asOfInput.min = state.payload.meta.researchStart;
    dom.asOfInput.max = state.payload.meta.researchEnd;
    dom.datasetMeta.textContent = `${state.payload.meta.snapshotId} · ${state.payload.meta.researchStart} - ${state.payload.meta.researchEnd}`;
    bindControls();
    createIcons({ icons });
    updateReplay({ resetRange: true });
    const observer = new ResizeObserver(scheduleMarkers);
    observer.observe(dom.priceFrame);
    observer.observe(dom.efficiencyFrame);
    dom.loading.classList.add("ready");
  } catch (error) {
    setLoading(error instanceof Error ? error.message : "图表加载失败", true);
    console.error(error);
  }
}

boot();
