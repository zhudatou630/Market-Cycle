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
const DIRECTION_COLORS = { up: "#087f6f", down: "#c4573d" };
const INVALIDATED_COLOR = "#6d6961";
const PRICE_SCALE_MODES = { log: PriceScaleMode.Logarithmic, linear: PriceScaleMode.Normal };

const dom = {
  asOfInput: document.querySelector("#as-of-input"),
  asOfReadout: document.querySelector("#as-of-readout"),
  candidateToggle: document.querySelector("#candidate-toggle"),
  datasetMeta: document.querySelector("#dataset-meta"),
  efficiencyChecks: [...document.querySelectorAll("[data-window]")],
  efficiencyTimeMarker: document.querySelector("#efficiency-time-marker"),
  loading: document.querySelector("#loading-state"),
  multiplierButtons: [...document.querySelectorAll("[data-multiplier]")],
  nextButton: document.querySelector("#next-button"),
  pathOverlay: document.querySelector("#path-overlay"),
  pathFrame: document.querySelector("#path-frame"),
  pathTimeMarker: document.querySelector("#path-time-marker"),
  playButton: document.querySelector("#play-button"),
  previousButton: document.querySelector("#previous-button"),
  indicatorScaleButtons: [...document.querySelectorAll("[data-indicator-scale]")],
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
  crosshairTime: null,
  crosshairSource: null,
  currentMultiplier: "2",
  currentPriceScale: "log",
  currentRangeBars: DEFAULT_RANGE_BARS,
  currentIndicatorScale: "auto",
  payload: null,
  playbackTimer: null,
  showCandidateLedger: false,
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
  const path = createChart(document.querySelector("#path-chart"), chartOptions());

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

  const pathEfficiency = path.addSeries(LineSeries, {
    color: "#354541",
    lineWidth: 2,
    lastValueVisible: false,
    priceLineVisible: false,
    crosshairMarkerVisible: false,
  });

  const charts = {
    efficiency,
    efficiencyBounds: createUnitBounds(efficiency),
    efficiencySeries,
    efficiencyTimeAnchor: createTimeAnchor(efficiency),
    path,
    pathBounds: createUnitBounds(path),
    pathEfficiency,
    pathTimeAnchor: createTimeAnchor(path),
    price,
    priceSeries,
  };
  syncTimeScales([price, efficiency, path]);
  subscribeSnapshotCrosshairs(charts);
  bindChartInteractionFrames();
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
      schedulePathOverlay();
    });
  }
}

function subscribeSnapshotCrosshairs(charts) {
  for (const [source, chart] of [
    ["price", charts.price],
    ["efficiency", charts.efficiency],
    ["path", charts.path],
  ]) {
    chart.subscribeCrosshairMove((param) => {
      const time = timeKey(param.time);
      if (time && time <= asOfDate()) {
        state.chartInteraction = false;
        state.crosshairTime = time;
        state.crosshairSource = source;
        renderSnapshotDetails(time);
      } else if (state.crosshairSource === source) {
        state.crosshairTime = null;
        state.crosshairSource = null;
        renderSnapshotDetails(asOfDate());
      }
      schedulePathOverlay();
    });
  }
}

function bindChartInteractionFrames() {
  for (const [source, frame] of [
    ["price", dom.priceFrame],
    ["efficiency", dom.efficiencyTimeMarker.parentElement],
    ["path", dom.pathFrame],
  ]) {
    frame.addEventListener("pointerdown", () => {
      state.chartInteraction = true;
      state.crosshairTime = null;
      state.crosshairSource = null;
      schedulePathOverlay();
    });
    frame.addEventListener("pointerleave", () => {
      if (state.crosshairSource !== source) return;
      state.crosshairTime = null;
      state.crosshairSource = null;
      renderSnapshotDetails(asOfDate());
      schedulePathOverlay();
    });
  }
}

function timeKey(time) {
  if (!time) return null;
  if (typeof time === "string") return time;
  if (typeof time === "number") return new Date(time * 1000).toISOString().slice(0, 10);
  return `${time.year}-${String(time.month).padStart(2, "0")}-${String(time.day).padStart(2, "0")}`;
}

function activePath() {
  return state.payload.paths[state.currentMultiplier];
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

function eventByCandidateId(events, candidateId) {
  return events.find(
    (event) => event.kind === "candidate_created" && event.candidateId === candidateId,
  );
}

function barByTime(time) {
  const index = endIndex(state.payload.bars, time) - 1;
  const bar = state.payload.bars[index];
  return bar?.time === time ? bar : null;
}

function updateReplay({ resetRange = false } = {}) {
  const date = asOfDate();
  const { charts, payload } = state;
  const wasFollowingEnd = isFollowingAsOf();

  const visibleBars = payload.bars.slice(0, state.asOfIndex + 1);
  charts.priceSeries.setData(visibleBars);
  const calendar = visibleBars.map((bar) => ({ time: bar.time, value: 0 }));
  charts.efficiencyTimeAnchor.setData(calendar);
  charts.pathTimeAnchor.setData(calendar);
  const efficiency = recordsThrough(payload.efficiency, date);
  for (const [window, series] of Object.entries(charts.efficiencySeries)) {
    const windowNumber = Number(window);
    series.setData(
      state.visibleEfficiency.has(windowNumber)
        ? efficiency.map((row) => ({ time: row.time, value: row[`efficiency_${window}`] })).filter((row) => row.value !== null)
        : [],
    );
  }
  updateIndicatorBounds(charts.efficiencyBounds, efficiency, date);

  const snapshots = recordsThrough(activePath().snapshots, date);
  charts.pathEfficiency.setData(
    snapshots
      .filter((snapshot) => snapshot.pathEfficiency !== null)
      .map((snapshot) => ({
        time: snapshot.time,
        value: snapshot.pathEfficiency,
        color: DIRECTION_COLORS[snapshot.direction],
      })),
  );
  updateIndicatorBounds(charts.pathBounds, snapshots, date);

  dom.asOfInput.value = date;
  dom.asOfReadout.textContent = `AS OF ${date}`;
  dom.previousButton.disabled = state.asOfIndex === 0;
  dom.nextButton.disabled = state.asOfIndex === payload.bars.length - 1;
  renderSnapshotDetails(date);
  schedulePathOverlay();

  if (resetRange || wasFollowingEnd) setDefaultVisibleRange();
}

function updateIndicatorBounds(series, records, fallbackTime) {
  if (state.currentIndicatorScale === "auto") {
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
  const from = Math.max(0, state.asOfIndex - state.currentRangeBars);
  const to = state.asOfIndex;
  for (const chart of [state.charts.price, state.charts.efficiency, state.charts.path]) {
    chart.timeScale().setVisibleLogicalRange({ from, to });
  }
}

function renderSnapshotDetails(time) {
  const snapshot = lastRecordThrough(activePath().snapshots, time);
  const rows = snapshot
    ? [
        ["观察日", time],
        ["路径方向", snapshot.direction === "up" ? "上行" : "下行", `direction-${snapshot.direction}`, true],
        ["锚点", `${snapshot.anchorAt} / ${formatNumber(snapshot.anchorPrice)}`, null, true],
        ["确认日", snapshot.confirmedAt ?? "-"],
        ["活跃候选", `#${snapshot.currentCandidateId}`],
        ["路径效率", formatDecimal(snapshot.pathEfficiency), null, true],
        ["路径长度", formatNumber(snapshot.pathLengthMin)],
        ["候选年龄", `${snapshot.candidateAgeBars} bars`],
        ["最大反向移动", formatNumber(snapshot.maximumCounterMove)],
      ]
    : [["观察日", time], ["路径状态", "尚未齐套"]];
  dom.snapshotValues.replaceChildren(
    ...rows.map(([label, value, valueClass, primary]) => {
      const row = document.createElement("div");
      const term = document.createElement("dt");
      const definition = document.createElement("dd");
      term.textContent = label;
      definition.textContent = value;
      if (valueClass) definition.classList.add(valueClass);
      if (primary) row.classList.add("primary");
      row.append(term, definition);
      return row;
    }),
  );
}

function formatNumber(value) {
  return value === null || value === undefined ? "-" : Number(value).toFixed(2);
}

function formatDecimal(value) {
  return value === null || value === undefined ? "-" : Number(value).toFixed(3);
}

function schedulePathOverlay() {
  window.requestAnimationFrame(() => {
    renderPathOverlay();
    renderSynchronizedTimeMarkers();
  });
}

function svgElement(name, attributes) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  return element;
}

function chartPoint(time, price) {
  const x = state.charts.price.timeScale().timeToCoordinate(time);
  const y = state.charts.priceSeries.priceToCoordinate(price);
  return x === null || y === null ? null : { x, y };
}

function appendLine(group, start, end, color, dash = null, width = 2) {
  const line = svgElement("line", {
    x1: start.x,
    y1: start.y,
    x2: end.x,
    y2: end.y,
    stroke: color,
    "stroke-width": width,
    "stroke-linecap": "round",
  });
  if (dash) line.setAttribute("stroke-dasharray", dash);
  group.append(line);
}

function appendCircle(group, point, color, radius = 3, fill = "#fbfcfb") {
  group.append(
    svgElement("circle", {
      cx: point.x,
      cy: point.y,
      r: radius,
      fill,
      stroke: color,
      "stroke-width": 1.5,
    }),
  );
}

function renderPathOverlay() {
  if (!state.charts || !state.payload) return;
  const bounds = dom.priceFrame.getBoundingClientRect();
  dom.pathOverlay.setAttribute("viewBox", `0 0 ${bounds.width} ${bounds.height}`);
  dom.pathOverlay.replaceChildren();

  const date = asOfDate();
  const events = recordsThrough(activePath().events, date);
  const confirmed = events.filter((event) => event.kind === "candidate_confirmed");
  const completedGroup = svgElement("g", {});
  for (const event of confirmed) {
    const anchor = chartPoint(event.anchorAt, event.anchorPrice);
    const confirmationBar = barByTime(event.time);
    const confirmation = confirmationBar ? chartPoint(event.time, confirmationBar.close) : null;
    if (!anchor || !confirmation) continue;
    const color = DIRECTION_COLORS[event.direction];
    appendLine(completedGroup, anchor, confirmation, color);
    appendCircle(completedGroup, anchor, color);
    appendCircle(completedGroup, confirmation, color, 2.5, color);
  }
  dom.pathOverlay.append(completedGroup);

  const snapshot = lastRecordThrough(activePath().snapshots, date);
  const activeCandidate = snapshot
    ? eventByCandidateId(events, snapshot.currentCandidateId)
    : null;
  const currentBar = barByTime(date);
  if (activeCandidate && currentBar) {
    const anchor = chartPoint(activeCandidate.anchorAt, activeCandidate.anchorPrice);
    const current = chartPoint(date, currentBar.close);
    if (anchor && current) {
      const group = svgElement("g", {});
      const color = DIRECTION_COLORS[activeCandidate.direction];
      appendLine(group, anchor, current, color, "6 5", 1.5);
      appendCircle(group, anchor, color, 3.5);
      dom.pathOverlay.append(group);
    }
  }

  if (!state.showCandidateLedger) return;
  const auditGroup = svgElement("g", { opacity: "0.82" });
  for (const event of events) {
    if (event.kind === "candidate_confirmed") continue;
    const anchor = chartPoint(event.anchorAt, event.anchorPrice);
    if (!anchor) continue;
    const color = event.kind === "candidate_invalidated" ? INVALIDATED_COLOR : DIRECTION_COLORS[event.direction];
    appendCircle(auditGroup, anchor, color, event.kind === "candidate_invalidated" ? 4.5 : 3);
    if (event.kind === "candidate_invalidated") {
      const invalidationBar = barByTime(event.time);
      const invalidation = invalidationBar ? chartPoint(event.time, invalidationBar.close) : null;
      if (invalidation) appendLine(auditGroup, anchor, invalidation, color, "2 4", 1);
    }
  }
  dom.pathOverlay.append(auditGroup);
}

function renderSynchronizedTimeMarkers() {
  const targets = [
    ["price", state.charts?.price, dom.priceTimeMarker],
    ["efficiency", state.charts?.efficiency, dom.efficiencyTimeMarker],
    ["path", state.charts?.path, dom.pathTimeMarker],
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

function setMultiplier(multiplier) {
  state.currentMultiplier = multiplier;
  for (const button of dom.multiplierButtons) {
    button.setAttribute("aria-pressed", String(button.dataset.multiplier === multiplier));
  }
  updateReplay();
}

function setPriceScale(scale) {
  state.currentPriceScale = scale;
  state.charts.price.applyOptions({
    rightPriceScale: { mode: PRICE_SCALE_MODES[scale] },
  });
  for (const button of dom.priceScaleButtons) {
    button.setAttribute("aria-pressed", String(button.dataset.priceScale === scale));
  }
  schedulePathOverlay();
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
    const requested = dom.asOfInput.value;
    const index = endIndex(state.payload.bars, requested) - 1;
    setAsOfIndex(index < 0 ? 0 : index);
  });
  dom.candidateToggle.addEventListener("change", () => {
    state.showCandidateLedger = dom.candidateToggle.checked;
    schedulePathOverlay();
  });
  for (const button of dom.multiplierButtons) {
    button.addEventListener("click", () => setMultiplier(button.dataset.multiplier));
  }
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
      const step = event.shiftKey ? 21 : 1;
      setAsOfIndex(state.asOfIndex + direction * step);
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
    if (state.payload.schemaVersion !== "phase1a_replay_v1") {
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
    const observer = new ResizeObserver(schedulePathOverlay);
    observer.observe(dom.priceFrame);
    observer.observe(dom.pathFrame);
    observer.observe(dom.efficiencyTimeMarker.parentElement);
    dom.loading.classList.add("ready");
  } catch (error) {
    setLoading(error instanceof Error ? error.message : "图表加载失败", true);
    console.error(error);
  }
}

boot();