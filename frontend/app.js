/* -------------------------------------------------------------
   AguaVerde - Dashboard JS
   ------------------------------------------------------------- */

const API       = '';   // same origin - FastAPI at /
const MAP_CTR   = [20.5, -103.5];
const MAP_ZOOM  = 9;
const SAM2_BOUNDS_SCALE = 0.58;
const SAM2_ADJUSTED_BOUNDS_SCALE = 0.38;

async function readJsonOrThrow(res) {
  const text = await res.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error(text.slice(0, 240));
    }
  }
  if (!res.ok) {
    throw new Error(data?.detail || `HTTP ${res.status}`);
  }
  return data;
}

const COLOR = { 0: '#388E3C', 1: '#F9A825', 2: '#C62828', null: '#9E9E9E' };
const LABEL = { 0: ' Sin estres', 1: ' Estres moderado', 2: ' Estres severo' };
const SHORT_LABEL = { 0: 'sin estres', 1: 'moderado', 2: 'severo' };

/* -- State ---------------------------------------------------- */
let map, trendChart, sam2Layer, gpChart, lkChart;
let demoSeriesChart, demoProbChart, demoGpChart;
let markers  = {};   // parcel_id -> Leaflet marker
let results  = {};   // parcel_id -> analysis result
let sam2Masks = {};  // parcel_id -> pixel mask result
let sam2Visible = { 0: true, 1: true, 2: true, pending: true };
let sam2AnalysisVisible = { 0: true, 1: true, 2: true };
let sam2MarkerVisible = { 0: true, 1: true, 2: true, pending: true };
let sam2HiddenParcels = new Set();
let sam2AdjustedOverlap = false;
let sam2GpMleAdjusted = false;
let sam2GpMleLoading = false;
let sam2GpMleEstimates = {};  // parcel_id -> /parcels/{id}/trend response
let parcels  = [];   // all parcels from /parcels
let parcelsLoading = null;
let photoB64 = null;
let photoMime = 'image/jpeg';
let filterCls = null;   // null = show all
let scanning  = false;

/* -- Bootstrap ------------------------------------------------ */
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  initTabs();
  initSam2();
  initForm();
  initFilters();
  initTrend();
  initDemo();
  setTrendParcelSelectState('Cargando parcelas...');
  setDemoParcelSelectState('Cargando parcelas...');
  loadParcels();
});

/* ==============================================================
   MAP
============================================================== */
function initMap() {
  map = L.map('map', { center: MAP_CTR, zoom: MAP_ZOOM, zoomControl: true });
  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '(c) <a href="https://openstreetmap.org">OpenStreetMap</a> contributors (c) <a href="https://carto.com">CARTO</a>',
    subdomains: 'abcd', maxZoom: 20,
  }).addTo(map);
  sam2Layer = L.layerGroup().addTo(map);
  // Force tile recalculation after CSS layout settles
  setTimeout(() => map.invalidateSize(), 100);
}

function makeIcon(cls, pulse = false) {
  const c = COLOR[cls];
  const sz = cls !== null ? 14 : 10;
  const ring = pulse ? `
    <div style="
      position:absolute;inset:-6px;border-radius:50%;
      border:2px solid ${c};opacity:.5;
      animation:ripple 1.2s ease-out infinite;
    "></div>` : '';
  return L.divIcon({
    className: '',
    html: `<div style="position:relative;width:${sz}px;height:${sz}px">
      ${ring}
      <div style="
        width:${sz}px;height:${sz}px;
        background:${c};
        border:2px solid rgba(255,255,255,.85);
        border-radius:50%;
        box-shadow:0 1px 5px rgba(0,0,0,.30);
        cursor:pointer;
        transition:transform .15s;
      "></div>
    </div>`,
    iconSize:   [sz + 12, sz + 12],
    iconAnchor: [(sz + 12) / 2, (sz + 12) / 2],
  });
}

/* -- Ripple keyframe injection --- */
const rippleStyle = document.createElement('style');
rippleStyle.textContent = `
  @keyframes ripple {
    0%   { transform: scale(.6); opacity: .6; }
    100% { transform: scale(2);  opacity: 0; }
  }`;
document.head.appendChild(rippleStyle);

/* ==============================================================
   LOAD PARCELS
============================================================== */
async function loadParcels() {
  if (parcelsLoading) return parcelsLoading;
  if (parcels.length) {
    populateTrendParcelSelect();
    return parcels;
  }

  parcelsLoading = (async () => {
  try {
    const res  = await fetch(`${API}/parcels?limit=200`);
    const data = await readJsonOrThrow(res);
    parcels = Array.isArray(data.parcels) ? data.parcels : [];
    document.getElementById('v-total').textContent = data.total;
    populateTrendParcelSelect();
    populateDemoParcelSelect();

    parcels.forEach(p => {
      const m = L.marker([p.latitude, p.longitude], { icon: makeIcon(null) }).addTo(map);
      m.bindTooltip(`<b>${p.parcel_id}</b><br><span style="font-size:11px;color:#666">${p.state}</span>`, {
        direction: 'top', offset: [0, -6],
      });
      m.on('click', () => onMarkerClick(p));
      markers[p.parcel_id] = m;
    });

    showToast(`${data.total} parcelas cargadas`, 'success');
    return parcels;
  } catch (e) {
    setTrendParcelSelectState('No se pudieron cargar parcelas');
    setDemoParcelSelectState('No se pudieron cargar parcelas');
    showToast('No se pudo conectar con el backend: ' + e.message, 'error');
    return [];
  } finally {
    parcelsLoading = null;
  }
  })();

  return parcelsLoading;
}

function onMarkerClick(p) {
  if (activeTab() === 'sam2') {
    toggleSam2Parcel(p);
    return;
  }

  switchTab('parcel');
  const trendSelect = document.getElementById('trend-parcel');
  if (trendSelect) trendSelect.value = p.parcel_id;

  // Use cached result (with LLM report) if available
  if (results[p.parcel_id]?.llm_report != null) {
    document.getElementById('empty-state').classList.add('hidden');
    document.getElementById('parcel-card').classList.remove('hidden');
    renderResult(results[p.parcel_id]);
    return;
  }

  showLoadingCard(p.parcel_id, p.latitude, p.longitude, p.state);
  analyzeParcelAndRender(p.parcel_id, false);
}

/* ==============================================================
   ANALYSIS
============================================================== */
async function analyzeParcel(parcelId, skipLlm = false) {
  try {
    const res = await fetch(`${API}/analyze/parcel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ parcel_id: parcelId, skip_llm: skipLlm }),
    });
    if (!res.ok) {
      await readJsonOrThrow(res);
    }
    const data = await readJsonOrThrow(res);
    results[parcelId] = data;
    setMarkerClass(parcelId, data.stress.class);
    updateStats();
    if (activeTab() === 'sam2') {
      refreshSam2MarkerVisibility();
      renderSam2View();
    } else {
      applyFilter();
    }
    return data;
  } catch (e) {
    if (!skipLlm) showToast(`Error al analizar ${parcelId}: ${e.message}`, 'error');
    return null;
  }
}

async function analyzeCoords(lat, lon, photo, mime, skipLlm) {
  const res = await fetch(`${API}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      lat, lon,
      photo_b64:  photo || null,
      photo_mime: mime,
      skip_llm:   skipLlm,
    }),
  });
  if (!res.ok) {
    await readJsonOrThrow(res);
  }
  return readJsonOrThrow(res);
}

async function analyzeParcelAndRender(parcelId, skipLlm = false) {
  const data = await analyzeParcel(parcelId, skipLlm);
  if (data) renderResult(data);
}

/* ==============================================================
   RENDER
============================================================== */
function showLoadingCard(parcelId, lat, lon, state) {
  document.getElementById('empty-state').classList.add('hidden');
  document.getElementById('parcel-card').classList.remove('hidden');
  document.getElementById('parcel-id').textContent  = parcelId;
  document.getElementById('parcel-meta').textContent = (lat && lon)
    ? `${lat.toFixed(4)}, ${lon.toFixed(4)}${state ? ' - ' + state : ''}`
    : 'Cargando...';
  const badge = document.getElementById('stress-badge');
  badge.textContent = '...';
  badge.className   = 'stress-badge';
  document.getElementById('conf-value').textContent = '-';
  document.getElementById('conf-bar').style.width   = '0%';
  ['ndmi','ndvi','ndwi','ndre','evi'].forEach(k =>
    document.getElementById(`idx-${k}`).textContent = '...'
  );
  document.getElementById('trend-dir').textContent  = '-';
  document.getElementById('trend-dir').className    = 'trend-direction sin_datos';
  document.getElementById('report-body').innerHTML  =
    '<div class="loading-dots"><span></span><span></span><span></span></div>';
  document.getElementById('report-model').textContent = '';
  ['0','1','2'].forEach(i => {
    document.getElementById(`pb-${i}`).style.width = '0%';
    document.getElementById(`pct-${i}`).textContent = '-';
  });
  if (trendChart) { trendChart.destroy(); trendChart = null; }
}

function renderResult(data) {
  const { stress, indices, trend, llm_report, location } = data;
  const cls = stress.class;

  document.getElementById('empty-state').classList.add('hidden');
  document.getElementById('parcel-card').classList.remove('hidden');
  document.getElementById('parcel-id').textContent = location.parcel_id;

  const parts = [];
  if (location.dist_km != null) parts.push(`${location.dist_km.toFixed(2)} km`);
  if (location.state)           parts.push(location.state);
  if (location.parcel_lat)      parts.push(`${location.parcel_lat.toFixed(4)}, ${location.parcel_lon.toFixed(4)}`);
  document.getElementById('parcel-meta').textContent = parts.join(' - ');

  // Badge
  const badge = document.getElementById('stress-badge');
  badge.textContent = `${stress.emoji} ${stress.label}`;
  badge.className   = `stress-badge s${cls}`;

  // Confidence
  const conf = Math.round(stress.confidence * 100);
  document.getElementById('conf-value').textContent = `${conf}%`;
  const bar = document.getElementById('conf-bar');
  bar.style.width      = `${conf}%`;
  bar.style.background = COLOR[cls];

  // Indices
  document.getElementById('idx-ndmi').textContent = fmt(indices.NDMI);
  document.getElementById('idx-ndvi').textContent = fmt(indices.NDVI);
  document.getElementById('idx-ndwi').textContent = fmt(indices.NDWI);
  document.getElementById('idx-ndre').textContent = fmt(indices.NDRE);
  document.getElementById('idx-evi').textContent  = fmt(indices.EVI);

  // Trend
  const dir  = trend.direction || 'sin_datos';
  const dirEl = document.getElementById('trend-dir');
  dirEl.textContent = {
    descendente: 'v Tendencia descendente',
    ascendente:  '^ Tendencia ascendente',
    estable:     '-> Tendencia estable',
    sin_datos:   '- Sin datos suficientes',
  }[dir] || '-';
  dirEl.className = `trend-direction ${dir}`;
  renderTrendChart(trend.windows || []);

  // Probabilities
  const probs = stress.probabilities;
  const keys  = ['Sin estres', 'Moderado', 'Severo'];
  keys.forEach((k, i) => {
    const pct = Math.round((probs[k] || 0) * 100);
    document.getElementById(`pb-${i}`).style.width  = `${pct}%`;
    document.getElementById(`pct-${i}`).textContent = `${pct}%`;
  });

  // Report
  const reportEl = document.getElementById('report-body');
  const modelEl  = document.getElementById('report-model');
  if (llm_report) {
    reportEl.textContent  = llm_report.full_text;
    reportEl.className    = llm_report.fallback ? 'report-body fallback' : 'report-body';
    modelEl.textContent   = llm_report.model_used
      ? `(${llm_report.model_used})`
      : '';
  } else {
    reportEl.textContent  = 'Analisis en modo rapido - sin reporte LLM.';
    reportEl.className    = 'report-body fallback';
    modelEl.textContent   = '';
  }
}

/* -- Trend chart ----------------------------------------------- */
function renderTrendChart(windows) {
  if (trendChart) { trendChart.destroy(); trendChart = null; }
  if (!windows.length) return;

  const ctx    = document.getElementById('trend-chart').getContext('2d');
  const labels = windows.map((_, i) => `V${i + 1}`);
  const values = windows.map(w => +(w.ndmi_mean ?? 0).toFixed(4));
  const ptColors = windows.map(w => COLOR[w.label] ?? COLOR[null]);

  trendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: values,
        borderColor: '#388E3C',
        backgroundColor: 'rgba(56,142,60,.07)',
        pointBackgroundColor: ptColors,
        pointBorderColor: '#fff',
        pointBorderWidth: 2,
        pointRadius: 7,
        tension: 0.3,
        fill: true,
      }],
    },
    options: {
      responsive: true,
      animation: { duration: 350 },
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 10 } } },
        y: {
          ticks: { font: { size: 10 } },
          grid: { color: '#f0f0ec' },
          title: { display: true, text: 'NDMI', font: { size: 10 } },
        },
      },
    },
  });
}

/* ==============================================================
   GP TREND TAB (Experimento D - Gaussian Process por parcela)
============================================================== */
const Z_LABEL = {
  sin_estres: 'Sin estres (dentro de lo esperado)',
  moderado:   'Estres moderado (1-2 sigma por debajo de lo esperado)',
  severo:     'Estres severo (>2 sigma por debajo de lo esperado)',
};

function populateTrendParcelSelect() {
  const sel = document.getElementById('trend-parcel');
  if (!sel) return;

  if (!parcels.length) {
    setTrendParcelSelectState('Sin parcelas cargadas');
    return;
  }

  const previousValue = sel.value;
  sel.innerHTML = parcels
    .map(p => {
      const id = escapeHtml(p.parcel_id);
      const state = escapeHtml(p.state || 'Jalisco');
      return `<option value="${id}">${id} - ${state}</option>`;
    })
    .join('');
  sel.disabled = false;

  if (previousValue && parcels.some(p => p.parcel_id === previousValue)) {
    sel.value = previousValue;
  } else if (parcels[0]?.parcel_id) {
    sel.value = parcels[0].parcel_id;
  }

  updateTrendCalcState();
}

function setTrendParcelSelectState(message) {
  const sel = document.getElementById('trend-parcel');
  if (!sel) return;
  sel.innerHTML = `<option value="">${escapeHtml(message)}</option>`;
  sel.disabled = true;
  updateTrendCalcState();
}

function updateTrendCalcState() {
  const sel = document.getElementById('trend-parcel');
  const btn = document.getElementById('btn-trend-calc');
  if (!sel || !btn) return;
  btn.disabled = sel.disabled || !sel.value;
}

async function ensureTrendParcelsLoaded() {
  if (parcels.length) {
    populateTrendParcelSelect();
    return;
  }
  setTrendParcelSelectState('Cargando parcelas...');
  await loadParcels();
}

let lastTrendData = null;
let trendView = 'individual';   // 'individual' | 'group'

function initTrend() {
  const calcBtn = document.getElementById('btn-trend-calc');
  const parcelSelect = document.getElementById('trend-parcel');
  if (calcBtn) calcBtn.addEventListener('click', calcTrend);
  if (parcelSelect) {
    parcelSelect.addEventListener('change', () => {
      updateTrendCalcState();
      // Mostrar seccion Exp E al seleccionar parcela - independiente del GP
      document.getElementById('exp-e-section').classList.remove('hidden');
      document.getElementById('likelihood-result').classList.add('hidden');
      if (lkChart) { lkChart.destroy(); lkChart = null; }
    });
    ['focus', 'mousedown', 'click'].forEach(eventName => {
      parcelSelect.addEventListener(eventName, ensureTrendParcelsLoaded);
    });
  }
  const lkBtn = document.getElementById('btn-likelihood');
  if (lkBtn) lkBtn.addEventListener('click', calcLikelihood);
  document.querySelectorAll('.view-toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (btn.dataset.view === 'group' && !lastTrendData?.group) return;
      trendView = btn.dataset.view;
      document.querySelectorAll('.view-toggle-btn').forEach(b => b.classList.toggle('active', b === btn));
      if (lastTrendData) applyTrendView(lastTrendData);
    });
  });
}

async function calcTrend() {
  const parcelId = document.getElementById('trend-parcel').value;
  const index    = document.getElementById('trend-index').value;
  if (!parcelId) {
    showToast('Selecciona una parcela', 'error');
    return;
  }

  const btn = document.getElementById('btn-trend-calc');
  btn.disabled = true;
  btn.textContent = 'Calculando...';

  try {
    const res = await fetch(`${API}/parcels/${parcelId}/trend?index=${index}&horizon=5`);
    const data = await readJsonOrThrow(res);
    renderTrendResult(data);
  } catch (e) {
    showToast(`Error al calcular tendencia: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Calcular tendencia';
    updateTrendCalcState();
  }
}

function renderTrendResult(data) {
  document.getElementById('trend-empty').classList.add('hidden');
  document.getElementById('trend-result').classList.remove('hidden');
  // Exp E siempre visible al tener una parcela activa
  document.getElementById('exp-e-section').classList.remove('hidden');
  document.getElementById('likelihood-result').classList.add('hidden');
  if (lkChart) { lkChart.destroy(); lkChart = null; }

  lastTrendData = data;
  trendView = 'individual';
  document.querySelectorAll('.view-toggle-btn').forEach(b => b.classList.toggle('active', b.dataset.view === 'individual'));

  const groupBtn = document.querySelector('.view-toggle-btn[data-view="group"]');
  const hasGroup = !!data.group;
  groupBtn.disabled = !hasGroup;
  groupBtn.title = hasGroup
    ? `${data.group.n_members} parcelas con terreno similar (grupo ${data.group.group_id})`
    : 'Sin agrupacion por terreno disponible';
  document.getElementById('trend-no-group').classList.toggle('hidden', hasGroup);

  applyTrendView(data);
}

function applyTrendView(data) {
  const block = (trendView === 'group' && data.group) ? data.group : {
    forecast: data.forecast, last_observation: data.last_observation, gp_curve: data.gp_curve,
  };
  const { last_observation, forecast } = block;
  const index = data.index;

  document.getElementById('trend-last-date').textContent  = last_observation.date;
  document.getElementById('trend-last-value').textContent = last_observation.value.toFixed(4);
  document.getElementById('trend-last-z').textContent     = last_observation.z.toFixed(2);

  const labelEl = document.getElementById('trend-last-label');
  labelEl.textContent = Z_LABEL[last_observation.label] || last_observation.label;
  labelEl.className   = `trend-direction ${
    { sin_estres: 'ascendente', moderado: 'estable', severo: 'descendente' }[last_observation.label] || 'sin_datos'
  }`;

  const basis = trendView === 'group'
    ? `con base en ${data.group.n_members} parcelas de terreno similar (grupo ${data.group.group_id}), no solo el historial propio`
    : 'con base en el historial propio de esta parcela';
  document.getElementById('trend-forecast-text').textContent =
    `Para el ${forecast.date} (+5 dias), el GP espera ${index} = ${forecast.mean.toFixed(4)} +- ${forecast.std.toFixed(4)} (1 sigma), ${basis}.`;

  const infoEl = document.getElementById('trend-group-info');
  infoEl.textContent = trendView === 'group'
    ? `Grupo de terreno ${data.group.group_id} - ${data.group.n_members} parcelas - baseline propio = ${data.group.baseline.toFixed(4)}`
    : '';

  renderGpChart(data, block);
}

function renderGpChart(data, block) {
  if (gpChart) { gpChart.destroy(); gpChart = null; }

  const { history, index } = data;
  const { gp_curve, forecast } = block;
  const upper2 = gp_curve.days.map((d, i) => ({ x: d, y: gp_curve.mean[i] + 2 * gp_curve.std[i] }));
  const upper1 = gp_curve.days.map((d, i) => ({ x: d, y: gp_curve.mean[i] + gp_curve.std[i] }));
  const meanLn = gp_curve.days.map((d, i) => ({ x: d, y: gp_curve.mean[i] }));
  const lower1 = gp_curve.days.map((d, i) => ({ x: d, y: gp_curve.mean[i] - gp_curve.std[i] }));
  const lower2 = gp_curve.days.map((d, i) => ({ x: d, y: gp_curve.mean[i] - 2 * gp_curve.std[i] }));
  const observed = history.days.map((d, i) => ({ x: d, y: history.values[i] }));
  const fcPoint = [{ x: forecast.day, y: forecast.mean }];
  const isGroup  = trendView === 'group';
  const lineColor = isGroup ? '#E65100' : '#388E3C';
  const bandRgba  = isGroup ? '230,81,0' : '56,142,60';
  const meanLabel = isGroup ? 'GP grupo' : 'GP individual';

  const ctx = document.getElementById('gp-chart').getContext('2d');
  gpChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        { label: '-2 sigma', data: lower2, borderWidth: 0, pointRadius: 0, fill: false },
        { label: '+2 sigma', data: upper2, borderWidth: 0, pointRadius: 0, fill: '-1', backgroundColor: `rgba(${bandRgba},.10)` },
        { label: '-1 sigma', data: lower1, borderWidth: 0, pointRadius: 0, fill: false },
        { label: '+1 sigma', data: upper1, borderWidth: 0, pointRadius: 0, fill: '-1', backgroundColor: `rgba(${bandRgba},.22)` },
        { label: meanLabel, data: meanLn, borderColor: lineColor, borderWidth: 2, pointRadius: 0, fill: false, tension: .15 },
        { label: index, data: observed, borderWidth: 0, pointRadius: 2.5, pointBackgroundColor: '#1C2018', showLine: false },
        { label: 'Pronostico', data: fcPoint, pointRadius: 6, pointStyle: 'star', pointBackgroundColor: '#1565C0', showLine: false },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: {
          display: true,
          labels: { filter: l => [meanLabel, index, 'Pronostico'].includes(l.text), font: { size: 10 } },
        },
      },
      scales: {
        x: {
          type: 'linear',
          ticks: {
            font: { size: 10 },
            callback: v => dayOffsetToDate(data.date0, v),
          },
          grid: { display: false },
        },
        y: {
          ticks: { font: { size: 10 } },
          grid: { color: '#f0f0ec' },
          title: { display: true, text: index, font: { size: 10 } },
        },
      },
    },
  });
}

function dayOffsetToDate(date0, dayOffset) {
  const d = new Date(date0 + 'T00:00:00Z');
  d.setUTCDate(d.getUTCDate() + Math.round(dayOffset));
  return d.toISOString().slice(0, 7);   // YYYY-MM
}

/* ==============================================================
   STATS & FILTER
============================================================== */
function updateStats() {
  let g = 0, y = 0, r = 0;
  Object.values(results).forEach(d => {
    if (d.stress.class === 0)      g++;
    else if (d.stress.class === 1) y++;
    else if (d.stress.class === 2) r++;
  });
  document.getElementById('v-green').textContent  = g;
  document.getElementById('v-yellow').textContent = y;
  document.getElementById('v-red').textContent    = r;
}

function setMarkerClass(parcelId, cls) {
  if (markers[parcelId]) markers[parcelId].setIcon(makeIcon(markerClassForParcelId(parcelId, cls)));
}

function markerClassForParcelId(parcelId, fallbackClass = null) {
  if (sam2GpMleAdjusted) {
    const gpMleClass = gpMleClassForParcel(parcelId);
    if (gpMleClass !== null) return gpMleClass;
  }
  return fallbackClass ?? results[parcelId]?.stress?.class ?? null;
}

function syncMarkerIcons() {
  Object.keys(markers).forEach(parcelId => setMarkerClass(parcelId));
}

function applyFilter() {
  Object.entries(markers).forEach(([id, m]) => {
    const cls = markerClassForParcelId(id);
    if (filterCls === null || cls === null || cls === filterCls) {
      if (!map.hasLayer(m)) map.addLayer(m);
    } else {
      if (map.hasLayer(m)) map.removeLayer(m);
    }
  });
}

/* ==============================================================
   SCAN ALL
============================================================== */
async function scanAll() {
  if (scanning) return;
  scanning = true;

  const btn = document.getElementById('btn-scan');
  btn.disabled = true;

  const overlay = document.createElement('div');
  overlay.className = 'scan-overlay';
  document.querySelector('.map-container').appendChild(overlay);

  const toScan = visibleParcels().filter(p => !results[p.parcel_id]);
  let done = 0;

  if (!toScan.length) {
    overlay.textContent = 'No hay parcelas pendientes en esta vista';
    await sleep(900);
    overlay.remove();
    btn.disabled = false;
    scanning = false;
    return;
  }

  for (const p of toScan) {
    overlay.textContent = ` Escaneando ${done + 1}/${toScan.length} - ${p.parcel_id}`;
    await analyzeParcel(p.parcel_id, true);
    done++;
    await sleep(80);   // avoid overwhelming the server
  }

  overlay.remove();
  btn.disabled = false;
  scanning = false;
  showToast(` Escaneo completo: ${done} parcelas analizadas`, 'success');
  if (activeTab() === 'sam2') renderSam2View();
}

function visibleParcels() {
  const bounds = map.getBounds();
  return parcels.filter(p => {
    if (!bounds.contains([p.latitude, p.longitude])) return false;
    const marker = markers[p.parcel_id];
    return !marker || map.hasLayer(marker);
  });
}

/* ==============================================================
   TABS
============================================================== */
function initTabs() {
  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });
}

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === name)
  );
  document.querySelectorAll('.tab-content').forEach(c =>
    c.classList.toggle('hidden', c.id !== `tab-${name}`)
  );
  if (name === 'sam2') {
    refreshSam2MarkerVisibility();
    renderSam2View();
  } else if (name === 'trend') {
    ensureTrendParcelsLoaded();
  } else if (name === 'demo') {
    if (sam2Layer) {
      resetSam2MarkerState();
      sam2Layer.clearLayers();
      applyFilter();
    }
    ensureDemoLoaded();
  } else if (sam2Layer) {
    resetSam2MarkerState();
    sam2Layer.clearLayers();
    applyFilter();
  }
}

function activeTab() {
  return document.querySelector('.tab.active')?.dataset.tab || 'parcel';
}

function initSam2() {
  const refreshBtn = document.getElementById('btn-sam2-refresh');
  const scanBtn = document.getElementById('btn-sam2-scan-all');
  const overlapBtn = document.getElementById('btn-sam2-adjust-overlap');
  const gpMleBtn = document.getElementById('btn-sam2-gp-mle');
  if (refreshBtn) refreshBtn.addEventListener('click', renderSam2View);
  if (scanBtn) scanBtn.addEventListener('click', scanSam2All);
  if (overlapBtn) {
    overlapBtn.addEventListener('click', () => {
      sam2AdjustedOverlap = !sam2AdjustedOverlap;
      updateSam2OverlapButton();
      renderSam2View();
    });
    updateSam2OverlapButton();
  }
  if (gpMleBtn) {
    gpMleBtn.addEventListener('click', toggleSam2GpMleAdjustment);
    updateSam2GpMleButton();
  }
  if (map) {
    map.on('moveend zoomend', () => {
      if (activeTab() === 'sam2') renderSam2View();
    });
  }
  document.querySelectorAll('[data-sam2-filter]').forEach(input => {
    input.addEventListener('change', () => {
      const key = input.dataset.sam2Filter;
      sam2Visible[key] = input.checked;
      renderSam2View();
    });
  });
  document.querySelectorAll('[data-sam2-analysis-filter]').forEach(input => {
    input.addEventListener('change', () => {
      const key = input.dataset.sam2AnalysisFilter;
      sam2AnalysisVisible[key] = input.checked;
      renderSam2View();
    });
  });
  document.querySelectorAll('[data-sam2-marker-filter]').forEach(input => {
    input.addEventListener('change', () => {
      const key = input.dataset.sam2MarkerFilter;
      sam2MarkerVisible[key] = input.checked;
      refreshSam2MarkerVisibility();
    });
  });
}

function renderSam2View() {
  if (!sam2Layer) return;

  sam2Layer.clearLayers();

  const analyzed = Object.values(results).filter(Boolean);
  const totalMaskCount = Object.keys(sam2Masks).length;
  let visibleMaskCount = 0;
  const counts = { 0: 0, 1: 0, 2: 0 };
  const compositeItems = [];

  parcels.forEach(parcel => {
    const data = results[parcel.parcel_id];
    const mask = sam2Masks[parcel.parcel_id];
    const cls = parcelStressClass(parcel);
    const color = COLOR[cls];

    if (sam2HiddenParcels.has(parcel.parcel_id)) return;

    if (mask) {
      if (!sam2ParcelClassVisible(cls)) return;

      visibleMaskCount++;
      addMaskCounts(mask, counts, cls);
      if (maskHasVisiblePixels(mask, cls)) {
        compositeItems.push({ parcel, mask, parcelClass: cls });
      }
      return;
    }

    if (!sam2Visible.pending) return;
    if (!sam2ParcelClassVisible(cls)) return;

    const cell = L.rectangle(parcelCellBounds(parcel), {
      color,
      fillColor: color,
      fillOpacity: cls === null ? 0.08 : 0.28,
      opacity: cls === null ? 0.35 : 0.9,
      weight: cls === null ? 1 : 2,
      interactive: true,
    }).addTo(sam2Layer);

    cell.bindTooltip(sam2Tooltip(parcel, data), {
      direction: 'top',
      offset: [0, -4],
    });
    cell.on('click', () => toggleSam2Parcel(parcel));
  });

  renderSam2Composite(compositeItems);

  const coverage = parcels.length
    ? Math.round((visibleMaskCount / parcels.length) * 100)
    : 0;

  setText('sam2-total', visibleMaskCount);
  setText('sam2-coverage', `${coverage}%`);
  setText('sam2-green', compactNumber(counts[0]));
  setText('sam2-yellow', compactNumber(counts[1]));
  setText('sam2-red', compactNumber(counts[2]));
  const sam2MetaParts = [`${visibleMaskCount} mascaras pixel visibles`];
  if (sam2AdjustedOverlap) sam2MetaParts.push('overlap ajustado');
  if (sam2GpMleAdjusted) sam2MetaParts.push('GP + MLE');
  setText('sam2-meta', sam2MetaParts.join(' - '));

  const status = document.getElementById('sam2-status');
  if (status) {
    status.textContent = visibleMaskCount
      ? `Raster pixel activo sobre el mapa${sam2GpMleAdjusted ? ' - calibrado con GP + MLE' : ''}`
      : totalMaskCount
        ? 'No hay mascaras visibles con los filtros actuales'
        : 'Genera mascaras para activar el raster';
  }

  renderSam2List();
}

async function scanSam2All() {
  if (scanning) return;
  scanning = true;

  const btn = document.getElementById('btn-sam2-scan-all');
  if (btn) btn.disabled = true;

  const overlay = document.createElement('div');
  overlay.className = 'scan-overlay';
  document.querySelector('.map-container').appendChild(overlay);

  const pending = parcels.filter(p => !sam2Masks[p.parcel_id] || !results[p.parcel_id]);
  let done = 0;

  for (const p of pending) {
    overlay.textContent = `SAM2 analisis ${done + 1}/${pending.length} - ${p.parcel_id}`;
    if (!results[p.parcel_id]) {
      await analyzeParcel(p.parcel_id, true);
    }
    if (!sam2Masks[p.parcel_id]) {
      await loadSam2Mask(p.parcel_id);
    }
    done++;
    if (done % 5 === 0) renderSam2View();
    await sleep(40);
  }

  overlay.remove();
  if (btn) btn.disabled = false;
  scanning = false;
  renderSam2View();
  showToast(`SAM2 actualizado: ${done} parcelas listas`, 'success');
}

async function toggleSam2GpMleAdjustment() {
  if (sam2GpMleLoading) return;

  if (sam2GpMleAdjusted) {
    sam2GpMleAdjusted = false;
    updateSam2GpMleButton();
    syncMarkerIcons();
    refreshSam2MarkerVisibility();
    renderSam2View();
    showToast('Ajuste GP + MLE desactivado', 'info');
    return;
  }

  sam2GpMleLoading = true;
  updateSam2GpMleButton('Calculando...');

  const overlay = document.createElement('div');
  overlay.className = 'scan-overlay';
  document.querySelector('.map-container').appendChild(overlay);

  const candidates = sam2GpMleCandidates();
  let done = 0;
  let ok = 0;

  try {
    for (const parcel of candidates) {
      overlay.textContent = `GP + MLE ${done + 1}/${candidates.length} - ${parcel.parcel_id}`;
      const estimate = await loadSam2GpMleEstimate(parcel.parcel_id);
      if (estimate) ok++;
      done++;
      if (done % 5 === 0) renderSam2View();
      await sleep(25);
    }

    sam2GpMleAdjusted = ok > 0;
    syncMarkerIcons();
    refreshSam2MarkerVisibility();
    renderSam2View();
    showToast(
      ok
        ? `Ajuste GP + MLE activo: ${ok} parcelas calibradas`
        : 'No se pudieron obtener estimaciones GP + MLE',
      ok ? 'success' : 'error',
    );
  } finally {
    overlay.remove();
    sam2GpMleLoading = false;
    updateSam2GpMleButton();
  }
}

function sam2GpMleCandidates() {
  const active = parcels.filter(p => sam2Masks[p.parcel_id] || results[p.parcel_id]);
  return active.length ? active : parcels;
}

async function loadSam2GpMleEstimate(parcelId) {
  if (sam2GpMleEstimates[parcelId]) return sam2GpMleEstimates[parcelId];

  try {
    const res = await fetch(`${API}/parcels/${encodeURIComponent(parcelId)}/trend?index=NDMI&horizon=5`);
    const data = await readJsonOrThrow(res);
    sam2GpMleEstimates[parcelId] = data;
    return data;
  } catch (e) {
    return null;
  }
}

function updateSam2GpMleButton(label = null) {
  const btn = document.getElementById('btn-sam2-gp-mle');
  if (!btn) return;
  btn.disabled = sam2GpMleLoading;
  btn.classList.toggle('active', sam2GpMleAdjusted);
  btn.textContent = label || (sam2GpMleAdjusted ? 'GP + MLE activo' : 'Ajuste GP + MLE');
}

function focusSam2Parcel(parcel) {
  map.flyTo([parcel.latitude, parcel.longitude], 14, { duration: 0.8 });
  const marker = markers[parcel.parcel_id];
  if (marker) marker.openTooltip();
}

function openNewLocationFromParcel(parcel) {
  setNewLocationInputs(parcel.latitude, parcel.longitude);
  switchTab('new');
  map.flyTo([parcel.latitude, parcel.longitude], 14, { duration: 0.8 });
  showToast(`Coordenadas cargadas: ${parcel.parcel_id}`, 'success');
}

function setNewLocationInputs(lat, lon) {
  const latInput = document.getElementById('inp-lat');
  const lonInput = document.getElementById('inp-lon');
  if (latInput) latInput.value = Number(lat).toFixed(6);
  if (lonInput) lonInput.value = Number(lon).toFixed(6);
}

function toggleSam2Parcel(parcel) {
  if (sam2HiddenParcels.has(parcel.parcel_id)) {
    sam2HiddenParcels.delete(parcel.parcel_id);
  } else {
    sam2HiddenParcels.add(parcel.parcel_id);
  }
  refreshSam2MarkerVisibility();
  renderSam2View();
}

function setSam2MarkerOpacity(parcelId, opacity) {
  const marker = markers[parcelId];
  if (marker && typeof marker.setOpacity === 'function') {
    marker.setOpacity(opacity);
  }
}

function refreshSam2MarkerVisibility() {
  Object.entries(markers).forEach(([parcelId, marker]) => {
    setMarkerClass(parcelId);
    if (sam2MarkerAllowed(parcelId)) {
      if (!map.hasLayer(marker)) map.addLayer(marker);
      setSam2MarkerOpacity(parcelId, sam2HiddenParcels.has(parcelId) ? 0.35 : 1);
    } else if (map.hasLayer(marker)) {
      map.removeLayer(marker);
    }
  });
}

function resetSam2MarkerState() {
  Object.keys(markers).forEach(parcelId => setSam2MarkerOpacity(parcelId, 1));
}

function sam2MarkerAllowed(parcelId) {
  const parcel = parcels.find(p => p.parcel_id === parcelId);
  const cls = parcel ? parcelStressClass(parcel) : (results[parcelId]?.stress?.class ?? null);
  const key = cls === null ? 'pending' : cls;
  return sam2MarkerVisible[key] !== false;
}

async function loadSam2Mask(parcelId) {
  if (sam2Masks[parcelId]) return sam2Masks[parcelId];
  try {
    const res = await fetch(`${API}/sam2/mask/${encodeURIComponent(parcelId)}`);
    const data = await readJsonOrThrow(res);
    sam2Masks[parcelId] = data;
    return data;
  } catch (e) {
    showToast(`Error SAM2 ${parcelId}: ${e.message}`, 'error');
    return null;
  }
}

function renderSam2List() {
  const list = document.getElementById('sam2-list');
  if (!list) return;

  const items = parcels.filter(p => {
    const mask = sam2Masks[p.parcel_id];
    const cls = parcelStressClass(p);
    return !sam2HiddenParcels.has(p.parcel_id)
      && mask
      && sam2ParcelClassVisible(cls)
      && maskHasVisiblePixels(mask, cls);
  });
  if (!items.length) {
    list.innerHTML = '<div class="sam2-empty">Sin mascaras activas</div>';
    return;
  }

  list.innerHTML = items
    .map(parcel => {
      const summary = summarizeMask(sam2Masks[parcel.parcel_id], parcelStressClass(parcel));
      const cls = summary.majority;
      const gpMleText = sam2GpMleAdjusted ? gpMleTextForParcel(parcel.parcel_id) : '';
      return `
        <button type="button" class="sam2-row s${cls}" data-parcel="${parcel.parcel_id}">
          <span>${parcel.parcel_id}</span>
          <strong>${summary.majorityPct}% ${SHORT_LABEL[cls]}${gpMleText ? ' · GP+MLE' : ''}</strong>
        </button>
      `;
    })
    .join('');

  list.querySelectorAll('.sam2-row').forEach(row => {
    row.addEventListener('click', () => {
      const parcel = parcels.find(p => p.parcel_id === row.dataset.parcel);
      if (parcel) toggleSam2Parcel(parcel);
    });
  });
}

function parcelStressClass(parcel) {
  const fallbackClass = results[parcel.parcel_id]?.stress?.class ?? null;
  if (!sam2GpMleAdjusted) return fallbackClass;
  return gpMleClassForParcel(parcel.parcel_id) ?? fallbackClass;
}

function gpMleClassForParcel(parcelId) {
  const estimate = sam2GpMleEstimates[parcelId];
  if (!estimate || estimate.error) return null;

  const block = estimate.group || estimate;
  return {
    sin_estres: 0,
    moderado: 1,
    severo: 2,
  }[block.last_observation?.label] ?? null;
}

function gpMleTextForParcel(parcelId) {
  const estimate = sam2GpMleEstimates[parcelId];
  if (!estimate || estimate.error) return '';

  const block = estimate.group || estimate;
  const label = block.last_observation?.label;
  const z = block.last_observation?.z;
  const basis = estimate.group ? `grupo ${estimate.group.group_id}` : 'individual';
  if (label == null || z == null) return '';

  return `GP + MLE (${basis}): ${label.replace('_', ' ')} | z=${Number(z).toFixed(2)}`;
}

function sam2ParcelClassVisible(parcelClass) {
  if (parcelClass === null) return true;
  return sam2AnalysisVisible[parcelClass] !== false;
}

function calibratedMaskClass(rawClass, parcelClass) {
  if (![0, 1, 2].includes(rawClass)) return rawClass;

  if (parcelClass === 0) {
    return rawClass === 2 ? 1 : 0;
  }

  if (parcelClass === 1) {
    return rawClass === 0 ? 0 : 1;
  }

  return rawClass;
}

function addMaskCounts(mask, counts, parcelClass = null) {
  for (let y = 0; y < mask.height; y++) {
    for (let x = 0; x < mask.width; x++) {
      const cls = calibratedMaskClass(mask.classes[y][x], parcelClass);
      if ([0, 1, 2].includes(cls)) counts[cls]++;
    }
  }
}

function summarizeMask(mask, parcelClass = null) {
  const counts = { 0: 0, 1: 0, 2: 0 };
  addMaskCounts(mask, counts, parcelClass);
  const total = counts[0] + counts[1] + counts[2] || 1;
  const majority = [0, 1, 2].sort((a, b) => counts[b] - counts[a])[0];
  return {
    majority,
    majorityPct: Math.round((counts[majority] / total) * 100),
    severePct: Math.round((counts[2] / total) * 100),
  };
}

function maskHasVisiblePixels(mask, parcelClass = null) {
  for (let y = 0; y < mask.height; y++) {
    for (let x = 0; x < mask.width; x++) {
      const cls = calibratedMaskClass(mask.classes[y][x], parcelClass);
      if (sam2Visible[cls]) return true;
    }
  }
  return false;
}

function parcelCellBounds(parcel) {
  const lat = parcel.latitude;
  const lon = parcel.longitude;
  const dLat = 0.010;
  const dLon = 0.012;
  return [
    [lat - dLat, lon - dLon],
    [lat + dLat, lon + dLon],
  ];
}

function sam2OverlayBounds(parcel, mask) {
  const bounds = Array.isArray(mask.bounds) && mask.bounds.length === 2
    ? mask.bounds
    : parcelCellBounds(parcel);
  return compactSam2Bounds(bounds, parcel, sam2BoundsScale());
}

function sam2BoundsScale() {
  return sam2AdjustedOverlap ? SAM2_ADJUSTED_BOUNDS_SCALE : SAM2_BOUNDS_SCALE;
}

function compactSam2Bounds(bounds, parcel, scale) {
  const south = bounds[0][0];
  const west = bounds[0][1];
  const north = bounds[1][0];
  const east = bounds[1][1];
  if (![south, west, north, east].every(Number.isFinite)) return bounds;

  const centerLat = parcel.latitude >= south && parcel.latitude <= north
    ? parcel.latitude
    : (south + north) / 2;
  const centerLon = parcel.longitude >= west && parcel.longitude <= east
    ? parcel.longitude
    : (west + east) / 2;
  const halfLat = ((north - south) * scale) / 2;
  const halfLon = ((east - west) * scale) / 2;

  return [
    [centerLat - halfLat, centerLon - halfLon],
    [centerLat + halfLat, centerLon + halfLon],
  ];
}

function sam2Tooltip(parcel, data) {
  const gpMleText = sam2GpMleAdjusted ? gpMleTextForParcel(parcel.parcel_id) : '';
  if (!data) return `<b>${parcel.parcel_id}</b><br>${gpMleText || 'Sin analizar'}`;
  return `<b>${parcel.parcel_id}</b><br>${data.stress.label}${gpMleText ? `<br>${gpMleText}` : ''}`;
}

function renderSam2Composite(items) {
  if (!items.length) return;

  const size = map.getSize();
  const width = Math.max(1, Math.round(size.x));
  const height = Math.max(1, Math.round(size.y));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');
  const finalClasses = new Int8Array(width * height);
  finalClasses.fill(-1);

  if (sam2AdjustedOverlap) {
    resolveAdjustedComposite(items, finalClasses, width, height);
  } else {
    items.forEach(({ parcel, mask, parcelClass }) => {
      resolveMaskIntoComposite(mask, sam2OverlayBounds(parcel, mask), finalClasses, width, height, parcelClass);
    });
  }

  const image = renderCompositeImage(ctx, finalClasses, width, height);
  ctx.putImageData(image, 0, 0);
  L.imageOverlay(canvas.toDataURL('image/png'), map.getBounds(), {
    opacity: 0.78,
    interactive: false,
    zIndex: 430,
  }).addTo(sam2Layer);
}

function updateSam2OverlapButton() {
  const btn = document.getElementById('btn-sam2-adjust-overlap');
  if (!btn) return;
  btn.classList.toggle('active', sam2AdjustedOverlap);
  btn.textContent = sam2AdjustedOverlap ? 'Overlap ajustado' : 'Ajustar overlap';
}

function resolveAdjustedComposite(items, finalClasses, width, height) {
  const votes = new Uint16Array(width * height * 3);
  items.forEach(({ parcel, mask, parcelClass }) => {
    addMaskVotes(mask, sam2OverlayBounds(parcel, mask), votes, width, height, parcelClass);
  });

  for (let idx = 0; idx < finalClasses.length; idx++) {
    const offset = idx * 3;
    const green = votes[offset];
    const yellow = votes[offset + 1];
    const red = votes[offset + 2];
    if (!green && !yellow && !red) continue;

    if (green >= yellow && green >= red) {
      finalClasses[idx] = 0;
    } else if (yellow >= red) {
      finalClasses[idx] = 1;
    } else {
      finalClasses[idx] = 2;
    }
  }
}

function addMaskVotes(mask, bounds, votes, width, height, parcelClass = null) {
  const south = bounds[0][0];
  const west = bounds[0][1];
  const north = bounds[1][0];
  const east = bounds[1][1];
  const topLeft = map.latLngToContainerPoint([north, west]);
  const bottomRight = map.latLngToContainerPoint([south, east]);
  const x0 = Math.floor(Math.min(topLeft.x, bottomRight.x));
  const x1 = Math.ceil(Math.max(topLeft.x, bottomRight.x));
  const y0 = Math.floor(Math.min(topLeft.y, bottomRight.y));
  const y1 = Math.ceil(Math.max(topLeft.y, bottomRight.y));
  if (x1 < 0 || y1 < 0 || x0 >= width || y0 >= height) return;

  const clippedX0 = Math.max(0, x0);
  const clippedY0 = Math.max(0, y0);
  const clippedX1 = Math.min(width, x1);
  const clippedY1 = Math.min(height, y1);
  const boxW = Math.max(1, x1 - x0);
  const boxH = Math.max(1, y1 - y0);

  for (let sy = clippedY0; sy < clippedY1; sy++) {
    const my = Math.min(mask.height - 1, Math.max(0, Math.floor(((sy - y0) / boxH) * mask.height)));
    for (let sx = clippedX0; sx < clippedX1; sx++) {
      const mx = Math.min(mask.width - 1, Math.max(0, Math.floor(((sx - x0) / boxW) * mask.width)));
      const cls = calibratedMaskClass(mask.classes[my][mx], parcelClass);
      if (![0, 1, 2].includes(cls)) continue;

      votes[(sy * width + sx) * 3 + cls]++;
    }
  }
}

function resolveMaskIntoComposite(mask, bounds, finalClasses, width, height, parcelClass = null) {
  const south = bounds[0][0];
  const west = bounds[0][1];
  const north = bounds[1][0];
  const east = bounds[1][1];
  const topLeft = map.latLngToContainerPoint([north, west]);
  const bottomRight = map.latLngToContainerPoint([south, east]);
  const x0 = Math.floor(Math.min(topLeft.x, bottomRight.x));
  const x1 = Math.ceil(Math.max(topLeft.x, bottomRight.x));
  const y0 = Math.floor(Math.min(topLeft.y, bottomRight.y));
  const y1 = Math.ceil(Math.max(topLeft.y, bottomRight.y));
  if (x1 < 0 || y1 < 0 || x0 >= width || y0 >= height) return;

  const clippedX0 = Math.max(0, x0);
  const clippedY0 = Math.max(0, y0);
  const clippedX1 = Math.min(width, x1);
  const clippedY1 = Math.min(height, y1);
  const boxW = Math.max(1, x1 - x0);
  const boxH = Math.max(1, y1 - y0);

  for (let sy = clippedY0; sy < clippedY1; sy++) {
    const my = Math.min(mask.height - 1, Math.max(0, Math.floor(((sy - y0) / boxH) * mask.height)));
    for (let sx = clippedX0; sx < clippedX1; sx++) {
      const mx = Math.min(mask.width - 1, Math.max(0, Math.floor(((sx - x0) / boxW) * mask.width)));
      const cls = calibratedMaskClass(mask.classes[my][mx], parcelClass);
      if (![0, 1, 2].includes(cls)) continue;

      const idx = sy * width + sx;
      if (cls < finalClasses[idx]) continue;

      finalClasses[idx] = cls;
    }
  }
}

function renderCompositeImage(ctx, finalClasses, width, height) {
  const image = ctx.createImageData(width, height);
  const colors = {
    0: [56, 142, 60, 135],
    1: [249, 168, 37, 155],
    2: [198, 40, 40, 175],
  };

  for (let idx = 0; idx < finalClasses.length; idx++) {
    const cls = finalClasses[idx];
    if (![0, 1, 2].includes(cls) || !sam2Visible[cls]) continue;

    const rgba = colors[cls];
    const out = idx * 4;
    image.data[out] = rgba[0];
    image.data[out + 1] = rgba[1];
    image.data[out + 2] = rgba[2];
    image.data[out + 3] = rgba[3];
  }

  return image;
}

function maskToDataUrl(mask, onlyClass = null) {
  const canvas = document.createElement('canvas');
  canvas.width = mask.width;
  canvas.height = mask.height;

  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(mask.width, mask.height);
  const colors = {
    0: [56, 142, 60, 120],
    1: [249, 168, 37, 145],
    2: [198, 40, 40, 165],
  };

  for (let y = 0; y < mask.height; y++) {
    for (let x = 0; x < mask.width; x++) {
      const idx = (y * mask.width + x) * 4;
      const cls = mask.classes[y][x];
      const rgba = (onlyClass === null || cls === onlyClass) && sam2Visible[cls]
        ? (colors[cls] || [158, 158, 158, 40])
        : [0, 0, 0, 0];
      img.data[idx] = rgba[0];
      img.data[idx + 1] = rgba[1];
      img.data[idx + 2] = rgba[2];
      img.data[idx + 3] = rgba[3];
    }
  }

  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL('image/png');
}

/* ==============================================================
   FORM (nueva ubicacion)
============================================================== */
function initForm() {
  const fileIn   = document.getElementById('inp-photo');
  const preview  = document.getElementById('photo-preview');
  const hint     = document.getElementById('drop-hint');
  const clearBtn = document.getElementById('clear-photo');

  fileIn.addEventListener('change', () => {
    const file = fileIn.files[0];
    if (!file) return;
    photoMime = file.type;
    const reader = new FileReader();
    reader.onload = e => {
      photoB64  = e.target.result.split(',')[1];
      preview.src = e.target.result;
      preview.classList.remove('hidden');
      hint.classList.add('hidden');
      clearBtn.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  });

  clearBtn.addEventListener('click', e => {
    e.stopPropagation();
    photoB64 = null; photoMime = 'image/jpeg';
    fileIn.value      = '';
    preview.src       = '';
    preview.classList.add('hidden');
    hint.classList.remove('hidden');
    clearBtn.classList.add('hidden');
  });

  document.getElementById('new-form').addEventListener('submit', async e => {
    e.preventDefault();
    const lat  = parseFloat(document.getElementById('inp-lat').value);
    const lon  = parseFloat(document.getElementById('inp-lon').value);
    const skip = document.getElementById('skip-llm').checked;
    const btn  = document.getElementById('btn-analyze');

    if (isNaN(lat) || isNaN(lon)) {
      showToast('Ingresa coordenadas validas', 'error');
      return;
    }
    if (lat < 15 || lat > 25 || lon < -108 || lon > -95) {
      showToast('Las coordenadas estan fuera de Jalisco / Mexico', 'info');
    }

    btn.disabled    = true;
    btn.textContent = ' Analizando...';

    switchTab('parcel');
    showLoadingCard('Nueva ubicacion', lat, lon, null);

    let tempMarker = L.marker([lat, lon], { icon: makeIcon(null, true) }).addTo(map);
    map.flyTo([lat, lon], 13, { duration: 1.2 });

    try {
      const data = await analyzeCoords(lat, lon, photoB64, photoMime, skip);
      renderResult(data);
      const cls = data.stress.class;
      tempMarker.setIcon(makeIcon(cls, false));
      tempMarker.bindPopup(`
        <b> ${lat.toFixed(4)}, ${lon.toFixed(4)}</b><br>
        Parcela ref: ${data.location.parcel_id}<br>
        ${LABEL[cls]}
      `).openPopup();
    } catch (err) {
      showToast('Error: ' + err.message, 'error');
      tempMarker.remove();
      document.getElementById('empty-state').classList.remove('hidden');
      document.getElementById('parcel-card').classList.add('hidden');
    } finally {
      btn.disabled    = false;
      btn.textContent = ' Analizar ubicacion';
    }
  });
}

/* ==============================================================
   FILTERS
============================================================== */
function initFilters() {
  document.getElementById('btn-scan').addEventListener('click', scanAll);

  document.getElementById('btn-filter-all').addEventListener('click', () => {
    filterCls = null;
    setFilterActive(null);
    applyFilter();
  });
  [0, 1, 2].forEach(cls => {
    document.getElementById(`btn-filter-${cls}`).addEventListener('click', () => {
      filterCls = cls;
      setFilterActive(cls);
      applyFilter();
    });
  });
}

function setFilterActive(cls) {
  document.getElementById('btn-filter-all').classList.toggle('active', cls === null);
  [0, 1, 2].forEach(c =>
    document.getElementById(`btn-filter-${c}`).classList.toggle('active', c === cls)
  );
}

/* ==============================================================
   DEMO GUIADA
============================================================== */
let demoLoadedParcel = null;
let demoCurrentMask = null;

const DEMO_STRESS_LABELS = ['Sin estres', 'Moderado', 'Severo'];
const DEMO_STRESS_COUNT_KEYS = ['sin_estres', 'moderado', 'severo'];

const DEMO_INDEX_META = {
  NDVI: {
    title: 'NDVI',
    bands: 'B08 NIR + B04 rojo',
    meaning: 'Mide vigor vegetal: valores altos suelen indicar vegetacion sana y densa.',
    palette: 'green',
  },
  NDWI: {
    title: 'NDWI',
    bands: 'B03 verde + B08 NIR',
    meaning: 'Mide contenido de agua en vegetacion. Ayuda a detectar cambios de humedad.',
    palette: 'blue',
  },
  NDMI: {
    title: 'NDMI',
    bands: 'B08 NIR + B11 SWIR',
    meaning: 'Indice clave: estima humedad de hoja y dosel. Si baja, puede indicar estres hidrico.',
    palette: 'teal',
  },
  NDRE: {
    title: 'NDRE',
    bands: 'B08 NIR + B05 red edge',
    meaning: 'Observa clorofila y senales tempranas de estres antes de que sean visibles.',
    palette: 'purple',
  },
  EVI: {
    title: 'EVI',
    bands: 'B08 NIR + B04 rojo + B02 azul',
    meaning: 'Refuerza la lectura de vigor en vegetacion densa, donde NDVI puede saturarse.',
    palette: 'orange',
  },
};

const DEMO_TERMS = [
  ['NDMI', 'Humedad del dosel', 'Indice principal para estres hidrico; usa NIR y SWIR para estimar agua en hoja/dosel.'],
  ['NDVI', 'Vigor vegetal', 'Compara NIR y rojo para medir verdor y actividad fotosintetica general.'],
  ['NDWI', 'Agua en vegetacion', 'Ayuda a observar contenido de agua y cambios hidricos en la cubierta vegetal.'],
  ['NDRE', 'Clorofila temprana', 'Usa red edge para detectar estres antes de que cambie mucho el verdor visible.'],
  ['EVI', 'Vigor en zonas densas', 'Indice de vegetacion mas estable cuando el dosel es muy cerrado.'],
  ['GP', 'Gaussian Process', 'Modelo que aprende una curva esperada y una banda de incertidumbre para cada fecha.'],
  ['MLE', 'Maxima verosimilitud', 'Ajuste que busca los parametros que hacen mas probable el historial observado.'],
  ['z-score', 'Desviaciones estandar', 'Dice que tan lejos esta la observacion de lo esperado por el modelo.'],
  ['Percentil', 'Posicion probabilistica', 'Convierte una senal estadistica a una escala interpretable de 0 a 100%.'],
  ['SAM2', 'Segmentacion visual', 'Aqui se usa como vista pixelada/preliminar para ubicar estres dentro de la parcela.'],
  ['E3 Stacking', 'Ensamble final', 'Combina Random Forest, XGBoost y SVM con una regresion logistica.'],
  ['Verosimilitud', 'Probabilidad del dato', 'Mide que tan probable es observar los indices actuales bajo la distribucion esperada.'],
];

function initDemo() {
  const btn = document.getElementById('btn-demo-load');
  const sel = document.getElementById('demo-parcel');
  if (btn) btn.addEventListener('click', () => loadDemoParcel());
  if (sel) sel.addEventListener('change', () => updateDemoLoadState());
  renderDemoGlossary();
}

function populateDemoParcelSelect() {
  const sel = document.getElementById('demo-parcel');
  if (!sel) return;

  if (!parcels.length) {
    setDemoParcelSelectState('Sin parcelas cargadas');
    return;
  }

  const previousValue = sel.value;
  sel.innerHTML = parcels
    .map(p => {
      const id = escapeHtml(p.parcel_id);
      const state = escapeHtml(p.state || 'Jalisco');
      return `<option value="${id}">${id} - ${state}</option>`;
    })
    .join('');
  sel.disabled = false;

  if (previousValue && parcels.some(p => p.parcel_id === previousValue)) {
    sel.value = previousValue;
  } else if (parcels.some(p => p.parcel_id === 'H1')) {
    sel.value = 'H1';
  } else if (parcels[0]?.parcel_id) {
    sel.value = parcels[0].parcel_id;
  }

  updateDemoLoadState();
}

function setDemoParcelSelectState(message) {
  const sel = document.getElementById('demo-parcel');
  if (!sel) return;
  sel.innerHTML = `<option value="">${escapeHtml(message)}</option>`;
  sel.disabled = true;
  updateDemoLoadState();
}

function updateDemoLoadState() {
  const sel = document.getElementById('demo-parcel');
  const btn = document.getElementById('btn-demo-load');
  if (!sel || !btn) return;
  btn.disabled = sel.disabled || !sel.value;
}

async function ensureDemoLoaded() {
  if (!parcels.length) {
    setDemoParcelSelectState('Cargando parcelas...');
    await loadParcels();
  } else {
    populateDemoParcelSelect();
  }
  if (!demoLoadedParcel && document.getElementById('demo-parcel')?.value) {
    await loadDemoParcel();
  }
}

async function loadDemoParcel() {
  const sel = document.getElementById('demo-parcel');
  const btn = document.getElementById('btn-demo-load');
  const parcelId = sel?.value;
  if (!parcelId) return;

  btn.disabled = true;
  btn.textContent = 'Cargando...';
  setText('demo-status', 'Cargando artefactos locales y preparando visualizaciones...');

  try {
    const res = await fetch(`${API}/demo/parcel/${encodeURIComponent(parcelId)}`);
    const demo = await readJsonOrThrow(res);
    demoLoadedParcel = demo.parcel_id;
    renderDemoBase(demo);

    const [diagnosis, trend, likelihood] = await Promise.allSettled([
      loadDemoDiagnosis(parcelId),
      loadDemoTrend(parcelId),
      loadDemoLikelihood(parcelId),
    ]);
    if (diagnosis.status === 'rejected') {
      renderDemoDiagnosisError(diagnosis.reason);
    }
    if (trend.status === 'rejected') {
      renderDemoTrendError(trend.reason);
    }
    if (likelihood.status === 'rejected') {
      renderDemoLikelihoodMessage(`No se pudo calcular verosimilitud: ${likelihood.reason.message}`);
    }

    setText('demo-status', `Demo lista para ${demo.parcel_id}. Scroll hacia abajo para ver todo el analisis.`);
  } catch (e) {
    setText('demo-status', `No se pudo cargar la demo: ${e.message}`);
    showToast(`Error demo: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Cargar ejemplo';
  }
}

function renderDemoBase(demo) {
  demoCurrentMask = demo.mask;
  const src = demo.source;
  setText('demo-source-name', src.satellite);
  setText(
    'demo-source-meta',
    `${src.n_dates} observaciones | ${src.date_min} a ${src.date_max} | patch ${src.shape.height}x${src.shape.width}x${src.shape.channels}`,
  );
  setText(
    'demo-composite-caption',
    `Parcela ${demo.parcel_id}, fecha ${src.latest_date}. Composicion visual generada desde indices procesados.`,
  );

  drawCompositeCanvas('demo-composite', demo.indices);
  renderDemoIndexGrid(demo);
  renderDemoSeriesChart(demo);
  renderDemoFeatures(demo);
  drawMaskCanvas('demo-mask', demo.mask);
  renderDemoMaskLegend(demo.mask);
  renderDemoSam2Explanation(demo);
}

function renderDemoIndexGrid(demo) {
  const grid = document.getElementById('demo-index-grid');
  grid.innerHTML = Object.entries(DEMO_INDEX_META).map(([key, meta]) => `
    <div class="demo-index-card">
      <div class="demo-index-head">
        <strong>${meta.title}</strong>
        <span>${meta.bands}</span>
      </div>
      <canvas id="demo-idx-${key}"></canvas>
      <p>${meta.meaning}</p>
    </div>
  `).join('');

  Object.entries(DEMO_INDEX_META).forEach(([key, meta]) => {
    drawGridCanvas(`demo-idx-${key}`, demo.indices[key].grid, meta.palette);
  });
}

function renderDemoSeriesChart(demo) {
  const ctx = document.getElementById('demo-series-chart').getContext('2d');
  if (demoSeriesChart) demoSeriesChart.destroy();

  const labels = demo.timeseries.dates;
  const colors = {
    NDVI: '#388E3C',
    NDWI: '#1565C0',
    NDMI: '#00897B',
    NDRE: '#6A1B9A',
    EVI: '#EF6C00',
  };

  demoSeriesChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: Object.keys(DEMO_INDEX_META).map(key => ({
        label: key,
        data: demo.indices[key].series,
        borderColor: colors[key],
        borderWidth: key === 'NDMI' ? 2 : 1,
        tension: .25,
        pointRadius: 0,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } } },
      scales: {
        x: { ticks: { display: false }, grid: { display: false } },
        y: { min: 0, max: 1, ticks: { font: { size: 10 } }, grid: { color: '#f0f0ec' } },
      },
    },
  });
}

function renderDemoFeatures(demo) {
  const el = document.getElementById('demo-feature-grid');
  const features = demo.features.by_index;
  el.innerHTML = ['NDMI', 'NDVI', 'NDWI', 'NDRE', 'EVI'].map(key => {
    const f = features[key];
    return `
      <div class="demo-feature">
        <strong>${key}</strong>
        <span>media ${fmt(f.mean)} | std ${fmt(f.std)}</span>
        <span>p25 ${fmt(f.p25)} | p75 ${fmt(f.p75)}</span>
        <span>tendencia ${f.trend >= 0 ? '+' : ''}${f.trend}</span>
      </div>
    `;
  }).join('');
}

async function loadDemoDiagnosis(parcelId) {
  const res = await fetch(`${API}/analyze/parcel`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ parcel_id: parcelId, skip_llm: true }),
  });
  const data = await readJsonOrThrow(res);
  renderDemoDiagnosis(data);
}

function renderDemoDiagnosis(data) {
  const stress = data.stress || {};
  const el = document.getElementById('demo-diagnosis');
  const cls = Number(stress.class);
  const label = DEMO_STRESS_LABELS[cls] || stress.label || 'Sin clasificar';
  const confidence = Math.round(clamp01(stress.confidence) * 100);
  const dominant = getDemoDominantMaskClass(demoCurrentMask);
  const segmentText = dominant
    ? `${dominant.label} (${Math.round(dominant.percent)}% de pixeles validos)`
    : 'Sin mascara disponible';

  el.innerHTML = `
    <div class="demo-diagnosis-card">
      <span>Diagnostico global E3</span>
      <strong>${escapeHtml(label)} (${confidence}%)</strong>
    </div>
    <div class="demo-diagnosis-card">
      <span>Segmentacion dominante</span>
      <strong>${escapeHtml(segmentText)}</strong>
    </div>
    <div class="demo-diagnosis-card">
      <span>Ventana temporal</span>
      <strong>${data.n_dates} fechas disponibles</strong>
    </div>
  `;

  const probValues = getDemoProbabilityValues(stress);
  const e3Prob = Number.isFinite(cls) ? probValues[cls] : null;
  const maskShare = Number.isFinite(cls) ? getDemoMaskShare(demoCurrentMask, cls) : null;
  const classDistance = dominant && Number.isFinite(cls) ? Math.abs(cls - dominant.classId) : null;
  const percentGap = e3Prob != null && maskShare != null
    ? Math.abs(e3Prob * 100 - maskShare)
    : null;
  const showProbChart = classDistance === 0 && percentGap != null && percentGap <= 25;

  renderDemoConsistencyNote({ dominant, cls, confidence, maskShare, percentGap, showProbChart });

  const wrap = document.getElementById('demo-prob-wrap');
  if (!showProbChart) {
    if (demoProbChart) {
      demoProbChart.destroy();
      demoProbChart = null;
    }
    wrap?.classList.add('hidden');
    return;
  }

  wrap?.classList.remove('hidden');
  const labels = DEMO_STRESS_LABELS;
  const probs = labels.map((_, i) => Math.round((probValues[i] || 0) * 100));
  const ctx = document.getElementById('demo-prob-chart').getContext('2d');
  if (demoProbChart) demoProbChart.destroy();
  demoProbChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: probs,
        backgroundColor: ['#388E3C', '#F9A825', '#C62828'],
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 10 } } },
        y: { min: 0, max: 100, ticks: { callback: v => `${v}%`, font: { size: 10 } } },
      },
    },
  });
}

function renderDemoDiagnosisError(err) {
  const note = document.getElementById('demo-consistency-note');
  if (note) {
    note.className = 'demo-consistency-note complementary';
    note.textContent = 'La comparacion se omite porque no se pudo ejecutar el diagnostico global.';
  }
  const wrap = document.getElementById('demo-prob-wrap');
  wrap?.classList.add('hidden');
  if (demoProbChart) {
    demoProbChart.destroy();
    demoProbChart = null;
  }
  document.getElementById('demo-diagnosis').innerHTML = `
    <div class="demo-mini-result">No se pudo calcular diagnostico: ${escapeHtml(err.message)}</div>
  `;
}

function renderDemoConsistencyNote({ dominant, cls, confidence, maskShare, percentGap, showProbChart }) {
  const note = document.getElementById('demo-consistency-note');
  if (!note) return;

  note.className = 'demo-consistency-note';
  if (!dominant || !Number.isFinite(cls)) {
    note.classList.add('complementary');
    note.textContent = 'Lecturas complementarias: aun no hay una mascara valida para comparar contra el diagnostico global.';
    return;
  }

  const globalLabel = DEMO_STRESS_LABELS[cls] || 'Sin clasificar';
  const base = `Diagnostico global: ${globalLabel} (${confidence}%). Segmentacion dominante: ${dominant.label} (${Math.round(dominant.percent)}%).`;

  if (showProbChart) {
    note.classList.add('consistent');
    note.textContent = `${base} Las clases coinciden y la barra de probabilidades se muestra como apoyo visual.`;
    return;
  }

  note.classList.add('complementary');
  if (cls === dominant.classId) {
    const gapText = percentGap == null ? '' : ` La diferencia numerica es de ${Math.round(percentGap)} puntos, asi que no se compara como porcentaje directo.`;
    note.textContent = `${base} Coinciden en clase, pero siguen siendo lecturas complementarias: E3 resume la ventana temporal completa; SAM2-preview muestra distribucion espacial por NDMI reciente.${gapText}`;
    return;
  }

  if (Math.abs(cls - dominant.classId) <= 1) {
    const gapText = percentGap == null ? '' : ` La diferencia numerica es de ${Math.round(percentGap)} puntos, asi que no se compara como porcentaje directo.`;
    note.textContent = `${base} Son lecturas cercanas pero complementarias: E3 resume la ventana temporal completa; SAM2-preview muestra distribucion espacial por NDMI reciente.${gapText}`;
    return;
  }

  const shareText = maskShare == null ? '' : ` La clase global ocupa ${Math.round(maskShare)}% de los pixeles validos en la mascara.`;
  note.textContent = `${base} Estas lecturas son complementarias: E3 resume la ventana temporal completa; SAM2-preview muestra distribucion espacial por NDMI reciente.${shareText}`;
}

function getDemoProbabilityValues(stress) {
  const probs = stress.probabilities || {};
  const keys = Object.keys(probs);
  const direct = DEMO_STRESS_LABELS.map(label => {
    const exact = toDemoProbability(probs[label]);
    if (exact != null) return exact;
    const normalizedLabel = normalizeDemoStressLabel(label);
    const foundKey = keys.find(key => normalizeDemoStressLabel(key) === normalizedLabel);
    return foundKey ? toDemoProbability(probs[foundKey]) : null;
  });

  if (direct.every(value => value != null)) return direct;

  const ordered = Object.values(probs).map(toDemoProbability);
  return DEMO_STRESS_LABELS.map((_, i) => direct[i] ?? ordered[i] ?? 0);
}

function toDemoProbability(value) {
  const n = Number(value);
  return Number.isFinite(n) ? clamp01(n) : null;
}

function normalizeDemoStressLabel(value) {
  return String(value || '')
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z]/g, '');
}

function getDemoDominantMaskClass(mask) {
  const entries = DEMO_STRESS_COUNT_KEYS.map((key, classId) => ({
    key,
    classId,
    label: DEMO_STRESS_LABELS[classId],
    count: Number(mask?.counts?.[key]) || 0,
  }));
  const validTotal = entries.reduce((sum, item) => sum + item.count, 0);
  if (!validTotal) return null;
  const dominant = entries.reduce((best, item) => (item.count > best.count ? item : best), entries[0]);
  return {
    ...dominant,
    percent: (dominant.count / validTotal) * 100,
    validTotal,
  };
}

function getDemoMaskShare(mask, classId) {
  const entries = DEMO_STRESS_COUNT_KEYS.map(key => Number(mask?.counts?.[key]) || 0);
  const validTotal = entries.reduce((sum, count) => sum + count, 0);
  if (!validTotal || classId < 0 || classId >= entries.length) return null;
  return (entries[classId] / validTotal) * 100;
}

async function loadDemoTrend(parcelId) {
  const res = await fetch(`${API}/parcels/${encodeURIComponent(parcelId)}/trend?index=NDMI&horizon=5`);
  const data = await readJsonOrThrow(res);
  renderDemoTrend(data);
}

function renderDemoTrend(data) {
  const ctx = document.getElementById('demo-gp-chart').getContext('2d');
  if (demoGpChart) demoGpChart.destroy();
  const historyPoints = data.history.days.map((day, i) => ({ x: day, y: data.history.values[i] }));
  const gpPoints = data.gp_curve.days.map((day, i) => ({ x: day, y: data.gp_curve.mean[i] }));
  demoGpChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'Historial NDMI',
          data: historyPoints,
          borderColor: '#388E3C',
          backgroundColor: '#388E3C',
          pointRadius: 1.5,
          showLine: false,
        },
        {
          label: 'Media GP',
          data: gpPoints,
          borderColor: '#1565C0',
          pointRadius: 0,
          borderWidth: 2,
          tension: .25,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } } },
      scales: {
        x: { type: 'linear', ticks: { display: false }, grid: { display: false } },
        y: { ticks: { font: { size: 10 } }, grid: { color: '#f0f0ec' } },
      },
    },
  });

  const last = data.last_observation;
  const group = data.group
    ? ` Grupo ${data.group.group_id}: ${data.group.n_members} parcelas similares.`
    : ' Sin grupo de terreno disponible.';
  setText(
    'demo-gp-summary',
    `Ultima observacion NDMI ${last.value} | z-score ${last.z} | clase ${last.label}.${group}`,
  );
}

function renderDemoTrendError(err) {
  setText('demo-gp-summary', `No se pudo calcular GP/MLE: ${err.message}`);
}

async function loadDemoLikelihood(parcelId) {
  const res = await fetch(`${API}/parcels/${encodeURIComponent(parcelId)}/likelihood`);
  if (res.status === 503) {
    renderDemoLikelihoodMessage('Modelo no entrenado en este equipo. La demo conserva la explicacion: la red estima mu/sigma y calcula un percentil de verosimilitud.');
    return;
  }
  const data = await readJsonOrThrow(res);
  const stress = data.stress;
  const ndmi = stress.distribution.NDMI;
  renderDemoLikelihoodMessage(
    `${stress.stress_label} | percentil ${stress.likelihood_percentile}% | z combinado ${stress.combined_z_score}. NDMI observado ${ndmi.observed}, esperado ${ndmi.mu} +/- ${ndmi.sigma}.`,
  );
}

function renderDemoLikelihoodMessage(message) {
  setText('demo-likelihood-summary', message);
}

function renderDemoMaskLegend(mask) {
  const total = Math.max(1, mask.width * mask.height);
  const items = [
    ['Sin estres', mask.counts.sin_estres, '#388E3C'],
    ['Moderado', mask.counts.moderado, '#F9A825'],
    ['Severo', mask.counts.severo, '#C62828'],
    ['Sin dato', mask.counts.sin_dato, '#9E9E9E'],
  ];
  document.getElementById('demo-mask-legend').innerHTML = items.map(([label, count, color]) => `
    <span><b style="color:${color}">${label}</b>: ${Math.round((count / total) * 100)}%</span>
  `).join('');
}

function renderDemoSam2Explanation(demo) {
  const el = document.getElementById('demo-sam2-gpmle-note');
  if (!el) return;

  const dominant = getDemoDominantMaskClass(demo.mask);
  const thresholds = demo.mask?.thresholds || {};
  const group = demo.group
    ? ` Al usar Grupo (Terreno similar), GP/MLE compara contra ${demo.group.n_members} parcelas del grupo ${demo.group.group_id}.`
    : ' Si no existe grupo de terreno, GP/MLE usa solo la historia local disponible.';
  const domText = dominant
    ? ` En este ejemplo domina ${dominant.label.toLowerCase()} con ${Math.round(dominant.percent)}% de pixeles validos.`
    : '';
  const thresholdText = thresholds.moderado != null && thresholds.severo != null
    ? ` Umbrales NDMI: moderado <= ${thresholds.moderado}, severo <= ${thresholds.severo}.`
    : '';

  el.textContent = `GP + MLE recalibra la lectura temporal: compara la mascara reciente contra una curva esperada y su incertidumbre.${domText}${thresholdText}${group}`;
}

function renderDemoGlossary() {
  const grid = document.getElementById('demo-term-grid');
  if (!grid) return;
  grid.innerHTML = DEMO_TERMS.map(([term, short, desc]) => `
    <div class="demo-term-card">
      <strong>${term} <em>${short}</em></strong>
      <p>${desc}</p>
    </div>
  `).join('');
}

function drawCompositeCanvas(id, indices) {
  const ndvi = indices.NDVI.grid;
  const ndmi = indices.NDMI.grid;
  const evi = indices.EVI.grid;
  const ndwi = indices.NDWI.grid;
  const canvas = document.getElementById(id);
  const h = ndvi.length;
  const w = ndvi[0].length;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      const vigor = clamp01(ndvi[y][x]);
      const moisture = clamp01(ndmi[y][x]);
      const density = clamp01(evi[y][x]);
      const water = clamp01(ndwi[y][x]);
      img.data[idx] = Math.round(35 + density * 165);
      img.data[idx + 1] = Math.round(45 + vigor * 195);
      img.data[idx + 2] = Math.round(35 + moisture * 120 + water * 55);
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function drawGridCanvas(id, grid, palette) {
  const canvas = document.getElementById(id);
  const h = grid.length;
  const w = grid[0].length;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      const [r, g, b] = paletteColor(clamp01(grid[y][x]), palette);
      img.data[idx] = r;
      img.data[idx + 1] = g;
      img.data[idx + 2] = b;
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function drawMaskCanvas(id, mask) {
  const canvas = document.getElementById(id);
  canvas.width = mask.width;
  canvas.height = mask.height;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(mask.width, mask.height);
  const colors = {
    '-1': [158, 158, 158, 80],
    0: [56, 142, 60, 210],
    1: [249, 168, 37, 220],
    2: [198, 40, 40, 235],
  };
  for (let y = 0; y < mask.height; y++) {
    for (let x = 0; x < mask.width; x++) {
      const idx = (y * mask.width + x) * 4;
      const rgba = colors[mask.classes[y][x]] || colors['-1'];
      img.data[idx] = rgba[0];
      img.data[idx + 1] = rgba[1];
      img.data[idx + 2] = rgba[2];
      img.data[idx + 3] = rgba[3];
    }
  }
  ctx.putImageData(img, 0, 0);
}

function paletteColor(value, palette) {
  const stops = {
    green: [[244, 241, 198], [46, 125, 50]],
    blue: [[232, 245, 255], [21, 101, 192]],
    teal: [[232, 245, 233], [0, 121, 107]],
    purple: [[246, 238, 249], [106, 27, 154]],
    orange: [[255, 243, 224], [230, 81, 0]],
  }[palette] || [[245, 245, 245], [40, 40, 40]];
  return stops[0].map((c, i) => Math.round(c + (stops[1][i] - c) * value));
}

function clamp01(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

/* ==============================================================
   LIKELIHOOD (Experimento E -- Red Neuronal Probabilistica)
============================================================== */
const LK_INDEX_ORDER = ['NDVI', 'NDWI', 'NDMI', 'NDRE', 'EVI'];

async function calcLikelihood() {
  const parcelId = document.getElementById('trend-parcel').value;
  if (!parcelId) {
    showToast('Selecciona una parcela primero', 'error');
    return;
  }

  const btn = document.getElementById('btn-likelihood');
  btn.disabled = true;
  btn.textContent = 'Calculando...';

  try {
    const res = await fetch(`${API}/parcels/${parcelId}/likelihood`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      if (res.status === 503) {
        showToast('Modelo no entrenado aun -- ejecuta: make train-likelihood', 'error');
        return;
      }
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    renderLikelihoodResult(data);
  } catch (e) {
    showToast(`Error al calcular verosimilitud: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Calcular verosimilitud';
  }
}

function renderLikelihoodResult(data) {
  const { stress } = data;
  const cls = stress.stress_class;
  const pct = stress.likelihood_percentile;

  document.getElementById('likelihood-result').classList.remove('hidden');

  const badge = document.getElementById('lk-badge');
  badge.textContent = stress.stress_label;
  badge.className   = `stress-badge s${cls}`;

  document.getElementById('lk-pct-value').textContent = `${pct.toFixed(1)}%`;
  const bar = document.getElementById('lk-pct-bar');
  bar.style.width      = `${Math.min(pct, 100)}%`;
  bar.style.background = COLOR[cls];

  const grid = document.getElementById('lk-dist-grid');
  grid.innerHTML = LK_INDEX_ORDER.map(idx => {
    const d    = stress.distribution[idx];
    const z    = d.z_score;
    const zCls = z < -2 ? 'z-severe' : z < -1 ? 'z-moderate' : z > 1 ? 'z-high' : 'z-ok';
    const zSign = z >= 0 ? '+' : '';
    const key  = idx === 'NDMI';
    return `<tr class="${key ? 'lk-key-row' : ''}">
      <td class="lk-idx-name">${idx}${key ? ' <span class="idx-tag">clave</span>' : ''}</td>
      <td class="lk-num">${d.observed.toFixed(4)}</td>
      <td class="lk-num">${d.mu.toFixed(4)}</td>
      <td class="lk-num lk-sigma">+/-${d.sigma.toFixed(4)}</td>
      <td class="lk-num"><span class="lk-z ${zCls}">${zSign}${z.toFixed(3)}</span></td>
    </tr>`;
  }).join('');

  renderLikelihoodChart(stress.distribution);
}

function renderLikelihoodChart(dist) {
  if (lkChart) { lkChart.destroy(); lkChart = null; }

  const labels  = LK_INDEX_ORDER;
  const zScores = labels.map(k => dist[k].z_score);
  const barColors = zScores.map(z =>
    z < -2  ? 'rgba(198,40,40,.85)'  :
    z < -1  ? 'rgba(249,168,37,.85)' :
    z >= 0  ? 'rgba(56,142,60,.75)'  :
              'rgba(56,142,60,.45)'
  );

  const ctx = document.getElementById('lk-chart').getContext('2d');
  lkChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'z-score',
        data: zScores,
        backgroundColor: barColors,
        borderRadius: 4,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      animation: { duration: 300 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const z = ctx.raw;
              const sign = z >= 0 ? '+' : '';
              return `z = ${sign}${z.toFixed(3)}`;
            },
          },
        },
        annotation: {
          annotations: {
            line1pos: { type: 'line', yMin:  1, yMax:  1, borderColor: '#9E9E9E', borderWidth: 1, borderDash: [4,4] },
            line1neg: { type: 'line', yMin: -1, yMax: -1, borderColor: '#9E9E9E', borderWidth: 1, borderDash: [4,4] },
            line2pos: { type: 'line', yMin:  2, yMax:  2, borderColor: '#757575', borderWidth: 1, borderDash: [2,3] },
            line2neg: { type: 'line', yMin: -2, yMax: -2, borderColor: '#757575', borderWidth: 1, borderDash: [2,3] },
          },
        },
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 11, weight: '600' } } },
        y: {
          grid: { color: '#f0f0ec' },
          ticks: { font: { size: 10 } },
          title: { display: true, text: 'desviaciones estandar (z)', font: { size: 10 } },
        },
      },
    },
  });
}

/* ==============================================================
   UTILS
============================================================== */
function fmt(v) {
  return (v != null && !isNaN(v)) ? (+v).toFixed(4) : '-';
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[ch]));
}

function showToast(msg, type = '') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toasts').appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function compactNumber(value) {
  return new Intl.NumberFormat('es-MX', {
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(value);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
