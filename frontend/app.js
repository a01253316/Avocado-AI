/* ─────────────────────────────────────────────────────────────
   AguaVerde — Dashboard JS
   ───────────────────────────────────────────────────────────── */

const API       = '';   // same origin — FastAPI at /
const MAP_CTR   = [20.5, -103.5];
const MAP_ZOOM  = 9;
const SCAN_N    = 30;   // parcels to batch-scan

const COLOR = { 0: '#388E3C', 1: '#F9A825', 2: '#C62828', null: '#9E9E9E' };
const LABEL = { 0: '🟢 Sin estrés', 1: '🟡 Estrés moderado', 2: '🔴 Estrés severo' };

/* ── State ──────────────────────────────────────────────────── */
let map, trendChart;
let markers  = {};   // parcel_id → Leaflet marker
let results  = {};   // parcel_id → analysis result
let parcels  = [];   // all parcels from /parcels
let photoB64 = null;
let photoMime = 'image/jpeg';
let filterCls = null;   // null = show all
let scanning  = false;

/* ── Bootstrap ──────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  initTabs();
  initForm();
  initFilters();
  loadParcels();
});

/* ══════════════════════════════════════════════════════════════
   MAP
══════════════════════════════════════════════════════════════ */
function initMap() {
  map = L.map('map', { center: MAP_CTR, zoom: MAP_ZOOM, zoomControl: true });
  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '© <a href="https://openstreetmap.org">OpenStreetMap</a> contributors © <a href="https://carto.com">CARTO</a>',
    subdomains: 'abcd', maxZoom: 20,
  }).addTo(map);
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

/* ── Ripple keyframe injection ─── */
const rippleStyle = document.createElement('style');
rippleStyle.textContent = `
  @keyframes ripple {
    0%   { transform: scale(.6); opacity: .6; }
    100% { transform: scale(2);  opacity: 0; }
  }`;
document.head.appendChild(rippleStyle);

/* ══════════════════════════════════════════════════════════════
   LOAD PARCELS
══════════════════════════════════════════════════════════════ */
async function loadParcels() {
  try {
    const res  = await fetch(`${API}/parcels?limit=200`);
    const data = await res.json();
    parcels = data.parcels;
    document.getElementById('v-total').textContent = data.total;

    parcels.forEach(p => {
      const m = L.marker([p.latitude, p.longitude], { icon: makeIcon(null) }).addTo(map);
      m.bindTooltip(`<b>${p.parcel_id}</b><br><span style="font-size:11px;color:#666">${p.state}</span>`, {
        direction: 'top', offset: [0, -6],
      });
      m.on('click', () => onMarkerClick(p));
      markers[p.parcel_id] = m;
    });

    showToast(`${data.total} parcelas cargadas`, 'success');
  } catch (e) {
    showToast('No se pudo conectar con el backend: ' + e.message, 'error');
  }
}

function onMarkerClick(p) {
  switchTab('parcel');

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

/* ══════════════════════════════════════════════════════════════
   ANALYSIS
══════════════════════════════════════════════════════════════ */
async function analyzeParcel(parcelId, skipLlm = false) {
  try {
    const res = await fetch(`${API}/analyze/parcel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ parcel_id: parcelId, skip_llm: skipLlm }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    results[parcelId] = data;
    setMarkerClass(parcelId, data.stress.class);
    updateStats();
    applyFilter();
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
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function analyzeParcelAndRender(parcelId, skipLlm = false) {
  const data = await analyzeParcel(parcelId, skipLlm);
  if (data) renderResult(data);
}

/* ══════════════════════════════════════════════════════════════
   RENDER
══════════════════════════════════════════════════════════════ */
function showLoadingCard(parcelId, lat, lon, state) {
  document.getElementById('empty-state').classList.add('hidden');
  document.getElementById('parcel-card').classList.remove('hidden');
  document.getElementById('parcel-id').textContent  = parcelId;
  document.getElementById('parcel-meta').textContent = (lat && lon)
    ? `${lat.toFixed(4)}, ${lon.toFixed(4)}${state ? ' · ' + state : ''}`
    : 'Cargando…';
  const badge = document.getElementById('stress-badge');
  badge.textContent = '…';
  badge.className   = 'stress-badge';
  document.getElementById('conf-value').textContent = '—';
  document.getElementById('conf-bar').style.width   = '0%';
  ['ndmi','ndvi','ndwi','ndre','evi'].forEach(k =>
    document.getElementById(`idx-${k}`).textContent = '…'
  );
  document.getElementById('trend-dir').textContent  = '—';
  document.getElementById('trend-dir').className    = 'trend-direction sin_datos';
  document.getElementById('report-body').innerHTML  =
    '<div class="loading-dots"><span></span><span></span><span></span></div>';
  document.getElementById('report-model').textContent = '';
  ['0','1','2'].forEach(i => {
    document.getElementById(`pb-${i}`).style.width = '0%';
    document.getElementById(`pct-${i}`).textContent = '—';
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
  document.getElementById('parcel-meta').textContent = parts.join(' · ');

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
    descendente: '⬇ Tendencia descendente',
    ascendente:  '⬆ Tendencia ascendente',
    estable:     '➡ Tendencia estable',
    sin_datos:   '— Sin datos suficientes',
  }[dir] || '—';
  dirEl.className = `trend-direction ${dir}`;
  renderTrendChart(trend.windows || []);

  // Probabilities
  const probs = stress.probabilities;
  const keys  = ['Sin estrés', 'Moderado', 'Severo'];
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
    reportEl.textContent  = 'Análisis en modo rápido — sin reporte Claude.';
    reportEl.className    = 'report-body fallback';
    modelEl.textContent   = '';
  }
}

/* ── Trend chart ─────────────────────────────────────────────── */
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

/* ══════════════════════════════════════════════════════════════
   STATS & FILTER
══════════════════════════════════════════════════════════════ */
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
  if (markers[parcelId]) markers[parcelId].setIcon(makeIcon(cls));
}

function applyFilter() {
  Object.entries(markers).forEach(([id, m]) => {
    const res = results[id];
    const cls = res ? res.stress.class : null;
    if (filterCls === null || cls === null || cls === filterCls) {
      if (!map.hasLayer(m)) map.addLayer(m);
    } else {
      if (map.hasLayer(m)) map.removeLayer(m);
    }
  });
}

/* ══════════════════════════════════════════════════════════════
   SCAN ALL
══════════════════════════════════════════════════════════════ */
async function scanAll() {
  if (scanning) return;
  scanning = true;

  const btn = document.getElementById('btn-scan');
  btn.disabled = true;

  const overlay = document.createElement('div');
  overlay.className = 'scan-overlay';
  document.querySelector('.map-container').appendChild(overlay);

  const toScan = parcels.filter(p => !results[p.parcel_id]).slice(0, SCAN_N);
  let done = 0;

  for (const p of toScan) {
    overlay.textContent = `⚡ Escaneando ${done + 1}/${toScan.length} · ${p.parcel_id}`;
    await analyzeParcel(p.parcel_id, true);
    done++;
    await sleep(80);   // avoid overwhelming the server
  }

  overlay.remove();
  btn.disabled = false;
  scanning = false;
  showToast(`✅ Escaneo completo: ${done} parcelas analizadas`, 'success');
}

/* ══════════════════════════════════════════════════════════════
   TABS
══════════════════════════════════════════════════════════════ */
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
}

/* ══════════════════════════════════════════════════════════════
   FORM (nueva ubicación)
══════════════════════════════════════════════════════════════ */
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
      showToast('Ingresa coordenadas válidas', 'error');
      return;
    }
    if (lat < 15 || lat > 25 || lon < -108 || lon > -95) {
      showToast('Las coordenadas están fuera de Jalisco / México', 'info');
    }

    btn.disabled    = true;
    btn.textContent = '⏳ Analizando…';

    switchTab('parcel');
    showLoadingCard('Nueva ubicación', lat, lon, null);

    let tempMarker = L.marker([lat, lon], { icon: makeIcon(null, true) }).addTo(map);
    map.flyTo([lat, lon], 13, { duration: 1.2 });

    try {
      const data = await analyzeCoords(lat, lon, photoB64, photoMime, skip);
      renderResult(data);
      const cls = data.stress.class;
      tempMarker.setIcon(makeIcon(cls, false));
      tempMarker.bindPopup(`
        <b>📍 ${lat.toFixed(4)}, ${lon.toFixed(4)}</b><br>
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
      btn.textContent = '🔍 Analizar ubicación';
    }
  });
}

/* ══════════════════════════════════════════════════════════════
   FILTERS
══════════════════════════════════════════════════════════════ */
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

/* ══════════════════════════════════════════════════════════════
   UTILS
══════════════════════════════════════════════════════════════ */
function fmt(v) {
  return (v != null && !isNaN(v)) ? (+v).toFixed(4) : '—';
}

function showToast(msg, type = '') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toasts').appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
