/* ─────────────────────────────────────────────────────────────
   AguaVerde — Dashboard JS
   ───────────────────────────────────────────────────────────── */

const API       = '';   // same origin — FastAPI at /
const MAP_CTR   = [20.5, -103.5];
const MAP_ZOOM  = 9;

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
const LABEL = { 0: '🟢 Sin estrés', 1: '🟡 Estrés moderado', 2: '🔴 Estrés severo' };

/* ── State ──────────────────────────────────────────────────── */
let map, trendChart, sam2Layer;
let markers  = {};   // parcel_id → Leaflet marker
let results  = {};   // parcel_id → analysis result
let sam2Masks = {};  // parcel_id -> pixel mask result
let sam2Visible = { 0: true, 1: true, 2: true, pending: true };
let parcels  = [];   // all parcels from /parcels
let photoB64 = null;
let photoMime = 'image/jpeg';
let filterCls = null;   // null = show all
let scanning  = false;

/* ── Bootstrap ──────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  initTabs();
  initSam2();
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
    const data = await readJsonOrThrow(res);
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
      await readJsonOrThrow(res);
    }
    const data = await readJsonOrThrow(res);
    results[parcelId] = data;
    setMarkerClass(parcelId, data.stress.class);
    updateStats();
    applyFilter();
    if (activeTab() === 'sam2') renderSam2View();
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
    reportEl.textContent  = 'Analisis en modo rapido - sin reporte LLM.';
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
    overlay.textContent = `⚡ Escaneando ${done + 1}/${toScan.length} · ${p.parcel_id}`;
    await analyzeParcel(p.parcel_id, true);
    done++;
    await sleep(80);   // avoid overwhelming the server
  }

  overlay.remove();
  btn.disabled = false;
  scanning = false;
  showToast(`✅ Escaneo completo: ${done} parcelas analizadas`, 'success');
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
  if (name === 'sam2') {
    renderSam2View();
  } else if (sam2Layer) {
    sam2Layer.clearLayers();
  }
}

function activeTab() {
  return document.querySelector('.tab.active')?.dataset.tab || 'parcel';
}

function initSam2() {
  const refreshBtn = document.getElementById('btn-sam2-refresh');
  const scanBtn = document.getElementById('btn-sam2-scan-all');
  if (refreshBtn) refreshBtn.addEventListener('click', renderSam2View);
  if (scanBtn) scanBtn.addEventListener('click', scanSam2All);
  document.querySelectorAll('[data-sam2-filter]').forEach(input => {
    input.addEventListener('change', () => {
      const key = input.dataset.sam2Filter;
      sam2Visible[key] = input.checked;
      renderSam2View();
    });
  });
}

function renderSam2View() {
  if (!sam2Layer) return;

  sam2Layer.clearLayers();

  const analyzed = Object.values(results).filter(Boolean);
  const masks = Object.values(sam2Masks);
  const counts = { 0: 0, 1: 0, 2: 0 };

  parcels.forEach(parcel => {
    const data = results[parcel.parcel_id];
    const mask = sam2Masks[parcel.parcel_id];
    const cls = data?.stress?.class ?? null;
    const color = COLOR[cls];

    if (mask) {
      addMaskCounts(mask, counts);
      if (!maskHasVisiblePixels(mask)) return;
      L.imageOverlay(maskToDataUrl(mask), parcelCellBounds(parcel), {
        opacity: 0.72,
        interactive: true,
      }).addTo(sam2Layer).on('click', () => focusSam2Parcel(parcel));
      return;
    }

    if (!sam2Visible.pending) return;

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
    cell.on('click', () => focusSam2Parcel(parcel));
  });

  const coverage = parcels.length
    ? Math.round((masks.length / parcels.length) * 100)
    : 0;

  setText('sam2-total', masks.length);
  setText('sam2-coverage', `${coverage}%`);
  setText('sam2-green', compactNumber(counts[0]));
  setText('sam2-yellow', compactNumber(counts[1]));
  setText('sam2-red', compactNumber(counts[2]));
  setText('sam2-meta', `${masks.length} mascaras pixel`);

  const status = document.getElementById('sam2-status');
  if (status) {
    status.textContent = masks.length
      ? 'Raster pixel activo sobre el mapa'
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

  const pending = parcels.filter(p => !sam2Masks[p.parcel_id]);
  let done = 0;

  for (const p of pending) {
    overlay.textContent = `SAM2 pixeles ${done + 1}/${pending.length} · ${p.parcel_id}`;
    await loadSam2Mask(p.parcel_id);
    done++;
    if (done % 5 === 0) renderSam2View();
    await sleep(40);
  }

  overlay.remove();
  if (btn) btn.disabled = false;
  scanning = false;
  renderSam2View();
  showToast(`SAM2 pixel actualizado: ${done} mascaras generadas`, 'success');
}

function focusSam2Parcel(parcel) {
  map.flyTo([parcel.latitude, parcel.longitude], 14, { duration: 0.8 });
  const marker = markers[parcel.parcel_id];
  if (marker) marker.openTooltip();
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
    return mask && maskHasVisiblePixels(mask);
  });
  if (!items.length) {
    list.innerHTML = '<div class="sam2-empty">Sin mascaras activas</div>';
    return;
  }

  list.innerHTML = items
    .map(parcel => {
      const summary = summarizeMask(sam2Masks[parcel.parcel_id]);
      const cls = summary.majority;
      return `
        <button type="button" class="sam2-row s${cls}" data-parcel="${parcel.parcel_id}">
          <span>${parcel.parcel_id}</span>
          <strong>${summary.severePct}% severo</strong>
        </button>
      `;
    })
    .join('');

  list.querySelectorAll('.sam2-row').forEach(row => {
    row.addEventListener('click', () => {
      const parcel = parcels.find(p => p.parcel_id === row.dataset.parcel);
      if (parcel) focusSam2Parcel(parcel);
    });
  });
}

function addMaskCounts(mask, counts) {
  for (let y = 0; y < mask.height; y++) {
    for (let x = 0; x < mask.width; x++) {
      const cls = mask.classes[y][x];
      if ([0, 1, 2].includes(cls)) counts[cls]++;
    }
  }
}

function summarizeMask(mask) {
  const counts = { 0: 0, 1: 0, 2: 0 };
  addMaskCounts(mask, counts);
  const total = counts[0] + counts[1] + counts[2] || 1;
  const majority = [0, 1, 2].sort((a, b) => counts[b] - counts[a])[0];
  return {
    majority,
    severePct: Math.round((counts[2] / total) * 100),
  };
}

function maskHasVisiblePixels(mask) {
  for (let y = 0; y < mask.height; y++) {
    for (let x = 0; x < mask.width; x++) {
      if (sam2Visible[mask.classes[y][x]]) return true;
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

function sam2Tooltip(parcel, data) {
  if (!data) return `<b>${parcel.parcel_id}</b><br>Sin analizar`;
  return `<b>${parcel.parcel_id}</b><br>${data.stress.label}`;
}

function maskToDataUrl(mask) {
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
      const rgba = sam2Visible[cls]
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
