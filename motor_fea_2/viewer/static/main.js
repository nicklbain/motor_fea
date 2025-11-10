const canvas = document.getElementById("mesh-canvas");
const ctx = canvas.getContext("2d");
const overlay = document.getElementById("overlay-message");
const statusBar = document.getElementById("status-bar");
const summaryContent = document.getElementById("summary-content");
const cellInfoContent = document.getElementById("cell-info-content");
const caseSelect = document.getElementById("case-select");
const refreshBtn = document.getElementById("refresh-cases");
const resetBtn = document.getElementById("reset-view");
const layerMaterialsToggle = document.getElementById("layer-materials");
const layerTargetToggle = document.getElementById("layer-target");
const layerMeshToggle = document.getElementById("layer-mesh");
const layerMagnetizationToggle = document.getElementById("layer-magnetization");
const layerJfreeToggle = document.getElementById("layer-jfree");
const layerBmagToggle = document.getElementById("layer-bmag");
const layerBvecToggle = document.getElementById("layer-bvec");
const bmagScaleSelect = document.getElementById("bmag-scale");
const bmagScaleControl = document.getElementById("bmag-scale-control");

const state = {
  mesh: null,
  edges: [],
  caseName: null,
  bField: null,
  centroids: [],
  designShapes: [],
  viewport: { width: 0, height: 0 },
  view: {
    centerX: 0,
    centerY: 0,
    baseScale: 1,
    zoom: 1,
    offsetX: 0,
    offsetY: 0,
  },
  layers: {
    materials: true,
    meshLines: true,
    magnetization: false,
    jfree: false,
    bMagnitude: false,
    bVectors: false,
    targetShapes: false,
  },
  bMagnitudeScale: "log",
  dragging: false,
  dragPointerId: null,
  lastPointer: { x: 0, y: 0 },
  dragMoved: false,
  selectedCellIndex: null,
};

const VIEWER_EVENT_PREFIX = "meshviewer:";

function emitViewerEvent(name, detail = {}) {
  window.dispatchEvent(new CustomEvent(`${VIEWER_EVENT_PREFIX}${name}`, { detail }));
}

const REGION_COLORS = {
  0: "#f7fbff", // air
  1: "#ffd166", // magnet (warm gold)
  2: "#a2a6af", // steel / iron
};
const WIRE_COLOR = "#b87333";
const EDGE_COLOR = "#000";
const MAG_ARROW_COLOR = "#c2185b";
const B_VECTOR_COLOR = "#005cb2";
const SELECTED_CELL_STROKE = "#ff4081";
const SELECTED_CELL_FILL = "rgba(255, 64, 129, 0.2)";
const TARGET_MAGNET_COLOR = "#c56b1a";
const TARGET_STEEL_COLOR = "#4b5563";
const TARGET_WIRE_COLOR = "#a62019";
const B_MAG_MIN_COLOR = { r: 230, g: 244, b: 255 };
const B_MAG_MAX_COLOR = { r: 178, g: 24, b: 43 };
const JFREE_NEG_COLOR = { r: 33, g: 94, b: 189 };
const JFREE_POS_COLOR = { r: 215, g: 38, b: 61 };
const JFREE_ZERO_COLOR = { r: 255, g: 255, b: 255 };

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  state.viewport.width = rect.width;
  state.viewport.height = rect.height;
  render();
}

window.addEventListener("resize", resizeCanvas);

async function init() {
  resizeCanvas();
  if (bmagScaleSelect) {
    bmagScaleSelect.value = state.bMagnitudeScale;
  }
  const bootstrap = window.VIEWER_BOOTSTRAP ?? {};
  setCaseOptions(bootstrap.cases || [], bootstrap.defaultCase);

  if (!caseSelect.value) {
    await refreshCaseList(false);
  }

  const initialCase = caseSelect.value;
  if (initialCase) {
    loadMesh(initialCase);
  } else {
    showOverlay("No mesh cases found. Run mesh_and_sources.py first.");
  }
  emitViewerEvent("ready", {
    caseName: state.caseName || initialCase || null,
    cases: currentCaseOptions(),
  });
}

function setCaseOptions(cases, preferred) {
  caseSelect.innerHTML = "";
  if (!cases || cases.length === 0) {
    const option = document.createElement("option");
    option.textContent = "No cases available";
    option.value = "";
    option.disabled = true;
    option.selected = true;
    caseSelect.appendChild(option);
    caseSelect.disabled = true;
    return;
  }

  caseSelect.disabled = false;
  cases.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    caseSelect.appendChild(option);
  });

  if (preferred && cases.includes(preferred)) {
    caseSelect.value = preferred;
  } else {
    caseSelect.selectedIndex = 0;
  }
}

function currentCaseOptions() {
  if (!caseSelect) {
    return [];
  }
  return Array.from(caseSelect.options)
    .map((option) => option.value)
    .filter((value) => value);
}

function setCaseSelection(caseName) {
  if (!caseSelect) {
    return false;
  }
  const match = Array.from(caseSelect.options).find((option) => option.value === caseName);
  if (match) {
    caseSelect.value = caseName;
    return true;
  }
  return false;
}

async function refreshCaseList(andNotify = true) {
  const previousSelection = state.caseName || caseSelect.value || null;
  try {
    const response = await fetch("/api/cases");
    if (!response.ok) {
      throw new Error("Failed to fetch case list");
    }
    const data = await response.json();
    const caseList = data.cases || [];
    setCaseOptions(caseList, data.default);
    if (previousSelection) {
      setCaseSelection(previousSelection);
    }
    if (andNotify) {
      updateStatus(`Found ${caseList.length} case(s).`);
    }
    emitViewerEvent("casesChanged", {
      cases: caseList,
      defaultCase: data.default || null,
    });
    return data;
  } catch (error) {
    updateStatus(`Error fetching case list: ${error.message}`);
    emitViewerEvent("casesChanged", {
      cases: currentCaseOptions(),
      error: error.message,
    });
    return null;
  }
}

async function refreshActiveCase() {
  updateStatus("Refreshing cases…");
  const activeCase = state.caseName || caseSelect.value || null;
  const data = await refreshCaseList();
  if (!data) {
    return null;
  }
  if (!activeCase) {
    if (caseSelect.value) {
      return loadMesh(caseSelect.value);
    }
    showOverlay("No mesh cases found. Run mesh_and_sources.py first.");
    return data;
  }

  const stillPresent = setCaseSelection(activeCase);
  if (stillPresent) {
    return loadMesh(activeCase, { preserveView: true, preserveSelection: true });
  }

  if (caseSelect.value) {
    updateStatus(`Active case '${activeCase}' missing. Loading ${caseSelect.value} instead.`);
    return loadMesh(caseSelect.value);
  }

  showOverlay("No mesh cases found. Run mesh_and_sources.py first.");
  updateStatus("No cases available.");
  return data;
}

async function loadMesh(caseName, options = {}) {
  if (!caseName) {
    return null;
  }
  const { preserveView = false, preserveSelection = false } = options;
  const previousSelection =
    preserveSelection && state.selectedCellIndex !== null
      ? state.selectedCellIndex
      : null;
  showOverlay("Loading mesh…");
  updateStatus(`Loading ${caseName}…`);
  try {
    const response = await fetch(`/api/mesh?case=${encodeURIComponent(caseName)}`);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || response.statusText);
    }
    const mesh = await response.json();
    state.mesh = mesh;
    state.bField = prepareBField(mesh.bField);
    state.designShapes = extractDesignShapes(mesh.meta);
    if (state.bMagnitudeScale === "log" && !state.bField?.logExtent) {
      state.bMagnitudeScale = "linear";
      if (bmagScaleSelect) {
        bmagScaleSelect.value = "linear";
      }
    }
    state.caseName = caseName;
    state.edges = buildEdges(mesh.tris || []);
    state.centroids = computeCentroids(mesh.nodes || [], mesh.tris || []);
    if (
      preserveSelection &&
      previousSelection !== null &&
      previousSelection >= 0 &&
      previousSelection < state.centroids.length
    ) {
      state.selectedCellIndex = previousSelection;
    } else {
      state.selectedCellIndex = null;
    }
    if (!preserveView) {
      fitViewToBounds(mesh.bounds);
    }
    updateSummary(mesh);
    updateCellInfo(state.selectedCellIndex);
    syncLayerAvailability();
    showOverlay("");
    updateStatus(
      `Loaded ${caseName} — ${mesh.summary?.node_count ?? 0} nodes, ${
        mesh.summary?.tri_count ?? 0
      } tris.`
    );
    render();
    emitViewerEvent("caseLoaded", {
      caseName,
      summary: mesh.summary || null,
      bounds: mesh.bounds || null,
    });
    return mesh;
  } catch (error) {
    const message = error?.message || String(error);
    showOverlay("Failed to load mesh.");
    updateStatus(`Error: ${message}`);
    emitViewerEvent("caseLoadFailed", { caseName, error: message });
    return null;
  }
}

function buildEdges(tris) {
  const seen = new Set();
  const edges = [];
  for (const tri of tris) {
    if (!Array.isArray(tri) || tri.length !== 3) continue;
    const pairs = [
      [tri[0], tri[1]],
      [tri[1], tri[2]],
      [tri[2], tri[0]],
    ];
    for (const [a, b] of pairs) {
      const i = Number(a);
      const j = Number(b);
      if (!Number.isFinite(i) || !Number.isFinite(j)) continue;
      const key = i < j ? `${i}-${j}` : `${j}-${i}`;
      if (!seen.has(key)) {
        seen.add(key);
        edges.push([i, j]);
      }
    }
  }
  return edges;
}

function computeCentroids(nodes, tris) {
  const centroids = [];
  for (const tri of tris || []) {
    if (!Array.isArray(tri) || tri.length !== 3) {
      centroids.push([0, 0]);
      continue;
    }
    const pa = nodes[tri[0]];
    const pb = nodes[tri[1]];
    const pc = nodes[tri[2]];
    if (!pa || !pb || !pc) {
      centroids.push([0, 0]);
      continue;
    }
    centroids.push([
      (pa[0] + pb[0] + pc[0]) / 3,
      (pa[1] + pb[1] + pc[1]) / 3,
    ]);
  }
  return centroids;
}

function fitViewToBounds(bounds) {
  if (!bounds || state.viewport.width === 0 || state.viewport.height === 0) {
    return;
  }
  const width = Math.max(bounds.maxX - bounds.minX, 1e-9);
  const height = Math.max(bounds.maxY - bounds.minY, 1e-9);
  const scaleX = state.viewport.width / width;
  const scaleY = state.viewport.height / height;
  const baseScale = 0.9 * Math.min(scaleX, scaleY);
  state.view = {
    centerX: (bounds.minX + bounds.maxX) / 2,
    centerY: (bounds.minY + bounds.maxY) / 2,
    baseScale: baseScale,
    zoom: 1,
    offsetX: 0,
    offsetY: 0,
  };
  render();
}

function worldToScreen(x, y) {
  const view = state.view;
  const scale = Math.max(view.baseScale * view.zoom, 1e-9);
  return {
    x: (x - view.centerX) * scale + state.viewport.width / 2 + view.offsetX,
    y: state.viewport.height / 2 - (y - view.centerY) * scale + view.offsetY,
  };
}

function screenToWorld(px, py) {
  const view = state.view;
  const scale = Math.max(view.baseScale * view.zoom, 1e-9);
  return {
    x: (px - state.viewport.width / 2 - view.offsetX) / scale + view.centerX,
    y: -(py - state.viewport.height / 2 - view.offsetY) / scale + view.centerY,
  };
}

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.mesh || !state.mesh.nodes || state.mesh.nodes.length === 0) {
    return;
  }

  drawMaterials();
  drawBFieldMagnitude();
  drawJfreeHeatmap();
  drawEdges();
  drawTargetGeometry();
  drawMagnetization();
  drawBFieldVectors();
  drawSelectedCellOutline();
}

function updateSummary(mesh) {
  if (!mesh) {
    summaryContent.textContent = "No mesh loaded.";
    return;
  }
  const summary = mesh.summary || {};
  const bounds = mesh.bounds || {};
  const stats = summary.field_stats || {};
  const formatRange = (label, data) =>
    data ? `${label}: ${data.min.toFixed(3)} – ${data.max.toFixed(3)}` : "";

  summaryContent.innerHTML = `
    <dl>
      <dt>Nodes / Elements</dt>
      <dd>${summary.node_count ?? 0} nodes, ${summary.tri_count ?? 0} tris</dd>
      <dt>Domain</dt>
      <dd>X: ${Number(bounds.minX ?? 0).toFixed(3)} → ${Number(bounds.maxX ?? 0).toFixed(3)} m<br/>
          Y: ${Number(bounds.minY ?? 0).toFixed(3)} → ${Number(bounds.maxY ?? 0).toFixed(3)} m</dd>
      <dt>Field snapshots</dt>
      <dd>
        ${formatRange("μ_r", stats.mu_r) || "μ_r: n/a"}<br/>
        ${formatRange("|M|", stats.M_mag) || "|M|: n/a"}<br/>
        ${formatRange("J_z", stats.Jz) || "J_z: n/a"}<br/>
        ${formatRange("|B|", stats.Bmag) || "|B|: n/a"}
      </dd>
    </dl>
  `;
}

function updateCellInfo(cellIndex) {
  if (!cellInfoContent) {
    return;
  }
  if (!state.mesh) {
    cellInfoContent.textContent = "Load a mesh to inspect cells.";
    return;
  }
  if (cellIndex === null || cellIndex === undefined || cellIndex < 0) {
    cellInfoContent.textContent = "Click a cell to inspect local fields.";
    return;
  }
  const nodes = state.mesh.nodes || [];
  const tris = state.mesh.tris || [];
  if (!tris[cellIndex]) {
    cellInfoContent.textContent = "Select a valid cell.";
    return;
  }
  const centroid = state.centroids[cellIndex] || [NaN, NaN];
  const regionId = Array.isArray(state.mesh.region_id)
    ? state.mesh.region_id[cellIndex]
    : null;

  const bx = valueAt(state.bField?.Bx, cellIndex);
  const by = valueAt(state.bField?.By, cellIndex);
  let bMag = valueAt(state.bField?.Bmag, cellIndex);
  if (!Number.isFinite(bMag) && Number.isFinite(bx) && Number.isFinite(by)) {
    bMag = Math.hypot(bx, by);
  }
  const directionDeg =
    Number.isFinite(bx) && Number.isFinite(by) && (Math.abs(bx) > 0 || Math.abs(by) > 0)
      ? ((Math.atan2(by, bx) * 180) / Math.PI).toFixed(1)
      : null;

  const muR = valueAt(state.mesh?.fields?.mu_r, cellIndex);
  const mx = valueAt(state.mesh?.fields?.Mx, cellIndex);
  const my = valueAt(state.mesh?.fields?.My, cellIndex);
  const mMag =
    Number.isFinite(mx) && Number.isFinite(my) ? Math.hypot(mx, my) : null;
  const jz = valueAt(state.mesh?.fields?.Jz, cellIndex);

  const materialLines = [];
  if (Number.isFinite(muR)) materialLines.push(`μ_r = ${formatNumber(muR)}`);
  if (Number.isFinite(mMag)) materialLines.push(`|M| = ${formatNumber(mMag)} A/m`);
  if (Number.isFinite(mx) || Number.isFinite(my)) {
    materialLines.push(
      `Mx = ${formatNumber(mx)}, My = ${formatNumber(my)} A/m`
    );
  }
  const fieldLines = [];
  if (Number.isFinite(bMag)) fieldLines.push(`|B| = ${formatNumber(bMag)} T`);
  if (Number.isFinite(bx) || Number.isFinite(by)) {
    fieldLines.push(`Bx = ${formatNumber(bx)} T, By = ${formatNumber(by)} T`);
  }
  if (directionDeg !== null) {
    fieldLines.push(`Dir = ${directionDeg}°`);
  }
  const jLines = [];
  if (Number.isFinite(jz)) {
    jLines.push(`${formatNumber(jz)} A/m²`);
  }

  cellInfoContent.innerHTML = `
    <dl>
      <dt>Cell</dt>
      <dd>#${cellIndex} (region ${regionId ?? "n/a"})</dd>
      <dt>Centroid</dt>
      <dd>x = ${formatNumber(centroid[0])} m<br/>y = ${formatNumber(
        centroid[1]
      )} m</dd>
      <dt>B field</dt>
      <dd>${fieldLines.join("<br/>") || "n/a"}</dd>
      <dt>Material</dt>
      <dd>${materialLines.join("<br/>") || "n/a"}</dd>
      <dt>J<sub>z</sub></dt>
      <dd>${jLines.join("<br/>") || "n/a"}</dd>
    </dl>
  `;
}

function showOverlay(message) {
  overlay.textContent = message;
  overlay.style.display = message ? "flex" : "none";
}

function updateStatus(message) {
  statusBar.textContent = message;
}

function syncLayerAvailability() {
  const hasMag = hasField("Mx") && hasField("My");
  const hasJ = hasField("Jz");
  setToggleAvailability(layerMagnetizationToggle, "magnetization", hasMag);
  setToggleAvailability(layerJfreeToggle, "jfree", hasJ);
  setToggleAvailability(layerMaterialsToggle, "materials", true);
  const hasB =
    Array.isArray(state.bField?.Bmag) && state.bField.Bmag.length > 0;
  setToggleAvailability(layerBmagToggle, "bMagnitude", hasB);
  setToggleAvailability(layerBvecToggle, "bVectors", hasB);
  setToggleAvailability(layerMeshToggle, "meshLines", true);
  const hasTargets = Array.isArray(state.designShapes) && state.designShapes.length > 0;
  setToggleAvailability(layerTargetToggle, "targetShapes", hasTargets);
  if (bmagScaleSelect) {
    bmagScaleSelect.disabled = !hasB;
    if (bmagScaleControl) {
      bmagScaleControl.classList.toggle("disabled", !hasB);
    }
  }
}

function hasField(name) {
  const arr = state.mesh?.fields?.[name];
  return Array.isArray(arr) && arr.length > 0;
}

function setToggleAvailability(toggle, layerKey, available) {
  if (!toggle) return;
  toggle.disabled = !available;
  const label = toggle.closest("label");
  if (label) {
    label.classList.toggle("disabled", !available);
  }
  if (!available) {
    toggle.checked = false;
    state.layers[layerKey] = false;
  } else {
    state.layers[layerKey] = toggle.checked;
  }
}

function prepareBField(rawField) {
  if (!rawField || typeof rawField !== "object") {
    return null;
  }
  const enriched = { ...rawField };
  const values = Array.isArray(rawField.Bmag) ? rawField.Bmag : null;
  enriched.logExtent = values ? computeLogExtent(values) : null;
  enriched.percentileBmag = values ? computePercentile(values, 99) : null;
  return enriched;
}

function extractDesignShapes(meta) {
  if (!meta || typeof meta !== "object") {
    return [];
  }

  const caseDefObjects = meta.case_definition?.objects;
  if (Array.isArray(caseDefObjects) && caseDefObjects.length) {
    return caseDefObjects.map((obj) => normalizeDesignShape(obj)).filter(Boolean);
  }

  const geometry = meta.geometry;
  if (Array.isArray(geometry) && geometry.length) {
    return geometry.map((obj) => normalizeDesignShape(obj)).filter(Boolean);
  }

  if (geometry && typeof geometry === "object") {
    const shapes = [];
    if (geometry.magnet_rect) {
      const norm = normalizeDesignShape(geometry.magnet_rect, "magnet");
      if (norm) shapes.push(norm);
    }
    if (geometry.steel_rect) {
      const norm = normalizeDesignShape(geometry.steel_rect, "steel");
      if (norm) shapes.push(norm);
    }
    if (Array.isArray(geometry.wire_disks)) {
      geometry.wire_disks.forEach((disk) => {
        const norm = normalizeDesignShape(disk, "wire");
        if (norm) shapes.push(norm);
      });
    }
    return shapes;
  }
  return [];
}

function normalizeDesignShape(entry, fallbackRole = "magnet") {
  if (!entry || typeof entry !== "object") {
    return null;
  }
  const role = entry.material || entry.role || fallbackRole;
  const shapeDef = entry.shape && typeof entry.shape === "object" ? entry.shape : entry;
  const typeRaw = shapeDef.type || entry.type;
  const type = typeof typeRaw === "string" ? typeRaw.toLowerCase() : "rect";
  const center = normalizeCenter(shapeDef.center ?? entry.center);
  if (!center) {
    return null;
  }
  if (type === "rect") {
    const width = firstFiniteNumber(
      shapeDef.width,
      shapeDef.size && typeof shapeDef.size === "object" ? shapeDef.size.width : null,
      typeof shapeDef.size === "number" ? shapeDef.size : null
    );
    const height = firstFiniteNumber(
      shapeDef.height,
      shapeDef.size && typeof shapeDef.size === "object" ? shapeDef.size.height : null,
      typeof shapeDef.size === "number" ? shapeDef.size : null
    );
    if (!(width > 0 && height > 0)) {
      return null;
    }
    const angle = normalizeAngleDegrees(
      firstFiniteNumber(
        shapeDef.angle,
        entry.angle,
        shapeDef.rotation,
        entry.rotation,
        shapeDef.theta,
        entry.theta
      )
    );
    return { type: "rect", center, width, height, role, angle };
  }
  if (type === "circle") {
    const radius = firstFiniteNumber(
      shapeDef.radius,
      shapeDef.size && typeof shapeDef.size === "object" ? shapeDef.size.radius : null,
      typeof shapeDef.size === "number" ? shapeDef.size : null,
      shapeDef.width ? Number(shapeDef.width) / 2 : null,
      shapeDef.height ? Number(shapeDef.height) / 2 : null
    );
    if (!(radius > 0)) {
      return null;
    }
    return { type: "circle", center, radius, role };
  }
  if (type === "ring") {
    const outer = firstFiniteNumber(
      shapeDef.outer_radius,
      shapeDef.outerRadius,
      shapeDef.radius,
      shapeDef.size && typeof shapeDef.size === "object" ? shapeDef.size.outer : null,
      typeof shapeDef.size === "number" ? shapeDef.size : null,
      shapeDef.od ? Number(shapeDef.od) / 2 : null,
      shapeDef.width ? Number(shapeDef.width) / 2 : null,
      shapeDef.height ? Number(shapeDef.height) / 2 : null
    );
    const inner = firstFiniteNumber(
      shapeDef.inner_radius,
      shapeDef.innerRadius,
      shapeDef.radius_inner,
      shapeDef.inner,
      shapeDef.id ? Number(shapeDef.id) / 2 : null,
      shapeDef.t ? outer && Number.isFinite(Number(shapeDef.t)) ? outer - Number(shapeDef.t) : null : null
    );
    if (!(outer > 0)) {
      return null;
    }
    const clampedInner = Math.max(0, Math.min(inner ?? 0, outer - 1e-9));
    return { type: "ring", center, outer_radius: outer, inner_radius: clampedInner, role };
  }
  return null;
}

function normalizeCenter(raw) {
  if (Array.isArray(raw) && raw.length >= 2) {
    const x = Number(raw[0]);
    const y = Number(raw[1]);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      return [x, y];
    }
  } else if (raw && typeof raw === "object") {
    const x = firstFiniteNumber(raw.x, raw.X, raw.cx, raw.centerX, raw[0]);
    const y = firstFiniteNumber(raw.y, raw.Y, raw.cy, raw.centerY, raw[1]);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      return [x, y];
    }
  }
  return null;
}

function normalizeAngleDegrees(value) {
  const angle = Number(value);
  if (!Number.isFinite(angle)) {
    return 0;
  }
  let normalized = angle % 360;
  if (normalized > 180) {
    normalized -= 360;
  } else if (normalized <= -180) {
    normalized += 360;
  }
  return normalized === -0 ? 0 : normalized;
}

function firstFiniteNumber(...values) {
  for (const value of values) {
    if (value === undefined || value === null) {
      continue;
    }
    const num = Number(value);
    if (Number.isFinite(num)) {
      return num;
    }
  }
  return null;
}

function computeLogExtent(values) {
  let minLog = Infinity;
  let maxLog = -Infinity;
  let hasPositive = false;
  for (const raw of values) {
    const value = Number(raw);
    if (!Number.isFinite(value) || value <= 0) {
      continue;
    }
    const logValue = safeLog10(value);
    if (logValue < minLog) {
      minLog = logValue;
    }
    if (logValue > maxLog) {
      maxLog = logValue;
    }
    hasPositive = true;
  }
  if (!hasPositive) {
    return null;
  }
  if (Math.abs(maxLog - minLog) < 1e-9) {
    maxLog = minLog + 1;
  }
  return { minLog, maxLog, range: maxLog - minLog };
}

function computePercentile(values, percentile) {
  if (!Array.isArray(values) || values.length === 0) {
    return null;
  }
  const cleaned = [];
  for (const raw of values) {
    const value = Number(raw);
    if (Number.isFinite(value)) {
      cleaned.push(value);
    }
  }
  if (cleaned.length === 0) {
    return null;
  }
  cleaned.sort((a, b) => a - b);
  const t = clamp(percentile, 0, 100) / 100;
  const maxIndex = cleaned.length - 1;
  const index = t * maxIndex;
  const lower = Math.floor(index);
  const upper = Math.min(maxIndex, Math.ceil(index));
  const weight = index - lower;
  if (lower === upper || weight <= 0) {
    return cleaned[lower];
  }
  return cleaned[lower] * (1 - weight) + cleaned[upper] * weight;
}

function drawMaterials() {
  if (!state.layers.materials) {
    return;
  }
  const tris = state.mesh.tris || [];
  const nodes = state.mesh.nodes || [];
  const regionIds = state.mesh.region_id || [];
  const jzField = state.mesh.fields?.Jz || null;
  const jzThreshold = jzCopperThreshold(jzField);

  ctx.save();
  for (let i = 0; i < tris.length; i += 1) {
    const tri = tris[i];
    const pa = nodes[tri[0]];
    const pb = nodes[tri[1]];
    const pc = nodes[tri[2]];
    if (!pa || !pb || !pc) continue;
    const fillStyle = materialColor(regionIds[i], jzField ? jzField[i] : 0, jzThreshold);
    ctx.beginPath();
    const A = worldToScreen(pa[0], pa[1]);
    const B = worldToScreen(pb[0], pb[1]);
    const C = worldToScreen(pc[0], pc[1]);
    ctx.moveTo(A.x, A.y);
    ctx.lineTo(B.x, B.y);
    ctx.lineTo(C.x, C.y);
    ctx.closePath();
    ctx.fillStyle = fillStyle;
    ctx.fill();
  }
  ctx.restore();
}

function drawTargetGeometry() {
  if (!state.layers.targetShapes || !state.designShapes?.length) {
    return;
  }
  ctx.save();
  ctx.setLineDash([6, 4]);
  ctx.lineWidth = 2;
  for (const shape of state.designShapes) {
    const role = shape.role || "magnet";
    ctx.strokeStyle = designStrokeColor(role);
    if (shape.type === "rect") {
      drawDesignRect(shape);
    } else if (shape.type === "circle") {
      drawDesignCircle(shape);
    } else if (shape.type === "ring") {
      drawDesignRing(shape);
    }
  }
  ctx.restore();
}

function designStrokeColor(role) {
  switch (role) {
    case "steel":
      return TARGET_STEEL_COLOR;
    case "wire":
      return TARGET_WIRE_COLOR;
    default:
      return TARGET_MAGNET_COLOR;
  }
}

function drawDesignRect(shape) {
  const halfW = (shape.width || 0) / 2;
  const halfH = (shape.height || 0) / 2;
  const cx = shape.center?.[0] ?? 0;
  const cy = shape.center?.[1] ?? 0;
  if (!(halfW > 0 && halfH > 0)) {
    return;
  }
  const angleDeg = Number(shape.angle) || 0;
  let cosA = 1;
  let sinA = 0;
  if (angleDeg) {
    const angleRad = (angleDeg * Math.PI) / 180;
    cosA = Math.cos(angleRad);
    sinA = Math.sin(angleRad);
  }
  const offsets = [
    [-halfW, -halfH],
    [halfW, -halfH],
    [halfW, halfH],
    [-halfW, halfH],
  ];
  const corners = offsets.map(([dx, dy]) => {
    const rx = dx * cosA - dy * sinA;
    const ry = dx * sinA + dy * cosA;
    return worldToScreen(cx + rx, cy + ry);
  });
  ctx.beginPath();
  ctx.moveTo(corners[0].x, corners[0].y);
  for (let i = 1; i < corners.length; i += 1) {
    ctx.lineTo(corners[i].x, corners[i].y);
  }
  ctx.closePath();
  ctx.stroke();
}

function drawDesignCircle(shape) {
  const cx = shape.center?.[0] ?? 0;
  const cy = shape.center?.[1] ?? 0;
  const radius = shape.radius ?? 0;
  const centerScreen = worldToScreen(cx, cy);
  const edgeScreen = worldToScreen(cx + radius, cy);
  const screenRadius = Math.hypot(edgeScreen.x - centerScreen.x, edgeScreen.y - centerScreen.y);
  ctx.beginPath();
  ctx.ellipse(centerScreen.x, centerScreen.y, screenRadius, screenRadius, 0, 0, Math.PI * 2);
  ctx.stroke();
}

function drawDesignRing(shape) {
  const outer = Math.max(0, shape.outer_radius ?? shape.radius ?? 0);
  const inner = Math.max(0, Math.min(shape.inner_radius ?? 0, outer));
  if (!(outer > 0)) {
    if (inner > 0) {
      drawDesignCircle({ center: shape.center, radius: inner });
    }
    return;
  }
  drawDesignCircle({ center: shape.center, radius: outer });
  if (inner > 0) {
    drawDesignCircle({ center: shape.center, radius: inner });
  }
}

function materialColor(regionId, jzValue, threshold) {
  if (Math.abs(jzValue || 0) > threshold) {
    return WIRE_COLOR;
  }
  return REGION_COLORS[regionId] || "#f0f0f0";
}

function jzCopperThreshold(values) {
  if (!values || values.length === 0) {
    return Infinity;
  }
  const stats = state.mesh?.summary?.field_stats?.Jz;
  if (!stats) return 1e-9;
  const range = Math.abs(stats.max - stats.min);
  return Math.max(range * 1e-3, 1e-9);
}

function drawEdges() {
  if (!state.layers.meshLines) {
    return;
  }
  ctx.save();
  ctx.lineWidth = 1;
  ctx.strokeStyle = EDGE_COLOR;
  const nodes = state.mesh.nodes;

  for (const [a, b] of state.edges) {
    const pa = nodes[a];
    const pb = nodes[b];
    if (!pa || !pb) continue;
    const A = worldToScreen(pa[0], pa[1]);
    const B = worldToScreen(pb[0], pb[1]);
    ctx.beginPath();
    ctx.moveTo(A.x, A.y);
    ctx.lineTo(B.x, B.y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawMagnetization() {
  if (!state.layers.magnetization) {
    return;
  }
  const Mx = state.mesh?.fields?.Mx;
  const My = state.mesh?.fields?.My;
  if (!Array.isArray(Mx) || !Array.isArray(My)) {
    return;
  }
  const centroids = state.centroids;
  const stats = state.mesh?.summary?.field_stats?.M_mag;
  const maxMag = Math.max(1e-9, stats?.max ?? 0);
  if (maxMag <= 0) {
    return;
  }
  const bounds = state.mesh?.bounds || { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  const domainDiag = Math.hypot(
    bounds.maxX - bounds.minX,
    bounds.maxY - bounds.minY
  );
  const baseLength = domainDiag * 0.05;

  ctx.save();
  ctx.strokeStyle = MAG_ARROW_COLOR;
  ctx.fillStyle = MAG_ARROW_COLOR;
  ctx.lineWidth = 1.3;

  for (let i = 0; i < centroids.length; i += 1) {
    const cx = centroids[i][0];
    const cy = centroids[i][1];
    const mx = Mx[i] || 0;
    const my = My[i] || 0;
    const mag = Math.hypot(mx, my);
    if (mag <= 1e-6) continue;
    const scale = (mag / maxMag) * baseLength;
    const vx = (mx / mag) * scale;
    const vy = (my / mag) * scale;
    const start = worldToScreen(cx, cy);
    const end = worldToScreen(cx + vx, cy + vy);
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    drawArrowHead(end, start);
  }

  ctx.restore();
}

function drawArrowHead(tip, tail) {
  const angle = Math.atan2(tip.y - tail.y, tip.x - tail.x);
  const headLength = 7;
  ctx.beginPath();
  ctx.moveTo(tip.x, tip.y);
  ctx.lineTo(
    tip.x - headLength * Math.cos(angle - Math.PI / 7),
    tip.y - headLength * Math.sin(angle - Math.PI / 7)
  );
  ctx.lineTo(
    tip.x - headLength * Math.cos(angle + Math.PI / 7),
    tip.y - headLength * Math.sin(angle + Math.PI / 7)
  );
  ctx.closePath();
  ctx.fill();
}

function drawBFieldMagnitude() {
  if (!state.layers.bMagnitude || !state.bField) {
    return;
  }
  const { Bmag } = state.bField;
  if (!Array.isArray(Bmag) || Bmag.length === 0) {
    return;
  }
  const tris = state.mesh.tris || [];
  const nodes = state.mesh.nodes || [];
  if (tris.length === 0 || Bmag.length !== tris.length) {
    return;
  }
  const stats = state.mesh?.summary?.field_stats?.Bmag;
  const statsMin = stats?.min ?? 0;
  const statsMax = stats?.max ?? statsMin;
  const statsRange = Math.max(statsMax - statsMin, 1e-12);
  const linearMin = 0;
  const linearCap = Math.max(state.bField.percentileBmag ?? statsMax ?? 0, 1e-12);
  const linearRange = Math.max(linearCap - linearMin, 1e-12);
  const mode =
    state.bMagnitudeScale === "log" && state.bField.logExtent ? "log" : "linear";
  const logExtent = state.bField.logExtent;
  const scaleMinVal = mode === "log" ? statsMin : linearMin;
  const scaleRange = mode === "log" ? statsRange : linearRange;

  ctx.save();
  ctx.globalAlpha = state.layers.materials ? 0.65 : 0.9;
  for (let i = 0; i < tris.length; i += 1) {
    const tri = tris[i];
    const pa = nodes[tri[0]];
    const pb = nodes[tri[1]];
    const pc = nodes[tri[2]];
    if (!pa || !pb || !pc) continue;
    const A = worldToScreen(pa[0], pa[1]);
    const B = worldToScreen(pb[0], pb[1]);
    const C = worldToScreen(pc[0], pc[1]);
    ctx.beginPath();
    ctx.moveTo(A.x, A.y);
    ctx.lineTo(B.x, B.y);
    ctx.lineTo(C.x, C.y);
    ctx.closePath();
    ctx.fillStyle = bMagnitudeColor(Bmag[i] || 0, scaleMinVal, scaleRange, mode, logExtent);
    ctx.fill();
  }
  ctx.restore();
}

function drawBFieldVectors() {
  if (!state.layers.bVectors || !state.bField) {
    return;
  }
  const { Bx, By } = state.bField;
  if (!Array.isArray(Bx) || !Array.isArray(By)) {
    return;
  }
  const centroids = state.centroids;
  const triCount = Math.min(centroids.length, Bx.length, By.length);
  if (triCount === 0) {
    return;
  }
  const bounds = state.mesh?.bounds || { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  const width = Math.max(bounds.maxX - bounds.minX, 1e-9);
  const height = Math.max(bounds.maxY - bounds.minY, 1e-9);
  const domainArea = width * height;
  const avgSpacing = Math.sqrt(domainArea / triCount);
  const baseLength = avgSpacing * 0.6;

  ctx.save();
  ctx.strokeStyle = B_VECTOR_COLOR;
  ctx.fillStyle = B_VECTOR_COLOR;
  ctx.lineWidth = 1;

  for (let i = 0; i < triCount; i += 1) {
    const cx = centroids[i][0];
    const cy = centroids[i][1];
    const bx = Bx[i] || 0;
    const by = By[i] || 0;
    const mag = Math.hypot(bx, by);
    if (mag <= 1e-9) continue;
    const vx = (bx / mag) * baseLength;
    const vy = (by / mag) * baseLength;
    const start = worldToScreen(cx, cy);
    const end = worldToScreen(cx + vx, cy + vy);
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    drawArrowHead(end, start);
  }

  ctx.restore();
}

function drawSelectedCellOutline() {
  const index = state.selectedCellIndex;
  if (index === null || index === undefined || index < 0) {
    return;
  }
  const tri = state.mesh?.tris?.[index];
  const nodes = state.mesh?.nodes;
  if (!tri || !nodes) {
    return;
  }
  const pa = nodes[tri[0]];
  const pb = nodes[tri[1]];
  const pc = nodes[tri[2]];
  if (!pa || !pb || !pc) {
    return;
  }
  const A = worldToScreen(pa[0], pa[1]);
  const B = worldToScreen(pb[0], pb[1]);
  const C = worldToScreen(pc[0], pc[1]);
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = SELECTED_CELL_STROKE;
  ctx.fillStyle = SELECTED_CELL_FILL;
  ctx.beginPath();
  ctx.moveTo(A.x, A.y);
  ctx.lineTo(B.x, B.y);
  ctx.lineTo(C.x, C.y);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function bMagnitudeColor(value, minVal, range, mode, logExtent) {
  if (mode === "log" && logExtent) {
    return bMagnitudeColorLog(value, logExtent);
  }
  return bMagnitudeColorLinear(value, minVal, range);
}

function bMagnitudeColorLinear(value, minVal, range) {
  const t = clamp((value - minVal) / range, 0, 1);
  return lerpColor(B_MAG_MIN_COLOR, B_MAG_MAX_COLOR, t);
}

function bMagnitudeColorLog(value, logExtent) {
  if (!logExtent) {
    return lerpColor(B_MAG_MIN_COLOR, B_MAG_MAX_COLOR, 0);
  }
  if (!(value > 0)) {
    return lerpColor(B_MAG_MIN_COLOR, B_MAG_MAX_COLOR, 0);
  }
  const t = clamp(
    (safeLog10(value) - logExtent.minLog) / logExtent.range,
    0,
    1
  );
  return lerpColor(B_MAG_MIN_COLOR, B_MAG_MAX_COLOR, t);
}

function drawJfreeHeatmap() {
  if (!state.layers.jfree) {
    return;
  }
  const jz = state.mesh?.fields?.Jz;
  if (!Array.isArray(jz) || jz.length === 0) {
    return;
  }
  const tris = state.mesh.tris || [];
  const nodes = state.mesh.nodes || [];
  const stats = state.mesh?.summary?.field_stats?.Jz;
  const extent = Math.max(
    1e-9,
    Math.abs(stats?.max ?? 0),
    Math.abs(stats?.min ?? 0)
  );
  if (extent <= 0) {
    return;
  }

  ctx.save();
  ctx.globalAlpha = 0.6;
  for (let i = 0; i < tris.length; i += 1) {
    const value = jz[i] || 0;
    if (Math.abs(value) <= 1e-9) continue;
    const color = jfreeColor(value, extent);
    const tri = tris[i];
    const pa = nodes[tri[0]];
    const pb = nodes[tri[1]];
    const pc = nodes[tri[2]];
    if (!pa || !pb || !pc) continue;
    const A = worldToScreen(pa[0], pa[1]);
    const B = worldToScreen(pb[0], pb[1]);
    const C = worldToScreen(pc[0], pc[1]);
    ctx.beginPath();
    ctx.moveTo(A.x, A.y);
    ctx.lineTo(B.x, B.y);
    ctx.lineTo(C.x, C.y);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }
  ctx.restore();
}

function jfreeColor(value, extent) {
  if (value >= 0) {
    const t = clamp(value / extent, 0, 1);
    return lerpColor(JFREE_ZERO_COLOR, JFREE_POS_COLOR, t);
  }
  const t = clamp(-value / extent, 0, 1);
  return lerpColor(JFREE_ZERO_COLOR, JFREE_NEG_COLOR, t);
}

function lerpColor(a, b, t) {
  const r = Math.round(a.r + (b.r - a.r) * t);
  const g = Math.round(a.g + (b.g - a.g) * t);
  const bChan = Math.round(a.b + (b.b - a.b) * t);
  return `rgb(${r}, ${g}, ${bChan})`;
}

caseSelect.addEventListener("change", () => {
  const value = caseSelect.value;
  if (value) {
    emitViewerEvent("caseSelected", { caseName: value });
    loadMesh(value);
  }
});

refreshBtn.addEventListener("click", () => {
  refreshActiveCase().catch((error) => {
    const message = error?.message || String(error);
    updateStatus(`Refresh failed: ${message}`);
  });
});

resetBtn.addEventListener("click", () => {
  if (state.mesh) {
    fitViewToBounds(state.mesh.bounds);
    updateStatus("View reset to fit mesh.");
  }
});

layerMaterialsToggle?.addEventListener("change", () => {
  state.layers.materials = layerMaterialsToggle.checked;
  render();
});

layerTargetToggle?.addEventListener("change", () => {
  state.layers.targetShapes = layerTargetToggle.checked;
  render();
});

layerMeshToggle?.addEventListener("change", () => {
  state.layers.meshLines = layerMeshToggle.checked;
  render();
});

layerMagnetizationToggle?.addEventListener("change", () => {
  state.layers.magnetization = layerMagnetizationToggle.checked;
  render();
});

layerJfreeToggle?.addEventListener("change", () => {
  state.layers.jfree = layerJfreeToggle.checked;
  render();
});

layerBmagToggle?.addEventListener("change", () => {
  state.layers.bMagnitude = layerBmagToggle.checked;
  render();
});

layerBvecToggle?.addEventListener("change", () => {
  state.layers.bVectors = layerBvecToggle.checked;
  render();
});

bmagScaleSelect?.addEventListener("change", () => {
  const mode = bmagScaleSelect.value === "linear" ? "linear" : "log";
  state.bMagnitudeScale = mode;
  render();
});

canvas.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  state.dragging = true;
  state.dragPointerId = event.pointerId;
  state.lastPointer = { x: event.clientX, y: event.clientY };
  state.dragMoved = false;
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
  if (!state.dragging || event.pointerId !== state.dragPointerId) {
    return;
  }
  const dx = event.clientX - state.lastPointer.x;
  const dy = event.clientY - state.lastPointer.y;
  if (!state.dragMoved && Math.hypot(dx, dy) > 3) {
    state.dragMoved = true;
  }
  state.view.offsetX += dx;
  state.view.offsetY += dy;
  state.lastPointer = { x: event.clientX, y: event.clientY };
  render();
});

function endDrag(event) {
  if (state.dragging && event.pointerId === state.dragPointerId) {
    state.dragging = false;
    state.dragPointerId = null;
    canvas.releasePointerCapture(event.pointerId);
    if (!state.dragMoved && event.button === 0) {
      handleCanvasClick(event);
    }
  }
}

canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);

canvas.addEventListener(
  "wheel",
  (event) => {
    if (!state.mesh) return;
    event.preventDefault();
    const zoomFactor = event.deltaY < 0 ? 1.1 : 0.9;
    const before = screenToWorld(event.offsetX, event.offsetY);
    state.view.zoom = clamp(state.view.zoom * zoomFactor, 0.05, 40);
    const after = worldToScreen(before.x, before.y);
    state.view.offsetX += event.offsetX - after.x;
    state.view.offsetY += event.offsetY - after.y;
    render();
  },
  { passive: false }
);

function safeLog10(value) {
  if (value <= 0) {
    return -Infinity;
  }
  if (typeof Math.log10 === "function") {
    return Math.log10(value);
  }
  return Math.log(value) / Math.LN10;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function handleCanvasClick(event) {
  if (!state.mesh) {
    return;
  }
  const px =
    typeof event.offsetX === "number"
      ? event.offsetX
      : event.clientX - canvas.getBoundingClientRect().left;
  const py =
    typeof event.offsetY === "number"
      ? event.offsetY
      : event.clientY - canvas.getBoundingClientRect().top;
  const world = screenToWorld(px, py);
  const triIndex = findTriangleAt(world.x, world.y);
  state.selectedCellIndex = triIndex;
  if (triIndex === null) {
    updateStatus("No cell under cursor.");
    updateCellInfo(null);
  } else {
    updateStatus(`Selected cell #${triIndex}.`);
    updateCellInfo(triIndex);
  }
  render();
}

function findTriangleAt(x, y) {
  const tris = state.mesh?.tris;
  const nodes = state.mesh?.nodes;
  if (!Array.isArray(tris) || !Array.isArray(nodes)) {
    return null;
  }
  for (let i = 0; i < tris.length; i += 1) {
    const tri = tris[i];
    const pa = nodes[tri[0]];
    const pb = nodes[tri[1]];
    const pc = nodes[tri[2]];
    if (!pa || !pb || !pc) continue;
    if (pointInTriangle(x, y, pa, pb, pc)) {
      return i;
    }
  }
  return null;
}

function pointInTriangle(px, py, a, b, c) {
  const v0x = c[0] - a[0];
  const v0y = c[1] - a[1];
  const v1x = b[0] - a[0];
  const v1y = b[1] - a[1];
  const v2x = px - a[0];
  const v2y = py - a[1];

  const dot00 = v0x * v0x + v0y * v0y;
  const dot01 = v0x * v1x + v0y * v1y;
  const dot02 = v0x * v2x + v0y * v2y;
  const dot11 = v1x * v1x + v1y * v1y;
  const dot12 = v1x * v2x + v1y * v2y;
  const denom = dot00 * dot11 - dot01 * dot01;
  if (Math.abs(denom) < 1e-18) {
    return false;
  }
  const invDenom = 1 / denom;
  const u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  const v = (dot00 * dot12 - dot01 * dot02) * invDenom;
  return u >= -1e-6 && v >= -1e-6 && u + v <= 1 + 1e-6;
}

function valueAt(arr, index) {
  if (!Array.isArray(arr) || index < 0 || index >= arr.length) {
    return null;
  }
  const value = Number(arr[index]);
  return Number.isFinite(value) ? value : null;
}

function formatNumber(value, digits = 3) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return "n/a";
  }
  return num.toFixed(digits);
}

init();

const MeshViewerAPI = {
  getCaseName: () => state.caseName || null,
  getCases: () => [...currentCaseOptions()],
  loadCase(caseName, options = {}) {
    if (!caseName) {
      return Promise.resolve(null);
    }
    const { syncSelection = true, ...loadOptions } = options;
    if (syncSelection) {
      setCaseSelection(caseName);
    }
    return loadMesh(caseName, loadOptions);
  },
  refreshCases({ notify = true } = {}) {
    return refreshCaseList(notify);
  },
  refreshActiveCase,
  setCaseSelection,
};

window.MeshViewerAPI = MeshViewerAPI;
