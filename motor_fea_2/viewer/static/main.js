const canvas = document.getElementById("mesh-canvas");
const ctx = canvas.getContext("2d");
const overlay = document.getElementById("overlay-message");
const statusBar = document.getElementById("status-bar");
const summaryContent = document.getElementById("summary-content");
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
};

const REGION_COLORS = {
  0: "#f7fbff", // air
  1: "#ffd166", // magnet (warm gold)
  2: "#a2a6af", // steel / iron
};
const WIRE_COLOR = "#b87333";
const EDGE_COLOR = "#000";
const MAG_ARROW_COLOR = "#c2185b";
const B_VECTOR_COLOR = "#005cb2";
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

async function refreshCaseList(andNotify = true) {
  try {
    const response = await fetch("/api/cases");
    if (!response.ok) {
      throw new Error("Failed to fetch case list");
    }
    const data = await response.json();
    setCaseOptions(data.cases || [], data.default);
    if (andNotify) {
      updateStatus(`Found ${data.cases.length} case(s).`);
    }
  } catch (error) {
    updateStatus(`Error fetching case list: ${error.message}`);
  }
}

async function loadMesh(caseName) {
  if (!caseName) {
    return;
  }
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
    fitViewToBounds(mesh.bounds);
    updateSummary(mesh);
    syncLayerAvailability();
    showOverlay("");
    updateStatus(
      `Loaded ${caseName} — ${mesh.summary?.node_count ?? 0} nodes, ${
        mesh.summary?.tri_count ?? 0
      } tris.`
    );
    render();
  } catch (error) {
    showOverlay("Failed to load mesh.");
    updateStatus(`Error: ${error.message}`);
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
  return enriched;
}

function extractDesignShapes(meta) {
  if (!meta || typeof meta !== "object") {
    return [];
  }
  const geometry = meta.geometry;
  if (!geometry || typeof geometry !== "object") {
    return [];
  }
  const shapes = [];
  if (geometry.magnet_rect) {
    shapes.push({ ...geometry.magnet_rect, role: "magnet" });
  }
  if (geometry.steel_rect) {
    shapes.push({ ...geometry.steel_rect, role: "steel" });
  }
  if (Array.isArray(geometry.wire_disks)) {
    geometry.wire_disks.forEach((disk) => {
      shapes.push({ ...disk, role: "wire" });
    });
  }
  return shapes;
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
  const corners = [
    worldToScreen(cx - halfW, cy - halfH),
    worldToScreen(cx + halfW, cy - halfH),
    worldToScreen(cx + halfW, cy + halfH),
    worldToScreen(cx - halfW, cy + halfH),
  ];
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
  const minVal = stats?.min ?? 0;
  const maxVal = stats?.max ?? minVal;
  const range = Math.max(maxVal - minVal, 1e-12);
  const mode =
    state.bMagnitudeScale === "log" && state.bField.logExtent ? "log" : "linear";
  const logExtent = state.bField.logExtent;

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
    ctx.fillStyle = bMagnitudeColor(Bmag[i] || 0, minVal, range, mode, logExtent);
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
    loadMesh(value);
  }
});

refreshBtn.addEventListener("click", () => {
  refreshCaseList().then(() => {
    if (!state.caseName && caseSelect.value) {
      loadMesh(caseSelect.value);
    }
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
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
  if (!state.dragging || event.pointerId !== state.dragPointerId) {
    return;
  }
  const dx = event.clientX - state.lastPointer.x;
  const dy = event.clientY - state.lastPointer.y;
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

init();
