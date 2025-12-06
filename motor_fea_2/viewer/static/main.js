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
const layerContourBToggle = document.getElementById("layer-contour-b");
const layerContourForceToggle = document.getElementById("layer-contour-force");

const state = {
  mesh: null,
  edges: [],
  caseName: null,
  bField: null,
  indicator: null,
  contours: { segments: [], totals: [] },
  quickContours: null,
  quickContoursCase: null,
  centroids: [],
  neighbors: [],
  indicatorStats: null,
  designShapes: [],
  indicatorPreview: null,
  viewport: { width: 0, height: 0 },
  materialTriangles: null,
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
    contourB: false,
    contourForce: false,
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
const TARGET_CONTOUR_COLOR = "#111827";
const CONTOUR_B_COLOR = "#0f766e";
const CONTOUR_FORCE_COLOR = "#b45309";
const B_MAG_MIN_COLOR = { r: 255, g: 255, b: 255 }; // 0 T = white
const B_MAG_LOW_COLOR = { r: 255, g: 243, b: 205 }; // faint warm tint around 0.2–0.5 T
const B_MAG_MIDWARM_COLOR = { r: 252, g: 174, b: 74 }; // orange around 1 T
const B_MAG_MID_COLOR = { r: 220, g: 38, b: 38 }; // ~2 T turns red
const B_MAG_HIGH_COLOR = { r: 120, g: 0, b: 0 }; // deep maroon for a steep ramp to black
const B_MAG_MAX_COLOR = { r: 0, g: 0, b: 0 }; // highest fields fade to black
const B_MAG_RED_POINT_TESLA = 2;
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
  state.quickContours = null;
  state.quickContoursCase = caseName;
  fetchQuickContours(caseName).catch(() => {});
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
    state.contours = mesh.contours || { segments: [], totals: [] };
    state.indicator = mesh.indicator || null;
    state.indicatorStats = mesh.summary?.indicator_stats || null;
    state.indicatorPreview = computeIndicatorPreview(mesh);
    if (state.bMagnitudeScale === "log" && !state.bField?.logExtent) {
      state.bMagnitudeScale = "linear";
      if (bmagScaleSelect) {
        bmagScaleSelect.value = "linear";
      }
    }
    state.caseName = caseName;
    state.materialTriangles = buildMaterialTriangles(
      mesh.region_id,
      mesh.tris ? mesh.tris.length : 0,
      mesh.fields?.Jz
    );
    // Always build full edge list so mesh lines cover the whole domain.
    state.edges = buildEdges(mesh.tris || []);
    state.neighbors = buildNeighbors(mesh.tris || []);
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
    state.contours = { segments: [], totals: [] };
    state.indicator = null;
    state.indicatorStats = null;
    state.indicatorPreview = null;
    emitViewerEvent("caseLoadFailed", { caseName, error: message });
    return null;
  }
}

function buildMaterialTriangles(regionIds, triCount, jzField = null) {
  const hasRegions = Array.isArray(regionIds) && regionIds.length;
  const hasJz = Array.isArray(jzField) && jzField.length;
  if (!hasRegions && !hasJz) {
    return null;
  }
  const limit = Math.min(triCount || 0, hasRegions ? regionIds.length : jzField.length);
  if (!(limit > 0)) {
    return null;
  }
  const indices = [];
  for (let i = 0; i < limit; i += 1) {
    const region = hasRegions ? Number(regionIds[i]) : 0;
    const jz = hasJz ? Number(jzField[i]) : 0;
    if ((Number.isFinite(region) && region !== 0) || (Number.isFinite(jz) && Math.abs(jz) > 0)) {
      indices.push(i);
    }
  }
  return indices.length && indices.length < limit ? indices : null;
}

function buildEdges(tris, includeIndices = null) {
  const seen = new Set();
  const edges = [];
  const indexList =
    Array.isArray(includeIndices) && includeIndices.length
      ? includeIndices
      : null;
  if (!Array.isArray(tris)) {
    return edges;
  }
  if (indexList) {
    for (const idx of indexList) {
      const tri = tris[idx];
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

  for (let idx = 0; idx < tris.length; idx += 1) {
    const tri = tris[idx];
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

function buildNeighbors(tris) {
  if (!Array.isArray(tris)) {
    return [];
  }
  const neighbors = tris.map(() => []);
  const edgeOwners = new Map();
  tris.forEach((tri, idx) => {
    if (!Array.isArray(tri) || tri.length !== 3) {
      return;
    }
    const pairs = [
      [tri[0], tri[1]],
      [tri[1], tri[2]],
      [tri[2], tri[0]],
    ];
    pairs.forEach(([a, b]) => {
      const i = Number(a);
      const j = Number(b);
      if (!Number.isFinite(i) || !Number.isFinite(j)) {
        return;
      }
      const key = i < j ? `${i}-${j}` : `${j}-${i}`;
      if (edgeOwners.has(key)) {
        const other = edgeOwners.get(key);
        if (other !== idx) {
          neighbors[idx].push(other);
          neighbors[other].push(idx);
        }
      } else {
        edgeOwners.set(key, idx);
      }
    });
  });
  return neighbors;
}

async function fetchQuickContours(caseName) {
  try {
    const response = await fetch(`/api/mesh/summary?case=${encodeURIComponent(caseName)}`);
    if (!response.ok) {
      return null;
    }
    const data = await response.json();
    if (state.quickContoursCase !== caseName) {
      return data;
    }
    state.quickContours = data;
    updateSummary(null);
    emitViewerEvent("quickContours", { caseName, contours: data });
    return data;
  } catch (error) {
    return null;
  }
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

function extractFieldFocusParams(meta) {
  const gen = meta && meta.mesh_generation ? meta.mesh_generation : {};
  const focus =
    (meta && (meta.field_focus_params || meta.field_focus)) ||
    gen.field_focus_params ||
    gen.field_focus ||
    {};
  const axisScaling = focus.axis_scaling || {};
  const sizeMap = focus.size_map || {};
  const magnitudeWeight = finiteOr(
    focus.magnitude_weight,
    focus.magnitudeWeight,
    1.0
  );
  const directionWeight = finiteOr(
    focus.direction_weight,
    focus.directionWeight,
    1.0
  );
  const gain = finiteOr(
    focus.indicator_gain,
    focus.indicatorGain,
    sizeMap.alpha,
    focus.alpha,
    0.4
  );
  const neutral = finiteOr(
    focus.indicator_neutral,
    focus.indicatorNeutral,
    null
  );
  const percentileRef = finiteOr(
    focus.indicator_percentile,
    focus.indicatorPercentile,
    focus.indicator_percentile_ref,
    85
  );
  const alphaMin = finiteOr(
    focus.scale_min,
    focus.scaleMin,
    axisScaling.scale_min,
    axisScaling.scaleMin,
    0.5
  );
  const alphaMax = finiteOr(
    focus.scale_max,
    focus.scaleMax,
    axisScaling.scale_max,
    axisScaling.scaleMax,
    2.0
  );
  return {
    magnitudeWeight: Number.isFinite(magnitudeWeight) ? magnitudeWeight : 1.0,
    directionWeight: Number.isFinite(directionWeight) ? directionWeight : 1.0,
    indicatorGain: Math.max(Number.isFinite(gain) ? gain : 0.4, 0),
    indicatorNeutral: Number.isFinite(neutral) && neutral > 0 ? neutral : null,
    indicatorPercentile: clamp(
      Number.isFinite(percentileRef) ? percentileRef : 85,
      0,
      100
    ),
    alphaMin: Math.max(
      Number.isFinite(alphaMin) ? alphaMin : 0.5,
      1e-6
    ),
    alphaMax: Math.max(
      Number.isFinite(alphaMax) ? alphaMax : 2.0,
      1e-6
    ),
  };
}

function computeIndicatorPreview(mesh) {
  const indicator = mesh?.indicator || null;
  if (
    !indicator ||
    !Array.isArray(indicator.magnitude) ||
    !Array.isArray(indicator.direction)
  ) {
    return null;
  }
  const paramsFromIndicator = indicator.params
    ? {
        directionWeight: Number(indicator.params.direction_weight ?? indicator.params.directionWeight ?? indicator.params.dirWeight ?? indicator.params.direction) || indicator.params.directionWeight || undefined,
        magnitudeWeight: Number(indicator.params.magnitude_weight ?? indicator.params.magnitudeWeight ?? indicator.params.magWeight ?? indicator.params.magnitude) || indicator.params.magnitudeWeight || undefined,
        indicatorGain: Number(indicator.params.indicator_gain ?? indicator.params.gain ?? indicator.params.alpha) || indicator.params.indicatorGain || undefined,
        indicatorNeutral: indicator.params.indicator_neutral ?? indicator.params.neutral ?? null,
        indicatorPercentile: indicator.params.indicator_percentile ?? indicator.params.percentile ?? indicator.params.percentile_ref,
        alphaMin: indicator.params.alpha_min ?? indicator.params.scale_min ?? indicator.params.alphaMin,
        alphaMax: indicator.params.alpha_max ?? indicator.params.scale_max ?? indicator.params.alphaMax,
      }
    : null;
  const params = paramsFromIndicator
    ? {
        directionWeight: Number.isFinite(paramsFromIndicator.directionWeight)
          ? paramsFromIndicator.directionWeight
          : 1.0,
        magnitudeWeight: Number.isFinite(paramsFromIndicator.magnitudeWeight)
          ? paramsFromIndicator.magnitudeWeight
          : 1.0,
        indicatorGain: Math.max(
          Number.isFinite(paramsFromIndicator.indicatorGain)
            ? paramsFromIndicator.indicatorGain
            : 0.4,
          0
        ),
        indicatorNeutral:
          paramsFromIndicator.indicatorNeutral !== null &&
          paramsFromIndicator.indicatorNeutral !== undefined &&
          Number.isFinite(Number(paramsFromIndicator.indicatorNeutral))
            ? Number(paramsFromIndicator.indicatorNeutral)
            : null,
        indicatorPercentile: Number.isFinite(
          Number(paramsFromIndicator.indicatorPercentile)
        )
          ? Number(paramsFromIndicator.indicatorPercentile)
          : 85,
        alphaMin: Math.max(
          Number.isFinite(Number(paramsFromIndicator.alphaMin))
            ? Number(paramsFromIndicator.alphaMin)
            : 0.5,
          1e-6
        ),
        alphaMax: Math.max(
          Number.isFinite(Number(paramsFromIndicator.alphaMax))
            ? Number(paramsFromIndicator.alphaMax)
            : 2.0,
          1e-6
        ),
      }
    : extractFieldFocusParams(mesh?.meta || {});
  const len = Math.min(indicator.magnitude.length, indicator.direction.length);
  if (len === 0) {
    return null;
  }
  const hasPreAlpha = Array.isArray(indicator.alpha) && indicator.alpha.length >= len;
  const combined = Array.isArray(indicator.combined) && indicator.combined.length >= len
    ? indicator.combined.slice(0, len)
    : new Array(len);
  const finiteVals = [];
  if (!Array.isArray(indicator.combined) || indicator.combined.length < len) {
    for (let i = 0; i < len; i += 1) {
      const mag = Number(indicator.magnitude[i]);
      const dir = Number(indicator.direction[i]);
      const val = params.magnitudeWeight * mag + params.directionWeight * dir;
      const safeVal = Number.isFinite(val) ? val : NaN;
      combined[i] = safeVal;
      if (Number.isFinite(safeVal)) {
        finiteVals.push(safeVal);
      }
    }
  } else {
    for (let i = 0; i < len; i += 1) {
      const val = Number(combined[i]);
      const safeVal = Number.isFinite(val) ? val : NaN;
      combined[i] = safeVal;
      if (Number.isFinite(safeVal)) {
        finiteVals.push(safeVal);
      }
    }
  }
  if (finiteVals.length === 0) {
    return null;
  }
  const refPercentile = clamp(
    Number.isFinite(params.indicatorPercentile)
      ? params.indicatorPercentile
      : 85,
    0,
    100
  );
  const ref =
    params.indicatorNeutral && params.indicatorNeutral > 0
      ? params.indicatorNeutral
      : percentile(finiteVals, refPercentile);
  const refSafe =
    Number.isFinite(ref) && ref > 0
      ? ref
      : Math.max(percentile(finiteVals, 50) || 1.0, 1e-9);
  const alphaMin = Math.max(params.alphaMin, 1e-6);
  const alphaMax = Math.max(params.alphaMax, alphaMin);
  const gain = Math.max(params.indicatorGain, 0);
  const alphas = hasPreAlpha ? indicator.alpha.slice(0, len) : new Array(len);
  let alphaMinSeen = Number.POSITIVE_INFINITY;
  let alphaMaxSeen = Number.NEGATIVE_INFINITY;
  let combinedMin = Number.POSITIVE_INFINITY;
  let combinedMax = Number.NEGATIVE_INFINITY;
  finiteVals.forEach((v) => {
    if (Number.isFinite(v)) {
      combinedMin = Math.min(combinedMin, v);
      combinedMax = Math.max(combinedMax, v);
    }
  });
  if (!hasPreAlpha) {
    for (let i = 0; i < len; i += 1) {
      const val = combined[i];
      let alphaVal = alphaMax;
      if (Number.isFinite(val) && refSafe > 0) {
        const ratio = Math.max(val, 1e-12) / refSafe;
        alphaVal = Math.pow(ratio, -gain);
      }
      if (!Number.isFinite(alphaVal)) {
        alphaVal = alphaMax;
      }
      alphaVal = clamp(alphaVal, alphaMin, alphaMax);
      alphas[i] = alphaVal;
    }
  }
  for (let i = 0; i < len; i += 1) {
    const alphaVal = Number(alphas[i]);
    if (Number.isFinite(alphaVal)) {
      alphaMinSeen = Math.min(alphaMinSeen, alphaVal);
      alphaMaxSeen = Math.max(alphaMaxSeen, alphaVal);
    }
  }
  return {
    params: {
      directionWeight: params.directionWeight,
      magnitudeWeight: params.magnitudeWeight,
      indicatorGain: gain,
      indicatorNeutral: params.indicatorNeutral,
      indicatorPercentile: refPercentile,
      alphaMin,
      alphaMax,
    },
    ref: refSafe,
    combined,
    alphas,
    stats: {
      combined_min: Number.isFinite(combinedMin) ? combinedMin : 0,
      combined_max: Number.isFinite(combinedMax) ? combinedMax : 0,
      ref: refSafe,
      gain,
      ref_percentile: refPercentile,
      weights: {
        direction: params.directionWeight,
        magnitude: params.magnitudeWeight,
      },
      alpha_min: Number.isFinite(alphaMinSeen) ? alphaMinSeen : alphaMin,
      alpha_max: Number.isFinite(alphaMaxSeen) ? alphaMaxSeen : alphaMax,
    },
  };
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
  drawContourB();
  drawContourForces();
  drawMagnetization();
  drawBFieldVectors();
  drawSelectedCellOutline();
}

function updateSummary(mesh) {
  if (!mesh) {
    const quickContours = state.quickContours?.totals || [];
    if (quickContours.length) {
      const contourSummary = quickContours
        .map((entry) => {
          const label = entry.contour_label || `Contour ${entry.contour_index}`;
          const fx = entry.net_force?.[0] ?? 0;
          const fy = entry.net_force?.[1] ?? 0;
          const torque = entry.net_torque ?? 0;
          return `${label}: Fx=${formatNumber(fx, 4)} N/m, Fy=${formatNumber(
            fy,
            4
          )} N/m, τ=${formatNumber(torque, 4)} N·m/m`;
        })
        .join("<br/>");
      summaryContent.innerHTML = `
        <dl>
          <dt>Contours (quick)</dt>
          <dd>${contourSummary}</dd>
          <dt>Status</dt>
          <dd>Loading full mesh…</dd>
        </dl>
      `;
      return;
    }
    summaryContent.textContent = "No mesh loaded.";
    return;
  }
  const summary = mesh.summary || {};
  const bounds = mesh.bounds || {};
  const stats = summary.field_stats || {};
  const formatRange = (label, data) =>
    data ? `${label}: ${data.min.toFixed(3)} – ${data.max.toFixed(3)}` : "";

  const indicatorStats = summary.indicator_stats || state.indicatorStats || null;
  const previewStats = state.indicatorPreview?.stats || null;
  const indicatorLines = indicatorStats
    ? `Magnitude indicator: ${formatNumber(indicatorStats.magnitude_min)} – ${formatNumber(
        indicatorStats.magnitude_max
      )}<br/>Direction indicator: ${formatNumber(indicatorStats.direction_min)} – ${formatNumber(
        indicatorStats.direction_max
      )}<br/>Combined: ${formatNumber(indicatorStats.combined_min)} – ${formatNumber(
        indicatorStats.combined_max
      )}`
    : "";
  const alphaLines = previewStats
    ? `Alpha preview: ${formatNumber(previewStats.alpha_min, 4)} – ${formatNumber(
        previewStats.alpha_max,
        4
      )}× (weights: dir=${formatNumber(previewStats.weights.direction, 3)}, mag=${formatNumber(
        previewStats.weights.magnitude,
        3
      )}, gain=${formatNumber(previewStats.gain, 3)}, ref=${formatNumber(
        previewStats.ref,
        4
      )}@p${formatNumber(previewStats.ref_percentile, 1)})`
    : "";
  const indicatorBlock = [indicatorLines, alphaLines].filter(Boolean).join("<br/>");
  const contourTotals =
    (state.contours?.totals && state.contours.totals.length
      ? state.contours.totals
      : null) ||
    (state.quickContours?.totals && state.quickContours.totals.length
      ? state.quickContours.totals
      : []);
  const contourSummary = contourTotals.length
    ? contourTotals
        .map((entry) => {
          const label = entry.contour_label || `Contour ${entry.contour_index}`;
          const fx = entry.net_force?.[0] ?? 0;
          const fy = entry.net_force?.[1] ?? 0;
          const torque = entry.net_torque ?? 0;
          return `${label}: Fx=${formatNumber(fx, 4)} N/m, Fy=${formatNumber(
            fy,
            4
          )} N/m, τ=${formatNumber(torque, 4)} N·m/m`;
        })
        .join("<br/>")
    : "None";

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
        ${formatRange("|B|", stats.Bmag) || "|B|: n/a"}<br/>
        ${indicatorBlock || "Indicator: n/a"}
      </dd>
      <dt>Contours</dt>
      <dd>${contourSummary}</dd>
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
  const indicatorData = state.indicator || {};
  const indicatorLines = [];
  const indicatorPreview = state.indicatorPreview || null;
  const previewParams = indicatorPreview?.params || null;
  const magIndicator = valueAt(indicatorData.magnitude, cellIndex);
  const dirIndicator = valueAt(indicatorData.direction, cellIndex);
  const combinedIndicator = valueAt(indicatorData.combined, cellIndex);
  const weightedCombined = indicatorPreview ? valueAt(indicatorPreview.combined, cellIndex) : null;
  const alphaPreview = indicatorPreview ? valueAt(indicatorPreview.alphas, cellIndex) : null;
  if (Number.isFinite(weightedCombined)) {
    indicatorLines.push(`Combined (weighted) = ${formatNumber(weightedCombined, 4)}`);
  } else if (Number.isFinite(combinedIndicator)) {
    indicatorLines.push(`Combined = ${formatNumber(combinedIndicator, 4)}`);
  }
  if (Number.isFinite(magIndicator)) {
    indicatorLines.push(`Magnitude = ${formatNumber(magIndicator, 4)} 1/m`);
  }
  if (Number.isFinite(dirIndicator)) {
    indicatorLines.push(`Direction = ${formatNumber(dirIndicator, 4)} 1/m`);
  }
  if (Number.isFinite(alphaPreview) && previewParams) {
    indicatorLines.push(
      `Alpha preview = ${formatNumber(alphaPreview, 4)}× (limits ${formatNumber(
        previewParams.alphaMin,
        3
      )}–${formatNumber(previewParams.alphaMax, 3)}×, ref=${formatNumber(
        indicatorPreview.ref,
        4
      )}, gain=${formatNumber(previewParams.indicatorGain, 3)})`
    );
  }
  const neighborLines = [];
  const neighborList =
    Array.isArray(state.neighbors?.[cellIndex]) && state.neighbors[cellIndex]
      ? state.neighbors[cellIndex]
      : [];
  neighborList.forEach((nb) => {
    const nbCombined = indicatorPreview
      ? valueAt(indicatorPreview.combined, nb)
      : valueAt(indicatorData.combined, nb);
    if (Number.isFinite(nbCombined)) {
      neighborLines.push(`#${nb}: ${formatNumber(nbCombined, 4)}`);
    }
  });
  if (neighborLines.length) {
    indicatorLines.push(`Neighbors → ${neighborLines.join(", ")}`);
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
      <dt>Indicator ${
        previewParams
          ? `(weights dir=${formatNumber(previewParams.directionWeight, 3)}, mag=${formatNumber(
              previewParams.magnitudeWeight,
              3
            )})`
          : "(weights=1)"
      }</dt>
      <dd>${indicatorLines.join("<br/>") || "n/a"}</dd>
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
  const hasContours =
    Array.isArray(state.contours?.segments) &&
    state.contours.segments.length > 0;
  setToggleAvailability(layerContourBToggle, "contourB", hasContours);
  setToggleAvailability(layerContourForceToggle, "contourForce", hasContours);
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
  if (type === "polygon") {
    const explicitVertices = normalizeVertices(shapeDef.vertices || entry.vertices);
    const explicitHoles = normalizeHoles(shapeDef.holes || entry.holes);
    if (explicitVertices && explicitVertices.length >= 3) {
      return {
        type: "polygon",
        vertices: explicitVertices,
        holes: explicitHoles && explicitHoles.length ? explicitHoles : undefined,
        role,
      };
    }
    const radius = firstFiniteNumber(
      shapeDef.radius,
      shapeDef.size && typeof shapeDef.size === "object" ? shapeDef.size.radius : null,
      typeof shapeDef.size === "number" ? shapeDef.size : null
    );
    let sides = Math.round(
      firstFiniteNumber(shapeDef.sides, shapeDef.n, entry.sides, entry.n, 6)
    );
    if (!Number.isFinite(sides)) {
      sides = 6;
    }
    sides = Math.max(3, Math.min(4096, sides));
    if (!(radius > 0) || sides < 3) {
      return null;
    }
    const rotation = normalizeAngleDegrees(
      firstFiniteNumber(shapeDef.rotation, entry.rotation, shapeDef.angle, entry.angle, 0)
    );
    return { type: "polygon", center, radius, sides, rotation, role };
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

function normalizeVertices(raw) {
  if (!Array.isArray(raw) || raw.length < 3) {
    return null;
  }
  const verts = [];
  for (const v of raw) {
    if (Array.isArray(v) && v.length >= 2) {
      const x = Number(v[0]);
      const y = Number(v[1]);
      if (Number.isFinite(x) && Number.isFinite(y)) {
        verts.push([x, y]);
      }
    } else if (v && typeof v === "object") {
      const x = firstFiniteNumber(v.x, v.X, v[0]);
      const y = firstFiniteNumber(v.y, v.Y, v[1]);
      if (Number.isFinite(x) && Number.isFinite(y)) {
        verts.push([x, y]);
      }
    }
  }
  return verts.length >= 3 ? verts : null;
}

function normalizeHoles(raw) {
  if (!Array.isArray(raw) || raw.length === 0) {
    return null;
  }
  const holes = [];
  for (const hole of raw) {
    const verts = normalizeVertices(hole);
    if (verts && verts.length >= 3) {
      holes.push(verts);
    }
  }
  return holes.length ? holes : null;
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
  const materialTris =
    Array.isArray(state.materialTriangles) && state.materialTriangles.length
      ? state.materialTriangles
      : null;

  ctx.save();
  const drawTri = (i) => {
    const tri = tris[i];
    const pa = nodes[tri[0]];
    const pb = nodes[tri[1]];
    const pc = nodes[tri[2]];
    if (!pa || !pb || !pc) return;
    const regionId = Array.isArray(regionIds) ? regionIds[i] : null;
    const fillStyle = materialColor(regionId, jzField ? jzField[i] : 0, jzThreshold);
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
  };

  if (materialTris) {
    for (const idx of materialTris) {
      drawTri(idx);
    }
  } else {
    for (let i = 0; i < tris.length; i += 1) {
      drawTri(i);
    }
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
    } else if (shape.type === "polygon") {
      drawDesignPolygon(shape);
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
    case "contour":
      return TARGET_CONTOUR_COLOR;
    default:
      return TARGET_MAGNET_COLOR;
  }
}

function regularPolygonVertices(center, radius, sides, rotationDeg = 0) {
  if (!Array.isArray(center) || center.length < 2 || !(radius > 0) || sides < 3) {
    return [];
  }
  const verts = [];
  const rotationRad = (rotationDeg * Math.PI) / 180;
  for (let i = 0; i < sides; i += 1) {
    const theta = rotationRad + (i * 2 * Math.PI) / sides;
    verts.push([
      center[0] + radius * Math.cos(theta),
      center[1] + radius * Math.sin(theta),
    ]);
  }
  return verts;
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

function drawDesignPolygon(shape) {
  if (Array.isArray(shape.vertices) && shape.vertices.length >= 3) {
    const outer = shape.vertices.map(([x, y]) => worldToScreen(x, y));
    ctx.beginPath();
    outer.forEach((pt, idx) => {
      if (idx === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    });
    ctx.closePath();
    ctx.stroke();
    if (Array.isArray(shape.holes)) {
      shape.holes.forEach((hole) => {
        const verts = hole.map(([x, y]) => worldToScreen(x, y));
        if (verts.length < 3) return;
        ctx.beginPath();
        verts.forEach((pt, idx) => {
          if (idx === 0) ctx.moveTo(pt.x, pt.y);
          else ctx.lineTo(pt.x, pt.y);
        });
        ctx.closePath();
        ctx.stroke();
      });
    }
    return;
  }
  const sides = Math.max(3, Math.round(shape.sides || 0));
  const radius = shape.radius ?? 0;
  if (!(radius > 0) || sides < 3) {
    return;
  }
  const vertsWorld = regularPolygonVertices(
    shape.center,
    radius,
    sides,
    shape.rotation ?? shape.angle ?? 0
  );
  if (!vertsWorld.length) return;
  const verts = vertsWorld.map(([x, y]) => worldToScreen(x, y));
  ctx.beginPath();
  verts.forEach((pt, idx) => {
    if (idx === 0) ctx.moveTo(pt.x, pt.y);
    else ctx.lineTo(pt.x, pt.y);
  });
  ctx.closePath();
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

function drawContourB() {
  if (!state.layers.contourB) return;
  const segments = state.contours?.segments || [];
  if (!segments.length) return;
  const bounds = state.mesh?.bounds || { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  const domainDiag = Math.hypot(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY);
  const maxB = Math.max(
    ...segments.map((seg) => {
      const mag = typeof seg.Bmag === "number" ? seg.Bmag : null;
      if (Number.isFinite(mag)) return Math.abs(mag);
      if (Array.isArray(seg.B) && seg.B.length >= 2) return Math.hypot(seg.B[0], seg.B[1]);
      return 0;
    }),
    0
  );
  if (!(maxB > 0)) return;
  const baseLength = domainDiag * 0.045;
  ctx.save();
  ctx.strokeStyle = CONTOUR_B_COLOR;
  ctx.fillStyle = CONTOUR_B_COLOR;
  ctx.lineWidth = 1.2;
  for (const seg of segments) {
    const mid = seg.mid;
    const vec = seg.B || [];
    const bx = Number(vec[0] ?? 0);
    const by = Number(vec[1] ?? 0);
    const mag = Number.isFinite(seg.Bmag) ? Math.abs(seg.Bmag) : Math.hypot(bx, by);
    if (!(mag > 0) || !Array.isArray(mid) || mid.length < 2) continue;
    const scale = (mag / maxB) * baseLength;
    const dirX = bx / mag;
    const dirY = by / mag;
    const start = worldToScreen(mid[0], mid[1]);
    const end = worldToScreen(mid[0] + dirX * scale, mid[1] + dirY * scale);
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    drawArrowHead(end, start);
  }
  ctx.restore();
}

function drawContourForces() {
  if (!state.layers.contourForce) return;
  const segments = state.contours?.segments || [];
  if (!segments.length) return;
  const bounds = state.mesh?.bounds || { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  const domainDiag = Math.hypot(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY);
  const maxF = Math.max(
    ...segments.map((seg) => {
      const f = seg.force || [];
      if (Array.isArray(f) && f.length >= 2) return Math.hypot(f[0], f[1]);
      return 0;
    }),
    0
  );
  if (!(maxF > 0)) return;
  const baseLength = domainDiag * 0.06;
  ctx.save();
  ctx.strokeStyle = CONTOUR_FORCE_COLOR;
  ctx.fillStyle = CONTOUR_FORCE_COLOR;
  ctx.lineWidth = 1.4;
  for (const seg of segments) {
    const mid = seg.mid;
    const f = seg.force || [];
    const fx = Number(f[0] ?? 0);
    const fy = Number(f[1] ?? 0);
    const mag = Math.hypot(fx, fy);
    if (!(mag > 0) || !Array.isArray(mid) || mid.length < 2) continue;
    const scale = (mag / maxF) * baseLength;
    const dirX = fx / mag;
    const dirY = fy / mag;
    const start = worldToScreen(mid[0], mid[1]);
    const end = worldToScreen(mid[0] + dirX * scale, mid[1] + dirY * scale);
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
  const safeRange = Math.max(range, 1e-12);
  const maxVal = minVal + safeRange;
  const t = normalizeToUnit(value, minVal, maxVal);
  const redPoint = clamp(normalizeToUnit(B_MAG_RED_POINT_TESLA, minVal, maxVal), 0.05, 0.95);
  return bMagnitudeGradient(t, redPoint);
}

function bMagnitudeColorLog(value, logExtent) {
  if (!logExtent) {
    return bMagnitudeGradient(0, 1);
  }
  const safeRange = Math.max(logExtent.range, 1e-12);
  const minLog = logExtent.minLog;
  const logVal = value > 0 ? safeLog10(value) : -Infinity;
  const t = Number.isFinite(logVal)
    ? clamp((logVal - minLog) / safeRange, 0, 1)
    : 0;
  const redLog = safeLog10(B_MAG_RED_POINT_TESLA);
  const redPoint = clamp(
    Number.isFinite(redLog) ? (redLog - minLog) / safeRange : 0,
    0.05,
    0.95
  );
  return bMagnitudeGradient(t, redPoint);
}

function bMagnitudeGradient(t, redPoint) {
  const clampedRed = clamp(redPoint, 0, 1);
  const clampedValue = clamp(t, 0, 1);
  const eased = Math.pow(clampedValue, 0.7); // spread lower values apart so 0.5T isn't lost next to 2T
  const lowStop = clampedRed * 0.35;
  const warmStop = clampedRed * 0.7;
  const redStop = clampedRed;
  const highStop = redStop + (1 - redStop) * 0.55;

  if (eased <= lowStop) {
    const localT = lowStop > 0 ? eased / lowStop : 0;
    return lerpColor(B_MAG_MIN_COLOR, B_MAG_LOW_COLOR, localT);
  }
  if (eased <= warmStop) {
    const localT = (eased - lowStop) / Math.max(warmStop - lowStop, 1e-9);
    return lerpColor(B_MAG_LOW_COLOR, B_MAG_MIDWARM_COLOR, localT);
  }
  if (eased <= redStop) {
    const localT = (eased - warmStop) / Math.max(redStop - warmStop, 1e-9);
    return lerpColor(B_MAG_MIDWARM_COLOR, B_MAG_MID_COLOR, localT);
  }
  if (eased <= highStop) {
    const localT = (eased - redStop) / Math.max(highStop - redStop, 1e-9);
    return lerpColor(B_MAG_MID_COLOR, B_MAG_HIGH_COLOR, localT);
  }
  const localT = (eased - highStop) / Math.max(1 - highStop, 1e-9);
  return lerpColor(B_MAG_HIGH_COLOR, B_MAG_MAX_COLOR, localT);
}

function normalizeToUnit(value, minVal, maxVal) {
  const range = maxVal - minVal;
  if (!(range > 0)) {
    return 0;
  }
  return clamp((value - minVal) / range, 0, 1);
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

layerContourBToggle?.addEventListener("change", () => {
  state.layers.contourB = layerContourBToggle.checked;
  render();
});

layerContourForceToggle?.addEventListener("change", () => {
  state.layers.contourForce = layerContourForceToggle.checked;
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

function percentile(values, pct) {
  if (!Array.isArray(values) || values.length === 0) {
    return NaN;
  }
  const clipped = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (clipped.length === 0) {
    return NaN;
  }
  const rank = clamp((pct / 100) * (clipped.length - 1), 0, clipped.length - 1);
  const low = Math.floor(rank);
  const high = Math.ceil(rank);
  const frac = rank - low;
  if (high >= clipped.length) {
    return clipped[clipped.length - 1];
  }
  return clipped[low] * (1 - frac) + clipped[high] * frac;
}

function finiteOr(...vals) {
  for (const v of vals) {
    const num = Number(v);
    if (Number.isFinite(num)) {
      return num;
    }
  }
  return null;
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
