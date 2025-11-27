const bootstrap = window.__BOOTSTRAP__ || {};
const BASE_GRID = bootstrap.defaultGrid || { Nx: 120, Ny: 120, Lx: 1, Ly: 1 };
const DEFAULT_GRID_SANITIZED = sanitizeGrid(BASE_GRID, BASE_GRID);
const MATERIAL_DEFAULTS = {
  magnet: { label: 'Permanent Magnet', color: '#e4572e', params: { mu_r: 1.05, Mx: 0, My: 800000 } },
  steel: { label: 'Steel', color: '#2e86de', params: { mu_r: 1000 } },
  wire: { label: 'Wire', color: '#f5a623', params: { current: 5000 } },
  air: { label: 'Air', color: '#7b8ba1', params: { mu_r: 1.0 } },
  contour: { label: 'Contour', color: '#111827', params: {} },
};

const state = {
  cases: bootstrap.cases || [],
  caseName: bootstrap.defaultCase || '',
  definitionName: '',
  grid: deepCopy(DEFAULT_GRID_SANITIZED),
  objects: [],
  selectedId: null,
  selectedIds: [],
  tool: 'select',
  rotatePivot: 'selection',
  dirty: false,
  status: '',
  view: null,
  lastAdaptiveMesh: null,
  lastRunSnapshot: null,
  lastSolveSnapshot: null,
  meshVersion: 0,
  lastSolvedMeshVersion: 0,
  hasBField: false,
  runBusy: false,
  adaptiveBusy: false,
  runStartedAt: null,
  debug: {
    showDxfPoints: false,
    previewPoints: null,
    previewSegments: null,
  },
};
state.view = createDefaultView(state.grid);
state.lastAdaptiveMesh = state.grid.mesh?.type === 'uniform' ? null : deepCopy(state.grid.mesh);

const elements = {};
const canvasState = {
  mode: null,
  pointerId: null,
  start: null,
  shapeId: null,
  offset: null,
  groupOffsets: null,
  panAnchor: null,
  panStartCanvas: null,
  panHasMoved: false,
  pendingDeselect: false,
};
const canvasPadding = 30;
const VIEW_MIN_ZOOM = 0.25;
const VIEW_MAX_ZOOM = 20;
const VIEW_STEP = 1.2;
const WHEEL_ZOOM_SENSITIVITY = 420;
const WHEEL_DELTA_CLAMP = 600;
const PAN_DRAG_THRESHOLD_PX = 4;
const PAN_DRAG_THRESHOLD_SQ = PAN_DRAG_THRESHOLD_PX * PAN_DRAG_THRESHOLD_PX;
const DEG2RAD = Math.PI / 180;
const POLYGON_MIN_SIDES = 3;
const POLYGON_MAX_SIDES = 4096;
const DXF_UNIT_SCALE = 0.001;
let drawScheduled = false;
const BUILDER_EVENT_PREFIX = 'casebuilder:';

function emitBuilderEvent(name, detail = {}) {
  window.dispatchEvent(new CustomEvent(`${BUILDER_EVENT_PREFIX}${name}`, { detail }));
}

function deepCopy(value) {
  return JSON.parse(JSON.stringify(value));
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function formatDuration(ms) {
  if (!Number.isFinite(ms) || ms < 0) return null;
  const seconds = ms / 1000;
  if (seconds < 60) {
    return seconds < 10 ? `${seconds.toFixed(2)}s` : `${seconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const rem = seconds % 60;
  const remStr = rem < 10 ? rem.toFixed(1).padStart(4, '0') : rem.toFixed(1);
  return `${minutes}m ${remStr}s`;
}

function startRunTimer() {
  state.runStartedAt = performance.now();
}

function stopRunTimer() {
  if (state.runStartedAt === null) return null;
  const elapsedMs = performance.now() - state.runStartedAt;
  state.runStartedAt = null;
  return elapsedMs;
}

function createDefaultView(grid) {
  return {
    centerX: grid.Lx / 2,
    centerY: grid.Ly / 2,
    zoom: 1,
  };
}

function getViewWindow() {
  const width = state.grid.Lx / state.view.zoom;
  const height = state.grid.Ly / state.view.zoom;
  return {
    minX: state.view.centerX - width / 2,
    minY: state.view.centerY - height / 2,
    maxX: state.view.centerX + width / 2,
    maxY: state.view.centerY + height / 2,
    width,
    height,
  };
}

function clampViewCenter() {
  const viewWidth = state.grid.Lx / state.view.zoom;
  const viewHeight = state.grid.Ly / state.view.zoom;
  const halfW = viewWidth / 2;
  const halfH = viewHeight / 2;
  if (viewWidth >= state.grid.Lx) {
    state.view.centerX = state.grid.Lx / 2;
  } else {
    state.view.centerX = clamp(state.view.centerX, halfW, state.grid.Lx - halfW);
  }
  if (viewHeight >= state.grid.Ly) {
    state.view.centerY = state.grid.Ly / 2;
  } else {
    state.view.centerY = clamp(state.view.centerY, halfH, state.grid.Ly - halfH);
  }
}

function updateZoomLabel() {
  if (!elements.zoomLevelLabel) return;
  const percent = Math.round(state.view.zoom * 100);
  elements.zoomLevelLabel.textContent = `${percent}%`;
}

function resetView(options = {}) {
  const { schedule = true } = options;
  state.view = createDefaultView(state.grid);
  if (state.debug) {
  state.debug.dxfSource = null;
    state.debug.previewPoints = null;
    state.debug.previewSegments = null;
  }
  updateZoomLabel();
  if (schedule) {
    scheduleDraw();
  }
}

function setZoom(nextZoom, focus = null) {
  const zoom = clamp(nextZoom, VIEW_MIN_ZOOM, VIEW_MAX_ZOOM);
  const prevZoom = state.view.zoom;
  if (Math.abs(prevZoom - zoom) < 1e-4) {
    updateZoomLabel();
    return;
  }
  const prevRect = getViewWindow();
  state.view.zoom = zoom;
  if (focus) {
    const newWidth = state.grid.Lx / state.view.zoom;
    const newHeight = state.grid.Ly / state.view.zoom;
    const relX = (focus.x - prevRect.minX) / prevRect.width;
    const relY = (focus.y - prevRect.minY) / prevRect.height;
    const minX = focus.x - relX * newWidth;
    const minY = focus.y - relY * newHeight;
    state.view.centerX = minX + newWidth / 2;
    state.view.centerY = minY + newHeight / 2;
  }
  clampViewCenter();
  updateZoomLabel();
  scheduleDraw();
}

function zoomBy(multiplier, focus = null) {
  setZoom(state.view.zoom * multiplier, focus);
}

function normalizeWheelDelta(event) {
  let delta = event.deltaY;
  if (event.deltaMode === (window.WheelEvent ? window.WheelEvent.DOM_DELTA_LINE : 1)) {
    delta *= 16;
  } else if (event.deltaMode === (window.WheelEvent ? window.WheelEvent.DOM_DELTA_PAGE : 2)) {
    delta *= 800;
  }
  return clamp(delta, -WHEEL_DELTA_CLAMP, WHEEL_DELTA_CLAMP);
}

function normalizeAngleDegrees(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return 0;
  }
  let angle = num % 360;
  if (angle > 180) {
    angle -= 360;
  } else if (angle <= -180) {
    angle += 360;
  }
  return angle;
}

function normalizeAngle(value) {
  let deg = Number(value);
  if (!Number.isFinite(deg)) deg = 0;
  deg = deg % 360;
  if (deg < 0) deg += 360;
  return deg;
}

function rectAngleRadians(shape) {
  const angle = Number(shape?.angle) || 0;
  return angle * DEG2RAD;
}

function rotatePoint(dx, dy, angleRad) {
  const cosA = Math.cos(angleRad);
  const sinA = Math.sin(angleRad);
  return {
    x: dx * cosA - dy * sinA,
    y: dx * sinA + dy * cosA,
  };
}

function rotatePointAround(px, py, origin, angleRad) {
  const dx = px - origin.x;
  const dy = py - origin.y;
  const rotated = rotatePoint(dx, dy, angleRad);
  return { x: origin.x + rotated.x, y: origin.y + rotated.y };
}

function hasExplicitVertices(shape) {
  return (
    (Array.isArray(shape?.vertices) && shape.vertices.length >= 3) ||
    (Array.isArray(shape?.holes) && shape.holes.length > 0)
  );
}

function sanitizeVertices(rawVerts) {
  if (!Array.isArray(rawVerts)) return [];
  const verts = rawVerts
    .map((v) => {
      if (Array.isArray(v) && v.length >= 2) return [Number(v[0]) || 0, Number(v[1]) || 0];
      if (v && typeof v === 'object') return [Number(v.x) || 0, Number(v.y) || 0];
      return null;
    })
    .filter(Boolean);
  return verts;
}

function sanitizeLoops(rawLoops) {
  if (!Array.isArray(rawLoops)) return [];
  return rawLoops
    .map((loop) => sanitizeVertices(loop))
    .filter((loop) => loop.length >= 3);
}

function polygonCentroid(vertices) {
  if (!vertices.length) return { x: 0, y: 0 };
  let areaAcc = 0;
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < vertices.length; i += 1) {
    const [x0, y0] = vertices[i];
    const [x1, y1] = vertices[(i + 1) % vertices.length];
    const cross = x0 * y1 - x1 * y0;
    areaAcc += cross;
    cx += (x0 + x1) * cross;
    cy += (y0 + y1) * cross;
  }
  if (Math.abs(areaAcc) < 1e-9) {
    const sum = vertices.reduce(
      (acc, v) => ({ x: acc.x + v[0], y: acc.y + v[1] }),
      { x: 0, y: 0 }
    );
    return { x: sum.x / vertices.length, y: sum.y / vertices.length };
  }
  const factor = 1 / (3 * areaAcc);
  return { x: cx * factor, y: cy * factor };
}

function polygonSignedArea(vertices) {
  if (!vertices.length) return 0;
  let area = 0;
  for (let i = 0; i < vertices.length; i += 1) {
    const [x1, y1] = vertices[i];
    const [x2, y2] = vertices[(i + 1) % vertices.length];
    area += x1 * y2 - x2 * y1;
  }
  return 0.5 * area;
}

function polygonRadius(shape) {
  const loops = hasExplicitVertices(shape) ? getPolygonLoops(shape) : [getPolygonVertices(shape)];
  const verts = loops.flat();
  if (!verts.length) return 0;
  const center = shapeCenter(shape);
  return verts.reduce((max, pt) => {
    const dx = pt.x - center.x;
    const dy = pt.y - center.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    return Math.max(max, dist);
  }, 0);
}

function translateExplicitPolygon(shape, dx, dy) {
  if (!hasExplicitVertices(shape) || (!dx && !dy)) return;
  if (Array.isArray(shape.vertices)) {
    shape.vertices = sanitizeVertices(shape.vertices).map(([x, y]) => [x + dx, y + dy]);
  }
  if (Array.isArray(shape.holes)) {
    shape.holes = sanitizeLoops(shape.holes).map((loop) => loop.map(([x, y]) => [x + dx, y + dy]));
  }
}

function shapeBounds(shape) {
  if (!shape) return null;
  const type = typeof shape.type === 'string' ? shape.type.toLowerCase() : '';
  if (type === 'circle') {
    const center = shapeCenter(shape);
    const r = Math.max(0, Number(shape.radius) || 0);
    return { minX: center.x - r, maxX: center.x + r, minY: center.y - r, maxY: center.y + r };
  }
  if (type === 'ring') {
    const center = shapeCenter(shape);
    const r = Math.max(0, Number(shape.outer_radius || shape.outerRadius || 0));
    return { minX: center.x - r, maxX: center.x + r, minY: center.y - r, maxY: center.y + r };
  }
  if (type === 'rect') {
    const center = shapeCenter(shape);
    const halfW = (Number(shape.width) || 0) / 2;
    const halfH = (Number(shape.height) || 0) / 2;
    return {
      minX: center.x - halfW,
      maxX: center.x + halfW,
      minY: center.y - halfH,
      maxY: center.y + halfH,
    };
  }
  if (type === 'polygon') {
    const loops = getPolygonLoops(shape);
    if (!loops.length) return null;
    const xs = loops.flat().map((v) => v.x);
    const ys = loops.flat().map((v) => v.y);
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys),
    };
  }
  return null;
}

function shapeSignedArea(shape) {
  if (!shape) return 0;
  const type = typeof shape.type === 'string' ? shape.type.toLowerCase() : '';
  if (type === 'polygon') {
    const loops = getPolygonLoops(shape).map((loop) => loop.map((v) => [v.x, v.y]));
    if (!loops.length) return 0;
    let area = polygonSignedArea(loops[0]);
    for (let i = 1; i < loops.length; i += 1) {
      area -= Math.abs(polygonSignedArea(loops[i]));
    }
    return area;
  }
  if (type === 'circle') {
    const r = Math.max(0, Number(shape.radius) || 0);
    return Math.PI * r * r;
  }
  if (type === 'ring') {
    const outer = Math.max(0, Number(shape.outer_radius || shape.outerRadius || 0));
    const inner = Math.max(0, Number(shape.inner_radius || shape.innerRadius || 0));
    return Math.PI * (outer * outer - inner * inner);
  }
  if (type === 'rect') {
    const w = Math.max(0, Number(shape.width) || 0);
    const h = Math.max(0, Number(shape.height) || 0);
    return w * h;
  }
  return 0;
}

function getPolygonVertices(shape) {
  if (hasExplicitVertices(shape)) {
    return sanitizeVertices(shape.vertices).map(([x, y]) => ({ x, y }));
  }
  if (!shape?.center) return [];
  const sides = clamp(Math.round(shape.sides || 0), POLYGON_MIN_SIDES, POLYGON_MAX_SIDES);
  const radius = Math.max(0, shape.radius || 0);
  if (!(radius > 0) || sides < POLYGON_MIN_SIDES) return [];
  const rotation = ((Number(shape.rotation ?? shape.angle) || 0) % 360) * DEG2RAD;
  const verts = [];
  for (let i = 0; i < sides; i += 1) {
    const theta = rotation + (i * Math.PI * 2) / sides;
    verts.push({
      x: shape.center[0] + radius * Math.cos(theta),
      y: shape.center[1] + radius * Math.sin(theta),
    });
  }
  return verts;
}

function getPolygonLoops(shape) {
  if (hasExplicitVertices(shape)) {
    const outer = sanitizeVertices(shape.vertices).map(([x, y]) => ({ x, y }));
    const holes = sanitizeLoops(shape.holes).map((loop) => loop.map(([x, y]) => ({ x, y })));
    const loops = [outer, ...holes].filter((loop) => loop.length >= 3);
    return loops;
  }
  return [getPolygonVertices(shape)].filter((loop) => loop.length >= 3);
}

function getRectCorners(shape) {
  if (!shape?.center) return [];
  const halfW = (shape.width || 0) / 2;
  const halfH = (shape.height || 0) / 2;
  const angle = rectAngleRadians(shape);
  const offsets = [
    [-halfW, -halfH],
    [halfW, -halfH],
    [halfW, halfH],
    [-halfW, halfH],
  ];
  return offsets.map(([dx, dy]) => {
    const rotated = rotatePoint(dx, dy, angle);
    return {
      x: shape.center[0] + rotated.x,
      y: shape.center[1] + rotated.y,
    };
  });
}

function pointInRotatedRect(point, shape) {
  if (!shape?.center) return false;
  const angle = rectAngleRadians(shape);
  const rotated = rotatePoint(point.x - shape.center[0], point.y - shape.center[1], -angle);
  const halfW = (shape.width || 0) / 2;
  const halfH = (shape.height || 0) / 2;
  return Math.abs(rotated.x) <= halfW && Math.abs(rotated.y) <= halfH;
}

function pointInPolygon(point, vertices) {
  if (!Array.isArray(vertices) || vertices.length < 3) return false;
  let inside = false;
  for (let i = 0, j = vertices.length - 1; i < vertices.length; j = i, i += 1) {
    const xi = vertices[i].x;
    const yi = vertices[i].y;
    const xj = vertices[j].x;
    const yj = vertices[j].y;
    const intersect =
      yi > point.y !== yj > point.y &&
      point.x < ((xj - xi) * (point.y - yi)) / (yj - yi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function pointInPolygonLoops(point, loops) {
  if (!Array.isArray(loops) || !loops.length) return false;
  let inside = false;
  loops.forEach((loop) => {
    if (pointInPolygon(point, loop)) {
      inside = !inside;
    }
  });
  return inside;
}

function init() {
  cacheElements();
  bindEvents();
  renderCaseSelect();
  refreshCaseList({ silent: true });
  if (state.caseName) {
    elements.caseNameInput.value = state.caseName;
    loadCase(state.caseName);
  } else {
    setActiveTool('select');
    renderAll();
  }
  updateAdaptiveButtonState();
  updateRunButtonState();
  emitBuilderEvent('ready', { caseName: state.caseName || null, cases: [...state.cases] });
}

function cacheElements() {
  elements.caseSelect = document.getElementById('caseSelect');
  elements.caseNameInput = document.getElementById('caseNameInput');
  elements.definitionNameInput = document.getElementById('definitionNameInput');
  elements.gridNx = document.getElementById('gridNx');
  elements.gridNy = document.getElementById('gridNy');
  elements.gridLx = document.getElementById('gridLx');
  elements.gridLy = document.getElementById('gridLy');
  elements.gridMeshMode = document.getElementById('gridMeshMode');
  elements.gridFineX = document.getElementById('gridFineX');
  elements.gridFineY = document.getElementById('gridFineY');
  elements.gridCoarseX = document.getElementById('gridCoarseX');
  elements.gridCoarseY = document.getElementById('gridCoarseY');
  elements.expFinePitch = document.getElementById('expFinePitch');
  elements.expCoarsePitch = document.getElementById('expCoarsePitch');
  elements.gridFocusPad = document.getElementById('gridFocusPad');
  elements.gridFocusFalloff = document.getElementById('gridFocusFalloff');
  elements.gridQualityAngle = document.getElementById('gridQualityAngle');
  elements.meshFocusMagnet = document.getElementById('meshFocusMagnet');
  elements.meshFocusSteel = document.getElementById('meshFocusSteel');
  elements.meshFocusWire = document.getElementById('meshFocusWire');
  elements.meshBtn = document.getElementById('adaptiveRunBtn');
  elements.fieldFocusEnabled = document.getElementById('fieldFocusEnabled');
  elements.fieldDirectionWeight = document.getElementById('fieldDirectionWeight');
  elements.fieldMagnitudeWeight = document.getElementById('fieldMagnitudeWeight');
  elements.fieldIndicatorNeutral = document.getElementById('fieldIndicatorNeutral');
  elements.fieldIndicatorPercentile = document.getElementById('fieldIndicatorPercentile');
  elements.fieldIndicatorGain = document.getElementById('fieldIndicatorGain');
  elements.fieldScaleMin = document.getElementById('fieldScaleMin');
  elements.fieldScaleMax = document.getElementById('fieldScaleMax');
  elements.fieldIndicatorClipLow = document.getElementById('fieldIndicatorClipLow');
  elements.fieldIndicatorClipHigh = document.getElementById('fieldIndicatorClipHigh');
  elements.fieldRatioLimit = document.getElementById('fieldRatioLimit');
  elements.fieldSizeMin = document.getElementById('fieldSizeMin');
  elements.fieldSizeMax = document.getElementById('fieldSizeMax');
  elements.gradedMeshControls = document.getElementById('gradedMeshControls');
  elements.experimentalMeshControls = document.getElementById('experimentalMeshControls');
  elements.uniformMeshControls = document.getElementById('uniformMeshControls');
  elements.toolButtons = Array.from(document.querySelectorAll('.tool-btn'));
  elements.shapeList = document.getElementById('shapeList');
  elements.deleteShapeBtn = document.getElementById('deleteShapeBtn');
  elements.duplicateShapeBtn = document.getElementById('duplicateShapeBtn');
  elements.groupSelectedBtn = document.getElementById('groupSelectedBtn');
  elements.rotateSelectedBtn = document.getElementById('rotateSelectedBtn');
  elements.rotateAngle = document.getElementById('rotateAngle');
  elements.rotatePivot = document.getElementById('rotatePivot');
  elements.dxfFileInput = document.getElementById('dxfFileInput');
  elements.dxfMaterialSelect = document.getElementById('dxfMaterialSelect');
  elements.dxfDebugPoints = document.getElementById('dxfDebugPoints');
  elements.dxfDebugPanel = document.getElementById('dxfDebugPanel');
  elements.dxfDebugText = document.getElementById('dxfDebugText');
  elements.buildDxfSolidBtn = document.getElementById('buildDxfSolidBtn');
  elements.inspectorEmpty = document.getElementById('inspectorEmpty');
  elements.inspectorFields = document.getElementById('inspectorFields');
  elements.shapeLabel = document.getElementById('shapeLabel');
  elements.shapeGrouped = document.getElementById('shapeGrouped');
  elements.materialSelect = document.getElementById('materialSelect');
  elements.rectInputs = {
    centerX: document.getElementById('rectCenterX'),
    centerY: document.getElementById('rectCenterY'),
    width: document.getElementById('rectWidth'),
    height: document.getElementById('rectHeight'),
    angle: document.getElementById('rectAngle'),
  };
  elements.circleInputs = {
    centerX: document.getElementById('circleCenterX'),
    centerY: document.getElementById('circleCenterY'),
    radius: document.getElementById('circleRadius'),
  };
  elements.ringInputs = {
    centerX: document.getElementById('ringCenterX'),
    centerY: document.getElementById('ringCenterY'),
    outerRadius: document.getElementById('ringOuterRadius'),
    innerRadius: document.getElementById('ringInnerRadius'),
  };
  elements.polyInputs = {
    centerX: document.getElementById('polyCenterX'),
    centerY: document.getElementById('polyCenterY'),
    radius: document.getElementById('polyRadius'),
    sides: document.getElementById('polySides'),
    rotation: document.getElementById('polyRotation'),
  };
  elements.paramInputs = {
    magnet: {
      mu: document.getElementById('magnetMu'),
      mx: document.getElementById('magnetMx'),
      my: document.getElementById('magnetMy'),
    },
    steel: {
      mu: document.getElementById('steelMu'),
    },
    air: {
      mu: document.getElementById('airMu'),
    },
    wire: {
      current: document.getElementById('wireCurrent'),
    },
  };
  elements.statusLine = document.getElementById('statusLine');
  elements.canvas = document.getElementById('builderCanvas');
  elements.zoomInBtn = document.getElementById('zoomInBtn');
  elements.zoomOutBtn = document.getElementById('zoomOutBtn');
  elements.zoomResetBtn = document.getElementById('zoomResetBtn');
  elements.zoomLevelLabel = document.getElementById('zoomLevelLabel');
  elements.loadCaseBtn = document.getElementById('loadCaseBtn');
  elements.newCaseBtn = document.getElementById('newCaseBtn');
  elements.saveCaseBtn = document.getElementById('saveCaseBtn');
  elements.runCaseBtn = document.getElementById('runCaseBtn');
  elements.downloadBtn = document.getElementById('downloadBtn');
}

function bindEvents() {
  elements.loadCaseBtn.addEventListener('click', () => {
    const target = elements.caseSelect.value || elements.caseNameInput.value.trim();
    if (target) {
      loadCase(target);
    }
  });
  elements.newCaseBtn.addEventListener('click', () => {
    const proposal = elements.caseNameInput.value.trim();
    newCase(proposal || 'new_case');
  });
  elements.caseSelect.addEventListener('change', () => {
    const selected = elements.caseSelect.value;
    if (selected) {
      elements.caseNameInput.value = selected;
      state.caseName = selected;
      updateAdaptiveButtonState();
    }
  });
  elements.caseNameInput.addEventListener('input', () => {
    state.caseName = elements.caseNameInput.value.trim();
    updateAdaptiveButtonState();
  });
  elements.definitionNameInput.addEventListener('input', () => {
    state.definitionName = elements.definitionNameInput.value.trim();
    markDirty();
  });
  const gridInputs = [
    elements.gridLx,
    elements.gridLy,
    elements.gridFineX,
    elements.gridFineY,
    elements.gridCoarseX,
    elements.gridCoarseY,
    elements.expFinePitch,
    elements.expCoarsePitch,
    elements.gridFocusPad,
    elements.gridFocusFalloff,
    elements.gridQualityAngle,
    elements.fieldDirectionWeight,
    elements.fieldMagnitudeWeight,
    elements.fieldIndicatorNeutral,
    elements.fieldIndicatorPercentile,
    elements.fieldIndicatorGain,
    elements.fieldScaleMin,
    elements.fieldScaleMax,
    elements.fieldIndicatorClipLow,
    elements.fieldIndicatorClipHigh,
    elements.fieldRatioLimit,
    elements.fieldSizeMin,
    elements.fieldSizeMax,
    elements.gridNx,
    elements.gridNy,
  ].filter(Boolean);
  gridInputs.forEach((input) => {
    input.addEventListener('input', () => updateGridFromInputs());
  });
  if (elements.gridMeshMode) {
    elements.gridMeshMode.addEventListener('change', () => updateGridFromInputs());
  }
  ['meshFocusMagnet', 'meshFocusSteel', 'meshFocusWire'].forEach((key) => {
    const checkbox = elements[key];
    if (checkbox) {
      checkbox.addEventListener('change', () => updateGridFromInputs());
    }
  });
  if (elements.fieldFocusEnabled) {
    elements.fieldFocusEnabled.addEventListener('change', () => updateGridFromInputs());
  }
  elements.toolButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      setActiveTool(btn.dataset.tool);
    });
  });
  elements.shapeList.addEventListener('click', (event) => {
    const item = event.target.closest('.shape-item');
    if (item) {
      selectShape(item.dataset.id, { additive: event.shiftKey, toggle: event.shiftKey });
    }
  });
  elements.deleteShapeBtn.addEventListener('click', deleteSelectedShape);
  elements.duplicateShapeBtn.addEventListener('click', duplicateSelectedShape);
  elements.groupSelectedBtn?.addEventListener('click', groupSelectedShapes);
  elements.rotateSelectedBtn?.addEventListener('click', rotateSelectedShapes);
  if (elements.rotatePivot) {
    elements.rotatePivot.addEventListener('change', () => {
      state.rotatePivot = elements.rotatePivot.value;
    });
  }
  elements.materialSelect.addEventListener('change', () => updateSelectedMaterial(elements.materialSelect.value));
  elements.shapeLabel.addEventListener('input', () => {
    const shape = getSelectedShape();
    if (!shape) return;
    shape.label = elements.shapeLabel.value;
    markDirty();
    renderShapeList();
  });
  elements.shapeGrouped?.addEventListener('change', () => {
    const shape = getSelectedShape();
    if (!shape) return;
    if (!elements.shapeGrouped.checked) {
      shape.group = '';
    }
    markDirty();
    renderShapeList();
    scheduleDraw();
  });
  Object.entries(elements.rectInputs).forEach(([key, input]) => {
    input.addEventListener('input', () => updateRectField(key));
  });
  Object.entries(elements.circleInputs).forEach(([key, input]) => {
    input.addEventListener('input', () => updateCircleField(key));
  });
  Object.entries(elements.ringInputs).forEach(([key, input]) => {
    input.addEventListener('input', () => updateRingField(key));
  });
  Object.entries(elements.polyInputs).forEach(([key, input]) => {
    input.addEventListener('input', () => updatePolygonField(key));
  });
  elements.paramInputs.magnet.mu.addEventListener('input', () => updateParam('magnet', 'mu_r', elements.paramInputs.magnet.mu.value));
  elements.paramInputs.magnet.mx.addEventListener('input', () => updateParam('magnet', 'Mx', elements.paramInputs.magnet.mx.value));
  elements.paramInputs.magnet.my.addEventListener('input', () => updateParam('magnet', 'My', elements.paramInputs.magnet.my.value));
  elements.paramInputs.steel.mu.addEventListener('input', () => updateParam('steel', 'mu_r', elements.paramInputs.steel.mu.value));
  elements.paramInputs.air.mu.addEventListener('input', () => updateParam('air', 'mu_r', elements.paramInputs.air.mu.value));
  elements.paramInputs.wire.current.addEventListener('input', () => updateParam('wire', 'current', elements.paramInputs.wire.current.value));
  elements.saveCaseBtn.addEventListener('click', saveCaseDefinition);
  elements.runCaseBtn.addEventListener('click', runCasePipeline);
  if (elements.meshBtn) {
    elements.meshBtn.addEventListener('click', runMeshPipeline);
  }
  elements.downloadBtn.addEventListener('click', downloadDefinition);
  if (elements.buildDxfSolidBtn) {
    elements.buildDxfSolidBtn.addEventListener('click', importAndBuildDxf);
  }
  if (elements.dxfMaterialSelect && elements.materialSelect) {
    elements.dxfMaterialSelect.value = elements.materialSelect.value;
  }
  if (elements.dxfDebugPoints) {
    elements.dxfDebugPoints.addEventListener('change', () => {
      state.debug.showDxfPoints = !!elements.dxfDebugPoints.checked;
      renderDxfDebugPanel();
      scheduleDraw();
    });
  }
  renderDxfDebugPanel();

  elements.canvas.addEventListener('pointerdown', onPointerDown);
  elements.canvas.addEventListener('pointermove', onPointerMove);
  elements.canvas.addEventListener('pointerup', onPointerUp);
  elements.canvas.addEventListener('pointerleave', onPointerUp);
  elements.canvas.addEventListener('wheel', onCanvasWheel, { passive: false });
  if (elements.zoomInBtn) {
    elements.zoomInBtn.addEventListener('click', () => zoomBy(VIEW_STEP));
  }
  if (elements.zoomOutBtn) {
    elements.zoomOutBtn.addEventListener('click', () => zoomBy(1 / VIEW_STEP));
  }
  if (elements.zoomResetBtn) {
    elements.zoomResetBtn.addEventListener('click', () => resetView());
  }
}

function renderCaseSelect() {
  elements.caseSelect.innerHTML = '';
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = '— select —';
  elements.caseSelect.appendChild(placeholder);
  state.cases.forEach((name) => {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    if (name === state.caseName) {
      opt.selected = true;
    }
    elements.caseSelect.appendChild(opt);
  });
}

function renderAll() {
  updateGridInputs();
  renderShapeList();
  updateInspector();
  renderRotatePivotOptions();
  updateZoomLabel();
  scheduleDraw();
}

function updateGridInputs() {
  const mesh = state.grid.mesh || {};
  const fallbackMesh = state.lastAdaptiveMesh || {};
  elements.gridNx.value = state.grid.Nx;
  elements.gridNy.value = state.grid.Ny;
  elements.gridLx.value = state.grid.Lx;
  elements.gridLy.value = state.grid.Ly;
  if (elements.gridMeshMode) {
    const mode =
      mesh.type === 'uniform' ? 'uniform' : mesh.type === 'experimental' ? 'experimental' : 'point_cloud';
    elements.gridMeshMode.value = mode;
  }
  if (elements.expFinePitch) {
    elements.expFinePitch.value = mesh.fine ?? fallbackMesh.fine ?? '';
  }
  if (elements.expCoarsePitch) {
    elements.expCoarsePitch.value = mesh.coarse ?? fallbackMesh.coarse ?? '';
  }
  if (elements.gridFineX) {
    elements.gridFineX.value = mesh.fine ?? fallbackMesh.fine ?? '';
  }
  if (elements.gridCoarseX) {
    elements.gridCoarseX.value = mesh.coarse ?? fallbackMesh.coarse ?? '';
  }
  const meshY = mesh.y || {};
  const fallbackY = fallbackMesh.y || {};
  if (elements.gridFineY) {
    elements.gridFineY.value = meshY.fine ?? fallbackY.fine ?? '';
  }
  if (elements.gridCoarseY) {
    elements.gridCoarseY.value = meshY.coarse ?? fallbackY.coarse ?? '';
  }
  if (elements.gridFocusPad) {
    elements.gridFocusPad.value = mesh.focus_pad ?? fallbackMesh.focus_pad ?? '';
  }
  if (elements.gridFocusFalloff) {
    elements.gridFocusFalloff.value = mesh.focus_falloff ?? fallbackMesh.focus_falloff ?? '';
  }
  if (elements.gridQualityAngle) {
    elements.gridQualityAngle.value = mesh.quality_min_angle ?? fallbackMesh.quality_min_angle ?? '';
  }
  const focusMaterials = Array.isArray(mesh.focus_materials)
    ? mesh.focus_materials
    : ['magnet', 'steel', 'wire'];
  if (mesh.type !== 'uniform') {
    const focusSet = new Set(focusMaterials);
    if (elements.meshFocusMagnet) {
      elements.meshFocusMagnet.checked = focusSet.has('magnet');
    }
    if (elements.meshFocusSteel) {
      elements.meshFocusSteel.checked = focusSet.has('steel');
    }
    if (elements.meshFocusWire) {
      elements.meshFocusWire.checked = focusSet.has('wire');
    }
  } else {
    const fallbackMaterials = Array.isArray(state.lastAdaptiveMesh?.focus_materials)
      ? state.lastAdaptiveMesh.focus_materials
      : ['magnet', 'steel', 'wire'];
    const cachedSet = new Set(fallbackMaterials);
    if (elements.meshFocusMagnet) {
      elements.meshFocusMagnet.checked = cachedSet.has('magnet');
    }
    if (elements.meshFocusSteel) {
      elements.meshFocusSteel.checked = cachedSet.has('steel');
    }
    if (elements.meshFocusWire) {
      elements.meshFocusWire.checked = cachedSet.has('wire');
    }
  }
  const defaultFieldFocus = DEFAULT_GRID_SANITIZED.mesh?.field_focus || {};
  const fieldFocus =
    mesh.field_focus ||
    state.lastAdaptiveMesh?.field_focus ||
    defaultFieldFocus;
  if (elements.fieldFocusEnabled) {
    const enabled =
      fieldFocus.enabled === undefined ? true : !!fieldFocus.enabled;
    elements.fieldFocusEnabled.checked = enabled;
  }
  if (elements.fieldDirectionWeight) {
    elements.fieldDirectionWeight.value =
      fieldFocus.direction_weight ??
      state.lastAdaptiveMesh?.field_focus?.direction_weight ??
      defaultFieldFocus.direction_weight ??
      1;
  }
  if (elements.fieldMagnitudeWeight) {
    elements.fieldMagnitudeWeight.value =
      fieldFocus.magnitude_weight ??
      state.lastAdaptiveMesh?.field_focus?.magnitude_weight ??
      defaultFieldFocus.magnitude_weight ??
      1;
  }
  const fallbackFocus = state.lastAdaptiveMesh?.field_focus || defaultFieldFocus;
  if (elements.fieldIndicatorNeutral) {
    const neutral =
      fieldFocus.indicator_neutral ??
      fallbackFocus?.indicator_neutral ??
      null;
    elements.fieldIndicatorNeutral.value =
      neutral === null || neutral === undefined ? '' : neutral;
  }
  if (elements.fieldIndicatorGain) {
    elements.fieldIndicatorGain.value =
      fieldFocus.indicator_gain ??
      fallbackFocus?.indicator_gain ??
      0.4;
  }
  if (elements.fieldIndicatorPercentile) {
    elements.fieldIndicatorPercentile.value =
      fieldFocus.indicator_percentile ??
      fallbackFocus?.indicator_percentile ??
      85;
  }
  if (elements.fieldScaleMin) {
    elements.fieldScaleMin.value =
      fieldFocus.scale_min ??
      fallbackFocus?.scale_min ??
      0.5;
  }
  if (elements.fieldScaleMax) {
    elements.fieldScaleMax.value =
      fieldFocus.scale_max ??
      fallbackFocus?.scale_max ??
      2;
  }
  if (elements.fieldRatioLimit) {
    elements.fieldRatioLimit.value =
      fieldFocus.ratio_limit ??
      fallbackFocus?.ratio_limit ??
      1.7;
  }
  if (elements.fieldIndicatorClipLow) {
    elements.fieldIndicatorClipLow.value =
      (fieldFocus.indicator_clip && fieldFocus.indicator_clip[0]) ??
      (fallbackFocus.indicator_clip && fallbackFocus.indicator_clip[0]) ??
      5;
  }
  if (elements.fieldIndicatorClipHigh) {
    elements.fieldIndicatorClipHigh.value =
      (fieldFocus.indicator_clip && fieldFocus.indicator_clip[1]) ??
      (fallbackFocus.indicator_clip && fallbackFocus.indicator_clip[1]) ??
      95;
  }
  if (elements.fieldSizeMin) {
    const val = fieldFocus.size_min ?? fallbackFocus.size_min;
    elements.fieldSizeMin.value = Number.isFinite(val) ? val : '';
  }
  if (elements.fieldSizeMax) {
    const val = fieldFocus.size_max ?? fallbackFocus.size_max;
    elements.fieldSizeMax.value = Number.isFinite(val) ? val : '';
  }
  elements.definitionNameInput.value = state.definitionName || '';
  elements.caseNameInput.value = state.caseName || '';
  updateMeshModeVisibility();
}

function updateGridFromInputs() {
  const prevGrid = state.grid;
  const mesh = prevGrid.mesh || {};
  const rawMode = elements.gridMeshMode?.value || mesh.type || 'point_cloud';
  const mode = rawMode === 'uniform' ? 'uniform' : rawMode === 'experimental' ? 'experimental' : 'point_cloud';
  const Lx = Math.max(0.01, Number(elements.gridLx?.value) || prevGrid.Lx);
  const Ly = Math.max(0.01, Number(elements.gridLy?.value) || prevGrid.Ly);
  const focusBoxes = Array.isArray(mesh.focus_boxes) ? deepCopy(mesh.focus_boxes) : undefined;
  let Nx = prevGrid.Nx;
  let Ny = prevGrid.Ny;
  let nextMesh = mesh;
  const prevFieldFocus = mesh.field_focus || {};

  if (mode === 'uniform') {
    Nx = Math.max(4, parseInt(elements.gridNx?.value, 10) || prevGrid.Nx);
    Ny = Math.max(4, parseInt(elements.gridNy?.value, 10) || prevGrid.Ny);
    nextMesh = { type: 'uniform' };
  } else if (mode === 'experimental') {
    const minDim = Math.max(Math.min(Lx, Ly), 0.01);
    let fine = readPositiveNumber(elements.expFinePitch?.value, mesh.fine ?? minDim / 150, 1e-5) || minDim / 150;
    let coarse =
      readPositiveNumber(elements.expCoarsePitch?.value, mesh.coarse ?? minDim / 40, 1e-5) ||
      Math.max(minDim / 40, fine);
    fine = Math.min(fine, coarse);
    const focusMaterials = Array.isArray(mesh.focus_materials)
      ? mesh.focus_materials
      : ['magnet', 'steel', 'wire'];
    const focusPadSource =
      mesh.type === 'experimental'
        ? mesh.focus_pad
        : state.lastAdaptiveMesh?.focus_pad;
    const focusPad = Math.max(0, Number(focusPadSource ?? 0.01) || 0.01);
    const focusFalloff = Math.max(
      0,
      Number(
        mesh.type === 'experimental'
          ? mesh.focus_falloff
          : state.lastAdaptiveMesh?.focus_falloff ?? 0.5 * focusPad
      ) || 0.5 * focusPad
    );
    if (elements.fieldFocusEnabled) {
      elements.fieldFocusEnabled.checked = false;
    }
    nextMesh = {
      type: 'experimental',
      fine,
      coarse,
      focus_pad: focusPad,
      focus_falloff: focusFalloff,
      focus_materials: focusMaterials,
    };
    if (focusBoxes) {
      nextMesh.focus_boxes = focusBoxes;
    }
    state.lastAdaptiveMesh = deepCopy(nextMesh);
  } else {
    const minDim = Math.max(Math.min(Lx, Ly), 0.01);
    let fineX = readPositiveNumber(elements.gridFineX?.value, mesh.fine ?? minDim / 150, 1e-5) || minDim / 150;
    let coarseX =
      readPositiveNumber(elements.gridCoarseX?.value, mesh.coarse ?? minDim / 40, 1e-5) ||
      Math.max(minDim / 40, fineX);
    fineX = Math.min(fineX, coarseX);
    let fineY = readPositiveNumber(elements.gridFineY?.value, mesh.y?.fine, 1e-5);
    let coarseY = readPositiveNumber(elements.gridCoarseY?.value, mesh.y?.coarse, 1e-5);
    if (fineY && coarseY) {
      if (fineY > coarseY) fineY = coarseY;
    } else if (coarseY && !fineY) {
      fineY = Math.min(coarseY, fineX);
    } else if (fineY && !coarseY) {
      coarseY = Math.max(fineY, coarseX);
    }
    const focusPad = Math.max(0, Number(elements.gridFocusPad?.value) || mesh.focus_pad || 0.02 * minDim);
    const focusFalloff = Math.max(
      0,
      Number(elements.gridFocusFalloff?.value) || mesh.focus_falloff || 0.5 * focusPad
    );
    const focusMaterials = [];
    if (elements.meshFocusMagnet?.checked) focusMaterials.push('magnet');
    if (elements.meshFocusSteel?.checked) focusMaterials.push('steel');
    if (elements.meshFocusWire?.checked) focusMaterials.push('wire');

    nextMesh = {
      type: 'point_cloud',
      fine: fineX,
      coarse: coarseX,
      focus_pad: focusPad,
      focus_falloff: focusFalloff,
      focus_materials: focusMaterials,
    };
    if (fineY || coarseY) {
      nextMesh.y = {};
      if (fineY) nextMesh.y.fine = fineY;
      if (coarseY) nextMesh.y.coarse = coarseY;
    }
    if (focusBoxes) {
      nextMesh.focus_boxes = focusBoxes;
    }
    const minAngle = readNonNegativeNumber(
      elements.gridQualityAngle?.value,
      mesh.quality_min_angle ?? state.lastAdaptiveMesh?.quality_min_angle,
      28
    );
    if (Number.isFinite(minAngle)) {
      nextMesh.quality_min_angle = minAngle;
    } else if ('quality_min_angle' in nextMesh) {
      delete nextMesh.quality_min_angle;
    }
    const indicatorPercentile = readNonNegativeNumber(
      elements.fieldIndicatorPercentile?.value,
      prevFieldFocus.indicator_percentile ?? state.lastAdaptiveMesh?.field_focus?.indicator_percentile,
      85
    );
    let indicatorNeutral =
      prevFieldFocus.indicator_neutral ??
      state.lastAdaptiveMesh?.field_focus?.indicator_neutral ??
      null;
    if (elements.fieldIndicatorNeutral) {
      const rawNeutral = elements.fieldIndicatorNeutral.value;
      if (typeof rawNeutral === 'string') {
        if (rawNeutral.trim() === '') {
          indicatorNeutral = null;
        } else {
          const parsed = Number(rawNeutral);
          if (Number.isFinite(parsed)) {
            indicatorNeutral = parsed;
          }
        }
      }
    }
    const indicatorGain = readNonNegativeNumber(
      elements.fieldIndicatorGain?.value,
      prevFieldFocus.indicator_gain ?? state.lastAdaptiveMesh?.field_focus?.indicator_gain,
      0.4
    );
    const scaleMin =
      readPositiveNumber(
        elements.fieldScaleMin?.value,
        prevFieldFocus.scale_min ?? state.lastAdaptiveMesh?.field_focus?.scale_min,
        1e-3
      ) ?? 0.5;
    let scaleMax =
      readPositiveNumber(
        elements.fieldScaleMax?.value,
        prevFieldFocus.scale_max ?? state.lastAdaptiveMesh?.field_focus?.scale_max,
        scaleMin
      ) ?? scaleMin;
    if (scaleMax < scaleMin) {
      scaleMax = scaleMin;
    }
    const ratioLimit =
      readPositiveNumber(
        elements.fieldRatioLimit?.value,
        prevFieldFocus.ratio_limit ?? state.lastAdaptiveMesh?.field_focus?.ratio_limit,
        1
      ) ?? 1.7;
    const clipLow =
      readNonNegativeNumber(
        elements.fieldIndicatorClipLow?.value,
        prevFieldFocus.indicator_clip?.[0] ?? state.lastAdaptiveMesh?.field_focus?.indicator_clip?.[0],
        5
      ) ?? 5;
    const clipHigh =
      readNonNegativeNumber(
        elements.fieldIndicatorClipHigh?.value,
        prevFieldFocus.indicator_clip?.[1] ?? state.lastAdaptiveMesh?.field_focus?.indicator_clip?.[1],
        95
      ) ?? 95;
    let sizeMinOverride = readPositiveNumber(
      elements.fieldSizeMin?.value,
      prevFieldFocus.size_min ?? state.lastAdaptiveMesh?.field_focus?.size_min,
      undefined
    );
    let sizeMaxOverride = readPositiveNumber(
      elements.fieldSizeMax?.value,
      prevFieldFocus.size_max ?? state.lastAdaptiveMesh?.field_focus?.size_max,
      undefined
    );
    if (sizeMaxOverride && sizeMinOverride && sizeMaxOverride < sizeMinOverride) {
      sizeMaxOverride = sizeMinOverride;
    }
    const fieldFocusSettings = {
      enabled: elements.fieldFocusEnabled
        ? !!elements.fieldFocusEnabled.checked
        : (prevFieldFocus.enabled ?? true),
      direction_weight: readNonNegativeNumber(
        elements.fieldDirectionWeight?.value,
        prevFieldFocus.direction_weight ?? state.lastAdaptiveMesh?.field_focus?.direction_weight,
        1
      ),
      magnitude_weight: readNonNegativeNumber(
        elements.fieldMagnitudeWeight?.value,
        prevFieldFocus.magnitude_weight ?? state.lastAdaptiveMesh?.field_focus?.magnitude_weight,
        1
      ),
      indicator_gain: indicatorGain,
      indicator_neutral: indicatorNeutral,
      indicator_percentile: indicatorPercentile,
      scale_min: scaleMin,
      scale_max: scaleMax,
      ratio_limit: ratioLimit,
      indicator_clip: [clipLow, clipHigh],
    };
    if (sizeMinOverride) fieldFocusSettings.size_min = sizeMinOverride;
    if (sizeMaxOverride) fieldFocusSettings.size_max = sizeMaxOverride;
    nextMesh.field_focus = fieldFocusSettings;
    state.lastAdaptiveMesh = deepCopy(nextMesh);
  }

  state.grid = {
    ...prevGrid,
    Nx,
    Ny,
    Lx,
    Ly,
    mesh: nextMesh,
  };
  clampViewCenter();
  updateZoomLabel();
  markDirty();
  updateMeshModeVisibility();
  scheduleDraw();
}

function updateMeshModeVisibility() {
  const modeRaw = elements.gridMeshMode?.value || state.grid.mesh?.type || 'point_cloud';
  const isUniform = modeRaw === 'uniform';
  const isExperimental = modeRaw === 'experimental';
  if (elements.gradedMeshControls) {
    elements.gradedMeshControls.classList.toggle('hidden', isUniform || isExperimental);
  }
  if (elements.experimentalMeshControls) {
    elements.experimentalMeshControls.classList.toggle('hidden', !isExperimental);
  }
  if (elements.uniformMeshControls) {
    elements.uniformMeshControls.classList.toggle('hidden', !isUniform);
  }
}

function setActiveTool(tool) {
  state.tool = tool;
  elements.toolButtons.forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.tool === tool);
  });
}

function newCase(name) {
  state.caseName = name;
  state.definitionName = name;
  state.grid = deepCopy(DEFAULT_GRID_SANITIZED);
  state.lastAdaptiveMesh = state.grid.mesh?.type === 'uniform' ? null : deepCopy(state.grid.mesh);
  state.objects = [];
  state.selectedId = null;
  state.selectedIds = [];
  state.dirty = false;
  state.lastRunSnapshot = null;
  state.lastSolveSnapshot = null;
  state.meshVersion = 0;
  state.lastSolvedMeshVersion = 0;
  state.hasBField = false;
  elements.caseNameInput.value = name;
  elements.definitionNameInput.value = name;
  resetView({ schedule: false });
  renderAll();
  setActiveTool('select');
  setStatus(`Started new case '${name}'.`, 'info');
  updateBFieldToggleAvailability();
  updateAdaptiveButtonState();
  updateRunButtonState();
  emitBuilderEvent('caseLoaded', { caseName: state.caseName, isNew: true });
}

async function loadCase(caseName) {
  try {
    const response = await fetch(`/api/case/${encodeURIComponent(caseName)}/definition`);
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const data = await response.json();
    const def = data.definition || {};
    state.caseName = data.case || caseName;
    state.definitionName = def.name || state.caseName;
    state.grid = sanitizeGrid(def.grid || {}, BASE_GRID);
    if (state.grid.mesh?.type !== 'uniform') {
      state.lastAdaptiveMesh = deepCopy(state.grid.mesh);
    } else if (!state.lastAdaptiveMesh && DEFAULT_GRID_SANITIZED.mesh?.type !== 'uniform') {
      state.lastAdaptiveMesh = deepCopy(DEFAULT_GRID_SANITIZED.mesh);
    }
    state.objects = (def.objects || []).map((obj, idx) => hydrateObject(obj, idx));
    resetView({ schedule: false });
    state.selectedId = state.objects.length ? state.objects[0].id : null;
    state.selectedIds = state.selectedId ? [state.selectedId] : [];
    state.dirty = false;
    state.lastRunSnapshot = null;
    state.lastSolveSnapshot = null;
    state.meshVersion = 0;
    state.lastSolvedMeshVersion = 0;
    renderCaseSelect();
    renderAll();
    setActiveTool('select');
    setStatus(`Loaded ${state.caseName}`, 'info');
    await refreshCaseStatus(state.caseName);
    updateAdaptiveButtonState();
    updateRunButtonState();
    emitBuilderEvent('caseLoaded', { caseName: state.caseName });
  } catch (err) {
    console.error(err);
    setStatus(`Failed to load case: ${err}`, 'error');
    emitBuilderEvent('caseLoadFailed', {
      caseName,
      error: err?.message || String(err),
    });
  }
}

async function refreshCaseStatus(caseName) {
  if (!caseName) {
    state.hasBField = false;
    updateBFieldToggleAvailability();
    updateAdaptiveButtonState();
    return;
  }
  try {
    const resp = await fetch(`/api/case/${encodeURIComponent(caseName)}/status`);
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    state.hasBField = !!data.has_solution;
    if (state.hasBField) {
      const fingerprint = fingerprintDefinition(serializeDefinition());
      state.lastSolveSnapshot = { caseName: state.caseName, fingerprint };
      state.lastSolvedMeshVersion = state.meshVersion;
    }
    updateBFieldToggleAvailability();
    updateAdaptiveButtonState();
    updateRunButtonState();
  } catch (err) {
    console.warn('Failed to refresh case status', err);
  }
}

function readPositiveNumber(value, fallback, min = 1e-5) {
  const num = Number(value);
  if (Number.isFinite(num) && num >= min) {
    return num;
  }
  if (Number.isFinite(fallback) && fallback >= min) {
    return fallback;
  }
  return null;
}

function readNonNegativeNumber(value, fallback, defaultValue = 0) {
  const num = Number(value);
  if (Number.isFinite(num) && num >= 0) {
    return num;
  }
  if (Number.isFinite(fallback) && fallback >= 0) {
    return fallback;
  }
  if (Number.isFinite(defaultValue) && defaultValue >= 0) {
    return defaultValue;
  }
  return 0;
}

function readNonNegativeInt(value, fallback, defaultValue = 0) {
  const num = parseInt(value, 10);
  if (Number.isFinite(num) && num >= 0) {
    return num;
  }
  if (Number.isFinite(fallback) && fallback >= 0) {
    return Math.floor(fallback);
  }
  if (Number.isFinite(defaultValue) && defaultValue >= 0) {
    return Math.floor(defaultValue);
  }
  return 0;
}

function sanitizeGrid(grid, fallback = BASE_GRID) {
  const source = typeof grid === 'object' && grid !== null ? grid : {};
  const fb = typeof fallback === 'object' && fallback !== null ? fallback : BASE_GRID;
  const Nx = Math.max(4, parseInt(source.Nx ?? fb.Nx ?? 120, 10) || 120);
  const Ny = Math.max(4, parseInt(source.Ny ?? fb.Ny ?? 120, 10) || 120);
  const Lx = Math.max(0.01, parseFloat(source.Lx ?? fb.Lx ?? 1.0) || 1.0);
  const Ly = Math.max(0.01, parseFloat(source.Ly ?? fb.Ly ?? 1.0) || 1.0);
  const mesh = sanitizeMesh(source.mesh, fb.mesh);
  return { Nx, Ny, Lx, Ly, mesh };
}

function sanitizeMesh(mesh, fallbackMesh = {}) {
  const source = typeof mesh === 'object' && mesh !== null ? mesh : {};
  const fallback = typeof fallbackMesh === 'object' && fallbackMesh !== null ? fallbackMesh : {};
  const typeRaw =
    typeof source.type === 'string'
      ? source.type.toLowerCase()
      : typeof fallback.type === 'string'
      ? fallback.type.toLowerCase()
      : 'point_cloud';
  const adaptiveTypes = new Set(['point_cloud', 'point', 'pointcloud', 'graded', 'adaptive', 'delaunay']);
  let type = 'point_cloud';
  if (typeRaw === 'uniform') {
    type = 'uniform';
  } else if (typeRaw === 'experimental' || typeRaw === 'equilateral') {
    type = 'experimental';
  } else if (adaptiveTypes.has(typeRaw)) {
    type = 'point_cloud';
  }
  if (type === 'uniform') {
    return { type: 'uniform' };
  }
  const coarse = readPositiveNumber(
    source.coarse ?? source.max_dx ?? fallback.coarse ?? fallback.max_dx ?? 0.02,
    0.02
  ) || 0.02;
  const fineDefault = Math.min(coarse / 3, coarse);
  const fine = readPositiveNumber(
    source.fine ?? source.min_dx ?? fallback.fine ?? fallback.min_dx ?? fineDefault,
    fineDefault
  ) || fineDefault;
  const ySource = source.y || {};
  const yFallback = fallback.y || {};
  const fineY = readPositiveNumber(ySource.fine ?? source.fine_y ?? yFallback.fine, null);
  const coarseY = readPositiveNumber(ySource.coarse ?? source.coarse_y ?? yFallback.coarse, null);
  const defaultFocusPad = type === 'experimental' ? 0.01 : 0.02;
  const focusPadSource =
    type === 'experimental'
      ? source.focus_pad ?? defaultFocusPad
      : source.focus_pad ?? fallback.focus_pad ?? defaultFocusPad;
  const focusPad = Math.max(0, Number(focusPadSource) || defaultFocusPad);
  const focusFalloffSource =
    type === 'experimental'
      ? source.focus_falloff ?? 0.5 * focusPad
      : source.focus_falloff ?? fallback.focus_falloff ?? 0.5 * focusPad;
  const focusFalloff = Math.max(0, Number(focusFalloffSource) || 0.5 * focusPad);
  const focusMaterialsRaw = Array.isArray(source.focus_materials)
    ? source.focus_materials
    : fallback.focus_materials;
  const focusMaterials = Array.isArray(focusMaterialsRaw)
    ? focusMaterialsRaw
        .map((entry) => (typeof entry === 'string' ? entry.toLowerCase() : ''))
        .filter((entry, idx, arr) => entry && arr.indexOf(entry) === idx)
    : ['magnet', 'steel', 'wire'];
  const meshSpec = {
    type,
    fine,
    coarse,
    focus_pad: focusPad,
    focus_falloff: focusFalloff,
    focus_materials: focusMaterials,
  };
  if (fineY || coarseY) {
    meshSpec.y = {};
    if (fineY) meshSpec.y.fine = fineY;
    if (coarseY) meshSpec.y.coarse = coarseY;
  }
  if (Array.isArray(source.focus_boxes)) {
    meshSpec.focus_boxes = deepCopy(source.focus_boxes);
  } else if (Array.isArray(fallback.focus_boxes)) {
    meshSpec.focus_boxes = deepCopy(fallback.focus_boxes);
  }
  const minAngle = readNonNegativeNumber(
    source.quality_min_angle,
    fallback.quality_min_angle,
    28
  );
  if (minAngle) {
    meshSpec.quality_min_angle = minAngle;
  }
  const fieldFocus = sanitizeFieldFocus(source.field_focus, fallback.field_focus);
  if (fieldFocus) {
    meshSpec.field_focus = fieldFocus;
  }
  return meshSpec;
}

function sanitizeFieldFocus(spec, fallback) {
  const src = typeof spec === 'object' && spec !== null ? spec : {};
  const fb = typeof fallback === 'object' && fallback !== null ? fallback : {};
  const enabled =
    src.enabled !== undefined
      ? !!src.enabled
      : fb.enabled !== undefined
      ? !!fb.enabled
      : true;
  const directionWeight = readNonNegativeNumber(
    src.direction_weight,
    fb.direction_weight,
    1
  );
  const magnitudeWeight = readNonNegativeNumber(
    src.magnitude_weight,
    fb.magnitude_weight,
    1
  );
  const indicatorPercentile = clamp(
    readNonNegativeNumber(src.indicator_percentile, fb.indicator_percentile, 85) ?? 85,
    0,
    100
  );
  const indicatorGain = readNonNegativeNumber(
    src.indicator_gain,
    fb.indicator_gain,
    0.4
  );
  let indicatorNeutral =
    src.indicator_neutral !== undefined
      ? src.indicator_neutral
      : fb.indicator_neutral;
  if (typeof indicatorNeutral === 'string' && indicatorNeutral.trim() === '') {
    indicatorNeutral = null;
  } else if (indicatorNeutral !== null && indicatorNeutral !== undefined) {
    const parsed = Number(indicatorNeutral);
    indicatorNeutral = Number.isFinite(parsed) ? parsed : null;
  } else {
    indicatorNeutral = null;
  }
  const scaleMin =
    readPositiveNumber(src.scale_min, fb.scale_min ?? 0.5, 1e-3) ?? 0.5;
  let scaleMax =
    readPositiveNumber(src.scale_max, fb.scale_max ?? 2.0, scaleMin) ??
    scaleMin;
  if (scaleMax < scaleMin) {
    scaleMax = scaleMin;
  }
  const smoothPasses = readNonNegativeInt(
    src.smooth_passes,
    fb.smooth_passes,
    2
  );
  const sizeSmoothPasses = readNonNegativeInt(
    src.size_smooth_passes,
    fb.size_smooth_passes,
    smoothPasses
  );
  const ratioLimit =
    readPositiveNumber(src.ratio_limit, fb.ratio_limit, 1.7) ?? 1.7;
  const clipLow =
    readNonNegativeNumber(
      Array.isArray(src.indicator_clip) ? src.indicator_clip[0] : undefined,
      Array.isArray(fb.indicator_clip) ? fb.indicator_clip[0] : undefined,
      5
    ) ?? 5;
  const clipHigh =
    readNonNegativeNumber(
      Array.isArray(src.indicator_clip) ? src.indicator_clip[1] : undefined,
      Array.isArray(fb.indicator_clip) ? fb.indicator_clip[1] : undefined,
      95
    ) ?? 95;
  let sizeMin = readPositiveNumber(src.size_min, fb.size_min, undefined);
  let sizeMax = readPositiveNumber(src.size_max, fb.size_max, undefined);
  if (sizeMax && sizeMin && sizeMax < sizeMin) {
    sizeMax = sizeMin;
  }
  const fieldFocus = {
    enabled,
    direction_weight: directionWeight,
    magnitude_weight: magnitudeWeight,
    indicator_percentile: indicatorPercentile,
    indicator_gain: indicatorGain,
    indicator_neutral: indicatorNeutral,
    scale_min: scaleMin,
    scale_max: scaleMax,
    smooth_passes: smoothPasses,
    size_smooth_passes: sizeSmoothPasses,
    ratio_limit: ratioLimit,
    indicator_clip: [clipLow, clipHigh],
  };
  if (sizeMin) fieldFocus.size_min = sizeMin;
  if (sizeMax) fieldFocus.size_max = sizeMax;
  return fieldFocus;
}

function hydrateObject(raw, idx) {
  const material = MATERIAL_DEFAULTS[raw.material] ? raw.material : 'air';
  const id = raw.id || `obj-${Date.now()}-${idx}`;
  const label = raw.label || `${MATERIAL_DEFAULTS[material].label} ${idx + 1}`;
  const params = { ...deepCopy(MATERIAL_DEFAULTS[material].params), ...(raw.params || {}) };
  const shape = normalizeShape(raw.shape || {}, material);
  const group = typeof raw.group === 'string' ? raw.group : '';
  return { id, label, material, params, shape, group };
}

function normalizeShape(shape, material) {
  const typeRaw = typeof shape.type === 'string' ? shape.type.toLowerCase() : 'rect';
  const type = ['rect', 'circle', 'ring', 'polygon'].includes(typeRaw) ? typeRaw : 'rect';
  const center = Array.isArray(shape.center)
    ? { x: Number(shape.center[0]) || 0, y: Number(shape.center[1]) || 0 }
    : typeof shape.center === 'object'
    ? { x: Number(shape.center.x) || 0, y: Number(shape.center.y) || 0 }
    : { x: 0, y: 0 };
  if (type === 'rect') {
    const width = Math.abs(Number(shape.width) || Number(shape.size) || 0.05 * state.grid.Lx);
    const height = Math.abs(Number(shape.height) || Number(shape.size) || 0.05 * state.grid.Ly);
    const rawAngle = Number(shape.angle);
    const angle = Number.isFinite(rawAngle) ? normalizeAngleDegrees(rawAngle) : 0;
    return { type, center: [center.x, center.y], width, height, angle };
  }
  if (type === 'ring') {
    const maxDim = Math.max(state.grid.Lx, state.grid.Ly);
    let outer = Math.abs(
      Number(shape.outer_radius ?? shape.outerRadius ?? shape.radius ?? 0.03 * maxDim)
    );
    if (!(outer > 0)) {
      outer = 0.03 * maxDim;
    }
    let inner = Math.abs(Number(shape.inner_radius ?? shape.innerRadius ?? 0.5 * outer));
    if (!(inner < outer)) {
      inner = Math.max(0, outer - 0.001 * maxDim);
    }
    return { type, center: [center.x, center.y], outer_radius: outer, inner_radius: inner };
  }
  if (type === 'polygon') {
    const explicitVerts = sanitizeVertices(shape.vertices);
    const explicitHoles = sanitizeLoops(shape.holes);
    if (explicitVerts.length >= POLYGON_MIN_SIDES) {
      const centroid = polygonCentroid(explicitVerts);
      const radius = polygonRadius({
        type,
        center: [centroid.x, centroid.y],
        vertices: explicitVerts,
        holes: explicitHoles,
      });
      return {
        type,
        center: [centroid.x, centroid.y],
        vertices: explicitVerts,
        holes: explicitHoles,
        sides: explicitVerts.length,
        rotation: 0,
        radius,
      };
    }
    const maxDim = Math.max(state.grid.Lx, state.grid.Ly);
    let radius = Math.abs(Number(shape.radius) || 0.02 * maxDim);
    if (!(radius > 0)) {
      radius = 0.02 * maxDim;
    }
    let sides = Math.round(Number(shape.sides) || 6);
    if (!Number.isFinite(sides)) {
      sides = 6;
    }
    sides = clamp(sides, POLYGON_MIN_SIDES, POLYGON_MAX_SIDES);
    const rotation = normalizeAngleDegrees(Number(shape.rotation ?? shape.angle) || 0);
    return { type, center: [center.x, center.y], radius, sides, rotation };
  }
  const radius = Math.abs(Number(shape.radius) || 0.02 * Math.max(state.grid.Lx, state.grid.Ly));
  return { type: 'circle', center: [center.x, center.y], radius };
}

function shapeCenter(shape) {
  if (!shape) return { x: 0, y: 0 };
  if (hasExplicitVertices(shape)) {
    const cx = Array.isArray(shape.center) ? Number(shape.center[0]) : null;
    const cy = Array.isArray(shape.center) ? Number(shape.center[1]) : null;
    if (Number.isFinite(cx) && Number.isFinite(cy)) {
      return { x: cx, y: cy };
    }
    const centroid = polygonCentroid(sanitizeVertices(shape.vertices));
    shape.center = [centroid.x, centroid.y];
    return centroid;
  }
  if (!shape.center) return { x: 0, y: 0 };
  if (Array.isArray(shape.center)) {
    return { x: Number(shape.center[0]) || 0, y: Number(shape.center[1]) || 0 };
  }
  if (typeof shape.center === 'object') {
    return { x: Number(shape.center.x) || 0, y: Number(shape.center.y) || 0 };
  }
  return { x: 0, y: 0 };
}

function renderShapeList() {
  elements.shapeList.innerHTML = '';
  const selectedSet = new Set(state.selectedIds || []);
  state.objects.forEach((obj) => {
    const div = document.createElement('div');
    div.className = 'shape-item' + (selectedSet.has(obj.id) ? ' selected' : '');
    div.dataset.id = obj.id;
    const color = MATERIAL_DEFAULTS[obj.material]?.color || '#999';
    const typeLabel = obj.shape.type === 'polygon'
      ? `polygon (${obj.shape.sides}${obj.shape.holes?.length ? ` + ${obj.shape.holes.length} holes` : ''})`
      : obj.shape.type;
    const groupLabel = obj.group ? ` · ${obj.group}` : '';
    div.innerHTML = `
      <span><strong style="color:${color}">●</strong> ${obj.label}</span>
      <span>${typeLabel}${groupLabel}</span>
    `;
    elements.shapeList.appendChild(div);
  });
  const hasSelection = selectedSet.size > 0;
  elements.deleteShapeBtn.disabled = !hasSelection;
  elements.duplicateShapeBtn.disabled = !(selectedSet.size === 1);
  if (elements.groupSelectedBtn) {
    elements.groupSelectedBtn.disabled = selectedSet.size < 2;
  }
  if (elements.rotateSelectedBtn) {
    elements.rotateSelectedBtn.disabled = selectedSet.size === 0;
  }
  renderRotatePivotOptions();
}

function renderRotatePivotOptions() {
  if (!elements.rotatePivot) return;
  const prev = elements.rotatePivot.value || state.rotatePivot || 'selection';
  elements.rotatePivot.innerHTML = '';
  const optCentroid = document.createElement('option');
  optCentroid.value = 'selection';
  optCentroid.textContent = 'Selection centroid';
  elements.rotatePivot.appendChild(optCentroid);
  state.objects
    .filter((obj) => obj.shape?.type === 'circle' || obj.shape?.type === 'ring')
    .forEach((obj) => {
      const opt = document.createElement('option');
      opt.value = `shape:${obj.id}`;
      opt.textContent = `Center of ${obj.label || obj.id}`;
      elements.rotatePivot.appendChild(opt);
    });
  const valid = Array.from(elements.rotatePivot.options).some((o) => o.value === prev);
  elements.rotatePivot.value = valid ? prev : 'selection';
  state.rotatePivot = elements.rotatePivot.value;
}

function updateInspector() {
  const shape = getSelectedShape();
  if (!shape) {
    elements.inspectorEmpty.classList.remove('hidden');
    elements.inspectorFields.classList.add('hidden');
    return;
  }
  elements.inspectorEmpty.classList.add('hidden');
  elements.inspectorFields.classList.remove('hidden');
  elements.shapeLabel.value = shape.label;
  if (elements.shapeGrouped) {
    elements.shapeGrouped.checked = !!shape.group;
  }
  elements.materialSelect.value = shape.material;
  document.querySelectorAll('.geom-fields').forEach((el) => {
    el.classList.toggle('active', el.dataset.shape === shape.shape.type);
  });
  document.querySelectorAll('.param-fields').forEach((el) => {
    el.classList.toggle('active', el.dataset.material === shape.material);
  });
  if (shape.shape.type === 'rect') {
    elements.rectInputs.centerX.value = shape.shape.center[0].toFixed(4);
    elements.rectInputs.centerY.value = shape.shape.center[1].toFixed(4);
    elements.rectInputs.width.value = shape.shape.width.toFixed(4);
    elements.rectInputs.height.value = shape.shape.height.toFixed(4);
    const angle = Number(shape.shape.angle) || 0;
    elements.rectInputs.angle.value = angle.toFixed(2);
  } else if (shape.shape.type === 'circle') {
    elements.circleInputs.centerX.value = shape.shape.center[0].toFixed(4);
    elements.circleInputs.centerY.value = shape.shape.center[1].toFixed(4);
    elements.circleInputs.radius.value = shape.shape.radius.toFixed(4);
  } else if (shape.shape.type === 'ring') {
    elements.ringInputs.centerX.value = shape.shape.center[0].toFixed(4);
    elements.ringInputs.centerY.value = shape.shape.center[1].toFixed(4);
    elements.ringInputs.outerRadius.value = shape.shape.outer_radius.toFixed(4);
    elements.ringInputs.innerRadius.value = shape.shape.inner_radius.toFixed(4);
  } else if (shape.shape.type === 'polygon') {
    const center = shapeCenter(shape.shape);
    const explicit = hasExplicitVertices(shape.shape);
    const radiusVal = explicit ? polygonRadius(shape.shape) : shape.shape.radius;
    elements.polyInputs.centerX.value = center.x.toFixed(4);
    elements.polyInputs.centerY.value = center.y.toFixed(4);
    elements.polyInputs.radius.value = (radiusVal || 0).toFixed(4);
    elements.polyInputs.sides.value = explicit
      ? sanitizeVertices(shape.shape.vertices).length
      : shape.shape.sides;
    const rotation = explicit ? 0 : Number(shape.shape.rotation ?? shape.shape.angle) || 0;
    elements.polyInputs.rotation.value = rotation.toFixed(2);
    elements.polyInputs.radius.disabled = explicit;
    elements.polyInputs.sides.disabled = explicit;
    elements.polyInputs.rotation.disabled = explicit;
  }
  const params = shape.params || {};
  elements.paramInputs.magnet.mu.value = shape.material === 'magnet' && params.mu_r !== undefined ? params.mu_r : '';
  elements.paramInputs.magnet.mx.value = shape.material === 'magnet' && params.Mx !== undefined ? params.Mx : '';
  elements.paramInputs.magnet.my.value = shape.material === 'magnet' && params.My !== undefined ? params.My : '';
  elements.paramInputs.steel.mu.value = shape.material === 'steel' && params.mu_r !== undefined ? params.mu_r : '';
  elements.paramInputs.air.mu.value = shape.material === 'air' && params.mu_r !== undefined ? params.mu_r : '';
  elements.paramInputs.wire.current.value = shape.material === 'wire' && params.current !== undefined ? params.current : '';
}

function getSelectedShape() {
  return state.objects.find((obj) => obj.id === state.selectedId) || null;
}

function selectShape(id, options = {}) {
  const { additive = false, toggle = false } = options;
  if (!id) {
    state.selectedId = null;
    state.selectedIds = [];
  } else if (additive) {
    const already = state.selectedIds.includes(id);
    let next = [...state.selectedIds];
    if (toggle && already) {
      next = next.filter((entry) => entry !== id);
    } else if (!already) {
      next.push(id);
    }
    state.selectedIds = next;
    state.selectedId = next[next.length - 1] || null;
  } else {
    state.selectedIds = [id];
    state.selectedId = id;
  }
  renderShapeList();
  updateInspector();
  scheduleDraw();
}

function deleteSelectedShape() {
  if (!state.selectedIds.length) return;
  const toDelete = new Set(state.selectedIds);
  state.objects = state.objects.filter((obj) => !toDelete.has(obj.id));
  state.selectedId = state.objects.length ? state.objects[0].id : null;
  state.selectedIds = state.selectedId ? [state.selectedId] : [];
  markDirty();
  renderAll();
}

function duplicateSelectedShape() {
  const shape = getSelectedShape();
  if (!shape) return;
  const copy = deepCopy(shape);
  copy.id = uniqueId();
  copy.label = `${shape.label} copy`;
  const prevCenter = shapeCenter(copy.shape);
  if (copy.shape.center) {
    copy.shape.center[0] = clamp(copy.shape.center[0] + 0.01 * state.grid.Lx, 0, state.grid.Lx);
    copy.shape.center[1] = clamp(copy.shape.center[1] + 0.01 * state.grid.Ly, 0, state.grid.Ly);
  }
  const nextCenter = shapeCenter(copy.shape);
  translateExplicitPolygon(copy.shape, nextCenter.x - prevCenter.x, nextCenter.y - prevCenter.y);
  state.objects.push(copy);
  state.selectedId = copy.id;
  state.selectedIds = [copy.id];
  markDirty();
  renderAll();
}

function groupSelectedShapes() {
  if (!state.selectedIds || state.selectedIds.length < 2) return;
  const groupId = `group-${Math.random().toString(36).slice(2, 8)}`;
  const selectedSet = new Set(state.selectedIds);
  state.objects.forEach((obj) => {
    if (selectedSet.has(obj.id)) {
      obj.group = groupId;
    }
  });
  markDirty();
  renderShapeList();
  updateInspector();
  scheduleDraw();
}

function rotationPivotPoint(pivotValue, targetIds) {
  const ids = Array.isArray(targetIds) && targetIds.length ? targetIds : state.selectedIds;
  if (pivotValue === 'selection') {
    const centers = ids
      .map((id) => state.objects.find((obj) => obj.id === id))
      .filter(Boolean)
      .map((obj) => shapeCenter(obj.shape));
    if (!centers.length) return { x: 0, y: 0 };
    const sum = centers.reduce((acc, c) => ({ x: acc.x + c.x, y: acc.y + c.y }), { x: 0, y: 0 });
    return { x: sum.x / centers.length, y: sum.y / centers.length };
  }
  if (pivotValue.startsWith('shape:')) {
    const pivotId = pivotValue.slice('shape:'.length);
    const pivotShape = state.objects.find((obj) => obj.id === pivotId);
    if (pivotShape?.shape?.center) {
      return shapeCenter(pivotShape.shape);
    }
  }
  return { x: 0, y: 0 };
}

function rotateSelectedShapes() {
  if (!state.selectedIds.length) return;
  const angleDeg = parseFloat(elements.rotateAngle?.value);
  const delta = Number.isFinite(angleDeg) ? angleDeg : 0;
  const angleRad = delta * DEG2RAD;
  const groupMap = new Map();
  state.objects.forEach((obj) => {
    if (obj.group) {
      const arr = groupMap.get(obj.group) || [];
      arr.push(obj);
      groupMap.set(obj.group, arr);
    }
  });
  const rotationTargets = new Map();
  state.selectedIds.forEach((id) => {
    const obj = state.objects.find((o) => o.id === id);
    if (!obj) return;
    rotationTargets.set(obj.id, obj);
    if (obj.group && groupMap.has(obj.group)) {
      groupMap.get(obj.group).forEach((member) => rotationTargets.set(member.id, member));
    }
  });
  const targets = Array.from(rotationTargets.values());
  const targetIds = targets.map((t) => t.id);
  const pivotValue = elements.rotatePivot?.value || state.rotatePivot || 'selection';
  const pivot = rotationPivotPoint(pivotValue, targetIds);
  targets.forEach((obj) => {
    if (!obj.shape?.center) return;
    const center = shapeCenter(obj.shape);
    const rotated = rotatePointAround(center.x, center.y, pivot, angleRad);
    if (obj.shape.type === 'polygon' && hasExplicitVertices(obj.shape)) {
      obj.shape.vertices = sanitizeVertices(obj.shape.vertices).map(([x, y]) => {
        const rp = rotatePointAround(x, y, pivot, angleRad);
        return [rp.x, rp.y];
      });
      obj.shape.center = [
        clamp(rotated.x, 0, state.grid.Lx),
        clamp(rotated.y, 0, state.grid.Ly),
      ];
      constrainShape(obj);
    } else {
      obj.shape.center[0] = clamp(rotated.x, 0, state.grid.Lx);
      obj.shape.center[1] = clamp(rotated.y, 0, state.grid.Ly);
      if (obj.shape.type === 'rect') {
        obj.shape.angle = normalizeAngleDegrees((obj.shape.angle || 0) + delta);
        constrainShape(obj);
      } else if (obj.shape.type === 'polygon') {
        obj.shape.rotation = normalizeAngleDegrees((obj.shape.rotation ?? obj.shape.angle ?? 0) + delta);
        constrainShape(obj);
      } else {
        constrainShape(obj);
      }
    }
  });
  markDirty();
  renderShapeList();
  updateInspector();
  scheduleDraw();
  setStatus(`Rotated ${targets.length} shape(s) by ${delta.toFixed(2)}°`, 'info');
}

function uniqueId() {
  return `obj-${Math.random().toString(36).slice(2, 9)}`;
}

function updateRectField(field) {
  const shape = getSelectedShape();
  if (!shape || shape.shape.type !== 'rect') return;
  const value = parseFloat(elements.rectInputs[field].value);
  if (Number.isNaN(value)) return;
  if (field === 'centerX') shape.shape.center[0] = clamp(value, 0, state.grid.Lx);
  if (field === 'centerY') shape.shape.center[1] = clamp(value, 0, state.grid.Ly);
  if (field === 'width') shape.shape.width = Math.max(0, value);
  if (field === 'height') shape.shape.height = Math.max(0, value);
  if (field === 'angle') shape.shape.angle = normalizeAngleDegrees(value);
  markDirty();
  scheduleDraw();
}

function updateCircleField(field) {
  const shape = getSelectedShape();
  if (!shape || shape.shape.type !== 'circle') return;
  const value = parseFloat(elements.circleInputs[field].value);
  if (Number.isNaN(value)) return;
  if (field === 'centerX') shape.shape.center[0] = clamp(value, 0, state.grid.Lx);
  if (field === 'centerY') shape.shape.center[1] = clamp(value, 0, state.grid.Ly);
  if (field === 'radius') shape.shape.radius = Math.max(0, value);
  markDirty();
  scheduleDraw();
}

function updateRingField(field) {
  const shape = getSelectedShape();
  if (!shape || shape.shape.type !== 'ring') return;
  const value = parseFloat(elements.ringInputs[field].value);
  if (Number.isNaN(value)) return;
  if (field === 'centerX') shape.shape.center[0] = clamp(value, 0, state.grid.Lx);
  if (field === 'centerY') shape.shape.center[1] = clamp(value, 0, state.grid.Ly);
  if (field === 'outerRadius') shape.shape.outer_radius = Math.max(0, value);
  if (field === 'innerRadius') shape.shape.inner_radius = Math.max(0, value);
  constrainShape(shape);
  markDirty();
  scheduleDraw();
}

function updatePolygonField(field) {
  const shape = getSelectedShape();
  if (!shape || shape.shape.type !== 'polygon') return;
  const explicit = hasExplicitVertices(shape.shape);
  const value = parseFloat(elements.polyInputs[field].value);
  if (Number.isNaN(value)) return;
  if (field === 'centerX' || field === 'centerY') {
    const prev = shapeCenter(shape.shape);
    const next = { x: prev.x, y: prev.y };
    if (field === 'centerX') next.x = clamp(value, 0, state.grid.Lx);
    if (field === 'centerY') next.y = clamp(value, 0, state.grid.Ly);
    const dx = next.x - prev.x;
    const dy = next.y - prev.y;
    shape.shape.center = [next.x, next.y];
    if (explicit) {
      translateExplicitPolygon(shape.shape, dx, dy);
    }
  } else if (explicit) {
    return;
  } else if (field === 'radius') {
    shape.shape.radius = Math.max(0, value);
  } else if (field === 'sides') {
    shape.shape.sides = clamp(Math.round(value), POLYGON_MIN_SIDES, POLYGON_MAX_SIDES);
  } else if (field === 'rotation') {
    shape.shape.rotation = normalizeAngleDegrees(value);
  }
  constrainShape(shape);
  markDirty();
  scheduleDraw();
}

function updateParam(material, key, rawValue) {
  const shape = getSelectedShape();
  if (!shape || shape.material !== material) return;
  const value = parseFloat(rawValue);
  if (Number.isNaN(value)) return;
  shape.params = shape.params || {};
  shape.params[key] = value;
  markDirty();
}

function updateSelectedMaterial(material) {
  const shape = getSelectedShape();
  if (!shape || !MATERIAL_DEFAULTS[material]) return;
  shape.material = material;
  shape.params = { ...deepCopy(MATERIAL_DEFAULTS[material].params), ...(shape.params || {}) };
  markDirty();
  updateInspector();
  renderShapeList();
  scheduleDraw();
}

function markDirty() {
  if (!state.dirty) {
    state.dirty = true;
  }
  updateAdaptiveButtonState();
  updateRunButtonState();
}

function scheduleDraw() {
  if (drawScheduled) return;
  drawScheduled = true;
  window.requestAnimationFrame(() => {
    drawScheduled = false;
    drawScene();
  });
}

function drawScene() {
  const ctx = elements.canvas.getContext('2d');
  const { width, height } = elements.canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#fdfdff';
  ctx.fillRect(0, 0, width, height);
  const drawableWidth = width - canvasPadding * 2;
  const drawableHeight = height - canvasPadding * 2;
  const viewRect = getViewWindow();
  ctx.strokeStyle = '#6b7280';
  ctx.strokeRect(canvasPadding, canvasPadding, drawableWidth, drawableHeight);
  drawDomainOutline(ctx, drawableWidth, drawableHeight, viewRect);
  drawDxfDebug(ctx, drawableWidth, drawableHeight, viewRect);
  state.objects.forEach((obj) => {
    drawObject(ctx, obj, drawableWidth, drawableHeight, viewRect);
  });
}

function drawDxfDebug(ctx, drawableWidth, drawableHeight, viewRect) {
  if (!state.debug?.showDxfPoints || !state.debug?.dxfSource) return;
  const src = state.debug.dxfSource;
  ctx.save();
  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(255,0,0,0.35)';
  ctx.fillStyle = 'rgba(255,0,0,0.6)';
  if (src.points && Array.isArray(src.points)) {
    src.points.forEach((pt) => {
      const c = physToCanvas(pt.x, pt.y, drawableWidth, drawableHeight, viewRect);
      ctx.beginPath();
      ctx.arc(c.x, c.y, 1.6, 0, Math.PI * 2);
      ctx.fill();
    });
    if (src.segments && Array.isArray(src.segments)) {
      ctx.strokeStyle = 'rgba(255,0,0,0.35)';
      src.segments.forEach((seg) => {
        const a = src.points[seg.start];
        const b = src.points[seg.end];
        if (!a || !b) return;
        if (seg.type === 'ARC' || seg.type === 'CIRCLE') {
          drawArcSegment(ctx, seg, a, b, drawableWidth, drawableHeight, viewRect);
        } else {
          const ca = physToCanvas(a.x, a.y, drawableWidth, drawableHeight, viewRect);
          const cb = physToCanvas(b.x, b.y, drawableWidth, drawableHeight, viewRect);
          ctx.beginPath();
          ctx.moveTo(ca.x, ca.y);
          ctx.lineTo(cb.x, cb.y);
          ctx.stroke();
        }
      });
    }
  } else if (Array.isArray(src.shapes)) {
    // fallback: draw loops from shapes
    const drawLoop = (loop) => {
      if (!Array.isArray(loop) || loop.length < 2) return;
      ctx.beginPath();
      loop.forEach((pt, idx) => {
        const p = Array.isArray(pt) ? { x: pt[0], y: pt[1] } : pt;
        const c = physToCanvas(p.x, p.y, drawableWidth, drawableHeight, viewRect);
        if (idx === 0) ctx.moveTo(c.x, c.y);
        else ctx.lineTo(c.x, c.y);
      });
      ctx.stroke();
      loop.forEach((pt) => {
        const p = Array.isArray(pt) ? { x: pt[0], y: pt[1] } : pt;
        const c = physToCanvas(p.x, p.y, drawableWidth, drawableHeight, viewRect);
        ctx.beginPath();
        ctx.arc(c.x, c.y, 1.6, 0, Math.PI * 2);
        ctx.fill();
      });
    };
    src.shapes.forEach((shape) => {
      if (hasExplicitVertices(shape)) {
        drawLoop(sanitizeVertices(shape.vertices));
        sanitizeLoops(shape.holes).forEach((hole) => drawLoop(hole));
      }
    });
  }
  ctx.restore();
}

function sampleArcForDebug(seg) {
  if (!seg || !seg.center || !Number.isFinite(seg.radius)) return [];
  const cx = Number(seg.center.x || 0);
  const cy = Number(seg.center.y || 0);
  const angles = computeArcAngles(seg, cx, cy);
  let start = angles.start;
  let span = angles.span;
  const steps = Math.max(12, Math.ceil(span / 6));
  const pts = [];
  for (let i = 0; i <= steps; i += 1) {
    const t = i / steps;
    const deg = start + span * t;
    const rad = (deg * Math.PI) / 180;
    const x = cx + seg.radius * Math.cos(rad);
    const y = cy + seg.radius * Math.sin(rad);
    pts.push({ x, y });
  }
  if (!pts.length && seg.start !== undefined && seg.end !== undefined && Array.isArray(seg.points)) {
    const a = seg.points[seg.start];
    const b = seg.points[seg.end];
    if (a && b) return [a, b];
  }
  return pts;
}

function drawArcSegment(ctx, seg, a, b, drawableWidth, drawableHeight, viewRect) {
  const stype = (seg.type || '').toString().toUpperCase();
  const samples = sampleArcForDebug(seg);
  if (samples.length >= 2) {
    ctx.beginPath();
    samples.forEach((pt, idx) => {
      const cp = physToCanvas(pt.x, pt.y, drawableWidth, drawableHeight, viewRect);
      if (idx === 0) ctx.moveTo(cp.x, cp.y);
      else ctx.lineTo(cp.x, cp.y);
    });
    ctx.stroke();
    // Mark sampled points for debug
    ctx.fillStyle = '#a855f7';
    samples.forEach((pt) => {
      const cp = physToCanvas(pt.x, pt.y, drawableWidth, drawableHeight, viewRect);
      ctx.beginPath();
      ctx.arc(cp.x, cp.y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}

function scaleWirePoints(points, transform) {
  if (!Array.isArray(points) || !transform) return points || [];
  const scale = Number(transform.scale) || 1;
  const dx = Number(transform.dx) || 0;
  const dy = Number(transform.dy) || 0;
  return points.map((pt) => ({
    x: pt.x * scale + dx,
    y: pt.y * scale + dy,
  }));
}

function scaleWireSegments(segments, transform, scaledPoints = null) {
  if (!Array.isArray(segments) || !transform) return segments || [];
  const scale = Number(transform.scale) || 1;
  const dx = Number(transform.dx) || 0;
  const dy = Number(transform.dy) || 0;
  return segments.map((seg) => {
    const copy = { ...seg };
    if (copy.center) {
      copy.center = {
        x: copy.center.x * scale + dx,
        y: copy.center.y * scale + dy,
      };
    }
    if (copy.radius !== undefined) {
      copy.radius = copy.radius * scale;
    }
    if (scaledPoints && copy.type && copy.type.toString().toUpperCase() === 'ARC') {
      copy.points = scaledPoints;
    }
    return copy;
  });
}

function polylineArea(pts) {
  if (!Array.isArray(pts) || pts.length < 3) return 0;
  let area = 0;
  for (let i = 0; i < pts.length; i += 1) {
    const j = (i + 1) % pts.length;
    area += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
  }
  return 0.5 * area;
}

function pointsEqual(a, b, tol = 1e-6) {
  if (!a || !b) return false;
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy <= tol * tol;
}

function sampleSegment(seg, points) {
  const stype = (seg.type || '').toString().toUpperCase();
  if (stype === 'LINE') {
    const a = points[seg.start];
    const b = points[seg.end];
    return a && b ? [a, b] : [];
  }
  if (stype === 'ARC' || stype === 'CIRCLE') {
    return sampleArcForDebug({ ...seg, points });
  }
  return [];
}

function segmentsToLoops(points, segments) {
  const tol = 1e-6;
  const polylines = segments
    .map((seg) => sampleSegment(seg, points))
    .filter((pl) => Array.isArray(pl) && pl.length >= 2);
  const loops = [];
  while (polylines.length) {
    let current = polylines.shift();
    let extended = true;
    while (extended && polylines.length) {
      extended = false;
      for (let i = 0; i < polylines.length; i += 1) {
        const candidate = polylines[i];
        if (!candidate.length) continue;
        if (pointsEqual(current[current.length - 1], candidate[0], tol)) {
          current = current.concat(candidate.slice(1));
          polylines.splice(i, 1);
          extended = true;
          break;
        }
        if (pointsEqual(current[current.length - 1], candidate[candidate.length - 1], tol)) {
          const reversed = candidate.slice().reverse();
          current = current.concat(reversed.slice(1));
          polylines.splice(i, 1);
          extended = true;
          break;
        }
        if (pointsEqual(current[0], candidate[candidate.length - 1], tol)) {
          current = candidate.concat(current.slice(1));
          polylines.splice(i, 1);
          extended = true;
          break;
        }
        if (pointsEqual(current[0], candidate[0], tol)) {
          const reversed = candidate.slice().reverse();
          current = reversed.concat(current.slice(1));
          polylines.splice(i, 1);
          extended = true;
          break;
        }
      }
    }
    if (current.length >= 3 && pointsEqual(current[0], current[current.length - 1], tol) === false) {
      current = current.concat([current[0]]);
    }
    if (current.length >= 3) {
      loops.push(current);
    }
  }
  return loops;
}

function loopsToPolygon(loops) {
  if (!loops.length) return null;
  const sorted = loops
    .map((loop) => ({ loop, area: polylineArea(loop) }))
    .filter((item) => Math.abs(item.area) > 0);
  if (!sorted.length) return null;
  sorted.sort((a, b) => Math.abs(b.area) - Math.abs(a.area));
  const outer = sorted[0].loop;
  const holes = sorted.slice(1).map((item) => item.loop);
  return { outer, holes };
}

function buildDxfSolidFromPreview(source = null) {
  const srcPoints = source?.points || state.debug?.previewPoints;
  const srcSegments = source?.segments || state.debug?.previewSegments;
  if (!srcPoints || !srcSegments) {
    setStatus('Import a DXF first to populate debug data.', 'error');
    return;
  }
  setStatus('Building solid from DXF wireframe...', 'info');
  const points = scaleWirePoints(srcPoints, { scale: 1, dx: 0, dy: 0 }); // already scaled on import
  const segments = scaleWireSegments(srcSegments, { scale: 1, dx: 0, dy: 0 }, points);
  const loops = segmentsToLoops(points, segments);
  const poly = loopsToPolygon(loops);
  if (!poly) {
    setStatus('Could not build a closed polygon from DXF wireframe.', 'error');
    return;
  }
  let polygonShape = {
    type: 'polygon',
    vertices: poly.outer.map((p) => [p.x, p.y]),
    holes: poly.holes.map((hole) => hole.map((p) => [p.x, p.y])),
  };
  const fitToDomain = elements.dxfFitToDomain ? !!elements.dxfFitToDomain.checked : true;
  const marginPct = elements.dxfFitMargin ? Number(elements.dxfFitMargin.value) || 2 : 2;
  if (fitToDomain) {
    const fitResult = fitShapesWithTransform([polygonShape], marginPct);
    polygonShape = fitResult.shapes[0];
  }
  const material = MATERIAL_DEFAULTS[elements.dxfMaterialSelect?.value]
    ? elements.dxfMaterialSelect.value
    : elements.materialSelect?.value || 'steel';
  const metaInfo = MATERIAL_DEFAULTS[material] || MATERIAL_DEFAULTS.steel;
  const obj = {
    id: uniqueId(),
    label: `DXF Solid ${state.objects.length + 1}`,
    material,
    params: deepCopy(metaInfo.params),
    shape: normalizeShape(polygonShape, material),
    group: '',
  };
  state.objects.push(obj);
  state.selectedId = obj.id;
  state.selectedIds = [obj.id];
  markDirty();
  renderShapeList();
  scheduleDraw();
  setStatus('Built solid from DXF wireframe.', 'success');
}

function drawDomainOutline(ctx, drawableWidth, drawableHeight, viewRect) {
  ctx.save();
  ctx.strokeStyle = '#94a3b8';
  ctx.setLineDash([6, 4]);
  const corners = [
    physToCanvas(0, 0, drawableWidth, drawableHeight, viewRect),
    physToCanvas(state.grid.Lx, 0, drawableWidth, drawableHeight, viewRect),
    physToCanvas(state.grid.Lx, state.grid.Ly, drawableWidth, drawableHeight, viewRect),
    physToCanvas(0, state.grid.Ly, drawableWidth, drawableHeight, viewRect),
  ];
  ctx.beginPath();
  corners.forEach((point, idx) => {
    if (idx === 0) {
      ctx.moveTo(point.x, point.y);
    } else {
      ctx.lineTo(point.x, point.y);
    }
  });
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

function drawObject(ctx, obj, drawableWidth, drawableHeight, viewRect) {
  const color = MATERIAL_DEFAULTS[obj.material]?.color || '#777';
  ctx.save();
  ctx.strokeStyle = color;
  const isContour = obj.material === 'contour';
  ctx.fillStyle = isContour ? 'rgba(0,0,0,0)' : `${color}33`;
  if (isContour) {
    ctx.setLineDash([6, 3]);
    ctx.lineWidth = 1.2;
  }
  const sizeScale = Math.min(drawableWidth / viewRect.width, drawableHeight / viewRect.height);
  if (obj.shape.type === 'rect') {
    const corners = getRectCorners(obj.shape).map((corner) =>
      physToCanvas(corner.x, corner.y, drawableWidth, drawableHeight, viewRect)
    );
    if (corners.length) {
      ctx.beginPath();
      corners.forEach((pt, idx) => {
        if (idx === 0) {
          ctx.moveTo(pt.x, pt.y);
        } else {
          ctx.lineTo(pt.x, pt.y);
        }
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
  } else if (obj.shape.type === 'circle') {
    const { x, y } = physToCanvas(
      obj.shape.center[0],
      obj.shape.center[1],
      drawableWidth,
      drawableHeight,
      viewRect
    );
    const r = Math.max(0, obj.shape.radius || 0) * sizeScale;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  } else if (obj.shape.type === 'ring') {
    const { x, y } = physToCanvas(
      obj.shape.center[0],
      obj.shape.center[1],
      drawableWidth,
      drawableHeight,
      viewRect
    );
    const outer = Math.max(0, obj.shape.outer_radius || 0) * sizeScale;
    const inner = Math.max(0, Math.min(obj.shape.inner_radius || 0, obj.shape.outer_radius || 0)) * sizeScale;
    ctx.beginPath();
    ctx.arc(x, y, outer, 0, Math.PI * 2);
    if (inner > 0) {
      ctx.moveTo(x + inner, y);
      ctx.arc(x, y, inner, 0, Math.PI * 2, true);
    }
    ctx.fill('evenodd');
    ctx.stroke();
  } else if (obj.shape.type === 'polygon') {
    const loops = getPolygonLoops(obj.shape).map((loop) =>
      loop.map((pt) => physToCanvas(pt.x, pt.y, drawableWidth, drawableHeight, viewRect))
    );
    if (loops.length && loops[0].length) {
      ctx.beginPath();
      loops.forEach((loop) => {
        loop.forEach((pt, idx) => {
          if (idx === 0) ctx.moveTo(pt.x, pt.y);
          else ctx.lineTo(pt.x, pt.y);
        });
        ctx.closePath();
      });
      ctx.fill('evenodd');
      ctx.stroke();
    }
  }
  if (obj.wireframe && obj.wireframe.points && obj.wireframe.segments) {
    drawWireframe(ctx, obj.wireframe, drawableWidth, drawableHeight, viewRect, color);
  }
  if (state.selectedIds && state.selectedIds.includes(obj.id)) {
    ctx.strokeStyle = '#111';
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.2;
    if (obj.shape.type === 'rect') {
      const corners = getRectCorners(obj.shape).map((corner) =>
        physToCanvas(corner.x, corner.y, drawableWidth, drawableHeight, viewRect)
      );
      if (corners.length) {
        ctx.beginPath();
        corners.forEach((pt, idx) => {
          if (idx === 0) {
            ctx.moveTo(pt.x, pt.y);
          } else {
            ctx.lineTo(pt.x, pt.y);
          }
        });
        ctx.closePath();
        ctx.stroke();
      }
    } else if (obj.shape.type === 'circle') {
      const { x, y } = physToCanvas(
        obj.shape.center[0],
        obj.shape.center[1],
        drawableWidth,
        drawableHeight,
        viewRect
      );
      const r = Math.max(0, obj.shape.radius || 0) * sizeScale;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.stroke();
    } else if (obj.shape.type === 'ring') {
      const { x, y } = physToCanvas(
        obj.shape.center[0],
        obj.shape.center[1],
        drawableWidth,
        drawableHeight,
        viewRect
      );
      const outer = Math.max(0, obj.shape.outer_radius || 0) * sizeScale;
      const inner = Math.max(0, Math.min(obj.shape.inner_radius || 0, obj.shape.outer_radius || 0)) * sizeScale;
      ctx.beginPath();
      ctx.arc(x, y, outer, 0, Math.PI * 2);
      ctx.stroke();
      if (inner > 0) {
        ctx.beginPath();
        ctx.arc(x, y, inner, 0, Math.PI * 2);
        ctx.stroke();
      }
    } else if (obj.shape.type === 'polygon') {
      const loops = getPolygonLoops(obj.shape).map((loop) =>
        loop.map((pt) => physToCanvas(pt.x, pt.y, drawableWidth, drawableHeight, viewRect))
      );
      if (loops.length && loops[0].length) {
        ctx.beginPath();
        loops.forEach((loop) => {
          loop.forEach((pt, idx) => {
            if (idx === 0) ctx.moveTo(pt.x, pt.y);
            else ctx.lineTo(pt.x, pt.y);
          });
          ctx.closePath();
        });
        ctx.stroke();
      }
    }
  }
  ctx.restore();
}

function physToCanvas(px, py, drawableWidth, drawableHeight, viewRect = getViewWindow()) {
  const normX = (px - viewRect.minX) / viewRect.width;
  const normY = (py - viewRect.minY) / viewRect.height;
  const x = canvasPadding + normX * drawableWidth;
  const y = canvasPadding + (1 - normY) * drawableHeight;
  return { x, y };
}

function eventToCanvasPoint(event) {
  const rect = elements.canvas.getBoundingClientRect();
  const scaleX = elements.canvas.width / rect.width;
  const scaleY = elements.canvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function canvasToPhys(canvasX, canvasY, options = {}) {
  const { clampResult = true } = options;
  const drawableWidth = elements.canvas.width - canvasPadding * 2;
  const drawableHeight = elements.canvas.height - canvasPadding * 2;
  const viewRect = getViewWindow();
  let x = viewRect.minX + ((canvasX - canvasPadding) / drawableWidth) * viewRect.width;
  let y = viewRect.minY + (1 - (canvasY - canvasPadding) / drawableHeight) * viewRect.height;
  if (clampResult) {
    x = clamp(x, 0, state.grid.Lx);
    y = clamp(y, 0, state.grid.Ly);
  }
  return {
    x,
    y,
  };
}

function eventToPhys(event, options) {
  const point = eventToCanvasPoint(event);
  return canvasToPhys(point.x, point.y, options);
}

function startPan(event, options = {}) {
  canvasState.mode = 'pan';
  canvasState.panAnchor = eventToPhys(event, { clampResult: false });
  canvasState.panStartCanvas = eventToCanvasPoint(event);
  canvasState.panHasMoved = false;
  canvasState.pendingDeselect = options.deselectOnRelease || false;
}

function updatePan(event) {
  if (!canvasState.panAnchor || !elements.canvas) return;
  const canvasPoint = eventToCanvasPoint(event);
  if (!canvasState.panHasMoved) {
    const dx = canvasPoint.x - canvasState.panStartCanvas.x;
    const dy = canvasPoint.y - canvasState.panStartCanvas.y;
    if (dx * dx + dy * dy < PAN_DRAG_THRESHOLD_SQ) {
      return;
    }
    canvasState.panHasMoved = true;
    canvasState.pendingDeselect = false;
  }
  const drawableWidth = elements.canvas.width - canvasPadding * 2;
  const drawableHeight = elements.canvas.height - canvasPadding * 2;
  if (drawableWidth <= 0 || drawableHeight <= 0) return;
  const viewRect = getViewWindow();
  const normX = (canvasPoint.x - canvasPadding) / drawableWidth;
  const normY = (canvasPoint.y - canvasPadding) / drawableHeight;
  const minX = canvasState.panAnchor.x - normX * viewRect.width;
  const minY = canvasState.panAnchor.y - (1 - normY) * viewRect.height;
  state.view.centerX = minX + viewRect.width / 2;
  state.view.centerY = minY + viewRect.height / 2;
  clampViewCenter();
  scheduleDraw();
}

function onCanvasWheel(event) {
  if (!elements.canvas) return;
  const normalized = normalizeWheelDelta(event);
  if (normalized === 0) return;
  event.preventDefault();
  const focus = eventToPhys(event, { clampResult: false });
  const exponent = -normalized / WHEEL_ZOOM_SENSITIVITY;
  const multiplier = Math.pow(VIEW_STEP, exponent);
  zoomBy(multiplier, focus);
}

function onPointerDown(event) {
  if (canvasState.pointerId) return;
  const isPrimary = event.button === 0 || event.pointerType === 'touch';
  const isMiddleButton = event.button === 1;
  if (!isPrimary && !isMiddleButton) {
    return;
  }
  elements.canvas.setPointerCapture(event.pointerId);
  canvasState.pointerId = event.pointerId;
  canvasState.mode = null;
  canvasState.shapeId = null;
  canvasState.start = null;
  canvasState.offset = null;
  canvasState.groupOffsets = null;
  canvasState.panAnchor = null;
  canvasState.panStartCanvas = null;
  canvasState.panHasMoved = false;
  canvasState.pendingDeselect = false;
  if (isMiddleButton) {
    startPan(event);
    return;
  }
  const phys = eventToPhys(event);
  if (state.tool === 'rect') {
    startDrawingShape('rect', phys);
    return;
  }
  if (state.tool === 'circle') {
    startDrawingShape('circle', phys);
    return;
  }
  if (state.tool === 'ring') {
    startDrawingShape('ring', phys);
    return;
  }
  if (state.tool === 'polygon') {
    startDrawingShape('polygon', phys);
    return;
  }
  const hit = hitTest(phys);
  if (hit) {
    selectShape(hit.id, { additive: event.shiftKey, toggle: event.shiftKey });
    canvasState.mode = 'drag';
    canvasState.shapeId = hit.id;
    canvasState.offset = {
      x: phys.x - hit.shape.center[0],
      y: phys.y - hit.shape.center[1],
    };
    const moveSetIds = new Set();
    if (hit.group) {
      state.objects.forEach((obj) => {
        if (obj.group && obj.group === hit.group) moveSetIds.add(obj.id);
      });
    }
    if (state.selectedIds?.length) {
      state.selectedIds.forEach((idVal) => moveSetIds.add(idVal));
    }
    const moveTargets = state.objects.filter((obj) => moveSetIds.has(obj.id));
    canvasState.groupOffsets = moveTargets.map((obj) => ({
      id: obj.id,
      offset: {
        x: phys.x - obj.shape.center[0],
        y: phys.y - obj.shape.center[1],
      },
    }));
  } else if (state.tool === 'select') {
    startPan(event, { deselectOnRelease: true });
  } else {
    state.selectedId = null;
    state.selectedIds = [];
    renderShapeList();
    updateInspector();
  }
}

function startDrawingShape(type, phys) {
  const material = MATERIAL_DEFAULTS[elements.materialSelect.value] ? elements.materialSelect.value : 'magnet';
  const meta = MATERIAL_DEFAULTS[material];
  const shape = {
    id: uniqueId(),
    label: `${meta.label} ${state.objects.length + 1}`,
    material,
    params: deepCopy(meta.params),
    shape: null,
    group: (elements.shapeGroup?.value || '').trim(),
  };
  if (type === 'rect') {
    shape.shape = { type: 'rect', center: [phys.x, phys.y], width: 0, height: 0, angle: 0 };
    canvasState.mode = 'draw-rect';
  } else if (type === 'circle') {
    shape.shape = { type: 'circle', center: [phys.x, phys.y], radius: 0 };
    canvasState.mode = 'draw-circle';
  } else if (type === 'ring') {
    shape.shape = {
      type: 'ring',
      center: [phys.x, phys.y],
      outer_radius: 0,
      inner_radius: 0,
    };
    canvasState.mode = 'draw-ring';
  } else if (type === 'polygon') {
    shape.shape = {
      type: 'polygon',
      center: [phys.x, phys.y],
      radius: 0,
      sides: 6,
      rotation: 0,
    };
    canvasState.mode = 'draw-polygon';
  }
  canvasState.shapeId = shape.id;
  canvasState.start = phys;
  state.objects.push(shape);
  state.selectedId = shape.id;
  state.selectedIds = [shape.id];
  renderShapeList();
  updateInspector();
  markDirty();
  scheduleDraw();
}

function onPointerMove(event) {
  if (!canvasState.pointerId) return;
  if (canvasState.mode === 'pan') {
    updatePan(event);
    return;
  }
  const phys = eventToPhys(event);
  if (canvasState.mode === 'draw-rect') {
    const shape = state.objects.find((obj) => obj.id === canvasState.shapeId);
    if (!shape) return;
    const dx = phys.x - canvasState.start.x;
    const dy = phys.y - canvasState.start.y;
    let width = Math.abs(dx);
    let height = Math.abs(dy);
    let cx;
    let cy;
    if (event.shiftKey) {
      const size = Math.max(width, height);
      width = height = size;
      cx = canvasState.start.x + Math.sign(dx || 1) * width / 2;
      cy = canvasState.start.y + Math.sign(dy || 1) * height / 2;
    } else {
      cx = canvasState.start.x + dx / 2;
      cy = canvasState.start.y + dy / 2;
    }
    shape.shape.width = width;
    shape.shape.height = height;
    shape.shape.center[0] = cx;
    shape.shape.center[1] = cy;
    constrainShape(shape);
    scheduleDraw();
    updateInspector();
  } else if (canvasState.mode === 'draw-circle') {
    const shape = state.objects.find((obj) => obj.id === canvasState.shapeId);
    if (!shape) return;
    const dx = phys.x - canvasState.start.x;
    const dy = phys.y - canvasState.start.y;
    shape.shape.radius = Math.sqrt(dx * dx + dy * dy);
    constrainShape(shape);
    scheduleDraw();
    updateInspector();
  } else if (canvasState.mode === 'draw-ring') {
    const shape = state.objects.find((obj) => obj.id === canvasState.shapeId);
    if (!shape) return;
    const dx = phys.x - canvasState.start.x;
    const dy = phys.y - canvasState.start.y;
    const r = Math.sqrt(dx * dx + dy * dy);
    shape.shape.outer_radius = r;
    if (!(shape.shape.inner_radius > 0)) {
      shape.shape.inner_radius = 0.7 * r;
    }
    constrainShape(shape);
    scheduleDraw();
    updateInspector();
  } else if (canvasState.mode === 'draw-polygon') {
    const shape = state.objects.find((obj) => obj.id === canvasState.shapeId);
    if (!shape) return;
    const dx = phys.x - canvasState.start.x;
    const dy = phys.y - canvasState.start.y;
    shape.shape.radius = Math.sqrt(dx * dx + dy * dy);
    shape.shape.rotation = normalizeAngleDegrees((Math.atan2(dy, dx) / Math.PI) * 180);
    constrainShape(shape);
    scheduleDraw();
    updateInspector();
  } else if (canvasState.mode === 'drag') {
    const shape = state.objects.find((obj) => obj.id === canvasState.shapeId);
    if (!shape) return;
    if (Array.isArray(canvasState.groupOffsets) && canvasState.groupOffsets.length > 0) {
      canvasState.groupOffsets.forEach((entry) => {
        const target = state.objects.find((obj) => obj.id === entry.id);
        if (!target || !target.shape?.center) return;
        const prev = shapeCenter(target.shape);
        const nextX = clamp(phys.x - entry.offset.x, 0, state.grid.Lx);
        const nextY = clamp(phys.y - entry.offset.y, 0, state.grid.Ly);
        target.shape.center[0] = nextX;
        target.shape.center[1] = nextY;
        translateExplicitPolygon(target.shape, nextX - prev.x, nextY - prev.y);
      });
    } else {
      const prev = shapeCenter(shape.shape);
      const nextX = clamp(phys.x - canvasState.offset.x, 0, state.grid.Lx);
      const nextY = clamp(phys.y - canvasState.offset.y, 0, state.grid.Ly);
      shape.shape.center[0] = nextX;
      shape.shape.center[1] = nextY;
      translateExplicitPolygon(shape.shape, nextX - prev.x, nextY - prev.y);
    }
    markDirty();
    scheduleDraw();
    updateInspector();
  }
}

function constrainShape(shape) {
  if (shape.shape.type === 'rect') {
    shape.shape.center[0] = clamp(shape.shape.center[0], 0, state.grid.Lx);
    shape.shape.center[1] = clamp(shape.shape.center[1], 0, state.grid.Ly);
    shape.shape.width = Math.max(0.0005, shape.shape.width);
    shape.shape.height = Math.max(0.0005, shape.shape.height);
    shape.shape.angle = normalizeAngleDegrees(shape.shape.angle || 0);
  } else if (shape.shape.type === 'circle') {
    shape.shape.center[0] = clamp(shape.shape.center[0], 0, state.grid.Lx);
    shape.shape.center[1] = clamp(shape.shape.center[1], 0, state.grid.Ly);
    shape.shape.radius = Math.max(0.0005, shape.shape.radius);
  } else if (shape.shape.type === 'ring') {
    shape.shape.center[0] = clamp(shape.shape.center[0], 0, state.grid.Lx);
    shape.shape.center[1] = clamp(shape.shape.center[1], 0, state.grid.Ly);
    shape.shape.outer_radius = Math.max(0.0005, shape.shape.outer_radius);
    const maxInner = Math.max(0, shape.shape.outer_radius - 1e-5 * Math.max(state.grid.Lx, state.grid.Ly));
    shape.shape.inner_radius = clamp(shape.shape.inner_radius, 0, maxInner);
  } else if (shape.shape.type === 'polygon') {
    if (hasExplicitVertices(shape.shape)) {
      const center = shapeCenter(shape.shape);
      const clampedX = clamp(center.x, 0, state.grid.Lx);
      const clampedY = clamp(center.y, 0, state.grid.Ly);
      translateExplicitPolygon(shape.shape, clampedX - center.x, clampedY - center.y);
      shape.shape.center = [clampedX, clampedY];
      shape.shape.sides = sanitizeVertices(shape.shape.vertices).length;
      shape.shape.radius = polygonRadius(shape.shape);
    } else {
      shape.shape.center[0] = clamp(shape.shape.center[0], 0, state.grid.Lx);
      shape.shape.center[1] = clamp(shape.shape.center[1], 0, state.grid.Ly);
      shape.shape.radius = Math.max(0.0005, shape.shape.radius);
      shape.shape.sides = clamp(Math.round(shape.shape.sides || POLYGON_MIN_SIDES), POLYGON_MIN_SIDES, POLYGON_MAX_SIDES);
      shape.shape.rotation = normalizeAngleDegrees(shape.shape.rotation ?? shape.shape.angle ?? 0);
    }
  }
}

function onPointerUp() {
  if (!canvasState.pointerId) return;
  elements.canvas.releasePointerCapture(canvasState.pointerId);
  const mode = canvasState.mode;
  const shape = state.objects.find((obj) => obj.id === canvasState.shapeId);
  if (shape && (mode === 'draw-rect' || mode === 'draw-circle' || mode === 'draw-ring' || mode === 'draw-polygon')) {
    let minimal = false;
    if (mode === 'draw-rect') {
      minimal = shape.shape.width * shape.shape.height < 1e-6;
    } else if (mode === 'draw-circle') {
      minimal = shape.shape.radius < 1e-3;
    } else if (mode === 'draw-ring') {
      minimal = shape.shape.outer_radius < 1e-3 || shape.shape.outer_radius - shape.shape.inner_radius < 1e-4;
    } else if (mode === 'draw-polygon') {
      minimal = shape.shape.radius < 1e-3;
    }
    if (minimal) {
      state.objects = state.objects.filter((obj) => obj.id !== shape.id);
      state.selectedId = null;
      state.selectedIds = [];
      renderShapeList();
      updateInspector();
    }
  }
  if (mode === 'pan' && canvasState.pendingDeselect) {
    state.selectedId = null;
    state.selectedIds = [];
    renderShapeList();
    updateInspector();
  }
  canvasState.mode = null;
  canvasState.pointerId = null;
  canvasState.shapeId = null;
  canvasState.start = null;
  canvasState.offset = null;
  canvasState.panAnchor = null;
  canvasState.panStartCanvas = null;
  canvasState.panHasMoved = false;
  canvasState.pendingDeselect = false;
}

function hitTest(point) {
  let contourHit = null;
  let nonContourHit = null;
  for (let i = state.objects.length - 1; i >= 0; i -= 1) {
    const obj = state.objects[i];
    const isContour = obj.material === 'contour';
    if (obj.shape.type === 'rect') {
      if (pointInRotatedRect(point, obj.shape)) {
        if (isContour && contourHit === null) contourHit = obj;
        else if (!isContour && nonContourHit === null) nonContourHit = obj;
        continue;
      }
    } else if (obj.shape.type === 'circle') {
      const [cx, cy] = obj.shape.center;
      const dx = point.x - cx;
      const dy = point.y - cy;
      if (Math.sqrt(dx * dx + dy * dy) <= (obj.shape.radius || 0)) {
        if (isContour && contourHit === null) contourHit = obj;
        else if (!isContour && nonContourHit === null) nonContourHit = obj;
        continue;
      }
    } else if (obj.shape.type === 'ring') {
      const [cx, cy] = obj.shape.center;
      const dx = point.x - cx;
      const dy = point.y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const outer = obj.shape.outer_radius || 0;
      const inner = Math.max(0, obj.shape.inner_radius || 0);
      if (dist <= outer && dist >= inner) {
        if (isContour && contourHit === null) contourHit = obj;
        else if (!isContour && nonContourHit === null) nonContourHit = obj;
        continue;
      }
    } else if (obj.shape.type === 'polygon') {
      const loops = getPolygonLoops(obj.shape);
      if (loops.length && pointInPolygonLoops(point, loops)) {
        if (isContour && contourHit === null) contourHit = obj;
        else if (!isContour && nonContourHit === null) nonContourHit = obj;
        continue;
      }
    }
  }
  return nonContourHit || contourHit;
}

function setStatus(message, variant = 'info') {
  elements.statusLine.textContent = message;
  elements.statusLine.dataset.variant = variant;
}

function ensureCaseTracked(caseName) {
  if (!caseName) {
    return false;
  }
  const alreadyPresent = state.cases.includes(caseName);
  if (!alreadyPresent) {
    state.cases.push(caseName);
    state.cases.sort();
    renderCaseSelect();
    emitBuilderEvent('casesChanged', { cases: [...state.cases] });
  }
  return !alreadyPresent;
}

async function refreshCaseList(options = {}) {
  const { silent = false } = options;
  if (!elements.caseSelect || !elements.caseNameInput) return;
  const previousSelection = state.caseName || elements.caseSelect.value || '';
  const before = state.cases.join(',');
  try {
    const response = await fetch('/api/cases');
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }
    const payload = await response.json();
    const cases = Array.isArray(payload.cases) ? payload.cases.filter(Boolean) : [];
    state.cases = [...new Set(cases)].sort();
    renderCaseSelect();
    if (previousSelection && state.cases.includes(previousSelection)) {
      elements.caseSelect.value = previousSelection;
    } else if (!state.caseName && state.cases.length) {
      elements.caseSelect.value = state.cases[0];
      if (!elements.caseNameInput.value) {
        elements.caseNameInput.value = state.cases[0];
      }
      state.caseName = elements.caseNameInput.value;
    }
    if (state.cases.join(',') !== before) {
      emitBuilderEvent('casesChanged', { cases: [...state.cases] });
    }
  } catch (err) {
    console.error('Failed to refresh case list', err);
    if (!silent) {
      setStatus(`Could not load cases: ${err.message || err}`, 'error');
    }
  }
}

function serializeDefinition() {
  const defaults = Object.fromEntries(
    Object.entries(MATERIAL_DEFAULTS).map(([key, meta]) => [key, deepCopy(meta.params)])
  );
  return {
    name: elements.definitionNameInput.value.trim() || state.caseName || 'new_case',
    grid: state.grid,
    defaults,
    objects: state.objects.map((obj) => ({
      id: obj.id,
      label: obj.label,
      material: obj.material,
      params: obj.params,
      shape: obj.shape,
      group: obj.group || '',
    })),
  };
}

function combinedShapeBounds(shapes) {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  shapes.forEach((shape) => {
    const b = shapeBounds(shape);
    if (!b) return;
    minX = Math.min(minX, b.minX);
    minY = Math.min(minY, b.minY);
    maxX = Math.max(maxX, b.maxX);
    maxY = Math.max(maxY, b.maxY);
  });
  if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
    return null;
  }
  return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY };
}

function scaleAndTranslateShape(shape, scale, dx, dy) {
  const copy = deepCopy(shape);
  const type = typeof copy.type === 'string' ? copy.type.toLowerCase() : '';
  const normalizePt = (pt) => {
    if (Array.isArray(pt)) return [Number(pt[0]) || 0, Number(pt[1]) || 0];
    if (pt && typeof pt === 'object') return [Number(pt.x) || 0, Number(pt.y) || 0];
    return [0, 0];
  };
  const applyPoint = (pt) => {
    const [px, py] = normalizePt(pt);
    return [px * scale + dx, py * scale + dy];
  };
  if (type === 'polygon') {
    if (hasExplicitVertices(copy)) {
      copy.vertices = sanitizeVertices(copy.vertices).map((v) => applyPoint(v));
    }
    if (Array.isArray(copy.holes)) {
      copy.holes = sanitizeLoops(copy.holes).map((loop) => loop.map((pt) => applyPoint(pt)));
    }
    if (Array.isArray(copy.center)) {
      copy.center = applyPoint(copy.center);
    }
    if (copy.radius !== undefined) {
      copy.radius = Math.abs(copy.radius * scale);
    }
  } else if (type === 'rect') {
    copy.center = applyPoint(copy.center || [0, 0]);
    copy.width = (copy.width || 0) * scale;
    copy.height = (copy.height || 0) * scale;
  } else if (type === 'circle') {
    copy.center = applyPoint(copy.center || [0, 0]);
    copy.radius = (copy.radius || 0) * scale;
  } else if (type === 'ring') {
    copy.center = applyPoint(copy.center || [0, 0]);
    copy.outer_radius = (copy.outer_radius || copy.outerRadius || 0) * scale;
    copy.inner_radius = (copy.inner_radius || copy.innerRadius || 0) * scale;
  }
  return copy;
}

function fitShapesToDomain(shapes, marginPct = 2) {
  return fitShapesWithTransform(shapes, marginPct).shapes;
}

function fitShapesWithTransform(shapes, marginPct = 2) {
  const sanitized = shapes.map((shape) => deepCopy(shape));
  const bounds = combinedShapeBounds(sanitized);
  if (!bounds || bounds.width <= 0 || bounds.height <= 0) {
    return { shapes: sanitized, transform: { scale: 1, dx: 0, dy: 0 } };
  }
  const padFrac = clamp((Number(marginPct) || 0) / 100, 0, 0.49);
  const padX = state.grid.Lx * padFrac;
  const padY = state.grid.Ly * padFrac;
  const targetW = Math.max(0, state.grid.Lx - 2 * padX);
  const targetH = Math.max(0, state.grid.Ly - 2 * padY);
  if (targetW <= 0 || targetH <= 0) {
    return { shapes: sanitized, transform: { scale: 1, dx: 0, dy: 0 } };
  }
  let scale = Math.min(targetW / bounds.width, targetH / bounds.height);
  if (!Number.isFinite(scale) || scale <= 0) {
    return { shapes: sanitized, transform: { scale: 1, dx: 0, dy: 0 } };
  }
  // Avoid scaling up; only shrink to fit.
  if (scale > 1) {
    scale = 1;
  }
  const offsetX = padX - bounds.minX * scale + (targetW - bounds.width * scale) / 2;
  const offsetY = padY - bounds.minY * scale + (targetH - bounds.height * scale) / 2;
  return {
    shapes: sanitized.map((shape) => scaleAndTranslateShape(shape, scale, offsetX, offsetY)),
    transform: { scale, dx: offsetX, dy: offsetY },
  };
}

async function importAndBuildDxf() {
  const file = elements.dxfFileInput?.files?.[0];
  if (!file) {
    setStatus('Choose a DXF file to import.', 'error');
    return;
  }
  const unitScale = DXF_UNIT_SCALE;
  // Parse locally for debugging points/segments identical to dxf_viewer_index.html
  try {
    const text = await file.text();
    const debugGeom = parseDxfText(text);
    state.debug.previewPoints = scaleWirePoints(debugGeom.points, { scale: unitScale, dx: 0, dy: 0 });
    state.debug.previewSegments = scaleWireSegments(
      debugGeom.segments,
      { scale: unitScale, dx: 0, dy: 0 },
      state.debug.previewPoints
    );
    renderDxfDebugPanel();
    // Directly build the solid from the parsed wireframe
    buildDxfSolidFromPreview({
      points: state.debug.previewPoints,
      segments: state.debug.previewSegments,
    });
    scheduleDraw();
  } catch (err) {
    console.warn('DXF debug parse failed', err);
    setStatus(`DXF import failed: ${err.message || err}`, 'error');
  }
}
function applyImportedShapes(entries, bounds, warnings = [], meta = {}) {
  const preferredMaterial = MATERIAL_DEFAULTS[elements.dxfMaterialSelect?.value]
    ? elements.dxfMaterialSelect.value
    : elements.materialSelect?.value || 'steel';
  const material = MATERIAL_DEFAULTS[preferredMaterial] ? preferredMaterial : 'steel';
  const marginPct = 2;
  const unitScale = meta?.unitScale && Number.isFinite(meta.unitScale) && meta.unitScale > 0 ? meta.unitScale : DXF_UNIT_SCALE;
  const rawShapes = entries.map((entry) => scaleAndTranslateShape(deepCopy(entry.shape || entry), unitScale, 0, 0));
  const fitResult = fitShapesWithTransform(rawShapes, marginPct);
  const transformed = fitResult.shapes;
  const wireTransform = fitResult.transform || { scale: 1, dx: 0, dy: 0 };
  const startCount = state.objects.length;
  const importGroup = `import-${Date.now().toString(36)}`;
  if (groupAsSingle) {
    const normalizedShapes = transformed.map((shapeDef) => normalizeShape(shapeDef, material));
    const areas = normalizedShapes.map((shape) => shapeSignedArea(shape));
    let outerIdx = 0;
    let maxArea = -Infinity;
    areas.forEach((val, idx) => {
      const a = Math.abs(val);
      if (a > maxArea) {
        maxArea = a;
        outerIdx = idx;
      }
    });
    const outerShape = normalizedShapes[outerIdx];
    const holeLoops = [];
    if (hasExplicitVertices(outerShape) && Array.isArray(outerShape.holes)) {
      holeLoops.push(...sanitizeLoops(outerShape.holes));
    }
    normalizedShapes.forEach((shape, idx) => {
      if (idx === outerIdx) return;
      const outerLoop = hasExplicitVertices(shape)
        ? sanitizeVertices(shape.vertices)
        : getPolygonVertices(shape).map((v) => [v.x, v.y]);
      if (outerLoop.length >= 3) {
        holeLoops.push(outerLoop);
      }
      if (hasExplicitVertices(shape) && Array.isArray(shape.holes)) {
        holeLoops.push(...sanitizeLoops(shape.holes));
      }
    });
    const combined = {
      type: 'polygon',
      vertices: hasExplicitVertices(outerShape)
        ? sanitizeVertices(outerShape.vertices)
        : getPolygonVertices(outerShape).map((v) => [v.x, v.y]),
      holes: holeLoops,
    };
    const normalizedCombined = normalizeShape(combined, material);
    const outerMeta = MATERIAL_DEFAULTS[material] || MATERIAL_DEFAULTS.steel;
    state.objects.push({
      id: uniqueId(),
      label: `${outerMeta.label} ${state.objects.length + 1}`,
      material,
      params: deepCopy(outerMeta.params),
      shape: normalizedCombined,
      group: importGroup,
      wireframe: meta?.segments && meta?.points ? { segments: meta.segments, points: scaleWirePoints(meta.points, wireTransform) } : null,
    });
  } else {
    transformed.forEach((shapeDef, idx) => {
      const meta = entries[idx] || {};
      const layerGroup = groupByLayer && meta.layer ? String(meta.layer).trim() : '';
      const group = layerGroup || (elements.shapeGroup?.value || '').trim();
      const normalized = normalizeShape(shapeDef, material);
      const metaInfo = MATERIAL_DEFAULTS[material] || MATERIAL_DEFAULTS.steel;
      const obj = {
        id: uniqueId(),
        label: `${metaInfo.label} ${state.objects.length + 1}`,
        material,
        params: deepCopy(metaInfo.params),
        shape: normalized,
        group,
        wireframe: meta?.segments && meta?.points ? { segments: meta.segments, points: scaleWirePoints(meta.points, wireTransform) } : null,
      };
      state.objects.push(obj);
    });
  }
  // Store debug points/segments for rendering aid when requested.
  if (state.debug && state.debug.showDxfPoints) {
    const combinedTransform = {
      scale: unitScale * (wireTransform.scale || 1),
      dx: wireTransform.dx || 0,
      dy: wireTransform.dy || 0,
    };
    const scaledPoints = meta?.points ? scaleWirePoints(meta.points, combinedTransform) : state.debug.previewPoints;
    const segmentsRaw = meta?.segments || state.debug.previewSegments;
    const scaledSegments = scaleWireSegments(segmentsRaw, combinedTransform);
    state.debug.dxfSource = scaledPoints
      ? { points: scaledPoints, segments: scaledSegments }
      : { shapes: transformed };
  } else if (state.debug) {
    state.debug.dxfSource = null;
  }
  if (state.objects.length > startCount) {
    state.selectedId = state.objects[state.objects.length - 1].id;
    state.selectedIds = [state.selectedId];
    markDirty();
    renderAll();
    const summaryBounds = bounds || combinedShapeBounds(transformed) || {};
    const warningText = warnings.length ? ` Warnings: ${warnings.join(' ')}` : '';
    const labelCount = groupAsSingle ? 1 : state.objects.length - startCount;
    const spanText =
      summaryBounds.width && summaryBounds.height
        ? ` Source span ~${summaryBounds.width.toFixed(4)} × ${summaryBounds.height.toFixed(4)}.`
        : '';
    const statusText = `Imported ${labelCount} grouped shape${labelCount === 1 ? '' : 's'} from DXF.` + spanText + warningText;
    setStatus(statusText, 'success');
  } else {
    setStatus('DXF import did not add any shapes.', 'error');
  }
}

async function saveCaseDefinition() {
  const caseName = elements.caseNameInput.value.trim();
  if (!caseName) {
    setStatus('Enter a case folder name before saving.', 'error');
    return;
  }
  const definition = serializeDefinition();
  try {
    const response = await fetch(`/api/case/${encodeURIComponent(caseName)}/definition`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(definition),
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const data = await response.json();
    state.caseName = data.case;
    state.definitionName = definition.name;
    state.dirty = false;
    const added = ensureCaseTracked(state.caseName);
    if (!added) {
      renderCaseSelect();
    }
    setStatus(`Saved to ${data.path}`, 'success');
    emitBuilderEvent('definitionSaved', { caseName: state.caseName, path: data.path });
  } catch (err) {
    console.error(err);
    setStatus(`Save failed: ${err}`, 'error');
  }
}

function setRunButtonBusy(isBusy) {
  state.runBusy = isBusy;
  if (elements.runCaseBtn) {
    elements.runCaseBtn.dataset.busy = isBusy ? 'true' : 'false';
  }
  updateRunButtonState();
  updateAdaptiveButtonState();
}

function setAdaptiveButtonBusy(isBusy) {
  state.adaptiveBusy = isBusy;
  updateAdaptiveButtonState();
  updateRunButtonState();
}

function stableStringify(value) {
  return JSON.stringify(
    value,
    (_, val) => {
      if (val && typeof val === 'object' && !Array.isArray(val)) {
        return Object.keys(val)
          .sort()
          .reduce((acc, key) => {
            acc[key] = val[key];
            return acc;
          }, {});
      }
      return val;
    },
    0
  );
}

function fingerprintDefinition(definition) {
  return stableStringify(definition);
}

function evaluateMeshState() {
  if (state.runBusy || state.adaptiveBusy) {
    return { ok: false, reason: 'A job is currently running.' };
  }
  if (!state.caseName) {
    return { ok: false, reason: 'Enter a case name first.' };
  }
  const modeRaw = elements.gridMeshMode?.value || state.grid.mesh?.type || 'point_cloud';
  const usesBFocus = modeRaw !== 'uniform' && modeRaw !== 'experimental';
  const bFocusChecked = !!elements.fieldFocusEnabled?.checked;
  if (usesBFocus && bFocusChecked && !state.hasBField) {
    return { ok: false, reason: 'Solve once before B-field driven meshing.' };
  }
  return { ok: true, reason: '' };
}

function updateAdaptiveButtonState() {
  if (!elements.meshBtn) return;
  const busy = state.adaptiveBusy || state.runBusy;
  const readiness = evaluateMeshState();
  const enabled = readiness.ok && !busy;
  elements.meshBtn.disabled = !enabled;
  elements.meshBtn.dataset.busy = state.adaptiveBusy ? 'true' : 'false';
  elements.meshBtn.title = enabled ? '' : readiness.reason || 'Mesh unavailable.';
}

function evaluateRunState() {
  if (state.runBusy || state.adaptiveBusy) {
    return { ok: false, reason: 'A job is currently running.' };
  }
  if (state.dirty) {
    return { ok: true, reason: '' };
  }
  const fingerprint = fingerprintDefinition(serializeDefinition());
  const lastSolve = state.lastSolveSnapshot;
  const meshVersion = state.meshVersion ?? 0;
  const solvedVersion = state.lastSolvedMeshVersion ?? 0;
  if (!lastSolve) {
    return { ok: true, reason: '' };
  }
  if (meshVersion > solvedVersion) {
    return { ok: true, reason: '' };
  }
  if (lastSolve.fingerprint && lastSolve.fingerprint !== fingerprint) {
    return { ok: true, reason: '' };
  }
  return { ok: false, reason: 'No changes since last solve.' };
}

function updateRunButtonState() {
  if (!elements.runCaseBtn) return;
  const readiness = evaluateRunState();
  elements.runCaseBtn.disabled = !readiness.ok;
  elements.runCaseBtn.title = readiness.ok ? '' : readiness.reason || 'Already up to date.';
}

function updateBFieldToggleAvailability() {
  if (!elements.fieldFocusEnabled) return;
  const has = !!state.hasBField;
  elements.fieldFocusEnabled.disabled = !has;
  if (!has) {
    elements.fieldFocusEnabled.checked = false;
  }
  elements.fieldFocusEnabled.title = has ? '' : 'Solve once before using B-field driven refinement.';
}

function summarizeSteps(steps) {
  if (!Array.isArray(steps) || !steps.length) return '';
  return steps
    .map((step) => {
      const label = step.step || 'step';
      return `${label}: ${step.success ? 'ok' : 'failed'}`;
    })
    .join(' | ');
}

async function runCasePipeline() {
  const caseName = elements.caseNameInput.value.trim();
  if (!caseName) {
    setStatus('Enter a case folder name before running.', 'error');
    return;
  }
  const definition = serializeDefinition();
  const fingerprint = fingerprintDefinition(definition);
  setRunButtonBusy(true);
  setStatus('Saving case and running pipeline...', 'info');
  startRunTimer();
  emitBuilderEvent('runStarted', { caseName, definition });
  try {
    const response = await fetch(`/api/case/${encodeURIComponent(caseName)}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(definition),
    });
    const raw = await response.text();
    let data = null;
    if (raw) {
      try {
        data = JSON.parse(raw);
      } catch (parseErr) {
        console.warn('Failed to parse run response JSON', parseErr);
      }
    }
    if (!response.ok) {
      const message =
        (data && (data.error || data.message)) ||
        raw ||
        `Run failed with status ${response.status}`;
      throw new Error(message);
    }
    state.caseName = data.case || caseName;
    state.definitionName = definition.name;
    state.dirty = false;
    state.lastRunSnapshot = {
      caseName: state.caseName,
      fingerprint,
    };
    state.meshVersion = (state.meshVersion || 0) + 1;
    state.lastSolvedMeshVersion = state.meshVersion;
    state.lastSolveSnapshot = {
      caseName: state.caseName,
      fingerprint,
    };
    updateAdaptiveButtonState();
    const added = ensureCaseTracked(state.caseName);
    if (!added) {
      renderCaseSelect();
    }
    const summary = summarizeSteps(data.steps);
    const elapsedMs = stopRunTimer();
    const elapsedLabel = formatDuration(elapsedMs);
    const suffixParts = [];
    if (summary) suffixParts.push(summary);
    if (elapsedLabel) suffixParts.push(`elapsed ${elapsedLabel}`);
    const suffix = suffixParts.length ? ` (${suffixParts.join(' · ')})` : '';
    setStatus(`Run complete${suffix}.`, 'success');
    emitBuilderEvent('runCompleted', {
      caseName: state.caseName,
      succeeded: true,
      steps: data.steps || [],
      durationMs: elapsedMs ?? undefined,
    });
    state.hasBField = true;
    state.lastSolveSnapshot = {
      caseName: state.caseName,
      fingerprint,
    };
    state.lastSolvedMeshVersion = state.meshVersion;
    updateBFieldToggleAvailability();
  } catch (err) {
    console.error(err);
    const elapsedMs = stopRunTimer();
    const elapsedLabel = formatDuration(elapsedMs);
    const timing = elapsedLabel ? ` after ${elapsedLabel}` : '';
    setStatus(`Run failed${timing}: ${err.message || err}`, 'error');
    emitBuilderEvent('runCompleted', {
      caseName: state.caseName || caseName,
      succeeded: false,
      error: err.message || String(err),
      durationMs: elapsedMs ?? undefined,
    });
  } finally {
    stopRunTimer();
    setRunButtonBusy(false);
    updateRunButtonState();
  }
}

async function runMeshPipeline() {
  const caseName = elements.caseNameInput.value.trim() || state.caseName;
  if (!caseName) {
    setStatus('Enter a case folder name before meshing.', 'error');
    return;
  }
  const meshReady = evaluateMeshState();
  if (!meshReady.ok) {
    setStatus(meshReady.reason || 'Mesh unavailable.', 'error');
    return;
  }
  const definition = serializeDefinition();
  const fingerprint = fingerprintDefinition(definition);
  setAdaptiveButtonBusy(true);
  setStatus('Building mesh...', 'info');
  startRunTimer();
  emitBuilderEvent('adaptiveRunStarted', { caseName, definition });
  try {
    const saveResp = await fetch(`/api/case/${encodeURIComponent(caseName)}/definition`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(definition),
    });
    if (!saveResp.ok) {
      throw new Error(await saveResp.text());
    }
    const useBField = !!elements.fieldFocusEnabled?.checked && state.hasBField;
    const endpoint = useBField ? 'mesh-adaptive' : 'mesh';
    const response = await fetch(`/api/case/${encodeURIComponent(caseName)}/${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(definition),
    });
    const raw = await response.text();
    let data = null;
    if (raw) {
      try {
        data = JSON.parse(raw);
      } catch (parseErr) {
        console.warn('Failed to parse mesh response JSON', parseErr);
      }
    }
    if (!response.ok) {
      const message =
        (data && (data.error || data.message)) ||
        raw ||
        `Mesh failed with status ${response.status}`;
      throw new Error(message);
    }
    state.caseName = data.case || caseName;
    state.definitionName = definition.name;
    state.dirty = false;
    state.meshVersion = (state.meshVersion || 0) + 1;
    state.lastRunSnapshot = {
      caseName: state.caseName,
      fingerprint,
    };
    updateAdaptiveButtonState();
    updateRunButtonState();
    const summary = summarizeSteps(data.steps);
    const suffix = summary ? ` (${summary})` : '';
    const elapsedMs = stopRunTimer();
    const elapsedLabel = formatDuration(elapsedMs);
    const timing = elapsedLabel ? ` · elapsed ${elapsedLabel}` : '';
    setStatus(`Mesh complete${suffix}${timing}.`, 'success');
    emitBuilderEvent('adaptiveRunCompleted', {
      caseName: state.caseName,
      succeeded: true,
      steps: data.steps || [],
      durationMs: elapsedMs ?? undefined,
    });
  } catch (err) {
    console.error(err);
    const elapsedMs = stopRunTimer();
    const elapsedLabel = formatDuration(elapsedMs);
    const timing = elapsedLabel ? ` after ${elapsedLabel}` : '';
    setStatus(`Mesh failed${timing}: ${err.message || err}`, 'error');
    emitBuilderEvent('adaptiveRunCompleted', {
      caseName: state.caseName || caseName,
      succeeded: false,
      error: err.message || String(err),
      durationMs: elapsedMs ?? undefined,
    });
  } finally {
    stopRunTimer();
    setAdaptiveButtonBusy(false);
  }
}

function downloadDefinition() {
  const definition = serializeDefinition();
  const blob = new Blob([JSON.stringify(definition, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${definition.name || 'case'}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

window.addEventListener('DOMContentLoaded', init);

const CaseBuilderAPI = {
  getCaseName: () => state.caseName || null,
  getCases: () => [...state.cases],
  loadCase,
  newCase,
  serializeDefinition,
  on(event, handler) {
    window.addEventListener(`${BUILDER_EVENT_PREFIX}${event}`, handler);
  },
  off(event, handler) {
    window.removeEventListener(`${BUILDER_EVENT_PREFIX}${event}`, handler);
  },
};

window.CaseBuilderAPI = CaseBuilderAPI;
function renderDxfDebugPanel() {
  if (!elements.dxfDebugPanel || !elements.dxfDebugText) return;
  if (!state.debug?.showDxfPoints) {
    elements.dxfDebugPanel.classList.add('hidden');
    elements.dxfDebugText.textContent = '';
    return;
  }
  const points = state.debug?.previewPoints || [];
  if (!points.length) {
    elements.dxfDebugPanel.classList.add('hidden');
    elements.dxfDebugText.textContent = '';
    return;
  }
  const maxRows = 200;
  const lines = [];
  lines.push('# X Y');
  points.slice(0, maxRows).forEach((p, idx) => {
    lines.push(`${idx + 1} ${p.x.toFixed(5)} ${p.y.toFixed(5)}`);
  });
  if (points.length > maxRows) {
    lines.push(`… ${points.length - maxRows} more`);
  }
  elements.dxfDebugPanel.classList.remove('hidden');
  elements.dxfDebugText.textContent = lines.join('\n');
}
function parseDxfText(text) {
  const pairs = [];
  const raw = text.replace(/\r\n/g, '\n').split(/\n/);
  for (let i = 0; i < raw.length - 1; i += 2) {
    pairs.push([raw[i].trim(), raw[i + 1].trim()]);
  }

  let inEntities = false;
  const segments = [];
  const points = [];
  const pointIndex = new Map();
  const bounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };

  const addPoint = (x, y) => {
    const key = x.toFixed(6) + ',' + y.toFixed(6);
    let idx = pointIndex.get(key);
    if (idx === undefined) {
      idx = points.length;
      points.push({ id: points.length + 1, x, y });
      pointIndex.set(key, idx);
    }
    includeInBounds(x, y);
    return idx;
  };

  const includeInBounds = (x, y) => {
    bounds.minX = Math.min(bounds.minX, x);
    bounds.maxX = Math.max(bounds.maxX, x);
    bounds.minY = Math.min(bounds.minY, y);
    bounds.maxY = Math.max(bounds.maxY, y);
  };

  const polarPoint = (cx, cy, r, deg) => {
    const rad = (deg * Math.PI) / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  };

  const parseEntity = (type, startIdx) => {
    const data = {};
    let i = startIdx + 1;
    for (; i < pairs.length; i++) {
      const [code, value] = pairs[i];
      if (code === '0') break;
      data[code] = value;
    }
    if (type === 'LINE') {
      const sx = parseFloat(data['10']);
      const sy = parseFloat(data['20']);
      const ex = parseFloat(data['11']);
      const ey = parseFloat(data['21']);
      if ([sx, sy, ex, ey].every(Number.isFinite)) {
        const a = addPoint(sx, sy);
        const b = addPoint(ex, ey);
        segments.push({ type: 'LINE', start: a, end: b });
      }
    } else if (type === 'ARC') {
      const cx = parseFloat(data['10']);
      const cy = parseFloat(data['20']);
      const r = parseFloat(data['40']);
      const startAngle = parseFloat(data['50']);
      const endAngle = parseFloat(data['51']);
      if ([cx, cy, r, startAngle, endAngle].every(Number.isFinite)) {
        includeInBounds(cx - r, cy - r);
        includeInBounds(cx + r, cy + r);
        const startPt = polarPoint(cx, cy, r, startAngle);
        const endPt = polarPoint(cx, cy, r, endAngle);
        const a = addPoint(startPt.x, startPt.y);
        const b = addPoint(endPt.x, endPt.y);
        segments.push({ type: 'ARC', start: a, end: b, center: { x: cx, y: cy }, radius: r, startAngle, endAngle });
      }
    } else if (type === 'CIRCLE') {
      const cx = parseFloat(data['10']);
      const cy = parseFloat(data['20']);
      const r = parseFloat(data['40']);
      if ([cx, cy, r].every(Number.isFinite)) {
        includeInBounds(cx - r, cy - r);
        includeInBounds(cx + r, cy + r);
        const startPt = polarPoint(cx, cy, r, 0);
        const a = addPoint(startPt.x, startPt.y);
        segments.push({ type: 'CIRCLE', start: a, end: a, center: { x: cx, y: cy }, radius: r, startAngle: 0, endAngle: 360 });
      }
    }
    return i;
  };

  for (let i = 0; i < pairs.length; i++) {
    const [code, value] = pairs[i];
    if (code === '0' && value === 'SECTION' && pairs[i + 1]?.[1] === 'ENTITIES') {
      inEntities = true;
      i += 1;
      continue;
    }
    if (code === '0' && value === 'ENDSEC' && inEntities) {
      inEntities = false;
    }
    if (!inEntities || code !== '0') continue;
    i = parseEntity(value, i) - 1;
  }

  return { points, segments, bounds };
}
function drawWireframe(ctx, wireframe, drawableWidth, drawableHeight, viewRect, color) {
  const pts = wireframe.points || [];
  const segments = wireframe.segments || [];
  ctx.save();
  ctx.strokeStyle = color || 'rgba(0,0,0,0.6)';
  ctx.lineWidth = Math.max(1, 2 / (state.view?.zoom || 1));
  ctx.setLineDash([]);
  // draw points as markers to verify placement
  ctx.fillStyle = '#1d4ed8';
  pts.forEach((pt) => {
    const c = physToCanvas(pt.x, pt.y, drawableWidth, drawableHeight, viewRect);
    ctx.beginPath();
    ctx.arc(c.x, c.y, 2.2, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.strokeStyle = color || 'rgba(0,0,0,0.6)';
  segments.forEach((seg) => {
    const startIdx = Number(seg.start);
    const endIdx = Number(seg.end);
    const a = pts[startIdx];
    const b = pts[endIdx];
    if (!a || !b) return;
    const stype = (seg.type || '').toString().toUpperCase();
    if (stype === 'ARC' || stype === 'CIRCLE') {
      ctx.strokeStyle = '#e11d48';
      drawArcSegment(ctx, seg, a, b, drawableWidth, drawableHeight, viewRect);
    } else {
      ctx.strokeStyle = color || 'rgba(0,0,0,0.6)';
      const ca = physToCanvas(a.x, a.y, drawableWidth, drawableHeight, viewRect);
      const cb = physToCanvas(b.x, b.y, drawableWidth, drawableHeight, viewRect);
      ctx.beginPath();
      ctx.moveTo(ca.x, ca.y);
      ctx.lineTo(cb.x, cb.y);
      ctx.stroke();
    }
  });
  ctx.restore();
}
function computeArcAngles(seg, cx, cy) {
  const stype = (seg.type || '').toString().toUpperCase();
  if (stype === 'CIRCLE') {
    return { start: 0, span: 360 };
  }
  let startDeg = normalizeAngle(seg.startAngle ?? seg.start_angle ?? 0);
  let endDeg = normalizeAngle(seg.endAngle ?? seg.end_angle ?? 0);
  // If we have endpoints, derive from them.
  if (Number.isFinite(seg.start) && Number.isFinite(seg.end) && Array.isArray(seg.points)) {
    const a = seg.points[seg.start];
    const b = seg.points[seg.end];
    if (a && b) {
      startDeg = normalizeAngle((Math.atan2(a.y - cy, a.x - cx) * 180) / Math.PI);
      endDeg = normalizeAngle((Math.atan2(b.y - cy, b.x - cx) * 180) / Math.PI);
    }
  }
  let span = endDeg - startDeg;
  if (span <= 0) span += 360;
  // Take the smaller span for arcs
  if (span > 180) {
    span = 360 - span;
    const tmp = startDeg;
    startDeg = endDeg;
    endDeg = tmp;
  }
  // If explicit start/end angles produce a smaller span, use that
  if (Number.isFinite(seg.startAngle) && Number.isFinite(seg.endAngle)) {
    let altStart = normalizeAngle(seg.startAngle);
    let altEnd = normalizeAngle(seg.endAngle);
    let altSpan = altEnd - altStart;
    if (altSpan <= 0) altSpan += 360;
    if (altSpan > 180) {
      altSpan = 360 - altSpan;
      const t2 = altStart;
      altStart = altEnd;
      altEnd = t2;
    }
    if (altSpan < span) {
      span = altSpan;
      startDeg = altStart;
    }
  }
  return { start: startDeg, span };
}
