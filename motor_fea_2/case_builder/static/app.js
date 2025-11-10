const bootstrap = window.__BOOTSTRAP__ || {};
const MATERIAL_DEFAULTS = {
  magnet: { label: 'Permanent Magnet', color: '#e4572e', params: { mu_r: 1.05, Mx: 0, My: 800000 } },
  steel: { label: 'Steel', color: '#2e86de', params: { mu_r: 1000 } },
  wire: { label: 'Wire', color: '#f5a623', params: { current: 5000 } },
  air: { label: 'Air', color: '#7b8ba1', params: { mu_r: 1.0 } },
};

const state = {
  cases: bootstrap.cases || [],
  caseName: bootstrap.defaultCase || '',
  definitionName: '',
  grid: sanitizeGrid(bootstrap.defaultGrid || { Nx: 120, Ny: 120, Lx: 1, Ly: 1 }),
  objects: [],
  selectedId: null,
  tool: 'select',
  dirty: false,
  status: '',
  view: null,
};
state.view = createDefaultView(state.grid);

const elements = {};
const canvasState = {
  mode: null,
  pointerId: null,
  start: null,
  shapeId: null,
  offset: null,
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

function init() {
  cacheElements();
  bindEvents();
  renderCaseSelect();
  if (state.caseName) {
    elements.caseNameInput.value = state.caseName;
    loadCase(state.caseName);
  } else {
    setActiveTool('select');
    renderAll();
  }
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
  elements.toolButtons = Array.from(document.querySelectorAll('.tool-btn'));
  elements.shapeList = document.getElementById('shapeList');
  elements.deleteShapeBtn = document.getElementById('deleteShapeBtn');
  elements.duplicateShapeBtn = document.getElementById('duplicateShapeBtn');
  elements.inspectorEmpty = document.getElementById('inspectorEmpty');
  elements.inspectorFields = document.getElementById('inspectorFields');
  elements.shapeLabel = document.getElementById('shapeLabel');
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
    }
  });
  elements.caseNameInput.addEventListener('input', () => {
    state.caseName = elements.caseNameInput.value.trim();
  });
  elements.definitionNameInput.addEventListener('input', () => {
    state.definitionName = elements.definitionNameInput.value.trim();
    markDirty();
  });
  ['Nx', 'Ny', 'Lx', 'Ly'].forEach((axis) => {
    elements[`grid${axis}`].addEventListener('input', () => {
      updateGridFromInputs();
    });
  });
  elements.toolButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      setActiveTool(btn.dataset.tool);
    });
  });
  elements.shapeList.addEventListener('click', (event) => {
    const item = event.target.closest('.shape-item');
    if (item) {
      selectShape(item.dataset.id);
    }
  });
  elements.deleteShapeBtn.addEventListener('click', deleteSelectedShape);
  elements.duplicateShapeBtn.addEventListener('click', duplicateSelectedShape);
  elements.materialSelect.addEventListener('change', () => updateSelectedMaterial(elements.materialSelect.value));
  elements.shapeLabel.addEventListener('input', () => {
    const shape = getSelectedShape();
    if (!shape) return;
    shape.label = elements.shapeLabel.value;
    markDirty();
    renderShapeList();
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
  elements.paramInputs.magnet.mu.addEventListener('input', () => updateParam('magnet', 'mu_r', elements.paramInputs.magnet.mu.value));
  elements.paramInputs.magnet.mx.addEventListener('input', () => updateParam('magnet', 'Mx', elements.paramInputs.magnet.mx.value));
  elements.paramInputs.magnet.my.addEventListener('input', () => updateParam('magnet', 'My', elements.paramInputs.magnet.my.value));
  elements.paramInputs.steel.mu.addEventListener('input', () => updateParam('steel', 'mu_r', elements.paramInputs.steel.mu.value));
  elements.paramInputs.air.mu.addEventListener('input', () => updateParam('air', 'mu_r', elements.paramInputs.air.mu.value));
  elements.paramInputs.wire.current.addEventListener('input', () => updateParam('wire', 'current', elements.paramInputs.wire.current.value));
  elements.saveCaseBtn.addEventListener('click', saveCaseDefinition);
  elements.runCaseBtn.addEventListener('click', runCasePipeline);
  elements.downloadBtn.addEventListener('click', downloadDefinition);

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
  updateZoomLabel();
  scheduleDraw();
}

function updateGridInputs() {
  elements.gridNx.value = state.grid.Nx;
  elements.gridNy.value = state.grid.Ny;
  elements.gridLx.value = state.grid.Lx;
  elements.gridLy.value = state.grid.Ly;
  elements.definitionNameInput.value = state.definitionName || '';
  elements.caseNameInput.value = state.caseName || '';
}

function updateGridFromInputs() {
  const Nx = Math.max(4, parseInt(elements.gridNx.value, 10) || state.grid.Nx);
  const Ny = Math.max(4, parseInt(elements.gridNy.value, 10) || state.grid.Ny);
  const Lx = Math.max(0.01, parseFloat(elements.gridLx.value) || state.grid.Lx);
  const Ly = Math.max(0.01, parseFloat(elements.gridLy.value) || state.grid.Ly);
  state.grid = { Nx, Ny, Lx, Ly };
  clampViewCenter();
  updateZoomLabel();
  markDirty();
  scheduleDraw();
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
  state.objects = [];
  state.selectedId = null;
  state.dirty = false;
  elements.caseNameInput.value = name;
  elements.definitionNameInput.value = name;
  resetView({ schedule: false });
  renderAll();
  setActiveTool('select');
  setStatus(`Started new case '${name}'.`, 'info');
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
    state.grid = sanitizeGrid(def.grid || {});
    state.objects = (def.objects || []).map((obj, idx) => hydrateObject(obj, idx));
    resetView({ schedule: false });
    state.selectedId = state.objects.length ? state.objects[0].id : null;
    state.dirty = false;
    renderCaseSelect();
    renderAll();
    setActiveTool('select');
    setStatus(`Loaded ${state.caseName}`, 'info');
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

function sanitizeGrid(grid) {
  return {
    Nx: Math.max(4, parseInt(grid.Nx, 10) || 120),
    Ny: Math.max(4, parseInt(grid.Ny, 10) || 120),
    Lx: Math.max(0.01, parseFloat(grid.Lx) || 1.0),
    Ly: Math.max(0.01, parseFloat(grid.Ly) || 1.0),
  };
}

function hydrateObject(raw, idx) {
  const material = MATERIAL_DEFAULTS[raw.material] ? raw.material : 'air';
  const id = raw.id || `obj-${Date.now()}-${idx}`;
  const label = raw.label || `${MATERIAL_DEFAULTS[material].label} ${idx + 1}`;
  const params = { ...deepCopy(MATERIAL_DEFAULTS[material].params), ...(raw.params || {}) };
  const shape = normalizeShape(raw.shape || {}, material);
  return { id, label, material, params, shape };
}

function normalizeShape(shape, material) {
  const typeRaw = typeof shape.type === 'string' ? shape.type.toLowerCase() : 'rect';
  const type = ['rect', 'circle', 'ring'].includes(typeRaw) ? typeRaw : 'rect';
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
  const radius = Math.abs(Number(shape.radius) || 0.02 * Math.max(state.grid.Lx, state.grid.Ly));
  return { type: 'circle', center: [center.x, center.y], radius };
}

function renderShapeList() {
  elements.shapeList.innerHTML = '';
  state.objects.forEach((obj) => {
    const div = document.createElement('div');
    div.className = 'shape-item' + (obj.id === state.selectedId ? ' selected' : '');
    div.dataset.id = obj.id;
    const color = MATERIAL_DEFAULTS[obj.material]?.color || '#999';
    div.innerHTML = `
      <span><strong style="color:${color}">●</strong> ${obj.label}</span>
      <span>${obj.shape.type}</span>
    `;
    elements.shapeList.appendChild(div);
  });
  elements.deleteShapeBtn.disabled = !state.selectedId;
  elements.duplicateShapeBtn.disabled = !state.selectedId;
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

function selectShape(id) {
  state.selectedId = id;
  renderShapeList();
  updateInspector();
  scheduleDraw();
}

function deleteSelectedShape() {
  if (!state.selectedId) return;
  state.objects = state.objects.filter((obj) => obj.id !== state.selectedId);
  state.selectedId = state.objects.length ? state.objects[0].id : null;
  markDirty();
  renderAll();
}

function duplicateSelectedShape() {
  const shape = getSelectedShape();
  if (!shape) return;
  const copy = deepCopy(shape);
  copy.id = uniqueId();
  copy.label = `${shape.label} copy`;
  if (copy.shape.center) {
    copy.shape.center[0] = clamp(copy.shape.center[0] + 0.01 * state.grid.Lx, 0, state.grid.Lx);
    copy.shape.center[1] = clamp(copy.shape.center[1] + 0.01 * state.grid.Ly, 0, state.grid.Ly);
  }
  state.objects.push(copy);
  state.selectedId = copy.id;
  markDirty();
  renderAll();
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
  state.dirty = true;
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
  state.objects.forEach((obj) => {
    drawObject(ctx, obj, drawableWidth, drawableHeight, viewRect);
  });
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
  ctx.fillStyle = `${color}33`;
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
  }
  if (obj.id === state.selectedId) {
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
  const hit = hitTest(phys);
  if (hit) {
    selectShape(hit.id);
    canvasState.mode = 'drag';
    canvasState.shapeId = hit.id;
    canvasState.offset = {
      x: phys.x - hit.shape.center[0],
      y: phys.y - hit.shape.center[1],
    };
  } else if (state.tool === 'select') {
    startPan(event, { deselectOnRelease: true });
  } else {
    state.selectedId = null;
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
  }
  canvasState.shapeId = shape.id;
  canvasState.start = phys;
  state.objects.push(shape);
  state.selectedId = shape.id;
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
  } else if (canvasState.mode === 'drag') {
    const shape = state.objects.find((obj) => obj.id === canvasState.shapeId);
    if (!shape) return;
    shape.shape.center[0] = clamp(phys.x - canvasState.offset.x, 0, state.grid.Lx);
    shape.shape.center[1] = clamp(phys.y - canvasState.offset.y, 0, state.grid.Ly);
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
  }
}

function onPointerUp() {
  if (!canvasState.pointerId) return;
  elements.canvas.releasePointerCapture(canvasState.pointerId);
  const mode = canvasState.mode;
  const shape = state.objects.find((obj) => obj.id === canvasState.shapeId);
  if (shape && (mode === 'draw-rect' || mode === 'draw-circle' || mode === 'draw-ring')) {
    let minimal = false;
    if (mode === 'draw-rect') {
      minimal = shape.shape.width * shape.shape.height < 1e-6;
    } else if (mode === 'draw-circle') {
      minimal = shape.shape.radius < 1e-3;
    } else if (mode === 'draw-ring') {
      minimal = shape.shape.outer_radius < 1e-3 || shape.shape.outer_radius - shape.shape.inner_radius < 1e-4;
    }
    if (minimal) {
      state.objects = state.objects.filter((obj) => obj.id !== shape.id);
      state.selectedId = null;
      renderShapeList();
      updateInspector();
    }
  }
  if (mode === 'pan' && canvasState.pendingDeselect) {
    state.selectedId = null;
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
  for (let i = state.objects.length - 1; i >= 0; i -= 1) {
    const obj = state.objects[i];
    if (obj.shape.type === 'rect') {
      if (pointInRotatedRect(point, obj.shape)) {
        return obj;
      }
    } else if (obj.shape.type === 'circle') {
      const [cx, cy] = obj.shape.center;
      const dx = point.x - cx;
      const dy = point.y - cy;
      if (Math.sqrt(dx * dx + dy * dy) <= (obj.shape.radius || 0)) {
        return obj;
      }
    } else if (obj.shape.type === 'ring') {
      const [cx, cy] = obj.shape.center;
      const dx = point.x - cx;
      const dy = point.y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const outer = obj.shape.outer_radius || 0;
      const inner = Math.max(0, obj.shape.inner_radius || 0);
      if (dist <= outer && dist >= inner) {
        return obj;
      }
    }
  }
  return null;
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
    })),
  };
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
  if (!elements.runCaseBtn) return;
  elements.runCaseBtn.disabled = isBusy;
  elements.runCaseBtn.dataset.busy = isBusy ? 'true' : 'false';
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
  setRunButtonBusy(true);
  setStatus('Saving case and running pipeline...', 'info');
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
    const added = ensureCaseTracked(state.caseName);
    if (!added) {
      renderCaseSelect();
    }
    const summary = summarizeSteps(data.steps);
    setStatus(summary ? `Run complete (${summary}).` : 'Run complete.', 'success');
    emitBuilderEvent('runCompleted', {
      caseName: state.caseName,
      succeeded: true,
      steps: data.steps || [],
    });
  } catch (err) {
    console.error(err);
    setStatus(`Run failed: ${err.message || err}`, 'error');
    emitBuilderEvent('runCompleted', {
      caseName: state.caseName || caseName,
      succeeded: false,
      error: err.message || String(err),
    });
  } finally {
    setRunButtonBusy(false);
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
