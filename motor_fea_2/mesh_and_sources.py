#!/usr/bin/env python3
"""
mesh_and_sources.py
-------------------
Generate a 2-D triangle mesh (from a square grid) plus material & source fields
for magnetostatic Az formulation, then save to NPZ. Designed so you can later
swap in an unstructured tri mesh without changing the solver.

Regions:
  0 = air, 1 = permanent magnet, 2 = steel

Saved fields (per element unless noted):
  nodes: (Nn,2) float64
  tris:  (Ne,3) int32    (CCW vertex order)
  region_id: (Ne,) int8  {0,1,2}
  mu_r:  (Ne,) float64
  Mx, My: (Ne,) float64  magnetization (A/m)
  Jz:    (Ne,) float64   free current density (A/m^2)
  meta:  dict (domain size, flags, etc.)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

AIR, PMAG, STEEL = 0, 1, 2

CASE_DEFINITION_FILENAME = "case_definition.json"

def square_tri_mesh(Nx=100, Ny=100, Lx=1.0, Ly=1.0, *,
                    x_coords: np.ndarray | None = None,
                    y_coords: np.ndarray | None = None):
    """Structured square grid split into two CCW triangles per cell."""
    if x_coords is not None:
        x = np.asarray(x_coords, dtype=float)
        if x.ndim != 1 or x.size < 2:
            raise ValueError("x_coords must be 1-D with at least 2 entries")
        if abs(x[0]) > 1e-12 or abs(x[-1] - Lx) > 1e-9:
            raise ValueError("x_coords must span [0, Lx]")
        Nx = x.size
    else:
        x = np.linspace(0, Lx, Nx)
    if y_coords is not None:
        y = np.asarray(y_coords, dtype=float)
        if y.ndim != 1 or y.size < 2:
            raise ValueError("y_coords must be 1-D with at least 2 entries")
        if abs(y[0]) > 1e-12 or abs(y[-1] - Ly) > 1e-9:
            raise ValueError("y_coords must span [0, Ly]")
        Ny = y.size
    else:
        y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    def nid(i,j): return i*Ny + j

    tris = []
    for i in range(Nx-1):
        for j in range(Ny-1):
            n00 = nid(i,   j)
            n10 = nid(i+1, j)
            n01 = nid(i,   j+1)
            n11 = nid(i+1, j+1)
            # Two CCW triangles per quad
            tris.append([n00, n10, n11])
            tris.append([n00, n11, n01])
    tris = np.asarray(tris, dtype=np.int32)

    # Ensure CCW orientation (robust even if re-meshed later)
    def tri_area(p, q, r):
        return 0.5*((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]))
    P = nodes
    for k in range(tris.shape[0]):
        a,b,c = tris[k]
        if tri_area(P[a],P[b],P[c]) <= 0.0:
            tris[k,[1,2]] = tris[k,[2,1]]
    return nodes, tris, (Lx, Ly)


def _shape_bounds(shape: dict | None) -> tuple[float, float, float, float] | None:
    """Return (xmin, xmax, ymin, ymax) for a supported shape."""
    if not isinstance(shape, dict):
        return None
    stype = str(shape.get("type", "")).lower()
    if stype == "rect":
        center = shape.get("center", [0.0, 0.0])
        if isinstance(center, dict):
            cx = float(center.get("x", 0.0))
            cy = float(center.get("y", 0.0))
        else:
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                cx, cy = float(center[0]), float(center[1])
            else:
                cx = cy = 0.0
        size = shape.get("size")
        if isinstance(size, dict):
            size_w = size.get("width")
            size_h = size.get("height")
        elif size is not None:
            size_w = size_h = size
        else:
            size_w = size_h = None
        width = float(shape.get("width", size_w if size_w is not None else 0.0))
        height = float(shape.get("height", size_h if size_h is not None else 0.0))
        if width <= 0 or height <= 0:
            return None
        half_w = 0.5 * width
        half_h = 0.5 * height
        corners = np.array([
            [ half_w,  half_h],
            [ half_w, -half_h],
            [-half_w, -half_h],
            [-half_w,  half_h],
        ])
        angle = float(shape.get("angle", 0.0))
        if angle:
            ang = math.radians(angle)
            rot = np.array([[math.cos(ang), -math.sin(ang)],
                            [math.sin(ang),  math.cos(ang)]])
            corners = corners @ rot.T
        corners += np.array([cx, cy])
        xmin = float(corners[:, 0].min())
        xmax = float(corners[:, 0].max())
        ymin = float(corners[:, 1].min())
        ymax = float(corners[:, 1].max())
        return xmin, xmax, ymin, ymax
    if stype == "circle":
        center = shape.get("center", [0.0, 0.0])
        if isinstance(center, dict):
            cx = float(center.get("x", 0.0))
            cy = float(center.get("y", 0.0))
        else:
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                cx, cy = float(center[0]), float(center[1])
            else:
                cx = cy = 0.0
        radius = float(shape.get("radius", 0.0))
        if radius <= 0:
            return None
        return cx - radius, cx + radius, cy - radius, cy + radius
    if stype == "ring":
        center = shape.get("center", [0.0, 0.0])
        if isinstance(center, dict):
            cx = float(center.get("x", 0.0))
            cy = float(center.get("y", 0.0))
        else:
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                cx, cy = float(center[0]), float(center[1])
            else:
                cx = cy = 0.0
        radius = float(shape.get("outer_radius",
                                 shape.get("outerRadius",
                                           shape.get("radius", 0.0))))
        if radius <= 0:
            return None
        return cx - radius, cx + radius, cy - radius, cy + radius
    return None


def _merge_intervals(intervals: list[tuple[float, float]], *, eps: float = 1e-12
                    ) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda ab: ab[0])
    merged: list[tuple[float, float]] = []
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end + eps:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def _collect_focus_spans(definition: dict | None,
                         axis: str,
                         *,
                         length: float,
                         materials: list[str] | None,
                         pad: float,
                         manual_boxes: list[dict] | None) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    if isinstance(definition, dict):
        objs = definition.get("objects", [])
        if isinstance(objs, list):
            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                material = str(obj.get("material", "air")).lower()
                if materials and material not in materials:
                    continue
                bounds = _shape_bounds(obj.get("shape"))
                if bounds is None:
                    continue
                if axis == "x":
                    lo, hi = bounds[0], bounds[1]
                else:
                    lo, hi = bounds[2], bounds[3]
                span_start = lo - pad
                span_end = hi + pad
                spans.append((span_start, span_end))
    if manual_boxes:
        for box in manual_boxes:
            if not isinstance(box, dict):
                continue
            coords = box.get(axis)
            if isinstance(coords, (list, tuple)) and len(coords) == 2:
                spans.append((float(coords[0]), float(coords[1])))
    clipped: list[tuple[float, float]] = []
    for start, end in spans:
        lo = float(min(start, end))
        hi = float(max(start, end))
        lo = max(0.0, lo)
        hi = min(length, hi)
        if hi - lo > 1e-9:
            clipped.append((lo, hi))
    return _merge_intervals(clipped)


def _axis_setting(mesh_spec: dict, axis: str, key: str, default=None):
    axis_dict = mesh_spec.get(axis)
    if isinstance(axis_dict, dict) and key in axis_dict:
        return axis_dict[key]
    axis_key = f"{key}_{axis}"
    if axis_key in mesh_spec:
        return mesh_spec[axis_key]
    return mesh_spec.get(key, default)


def _axis_spacing(mesh_spec: dict, axis: str, *, key_candidates: list[str], default: float):
    for candidate in key_candidates:
        value = _axis_setting(mesh_spec, axis, candidate)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def _graded_axis(length: float,
                 coarse: float,
                 fine: float,
                 spans: list[tuple[float, float]],
                 falloff: float) -> np.ndarray:
    """Return monotonically increasing coordinates from 0..length."""
    length = float(length)
    coarse = float(coarse)
    fine = float(fine)
    if coarse <= 0 or length <= 0:
        raise ValueError("Domain length and coarse spacing must be positive")
    fine = max(min(fine, coarse), 1e-6)
    sample_count = max(1024, int(length / max(fine, 1e-6)) * 8)
    xs = np.linspace(0.0, length, sample_count)
    influence_total = np.zeros_like(xs)
    falloff = max(float(falloff), 0.0)
    if spans:
        for start, end in spans:
            s = max(0.0, min(length, float(start)))
            e = max(0.0, min(length, float(end)))
            if e <= s:
                continue
            dist = np.zeros_like(xs)
            left_mask = xs < s
            right_mask = xs > e
            dist[left_mask] = s - xs[left_mask]
            dist[right_mask] = xs[right_mask] - e
            if falloff > 0:
                influence = np.exp(-0.5 * (dist / falloff) ** 2)
            else:
                influence = (dist == 0.0).astype(float)
            influence_total += influence
        influence_total = np.clip(influence_total, 0.0, 1.0)
    strength = max(coarse / fine - 1.0, 0.0)
    relative_density = 1.0 + strength * influence_total
    base_density = 1.0 / coarse
    density = base_density * relative_density
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(xs))])
    total_cells = cumulative[-1]
    n_cells = max(1, int(math.ceil(total_cells)))
    targets = np.linspace(0.0, total_cells, n_cells + 1)
    coords = np.interp(targets, cumulative, xs)
    coords[0] = 0.0
    coords[-1] = length
    return coords


def _build_adaptive_axes(grid: dict, definition: dict | None
                        ) -> tuple[np.ndarray, np.ndarray, dict] | None:
    mesh_spec = grid.get("mesh") or grid.get("meshing")
    if not isinstance(mesh_spec, dict):
        return None
    if mesh_spec.get("enabled") is False:
        return None
    mesh_type = str(mesh_spec.get("type", "graded")).lower() or "graded"
    if mesh_type != "graded":
        return None
    Lx = float(grid["Lx"])
    Ly = float(grid["Ly"])
    approx_Nx = int(grid.get("Nx", max(int(Lx / 0.01), 8)))
    approx_Ny = int(grid.get("Ny", max(int(Ly / 0.01), 8)))
    default_coarse_x = Lx / max(approx_Nx - 1, 1)
    default_coarse_y = Ly / max(approx_Ny - 1, 1)
    default_fine_x = default_coarse_x / 3.0
    default_fine_y = default_coarse_y / 3.0

    coarse_x = _axis_spacing(mesh_spec, "x", key_candidates=["coarse", "coarse_dx", "max_dx", "dx_far"], default=default_coarse_x)
    coarse_y = _axis_spacing(mesh_spec, "y", key_candidates=["coarse", "coarse_dy", "max_dy", "dy_far"], default=default_coarse_y)
    fine_x = _axis_spacing(mesh_spec, "x", key_candidates=["fine", "fine_dx", "min_dx", "dx_near"], default=default_fine_x)
    fine_y = _axis_spacing(mesh_spec, "y", key_candidates=["fine", "fine_dy", "min_dy", "dy_near"], default=default_fine_y)

    pad = float(mesh_spec.get("focus_pad", 0.0))
    falloff = mesh_spec.get("focus_falloff")
    if falloff is None:
        falloff = 0.5 * pad if pad > 0 else 0.05 * min(Lx, Ly)
    falloff = max(float(falloff), 0.0)
    focus_materials = mesh_spec.get("focus_materials")
    if focus_materials is None:
        focus_materials = ["magnet", "steel", "wire"]
    elif isinstance(focus_materials, (set, tuple)):
        focus_materials = list(focus_materials)
    elif isinstance(focus_materials, str):
        focus_materials = [focus_materials]
    focus_materials = [str(m).lower() for m in focus_materials]
    manual_boxes = mesh_spec.get("focus_boxes")

    spans_x = _collect_focus_spans(definition, "x", length=Lx,
                                   materials=focus_materials,
                                   pad=pad,
                                   manual_boxes=manual_boxes if isinstance(manual_boxes, list) else None)
    spans_y = _collect_focus_spans(definition, "y", length=Ly,
                                   materials=focus_materials,
                                   pad=pad,
                                   manual_boxes=manual_boxes if isinstance(manual_boxes, list) else None)
    x_coords = _graded_axis(Lx, coarse_x, fine_x, spans_x, falloff)
    y_coords = _graded_axis(Ly, coarse_y, fine_y, spans_y, falloff)
    meta = {
        "type": mesh_type,
        "x": {
            "coarse": coarse_x,
            "fine": fine_x,
            "nodes": int(x_coords.size),
            "focus_spans": spans_x,
        },
        "y": {
            "coarse": coarse_y,
            "fine": fine_y,
            "nodes": int(y_coords.size),
            "focus_spans": spans_y,
        },
        "focus_pad": pad,
        "focus_falloff": falloff,
        "focus_materials": focus_materials,
    }
    if manual_boxes:
        meta["focus_boxes"] = manual_boxes
    return x_coords, y_coords, meta

def _safe_meta_copy(data):
    try:
        return json.loads(json.dumps(data))
    except (TypeError, ValueError):
        return data


def build_fields(nodes, tris, LxLy,
                 include_magnet=True,
                 include_steel=True,
                 include_wire=True,
                 magnet_My=8e5,           # A/m (μ0*M ~ 1.0 T)
                 mu_r_magnet=1.05,
                 mu_r_steel=1000.0,
                 wire_current=5000.0,     # A (per unit depth)
                 wire_radius=0.02,        # m
                 seed=0,
                 grid_spec: dict | None = None,
                 mesh_meta: dict | None = None):
    rng = np.random.default_rng(seed)
    Lx, Ly = LxLy
    Ne = tris.shape[0]
    P = nodes
    tri_area = tri_areas(nodes, tris)

    # Element centroids
    c = P[tris].mean(axis=1)      # (Ne,3,2)->(Ne,2)
    cx = c[:,0]; cy = c[:,1]

    # Default: air
    region = np.full(Ne, AIR, dtype=np.int8)
    mu_r   = np.ones(Ne, dtype=float)
    Mx     = np.zeros(Ne, dtype=float)
    My     = np.zeros(Ne, dtype=float)
    Jz     = np.zeros(Ne, dtype=float)

    # Layout: center things near the middle
    gx, gy = 0.5*Lx, 0.5*Ly

    # Parameterize rectangles so we can describe them in metadata later.
    mag_rect = dict(
        width=0.12 * Lx,
        height=0.06 * Ly,
        center=(gx - 0.08 * Lx, gy),
    )
    steel_rect = dict(
        width=0.10 * Lx,
        height=0.08 * Ly,
        center=(gx + 0.10 * Lx, gy + 0.05 * Ly),
    )

    # Permanent magnet: small rectangle, uniform My
    if include_magnet:
        w, h = mag_rect["width"], mag_rect["height"]
        cxm, cym = mag_rect["center"]
        mag_frac = rect_overlap_fraction(nodes, tris, cxm, cym, w, h, tri_area)
        mag_mask = mag_frac >= 0.5
        region[mag_mask] = PMAG
        mu_r = np.where(
            mag_frac > 0,
            (1.0 - mag_frac) * mu_r + mag_frac * mu_r_magnet,
            mu_r,
        )
        My += magnet_My * mag_frac  # area-weighted magnetization
    else:
        mag_frac = np.zeros(Ne, dtype=float)

    # Steel insert: another rectangle
    if include_steel:
        w, h = steel_rect["width"], steel_rect["height"]
        cxs, cys = steel_rect["center"]
        steel_frac = rect_overlap_fraction(nodes, tris, cxs, cys, w, h, tri_area)
        mask_major = (steel_frac >= 0.5) & (region != PMAG)
        region[mask_major] = STEEL
        mu_r = np.where(
            (steel_frac > 0) & (region != PMAG),
            (1.0 - steel_frac) * mu_r + steel_frac * mu_r_steel,
            mu_r,
        )
    else:
        steel_frac = np.zeros(Ne, dtype=float)

    # Wire pair (+I, -I) as uniform Jz over discs (to keep Neumann compatible)
    if include_wire:
        r2 = wire_radius**2
        # +I below center
        cx1, cy1 = gx, gy - 0.10*Ly
        # -I above center (return)
        cx2, cy2 = gx, gy + 0.20*Ly
        in_w1 = (cx - cx1)**2 + (cy - cy1)**2 <= r2
        in_w2 = (cx - cx2)**2 + (cy - cy2)**2 <= r2

        # Element areas
        A = tri_area
        A1 = A[in_w1].sum()
        A2 = A[in_w2].sum()
        if A1 > 0:
            Jz[in_w1] += wire_current / A1
        if A2 > 0:
            Jz[in_w2] -= wire_current / A2

    geometry = {}
    if include_magnet:
        geometry["magnet_rect"] = dict(
            type="rect",
            center=[float(cxm), float(cym)],
            width=float(0.12 * Lx),
            height=float(0.06 * Ly),
        )
    if include_steel:
        geometry["steel_rect"] = dict(
            type="rect",
            center=[float(cxs), float(cys)],
            width=float(0.10 * Lx),
            height=float(0.08 * Ly),
        )
    if include_wire:
        geometry["wire_disks"] = [
            dict(type="circle", center=[float(cx1), float(cy1)], radius=float(wire_radius), current=float(wire_current)),
            dict(type="circle", center=[float(cx2), float(cy2)], radius=float(wire_radius), current=float(-wire_current)),
        ]

    meta = dict(
        Lx=Lx, Ly=Ly,
        include_magnet=bool(include_magnet),
        include_steel=bool(include_steel),
        include_wire=bool(include_wire),
        mu_r_magnet=float(mu_r_magnet),
        mu_r_steel=float(mu_r_steel),
        magnet_My=float(magnet_My),
        wire_current=float(wire_current),
        wire_radius=float(wire_radius),
        geometry=geometry,
    )
    if grid_spec is not None:
        meta["grid"] = _safe_meta_copy(grid_spec)
    if mesh_meta is not None:
        meta["mesh_generation"] = _safe_meta_copy(mesh_meta)
    return dict(nodes=nodes, tris=tris, region_id=region,
                mu_r=mu_r, Mx=Mx, My=My, Jz=Jz, meta=meta)


def build_fields_from_definition(nodes, tris, LxLy, definition, *,
                                 definition_source=None,
                                 grid_spec: dict | None = None,
                                 mesh_meta: dict | None = None):
    """Build material and source fields from a JSON case definition."""
    Lx, Ly = LxLy
    tri_area_vals = tri_areas(nodes, tris)
    Ne = tris.shape[0]
    region = np.full(Ne, AIR, dtype=np.int8)
    mu_r = np.ones(Ne, dtype=float)
    Mx = np.zeros(Ne, dtype=float)
    My = np.zeros(Ne, dtype=float)
    Jz = np.zeros(Ne, dtype=float)

    defaults = definition.get("defaults", {})
    objects = definition.get("objects", [])
    geometry = []

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        material = str(obj.get("material", "air")).lower()
        shape = obj.get("shape") or {}
        params = obj.get("params") or {}
        min_fill = float(obj.get("min_fill", 0.5))
        fraction = _shape_overlap_fraction(shape, nodes, tris, tri_area_vals)
        if fraction is None:
            continue
        has_overlap = fraction > 0
        if not np.any(has_overlap):
            geometry.append(obj)
            continue

        mat_defaults = defaults.get(material, {}) if isinstance(defaults, dict) else {}

        def _param(key, fallback):
            if key in params:
                return params[key]
            if isinstance(mat_defaults, dict) and key in mat_defaults:
                return mat_defaults[key]
            return fallback

        if material in {"air", "magnet", "steel"}:
            target_mu = float(_param("mu_r", 1.0 if material == "air" else 1.05 if material == "magnet" else 1000.0))
            mu_r = np.where(has_overlap, (1.0 - fraction) * mu_r + fraction * target_mu, mu_r)
            region_id = {"air": AIR, "magnet": PMAG, "steel": STEEL}[material]
            region = np.where(fraction >= min_fill, region_id, region)
            if material == "magnet":
                target_mx_local = float(_param("Mx", 0.0))
                target_my_local = float(_param("My", 8e5))
                angle_deg = _shape_angle_degrees(shape)
                target_mx, target_my = _rotate_xy(target_mx_local, target_my_local, angle_deg)
                Mx = np.where(has_overlap, (1.0 - fraction) * Mx + fraction * target_mx, Mx)
                My = np.where(has_overlap, (1.0 - fraction) * My + fraction * target_my, My)

        if material == "wire":
            total_current = float(_param("current", 0.0))
            effective_area = float(np.dot(tri_area_vals, fraction))
            if abs(effective_area) > 0:
                Jz += (fraction * total_current / effective_area)

        geometry.append(obj)

    meta = dict(
        Lx=Lx,
        Ly=Ly,
        defaults=defaults,
        geometry=geometry,
        case_definition=definition,
    )
    if definition_source:
        meta["case_definition_source"] = str(definition_source)
    if grid_spec is not None:
        meta["grid"] = _safe_meta_copy(grid_spec)
    if mesh_meta is not None:
        meta["mesh_generation"] = _safe_meta_copy(mesh_meta)

    return dict(
        nodes=nodes,
        tris=tris,
        region_id=region,
        mu_r=mu_r,
        Mx=Mx,
        My=My,
        Jz=Jz,
        meta=meta,
    )

def tri_areas(nodes, tris):
    P = nodes; T = tris
    x1,y1 = P[T[:,0],0], P[T[:,0],1]
    x2,y2 = P[T[:,1],0], P[T[:,1],1]
    x3,y3 = P[T[:,2],0], P[T[:,2],1]
    return 0.5*np.abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))


def rect_overlap_fraction(nodes, tris, cx, cy, width, height, tri_area=None, angle_deg=0.0):
    """Return area fraction of each triangle covered by a (possibly rotated) rectangle."""
    if width <= 0 or height <= 0:
        return np.zeros(tris.shape[0], dtype=float)

    tri_pts = nodes[tris]
    areas = tri_area if tri_area is not None else tri_areas(nodes, tris)
    fractions = np.zeros(tris.shape[0], dtype=float)
    angle_rad = math.radians(angle_deg)

    if abs(angle_rad) < 1e-9:
        xmin = cx - width / 2
        xmax = cx + width / 2
        ymin = cy - height / 2
        ymax = cy + height / 2
        pts_iter = tri_pts
    else:
        center = np.array([cx, cy])
        local = tri_pts - center
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot_x = local[:, :, 0] * cos_a + local[:, :, 1] * sin_a
        rot_y = -local[:, :, 0] * sin_a + local[:, :, 1] * cos_a
        pts_iter = np.stack((rot_x, rot_y), axis=2)
        xmin = -width / 2
        xmax = width / 2
        ymin = -height / 2
        ymax = height / 2

    for idx, pts in enumerate(pts_iter):
        area = polygon_rect_overlap_area(pts, xmin, xmax, ymin, ymax)
        if areas[idx] > 0:
            fractions[idx] = area / areas[idx]
    return fractions


def circle_overlap_fraction(nodes, tris, cx, cy, radius):
    """Approximate area fraction covered by a circle using barycentric samples."""
    if radius <= 0:
        return np.zeros(tris.shape[0], dtype=float)
    pts = nodes[tris]  # (Ne,3,2)
    bary_samples = (
        (1 / 3, 1 / 3, 1 / 3),  # centroid
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
        (0.8, 0.1, 0.1),
        (0.1, 0.8, 0.1),
        (0.1, 0.1, 0.8),
    )
    frac = np.zeros(tris.shape[0], dtype=float)
    r2 = radius * radius
    for bary in bary_samples:
        sample = bary[0] * pts[:, 0, :] + bary[1] * pts[:, 1, :] + bary[2] * pts[:, 2, :]
        inside = ((sample[:, 0] - cx) ** 2 + (sample[:, 1] - cy) ** 2) <= r2
        frac += inside.astype(float)
    frac /= len(bary_samples)
    return frac


def ring_overlap_fraction(nodes, tris, cx, cy, outer_radius, inner_radius):
    """Area fraction covered by a ring = outer disk minus inner disk."""
    if outer_radius <= 0:
        return np.zeros(tris.shape[0], dtype=float)
    outer = circle_overlap_fraction(nodes, tris, cx, cy, outer_radius)
    if inner_radius <= 0:
        return outer
    inner = circle_overlap_fraction(nodes, tris, cx, cy, min(inner_radius, outer_radius))
    frac = outer - inner
    return np.clip(frac, 0.0, 1.0)


def _shape_overlap_fraction(shape, nodes, tris, tri_area_vals):
    if not isinstance(shape, dict):
        return None
    stype = str(shape.get("type", "")).lower()
    if stype == "rect":
        center = shape.get("center", [0.0, 0.0])
        if isinstance(center, dict):
            cx = center.get("x", 0.0)
            cy = center.get("y", 0.0)
        else:
            cx, cy = center
        size = shape.get("size")
        if isinstance(size, dict):
            size_w = size.get("width")
            size_h = size.get("height")
        elif size is not None:
            size_w = size_h = size
        else:
            size_w = size_h = None
        width = shape.get("width", size_w if size_w is not None else 0.0)
        height = shape.get("height", size_h if size_h is not None else 0.0)
        angle = shape.get("angle", 0.0)
        return rect_overlap_fraction(
            nodes,
            tris,
            float(cx),
            float(cy),
            float(width),
            float(height),
            tri_area_vals,
            float(angle),
        )
    if stype == "circle":
        center = shape.get("center", [0.0, 0.0])
        if isinstance(center, dict):
            cx = center.get("x", 0.0)
            cy = center.get("y", 0.0)
        else:
            cx, cy = center
        radius = shape.get("radius", 0.0)
        return circle_overlap_fraction(nodes, tris, float(cx), float(cy), float(radius))
    if stype == "ring":
        center = shape.get("center", [0.0, 0.0])
        if isinstance(center, dict):
            cx = center.get("x", 0.0)
            cy = center.get("y", 0.0)
        else:
            cx, cy = center
        outer = shape.get("outer_radius", shape.get("outerRadius", shape.get("radius", 0.0)))
        inner = shape.get("inner_radius", shape.get("innerRadius", 0.0))
        return ring_overlap_fraction(
            nodes,
            tris,
            float(cx),
            float(cy),
            float(outer),
            float(inner),
        )
    return None


def _shape_angle_degrees(shape) -> float:
    """Return the rotation angle declared on a shape, defaulting to 0."""
    if isinstance(shape, dict) and "angle" in shape:
        try:
            return float(shape.get("angle", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _rotate_xy(mx: float, my: float, angle_deg: float) -> tuple[float, float]:
    """Rotate a 2-D vector by angle_deg (CCW, degrees)."""
    if not angle_deg:
        return mx, my
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return mx * cos_a - my * sin_a, mx * sin_a + my * cos_a


def polygon_rect_overlap_area(tri_pts, xmin, xmax, ymin, ymax):
    """Compute the area of a triangle clipped by an axis-aligned rectangle."""
    poly = clip_polygon_to_halfspace(tri_pts, axis=0, value=xmin, keep_greater=True)
    if not poly:
        return 0.0
    poly = clip_polygon_to_halfspace(poly, axis=0, value=xmax, keep_greater=False)
    if not poly:
        return 0.0
    poly = clip_polygon_to_halfspace(poly, axis=1, value=ymin, keep_greater=True)
    if not poly:
        return 0.0
    poly = clip_polygon_to_halfspace(poly, axis=1, value=ymax, keep_greater=False)
    if not poly:
        return 0.0
    return polygon_area(poly)


def clip_polygon_to_halfspace(vertices, axis, value, keep_greater):
    """Clip a polygon to x>=value / x<=value / y>=value / y<=value."""
    if len(vertices) == 0:
        return []
    idx = int(axis)
    result = []
    prev = vertices[-1]
    prev_inside = _halfspace_condition(prev[idx], value, keep_greater)
    for curr in vertices:
        curr_inside = _halfspace_condition(curr[idx], value, keep_greater)
        if curr_inside:
            if not prev_inside:
                result.append(_intersect(prev, curr, idx, value))
            result.append(curr.tolist() if isinstance(curr, np.ndarray) else list(curr))
        elif prev_inside:
            result.append(_intersect(prev, curr, idx, value))
        prev = curr
        prev_inside = curr_inside
    return result


def _halfspace_condition(coord, value, keep_greater):
    return coord >= value if keep_greater else coord <= value


def _intersect(p1, p2, axis_idx, value):
    p1x, p1y = p1
    p2x, p2y = p2
    coord1 = p1x if axis_idx == 0 else p1y
    coord2 = p2x if axis_idx == 0 else p2y
    d = coord2 - coord1
    if abs(d) < 1e-12:
        t = 0.0
    else:
        t = (value - coord1) / d
    t = np.clip(t, 0.0, 1.0)
    return [
        float(p1x + t * (p2x - p1x)),
        float(p1y + t * (p2y - p1y)),
    ]


def polygon_area(vertices):
    if len(vertices) < 3:
        return 0.0
    area = 0.0
    for i in range(len(vertices)):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % len(vertices)]
        area += x1 * y2 - x2 * y1
    return 0.5 * abs(area)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a magnetostatic test case")
    parser.add_argument("--case", default="mag2d_case",
                        help="Case subfolder name under --cases-dir (default: mag2d_case)")
    parser.add_argument("--cases-dir", default="cases",
                        help="Directory that stores case subfolders (default: cases)")
    parser.add_argument("--case-config",
                        help=(
                            "Path to a JSON case definition. If omitted, the generator looks "
                            "for cases/<case>/case_definition.json."
                        ))
    parser.add_argument("--Nx", type=int, default=100, help="Grid count along X (default: 100)")
    parser.add_argument("--Ny", type=int, default=100, help="Grid count along Y (default: 100)")
    parser.add_argument("--Lx", type=float, default=1.0, help="Domain width (m)")
    parser.add_argument("--Ly", type=float, default=1.0, help="Domain height (m)")
    parser.add_argument("--magnet-My", type=float, default=8e5,
                        help="Permanent magnet My (A/m, default: 8e5)")
    parser.add_argument("--mu-r-magnet", type=float, default=1.05,
                        help="Relative permeability used for magnet (default: 1.05)")
    parser.add_argument("--mu-r-steel", type=float, default=1000.0,
                        help="Relative permeability of steel insert (default: 1000)")
    parser.add_argument("--wire-current", type=float, default=5000.0,
                        help="Total coil current per conductor (A, default: 5000)")
    parser.add_argument("--wire-radius", type=float, default=0.02,
                        help="Radius of each current disk source (m, default: 0.02)")
    parser.add_argument("--no-magnet", action="store_true",
                        help="Omit the permanent magnet region")
    parser.add_argument("--no-steel", action="store_true",
                        help="Omit the steel insert region")
    parser.add_argument("--no-wire", action="store_true",
                        help="Omit the current-carrying coils")
    return parser.parse_args()


def _write_case(payload: dict, case_dir: Path, filename: str = "mesh.npz") -> Path:
    case_dir.mkdir(parents=True, exist_ok=True)
    outpath = case_dir / filename
    np.savez_compressed(outpath, **payload)
    return outpath


def _resolve_case_definition_path(arg_path: str | None, case_dir: Path) -> Path | None:
    if arg_path:
        path = Path(arg_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Case definition '{path}' was not found")
        return path
    candidate = (case_dir / CASE_DEFINITION_FILENAME).resolve()
    return candidate if candidate.exists() else None


def _load_case_definition(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _grid_from_args(args: argparse.Namespace, definition: dict | None) -> dict:
    grid = {
        "Nx": int(args.Nx),
        "Ny": int(args.Ny),
        "Lx": float(args.Lx),
        "Ly": float(args.Ly),
    }
    if definition and isinstance(definition, dict):
        overrides = definition.get("grid", {})
        if isinstance(overrides, dict):
            for key, value in overrides.items():
                if key in {"Nx", "Ny"}:
                    try:
                        grid[key] = int(value)
                    except (TypeError, ValueError):
                        continue
                elif key in {"Lx", "Ly"}:
                    try:
                        grid[key] = float(value)
                    except (TypeError, ValueError):
                        continue
                else:
                    grid[key] = value
    grid["Nx"] = max(2, int(grid.get("Nx", 2)))
    grid["Ny"] = max(2, int(grid.get("Ny", 2)))
    grid["Lx"] = float(grid.get("Lx", 1.0))
    grid["Ly"] = float(grid.get("Ly", 1.0))
    return grid


if __name__ == "__main__":
    args = _parse_args()
    case_dir = (Path(args.cases_dir) / args.case).resolve()
    case_def_path = _resolve_case_definition_path(args.case_config, case_dir)
    case_definition = _load_case_definition(case_def_path) if case_def_path else None

    grid = _grid_from_args(args, case_definition)
    x_coords = y_coords = None
    mesh_meta = None
    try:
        axes = _build_adaptive_axes(grid, case_definition)
    except ValueError as exc:
        raise SystemExit(f"Failed to build mesh coordinates: {exc}") from exc
    else:
        if axes is not None:
            x_coords, y_coords, mesh_meta = axes
            if isinstance(x_coords, np.ndarray):
                grid["Nx"] = int(x_coords.size)
            if isinstance(y_coords, np.ndarray):
                grid["Ny"] = int(y_coords.size)
    nodes, tris, L = square_tri_mesh(
        Nx=grid["Nx"],
        Ny=grid["Ny"],
        Lx=grid["Lx"],
        Ly=grid["Ly"],
        x_coords=x_coords,
        y_coords=y_coords,
    )

    if case_definition:
        source = case_def_path
        try:
            source = case_def_path.relative_to(case_dir)
        except ValueError:
            pass
        payload = build_fields_from_definition(
            nodes,
            tris,
            (grid["Lx"], grid["Ly"]),
            case_definition,
            definition_source=str(source),
            grid_spec=grid,
            mesh_meta=mesh_meta,
        )
    else:
        payload = build_fields(
            nodes,
            tris,
            (grid["Lx"], grid["Ly"]),
            include_magnet=not args.no_magnet,
            include_steel=not args.no_steel,
            include_wire=not args.no_wire,
            magnet_My=args.magnet_My,
            mu_r_magnet=args.mu_r_magnet,
            mu_r_steel=args.mu_r_steel,
            wire_current=args.wire_current,
            wire_radius=args.wire_radius,
            grid_spec=grid,
            mesh_meta=mesh_meta,
        )

    outpath = _write_case(payload, case_dir)
    print(
        f"Wrote {outpath} with:",
        f"\n  nodes={payload['nodes'].shape}",
        f"\n  tris={payload['tris'].shape}",
        f"\n  mu_r:  [{payload['mu_r'].min():.3g}, {payload['mu_r'].max():.3g}]",
        f"\n  |M|:   [{np.hypot(payload['Mx'],payload['My']).min():.3g}, "
        f"{np.hypot(payload['Mx'],payload['My']).max():.3g}] A/m",
        f"\n  Jz sum={payload['Jz'].sum() * (grid['Lx'] * grid['Ly']):.6g} A (approx)",
    )
