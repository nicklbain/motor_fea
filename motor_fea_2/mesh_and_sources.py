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
from scipy.spatial import Delaunay

try:
    import triangle as _triangle_lib
except Exception:  # noqa: BLE001
    _triangle_lib = None

from field_adapt import FieldAdaptError, build_axes_from_field
AIR, PMAG, STEEL = 0, 1, 2
UNSTRUCTURED_MESH_TYPES = {"point_cloud", "pointcloud", "delaunay", "unstructured", "experimental", "equilateral"}
# Large explicit polygons are downsampled to keep overlap clipping affordable.
MAX_EXPLICIT_POLYGON_SIDES = 256

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
    if stype == "polygon":
        verts_raw = shape.get("vertices")
        if isinstance(verts_raw, (list, tuple)) and len(verts_raw) >= 3:
            xs: list[float] = []
            ys: list[float] = []
            for v in verts_raw:
                try:
                    if isinstance(v, dict):
                        xs.append(float(v.get("x", 0.0)))
                        ys.append(float(v.get("y", 0.0)))
                    elif isinstance(v, (list, tuple)) and len(v) >= 2:
                        xs.append(float(v[0]))
                        ys.append(float(v[1]))
                except (TypeError, ValueError):
                    continue
            if xs and ys:
                return min(xs), max(xs), min(ys), max(ys)
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


def _collect_focus_regions(definition: dict | None,
                           *,
                           Lx: float,
                           Ly: float,
                           materials: list[str] | None,
                           pad: float,
                           manual_boxes: list[dict] | None) -> list[tuple[float, float, float, float]]:
    """Gather axis-aligned boxes that should receive fine sampling."""
    regions: list[tuple[float, float, float, float]] = []
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
                xmin = max(0.0, bounds[0] - pad)
                xmax = min(Lx, bounds[1] + pad)
                ymin = max(0.0, bounds[2] - pad)
                ymax = min(Ly, bounds[3] + pad)
                if xmax - xmin > 1e-9 and ymax - ymin > 1e-9:
                    regions.append((xmin, xmax, ymin, ymax))
    if manual_boxes:
        for box in manual_boxes:
            if not isinstance(box, dict):
                continue
            x_span = box.get("x")
            y_span = box.get("y")
            if (
                isinstance(x_span, (list, tuple)) and len(x_span) == 2
                and isinstance(y_span, (list, tuple)) and len(y_span) == 2
            ):
                xmin = max(0.0, min(float(x_span[0]), float(x_span[1])))
                xmax = min(Lx, max(float(x_span[0]), float(x_span[1])))
                ymin = max(0.0, min(float(y_span[0]), float(y_span[1])))
                ymax = min(Ly, max(float(y_span[0]), float(y_span[1])))
                if xmax - xmin > 1e-9 and ymax - ymin > 1e-9:
                    regions.append((xmin, xmax, ymin, ymax))
    return regions


def _unique_rows(points: np.ndarray) -> np.ndarray:
    """Remove duplicate 2-D points while preserving order."""
    if points.size == 0:
        return points
    pts = np.ascontiguousarray(points, dtype=float)
    dtype = np.dtype([("x", float), ("y", float)])
    view = pts.view(dtype)
    _, unique_idx = np.unique(view, return_index=True)
    unique_idx.sort()
    return pts[unique_idx]


def _orient_tris_ccw(nodes: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Ensure CCW orientation and drop zero-area triangles."""
    if tris.size == 0:
        return tris
    p = nodes
    a = p[tris[:, 0]]
    b = p[tris[:, 1]]
    c = p[tris[:, 2]]
    two_area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flip = two_area < 0.0
    if np.any(flip):
        flipped = tris[flip].copy()
        flipped[:, [1, 2]] = flipped[:, [2, 1]]
        tris[flip] = flipped
    keep = np.abs(two_area) > 1e-16
    return tris[keep]


def _quality_triangulation(points: np.ndarray,
                           *,
                           min_angle: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Attempt to remesh the point cloud with Triangle to improve minimum angles."""
    if _triangle_lib is None:
        return None, None
    if points.shape[0] < 3:
        return None, None
    data = {"vertices": points}
    min_angle = max(float(min_angle), 0.0)
    opts = f"Qq{min_angle:.6g}"
    try:
        triangulation = _triangle_lib.triangulate(data, opts)
    except Exception:  # noqa: BLE001
        return None, None
    verts = triangulation.get("vertices")
    tris = triangulation.get("triangles")
    if verts is None or tris is None:
        return None, None
    verts = np.asarray(verts, dtype=float)
    tris = np.asarray(tris, dtype=np.int32)
    tris = _orient_tris_ccw(verts, tris)
    if tris.size == 0:
        return None, None
    return verts, tris


def _build_point_cloud_mesh(grid: dict,
                            definition: dict | None,
                            *,
                            adapt_from: str | Path | None = None,
                            field_focus_spec: dict | None = None,
                            field_focus_defaults: dict | None = None,
                            mesh_only: bool = False
                            ) -> tuple[np.ndarray, np.ndarray, tuple[float, float], dict]:
    mesh_spec = grid.get("mesh") or grid.get("meshing") or {}
    if not isinstance(mesh_spec, dict):
        mesh_spec = {}
    mesh_type = str(mesh_spec.get("type", "")).lower() or "point_cloud"
    is_experimental = mesh_type in {"experimental", "equilateral"}
    Lx = float(grid["Lx"])
    Ly = float(grid["Ly"])
    approx_Nx = max(int(grid.get("Nx", 50)), 2)
    approx_Ny = max(int(grid.get("Ny", 50)), 2)
    default_coarse_x = Lx / max(approx_Nx - 1, 1)
    default_coarse_y = Ly / max(approx_Ny - 1, 1)
    default_fine_x = default_coarse_x / 3.0
    default_fine_y = default_coarse_y / 3.0

    coarse_x = _axis_spacing(mesh_spec, "x",
                             key_candidates=["coarse", "coarse_dx", "max_dx", "dx_far"],
                             default=default_coarse_x)
    coarse_y = _axis_spacing(mesh_spec, "y",
                             key_candidates=["coarse", "coarse_dy", "max_dy", "dy_far"],
                             default=default_coarse_y)
    if coarse_x <= 0 or coarse_y <= 0:
        raise ValueError("point-cloud mesh requires positive coarse spacings")

    fine_x = _axis_spacing(mesh_spec, "x",
                           key_candidates=["fine", "fine_dx", "min_dx", "dx_near"],
                           default=default_fine_x)
    fine_y = _axis_spacing(mesh_spec, "y",
                           key_candidates=["fine", "fine_dy", "min_dy", "dy_near"],
                           default=default_fine_y)
    fine_x = max(min(fine_x, coarse_x), 1e-6)
    fine_y = max(min(fine_y, coarse_y), 1e-6)

    default_pad = 0.01 if is_experimental else 0.0
    pad = float(mesh_spec.get("focus_pad", mesh_spec.get("focusPad", default_pad)))
    if is_experimental and pad <= 0.0:
        pad = 0.01
    focus_materials = mesh_spec.get("focus_materials")
    if focus_materials is None:
        focus_materials = ["magnet", "steel", "wire"]
    elif isinstance(focus_materials, str):
        focus_materials = [focus_materials]
    elif isinstance(focus_materials, (set, tuple)):
        focus_materials = list(focus_materials)
    focus_materials = [str(m).lower() for m in focus_materials]
    manual_boxes = mesh_spec.get("focus_boxes")
    if not isinstance(manual_boxes, list):
        manual_boxes = None
    focus_regions = _collect_focus_regions(
        definition,
        Lx=Lx,
        Ly=Ly,
        materials=focus_materials,
        pad=pad,
        manual_boxes=manual_boxes,
    )
    field_focus_defaults = field_focus_defaults or {}
    focus_overrides = field_focus_spec if isinstance(field_focus_spec, dict) else {}
    field_focus_cfg = {
        "enabled": True,
        "direction_weight": 1.0,
        "magnitude_weight": 1.0,
        "indicator_gain": 0.4,  # exponent on indicator -> size mapping
        "indicator_neutral": None,
        "indicator_percentile": 85.0,
        "scale_min": 0.5,
        "scale_max": 2.0,
        "smooth_passes": 2,
        "size_smooth_passes": 2,
        "indicator_clip": [5.0, 95.0],
        "ratio_limit": 1.7,
    }
    field_focus_cfg.update(field_focus_defaults)
    if focus_overrides:
        field_focus_cfg.update(focus_overrides)
    field_focus_params = {
        "enabled": bool(field_focus_cfg.get("enabled", True)),
        "direction_weight": float(field_focus_cfg.get("direction_weight", 1.0)),
        "magnitude_weight": float(field_focus_cfg.get("magnitude_weight", 1.0)),
        "indicator_gain": float(field_focus_cfg.get("indicator_gain", 0.4)),
        "indicator_neutral": field_focus_cfg.get("indicator_neutral"),
        "indicator_percentile": float(field_focus_cfg.get("indicator_percentile", 85.0)),
        "scale_min": float(field_focus_cfg.get("scale_min", 0.5)),
        "scale_max": float(field_focus_cfg.get("scale_max", 2.0)),
    }
    field_focus_scaling = None
    field_focus_meta = None
    field_focus_size = None
    field_focus_alpha = None
    experimental_meta = None

    base_nx = max(int(math.ceil(Lx / coarse_x)) + 1, 2)
    base_ny = max(int(math.ceil(Ly / coarse_y)) + 1, 2)
    if is_experimental:
        band_ref = max(pad, 0.01)
        target_x = max(min(fine_x, band_ref), 1e-4)
        target_y = max(min(fine_y, band_ref), 1e-4)
        base_nx = max(base_nx, int(math.ceil(Lx / target_x)) + 1)
        base_ny = max(base_ny, int(math.ceil(Ly / target_y)) + 1)
    if adapt_from and field_focus_cfg.get("enabled", True):
        try:
            neutral_value = field_focus_cfg.get("indicator_neutral")
            try:
                neutral_value = float(neutral_value)
            except (TypeError, ValueError):
                neutral_value = None
            clip_cfg = field_focus_cfg.get("indicator_clip", (5.0, 95.0))
            if isinstance(clip_cfg, (list, tuple)) and len(clip_cfg) >= 2:
                clip_tuple = (float(clip_cfg[0]), float(clip_cfg[1]))
            else:
                clip_tuple = (5.0, 95.0)
            size_min_override = field_focus_cfg.get("size_min")
            size_max_override = field_focus_cfg.get("size_max")
            scale_min = float(field_focus_cfg.get("scale_min", 0.5))
            scale_max = float(field_focus_cfg.get("scale_max", 2.0))
            size_min_val = (
                float(size_min_override)
                if size_min_override is not None
                else scale_min * min(coarse_x, coarse_y)
            )
            size_max_val = (
                float(size_max_override)
                if size_max_override is not None
                else scale_max * max(coarse_x, coarse_y)
            )
            size_min_val = max(size_min_val, 1e-6)
            size_max_val = max(size_max_val, size_min_val)
            alpha = float(field_focus_cfg.get("indicator_gain", 0.4))
            ratio_limit = float(field_focus_cfg.get("ratio_limit", 1.7))
            smooth_passes = int(field_focus_cfg.get("smooth_passes", 2))
            size_smooth_passes = int(field_focus_cfg.get("size_smooth_passes", smooth_passes))
            field_focus_scaling, field_focus_meta, field_focus_size, field_focus_alpha = _field_density_scaling_from_solution(
                adapt_from,
                Lx=Lx,
                Ly=Ly,
                ncols=base_nx - 1,
                nrows=base_ny - 1,
                coarse_x=coarse_x,
                coarse_y=coarse_y,
                magnitude_weight=float(field_focus_cfg.get("magnitude_weight", 1.0)),
                direction_weight=float(field_focus_cfg.get("direction_weight", 1.0)),
                size_min=size_min_val,
                size_max=size_max_val,
                alpha=alpha,
                neutral_indicator=neutral_value,
                indicator_percentile=field_focus_cfg.get("indicator_percentile", 85.0),
                clip_percentiles=clip_tuple,
                smooth_passes=smooth_passes,
                ratio_limit=ratio_limit,
                scale_min=scale_min,
                scale_max=scale_max,
                size_smooth_passes=size_smooth_passes,
                size_min_override=size_min_override is not None,
                size_max_override=size_max_override is not None,
            )
        except FieldAdaptError as exc:
            raise SystemExit(f"Field-driven focus extraction failed: {exc}") from exc

    x_base = _axis_from_spacing(Lx, coarse_x, base_nx, field_focus_scaling.get("x") if field_focus_scaling else None)
    y_base = _axis_from_spacing(Ly, coarse_y, base_ny, field_focus_scaling.get("y") if field_focus_scaling else None)
    Xb, Yb = np.meshgrid(x_base, y_base, indexing="ij")
    base_points = np.column_stack([Xb.ravel(), Yb.ravel()])
    base_keep = np.ones(base_points.shape[0], dtype=bool)
    point_chunks: list[np.ndarray] = []
    focus_point_raw = 0
    base_removed = 0
    size_field_added = 0
    size_driven_points = None
    boundary_points: list[np.ndarray] = []

    if is_experimental and field_focus_size is None:
        falloff_val = mesh_spec.get("focus_falloff", mesh_spec.get("focusFalloff"))
        if falloff_val is None:
            falloff_val = max(0.5 * pad, 0.002 * max(Lx, Ly))
        falloff_val = max(float(falloff_val), 0.0)
        fine_target = min(fine_x, fine_y)
        coarse_target = max(coarse_x, coarse_y)
        field_focus_size, experimental_meta = _material_proximity_size_grid(
            Lx,
            Ly,
            x_grid=x_base,
            y_grid=y_base,
            regions=focus_regions,
            fine=fine_target,
            coarse=coarse_target,
            band=pad,
            falloff=falloff_val if falloff_val > 0.0 else max(pad * 0.5, 1e-3),
            smooth_passes=2,
            ratio_limit=1.35,
        )

    if field_focus_size is not None and x_base.size >= 2 and y_base.size >= 2:
        ncols_sf, nrows_sf = field_focus_size.shape
        if ncols_sf != x_base.size - 1 or nrows_sf != y_base.size - 1:
            # Best-effort crop to overlapping region (shapes should normally match).
            ncols_sf = min(ncols_sf, x_base.size - 1)
            nrows_sf = min(nrows_sf, y_base.size - 1)
            field_focus_size = field_focus_size[:ncols_sf, :nrows_sf]
            if field_focus_size.size == 0:
                field_focus_size = None
        if field_focus_size is not None:
            # Use Poisson sampling driven by the target size field to keep triangles
            # closer to equilateral while honoring local alpha (refinement and coarsening).
            focus_size_cap = min(fine_x, fine_y)
            # Coarsen the size grid to reduce overhead; target ~120 cells on the long side.
            max_dim = max(field_focus_size.shape)
            coarsen = max(1, min(4, max_dim // 120))
            if coarsen > 1:
                field_focus_size = _coarsen_grid_mean(field_focus_size, coarsen)
                x_base = x_base[::coarsen] if x_base.size > coarsen else x_base
                y_base = y_base[::coarsen] if y_base.size > coarsen else y_base
                if x_base[-1] != Lx:
                    x_base = np.append(x_base, Lx)
                if y_base[-1] != Ly:
                    y_base = np.append(y_base, Ly)
            # Extra smoothing and ratio clamp on the size grid to soften transitions.
            field_focus_size = _smooth_grid_box(field_focus_size, passes=1)
            field_focus_size = _limit_neighbor_ratio(field_focus_size, ratio=1.35, passes=2)
            # Respect material focus regions by capping size within those boxes.
            if focus_regions:
                field_focus_size = _apply_focus_caps_to_size_grid(
                    field_focus_size,
                    x_base,
                    y_base,
                    focus_regions,
                    focus_size_cap,
                )
            # Fast fill: deterministic hex lattice per size cell (no rejection).
            size_driven_points = _hex_fill_from_size_grid(
                Lx, Ly,
                x_grid=x_base,
                y_grid=y_base,
                size_grid=field_focus_size,
                jitter=0.05,
            )
            size_driven_points = _prune_close_points(
                size_driven_points,
                x_grid=x_base,
                y_grid=y_base,
                size_grid=field_focus_size,
                tol=0.9,
            )
            # Sprinkle boundary points so the hull is well represented.
            boundary_spacing = max(min(field_focus_size.min() * 1.5, focus_size_cap * 1.5), 1e-6)
            xs_edge = np.arange(0.0, Lx + boundary_spacing * 0.5, boundary_spacing)
            ys_edge = np.arange(0.0, Ly + boundary_spacing * 0.5, boundary_spacing)
            left = np.column_stack([np.zeros_like(ys_edge), ys_edge])
            right = np.column_stack([np.full_like(ys_edge, Lx), ys_edge])
            bottom = np.column_stack([xs_edge, np.zeros_like(xs_edge)])
            top = np.column_stack([xs_edge, np.full_like(xs_edge, Ly)])
            boundary_points = [left, right, bottom, top]

    for (xmin, xmax, ymin, ymax) in focus_regions:
        width = xmax - xmin
        height = ymax - ymin
        if width <= 0 or height <= 0:
            continue
        in_region = (
            (base_points[:, 0] >= xmin)
            & (base_points[:, 0] <= xmax)
            & (base_points[:, 1] >= ymin)
            & (base_points[:, 1] <= ymax)
        )
        removed_here = int(np.count_nonzero(in_region & base_keep))
        if removed_here:
            base_keep[in_region] = False
            base_removed += removed_here
        nx = max(int(math.ceil(width / fine_x)) + 1, 2)
        ny = max(int(math.ceil(height / fine_y)) + 1, 2)
        x_local = np.linspace(xmin, xmax, nx)
        y_local = np.linspace(ymin, ymax, ny)
        Xf, Yf = np.meshgrid(x_local, y_local, indexing="ij")
        fine_points = np.column_stack([Xf.ravel(), Yf.ravel()])
        point_chunks.append(fine_points)
        focus_point_raw += fine_points.shape[0]

    if not point_chunks:
        point_chunks.append(np.empty((0, 2), dtype=float))
    kept_base = base_points[base_keep]
    if size_driven_points is not None:
        points = _unique_rows(np.vstack([*boundary_points, size_driven_points]))
        size_field_added = points.shape[0]
        base_removed = int(base_points.shape[0])  # treat base as fully replaced
    else:
        points = _unique_rows(np.vstack([kept_base, *point_chunks]))
    if points.shape[0] < 3:
        raise ValueError("Point-cloud mesher produced fewer than 3 unique points")

    min_angle = float(mesh_spec.get("quality_min_angle", 28.0))
    quality_nodes = quality_tris = None
    quality_used = False
    # If we're building a preview-only mesh and the point cloud is huge, skip Triangle quality meshing.
    point_count = int(points.shape[0])
    max_quality_points = int(mesh_spec.get("quality_point_limit", 150_000))
    allow_quality = _triangle_lib is not None and min_angle > 0.0
    if mesh_only and point_count > max_quality_points:
        allow_quality = False

    if allow_quality:
        quality_nodes, quality_tris = _quality_triangulation(points, min_angle=min_angle)
        quality_used = quality_nodes is not None and quality_tris is not None

    if quality_used:
        nodes = quality_nodes
        tris = quality_tris
    else:
        nodes = points
        tri = Delaunay(points)
        tris = np.asarray(tri.simplices, dtype=np.int32)
        tris = _orient_tris_ccw(nodes, tris)
        if tris.size == 0:
            raise ValueError("Point-cloud mesher produced only degenerate triangles")

    generator = "point_cloud_hex" if size_driven_points is not None else "point_cloud"
    mesh_meta = {
        "type": mesh_type,
        "generator": f"{generator}_triangle" if quality_used else f"{generator}_delaunay",
        "coarse_spacing": {"x": float(coarse_x), "y": float(coarse_y)},
        "fine_spacing": {"x": float(fine_x), "y": float(fine_y)},
        "point_counts": {
            "base": int(base_points.shape[0]),
            "base_removed": int(base_removed),
            "focus_raw": int(focus_point_raw),
            "size_field": int(size_field_added),
            "unique": int(points.shape[0]),
        },
        "focus_pad": float(pad),
        "focus_materials": focus_materials,
        "focus_regions": [
            [float(xmin), float(xmax), float(ymin), float(ymax)]
            for (xmin, xmax, ymin, ymax) in focus_regions
        ],
    }
    if field_focus_params:
        mesh_meta["field_focus_params"] = field_focus_params
    if manual_boxes:
        mesh_meta["focus_boxes"] = manual_boxes
    if field_focus_meta:
        mesh_meta["field_focus"] = field_focus_meta
        if field_focus_alpha is not None:
            mesh_meta["field_focus"]["alpha_stats"] = {
                "min": float(np.min(field_focus_alpha)),
                "max": float(np.max(field_focus_alpha)),
                "median": float(np.median(field_focus_alpha)),
            }
    if quality_used:
        mesh_meta["quality_mesher"] = {
            "backend": "triangle",
            "min_angle": min_angle,
        }
    else:
        mesh_meta["quality_mesher"] = {
            "backend": "scipy_delaunay",
            "quality_skipped": bool(mesh_only and point_count > max_quality_points),
        }
    if size_driven_points is not None and field_focus_size is not None:
        mesh_meta["field_fill"] = {
            "size_min": float(np.nanmin(field_focus_size)),
            "size_max": float(np.nanmax(field_focus_size)),
            "boundary_points": int(sum(bp.shape[0] for bp in boundary_points)) if boundary_points else 0,
            "grid_coarsen": int(max(1, coarsen if "coarsen" in locals() else 1)),
            "method": "hex_lattice",
        }
    if experimental_meta:
        mesh_meta["experimental"] = experimental_meta
    return nodes, tris, (Lx, Ly), mesh_meta


def _triangle_neighbors(tris: np.ndarray) -> list[list[int]]:
    neighbors: list[list[int]] = [[] for _ in range(tris.shape[0])]
    edges: dict[tuple[int, int], int] = {}
    for idx, tri in enumerate(tris):
        for corner in range(3):
            a = int(tri[corner])
            b = int(tri[(corner + 1) % 3])
            if a > b:
                a, b = b, a
            key = (a, b)
            other = edges.get(key)
            if other is None:
                edges[key] = idx
            else:
                neighbors[idx].append(other)
                neighbors[other].append(idx)
    return neighbors


def _angle_diff(rad_a: float, rad_b: float) -> float:
    diff = rad_a - rad_b
    return abs(math.atan2(math.sin(diff), math.cos(diff)))


def _triangle_mean_edge_length(nodes: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Characteristic triangle size: mean edge length per element."""
    if nodes.ndim != 2 or nodes.shape[1] != 2 or tris.ndim != 2 or tris.shape[1] != 3:
        return np.array([], dtype=float)
    pts = nodes[tris]
    edges = np.linalg.norm(pts[:, [1, 2, 0], :] - pts[:, [0, 1, 2], :], axis=2)
    return edges.mean(axis=1)


def _field_gradient_components(Bmag: np.ndarray,
                               Bx: np.ndarray,
                               By: np.ndarray,
                               centroids: np.ndarray,
                               neighbors: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    Ne = Bmag.shape[0]
    mag_comp = np.zeros(Ne, dtype=float)
    dir_comp = np.zeros(Ne, dtype=float)
    angles = np.arctan2(By, Bx)
    for idx in range(Ne):
        c = centroids[idx]
        best_mag = 0.0
        best_dir = 0.0
        for nb in neighbors[idx]:
            dist = float(np.hypot(c[0] - centroids[nb, 0], c[1] - centroids[nb, 1]))
            if dist <= 1e-12:
                dist = 1e-12
            best_mag = max(best_mag, abs(Bmag[idx] - Bmag[nb]) / dist)
            angle_change = _angle_diff(angles[idx], angles[nb]) / dist
            if math.isfinite(angle_change):
                best_dir = max(best_dir, angle_change)
        mag_comp[idx] = best_mag
        dir_comp[idx] = best_dir
    return mag_comp, dir_comp


def _smooth_1d(values: np.ndarray, passes: int) -> np.ndarray:
    if passes <= 0 or values.size == 0:
        return values
    kernel = np.array([0.25, 0.5, 0.25], dtype=float)
    out = values.astype(float)
    for _ in range(passes):
        padded = np.pad(out, (1, 1), mode="edge")
        out = kernel[0] * padded[:-2] + kernel[1] * padded[1:-1] + kernel[2] * padded[2:]
    return out


def _smooth_2d(values: np.ndarray, passes: int, *, log_space: bool = False) -> np.ndarray:
    if passes <= 0 or values.size == 0:
        return values
    arr = values.astype(float)
    if log_space:
        arr = np.log(np.clip(arr, 1e-16, None))
    kernel = np.array([[0.05, 0.1, 0.05],
                       [0.1,  0.4, 0.1 ],
                       [0.05, 0.1, 0.05]], dtype=float)
    for _ in range(passes):
        padded = np.pad(arr, 1, mode="edge")
        window = (
            kernel[0, 0] * padded[:-2, :-2] + kernel[0, 1] * padded[:-2, 1:-1] + kernel[0, 2] * padded[:-2, 2:] +
            kernel[1, 0] * padded[1:-1, :-2] + kernel[1, 1] * padded[1:-1, 1:-1] + kernel[1, 2] * padded[1:-1, 2:] +
            kernel[2, 0] * padded[2:, :-2] + kernel[2, 1] * padded[2:, 1:-1] + kernel[2, 2] * padded[2:, 2:]
        )
        arr = window
    if log_space:
        arr = np.exp(arr)
    return arr


def _limit_neighbor_ratio(field: np.ndarray, *, ratio: float, passes: int = 2) -> np.ndarray:
    """Clamp neighboring entries so their ratio never exceeds `ratio`."""
    if ratio <= 1.0 or field.size == 0:
        return field
    out = field.astype(float)
    r = float(ratio)
    for _ in range(max(1, passes)):
        # Sweep rows
        left = out[:, :-1]
        right = out[:, 1:]
        too_big = right > left * r
        right[too_big] = left[too_big] * r
        too_small = right * r < left
        left[too_small] = right[too_small] * r
        # Sweep cols
        top = out[:-1, :]
        bot = out[1:, :]
        too_big = bot > top * r
        bot[too_big] = top[too_big] * r
        too_small = bot * r < top
        top[too_small] = bot[too_small] * r
    return out


def _field_density_scaling_from_solution(field_path: str | Path,
                                         *,
                                         Lx: float,
                                         Ly: float,
                                         ncols: int,
                                         nrows: int,
                                         coarse_x: float,
                                         coarse_y: float,
                                         magnitude_weight: float,
                                         direction_weight: float,
                                         size_min: float,
                                         size_max: float,
    alpha: float,
    neutral_indicator: float | None,
    indicator_percentile: float | None,
    clip_percentiles: tuple[float, float],
    smooth_passes: int,
                                         ratio_limit: float,
                                         scale_min: float,
                                         scale_max: float,
                                         size_smooth_passes: int = 1,
                                         size_min_override: bool = False,
                                         size_max_override: bool = False) -> tuple[dict[str, np.ndarray], dict, np.ndarray, np.ndarray]:
    path = Path(field_path).expanduser().resolve()
    if not path.exists():
        raise FieldAdaptError(f"adapt-from file '{path}' was not found")
    mag_pre = dir_pre = combined_pre = alpha_pre = None
    with np.load(path, allow_pickle=True) as payload:
        try:
            nodes = np.asarray(payload["nodes"], dtype=float)
            tris = np.asarray(payload["tris"], dtype=np.int32)
            Bmag = np.asarray(payload["Bmag"], dtype=float)
            Bx = np.asarray(payload["Bx"], dtype=float)
            By = np.asarray(payload["By"], dtype=float)
        except KeyError as exc:
            raise FieldAdaptError(f"{path} is missing nodes/tris/B-field arrays") from exc
        mag_pre = payload.get("indicator_magnitude")
        if mag_pre is not None:
            mag_pre = np.asarray(mag_pre, dtype=float)
        dir_pre = payload.get("indicator_direction")
        if dir_pre is not None:
            dir_pre = np.asarray(dir_pre, dtype=float)
        combined_pre = payload.get("indicator_combined")
        if combined_pre is not None:
            combined_pre = np.asarray(combined_pre, dtype=float)
        alpha_pre = payload.get("alpha")
        if alpha_pre is not None:
            alpha_pre = np.asarray(alpha_pre, dtype=float)
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise FieldAdaptError("B-field mesh must contain triangles")
    if Bmag.size != tris.shape[0]:
        raise FieldAdaptError("Bmag must be per triangle for field-driven refinement")
    tri_size = _triangle_mean_edge_length(nodes, tris)
    centroids = nodes[tris].mean(axis=1)
    use_precomputed = (
        mag_pre is not None
        and dir_pre is not None
        and mag_pre.size == tris.shape[0]
        and dir_pre.size == tris.shape[0]
    )
    if use_precomputed:
        mag_comp = np.asarray(mag_pre, dtype=float)
        dir_comp = np.asarray(dir_pre, dtype=float)
        indicator = magnitude_weight * mag_comp + direction_weight * dir_comp
    else:
        neighbors = _triangle_neighbors(tris)
        mag_comp, dir_comp = _field_gradient_components(Bmag, Bx, By, centroids, neighbors)
        indicator = magnitude_weight * mag_comp + direction_weight * dir_comp
    finite_indicator = indicator[np.isfinite(indicator)]
    if finite_indicator.size == 0:
        raise FieldAdaptError("Field indicator is degenerate (no finite entries)")
    clip_lo, clip_hi = clip_percentiles
    clip_lo = max(0.0, float(clip_lo))
    clip_hi = min(100.0, float(clip_hi))
    if clip_hi < clip_lo:
        clip_hi = clip_lo
    if finite_indicator.size and (clip_lo > 0 or clip_hi < 100):
        lo = np.percentile(finite_indicator, clip_lo)
        hi = np.percentile(finite_indicator, clip_hi)
        indicator = np.clip(indicator, lo, hi)
        finite_indicator = indicator[np.isfinite(indicator)]
    if finite_indicator.size == 0:
        raise FieldAdaptError("Field indicator is degenerate after clipping")

    # Bin indicator onto a coarse lattice using max within each bin.
    grid = np.full((max(ncols, 1), max(nrows, 1)), np.nan, dtype=float)
    xi = np.clip(np.floor(centroids[:, 0] / max(Lx, 1e-12) * grid.shape[0]).astype(int), 0, grid.shape[0] - 1)
    yi = np.clip(np.floor(centroids[:, 1] / max(Ly, 1e-12) * grid.shape[1]).astype(int), 0, grid.shape[1] - 1)
    np.maximum.at(grid, (xi, yi), indicator)
    fill_value = float(np.median(finite_indicator))
    grid = np.where(np.isfinite(grid), grid, fill_value)
    base_spacing_grid = None
    spacing_samples = tri_size[np.isfinite(tri_size) & (tri_size > 0.0)]
    base_spacing_ref = min(coarse_x, coarse_y)
    spacing_lo = base_spacing_ref
    spacing_hi = base_spacing_ref
    spacing_min = base_spacing_ref
    if spacing_samples.size:
        base_spacing_ref = float(np.median(spacing_samples))
        spacing_lo = float(np.percentile(spacing_samples, 5.0))
        spacing_hi = float(np.percentile(spacing_samples, 90.0))
        spacing_min = float(np.percentile(spacing_samples, 1.0))
        spacing_lo = max(spacing_lo, 1e-12)
        spacing_min = max(spacing_min, 1e-12)
        spacing_hi = max(spacing_hi, spacing_lo)
    else:
        spacing_min = base_spacing_ref
    if tri_size.size == indicator.size:
        spacing_min_grid = np.full_like(grid, np.inf, dtype=float)
        spacing_count = np.zeros_like(grid)
        np.minimum.at(spacing_min_grid, (xi, yi), tri_size)
        np.add.at(spacing_count, (xi, yi), 1.0)
        spacing_min_grid = np.where(spacing_count > 0, spacing_min_grid, np.inf)
        finite_spacing = spacing_min_grid[np.isfinite(spacing_min_grid) & (spacing_min_grid < np.inf)]
        if finite_spacing.size:
            filler = float(np.median(finite_spacing))
            base_spacing_grid = np.where(np.isfinite(spacing_min_grid), spacing_min_grid, filler)
        else:
            base_spacing_grid = None

    alpha_grid_pre = None
    if alpha_pre is not None and alpha_pre.size == indicator.size:
        alpha_sum = np.zeros_like(grid)
        alpha_count = np.zeros_like(grid)
        alpha_flat = np.asarray(alpha_pre, dtype=float)
        np.add.at(alpha_sum, (xi, yi), alpha_flat)
        np.add.at(alpha_count, (xi, yi), 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha_grid_pre = alpha_sum / np.where(alpha_count > 0, alpha_count, 1.0)
        finite_alpha = alpha_grid_pre[np.isfinite(alpha_grid_pre)]
        if finite_alpha.size:
            fill_alpha = float(np.median(finite_alpha))
            alpha_grid_pre = np.where(np.isfinite(alpha_grid_pre), alpha_grid_pre, fill_alpha)
            alpha_grid_pre = np.clip(alpha_grid_pre, scale_min, scale_max)
        else:
            alpha_grid_pre = None

    # Smooth indicator a bit to avoid single-element spikes.
    indicator_grid = _smooth_2d(grid, max(int(smooth_passes), 0))
    pct = 85.0 if indicator_percentile is None else float(indicator_percentile)
    if not math.isfinite(pct):
        pct = 85.0
    pct = min(max(pct, 0.0), 100.0)
    if neutral_indicator is not None and math.isfinite(neutral_indicator) and neutral_indicator > 0:
        ref = float(neutral_indicator)
    else:
        ref = float(np.percentile(indicator_grid, pct)) if indicator_grid.size else fill_value
    ref = ref if math.isfinite(ref) and ref > 0 else fill_value if fill_value > 0 else 1.0
    base_spacing = max(float(base_spacing_ref), 1e-12)
    size_min = (
        max(float(size_min), 1e-6)
        if size_min_override is not None
        else max(scale_min * spacing_min, 1e-6)
    )
    size_max = (
        max(float(size_max), size_min)
        if size_max_override is not None
        else max(scale_max * spacing_hi, size_min)
    )
    alpha = max(float(alpha), 0.0)
    ratio_limit = max(float(ratio_limit), 1.0)
    size_smooth_passes = max(int(size_smooth_passes), 0)
    scale_min = max(float(scale_min), 1e-3)
    scale_max = max(float(scale_max), scale_min)
    if base_spacing_grid is None and indicator_grid.size:
        base_spacing_grid = np.full_like(indicator_grid, base_spacing)
    elif base_spacing_grid is not None:
        base_spacing_grid = np.where(np.isfinite(base_spacing_grid), base_spacing_grid, base_spacing)

    # Map indicator -> target size (work in log space for smoother variation).
    base_map = base_spacing_grid if base_spacing_grid is not None else np.full_like(indicator_grid, base_spacing)
    size_grid = base_map * (indicator_grid / ref) ** (-alpha)
    size_grid = np.clip(size_grid, size_min, size_max)
    if alpha_grid_pre is not None:
        size_grid = np.clip(base_map * alpha_grid_pre, size_min, size_max)
    if size_smooth_passes:
        size_grid = _smooth_2d(size_grid, size_smooth_passes, log_space=True)
    if ratio_limit > 1.0:
        size_grid = _limit_neighbor_ratio(size_grid, ratio=ratio_limit, passes=2)
    size_grid = np.clip(size_grid, size_min, size_max)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_grid = size_grid / np.clip(base_map, 1e-12, None)
    alpha_grid = np.clip(alpha_grid, scale_min, scale_max)

    col_target = np.median(size_grid, axis=1) if size_grid.size else np.array([])
    row_target = np.median(size_grid, axis=0) if size_grid.size else np.array([])
    x_scales = np.array([])
    y_scales = np.array([])
    if col_target.size:
        x_scales = np.clip(col_target / max(coarse_x, 1e-12), scale_min, scale_max)
    if row_target.size:
        y_scales = np.clip(row_target / max(coarse_y, 1e-12), scale_min, scale_max)

    meta = {
        "source_field": str(path),
        "indicator": "B_gradient_directional",
        "magnitude_weight": float(magnitude_weight),
        "direction_weight": float(direction_weight),
        "indicator_neutral": float(neutral_indicator) if neutral_indicator is not None else None,
        "indicator_percentile": float(pct),
        "indicator_stats": {
            "min": float(np.min(indicator_grid)),
            "max": float(np.max(indicator_grid)),
            "median": float(np.median(indicator_grid)),
            "ref_percentile": float(pct),
            "ref_value": float(ref),
        },
        "size_map": {
            "min": float(np.min(size_grid)),
            "max": float(np.max(size_grid)),
            "median": float(np.median(size_grid)),
            "size_min": size_min,
            "size_max": size_max,
            "base_spacing": base_spacing,
            "base_spacing_source": "mesh_min" if spacing_samples.size else "coarse_spacing",
            "base_spacing_percentiles": [spacing_min, base_spacing_ref, spacing_hi],
            "alpha": alpha,
            "smooth_passes": size_smooth_passes,
            "ratio_limit": ratio_limit,
            "clip_percentiles": [float(clip_lo), float(clip_hi)],
        },
        "alpha_stats": {
            "min": float(np.min(alpha_grid)),
            "max": float(np.max(alpha_grid)),
            "median": float(np.median(alpha_grid)),
        },
        "alpha_source": "B_field_alpha" if alpha_grid_pre is not None else "indicator_mapping",
        "axis_scaling": {
            "x": {
                "samples": int(x_scales.size),
                "min": float(x_scales.min()) if x_scales.size else 1.0,
                "max": float(x_scales.max()) if x_scales.size else 1.0,
            },
            "y": {
                "samples": int(y_scales.size),
                "min": float(y_scales.min()) if y_scales.size else 1.0,
                "max": float(y_scales.max()) if y_scales.size else 1.0,
            },
            "scale_min": float(scale_min),
            "scale_max": float(scale_max),
        },
    }
    if spacing_samples.size:
        meta["size_map"]["base_spacing_stats"] = {
            "min": float(np.min(spacing_samples)),
            "max": float(np.max(spacing_samples)),
            "median": float(np.median(spacing_samples)),
        }
    return {"x": x_scales, "y": y_scales}, meta, size_grid, alpha_grid


def _axis_from_spacing(total_length: float,
                       coarse_spacing: float,
                       nodes: int,
                       scaling: np.ndarray | None) -> np.ndarray:
    nodes = max(int(nodes), 2)
    intervals = np.full(nodes - 1, float(coarse_spacing), dtype=float)
    if scaling is not None and intervals.size:
        scale_arr = np.asarray(scaling, dtype=float)
        if scale_arr.size != intervals.size:
            if scale_arr.size == 0:
                scale_arr = np.ones_like(intervals)
            else:
                old_param = np.linspace(0.0, 1.0, scale_arr.size)
                new_param = np.linspace(0.0, 1.0, intervals.size)
                scale_arr = np.interp(new_param, old_param, scale_arr)
        intervals = intervals * scale_arr
    total = intervals.sum()
    if not np.isfinite(total) or total <= 0:
        intervals[:] = total_length / max(intervals.size, 1)
        total = intervals.sum()
    if total <= 0:
        return np.linspace(0.0, total_length, nodes)
    intervals *= total_length / total
    coords = np.concatenate([[0.0], np.cumsum(intervals)])
    coords[-1] = total_length
    return coords


def _size_at_point(x: float,
                   y: float,
                   x_grid: np.ndarray,
                   y_grid: np.ndarray,
                   size_grid: np.ndarray,
                   *,
                   focus_regions: list[tuple[float, float, float, float]] | None = None,
                   focus_size_cap: float | None = None) -> float:
    """Piecewise-constant lookup of the target size for a point."""
    xi = int(np.clip(np.searchsorted(x_grid, x, side="right") - 1, 0, size_grid.shape[0] - 1))
    yi = int(np.clip(np.searchsorted(y_grid, y, side="right") - 1, 0, size_grid.shape[1] - 1))
    base = float(size_grid[xi, yi])
    if focus_regions and focus_size_cap is not None:
        for (xmin, xmax, ymin, ymax) in focus_regions:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return min(base, focus_size_cap)
    return base


def _apply_focus_caps_to_size_grid(size_grid: np.ndarray,
                                   x_grid: np.ndarray,
                                   y_grid: np.ndarray,
                                   regions: list[tuple[float, float, float, float]],
                                   cap: float | None) -> np.ndarray:
    """Clamp size grid inside focus regions to a maximum of `cap`."""
    if cap is None or not regions or size_grid.size == 0:
        return size_grid
    grid = size_grid.copy()
    cx = 0.5 * (x_grid[:-1] + x_grid[1:])
    cy = 0.5 * (y_grid[:-1] + y_grid[1:])
    for (xmin, xmax, ymin, ymax) in regions:
        if xmax <= xmin or ymax <= ymin:
            continue
        mask_x = (cx >= xmin) & (cx <= xmax)
        mask_y = (cy >= ymin) & (cy <= ymax)
        if not mask_x.any() or not mask_y.any():
            continue
        region_mask = np.ix_(mask_x, mask_y)
        grid[region_mask] = np.minimum(grid[region_mask], cap)
    return grid


def _material_proximity_size_grid(Lx: float,
                                  Ly: float,
                                  *,
                                  x_grid: np.ndarray,
                                  y_grid: np.ndarray,
                                  regions: list[tuple[float, float, float, float]],
                                  fine: float,
                                  coarse: float,
                                  band: float,
                                  falloff: float,
                                  smooth_passes: int = 1,
                                  ratio_limit: float = 1.35) -> tuple[np.ndarray, dict]:
    """Size field that grades from fine near materials to coarse elsewhere."""
    if x_grid.size < 2 or y_grid.size < 2:
        return np.zeros((0, 0), dtype=float), {}
    fine = max(float(fine), 1e-6)
    coarse = max(float(coarse), fine)
    band = max(float(band), 0.0)
    falloff = max(float(falloff), max(0.5 * band, 1e-6))
    cx = 0.5 * (x_grid[:-1] + x_grid[1:])
    cy = 0.5 * (y_grid[:-1] + y_grid[1:])
    Xc, Yc = np.meshgrid(cx, cy, indexing="ij")
    grid = np.full_like(Xc, coarse, dtype=float)
    applied = 0
    for (xmin, xmax, ymin, ymax) in regions:
        if xmax <= xmin or ymax <= ymin:
            continue
        applied += 1
        dx = np.where(Xc < xmin, xmin - Xc, np.where(Xc > xmax, Xc - xmax, 0.0))
        dy = np.where(Yc < ymin, ymin - Yc, np.where(Yc > ymax, Yc - ymax, 0.0))
        dist = np.hypot(dx, dy)
        influence = np.exp(-0.5 * (dist / falloff) ** 2)
        candidate = coarse - (coarse - fine) * influence
        if band > 0.0:
            candidate = np.where(dist <= band, fine, candidate)
        grid = np.minimum(grid, candidate)
    if smooth_passes:
        grid = _smooth_grid_box(grid, passes=max(int(smooth_passes), 0))
    if ratio_limit > 1.0:
        grid = _limit_neighbor_ratio(grid, ratio=max(float(ratio_limit), 1.0), passes=2)
    grid = np.clip(grid, fine, coarse)
    meta = {
        "strategy": "material_band",
        "regions": int(applied),
        "band": band,
        "falloff": falloff,
        "size_stats": {
            "min": float(np.nanmin(grid)),
            "max": float(np.nanmax(grid)),
            "median": float(np.nanmedian(grid)),
        },
        "grid": [int(grid.shape[0]), int(grid.shape[1])],
        "fine": fine,
        "coarse": coarse,
    }
    return grid, meta


def _coarsen_grid_mean(grid: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 2-D grid by averaging non-overlapping tiles."""
    if factor <= 1:
        return grid
    ncols, nrows = grid.shape
    c = max(1, int(factor))
    new_cols = max(1, ncols // c)
    new_rows = max(1, nrows // c)
    trimmed = grid[: new_cols * c, : new_rows * c]
    reshaped = trimmed.reshape(new_cols, c, new_rows, c)
    return np.nanmean(reshaped, axis=(1, 3))


def _smooth_grid_box(grid: np.ndarray, passes: int = 1) -> np.ndarray:
    """Apply a small box-blur kernel to soften a 2-D grid."""
    if passes <= 0 or grid.size == 0:
        return grid
    arr = grid.astype(float)
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=float)
    kernel /= kernel.sum()
    for _ in range(passes):
        padded = np.pad(arr, 1, mode="edge")
        window = (
            kernel[0, 0] * padded[:-2, :-2] + kernel[0, 1] * padded[:-2, 1:-1] + kernel[0, 2] * padded[:-2, 2:] +
            kernel[1, 0] * padded[1:-1, :-2] + kernel[1, 1] * padded[1:-1, 1:-1] + kernel[1, 2] * padded[1:-1, 2:] +
            kernel[2, 0] * padded[2:, :-2] + kernel[2, 1] * padded[2:, 1:-1] + kernel[2, 2] * padded[2:, 2:]
        )
        arr = window
    return arr


def _hex_fill_from_size_grid(Lx: float,
                             Ly: float,
                             x_grid: np.ndarray,
                             y_grid: np.ndarray,
                             size_grid: np.ndarray,
                             *,
                             jitter: float = 0.05) -> np.ndarray:
    """
    Fast deterministic fill: build a near-hex lattice in each size cell.

    We respect the local target size (cell-constant) and rely on the upstream
    ratio limiter to keep adjacent cells from diverging too much.
    """
    size_min = max(float(np.nanmin(size_grid)), 1e-9)
    pitch_hash = size_min * 0.6  # hash cell size for deduplication
    inv_hash = 1.0 / pitch_hash
    seen: dict[tuple[int, int], int] = {}
    pts: list[tuple[float, float]] = []
    row_factor = math.sqrt(3) / 2.0
    rng = np.random.default_rng(0)
    nx = size_grid.shape[0]
    ny = size_grid.shape[1]
    for i in range(nx):
        x0 = x_grid[i]
        x1 = x_grid[i + 1]
        for j in range(ny):
            y0 = y_grid[j]
            y1 = y_grid[j + 1]
            s = float(size_grid[i, j])
            if not math.isfinite(s) or s <= 0:
                continue
            dx = s
            dy = s * row_factor
            row = 0
            y = y0
            while y <= y1 + 1e-12:
                x_offset = 0.5 * dx if (row % 2) else 0.0
                x = x0 + x_offset
                while x <= x1 + 1e-12:
                    jx = (rng.random() - 0.5) * dx * jitter
                    jy = (rng.random() - 0.5) * dy * jitter
                    px = x + jx
                    py = y + jy
                    if px < 0.0 or px > Lx or py < 0.0 or py > Ly:
                        x += dx
                        continue
                    hx = int(px * inv_hash)
                    hy = int(py * inv_hash)
                    key = (hx, hy)
                    if key not in seen:
                        seen[key] = 1
                        pts.append((px, py))
                    x += dx
                row += 1
                y += dy
    if not pts:
        return np.empty((0, 2), dtype=float)
    return np.asarray(pts, dtype=float)


def _prune_close_points(points: np.ndarray,
                        *,
                        x_grid: np.ndarray,
                        y_grid: np.ndarray,
                        size_grid: np.ndarray,
                        tol: float = 0.9) -> np.ndarray:
    """Single pass culling of points closer than tol * target size to smooth seams."""
    if points.size == 0:
        return points
    size_min = max(float(np.nanmin(size_grid)), 1e-9)
    pitch_hash = size_min * 0.6
    inv_hash = 1.0 / pitch_hash
    buckets: dict[tuple[int, int], list[int]] = {}
    keep = np.ones(points.shape[0], dtype=bool)
    for idx, (px, py) in enumerate(points):
        gx = int(px * inv_hash)
        gy = int(py * inv_hash)
        buckets.setdefault((gx, gy), []).append(idx)
    for (gx, gy), indices in buckets.items():
        neighbor_keys = [(gx + dx, gy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        for key in neighbor_keys:
            for i in indices:
                if not keep[i]:
                    continue
                for j in buckets.get(key, []):
                    if j <= i or not keep[j]:
                        continue
                    dx = points[i, 0] - points[j, 0]
                    dy = points[i, 1] - points[j, 1]
                    dist = math.hypot(dx, dy)
                    if dist <= 1e-12:
                        keep[j] = False
                        continue
                    si = _size_at_point(points[i, 0], points[i, 1], x_grid, y_grid, size_grid, focus_regions=None, focus_size_cap=None)
                    sj = _size_at_point(points[j, 0], points[j, 1], x_grid, y_grid, size_grid, focus_regions=None, focus_size_cap=None)
                    target = tol * 0.5 * (si + sj)
                    if dist < target:
                        if si >= sj:
                            keep[i] = False
                            break
                        else:
                            keep[j] = False
    return points[keep]


def _poisson_sample_variable_radius(Lx: float,
                                    Ly: float,
                                    *,
                                    x_grid: np.ndarray,
                                    y_grid: np.ndarray,
                                    size_grid: np.ndarray,
                                    focus_regions: list[tuple[float, float, float, float]] | None,
                                    focus_size_cap: float | None,
                                    k: int = 12) -> np.ndarray:
    """
    Bridson Poisson-disk sampler with spatially varying minimum spacing.

    The minimum distance between samples near (x,y) is
    radius_factor * size_grid_at(x,y), so density tracks the target size
    field and triangles remain closer to equilateral than a stretched base
    tensor grid.
    """
    size_min = float(np.nanmin(size_grid))
    size_max = float(np.nanmax(size_grid))
    if not np.isfinite(size_min) or size_min <= 0:
        return np.empty((0, 2), dtype=float)
    radius_factor = 0.55  # converts desired edge length -> Poisson radius
    def radius_fn(pt: np.ndarray) -> float:
        s = _size_at_point(float(pt[0]), float(pt[1]), x_grid, y_grid, size_grid,
                           focus_regions=focus_regions, focus_size_cap=focus_size_cap)
        return max(radius_factor * max(s, size_min), 1e-9)

    min_radius = radius_factor * size_min
    cell_size = min_radius / math.sqrt(2.0)
    grid: dict[tuple[int, int], list[int]] = {}
    points: list[np.ndarray] = []
    radii: list[float] = []

    def fits(pt: np.ndarray, r_pt: float) -> bool:
        gx = int(pt[0] / cell_size)
        gy = int(pt[1] / cell_size)
        neighborhood = range(-2, 3)
        for dx in neighborhood:
            for dy in neighborhood:
                bucket = grid.get((gx + dx, gy + dy))
                if not bucket:
                    continue
                for idx in bucket:
                    other = points[idx]
                    dist = math.hypot(pt[0] - other[0], pt[1] - other[1])
                    min_allowed = max(r_pt, radii[idx])
                    if dist < min_allowed:
                        return False
        return True

    # Seed with a deterministic corner and a random point to avoid empty sets.
    seed = np.array([0.5 * Lx, 0.5 * Ly], dtype=float)
    r_seed = radius_fn(seed)
    points.append(seed)
    radii.append(r_seed)
    grid[(int(seed[0] / cell_size), int(seed[1] / cell_size))] = [0]
    active = [0]

    rng = np.random.default_rng()
    while active:
        idx = int(rng.choice(active))
        base_pt = points[idx]
        base_r = radii[idx]
        accepted = False
        for _ in range(max(1, k)):
            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            radial = float(rng.uniform(base_r, 2.0 * base_r))
            candidate = base_pt + radial * np.array([math.cos(ang), math.sin(ang)])
            if not (0.0 <= candidate[0] <= Lx and 0.0 <= candidate[1] <= Ly):
                continue
            r_cand = radius_fn(candidate)
            if fits(candidate, r_cand):
                pt_idx = len(points)
                points.append(candidate)
                radii.append(r_cand)
                cell = (int(candidate[0] / cell_size), int(candidate[1] / cell_size))
                grid.setdefault(cell, []).append(pt_idx)
                active.append(pt_idx)
                accepted = True
                break
        if not accepted:
            active.remove(idx)

    return np.asarray(points, dtype=float)
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
                                 mesh_meta: dict | None = None,
                                 mesh_only: bool = False):
    """Build material and source fields from a JSON case definition.

    When `mesh_only` is True we switch to a fast centroid-based classifier
    (no per-triangle polygon clipping) to keep mesh previews fast. The full
    expensive painting runs only for a true solve.
    """
    Lx, Ly = LxLy
    tri_area_vals = tri_areas(nodes, tris)
    Ne = tris.shape[0]
    region = np.full(Ne, AIR, dtype=np.int8)
    mu_r = np.ones(Ne, dtype=float)
    Mx = np.zeros(Ne, dtype=float)
    My = np.zeros(Ne, dtype=float)
    Jz = np.zeros(Ne, dtype=float)

    def _fast_contains_polygon(points: np.ndarray, verts: list[list[float]], holes: list[list[list[float]]] | None):
        """Fast point-in-polygon with chunking; returns boolean mask."""
        poly = np.asarray(verts, dtype=float)
        if poly.ndim != 2 or poly.shape[0] < 3:
            return np.zeros(points.shape[0], dtype=bool)
        edges_x0 = poly[:, 0]
        edges_y0 = poly[:, 1]
        edges_x1 = np.roll(edges_x0, -1)
        edges_y1 = np.roll(edges_y0, -1)
        chunk = 8192
        inside = np.zeros(points.shape[0], dtype=bool)
        for start in range(0, points.shape[0], chunk):
            stop = min(points.shape[0], start + chunk)
            px = points[start:stop, 0][:, None]
            py = points[start:stop, 1][:, None]
            cond = ((edges_y0 > py) != (edges_y1 > py)) & (
                px < (edges_x1 - edges_x0) * (py - edges_y0) / (edges_y1 - edges_y0 + 1e-16) + edges_x0
            )
            cnt = cond.sum(axis=1)
            inside[start:stop] = (cnt % 2) == 1
        if holes:
            for hole in holes:
                hole_arr = np.asarray(hole, dtype=float)
                if hole_arr.shape[0] < 3:
                    continue
                edges_x0 = hole_arr[:, 0]
                edges_y0 = hole_arr[:, 1]
                edges_x1 = np.roll(edges_x0, -1)
                edges_y1 = np.roll(edges_y0, -1)
                for start in range(0, points.shape[0], chunk):
                    stop = min(points.shape[0], start + chunk)
                    px = points[start:stop, 0][:, None]
                    py = points[start:stop, 1][:, None]
                    cond = ((edges_y0 > py) != (edges_y1 > py)) & (
                        px < (edges_x1 - edges_x0) * (py - edges_y0) / (edges_y1 - edges_y0 + 1e-16) + edges_x0
                    )
                    cnt = cond.sum(axis=1)
                    hole_inside = (cnt % 2) == 1
                    inside[start:stop] &= ~hole_inside
        return inside

    def _fast_contains(shape: dict, pts: np.ndarray) -> np.ndarray:
        """Centroid-based containment checks for fast mesh-only painting."""
        stype = str(shape.get("type", "")).lower()
        if stype == "polygon":
            verts = shape.get("vertices") or []
            holes_raw = shape.get("holes") or []
            holes: list[list[list[float]]] = []
            for h in holes_raw:
                if isinstance(h, (list, tuple)) and len(h) >= 3:
                    holes.append([[float(v[0]), float(v[1])] if not isinstance(v, dict) else [float(v.get("x", 0.0)), float(v.get("y", 0.0))] for v in h])
            verts_arr: list[list[float]] = []
            for v in verts:
                try:
                    if isinstance(v, dict):
                        verts_arr.append([float(v.get("x", 0.0)), float(v.get("y", 0.0))])
                    elif isinstance(v, (list, tuple)) and len(v) >= 2:
                        verts_arr.append([float(v[0]), float(v[1])])
                except (TypeError, ValueError):
                    continue
            if len(verts_arr) < 3:
                return np.zeros(pts.shape[0], dtype=bool)
            return _fast_contains_polygon(pts, verts_arr, holes)
        if stype == "rect":
            center = shape.get("center", [0.0, 0.0])
            if isinstance(center, dict):
                cx = float(center.get("x", 0.0))
                cy = float(center.get("y", 0.0))
            else:
                cx, cy = (float(center[0]), float(center[1])) if isinstance(center, (list, tuple)) and len(center) >= 2 else (0.0, 0.0)
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
            angle = float(shape.get("angle", 0.0))
            if width <= 0 or height <= 0:
                return np.zeros(pts.shape[0], dtype=bool)
            if abs(angle) > 1e-9:
                ang = math.radians(angle)
                cos_a = math.cos(ang)
                sin_a = math.sin(ang)
                rel = pts - np.array([cx, cy])
                x_local = rel[:, 0] * cos_a + rel[:, 1] * sin_a
                y_local = -rel[:, 0] * sin_a + rel[:, 1] * cos_a
            else:
                x_local = pts[:, 0] - cx
                y_local = pts[:, 1] - cy
            return (
                (x_local >= -width / 2)
                & (x_local <= width / 2)
                & (y_local >= -height / 2)
                & (y_local <= height / 2)
            )
        if stype == "circle":
            center = shape.get("center", [0.0, 0.0])
            if isinstance(center, dict):
                cx = float(center.get("x", 0.0))
                cy = float(center.get("y", 0.0))
            else:
                cx, cy = (float(center[0]), float(center[1])) if isinstance(center, (list, tuple)) and len(center) >= 2 else (0.0, 0.0)
            radius = float(shape.get("radius", 0.0))
            if radius <= 0:
                return np.zeros(pts.shape[0], dtype=bool)
            r2 = radius * radius
            dx = pts[:, 0] - cx
            dy = pts[:, 1] - cy
            return (dx * dx + dy * dy) <= r2
        if stype == "ring":
            center = shape.get("center", [0.0, 0.0])
            if isinstance(center, dict):
                cx = float(center.get("x", 0.0))
                cy = float(center.get("y", 0.0))
            else:
                cx, cy = (float(center[0]), float(center[1])) if isinstance(center, (list, tuple)) and len(center) >= 2 else (0.0, 0.0)
            outer = float(shape.get("outer_radius", shape.get("outerRadius", shape.get("radius", 0.0))))
            inner = float(shape.get("inner_radius", shape.get("innerRadius", 0.0)))
            if outer <= 0:
                return np.zeros(pts.shape[0], dtype=bool)
            dx = pts[:, 0] - cx
            dy = pts[:, 1] - cy
            r2 = dx * dx + dy * dy
            return (r2 <= outer * outer) & (r2 >= inner * inner)
        return np.zeros(pts.shape[0], dtype=bool)

    if mesh_only:
        # Fast centroid classification: no clipping, but assigns materials.
        centroids = nodes[tris].mean(axis=1)
        defaults = definition.get("defaults", {}) if isinstance(definition, dict) else {}
        objects = definition.get("objects", []) if isinstance(definition, dict) else []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            material = str(obj.get("material", "air")).lower()
            if material not in {"air", "magnet", "steel", "wire"}:
                continue
            shape = obj.get("shape") or {}
            mask = _fast_contains(shape, centroids)
            if not np.any(mask):
                continue
            params = obj.get("params") or {}
            mat_defaults = defaults.get(material, {}) if isinstance(defaults, dict) else {}

            def _param(key, fallback):
                if key in params:
                    return params[key]
                if isinstance(mat_defaults, dict) and key in mat_defaults:
                    return mat_defaults[key]
                return fallback

            if material in {"air", "magnet", "steel"}:
                target_mu = float(_param("mu_r", 1.0 if material == "air" else 1.05 if material == "magnet" else 1000.0))
                mu_r[mask] = target_mu
                if material == "magnet":
                    target_mx = float(_param("Mx", 0.0))
                    target_my = float(_param("My", 8e5))
                    angle_deg = _shape_angle_degrees(shape)
                    mx_rot, my_rot = _rotate_xy(target_mx, target_my, angle_deg)
                    Mx[mask] = mx_rot
                    My[mask] = my_rot
                region_id = {"air": AIR, "magnet": PMAG, "steel": STEEL}[material]
                region[mask] = region_id
            if material == "wire":
                total_current = float(_param("current", 0.0))
                A = tri_area_vals[mask].sum()
                if abs(A) > 0:
                    Jz[mask] += total_current / A

        meta = dict(
            Lx=Lx,
            Ly=Ly,
            defaults=defaults,
            geometry=objects,
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

    defaults = definition.get("defaults", {})
    objects = definition.get("objects", [])
    geometry = []

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        material = str(obj.get("material", "air")).lower()
        if material == "contour":
            # Contour-only shapes are kept for post-processing but never affect fields,
            # so skip any overlap/area computations that scale with polygon sides.
            geometry.append(obj)
            continue
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
    contours = [
        obj for obj in geometry
        if isinstance(obj, dict) and str(obj.get("material", "")).lower() == "contour"
    ]
    if contours:
        meta["contours"] = contours
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

    # Bounding-box based culling: quickly mark tris entirely outside or inside.
    tri_min = pts_iter.min(axis=1)
    tri_max = pts_iter.max(axis=1)

    outside = (
        (tri_max[:, 0] < xmin)
        | (tri_min[:, 0] > xmax)
        | (tri_max[:, 1] < ymin)
        | (tri_min[:, 1] > ymax)
    )
    inside = (
        (tri_min[:, 0] >= xmin)
        & (tri_max[:, 0] <= xmax)
        & (tri_min[:, 1] >= ymin)
        & (tri_max[:, 1] <= ymax)
    )

    fractions[inside] = 1.0
    boundary_idxs = np.nonzero(~outside & ~inside)[0]

    for idx in boundary_idxs:
        pts = pts_iter[idx]
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
    if stype == "polygon":
        if "holes" in shape:
            verts = shape.get("vertices")
            holes = shape.get("holes") or []
            try:
                loops: list[list[list[float]]] = []
                if isinstance(verts, (list, tuple)) and len(verts) >= 3:
                    loops.append([[float(v[0]), float(v[1])] if not isinstance(v, dict) else [float(v.get("x", 0.0)), float(v.get("y", 0.0))] for v in verts])
                for hole in holes:
                    if isinstance(hole, (list, tuple)) and len(hole) >= 3:
                        loops.append([[float(v[0]), float(v[1])] if not isinstance(v, dict) else [float(v.get("x", 0.0)), float(v.get("y", 0.0))] for v in hole])
            except (TypeError, ValueError):
                loops = []
            if loops:
                return polygon_overlap_fraction_loops(nodes, tris, loops, tri_area_vals)
        verts_raw = shape.get("vertices")
        if isinstance(verts_raw, (list, tuple)) and len(verts_raw) >= 3:
            verts: list[list[float]] = []
            for v in verts_raw:
                try:
                    if isinstance(v, dict):
                        verts.append([float(v.get("x", 0.0)), float(v.get("y", 0.0))])
                    elif isinstance(v, (list, tuple)) and len(v) >= 2:
                        verts.append([float(v[0]), float(v[1])])
                except (TypeError, ValueError):
                    continue
            if len(verts) >= 3:
                return polygon_overlap_fraction_vertices(
                    nodes,
                    tris,
                    verts,
                    tri_area_vals,
                )
        center = shape.get("center", [0.0, 0.0])
        if isinstance(center, dict):
            cx = center.get("x", 0.0)
            cy = center.get("y", 0.0)
        elif isinstance(center, (list, tuple)) and len(center) >= 2:
            cx, cy = center
        else:
            cx = cy = 0.0
        radius = float(shape.get("radius", 0.0))
        sides = int(shape.get("sides", 3))
        rotation = float(shape.get("rotation", shape.get("angle", 0.0)))
        return polygon_overlap_fraction(
            nodes,
            tris,
            float(cx),
            float(cy),
            float(radius),
            int(max(3, sides)),
            float(rotation),
            tri_area_vals,
        )
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
    if isinstance(shape, dict) and "rotation" in shape:
        try:
            return float(shape.get("rotation", 0.0))
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


def _regular_polygon_vertices(cx: float,
                              cy: float,
                              radius: float,
                              sides: int,
                              rotation_deg: float = 0.0) -> list[list[float]]:
    """Return CCW vertices for a regular polygon centered at (cx, cy)."""
    if radius <= 0 or sides < 3:
        return []
    angles = np.linspace(0.0, 2 * math.pi, sides, endpoint=False) + math.radians(rotation_deg)
    verts = []
    for ang in angles:
        verts.append([float(cx + radius * math.cos(ang)), float(cy + radius * math.sin(ang))])
    return verts


def _clip_polygon_to_edge(vertices: list[list[float]],
                          edge_start: list[float],
                          edge_end: list[float],
                          *,
                          eps: float = 1e-12) -> list[list[float]]:
    """Clip a polygon against a single directed edge (keep left side)."""
    if len(vertices) < 2:
        return vertices
    out: list[list[float]] = []
    ex = edge_end[0] - edge_start[0]
    ey = edge_end[1] - edge_start[1]

    def _inside(pt: list[float]) -> bool:
        return (ex * (pt[1] - edge_start[1]) - ey * (pt[0] - edge_start[0])) >= -eps

    def _segment_line_intersection(p1: list[float], p2: list[float]) -> list[float] | None:
        r1x, r1y = p1[0], p1[1]
        r2x, r2y = p2[0], p2[1]
        dx = r2x - r1x
        dy = r2y - r1y
        denom = dx * ey - dy * ex
        if abs(denom) < eps:
            return None
        t = ((edge_start[0] - r1x) * ey - (edge_start[1] - r1y) * ex) / denom
        if t < -eps or t > 1 + eps:
            return None
        return [
            float(r1x + t * dx),
            float(r1y + t * dy),
        ]

    prev = vertices[-1]
    prev_inside = _inside(prev)
    for cur in vertices:
        cur_inside = _inside(cur)
        if cur_inside:
            if not prev_inside:
                inter = _segment_line_intersection(prev, cur)
                if inter is not None:
                    out.append(inter)
            out.append(cur)
        elif prev_inside:
            inter = _segment_line_intersection(prev, cur)
            if inter is not None:
                out.append(inter)
        prev = cur
        prev_inside = cur_inside
    return out


def _clip_polygon_to_triangle(vertices: list[list[float]],
                              tri_pts: np.ndarray) -> list[list[float]]:
    """Clip an arbitrary polygon against a (convex) triangle."""
    clipped = vertices
    for k in range(3):
        a = tri_pts[k].tolist()
        b = tri_pts[(k + 1) % 3].tolist()
        clipped = _clip_polygon_to_edge(clipped, a, b)
        if len(clipped) < 3:
            break
    return clipped


def polygon_overlap_fraction_vertices(nodes,
                                      tris,
                                      vertices: list[list[float]],
                                      tri_area_vals=None):
    """Return area fraction of each triangle covered by an explicit polygon."""
    verts = [[float(v[0]), float(v[1])] for v in vertices if len(v) >= 2]
    if len(verts) < 3:
        return np.zeros(tris.shape[0], dtype=float)
    if len(verts) > MAX_EXPLICIT_POLYGON_SIDES:
        step = math.ceil(len(verts) / MAX_EXPLICIT_POLYGON_SIDES)
        verts = verts[::step]
        if len(verts) < 3:
            return np.zeros(tris.shape[0], dtype=float)
    tri_pts = nodes[tris]
    tri_min = tri_pts.min(axis=1)
    tri_max = tri_pts.max(axis=1)
    poly_bounds = np.asarray(verts)
    poly_min = poly_bounds.min(axis=0)
    poly_max = poly_bounds.max(axis=0)
    # Quick reject using bounding boxes before expensive clipping.
    overlaps_bbox = (
        (tri_max[:, 0] >= poly_min[0])
        & (tri_min[:, 0] <= poly_max[0])
        & (tri_max[:, 1] >= poly_min[1])
        & (tri_min[:, 1] <= poly_max[1])
    )
    areas = tri_area_vals if tri_area_vals is not None else tri_areas(nodes, tris)
    fractions = np.zeros(tris.shape[0], dtype=float)
    for idx in np.nonzero(overlaps_bbox)[0]:
        pts = tri_pts[idx]
        clipped = _clip_polygon_to_triangle(verts, pts)
        if len(clipped) >= 3 and areas[idx] > 0:
            area = polygon_area(clipped)
            fractions[idx] = min(1.0, max(0.0, area / areas[idx]))
    return fractions


def polygon_overlap_fraction(nodes,
                             tris,
                             cx: float,
                             cy: float,
                             radius: float,
                             sides: int,
                             rotation_deg: float,
                             tri_area_vals=None):
    """Return area fraction of each triangle covered by a regular polygon."""
    if radius <= 0 or sides < 3:
        return np.zeros(tris.shape[0], dtype=float)
    sides_int = int(max(3, sides))
    if sides_int > MAX_EXPLICIT_POLYGON_SIDES:
        return circle_overlap_fraction(nodes, tris, cx, cy, radius)
    poly = _regular_polygon_vertices(cx, cy, radius, sides_int, rotation_deg)
    if not poly:
        return np.zeros(tris.shape[0], dtype=float)
    tri_pts = nodes[tris]
    areas = tri_area_vals if tri_area_vals is not None else tri_areas(nodes, tris)
    fractions = np.zeros(tris.shape[0], dtype=float)
    for idx, pts in enumerate(tri_pts):
        clipped: list[list[float]] = pts.tolist()
        for i in range(len(poly)):
            a = poly[i]
            b = poly[(i + 1) % len(poly)]
            clipped = _clip_polygon_to_edge(clipped, a, b)
            if len(clipped) < 3:
                break
        area = polygon_area(clipped)
        if areas[idx] > 0:
            fractions[idx] = area / areas[idx]
    return fractions


def polygon_overlap_fraction_loops(nodes,
                                   tris,
                                   loops: list[list[list[float]]],
                                   tri_area_vals=None):
    """Return area fraction for a polygon with holes (outer + hole loops)."""
    if not loops or len(loops[0]) < 3:
        return np.zeros(tris.shape[0], dtype=float)
    outer = loops[0]
    holes = loops[1:]
    if len(outer) > MAX_EXPLICIT_POLYGON_SIDES:
        step = math.ceil(len(outer) / MAX_EXPLICIT_POLYGON_SIDES)
        outer = outer[::step]
    if len(outer) < 3:
        return np.zeros(tris.shape[0], dtype=float)
    downsampled_holes: list[list[list[float]]] = []
    for hole in holes:
        if len(hole) > MAX_EXPLICIT_POLYGON_SIDES:
            step = math.ceil(len(hole) / MAX_EXPLICIT_POLYGON_SIDES)
            hole = hole[::step]
        if len(hole) >= 3:
            downsampled_holes.append(hole)
    holes = downsampled_holes
    tri_pts = nodes[tris]
    tri_min = tri_pts.min(axis=1)
    tri_max = tri_pts.max(axis=1)
    bounds = np.asarray(outer)
    poly_min = bounds.min(axis=0)
    poly_max = bounds.max(axis=0)
    # Quick reject using bounding boxes before expensive clipping.
    overlaps_bbox = (
        (tri_max[:, 0] >= poly_min[0])
        & (tri_min[:, 0] <= poly_max[0])
        & (tri_max[:, 1] >= poly_min[1])
        & (tri_min[:, 1] <= poly_max[1])
    )
    areas = tri_area_vals if tri_area_vals is not None else tri_areas(nodes, tris)
    fractions = np.zeros(tris.shape[0], dtype=float)

    for idx in np.nonzero(overlaps_bbox)[0]:
        pts = tri_pts[idx]
        clipped_outer = _clip_polygon_to_triangle(outer, pts)
        if len(clipped_outer) < 3 or areas[idx] <= 0:
            continue
        area = polygon_area(clipped_outer)
        for hole in holes:
            clipped_hole = _clip_polygon_to_triangle(hole, pts)
            if len(clipped_hole) >= 3:
                area -= polygon_area(clipped_hole)
        fractions[idx] = max(0.0, min(1.0, area / areas[idx]))
    return fractions

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
    parser.add_argument("--adapt-from",
                        help="Path to an existing B_field.npz whose |∇B| should drive spacing")
    parser.add_argument("--adapt-refine-q", type=float, default=0.75,
                        help="Quantile (0-1) above which intervals are refined (default: 0.75)")
    parser.add_argument("--adapt-coarsen-q", type=float, default=0.25,
                        help="Quantile (0-1) below which intervals are coarsened (default: 0.25)")
    parser.add_argument("--adapt-refine-factor", type=float, default=0.5,
                        help="Multiplier applied to refined intervals (default: 0.5)")
    parser.add_argument("--adapt-coarsen-factor", type=float, default=2.0,
                        help="Multiplier applied to coarsened intervals (default: 2.0)")
    parser.add_argument("--adapt-smooth-passes", type=int, default=1,
                        help="How many 1-D smoothing passes to apply to the indicator (default: 1)")
    parser.add_argument("--adapt-material-pad", type=float, default=0.0,
                        help="Pad distance (m) around non-air materials where max spacing caps apply (default: 0)")
    parser.add_argument("--adapt-material-max-scale", type=float, default=1.0,
                        help="Limit growth near materials to at most this multiple of the previous spacing (default: 1.0)")
    parser.add_argument("--adapt-allow-coarsen", action="store_true",
                        help="Allow the field-driven mesh to coarsen low-gradient regions (default: refine-only)")
    parser.add_argument("--adapt-direction-weight", type=float, default=1.0,
                        help="Weight of directional change (angle of B) when deriving density scalars from a B-field")
    parser.add_argument("--adapt-magnitude-weight", type=float, default=1.0,
                        help="Weight of |B| magnitude gradients when deriving density scalars from a B-field")
    parser.add_argument("--adapt-density-gain", type=float, default=0.4,
                        help="Exponent applied when mapping indicator to target size (default: 0.4)")
    parser.add_argument("--adapt-density-neutral", type=float, default=float("nan"),
                        help="Indicator value that maps to a neutral (1×) spacing; NaN means auto median")
    parser.add_argument("--adapt-density-min-scale", type=float, default=0.5,
                        help="Minimum spacing multiplier applied by the field-driven density (default: 0.5)")
    parser.add_argument("--adapt-density-max-scale", type=float, default=2.0,
                        help="Maximum spacing multiplier applied by the field-driven density (default: 2.0)")
    parser.add_argument("--adapt-density-smooth", type=int, default=2,
                        help="Number of smoothing passes applied to the indicator/size maps (default: 2)")
    parser.add_argument("--mesh-only", action="store_true",
                        help="Generate mesh geometry only (skip material/field assignment; for previews)")
    return parser.parse_args()


def _write_case(payload: dict, case_dir: Path, filename: str = "mesh.npz", *, compress: bool = True) -> Path:
    case_dir.mkdir(parents=True, exist_ok=True)
    outpath = case_dir / filename
    if compress:
        np.savez_compressed(outpath, **payload)
    else:
        np.savez(outpath, **payload)
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
    mesh_spec = grid.get("mesh") or grid.get("meshing") or {}
    mesh_type = str(mesh_spec.get("type", "graded")).lower() if isinstance(mesh_spec, dict) else "graded"
    use_point_cloud = mesh_type in UNSTRUCTURED_MESH_TYPES

    x_coords = y_coords = None
    mesh_meta = None
    nodes = tris = None
    L = (grid["Lx"], grid["Ly"])

    if use_point_cloud:
        field_focus_spec = None
        if isinstance(mesh_spec, dict):
            spec_focus = mesh_spec.get("field_focus")
            if isinstance(spec_focus, dict):
                field_focus_spec = spec_focus
        neutral_arg = None if math.isnan(getattr(args, "adapt_density_neutral", float("nan"))) else float(args.adapt_density_neutral)
        field_focus_defaults = {
            "enabled": True,
            "direction_weight": float(args.adapt_direction_weight),
            "magnitude_weight": float(args.adapt_magnitude_weight),
            "indicator_gain": max(float(args.adapt_density_gain), 0.0),
            "indicator_neutral": neutral_arg,
            "scale_min": max(float(args.adapt_density_min_scale), 1e-3),
            "scale_max": max(float(args.adapt_density_max_scale), 1e-3),
            "smooth_passes": max(int(args.adapt_density_smooth), 0),
            "size_smooth_passes": max(int(args.adapt_density_smooth), 0),
            "indicator_clip": [5.0, 95.0],
            "ratio_limit": 1.7,
        }
        if field_focus_defaults["scale_max"] < field_focus_defaults["scale_min"]:
            field_focus_defaults["scale_max"] = field_focus_defaults["scale_min"]
        try:
            nodes, tris, L, mesh_meta = _build_point_cloud_mesh(
                grid,
                case_definition,
                adapt_from=args.adapt_from,
                field_focus_spec=field_focus_spec,
                field_focus_defaults=field_focus_defaults,
            )
        except ValueError as exc:
            raise SystemExit(f"Failed to build point-cloud mesh: {exc}") from exc
    else:
        if args.adapt_from:
            try:
                x_coords, y_coords, mesh_meta = build_axes_from_field(
                    args.adapt_from,
                    target_size=(grid["Lx"], grid["Ly"]),
                    refine_quantile=args.adapt_refine_q,
                    coarsen_quantile=args.adapt_coarsen_q,
                    refine_factor=args.adapt_refine_factor,
                    coarsen_factor=args.adapt_coarsen_factor,
                    smooth_passes=args.adapt_smooth_passes,
                    material_pad=args.adapt_material_pad,
                    material_max_scale=args.adapt_material_max_scale,
                    allow_coarsen=args.adapt_allow_coarsen,
                )
            except FieldAdaptError as exc:
                raise SystemExit(f"Field-driven adaptivity failed: {exc}") from exc
            grid["Nx"] = int(x_coords.size)
            grid["Ny"] = int(y_coords.size)
        else:
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
            mesh_only=bool(args.mesh_only),
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

    # Mesh-only previews prioritize speed over file size: skip compression.
    outpath = _write_case(payload, case_dir, compress=not args.mesh_only)
    if args.mesh_only:
        print(
            f"Wrote preview mesh {outpath} with:",
            f"\n  nodes={payload['nodes'].shape}",
            f"\n  tris={payload['tris'].shape}",
            f"\n  mu_r:  [{payload['mu_r'].min():.3g}, {payload['mu_r'].max():.3g}]",
        )
    else:
        print(
            f"Wrote {outpath} with:",
            f"\n  nodes={payload['nodes'].shape}",
            f"\n  tris={payload['tris'].shape}",
            f"\n  mu_r:  [{payload['mu_r'].min():.3g}, {payload['mu_r'].max():.3g}]",
            f"\n  |M|:   [{np.hypot(payload['Mx'],payload['My']).min():.3g}, "
            f"{np.hypot(payload['Mx'],payload['My']).max():.3g}] A/m",
            f"\n  Jz sum={payload['Jz'].sum() * (grid['Lx'] * grid['Ly']):.6g} A (approx)",
        )
