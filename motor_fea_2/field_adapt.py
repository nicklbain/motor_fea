"""Field-driven adaptive axis builder for the structured square mesh."""
from __future__ import annotations

from pathlib import Path

import numpy as np

EPS = 1e-12

class FieldAdaptError(RuntimeError):
    """Raised when we cannot derive structured axes from a field payload."""


def build_axes_from_field(
    field_path: Path | str,
    *,
    target_size: tuple[float, float],
    refine_quantile: float = 0.75,
    coarsen_quantile: float = 0.25,
    refine_factor: float = 0.5,
    coarsen_factor: float = 2.0,
    smooth_passes: int = 1,
    material_pad: float = 0.0,
    material_max_scale: float = 1.0,
    allow_coarsen: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Derive nonuniform x/y coordinates from an existing B-field solution.

    We treat the mesh as structured (same layout as square_tri_mesh) and
    compute a per-cell |∇B| indicator. Columns / rows with large indicators
    get their spacing shrunk (refine_factor) while smooth regions expand
    (coarsen_factor). The number of intervals stays constant; only their
    lengths change, and we renormalize to the requested target size.
    """
    path = Path(field_path).expanduser().resolve()
    if not path.exists():
        raise FieldAdaptError(f"adapt-from file '{path}' was not found")
    target_Lx = float(target_size[0])
    target_Ly = float(target_size[1])
    if target_Lx <= 0 or target_Ly <= 0:
        raise FieldAdaptError("target domain must have positive Lx/Ly")
    with np.load(path, allow_pickle=True) as payload:
        try:
            nodes = np.asarray(payload["nodes"], dtype=float)
            Bmag = np.asarray(payload["Bmag"], dtype=float)
        except KeyError as exc:
            raise FieldAdaptError(
                f"{path} is missing required arrays (needs nodes + Bmag)"
            ) from exc
        region = payload.get("region_id")
        if region is not None:
            region = np.asarray(region)
    x_coords, y_coords = _extract_structured_axes(nodes)
    nx = x_coords.size
    ny = y_coords.size
    ncols = max(nx - 1, 1)
    nrows = max(ny - 1, 1)
    cell_field = _cell_field(Bmag, ncols, nrows)
    centers_x = 0.5 * (x_coords[:-1] + x_coords[1:])
    centers_y = 0.5 * (y_coords[:-1] + y_coords[1:])
    grad_x = _axis_gradient(cell_field, centers_x, axis=0)
    grad_y = _axis_gradient(cell_field, centers_y, axis=1)
    grad_mag = np.hypot(grad_x, grad_y)
    col_scores = grad_mag.max(axis=1) if grad_mag.size else np.array([])
    row_scores = grad_mag.max(axis=0) if grad_mag.size else np.array([])
    if smooth_passes > 0:
        col_scores = _smooth_scores(col_scores, smooth_passes)
        row_scores = _smooth_scores(row_scores, smooth_passes)
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    cell_material_mask = None
    if region is not None and region.size == 2 * ncols * nrows:
        cell_material_mask = region.reshape((ncols, nrows, 2))
        cell_material_mask = np.any(cell_material_mask != 0, axis=2)
    new_dx, x_stats = _scaled_intervals(
        dx,
        col_scores,
        refine_quantile,
        coarsen_quantile,
        refine_factor,
        coarsen_factor,
        allow_coarsen=allow_coarsen,
    )
    new_dy, y_stats = _scaled_intervals(
        dy,
        row_scores,
        refine_quantile,
        coarsen_quantile,
        refine_factor,
        coarsen_factor,
        allow_coarsen=allow_coarsen,
    )
    pad = max(float(material_pad), 0.0)
    max_scale = max(float(material_max_scale), 1.0)
    caps_x = caps_y = None
    material_meta = {}
    if cell_material_mask is not None:
        col_mask = np.any(cell_material_mask, axis=1)
        row_mask = np.any(cell_material_mask, axis=0)
        caps_x, x_cap_info = _material_caps_for_axis(dx, centers_x, col_mask, pad, max_scale)
        caps_y, y_cap_info = _material_caps_for_axis(dy, centers_y, row_mask, pad, max_scale)
        if x_cap_info:
            material_meta["x"] = x_cap_info
        if y_cap_info:
            material_meta["y"] = y_cap_info
    new_x, clamp_x = _normalize_intervals(new_dx, target_Lx, origin=float(x_coords[0]), caps=caps_x)
    new_y, clamp_y = _normalize_intervals(new_dy, target_Ly, origin=float(y_coords[0]), caps=caps_y)
    mesh_meta = {
        "type": "field_adapt",
        "source_field": str(path),
        "indicator": "Bmag_gradient",
        "refine_quantile": float(refine_quantile),
        "coarsen_quantile": float(coarsen_quantile),
        "refine_factor": float(refine_factor),
        "coarsen_factor": float(coarsen_factor),
        "smooth_passes": int(max(smooth_passes, 0)),
        "material_pad": pad,
        "material_max_scale": max_scale,
        "allow_coarsen": bool(allow_coarsen),
        "x": _axis_meta(new_x, x_stats, target_Lx),
        "y": _axis_meta(new_y, y_stats, target_Ly),
    }
    mesh_meta["x"]["clamped_intervals"] = clamp_x
    mesh_meta["y"]["clamped_intervals"] = clamp_y
    if material_meta:
        mesh_meta["material_caps"] = material_meta
    return new_x, new_y, mesh_meta


def _extract_structured_axes(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise FieldAdaptError(f"nodes must be (N,2), got shape {nodes.shape}")
    xs = np.unique(np.round(nodes[:, 0], 12))
    ys = np.unique(np.round(nodes[:, 1], 12))
    xs.sort()
    ys.sort()
    if xs.size * ys.size != nodes.shape[0]:
        raise FieldAdaptError(
            "field mesh does not look like a structured grid "
            f"(len(xs) * len(ys) = {xs.size * ys.size}, nodes={nodes.shape[0]})"
        )
    if np.any(np.diff(xs) <= 0.0) or np.any(np.diff(ys) <= 0.0):
        raise FieldAdaptError("structured coordinates must be strictly increasing")
    return xs, ys


def _cell_field(values: np.ndarray, ncols: int, nrows: int) -> np.ndarray:
    expected = 2 * ncols * nrows
    if values.size != expected:
        raise FieldAdaptError(
            f"expected {expected} triangle samples, found {values.size}. "
            "Make sure you pass a B_field.npz generated from square_tri_mesh."
        )
    cells = values.reshape((ncols, nrows, 2))
    return cells.mean(axis=2)


def _axis_gradient(field: np.ndarray, centers: np.ndarray, axis: int) -> np.ndarray:
    if field.size == 0:
        return np.zeros_like(field)
    grad = np.zeros_like(field)
    if centers.size < 2:
        return grad
    if axis == 0:
        interior = centers[2:] - centers[:-2]
        interior = np.where(np.abs(interior) < EPS, np.sign(interior) * EPS, interior)
        grad[1:-1, :] = (field[2:, :] - field[:-2, :]) / interior[:, None]
        edge_left = centers[1] - centers[0]
        edge_right = centers[-1] - centers[-2]
        if abs(edge_left) < EPS:
            edge_left = EPS
        if abs(edge_right) < EPS:
            edge_right = EPS
        grad[0, :] = (field[1, :] - field[0, :]) / edge_left
        grad[-1, :] = (field[-1, :] - field[-2, :]) / edge_right
    else:
        interior = centers[2:] - centers[:-2]
        interior = np.where(np.abs(interior) < EPS, np.sign(interior) * EPS, interior)
        grad[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / interior[None, :]
        edge_bot = centers[1] - centers[0]
        edge_top = centers[-1] - centers[-2]
        if abs(edge_bot) < EPS:
            edge_bot = EPS
        if abs(edge_top) < EPS:
            edge_top = EPS
        grad[:, 0] = (field[:, 1] - field[:, 0]) / edge_bot
        grad[:, -1] = (field[:, -1] - field[:, -2]) / edge_top
    return grad


def _smooth_scores(values: np.ndarray, passes: int) -> np.ndarray:
    if values.size == 0 or passes <= 0:
        return values
    out = values.astype(float)
    kernel = np.array([0.25, 0.5, 0.25])
    for _ in range(passes):
        padded = np.pad(out, (1, 1), mode="edge")
        out = (
            kernel[0] * padded[:-2]
            + kernel[1] * padded[1:-1]
            + kernel[2] * padded[2:]
        )
    return out


def _material_caps_for_axis(
    intervals: np.ndarray,
    centers: np.ndarray,
    core_mask: np.ndarray,
    pad: float,
    max_scale: float,
) -> tuple[np.ndarray | None, dict | None]:
    if core_mask is None or not np.any(core_mask):
        return None, None
    centers = np.asarray(centers, dtype=float)
    expanded_mask = core_mask.copy()
    if pad > 1e-12:
        expanded_mask = _expand_mask_by_distance(core_mask, centers, pad)
    intervals = np.asarray(intervals, dtype=float)
    caps = np.full_like(intervals, np.inf)
    scale = max(float(max_scale), 1.0)
    caps[core_mask] = np.maximum(intervals[core_mask], EPS) * scale
    pad_only = expanded_mask & ~core_mask
    if pad_only.any():
        core_indices = np.flatnonzero(core_mask)
        core_centers = centers[core_indices]
        for idx in np.flatnonzero(pad_only):
            distances = np.abs(core_centers - centers[idx])
            nearest = core_indices[np.argmin(distances)]
            caps[idx] = intervals[nearest] * scale
    info = {
        "core_intervals": int(np.count_nonzero(core_mask)),
        "padded_intervals": int(np.count_nonzero(expanded_mask)),
        "pad": float(pad),
        "max_scale": float(scale),
    }
    return caps, info


def _expand_mask_by_distance(mask: np.ndarray, centers: np.ndarray, pad: float) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    centers = np.asarray(centers, dtype=float)
    pad = max(float(pad), 0.0)
    if pad <= 1e-12:
        return mask
    result = mask.copy()
    core_positions = centers[mask]
    for idx, center in enumerate(centers):
        if result[idx]:
            continue
        if np.any(np.abs(core_positions - center) <= pad + 1e-12):
            result[idx] = True
    return result


def _scaled_intervals(
    intervals: np.ndarray,
    scores: np.ndarray,
    refine_q: float,
    coarsen_q: float,
    refine_factor: float,
    coarsen_factor: float,
    *,
    allow_coarsen: bool,
) -> tuple[np.ndarray, dict]:
    intervals = np.asarray(intervals, dtype=float)
    if intervals.size == 0:
        return intervals, {"refined": 0, "coarsened": 0, "kept": 0}
    scores = np.asarray(scores, dtype=float)
    if scores.size != intervals.size:
        # Fallback: broadcast best effort
        if scores.size == 0:
            scores = np.zeros_like(intervals)
        else:
            scores = np.interp(
                np.linspace(0, 1, intervals.size),
                np.linspace(0, 1, scores.size),
                scores,
            )
    valid = scores[np.isfinite(scores)]
    if valid.size == 0:
        thresholds = (np.inf, -np.inf)
    else:
        coarsen_thr = np.quantile(valid, np.clip(coarsen_q, 0.0, 1.0))
        refine_thr = np.quantile(valid, np.clip(refine_q, 0.0, 1.0))
        if refine_thr <= coarsen_thr:
            thresholds = (np.inf, -np.inf)
        else:
            thresholds = (refine_thr, coarsen_thr)
    refine_thr, coarsen_thr = thresholds
    min_allowed = max(intervals.min() * min(refine_factor, 1.0), intervals.min() * 0.25)
    min_allowed = max(min_allowed, EPS)
    max_allowed = min(intervals.max() * max(coarsen_factor, 1.0), intervals.sum())
    new_intervals = intervals.copy()
    refined = coarsened = kept = 0
    for idx, (length, score) in enumerate(zip(intervals, scores)):
        if score >= refine_thr and length > min_allowed * 1.02:
            new_intervals[idx] = max(length * refine_factor, min_allowed)
            refined += 1
        elif allow_coarsen and score <= coarsen_thr and length < max_allowed / 1.02:
            new_intervals[idx] = min(length * coarsen_factor, max_allowed)
            coarsened += 1
        else:
            kept += 1
    stats = {"refined": refined, "coarsened": coarsened, "kept": kept}
    return new_intervals, stats


def _normalize_intervals(
    intervals: np.ndarray,
    total_length: float,
    *,
    origin: float = 0.0,
    caps: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    intervals = np.asarray(intervals, dtype=float)
    if intervals.size == 0:
        return np.array([origin, origin + total_length], dtype=float), 0
    if total_length <= 0:
        raise FieldAdaptError("target axis length must be positive")
    if caps is not None:
        caps_arr = np.asarray(caps, dtype=float)
        if caps_arr.shape != intervals.shape:
            raise FieldAdaptError("cap array shape mismatch for axis intervals")
    else:
        caps_arr = np.full_like(intervals, np.inf)
    final = _scale_with_caps(intervals, caps_arr, total_length)
    coords = origin + np.concatenate([[0.0], np.cumsum(final)])
    coords[-1] = origin + total_length
    clamped = int(np.sum(np.isfinite(caps_arr) & (final >= caps_arr - 1e-12)))
    return coords, clamped


def _scale_with_caps(base: np.ndarray, caps: np.ndarray, target: float) -> np.ndarray:
    base = np.asarray(base, dtype=float)
    caps = np.asarray(caps, dtype=float)
    if base.size == 0:
        return base
    final = np.zeros_like(base)
    remaining_idx = np.arange(base.size)
    remaining_target = float(target)
    tol = 1e-12
    while remaining_idx.size:
        subset = base[remaining_idx]
        subset_sum = subset.sum()
        if subset_sum <= tol:
            fill_value = remaining_target / remaining_idx.size if remaining_idx.size else 0.0
            final[remaining_idx] = fill_value
            remaining_target = 0.0
            break
        scale = remaining_target / subset_sum
        scaled = subset * scale
        caps_subset = caps[remaining_idx]
        over_mask = scaled > caps_subset + tol
        if over_mask.any():
            fixed_indices = remaining_idx[over_mask]
            final[fixed_indices] = caps_subset[over_mask]
            remaining_target -= final[fixed_indices].sum()
            remaining_target = max(remaining_target, 0.0)
            remaining_idx = remaining_idx[~over_mask]
            continue
        final[remaining_idx] = scaled
        remaining_target -= final[remaining_idx].sum()
        remaining_target = max(remaining_target, 0.0)
        break
    residue = target - final.sum()
    if abs(residue) > 1e-9 and final.size:
        final += residue / final.size
    return final


def _axis_meta(coords: np.ndarray, stats: dict, target_length: float) -> dict:
    diffs = np.diff(coords)
    min_spacing = float(diffs.min()) if diffs.size else 0.0
    max_spacing = float(diffs.max()) if diffs.size else 0.0
    return {
        "nodes": int(coords.size),
        "min_spacing": min_spacing,
        "max_spacing": max_spacing,
        "target_length": float(target_length),
        "refined_intervals": int(stats.get("refined", 0)),
        "coarsened_intervals": int(stats.get("coarsened", 0)),
        "kept_intervals": int(stats.get("kept", 0)),
    }
