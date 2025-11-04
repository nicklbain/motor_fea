#!/usr/bin/env python3
"""
Shared utilities for working with magnetostatic field snapshots exported by
magnetostatics_solver.py.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

mu0 = 4.0 * np.pi * 1e-7


def load_field_snapshot(npz_path: Path) -> Dict[str, np.ndarray]:
    """Load the NPZ archive produced by magnetostatics_solver.py."""
    data = np.load(npz_path)
    required = {"Bx", "By", "Bmag", "x_centers", "y_centers", "Lx", "Ly", "dx", "dy"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Snapshot missing required arrays: {sorted(missing)}")
    snapshot = {key: data[key] for key in data.files}
    snapshot["Lx"] = float(snapshot["Lx"])
    snapshot["Ly"] = float(snapshot["Ly"])
    snapshot["dx"] = float(snapshot["dx"])
    snapshot["dy"] = float(snapshot["dy"])
    snapshot["Nx"] = snapshot["Bx"].shape[1]
    snapshot["Ny"] = snapshot["Bx"].shape[0]
    return snapshot


def bilinear_interp(field: np.ndarray, x: float, y: float, *, dx: float, dy: float) -> float:
    """Sample a cell-centred field via bilinear interpolation."""
    ny, nx = field.shape
    xi = (x / dx) - 0.5
    yi = (y / dy) - 0.5
    i0 = int(np.floor(xi))
    j0 = int(np.floor(yi))
    if i0 < 0 or i0 >= nx - 1 or j0 < 0 or j0 >= ny - 1:
        raise ValueError("Sample point lies outside interpolation domain.")
    tx = xi - i0
    ty = yi - j0
    v00 = field[j0, i0]
    v10 = field[j0, i0 + 1]
    v01 = field[j0 + 1, i0]
    v11 = field[j0 + 1, i0 + 1]
    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


def polygon_signed_area(vertices: Sequence[Tuple[float, float]]) -> float:
    """Return the signed area (positive for CCW ordering)."""
    area = 0.0
    n = len(vertices)
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return 0.5 * area


def compute_polygon_mst(
    snapshot: Dict[str, np.ndarray],
    vertices: Sequence[Tuple[float, float]],
    *,
    torque_origin: Tuple[float, float],
    samples_per_cell: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Integrate Maxwell stress around a polygon defined by `vertices`.

    Returns per-edge force decomposition and total force/torque (per unit depth).
    """
    if len(vertices) < 3:
        raise ValueError("Polygon requires at least three vertices.")

    Lx = snapshot["Lx"]
    Ly = snapshot["Ly"]
    dx = snapshot["dx"]
    dy = snapshot["dy"]
    Bx_field = snapshot["Bx"]
    By_field = snapshot["By"]
    min_spacing = min(dx, dy)

    margin_x = 0.5 * dx
    margin_y = 0.5 * dy
    vertices_arr = np.asarray(vertices, dtype=float)
    if not (
        (vertices_arr[:, 0] >= margin_x).all()
        and (vertices_arr[:, 0] <= Lx - margin_x).all()
        and (vertices_arr[:, 1] >= margin_y).all()
        and (vertices_arr[:, 1] <= Ly - margin_y).all()
    ):
        raise ValueError("Polygon must stay within the domain (leaving space for interpolation).")

    area = polygon_signed_area(vertices)
    if math.isclose(area, 0.0):
        raise ValueError("Polygon is degenerate (zero area).")
    orientation = np.sign(area)  # +1 for CCW, -1 for CW

    torque_origin = np.asarray(torque_origin, dtype=float)
    total_force = np.zeros(2)
    total_torque = 0.0

    edge_reports: List[Dict[str, np.ndarray]] = []

    for idx in range(len(vertices)):
        p0 = vertices_arr[idx]
        p1 = vertices_arr[(idx + 1) % len(vertices)]
        edge_vec = p1 - p0
        length = float(np.hypot(edge_vec[0], edge_vec[1]))
        if math.isclose(length, 0.0):
            continue

        t_hat = edge_vec / length
        if orientation > 0.0:
            n_hat = np.array([t_hat[1], -t_hat[0]])
        else:
            n_hat = np.array([-t_hat[1], t_hat[0]])

        n_samples = max(
            4,
            int(np.ceil((length / min_spacing) * samples_per_cell)),
        )
        ds = length / n_samples

        edge_force = np.zeros(2)
        edge_normal_force = 0.0
        edge_shear_force = 0.0
        edge_torque = 0.0

        for k in range(n_samples):
            s = (k + 0.5) / n_samples
            point = p0 + edge_vec * s
            x_pt, y_pt = point

            Bx_val = bilinear_interp(Bx_field, x_pt, y_pt, dx=dx, dy=dy)
            By_val = bilinear_interp(By_field, x_pt, y_pt, dx=dx, dy=dy)
            Bsq = Bx_val * Bx_val + By_val * By_val
            Txx = (Bx_val * Bx_val - 0.5 * Bsq) / mu0
            Tyy = (By_val * By_val - 0.5 * Bsq) / mu0
            Txy = (Bx_val * By_val) / mu0

            traction = np.array([
                Txx * n_hat[0] + Txy * n_hat[1],
                Txy * n_hat[0] + Tyy * n_hat[1],
            ])

            edge_force += traction * ds
            edge_normal_force += float(np.dot(traction, n_hat) * ds)
            edge_shear_force += float(np.dot(traction, t_hat) * ds)
            rel = point - torque_origin
            edge_torque += float((rel[0] * traction[1] - rel[1] * traction[0]) * ds)

        edge_reports.append(
            {
                "edge_index": idx,
                "start": p0,
                "end": p1,
                "force": edge_force,
                "normal_force": edge_normal_force,
                "shear_force": edge_shear_force,
                "torque": edge_torque,
            }
        )

        total_force += edge_force
        total_torque += edge_torque

    return {
        "edge_reports": edge_reports,
        "total_force": total_force,
        "total_torque": total_torque,
        "orientation": orientation,
        "area": area,
    }
