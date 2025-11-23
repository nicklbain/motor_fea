#!/usr/bin/env python3
"""
solve_B.py
----------
Read a mesh+fields NPZ (from mesh_and_sources.py) and solve:

   -∇·( ν ∇A_z ) = Jz + [∇×M]_z

with homogeneous Neumann on the outer boundary (natural BC).
Permanent magnets contribute via the *surface* bound current
K_b = (M × n̂)·ẑ on PM/non-PM interfaces (edge load).
We pin one node to fix the gauge (A_z up to a constant).

Outputs (per case subfolder):
  - Az_field.npz : nodal solution (A_z).
  - B_field.npz  : triangle-centered B = (Bx, By, |B|).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

AIR, PMAG, STEEL = 0, 1, 2
DEFAULT_CASES_DIR = Path("cases")
DEFAULT_CASE_NAME = "mag2d_case"
DEFAULT_MESH_FILENAME = "mesh.npz"
LEGACY_FLAT_FILENAME = "mag2d_case.npz"
MU0 = 4e-7 * np.pi
MAX_POLY_SIDES_EXPLICIT = 4096

def load_case(path):
    data = np.load(path, allow_pickle=True)
    nodes = data["nodes"]
    tris  = data["tris"]
    region= data["region_id"]
    mu_r  = data["mu_r"]
    Mx    = data["Mx"]
    My    = data["My"]
    Jz    = data["Jz"]
    meta  = dict(data["meta"].item())
    return nodes, tris, region, mu_r, Mx, My, Jz, meta

def triangle_metrics(nodes: np.ndarray, tris: np.ndarray):
    """Return signed area*2, absolute area, and gradient helpers (b,c) per triangle."""
    P = nodes
    T = tris
    x1, y1 = P[T[:, 0], 0], P[T[:, 0], 1]
    x2, y2 = P[T[:, 1], 0], P[T[:, 1], 1]
    x3, y3 = P[T[:, 2], 0], P[T[:, 2], 1]
    twoA = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    A = 0.5 * np.abs(twoA)
    b = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1)
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1)
    return twoA, A, b, c


def assemble_A_system(nodes, tris, region, mu_r, Mx, My, Jz):
    """Build sparse K and RHS f for -div(nu grad A) = Jz + curl(M)_z + edge(M×n̂)."""
    nu_elem = 1.0 / (MU0 * mu_r)    # per element
    Nn = nodes.shape[0]
    Ne = tris.shape[0]
    T = tris
    twoA, A, b, c = triangle_metrics(nodes, tris)

    # Stiffness assembly (P1 triangles)
    rows, cols, vals = [], [], []
    for e in range(Ne):
        Ae = A[e]
        if Ae <= 0.0:
            continue
        factor = nu_elem[e] / (4.0 * Ae)
        Ke = factor * (np.outer(b[e], b[e]) + np.outer(c[e], c[e]))  # 3x3
        for ii in range(3):
            I = T[e, ii]
            for jj in range(3):
                J = T[e, jj]
                rows.append(I); cols.append(J); vals.append(Ke[ii, jj])

    K = sp.coo_matrix((vals, (rows, cols)), shape=(Nn, Nn)).tocsr()

    # RHS: volume Jz (piecewise constant) + volume curl(M) if provided (usually zero for uniform M)
    f = np.zeros(Nn, dtype=float)
    # Volume Jz
    for e in range(Ne):
        Fe = Jz[e] * A[e] / 3.0    # equally to each vertex for P1
        for ii in range(3):
            f[T[e, ii]] += Fe

    # Volume curl(M) term: f += (∂x My - ∂y Mx) integrated over element
    # Approximate with piecewise-constant gradients of M per element if needed.
    # Here M is uniform per element -> volume curl is zero inside each PM (usual choice).
    # If you *do* populate spatially varying M per element, uncomment below to include it.
    # dMy_dx, dMx_dy = elem_grad_piecewise(nodes, tris, My), elem_grad_piecewise(nodes, tris, Mx)
    # for e in range(Ne):
    #     g = (dMy_dx[e] - dMx_dy[e]) * A[e] / 3.0
    #     for ii in range(3):
    #         f[T[e, ii]] += g

    # Surface bound current from PM boundaries: K_b = (M × n̂)·ẑ = Mx*n_y - My*n_x
    add_magnet_edge_load(f, nodes, tris, region, Mx, My)

    return K, f

def add_magnet_edge_load(f, nodes, tris, region, Mx, My):
    """
    For every interior edge shared by (PMAG, non-PMAG), add edge load:
      ∫_edge K_b * v ds  with K_b = Mx * n_y - My * n_x
    n̂ is the unit normal pointing *outward* from the PM triangle.
    Each edge contributes equally to its two end nodes: +K_b * |edge| / 2.
    """
    # Build edge adjacency: key=(min(u,v), max(u,v)) -> list of (elem_index, oriented_pair(u->v))
    edges = {}
    T = tris; P = nodes
    for e in range(T.shape[0]):
        tri = T[e]
        oriented = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for (u,v) in oriented:
            key = (min(u,v), max(u,v))
            edges.setdefault(key, []).append((e, (u,v)))

    for key, adj in edges.items():
        if len(adj) != 2:
            continue  # boundary of the whole domain; PM likely not touching it in typical setup
        (e1, pair1), (e2, pair2) = adj
        r1, r2 = region[e1], region[e2]
        # Need exactly one PM and one non-PM for a PM boundary
        if (r1 == PMAG) ^ (r2 == PMAG):
            # Choose the PM element as "inside", get its oriented edge (u->v) along triangle boundary
            e_in  = e1 if r1 == PMAG else e2
            (u,v) = pair1 if r1 == PMAG else pair2
            # Edge geometry and outward normal for *this triangle* (CCW ensures left normal is outward)
            xi, yi = P[u]
            xj, yj = P[v]
            tx, ty = (xj - xi), (yj - yi)
            elen = (tx*tx + ty*ty)**0.5
            if elen == 0.0:
                continue
            # Outward unit normal (left turn of edge vector for CCW triangle)
            nx, ny = ty/elen, -tx/elen

            # Magnetization in the PM element (uniform per element)
            Mxe, Mye = Mx[e_in], My[e_in]
            Kb = Mxe*ny - Mye*nx  # (M × n̂)·ẑ

            # Lumped edge load: +Kb * |edge|/2 to both end nodes
            load = Kb * elen * 0.5
            f[u] += load
            f[v] += load

def solve_neumann_pinned(K, f, pin_node=0):
    """Pin one node (gauge fix) to eliminate the constant nullspace."""
    K = K.tolil()
    K[pin_node,:] = 0.0
    K[:,pin_node] = 0.0
    K[pin_node, pin_node] = 1.0
    f = f.copy()
    f[pin_node] = 0.0
    K = K.tocsr()
    A = spla.spsolve(K, f)
    return A


def compute_triangle_B(nodes: np.ndarray, tris: np.ndarray, Az: np.ndarray):
    """Compute B field per triangle from nodal Az (Bx=∂Az/∂y, By=-∂Az/∂x)."""
    twoA, _, b, c = triangle_metrics(nodes, tris)
    Ne = tris.shape[0]
    grad = np.zeros((Ne, 2), dtype=float)
    for e in range(Ne):
        denom = twoA[e]
        if denom == 0.0:
            continue
        a_vals = Az[tris[e]]
        dAdx = np.dot(a_vals, b[e]) / denom
        dAdy = np.dot(a_vals, c[e]) / denom
        grad[e, 0] = dAdx
        grad[e, 1] = dAdy
    Bx = grad[:, 1]
    By = -grad[:, 0]
    return Bx, By, np.hypot(Bx, By)


def _regular_polygon_vertices(cx: float, cy: float, radius: float, sides: int, rotation_deg: float):
    if not (radius > 0.0) or sides < 3:
        return []
    sides = int(max(3, sides))
    angles = np.linspace(0.0, 2 * np.pi, sides, endpoint=False) + np.radians(rotation_deg)
    verts = []
    for ang in angles:
        verts.append([cx + radius * np.cos(ang), cy + radius * np.sin(ang)])
    return verts


def _point_in_triangle(px: float, py: float, tri_pts: np.ndarray) -> bool:
    ax, ay = tri_pts[0]
    bx, by = tri_pts[1]
    cx, cy = tri_pts[2]
    v0x, v0y = cx - ax, cy - ay
    v1x, v1y = bx - ax, by - ay
    v2x, v2y = px - ax, py - ay
    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-18:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u >= -1e-10 and v >= -1e-10 and (u + v) <= 1.0 + 1e-10


def _build_triangle_lookup(nodes: np.ndarray,
                           tris: np.ndarray) -> tuple[np.ndarray, np.ndarray, cKDTree | None]:
    """Precompute triangle points/centroids and a KD-tree for spatial lookup."""
    if tris.size == 0:
        return np.zeros((0, 3, 2), dtype=float), np.zeros((0, 2), dtype=float), None
    tri_pts = nodes[tris]  # (Ne,3,2)
    centroids = tri_pts.mean(axis=1)
    try:
        tree = cKDTree(centroids)
    except Exception:
        tree = None
    return tri_pts, centroids, tree


def _sample_B_at_point(px: float,
                       py: float,
                       tri_pts: np.ndarray,
                       centroids: np.ndarray,
                       tree: cKDTree | None,
                       Bx: np.ndarray,
                       By: np.ndarray,
                       *,
                       initial_k: int = 12) -> tuple[float, float, float, int]:
    """
    Return (Bx, By, |B|, tri_index) at a point using a KD-tree to limit triangle tests.
    Falls back to nearest centroid if no containing triangle is found.
    """
    num_tris = tri_pts.shape[0]
    if num_tris == 0:
        return 0.0, 0.0, 0.0, -1

    if tree is not None:
        k = min(max(1, initial_k), num_tris)
        _, idxs = tree.query([px, py], k=k)
        if np.isscalar(idxs):
            idxs = [int(idxs)]
        else:
            idxs = [int(i) for i in np.atleast_1d(idxs)]
        for idx in idxs:
            if _point_in_triangle(px, py, tri_pts[idx]):
                bx = float(Bx[idx])
                by = float(By[idx])
                return bx, by, float(np.hypot(bx, by)), idx

        nearest_dist, nearest_idx = tree.query([px, py], k=1)
        search_radii = [nearest_dist * factor + 1e-12 for factor in (1.5, 3.0, 6.0)]
        for radius in search_radii:
            if not np.isfinite(radius):
                continue
            candidates = tree.query_ball_point([px, py], r=radius)
            for idx in candidates:
                if _point_in_triangle(px, py, tri_pts[idx]):
                    bx = float(Bx[idx])
                    by = float(By[idx])
                    return bx, by, float(np.hypot(bx, by)), idx
        nearest = int(nearest_idx)
    else:
        dx = centroids[:, 0] - px
        dy = centroids[:, 1] - py
        nearest = int(np.argmin(dx * dx + dy * dy))

    bx = float(Bx[nearest])
    by = float(By[nearest])
    return bx, by, float(np.hypot(bx, by)), nearest


def _contour_vertices_from_shape(shape: dict) -> list[list[float]]:
    stype = str(shape.get("type", "")).lower()
    center = shape.get("center", [0.0, 0.0])
    if isinstance(center, dict):
        cx = float(center.get("x", 0.0))
        cy = float(center.get("y", 0.0))
    elif isinstance(center, (list, tuple)) and len(center) >= 2:
        cx, cy = float(center[0]), float(center[1])
    else:
        cx = cy = 0.0
    if stype == "polygon":
        radius = float(shape.get("radius", 0.0))
        sides = int(shape.get("sides", 3))
        rotation = float(shape.get("rotation", shape.get("angle", 0.0)))
        sides = min(max(3, sides), MAX_POLY_SIDES_EXPLICIT)
        return _regular_polygon_vertices(cx, cy, radius, sides, rotation)
    if stype == "circle":
        radius = float(shape.get("radius", 0.0))
        if not (radius > 0.0):
            return []
        sides = min(MAX_POLY_SIDES_EXPLICIT, 256)
        return _regular_polygon_vertices(cx, cy, radius, sides, 0.0)
    if stype == "rect":
        width = float(shape.get("width", 0.0))
        height = float(shape.get("height", 0.0))
        angle = float(shape.get("angle", 0.0))
        if not (width > 0.0 and height > 0.0):
            return []
        half_w = 0.5 * width
        half_h = 0.5 * height
        corners = np.array([
            [half_w, half_h],
            [half_w, -half_h],
            [-half_w, -half_h],
            [-half_w, half_h],
        ])
        if angle:
            ang = np.radians(angle)
            rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            corners = corners @ rot.T
        corners += np.array([cx, cy])
        return corners.tolist()
    return []


def _compute_contour_forces(meta: dict[str, object],
                            nodes: np.ndarray,
                            tris: np.ndarray,
                            Bx: np.ndarray,
                            By: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    contours = []
    if isinstance(meta, dict):
        raw = meta.get("contours")
        if isinstance(raw, list):
            contours = raw
    if not contours:
        return [], []
    tri_pts, centroids, tree = _build_triangle_lookup(nodes, tris)
    segments: list[dict[str, object]] = []
    totals: list[dict[str, object]] = []
    for c_idx, contour in enumerate(contours):
        if not isinstance(contour, dict):
            continue
        shape = contour.get("shape") if isinstance(contour.get("shape"), dict) else contour.get("shape") or contour
        verts = _contour_vertices_from_shape(shape or {})
        if len(verts) < 3:
            continue
        center = np.mean(np.asarray(verts), axis=0)
        total_force = np.zeros(2, dtype=float)
        total_torque = 0.0
        for s_idx in range(len(verts)):
            a = np.asarray(verts[s_idx])
            b = np.asarray(verts[(s_idx + 1) % len(verts)])
            mid = 0.5 * (a + b)
            length = float(np.hypot(*(b - a)))
            if length <= 0.0:
                continue
            bx, by, bmag, tri_idx = _sample_B_at_point(mid[0], mid[1],
                                                      tri_pts, centroids, tree,
                                                      Bx, By)
            b2 = bx * bx + by * by
            # Maxwell stress tensor traction: t = (1/mu0)*(B B^T - 0.5|B|^2 I) · n
            normal = mid - center
            n_norm = float(np.hypot(normal[0], normal[1]))
            if n_norm == 0.0:
                tangent = b - a
                normal = np.array([tangent[1], -tangent[0]])
                n_norm = float(np.hypot(normal[0], normal[1]))
            nx, ny = (normal / max(n_norm, 1e-12)).tolist()
            t_x = (bx * bx * nx + bx * by * ny - 0.5 * b2 * nx) / MU0
            t_y = (bx * by * nx + by * by * ny - 0.5 * b2 * ny) / MU0
            fx = t_x * length
            fy = t_y * length
            total_force[0] += fx
            total_force[1] += fy
            lever = mid - center
            torque_z = lever[0] * fy - lever[1] * fx  # r × F (out of plane)
            total_torque += torque_z
            segments.append({
                "contour_index": c_idx,
                "contour_id": contour.get("id"),
                "contour_label": contour.get("label"),
                "segment_index": s_idx,
                "p0": a.tolist(),
                "p1": b.tolist(),
                "mid": mid.tolist(),
                "length": length,
                "normal": [nx, ny],
                "B": [bx, by],
                "Bmag": bmag,
                "traction": [t_x, t_y],
                "force": [fx, fy],
                "torque": torque_z,
                "tri_index": int(tri_idx),
            })
        totals.append({
            "contour_index": c_idx,
            "contour_id": contour.get("id"),
            "contour_label": contour.get("label"),
            "net_force": total_force.tolist(),
            "net_torque": float(total_torque),
        })
    return segments, totals


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve Az for a generated case",
        allow_abbrev=False
    )
    parser.add_argument("case", nargs="?",
                        help="Case folder, mesh file path, or case-relative npz")
    parser.add_argument("--case", dest="case_override",
                        help="Explicit case folder or mesh file path")
    parser.add_argument("--cases-dir", default=str(DEFAULT_CASES_DIR),
                        help="Base directory containing case subfolders (default: cases)")
    parser.add_argument("--pin-node", type=int, default=0,
                        help="Index of node to pin for gauge fixing (default: 0)")
    return parser.parse_args()


def _resolve_mesh_path(arg: str | None, cases_dir: Path) -> Path:
    cases_dir = cases_dir.resolve()
    new_default = cases_dir / DEFAULT_CASE_NAME / DEFAULT_MESH_FILENAME
    legacy_default = cases_dir / LEGACY_FLAT_FILENAME

    def _path_from_case(case_name: str) -> Path:
        candidate = cases_dir / case_name
        if candidate.is_dir():
            return candidate / DEFAULT_MESH_FILENAME
        return candidate

    if arg is None:
        if new_default.exists():
            return new_default
        if legacy_default.exists():
            return legacy_default
        return new_default

    user_path = Path(arg)
    if user_path.is_dir():
        return (user_path / DEFAULT_MESH_FILENAME).resolve()
    if not user_path.suffix:
        candidate = _path_from_case(arg)
        if candidate.exists() or candidate.parent.exists():
            return candidate.resolve()
    if not user_path.is_absolute():
        candidate = cases_dir / user_path
        if candidate.exists():
            return candidate.resolve()
    return user_path.resolve()


def _write_solution_files(case_mesh_path: Path,
                          nodes: np.ndarray,
                          tris: np.ndarray,
                          region: np.ndarray,
                          mu_r: np.ndarray,
                          Az: np.ndarray,
                          Bx: np.ndarray,
                          By: np.ndarray,
                          Bmag: np.ndarray,
                          meta: dict[str, object],
                          *,
                          contour_segments: list[dict[str, object]] | None = None,
                          contour_totals: list[dict[str, object]] | None = None):
    case_dir = case_mesh_path.parent
    case_dir.mkdir(parents=True, exist_ok=True)
    az_path = case_dir / "Az_field.npz"
    np.savez_compressed(az_path, Az=Az, nodes=nodes, tris=tris,
                        region_id=region, mu_r=mu_r, meta=meta)

    b_path = case_dir / "B_field.npz"
    payload = dict(
        Bx=Bx, By=By, Bmag=Bmag,
        tris=tris, nodes=nodes,
        region_id=region, mu_r=mu_r, meta=meta,
    )
    if contour_segments is not None:
        payload["contour_segments"] = np.array(contour_segments, dtype=object)
    if contour_totals is not None:
        payload["contour_totals"] = np.array(contour_totals, dtype=object)
    np.savez_compressed(b_path, **payload)
    return az_path, b_path

if __name__ == "__main__":
    args = _parse_args()
    case_arg = args.case_override if args.case_override is not None else args.case
    mesh_path = _resolve_mesh_path(case_arg, Path(args.cases_dir))
    nodes, tris, region, mu_r, Mx, My, Jz, meta = load_case(mesh_path)
    K, f = assemble_A_system(nodes, tris, region, mu_r, Mx, My, Jz)
    Az = solve_neumann_pinned(K, f, pin_node=args.pin_node)
    Bx, By, Bmag = compute_triangle_B(nodes, tris, Az)
    contour_segments, contour_totals = _compute_contour_forces(meta, nodes, tris, Bx, By)
    az_path, b_path = _write_solution_files(mesh_path, nodes, tris, region,
                                            mu_r, Az, Bx, By, Bmag, meta,
                                            contour_segments=contour_segments or None,
                                            contour_totals=contour_totals or None)
    print(f"Solved A_z on {nodes.shape[0]} nodes (tris={tris.shape[0]})")
    print(f"  A_z range: [{Az.min():.6g}, {Az.max():.6g}]  -> {az_path}")
    print(f"  |B| range: [{Bmag.min():.6g}, {Bmag.max():.6g}]  -> {b_path}")
    if contour_totals:
        for entry in contour_totals:
            label = entry.get("contour_label") or f"contour {entry.get('contour_index')}"
            fx, fy = entry.get("net_force", [0.0, 0.0])
            torque = entry.get("net_torque", 0.0)
            print(f"  Net force on {label}: Fx={fx:.6g} N/m, Fy={fy:.6g} N/m, Torque={torque:.6g} N·m/m")
