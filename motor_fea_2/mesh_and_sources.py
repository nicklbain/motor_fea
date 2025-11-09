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
from pathlib import Path

import numpy as np

AIR, PMAG, STEEL = 0, 1, 2

def square_tri_mesh(Nx=100, Ny=100, Lx=1.0, Ly=1.0):
    """Structured square grid split into two CCW triangles per cell."""
    x = np.linspace(0, Lx, Nx)
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

def build_fields(nodes, tris, LxLy,
                 include_magnet=True,
                 include_steel=True,
                 include_wire=True,
                 magnet_My=8e5,           # A/m (μ0*M ~ 1.0 T)
                 mu_r_magnet=1.05,
                 mu_r_steel=1000.0,
                 wire_current=5000.0,     # A (per unit depth)
                 wire_radius=0.02,        # m
                 seed=0):
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
    return dict(nodes=nodes, tris=tris, region_id=region,
                mu_r=mu_r, Mx=Mx, My=My, Jz=Jz, meta=meta)

def tri_areas(nodes, tris):
    P = nodes; T = tris
    x1,y1 = P[T[:,0],0], P[T[:,0],1]
    x2,y2 = P[T[:,1],0], P[T[:,1],1]
    x3,y3 = P[T[:,2],0], P[T[:,2],1]
    return 0.5*np.abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))


def rect_overlap_fraction(nodes, tris, cx, cy, width, height, tri_area=None):
    """Return area fraction of each triangle covered by an axis-aligned rectangle."""
    if width <= 0 or height <= 0:
        return np.zeros(tris.shape[0], dtype=float)
    xmin = cx - width / 2
    xmax = cx + width / 2
    ymin = cy - height / 2
    ymax = cy + height / 2
    tri_pts = nodes[tris]
    areas = tri_area if tri_area is not None else tri_areas(nodes, tris)
    fractions = np.zeros(tris.shape[0], dtype=float)
    for idx, pts in enumerate(tri_pts):
        area = polygon_rect_overlap_area(pts, xmin, xmax, ymin, ymax)
        if areas[idx] > 0:
            fractions[idx] = area / areas[idx]
    return fractions


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


if __name__ == "__main__":
    args = _parse_args()

    nodes, tris, L = square_tri_mesh(Nx=args.Nx, Ny=args.Ny, Lx=args.Lx, Ly=args.Ly)
    payload = build_fields(nodes, tris, L,
                           include_magnet=not args.no_magnet,
                           include_steel=not args.no_steel,
                           include_wire=not args.no_wire,
                           magnet_My=args.magnet_My,
                           mu_r_magnet=args.mu_r_magnet,
                           mu_r_steel=args.mu_r_steel,
                           wire_current=args.wire_current,
                           wire_radius=args.wire_radius)
    case_dir = Path(args.cases_dir) / args.case
    outpath = _write_case(payload, case_dir)
    print(f"Wrote {outpath} with:",
          f"\n  nodes={payload['nodes'].shape}",
          f"\n  tris={payload['tris'].shape}",
          f"\n  mu_r:  [{payload['mu_r'].min():.3g}, {payload['mu_r'].max():.3g}]",
          f"\n  |M|:   [{np.hypot(payload['Mx'],payload['My']).min():.3g}, "
          f"{np.hypot(payload['Mx'],payload['My']).max():.3g}] A/m",
          f"\n  Jz sum={payload['Jz'].sum()* (args.Lx * args.Ly):.6g} A (approx)")
