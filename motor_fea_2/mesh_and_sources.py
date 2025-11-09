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
import numpy as np
import os

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
                 wire_current=10.0,       # A (per unit depth)
                 wire_radius=0.02,        # m
                 seed=0):
    rng = np.random.default_rng(seed)
    Lx, Ly = LxLy
    Ne = tris.shape[0]
    P = nodes

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

    # Permanent magnet: small rectangle, uniform My
    if include_magnet:
        w, h = 0.12*Lx, 0.06*Ly
        cxm, cym = gx - 0.08*Lx, gy
        in_mag = (np.abs(cx - cxm) <= w/2) & (np.abs(cy - cym) <= h/2)
        region[in_mag] = PMAG
        mu_r[in_mag]   = mu_r_magnet
        My[in_mag]     = magnet_My  # flip sign to invert poles

    # Steel insert: another rectangle
    if include_steel:
        w, h = 0.10*Lx, 0.08*Ly
        cxs, cys = gx + 0.10*Lx, gy + 0.05*Ly
        in_st = (np.abs(cx - cxs) <= w/2) & (np.abs(cy - cys) <= h/2)
        # Avoid overwriting PM elements if rectangles overlap
        mask = in_st & (region != PMAG)
        region[mask] = STEEL
        mu_r[mask]   = mu_r_steel

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
        A = tri_areas(nodes, tris)
        A1 = A[in_w1].sum()
        A2 = A[in_w2].sum()
        if A1 > 0:
            Jz[in_w1] += wire_current / A1
        if A2 > 0:
            Jz[in_w2] -= wire_current / A2

    meta = dict(
        Lx=Lx, Ly=Ly,
        include_magnet=bool(include_magnet),
        include_steel=bool(include_steel),
        include_wire=bool(include_wire),
        mu_r_magnet=float(mu_r_magnet),
        mu_r_steel=float(mu_r_steel),
        magnet_My=float(magnet_My),
        wire_current=float(wire_current),
        wire_radius=float(wire_radius)
    )
    return dict(nodes=nodes, tris=tris, region_id=region,
                mu_r=mu_r, Mx=Mx, My=My, Jz=Jz, meta=meta)

def tri_areas(nodes, tris):
    P = nodes; T = tris
    x1,y1 = P[T[:,0],0], P[T[:,0],1]
    x2,y2 = P[T[:,1],0], P[T[:,1],1]
    x3,y3 = P[T[:,2],0], P[T[:,2],1]
    return 0.5*np.abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))

if __name__ == "__main__":
    # Build a default case and write it out
    nodes, tris, L = square_tri_mesh(Nx=100, Ny=100, Lx=1.0, Ly=1.0)
    payload = build_fields(nodes, tris, L,
                           include_magnet=True,
                           include_steel=True,
                           include_wire=True)
    os.makedirs("cases", exist_ok=True)
    outpath = "cases/mag2d_case.npz"
    np.savez_compressed(outpath, **payload)
    print(f"Wrote {outpath} with:",
          f"\n  nodes={payload['nodes'].shape}",
          f"\n  tris={payload['tris'].shape}",
          f"\n  mu_r:  [{payload['mu_r'].min():.3g}, {payload['mu_r'].max():.3g}]",
          f"\n  |M|:   [{np.hypot(payload['Mx'],payload['My']).min():.3g}, "
          f"{np.hypot(payload['Mx'],payload['My']).max():.3g}] A/m",
          f"\n  Jz sum={payload['Jz'].sum()* (1.0 * 1.0):.6g} A (approx)")

