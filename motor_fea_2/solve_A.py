#!/usr/bin/env python3
"""
solve_A.py
----------
Read a mesh+fields NPZ (from mesh_and_sources.py) and solve:

   -∇·( ν ∇A_z ) = Jz + [∇×M]_z

with homogeneous Neumann on the outer boundary (natural BC).
Permanent magnets contribute via the *surface* bound current
K_b = (M × n̂)·ẑ on PM/non-PM interfaces (edge load).
We pin one node to fix the gauge (A_z up to a constant).

Outputs:
  A_z (per node) written next to the case file.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os
import sys

AIR, PMAG, STEEL = 0, 1, 2

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

def assemble_A_system(nodes, tris, region, mu_r, Mx, My, Jz):
    """Build sparse K and RHS f for -div(nu grad A) = Jz + curl(M)_z + edge(M×n̂)."""
    mu0 = 4e-7*np.pi
    nu_elem = 1.0/(mu0 * mu_r)    # per element
    Nn = nodes.shape[0]
    Ne = tris.shape[0]

    # Element geometry
    P = nodes; T = tris
    x1,y1 = P[T[:,0],0], P[T[:,0],1]
    x2,y2 = P[T[:,1],0], P[T[:,1],1]
    x3,y3 = P[T[:,2],0], P[T[:,2],1]
    # Twice area & area
    twoA = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    A    = 0.5*np.abs(twoA)
    # Grad basis coefficients (b_i, c_i)
    b = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1)  # (Ne,3)
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1)

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

if __name__ == "__main__":
    case_path = sys.argv[1] if len(sys.argv) > 1 else "cases/mag2d_case.npz"
    nodes, tris, region, mu_r, Mx, My, Jz, meta = load_case(case_path)
    K, f = assemble_A_system(nodes, tris, region, mu_r, Mx, My, Jz)
    Az = solve_neumann_pinned(K, f, pin_node=0)

    outpath = case_path.replace(".npz", "_Az.npz")
    np.savez_compressed(outpath, Az=Az, nodes=nodes, tris=tris, meta=meta)
    print(f"Solved A_z on {nodes.shape[0]} nodes, wrote {outpath}\n"
          f"  A_z range: [{Az.min():.6g}, {Az.max():.6g}]")

