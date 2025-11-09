#!/usr/bin/env python3
"""
magnetostatics_solver.py
------------------------
Vector-potential magnetostatics demo on a 2-D grid (node-based Az).
- Unknowns: Az at nodes
- Materials: nu (=1/mu) per cell
- Magnetization: uniform rectangular region defined in physical space;
  it is stamped as sheet currents on the cell edges enclosing the region.

The domain is defined via total x/y extents and a target grid spacing so you
can easily adjust the discretisation without touching cell counts directly.

Usage: python3 magnetostatics_solver.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception as exc:  # pragma: no cover - SciPy optional
    sp = None
    spla = None
    SCIPY_IMPORT_ERROR = exc
else:
    SCIPY_IMPORT_ERROR = None

ENABLE_PLOTTING = False  # toggle to skip figure rendering when profiling performance

# -----------------------------
# Problem setup
# -----------------------------
Lx_total = 0.2      # [m] total width of the modelled window
Ly_total = 0.10      # [m] total height of the modelled window
grid_size = 0.0005     # [m] nominal square cell edge length (dx = dy here)

grid_override = os.getenv("MAGNETO_GRID_SIZE")
if grid_override is not None:
    try:
        grid_size = float(grid_override)
        print(f"[info] Using MAGNETO_GRID_SIZE override: {grid_size:.6f} m")
    except ValueError:
        print(f"[warn] Invalid MAGNETO_GRID_SIZE={grid_override!r}; using default {grid_size:.6f} m.")

# Round the domain to an integer number of cells based on the requested spacing.
Nx = max(1, int(np.ceil(Lx_total / grid_size)))
Ny = max(1, int(np.ceil(Ly_total / grid_size)))
dx = grid_size
dy = grid_size

Lx = Nx * dx
Ly = Ny * dy

if not np.isclose(Lx, Lx_total):
    print(f"[info] Rounded Lx from {Lx_total:.6f} m to {Lx:.6f} m to match {Nx} cells.")
if not np.isclose(Ly, Ly_total):
    print(f"[info] Rounded Ly from {Ly_total:.6f} m to {Ly:.6f} m to match {Ny} cells.")

Nx_nodes = Nx + 1
Ny_nodes = Ny + 1
Nnodes = Nx_nodes * Ny_nodes

mu0 = 4.0 * np.pi * 1e-7
nu_air = 1.0 / mu0
nu0 = nu_air
mu_r_magnet = 1.05
nu_magnet = 1.0 / (mu0 * mu_r_magnet)

# Magnet definitions in physical coordinates (lower-left corner + size)
Br = 2.0  # Tesla
M_default = np.array([0.0, Br / mu0, 0.0])  # +y

magnet_1_origin = (0.04, 0.05)   # [m] (x0, y0)
magnet_1_size = (0.01, 0.01)     # [m] (width_x, height_y)

include_second_magnet = True
magnet_2_origin = (0.08, 0.05)   # [m] (x0, y0)
magnet_2_size = (0.01, 0.01)     # [m] (width_x, height_y)

magnet_configs = [
    {
        "name": "Magnet 1",
        "origin": magnet_1_origin,
        "size": magnet_1_size,
        "magnetization": M_default,
    },
]

if include_second_magnet:
    magnet_configs.append(
        {
            "name": "Magnet 2",
            "origin": magnet_2_origin,
            "size": magnet_2_size,
            "magnetization": M_default,
        }
    )

def compute_mst_force_torque(Bx, By, dx, dy, magnet_extent, r0=None):
    """
    Compute (Fx, Fy, tau_z) per unit depth for a rectangular magnetized region
    using the Maxwell stress tensor integrated over concentric loops in air.
    """
    Bx = np.asarray(Bx, dtype=float)
    By = np.asarray(By, dtype=float)
    if Bx.shape != By.shape:
        raise ValueError("Bx and By must share the same shape.")

    Ny, Nx = Bx.shape
    if Nx < 2 or Ny < 2:
        raise ValueError("Field arrays must be at least 2x2 for bilinear interpolation.")

    Lx = Nx * dx
    Ly = Ny * dy
    x_min, x_max, y_min, y_max = magnet_extent

    if r0 is None:
        r0 = (0.5 * (x_min + x_max), 0.5 * (y_min + y_max))
    r0x, r0y = r0

    margin_x = 0.5 * dx
    margin_y = 0.5 * dy

    def bilinear_interp(field, x, y):
        xi = (x / dx) - 0.5
        yi = (y / dy) - 0.5
        i0 = int(np.floor(xi))
        j0 = int(np.floor(yi))
        i0 = np.clip(i0, 0, Nx - 2)
        j0 = np.clip(j0, 0, Ny - 2)
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

    # Offsets (in cells) for concentric loops to reduce numerical noise.
    offsets = [
        (0.5 * dx, 0.5 * dy),
        (1.0 * dx, 1.0 * dy),
        (1.5 * dx, 1.5 * dy),
    ]

    results = []
    max_cell_size = max(dx, dy)

    for offset_x, offset_y in offsets:
        x_left = x_min - offset_x
        x_right = x_max + offset_x
        y_bottom = y_min - offset_y
        y_top = y_max + offset_y

        # Skip loops that would touch the outer boundary or leave the interpolation domain.
        if (
            x_left < margin_x
            or x_right > Lx - margin_x
            or y_bottom < margin_y
            or y_top > Ly - margin_y
        ):
            continue

        Fx = 0.0
        Fy = 0.0
        tau_z = 0.0

        edges = [
            ((x_left, y_bottom), (x_right, y_bottom), (0.0, -1.0)),  # bottom edge
            ((x_right, y_bottom), (x_right, y_top), (1.0, 0.0)),    # right edge
            ((x_right, y_top), (x_left, y_top), (0.0, 1.0)),        # top edge
            ((x_left, y_top), (x_left, y_bottom), (-1.0, 0.0)),     # left edge
        ]

        for (start, end, normal) in edges:
            sx, sy = start
            ex, ey = end
            length = np.hypot(ex - sx, ey - sy)
            if length <= 0.0:
                continue

            n_samples = max(4, int(np.ceil(length / max_cell_size)))
            ds = length / n_samples
            nx, ny = normal

            for k in range(n_samples):
                t = (k + 0.5) / n_samples
                x = sx + (ex - sx) * t
                y = sy + (ey - sy) * t

                Bx_val = bilinear_interp(Bx, x, y)
                By_val = bilinear_interp(By, x, y)

                Bsq = Bx_val * Bx_val + By_val * By_val
                Txx = (Bx_val * Bx_val - 0.5 * Bsq) / mu0
                Tyy = (By_val * By_val - 0.5 * Bsq) / mu0
                Txy = (Bx_val * By_val) / mu0  # identical to Tyx

                tx = Txx * nx + Txy * ny
                ty = Txy * nx + Tyy * ny

                Fx += tx * ds
                Fy += ty * ds
                tau_z += ((x - r0x) * ty - (y - r0y) * tx) * ds

        results.append((Fx, Fy, tau_z))

    if not results:
        raise ValueError(
            "No valid integration loops found; ensure the magnet is sufficiently far from boundaries."
        )

    results = np.asarray(results)
    Fx_mean = float(np.mean(results[:, 0]))
    Fy_mean = float(np.mean(results[:, 1]))
    tau_mean = float(np.mean(results[:, 2]))
    return Fx_mean, Fy_mean, tau_mean

def node_index(i, j):
    return j * Nx_nodes + i

def node_coords(idx):
    i = idx % Nx_nodes
    j = idx // Nx_nodes
    return i, j

def ordered_boundary_nodes():
    """Return boundary node indices in CCW order without duplicating the start node."""
    nodes = []
    # Bottom edge (left to right)
    for i in range(Nx_nodes):
        nodes.append(node_index(i, 0))
    # Right edge (bottom to top, skipping bottom corner)
    for j in range(1, Ny_nodes):
        nodes.append(node_index(Nx_nodes - 1, j))
    # Top edge (right to left, skipping top-right corner)
    for i in range(Nx_nodes - 2, -1, -1):
        nodes.append(node_index(i, Ny_nodes - 1))
    # Left edge (top to bottom, skipping both corners)
    for j in range(Ny_nodes - 2, 0, -1):
        nodes.append(node_index(0, j))
    return np.array(nodes, dtype=int)

def compute_boundary_geometry(boundary_nodes):
    """Compute boundary segment midpoints, lengths, and normals."""
    Nb = boundary_nodes.size
    node_points = np.zeros((Nb, 2), dtype=float)
    for k, idx in enumerate(boundary_nodes):
        i, j = node_coords(idx)
        node_points[k, 0] = i * dx
        node_points[k, 1] = j * dy

    seg_midpoints = np.zeros((Nb, 2), dtype=float)
    seg_lengths = np.zeros(Nb, dtype=float)
    seg_normals = np.zeros((Nb, 2), dtype=float)

    for k in range(Nb):
        start_pt = node_points[k]
        end_pt = node_points[(k + 1) % Nb]
        vec = end_pt - start_pt
        length = np.hypot(vec[0], vec[1])
        if length <= 0.0:
            raise ValueError("Degenerate boundary segment encountered.")
        tangent = vec / length
        normal = np.array([tangent[1], -tangent[0]])
        seg_midpoints[k] = 0.5 * (start_pt + end_pt)
        seg_lengths[k] = length
        seg_normals[k] = normal

    node_normals = np.zeros((Nb, 2), dtype=float)
    for k in range(Nb):
        n_prev = seg_normals[(k - 1) % Nb]
        n_curr = seg_normals[k]
        n_vec = n_prev + n_curr
        norm = np.hypot(n_vec[0], n_vec[1])
        if norm <= 0.0:
            n_vec = n_curr
            norm = np.hypot(n_vec[0], n_vec[1])
        node_normals[k] = n_vec / norm

    return node_points, seg_midpoints, seg_lengths, seg_normals, node_normals

def build_normal_derivative_operators(boundary_nodes, node_normals, interior_lookup, boundary_lookup):
    """Assemble one-sided interior normal derivative matrices Dbi and Dbb."""
    Nb = boundary_nodes.size
    Ni = len(interior_lookup)
    Dbi = np.zeros((Nb, Ni), dtype=float)
    Dbb = np.zeros((Nb, Nb), dtype=float)

    for row, g_idx in enumerate(boundary_nodes):
        i, j = node_coords(g_idx)
        is_left = i == 0
        is_right = i == Nx_nodes - 1
        is_bottom = j == 0
        is_top = j == Ny_nodes - 1

        if (is_left or is_right) and (is_bottom or is_top):
            di = 1 if is_left else -1
            dj = 1 if is_bottom else -1
            neighbor_idx = node_index(i + di, j + dj)
            distance = np.hypot(di * dx, dj * dy)
            coeff = 1.0 / distance
            if neighbor_idx in boundary_lookup:
                Dbb[row, boundary_lookup[neighbor_idx]] -= coeff
            else:
                Dbi[row, interior_lookup[neighbor_idx]] -= coeff
            Dbb[row, row] += coeff
            continue

        if is_left or is_right:
            di = 1 if is_left else -1
            neighbor_idx = node_index(i + di, j)
            distance = abs(di) * dx
            coeff = 1.0 / distance
            if neighbor_idx in boundary_lookup:
                Dbb[row, boundary_lookup[neighbor_idx]] -= coeff
            else:
                Dbi[row, interior_lookup[neighbor_idx]] -= coeff
            Dbb[row, row] += coeff
            continue

        if is_bottom or is_top:
            dj = 1 if is_bottom else -1
            neighbor_idx = node_index(i, j + dj)
            distance = abs(dj) * dy
            coeff = 1.0 / distance
            if neighbor_idx in boundary_lookup:
                Dbb[row, boundary_lookup[neighbor_idx]] -= coeff
            else:
                Dbi[row, interior_lookup[neighbor_idx]] -= coeff
            Dbb[row, row] += coeff
            continue

        raise RuntimeError("Boundary node classification failed.")

    return Dbi, Dbb

def assemble_boundary_integrals(boundary_points, seg_midpoints, seg_lengths, node_normals):
    """Assemble dense V and K' matrices via midpoint collocation for 2-D Laplace."""
    Nb = boundary_points.shape[0]
    V = np.zeros((Nb, Nb), dtype=float)
    Kprime = np.zeros((Nb, Nb), dtype=float)
    const = -1.0 / (2.0 * np.pi)

    for i in range(Nb):
        x_i = boundary_points[i]
        n_i = node_normals[i]
        for j in range(Nb):
            y_mid = seg_midpoints[j]
            r_vec = x_i - y_mid
            r_norm = np.hypot(r_vec[0], r_vec[1])
            if r_norm <= 0.0:
                raise ValueError("Coincident collocation and source points encountered.")
            kernel = const * np.log(r_norm)
            V[i, j] = kernel * seg_lengths[j]
            grad_dot_n = const * np.dot(r_vec, n_i) / (r_norm * r_norm)
            Kprime[i, j] = grad_dot_n * seg_lengths[j]

    return V, Kprime

# Per-cell nu (1/mu); start with air everywhere.
nu_cell = np.full((Nx, Ny), nu_air, dtype=float)

# Determine which cells fall inside the magnet by checking their centres.
x_centres = (np.arange(Nx) + 0.5) * dx
y_centres = (np.arange(Ny) + 0.5) * dy
Xc = x_centres[:, None]
Yc = y_centres[None, :]

if not magnet_configs:
    raise ValueError("At least one magnet configuration is required.")

combined_magnet_mask = np.zeros((Nx, Ny), dtype=bool)

for magnet in magnet_configs:
    x0, y0 = magnet["origin"]
    mx, my = magnet["size"]
    x1 = x0 + mx
    y1 = y0 + my

    mask = (Xc >= x0) & (Xc <= x1) & (Yc >= y0) & (Yc <= y1)
    if not np.any(mask):
        raise ValueError(
            f"{magnet['name']} definition does not cover any cells; "
            "adjust origin/size or grid_size."
        )

    magnet_i_idx, magnet_j_idx = np.where(mask)
    magnet["mask"] = mask
    magnet["cell_indices"] = (magnet_i_idx, magnet_j_idx)
    magnet["extent_grid"] = (
        magnet_i_idx.min() * dx,
        (magnet_i_idx.max() + 1) * dx,
        magnet_j_idx.min() * dy,
        (magnet_j_idx.max() + 1) * dy,
    )

    combined_magnet_mask |= mask
    nu_cell[mask] = nu_magnet

if not np.any(combined_magnet_mask):
    raise ValueError("Magnet definitions do not cover any cells; adjust configuration.")

# Allocate K and b
if sp is not None:
    csr_rows = []
    csr_cols = []
    csr_data = []
else:
    K = np.zeros((Nnodes, Nnodes), dtype=float)
b = np.zeros(Nnodes, dtype=float)

def nu_face_horiz(i_face, j):
    vals = []
    if 0 <= i_face < Nx and 0 <= j-1 < Ny:
        vals.append(nu_cell[i_face, j-1])
    if 0 <= i_face < Nx and 0 <= j < Ny:
        vals.append(nu_cell[i_face, j])
    if not vals:
        return nu_air
    if len(vals) == 2:
        a, c = vals
        return 2.0 * a * c / (a + c)
    return vals[0]

def nu_face_vert(i, j_face):
    vals = []
    if 0 <= i-1 < Nx and 0 <= j_face < Ny:
        vals.append(nu_cell[i-1, j_face])
    if 0 <= i < Nx and 0 <= j_face < Ny:
        vals.append(nu_cell[i, j_face])
    if not vals:
        return nu_air
    if len(vals) == 2:
        a, c = vals
        return 2.0 * a * c / (a + c)
    return vals[0]

# Assemble
for j in range(1, Ny_nodes - 1):
    for i in range(1, Nx_nodes - 1):
        idx = node_index(i, j)
        nu_e = nu_face_horiz(i, j)
        nu_w = nu_face_horiz(i-1, j)
        nu_n = nu_face_vert(i, j)
        nu_s = nu_face_vert(i, j-1)

        diag = nu_e + nu_w + nu_n + nu_s
        if sp is not None:
            csr_rows.extend(
                [
                    idx,
                    idx,
                    idx,
                    idx,
                    idx,
                ]
            )
            csr_cols.extend(
                [
                    idx,
                    node_index(i+1, j),
                    node_index(i-1, j),
                    node_index(i, j+1),
                    node_index(i, j-1),
                ]
            )
            csr_data.extend(
                [
                    diag,
                    -nu_e,
                    -nu_w,
                    -nu_n,
                    -nu_s,
                ]
            )
        else:
            K[idx, idx] += diag
            K[idx, node_index(i+1, j)] += -nu_e
            K[idx, node_index(i-1, j)] += -nu_w
            K[idx, node_index(i, j+1)] += -nu_n
            K[idx, node_index(i, j-1)] += -nu_s

if sp is not None:
    K = sp.csr_matrix(
        (np.array(csr_data), (np.array(csr_rows), np.array(csr_cols))),
        shape=(Nnodes, Nnodes),
    )

# Sheet-current stamping
def stamp_edge(M_vec, n_hat, n0, n1, edge_length):
    nx, ny = n_hat
    Mx, My, _ = M_vec
    Kz = Mx*ny - My*nx    # A/m
    I_edge = Kz * edge_length      # line current [A]
    b[n0] += 0.5 * I_edge
    b[n1] += 0.5 * I_edge

# Stamp each boundary edge around the magnetised regions.
for magnet in magnet_configs:
    mask = magnet["mask"]
    M_vec = magnet["magnetization"]
    magnet_i_idx, magnet_j_idx = magnet["cell_indices"]
    for i, j in zip(magnet_i_idx, magnet_j_idx):
        # Left face
        if i == 0 or not mask[i-1, j]:
            stamp_edge(
                M_vec,
                n_hat=(-1.0, 0.0),
                n0=node_index(i, j),
                n1=node_index(i, j+1),
                edge_length=dy,
            )
        # Right face
        if i == Nx-1 or not mask[i+1, j]:
            stamp_edge(
                M_vec,
                n_hat=(+1.0, 0.0),
                n0=node_index(i+1, j),
                n1=node_index(i+1, j+1),
                edge_length=dy,
            )
        # Bottom face
        if j == 0 or not mask[i, j-1]:
            stamp_edge(
                M_vec,
                n_hat=(0.0, -1.0),
                n0=node_index(i, j),
                n1=node_index(i+1, j),
                edge_length=dx,
            )
        # Top face
        if j == Ny-1 or not mask[i, j+1]:
            stamp_edge(
                M_vec,
                n_hat=(0.0, +1.0),
                n0=node_index(i, j+1),
                n1=node_index(i+1, j+1),
                edge_length=dx,
            )

# Boundary traces and coupling operators
boundary_nodes = ordered_boundary_nodes()
is_boundary = np.zeros(Nnodes, dtype=bool)
is_boundary[boundary_nodes] = True
interior_nodes = np.nonzero(~is_boundary)[0]

boundary_lookup = {idx: pos for pos, idx in enumerate(boundary_nodes)}
interior_lookup = {idx: pos for pos, idx in enumerate(interior_nodes)}

boundary_points, seg_midpoints, seg_lengths, _, node_normals = compute_boundary_geometry(boundary_nodes)
V_dense, Kprime_dense = assemble_boundary_integrals(boundary_points, seg_midpoints, seg_lengths, node_normals)
Dbi, Dbb = build_normal_derivative_operators(boundary_nodes, node_normals, interior_lookup, boundary_lookup)

Ni = interior_nodes.size
Nb = boundary_nodes.size

if Ni == 0 or Nb == 0:
    raise RuntimeError("Invalid node partitioning; boundary coupling requires interior and boundary nodes.")

if sp is None:
    raise RuntimeError(
        "SciPy sparse is required for this grid size. Install scipy or lower Nx, Ny."
    )

Kii = K[interior_nodes][:, interior_nodes]
Kib = K[interior_nodes][:, boundary_nodes]
fi = b[interior_nodes]

zero_ib = sp.csr_matrix((Ni, Nb))
zero_bi = sp.csr_matrix((Nb, Ni))
I_nb = sp.identity(Nb, format="csr")

V_mat = sp.csr_matrix(V_dense)
Kp = sp.csr_matrix(Kprime_dense)
BEM_flux = nu0 * (-0.5 * I_nb + Kp)
Dbi_mat = nu0 * sp.csr_matrix(Dbi)
Dbb_mat = nu0 * sp.csr_matrix(Dbb)

w = sp.csr_matrix(seg_lengths.reshape(1, -1))  # 1×Nb
zero_Ni1 = sp.csr_matrix((1, Ni))
zero_Nb1 = sp.csr_matrix((1, Nb))
zero_11 = sp.csr_matrix((1, 1))  # zero block for Lagrange multiplier
col_lambda_ui = sp.csr_matrix((Ni, 1))
col_lambda_ub = sp.csr_matrix((Nb, 1))
col_lambda_sigma = w.T

block_matrix = sp.bmat(
    [
        [Kii, Kib, zero_ib, col_lambda_ui],
        [zero_bi, I_nb, -V_mat, col_lambda_ub],
        [Dbi_mat, Dbb_mat, BEM_flux, col_lambda_sigma],
        [zero_Ni1, zero_Nb1, w, zero_11],
    ],
    format="csr",
)
rhs = np.concatenate([fi, np.zeros(Nb), np.zeros(Nb), np.zeros(1)])

solution = spla.spsolve(block_matrix, rhs)
ui = solution[:Ni]
ub = solution[Ni:Ni+Nb]
sigma = solution[Ni+Nb:Ni+2*Nb]
zeta = solution[-1]  # Lagrange multiplier for decay constraint

A_vec = np.zeros(Nnodes, dtype=float)
A_vec[interior_nodes] = ui
A_vec[boundary_nodes] = ub
A = A_vec.reshape((Ny_nodes, Nx_nodes))
print("[info] Solved coupled FEM–BEM system (spsolve).")

# Compute B at cell centers
Bx = np.zeros((Ny, Nx))
By = np.zeros((Ny, Nx))
for j in range(Ny):
    for i in range(Nx):
        dA_dy_left  = (A[j+1, i]   - A[j,   i])   / dy
        dA_dy_right = (A[j+1, i+1] - A[j, i+1])  / dy
        Bx[j, i] = 0.5 * (dA_dy_left + dA_dy_right)

        dA_dx_bottom = (A[j,   i+1] - A[j,   i]) / dx
        dA_dx_top    = (A[j+1, i+1] - A[j+1, i]) / dx
        By[j, i] = -0.5 * (dA_dx_bottom + dA_dx_top)

Bmag = np.sqrt(Bx**2 + By**2)

xc = (np.arange(Nx) + 0.5) * dx
yc = (np.arange(Ny) + 0.5) * dy
XX, YY = np.meshgrid(xc, yc, indexing="xy")

# Export B-field snapshot for downstream processing
output_path = Path("magnetostatics_solver_field_data.npz")
np.savez(
    output_path,
    x_centers=xc,
    y_centers=yc,
    x_grid=XX,
    y_grid=YY,
    Bx=Bx,
    By=By,
    Bmag=Bmag,
    dx=np.array(dx),
    dy=np.array(dy),
    Lx=np.array(Lx),
    Ly=np.array(Ly),
)
print(f"[info] B-field data exported to {output_path.resolve()}")

if ENABLE_PLOTTING:
    # -----------------------------
    # Multi-subplot figure (for local use)
    # -----------------------------
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle("Magnetostatics demo: A_z and B fields")

    im0 = axs[0,0].imshow(A, origin='lower', extent=[0, Lx, 0, Ly])
    axs[0,0].set_title("A_z at nodes")
    axs[0,0].set_xlabel("x (m)"); axs[0,0].set_ylabel("y (m)")
    fig.colorbar(im0, ax=axs[0,0])

    im1 = axs[0,1].imshow(Bmag, origin='lower', extent=[0, Lx, 0, Ly])
    axs[0,1].set_title("|B| at cell centers")
    axs[0,1].set_xlabel("x (m)"); axs[0,1].set_ylabel("y (m)")
    fig.colorbar(im1, ax=axs[0,1])

    # Uniform-length quiver colored by |B|
    target_vectors_per_axis = 15
    step = max(1, int(np.ceil(max(Nx, Ny) / target_vectors_per_axis)))
    Xq = XX[::step, ::step]
    Yq = YY[::step, ::step]
    Bx_sample = Bx[::step, ::step]
    By_sample = By[::step, ::step]
    Bmag_sample = np.sqrt(Bx_sample**2 + By_sample**2)

    # Avoid division by zero at near-zero field points
    direction_scale = np.where(Bmag_sample > 0.0, Bmag_sample, 1.0)
    Ux_dir = np.divide(Bx_sample, direction_scale, out=np.zeros_like(Bx_sample), where=direction_scale > 0.0)
    Uy_dir = np.divide(By_sample, direction_scale, out=np.zeros_like(By_sample), where=direction_scale > 0.0)

    arrow_len = 0.6 * step * min(dx, dy)
    Uq = Ux_dir * arrow_len
    Vq = Uy_dir * arrow_len

    quiv_norm = Normalize(vmin=Bmag.min(), vmax=Bmag.max())
    quiv = axs[0,2].quiver(
        Xq,
        Yq,
        Uq,
        Vq,
        Bmag_sample,
        cmap="viridis",
        norm=quiv_norm,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        pivot="mid",
    )
    axs[0,2].set_title("B direction (uniform arrows)")
    axs[0,2].set_xlabel("x (m)"); axs[0,2].set_ylabel("y (m)")
    fig.colorbar(quiv, ax=axs[0,2], label="|B| (T)")

    im2 = axs[1,0].imshow(Bx, origin='lower', extent=[0, Lx, 0, Ly])
    axs[1,0].set_title("B_x")
    axs[1,0].set_xlabel("x (m)"); axs[1,0].set_ylabel("y (m)")
    fig.colorbar(im2, ax=axs[1,0])

    im3 = axs[1,1].imshow(By, origin='lower', extent=[0, Lx, 0, Ly])
    axs[1,1].set_title("B_y")
    axs[1,1].set_xlabel("x (m)"); axs[1,1].set_ylabel("y (m)")
    fig.colorbar(im3, ax=axs[1,1])

    axs[1,2].axis('off')
    axs[1,2].text(0.05, 0.9, "Grid: {}x{} cells\nCell size: {:.3f} m\nDomain: {:.3f} x {:.3f} m".format(
        Nx, Ny, dx, Lx, Ly))
    magnet_info_lines = []
    for magnet in magnet_configs:
        mx, my = magnet["size"]
        x0, y0 = magnet["origin"]
        magnet_info_lines.append(
            "{} origin (m): ({:.3f}, {:.3f})\n   size (m): ({:.3f}, {:.3f})".format(
                magnet["name"], x0, y0, mx, my
            )
        )
    axs[1,2].text(0.05, 0.74, "Magnets:\n" + "\n".join(magnet_info_lines), va="top")
    axs[1,2].text(0.05, 0.32, "mu_r(magnet) = {:.2f}".format(mu_r_magnet))
    axs[1,2].text(0.05, 0.24, "M = {:.3e} A/m (along +y)".format(M_default[1]))

    plt.tight_layout()
    plt.show()
else:
    print("[info] Plotting disabled (ENABLE_PLOTTING=False).")

# Print small summaries
print("Solved A_z (nodes):", A.shape, "  B fields (cells):", Bx.shape)
for magnet in magnet_configs:
    extent = magnet["extent_grid"]
    print(
        "{} grid-aligned extent: x=[{:.3f}, {:.3f}] m, y=[{:.3f}, {:.3f}] m".format(
            magnet["name"], extent[0], extent[1], extent[2], extent[3]
        )
    )

    try:
        Fx_mst, Fy_mst, tau_mst = compute_mst_force_torque(Bx, By, dx, dy, extent)
        print(
            "  Maxwell stress: Fx = {:.4e} N/m, Fy = {:.4e} N/m, tau_z = {:.4e} N·m/m".format(
                Fx_mst, Fy_mst, tau_mst
            )
        )
    except ValueError as exc:
        print(f"  Maxwell stress computation skipped: {exc}")

    magnet_i_idx, magnet_j_idx = magnet["cell_indices"]
    if magnet_i_idx.size:
        mid_idx = magnet_i_idx.size // 2
        i_mid = magnet_i_idx[mid_idx]
        j_mid = magnet_j_idx[mid_idx]
        B_sample = np.sqrt(Bx[j_mid, i_mid]**2 + By[j_mid, i_mid]**2)
        print(
            "  Representative magnet cell (Bx, By, |B|):",
            Bx[j_mid, i_mid],
            By[j_mid, i_mid],
            B_sample,
        )
