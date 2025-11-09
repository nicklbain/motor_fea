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
Lx_total = 0.4      # [m] total width of the modelled window
Ly_total = 0.25      # [m] total height of the modelled window
grid_size = 0.0005     # [m] nominal square cell edge length (dx = dy here)

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
mu_r_magnet = 1.05
nu_magnet = 1.0 / (mu0 * mu_r_magnet)

# Magnet definitions in physical coordinates (lower-left corner + size)
Br = 2.0  # Tesla
M_default = np.array([Br / mu0, 0.0, 0.0])  # +y

magnet_1_origin = (0.16, 0.12)   # [m] (x0, y0)
magnet_1_size = (0.01, 0.01)     # [m] (width_x, height_y)

include_second_magnet = True
magnet_2_origin = (0.2, 0.12)   # [m] (x0, y0)
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


# My understanding is that most everything we've done by this point is just assigning either nu_magnet or nu_air to each cell
# Basically finding this array:
#     nu_cell[node] = assigned nu
# Seems like this could be significantly improved because grid doesn't necessarily align with magnet locations
# Seems much better to mesh in a smart way and then assign nu_air and nu_magnet to each node


# Allocate K and b
if sp is not None:
    csr_rows = []
    csr_cols = []
    csr_data = []
else:
    K = np.zeros((Nnodes, Nnodes), dtype=float) # Seems important to note that K appears to be an Nnodes * Nnodes matrix 
    # Going from 5x5 to 10x10 -> array with: (5*5*5*5) -> (10*10*10*10) elements (explodes)

b = np.zeros(Nnodes, dtype=float)


# Not totally sure but seems to get the nu on a face, and if face is on two different nu's -> does some sort of combination of two (how?)
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
for j in range(Ny_nodes):
    for i in range(Nx_nodes):
        if i == 0 or i == Nx_nodes - 1 or j == 0 or j == Ny_nodes - 1:
            idx = node_index(i, j)
            if sp is not None:
                csr_rows.append(idx)
                csr_cols.append(idx)
                csr_data.append(1.0)
            else:
                K[idx, idx] = 1.0  # Dirichlet A=0
            b[idx] = 0.0

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


# K is only generated here
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

# Solve
if sp is not None:
    A_vec = spla.spsolve(K, b)
    print("[info] Solved linear system with SciPy sparse solver (spsolve).")
else:
    raise RuntimeError(
        "SciPy sparse is required for this grid size. Install scipy or lower Nx, Ny."
    )
A = A_vec.reshape((Ny_nodes, Nx_nodes))

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