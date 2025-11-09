#!/usr/bin/env python3
import numpy as np

# --- Geometry & grid
Lx, Ly   = 0.19, 0.19          # domain size [m]
dx = dy  = 0.05              # cell size [m]  (if running for real, make this ~0.02)
Nx, Ny   = int(Lx/dx), int(Ly/dy)
Nx_n, Ny_n = Nx+1, Ny+1
Nnodes = Nx_n * Ny_n
mu0 = 4*np.pi*1e-7

# --- Materials
mu_r_air, mu_r_mag = 1.0, 1.05
nu_air   = 1.0/(mu0*mu_r_air)
nu_mag   = 1.0/(mu0*mu_r_mag)
nu_cell  = np.full((Nx, Ny), nu_air, dtype=float)   # stored as [i, j] = [x-index, y-index]

# --- One rectangular magnet (origin+size), magnetization M (A/m)
mag_origin = (0.08, 0.06)   # x0, y0 [m]
mag_size   = (0.08, 0.08)   # wx, wy [m]
Br = 1.2                    # Tesla
# Magnetization along +y:
M = np.array([0.0, Br/mu0, 0.0], dtype=float)

# Mark magnet cells (keep mask indexed as [i, j] to match nu_cell)
x_centers = (np.arange(Nx) + 0.5)*dx   # length Nx
y_centers = (np.arange(Ny) + 0.5)*dy   # length Ny
Xc = x_centers[:, None]                # (Nx, 1)
Yc = y_centers[None, :]                # (1, Ny)
x0, y0 = mag_origin
wx, wy = mag_size
mask = (Xc >= x0) & (Xc <= x0 + wx) & (Yc >= y0) & (Yc <= y0 + wy)
nu_cell[mask] = nu_mag

# --- Helpers
def idx(i, j):
    """Node (i,j) -> flat index (row-major with j as row)."""
    return j * Nx_n + i

# Face harmonic means (defined in the same [i, j] convention as nu_cell)
def nu_face_horiz(i_face, j):
    """ν at horizontal face between rows j-1 and j at column i_face (faces run along x)."""
    vals = []
    if 0 <= i_face < Nx and 0 <= j-1 < Ny:
        vals.append(nu_cell[i_face, j-1])
    if 0 <= i_face < Nx and 0 <= j   < Ny:
        vals.append(nu_cell[i_face, j])
    if not vals:
        return nu_air
    return (2*vals[0]*vals[1]/(vals[0]+vals[1])) if len(vals) == 2 else vals[0]

def nu_face_vert(i, j_face):
    """ν at vertical face between columns i-1 and i at row j_face (faces run along y)."""
    vals = []
    if 0 <= i-1 < Nx and 0 <= j_face < Ny:
        vals.append(nu_cell[i-1, j_face])
    if 0 <= i   < Nx and 0 <= j_face < Ny:
        vals.append(nu_cell[i,   j_face])
    if not vals:
        return nu_air
    return (2*vals[0]*vals[1]/(vals[0]+vals[1])) if len(vals) == 2 else vals[0]

# --- Assemble dense K and b
K = np.zeros((Nnodes, Nnodes), dtype=float)
b = np.zeros(Nnodes, dtype=float)

print("K: ", K)
print("b: ", b)

# Dirichlet A=0 on outer boundary (clobber rows to identity)
for j in range(Ny_n):
    for i in range(Nx_n):
        if i in (0, Nx_n-1) or j in (0, Ny_n-1):
            k = idx(i, j)
            K[k, k] = 1.0
            b[k]    = 0.0

# Interior 5-point stencil (matching your earlier scaling: no explicit 1/dx^2, 1/dy^2 factors)
for j in range(1, Ny_n-1):
    for i in range(1, Nx_n-1):
        k = idx(i, j)
        # Skip if this row was a boundary identity (it isn't, by loop limits)
        nuE = nu_face_horiz(i,   j)
        nuW = nu_face_horiz(i-1, j)
        nuN = nu_face_vert (i,   j)
        nuS = nu_face_vert (i,   j-1)

        print("nuE: ", nuE, nu_air, nu_mag)
        diag = nuE + nuW + nuN + nuS

        K[k, k]               += diag
        K[k, idx(i+1, j)]     += -nuE
        K[k, idx(i-1, j)]     += -nuW
        K[k, idx(i,   j+1)]   += -nuN
        K[k, idx(i,   j-1)]   += -nuS

# --- Magnetization edge stamping: K_m = M × n  (we need (K_m)_z)
def stamp_segment(n_hat, n0, n1, length):
    nx, ny = n_hat
    Mx, My, _ = M
    Kz = Mx*ny - My*nx   # A/m
    I  = Kz * length     # A (line current in 2-D)
    b[n0] += 0.5 * I
    b[n1] += 0.5 * I

# Loop magnet cell edges and stamp where neighbor is non-magnet
Ii, Jj = np.where(mask)   # NOTE: here Ii indexes x (i), Jj indexes y (j)
for i, j in zip(Ii, Jj):
    # left edge
    if i == 0 or not mask[i-1, j]:
        stamp_segment((-1.0, 0.0), idx(i,   j),   idx(i,   j+1), dy)
    # right edge
    if i == Nx-1 or not mask[i+1, j]:
        stamp_segment((+1.0, 0.0), idx(i+1, j),   idx(i+1, j+1), dy)
    # bottom edge
    if j == 0 or not mask[i, j-1]:
        stamp_segment((0.0, -1.0), idx(i,   j),   idx(i+1, j),   dx)
    # top edge
    if j == Ny-1 or not mask[i, j+1]:
        stamp_segment((0.0, +1.0), idx(i,   j+1), idx(i+1, j+1), dx)


np.set_printoptions(precision=3, suppress=True, linewidth=200)  # 3 decimals, no sci-notation
print("K:\n", K.toarray().round(3) if hasattr(K, "toarray") else K.round(3))
print("b:\n", b.round(3))


# --- Solve (dense)
A_vec = np.linalg.solve(K, b)
A = A_vec.reshape(Ny_n, Nx_n)  # rows = y index j, cols = x index i

# --- Field recovery at cell centers (Bx = dA/dy, By = -dA/dx)
Bx = ((A[1:, 0:-1] - A[0:-1, 0:-1]) + (A[1:, 1:] - A[0:-1, 1:])) * 0.5 / dy
By = -((A[0:-1, 1:] - A[0:-1, 0:-1]) + (A[1:, 1:] - A[1:, 0:-1])) * 0.5 / dx

print("A shape:", A.shape, "  Bx/By shape:", Bx.shape)
