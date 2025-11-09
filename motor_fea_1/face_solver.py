#!/usr/bin/env python3
# Minimal, readable div-free completion on a 10x10x10 MAC grid
# - Face-normal unknowns: Bx(i+1/2,j,k), By(i,j+1/2,k), Bz(i,j,k+1/2)
# - Two interior z-faces are fixed: +1 and -1 on the center cell's bottom/top faces
# - All outer-box faces are fixed to 0
# - We iteratively reduce divergence using a Jacobi Poisson solve on cell centers
#   and subtract the discrete gradient from *unfixed* faces.

import numpy as np

# -------------------------
# Grid sizes (cells)
# -------------------------
Nx = Ny = Nz = 10

# Face arrays (face-normal components)
Bx = np.zeros((Nx+1, Ny,   Nz  ), dtype=float)
By = np.zeros((Nx,   Ny+1, Nz  ), dtype=float)
Bz = np.zeros((Nx,   Ny,   Nz+1), dtype=float)

# Masks of fixed (pinned) faces: True means "do not update this face"
fixed_x = np.zeros_like(Bx, dtype=bool)
fixed_y = np.zeros_like(By, dtype=bool)
fixed_z = np.zeros_like(Bz, dtype=bool)

# Pin all outer-box faces to 0
fixed_x[ 0, :, :] = True
fixed_x[-1, :, :] = True
fixed_y[:,  0, :] = True
fixed_y[:, -1, :] = True
fixed_z[:, :,  0] = True
fixed_z[:, :, -1] = True

# Pin two *interior* z-faces on the center cell
i0, j0, k0 = Nx//2, Ny//2, Nz//2   # center cell index
Bz[i0, j0, k0    ] = +1.0          # bottom face of that cell
Bz[i0, j0, k0 + 1] = -1.0          # top face of that cell
fixed_z[i0, j0, k0    ] = True
fixed_z[i0, j0, k0 + 1] = True

# -------------------------
# Helpers: divergence, Jacobi Poisson solve, and one projection step
# -------------------------
def divergence(Bx, By, Bz):
    """Cell-centered divergence on an (Nx,Ny,Nz) array."""
    return (Bx[1:, :, :] - Bx[:-1, :, :]) \
         + (By[:, 1:, :] - By[:, :-1, :]) \
         + (Bz[:, :, 1:] - Bz[:, :, :-1])

def solve_poisson_jacobi(rhs, iters=200):
    """
    Solve Δp = rhs on an (Nx,Ny,Nz) cell grid using simple Jacobi.
    Dirichlet p=0 on the outer cell layer (i.e., p fixed to 0 on boundary cells).
    """
    p = np.zeros_like(rhs)
    for _ in range(iters):
        # 6-point Jacobi: p_new = (sum of 6 neighbors - rhs)/6 on interior cells
        p_new = np.zeros_like(p)
        p_new[1:-1,1:-1,1:-1] = (
            p[2:  ,1:-1,1:-1] + p[:-2 ,1:-1,1:-1] +
            p[1:-1,2:  ,1:-1] + p[1:-1,:-2 ,1:-1] +
            p[1:-1,1:-1,2:  ] + p[1:-1,1:-1,:-2 ] - rhs[1:-1,1:-1,1:-1]
        ) / 6.0
        p = p_new
    return p

def project_divergence_once(Bx, By, Bz, fixed_x, fixed_y, fixed_z, jacobi_iters=200):
    """
    One projection pass:
      1) compute d = div(B)
      2) solve Δp = d  (Dirichlet p=0 on outer cell layer)
      3) subtract gradient of p from *unfixed* faces
    """
    d = divergence(Bx, By, Bz)                         # shape (Nx,Ny,Nz)
    p = solve_poisson_jacobi(d, iters=jacobi_iters)    # cell-centered (Nx,Ny,Nz)

    # Discrete gradients on faces (unit spacing)
    grad_x = np.zeros_like(Bx); grad_y = np.zeros_like(By); grad_z = np.zeros_like(Bz)
    grad_x[1:Nx, :, :] = p[1:, :, :] - p[:-1, :, :]
    grad_y[:, 1:Ny, :] = p[:, 1:, :] - p[:, :-1, :]
    grad_z[:, :, 1:Nz] = p[:, :, 1:] - p[:, :, :-1]

    # Update only *unfixed* faces
    Bx[~fixed_x] -= grad_x[~fixed_x]
    By[~fixed_y] -= grad_y[~fixed_y]
    Bz[~fixed_z] -= grad_z[~fixed_z]

    return Bx, By, Bz

# -------------------------
# Iterate a few projection passes
# -------------------------
max_passes = 20
tol = 1e-6

for it in range(1, max_passes+1):
    Bx, By, Bz = project_divergence_once(Bx, By, Bz, fixed_x, fixed_y, fixed_z, jacobi_iters=200)
    div_res = divergence(Bx, By, Bz)
    max_div = float(np.max(np.abs(div_res)))
    energy = float(np.sqrt((Bx**2).sum() + (By**2).sum() + (Bz**2).sum()))
    print(f"pass {it:2d}: max|div| = {max_div:.3e},  ||B||_2 = {energy:.6f}")
    if max_div < tol:
        break

# -------------------------
# Diagnostics
# -------------------------
print("\nPinned center z-faces remain:")
print("  Bz[center,bottom] =", Bz[i0, j0, k0])
print("  Bz[center,top]    =", Bz[i0, j0, k0+1])

print("\nFinal checks:")
div_res = divergence(Bx, By, Bz)
print("  Max |divergence| over all cells:", float(np.max(np.abs(div_res))))
print("  L2 norm ||B||:", float(np.sqrt((Bx**2).sum() + (By**2).sum() + (Bz**2).sum())))
