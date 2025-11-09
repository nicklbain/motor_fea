#!/usr/bin/env python3
"""
minnorm_Au_eq_c.py
------------------
Builds the discrete divergence operator A and RHS c for a 3-D MAC-style grid of cells,
with face-centered unknowns u = [Bx | By | Bz_free], where two central Bz faces are "fixed"
to emulate a simple "magnet" source. We then solve

    minimize ||u||_2  subject to   A u = c

in two ways:
  (1) Accurate/basic: SVD pseudoinverse (dense)
  (2) Fast/advanced: Sparse CSR with a direct solve on (A A^T) lambda = c, then u = A^T lambda

Both methods yield the (unique) minimum-L2-norm solution.
"""

from dataclasses import dataclass
import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

@dataclass
class ACSystem:
    A_csr: "sp.csr_matrix"   # (num_cells x num_free_faces)
    c: np.ndarray            # (num_cells,)
    Nx: int; Ny: int; Nz: int
    nBx: int; nBy: int; nBz_free: int

def build_A_c(Nx: int, Ny: int, Nz: int) -> ACSystem:
    """
    Construct A and c for the given grid size.
    Unknown vector u = [Bx (all) | By (all) | Bz_free], where two Bz faces are fixed:
        at (i0,j0,k0) value +1, and at (i0,j0,k0+1) value -1
    These two fixed faces are omitted from A's columns; their contributions move to c.
    """
    assert SCIPY_OK, "SciPy is required for the CSR build."

    i0, j0, k0 = Nx//2, Ny//2, Nz//2
    fixed_bz = {(i0,j0,k0): +1.0, (i0,j0,k0+1): -1.0}

    nBx = (Nx+1)*Ny*Nz
    nBy = Nx*(Ny+1)*Nz
    # map free Bz faces to a compact index (skip the 2 fixed ones)
    bz_free_index = -np.ones((Nx,Ny,Nz+1), dtype=np.int64)
    free_count = 0
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz+1):
                if (i,j,k) in fixed_bz:  # skip fixed
                    continue
                bz_free_index[i,j,k] = free_count
                free_count += 1
    nBz_free = free_count
    n_cols = nBx + nBy + nBz_free
    n_rows = Nx * Ny * Nz

    def idx_cell(i,j,k): return (i*Ny + j)*Nz + k
    def idx_Bx(i_face,j,k): return (i_face*(Ny*Nz) + j*Nz + k)            # 0..nBx-1
    def idx_By(i,j_face,k): return nBx + ((i*(Ny+1) + j_face)*Nz + k)     # nBx..nBx+nBy-1
    def idx_Bz_free(i,j,k_face): return nBx + nBy + int(bz_free_index[i,j,k_face])

    # We'll assemble row by row, keeping track of col indices and values
    colind = []
    data   = []
    rowptr = np.zeros(n_rows+1, dtype=np.int64)
    c = np.zeros(n_rows, dtype=np.float64)

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                r = idx_cell(i,j,k)
                rowptr[r] = len(colind)

                # Divergence stencil: +Bx[i+1] - Bx[i]
                colind.append(idx_Bx(i+1,j,k)); data.append(+1.0)  # right x-face
                colind.append(idx_Bx(i  ,j,k)); data.append(-1.0)  # left  x-face

                # +By[j+1] - By[j]
                colind.append(idx_By(i,j+1,k)); data.append(+1.0)  # top y-face
                colind.append(idx_By(i,j  ,k)); data.append(-1.0)  # bottom y-face

                # +Bz[k+1] - Bz[k]  (handle fixed faces by moving to c)
                # front face (k+1)
                if (i,j,k+1) in fixed_bz:
                    c[r] += +fixed_bz[(i,j,k+1)]
                else:
                    colind.append(idx_Bz_free(i,j,k+1)); data.append(+1.0)
                # back face (k)
                if (i,j,k) in fixed_bz:
                    c[r] += -fixed_bz[(i,j,k)]
                else:
                    colind.append(idx_Bz_free(i,j,k)); data.append(-1.0)

    rowptr[-1] = len(colind)
    A_csr = sp.csr_matrix((np.array(data, float),
                           np.array(colind, np.int64),
                           rowptr),
                          shape=(n_rows, n_cols))
    return ACSystem(A_csr=A_csr, c=c, Nx=Nx, Ny=Ny, Nz=Nz, nBx=nBx, nBy=nBy, nBz_free=nBz_free)

# -----------------------------
# (1) Accurate/basic: SVD pseudoinverse
# -----------------------------
def solve_min_norm_svd(A_csr, c):
    A = A_csr.toarray()
    # Use pseudoinverse for the exact minimum-norm solution: u = A^+ c
    # np.linalg.pinv uses SVD under the hood
    u = np.linalg.pinv(A) @ c
    return u

# -----------------------------
# (2) Fast/advanced: sparse direct via (A A^T)
# -----------------------------
def solve_min_norm_via_normal_eq(A_csr, c):
    # Solve (A A^T) λ = c (projected to mean-zero to handle the constant nullspace),
    # then u = A^T λ  → the exact min-||u|| solution of Au=c.
    M = (A_csr @ A_csr.T).tocsr()
    c0 = c - c.mean()   # remove constant mode (nullspace of M)
    lam = spla.spsolve(M, c0)
    u = A_csr.T @ lam
    return u

def main():
    Nx=Ny=Nz=3
    ac = build_A_c(Nx,Ny,Nz)
    A, c = ac.A_csr, ac.c

    print("#" * 72)
    print(f"Grid: {Nx} x {Ny} x {Nz}")
    print(f"A shape: {A.shape}  (rows=cells={Nx*Ny*Nz}, cols=faces_free={A.shape[1]})")
    print(f"Blocks: nBx={ac.nBx}, nBy={ac.nBy}, nBz_free={ac.nBz_free}")
    print("Note: u = [Bx (all) | By (all) | Bz (all except the 2 fixed center faces)]")
    print("#" * 72)

    # Accurate/basic
    print("\n=== Accurate/basic method: SVD pseudoinverse ===")
    u_svd = solve_min_norm_svd(A, c)
    r_svd = A @ u_svd - c
    print(f"‖u_svd‖₂ = {np.linalg.norm(u_svd):.6f}")
    print(f"Residual: max|Au-c| = {np.max(np.abs(r_svd)):.3e}, "
          f"mean|Au-c| = {np.mean(np.abs(r_svd)):.3e}")
    print("u_svd (first 40 entries):")
    print(np.array2string(u_svd[:40], precision=6, suppress_small=False))

    # Fast/advanced
    print("\n=== Fast/advanced method: sparse direct on (A A^T) ===")
    if not SCIPY_OK:
        print("SciPy not available; skipping fast method.")
        return
    u_fast = solve_min_norm_via_normal_eq(A, c)
    r_fast = A @ u_fast - c
    print(f"‖u_fast‖₂ = {np.linalg.norm(u_fast):.6f}")
    print(f"Residual: max|Au-c| = {np.max(np.abs(r_fast)):.3e}, "
          f"mean|Au-c| = {np.mean(np.abs(r_fast)):.3e}")

    # Compare
    diff = u_svd - u_fast
    print("\n=== Comparison (SVD vs Fast) ===")
    print(f"‖u_svd - u_fast‖₂ = {np.linalg.norm(diff):.3e}, "
          f"max|Δ| = {np.max(np.abs(diff)):.3e}")
    print("u_fast (first 40 entries):")
    print(np.array2string(u_fast[:40], precision=6, suppress_small=False))

if __name__ == "__main__":
    main()