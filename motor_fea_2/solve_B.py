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
import json
import time
import datetime

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from scipy.interpolate import PchipInterpolator

AIR, PMAG, STEEL = 0, 1, 2
DEFAULT_CASES_DIR = Path("cases")
DEFAULT_CASE_NAME = "mag2d_case"
DEFAULT_MESH_FILENAME = "mesh.npz"
LEGACY_FLAT_FILENAME = "mag2d_case.npz"
MU0 = 4e-7 * np.pi
MAX_POLY_SIDES_EXPLICIT = 4096
INDICATOR_CLIP_DEFAULT = (5.0, 95.0)
ALPHA_MIN_DEFAULT = 0.5
ALPHA_MAX_DEFAULT = 2.0
INDICATOR_GAIN_DEFAULT = 0.4
BH_MU_MIN = 1.0
BH_RELAX_DEFAULT = 0.33
BH_MAX_ITERS = 15
BH_TOL = 1e-3
NEWTON_MAX_ITERS = 4
NEWTON_LINESEARCH_STEPS = 6
NEWTON_RES_TOL = 1e-4
NEWTON_MIN_DROP = 0.01
TRUST_STEP_NORM_MAX = 5.0   # cap on ||ΔA|| in Newton to improve robustness
MU_LOG_STEP_MAX = 0.5       # cap per-iteration log-change in mu updates (used for BH damping)
MU_DIFF_BLEND = 0.5         # blend between mu_sec and mu_diff to tame spikes
BH_B_CLAMP = 2.0            # cap |B| fed into BH curves to avoid runaway first iterations
BH_PLATEAU_DROP = 0.05      # if rel_change fails to drop by 5% over recent Picard steps, bail early
BH_ENABLE_NEWTON_DEFAULT = False  # default to Picard-only unless explicitly enabled

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
    bh_id = data["bh_id"] if "bh_id" in data else np.full(tris.shape[0], -1, dtype=np.int32)
    return nodes, tris, region, mu_r, Mx, My, Jz, meta, bh_id


def _bh_curves_from_meta(meta: dict[str, object]) -> list[dict[str, object]]:
    """Extract and sanitize BH curves from mesh metadata."""
    curves: list[dict[str, object]] = []
    if not isinstance(meta, dict):
        return curves
    raw_curves = meta.get("bh_curves")
    if not isinstance(raw_curves, list):
        return curves
    for idx, entry in enumerate(raw_curves):
        if not isinstance(entry, dict):
            continue
        B = entry.get("B")
        H = entry.get("H")
        if B is None or H is None:
            points = entry.get("points")
            if isinstance(points, (list, tuple)) and len(points) >= 2:
                B = [p[0] for p in points]
                H = [p[1] for p in points]
        try:
            B_arr = np.asarray(B, dtype=float)
            H_arr = np.asarray(H, dtype=float)
        except Exception:
            continue
        if B_arr.size < 2 or H_arr.size != B_arr.size:
            continue
        finite = np.isfinite(B_arr) & np.isfinite(H_arr)
        if not np.any(finite):
            continue
        B_arr = np.abs(B_arr[finite])
        H_arr = np.abs(H_arr[finite])
        order = np.argsort(B_arr)
        B_arr = B_arr[order]
        H_arr = H_arr[order]
        keep = np.concatenate([[True], np.diff(B_arr) > 1e-12])
        B_arr = B_arr[keep]
        H_arr = H_arr[keep]
        if B_arr.size < 2:
            continue
        # Light smoothing + monotonic enforcement for robustness
        # Smooth H vs B with a small window, then enforce monotone H, recompute B from that.
        def _smooth_monotone(x: np.ndarray, y: np.ndarray, window: int = 5) -> tuple[np.ndarray, np.ndarray]:
            if x.size < 3:
                return x, y
            k = max(1, min(window, x.size))
            kernel = np.ones(k) / k
            y_pad = np.pad(y, (k//2, k-1-k//2), mode="edge")
            y_smooth = np.convolve(y_pad, kernel, mode="valid")
            # enforce monotone increasing y w.r.t x
            y_mono = np.maximum.accumulate(y_smooth)
            return x, y_mono

        B_arr, H_arr = _smooth_monotone(B_arr, H_arr, window=7)
        dB_dH = np.gradient(B_arr, H_arr, edge_order=2)
        # Blend derivative with secant slope to tame spikes
        with np.errstate(divide="ignore", invalid="ignore"):
            mu_sec_curve = B_arr / np.maximum(MU0 * H_arr, 1e-16)
            dB_dH = MU_DIFF_BLEND * dB_dH + (1.0 - MU_DIFF_BLEND) * (mu_sec_curve * MU0)
        mu_linear = entry.get("mu_r_linear")
        fit_diag = None
        fit_payload = None
        if B_arr.size >= 8:
            fit_payload, fit_diag = _fit_bh_curve(B_arr, H_arr, dB_dH)
        curve = {
            "B": B_arr,
            "H": H_arr,
            "dB_dH": dB_dH,
            "source": entry.get("source") or "unknown",
            "material": entry.get("material"),
            "label": entry.get("label"),
            "mu_r_linear": mu_linear,
            "b_sat": entry.get("b_sat"),
            "id": idx,
        }
        if fit_payload is not None:
            curve["fit"] = fit_payload
        if fit_diag is not None:
            curve["fit_diag"] = fit_diag
        curves.append(curve)
    return curves


def _fit_bh_curve(B_arr: np.ndarray, H_arr: np.ndarray, dB_dH_raw: np.ndarray | None = None,
                  *, rel_tol: float = 0.05, n_samples: int = 256) -> tuple[dict | None, dict | None]:
    """
    Fit a smooth monotone BH curve using PCHIP on B->H, then resample to a compact table.
    Uses slope-weighted sampling to preserve steep knees; grows n_samples until tolerance
    is met or a hard cap is reached. Returns (fit_payload, diagnostics). If the fit exceeds
    rel_tol on max relative error, returns (None, diagnostics) so callers can fall back to
    the raw curve.
    """
    diag = {
        "accepted": False,
        "n_raw": int(B_arr.size),
        "n_fit": int(n_samples),
        "rel_tol": float(rel_tol),
        "max_rel_err": None,
        "mean_rel_err": None,
        "status": "init",
    }
    try:
        pchip_raw = PchipInterpolator(B_arr, H_arr, extrapolate=True)

        def _fit_with(n_fit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
            # Slope-weighted parameterization to concentrate samples where dH/dB is large.
            dH_dB_raw = np.maximum(pchip_raw.derivative()(B_arr), 1e-12)
            weight = 1.0 + dH_dB_raw / np.maximum(np.max(dH_dB_raw), 1e-12)
            s = np.concatenate([[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(B_arr))])
            s /= s[-1] if s[-1] > 0 else 1.0
            s_targets = np.linspace(0.0, 1.0, n_fit)
            B_fit = np.interp(s_targets, s, B_arr)
            H_fit = pchip_raw(B_fit)
            dH_dB = np.maximum(pchip_raw.derivative()(B_fit), 1e-12)
            dB_dH = 1.0 / dH_dB
            pchip_fit = PchipInterpolator(B_fit, H_fit, extrapolate=True)
            H_recon = pchip_fit(B_arr)
            rel = np.abs(H_recon - H_arr) / np.maximum(np.abs(H_arr) + 1e-9, 1e-9)
            return B_fit, H_fit, dB_dH, float(np.max(rel)), float(np.mean(rel))

        max_fit = 2048
        while True:
            B_fit, H_fit, dB_dH, max_err, mean_err = _fit_with(n_samples)
            diag.update({"n_fit": int(n_samples), "max_rel_err": max_err, "mean_rel_err": mean_err})
            if np.any(~np.isfinite(H_fit)) or np.any(~np.isfinite(dB_dH)):
                diag["status"] = "non_finite"
                return None, diag
            if max_err <= rel_tol:
                diag["accepted"] = True
                diag["status"] = "ok"
                fit_payload = {
                    "B": B_fit,
                    "H": H_fit,
                    "dB_dH": dB_dH,
                    "rel_tol": rel_tol,
                    "max_rel_err": max_err,
                    "mean_rel_err": mean_err,
                }
                return fit_payload, diag
            if n_samples >= max_fit:
                diag["accepted"] = False
                diag["status"] = "rejected_err"
                return None, diag
            n_samples = min(max_fit, int(n_samples * 1.6))
    except Exception as exc:  # pragma: no cover - fit is best-effort
        diag["status"] = f"fit_failed:{type(exc).__name__}"
        return None, diag


def _mu_from_curve(curve: dict[str, object], B_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute H, secant mu_r, and differential mu_r for |B| samples with smoothing/blending."""
    # Prefer a fitted/smoothed curve when available and validated.
    if "fit" in curve:
        fit = curve["fit"]
        B_curve = np.asarray(fit["B"], dtype=float)
        H_curve = np.asarray(fit["H"], dtype=float)
        dBdH_curve = np.asarray(fit.get("dB_dH"), dtype=float)
    else:
        B_curve = np.asarray(curve["B"], dtype=float)
        H_curve = np.asarray(curve["H"], dtype=float)
        dBdH_curve = np.asarray(curve.get("dB_dH"), dtype=float)
    if dBdH_curve.size != B_curve.size:
        dBdH_curve = np.gradient(B_curve, H_curve, edge_order=2)
    if B_curve.size < 2 or H_curve.size != B_curve.size:
        base = float(curve.get("mu_r_linear", 1.0))
        return np.zeros_like(B_values), np.full_like(B_values, fill_value=base), np.full_like(B_values, fill_value=base)
    Babs = np.abs(B_values)
    if np.isfinite(BH_B_CLAMP):
        Babs = np.minimum(Babs, BH_B_CLAMP)

    def _slope(idx0: int, idx1: int) -> float:
        dH = H_curve[idx1] - H_curve[idx0]
        dB = B_curve[idx1] - B_curve[idx0]
        if abs(dH) < 1e-12:
            return dBdH_curve[idx0] if idx0 < dBdH_curve.size else MU0
        return max(dB / dH, MU0)

    slope_left = max(dBdH_curve[0], _slope(0, 1))
    # Enforce a “flat” tail so extrapolation past the last BH point quickly
    # collapses toward mu≈1 instead of riding a high secant slope.
    slope_right = MU0  # force tail toward air
    B_min = B_curve[0]
    B_max = B_curve[-1]
    H_min = H_curve[0]
    H_max = H_curve[-1]
    H_interp = np.interp(Babs, B_curve, H_curve)
    dBdH_interp = np.interp(Babs, B_curve, dBdH_curve)
    below = Babs < B_min
    above = Babs > B_max
    if np.any(below):
        H_interp[below] = H_min + (Babs[below] - B_min) / slope_left
        dBdH_interp[below] = slope_left
    if np.any(above):
        H_interp[above] = H_max + (Babs[above] - B_max) / slope_right
        dBdH_interp[above] = slope_right
    mu_sec = Babs / np.maximum(MU0 * H_interp, 1e-12)
    mu_diff_raw = dBdH_interp / MU0
    mu_diff = MU_DIFF_BLEND * mu_diff_raw + (1.0 - MU_DIFF_BLEND) * mu_sec
    # Cap the tail mu so over-saturation bumps fall back toward air quickly.
    if np.any(above):
        # Smoothly collapse mu toward air for B beyond the measured curve
        span = max(0.2 * B_max, 1e-12)
        ramp = np.clip((Babs[above] - B_max) / span, 0.0, 1.0)
        mu_sec[above] = (1.0 - ramp) * mu_sec[above] + ramp * BH_MU_MIN
        mu_diff[above] = (1.0 - ramp) * mu_diff[above] + ramp * BH_MU_MIN
    mu_max = curve.get("mu_r_linear")
    if mu_max is None or not np.isfinite(mu_max):
        mu_max = max(np.nanmax(mu_sec), 1.0) * 10.0
    mu_sec = np.clip(mu_sec, BH_MU_MIN, float(mu_max))
    mu_diff = np.clip(mu_diff, BH_MU_MIN, float(mu_max))
    return H_interp, mu_sec, mu_diff


def _picard_bh_solve(nodes, tris, region, mu_r_init, Mx, My, Jz, bh_id, bh_curves, *, pin_node=0,
                     relax: float = BH_RELAX_DEFAULT, max_iters: int = BH_MAX_ITERS, tol: float = BH_TOL):
    """Iterate mu_r based on BH curves with damping and adaptive relaxation."""
    mu_r = mu_r_init.copy()
    Az = np.zeros(nodes.shape[0], dtype=float)
    Bx = By = Bmag = None
    diag = []
    bh_curves_by_id = {c.get("id", idx): c for idx, c in enumerate(bh_curves)}
    prev_rel_change = None
    steady_count = 0
    plateau_guard = []
    for it in range(max_iters):
        t_iter = time.perf_counter()
        K, f = assemble_A_system(nodes, tris, region, mu_r, Mx, My, Jz)
        Az = solve_neumann_pinned(K, f, pin_node=pin_node)
        Bx, By, Bmag = compute_triangle_B(nodes, tris, Az)
        mu_next = mu_r.copy()
        # Adjust relaxation if the previous step blew up.
        relax_used = relax
        if prev_rel_change is not None:
            if prev_rel_change > 10.0:
                relax_used = min(relax_used, 0.15)
            elif prev_rel_change > 2.0:
                relax_used = min(relax_used, 0.35)
        for cid, curve in bh_curves_by_id.items():
            mask = bh_id == cid
            if not np.any(mask):
                continue
            _, mu_sec, _ = _mu_from_curve(curve, Bmag[mask])
            # If we're far into saturation, accelerate the drop; otherwise blend gently.
            if np.any(mu_sec < 0.5 * mu_r[mask]):
                relax_eff = 1.0
                down_cap = 8.0
            else:
                relax_eff = relax_used
                down_cap = MU_LOG_STEP_MAX
            blended = (1.0 - relax_eff) * mu_r[mask] + relax_eff * mu_sec
            mu_next[mask] = _damped_mu_update(mu_r[mask], blended,
                                              max_log_step=MU_LOG_STEP_MAX,
                                              max_log_step_down=down_cap,
                                              max_log_step_up=MU_LOG_STEP_MAX)
        denom = np.maximum(np.abs(mu_r), 1e-9)
        rel_change = float(np.max(np.abs(mu_next - mu_r) / denom))
        mu_r = mu_next
        prev_rel_change = rel_change
        if rel_change < 0.75:
            steady_count += 1
        else:
            steady_count = 0
        plateau_guard.append(rel_change)
        diag.append({
            "iter": it + 1,
            "method": "picard",
            "rel_change": rel_change,
            "relax": relax_used,
            "time_s": time.perf_counter() - t_iter,
            "B_max": float(np.max(Bmag)) if Bmag is not None else None,
            "mu_min": float(np.min(mu_r)),
            "mu_max": float(np.max(mu_r)),
        })
        # Plateau guard after logging to retain diagnostics; need at least 3 samples.
        if len(plateau_guard) >= 3 and rel_change < 0.5:
            drop_frac = (plateau_guard[-3] - plateau_guard[-1]) / max(plateau_guard[-3], 1e-12)
            if drop_frac < BH_PLATEAU_DROP:
                break
        if rel_change < tol:
            return Az, mu_r, Bx, By, Bmag, it + 1, True, diag
        if steady_count >= 3:
            break
        # Bail early if we're clearly diverging.
        if rel_change > 100.0 and it >= 2:
            break
    return Az, mu_r, Bx, By, Bmag, it + 1, False, diag


def _residual_norm(K: sp.spmatrix, A: np.ndarray, f: np.ndarray) -> float:
    r = K @ A - f
    return float(np.linalg.norm(r) / max(len(A), 1))


def _damped_mu_update(prev: np.ndarray | None,
                      new: np.ndarray,
                      max_log_step: float = MU_LOG_STEP_MAX,
                      max_log_step_down: float | None = None,
                      max_log_step_up: float | None = None) -> np.ndarray:
    """Limit per-entry multiplicative change; allow asymmetric caps for down/up steps."""
    if prev is None:
        return new
    prev = np.asarray(prev, dtype=float)
    new = np.asarray(new, dtype=float)
    max_log_step = max(float(max_log_step), 0.0)
    max_log_down = max_log_step_down if max_log_step_down is not None else max_log_step
    max_log_up = max_log_step_up if max_log_step_up is not None else max_log_step
    max_log_down = max(float(max_log_down), 0.0)
    max_log_up = max(float(max_log_up), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(prev > 0.0, new / prev, 1.0)
        log_ratio = np.log(np.clip(ratio, 1e-16, 1e16))
    log_ratio = np.where(log_ratio < 0, np.maximum(log_ratio, -max_log_down), np.minimum(log_ratio, max_log_up))
    return prev * np.exp(log_ratio)


def _bh_maps_from_B(Bmag: np.ndarray,
                    bh_id: np.ndarray,
                    curves: dict[int, dict],
                    *,
                    prev_mu_sec: np.ndarray | None = None,
                    prev_mu_diff: np.ndarray | None = None,
                    max_log_step: float = MU_LOG_STEP_MAX) -> tuple[np.ndarray, np.ndarray]:
    """Return secant and differential mu_r arrays for all elements with optional damping."""
    mu_sec = np.ones_like(Bmag, dtype=float)
    mu_diff = np.ones_like(Bmag, dtype=float)
    for cid, curve in curves.items():
        mask = bh_id == cid
        if not np.any(mask):
            continue
        _, mu_s, mu_d = _mu_from_curve(curve, Bmag[mask])
        mu_sec[mask] = mu_s
        mu_diff[mask] = mu_d
    if prev_mu_sec is not None:
        mu_sec = _damped_mu_update(prev_mu_sec, mu_sec,
                                   max_log_step=max_log_step,
                                   max_log_step_down=3.0,  # allow fast drops toward saturation
                                   max_log_step_up=max_log_step)
    if prev_mu_diff is not None:
        mu_diff = _damped_mu_update(prev_mu_diff, mu_diff,
                                    max_log_step=max_log_step,
                                    max_log_step_down=3.0,
                                    max_log_step_up=max_log_step)
    return mu_sec, mu_diff


def _newton_bh_solve(nodes, tris, region, mu_r_init, Mx, My, Jz, bh_id, bh_curves, *,
                     pin_node=0, max_iters: int = NEWTON_MAX_ITERS, res_tol: float = NEWTON_RES_TOL,
                     min_drop: float = NEWTON_MIN_DROP):
    """Newton solve using differential mu for Jacobian and secant mu for residual."""
    # Warm start: linear solve with mesh mu, then map mu from its B-field.
    K0, f = assemble_A_system(nodes, tris, region, mu_r_init, Mx, My, Jz)
    Az = solve_neumann_pinned(K0, f, pin_node=pin_node)
    Bx0, By0, Bmag0 = compute_triangle_B(nodes, tris, Az)
    bh_curves_by_id = {c.get("id", idx): c for idx, c in enumerate(bh_curves)}
    mu_sec, mu_diff = _bh_maps_from_B(
        Bmag0, bh_id, bh_curves_by_id,
        prev_mu_sec=mu_r_init, prev_mu_diff=mu_r_init, max_log_step=MU_LOG_STEP_MAX
    )
    bh_curves_by_id = {c.get("id", idx): c for idx, c in enumerate(bh_curves)}
    diag = []
    fallback_reason = None

    K_sec, f = assemble_A_system(nodes, tris, region, mu_sec, Mx, My, Jz)
    res_norm = _residual_norm(K_sec, Az, f)
    # If the warm-start residual is already close to tolerance, skip Newton (saves one expensive solve)
    if res_norm < res_tol * 5.0:
        Bx, By, Bmag = compute_triangle_B(nodes, tris, Az)
        diag.append({"iter": 0, "method": "newton", "res_norm": res_norm, "accepted": True, "time_s": 0.0, "reason": "skip_newton_low_res"})
        return Az, mu_sec, Bx, By, Bmag, 0, False, diag, "skip_newton"
    if res_norm < res_tol:
        Bx, By, Bmag = compute_triangle_B(nodes, tris, Az)
        diag.append({"iter": 0, "method": "newton", "res_norm": res_norm, "accepted": True, "time_s": 0.0})
        return Az, mu_sec, Bx, By, Bmag, 0, True, diag, fallback_reason

    for it in range(1, max_iters + 1):
        t_iter = time.perf_counter()
        # Build Jacobian with differential mu
        K_diff, _ = assemble_A_system(nodes, tris, region, mu_diff, Mx, My, Jz)
        res_vec = K_sec @ Az - f
        try:
            delta = spla.spsolve(K_diff, -res_vec)
        except Exception:
            fallback_reason = "jacobian_solve_failed"
            diag.append({"iter": it, "method": "newton", "res_norm": res_norm, "accepted": False, "time_s": time.perf_counter() - t_iter, "reason": fallback_reason})
            return Az, mu_sec, *compute_triangle_B(nodes, tris, Az), it, False, diag, fallback_reason
        # Trust-region style cap on step norm
        step_norm = float(np.linalg.norm(delta))
        if step_norm > TRUST_STEP_NORM_MAX and step_norm > 0:
            delta = delta * (TRUST_STEP_NORM_MAX / step_norm)

        alpha = 1.0
        prev_res = res_norm
        best = (prev_res, Az, mu_sec, mu_diff, 0.0)
        for _ in range(NEWTON_LINESEARCH_STEPS):
            Az_trial = Az + alpha * delta
            Bx_t, By_t, Bmag_t = compute_triangle_B(nodes, tris, Az_trial)
            mu_sec_t, mu_diff_t = _bh_maps_from_B(
                Bmag_t, bh_id, bh_curves_by_id,
                prev_mu_sec=mu_sec, prev_mu_diff=mu_diff, max_log_step=MU_LOG_STEP_MAX
            )
            K_sec_t, _ = assemble_A_system(nodes, tris, region, mu_sec_t, Mx, My, Jz)
            res_norm_t = _residual_norm(K_sec_t, Az_trial, f)
            if res_norm_t < best[0] - 1e-12:
                best = (res_norm_t, Az_trial, mu_sec_t, mu_diff_t, alpha)
            alpha *= 0.5
        res_norm, Az, mu_sec, mu_diff, alpha_used = best
        improved = res_norm < prev_res - 1e-12
        K_sec = assemble_A_system(nodes, tris, region, mu_sec, Mx, My, Jz)[0]
        if res_norm < res_tol:
            Bx, By, Bmag = compute_triangle_B(nodes, tris, Az)
            diag.append({"iter": it, "method": "newton", "res_norm": res_norm, "accepted": improved, "alpha": alpha_used, "time_s": time.perf_counter() - t_iter})
            return Az, mu_sec, Bx, By, Bmag, it, True, diag, fallback_reason
        drop_frac = (prev_res - res_norm) / max(prev_res, 1e-16)
        if not improved:
            fallback_reason = "line_search_failed"
            diag.append({"iter": it, "method": "newton", "res_norm": res_norm, "accepted": False, "alpha": alpha_used, "time_s": time.perf_counter() - t_iter, "reason": fallback_reason})
            break
        if drop_frac < min_drop:
            fallback_reason = "residual_drop_small"
            diag.append({"iter": it, "method": "newton", "res_norm": res_norm, "accepted": True, "alpha": alpha_used, "time_s": time.perf_counter() - t_iter, "reason": fallback_reason})
            break
        diag.append({"iter": it, "method": "newton", "res_norm": res_norm, "accepted": True, "alpha": alpha_used, "time_s": time.perf_counter() - t_iter})

    Bx, By, Bmag = compute_triangle_B(nodes, tris, Az)
    return Az, mu_sec, Bx, By, Bmag, max_iters, False, diag, fallback_reason

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
    T = tris
    twoA, A, b, c = triangle_metrics(nodes, tris)

    # Stiffness assembly (vectorized)
    valid = A > 0.0
    T_valid = T[valid]
    if T_valid.size == 0:
        K = sp.csr_matrix((Nn, Nn))
    else:
        factors = nu_elem[valid] / (4.0 * A[valid])
        Ke = factors[:, None, None] * (b[valid, :, None] * b[valid, None, :] + c[valid, :, None] * c[valid, None, :])
        rows = np.repeat(T_valid, 3, axis=1).ravel()
        cols = np.tile(T_valid, (1, 3)).ravel()
        data = Ke.reshape(-1)
        K = sp.coo_matrix((data, (rows, cols)), shape=(Nn, Nn)).tocsr()

    # RHS: volume Jz (piecewise constant) + volume curl(M) if provided (usually zero for uniform M)
    f = np.zeros(Nn, dtype=float)
    # Volume Jz
    Fe = (Jz * A) / 3.0
    np.add.at(f, T[:, 0], Fe)
    np.add.at(f, T[:, 1], Fe)
    np.add.at(f, T[:, 2], Fe)

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
    For every edge belonging to a PM element, add the bound surface current:
      ∫_edge K_b * v ds  with K_b = (M × n̂)·ẑ = Mx * n_y - My * n_x
    n̂ is the outward unit normal of that element along the edge.
    Each edge contributes equally to its two end nodes: +K_b * |edge| / 2.

    This now covers:
      - PM touching air/steel/wire (previous behavior).
      - PM touching the outer boundary (treated as PM vs "nothing").
      - PM touching another PM with different M (we add the contribution from
        each side, so the jump in M still appears).
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

    # For each edge, add contributions from each adjacent PM element (one for boundary edges, two for interior).
    for adj in edges.values():
        for e_idx, (u, v) in adj:
            if region[e_idx] != PMAG:
                continue
            xi, yi = P[u]
            xj, yj = P[v]
            tx, ty = (xj - xi), (yj - yi)
            elen = (tx * tx + ty * ty) ** 0.5
            if elen == 0.0:
                continue
            # For CCW triangles the left normal points *into* the element; flip to get outward.
            nx, ny = -ty / elen, tx / elen
            Mxe, Mye = Mx[e_idx], My[e_idx]
            Kb = Mxe * ny - Mye * nx  # (M × n̂)·ẑ
            load = Kb * elen * 0.5
            f[u] += load
            f[v] += load

def solve_neumann_pinned(K, f, pin_node=0):
    """Pin one node (gauge fix) to eliminate the constant nullspace."""
    # Add a diagonal entry for any isolated nodes so the linear system is not singular.
    # A handful of unreferenced nodes can show up from meshing; they should not influence
    # the solution, so we pin them alongside the gauge node.
    K = K.tocsc()
    f = f.copy()
    row_nnz = np.asarray(K.getnnz(axis=1)).ravel()
    extra_pins = np.flatnonzero(row_nnz == 0)
    pins = [pin_node] + [p for p in extra_pins if p != pin_node]
    K = K.tolil()
    for idx in pins:
        K[idx, :] = 0.0
        K[:, idx] = 0.0
        K[idx, idx] = 1.0
        f[idx] = 0.0
    K = K.tocsc()
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
        # Fallback for meshes saved before contours were split out of geometry.
        if not contours:
            geom = meta.get("geometry")
            if isinstance(geom, list):
                contours = [
                    obj
                    for obj in geom
                    if isinstance(obj, dict) and str(obj.get("material", "")).lower() == "contour"
                ]
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


def _triangle_neighbors(tris: np.ndarray) -> list[list[int]]:
    """Adjacency list where entry i contains neighboring triangle indices for tri i."""
    neighbors: list[list[int]] = [[] for _ in range(tris.shape[0])]
    edges: dict[tuple[int, int], int] = {}
    for idx, tri in enumerate(tris):
        for corner in range(3):
            a = int(tri[corner])
            b = int(tri[(corner + 1) % 3])
            if a > b:
                a, b = b, a
            key = (a, b)
            other = edges.get(key)
            if other is None:
                edges[key] = idx
            else:
                neighbors[idx].append(other)
                neighbors[other].append(idx)
    return neighbors


def _triangle_neighbor_pairs(tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of paired triangle indices that share an edge."""
    if tris.size == 0:
        empty = np.array([], dtype=int)
        return empty, empty
    edges = np.concatenate(
        [
            tris[:, [0, 1]],
            tris[:, [1, 2]],
            tris[:, [2, 0]],
        ],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    tri_idx = np.repeat(np.arange(tris.shape[0], dtype=int), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges_sorted = edges[order]
    tris_sorted = tri_idx[order]
    same = np.all(edges_sorted[1:] == edges_sorted[:-1], axis=1)
    if not np.any(same):
        empty = np.array([], dtype=int)
        return empty, empty
    return tris_sorted[:-1][same], tris_sorted[1:][same]


def _angle_diff(rad_a: float, rad_b: float) -> float:
    diff = rad_a - rad_b
    return abs(np.arctan2(np.sin(diff), np.cos(diff)))


def _field_gradient_components(
    Bmag: np.ndarray,
    Bx: np.ndarray,
    By: np.ndarray,
    centroids: np.ndarray,
    tris: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    Ne = Bmag.shape[0]
    mag_comp = np.zeros(Ne, dtype=float)
    dir_comp = np.zeros(Ne, dtype=float)
    angles = np.arctan2(By, Bx)
    i_idx, j_idx = _triangle_neighbor_pairs(tris)
    if i_idx.size == 0:
        return mag_comp, dir_comp
    delta = centroids[i_idx] - centroids[j_idx]
    dist = np.hypot(delta[:, 0], delta[:, 1])
    dist = np.where(dist < 1e-12, 1e-12, dist)
    mag_change = np.abs(Bmag[i_idx] - Bmag[j_idx]) / dist
    ang_diff = np.abs(np.arctan2(np.sin(angles[i_idx] - angles[j_idx]), np.cos(angles[i_idx] - angles[j_idx]))) / dist
    # Scatter max contributions back to each triangle.
    np.maximum.at(mag_comp, i_idx, mag_change)
    np.maximum.at(mag_comp, j_idx, mag_change)
    np.maximum.at(dir_comp, i_idx, ang_diff)
    np.maximum.at(dir_comp, j_idx, ang_diff)
    return mag_comp, dir_comp


def _focus_params_from_meta(meta: dict[str, object]) -> dict[str, float | None]:
    """Pull field-focus indicator settings out of solver/mesh metadata."""
    params: dict[str, float | None] = {
        "direction_weight": 1.0,
        "magnitude_weight": 1.0,
        "indicator_gain": INDICATOR_GAIN_DEFAULT,
        "indicator_neutral": None,
        "indicator_percentile": 85.0,
        "alpha_min": ALPHA_MIN_DEFAULT,
        "alpha_max": ALPHA_MAX_DEFAULT,
        "indicator_clip": INDICATOR_CLIP_DEFAULT,
    }
    if not isinstance(meta, dict):
        return params
    focus = None
    mesh_gen = meta.get("mesh_generation")
    if isinstance(mesh_gen, dict):
        if isinstance(mesh_gen.get("field_focus_params"), dict):
            focus = mesh_gen["field_focus_params"]
        elif isinstance(mesh_gen.get("field_focus"), dict):
            focus = mesh_gen["field_focus"]
    if focus is None and isinstance(meta.get("field_focus_params"), dict):
        focus = meta["field_focus_params"]
    if isinstance(focus, dict):
        params.update({
            "direction_weight": focus.get("direction_weight", params["direction_weight"]),
            "magnitude_weight": focus.get("magnitude_weight", params["magnitude_weight"]),
            "indicator_gain": focus.get("indicator_gain", params["indicator_gain"]),
            "indicator_neutral": focus.get("indicator_neutral", params["indicator_neutral"]),
            "indicator_percentile": focus.get("indicator_percentile", params["indicator_percentile"]),
            "alpha_min": focus.get("scale_min", focus.get("alpha_min", params["alpha_min"])),
            "alpha_max": focus.get("scale_max", focus.get("alpha_max", params["alpha_max"])),
            "indicator_clip": focus.get("indicator_clip", params["indicator_clip"]),
        })
    # Ensure numeric sanity
    params["direction_weight"] = float(params["direction_weight"] or 0.0)
    params["magnitude_weight"] = float(params["magnitude_weight"] or 0.0)
    try:
        params["indicator_gain"] = max(float(params["indicator_gain"]), 0.0)
    except Exception:
        params["indicator_gain"] = INDICATOR_GAIN_DEFAULT
    try:
        neutral = float(params["indicator_neutral"])
        params["indicator_neutral"] = neutral if np.isfinite(neutral) and neutral > 0 else None
    except Exception:
        params["indicator_neutral"] = None
    try:
        pct = float(params["indicator_percentile"])
        params["indicator_percentile"] = float(np.clip(pct, 0.0, 100.0))
    except Exception:
        params["indicator_percentile"] = 85.0
    try:
        params["alpha_min"] = max(float(params["alpha_min"]), 1e-6)
    except Exception:
        params["alpha_min"] = ALPHA_MIN_DEFAULT
    try:
        params["alpha_max"] = max(float(params["alpha_max"]), params["alpha_min"])
    except Exception:
        params["alpha_max"] = ALPHA_MAX_DEFAULT
    clip = params.get("indicator_clip", INDICATOR_CLIP_DEFAULT)
    if isinstance(clip, (list, tuple)) and len(clip) >= 2:
        try:
            lo, hi = float(clip[0]), float(clip[1])
            lo = max(lo, 0.0)
            hi = max(hi, lo)
            hi = min(hi, 100.0)
            params["indicator_clip"] = (lo, hi)
        except Exception:
            params["indicator_clip"] = INDICATOR_CLIP_DEFAULT
    else:
        params["indicator_clip"] = INDICATOR_CLIP_DEFAULT
    return params


def _compute_indicator_and_alpha(
    nodes: np.ndarray,
    tris: np.ndarray,
    Bx: np.ndarray,
    By: np.ndarray,
    Bmag: np.ndarray,
    meta: dict[str, object],
) -> dict[str, object] | None:
    """Derive indicator components and alpha map for reuse in downstream meshing."""
    if tris.ndim != 2 or tris.shape[1] != 3 or Bmag.shape[0] != tris.shape[0]:
        return None
    params = _focus_params_from_meta(meta)
    centroids = nodes[tris].mean(axis=1)
    mag_comp, dir_comp = _field_gradient_components(Bmag, Bx, By, centroids, tris)
    combined = params["magnitude_weight"] * mag_comp + params["direction_weight"] * dir_comp
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return None
    clip_lo, clip_hi = params["indicator_clip"]
    if clip_lo > 0 or clip_hi < 100:
        lo = np.percentile(finite, clip_lo)
        hi = np.percentile(finite, clip_hi)
        combined = np.clip(combined, lo, hi)
        mag_comp = np.clip(mag_comp, lo, hi)
        dir_comp = np.clip(dir_comp, lo, hi)
        finite = combined[np.isfinite(combined)]
    ref = params["indicator_neutral"]
    if ref is None or not np.isfinite(ref) or ref <= 0:
        ref = np.percentile(finite, params["indicator_percentile"])
    if not np.isfinite(ref) or ref <= 0:
        ref = max(np.median(finite), 1e-9)
    gain = params["indicator_gain"]
    alpha_min = params["alpha_min"]
    alpha_max = max(params["alpha_max"], alpha_min)
    alpha = np.full_like(combined, fill_value=alpha_max, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.maximum(combined, 1e-12) / ref
        alpha = np.power(ratio, -gain)
    alpha = np.clip(alpha, alpha_min, alpha_max)

    stats = {
        "magnitude_min": float(np.nanmin(mag_comp)),
        "magnitude_max": float(np.nanmax(mag_comp)),
        "direction_min": float(np.nanmin(dir_comp)),
        "direction_max": float(np.nanmax(dir_comp)),
        "combined_min": float(np.nanmin(combined)),
        "combined_max": float(np.nanmax(combined)),
        "alpha_min": float(np.nanmin(alpha)),
        "alpha_max": float(np.nanmax(alpha)),
        "ref_value": float(ref),
    }
    meta_entry = {
        "params": {
            "direction_weight": float(params["direction_weight"]),
            "magnitude_weight": float(params["magnitude_weight"]),
            "indicator_gain": float(gain),
            "indicator_neutral": params["indicator_neutral"],
            "indicator_percentile": float(params["indicator_percentile"]),
            "alpha_min": float(alpha_min),
            "alpha_max": float(alpha_max),
            "indicator_clip": list(params["indicator_clip"]),
        },
        "stats": stats,
        "clip_percentiles": list(params["indicator_clip"]),
        "ref_value": float(ref),
    }
    return {
        "magnitude": mag_comp,
        "direction": dir_comp,
        "combined": combined,
        "alpha": alpha,
        "meta": meta_entry,
    }


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
    parser.add_argument("--no-indicator", action="store_true",
                        help="Skip B-field indicator/alpha post-processing")
    parser.add_argument("--no-bh", action="store_true",
                        help="Skip BH curves and run a single linear solve using mu_r from the mesh")
    parser.add_argument("--bh-max-iters", type=int, default=NEWTON_MAX_ITERS,
                        help="Cap Newton BH iterations (default: 4)")
    parser.add_argument("--bh-residual-tol", type=float, default=NEWTON_RES_TOL,
                        help="Residual tolerance for Newton BH solve (default: 1e-4)")
    parser.add_argument("--bh-min-drop", type=float, default=NEWTON_MIN_DROP,
                        help="Early-stop if residual improves by less than this fraction (default: 0.05)")
    parser.add_argument("--bh-enable-newton", action="store_true", default=BH_ENABLE_NEWTON_DEFAULT,
                        help="Enable Newton BH solve (defaults to Picard-only for stability/speed)")
    parser.add_argument("--bh-allow-labels",
                        help="Comma-separated substrings; only BH curves whose label contains one survive (case-insensitive)")
    parser.add_argument("--freeze-mu-from",
                        help="Path to B_field.npz containing mu_r_effective to reuse (skips BH iteration)")
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
                          bh_id: np.ndarray,
                          Az: np.ndarray,
                          Bx: np.ndarray,
                          By: np.ndarray,
                          Bmag: np.ndarray,
                          meta: dict[str, object],
                          *,
                          mu_r_effective: np.ndarray | None = None,
                          contour_segments: list[dict[str, object]] | None = None,
                          contour_totals: list[dict[str, object]] | None = None,
                          indicator: dict[str, object] | None = None):
    case_dir = case_mesh_path.parent
    case_dir.mkdir(parents=True, exist_ok=True)
    az_path = case_dir / "Az_field.npz"
    az_payload = dict(Az=Az, nodes=nodes, tris=tris,
                      region_id=region, mu_r=mu_r, bh_id=bh_id, meta=meta)
    if mu_r_effective is not None:
        az_payload["mu_r_effective"] = mu_r_effective
    np.savez_compressed(az_path, **az_payload)

    b_path = case_dir / "B_field.npz"
    payload = dict(
        Bx=Bx, By=By, Bmag=Bmag,
        tris=tris, nodes=nodes,
        region_id=region, mu_r=mu_r, bh_id=bh_id, meta=meta,
    )
    if mu_r_effective is not None:
        payload["mu_r_effective"] = mu_r_effective
    if indicator:
        payload["indicator_magnitude"] = indicator.get("magnitude")
        payload["indicator_direction"] = indicator.get("direction")
        payload["indicator_combined"] = indicator.get("combined")
        payload["alpha"] = indicator.get("alpha")
    if contour_segments is not None:
        payload["contour_segments"] = np.array(contour_segments, dtype=object)
    if contour_totals is not None:
        payload["contour_totals"] = np.array(contour_totals, dtype=object)
    np.savez_compressed(b_path, **payload)
    return az_path, b_path


def _append_diagnostics(case_dir: Path, entry: dict):
    """Append a JSON line of diagnostics to cases/<case>/diagnostics.log."""
    try:
        path = case_dir / "diagnostics.log"
        entry = dict(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str))
            handle.write("\n")
    except Exception:
        pass

if __name__ == "__main__":
    t_start = time.perf_counter()
    args = _parse_args()
    case_arg = args.case_override if args.case_override is not None else args.case
    mesh_path = _resolve_mesh_path(case_arg, Path(args.cases_dir))
    nodes, tris, region, mu_r_mesh, Mx, My, Jz, meta, bh_id = load_case(mesh_path)
    bh_curves = _bh_curves_from_meta(meta)
    # Optional: allow only specific BH labels
    if args.bh_allow_labels:
        allowed = [s.strip().lower() for s in args.bh_allow_labels.split(",") if s.strip()]
        if allowed:
            keep_ids = []
            for curve in bh_curves:
                label = str(curve.get("label", "")).lower()
                if any(tok in label for tok in allowed):
                    keep_ids.append(curve.get("id"))
            keep_ids = [cid for cid in keep_ids if cid is not None]
            if keep_ids:
                mask_keep = np.isin(bh_id, keep_ids)
                bh_id = np.where(mask_keep, bh_id, -1)
                bh_curves = [c for c in bh_curves if c.get("id") in keep_ids]
    # If steel elements lack a BH curve but a steel BH is available, assign a default
    if bh_curves:
        steel_curve_ids = [c.get("id") for c in bh_curves if str(c.get("material", "")).lower() == "steel"]
        if steel_curve_ids:
            missing_mask = (region == STEEL) & (bh_id < 0)
            if np.any(missing_mask):
                bh_id = bh_id.copy()
                bh_id[missing_mask] = steel_curve_ids[0]

    # Optional: reuse frozen mu_r_effective from previous solve
    if args.freeze_mu_from:
        freeze_path = Path(args.freeze_mu_from).expanduser()
        try:
            with np.load(freeze_path, allow_pickle=True) as payload:
                mu_frozen = payload.get("mu_r_effective")
                tris_prev = payload.get("tris")
            if mu_frozen is not None and tris_prev is not None and mu_frozen.size == tris.shape[0] and np.array_equal(tris_prev, tris):
                mu_r_mesh = np.asarray(mu_frozen, dtype=float)
                bh_id = np.full_like(bh_id, -1)
                bh_curves = []
                meta = dict(meta)
                meta["bh_solver"] = {
                    "method": "frozen_mu",
                    "source": str(freeze_path),
                }
        except Exception:
            pass

    bh_curves_present = bool(bh_curves) and np.any(bh_id >= 0)
    use_bh = (not args.no_bh) and bh_curves_present
    bh_iters = 0
    bh_converged = True
    solve_method = "linear"
    t_solve_start = time.perf_counter()
    bh_diag = []
    bh_fallback_reason = None
    if use_bh:
        use_newton = bool(args.bh_enable_newton)
        method_used = "picard"
        if use_newton:
            Az, mu_r_eff, Bx, By, Bmag, bh_iters, bh_converged, bh_diag, bh_fallback_reason = _newton_bh_solve(
                nodes, tris, region, mu_r_mesh, Mx, My, Jz, bh_id, bh_curves,
                pin_node=args.pin_node, max_iters=max(int(args.bh_max_iters), 1),
                res_tol=max(float(args.bh_residual_tol), 0.0), min_drop=max(float(args.bh_min_drop), 0.0)
            )
            method_used = "newton"
        else:
            bh_fallback_reason = "skip_newton_default"
        if not use_newton or not bh_converged:
            # Short, strongly damped Picard to avoid long thrashing.
            picard_max_iters = min(5, BH_MAX_ITERS)
            picard_relax = 0.2
            start_mu = mu_r_mesh if not use_newton else mu_r_eff
            Az, mu_r_eff, Bx, By, Bmag, bh_iters, bh_converged, picard_diag = _picard_bh_solve(
                nodes, tris, region, start_mu, Mx, My, Jz, bh_id, bh_curves,
                pin_node=args.pin_node, relax=picard_relax, max_iters=picard_max_iters, tol=BH_TOL
            )
            bh_diag.extend(picard_diag)
            method_used = "picard" if bh_fallback_reason == "skip_newton_default" or not use_newton else "picard_fallback"
        solve_method = method_used
        relax_used = picard_relax if (not use_newton or not bh_converged) else BH_RELAX_DEFAULT
        meta = dict(meta)
        meta["bh_solver"] = {
            "iterations": int(bh_iters),
            "converged": bool(bh_converged),
            "method": method_used,
            "relax": float(relax_used),
            "tol": float(BH_TOL),
            "fallback_reason": bh_fallback_reason,
        }
        mu_r_effective = mu_r_eff
    else:
        K, f = assemble_A_system(nodes, tris, region, mu_r_mesh, Mx, My, Jz)
        Az = solve_neumann_pinned(K, f, pin_node=args.pin_node)
        Bx, By, Bmag = compute_triangle_B(nodes, tris, Az)
        mu_r_effective = None
        if args.no_bh and bh_curves_present:
            meta = dict(meta)
            meta["bh_solver"] = {"method": "disabled"}
    indicator = None
    if not args.no_indicator:
        indicator = _compute_indicator_and_alpha(nodes, tris, Bx, By, Bmag, meta)
    if indicator and indicator.get("meta"):
        # Store indicator/alpha provenance in the mesh metadata so downstream tools can reuse it.
        meta = dict(meta)
        meta["field_indicator"] = indicator["meta"]
    contour_segments, contour_totals = _compute_contour_forces(meta, nodes, tris, Bx, By)
    az_path, b_path = _write_solution_files(mesh_path, nodes, tris, region,
                                            mu_r_mesh, bh_id, Az, Bx, By, Bmag, meta,
                                            mu_r_effective=mu_r_effective,
                                            indicator=indicator,
                                            contour_segments=contour_segments or None,
                                            contour_totals=contour_totals or None)
    t_total = time.perf_counter() - t_start
    t_solve = time.perf_counter() - t_solve_start
    print(f"Solved A_z on {nodes.shape[0]} nodes (tris={tris.shape[0]})")
    print(f"  A_z range: [{Az.min():.6g}, {Az.max():.6g}]  -> {az_path}")
    print(f"  |B| range: [{Bmag.min():.6g}, {Bmag.max():.6g}]  -> {b_path}")
    if mu_r_effective is not None:
        status = "converged" if bh_converged else "max iters"
        print(f"  BH iterations: {bh_iters} ({status}), mu_r_eff in [{mu_r_effective.min():.4g}, {mu_r_effective.max():.4g}]")
    if contour_totals:
        for entry in contour_totals:
            label = entry.get("contour_label") or f"contour {entry.get('contour_index')}"
            fx, fy = entry.get("net_force", [0.0, 0.0])
            torque = entry.get("net_torque", 0.0)
            print(f"  Net force on {label}: Fx={fx:.6g} N/m, Fy={fy:.6g} N/m, Torque={torque:.6g} N·m/m")
    diag = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "case": mesh_path.parent.name,
        "mesh_path": str(mesh_path),
        "nodes": int(nodes.shape[0]),
        "tris": int(tris.shape[0]),
        "bh_enabled": bool(use_bh),
        "solver_method": solve_method,
        "bh_iterations": int(bh_iters),
        "bh_converged": bool(bh_converged),
        "bh_fallback_reason": bh_fallback_reason,
        "bh_args": {
            "max_iters": int(args.bh_max_iters),
            "residual_tol": float(args.bh_residual_tol),
            "min_drop": float(args.bh_min_drop),
            "enable_newton": bool(args.bh_enable_newton),
            "allow_labels": args.bh_allow_labels,
        },
        "bh_diag": bh_diag,
        "timing_s": {
            "solve": t_solve,
            "total": t_total,
        },
        "mu_r_mesh_range": [float(np.nanmin(mu_r_mesh)), float(np.nanmax(mu_r_mesh))],
        "mu_r_effective_range": [float(np.nanmin(mu_r_effective)), float(np.nanmax(mu_r_effective))] if mu_r_effective is not None else None,
    }
    bh_fit_diag = []
    for curve in bh_curves:
        if "fit_diag" not in curve:
            continue
        bh_fit_diag.append({
            "id": curve.get("id"),
            "label": curve.get("label"),
            "material": curve.get("material"),
            **curve["fit_diag"],
        })
    if bh_fit_diag:
        diag["bh_fit"] = bh_fit_diag
    _append_diagnostics(mesh_path.parent, diag)
