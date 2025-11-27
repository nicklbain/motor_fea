from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import gzip
import json

import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
CASES_DIR = (REPO_ROOT / "cases").resolve()
DEFAULT_CASE_SUBDIR = "mag2d_case"
DEFAULT_MESH_NAME = "mesh.npz"
LEGACY_DEFAULT_FILENAME = "mag2d_case.npz"

app = Flask(__name__, static_folder="static", template_folder="templates")


def _finite_array(arr: Any, *, fill: float = 0.0) -> np.ndarray:
    """Return a numpy array with NaN/Inf replaced by `fill` (or -fill)."""
    out = np.asarray(arr)
    if out.dtype.kind in {"f", "c"}:
        out = np.nan_to_num(out, nan=fill, posinf=fill, neginf=-fill)
    return out


def _to_finite_list(arr: Any, *, fill: float = 0.0) -> list:
    return _finite_array(arr, fill=fill).tolist()


def _sanitize_obj(value: Any, *, fill: float = 0.0) -> Any:
    """Recursively replace NaN/Inf in JSON-like objects."""
    if isinstance(value, dict):
        return {k: _sanitize_obj(v, fill=fill) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_obj(v, fill=fill) for v in value]
    if isinstance(value, np.ndarray):
        return _to_finite_list(value, fill=fill)
    if isinstance(value, np.generic):
        val = value.item()
        if isinstance(val, float) and not math.isfinite(val):
            return 0.0
        return val
    if isinstance(value, float):
        return 0.0 if not math.isfinite(value) else value
    return value


def _list_case_names() -> List[str]:
    """Return available case names, preferring directories with mesh.npz."""

    if not CASES_DIR.exists():
        return []

    dir_candidates = sorted(
        {
            path.parent.relative_to(CASES_DIR).as_posix()
            for path in CASES_DIR.rglob(DEFAULT_MESH_NAME)
            if path.is_file()
        }
    )
    dir_candidates = [name for name in dir_candidates if name]
    if dir_candidates:
        return dir_candidates

    # Fallback for legacy layouts with loose .npz files directly in cases/.
    return sorted(
        str(p.relative_to(CASES_DIR).as_posix())
        for p in CASES_DIR.glob("*.npz")
        if p.is_file()
    )


def _default_case_name() -> str | None:
    preferred = CASES_DIR / DEFAULT_CASE_SUBDIR / DEFAULT_MESH_NAME
    if preferred.exists():
        return DEFAULT_CASE_SUBDIR
    legacy = CASES_DIR / LEGACY_DEFAULT_FILENAME
    if legacy.exists():
        return legacy.relative_to(CASES_DIR).as_posix()
    cases = _list_case_names()
    return cases[0] if cases else None


def _resolve_case_path(case_name: str | None) -> Path:
    if not CASES_DIR.exists():
        raise FileNotFoundError(
            "The cases/ directory does not exist yet. Generate a mesh first."
        )
    if case_name in (None, "", "default"):
        case_name = _default_case_name()
        if not case_name:
            raise FileNotFoundError("No .npz meshes found in cases/.")
    candidate = (CASES_DIR / case_name).resolve()
    if candidate.is_dir():
        candidate = (candidate / DEFAULT_MESH_NAME).resolve()
    if not str(candidate).startswith(str(CASES_DIR)):
        raise ValueError("Case path must stay under the cases/ directory.")
    if not candidate.exists():
        raise FileNotFoundError(f"Mesh case '{case_name}' was not found.")
    return candidate


def _json_ready_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _sanitize_obj(v) for k, v in meta.items()}


def _field_stats(arr: np.ndarray) -> Dict[str, float]:
    arr = _finite_array(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"min": 0.0, "max": 0.0}
    return {"min": float(finite.min()), "max": float(finite.max())}


def _triangle_neighbors(tris: np.ndarray) -> list[list[int]]:
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


def _angle_diff(rad_a: float, rad_b: float) -> float:
    diff = rad_a - rad_b
    return abs(math.atan2(math.sin(diff), math.cos(diff)))


def _field_gradient_components(
    Bmag: np.ndarray,
    Bx: np.ndarray,
    By: np.ndarray,
    centroids: np.ndarray,
    neighbors: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    Ne = Bmag.shape[0]
    mag_comp = np.zeros(Ne, dtype=float)
    dir_comp = np.zeros(Ne, dtype=float)
    angles = np.arctan2(By, Bx)
    for idx in range(Ne):
        c = centroids[idx]
        best_mag = 0.0
        best_dir = 0.0
        for nb in neighbors[idx]:
            dist = float(np.hypot(c[0] - centroids[nb, 0], c[1] - centroids[nb, 1]))
            if dist <= 1e-12:
                dist = 1e-12
            best_mag = max(best_mag, abs(Bmag[idx] - Bmag[nb]) / dist)
            angle_change = _angle_diff(angles[idx], angles[nb]) / dist
            if math.isfinite(angle_change):
                best_dir = max(best_dir, angle_change)
        mag_comp[idx] = best_mag
        dir_comp[idx] = best_dir
    return mag_comp, dir_comp


def _indicator_payload(
    nodes: np.ndarray,
    tris: np.ndarray,
    Bmag: np.ndarray,
    Bx: np.ndarray,
    By: np.ndarray,
    *,
    indicator_arrays: Dict[str, np.ndarray] | None = None,
    indicator_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    if tris.ndim != 2 or tris.shape[1] != 3:
        return None
    if Bmag.shape[0] != tris.shape[0]:
        return None
    pre_mag = None
    pre_dir = None
    pre_combined = None
    pre_alpha = None
    if indicator_arrays:
        pre_mag = indicator_arrays.get("indicator_magnitude")
        pre_dir = indicator_arrays.get("indicator_direction")
        pre_combined = indicator_arrays.get("indicator_combined")
        pre_alpha = indicator_arrays.get("alpha")
    try:
        centroids = nodes[tris].mean(axis=1)
    except Exception:  # noqa: BLE001
        return None
    if (
        pre_mag is not None
        and pre_dir is not None
        and np.asarray(pre_mag).shape[0] == tris.shape[0]
        and np.asarray(pre_dir).shape[0] == tris.shape[0]
    ):
        mag_comp = np.asarray(pre_mag, dtype=float)
        dir_comp = np.asarray(pre_dir, dtype=float)
        if pre_combined is not None and np.asarray(pre_combined).shape[0] == mag_comp.shape[0]:
            combined = np.asarray(pre_combined, dtype=float)
        else:
            combined = mag_comp + dir_comp
    else:
        neighbors = _triangle_neighbors(tris)
        mag_comp, dir_comp = _field_gradient_components(Bmag, Bx, By, centroids, neighbors)
        combined = mag_comp + dir_comp
    mag_comp = _finite_array(mag_comp)
    dir_comp = _finite_array(dir_comp)
    combined = _finite_array(combined)
    stats = {
        "magnitude_min": float(np.min(mag_comp)),
        "magnitude_max": float(np.max(mag_comp)),
        "direction_min": float(np.min(dir_comp)),
        "direction_max": float(np.max(dir_comp)),
        "combined_min": float(np.min(combined)),
        "combined_max": float(np.max(combined)),
    }
    alpha_stats = None
    if pre_alpha is not None and np.asarray(pre_alpha).shape[0] == combined.shape[0]:
        alpha_arr = _finite_array(pre_alpha)
        alpha_stats = {
            "min": float(np.min(alpha_arr)),
            "max": float(np.max(alpha_arr)),
            "median": float(np.median(alpha_arr)),
        }
    payload = {
        "magnitude": mag_comp.tolist(),
        "direction": dir_comp.tolist(),
        "combined": combined.tolist(),
        "stats": stats,
    }
    if alpha_stats:
        payload["alpha_stats"] = alpha_stats
    if pre_alpha is not None and np.asarray(pre_alpha).shape[0] == combined.shape[0]:
        payload["alpha"] = alpha_arr.tolist()
    if indicator_meta:
        params = indicator_meta.get("params")
        if isinstance(params, dict):
            payload["params"] = params
        payload["meta"] = indicator_meta
    return payload


def _load_b_field_payload(mesh_path: Path) -> tuple[Dict[str, Any], Dict[str, float], Dict[str, np.ndarray]] | None:
    b_path = mesh_path.parent / "B_field.npz"
    if not b_path.exists():
        return None
    with np.load(b_path, allow_pickle=True) as data:
        if not {"Bx", "By", "Bmag"} <= set(data.files):
            return None
        Bx = np.asarray(data["Bx"], dtype=float)
        By = np.asarray(data["By"], dtype=float)
        Bmag = np.asarray(data["Bmag"], dtype=float)
        payload: Dict[str, Any] = {
            "Bx": _to_finite_list(Bx),
            "By": _to_finite_list(By),
            "Bmag": _to_finite_list(Bmag),
            "source": b_path.name,
        }
        indicator_arrays: Dict[str, np.ndarray] = {}
        if "indicator_magnitude" in data and "indicator_direction" in data:
            indicator_arrays["indicator_magnitude"] = _finite_array(data["indicator_magnitude"], fill=0.0)
            indicator_arrays["indicator_direction"] = _finite_array(data["indicator_direction"], fill=0.0)
        if "indicator_combined" in data:
            indicator_arrays["indicator_combined"] = _finite_array(data["indicator_combined"], fill=0.0)
        if "alpha" in data:
            indicator_arrays["alpha"] = _finite_array(data["alpha"], fill=0.0)
        if "meta" in data:
            payload["meta"] = _json_ready_meta(data["meta"].item())
        if "contour_segments" in data:
            payload["contour_segments"] = [_sanitize_obj(dict(entry)) for entry in data["contour_segments"].tolist()]
        if "contour_totals" in data:
            payload["contour_totals"] = [_sanitize_obj(dict(entry)) for entry in data["contour_totals"].tolist()]
        stats = _field_stats(Bmag)
    arrays: Dict[str, np.ndarray] = {"Bx": Bx, "By": By, "Bmag": Bmag}
    arrays.update(indicator_arrays)
    return payload, stats, arrays


def _load_mesh_payload(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        nodes = data["nodes"]
        tris = data["tris"]
        payload: Dict[str, Any] = {
            "nodes": _to_finite_list(nodes),
            "tris": tris.tolist(),
            "region_id": _to_finite_list(data.get("region_id", np.zeros(tris.shape[0]))),
            "fields": {},
            "summary": {},
        }

        if nodes.size:
            min_x = float(nodes[:, 0].min())
            max_x = float(nodes[:, 0].max())
            min_y = float(nodes[:, 1].min())
            max_y = float(nodes[:, 1].max())
        else:
            min_x = max_x = min_y = max_y = 0.0
        bounds = {"minX": min_x, "maxX": max_x, "minY": min_y, "maxY": max_y}
        payload["bounds"] = bounds
        payload["summary"].update(
            node_count=int(nodes.shape[0]),
            tri_count=int(tris.shape[0]),
            bbox_width=bounds["maxX"] - bounds["minX"],
            bbox_height=bounds["maxY"] - bounds["minY"],
        )

        field_names = ["mu_r", "Mx", "My", "Jz"]
        field_stats = {}
        for name in field_names:
            if name in data:
                arr = _finite_array(data[name])
                payload["fields"][name] = arr.tolist()
                field_stats[name] = _field_stats(arr)
        if "Mx" in data and "My" in data:
            mag = np.hypot(_finite_array(data["Mx"]), _finite_array(data["My"]))
            payload["fields"]["M_mag"] = mag.tolist()
            field_stats["M_mag"] = _field_stats(mag)

        b_field = _load_b_field_payload(path)
        if b_field:
            payload["bField"] = b_field[0]
            field_stats["Bmag"] = b_field[1]
            if "contour_segments" in b_field[0]:
                payload["contours"] = {
                    "segments": b_field[0].get("contour_segments", []),
                    "totals": b_field[0].get("contour_totals", []),
                }
            indicator_arrays = {
                key: b_field[2][key]
                for key in ("indicator_magnitude", "indicator_direction", "indicator_combined", "alpha")
                if key in b_field[2]
            }
            indicator_info = _indicator_payload(
                nodes,
                tris,
                b_field[2]["Bmag"],
                b_field[2]["Bx"],
                b_field[2]["By"],
                indicator_arrays=indicator_arrays or None,
                indicator_meta=(b_field[0].get("meta") or {}).get("field_indicator") if b_field[0].get("meta") else None,
            )
            if indicator_info:
                payload["indicator"] = indicator_info
                payload["summary"]["indicator_stats"] = indicator_info.get("stats")

        payload["summary"]["field_stats"] = field_stats
        if "meta" in data:
            meta_obj = data["meta"].item()
            payload["meta"] = _json_ready_meta(meta_obj)
        # Sanitize the full payload one layer deep (contours/meta already handled).
        payload = _sanitize_obj(payload)
    payload["source"] = path.name
    return payload


@app.route("/")
def index() -> str:
    bootstrap = {
        "cases": _list_case_names(),
        "defaultCase": _default_case_name(),
    }
    return render_template("index.html", bootstrap=bootstrap)


@app.route("/api/cases")
def api_cases():
    return jsonify({"cases": _list_case_names(), "default": _default_case_name()})


@app.route("/api/mesh")
def api_mesh():
    case_name = request.args.get("case")
    try:
        path = _resolve_case_path(case_name)
        payload = _load_mesh_payload(path)
        # Large meshes can exceed browser limits; compress when big.
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if len(body) > 1_000_000:
            gz = gzip.compress(body)
            return Response(gz, mimetype="application/json", headers={"Content-Encoding": "gzip"})
        return Response(body, mimetype="application/json")
    except FileNotFoundError as exc:
        abort(404, description=str(exc))
    except ValueError as exc:
        abort(400, description=str(exc))


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh viewer dev server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Bind port (default 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
