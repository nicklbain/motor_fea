from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from flask import Flask, abort, jsonify, render_template, request

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
CASES_DIR = (REPO_ROOT / "cases").resolve()
DEFAULT_CASE_SUBDIR = "mag2d_case"
DEFAULT_MESH_NAME = "mesh.npz"
LEGACY_DEFAULT_FILENAME = "mag2d_case.npz"

app = Flask(__name__, static_folder="static", template_folder="templates")


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
    ready: Dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, np.generic):
            ready[key] = value.item()
        elif isinstance(value, (np.ndarray, list, tuple)):
            ready[key] = np.asarray(value).tolist()
        else:
            ready[key] = value
    return ready


def _field_stats(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0}
    return {"min": float(arr.min()), "max": float(arr.max())}


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
) -> Dict[str, Any] | None:
    if tris.ndim != 2 or tris.shape[1] != 3:
        return None
    if Bmag.shape[0] != tris.shape[0]:
        return None
    try:
        centroids = nodes[tris].mean(axis=1)
    except Exception:  # noqa: BLE001
        return None
    neighbors = _triangle_neighbors(tris)
    mag_comp, dir_comp = _field_gradient_components(Bmag, Bx, By, centroids, neighbors)
    combined = mag_comp + dir_comp
    stats = {
        "magnitude_min": float(np.min(mag_comp)),
        "magnitude_max": float(np.max(mag_comp)),
        "direction_min": float(np.min(dir_comp)),
        "direction_max": float(np.max(dir_comp)),
        "combined_min": float(np.min(combined)),
        "combined_max": float(np.max(combined)),
    }
    return {
        "magnitude": mag_comp.tolist(),
        "direction": dir_comp.tolist(),
        "combined": combined.tolist(),
        "stats": stats,
    }


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
            "Bx": Bx.tolist(),
            "By": By.tolist(),
            "Bmag": Bmag.tolist(),
            "source": b_path.name,
        }
        if "meta" in data:
            payload["meta"] = _json_ready_meta(data["meta"].item())
        if "contour_segments" in data:
            payload["contour_segments"] = [dict(entry) for entry in data["contour_segments"].tolist()]
        if "contour_totals" in data:
            payload["contour_totals"] = [dict(entry) for entry in data["contour_totals"].tolist()]
        stats = _field_stats(Bmag)
    arrays = {"Bx": Bx, "By": By, "Bmag": Bmag}
    return payload, stats, arrays


def _load_mesh_payload(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        nodes = data["nodes"]
        tris = data["tris"]
        payload: Dict[str, Any] = {
            "nodes": nodes.tolist(),
            "tris": tris.tolist(),
            "region_id": data.get("region_id", np.zeros(tris.shape[0])).tolist(),
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
                arr = data[name]
                payload["fields"][name] = arr.tolist()
                field_stats[name] = _field_stats(arr)
        if "Mx" in data and "My" in data:
            mag = np.hypot(data["Mx"], data["My"])
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
            indicator_info = _indicator_payload(
                nodes,
                tris,
                b_field[2]["Bmag"],
                b_field[2]["Bx"],
                b_field[2]["By"],
            )
            if indicator_info:
                payload["indicator"] = indicator_info
                payload["summary"]["indicator_stats"] = indicator_info.get("stats")

        payload["summary"]["field_stats"] = field_stats
        if "meta" in data:
            meta_obj = data["meta"].item()
            payload["meta"] = _json_ready_meta(meta_obj)
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
        return jsonify(payload)
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
