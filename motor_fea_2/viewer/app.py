from __future__ import annotations

import argparse
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


def _load_b_field_payload(mesh_path: Path) -> tuple[Dict[str, Any], Dict[str, float]] | None:
    b_path = mesh_path.parent / "B_field.npz"
    if not b_path.exists():
        return None
    with np.load(b_path, allow_pickle=True) as data:
        if not {"Bx", "By", "Bmag"} <= set(data.files):
            return None
        payload: Dict[str, Any] = {
            "Bx": data["Bx"].tolist(),
            "By": data["By"].tolist(),
            "Bmag": data["Bmag"].tolist(),
            "source": b_path.name,
        }
        if "meta" in data:
            payload["meta"] = _json_ready_meta(data["meta"].item())
        stats = _field_stats(data["Bmag"])
    return payload, stats


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
