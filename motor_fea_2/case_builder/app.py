from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, List

from flask import Flask, abort, jsonify, render_template, request

CASE_DEFINITION_FILENAME = "case_definition.json"
DEFAULT_GRID = {"Nx": 120, "Ny": 120, "Lx": 1.0, "Ly": 1.0}

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
CASES_DIR = (REPO_ROOT / "cases").resolve()
PIPELINE_STEPS = (
    ("mesh", ["python3", "mesh_and_sources.py", "--case"]),
    ("solve_B", ["python3", "solve_B.py", "--case"]),
)

app = Flask(__name__, static_folder="static", template_folder="templates")


def _normalize_case_name(name: str | None) -> str:
    if not name:
        raise ValueError("Case name is required")
    candidate = Path(name)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise ValueError("Case name cannot traverse directories")
    return candidate.as_posix()


def _case_dir(case_name: str) -> Path:
    normalized = _normalize_case_name(case_name)
    case_path = (CASES_DIR / normalized).resolve()
    if not str(case_path).startswith(str(CASES_DIR)):
        raise ValueError("Case path must reside within the cases/ directory")
    return case_path


def _case_definition_path(case_name: str) -> Path:
    return _case_dir(case_name) / CASE_DEFINITION_FILENAME


def _list_case_names() -> List[str]:
    if not CASES_DIR.exists():
        return []
    mesh_dirs = {
        path.parent.relative_to(CASES_DIR).as_posix()
        for path in CASES_DIR.rglob("mesh.npz")
        if path.is_file()
    }
    if mesh_dirs:
        return sorted(mesh_dirs)
    def_dirs = {
        path.parent.relative_to(CASES_DIR).as_posix()
        for path in CASES_DIR.rglob(CASE_DEFINITION_FILENAME)
        if path.is_file()
    }
    if def_dirs:
        return sorted(def_dirs)
    return sorted(p.relative_to(CASES_DIR).as_posix() for p in CASES_DIR.iterdir() if p.is_dir())


def _load_definition(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return _default_definition(path.parent.name)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _default_definition(case_name: str) -> Dict[str, Any]:
    return {
        "name": case_name,
        "grid": DEFAULT_GRID.copy(),
        "objects": [],
        "defaults": {
            "magnet": {"mu_r": 1.05, "My": 8e5},
            "steel": {"mu_r": 1000.0},
            "wire": {"current": 5000.0},
        },
    }


def _write_definition(path: Path, definition: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(definition, handle, indent=2)
        handle.write("\n")


def _run_pipeline(case_name: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for step, base_command in PIPELINE_STEPS:
        command = [*base_command, case_name]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        step_result: Dict[str, Any] = {
            "step": step,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "success": completed.returncode == 0,
        }
        results.append(step_result)
        if completed.returncode != 0:
            break
    return results


@app.route("/")
def index():
    cases = _list_case_names()
    bootstrap = {
        "cases": cases,
        "defaultCase": cases[0] if cases else "",
        "defaultGrid": DEFAULT_GRID,
    }
    return render_template("index.html", bootstrap=bootstrap)


@app.get("/api/cases")
def api_cases():
    return jsonify({"cases": _list_case_names()})


@app.get("/api/case/<path:case_name>/definition")
def api_get_definition(case_name: str):
    try:
        path = _case_definition_path(case_name)
        definition = _load_definition(path)
        return jsonify({
            "case": _normalize_case_name(case_name),
            "definition": definition,
            "path": str(path),
        })
    except ValueError as exc:
        abort(400, description=str(exc))


@app.post("/api/case/<path:case_name>/definition")
def api_save_definition(case_name: str):
    try:
        path = _case_definition_path(case_name)
    except ValueError as exc:
        abort(400, description=str(exc))
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        abort(400, description="Expected JSON object for case definition")
    _write_definition(path, payload)
    return jsonify({"saved": True, "case": _normalize_case_name(case_name), "path": str(path)})


@app.post("/api/case/<path:case_name>/run")
def api_run_case(case_name: str):
    try:
        normalized = _normalize_case_name(case_name)
        path = _case_definition_path(normalized)
    except ValueError as exc:
        abort(400, description=str(exc))
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        abort(400, description="Expected JSON object for case definition")
    _write_definition(path, payload)
    steps = _run_pipeline(normalized)
    success = len(steps) == len(PIPELINE_STEPS) and all(step["success"] for step in steps)
    response: Dict[str, Any] = {
        "case": normalized,
        "path": str(path),
        "steps": steps,
        "succeeded": success,
    }
    if success:
        return jsonify(response)
    failed_step = next((step for step in reversed(steps) if not step["success"]), None)
    if failed_step:
        response["error"] = f"{failed_step['step']} failed with exit code {failed_step['returncode']}"
    else:
        response["error"] = "Run pipeline did not complete successfully."
    return jsonify(response), 500


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Case builder dev server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
