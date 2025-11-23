from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Tuple

from flask import Flask, abort, jsonify, render_template, request

try:
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union

    shapely_available = True
except Exception:  # noqa: BLE001
    shapely_available = False

CASE_DEFINITION_FILENAME = "case_definition.json"
DEFAULT_GRID = {
    "Nx": 160,
    "Ny": 160,
    "Lx": 1.0,
    "Ly": 1.0,
    "mesh": {
        "type": "point_cloud",
        "coarse": 0.02,
        "fine": 0.005,
        "focus_pad": 0.02,
        "focus_falloff": 0.01,
        "focus_materials": ["magnet", "steel", "wire"],
        "quality_min_angle": 28.0,
        "field_focus": {
            "enabled": True,
            "direction_weight": 1.0,
            "magnitude_weight": 1.0,
            "indicator_gain": 1.0,
            "indicator_neutral": None,
            "scale_min": 0.5,
            "scale_max": 2.0,
            "smooth_passes": 1,
        },
    },
}

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
CASES_DIR = (REPO_ROOT / "cases").resolve()
PIPELINE_STEPS = (
    ("mesh", ["python3", "mesh_and_sources.py", "--case"]),
    ("solve_B", ["python3", "solve_B.py", "--case"]),
)
DXF_MAX_VERTICES = 20000


def _approx_circle_vertices(center: Tuple[float, float], radius: float, segments: int = 128) -> List[Tuple[float, float]]:
    cx, cy = center
    pts: List[Tuple[float, float]] = []
    for i in range(max(8, segments)):
        ang = 2 * math.pi * i / segments
        pts.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang)))
    pts.append(pts[0])
    return pts

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
        "grid": json.loads(json.dumps(DEFAULT_GRID)),
        "objects": [],
        "defaults": {
            "magnet": {"mu_r": 1.05, "My": 8e5},
            "steel": {"mu_r": 1000.0},
            "wire": {"current": 5000.0},
            "contour": {},
        },
    }


def _write_definition(path: Path, definition: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(definition, handle, indent=2)
        handle.write("\n")


def _run_pipeline(case_name: str, *, mesh_extra_args: List[str] | None = None) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for step, base_command in PIPELINE_STEPS:
        command = [*base_command, case_name]
        if step == "mesh" and mesh_extra_args:
            command.extend(mesh_extra_args)
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


def _shape_bounds(shape: Dict[str, Any]) -> Tuple[float, float, float, float] | None:
    """Compute an axis-aligned bounding box for a shape dict."""
    stype = str(shape.get("type", "")).lower()
    if stype == "circle":
        center = shape.get("center", [0.0, 0.0])
        radius = float(shape.get("radius", 0.0))
        if radius <= 0:
            return None
        cx = float(center[0])
        cy = float(center[1])
        return cx - radius, cx + radius, cy - radius, cy + radius
    if stype == "polygon":
        verts = shape.get("vertices") or []
        holes = shape.get("holes") or []
        all_verts = list(verts) + [pt for hole in holes for pt in hole]
        if not all_verts:
            return None
        xs = [float(v[0]) for v in all_verts]
        ys = [float(v[1]) for v in all_verts]
        return min(xs), max(xs), min(ys), max(ys)
    if stype == "rect":
        center = shape.get("center", [0.0, 0.0])
        cx, cy = float(center[0]), float(center[1])
        width = float(shape.get("width", 0.0))
        height = float(shape.get("height", 0.0))
        if width <= 0 or height <= 0:
            return None
        return cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2
def _decode_dxf_text(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="ignore")


def _normalize_angle(deg: float) -> float:
    wrapped = deg % 360.0
    if wrapped < 0:
        wrapped += 360.0
    return wrapped


def _polar_point(cx: float, cy: float, radius: float, degrees: float) -> Tuple[float, float]:
    rad = math.radians(degrees)
    return cx + radius * math.cos(rad), cy + radius * math.sin(rad)


def _sample_arc_points(
    center: Tuple[float, float], radius: float, start_deg: float, end_deg: float
) -> List[Tuple[float, float]]:
    """Approximate a DXF ARC with a dense set of points along its sweep."""
    start = _normalize_angle(start_deg)
    end = _normalize_angle(end_deg)
    span = end - start
    if span <= 0:
        span += 360.0
    steps = max(24, int(math.ceil(span / 6.0)))
    pts: List[Tuple[float, float]] = []
    for i in range(steps + 1):
        t = i / steps
        ang = start + span * t
        pts.append(_polar_point(center[0], center[1], radius, ang))
    return pts


def _parse_ascii_dxf(
    file_bytes: bytes,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]], List[str], Dict[str, float] | None]:
    """Lightweight DXF parser mirroring the viewer logic (LINE/ARC/CIRCLE only).

    Returns segments, unique points, warnings, bounds.
    """
    text = _decode_dxf_text(file_bytes)
    lines = text.replace("\r\n", "\n").split("\n")
    if len(lines) < 2:
        return [], [], ["DXF file is empty or malformed."], None
    pairs: List[Tuple[str, str]] = []
    for i in range(0, len(lines) - 1, 2):
        pairs.append((lines[i].strip(), lines[i + 1].strip()))
    segments: List[Dict[str, Any]] = []
    points: List[Dict[str, float]] = []
    warnings: List[str] = []
    bounds = {"xmin": math.inf, "xmax": -math.inf, "ymin": math.inf, "ymax": -math.inf}
    point_index: Dict[Tuple[int, int], int] = {}

    def include(x: float, y: float) -> None:
        bounds["xmin"] = min(bounds["xmin"], x)
        bounds["xmax"] = max(bounds["xmax"], x)
        bounds["ymin"] = min(bounds["ymin"], y)
        bounds["ymax"] = max(bounds["ymax"], y)

    def add_point(x: float, y: float) -> int:
        key = (round(x * 1e6), round(y * 1e6))
        idx = point_index.get(key)
        if idx is None:
            idx = len(points)
            points.append({"id": idx + 1, "x": x, "y": y})
            point_index[key] = idx
        include(x, y)
        return idx

    in_entities = False
    i = 0
    while i < len(pairs):
        code, value = pairs[i]
        if code == "0" and value == "SECTION" and i + 1 < len(pairs) and pairs[i + 1][1] == "ENTITIES":
            in_entities = True
            i += 2
            continue
        if code == "0" and value == "ENDSEC" and in_entities:
            in_entities = False
        if not in_entities or code != "0":
            i += 1
            continue

        entity_type = value.upper()
        data: Dict[str, str] = {}
        i += 1
        while i < len(pairs):
            c, v = pairs[i]
            if c == "0":
                break
            data[c] = v
            i += 1

        def read_float(key: str, default: float | None = None) -> float | None:
            raw = data.get(key)
            if raw is None:
                return default
            try:
                return float(raw)
            except Exception:  # noqa: BLE001
                return default

        if entity_type == "LINE":
            sx = read_float("10")
            sy = read_float("20")
            ex = read_float("11")
            ey = read_float("21")
            if None not in (sx, sy, ex, ey):
                a_idx = add_point(sx, sy)
                b_idx = add_point(ex, ey)
                segments.append({"type": "LINE", "start": a_idx, "end": b_idx})
            else:
                warnings.append("LINE missing coordinates; skipped.")
        elif entity_type == "ARC":
            cx = read_float("10")
            cy = read_float("20")
            radius = read_float("40")
            start_ang = read_float("50", 0.0)
            end_ang = read_float("51", 0.0)
            if None not in (cx, cy, radius, start_ang, end_ang) and radius and radius > 0:
                include(cx - radius, cy - radius)
                include(cx + radius, cy + radius)
                start_pt = _polar_point(cx, cy, radius, start_ang or 0.0)
                end_pt = _polar_point(cx, cy, radius, end_ang or 0.0)
                a_idx = add_point(start_pt[0], start_pt[1])
                b_idx = add_point(end_pt[0], end_pt[1])
                segments.append(
                    {
                        "type": "ARC",
                        "start": a_idx,
                        "end": b_idx,
                        "center": {"x": cx, "y": cy},
                        "radius": radius,
                        "start_angle": start_ang or 0.0,
                        "end_angle": end_ang or 0.0,
                    }
                )
            else:
                warnings.append("ARC missing coordinates; skipped.")
        elif entity_type == "CIRCLE":
            cx = read_float("10")
            cy = read_float("20")
            radius = read_float("40")
            if None not in (cx, cy, radius) and radius and radius > 0:
                include(cx - radius, cy - radius)
                include(cx + radius, cy + radius)
                start_pt = _polar_point(cx, cy, radius, 0.0)
                idx = add_point(start_pt[0], start_pt[1])
                segments.append(
                    {
                        "type": "CIRCLE",
                        "start": idx,
                        "end": idx,
                        "center": {"x": cx, "y": cy},
                        "radius": radius,
                        "start_angle": 0.0,
                        "end_angle": 360.0,
                    }
                )
            else:
                warnings.append("CIRCLE missing coordinates; skipped.")
        else:
            warnings.append(f"Unsupported entity {entity_type}; skipped.")
        # Loop continues with i at next "0" (either ENDSEC or next entity)

    if not segments:
        warnings.append("No LINE/ARC/CIRCLE entities found in ENTITIES section.")
    finite_bounds = None
    if all(math.isfinite(bounds[k]) for k in bounds):
        finite_bounds = {"xmin": bounds["xmin"], "xmax": bounds["xmax"], "ymin": bounds["ymin"], "ymax": bounds["ymax"]}
    return segments, points, warnings, finite_bounds


def _segments_to_polygon_shape(segments: List[Dict[str, Any]]) -> Tuple[Dict[str, Any] | None, List[str]]:
    """Stitch segments into a single polygon with holes, mirroring the viewer output."""
    warnings: List[str] = []

    def _pts_from_segment(seg: Dict[str, Any]) -> List[Tuple[float, float]]:
        stype = str(seg.get("type", "")).upper()
        if stype == "LINE":
            start = seg.get("start")
            end = seg.get("end")
            if start and end:
                try:
                    return [(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))]
                except Exception:  # noqa: BLE001
                    return []
            return []
        if stype == "ARC":
            center_raw = seg.get("center")
            try:
                cx = float(center_raw[0]) if isinstance(center_raw, (list, tuple)) else float(center_raw.get("x", 0.0))
                cy = float(center_raw[1]) if isinstance(center_raw, (list, tuple)) else float(center_raw.get("y", 0.0))
            except Exception:  # noqa: BLE001
                return []
            radius = float(seg.get("radius", 0.0))
            start_ang = float(seg.get("start_angle", 0.0))
            end_ang = float(seg.get("end_angle", 0.0))
            if radius > 0:
                return _sample_arc_points((cx, cy), radius, start_ang, end_ang)
            return []
        if stype == "CIRCLE":
            center_raw = seg.get("center")
            try:
                cx = float(center_raw[0]) if isinstance(center_raw, (list, tuple)) else float(center_raw.get("x", 0.0))
                cy = float(center_raw[1]) if isinstance(center_raw, (list, tuple)) else float(center_raw.get("y", 0.0))
            except Exception:  # noqa: BLE001
                return []
            radius = float(seg.get("radius", 0.0))
            if radius > 0:
                pts = _approx_circle_vertices((cx, cy), radius, segments=256)
                return pts
            return []
        return []

    def _points_close(a: Tuple[float, float], b: Tuple[float, float], tol: float = 1e-5) -> bool:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= tol * tol

    def _poly_area(pts: List[Tuple[float, float]]) -> float:
        if len(pts) < 3:
            return 0.0
        area = 0.0
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            area += x0 * y1 - x1 * y0
        return 0.5 * area

    def _collect_polylines() -> List[List[Tuple[float, float]]]:
        polylines: List[List[Tuple[float, float]]] = []
        for seg in segments:
            pts = _pts_from_segment(seg)
            if len(pts) < 2:
                continue
            if not _points_close(pts[0], pts[-1]):
                polylines.append(pts)
            else:
                polylines.append(pts)
        return polylines

    def _stitch_graph() -> Tuple[Dict[str, Any] | None, List[str]]:
        polylines = _collect_polylines()
        if not polylines:
            return None, ["Could not build loops from DXF entities."]

        def key(pt: Tuple[float, float], tol: float = 1e-5) -> Tuple[int, int]:
            return (round(pt[0] / tol), round(pt[1] / tol))

        loops: List[List[Tuple[float, float]]] = []
        while polylines:
            current = polylines.pop(0)
            changed = True
            while changed and polylines:
                changed = False
                end = current[-1]
                start_key = key(end)
                for idx, poly in enumerate(polylines):
                    if _points_close(poly[0], end):
                        current.extend(poly[1:])
                        polylines.pop(idx)
                        changed = True
                        break
                    if _points_close(poly[-1], end):
                        current.extend(reversed(poly[:-1]))
                        polylines.pop(idx)
                        changed = True
                        break
                if changed:
                    continue
                # try connecting to start of current (loop closure)
                start = current[0]
                end_key = key(start)
                for idx, poly in enumerate(polylines):
                    if key(poly[-1]) == end_key and _points_close(poly[-1], start):
                        current = list(reversed(poly[:-1])) + current
                        polylines.pop(idx)
                        changed = True
                        break
            if _points_close(current[0], current[-1]):
                if not _points_close(current[0], current[-1], tol=0.0):
                    current.append(current[0])
                loops.append(current)
            else:
                # force close if near
                if _points_close(current[0], current[-1], tol=1e-3):
                    current.append(current[0])
                    loops.append(current)
        if not loops:
            return None, ["Could not assemble any closed loops from DXF entities."]
        loops = sorted(loops, key=lambda pts: abs(_poly_area(pts)), reverse=True)
        outer = loops[0]
        holes = loops[1:]
        return {"type": "polygon", "vertices": outer, "holes": holes}, ["Stitched without shapely (graph chaining)."]

    if shapely_available:
        lines: List[LineString] = []
        for seg in segments:
            pts = _pts_from_segment(seg)
            if len(pts) < 2:
                continue
            lines.extend(LineString([pts[i], pts[i + 1]]) for i in range(len(pts) - 1))
        if not lines:
            warnings.append("No DXF outlines found to stitch.")
            return None, warnings
        polygons = list(polygonize(lines))
        if not polygons:
            warnings.append("Polygonization produced no closed loops; check DXF integrity.")
            return None, warnings
        merged = unary_union(polygons)
        if merged.geom_type == "MultiPolygon":
            geoms = sorted(merged.geoms, key=lambda p: p.area, reverse=True)
            if len(geoms) > 1:
                warnings.append(f"DXF contained {len(geoms)} disconnected regions; keeping the largest.")
            merged = geoms[0]
        if merged.geom_type != "Polygon":
            warnings.append("DXF geometry could not be reduced to a polygon.")
            return None, warnings
        shape = {
            "type": "polygon",
            "vertices": list(merged.exterior.coords),
            "holes": [list(interior.coords) for interior in merged.interiors],
        }
        return shape, warnings

    fallback_shape, fallback_warnings = _stitch_graph()
    warnings.extend(fallback_warnings)
    return fallback_shape, warnings


def _import_dxf(file_bytes: bytes) -> Dict[str, Any]:
    segments, points, warnings, bounds = _parse_ascii_dxf(file_bytes)
    shape, stitch_warnings = _segments_to_polygon_shape(
        [
            {
                **seg,
                "start": points[seg["start"]],
                "end": points[seg["end"]],
            }
            for seg in segments
            if isinstance(seg.get("start"), int) and isinstance(seg.get("end"), int) and seg["start"] < len(points) and seg["end"] < len(points)
        ]
    )
    warnings.extend(stitch_warnings)
    shapes: List[Dict[str, Any]] = []
    if shape:
        shapes.append({"shape": shape, "source": "ascii-parser", "layer": ""})
    if len(shapes) > DXF_MAX_VERTICES:
        warnings.append(
            f"DXF produced many shapes ({len(shapes)}); only the first {DXF_MAX_VERTICES} are kept."
        )
        shapes = shapes[:DXF_MAX_VERTICES]
    return {
        "shapes": shapes,
        "warnings": warnings,
        "bounds": bounds,
        "segments": segments,
        "points": points,
    }


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


@app.post("/api/import/dxf")
def api_import_dxf():
    if "file" not in request.files:
        abort(400, description="DXF file is required (multipart/form-data with field 'file').")
    file = request.files["file"]
    data = file.read()
    if not data:
        abort(400, description="Uploaded DXF file was empty.")
    try:
        result = _import_dxf(data)
    except RuntimeError as exc:
        abort(500, description=str(exc))
    return jsonify(result)


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


@app.post("/api/case/<path:case_name>/run-adaptive")
def api_run_case_adaptive(case_name: str):
    try:
        normalized = _normalize_case_name(case_name)
        path = _case_definition_path(normalized)
    except ValueError as exc:
        abort(400, description=str(exc))
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        abort(400, description="Expected JSON object for case definition")
    if not path.exists():
        abort(400, description="No saved definition for this case. Run the normal pipeline first.")
    saved_definition = _load_definition(path)
    if payload != saved_definition:
        abort(409, description="Definition has changed since the last solve. Run the standard case first.")
    b_field_path = (_case_dir(normalized) / "B_field.npz").resolve()
    if not b_field_path.exists():
        abort(400, description="No B_field.npz found. Run the standard case once before adaptive refinement.")
    steps = _run_pipeline(normalized, mesh_extra_args=["--adapt-from", str(b_field_path)])
    success = len(steps) == len(PIPELINE_STEPS) and all(step["success"] for step in steps)
    response: Dict[str, Any] = {
        "case": normalized,
        "path": str(path),
        "steps": steps,
        "succeeded": success,
        "mode": "field_point_cloud",
        "adapt_from": str(b_field_path),
    }
    if success:
        return jsonify(response)
    failed_step = next((step for step in reversed(steps) if not step["success"]), None)
    if failed_step:
        response["error"] = f"{failed_step['step']} failed with exit code {failed_step['returncode']}"
    else:
        response["error"] = "Adaptive run did not complete successfully."
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
