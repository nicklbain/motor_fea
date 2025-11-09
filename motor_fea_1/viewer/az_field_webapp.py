#!/usr/bin/env python3
"""
az_field_webapp.py
------------------
Interactive B-field visualiser and Maxwell-stress explorer served via Bokeh.

Launch with (from repo root):
    python3 -m bokeh serve --show viewer/az_field_webapp.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    Div,
    LabelSet,
    LinearColorMapper,
    PointDrawTool,
    Segment,
)
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from viewer.az_field_utils import compute_polygon_mst, load_field_snapshot

SNAPSHOT_PATH = REPO_ROOT / "magnetostatics_solver_field_data.npz"

snapshot = load_field_snapshot(SNAPSHOT_PATH)

Bmag = snapshot["Bmag"]
Bx = snapshot["Bx"]
By = snapshot["By"]
Lx = snapshot["Lx"]
Ly = snapshot["Ly"]
dx = snapshot["dx"]
dy = snapshot["dy"]

# Prepare heatmap (flip vertically so y increases upwards in the figure).
Bmag_plot = Bmag[::-1, :]

color_mapper = LinearColorMapper(palette=Viridis256, low=float(Bmag.min()), high=float(Bmag.max()))

heatmap_source = ColumnDataSource(
    data=dict(image=[Bmag_plot], x=[0.0], y=[0.0], dw=[Lx], dh=[Ly])
)

# Prepare quiver-like segments
Ny, Nx = Bmag.shape
xc = snapshot["x_centers"]
yc = snapshot["y_centers"]
XX, YY = np.meshgrid(xc, yc, indexing="xy")

target_vectors_per_axis = 25
step = max(1, int(np.ceil(max(Nx, Ny) / target_vectors_per_axis)))
Bx_sample = Bx[::step, ::step]
By_sample = By[::step, ::step]
X_sample = XX[::step, ::step]
Y_sample = YY[::step, ::step]
Bmag_sample = np.sqrt(Bx_sample * Bx_sample + By_sample * By_sample)

direction_scale = np.where(Bmag_sample > 0.0, Bmag_sample, 1.0)
Ux_dir = np.divide(Bx_sample, direction_scale, out=np.zeros_like(Bx_sample), where=direction_scale > 0.0)
Uy_dir = np.divide(By_sample, direction_scale, out=np.zeros_like(By_sample), where=direction_scale > 0.0)

arrow_len = 0.5 * step * min(dx, dy)
Ux = Ux_dir * (0.5 * arrow_len)
Uy = Uy_dir * (0.5 * arrow_len)

segments_source = ColumnDataSource(
    data=dict(
        x0=(X_sample - Ux).ravel(),
        y0=(Y_sample - Uy).ravel(),
        x1=(X_sample + Ux).ravel(),
        y1=(Y_sample + Uy).ravel(),
        magnitude=Bmag_sample.ravel(),
    )
)

edges_source = ColumnDataSource(data=dict(x0=[], y0=[], x1=[], y1=[], edge_index=[]))
edge_labels_source = ColumnDataSource(data=dict(x=[], y=[], text=[]))

# Initial polygon vertices (modifiable by user)
initial_vertices: List[Tuple[float, float]] = [
    (0.04, 0.04),
    (0.08, 0.08),
    (0.08, 0.04),
    (0.04, 0.08),
]

points_source = ColumnDataSource(
    data=dict(
        x=[p[0] for p in initial_vertices],
        y=[p[1] for p in initial_vertices],
    )
)

poly_source = ColumnDataSource(data=dict(xs=[[p[0] for p in initial_vertices]], ys=[[p[1] for p in initial_vertices]]))

divergence_field = np.gradient(Bx, dx, axis=1) + np.gradient(By, dy, axis=0)
cell_area = dx * dy


def format_decimal(value: float, *, min_decimals: int = 4, max_decimals: int = 8) -> str:
    """Return a decimal string without scientific notation, keeping useful precision."""
    formatted = f"{value:.{max_decimals}f}"
    if "." not in formatted:
        return formatted
    integer_part, frac_part = formatted.split(".")
    frac_part = frac_part.rstrip("0")
    if len(frac_part) < min_decimals:
        frac_part = frac_part.ljust(min_decimals, "0")
    if not frac_part:
        frac_part = "0" * min_decimals
    return f"{integer_part}.{frac_part}"


fig = figure(
    title="B-field snapshot",
    tools="pan,wheel_zoom,reset",
    active_scroll="wheel_zoom",
    match_aspect=True,
    x_range=(0.0, Lx),
    y_range=(0.0, Ly),
)

fig.xaxis.axis_label = "x (m)"
fig.yaxis.axis_label = "y (m)"

fig.image(
    source=heatmap_source,
    image="image",
    x="x",
    y="y",
    dw="dw",
    dh="dh",
    color_mapper=color_mapper,
)

fig.add_glyph(
    segments_source,
    Segment(
        x0="x0",
        y0="y0",
        x1="x1",
        y1="y1",
        line_width=2,
        line_color={"field": "magnitude", "transform": color_mapper},
    ),
)

fig.segment(
    "x0",
    "y0",
    "x1",
    "y1",
    source=edges_source,
    line_color="red",
    line_width=3,
    line_alpha=0.7,
)

fig.add_layout(
    LabelSet(
        x="x",
        y="y",
        text="text",
        source=edge_labels_source,
        level="overlay",
        text_color="red",
        text_font_size="10pt",
        text_baseline="middle",
        text_align="center",
    )
)

patch_renderer = fig.patch(
    "xs",
    "ys",
    source=poly_source,
    fill_alpha=0.1,
    fill_color=None,
    line_color="white",
    line_width=2,
)

vertex_renderer = fig.circle("x", "y", source=points_source, size=8, color="white")

point_tool = PointDrawTool(renderers=[vertex_renderer], add=False, name="Edit Polygon")
fig.add_tools(point_tool)
info_div = Div(
    text=(
        "<b>Instructions</b><br>"
        "Use the toolbar's point icon (Edit Polygon) to drag vertices.<br>"
        "Pan/zoom with the standard tools when the point tool is inactive."
    ),
    width=350,
)

color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, location=(0, 0))
fig.add_layout(color_bar, "right")

force_div = Div(text="", width=350)


def polygon_mask(vertices: Sequence[Tuple[float, float]], xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Return a boolean mask of points inside the polygon."""
    if len(vertices) < 3:
        return np.zeros_like(xx, dtype=bool)
    px = np.asarray([v[0] for v in vertices], dtype=float)
    py = np.asarray([v[1] for v in vertices], dtype=float)
    inside = np.zeros_like(xx, dtype=bool)
    j_idx = len(px) - 1
    for i_idx in range(len(px)):
        xi = px[i_idx]
        yi = py[i_idx]
        xj = px[j_idx]
        yj = py[j_idx]
        intersect = ((yi > yy) != (yj > yy)) & (
            xx < (xj - xi) * (yy - yi) / (yj - yi + 1e-16) + xi
        )
        inside ^= intersect
        j_idx = i_idx
    return inside


def update_polygon_source(xs: List[float], ys: List[float]) -> None:
    """Update the polygon patch data to reflect the vertex list."""
    if not xs or len(xs) != len(ys):
        poly_source.data = dict(xs=[[]], ys=[[]])
        return
    xs_list = list(xs)
    ys_list = list(ys)
    xs_closed = xs_list + [xs_list[0]]
    ys_closed = ys_list + [ys_list[0]]
    poly_source.data = dict(xs=[xs_closed], ys=[ys_closed])


def compute_and_update() -> None:
    xs = list(points_source.data.get("x", []))
    ys = list(points_source.data.get("y", []))
    if len(xs) < 3 or len(xs) != len(ys):
        force_div.text = "<b>Polygon needs at least three valid vertices.</b>"
        return
    vertices = list(zip(xs, ys))
    torque_origin = (float(np.mean(xs)), float(np.mean(ys)))
    try:
        results = compute_polygon_mst(
            snapshot,
            vertices,
            torque_origin=torque_origin,
            samples_per_cell=1.5,
        )
    except ValueError as exc:
        force_div.text = f"<b>Maxwell stress error:</b> {exc}"
        return

    total_force = results["total_force"]
    total_torque = results["total_torque"]
    orientation = "CCW" if results["orientation"] > 0 else "CW"
    signed_area = results["area"]
    area = abs(signed_area)

    if results["edge_reports"]:
        starts = [edge["start"] for edge in results["edge_reports"]]
        ends = [edge["end"] for edge in results["edge_reports"]]
        edges_source.data = dict(
            x0=[start[0] for start in starts],
            y0=[start[1] for start in starts],
            x1=[end[0] for end in ends],
            y1=[end[1] for end in ends],
            edge_index=list(range(len(starts))),
        )
        edge_labels_source.data = dict(
            x=[0.5 * (s[0] + e[0]) for s, e in zip(starts, ends)],
            y=[0.5 * (s[1] + e[1]) for s, e in zip(starts, ends)],
            text=[str(edge["edge_index"]) for edge in results["edge_reports"]],
        )
    else:
        edges_source.data = dict(x0=[], y0=[], x1=[], y1=[], edge_index=[])
        edge_labels_source.data = dict(x=[], y=[], text=[])

    mask = polygon_mask(vertices, XX, YY)
    if mask.any():
        divergence_integral = float(np.sum(divergence_field[mask]) * cell_area)
        divergence_average = divergence_integral / area
    else:
        divergence_integral = float("nan")
        divergence_average = float("nan")

    rows = [
        "<b>Polygon properties</b>",
        f"Orientation: {orientation}",
        f"Area: {format_decimal(area)} m²",
        "<b>Net loads (per unit depth)</b>",
        f"Fx: {format_decimal(total_force[0])} N/m",
        f"Fy: {format_decimal(total_force[1])} N/m",
        f"τz: {format_decimal(total_torque)} N·m/m",
        "",
        "<b>Per-edge contributions</b>",
    ]
    for edge in results["edge_reports"]:
        idx = edge["edge_index"]
        rows.append(
            f"Edge {idx}: |F|={format_decimal(np.hypot(*edge['force']))} N/m, "
            f"Fn={format_decimal(edge['normal_force'])}, "
            f"Fs={format_decimal(edge['shear_force'])}, "
            f"τz={format_decimal(edge['torque'])}"
        )
    rows.extend(
        [
            "",
            "<b>Divergence check</b>",
            f"∬∇·B dA: {divergence_integral:.3e} T·m",
            f"Average ∇·B: {divergence_average:.3e} T/m",
        ]
    )

    force_div.text = "<br>".join(rows)
    update_polygon_source(xs, ys)


def on_points_change(attr: str, old, new) -> None:
    compute_and_update()


points_source.on_change("data", on_points_change)

compute_and_update()

layout = row(
    fig,
    column(info_div, force_div, sizing_mode="stretch_both"),
    sizing_mode="stretch_both",
)
curdoc().add_root(layout)
curdoc().title = "B-field Explorer"
