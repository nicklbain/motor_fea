# Motor FEA Sandbox

This repository contains a minimal 2-D magnetostatic workflow:

1. Generate a mesh plus material/source fields (`mesh_and_sources.py`).
2. Solve for the out-of-plane magnetic vector potential and derived flux density (`solve_B.py`).
3. Inspect any `.npz` payload (mesh or solution) with the lightweight Flask viewer in `viewer/`.

All scripts are pure Python and only depend on NumPy/SciPy (viewer additionally needs Flask/Plotly).

---

## Quick Start

```bash
# 1) (Recommended) create/activate a virtual environment
python3 -m venv .venv
. .venv/bin/activate

# 2) Install dependencies for solver + viewer (and case builder if you plan to use it)
pip install numpy scipy
pip install -r viewer/requirements.txt
pip install -r case_builder/requirements.txt  # optional UI for drawing geometries
# (Optional, improves unstructured mesh quality)
pip install triangle

# 3) Generate a case (writes cases/<case>/mesh.npz)
python3 mesh_and_sources.py --case demo_case

# 4) Solve for A and B (writes Az_field.npz + B_field.npz next to the mesh)
python3 solve_B.py --case demo_case

# 5) Launch the viewer (optional, in another shell)
python3 viewer/app.py --debug
# Visit http://127.0.0.1:5000 and pick cases/demo_case/mesh.npz (or Az/B files) from the dropdown.

# 6) (New) Launch the combined workbench UI
python3 workbench/app.py --debug
# Visit http://127.0.0.1:5173 to build, solve, and visualize in one place.
```

---

## Files and Case Layout

Each case lives under `cases/<case_name>/`:

| File | Description |
| --- | --- |
| `mesh.npz` | Nodes, triangles, region IDs, μ_r, magnetization `(Mx, My)`, source current density `Jz`, and metadata produced by `mesh_and_sources.py`. |
| `Az_field.npz` | Nodal vector potential solution `Az` computed by `solve_B.py` plus the original mesh data for reference. |
| `B_field.npz` | Triangle-centered flux density: `Bx = ∂Az/∂y`, `By = -∂Az/∂x`, and `Bmag = √(Bx²+By²)` (plus indicator/α breakdowns when available). |

`solve_B.py` automatically creates the case subfolder if it is missing. Running the solver without arguments tries, in order:

1. `cases/mag2d_case/mesh.npz`
2. `cases/mag2d_case.npz` (legacy flat layout)
3. The inferred path from whatever you pass (case folder, relative file, or absolute file).

---

## Useful CLI Flags

### `mesh_and_sources.py`

| Flag | Purpose |
| --- | --- |
| `--case <name>` | Case subfolder under `--cases-dir` (default: `mag2d_case`). |
| `--cases-dir <path>` | Where case folders live (default: `cases`). |
| `--case-config <file>` | Path to a JSON case definition. Defaults to `cases/<case>/case_definition.json` when present. |
| `--Nx`, `--Ny` | Legacy uniform grid resolution (default: 100×100). Adaptive case definitions override this via `grid.mesh`. |
| `--Lx`, `--Ly` | Domain size in meters. |
| `--magnet-My` | Permanent magnet strength `My` in A/m (default: \(8\times10^5\)). |
| `--mu-r-magnet`, `--mu-r-steel` | Override material permeabilities. |
| `--wire-current`, `--wire-radius` | Coil current per conductor (A) and disk radius (m). Defaults to 5000 A and 0.02 m to make the coil’s B-field visible. |
| `--no-magnet`, `--no-steel`, `--no-wire` | Drop individual regions/sources for debugging. |
| `--adapt-from <B_field.npz>` | Use an existing solution’s \( \|\nabla B\| \) field to shrink/expand grid intervals instead of the material-based focus heuristic. |
| `--adapt-refine-q`, `--adapt-coarsen-q` | Quantile cutoffs for the refine / coarsen buckets (defaults: 0.75 / 0.25). |
| `--adapt-refine-factor`, `--adapt-coarsen-factor`, `--adapt-smooth-passes` | Fine-tune how aggressively intervals are halved/doubled and how much 1-D smoothing is applied to the indicator. Coarsening is disabled unless you also pass `--adapt-allow-coarsen`. |
| `--adapt-material-pad`, `--adapt-material-max-scale` | Clamp spacing directly over (and optionally near) non-air materials so refined regions never grow beyond their previous cell size. |
| `--adapt-direction-weight`, `--adapt-magnitude-weight` | While extracting focus boxes from a solved B-field (point-cloud meshes), weight how strongly direction changes vs magnitude changes contribute to the indicator (defaults: 1 / 1). |
| `--adapt-min-component-area`, `--adapt-max-components` | Ignore tiny indicator blobs below a given area and/or cap how many focus boxes are spawned when sampling ∇B. |

### Meshing modes

- Structured (`grid.mesh.type: "graded"`): existing tensor grid with optional graded spacing along each axis. Works with `--adapt-from` and the focus-span heuristic already in the repo.
- Point-cloud Delaunay (`grid.mesh.type: "point_cloud"`): builds a coarse background point lattice (set by `coarse` / `coarse_dx`) and sprinkles extra points only inside focus regions (materials listed in `focus_materials`, B-field-driven focus boxes, or manual `focus_boxes`). A Delaunay triangulation stitches those points into an unstructured mesh, so refinement stays local instead of propagating along entire rows/columns. Works with the same case-definition objects used for materials.

Example case-definition snippet:

```json
"grid": {
  "Lx": 0.6,
  "Ly": 0.6,
  "mesh": {
    "type": "point_cloud",
    "coarse": 0.005,
    "fine": 0.0005,
    "focus_pad": 0.004,
    "focus_materials": ["steel", "magnet", "wire"],
    "focus_boxes": [
      { "x": [0.20, 0.32], "y": [0.22, 0.30] }
    ]
  }
}
```

`coarse`/`fine` have the same meaning as the structured builder (approximate spacing in meters). Omit `--adapt-from` when using `point_cloud`—field-driven adaptivity currently assumes a structured grid.

When you *do* pass `--adapt-from` while in `point_cloud` mode, the solver evaluates the stored `B_field.npz`, measures both the magnitude gradient and how rapidly the vector direction spins, clusters the hotspots into connected components, and spawns additional focus boxes for each component (after clipping to `grid.mesh.focus_pad` and your `field_focus` settings). That keeps the refined patch compact instead of scaling entire rows and columns.

### `solve_B.py`

| Flag | Purpose |
| --- | --- |
| positional / `--case <path>` | Case folder or .npz path to solve (defaults as noted above). |
| `--cases-dir <path>` | Base directory for case folders (default: `cases`). |
| `--pin-node <idx>` | Node index pinned to zero to fix the Neumann gauge (default: 0). |
| `--bh-max-iters`, `--bh-residual-tol`, `--bh-min-drop` | Control BH Newton iteration cap, residual tolerance, and early-stop threshold. |
| `--bh-allow-labels` | Comma-separated substrings to keep BH curves only on matching shape labels (others remain linear). |
| `--freeze-mu-from <B_field.npz>` | Reuse `mu_r_effective` from a prior solve (skips BH iteration and solves linearly). |

---

## Viewer Tips

* The viewer lists every `.npz` under `cases/`, so you can browse `mesh.npz`, `Az_field.npz`, or `B_field.npz`.
* To refresh after regenerating a case, just reload the browser—the API enumerates files on each request.
* The viewer is a simple Flask app; run `python3 viewer/app.py --help` for options.

---

## Adding Custom Cases

Feel free to replace `mesh_and_sources.py` with your own mesher as long as you emit the same fields in `mesh.npz`.
`solve_B.py` only requires that:

* `tris` reference `nodes` using CCW orientation.
* Material/source arrays (`region_id`, `mu_r`, `Mx`, `My`, `Jz`) are per triangle.

Everything else (edge magnet loads, Neumann solve, B-field computation) happens automatically.

---

## Case Builder (experimental)

The `case_builder/` Flask app lets you sketch rectangles, circles, rings, or regular polygons, tag them as magnets/steel/wires, or mark a path as a “contour” (kept for post-processing but not meshed as material), and save the geometry + material parameters to `cases/<case>/case_definition.json`. When that file exists (or when you pass `--case-config`), `mesh_and_sources.py` consumes it instead of the hard-coded demo geometry. Contour-only shapes are preserved in the solver metadata under `meta.contours` so downstream tooling can integrate along their sides later (torque, force, etc.).

When a contour is present and you run `solve_B.py` (or click “Run case” in the builder), the solver now samples the B-field at the midpoint of each contour segment, evaluates the Maxwell stress traction using that outward normal, and stores both per-segment B vectors and per-segment force (N/m) plus contour-level net force. The viewer exposes two toggles to overlay contour B arrows and force arrows. Contours no longer participate in material/field overlap calculations (so bumping the side count does not slow meshing), and segment sampling uses a KD-tree over triangle centroids to keep evaluation fast even with hundreds of sides.

You can also import DXF outlines: use the **Import DXF** panel in the builder to upload a file, optionally group shapes by layer, and auto-fit the geometry into your current domain. Closed polylines become explicit polygons, circles stay analytic, and arcs get faceted into segments (a warning is shown if anything is skipped).

**Usage**

```bash
pip install -r case_builder/requirements.txt
python3 case_builder/app.py --debug
# open http://127.0.0.1:5050, draw shapes, then "Save to folder"
python3 mesh_and_sources.py --case <your_case>
```

Each saved definition includes the physical domain and the mesh controls so the solver can reproduce the exact grid:

```json
{
  "name": "demo",
  "grid": {
    "Lx": 0.4,
    "Ly": 0.3,
    "mesh": {
      "type": "graded",
      "fine": 0.003,
      "coarse": 0.015,
      "focus_pad": 0.02,
      "focus_falloff": 0.01,
      "focus_materials": ["magnet", "steel", "wire"]
    }
  },
  "objects": [
    {
      "id": "magnet-1",
      "label": "Left magnet",
      "material": "magnet",
      "shape": {"type": "rect", "center": [0.15, 0.15], "width": 0.08, "height": 0.06},
      "params": {"mu_r": 1.05, "Mx": 0.0, "My": 800000.0}
    },
    {
      "id": "wire-upper",
      "material": "wire",
      "shape": {"type": "circle", "center": {"x": 0.25, "y": 0.22}, "radius": 0.02},
      "params": {"current": -5000.0}
    }
  ]
}
```

Fields that overlap get applied in list order, so later shapes win. Rectangles benefit from exact area clipping, while circles use barycentric sampling (good for grids ≥ ~50×50). The point-and-click UI is intentionally simple now but the JSON schema is human-readable, versionable, and forward-compatible with future features (DXF import, arcs, etc.).

Permanent magnet `params.Mx/My` are specified in the shape's local (unrotated) axes; if you set `shape.angle`, the magnetization vector is rotated by the same amount so the magnet's polarization stays aligned with the body.

### Adaptive mesh controls

The **Adaptive mesh** mode in the Case Builder now drives the new point-cloud mesher. You still choose coarse / fine pitches, a padding distance, and which materials count as “focus,” but refinement stays local:

### B–H curves and saturation

Steel objects can now carry a nonlinear B–H curve. In the Case Builder, either enter `µ_r` plus a saturation `B_sat` to auto-generate a simple knee, or upload a CSV with two columns (comma/space separated) `B_T, H_A_per_m` (header optional, comments starting with `#` are ignored). The mesh stores per-triangle `bh_id` plus the curve in metadata; the solver runs a Picard loop to update `µ_r` from the curve and writes the effective `mu_r_effective` alongside the usual outputs.

Each solve appends a lightweight diagnostics line to `cases/<case>/diagnostics.log` (NDJSON): timestamp, mesh path, node/tri counts, BH settings (method, iterations, convergence, flags), µ ranges, and timing. Use it to trace why a run was slow or which solver path was taken. The file grows over time; delete it if you want a fresh log.

1. A coarse lattice covers the entire domain using the requested `coarse` spacing (optionally anisotropic via `mesh.y`).
2. Each focus region (material bounds ± `focus_pad` or manually supplied `focus_boxes`) receives its own fine lattice sampled at `fine` spacing.
3. Coarse-lattice points that fall inside each focus region are dropped before the Delaunay pass, so the fine patch cleanly replaces the surrounding rows/columns instead of dragging refinement strips across the domain.

There’s also an **Experimental (equilateral-heavy)** mode in the Mesh mode dropdown. It builds a deterministic hex-like fill from a per-cell size grid, caps spacing near focus materials, and smoothly grades back to the coarse pitch to keep aspect ratios tame without invoking the legacy Triangle quality mesher.

Need to bias only certain shapes? Uncheck the materials you don’t care about or add custom `focus_boxes` in the JSON. Switch the UI to “Uniform (legacy Nx × Ny)” if you truly need the old structured grid—the JSON will store `grid.mesh.type = "uniform"` plus the classic counts. Prefer fatter triangles? Raise the “Min triangle angle (deg)” control (or `grid.mesh.quality_min_angle`) so Triangle inserts extra Steiner points until that angle target is met; set it to 0 to fall back to the plain Delaunay mesh.

Below those knobs you’ll now find a **B-field driven refinement** group. It lets you keep using the material-based focus boxes, *and* optionally add new boxes derived from the last solved `B_field.npz`. The UI writes those values to `grid.mesh.field_focus`, which the CLI interprets whenever you pass `--adapt-from`. You can tweak:

- `Refine quantile`: what fraction of the indicator (combined magnitude / direction change) should be marked as “interesting”.
- `Direction weight` vs `Magnitude weight`: emphasize how fast the vector rotates vs how much its magnitude grows/shrinks.
- `Min focus box area` and `Max focus boxes`: suppress noise speckles and keep the iteration manageable.

Each `.npz` still records `meta["mesh_generation"]` so you can confirm whether the mesh came from the point-cloud, whether it used material spans only, or whether ∇B contributed additional focus boxes.

### Field-driven adaptive refinement

If you already solved a case, you can let the field solution steer the next mesh by pointing the generator at the saved `B_field.npz`:

```bash
python3 mesh_and_sources.py --case demo_case_field \
    --adapt-from cases/demo_case/B_field.npz \
    --adapt-refine-q 0.8 --adapt-coarsen-q 0.2
```

The workflow is:

1. Generate + solve once to obtain `cases/<case>/B_field.npz`.
2. Re-run `mesh_and_sources.py` for a *new* case name while passing `--adapt-from` (optionally tweak the quantiles/factors).
3. Solve the new mesh, inspect the results, and iterate if needed.

Behind the scenes the helper treats the existing mesh as structured, averages the two triangles in every cell, computes per-cell
\( \|\nabla B\| \) with directional finite differences, smooths the indicator, and buckets each column/row into `refine`, `keep`, or `coarsen`.
Intervals assigned to `refine` get multiplied by `--adapt-refine-factor` (default 0.5), `coarsen` intervals get multiplied by `--adapt-coarsen-factor`
(default 2.0), and the whole axis is re-scaled to fit the requested `Lx/Ly`. The decision counts and spacing stats end up in
`meta["mesh_generation"]` with `type = "field_adapt"` so you can track what happened.

While doing this, any interval that sits on (or within `--adapt-material-pad`) a non-air material is prevented from growing beyond its previous size (scaled by `--adapt-material-max-scale`, default 1.0). This ensures magnets, steel, and coils always retain at least their prior resolution even if the gradient indicator goes quiet. By default the adaptive pass only refines (never coarsens); re-enable coarsening with `--adapt-allow-coarsen` if you need both directions.

Supplying `--adapt-from` overrides any `grid.mesh` directives inside a case definition. Clear the flag to revert to the material-focused adaptive mesher.

When you use the Case Builder UI to run a case, a follow-up **Adaptive mesh re-run** button becomes available (as long as you have not changed the geometry/materials since the last solve). That button now samples the previously saved `B_field.npz`, builds indicator-driven focus boxes using your `B-field driven refinement` settings, regenerates the point-cloud mesh, and re-solves—perfect for iterating on the tuning knobs without leaving the browser. If you want the legacy structured-axis remesher instead, run `mesh_and_sources.py` manually with `--adapt-from` while `grid.mesh.type` is `"graded"`.


## To do:

- Verify ok if materials are touching each other
- Add ability to draw rectangle, circle, ring, or regular polygon (including “contour-only” paths) and use as loop for integrating Maxwell Stress tensor -> display Fx, Fy, torque per Z
