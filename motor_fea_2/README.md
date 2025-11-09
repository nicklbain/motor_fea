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

# 3) Generate a case (writes cases/<case>/mesh.npz)
python3 mesh_and_sources.py --case demo_case

# 4) Solve for A and B (writes Az_field.npz + B_field.npz next to the mesh)
python3 solve_B.py --case demo_case

# 5) Launch the viewer (optional, in another shell)
python3 viewer/app.py --debug
# Visit http://127.0.0.1:5000 and pick cases/demo_case/mesh.npz (or Az/B files) from the dropdown.
```

---

## Files and Case Layout

Each case lives under `cases/<case_name>/`:

| File | Description |
| --- | --- |
| `mesh.npz` | Nodes, triangles, region IDs, μ_r, magnetization `(Mx, My)`, source current density `Jz`, and metadata produced by `mesh_and_sources.py`. |
| `Az_field.npz` | Nodal vector potential solution `Az` computed by `solve_B.py` plus the original mesh data for reference. |
| `B_field.npz` | Triangle-centered flux density: `Bx = ∂Az/∂y`, `By = -∂Az/∂x`, and `Bmag = √(Bx²+By²)`. |

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
| `--Nx`, `--Ny` | Structured grid resolution (default: 100×100). |
| `--Lx`, `--Ly` | Domain size in meters. |
| `--magnet-My` | Permanent magnet strength `My` in A/m (default: \(8\times10^5\)). |
| `--mu-r-magnet`, `--mu-r-steel` | Override material permeabilities. |
| `--wire-current`, `--wire-radius` | Coil current per conductor (A) and disk radius (m). Defaults to 5000 A and 0.02 m to make the coil’s B-field visible. |
| `--no-magnet`, `--no-steel`, `--no-wire` | Drop individual regions/sources for debugging. |

### `solve_B.py`

| Flag | Purpose |
| --- | --- |
| positional / `--case <path>` | Case folder or .npz path to solve (defaults as noted above). |
| `--cases-dir <path>` | Base directory for case folders (default: `cases`). |
| `--pin-node <idx>` | Node index pinned to zero to fix the Neumann gauge (default: 0). |

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

The `case_builder/` Flask app lets you sketch rectangles and circles, tag them as magnets/steel/wires, and save the geometry + material parameters to `cases/<case>/case_definition.json`. When that file exists (or when you pass `--case-config`), `mesh_and_sources.py` consumes it instead of the hard-coded demo geometry.

**Usage**

```bash
pip install -r case_builder/requirements.txt
python3 case_builder/app.py --debug
# open http://127.0.0.1:5050, draw shapes, then "Save to folder"
python3 mesh_and_sources.py --case <your_case>
```

Each saved definition looks like:

```json
{
  "name": "demo",
  "grid": {"Nx": 120, "Ny": 120, "Lx": 0.4, "Ly": 0.3},
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
