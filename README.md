# AuraFlow

Differentiable aeroacoustics simulations in JAX: fast approximate rotor-noise
models (BEMT, CONA) and full CFD + permeable-surface FW-H, sharing one geometry
/ frames / medium core. See [`docs/architecture.md`](docs/architecture.md) for
the design and module map.

Three simulation fidelities, fastest to most precise:

- **`auraflow.bemt`** — blade-element momentum theory loading + compact-chordwise
  Farassat 1A tonal noise.
- **`auraflow.cona`** — CONA-style end-to-end pipeline: 6-DOF flight, BEMT/analytic
  loading, tonal (F1A/1C) + BPM broadband noise, Griffin–Lim auralization.
- **`auraflow.cfd`** — compressible near-field CFD (JAX-Fluids) sampled on a static
  permeable sphere and propagated by permeable-surface FW-H.

Shared: `auraflow.core` (geometry, frames, airfoils), `auraflow.fwh` (FW-H
formulations), `auraflow.signal` (spectra, SPL). `auraflow.datasets` reproduces
the JASA flyover data-generation recipe.

`auraflow.body` makes *any* triangle mesh radiate — imported 3D models,
airframes, or a vibrating loudspeaker membrane — through the same FW-H core
(`mesh_pressure`). Build or import a speaker, play a waveform, and read the
pressure at listeners: see **[`docs/speaker.md`](docs/speaker.md)**.

## Install

```sh
uv sync                       # base (import auraflow always works)
uv sync --extra cfd           # + JAX-Fluids CFD backend
uv sync --extra viz-live      # + live in-browser visualization
```

Extras: `viz` (matplotlib), `viz-live` (websockets), `cfd` (jaxfluids), `mesh`
(trimesh, 3-D model import), `data` (dload-ml), `gpu` (CUDA JAX — never on the
CPU dev box).

## General bodies & speakers

`auraflow.body` turns *any* closed 3-D mesh (imported or parametric) plus a
motion into an FW-H source, a loudspeaker, or a CFD level-set solid.

**Import → radiate** an imported/parametric body flying past microphones
(thickness noise, Doppler; differentiable through vertices and motion):

```python
from auraflow.body import load_mesh, WaypointMotion, mesh_pressure
from auraflow.core.medium import Medium
mesh = load_mesh("body.stl")                      # or TriMesh.sphere(0.3)
motion = WaypointMotion([0, 1], [[-40, 0, 8], [40, 0, 8]])
p, t = mesh_pressure(mesh, motion, tau, mics, Medium())   # [O, T_obs] Pa
```

**Speaker**: drive a membrane with a waveform and record it at listeners:

```python
from auraflow.body import Speaker
spk = Speaker.circular_piston(radius=0.05, baffled=True)  # or Speaker.from_mesh(cabinet, ...)
p, t = spk.play(audio, fs, listeners, Medium(), gain=0.05)
```

**CFD level-set body**: immerse the mesh as a FLUID-SOLID solid in JAX-Fluids
(the SDF is the level-set field; static or prescribed-moving solids):

```python
from auraflow.cfd import levelset_body_case, permeable_mesh_surface, run_acoustic_case
case = levelset_body_case(mesh, box_lo=(-.5,-.5,-.5), box_hi=(.5,.5,.5), cells=(48,48,48))
surf = permeable_mesh_surface(TriMesh.sphere(0.4, subdivisions=3))   # sample on any closed mesh
hist = run_acoustic_case(case, surf, n_steps=200)                    # GPU for real resolutions
```

Runnable demos (tiny, CPU-safe): `scripts/speaker_demo.py` and
`scripts/body_flyby_demo.py` (both take `--viz` for the live mesh stream).

## Live 3-D visualization

Stream a running simulation to a self-contained three.js page in the browser,
live while it computes — see **[`docs/viz.md`](docs/viz.md)**.

```sh
uv run --extra cfd --extra viz-live python scripts/viz_demo_cfd.py      # live CFD pulse
uv run --extra viz-live python scripts/viz_demo_flyover.py              # CONA flyover replay
```

Then open the printed `http://localhost:8000`.

## Development

See [`CLAUDE.md`](CLAUDE.md) for the dev-box RAM rules (run pytest one file at a
time under a memory cap; heavy compute goes to GPU via omnirun). Tooling:
`uv run pytest`, `uv run ruff check src tests scripts`,
`uv run basedpyright src/auraflow/<module>`.
