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

## Install

```sh
uv sync                       # base (import auraflow always works)
uv sync --extra cfd           # + JAX-Fluids CFD backend
uv sync --extra viz-live      # + live in-browser visualization
```

Extras: `viz` (matplotlib), `viz-live` (websockets), `cfd` (jaxfluids), `data`
(dload-ml), `gpu` (CUDA JAX — never on the CPU dev box).

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
