# AuraFlow architecture

Differentiable aeroacoustics in JAX. Three simulation fidelities sharing one core:

```
                    ┌────────────────────────────────────────────────┐
                    │                auraflow.core                   │
                    │  frames, blade geometry, airfoil polars,       │
                    │  medium (ρ0, c0, ν), rotor/vehicle kinematics  │
                    └────────────────────────────────────────────────┘
                         ▲                 ▲                  ▲
        ┌────────────────┴──┐   ┌──────────┴─────────┐  ┌─────┴──────────────┐
        │  auraflow.bemt    │   │   auraflow.cona    │  │   auraflow.cfd     │
        │  BEMT loading +   │   │  end-to-end multi- │  │  compressible CFD  │
        │  compact F1A      │   │  rotor pipeline:   │  │  near field +      │
        │  tonal noise      │   │  6DOF/trim, HBEM,  │  │  permeable-surface │
        │  (fwh_rotor_sim   │   │  Beddoes wake,     │  │  sampling          │
        │   approach)       │   │  unsteady corr.,   │  │                    │
        └───────┬───────────┘   │  tonal + BPM       │  └─────┬──────────────┘
                │               │  broadband,        │        │
                │               │  Griffin–Lim       │        │
                │               └──────────┬─────────┘        │
                ▼                          ▼                  ▼
        ┌────────────────────────────────────────────────────────────┐
        │                       auraflow.fwh                         │
        │  retarded-time solvers, Farassat 1A (compact & surface),   │
        │  permeable-surface 1A (static + Garrick convective),       │
        │  Formulation 1C (uniformly moving medium)                  │
        └────────────────────────────┬───────────────────────────────┘
                                     ▼
        ┌────────────────────────────────────────────────────────────┐
        │                     auraflow.signal                        │
        │  Welch PSD, SPL/OASPL, A-weighting, 1/3-octave, STFT,      │
        │  Griffin–Lim synthesis, harmonic extraction                │
        └────────────────────────────────────────────────────────────┘

  auraflow.body      — general 3D bodies: triangle meshes, rigid+deforming motion,
                       model import (STL/OBJ/PLY/GLB), mesh→FW-H sources, mesh→SDF
  auraflow.datasets  — JASA-style dataset generation (any backend)
  auraflow.viz       — in-browser 3D live visualization (websocket streaming)
  auraflow.run       — omnirun helpers / experiment entry points
```

## Generalization principle (v2)

The acoustic core is **not rotor-specific**. `fwh.f1a_pressure` consumes per-source
histories `(y, v, a, L, Q_n)` on a uniform source-time grid — *anything* that produces
panel histories radiates: rotor blades, a full airframe, a loudspeaker membrane, any
imported 3D model moving through (or vibrating in) a compressible medium. The rotor
backends (bemt/cona) are one adapter family; `auraflow.body` is the general one:

```
TriMesh (imported or parametric)
   × Motion (rigid pose(t) ∘ optional surface deformation u_n(face, t))
   × surface data (p(face,t) from CFD/prescribed; or nothing for thickness-only)
   → panel source histories → auraflow.fwh → signals
```

A **speaker** is the degenerate-motion case: static rigid enclosure, membrane faces
carrying a prescribed normal velocity signal u_n(t) (e.g. decoded from an audio file)
→ thickness/monopole radiation; validated against the analytic baffled circular piston.
A **flying body** is the opposite case: rigid motion, loading from surface pressure.
Both compose (a vibrating surface on a moving body).

## Library conventions

- **Stack**: JAX + equinox (modules as pytrees). numpy/scipy allowed only at setup/IO
  boundaries, never inside jitted/differentiated code paths.
- **Precision**: all acoustics code must be float64-safe; tests enable
  `jax.config.update("jax_enable_x64", True)` via `tests/conftest.py`. Retarded-time and
  FW-H math is precision-sensitive: never hardcode float32.
- **Units**: SI throughout (m, s, kg, Pa, rad). Angles in radians. RPM appears only at
  user-facing constructors (converted immediately to Ω [rad/s]).
- **Frames** (right-handed):
  - *World frame*: z up. Observers, vehicle trajectory, microphones live here.
  - *Rotor frame*: origin at hub, z along thrust axis. Blade azimuth ψ measured from +x
    toward +y (counterclockwise seen from +z); rotation sense = sign of Ω.
  - *Blade section frame*: x spanwise outward, y chordwise toward leading edge in the
    direction of rotation, z thrust-normal (matches fwh_rotor_sim).
- **Array shapes** (leading batch axes, trailing xyz):
  - sources `[..., S, 3]`, observers `[O, 3]`, times `[T]`,
    pressure signals `[O, T]`, per-source quantities `[..., S, T]` before summation.
  - `vmap` over observers; `lax.scan`/vectorized time where possible.
- **Differentiability rules**: no data-dependent Python control flow in differentiated
  paths; `jnp.where` for branches (e.g. Beddoes wake pieces); soft clamps instead of hard
  clips where gradients matter (document any gradient dead zones); table lookups via
  differentiable interpolation (linear or cubic on regular grids).
- **Static vs traced**: geometry discretization counts, panel counts, blade counts are
  static (Python ints); everything physical (chord/twist params, Ω histories, positions,
  polar coefficients) is traced and differentiable.

## Modules

### auraflow.core
- `medium.py` — `Medium(rho0, c0, nu, p0)` eqx module; standard atmosphere constructor.
- `frames.py` — rotation matrices, `Rz(psi)`, Euler/body transforms, azimuth integration
  `psi(t) = ∫Ω dt` (differentiable cumulative trapezoid + interp), kinematics helpers.
- `blade.py` — `BladeGeometry`: radial stations (trapezoid-consistent dr), chord/twist as
  arrays or parametric callables, pitch-axis location, optional airfoil section profiles
  (for thickness noise & CFD levelsets); `Rotor`: n_blades, hub position/orientation,
  rotation sense; `Vehicle`: multiple rotors + mass properties.
- `airfoil.py` — polar protocols: `ThinAirfoilPolar` (Cl=2π(α−α0)/sqrt-beta corrections,
  Cd=cd0+k·Cl²), `TablePolar` (differentiable (α, M, Re) interpolation), stall softening.

### auraflow.body
- `mesh.py` — `TriMesh` (eqx): `vertices [V,3]` (traced, differentiable), `faces [F,3]`
  (static int); derived per-face centroids/normals/areas; outward-winding invariant;
  area-weighted single-point panel quadrature (document the compactness assumption:
  panels ≪ wavelength; refine the mesh, not the quadrature). Parametric primitives
  (`sphere`, `box`, `disk`, `cylinder`, `flat_plate`) for tests/validation gates.
- `io.py` — `load_mesh(path)` via trimesh (optional `mesh` extra; lazy import): STL, OBJ,
  PLY, GLB/GLTF, OFF. At import: merge duplicate vertices, repair winding, verify/report
  watertightness, convert to float64 `TriMesh`. numpy allowed here (IO boundary).
- `motion.py` — kinematics for anything:
  - `RigidMotion` protocol: `pose(t) -> (R [3,3], x [3])`; velocities/accelerations via
    `jax.jvp` (no finite differences). Implementations: `StaticPose`, `ConstantVelocity`,
    `SpinMotion` (axis+Ω(t), subsumes rotor spin), `WaypointMotion` (smooth spline),
    `ComposedMotion` (child frame in parent frame — blade in rotor in vehicle).
  - `SurfaceVibration`: prescribed normal displacement/velocity per face,
    `u_n(face_ids, t)` (traced) — the speaker membrane; superimposed on rigid motion.
  - `panel_histories(mesh, motion, tau) -> (y, v, a, n, area) [F,T,…]` — the single
    entry point every acoustic adapter uses.
- `sources.py` — mesh → FW-H adapters:
  - `impermeable_sources(mesh, motion, tau, p_surface=None, vibration=None)` →
    `(y, v, a, L, Q_n)`: thickness `Q_n = ρ0 (v·n + u_n)` per panel; loading
    `L = p·n·area` when surface pressure is given (from CFD or prescribed).
  - `permeable_surface(mesh)` → generalizes `cfd.sphere.PermeableSphere` to any closed
    mesh (points/normals/areas for CFD sampling + static-surface F1A fast path).
- `sdf.py` — mesh → signed distance: `sdf_grid(mesh, box, cells)` (trimesh proximity at
  setup, numpy OK) + differentiable trilinear `sdf_eval` — feeds JAX-Fluids level-set
  solids (resolved bodies in CFD) and viz.
- `speaker.py` — the speaker model on top of the above: `Speaker(enclosure: TriMesh,
  membrane_faces, baffled: bool)`; `radiate(audio_signal u_n(t) or cone velocity, fs,
  listeners) -> pressure [O,T]`; rigid-enclosure scattering neglected (documented;
  correct in the baffled/free-field limits used by the gates).

### auraflow.fwh
- `retarded.py` — vectorized Newton retarded-time solve g(τ)=τ+r/c0−t (fixed iters,
  implicit-diff friendly), plus source-time-marching "advanced time" projection.
- `farassat1a.py` — general moving-source F1A: compact point sources (loading; thickness
  via monopole pairs or ∂t of displaced volume) and mesh-surface variants.
- `permeable.py` — permeable-surface F1A for a *static* surface (closed-form retarded
  times), with optional uniform mean flow via Garrick triangle (per OpenCFD-FWH digest).
- `formulation1c.py` — convective FW-H (Najafi-Yazdi 1C) for wind cases in CONA.
- Validation gates: analytic monopole/dipole (static and in uniform flow), rotating
  point force vs fwh_rotor_sim results.

### auraflow.bemt
- `annulus.py` — proper per-annulus BEMT: local momentum/blade-element balance with
  Prandtl tip/root loss, swirl; fixed-point solve with `lax.while_loop`/unrolled iters
  (implicit function theorem for gradients via `jax.lax.custom_root` or equinox).
- `inflow.py` — uniform momentum inflow + Pitt–Peters linear inflow λ(r,ψ)
  (kx=(15π/32)tan(χ/2), ky=0).
- `wake.py` — parameterized Beddoes prescribed wake (piecewise via jnp.where),
  Lamb–Oseen vortex segments + Biot–Savart induced velocities.
- `unsteady.py` — Wagner/Jones deficiency-function recursion (lax.scan), apparent mass.

### auraflow.cona
- `flightsim.py` — 6-DOF quadrotor dynamics + backstepping/geometric trajectory-tracking
  controller, Dryden gusts; produces per-rotor Ω(t) and vehicle states.
- `reconstruct.py` — time reconstruction onto ~1°-azimuth grid, linear interp,
  unsteady load corrections.
- `tonal.py` — loading (chordwise-compact dipoles at pitch axis) + thickness noise
  via Formulation 1C / F1A.
- `broadband.py` — BPM self-noise model per blade section → 1/3-octave spectrogram.
- `auralize.py` — Griffin–Lim broadband synthesis + tonal summation → 44.1 kHz signal.
- `pipeline.py` — end-to-end `simulate(vehicle, mission, observers) -> signals`.

### auraflow.cfd
- Foundation library per ecosystem survey (JAXFLUIDS candidate — pending evaluation).
- `domain.py` — near-field box/sphere setup, sponge zones, non-reflecting BCs.
- `rotor_source.py` — rotor representation: actuator line forcing from BEMT loads
  (first target), immersed-boundary resolved blades (stretch goal).
- `sampling.py` — interpolation of (ρ, u, p) onto permeable-sphere panels each sample step.
- `coupling.py` — drive `auraflow.fwh.permeable` from sampled surface data.

### auraflow.signal
- `spectra.py` — rfft helpers, Welch PSD (Hann, 50% overlap), SPL/OASPL (re 20 µPa),
  band-integrated 1/3-octave levels, A-weighting, BPF harmonic extraction.
- `synthesis.py` — STFT/iSTFT, Griffin–Lim from magnitude spectrograms, 1/3-octave-band
  → STFT-magnitude energy spreading.

### auraflow.datasets
- `jasa2026.py` — NASA 1-Pax quadrotor flyover cases: V∞ ∈ {1..10} m/s, 30 m altitude,
  256 ground mics (10 m grid), 1 s @ 44.1 kHz; backend-selectable (cona | cfd).

### auraflow.viz
- Live: simulation loop pushes downsampled field slices/isosurfaces over websocket to a
  self-contained three.js page. Offline: replay saved snapshots. (Design TBD in task #10.)

## Testing strategy

- Analytic gates: monopole/dipole (±flow) for every FW-H formulation; momentum-theory
  limits for BEMT (hover induced velocity, thrust); Wagner step response for unsteady aero.
- Body/mesh gates (v2): pulsating sphere mesh (breathing u_n) vs exact monopole solution
  p(r) = (ρ0 c0 U0 k a²/r)·e^{ik(r−a)}/√(1+(ka)²); rigid oscillating sphere vs analytic
  dipole; baffled circular piston vs Rayleigh on-axis |p(z)| and far-field directivity
  2J1(ka·sinθ)/(ka·sinθ); imported-vs-parametric mesh equivalence (same sphere STL vs
  primitive → same radiated field); translating mesh Doppler; thin-plate mesh loading in
  the compact limit vs a compact F1A dipole; SDF sign/values on primitives.
- Cross-backend gates: BEMT+F1A vs CONA tonal path on the same rotor; CONA vs published
  validation numbers (DJI 9450 hover BPF directivity); CFD+FW-H vs CONA on JASA cases.
- Gradient tests: finite-difference checks through each backend's scalar outputs.
- All tests CPU-runnable at reduced resolution; GPU runs via omnirun for full cases.
