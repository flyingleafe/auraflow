# Reference #1 digest: fwh_rotor_sim (harmonic-noise-suppression repo)

Location: /home/flyingleafe/projects/harmonic-noise-suppression/src/fwh_rotor_sim
Stack: pure PyTorch (AuraFlow ports to JAX). Design doc worth reading:
docs/fwh_rotor_acoustic_simulator_plan.md (literature survey + phased plan; Phase 1 done).

## What it implements

- `geometry.py`: `Blade` (parametric chord/twist callables, radial strips w/ trapezoid-consistent
  dr, quarter-chord compact sources), `Rotor` (multi-blade; shaft_tilt stored but UNUSED).
  Frames: x = span outward, y = tangential (rotation dir), z = thrust; azimuth via R_z(ψ).
- `bemt.py`: ThinAirfoilPolar (Cl = 2π(α−α0), Cd = cd0 + k Cl²), NOT per-annulus BEMT —
  single uniform induced inflow from actuator-disk momentum theory, fixed-point solve_hover()
  (20 iters). α clamped ±20° (kills gradients outside).
- `fwh.py`: Farassat 1A **loading (dipole) terms only**, compact point-force sources:
  4π p' = Σ [ Ḟr/(c0 r(1−Mr)²) + (Fr−FM)/(r²(1−Mr)²) + Fr(r Ṁr + c0(Mr−M²))/(c0 r²(1−Mr)³) ]_ret
  Vectorized Newton–Raphson retarded time: g(τ) = τ + r/c0 − t, g' = 1 − Mr, 8 fixed iters,
  step clamp ±0.1, init τ = t − |x|/c0. Subsonic only. **Validated < 1e-3 Pa vs analytic dipole.**
- `solver.py`: per-blade closures (y, v, F, Ḟ, Ṁ) → FW-H kernel. Forces computed in body frame
  once (steady hover), rotated by R_z(ψ); Ḟ analytic via Ω dR/dψ.

## Validated

Stationary dipole vs analytic; hover BPF peak (166.6 vs 166.7 Hz) & periodicity; variable-speed
stability; multi-observer vectorization exactness (f32/f64). End-to-end autograd through chord/
twist/polar params verified. Audio notebook: DREGON-LM RPS → Ω(τ) → pressure → audio.

## Bugs / gaps to fix in the JAX port (do NOT replicate)

1. **Variable-Ω azimuth wrong**: ψ = Ω(τ)·τ instead of ∫Ω dτ. A proper _integrate_azimuth exists
   but is dead code (and runs no_grad). Fix: cumulative-trapezoid phase, differentiable interp.
2. **BEMT inflow never used in acoustics**: compute_forces called with v_induced=None → α = θ,
   overpredicted loads. solve_hover() never invoked by solver.
3. **Forces frozen at Omega.mean()**: instantaneous ∝Ω² load modulation missing for variable RPM.
4. No thickness noise, no broadband, no forward-flight kinematics, no unsteady aero.
5. Perf: ~15 redundant source-quantity+BEMT evaluations per pressure call (closure design).
6. Blade tensors built with default dtype/device at construction.

## Reuse

- FW-H 1A kernel + Newton retarded-time structure (port to JAX; lax.scan/fori_loop-shaped).
- Blade parametric geometry approach; trapezoid strips; validation-test patterns
  (analytic dipole, BPF/periodicity, vectorization equivalence at two dtypes).
- APC 10x7 real-geometry ingestion example (FLOWUnsteady/UIUC data inlined).
- Build scaffold: uv + hatchling flat-src multi-package layout, ruff/pyright/git-hooks flake.

## omnirun setup (port to auraflow for task #9)

- Repo-level `omnirun.toml`: outputs = ["results/**"], resources gpus=1 time=1h,
  env kind = "uv" (NOT "auto" — notebook backends rewrite auto→system, losing uv.lock pinning).
- User-global `~/.config/omnirun/config.toml` already configures backends (shared across
  projects, but `project_root` points at harmonic-noise-suppression — auraflow needs its own
  entries or overrides):
  - apocrita-short: slurm, host=apocrita, partition=gpushort, ≤1h, gpu_map V100/A100/H100.
  - apocrita-long: partition=sae, account=pilot_sae_gpu, ≤10d, A100-80/H100/H200/L40.
  - colab: default_gpu=T4 (needs local keep-alive daemon; T4 allocation is a lottery).
  - kaggle: 30 weekly GPU hours; ~1 MB kernel source cap → slim-snapshot clone recipe.
- Launch: `omnirun submit --backend apocrita-short --gpus 1 --time 30m --yes -- python ...`;
  ps/status/logs/pull; needs clean pushed HEAD; `.env` ships automatically.
- Known warts: same-SHA worktree reuse poisons retries (override results_root);
  outputs glob scoops sibling results; stale heartbeats after SSH ControlMaster expiry.
