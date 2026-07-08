# Differentiable CFD ecosystem survey (July 2026)

Decision: **JAX-Fluids** as the full-CFD backend (behind a thin `auraflow.cfd` facade);
**XLB** optional secondary for tip-Mach ≤ 0.3; everything acoustic/rotor-specific in-house.

## Shortlist verdicts

- **JAX-Fluids** (MIT, v0.2.1, active through 2026): only mature differentiable *compressible*
  NS solver in JAX with prescribed-motion level-set immersed solids. WENO/TENO high-order,
  ALDM ILES, sponge forcing, checkpointed reverse-mode AD, CPU/GPU/TPU (to 512 A100s /
  2048 TPUv3). Gaps: no NSCBC non-reflecting BCs (sponge + grid stretching only), Cartesian
  stretched grids only, no AMR, git-only install, docs lag source, **no published
  aeroacoustics/rotor applications — validation burden is ours**. Level-set reinit must run
  fixed-iteration mode (`is_jaxwhileloop: false`) for reverse AD.
- **XLB** (Apache-2.0, pip, active): LBM, JAX+Warp backends; rotating-turbine IBM example
  (Warp backend); Warp backend gradients broken as of May 2026 (issue #161); standard LBM
  credible for direct acoustics only to Ma≈0.3. Secondary backend candidate only.
- **jax-cfd**: officially unmaintained, incompressible-only. Excluded.
- **Exponax**: periodic semi-linear spectral; irrelevant.
- **PhiFlow**: JAX backend + convenient rotating obstacles but incompressible/low-order —
  prototyping sandbox only.
- **NVIDIA Warp** (Apache-2.0 since May 2025): differentiable kernel framework w/ JAX interop;
  option for custom kernels later.
- **Trixi.jl**: only surveyed code with built-in CAA coupling (Euler–APE) — design reference;
  Julia + forward-mode-first AD → not a dependency.
- **j-Wave** (LGPL, JAX): differentiable acoustic propagation w/ PML, quiescent medium —
  candidate for far-field scattering beyond FW-H someday.
- Watch list: JANC (JAX-AMR, 2D Euler), DiFVM (unstructured JAX FVM, no repo yet),
  JAX-Shock (no repo). FluidX3D fastest LBM but non-commercial license, non-differentiable.

## Gaps we fill (nothing exists in JAX; medium-confidence absence after systematic search)

1. **Permeable-surface FW-H in JAX** — flagship. Templates: AcousticAnalogies.jl (NASA
   Glenn, AD-compatible *compact* F1A + Brooks & Burley broadband, validated vs ANOPP2),
   OpenCFD-FWH (Fortran permeable, our reference #2), mcmehrtens/FW-H-Solver (archived,
   validation reference), libAcoustics (OpenFOAM). Canonical math: Farassat NASA/TM-2007-214853.
2. **NSCBC outflow for JAX-Fluids** — potential upstream contribution; sponge first.
3. **JAX BEMT** — port CCBlade.jl residual formulation (Ning 2014/2021 guaranteed-convergence
   residual → `jax.lax.custom_root` implicit diff). Cross-check pyBEMT.
4. **Actuator-line forcing** — Gaussian-projected blade-element forces as source terms
   (Tier A, cheap+robust); Tier B resolved rotating level-set blades (research-grade).
5. **BPM broadband in JAX** — legacy Python2/Julia ports exist as references.

## Validation campaign to budget

Convected monopole/dipole → cylinder Aeolian tone → APC propeller vs experiment →
JASA NASA 1-Pax cases vs CONA backend.
