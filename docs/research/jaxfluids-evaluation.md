# JAXFLUIDS hands-on evaluation (CPU, Python 3.12, uv)

Evaluated jaxfluids 0.2.1 @ commit cb46ed6, July 2026. License: MIT.

## Install

- NOT on PyPI. Works: `uv pip install "git+https://github.com/tumaer/JAXFLUIDS.git"`.
- Zero version pins in setup.py (bare flax/jax/h5py/optax/gymnasium/...); resolver pulled
  jax 0.10.2 and it runs fine — but pin jax + the jaxfluids commit ourselves.
- Ships jaxfluids_nn / _rl / _postprocess / _thirdparty as siblings; ~40 transitive deps.
- Cosmetics: SyntaxWarnings on import, mandatory ASCII-banner logger.

## Verified empirically

- 1D Sod shock tube (200 cells, WENO5-Z/HLLC/RK3): 9 s wall on CPU, HDF5 outputs read back
  via jaxfluids_postprocess.
- `InputManager` accepts plain Python dicts (no JSON files needed). Initial conditions are
  stringified lambdas that get eval'd — wrap behind our facade.
- Manual stepping loop: `InitializationManager(im).initialization()` → `JaxFluidsBuffers`
  pytree; per step `compute_control_flow_params` + `do_integration_step`. State readable
  each step: `simulation_buffers.material_fields.primitives[..., nhx, nhy, nhz]`.
- Per-step probing of state at arbitrary points inside the loop is jit/grad-compatible
  (use `jax.scipy.ndimage.map_coordinates` on cell centers for 3D sphere sampling).
- **Autodiff verified**: `jax.value_and_grad` through a 10-step `_feed_forward` rollout
  (is_scan + inner-step checkpointing) → finite sensible gradients, 39 s CPU incl. compile.
  Fixed dt in the differentiable path (no CFL adaptivity); pass interior domain w/o halos.
- Callback hooks (`before_step_start`, `after_step_end`) and ML-injection path
  (`ParametersSetup`/`CallablesSetup`) exist.

## Gaps relevant to AuraFlow

- Rotor forcing: `custom_forcing` callables get only (x,y,z,t), not flow state — fine for
  prescribed body-force disks; state-dependent actuator-line needs a ~10-line subclass of
  `Forcing.compute_custom_forcing` (which does receive primitives), or mutate buffers
  between steps (first-order splitting).
- Rotating solids: levelset FLUID-SOLID with prescribed solid velocity lambdas of (x,y,t)
  (moving-solid example exists: translating cylinder); rotating resolved blades expressible
  but resolution-hungry; `solid_coupling.dynamic` exists.
- BCs: no characteristic/non-reflecting BC; use the built-in **sponge-layer forcing**
  (target primitives + spatial strength callables).
- Compressible-only (good for acoustics, but low-Mach stiffness: acoustic CFL limits dt).
- 0.2.x API instability (`materials_setup` raises NotImplementedError); `simulate()`
  insists on output dirs/global logger — use `initialization()` + own loop.

## Verdict

Viable differentiable compressible backend. Vendor-pin the commit, wrap behind
`auraflow.cfd` facade, add: state-dependent actuator forcing patch, sphere sampling,
sponge configuration helpers.
