"""Simulation driver: march JAX-Fluids, sample the sphere, propagate via FW-H.

This is the coupling glue for the full-CFD backend. It

1. builds and initialises a JAX-Fluids simulation from a :class:`CFDCase`,
2. marches it with a **manual integration-step loop** (rather than
   ``SimulationManager.simulate``) so the permeable sphere can be sampled every
   ``k`` steps *in memory* -- no HDF5 written to disk
   (``docs/research/jaxfluids-evaluation.md``),
3. reduces the sampled surface fields to a :class:`SurfaceHistory` PyTree, and
4. feeds that history to :func:`auraflow.fwh.f1a_permeable_static` to obtain the
   far-field pressure at observers (:func:`propagate_to_observers`).

JAX-Fluids is imported lazily inside the driver so the base install works
without the ``cfd`` extra. The CFD state is single precision by default; it is
upcast to float64 at the FW-H boundary (retarded-time math is precision
sensitive -- ``docs/architecture.md``).

Differentiability: JAX-Fluids' integration step is jit/grad-compatible with a
fixed timestep (verified in the evaluation digest). The Python sampling loop here
is written for clarity and in-memory sampling; a fully differentiable rollout of
many steps should use JAX-Fluids' ``feed_forward`` (scan + inner-step
checkpointing) to bound memory -- see the module notes. All *post-CFD* steps
(sphere interpolation, FW-H) are differentiable as written.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from auraflow.cfd.case import CFDCase
from auraflow.cfd.sphere import PermeableSphere, sample_primitives
from auraflow.core.medium import Medium
from auraflow.fwh.f1a import f1a_permeable_static

if TYPE_CHECKING:  # pragma: no cover - typing only
    from auraflow.cfd.body_case import PermeableMeshSurface
    from auraflow.fwh.geometry import radiation_vectors  # noqa: F401
    from auraflow.viz.server import VizStreamer

__all__ = [
    "SurfaceHistory",
    "propagate_to_observers",
    "run_acoustic_case",
]


class SurfaceHistory(eqx.Module):
    """Time history of permeable-surface fields sampled from the CFD.

    A PyTree carrying exactly the arrays :func:`f1a_permeable_static` consumes
    (together with the static :class:`~auraflow.cfd.sphere.PermeableSphere`
    geometry). Shapes follow ``docs/architecture.md``: ``S`` surface points, ``T``
    samples.

    Attributes:
        tau: Source-time grid [s], shape ``[T]`` (uniform, spacing
            ``dt * sample_every``).
        rho: Density at the surface points [kg/m^3], shape ``[S, T]``.
        u: Velocity at the surface points [m/s], shape ``[S, T, 3]``.
        p: Pressure at the surface points [Pa], shape ``[S, T]``.
    """

    tau: Array
    rho: Array
    u: Array
    p: Array


def _interior_primitives(jxf_buffers: Any, domain_information: Any) -> Array:
    """Strip halo cells from the JAX-Fluids primitive buffer.

    Returns the interior primitives ``[5, Nx, Ny, Nz]`` ordered
    ``(rho, u, v, w, p)``.
    """
    primitives = jxf_buffers.simulation_buffers.material_fields.primitives
    slices = tuple(domain_information.domain_slices_conservatives)
    return primitives[(slice(None), *slices)]


def run_acoustic_case(
    case: CFDCase,
    sphere: PermeableSphere | PermeableMeshSurface,
    n_steps: int,
    sample_every: int = 1,
    warmup_steps: int = 0,
    viz: VizStreamer | None = None,
    viz_field: str = "p",
    viz_slice_axis: str = "z",
) -> SurfaceHistory:
    """Run a JAX-Fluids acoustic case and sample the permeable sphere.

    Marches ``n_steps`` fixed-``dt`` integration steps, sampling ``(rho, u, p)``
    on ``sphere`` every ``sample_every`` steps after an optional ``warmup_steps``
    transient discard. All sampling happens in memory.

    Args:
        case: The :class:`~auraflow.cfd.case.CFDCase` to run.
        sphere: Static permeable data surface -- a
            :class:`~auraflow.cfd.sphere.PermeableSphere` or a
            :class:`~auraflow.cfd.body_case.PermeableMeshSurface` (any closed
            mesh); must lie strictly inside ``case.domain``.
        n_steps: Number of integration steps (static int).
        sample_every: Sample the sphere every this many steps (static int).
        warmup_steps: Steps to run before the first sample (transient discard).
        viz: Optional :class:`~auraflow.viz.server.VizStreamer` for live in-browser
            visualization. When given, a downsampled mid-plane field slice and the
            sphere overpressure ``p'`` are pushed every sample step (best-effort,
            non-blocking; ``None`` = zero overhead). The scene (domain box, sphere
            point cloud, slice plane) is published once before the march.
        viz_field: Primitive field shown on the live slice (``"p"`` or ``"rho"``).
        viz_slice_axis: Axis the live field slice is taken normal to.

    Returns:
        A :class:`SurfaceHistory` with ``tau`` starting at the first sampled
        physical time and spacing ``dt * sample_every``.

    Raises:
        ImportError: if the ``cfd`` extra (jaxfluids) is not installed.
    """
    try:
        from jaxfluids import (  # type: ignore[import-untyped]
            InitializationManager,
            InputManager,
            SimulationManager,
        )
        from jaxfluids.data_types.ml_buffers import (  # type: ignore[import-untyped]
            CallablesSetup,
            ParametersSetup,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "auraflow.cfd.run requires the 'cfd' extra (jaxfluids). Install with "
            "`uv sync --extra cfd` or `pip install 'auraflow[cfd]'`."
        ) from exc

    input_manager = InputManager(case.case, case.numerical_setup)
    init_manager = InitializationManager(input_manager)
    sim_manager = SimulationManager(input_manager)
    domain_information = input_manager.domain_information

    # Level-set body cases carry an initial level-set field (the body SDF sampled
    # at the cell centres, negative-inside per JAX-Fluids' convention); inject it.
    levelset_init = getattr(case, "levelset_init", None)
    if levelset_init is not None:
        jxf_buffers = init_manager.initialization(user_levelset_init=jnp.asarray(levelset_init))
    else:
        jxf_buffers = init_manager.initialization()
    ml_parameters = ParametersSetup()
    ml_callables = CallablesSetup()

    x, y, z = case.domain.cell_centers()

    if viz is not None:
        from auraflow.viz.cfd import cfd_scene_kwargs

        viz.init_scene(
            **cfd_scene_kwargs(case.domain, sphere, field=viz_field, slice_axis=viz_slice_axis)
        )
    p0 = float(case.medium.p0)

    rho_list: list[Array] = []
    u_list: list[Array] = []
    p_list: list[Array] = []
    tau_list: list[float] = []

    for step in range(n_steps):
        control_flow_params = sim_manager.compute_control_flow_params(
            jxf_buffers.time_control_variables, jxf_buffers.step_information
        )
        jxf_buffers, _ = sim_manager.do_integration_step(
            jxf_buffers, control_flow_params, ml_parameters, ml_callables
        )
        if step < warmup_steps:
            continue
        if (step - warmup_steps) % sample_every != 0:
            continue
        interior = _interior_primitives(jxf_buffers, domain_information)
        rho_s, u_s, p_s = sample_primitives(interior, x, y, z, sphere.points)
        rho_list.append(rho_s)
        u_list.append(u_s)
        p_list.append(p_s)
        t_now = float(jxf_buffers.time_control_variables.physical_simulation_time)
        tau_list.append(t_now)

        if viz is not None and viz.active:
            from auraflow.viz.cfd import cfd_frame_kwargs

            interior_np = jax.device_get(interior)
            viz.push_frame(
                **cfd_frame_kwargs(
                    interior_np,
                    x,
                    y,
                    z,
                    jax.device_get(p_s),
                    p0,
                    t_now,
                    len(tau_list) - 1,
                    field=viz_field,
                    slice_axis=viz_slice_axis,
                )
            )

    if not rho_list:
        raise ValueError("no samples collected; check n_steps/sample_every/warmup_steps")

    rho = jnp.stack(rho_list, axis=1)  # [S, T]
    u = jnp.stack(u_list, axis=1)  # [S, T, 3]
    p = jnp.stack(p_list, axis=1)  # [S, T]
    tau = jnp.asarray(tau_list)  # [T]
    return SurfaceHistory(tau=tau, rho=rho, u=u, p=p)


def propagate_to_observers(
    surface_history: SurfaceHistory,
    sphere: PermeableSphere | PermeableMeshSurface,
    observers: Array,
    medium: Medium,
    n_obs: int | None = None,
) -> tuple[Array, Array]:
    """Propagate a sampled surface history to observers via permeable FW-H.

    Wraps :func:`auraflow.fwh.f1a_permeable_static` (static-sphere fast path).
    The surface fields are upcast to float64 here (retarded-time math is
    precision sensitive).

    Each observer gets its **own** time window spanning *all* panel arrivals,
    ``[min_s(tau_0 + r_s/c0), max_s(tau_-1 + r_s/c0)]`` (the union over the ``S``
    panels), rather than one window shared across observers. Two reasons:

    - observers at very different ranges have disjoint arrival windows, so a
      single shared grid would miss the signal at the nearer ones;
    - for a transient (e.g. a pulse) only some panels carry signal at any instant,
      and the physically-arriving contribution from the near panels falls *before*
      the intersection window ``[max_s(tau_0 + r_s/c0), ...]`` used by
      :func:`auraflow.fwh.geometry.default_observer_grid` opens. The union window
      captures it; per-panel constant extrapolation onto the shared grid clamps to
      the (near-zero) pre/post-pulse tails, so no spurious signal is introduced as
      long as the run is long enough that the surface fields return to ambient.

    Consequently ``t_obs`` is returned per observer.

    Args:
        surface_history: Sampled :class:`SurfaceHistory`.
        sphere: The static :class:`~auraflow.cfd.sphere.PermeableSphere` the
            history was sampled on.
        observers: Observer positions [m], shape ``[O, 3]``.
        medium: Ambient :class:`~auraflow.core.medium.Medium`.
        n_obs: Number of observer-time samples per observer (default: ``T``).

    Returns:
        ``(p_prime, t_obs)`` with total acoustic pressure ``p_prime`` [Pa] shape
        ``[O, T_obs]`` (thickness + loading) and the per-observer time grids
        ``t_obs`` [s] shape ``[O, T_obs]``.
    """
    observers = jnp.asarray(observers, dtype=jnp.float64)
    points = jnp.asarray(sphere.points, dtype=jnp.float64)
    normals = jnp.asarray(sphere.normals, dtype=jnp.float64)
    area = jnp.asarray(sphere.area, dtype=jnp.float64)
    rho = jnp.asarray(surface_history.rho, dtype=jnp.float64)
    u = jnp.asarray(surface_history.u, dtype=jnp.float64)
    p = jnp.asarray(surface_history.p, dtype=jnp.float64)
    tau = jnp.asarray(surface_history.tau, dtype=jnp.float64)
    c0 = jnp.asarray(medium.c0, dtype=jnp.float64)
    n_obs = tau.shape[0] if n_obs is None else n_obs

    def one_observer(x_o: Array) -> tuple[Array, Array]:
        r = jnp.linalg.norm(x_o - points, axis=-1)  # [S]
        t_lo = tau[0] + jnp.min(r) / c0
        t_hi = tau[-1] + jnp.max(r) / c0
        t_obs_o = jnp.linspace(t_lo, t_hi, n_obs)
        pt, pl = f1a_permeable_static(
            x_o[None, :], points, normals, area, rho, u, p, medium, tau, t_obs_o
        )
        return (pt + pl)[0], t_obs_o

    p_prime, t_obs = jax.vmap(one_observer)(observers)
    return p_prime, t_obs
