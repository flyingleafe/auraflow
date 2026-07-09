"""Mesh -> FW-H source adapters for ``auraflow.body``.

Turns a :class:`~auraflow.body.mesh.TriMesh` plus a :class:`Motion` (and,
optionally, a prescribed surface vibration and/or a surface-pressure history)
into the per-source histories that :func:`auraflow.fwh.f1a_pressure` consumes.
Three adapters live here:

- :func:`permeable_surface` -- the **permeable** data surface (points, outward
  normals, areas) that CFD sampling and :func:`auraflow.fwh.f1a_permeable_static`
  consume; the general-body counterpart of ``cfd.sphere.PermeableSphere``.
- :func:`impermeable_sources` -- the **impermeable / solid-body** adapter:
  panel kinematics (:func:`auraflow.body.panel_histories`) plus optional surface
  pressure become ``(y, v, a, L, Q_n)``, i.e. thickness (monopole) and loading
  (dipole) FW-H sources on each panel.
- :func:`mesh_pressure` -- the one-call radiation path: build sources, run the
  F1A kernel, and return the radiated pressure on a common observer grid.

Sign / normalization conventions (matching :mod:`auraflow.fwh.f1a`):

- Panel outward unit normal ``n`` points *into the fluid* (away from the body).
- Thickness source area density ``Q_n = rho0 * (v . n)`` [kg/(s m^2)] -- the
  panel-velocity ``v`` from :func:`panel_histories` already folds the rigid-body
  point velocity **and** any prescribed vibration ``u_n * n`` into its normal
  component, so this single expression covers a breathing/vibrating surface and
  a translating rigid body alike.
- Loading force area density ``L_i = p_surface * n_i`` [N/m^2] with
  ``p_surface`` the **gauge** surface pressure ``p - p0``; this is the force the
  surface exerts on the fluid (compressive stress ``P_ij = p_surface delta_ij``,
  ``L_i = P_ij n_j``), identical to the permeable-surface loading convention.
  The panel-area weighting (giving the physical panel force ``p_surface n dS``)
  is applied downstream by :func:`auraflow.fwh.f1a_pressure` through its ``area``
  argument, so both source densities pair with ``mesh.areas()``.

.. note::
   Static-motion fast path. For a :class:`StaticPose` the panel geometry is
   time-invariant, so radiation distances are constant and the retarded delays
   are closed-form; :func:`auraflow.fwh.f1a_permeable_static` exploits this. It
   is *not* used here because it derives ``Q_n``/``L`` from permeable panel
   fields ``(rho, u, p)`` and would inject a spurious ``rho0 u_n^2`` quadratic
   loading term for a purely vibrating (thickness) surface. The general
   :func:`auraflow.fwh.f1a_pressure` path handles static geometry correctly (the
   per-panel arrival times simply become constant shifts) and is used for every
   motion; the convective terms it carries are ``O(u_n / c0)`` and negligible at
   the acoustic Mach numbers of a loudspeaker membrane.
"""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from auraflow.body.mesh import TriMesh
from auraflow.body.motion import Motion, SurfaceVibration, panel_histories
from auraflow.core.medium import Medium
from auraflow.fwh.f1a import f1a_pressure
from auraflow.fwh.geometry import default_observer_grid

__all__ = ["impermeable_sources", "mesh_pressure", "permeable_surface"]


def permeable_surface(mesh: TriMesh) -> tuple[Array, Array, Array]:
    """Permeable FW-H data surface from a (closed) triangle mesh.

    Returns the per-face quadrature geometry the permeable-surface FW-H solver
    needs, matching the layout of :func:`auraflow.cfd.sphere.fibonacci_sphere`.
    The mesh should be watertight (a closed data surface); this is not enforced
    so open patches can still be sampled for diagnostics.

    Args:
        mesh: The permeable-surface :class:`TriMesh`.

    Returns:
        ``(points, normals, areas)`` with

        - ``points`` [m], shape ``[F, 3]`` -- face centroids (``y_panels``);
        - ``normals``, shape ``[F, 3]`` -- outward unit normals;
        - ``areas`` [m^2], shape ``[F]`` -- per-face areas.
    """
    return mesh.centroids(), mesh.normals(), mesh.areas()


def impermeable_sources(
    mesh: TriMesh,
    motion: Motion,
    tau: ArrayLike,
    medium: Medium,
    *,
    p_surface: ArrayLike | None = None,
    vibration: SurfaceVibration | None = None,
) -> tuple[Array, Array, Array, Array, Array]:
    """Impermeable-body FW-H source histories from mesh kinematics + pressure.

    Builds panel kinematic histories (:func:`panel_histories`) and reduces them
    to the thickness (monopole) and loading (dipole) F1A source **area
    densities**. Pair the returned ``Q_n``/``L`` with ``mesh.areas()`` when
    feeding :func:`auraflow.fwh.f1a_pressure` (see module docstring for the
    sign/area conventions); :func:`mesh_pressure` does this for you.

    Args:
        mesh: The body surface :class:`TriMesh`.
        motion: The rigid :class:`Motion` (pose history of the body).
        tau: Uniform source-time grid [s], shape ``[T]``.
        medium: Ambient :class:`Medium`.
        p_surface: Optional **gauge** surface pressure ``p - p0`` [Pa] per panel,
            shape ``[F, T]`` (from CFD or prescribed). ``None`` -> no loading
            (thickness-only radiation, e.g. a loudspeaker membrane).
        vibration: Optional prescribed :class:`SurfaceVibration` (membrane
            normal velocity), superimposed on the rigid motion.

    Returns:
        ``(y, v, a, L, Q_n)`` with panel-centroid kinematics ``y, v, a`` of shape
        ``[F, T, 3]`` [m, m/s, m/s^2], loading area density ``L`` shape
        ``[F, T, 3]`` [N/m^2] (zeros when ``p_surface`` is ``None``), and
        thickness area density ``Q_n`` shape ``[F, T]`` [kg/(s m^2)].
    """
    hist = panel_histories(mesh, motion, tau, vibration)
    qn = medium.rho0 * jnp.sum(hist.v * hist.n, axis=-1)  # [F, T]
    if p_surface is None:
        load = jnp.zeros_like(hist.y)  # [F, T, 3]
    else:
        load = jnp.asarray(p_surface, dtype=jnp.float64)[..., None] * hist.n
    return hist.y, hist.v, hist.a, load, qn


def _common_observer_grid(observers: Array, y: Array, tau: Array, c0: Array, n_obs: int) -> Array:
    """Uniform observer grid valid (no extrapolation) for **all** observers.

    Intersects the per-observer valid arrival windows of
    :func:`default_observer_grid` across every observer, so a single shared
    ``t_obs`` can be handed to :func:`f1a_pressure`. Computed one observer at a
    time (Python loop over the few microphones) to keep the peak memory at
    ``[F, T]`` rather than ``[O, F, T]``.
    """
    los: list[Array] = []
    his: list[Array] = []
    for k in range(observers.shape[0]):
        r = jnp.linalg.norm(observers[k] - y, axis=-1)  # [F, T]
        arrival = tau[None, :] + r / c0
        grid = default_observer_grid(arrival, 2)  # endpoints of the valid window
        los.append(grid[0])
        his.append(grid[-1])
    t_lo = jnp.max(jnp.stack(los))
    t_hi = jnp.min(jnp.stack(his))
    return jnp.linspace(t_lo, t_hi, n_obs)


def mesh_pressure(
    mesh: TriMesh,
    motion: Motion,
    tau: ArrayLike,
    observers: ArrayLike,
    medium: Medium,
    *,
    p_surface: ArrayLike | None = None,
    vibration: SurfaceVibration | None = None,
    n_obs: int | None = None,
) -> tuple[Array, Array]:
    """Radiated pressure of an arbitrary mesh -- the one-call FW-H path.

    Builds impermeable thickness + loading sources
    (:func:`impermeable_sources`), runs the Farassat 1A kernel
    (:func:`auraflow.fwh.f1a_pressure`) with area-weighting from
    ``mesh.areas()``, and returns the summed (thickness + loading) pressure on a
    shared observer-time grid (:func:`default_observer_grid`, intersected over
    observers). Differentiable through the mesh vertices, the motion parameters,
    the vibration and the surface pressure.

    All motions (including :class:`StaticPose`) route through the general kernel;
    see the module docstring for why the static permeable fast path is avoided.

    Args:
        mesh: The body surface :class:`TriMesh`.
        motion: The rigid :class:`Motion`.
        tau: Uniform source-time grid [s], shape ``[T]``.
        observers: Observer positions [m], shape ``[O, 3]``.
        medium: Ambient :class:`Medium`.
        p_surface: Optional gauge surface pressure [Pa], shape ``[F, T]``.
        vibration: Optional prescribed :class:`SurfaceVibration`.
        n_obs: Number of observer-time samples ``T_obs`` (default ``len(tau)``).

    Returns:
        ``(p, t_obs)``: total acoustic pressure [Pa], shape ``[O, T_obs]``, and
        the observer-time grid [s], shape ``[T_obs]``.
    """
    tau = jnp.asarray(tau, dtype=jnp.float64)
    observers = jnp.asarray(observers, dtype=jnp.float64)
    n_obs = tau.shape[0] if n_obs is None else n_obs
    y, v, a, load, qn = impermeable_sources(
        mesh, motion, tau, medium, p_surface=p_surface, vibration=vibration
    )
    area = mesh.areas()
    t_obs = _common_observer_grid(observers, y, tau, medium.c0, n_obs)
    p_t, p_l = f1a_pressure(observers, y, v, a, qn, load, medium, tau, t_obs, area)
    return p_t + p_l, t_obs
