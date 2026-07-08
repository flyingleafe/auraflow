r"""CONA tonal noise: SectionState -> compact FW-H sources -> observer pressure.

The acoustic back end of the CONA pipeline
(``docs/research/cona-reference.md`` module 4). It reuses the BEMT compact-source
adapters (:func:`auraflow.bemt.acoustics.section_loading_sources` /
:func:`~auraflow.bemt.acoustics.section_thickness_sources`) to turn each rotor's
:class:`~auraflow.bemt.unsteady.SectionState` (built here by
:func:`auraflow.cona.airloads.rotor_section_state`) into moving compact
monopole (thickness) and dipole (loading) sources, then propagates them with:

- **Farassat 1A** (:func:`auraflow.fwh.f1a_pressure`) for hover / static cases
  (no mean flow), directly in the world frame; or
- **Formulation 1C** (:func:`auraflow.fwh.f1c_pressure`) for forward flight:
  the mean flight speed sets ``mach0 = V/c0``; the scene is rotated so the
  effective mean flow (head wind ``-V``) is along ``+x1``, the vehicle
  translation is removed so the hub is (nearly) fixed in the mean-flow frame,
  and the pre-contracted ``qn`` / ``load`` sources are convected exactly.
  ``f1c`` reduces to ``f1a`` as ``mach0 -> 0``, so the two paths agree at low
  speed (a consistency gate in ``tests/cona/test_tonal.py``).

Multi-rotor cases sum coherently in the time domain on a shared observer grid,
so blade-passage interference and amplitude modulation emerge naturally.

Shapes: observers ``[O, 3]``; returned pressures ``[O, T_obs]``. SI, float64,
differentiable end to end.
"""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from auraflow.bemt.acoustics import section_loading_sources, section_thickness_sources
from auraflow.bemt.solver import Polar
from auraflow.bemt.unsteady import SectionState
from auraflow.cona.airloads import rotor_section_state
from auraflow.cona.flight import FlightHistory
from auraflow.core.blade import Vehicle
from auraflow.core.medium import Medium
from auraflow.fwh import default_observer_grid, f1a_pressure, f1c_pressure

__all__ = ["cona_tonal_noise", "mean_flow_frame"]

_EPS = 1.0e-12


def mean_flow_frame(v_mean: Array) -> tuple[Array, Array]:
    r"""Rotation to the Formulation-1C frame and the free-stream Mach number.

    Given the mean flight velocity ``v_mean`` (world frame), the air seen by the
    vehicle is a head wind ``U0 = -v_mean``. Formulation 1C requires ``U0`` along
    ``+x1``; this builds an orthonormal rotation ``Rw`` (world -> 1C frame) whose
    first row is the head-wind direction, so ``Rw @ (-v_mean/|v_mean|) = e1``.
    The flight speed ``|v_mean|`` is returned (the caller forms ``mach0 =
    speed/c0``). For a (near) hover ``|v_mean| ~ 0`` the identity rotation is
    returned.

    Args:
        v_mean: Mean flight velocity in the world frame [m/s], shape ``[3]``.

    Returns:
        ``(Rw, speed)``: rotation matrix ``[3, 3]`` (world -> 1C frame) and the
        flight speed ``|v_mean|`` [m/s].
    """
    speed = jnp.linalg.norm(v_mean)
    head = jnp.where(speed > 1.0e-6, -v_mean / (speed + _EPS), jnp.array([1.0, 0.0, 0.0]))
    # Pick a reference not parallel to the head-wind direction.
    ref = jnp.where(jnp.abs(head[2]) < 0.9, jnp.array([0.0, 0.0, 1.0]), jnp.array([0.0, 1.0, 0.0]))
    b1 = head
    b2 = ref - jnp.dot(ref, b1) * b1
    b2 = b2 / (jnp.linalg.norm(b2) + _EPS)
    b3 = jnp.cross(b1, b2)
    rw = jnp.stack([b1, b2, b3], axis=0)  # rows -> new-basis components
    rw = jnp.where(speed > 1.0e-6, rw, jnp.eye(3))
    return rw, speed


def _rotor_sources(
    state: SectionState, medium: Medium, thickness: bool, thickness_ratio: ArrayLike
) -> tuple[Array, Array, Array, Array, Array]:
    """Compact ``(y, v, a, load, qn)`` sources for one rotor's SectionState."""
    y, v, a, load = section_loading_sources(state)  # [S,T,3] x4, world frame
    if thickness:
        qn = section_thickness_sources(state, medium, thickness_ratio)[3]  # [S,T]
    else:
        qn = jnp.zeros(y.shape[:2])
    return y, v, a, load, qn


def cona_tonal_noise(
    vehicle: Vehicle,
    flight: FlightHistory,
    observers: ArrayLike,
    medium: Medium,
    collective: ArrayLike = 0.0,
    polar: Polar | None = None,
    gust: ArrayLike | None = None,
    thickness: bool = True,
    thickness_ratio: ArrayLike = 0.12,
    flow_model: str = "auto",
    n_obs: int | None = None,
    t_obs: ArrayLike | None = None,
    **airload_kwargs: object,
) -> tuple[Array, Array, Array, Array]:
    r"""Tonal (thickness + loading) noise of a multirotor over a flight history.

    Builds each rotor's :class:`SectionState` (:func:`rotor_section_state`),
    converts to compact FW-H sources (reusing the BEMT adapters), and propagates
    with Farassat 1A (hover) or Formulation 1C (forward flight), summing rotors
    coherently on a common observer-time grid.

    Args:
        vehicle: The vehicle (rotor placements / geometry).
        flight: Upstream :class:`~auraflow.cona.flight.FlightHistory`.
        observers: Observer positions in the world frame [m], shape ``[O, 3]``.
        medium: Ambient medium.
        collective: Collective pitch [rad]; scalar, ``[T]`` or ``[Nr]``.
        polar: Airfoil polar (shared across rotors).
        gust: Optional world-frame gust series [m/s], shape ``[T, 3]``.
        thickness: Include the compact thickness monopole.
        thickness_ratio: Airfoil ``t/c`` for the thickness source [-].
        flow_model: ``"auto"`` (F1C if the mean flight Mach exceeds ``1e-4``,
            else F1A), ``"f1a"``, or ``"f1c"`` (force one path).
        n_obs: Number of observer-time samples (default ``T``); ignored if
            ``t_obs`` is given.
        t_obs: Explicit uniform observer-time grid [s], shape ``[T_obs]``.
        **airload_kwargs: Forwarded to :func:`rotor_section_state` (wake /
            unsteady options, e.g. ``include_induced``, ``wake_params``).

    Returns:
        ``(p_total, p_thickness, p_loading, t_obs)``: pressures [Pa] each of
        shape ``[O, T_obs]`` and the observer-time grid ``[T_obs]``.
    """
    observers = jnp.asarray(observers, dtype=float)
    t = jnp.asarray(flight.t, dtype=float)
    n_rotors = vehicle.n_rotors
    coll = jnp.asarray(collective, dtype=float)

    # Build each rotor's SectionState and its compact world-frame sources.
    states: list[SectionState] = []
    world_sources: list[tuple[Array, Array, Array, Array, Array]] = []
    for i in range(n_rotors):
        coll_i = coll[i] if coll.ndim == 1 and coll.shape[0] == n_rotors else coll
        state = rotor_section_state(
            vehicle,
            flight,
            i,
            medium,
            collective=coll_i,
            polar=polar,
            gust=gust,
            **airload_kwargs,  # type: ignore[arg-type]
        )
        states.append(state)
        world_sources.append(_rotor_sources(state, medium, thickness, thickness_ratio))

    v_mean = jnp.mean(flight.v, axis=0)  # [3]
    rw, speed = mean_flow_frame(v_mean)
    mach0 = speed / medium.c0

    use_f1c = {
        "auto": bool(mach0 > 1.0e-4),
        "f1c": True,
        "f1a": False,
    }[flow_model]

    # Observer-time grid from the pooled world-frame arrival window.
    if t_obs is None:
        arrivals = []
        for y, *_ in world_sources:
            d = observers[:, None, None, :] - y[None, :, :, :]  # [O,S,T,3]
            r = jnp.linalg.norm(d, axis=-1)
            arrivals.append((t[None, None, :] + r / medium.c0).reshape(-1, t.shape[0]))
        arrival = jnp.concatenate(arrivals, axis=0)
        t_obs = default_observer_grid(arrival, t.shape[0] if n_obs is None else n_obs)
    else:
        t_obs = jnp.asarray(t_obs, dtype=float)

    p_thick = jnp.zeros((observers.shape[0], t_obs.shape[0]))
    p_load = jnp.zeros((observers.shape[0], t_obs.shape[0]))

    if not use_f1c:
        # Hover / static: propagate the world-frame sources directly with F1A.
        for y, v, a, load, qn in world_sources:
            pt, pl = f1a_pressure(observers, y, v, a, qn, load, medium, t, t_obs)
            p_thick = p_thick + pt
            p_load = p_load + pl
        return p_thick + p_load, p_thick, p_load, t_obs

    # Forward flight: transform to the mean-flow frame (hub ~ fixed, U0 along
    # +x1), then propagate with Formulation 1C.
    x_cg = flight.x  # [T, 3]
    v_cg = flight.v  # [T, 3]
    x_ref = x_cg[0]  # reference origin (observers fixed at the initial pose)
    obs_f = jnp.einsum("ij,oj->oi", rw, observers - x_ref)  # [O, 3]

    for y, v, a, load, qn in world_sources:
        # Remove the vehicle translation so the hub is ~fixed in this frame.
        y_vf = y - x_cg[None, :, :]  # [S,T,3]
        v_vf = v - v_cg[None, :, :]
        y_f = jnp.einsum("ij,stj->sti", rw, y_vf)
        v_f = jnp.einsum("ij,stj->sti", rw, v_vf)
        a_f = jnp.einsum("ij,stj->sti", rw, a)
        load_f = jnp.einsum("ij,stj->sti", rw, load)
        pt, pl = f1c_pressure(obs_f, y_f, v_f, a_f, qn, load_f, medium, mach0, t, t_obs)
        p_thick = p_thick + pt
        p_load = p_load + pl

    return p_thick + p_load, p_thick, p_load, t_obs
