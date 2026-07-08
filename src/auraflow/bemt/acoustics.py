r"""BEMT -> Farassat-1A coupling: rotor tonal (thickness + loading) noise.

Maps the quasi-steady blade-element state (:class:`~auraflow.bemt.unsteady.SectionState`)
to compact moving sources and propagates them with the Farassat Formulation 1A
kernel (:func:`auraflow.fwh.f1a_pressure`). This is where the predecessor's bug
(a) is fixed: the loading that drives the acoustics is the *aerodynamic reaction
force including the BEMT-induced velocity*, taken straight from the section
state, so switching the induced velocity on and off changes the radiated signal.

Source model
------------
Each blade station segment (span ``dr``) becomes one compact source at its
quarter-chord / pitch axis:

- **Loading (dipole)**: force on the fluid ``L_i = f_i^{fluid} dr`` [N], the
  reaction to the sectional lift+drag, already rotated to the world frame by
  :func:`march_bemt`.
- **Thickness (monopole, optional)**: a compact displacement monopole of
  strength ``Q_n = rho0 A(r) W`` [kg/s], where ``A(r)`` is the airfoil
  cross-sectional area and ``W`` the local resultant speed. This is the
  standard compact "volume-displacement" approximation -- the section sweeps
  fluid at volumetric rate ``A W`` -- and its blade-passing modulation radiates
  the thickness tone. ``A(r) = area_coeff * (t/c) * chord^2`` with
  ``area_coeff = 0.685`` (a NACA-4-digit-like section-area constant). This is an
  approximation: a rigorous thickness prediction needs the full blade surface
  mesh (the route the CONA backend will take), but the compact monopole is
  adequate for the tonal thickness peak at moderate tip Mach number.

The kinematic source velocity/acceleration are the exact rigid-body values from
the section state (never finite-differenced positions), as required by the F1A
kernel.

Shapes: observers ``[O, 3]``; returned pressures ``[O, T_obs]`` on a uniform
observer-time grid. SI units, float64-safe and differentiable end to end.
"""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from auraflow.bemt.solver import Polar
from auraflow.bemt.unsteady import SectionState, march_bemt
from auraflow.core.airfoil import ThinAirfoilPolar
from auraflow.core.blade import Rotor
from auraflow.core.medium import Medium
from auraflow.fwh import f1a_pressure
from auraflow.fwh.geometry import default_observer_grid

__all__ = [
    "rotor_tonal_noise",
    "section_loading_sources",
    "section_thickness_sources",
]

_AREA_COEFF = 0.685  # airfoil cross-sectional area ~ area_coeff * (t/c) * chord^2


def _flatten_sources(state: SectionState) -> tuple[Array, Array, Array]:
    """Flatten ``[B, S, T, 3]`` kinematics to ``[B*S, T, 3]`` compact sources."""
    b, s, t = state.phi.shape
    y = state.position.reshape(b * s, t, 3)
    v = state.velocity.reshape(b * s, t, 3)
    a = state.acceleration.reshape(b * s, t, 3)
    return y, v, a


def section_loading_sources(state: SectionState) -> tuple[Array, Array, Array, Array]:
    """Compact loading (dipole) sources from a section state.

    The per-span world-frame force on the fluid is integrated over each station
    segment width ``dr`` to give the compact segment force.

    Args:
        state: The marched :class:`SectionState`.

    Returns:
        ``(y, v, a, force)``: source positions/velocities/accelerations
        ``[B*S, T, 3]`` [m, m/s, m/s^2] and the force on the fluid ``[B*S, T, 3]``
        [N].
    """
    y, v, a = _flatten_sources(state)
    b, s, t = state.phi.shape
    force = (state.force_on_fluid * state.dr[None, :, None, None]).reshape(b * s, t, 3)
    return y, v, a, force


def section_thickness_sources(
    state: SectionState, medium: Medium, thickness_ratio: ArrayLike = 0.12
) -> tuple[Array, Array, Array, Array]:
    r"""Compact thickness (monopole) sources from a section state.

    Uses the volume-displacement approximation ``Q_n = rho0 A(r) W`` with
    ``A(r) = area_coeff (t/c) chord^2`` (see the module docstring).

    Args:
        state: The marched :class:`SectionState`.
        medium: Ambient medium (supplies ``rho0``).
        thickness_ratio: Airfoil thickness-to-chord ratio ``t/c`` [-], scalar
            or shape ``[S]``.

    Returns:
        ``(y, v, a, qn)``: source kinematics ``[B*S, T, 3]`` and monopole
        strength ``qn`` ``[B*S, T]`` [kg/s].
    """
    y, v, a = _flatten_sources(state)
    b, s, t = state.phi.shape
    area = _AREA_COEFF * jnp.asarray(thickness_ratio) * state.chord**2  # [S]
    qn = (medium.rho0 * area[None, :, None] * state.w).reshape(b * s, t)
    return y, v, a, qn


def rotor_tonal_noise(
    rotor: Rotor,
    medium: Medium,
    t: ArrayLike,
    omega: ArrayLike,
    observers: ArrayLike,
    collective: ArrayLike = 0.0,
    v_inf: ArrayLike | None = None,
    hub_velocity: ArrayLike | None = None,
    polar: Polar | None = None,
    include_induced: bool = True,
    thickness: bool = True,
    thickness_ratio: ArrayLike = 0.12,
    tip_loss: bool = True,
    root_loss: bool = True,
    n_obs: int | None = None,
    t_obs: ArrayLike | None = None,
) -> tuple[Array, Array, Array, Array]:
    r"""Tonal (thickness + loading) noise of a rotor via BEMT + Farassat 1A.

    Marches the BEMT state (:func:`~auraflow.bemt.unsteady.march_bemt`), builds
    compact loading (and optionally thickness) sources, and propagates them to
    the observers with the moving-source F1A kernel. The observer-time grid is
    the common valid arrival window across all source/observer pairs unless
    ``t_obs`` is supplied.

    Args:
        rotor: Rotor with hub placement in the world frame.
        medium: Ambient medium.
        t: Uniform source-time grid [s], shape ``[T]``.
        omega: Rotor-speed magnitude history [rad/s]; scalar or ``[T]``.
        observers: Observer positions in the world frame [m], shape ``[O, 3]``.
        collective: Collective pitch [rad]; scalar or ``[T]``.
        v_inf: World-frame free-stream velocity [m/s], shape ``[3]``.
        hub_velocity: World-frame hub translation [m/s], shape ``[3]``.
        polar: Airfoil polar; defaults to :class:`ThinAirfoilPolar`.
        include_induced: Feed the BEMT-induced velocity into the loads (bug (a)
            fix). Set ``False`` to isolate its acoustic effect.
        thickness: Include the compact thickness monopole.
        thickness_ratio: Airfoil ``t/c`` for the thickness source [-].
        tip_loss: Apply the Prandtl tip loss in the inflow solve.
        root_loss: Apply the Prandtl root loss in the inflow solve.
        n_obs: Number of observer-time samples (default ``T``); ignored if
            ``t_obs`` is given.
        t_obs: Explicit uniform observer-time grid [s], shape ``[T_obs]``.

    Returns:
        ``(p_total, p_thickness, p_loading, t_obs)``: pressures [Pa] each of
        shape ``[O, T_obs]`` and the observer-time grid ``[T_obs]``. When
        ``thickness=False`` the thickness term is zero.
    """
    if polar is None:
        polar = ThinAirfoilPolar()
    t = jnp.asarray(t, dtype=float)
    observers = jnp.asarray(observers, dtype=float)

    state = march_bemt(
        rotor,
        medium,
        t,
        omega,
        collective=collective,
        v_inf=v_inf,
        hub_velocity=hub_velocity,
        polar=polar,
        include_induced=include_induced,
        tip_loss=tip_loss,
        root_loss=root_loss,
    )
    y, v, a, force = section_loading_sources(state)
    if thickness:
        qn = section_thickness_sources(state, medium, thickness_ratio)[3]
    else:
        qn = jnp.zeros(y.shape[:2])

    if t_obs is None:
        n = y.shape[0]
        d = observers[:, None, None, :] - y[None, :, :, :]  # [O, S, T, 3]
        r = jnp.linalg.norm(d, axis=-1)  # [O, S, T]
        arrival = t[None, None, :] + r / medium.c0
        t_obs = default_observer_grid(
            arrival.reshape(observers.shape[0] * n, t.shape[0]),
            t.shape[0] if n_obs is None else n_obs,
        )
    else:
        t_obs = jnp.asarray(t_obs, dtype=float)

    p_thick, p_load = f1a_pressure(observers, y, v, a, qn, force, medium, t, t_obs)
    return p_thick + p_load, p_thick, p_load, t_obs
