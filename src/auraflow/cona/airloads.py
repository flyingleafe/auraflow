r"""CONA HBEM airloads: FlightHistory + Vehicle -> per-rotor SectionState.

The middle stage of the CONA backend (``docs/research/cona-reference.md``
module 2 "Aerodynamics" + module 3 "unsteady corrections"). It consumes the
:class:`~auraflow.cona.flight.FlightHistory` produced upstream and, for every
rotor of a :class:`~auraflow.core.blade.Vehicle`, builds the time-resolved
blade-element state as the **same** :class:`~auraflow.bemt.unsteady.SectionState`
PyTree the BEMT backend emits -- so the downstream tonal
(:mod:`auraflow.cona.tonal`) and BPM broadband stages consume it unchanged.

What CONA swaps relative to :func:`auraflow.bemt.unsteady.march_bemt`
--------------------------------------------------------------------
- **Inflow**: instead of the per-annulus momentum-balance induced velocity, the
  axial induced velocity comes from the Beddoes **prescribed wake**
  (:mod:`auraflow.cona.wake`): a tip-vortex helix built from the rotor's
  operating point, evaluated on the disk once and indexed by blade azimuth.
- **Unsteady aero**: the section angle of attack is passed through the Wagner
  deficiency march (:func:`auraflow.cona.unsteady_aero.effective_aoa`) so the
  circulatory load lags the quasi-steady value, and an apparent-mass term is
  added for the variable section speed.
- **Kinematics**: the hub follows the full 6-DOF vehicle motion (translation +
  attitude), not a constant hub velocity.

The rest -- velocity triangle, polar lookup, force-on-fluid assembly, world
compact-source kinematics -- mirrors ``march_bemt`` (whose conventions this
module reuses: rotor-frame ``+z`` thrust axis, azimuth from ``+x`` toward
``+y``, force on fluid ``= f_t e_motion - f_n z_rotor``).

Approximations (documented deviations)
--------------------------------------
- The prescribed wake is built once at the **mean** operating point (mean rotor
  speed / thrust / advance ratio) rather than trimmed each step; the disk
  inflow is evaluated on a fixed rotor-frame grid and indexed by azimuth (fore-
  aft asymmetry retained, higher harmonics of the wake distortion dropped).
- The section velocity triangle uses the **hub** relative-air velocity; the
  extra blade velocity from the (small) vehicle body rate ``Omega_body`` is
  neglected in the aero triangle (kept only in the hub translation). Valid when
  ``Omega_body << Omega_rotor`` -- the level-flight / hover CONA cases.

SI, float64-safe, ``scan``/``vmap``/``grad`` friendly.
"""

from typing import cast

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from auraflow.bemt.solver import Polar
from auraflow.bemt.unsteady import SectionState
from auraflow.cona.flight import FlightHistory
from auraflow.cona.unsteady_aero import effective_aoa
from auraflow.cona.wake import make_prescribed_wake
from auraflow.core.airfoil import ThinAirfoilPolar
from auraflow.core.blade import Vehicle
from auraflow.core.frames import integrate_azimuth
from auraflow.core.medium import Medium

__all__ = ["cona_airloads", "rotor_section_state"]

_EPS = 1.0e-9


def _periodic_interp(psi_q: Array, psi_grid: Array, values_1d: Array) -> Array:
    """Periodic linear interpolation of a 1-D azimuth series ``values_1d[Npsi]``.

    ``psi_grid`` is a uniform grid on ``[0, 2 pi)``; the series is closed with a
    wrap-around node so the interpolation is smooth across ``2 pi``. ``psi_q``
    (any shape) is wrapped into ``[0, 2 pi)`` and interpolated; the result has
    the shape of ``psi_q``.
    """
    two_pi = 2.0 * jnp.pi
    grid_closed = jnp.concatenate([psi_grid, psi_grid[:1] + two_pi])
    vals_closed = jnp.concatenate([values_1d, values_1d[:1]])
    return jnp.interp(jnp.mod(psi_q, two_pi), grid_closed, vals_closed)


def rotor_section_state(
    vehicle: Vehicle,
    flight: FlightHistory,
    rotor_index: int,
    medium: Medium,
    collective: ArrayLike = 0.0,
    polar: Polar | None = None,
    gust: ArrayLike | None = None,
    include_induced: bool = True,
    include_unsteady: bool = True,
    n_wake_azimuth: int = 24,
    n_wake_rev: int = 4,
    n_inflow_psi: int = 36,
    r_c0: ArrayLike | None = None,
    wake_params: tuple[float, float, float] = (1.0, 0.0, 0.5),
) -> SectionState:
    r"""Blade-element :class:`SectionState` for one rotor over a flight history.

    Args:
        vehicle: The vehicle carrying the rotor (body-frame hub placements).
        flight: Upstream :class:`~auraflow.cona.flight.FlightHistory`.
        rotor_index: Which rotor (static int).
        medium: Ambient medium.
        collective: Collective pitch added to blade twist [rad]; scalar or
            ``[T]``.
        polar: Airfoil polar; defaults to :class:`ThinAirfoilPolar`.
        gust: Optional world-frame gust-velocity series [m/s], shape ``[T, 3]``
            (added to the free stream seen by the rotor).
        include_induced: Feed the prescribed-wake induced velocity into the
            triangle (set ``False`` for geometry-only inflow).
        include_unsteady: Apply the Wagner unsteady-aero lag + apparent mass.
        n_wake_azimuth: Wake-age segments per revolution (static int).
        n_wake_rev: Revolutions of wake age retained (static int).
        n_inflow_psi: Azimuth resolution of the cached disk-inflow grid.
        r_c0: Initial tip-vortex core radius [m]; defaults to ``0.15 * chord``.
        wake_params: Beddoes settling parameters ``(w0, ws, wc)``.

    Returns:
        A :class:`~auraflow.bemt.unsteady.SectionState` with ``[B, S, T]``
        aerodynamic leaves and ``[B, S, T, 3]`` world-frame source kinematics.
    """
    if polar is None:
        polar = ThinAirfoilPolar()
    rotor = vehicle.rotors[rotor_index]
    blade = rotor.blade
    spin = rotor.spin_direction
    n_blades = rotor.n_blades
    n_stations = blade.n_stations

    t = jnp.asarray(flight.t, dtype=float)
    n_t = t.shape[0]
    dt = (t[-1] - t[0]) / (n_t - 1)

    omega_mag = jnp.asarray(flight.rotor_speeds)[:, rotor_index]  # [T]
    signed_omega = spin * omega_mag  # [T]
    omega_dot = cast(Array, jnp.gradient(signed_omega, t))

    # Azimuths by integration; equally spaced blades.
    psi_ref = integrate_azimuth(t, signed_omega, 0.0)  # [T]
    offsets = spin * 2.0 * jnp.pi * jnp.arange(n_blades) / n_blades  # [B]
    psi = psi_ref[None, :] + offsets[:, None]  # [B, T]

    # --- Hub world kinematics from the 6-DOF flight ---------------------------
    hub_body = rotor.hub_position  # [3]
    r_wr = jnp.einsum("tij,jk->tik", flight.R, rotor.hub_orientation)  # world<-rotor [T,3,3]
    x_hub = flight.x + jnp.einsum("tij,j->ti", flight.R, hub_body)  # [T, 3]
    # v_hub = v_cg + R (Omega_body x hub_body).
    omega_cross = jnp.cross(flight.Omega_body, jnp.broadcast_to(hub_body, flight.Omega_body.shape))
    v_hub = flight.v + jnp.einsum("tij,tj->ti", flight.R, omega_cross)  # [T, 3]

    # Relative air velocity at the hub in the rotor frame.
    wind = (
        jnp.zeros((n_t, 3))
        if gust is None
        else jnp.broadcast_to(jnp.asarray(gust, dtype=float), (n_t, 3))
    )
    v_rel_world = wind - v_hub  # [T, 3]
    v_rf = jnp.einsum("tji,tj->ti", r_wr, v_rel_world)  # rotor-frame [T,3] (R^T @ v)

    vx_climb = -v_rf[:, 2]  # axial through-flow (+ = downwash sense) [T]

    r = blade.r  # [S]
    dr = blade.dr
    chord = blade.chord
    coll = jnp.broadcast_to(jnp.asarray(collective, dtype=float), (n_t,))
    theta = blade.twist[:, None] + coll[None, :]  # [S, T]

    # Broadcast to [B, S, T].
    psi_bst = psi[:, None, :]
    cospsi = jnp.cos(psi_bst)
    sinpsi = jnp.sin(psi_bst)
    r_bst = r[None, :, None]
    chord_bst = chord[None, :, None]
    theta_bst = theta[None, :, :]

    # In-plane free stream projected on the blade motion direction.
    v_ip_dot_motion = spin * (
        -v_rf[:, 0][None, None, :] * sinpsi + v_rf[:, 1][None, None, :] * cospsi
    )  # [B,1,T] -> broadcast
    u_t = omega_mag[None, None, :] * r_bst - v_ip_dot_motion  # [B,S,T]

    # --- Prescribed-wake induced axial velocity ------------------------------
    if include_induced:
        mean_omega = jnp.maximum(jnp.mean(omega_mag), _EPS)
        vtip = mean_omega * blade.radius
        mean_thrust = jnp.mean(jnp.asarray(flight.rotor_thrusts)[:, rotor_index])
        disk_area = jnp.pi * blade.radius**2
        ct = mean_thrust / (medium.rho0 * disk_area * vtip**2 + _EPS)
        mean_inplane = jnp.mean(jnp.hypot(v_rf[:, 0], v_rf[:, 1]))
        mean_axial = jnp.mean(vx_climb)
        mu_x = mean_inplane / (vtip + _EPS)
        mu_z = mean_axial / (vtip + _EPS)
        wake = make_prescribed_wake(
            ct,
            mean_omega,
            blade.radius,
            n_blades,
            medium,
            mu_x=mu_x,
            mu_z=mu_z,
            spin=spin,
            n_azimuth=n_wake_azimuth,
            n_rev=n_wake_rev,
            chord_ref=jnp.mean(chord),
            r_c0=r_c0,
            w0=wake_params[0],
            ws=wake_params[1],
            wc=wake_params[2],
        )
        psi_grid = jnp.linspace(0.0, 2.0 * jnp.pi, n_inflow_psi, endpoint=False)
        uz_grid = wake.inflow_grid(r, psi_grid)  # [S, Npsi], axial induced vel
        va_grid = -uz_grid  # downwash-positive [S, Npsi]

        # Index by blade azimuth: v_a[b,s,t] = interp(va_grid[s], psi[b,t]).
        def interp_station(vals_s: Array) -> Array:
            return _periodic_interp(psi, psi_grid, vals_s)  # [B,T]

        va = jax.vmap(interp_station)(va_grid)  # [S, B, T]
        va = jnp.transpose(va, (1, 0, 2))  # [B, S, T]
    else:
        va = jnp.zeros((n_blades, n_stations, n_t))

    u_a = jnp.broadcast_to(vx_climb[None, None, :], (n_blades, n_stations, n_t)) + va
    phi = jnp.arctan2(u_a, u_t)
    w = jnp.hypot(u_a, u_t)
    alpha_qs = theta_bst - phi  # quasi-steady angle of attack [B,S,T]
    mach = w / medium.c0
    reynolds = w * chord_bst / medium.nu

    # --- Unsteady aero (Wagner lag on the effective AoA + apparent mass) ------
    cl_alpha = jnp.asarray(getattr(polar, "cl_alpha", 2.0 * jnp.pi))
    if include_unsteady:
        bs = n_blades * n_stations
        w_flat = w.reshape(bs, n_t)
        a_flat = alpha_qs.reshape(bs, n_t)
        m_flat = mach.reshape(bs, n_t)
        c_flat = jnp.broadcast_to(chord[None, :], (n_blades, n_stations)).reshape(bs)

        def per_section(vv: Array, aa: Array, mm: Array, cc: Array) -> Array:
            return effective_aoa(vv, aa, dt, cc, mm)

        alpha_eff = jax.vmap(per_section)(w_flat, a_flat, m_flat, c_flat).reshape(
            n_blades, n_stations, n_t
        )
        v_dot = cast(Array, jnp.gradient(w, dt, axis=-1))
        lift_nc = 0.5 * medium.rho0 * cl_alpha * chord_bst**2 * v_dot * alpha_qs
    else:
        alpha_eff = alpha_qs
        lift_nc = jnp.zeros_like(w)

    cl, cd = polar(alpha_eff, mach, reynolds)
    q_dyn = 0.5 * medium.rho0 * w**2 * chord_bst
    lift_per_span = q_dyn * cl + lift_nc
    drag_per_span = q_dyn * cd
    cl_eff = lift_per_span / (q_dyn + _EPS)

    # Resolve into thrust-normal (fn) and in-plane (ft) per-span forces.
    cosphi = jnp.cos(phi)
    sinphi = jnp.sin(phi)
    fn = lift_per_span * cosphi - drag_per_span * sinphi
    ft = lift_per_span * sinphi + drag_per_span * cosphi

    # --- World-frame compact-source kinematics -------------------------------
    pos_x = r_bst * cospsi
    pos_y = r_bst * sinpsi
    zeros = jnp.zeros_like(pos_x)
    pos_rotor = jnp.stack([pos_x, pos_y, zeros], axis=-1)  # [B,S,T,3]

    so = signed_omega.reshape(1, 1, n_t)
    so_dot = omega_dot.reshape(1, 1, n_t)
    vel_rotor = jnp.stack([-so * pos_y, so * pos_x, zeros], axis=-1)
    acc_rotor = jnp.stack(
        [-so_dot * pos_y - so**2 * pos_x, so_dot * pos_x - so**2 * pos_y, zeros], axis=-1
    )

    ones_bst = jnp.ones_like(pos_x)
    motion_hat = jnp.stack([spin * -sinpsi * ones_bst, spin * cospsi * ones_bst, zeros], axis=-1)
    z_hat = jnp.array([0.0, 0.0, 1.0])
    force_rotor = ft[..., None] * motion_hat - fn[..., None] * z_hat

    def to_world_vec(v_rotor: Array) -> Array:
        # r_wr is [T,3,3]; v_rotor is [B,S,T,3].
        return jnp.einsum("tij,bstj->bsti", r_wr, v_rotor)

    position = to_world_vec(pos_rotor) + x_hub[None, None, :, :]
    velocity = to_world_vec(vel_rotor) + v_hub[None, None, :, :]
    acceleration = to_world_vec(acc_rotor)
    force_on_fluid = to_world_vec(force_rotor)

    return SectionState(
        r=r,
        dr=dr,
        chord=chord,
        psi=psi,
        phi=phi,
        alpha=alpha_eff,
        w=w,
        reynolds=reynolds,
        mach=mach,
        cl=cl_eff,
        cd=cd,
        v_axial=va,
        v_swirl=jnp.zeros_like(va),
        lift_per_span=lift_per_span,
        drag_per_span=drag_per_span,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        force_on_fluid=force_on_fluid,
    )


def cona_airloads(
    vehicle: Vehicle,
    flight: FlightHistory,
    medium: Medium,
    collective: ArrayLike = 0.0,
    polar: Polar | None = None,
    gust: ArrayLike | None = None,
    **kwargs: object,
) -> tuple[SectionState, ...]:
    """Per-rotor :class:`SectionState` for every rotor of a vehicle.

    Thin loop over :func:`rotor_section_state` (the rotor count is static, so
    the Python loop is fine and each call is independently jittable).

    Args:
        vehicle: The vehicle.
        flight: The flight history.
        medium: Ambient medium.
        collective: Collective pitch [rad]; scalar, ``[T]``, or ``[Nr]`` (one
            per rotor).
        polar: Airfoil polar (shared across rotors).
        gust: Optional world-frame gust series [m/s], shape ``[T, 3]``.
        **kwargs: Forwarded to :func:`rotor_section_state` (wake / unsteady
            options).

    Returns:
        A tuple of :class:`SectionState`, one per rotor.
    """
    coll = jnp.asarray(collective, dtype=float)
    states = []
    for i in range(vehicle.n_rotors):
        coll_i = coll[i] if coll.ndim == 1 and coll.shape[0] == vehicle.n_rotors else coll
        states.append(
            rotor_section_state(
                vehicle,
                flight,
                i,
                medium,
                collective=coll_i,
                polar=polar,
                gust=gust,
                **kwargs,  # type: ignore[arg-type]
            )
        )
    return tuple(states)
