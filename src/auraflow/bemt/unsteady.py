r"""Quasi-steady time marching of the BEMT state around the azimuth.

Given a rotor, an ambient medium and a rotor-speed history ``Omega(t)`` (plus an
optional free-stream / hub translation), :func:`march_bemt` evaluates the full
blade-element state -- local velocity triangle *including the induced velocity*,
angle of attack, section Mach/Reynolds, sectional lift/drag and the world-frame
compact-source kinematics -- at every ``(blade, station, time)``.

This is the quasi-steady kernel that fixes the three predecessor bugs
(``docs/research/fwh-rotor-sim-audit.md``):

- **(b) time-varying speed**: forces use the instantaneous ``Omega(t)`` at each
  time sample, not a frozen ``mean(Omega)``;
- **(c) azimuth**: blade azimuths are the integral ``psi = int Omega dt`` via
  :func:`auraflow.core.frames.integrate_azimuth`, not ``Omega(t) t``;
- **(a) induced velocity**: the per-annulus BEMT induced axial/swirl velocity
  (or, with ``include_induced=False``, exactly zero) enters the velocity
  triangle and hence the angle of attack and the loads that drive the acoustics.

Frames follow ``docs/architecture.md``: rotor-frame ``+z`` is the thrust axis,
azimuth ``psi`` from ``+x`` toward ``+y``; ``rotor.hub_orientation`` maps
rotor-frame vectors to the parent (world) frame. The state is returned as a
single :class:`SectionState` PyTree so both the tonal acoustics
(:mod:`auraflow.bemt.acoustics`) and a later BPM broadband model can consume it.

SI units throughout; float64-safe and differentiable (the inflow solve uses
:func:`jax.lax.custom_root`).
"""

from typing import cast

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from auraflow.bemt.solver import Polar, prandtl_tip_root_loss, solve_inflow_angle
from auraflow.core.airfoil import ThinAirfoilPolar
from auraflow.core.blade import Rotor
from auraflow.core.frames import integrate_azimuth
from auraflow.core.medium import Medium

__all__ = ["SectionState", "march_bemt"]


class SectionState(eqx.Module):
    """Time-resolved blade-element state and compact-source kinematics.

    Aerodynamic leaves have shape ``[B, S, T]`` (blades, stations, times);
    vector kinematic/force leaves have shape ``[B, S, T, 3]``; per-station
    geometry is ``[S]`` and azimuth is ``[B, T]``.

    This is the clean hand-off PyTree between the BEMT aerodynamics and any
    acoustic model. The tonal coupling (:mod:`auraflow.bemt.acoustics`) turns
    ``(position, velocity, acceleration, force_on_fluid)`` directly into
    Farassat-1A compact loading sources; a broadband (BPM) model would instead
    consume ``(w, alpha, reynolds, mach, chord, r)``.

    Attributes:
        r: Station radii [m], shape ``[S]``.
        dr: Trapezoid-consistent station widths [m], shape ``[S]``.
        chord: Section chord [m], shape ``[S]``.
        psi: Blade azimuths ``psi_b(t)`` [rad], shape ``[B, T]`` (unwrapped).
        phi: Local inflow angle [rad], shape ``[B, S, T]``.
        alpha: Angle of attack ``theta - phi`` [rad], shape ``[B, S, T]``.
        w: Resultant section speed [m/s], shape ``[B, S, T]``.
        reynolds: Section Reynolds number [-], shape ``[B, S, T]``.
        mach: Section Mach number [-], shape ``[B, S, T]``.
        cl: Section lift coefficient [-], shape ``[B, S, T]``.
        cd: Section drag coefficient [-], shape ``[B, S, T]``.
        v_axial: Axial induced velocity [m/s], shape ``[B, S, T]``.
        v_swirl: Swirl induced velocity [m/s], shape ``[B, S, T]``.
        lift_per_span: Sectional lift per unit span [N/m], shape ``[B, S, T]``.
        drag_per_span: Sectional drag per unit span [N/m], shape ``[B, S, T]``.
        position: Quarter-chord world positions [m], shape ``[B, S, T, 3]``.
        velocity: Source world velocity [m/s], shape ``[B, S, T, 3]``.
        acceleration: Source world acceleration [m/s^2], shape ``[B, S, T, 3]``.
        force_on_fluid: World-frame force on the fluid **per unit span** [N/m],
            shape ``[B, S, T, 3]`` (reaction to the section aero force; multiply
            by ``dr`` to obtain the compact segment force).
    """

    r: Array
    dr: Array
    chord: Array
    psi: Array
    phi: Array
    alpha: Array
    w: Array
    reynolds: Array
    mach: Array
    cl: Array
    cd: Array
    v_axial: Array
    v_swirl: Array
    lift_per_span: Array
    drag_per_span: Array
    position: Array
    velocity: Array
    acceleration: Array
    force_on_fluid: Array


def march_bemt(
    rotor: Rotor,
    medium: Medium,
    t: ArrayLike,
    omega: ArrayLike,
    collective: ArrayLike = 0.0,
    v_inf: ArrayLike | None = None,
    hub_velocity: ArrayLike | None = None,
    polar: Polar | None = None,
    include_induced: bool = True,
    tip_loss: bool = True,
    root_loss: bool = True,
) -> SectionState:
    r"""March the quasi-steady BEMT state over a rotor revolution history.

    At each time the local velocity triangle uses the instantaneous rotor speed
    and (optionally) the per-annulus BEMT induced velocity. The in-plane free
    stream modulates the tangential speed with azimuth (advancing/retreating
    asymmetry); the disk-normal free stream / climb sets the axial through-flow.

    Args:
        rotor: Rotor with hub placement in the parent (world) frame.
        medium: Ambient medium.
        t: Uniformly spaced time grid [s], shape ``[T]``.
        omega: Rotor angular-rate **magnitude** ``|Omega|`` [rad/s]; scalar or
            shape ``[T]``. The signed rate is ``spin_direction * |Omega|``.
        collective: Collective pitch added to the blade twist [rad]; scalar or
            shape ``[T]``.
        v_inf: Free-stream (wind) velocity in the world frame [m/s], shape
            ``[3]`` (constant). Forward flight can be modelled either here (as a
            wind on a fixed hub) or via ``hub_velocity`` (a translating hub).
        hub_velocity: Constant hub translation velocity in the world frame
            [m/s], shape ``[3]``; the hub position is ``hub0 + hub_velocity t``.
        polar: Airfoil polar; defaults to :class:`ThinAirfoilPolar`.
        include_induced: If ``True`` (default) solve the per-annulus BEMT
            induced axial/swirl velocity and feed it into the triangle; if
            ``False`` set the induced velocity to zero (geometry-only inflow).
        tip_loss: Apply the Prandtl tip loss in the induced-velocity solve.
        root_loss: Apply the Prandtl root loss in the induced-velocity solve.

    Returns:
        A :class:`SectionState` with ``[B, S, T]`` aerodynamic leaves and
        ``[B, S, T, 3]`` source kinematics/force leaves.
    """
    if polar is None:
        polar = ThinAirfoilPolar()
    blade = rotor.blade
    spin = rotor.spin_direction
    t = jnp.asarray(t, dtype=float)
    n_t = t.shape[0]
    omega_t = jnp.broadcast_to(jnp.asarray(omega, dtype=float), (n_t,))
    signed_omega = spin * omega_t
    omega_dot = cast(Array, jnp.gradient(signed_omega, t))

    # Azimuth by integration (bug (c) fix), with equally spaced blades.
    psi_ref = integrate_azimuth(t, signed_omega, 0.0)  # [T]
    offsets = spin * 2.0 * jnp.pi * jnp.arange(rotor.n_blades) / rotor.n_blades  # [B]
    psi = psi_ref[None, :] + offsets[:, None]  # [B, T]

    r = blade.r  # [S]
    dr = blade.dr
    chord = blade.chord
    coll = jnp.asarray(collective, dtype=float)
    theta = blade.twist[:, None] + jnp.broadcast_to(coll, (n_t,))[None, :]  # [S, T]

    # Broadcast to [B, S, T].
    psi_bst = psi[:, None, :]  # [B, 1, T]
    r_bst = r[None, :, None]
    chord_bst = chord[None, :, None]
    theta_bst = theta[None, :, :]
    cospsi = jnp.cos(psi_bst)
    sinpsi = jnp.sin(psi_bst)

    # Free stream expressed in the rotor frame: R_hub^T (v_inf - hub_velocity).
    hub_vel = jnp.zeros(3) if hub_velocity is None else jnp.asarray(hub_velocity, dtype=float)
    v_inf_vec = jnp.zeros(3) if v_inf is None else jnp.asarray(v_inf, dtype=float)
    r_hub = rotor.hub_orientation
    v_rel = v_inf_vec - hub_vel
    v_rf = r_hub.T @ v_rel  # [3]
    vx_climb = -v_rf[2]  # axial through-flow (+ = same sense as downwash/climb)
    # In-plane free stream projected on the blade motion direction
    # t_hat = spin * (-sin psi, cos psi, 0):  v_ip . t_hat = spin(-vx sin + vy cos).
    v_ip_dot_motion = spin * (-v_rf[0] * sinpsi + v_rf[1] * cospsi)  # [B, 1, T]

    v_x = jnp.broadcast_to(vx_climb, (rotor.n_blades, blade.n_stations, n_t))
    u_t = omega_t[None, None, :] * r_bst - v_ip_dot_motion  # tangential inflow [B,S,T]
    u_t = jnp.broadcast_to(u_t, v_x.shape)

    if include_induced:
        phi = solve_inflow_angle(
            r_bst * jnp.ones_like(v_x),
            theta_bst * jnp.ones_like(v_x),
            chord_bst * jnp.ones_like(v_x),
            v_x,
            u_t,
            blade.radius,
            blade.hub_radius,
            rotor.n_blades,
            polar,
            medium,
            tip_loss=tip_loss,
            root_loss=root_loss,
        )
        sinphi = jnp.sin(phi)
        cosphi = jnp.cos(phi)
        tphi = jnp.tan(phi)
        loss = prandtl_tip_root_loss(
            r_bst,
            blade.radius,
            blade.hub_radius,
            sinphi,
            rotor.n_blades,
            tip=tip_loss,
            root=root_loss,
        )
        sigma = rotor.n_blades * chord_bst / (2.0 * jnp.pi * r_bst)
        w0 = jnp.maximum(jnp.hypot(v_x, u_t), jnp.abs(u_t))
        cl0, cd0 = polar(theta_bst - phi, w0 / medium.c0, w0 * chord_bst / medium.nu)
        ct_c0 = cl0 * sinphi + cd0 * cosphi
        kt = sigma * ct_c0 / (4.0 * loss * sinphi**2)
        u_a = u_t * tphi / (1.0 + kt * tphi)
        v_axial = u_a - v_x
        v_swirl = u_a * kt
        u_t_ind = u_t - v_swirl
    else:
        u_a = v_x
        phi = jnp.arctan2(u_a, u_t)
        v_axial = jnp.zeros_like(u_a)
        v_swirl = jnp.zeros_like(u_a)
        u_t_ind = u_t

    w = jnp.hypot(u_a, u_t_ind)
    alpha = theta_bst - phi
    mach = w / medium.c0
    reynolds = w * chord_bst / medium.nu
    cl, cd = polar(alpha, mach, reynolds)

    q_dyn = 0.5 * medium.rho0 * w**2 * chord_bst  # [B,S,T]
    lift_per_span = q_dyn * cl
    drag_per_span = q_dyn * cd
    fn = q_dyn * (cl * jnp.cos(phi) - cd * jnp.sin(phi))  # thrust dir, per span
    ft = q_dyn * (cl * jnp.sin(phi) + cd * jnp.cos(phi))  # in-plane, per span

    # --- Source kinematics (rigid rotation about a translating hub) ----------
    pos_x = r_bst * cospsi
    pos_y = r_bst * sinpsi
    zeros = jnp.zeros_like(pos_x)
    pos_rotor = jnp.stack([pos_x, pos_y, zeros], axis=-1)  # [B,S,T,3]

    so = signed_omega.reshape(1, 1, n_t)  # [1,1,T]
    so_dot = omega_dot.reshape(1, 1, n_t)
    vel_rotor = jnp.stack([-so * pos_y, so * pos_x, zeros], axis=-1)
    acc_rotor = jnp.stack(
        [-so_dot * pos_y - so**2 * pos_x, so_dot * pos_x - so**2 * pos_y, zeros], axis=-1
    )

    # Force on the fluid = reaction to blade aero force:
    #   +ft along the blade motion direction, -fn along the thrust axis (+z).
    ones_bst = jnp.ones_like(pos_x)
    motion_hat = jnp.stack([spin * -sinpsi * ones_bst, spin * cospsi * ones_bst, zeros], axis=-1)
    z_hat = jnp.array([0.0, 0.0, 1.0])
    force_rotor = ft[..., None] * motion_hat - fn[..., None] * z_hat

    hub0 = rotor.hub_position
    hub_pos_t = hub0[None, :] + hub_vel[None, :] * t[:, None]  # [T, 3]

    def to_world_vec(v_rotor: Array) -> Array:
        return jnp.einsum("ij,bstj->bsti", r_hub, v_rotor)

    position = jnp.einsum("ij,bstj->bsti", r_hub, pos_rotor) + hub_pos_t[None, None, :, :]
    velocity = to_world_vec(vel_rotor) + hub_vel
    acceleration = to_world_vec(acc_rotor)
    force_on_fluid = to_world_vec(force_rotor)

    return SectionState(
        r=r,
        dr=dr,
        chord=chord,
        psi=psi,
        phi=phi,
        alpha=alpha,
        w=w,
        reynolds=reynolds,
        mach=mach,
        cl=cl,
        cd=cd,
        v_axial=v_axial,
        v_swirl=v_swirl,
        lift_per_span=lift_per_span,
        drag_per_span=drag_per_span,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        force_on_fluid=force_on_fluid,
    )
