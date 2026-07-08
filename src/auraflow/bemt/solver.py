r"""Steady per-annulus blade-element momentum theory (BEMT).

Differentiable, hover-capable BEMT solved in a single unknown per annulus --
the local inflow angle ``phi`` -- following the CCBlade/Ning (2014) residual
approach, but written in the *induced-velocity* (not induction-factor) form so
that it is well conditioned across the whole operating envelope **including
hover** (``V_x -> 0``), where the classic ``a = v_a / V_x`` normalization is
singular.

Velocity triangle and momentum-balance
--------------------------------------
For an annulus at radius ``r`` the section sees an axial through-flow ``V_x``
(climb + disk-normal free stream) and a tangential speed ``V_y = Omega r``.
With axial and swirl induced velocities ``v_a`` and ``v_t`` the resultant seen
by the blade has components

    U_a = V_x + v_a      (axial, along the thrust axis +z),
    U_t = V_y - v_t      (in the rotor plane, opposing rotation),
    tan(phi) = U_a / U_t,   W = sqrt(U_a^2 + U_t^2),

and the section angle of attack is ``alpha = theta - phi`` with ``theta`` the
local geometric pitch (collective + twist). The sectional force coefficients
resolved into the thrust and in-plane directions are (propeller/rotor sign
convention -- drag *reduces* thrust and *opposes* rotation)

    c_n = C_l cos(phi) - C_d sin(phi)   (thrust direction),
    c_t = C_l sin(phi) + C_d cos(phi)   (in-plane / torque direction).

Combining annular momentum theory (with Prandtl loss ``F``) and blade-element
theory gives, with local solidity ``sigma' = B c / (2 pi r)``,

    v_a = U_a k_n,   v_t = U_a k_t,
    k_n = sigma' c_n / (4 F sin^2 phi),   k_t = sigma' c_t / (4 F sin^2 phi).

Eliminating ``U_a`` between the axial relation ``U_a (1 - k_n) = V_x`` and the
triangle ``U_a = V_y tan(phi) / (1 + k_t tan(phi))`` yields the scalar residual

    R(phi) = V_y tan(phi) (1 - k_n) - V_x (1 + k_t tan(phi)) = 0.       (*)

In hover (``V_x = 0``) this reduces to ``k_n = 1`` -- the standard combined
blade-element/momentum hover condition -- with no division by ``V_x`` anywhere,
so the solve stays finite. ``R`` runs from ``-inf`` (as ``phi -> 0+``,
``k_n -> +inf``) to ``+inf`` (as ``phi -> pi/2``) for a thrusting section, so a
bracketed bisection on ``phi in (0, pi/2)`` has exactly one sign change.

Differentiability
-----------------
The root of (*) is found with :func:`jax.lax.custom_root`, so gradients w.r.t.
any closed-over parameter (collective pitch, chord/twist, Omega, medium, polar
coefficients ...) flow by the implicit function theorem rather than through the
bisection iterations. The residual is elementwise over stations, so a single
vector ``custom_root`` handles the whole blade with a diagonal tangent solve.

Units are SI throughout (m, s, rad, kg, N). Shapes: per-station quantities are
``[S]``; scalars are 0-d arrays.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from auraflow.core.airfoil import ThinAirfoilPolar
from auraflow.core.blade import Rotor
from auraflow.core.medium import Medium

__all__ = [
    "AnnulusState",
    "RotorLoads",
    "prandtl_tip_root_loss",
    "solve_inflow_angle",
    "steady_bemt",
]

Polar = Callable[..., tuple[Array, Array]]

_PHI_LO = 1.0e-6
_PHI_HI = 0.5 * jnp.pi - 1.0e-6
_F_FLOOR = 1.0e-4  # keep the loss factor away from 0 so k_n, k_t stay finite


class AnnulusState(eqx.Module):
    """Per-station steady BEMT solution (all leaves shape ``[S]``).

    Attributes:
        r: Station radii [m].
        phi: Converged local inflow angle [rad].
        alpha: Section angle of attack ``theta - phi`` [rad].
        v_axial: Axial induced velocity ``v_a`` [m/s].
        v_swirl: Tangential (swirl) induced velocity ``v_t`` [m/s].
        inflow_axial: Total axial velocity at the disk ``U_a = V_x + v_a`` [m/s].
        inflow_tangential: In-plane velocity ``U_t = V_y - v_t`` [m/s].
        w: Resultant section speed ``W`` [m/s].
        reynolds: Section Reynolds number ``W c / nu`` [-].
        mach: Section Mach number ``W / c0`` [-].
        cl: Section lift coefficient [-].
        cd: Section drag coefficient [-].
        loss: Prandtl tip/root loss factor ``F`` [-].
        lift_per_span: Sectional lift per unit span [N/m].
        drag_per_span: Sectional drag per unit span [N/m].
        fn_per_span: Thrust-direction force per unit span (one blade) [N/m].
        ft_per_span: In-plane force per unit span (one blade) [N/m].
    """

    r: Array
    phi: Array
    alpha: Array
    v_axial: Array
    v_swirl: Array
    inflow_axial: Array
    inflow_tangential: Array
    w: Array
    reynolds: Array
    mach: Array
    cl: Array
    cd: Array
    loss: Array
    lift_per_span: Array
    drag_per_span: Array
    fn_per_span: Array
    ft_per_span: Array


class RotorLoads(eqx.Module):
    """Rotor-integrated steady loads and coefficients (scalars).

    Coefficients use the rotor (helicopter) normalization with tip speed
    ``Omega R`` and disk area ``A = pi R^2``:

        C_T = T / (rho0 A (Omega R)^2),
        C_Q = Q / (rho0 A (Omega R)^2 R),
        C_P = P / (rho0 A (Omega R)^3) = C_Q.

    Attributes:
        thrust: Rotor thrust [N] (sum over ``B`` blades).
        torque: Rotor torque [N m].
        power: Rotor power ``Omega Q`` [W].
        ct: Thrust coefficient [-].
        cq: Torque coefficient [-].
        cp: Power coefficient [-].
        figure_of_merit: Hover figure of merit ``C_T^{3/2} / (sqrt(2) C_P)`` [-].
        annulus: The per-station :class:`AnnulusState`.
    """

    thrust: Array
    torque: Array
    power: Array
    ct: Array
    cq: Array
    cp: Array
    figure_of_merit: Array
    annulus: AnnulusState


def prandtl_tip_root_loss(
    r: ArrayLike,
    radius: ArrayLike,
    hub_radius: ArrayLike,
    sinphi: ArrayLike,
    n_blades: int,
    tip: bool = True,
    root: bool = True,
) -> Array:
    r"""Prandtl tip and root loss factor ``F`` [-].

    ``F = F_tip * F_root`` with, for ``B`` blades,

        f_tip  = (B/2) (R - r) / (r |sin phi|),  F_tip  = (2/pi) acos(exp(-f_tip)),
        f_root = (B/2) (r - R_hub) / (r |sin phi|), F_root = (2/pi) acos(exp(-f_root)).

    ``F_tip -> 1`` well inboard (``f`` large, ``exp(-f) -> 0``, ``acos(0)=pi/2``)
    and ``F_tip -> 0`` at the tip (``r -> R``, ``f -> 0``, ``acos(1)=0``).

    Smooth-safe guards (documented gradient behaviour):

    - ``|sin phi|`` is floored by a tiny epsilon so the division is finite;
    - the exponent argument ``f`` is floored so ``exp(-f) <= 1`` and clipped
      strictly below 1, avoiding the infinite slope of ``acos`` at 1 (a
      negligible dead zone confined to the last epsilon at the very tip/root);
    - the product is floored at :data:`_F_FLOOR` so the momentum factors
      ``k_n, k_t`` (which divide by ``F``) stay finite at the extreme stations.

    Args:
        r: Station radii [m], shape ``[...]``.
        radius: Tip radius ``R`` [m], scalar.
        hub_radius: Root cutout radius ``R_hub`` [m], scalar.
        sinphi: ``sin(phi)`` at each station, shape ``[...]``.
        n_blades: Number of blades ``B`` (static int).
        tip: Include the tip loss factor.
        root: Include the root loss factor.

    Returns:
        Loss factor ``F``, shape ``[...]`` (broadcast of ``r`` and ``sinphi``).
    """
    r = jnp.asarray(r)
    sphi = jnp.abs(jnp.asarray(sinphi)) + 1.0e-12
    half_b = 0.5 * n_blades
    f = jnp.ones_like(r * sphi)

    def _factor(argument: ArrayLike) -> Array:
        arg = jnp.maximum(argument, 1.0e-10)
        return (2.0 / jnp.pi) * jnp.arccos(jnp.clip(jnp.exp(-arg), 0.0, 1.0 - 1.0e-12))

    if tip:
        f = f * _factor(half_b * (radius - r) / (r * sphi))
    if root:
        f = f * _factor(half_b * (r - hub_radius) / (r * sphi))
    return jnp.clip(f, _F_FLOOR, 1.0)


def solve_inflow_angle(
    r: Array,
    theta: Array,
    chord: Array,
    v_x: Array,
    v_y: Array,
    radius: Array,
    hub_radius: Array,
    n_blades: int,
    polar: Polar,
    medium: Medium,
    tip_loss: bool = True,
    root_loss: bool = True,
    n_bisect: int = 60,
) -> Array:
    """Solve the CCBlade residual (*) for the inflow angle at every station.

    A bracketed bisection on ``phi in (0, pi/2)`` (guaranteed sign change for a
    thrusting section) is wrapped in :func:`jax.lax.custom_root`, so the returned
    ``phi`` is differentiable w.r.t. all closed-over inputs by implicit
    differentiation (diagonal tangent solve, since the residual is elementwise).

    Args:
        r: Station radii [m], shape ``[S]``.
        theta: Local geometric pitch (collective + twist) [rad], shape ``[S]``.
        chord: Section chord [m], shape ``[S]``.
        v_x: Axial through-flow (climb + disk-normal free stream) [m/s], ``[S]``.
        v_y: Tangential speed ``Omega r`` [m/s], shape ``[S]``.
        radius: Tip radius [m], scalar.
        hub_radius: Root cutout radius [m], scalar.
        n_blades: Number of blades (static int).
        polar: Airfoil polar ``(alpha, mach, reynolds) -> (cl, cd)``.
        medium: Ambient medium.
        tip_loss: Apply the Prandtl tip loss.
        root_loss: Apply the Prandtl root loss.
        n_bisect: Bisection iterations (60 -> ~1e-18 bracket in float64).

    Returns:
        Converged inflow angle ``phi`` [rad], shape ``[S]``.
    """

    def residual(phi: Array) -> Array:
        sinphi = jnp.sin(phi)
        alpha = theta - phi
        w_est = jnp.maximum(jnp.hypot(v_x, v_y), jnp.abs(v_y))
        cl, cd = polar(alpha, w_est / medium.c0, w_est * chord / medium.nu)
        cn = cl * jnp.cos(phi) - cd * sinphi
        ct = cl * sinphi + cd * jnp.cos(phi)
        loss = prandtl_tip_root_loss(
            r, radius, hub_radius, sinphi, n_blades, tip=tip_loss, root=root_loss
        )
        sigma = n_blades * chord / (2.0 * jnp.pi * r)
        denom = 4.0 * loss * sinphi**2
        kn = sigma * cn / denom
        kt = sigma * ct / denom
        tphi = jnp.tan(phi)
        return v_y * tphi * (1.0 - kn) - v_x * (1.0 + kt * tphi)

    def solve(res: Callable[[Array], Array], _guess: Array) -> Array:
        lo = jnp.full_like(r, _PHI_LO)
        hi = jnp.full_like(r, _PHI_HI)
        f_lo = res(lo)

        def body(_i: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
            lo, hi = state
            mid = 0.5 * (lo + hi)
            f_mid = res(mid)
            same = jnp.sign(f_mid) == jnp.sign(f_lo)
            return jnp.where(same, mid, lo), jnp.where(same, hi, mid)

        lo, hi = jax.lax.fori_loop(0, n_bisect, body, (lo, hi))
        return 0.5 * (lo + hi)

    def tangent_solve(g: Callable[[Array], Array], y: Array) -> Array:
        return y / g(jnp.ones_like(y))

    guess = jnp.full_like(r, 0.05)
    return jax.lax.custom_root(residual, guess, solve, tangent_solve)


def steady_bemt(
    rotor: Rotor,
    medium: Medium,
    omega: ArrayLike,
    v_climb: ArrayLike = 0.0,
    collective: ArrayLike = 0.0,
    polar: Polar | None = None,
    tip_loss: bool = True,
    root_loss: bool = True,
) -> RotorLoads:
    r"""Steady axial-flight / hover BEMT for a whole rotor.

    Solves the per-annulus inflow angle (:func:`solve_inflow_angle`), rebuilds
    the velocity triangle and sectional loads, and integrates the rotor thrust,
    torque and power with the blade's trapezoid-consistent station widths
    ``dr`` (:attr:`~auraflow.core.blade.BladeGeometry.dr`).

    Args:
        rotor: The rotor (blade geometry + blade count).
        medium: Ambient medium.
        omega: Rotor angular rate magnitude ``|Omega|`` [rad/s], scalar.
        v_climb: Axial climb velocity [m/s], scalar (0 = hover; >0 climb,
            <0 descent -- descent may leave the momentum-theory validity range).
        collective: Collective pitch added to the blade twist [rad], scalar.
        polar: Airfoil polar; defaults to a thin-airfoil ``2*pi`` slope with
            small profile drag.
        tip_loss: Apply the Prandtl tip loss.
        root_loss: Apply the Prandtl root loss.

    Returns:
        A :class:`RotorLoads` with integrated coefficients and the per-station
        :class:`AnnulusState`.
    """
    blade = rotor.blade
    if polar is None:
        polar = ThinAirfoilPolar()
    omega = jnp.asarray(omega)
    r = blade.r
    theta = blade.twist + jnp.asarray(collective)
    chord = blade.chord
    v_x = jnp.broadcast_to(jnp.asarray(v_climb, dtype=r.dtype), r.shape)
    v_y = omega * r

    phi = solve_inflow_angle(
        r,
        theta,
        chord,
        v_x,
        v_y,
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
        r,
        blade.radius,
        blade.hub_radius,
        sinphi,
        rotor.n_blades,
        tip=tip_loss,
        root=root_loss,
    )
    sigma = rotor.n_blades * chord / (2.0 * jnp.pi * r)

    # First pass with a geometric speed estimate to get k_t, then recover the
    # momentum-consistent triangle. Recompute coefficients at the true W.
    w0 = jnp.maximum(jnp.hypot(v_x, v_y), jnp.abs(v_y))
    cl0, cd0 = polar(theta - phi, w0 / medium.c0, w0 * chord / medium.nu)
    ct_coeff0 = cl0 * sinphi + cd0 * cosphi
    kt = sigma * ct_coeff0 / (4.0 * loss * sinphi**2)

    u_a = v_y * tphi / (1.0 + kt * tphi)
    v_axial = u_a - v_x
    v_swirl = u_a * kt
    u_t = v_y - v_swirl
    w = jnp.hypot(u_a, u_t)

    alpha = theta - phi
    mach = w / medium.c0
    reynolds = w * chord / medium.nu
    cl, cd = polar(alpha, mach, reynolds)

    q_dyn = 0.5 * medium.rho0 * w**2 * chord
    lift_per_span = q_dyn * cl
    drag_per_span = q_dyn * cd
    fn = q_dyn * (cl * cosphi - cd * sinphi)  # thrust direction, one blade
    ft = q_dyn * (cl * sinphi + cd * cosphi)  # in-plane, one blade

    dr = blade.dr
    thrust = rotor.n_blades * jnp.sum(fn * dr)
    torque = rotor.n_blades * jnp.sum(ft * r * dr)
    power = omega * torque

    area = jnp.pi * blade.radius**2
    vtip = omega * blade.radius
    ct = thrust / (medium.rho0 * area * vtip**2)
    cq = torque / (medium.rho0 * area * vtip**2 * blade.radius)
    cp = power / (medium.rho0 * area * vtip**3)
    fom = jnp.clip(ct, 0.0, None) ** 1.5 / (jnp.sqrt(2.0) * jnp.abs(cp) + 1.0e-30)

    annulus = AnnulusState(
        r=r,
        phi=phi,
        alpha=alpha,
        v_axial=v_axial,
        v_swirl=v_swirl,
        inflow_axial=u_a,
        inflow_tangential=u_t,
        w=w,
        reynolds=reynolds,
        mach=mach,
        cl=cl,
        cd=cd,
        loss=loss,
        lift_per_span=lift_per_span,
        drag_per_span=drag_per_span,
        fn_per_span=fn,
        ft_per_span=ft,
    )
    return RotorLoads(
        thrust=thrust,
        torque=torque,
        power=power,
        ct=ct,
        cq=cq,
        cp=cp,
        figure_of_merit=fom,
        annulus=annulus,
    )
