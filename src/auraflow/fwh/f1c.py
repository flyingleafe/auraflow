"""FW-H Formulation 1C: convective (uniform mean flow) Farassat-type solution.

Najafi-Yazdi, Bres & Mongeau (2011), recovered verbatim in
``docs/research/cona-external-formulations.md`` §1 (McGill dissertation eqs.
3.45 thickness / 3.49 loading). Uniform mean flow ``U0 = M0 c0`` along ``+x1``,
``beta^2 = 1 - M0^2``. The convective Green's function delay is ``tau_e = t -
R/c0`` with the phase distance ``R`` and amplitude distance ``R*`` from
:func:`~auraflow.fwh.geometry.convective_radiation`; the (non-unit) radiation
vectors ``R~_i`` and ``R~*_i`` and the Doppler factor ``1 - M_R`` with
``M_R = v_i R~_i / c0`` complete the geometry.

Auxiliary source-time derivatives (only ``y(tau)`` moves, ``dy/dtau = v``):
``dR*/dtau = -v_i R~*_i``, ``dM_R/dtau = (dv_i/dtau R~_i + v_i dR~_i/dtau)/c0``.
The geometric derivatives ``dR~_i/dtau`` (and ``dR*/dtau``) are obtained by a
single forward-mode :func:`jax.jvp` of :func:`convective_radiation` along the
tangent ``v`` -- exact and consistent with the analytic formulas. Time
derivatives of the *source* quantities ``Q_n``, ``L_i`` use the same 2nd-order
central differences as :mod:`auraflow.fwh.f1a`, and the acceleration ``a`` is
supplied by the caller. All outputs are differentiable.

Thickness (eq. 3.45, ``S = Q_i n_i`` the scalar monopole strength, overdot the
full source-time derivative of the bracketed scalar):

    4 pi p'_T = int Sdot/(R* D^2)
              - int (dR*/dtau) S/(R*^2 D^2)
              + int S (dM_R/dtau)/(R* D^3)
              - M0 int (dR~1/dtau S + R~1 Sdot)/(R* D^2)
              + M0 int (dR*/dtau) R~1 S/(R*^2 D^2)
              - M0 int (dM_R/dtau) R~1 S/(R* D^3)
              - U0 int R~*1 S/(R*^2 D)

Loading (eq. 3.49, ``L_i = L_ij n_j`` the loading vector, ``D = 1 - M_R``):

    4 pi p'_L = (1/c0) int d/dtau(L_i R~_i)/(R* D^2)
              - (1/c0) int (dR*/dtau)(L_i R~_i)/(R*^2 D^2)
              + (1/c0) int (dM_R/dtau)(L_i R~_i)/(R* D^3)
              + int (L_i R~*_i)/(R*^2 D)

At ``M0 = 0`` (``R = R* = |x-y|``, ``R~ = R~* = rhat``) these collapse
term-by-term to Farassat 1A -- the ``f1c``->``f1a`` cross-check. For a static
source and observer (wind tunnel) the kinematic derivatives vanish and the
cheaper :func:`f1c_windtunnel` fast path applies.

Shapes match :mod:`auraflow.fwh.f1a`: observers ``[O, 3]``; per-source
histories ``[S, T]`` / ``[S, T, 3]`` on ``tau`` ``[T]``; output ``[O, T_obs]``.
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from auraflow.core.medium import Medium
from auraflow.fwh.geometry import (
    convective_radiation,
    resample_sum,
    source_time_derivative,
)

__all__ = [
    "f1c_pressure",
    "f1c_windtunnel",
]

_FOUR_PI = 4.0 * jnp.pi


def _f1c_integrands(
    x: Array,
    y: Array,
    v: Array,
    a: Array,
    qn: Array,
    load: Array,
    dtau: Array,
    c0: Array,
    mach0: Array,
) -> tuple[Array, Array, Array]:
    """Per-source, single-observer 1C integrands (thickness, loading).

    Returns ``(R, pT_integrand, pL_integrand)`` each shape ``[S, T]``; ``R`` is
    the phase distance used to form arrival times.
    """
    u0 = mach0 * c0

    def geom(yy: Array) -> tuple[Array, Array, Array, Array]:
        return convective_radiation(x, yy, mach0)

    # Forward-mode derivative along dy/dtau = v gives the source-time
    # derivatives d(.)/dtau of R, R*, R~, R~* exactly.
    (
        (r_phase, r_star, r_tilde, r_tilde_star),
        (
            _r_phase_dot,
            r_star_dot,
            r_tilde_dot,
            _r_tilde_star_dot,
        ),
    ) = jax.jvp(geom, (y,), (v,))

    mr = jnp.sum(v * r_tilde, axis=-1) / c0  # M_R
    dtau_rstar = r_star_dot  # dR*/dtau = -v_i R~*_i
    dtau_mr = (jnp.sum(a * r_tilde, axis=-1) + jnp.sum(v * r_tilde_dot, axis=-1)) / c0
    doppler = 1.0 - mr
    rt1 = r_tilde[..., 0]
    rt1_dot = r_tilde_dot[..., 0]
    rts1 = r_tilde_star[..., 0]

    # Thickness (scalar monopole S = Q_n; Sdot is its full source-time deriv).
    s = qn
    s_dot = source_time_derivative(qn, dtau, axis=-1)
    pt = (
        s_dot / (r_star * doppler**2)
        - dtau_rstar * s / (r_star**2 * doppler**2)
        + s * dtau_mr / (r_star * doppler**3)
        - mach0 * (rt1_dot * s + rt1 * s_dot) / (r_star * doppler**2)
        + mach0 * dtau_rstar * rt1 * s / (r_star**2 * doppler**2)
        - mach0 * dtau_mr * rt1 * s / (r_star * doppler**3)
        - u0 * rts1 * s / (r_star**2 * doppler)
    )

    # Loading (vector L_i projected on the radiation vectors R~, R~*).
    lr = jnp.sum(load * r_tilde, axis=-1)  # L_i R~_i
    load_dot = source_time_derivative(load, dtau, axis=1)
    lr_dot = jnp.sum(load_dot * r_tilde, axis=-1) + jnp.sum(load * r_tilde_dot, axis=-1)
    lrs = jnp.sum(load * r_tilde_star, axis=-1)  # L_i R~*_i
    pl = (
        lr_dot / (c0 * r_star * doppler**2)
        - dtau_rstar * lr / (c0 * r_star**2 * doppler**2)
        + dtau_mr * lr / (c0 * r_star * doppler**3)
        + lrs / (r_star**2 * doppler)
    )
    return r_phase, pt, pl


def f1c_pressure(
    x_obs: ArrayLike,
    y: ArrayLike,
    v: ArrayLike,
    a: ArrayLike,
    qn: ArrayLike,
    load: ArrayLike,
    medium: Medium,
    mach0: ArrayLike,
    tau: ArrayLike,
    t_obs: ArrayLike,
    area: ArrayLike | None = None,
) -> tuple[Array, Array]:
    """Formulation 1C thickness and loading pressure in a uniform mean flow.

    Mean flow ``U0 = mach0 * c0`` along ``+x1``. Source quantities ``qn`` and
    ``load`` follow the same sign/units convention as :func:`f1a.f1a_pressure`
    (``qn`` is the scalar ``Q_i n_i``; ``load`` is the loading vector
    ``L_ij n_j``). At ``mach0 = 0`` this reproduces
    :func:`auraflow.fwh.f1a.f1a_pressure` to near machine precision.

    Args:
        x_obs: Observer positions [m], shape ``[O, 3]``.
        y, v, a: Source position/velocity/acceleration, shape ``[S, T, 3]``
            (velocity/acceleration relative to the ground frame, not the flow).
        qn: Monopole source strength ``Q_i n_i``, shape ``[S, T]``.
        load: Loading vector ``L_ij n_j`` [N or N/m^2], shape ``[S, T, 3]``.
        medium: Ambient medium.
        mach0: Free-stream Mach number ``M0`` along +x1 (scalar, ``|M0| < 1``).
        tau: Uniform source-time grid [s], shape ``[T]``.
        t_obs: Uniform observer-time grid [s], shape ``[T_obs]``.
        area: Panel areas [m^2], shape ``[S]``; default all ones.

    Returns:
        ``(p_thickness, p_loading)`` [Pa], each shape ``[O, T_obs]``.
    """
    y = jnp.asarray(y)
    v = jnp.asarray(v)
    a = jnp.asarray(a)
    qn = jnp.asarray(qn)
    load = jnp.asarray(load)
    tau = jnp.asarray(tau)
    t_obs = jnp.asarray(t_obs)
    mach0 = jnp.asarray(mach0)
    c0 = medium.c0
    n_src = y.shape[0]
    area_arr = jnp.ones(n_src) if area is None else jnp.asarray(area)
    dtau = tau[1] - tau[0]
    w = area_arr[:, None]

    def single(x: Array) -> tuple[Array, Array]:
        r_phase, pt, pl = _f1c_integrands(x, y, v, a, qn, load, dtau, c0, mach0)
        arrival = tau[None, :] + r_phase / c0
        p_t = resample_sum(arrival, pt * w, t_obs) / _FOUR_PI
        p_l = resample_sum(arrival, pl * w, t_obs) / _FOUR_PI
        return p_t, p_l

    return jax.vmap(single)(jnp.asarray(x_obs))


def f1c_windtunnel(
    x_obs: ArrayLike,
    y_panels: ArrayLike,
    normal: ArrayLike,  # noqa: ARG001 -- kept for signature symmetry / documentation
    area: ArrayLike,
    qn: ArrayLike,
    load: ArrayLike,
    medium: Medium,
    mach0: ArrayLike,
    tau: ArrayLike,
    t_obs: ArrayLike,
) -> tuple[Array, Array]:
    """Wind-tunnel special case of 1C: static source and observer in uniform flow.

    With the geometry time-invariant (``v = 0``), ``M_R = 0`` and every
    kinematic derivative (``dR*/dtau``, ``dM_R/dtau``, ``dR~/dtau``) vanishes, so
    the 1C integrals reduce to (``docs/research/cona-external-formulations.md``
    §1 "Special cases"):

        4 pi p'_T = int [ (1 - M0 R~1) Qdot_n / R* - U0 R~*1 Q_n / R*^2 ],
        4 pi p'_L = int [ (1/c0) Ldot_i R~_i / R* + L_i R~*_i / R*^2 ].

    Equivalent to :func:`f1c_pressure` with ``v = a = 0`` but cheaper (radiation
    geometry is evaluated once, no ``jvp``).

    Args:
        x_obs: Observer positions [m], shape ``[O, 3]``.
        y_panels: Static source/panel positions [m], shape ``[S, 3]``.
        normal: Outward unit normals, shape ``[S, 3]``. Unused here (``qn`` and
            ``load`` are the already-contracted sources ``Q_i n_i`` and
            ``L_ij n_j``); accepted for signature symmetry with the permeable
            helpers.
        area: Panel areas [m^2], shape ``[S]``.
        qn: Monopole source strength ``Q_i n_i``, shape ``[S, T]``.
        load: Loading vector ``L_ij n_j``, shape ``[S, T, 3]``.
        medium: Ambient medium.
        mach0: Free-stream Mach number ``M0`` along +x1 (scalar).
        tau: Uniform source-time grid [s], shape ``[T]``.
        t_obs: Uniform observer-time grid [s], shape ``[T_obs]``.

    Returns:
        ``(p_thickness, p_loading)`` [Pa], each shape ``[O, T_obs]``.
    """
    y_panels = jnp.asarray(y_panels)
    area = jnp.asarray(area)
    qn = jnp.asarray(qn)
    load = jnp.asarray(load)
    tau = jnp.asarray(tau)
    t_obs = jnp.asarray(t_obs)
    mach0 = jnp.asarray(mach0)
    c0 = medium.c0
    u0 = mach0 * c0
    dtau = tau[1] - tau[0]

    qn_dot = source_time_derivative(qn, dtau, axis=-1)
    load_dot = source_time_derivative(load, dtau, axis=1)
    w = area[:, None]

    def single(x: Array) -> tuple[Array, Array]:
        r_phase, r_star, r_tilde, r_tilde_star = convective_radiation(x, y_panels, mach0)
        r_star_c = r_star[:, None]
        rt1 = r_tilde[:, 0:1]  # [S, 1]
        rts1 = r_tilde_star[:, 0:1]
        rt_bt = r_tilde[:, None, :]  # [S, 1, 3]
        rts_bt = r_tilde_star[:, None, :]
        pt = (1.0 - mach0 * rt1) * qn_dot / r_star_c - u0 * rts1 * qn / r_star_c**2
        lr_dot = jnp.sum(load_dot * rt_bt, axis=-1)
        lrs = jnp.sum(load * rts_bt, axis=-1)
        pl = lr_dot / (c0 * r_star_c) + lrs / r_star_c**2
        arrival = tau[None, :] + r_phase[:, None] / c0
        p_t = resample_sum(arrival, pt * w, t_obs) / _FOUR_PI
        p_l = resample_sum(arrival, pl * w, t_obs) / _FOUR_PI
        return p_t, p_l

    return jax.vmap(single)(jnp.asarray(x_obs))
