"""Farassat Formulation 1A (retarded-time, moving sources).

Implements the classical Farassat 1A solution of the FW-H equation with the
volume (quadrupole) term dropped, in the source-time-marching form. Two
physical inputs are supported through one shared kernel:

- **Compact point sources** (impermeable, BEMT backend): a monopole
  ("thickness") strength ``Q_n(tau)`` and/or a loading force vector
  ``L_i(tau)`` attached to a moving point ``y(tau)``.
- **Permeable surfaces** (CFD backend): panel data ``(rho, u, p, n, dS)`` on a
  surface that may move with velocity ``v``, reduced to ``Q_n`` and ``L_i`` via
  :func:`permeable_surface_sources` before entering the same kernel. The
  common case of a **static** surface with time-varying flow is handled by the
  closed-form-delay fast path :func:`f1a_permeable_static`.

Governing equations (``docs/research/cfd-fwh-reference.md`` and
``docs/research/fwh-rotor-sim-audit.md``), with ``d_i = x_i - y_i``,
``r = |d|``, ``rhat = d/r``, ``M_i = v_i/c0``, ``M_r = M_i rhat_i``,
``M^2 = M_i M_i`` and overdots source-time (``tau``) derivatives:

    4 pi p'_T = int [ Qdot_n / (r (1-M_r)^2) ]
              + int [ Q_n (r Mdot_r + c0 (M_r - M^2)) / (r^2 (1-M_r)^3) ]

    4 pi p'_L = int [ Ldot_r / (c0 r (1-M_r)^2) ]
              + int [ (L_r - L_M) / (r^2 (1-M_r)^2) ]
              + int [ L_r (r Mdot_r + c0 (M_r - M^2)) / (c0 r^2 (1-M_r)^3) ]

with ``L_r = L_i rhat_i``, ``L_M = L_i M_i``, ``Ldot_r = (dL_i/dtau) rhat_i``,
``Mdot_r = (dv_i/dtau) rhat_i / c0`` (the overdots act on the source vectors,
not on ``rhat`` -- the ``rhat``-motion contributions are already carried by the
``M_r - M^2`` and ``Mdot_r`` groupings; this is the canonical Brentner-Farassat
grouping and equals the fully-expanded Formulation 1C at ``M0 = 0``, which the
``f1c`` cross-check verifies).

**Time derivatives** of the source quantities ``Q_n``, ``L_i`` are taken by
2nd-order central differences on the (uniform) source-time grid
(:func:`~auraflow.fwh.geometry.source_time_derivative`); the source velocity
``v`` and acceleration ``a`` are supplied by the caller (kinematics are exact,
not differenced). All outputs are differentiable.

Shapes: observers ``x_obs`` ``[O, 3]``; per-source histories ``y, v, a, L``
``[S, T, 3]`` and ``Q_n`` ``[S, T]`` on the source-time grid ``tau`` ``[T]``;
panel areas ``dS`` ``[S]``; output pressures ``[O, T_obs]`` on ``t_obs``.
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from auraflow.core.medium import Medium
from auraflow.fwh.geometry import (
    radiation_vectors,
    resample_sum,
    source_time_derivative,
)

__all__ = [
    "f1a_loading",
    "f1a_permeable",
    "f1a_permeable_static",
    "f1a_pressure",
    "f1a_thickness",
    "permeable_surface_sources",
]

_FOUR_PI = 4.0 * jnp.pi


def _f1a_integrands(
    x: Array,
    y: Array,
    v: Array,
    a: Array,
    qn: Array,
    load: Array,
    dtau: Array,
    c0: Array,
) -> tuple[Array, Array, Array]:
    """Per-source, single-observer 1A integrands (thickness, loading).

    Returns ``(r, pT_integrand, pL_integrand)`` each shape ``[S, T]``; ``r`` is
    the radiation distance used to form arrival times.
    """
    r, rhat = radiation_vectors(x, y)  # [S, T], [S, T, 3]
    mr = jnp.sum(v * rhat, axis=-1) / c0  # [S, T]
    m2 = jnp.sum(v * v, axis=-1) / c0**2
    doppler = 1.0 - mr
    mr_dot = jnp.sum(a * rhat, axis=-1) / c0  # (dv/dtau).rhat / c0
    curv = r * mr_dot + c0 * (mr - m2)  # r Mdot_r + c0 (M_r - M^2)

    # Thickness (monopole) term.
    qn_dot = source_time_derivative(qn, dtau, axis=-1)
    pt = qn_dot / (r * doppler**2) + qn * curv / (r**2 * doppler**3)

    # Loading (dipole) term.
    lr = jnp.sum(load * rhat, axis=-1)
    lm = jnp.sum(load * v, axis=-1) / c0
    load_dot = source_time_derivative(load, dtau, axis=1)
    lr_dot = jnp.sum(load_dot * rhat, axis=-1)
    pl = (
        lr_dot / (c0 * r * doppler**2)
        + (lr - lm) / (r**2 * doppler**2)
        + lr * curv / (c0 * r**2 * doppler**3)
    )
    return r, pt, pl


def f1a_pressure(
    x_obs: ArrayLike,
    y: ArrayLike,
    v: ArrayLike,
    a: ArrayLike,
    qn: ArrayLike,
    load: ArrayLike,
    medium: Medium,
    tau: ArrayLike,
    t_obs: ArrayLike,
    area: ArrayLike | None = None,
) -> tuple[Array, Array]:
    """Farassat 1A thickness and loading pressure for moving compact sources.

    This is the core kernel. Each of the ``S`` sources carries a monopole
    strength ``qn(tau)`` and a loading force ``load(tau)`` while moving along
    ``y(tau)`` with velocity ``v(tau)`` and acceleration ``a(tau)``.
    Contributions are marched in source time to observer arrival times
    ``t = tau + r/c0`` and resampled onto ``t_obs``.

    Args:
        x_obs: Observer positions [m], shape ``[O, 3]``.
        y: Source positions [m], shape ``[S, T, 3]``.
        v: Source velocities [m/s], shape ``[S, T, 3]``.
        a: Source accelerations [m/s^2], shape ``[S, T, 3]``.
        qn: Monopole ("thickness") source strength [kg/s per unit area if
            ``area`` given, else kg/s], shape ``[S, T]``. Physically the
            area density of ``Q_n``; ``sum_s qn * area`` is the enclosed mass
            source rate.
        load: Loading force vector [N per unit area if ``area`` given, else N],
            shape ``[S, T, 3]`` (force exerted by the surface on the fluid).
        medium: Ambient :class:`~auraflow.core.medium.Medium`.
        tau: Uniform source-time grid [s], shape ``[T]``.
        t_obs: Uniform observer-time grid [s], shape ``[T_obs]``.
        area: Panel areas [m^2], shape ``[S]``; default all ones (point
            sources, ``qn``/``load`` already integrated).

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
    c0 = medium.c0
    n_src = y.shape[0]
    area_arr = jnp.ones(n_src) if area is None else jnp.asarray(area)
    dtau = tau[1] - tau[0]
    w = area_arr[:, None]

    def single(x: Array) -> tuple[Array, Array]:
        r, pt, pl = _f1a_integrands(x, y, v, a, qn, load, dtau, c0)
        arrival = tau[None, :] + r / c0
        p_t = resample_sum(arrival, pt * w, t_obs) / _FOUR_PI
        p_l = resample_sum(arrival, pl * w, t_obs) / _FOUR_PI
        return p_t, p_l

    return jax.vmap(single)(jnp.asarray(x_obs))


def f1a_loading(
    x_obs: ArrayLike,
    y: ArrayLike,
    v: ArrayLike,
    a: ArrayLike,
    force: ArrayLike,
    medium: Medium,
    tau: ArrayLike,
    t_obs: ArrayLike,
    area: ArrayLike | None = None,
) -> Array:
    """Loading (dipole) noise of moving compact point forces (BEMT backend).

    Convenience wrapper over :func:`f1a_pressure` with zero thickness source.

    Args:
        x_obs: Observer positions [m], shape ``[O, 3]``.
        y, v, a: Source position/velocity/acceleration, shape ``[S, T, 3]``.
        force: Loading force on the fluid [N], shape ``[S, T, 3]``.
        medium: Ambient medium.
        tau: Source-time grid [s], shape ``[T]``.
        t_obs: Observer-time grid [s], shape ``[T_obs]``.
        area: Optional panel areas [m^2], shape ``[S]``.

    Returns:
        Loading pressure [Pa], shape ``[O, T_obs]``.
    """
    qn = jnp.zeros(jnp.asarray(y).shape[:2])
    return f1a_pressure(x_obs, y, v, a, qn, force, medium, tau, t_obs, area)[1]


def f1a_thickness(
    x_obs: ArrayLike,
    y: ArrayLike,
    v: ArrayLike,
    a: ArrayLike,
    qn: ArrayLike,
    medium: Medium,
    tau: ArrayLike,
    t_obs: ArrayLike,
    area: ArrayLike | None = None,
) -> Array:
    """Thickness (monopole) noise of moving compact sources.

    Convenience wrapper over :func:`f1a_pressure` with zero loading.

    Args:
        x_obs: Observer positions [m], shape ``[O, 3]``.
        y, v, a: Source position/velocity/acceleration, shape ``[S, T, 3]``.
        qn: Monopole source strength, shape ``[S, T]``.
        medium: Ambient medium.
        tau: Source-time grid [s], shape ``[T]``.
        t_obs: Observer-time grid [s], shape ``[T_obs]``.
        area: Optional panel areas [m^2], shape ``[S]``.

    Returns:
        Thickness pressure [Pa], shape ``[O, T_obs]``.
    """
    load = jnp.zeros_like(jnp.asarray(y))
    return f1a_pressure(x_obs, y, v, a, qn, load, medium, tau, t_obs, area)[0]


def permeable_surface_sources(
    rho: ArrayLike,
    u: ArrayLike,
    p: ArrayLike,
    normal: ArrayLike,
    v: ArrayLike,
    medium: Medium,
) -> tuple[Array, Array]:
    """FW-H permeable-surface monopole and loading sources from panel fields.

    Per ``docs/research/cfd-fwh-reference.md`` (viscous stress neglected,
    ``P_ij = (p - p0) delta_ij``):

        Q_n = rho0 v_n + rho (u_n - v_n)          (= rho u_n for a static panel)
        L_i = (p - p0) n_i + rho u_i (u_n - v_n)   (= (p-p0) n_i + rho u_i u_n)

    with ``u_n = u . n``, ``v_n = v . n`` and ``n`` the outward unit normal.

    Args:
        rho: Panel density [kg/m^3], shape ``[S, T]``.
        u: Panel fluid velocity [m/s], shape ``[S, T, 3]``.
        p: Panel pressure [Pa], shape ``[S, T]``.
        normal: Outward unit normals, shape ``[S, 3]`` (static surface) or
            ``[S, T, 3]`` (moving surface).
        v: Surface (panel) velocity [m/s], scalar ``0.0`` for a static surface
            or shape ``[S, T, 3]``.
        medium: Ambient medium (supplies ``rho0`` and ``p0``).

    Returns:
        ``(Q_n, L_i)``: monopole strength [kg/(s m^2)] shape ``[S, T]`` and
        loading force per area [N/m^2] shape ``[S, T, 3]``.
    """
    rho = jnp.asarray(rho)
    u = jnp.asarray(u)
    p = jnp.asarray(p)
    normal = jnp.asarray(normal)
    if normal.ndim == 2:  # [S, 3] -> broadcast over time
        normal = normal[:, None, :]
    v = jnp.broadcast_to(jnp.asarray(v, dtype=u.dtype), u.shape)
    un = jnp.sum(u * normal, axis=-1)
    vn = jnp.sum(v * normal, axis=-1)
    qn = medium.rho0 * vn + rho * (un - vn)
    load = (p - medium.p0)[..., None] * normal + rho[..., None] * u * (un - vn)[..., None]
    return qn, load


def f1a_permeable(
    x_obs: ArrayLike,
    y: ArrayLike,
    v: ArrayLike,
    a: ArrayLike,
    rho: ArrayLike,
    u: ArrayLike,
    p: ArrayLike,
    normal: ArrayLike,
    area: ArrayLike,
    medium: Medium,
    tau: ArrayLike,
    t_obs: ArrayLike,
) -> tuple[Array, Array]:
    """Permeable-surface Farassat 1A for a (possibly moving) surface.

    Computes ``Q_n`` and ``L_i`` from panel fields
    (:func:`permeable_surface_sources`) and feeds them to :func:`f1a_pressure`.
    For a static surface with time-varying flow prefer
    :func:`f1a_permeable_static`, which uses closed-form delays.

    Args:
        x_obs: Observer positions [m], shape ``[O, 3]``.
        y, v, a: Panel-centre position/velocity/acceleration, shape
            ``[S, T, 3]``.
        rho, p: Panel density/pressure, shape ``[S, T]``.
        u: Panel fluid velocity [m/s], shape ``[S, T, 3]``.
        normal: Outward unit normals, shape ``[S, T, 3]`` (or ``[S, 3]``).
        area: Panel areas [m^2], shape ``[S]``.
        medium: Ambient medium.
        tau: Source-time grid [s], shape ``[T]``.
        t_obs: Observer-time grid [s], shape ``[T_obs]``.

    Returns:
        ``(p_thickness, p_loading)`` [Pa], each shape ``[O, T_obs]``.
    """
    qn, load = permeable_surface_sources(rho, u, p, normal, v, medium)
    return f1a_pressure(x_obs, y, v, a, qn, load, medium, tau, t_obs, area)


def f1a_permeable_static(
    x_obs: ArrayLike,
    y_panels: ArrayLike,
    normal: ArrayLike,
    area: ArrayLike,
    rho: ArrayLike,
    u: ArrayLike,
    p: ArrayLike,
    medium: Medium,
    tau: ArrayLike,
    t_obs: ArrayLike,
) -> tuple[Array, Array]:
    """Permeable-surface F1A fast path for a **static** surface (CFD coupling).

    The panel geometry is time-invariant, so the radiation distance ``r`` and
    unit vector ``rhat`` are constant per (panel, observer): ``M_r = 0`` and the
    delay ``t = tau + r/c0`` is a pure per-panel shift. The integrands collapse
    to

        4 pi p'_T = int [ Qdot_n / r ],
        4 pi p'_L = int [ Ldot_r / (c0 r) + L_r / r^2 ],

    with ``Q_n = rho u_n`` and ``L_i = (p - p0) n_i + rho u_i u_n`` (surface at
    rest). This is the OpenCFD-FWH stationary-surface algorithm with ``M0 = 0``
    (``docs/research/cfd-fwh-reference.md``); use :mod:`auraflow.fwh.f1c` for the
    uniform-mean-flow (wind-tunnel) convective variant.

    Args:
        x_obs: Observer positions [m], shape ``[O, 3]``.
        y_panels: Static panel-centre positions [m], shape ``[S, 3]``.
        normal: Outward unit normals, shape ``[S, 3]``.
        area: Panel areas [m^2], shape ``[S]``.
        rho, p: Panel density/pressure time histories, shape ``[S, T]``.
        u: Panel fluid velocity [m/s], shape ``[S, T, 3]``.
        medium: Ambient medium.
        tau: Source-time grid [s], shape ``[T]``.
        t_obs: Observer-time grid [s], shape ``[T_obs]``.

    Returns:
        ``(p_thickness, p_loading)`` [Pa], each shape ``[O, T_obs]``.
    """
    y_panels = jnp.asarray(y_panels)
    normal = jnp.asarray(normal)
    area = jnp.asarray(area)
    tau = jnp.asarray(tau)
    t_obs = jnp.asarray(t_obs)
    c0 = medium.c0
    dtau = tau[1] - tau[0]

    qn, load = permeable_surface_sources(rho, u, p, normal, 0.0, medium)  # [S, T], [S, T, 3]
    qn_dot = source_time_derivative(qn, dtau, axis=-1)
    load_dot = source_time_derivative(load, dtau, axis=1)
    w = area[:, None]

    def single(x: Array) -> tuple[Array, Array]:
        r, rhat = radiation_vectors(x, y_panels)  # [S], [S, 3]
        r_col = r[:, None]
        rhat_bt = rhat[:, None, :]  # [S, 1, 3]
        pt = qn_dot / r_col
        lr = jnp.sum(load * rhat_bt, axis=-1)
        lr_dot = jnp.sum(load_dot * rhat_bt, axis=-1)
        pl = lr_dot / (c0 * r_col) + lr / r_col**2
        arrival = tau[None, :] + r_col / c0
        p_t = resample_sum(arrival, pt * w, t_obs) / _FOUR_PI
        p_l = resample_sum(arrival, pl * w, t_obs) / _FOUR_PI
        return p_t, p_l

    return jax.vmap(single)(jnp.asarray(x_obs))
