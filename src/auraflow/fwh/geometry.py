"""Retarded-/emission-time machinery for the FW-H solvers.

This module holds the geometry and time-marching primitives shared by the
Farassat 1A (:mod:`auraflow.fwh.f1a`) and Formulation 1C
(:mod:`auraflow.fwh.f1c`) kernels. Two regimes are supported, following the
OpenCFD-FWH / PSU-WOPWOP "source-time-dominant" (advanced-time) approach
(see ``docs/research/cfd-fwh-reference.md`` §"Time algorithm"):

- **Static observers and static surface points**: the emission-time delay is
  closed form, ``t = tau + r / c0`` with ``r`` constant per (source, observer)
  pair; no root finding is needed.
- **Moving compact sources**: for each source time ``tau`` the observer arrival
  time is ``t = tau + R(tau) / c0``; per-source contributions are interpolated
  (differentiably) onto a shared uniform observer time grid and summed. This
  avoids retarded-time root finding entirely.

Conventions (see ``docs/architecture.md``):

- SI units (m, s, kg, Pa, rad); all math is float64-safe and differentiable.
- Separation vector ``d_i = x_i - y_i`` (observer minus source); ``rhat = d/r``
  points *from source to observer*.
- Overdots denote source-time (``tau``) derivatives.
- Shapes: single observer ``x`` is ``[3]``; per-source time histories are
  ``[S, T]`` (scalars) or ``[S, T, 3]`` (vectors); ``tau`` is ``[T]`` and must
  be **uniformly spaced** (central-difference derivatives assume this). The
  uniform observer grid is ``t_obs`` of shape ``[T_obs]``.
- Subsonic motion only: ``1 - M_r > 0``, so arrival times increase
  monotonically with ``tau`` and piecewise-linear resampling is well posed.
"""

from typing import cast

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

__all__ = [
    "arrival_times",
    "convective_radiation",
    "default_observer_grid",
    "doppler_factor",
    "mach_radial",
    "radiation_vectors",
    "resample_sum",
    "source_time_derivative",
]


def radiation_vectors(x_obs: ArrayLike, y_src: ArrayLike) -> tuple[Array, Array]:
    """Radiation distance and unit vector from sources to an observer.

    Args:
        x_obs: Observer position [m], shape ``[..., 3]`` (broadcasts against
            ``y_src``; typically a single ``[3]``).
        y_src: Source positions [m], shape ``[..., 3]`` (e.g. ``[S, T, 3]``).

    Returns:
        ``(r, rhat)``: distance ``r = |x - y|`` [m], shape ``[...]``, and unit
        radiation vector ``rhat = (x - y) / r``, shape ``[..., 3]``.
    """
    d = jnp.asarray(x_obs) - jnp.asarray(y_src)
    r = jnp.linalg.norm(d, axis=-1)
    return r, d / r[..., None]


def mach_radial(v: ArrayLike, rhat: ArrayLike, c0: ArrayLike) -> Array:
    """Radiation-direction Mach number ``M_r = v . rhat / c0``.

    Args:
        v: Source velocity [m/s], shape ``[..., 3]``.
        rhat: Unit radiation vector, shape ``[..., 3]``.
        c0: Speed of sound [m/s], scalar.

    Returns:
        ``M_r``, shape ``[...]``.
    """
    return jnp.sum(jnp.asarray(v) * jnp.asarray(rhat), axis=-1) / c0


def doppler_factor(v: ArrayLike, rhat: ArrayLike, c0: ArrayLike) -> Array:
    """Doppler amplification factor ``1 - M_r``.

    Args:
        v: Source velocity [m/s], shape ``[..., 3]``.
        rhat: Unit radiation vector, shape ``[..., 3]``.
        c0: Speed of sound [m/s], scalar.

    Returns:
        ``1 - M_r``, shape ``[...]``. Positive for subsonic radial motion.
    """
    return 1.0 - mach_radial(v, rhat, c0)


def source_time_derivative(f: ArrayLike, dtau: ArrayLike, axis: int = -1) -> Array:
    """Second-order central finite difference along the source-time axis.

    Thin wrapper over :func:`jnp.gradient`: 2nd-order central differences in the
    interior, 1st-order one-sided at the ends (per the OpenCFD-FWH algorithm,
    ``docs/research/cfd-fwh-reference.md``). Differentiable with respect to
    ``f``. **Assumes a uniform grid** with spacing ``dtau``.

    Args:
        f: Samples on a uniform source-time grid, any shape; time along ``axis``.
        dtau: Grid spacing [s], scalar.
        axis: Time axis of ``f``.

    Returns:
        ``df/dtau``, same shape as ``f``.
    """
    return cast(Array, jnp.gradient(jnp.asarray(f), dtau, axis=axis))


def arrival_times(tau: ArrayLike, r: ArrayLike, c0: ArrayLike) -> Array:
    """Observer arrival time of a contribution emitted at source time ``tau``.

    Advanced-time relation ``t = tau + r / c0`` (``r`` the phase distance).

    Args:
        tau: Source times [s], broadcastable with ``r``.
        r: Radiation (phase) distance [m], any shape.
        c0: Speed of sound [m/s], scalar.

    Returns:
        Arrival times [s], broadcast shape of ``tau`` and ``r``.
    """
    return jnp.asarray(tau) + jnp.asarray(r) / c0


def resample_sum(t_arrival: ArrayLike, contrib: ArrayLike, t_obs: ArrayLike) -> Array:
    """Resample per-source arrival series onto a shared grid and sum.

    Each source ``s`` contributes a value ``contrib[s, k]`` arriving at time
    ``t_arrival[s, k]``; these are linearly interpolated onto the common
    observer grid ``t_obs`` and summed over sources. Interpolation is
    differentiable with respect to both ``contrib`` and ``t_arrival``. Queries
    outside a source's arrival span contribute **zero** (there is no signal
    before a source's earliest emission arrives, nor after its latest): the
    interpolation is zero-filled, not clamped to the end values. Constant
    (clamped) extrapolation was a bug -- when a *shared* ``t_obs`` spans a wider
    window than a given source's arrivals (e.g. one grid serving mics at very
    different ranges, :func:`~auraflow.cfd.flyover.quadrotor_surface_flyover`),
    clamping freezes each source's endpoint integrand into a large DC plateau
    over the out-of-window tail, summing to a spurious low-frequency pedestal
    that can bury the real signal. Zero-fill instead leaves those regions
    silent. Within the common valid window (:func:`default_observer_grid`)
    every source has data everywhere, so the choice is immaterial there.

    Args:
        t_arrival: Per-source arrival times [s], shape ``[S, T]``, increasing
            along the time axis (guaranteed for subsonic motion).
        contrib: Per-source contributions, shape ``[S, T]``.
        t_obs: Shared uniform observer grid [s], shape ``[T_obs]``.

    Returns:
        Summed, resampled signal on ``t_obs``, shape ``[T_obs]``.
    """
    t_arrival = jnp.asarray(t_arrival)
    contrib = jnp.asarray(contrib)
    t_obs = jnp.asarray(t_obs)

    def one(ta: Array, c: Array) -> Array:
        return jnp.interp(t_obs, ta, c, left=0.0, right=0.0)

    return jnp.sum(jax.vmap(one)(t_arrival, contrib), axis=0)


def default_observer_grid(t_arrival: ArrayLike, n_obs: int) -> Array:
    """Uniform observer grid spanning the window covered by *all* sources.

    Following ``docs/research/cfd-fwh-reference.md``: the valid window starts at
    the latest of the per-source earliest arrivals and ends at the earliest of
    the per-source latest arrivals, so every source has data across the whole
    grid (no extrapolation).

    Args:
        t_arrival: Per-source arrival times [s], shape ``[S, T]``.
        n_obs: Number of grid points ``T_obs`` (static int).

    Returns:
        Uniform grid ``t_obs`` [s], shape ``[n_obs]``.
    """
    t_arrival = jnp.asarray(t_arrival)
    t_lo = jnp.max(jnp.min(t_arrival, axis=-1))
    t_hi = jnp.min(jnp.max(t_arrival, axis=-1))
    return jnp.linspace(t_lo, t_hi, n_obs)


def convective_radiation(
    x_obs: ArrayLike, y_src: ArrayLike, mach0: ArrayLike
) -> tuple[Array, Array, Array, Array]:
    """Convective radiation geometry for Formulation 1C (uniform flow along +x1).

    Mean flow ``U0 = M0 c0`` along ``+x1``, ``beta^2 = 1 - M0^2``. Returns the
    phase distance ``R``, amplitude distance ``R*`` and the (non-unit)
    radiation vectors ``R~_i = dR/dx_i`` and ``R~*_i = dR*/dx_i``, per
    ``docs/research/cona-external-formulations.md`` §1:

    - ``R* = sqrt(d1^2 + beta^2 (d2^2 + d3^2))``, ``d_i = x_i - y_i``
    - ``R  = (-M0 d1 + R*) / beta^2``
    - ``R~*  = (d1, beta^2 d2, beta^2 d3) / R*``
    - ``R~   = ((-M0 + d1/R*) / beta^2, d2/R*, d3/R*)``

    Setting ``M0 = 0`` gives ``R = R* = |x - y|`` and ``R~ = R~* = rhat``, i.e.
    the ordinary (non-convective) geometry.

    Args:
        x_obs: Observer position [m], shape ``[..., 3]`` (typically ``[3]``).
        y_src: Source positions [m], shape ``[..., 3]``.
        mach0: Free-stream Mach number ``M0`` along +x1 (scalar, ``|M0| < 1``).

    Returns:
        ``(R, R_star, R_tilde, R_tilde_star)`` with scalar fields shape
        ``[...]`` and vector fields shape ``[..., 3]``.
    """
    d = jnp.asarray(x_obs) - jnp.asarray(y_src)
    d1, d2, d3 = d[..., 0], d[..., 1], d[..., 2]
    mach0 = jnp.asarray(mach0)
    beta2 = 1.0 - mach0**2
    r_star = jnp.sqrt(d1**2 + beta2 * (d2**2 + d3**2))
    r = (-mach0 * d1 + r_star) / beta2
    r_tilde_star = jnp.stack([d1 / r_star, beta2 * d2 / r_star, beta2 * d3 / r_star], axis=-1)
    r_tilde = jnp.stack([(-mach0 + d1 / r_star) / beta2, d2 / r_star, d3 / r_star], axis=-1)
    return r, r_star, r_tilde, r_tilde_star
