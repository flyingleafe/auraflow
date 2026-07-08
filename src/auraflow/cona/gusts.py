r"""Dryden low-altitude atmospheric turbulence (MIL-F-8785C).

Discrete Dryden gust generator per ``docs/research/cona-external-formulations.md``
sect. 4 -- the wind model feeding the CONA flight simulation
(``docs/research/cona-reference.md`` module 1). Forming filters are driven by
Gaussian white noise; the longitudinal component ``u`` uses the exact first-order
(AR(1)) discretization, and the lateral/vertical components ``v, w`` use the exact
zero-order-hold discretization of the two-state forming filter (van Loan), so that
each component reproduces its target stationary variance exactly.

**Units.** SI in and out (m, m/s, s). The MIL-F-8785C scale-length and intensity
formulas are empirical in *feet*; the feet conversion happens internally.

Low-altitude spectra (``h < 1000`` ft, ``h`` the altitude, ``V`` the airspeed):

.. math::
   \Phi_u(\omega) &= \frac{2\sigma_u^2 L_u}{\pi V}
        \frac{1}{1 + (L_u\omega/V)^2} \\
   \Phi_{v,w}(\omega) &= \frac{\sigma^2 L}{\pi V}
        \frac{1 + 3(L\omega/V)^2}{\big(1 + (L\omega/V)^2\big)^2}

with (``h`` in feet) ``L_w = h``, ``L_u = L_v = h/(0.177 + 0.000823 h)^{1.2}``,
``sigma_w = 0.1 W_{20}``, ``sigma_u = sigma_v = sigma_w/(0.177 + 0.000823 h)^{0.4}``,
where ``W_{20}`` is the mean wind at 20 ft (light 15 kt, moderate 30 kt, severe
45 kt).

**Output frame.** The series is returned in the aircraft *stability/body* frame:
``u`` longitudinal (along the airspeed vector), ``v`` lateral, ``w`` vertical
(positive *down*, aero convention). For a level flight along world ``+x`` with a
level attitude this maps to world ``(+x, +y, -z)``; rotate by the vehicle
attitude to obtain a world-frame wind for :func:`auraflow.cona.flight.simulate`.

Requires ``dt << L/V`` (short step relative to the turbulence correlation time).
Float64-safe and differentiable in the atmosphere/airspeed parameters.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

__all__ = [
    "DrydenParameters",
    "W20_PRESETS",
    "dryden_gust",
    "dryden_parameters",
]

_FT_PER_M = 1.0 / 0.3048  # feet per metre
_KT_TO_MS = 0.514444  # knots to m/s

#: Mean wind at 20 ft for the MIL-F-8785C turbulence severities [m/s].
W20_PRESETS: dict[str, float] = {
    "light": 15.0 * _KT_TO_MS,
    "moderate": 30.0 * _KT_TO_MS,
    "severe": 45.0 * _KT_TO_MS,
}


class DrydenParameters(NamedTuple):
    """Dryden scale lengths [m] and turbulence intensities [m/s] (SI).

    Attributes:
        L_u, L_v, L_w: Longitudinal / lateral / vertical scale lengths [m].
        sigma_u, sigma_v, sigma_w: RMS turbulence intensities [m/s].
    """

    L_u: Array
    L_v: Array
    L_w: Array
    sigma_u: Array
    sigma_v: Array
    sigma_w: Array


def _resolve_w20(w20: float | str) -> Array:
    """Resolve a preset name or numeric mean-wind to an array [m/s] (traceable)."""
    if isinstance(w20, str):
        try:
            return jnp.asarray(W20_PRESETS[w20], dtype=float)
        except KeyError:
            raise ValueError(
                f"unknown W20 preset {w20!r}; choose from {sorted(W20_PRESETS)}"
            ) from None
    return jnp.asarray(w20, dtype=float)


def dryden_parameters(altitude: ArrayLike, w20: float | str) -> DrydenParameters:
    """MIL-F-8785C low-altitude scale lengths and turbulence intensities.

    Args:
        altitude: Altitude above ground ``h`` [m], scalar (``h < ~305`` m for the
            low-altitude form to be valid).
        w20: Mean wind at 20 ft: a value in [m/s], or a preset name
            (``"light"``, ``"moderate"``, ``"severe"``).

    Returns:
        A :class:`DrydenParameters` with SI scale lengths and intensities.
    """
    h_m = jnp.asarray(altitude, dtype=float)
    h_ft = h_m * _FT_PER_M
    w20_ms = _resolve_w20(w20)
    denom = 0.177 + 0.000823 * h_ft
    l_w = h_m
    l_u = h_m / denom**1.2
    sigma_w = 0.1 * w20_ms  # 0.1 * W20; the ft<->m factor cancels
    sigma_u = sigma_w / denom**0.4
    return DrydenParameters(
        L_u=l_u, L_v=l_u, L_w=l_w, sigma_u=sigma_u, sigma_v=sigma_u, sigma_w=sigma_w
    )


def _ar1_series(phi: Array, sigma: Array, eta: Array) -> Array:
    r"""Exact first-order (AR(1)) forming filter, stationary from ``n = 0``.

    ``x_0 = sigma * eta_0``; ``x_n = phi x_{n-1} + sigma sqrt(1 - phi^2) eta_n``.
    Stationary variance is ``sigma^2`` for all ``n``.

    Args:
        phi: AR(1) pole ``exp(-V dt / L)``, scalar in ``(0, 1)``.
        sigma: Target RMS intensity, scalar.
        eta: Unit-variance Gaussian innovations, shape ``[T]``.

    Returns:
        The filtered series, shape ``[T]``.
    """
    g = sigma * jnp.sqrt(1.0 - phi**2)
    x0 = sigma * eta[0]

    def step(x_prev: Array, eta_n: Array) -> tuple[Array, Array]:
        x = phi * x_prev + g * eta_n
        return x, x

    _, tail = jax.lax.scan(step, x0, eta[1:])
    return jnp.concatenate([x0[None], tail])


def _dryden_vw_series(scale: Array, sigma: Array, v_air: Array, dt: float, eta: Array) -> Array:
    r"""Exact ZOH discretization of the two-state ``v``/``w`` Dryden filter.

    Realizes ``H(s) = K (1 + sqrt(3) T s)/(1 + T s)^2`` with ``T = L/V`` and
    ``K = sigma sqrt(2T/pi)`` in controllable canonical form, discretizes it
    exactly for a piecewise-constant/white input via van Loan's method, and
    drives it with an innovation covariance calibrated so the stationary output
    variance equals ``sigma^2``. The state is initialized from the stationary
    covariance so the series is stationary from ``n = 0``.

    Args:
        scale: Turbulence scale length ``L`` [m], scalar.
        sigma: Target RMS intensity [m/s], scalar.
        v_air: Airspeed ``V`` [m/s], scalar (> 0).
        dt: Time step [s].
        eta: Unit-variance Gaussian innovations, shape ``[T, 2]``.

    Returns:
        The filtered series, shape ``[T]``.
    """
    tconst = scale / v_air
    kgain = sigma * jnp.sqrt(2.0 * tconst / jnp.pi)
    a0 = 1.0 / tconst**2
    a1 = 2.0 / tconst
    b0 = kgain / tconst**2
    b1 = kgain * jnp.sqrt(3.0) / tconst
    A = jnp.array([[0.0, 1.0], [-a0, -a1]])
    C = jnp.array([b0, b1])

    # Input intensity giving output variance sigma^2 in this realization is pi/2
    # (derived analytically; see module notes). Q = q * B B^T with B = [0, 1]^T.
    q = jnp.pi / 2.0
    Q = q * jnp.array([[0.0, 0.0], [0.0, 1.0]])

    # Van Loan: exact discrete transition Ad and process-noise covariance Qd.
    upper = jnp.concatenate([-A, Q], axis=1)
    lower = jnp.concatenate([jnp.zeros((2, 2)), A.T], axis=1)
    big = jnp.concatenate([upper, lower], axis=0) * dt
    G = jax.scipy.linalg.expm(big)
    Ad = G[2:, 2:].T
    Qd = Ad @ G[:2, 2:]
    Qd = 0.5 * (Qd + Qd.T)  # symmetrize
    Ld = jnp.linalg.cholesky(Qd + 1e-30 * jnp.eye(2))

    # Stationary state covariance P = diag(pi T^3 / 8, pi T / 8); output var = sigma^2.
    p11 = jnp.pi * tconst**3 / 8.0
    p22 = jnp.pi * tconst / 8.0
    L0 = jnp.diag(jnp.sqrt(jnp.array([p11, p22])))

    x0 = L0 @ eta[0]

    def step(x_prev: Array, eta_n: Array) -> tuple[Array, Array]:
        x = Ad @ x_prev + Ld @ eta_n
        return x, C @ x

    y0 = C @ x0
    _, tail = jax.lax.scan(step, x0, eta[1:])
    return jnp.concatenate([y0[None], tail])


def dryden_gust(
    key: Array,
    altitude: ArrayLike,
    airspeed: ArrayLike,
    w20: float | str,
    dt: float,
    n_steps: int,
) -> Array:
    r"""Generate a discrete Dryden low-altitude gust-velocity series.

    Args:
        key: JAX PRNG key.
        altitude: Altitude above ground ``h`` [m], scalar.
        airspeed: Airspeed ``V`` [m/s], scalar (> 0).
        w20: Mean wind at 20 ft [m/s] or preset (``"light"``/``"moderate"``/
            ``"severe"``). ``w20 = 0`` gives an identically-zero series.
        dt: Time step [s] (should satisfy ``dt << L/V``).
        n_steps: Number of samples ``T`` (static int, ``>= 1``).

    Returns:
        Gust velocities [m/s], shape ``[T, 3]``, in the stability/body frame
        ``(u, v, w)`` = (longitudinal, lateral, vertical-down). See the module
        docstring for the world-frame mapping.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    v_air = jnp.asarray(airspeed, dtype=float)
    params = dryden_parameters(altitude, w20)

    ku, kv, kw = jax.random.split(key, 3)
    eta_u = jax.random.normal(ku, (n_steps,))
    eta_v = jax.random.normal(kv, (n_steps, 2))
    eta_w = jax.random.normal(kw, (n_steps, 2))

    phi_u = jnp.exp(-v_air * dt / params.L_u)
    u = _ar1_series(phi_u, params.sigma_u, eta_u)
    v = _dryden_vw_series(params.L_v, params.sigma_v, v_air, dt, eta_v)
    w = _dryden_vw_series(params.L_w, params.sigma_w, v_air, dt, eta_w)
    return jnp.stack([u, v, w], axis=1)
