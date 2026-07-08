r"""Momentum-theory inflow models for forward-flight / edgewise operation.

Two levels of fidelity sit on top of the per-annulus solver
(:mod:`auraflow.bemt.solver`):

- **Uniform (Glauert) momentum inflow** -- one mean inflow ratio ``lambda0``
  for the whole disk, solved implicitly for arbitrary climb/edgewise flight.
- **Pitt-Peters linear inflow** -- a first-harmonic distribution
  ``lambda(r, psi) = lambda0 (1 + k_x (r/R) cos psi + k_y (r/R) sin psi)`` that
  captures the fore/aft inflow asymmetry of an edgewise rotor.

Non-dimensionalization (Leishman convention): all inflow ratios and advance
ratios are normalized by the tip speed ``Omega R``:

    mu   = V_inf cos(alpha_p) / (Omega R)   (in-plane advance ratio),
    mu_z = V_inf sin(alpha_p) / (Omega R)   (axial/climb advance ratio, +up),
    lambda = (v_i + V_inf sin alpha_p) / (Omega R)  (total axial inflow ratio),

with ``alpha_p`` the rotor-disk angle of attack (angle of the free stream above
the disk plane). ``C_T`` is the thrust coefficient
``T / (rho A (Omega R)^2)``.

Everything is SI, float64-safe and differentiable; the implicit Glauert solve
uses :func:`jax.lax.custom_root` so gradients flow by the implicit function
theorem.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

__all__ = ["glauert_inflow", "pitt_peters_inflow", "wake_skew_angle"]


def glauert_inflow(
    ct: ArrayLike,
    mu: ArrayLike = 0.0,
    mu_z: ArrayLike = 0.0,
    n_bisect: int = 80,
    lambda_max: float = 5.0,
) -> Array:
    r"""Uniform Glauert momentum inflow ratio ``lambda0``.

    Solves the implicit glauert/momentum relation for the total axial inflow

        lambda = mu_z + C_T / (2 sqrt(mu^2 + lambda^2)),                 (*)

    valid for hover, axial and edgewise flight. Limiting behaviour recovered:

    - **Hover** (``mu = mu_z = 0``): ``lambda = sqrt(C_T / 2)`` exactly.
    - **High-speed edgewise** (``mu >> lambda``): ``lambda ~ mu_z + C_T/(2 mu)``.

    The root is bracketed on ``[mu_z, mu_z + lambda_max]`` (the induced part is
    non-negative for ``C_T >= 0``) and found with a differentiable bisection
    wrapped in :func:`jax.lax.custom_root`, so ``d lambda / d(C_T, mu, mu_z)``
    is exact by implicit differentiation.

    Args:
        ct: Thrust coefficient ``C_T = T / (rho A (Omega R)^2)``, scalar or
            batched, ``>= 0``.
        mu: In-plane advance ratio ``V cos(alpha_p) / (Omega R)``, broadcastable.
        mu_z: Axial advance ratio ``V sin(alpha_p) / (Omega R)`` (+ = climb).
        n_bisect: Bisection iterations.
        lambda_max: Upper bound of the induced-inflow bracket [-].

    Returns:
        Total axial inflow ratio ``lambda0`` [-], broadcast shape of the inputs.
    """
    ct, mu, mu_z = jnp.broadcast_arrays(
        jnp.asarray(ct, dtype=float), jnp.asarray(mu, dtype=float), jnp.asarray(mu_z, dtype=float)
    )

    def residual(lam: Array) -> Array:
        return lam - mu_z - ct / (2.0 * jnp.sqrt(mu**2 + lam**2))

    def solve(res: Callable[[Array], Array], _guess: Array) -> Array:
        lo = mu_z + jnp.zeros_like(ct)
        hi = mu_z + jnp.full_like(ct, lambda_max)
        f_lo = res(lo)

        def body(_i: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
            lo, hi = state
            mid = 0.5 * (lo + hi)
            same = jnp.sign(res(mid)) == jnp.sign(f_lo)
            return jnp.where(same, mid, lo), jnp.where(same, hi, mid)

        lo, hi = jax.lax.fori_loop(0, n_bisect, body, (lo, hi))
        return 0.5 * (lo + hi)

    def tangent_solve(g: Callable[[Array], Array], y: Array) -> Array:
        return y / g(jnp.ones_like(y))

    guess = mu_z + jnp.sqrt(jnp.abs(ct) / 2.0)
    return jax.lax.custom_root(residual, guess, solve, tangent_solve)


def wake_skew_angle(mu: ArrayLike, lambda0: ArrayLike) -> Array:
    r"""Wake skew angle ``chi = atan2(mu, lambda0)`` [rad].

    ``chi`` is the tilt of the wake from the rotor axis: ``chi = 0`` in axial
    flight/hover (``mu = 0``) and ``chi -> pi/2`` in fast edgewise flight
    (``mu >> lambda``). Using :func:`jnp.arctan2` keeps it smooth through
    ``lambda0 = 0``.

    Args:
        mu: In-plane advance ratio [-], any shape.
        lambda0: Total axial inflow ratio [-], broadcastable.

    Returns:
        Skew angle ``chi`` [rad], broadcast shape.
    """
    return jnp.arctan2(jnp.asarray(mu), jnp.asarray(lambda0))


def pitt_peters_inflow(
    lambda0: ArrayLike,
    chi: ArrayLike,
    r_over_R: ArrayLike,
    psi: ArrayLike,
    ky: ArrayLike = 0.0,
) -> Array:
    r"""Pitt-Peters first-harmonic linear inflow ``lambda(r, psi)``.

    .. math::

        lambda(r, psi) = lambda_0 \left(1 + k_x (r/R) \cos psi
                                          + k_y (r/R) \sin psi\right),

    with the longitudinal weight ``k_x = (15 pi / 32) tan(chi / 2)`` (Coleman /
    Pitt-Peters) and the lateral weight ``k_y`` a free parameter (default ``0``,
    matching the CONA reference; a common alternative is ``k_y = -2 mu``).

    Sanity limits of ``k_x``: ``chi = 0`` (hover/axial) gives ``k_x = 0`` (uniform
    inflow); ``chi = pi/2`` (edgewise) gives ``k_x = 15 pi / 32`` -- more inflow
    at the trailing edge of the disk (``psi = 0`` downstream) and less at the
    leading edge, the classic longitudinal inflow gradient.

    Args:
        lambda0: Mean inflow ratio ``lambda_0`` [-], scalar or broadcastable.
        chi: Wake skew angle [rad] (see :func:`wake_skew_angle`).
        r_over_R: Non-dimensional radius ``r/R`` [-], any shape ``[...]``.
        psi: Blade azimuth [rad], broadcastable with ``r_over_R``.
        ky: Lateral inflow weight ``k_y`` [-] (default 0).

    Returns:
        Local inflow ratio ``lambda(r, psi)`` [-], broadcast shape.
    """
    lambda0 = jnp.asarray(lambda0)
    kx = (15.0 * jnp.pi / 32.0) * jnp.tan(0.5 * jnp.asarray(chi))
    rr = jnp.asarray(r_over_R)
    psi = jnp.asarray(psi)
    return lambda0 * (1.0 + kx * rr * jnp.cos(psi) + jnp.asarray(ky) * rr * jnp.sin(psi))
