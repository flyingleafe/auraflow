"""2D airfoil section profiles for lofting rotor-blade meshes (``auraflow.body``).

A *profile* is a closed, non-self-intersecting 2D loop of the airfoil section in
**unit-chord** coordinates ``(xi, eta)``:

- ``xi`` -- chordwise coordinate, ``0`` at the leading edge, ``1`` at the
  trailing edge (fraction of chord).
- ``eta`` -- section-normal (thickness) coordinate, positive on the upper
  surface, as a fraction of chord.

:func:`naca4_profile` builds the standard NACA 4-digit section (mean-camber line
plus thickness envelope). The vertices are returned ordered so the loop is
**counterclockwise** in the ``(xi, eta)`` plane (positive shoelace area), which
:func:`auraflow.body.blade.blade_mesh` relies on to produce an outward-wound
(``volume() > 0``) swept surface.

Trailing edge: the **closed** (blunt-closed) thickness polynomial is used
(``a4 = 0.1036``) so ``eta(1) = 0`` exactly and the upper/lower surfaces meet at
a single trailing-edge point -- the loop closes with no gap and no duplicate
vertex. The section is differentiable in ``(m, p, tc)`` (guarded division so the
symmetric ``m = 0`` case is finite; ``naca0012`` uses it).

All maths is JAX (differentiable); the returned array is ``[N, 2]`` float64.
"""

from functools import partial

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

__all__ = ["naca0012", "naca4_profile"]

# Closed-trailing-edge quartic coefficient for the NACA 4-digit thickness
# polynomial (``a4 = 0.1036`` closes eta(1) exactly; the classic open-TE value is
# 0.1015). Closed TE keeps the lofted loop watertight without a TE gap.
_A0 = 0.2969
_A1 = 0.1260
_A2 = 0.3516
_A3 = 0.2843
_A4 = 0.1036


def naca4_profile(m: ArrayLike, p: ArrayLike, tc: ArrayLike, n_points: int = 120) -> Array:
    """Closed NACA 4-digit airfoil section as a 2D loop, unit chord.

    Standard NACA 4-digit equations: a mean-camber line of maximum camber ``m``
    at chordwise location ``p`` plus a symmetric thickness envelope of
    maximum thickness ratio ``tc``. Chordwise sampling is cosine-clustered
    (dense at the leading and trailing edges) for an accurate leading-edge
    radius and thickness peak.

    The loop is ordered **counterclockwise** in ``(xi, eta)`` (lower surface
    from leading to trailing edge, then upper surface back), giving a positive
    shoelace area; :func:`auraflow.body.blade.blade_mesh` maps this to an
    outward-wound swept surface.

    Differentiable in ``(m, p, tc)`` (float64). The ``p`` denominator is guarded
    so ``m = 0`` (symmetric) sections are finite regardless of ``p``.

    Args:
        m: Maximum camber as a fraction of chord (e.g. ``0.02`` for 2%).
        p: Chordwise location of maximum camber as a fraction of chord in
            ``(0, 1)`` (unused when ``m = 0`` but must be passed).
        tc: Maximum thickness as a fraction of chord (e.g. ``0.12``).
        n_points: Number of chordwise samples **per surface** (static int,
            ``>= 3``). The returned loop has ``2 * n_points - 2`` vertices (the
            shared leading- and trailing-edge points are not duplicated).

    Returns:
        Section loop [-], shape ``[2 * n_points - 2, 2]`` (``(xi, eta)`` per
        row), a closed non-self-intersecting counterclockwise polygon in
        unit-chord coordinates.
    """
    if n_points < 3:
        raise ValueError(f"n_points must be >= 3, got {n_points}")
    m = jnp.asarray(m, dtype=jnp.float64)
    p = jnp.asarray(p, dtype=jnp.float64)
    tc = jnp.asarray(tc, dtype=jnp.float64)

    # Cosine-clustered chordwise stations, xi in [0, 1] (LE -> TE).
    beta = jnp.linspace(0.0, jnp.pi, n_points)
    xi = 0.5 * (1.0 - jnp.cos(beta))

    # Half-thickness envelope (closed TE: eta_t(1) = 0 exactly).
    eta_t = 5.0 * tc * (_A0 * jnp.sqrt(xi) - _A1 * xi - _A2 * xi**2 + _A3 * xi**3 - _A4 * xi**4)

    # Mean-camber line and its slope, guarded so m = 0 (p possibly 0 or 1) is
    # finite: p_safe stays in (0, 1) so neither the fore nor the aft denominator
    # (p_safe**2, (1 - p_safe)**2) vanishes. With m = 0 the camber is 0 anyway.
    p_safe = jnp.where((p > 0.0) & (p < 1.0), p, 0.5)
    fwd = xi < p_safe
    yc = jnp.where(
        fwd,
        m / p_safe**2 * (2.0 * p_safe * xi - xi**2),
        m / (1.0 - p_safe) ** 2 * ((1.0 - 2.0 * p_safe) + 2.0 * p_safe * xi - xi**2),
    )
    dyc = jnp.where(
        fwd,
        2.0 * m / p_safe**2 * (p_safe - xi),
        2.0 * m / (1.0 - p_safe) ** 2 * (p_safe - xi),
    )
    theta = jnp.arctan(dyc)
    ct, st = jnp.cos(theta), jnp.sin(theta)

    xu = xi - eta_t * st
    yu = yc + eta_t * ct
    xl = xi + eta_t * st
    yl = yc - eta_t * ct

    lower = jnp.stack([xl, yl], axis=-1)  # LE -> TE (below camber)
    upper = jnp.stack([xu, yu], axis=-1)  # LE -> TE (above camber)
    # CCW loop: lower LE->TE, then upper TE->LE, dropping the shared LE and TE
    # vertices so no point is duplicated.
    upper_back = upper[-2:0:-1]  # indices n-2 .. 1
    return jnp.concatenate([lower, upper_back], axis=0)


# NACA 0012: symmetric, 12% thick. ``p`` is irrelevant for a symmetric section
# (m = 0) but a value must be supplied; 0.0 is fine (division is guarded).
naca0012 = partial(naca4_profile, 0.0, 0.0, 0.12)
"""Symmetric 12%-thick NACA 0012 section: ``naca0012(n_points=...) -> [N, 2]``."""
