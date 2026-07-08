"""Airfoil polar models (lift/drag coefficient vs angle of attack).

All polars are equinox modules with the common call signature::

    cl, cd = polar(alpha, mach=None, reynolds=None)

where ``alpha`` is the angle of attack [rad] (any shape) and the returned
``cl``/``cd`` broadcast over ``alpha`` (and ``mach``/``reynolds`` where
applicable). All models are smooth and differentiable everywhere — no hard
clamps with zero-gradient dead zones inside the physical range (any remaining
dead zones are documented explicitly).
"""

import itertools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

__all__ = ["TablePolar", "ThinAirfoilPolar"]

_DEFAULT_ALPHA_STALL = math.radians(20.0)  # [rad]
_DEFAULT_STALL_WIDTH = math.radians(2.0)  # [rad]


def _soft_clip(x: Array, limit: Array, width: Array) -> Array:
    """Smoothly clip ``x`` to ``[-limit, +limit]`` with transition width ``width``.

    ``width * (softplus((x + limit)/width) - softplus((x - limit)/width)) - limit``.

    Since ``sigmoid(z) = (1 + tanh(z/2)) / 2``, the derivative is the
    tanh-based smooth boxcar ``sigmoid((x + limit)/width) - sigmoid((x -
    limit)/width)``, which is strictly positive for all finite ``x`` — no
    zero-gradient dead zones. For ``|x| < limit - a few * width`` the map is
    the identity up to an ``O(exp(-(limit - |x|)/width))`` error; for
    ``|x| >> limit`` it saturates at ``+-limit``.
    """
    return (
        width * (jax.nn.softplus((x + limit) / width) - jax.nn.softplus((x - limit) / width))
        - limit
    )


class ThinAirfoilPolar(eqx.Module):
    """Thin-airfoil lift with parabolic drag and smooth stall saturation.

    In the linear range,

    ``cl = cl_alpha * (alpha - alpha0)``, ``cd = cd0 + k * cl**2``.

    Instead of a hard clamp at stall (which kills gradients), the effective
    incidence ``alpha - alpha0`` is passed through a smooth (tanh-family)
    saturation that is the identity well below ``alpha_stall`` and levels off
    at ``+-alpha_stall`` over a transition of width ``stall_width`` (see
    ``_soft_clip``). The gradient ``d cl / d alpha`` is strictly positive for
    all finite ``alpha``.

    If ``mach`` is given, the Prandtl–Glauert compressibility correction
    ``cl / sqrt(1 - mach**2)`` is applied (valid for subsonic sections,
    roughly ``mach < 0.7``; no transonic limiting is performed). ``reynolds``
    is accepted for signature compatibility and ignored.

    Attributes:
        alpha0: Zero-lift angle of attack [rad].
        cl_alpha: Lift-curve slope [1/rad] (thin-airfoil theory: ``2 pi``).
        cd0: Minimum (profile) drag coefficient [-].
        k: Induced/parasite drag factor [-]: ``cd = cd0 + k cl^2``.
        alpha_stall: Saturation (stall) incidence [rad]; ``|cl|`` levels off
            at ``cl_alpha * alpha_stall`` (before compressibility correction).
        stall_width: Transition width of the smooth saturation [rad].
    """

    alpha0: Array
    cl_alpha: Array
    cd0: Array
    k: Array
    alpha_stall: Array
    stall_width: Array

    def __init__(
        self,
        alpha0: ArrayLike = 0.0,
        cl_alpha: ArrayLike = 2.0 * jnp.pi,
        cd0: ArrayLike = 0.01,
        k: ArrayLike = 0.0,
        alpha_stall: ArrayLike = _DEFAULT_ALPHA_STALL,
        stall_width: ArrayLike = _DEFAULT_STALL_WIDTH,
    ):
        """Construct a thin-airfoil polar.

        Args:
            alpha0: Zero-lift angle of attack [rad], scalar.
            cl_alpha: Lift-curve slope [1/rad], scalar. Default ``2 pi``.
            cd0: Minimum drag coefficient [-], scalar.
            k: Induced drag factor [-], scalar (``cd = cd0 + k cl^2``).
            alpha_stall: Stall (saturation) incidence [rad], scalar.
                Default 20 degrees.
            stall_width: Smooth-saturation transition width [rad], scalar.
                Default 2 degrees. Must be > 0.
        """
        self.alpha0 = jnp.asarray(alpha0)
        self.cl_alpha = jnp.asarray(cl_alpha)
        self.cd0 = jnp.asarray(cd0)
        self.k = jnp.asarray(k)
        self.alpha_stall = jnp.asarray(alpha_stall)
        self.stall_width = jnp.asarray(stall_width)

    def __call__(
        self,
        alpha: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        """Evaluate the polar.

        Args:
            alpha: Angle of attack [rad], any shape ``[...]``.
            mach: Optional section Mach number [-], broadcastable with
                ``alpha``. Applies Prandtl–Glauert ``1/sqrt(1 - mach^2)`` to
                ``cl``; must satisfy ``mach < 1``.
            reynolds: Ignored (accepted for signature compatibility).

        Returns:
            ``(cl, cd)`` [-, -], each of shape ``[...]`` (broadcast of
            ``alpha`` and ``mach``).
        """
        del reynolds  # not modeled by thin-airfoil theory
        incidence = jnp.asarray(alpha) - self.alpha0
        cl = self.cl_alpha * _soft_clip(incidence, self.alpha_stall, self.stall_width)
        if mach is not None:
            cl = cl / jnp.sqrt(1.0 - jnp.asarray(mach) ** 2)
        cd = self.cd0 + self.k * cl**2
        return cl, cd


def _locate(grid: Array, query: Array) -> tuple[Array, Array]:
    """Bracket ``query`` on a strictly increasing 1-D ``grid``.

    Returns ``(index, frac)`` such that the interpolated value is
    ``(1 - frac) * table[index] + frac * table[index + 1]``. Queries are
    clamped to the grid range (constant extrapolation), which is a documented
    gradient dead zone with respect to the query outside the table.
    """
    q = jnp.clip(query, grid[0], grid[-1])
    idx = jnp.clip(jnp.searchsorted(grid, q, side="right") - 1, 0, grid.shape[0] - 2)
    frac = (q - grid[idx]) / (grid[idx + 1] - grid[idx])
    return idx, frac


def _multilinear(table: Array, locs: list[tuple[Array, Array]]) -> Array:
    """Multilinear (gather + lerp) interpolation of ``table`` at bracketed points.

    ``locs`` holds one ``(index, frac)`` pair per table axis (all of a common
    broadcast shape ``[...]``); the result sums the ``2^d`` corner values of
    each surrounding cell weighted by the product of per-axis linear weights.
    Differentiable with respect to the fracs (hence the queries) and the table.
    """
    result = jnp.zeros(())
    for corner in itertools.product((0, 1), repeat=len(locs)):
        weight = jnp.ones(())
        idx = []
        for (i, frac), hi in zip(locs, corner, strict=True):
            idx.append(i + hi)
            weight = weight * jnp.where(hi, frac, 1.0 - frac)
        result = result + weight * table[tuple(idx)]
    return result


class TablePolar(eqx.Module):
    """Tabulated airfoil polar on a regular (alpha[, mach][, reynolds]) grid.

    Lookups use multilinear (1-/2-/3-linear) interpolation implemented with
    gather + lerp — differentiable with respect to the query coordinates and
    the table values (no scipy). Grids must be strictly increasing but need
    not be uniformly spaced.

    Queries outside a grid's range are clamped to the boundary (constant
    extrapolation). **Gradient dead zone**: outside the table range the
    derivative with respect to that query coordinate is zero; gradients with
    respect to the table values remain well defined.

    Attributes:
        alpha_grid: Angle-of-attack nodes [rad], shape ``[A]``.
        mach_grid: Optional Mach nodes [-], shape ``[M]``, or ``None``.
        reynolds_grid: Optional Reynolds nodes [-], shape ``[R]``, or ``None``.
        cl_table: Lift coefficients [-], shape ``[A]``, ``[A, M]``, ``[A, R]``,
            or ``[A, M, R]`` matching which grids are present (axis order:
            alpha, then mach, then reynolds).
        cd_table: Drag coefficients [-], same shape as ``cl_table``.
    """

    alpha_grid: Array
    cl_table: Array
    cd_table: Array
    mach_grid: Array | None
    reynolds_grid: Array | None

    def __init__(
        self,
        alpha_grid: ArrayLike,
        cl_table: ArrayLike,
        cd_table: ArrayLike,
        mach_grid: ArrayLike | None = None,
        reynolds_grid: ArrayLike | None = None,
    ):
        """Construct a table polar.

        Args:
            alpha_grid: Angle-of-attack nodes [rad], shape ``[A]``, strictly
                increasing, ``A >= 2``.
            cl_table: Lift coefficient table [-], shape ``[A]`` plus one axis
                per optional grid (mach before reynolds).
            cd_table: Drag coefficient table [-], same shape as ``cl_table``.
            mach_grid: Optional Mach nodes [-], shape ``[M]``, strictly
                increasing, ``M >= 2``.
            reynolds_grid: Optional Reynolds nodes [-], shape ``[R]``,
                strictly increasing, ``R >= 2``.
        """
        self.alpha_grid = jnp.asarray(alpha_grid)
        self.mach_grid = None if mach_grid is None else jnp.asarray(mach_grid)
        self.reynolds_grid = None if reynolds_grid is None else jnp.asarray(reynolds_grid)
        self.cl_table = jnp.asarray(cl_table)
        self.cd_table = jnp.asarray(cd_table)

        expected = tuple(
            g.shape[0]
            for g in (self.alpha_grid, self.mach_grid, self.reynolds_grid)
            if g is not None
        )
        if any(n < 2 for n in expected):
            raise ValueError("every grid must have at least 2 nodes")
        if self.cl_table.shape != expected or self.cd_table.shape != expected:
            raise ValueError(
                f"cl/cd tables must have shape {expected}, got "
                f"{self.cl_table.shape} and {self.cd_table.shape}"
            )

    def __call__(
        self,
        alpha: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        """Interpolate the tables.

        Args:
            alpha: Angle of attack [rad], any shape.
            mach: Section Mach number [-]; required iff the polar has a
                ``mach_grid`` (ignored otherwise). Broadcastable with ``alpha``.
            reynolds: Reynolds number [-]; required iff the polar has a
                ``reynolds_grid`` (ignored otherwise). Broadcastable with
                ``alpha``.

        Returns:
            ``(cl, cd)`` [-, -], each with the broadcast shape of the provided
            query coordinates.

        Raises:
            ValueError: If ``mach``/``reynolds`` is missing while the
                corresponding grid is present.
        """
        grids: list[Array] = [self.alpha_grid]
        queries: list[ArrayLike] = [alpha]
        if self.mach_grid is not None:
            if mach is None:
                raise ValueError("this TablePolar has a mach axis; pass mach=...")
            grids.append(self.mach_grid)
            queries.append(mach)
        if self.reynolds_grid is not None:
            if reynolds is None:
                raise ValueError("this TablePolar has a reynolds axis; pass reynolds=...")
            grids.append(self.reynolds_grid)
            queries.append(reynolds)

        qs = jnp.broadcast_arrays(*[jnp.asarray(q) for q in queries])
        locs = [_locate(g, q) for g, q in zip(grids, qs, strict=True)]
        return _multilinear(self.cl_table, locs), _multilinear(self.cd_table, locs)
