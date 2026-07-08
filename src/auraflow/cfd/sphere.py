"""Permeable data surface: a static sphere sampled from the CFD grid.

The full-CFD backend runs a compressible near-field simulation inside a
Cartesian box and hands the far-field propagation to the permeable-surface
FW-H solver (:func:`auraflow.fwh.f1a_permeable_static`). The coupling surface is
a **static** sphere enclosing the rotor(s): the blades move *inside* it, but the
surface itself never moves, so the closed-form-delay fast path applies.

This module provides

- :func:`fibonacci_sphere` / :class:`PermeableSphere`: a near-uniform spherical
  point set (Fibonacci/spiral lattice) with per-point **outward** unit normals
  and **equal-area** weights, i.e. exactly the ``y_panels [S, 3]``,
  ``normal [S, 3]``, ``area [S]`` arrays the FW-H solver needs.
- :func:`trilinear_interpolate` / :func:`sample_primitives`: differentiable
  trilinear interpolation of the CFD primitive fields ``(rho, u, p)`` from the
  cell-centred Cartesian grid onto the sphere points. Pure JAX -- usable as a
  per-step sampling callback inside the simulation driver
  (:mod:`auraflow.cfd.run`).

Frames and units follow ``docs/architecture.md``: SI throughout (m, s, kg, Pa),
world frame, trailing ``xyz`` axis. The sphere geometry is static (Python int
``n_points``); the centre/radius and all sampled fields are traced JAX arrays.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

__all__ = [
    "PermeableSphere",
    "fibonacci_sphere",
    "sample_primitives",
    "trilinear_interpolate",
]

# Golden angle used by the Fibonacci spiral lattice [rad].
_GOLDEN_ANGLE = jnp.pi * (3.0 - jnp.sqrt(jnp.asarray(5.0)))


def fibonacci_sphere(
    n_points: int,
    radius: ArrayLike = 1.0,
    center: ArrayLike | tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[Array, Array, Array]:
    """Near-uniform spherical point set (Fibonacci/spiral lattice).

    Points are placed at equal increments of the golden angle in azimuth and at
    equal spacing in ``z`` (equal-area bands), giving a near-uniform tiling of
    the sphere with no clustering at the poles. Each point is assigned the same
    area weight ``4 pi radius^2 / n_points`` so the weights sum exactly to the
    sphere area; the outward normal is the radial unit vector.

    Args:
        n_points: Number of surface points ``S`` (static int, ``>= 1``).
        radius: Sphere radius [m], scalar.
        center: Sphere centre [m], shape ``[3]``.

    Returns:
        ``(points, normals, area)`` with

        - ``points`` [m], shape ``[S, 3]`` -- the ``y_panels`` for FW-H;
        - ``normals``, shape ``[S, 3]`` -- outward unit normals;
        - ``area`` [m^2], shape ``[S]`` -- equal per-point weights summing to
          ``4 pi radius^2``.
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    radius = jnp.asarray(radius)
    center = jnp.asarray(center)
    i = jnp.arange(n_points)
    # Equal-area latitude bands: z in (-1, 1), centred sample per band.
    z = 1.0 - 2.0 * (i + 0.5) / n_points
    r_xy = jnp.sqrt(jnp.clip(1.0 - z * z, 0.0, 1.0))
    phi = _GOLDEN_ANGLE * i
    unit = jnp.stack([r_xy * jnp.cos(phi), r_xy * jnp.sin(phi), z], axis=-1)  # [S, 3]
    points = center + radius * unit
    area = jnp.full((n_points,), 4.0 * jnp.pi * radius**2 / n_points)
    return points, unit, area


class PermeableSphere(eqx.Module):
    """Static permeable FW-H data surface (a sampled sphere).

    Bundles the geometry the permeable-surface FW-H solver consumes. Construct
    with :meth:`fibonacci`. The point count is static; centre, radius, and the
    per-point arrays are traced (differentiable, e.g. for sensitivity of the
    far-field noise to the surface radius).

    Attributes:
        points: Surface point positions [m], shape ``[S, 3]`` (``y_panels``).
        normals: Outward unit normals, shape ``[S, 3]``.
        area: Per-point area weights [m^2], shape ``[S]``.
        center: Sphere centre [m], shape ``[3]``.
        radius: Sphere radius [m], scalar.
        n_points: Number of surface points ``S`` (static int).
    """

    points: Array
    normals: Array
    area: Array
    center: Array
    radius: Array
    n_points: int = eqx.field(static=True)

    @classmethod
    def fibonacci(
        cls,
        n_points: int,
        radius: ArrayLike = 1.0,
        center: ArrayLike | tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "PermeableSphere":
        """Build a Fibonacci-lattice permeable sphere.

        Args:
            n_points: Number of surface points ``S`` (static int).
            radius: Sphere radius [m], scalar. Choose so the sphere encloses the
                rotor tip vortices but sits inside the sponge-free core of the
                CFD box.
            center: Sphere centre [m], shape ``[3]`` (usually the rotor hub).

        Returns:
            A :class:`PermeableSphere`.
        """
        points, normals, area = fibonacci_sphere(n_points, radius, center)
        return cls(
            points=points,
            normals=normals,
            area=area,
            center=jnp.asarray(center),
            radius=jnp.asarray(radius),
            n_points=n_points,
        )


def _axis_index_coords(coord: Array, grid: Array) -> Array:
    """Fractional index coordinates of ``coord`` on a uniform 1-D ``grid``.

    ``grid`` holds monotonically increasing, uniformly spaced cell centres; the
    returned value is ``(coord - grid[0]) / dx`` (``dx = grid[1] - grid[0]``),
    i.e. the continuous index into ``grid`` used by ``map_coordinates``.
    """
    dx = grid[1] - grid[0]
    return (coord - grid[0]) / dx


def trilinear_interpolate(
    field: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    points: ArrayLike,
) -> Array:
    """Trilinear interpolation of a gridded field onto scattered points.

    Differentiable (pure JAX, ``jax.scipy.ndimage.map_coordinates`` order 1)
    with respect to both ``field`` and ``points``. Queries outside the grid are
    clamped to the nearest cell centre (``mode="nearest"``); keep the sphere
    strictly inside the CFD domain so no clamping occurs on physical points.

    Args:
        field: Cell-centred field on a uniform Cartesian grid. Shape
            ``[Nx, Ny, Nz]`` (scalar) or ``[Nx, Ny, Nz, C]`` (``C`` channels,
            interpolated channel-wise).
        x: Cell-centre coordinates along axis 0 [m], shape ``[Nx]``, uniform.
        y: Cell-centre coordinates along axis 1 [m], shape ``[Ny]``, uniform.
        z: Cell-centre coordinates along axis 2 [m], shape ``[Nz]``, uniform.
        points: Query points [m], shape ``[S, 3]``.

    Returns:
        Interpolated values, shape ``[S]`` (scalar field) or ``[S, C]``
        (multi-channel field).
    """
    field = jnp.asarray(field)
    points = jnp.asarray(points)
    ix = _axis_index_coords(points[:, 0], jnp.asarray(x))
    iy = _axis_index_coords(points[:, 1], jnp.asarray(y))
    iz = _axis_index_coords(points[:, 2], jnp.asarray(z))
    idx = jnp.stack([ix, iy, iz], axis=0)  # [3, S]

    def sample_scalar(f: Array) -> Array:
        return jax.scipy.ndimage.map_coordinates(f, list(idx), order=1, mode="nearest")

    if field.ndim == 3:
        return sample_scalar(field)
    if field.ndim == 4:
        return jax.vmap(sample_scalar, in_axes=-1, out_axes=-1)(field)
    raise ValueError(f"field must have 3 or 4 dims, got shape {field.shape}")


def sample_primitives(
    primitives: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    points: ArrayLike,
) -> tuple[Array, Array, Array]:
    """Interpolate JAX-Fluids primitive variables onto sphere points.

    JAX-Fluids stores single-phase primitives as a leading-variable-axis array
    ``[5, Nx, Ny, Nz]`` ordered ``(rho, u, v, w, p)`` (interior cells, halos
    already stripped -- see :mod:`auraflow.cfd.run`). This splits and
    trilinearly interpolates them onto the surface points.

    Args:
        primitives: Interior primitive field, shape ``[5, Nx, Ny, Nz]``
            ordered ``(rho, u, v, w, p)``.
        x, y, z: Uniform cell-centre coordinates [m], shapes ``[Nx]``,
            ``[Ny]``, ``[Nz]``.
        points: Surface point positions [m], shape ``[S, 3]``.

    Returns:
        ``(rho, u, p)`` at the points: density [kg/m^3] shape ``[S]``, velocity
        [m/s] shape ``[S, 3]``, pressure [Pa] shape ``[S]``.
    """
    primitives = jnp.asarray(primitives)
    # Move the variable axis to the trailing channel axis for interpolation.
    field = jnp.moveaxis(primitives, 0, -1)  # [Nx, Ny, Nz, 5]
    sampled = trilinear_interpolate(field, x, y, z, points)  # [S, 5]
    rho = sampled[:, 0]
    u = sampled[:, 1:4]
    p = sampled[:, 4]
    return rho, u, p
