"""Canonical-SDF composition: build one part's SDF, reuse it at every pose.

The reuse core behind issue #2. Building a mesh signed-distance grid is the
wall-clock bottleneck of resolved-body CFD, and a rotor is *the same blade*
repeated at several azimuths (and re-run at many RPMs, and -- ultimately -- at a
time-varying RPM). So we build the expensive thing **once**: a
:class:`CanonicalSDF` for a single blade in its own tight box, then place it
analytically.

- :class:`CanonicalSDF` -- a gridded SDF for one part with differentiable
  ``eval(points)``: trilinear inside its box and a safe, monotone far field
  outside (clamped-to-box trilinear value **plus** the Euclidean distance from
  the point to the box). The far field is a slight over-estimate of the true
  distance, which is exactly what a union ``min`` wants (a part never wrongly
  claims a point that belongs to a nearer part), and it is exact near the
  surface where it matters.
- :func:`compose_union` -- ``min`` over parts of each part's canonical SDF
  evaluated at the inverse of that part's placement transform.
- :func:`rotor_sdf` -- a rotor's SDF as a closure over ONE canonical blade
  evaluated at ``R_axis(-psi_k)`` for each blade azimuth ``psi_k``, unioned with
  an analytic capped-cylinder hub. So one small (cheap-to-build) blade SDF
  serves every azimuth, blade count and initial condition. The SDF itself is
  **RPM-independent**; only the *initial azimuth* enters the initial level-set.
- :func:`capped_cylinder_sdf` -- the analytic hub primitive (negative inside).

Sign convention matches :mod:`auraflow.body.sdf`: **negative inside**.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from auraflow.body.mesh import TriMesh
from auraflow.body.motion import axis_angle_matrix
from auraflow.body.sdf import cached_sdf_grid, sdf_eval, sdf_grid

__all__ = [
    "CanonicalSDF",
    "capped_cylinder_sdf",
    "compose_union",
    "rotor_sdf",
]

# A 3-vector accepted as an array or a plain float triple (user convenience),
# matching auraflow.body.motion.Vec3.
Vec3 = ArrayLike | tuple[float, float, float]

# A placement transform: rotation ``R`` [3, 3] and translation ``t`` [3] mapping
# a canonical-frame point ``x`` to world ``R @ x + t``.
Transform = tuple[Array, Array]


class CanonicalSDF(eqx.Module):
    """A gridded signed-distance field for ONE part, reusable at any pose.

    Attributes:
        grid: Signed distances [m] on the part's box, shape ``[Nx, Ny, Nz]``
            (negative inside).
        box_lo: Lower box corner [m], shape ``[3]``.
        box_hi: Upper box corner [m], shape ``[3]``.
    """

    grid: Array
    box_lo: Array
    box_hi: Array

    def __init__(self, grid: ArrayLike, box_lo: ArrayLike, box_hi: ArrayLike):
        self.grid = jnp.asarray(grid, dtype=jnp.float64)
        self.box_lo = jnp.asarray(box_lo, dtype=jnp.float64)
        self.box_hi = jnp.asarray(box_hi, dtype=jnp.float64)

    @classmethod
    def from_mesh(
        cls,
        mesh: TriMesh,
        *,
        padding: float | tuple[float, float, float] = 0.1,
        cells: int | tuple[int, int, int] = 48,
        method: str = "jax",
        cache: bool = True,
        cache_dir: str | None = None,
        batch_points: int = 4096,
    ) -> CanonicalSDF:
        """Build a canonical SDF for ``mesh`` in a tight, padded box.

        The box is the mesh vertex bounding box grown by ``padding`` on each side
        (generous padding keeps the surface well inside, so the trilinear field
        is accurate right up to the box faces and the far-field approximation
        only kicks in where the SDF is already large). The grid build goes
        through :func:`auraflow.body.sdf.cached_sdf_grid` (disk-memoized) by
        default.

        Args:
            mesh: The single-part :class:`~auraflow.body.mesh.TriMesh` (e.g. one
                blade in its section frame).
            padding: Padding [m] added to each side of the vertex bbox; a scalar
                or a per-axis triple.
            cells: SDF grid node counts per axis (int or triple).
            method: SDF build method (see :func:`auraflow.body.sdf.sdf_grid`).
            cache: Memoize the grid on disk (:func:`cached_sdf_grid`).
            cache_dir: Cache directory override.
            batch_points: Query-point chunk for the JAX build (memory knob).

        Returns:
            A :class:`CanonicalSDF`.
        """
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        pad = np.broadcast_to(np.asarray(padding, dtype=np.float64), (3,))
        lo = verts.min(axis=0) - pad
        hi = verts.max(axis=0) + pad
        if cache:
            grid = cached_sdf_grid(
                mesh, lo, hi, cells, method=method, cache_dir=cache_dir, batch_points=batch_points
            )
        else:
            grid = sdf_grid(mesh, lo, hi, cells, method=method, batch_points=batch_points)
        return cls(grid, lo, hi)

    def eval(self, points: ArrayLike) -> Array:
        """Signed distance at ``points`` (trilinear inside, safe far field outside).

        Inside the box the value is the trilinear interpolation of the grid.
        Outside, the point is clamped to the box, the trilinear value at that
        boundary point is taken, and the Euclidean distance from the point to the
        box is added -- a monotone, slightly conservative extension suitable for
        ``min``-composition. Differentiable in ``points``.

        Args:
            points: Query point(s) [m], shape ``[3]`` (single) or ``[S, 3]``.

        Returns:
            Signed distance(s) [m], scalar or shape ``[S]``.
        """
        pts = jnp.asarray(points, dtype=jnp.float64)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]
        clamped = jnp.clip(pts, self.box_lo, self.box_hi)
        inside_val = sdf_eval(self.grid, self.box_lo, self.box_hi, clamped)  # [S]
        outside = jnp.linalg.norm(pts - clamped, axis=-1)  # [S], 0 inside the box
        val = inside_val + outside
        return val[0] if single else val


def capped_cylinder_sdf(
    points: ArrayLike,
    *,
    center: Vec3 = (0.0, 0.0, 0.0),
    axis: Vec3 = (0.0, 0.0, 1.0),
    radius: ArrayLike = 1.0,
    half_height: ArrayLike = 0.5,
) -> Array:
    """Analytic signed distance to a finite (capped) cylinder, negative inside.

    The cylinder is centred at ``center`` with its symmetry axis along ``axis``,
    of the given ``radius`` and half-length ``half_height`` along the axis. Exact
    and differentiable; no mesh or grid needed -- used as the rotor hub.

    Args:
        points: Query point(s) [m], shape ``[3]`` or ``[S, 3]``.
        center: Cylinder centre [m], shape ``[3]``.
        axis: Cylinder axis (normalized internally), shape ``[3]``.
        radius: Cylinder radius [m], scalar.
        half_height: Half length along the axis [m], scalar.

    Returns:
        Signed distance(s) [m], scalar or shape ``[S]`` (negative inside).
    """
    pts = jnp.asarray(points, dtype=jnp.float64)
    single = pts.ndim == 1
    if single:
        pts = pts[None, :]
    ahat = jnp.asarray(axis, dtype=jnp.float64)
    ahat = ahat / jnp.linalg.norm(ahat)
    d = pts - jnp.asarray(center, dtype=jnp.float64)
    along = d @ ahat  # [S]
    radial = jnp.linalg.norm(d - along[:, None] * ahat, axis=-1)  # [S]
    dz = jnp.abs(along) - jnp.asarray(half_height, dtype=jnp.float64)
    dr = radial - jnp.asarray(radius, dtype=jnp.float64)
    outside = jnp.sqrt(jnp.maximum(dr, 0.0) ** 2 + jnp.maximum(dz, 0.0) ** 2)
    inside = jnp.minimum(jnp.maximum(dr, dz), 0.0)
    val = outside + inside
    return val[0] if single else val


def compose_union(
    parts_and_transforms: Sequence[tuple[CanonicalSDF, Array, Array]],
    points: ArrayLike,
) -> Array:
    """Union (``min``) of several placed canonical SDFs at ``points``.

    Each entry is ``(part, R, t)`` placing the canonical part at world
    ``R @ x + t``; the part is evaluated at the inverse-transformed points
    ``R^T (world - t)`` and the per-part signed distances are ``min``-reduced (a
    union of solids). Differentiable in the points and transforms.

    Args:
        parts_and_transforms: ``(CanonicalSDF, R [3, 3], t [3])`` per part.
        points: Query point(s) [m], shape ``[3]`` (single) or ``[S, 3]``.

    Returns:
        Signed distance(s) [m] of the union, scalar or shape ``[S]``.
    """
    pts = jnp.asarray(points, dtype=jnp.float64)
    single = pts.ndim == 1
    if single:
        pts = pts[None, :]
    vals = []
    for part, r, t in parts_and_transforms:
        r = jnp.asarray(r, dtype=jnp.float64)
        t = jnp.asarray(t, dtype=jnp.float64)
        local = (pts - t) @ r  # (R^T (p - t))^T == (p - t) @ R
        vals.append(part.eval(local))
    out = jnp.min(jnp.stack(vals, axis=0), axis=0)
    return out[0] if single else out


def rotor_sdf(
    blade_sdf: CanonicalSDF,
    n_blades: int,
    azimuth: ArrayLike = 0.0,
    *,
    axis: Vec3 = (0.0, 0.0, 1.0),
    center: Vec3 = (0.0, 0.0, 0.0),
    spin_direction: int = 1,
    hub: dict[str, Any] | None = None,
) -> Callable[[ArrayLike], Array]:
    """Whole-rotor SDF as a closure over ONE canonical blade + an analytic hub.

    Blade ``k`` sits at azimuth ``psi_k = azimuth + spin_direction * 2 pi k /
    n_blades`` about ``axis`` through ``center`` (the rotor-frame convention of
    :meth:`auraflow.core.blade.Rotor.blade_azimuths`), so its placement rotation
    is ``R_axis(psi_k)`` and the world point maps into the canonical blade frame
    by ``R_axis(-psi_k) (x - center)``. All blades share the single
    ``blade_sdf``; an optional capped-cylinder hub is unioned in. The returned
    callable evaluates the rotor SDF at arbitrary points -- fast trilinear
    lookups, no mesh queries.

    The rotor SDF is **independent of RPM**: ``azimuth`` is the only kinematic
    input, so one blade SDF serves every rate and every initial condition. For a
    CFD level-set, evaluate this at ``initial_azimuth`` on the cell centres; the
    solver then advects the field with the prescribed (possibly time-varying)
    solid velocity.

    Args:
        blade_sdf: The canonical single-blade :class:`CanonicalSDF` (section
            frame; its box is the rotor frame at ``psi = 0``).
        n_blades: Number of blades ``B`` (>= 1).
        azimuth: Reference-blade azimuth ``psi`` [rad] (scalar).
        axis: Rotation/thrust axis (normalized internally), shape ``[3]``.
        center: A point on the axis (the hub) [m], shape ``[3]``.
        spin_direction: ``+1`` (CCW from ``+axis``) or ``-1`` (CW).
        hub: ``None`` for no hub, or a dict of :func:`capped_cylinder_sdf`
            params (``radius``, ``half_height``; ``axis``/``center`` default to
            the rotor's).

    Returns:
        ``callable(points) -> signed distance`` (negative inside), accepting
        ``[3]`` or ``[S, 3]`` points.
    """
    if n_blades < 1:
        raise ValueError("n_blades must be >= 1")
    axis_a = jnp.asarray(axis, dtype=jnp.float64)
    center_a = jnp.asarray(center, dtype=jnp.float64)
    psi = jnp.asarray(azimuth, dtype=jnp.float64)
    psis = psi + spin_direction * 2.0 * jnp.pi * jnp.arange(n_blades) / n_blades

    parts: list[tuple[CanonicalSDF, Array, Array]] = []
    for k in range(int(n_blades)):
        r = axis_angle_matrix(axis_a, psis[k])  # canonical -> world rotation
        parts.append((blade_sdf, r, center_a))

    hub_params = None if hub is None else dict(hub)

    def evaluate(points: ArrayLike) -> Array:
        pts = jnp.asarray(points, dtype=jnp.float64)
        val = compose_union(parts, pts)
        if hub_params is not None:
            hub_val = capped_cylinder_sdf(
                pts,
                center=hub_params.get("center", center_a),
                axis=hub_params.get("axis", axis_a),
                radius=hub_params["radius"],
                half_height=hub_params["half_height"],
            )
            val = jnp.minimum(val, hub_val)
        return val

    return evaluate
