"""Mesh -> signed distance field for ``auraflow.body``.

Builds a Cartesian signed-distance grid from a :class:`~auraflow.body.mesh.TriMesh`
(via ``trimesh.proximity`` at setup, numpy allowed) and evaluates it with
differentiable trilinear interpolation. Feeds JAX-Fluids level-set solids
(resolved bodies in CFD) and the viz layer.

**Sign convention**: the field is **negative inside** the body, zero on the
surface, positive outside -- the standard level-set solid convention. trimesh's
``signed_distance`` is positive *inside*, so it is negated here.

``trimesh`` is imported lazily (optional ``mesh`` extra) inside :func:`sdf_grid`
only; :func:`sdf_eval` is pure JAX and needs neither trimesh nor the grid's
provenance. Grid construction uses numpy (setup boundary); evaluation is
differentiable with respect to the query points.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from auraflow.body.mesh import TriMesh
from auraflow.cfd.sphere import trilinear_interpolate

__all__ = ["sdf_eval", "sdf_grid"]


def _import_trimesh() -> "Any":
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "auraflow.body.sdf.sdf_grid requires the optional 'mesh' extra. "
            "Install it with:  pip install 'auraflow[mesh]'  (adds trimesh>=4)."
        ) from exc
    return trimesh


def _axes(box_lo: ArrayLike, box_hi: ArrayLike, cells: int | tuple[int, int, int]):
    lo = np.asarray(box_lo, dtype=np.float64)
    hi = np.asarray(box_hi, dtype=np.float64)
    n = (cells, cells, cells) if isinstance(cells, int) else tuple(cells)
    xs = np.linspace(lo[0], hi[0], n[0])
    ys = np.linspace(lo[1], hi[1], n[1])
    zs = np.linspace(lo[2], hi[2], n[2])
    return xs, ys, zs


def sdf_grid(
    mesh: TriMesh,
    box_lo: ArrayLike,
    box_hi: ArrayLike,
    cells: int | tuple[int, int, int],
    *,
    batch_size: int = 4096,
) -> Array:
    """Signed-distance grid sampled on a uniform Cartesian box.

    Nodes are placed at ``linspace(box_lo[i], box_hi[i], cells_i)`` on each axis
    (endpoints inclusive), so :func:`sdf_eval` reproduces the grid exactly at the
    nodes. Uses ``trimesh.proximity.signed_distance`` (exact, numpy) at setup,
    evaluated in ``batch_size`` chunks of query points: trimesh's closest-point
    machinery allocates candidate matrices proportional to the query count, and
    an unchunked multi-million-node grid (or even a 24^3 grid against a rotor
    mesh on the small dev box) exhausts host RAM.

    Args:
        mesh: The body :class:`TriMesh` (used for its vertices/faces geometry).
        box_lo: Lower box corner [m], shape ``[3]``.
        box_hi: Upper box corner [m], shape ``[3]``.
        cells: Node counts per axis; an int applies to all three axes.
        batch_size: Query points per ``signed_distance`` call (memory knob;
            result is identical for any value).

    Returns:
        Signed distances [m], shape ``[Nx, Ny, Nz]``, **negative inside** the
        body and positive outside (level-set convention).
    """
    trimesh = _import_trimesh()
    tm = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )
    xs, ys, zs = _axes(box_lo, box_hi, cells)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    # trimesh: +inside, -outside. Negate for the -inside level-set convention.
    out = np.empty(pts.shape[0], dtype=np.float64)
    for lo_i in range(0, pts.shape[0], batch_size):
        chunk = pts[lo_i : lo_i + batch_size]
        out[lo_i : lo_i + chunk.shape[0]] = -np.asarray(
            trimesh.proximity.signed_distance(tm, chunk), dtype=np.float64
        )
    return jnp.asarray(out.reshape(gx.shape))


def sdf_eval(
    grid: ArrayLike,
    box_lo: ArrayLike,
    box_hi: ArrayLike,
    points: ArrayLike,
) -> Array:
    """Differentiable trilinear evaluation of a signed-distance grid.

    Reuses the trilinear interpolation of :func:`auraflow.cfd.sphere.trilinear_interpolate`
    (``map_coordinates`` order 1, ``nearest`` clamping outside the box). The
    gradient with respect to ``points`` is finite everywhere and approximates the
    unit outward direction away from the surface (``|grad SDF| = 1``).

    Args:
        grid: Signed-distance grid from :func:`sdf_grid`, shape ``[Nx, Ny, Nz]``.
        box_lo: Lower box corner [m], shape ``[3]`` (same as used for the grid).
        box_hi: Upper box corner [m], shape ``[3]``.
        points: Query point(s) [m], shape ``[3]`` (single) or ``[S, 3]``.

    Returns:
        Signed distance(s) [m], scalar for a single point or shape ``[S]``.
    """
    grid = jnp.asarray(grid)
    pts = jnp.asarray(points, dtype=jnp.float64)
    single = pts.ndim == 1
    if single:
        pts = pts[None, :]
    nx, ny, nz = grid.shape
    lo = jnp.asarray(box_lo, dtype=jnp.float64)
    hi = jnp.asarray(box_hi, dtype=jnp.float64)
    xs = jnp.linspace(lo[0], hi[0], nx)
    ys = jnp.linspace(lo[1], hi[1], ny)
    zs = jnp.linspace(lo[2], hi[2], nz)
    vals = trilinear_interpolate(grid, xs, ys, zs, pts)  # [S]
    return vals[0] if single else vals
