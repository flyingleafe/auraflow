"""Mesh -> signed distance field for ``auraflow.body``.

Two ways to build a Cartesian signed-distance grid from a
:class:`~auraflow.body.mesh.TriMesh`:

- :func:`sdf_grid_jax` (the **default**): a chunked, brute-force, pure-JAX
  point->triangle distance with a **generalized-winding-number** sign. It runs
  correctly on CPU (the tests) and *fast* on GPU (deployment) -- the whole
  grid-points x faces reduction is one vectorized kernel, jitted once and looped
  over point chunks in Python. This is the answer to the CFD-bottleneck issue
  (#2): the trimesh path took ~1h46m for a 192^3-ish grid against a ~15k-face
  rotor on one CPU core; the JAX kernel is seconds on any GPU.
- the legacy ``trimesh.proximity`` path (``method="trimesh"``): exact but
  single-threaded CPU, kept for cross-validation.

:func:`sdf_grid` dispatches on ``method=`` (default ``"jax"``); everything that
called ``sdf_grid`` before (``cfd.body_case`` -> the CFD level-set) now goes
through the JAX kernel automatically.

**Sign convention**: the field is **negative inside** the body, zero on the
surface, positive outside -- the standard level-set solid convention (and what
JAX-Fluids ingests directly). The winding-number sign is robust for watertight
meshes and, crucially, for *thin* watertight bodies (rotor blades), where naive
ray-parity is fragile.

**Precision**: distances are accumulated in the query points' dtype (float64
when ``jax_enable_x64`` is on, as in the tests and acoustics; float32 on a GPU
deployment without x64 -- fine for a level-set field whose cell size dwarfs the
float32 epsilon). The grid is returned as float64 when x64 is enabled.

:func:`sdf_eval` is pure JAX and needs neither trimesh nor the grid's
provenance; its gradient w.r.t. the query points is finite everywhere and
approximates the unit outward direction away from the surface (``|grad| = 1``).

:func:`cached_sdf_grid` memoizes a built grid on disk keyed by a content hash of
(vertices, faces, box, cells, method-version), so repeat runs (and multi-case
GPU jobs) skip the build entirely.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from auraflow.body.mesh import TriMesh
from auraflow.cfd.sphere import trilinear_interpolate

__all__ = [
    "cached_sdf_grid",
    "sdf_eval",
    "sdf_grid",
    "sdf_grid_jax",
    "winding_number",
]

# Bump when the numerics of the JAX build change so cached grids invalidate.
_SDF_METHOD_VERSION = "jax-bruteforce-winding-v1"


def _import_trimesh() -> Any:
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "auraflow.body.sdf.sdf_grid(method='trimesh') requires the optional 'mesh' "
            "extra. Install it with:  pip install 'auraflow[mesh]'  (adds trimesh>=4). "
            "The default method='jax' has no such dependency."
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


# --- Pure-JAX brute-force kernels ------------------------------------------


def _dot(u: Array, v: Array) -> Array:
    return jnp.sum(u * v, axis=-1)


def _closest_point_on_triangle(p: Array, a: Array, b: Array, c: Array) -> Array:
    """Closest point on triangle ``(a, b, c)`` to ``p`` (Ericson's regions).

    Fully vectorized (broadcasting over any leading dims): ``p, a, b, c`` are
    ``[..., 3]`` and the result is ``[..., 3]``. The seven Voronoi regions
    (three vertices, three edges, interior) are selected with ``where`` in
    priority order, so it is branch-free and robust for obtuse triangles --
    unlike a plain barycentric clamp.
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    bp = p - b
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    cp = p - c
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    denom_safe = jnp.where(denom == 0.0, 1.0, denom)
    v = vb / denom_safe
    w = vc / denom_safe
    interior = a + v[..., None] * ab + w[..., None] * ac

    def edge(t_num: Array, t_den: Array, base: Array, dir_: Array) -> Array:
        t = t_num / jnp.where(t_den == 0.0, 1.0, t_den)
        t = jnp.clip(t, 0.0, 1.0)
        return base + t[..., None] * dir_

    pt_ab = edge(d1, d1 - d3, a, ab)
    pt_ac = edge(d2, d2 - d6, a, ac)
    pt_bc = edge(d4 - d3, (d4 - d3) + (d5 - d6), b, c - b)

    cond_a = (d1 <= 0.0) & (d2 <= 0.0)
    cond_b = (d3 >= 0.0) & (d4 <= d3)
    cond_c = (d6 >= 0.0) & (d5 <= d6)
    cond_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    cond_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    cond_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)

    out = interior
    out = jnp.where(cond_bc[..., None], pt_bc, out)
    out = jnp.where(cond_ac[..., None], pt_ac, out)
    out = jnp.where(cond_ab[..., None], pt_ab, out)
    out = jnp.where(cond_c[..., None], c, out)
    out = jnp.where(cond_b[..., None], b, out)
    out = jnp.where(cond_a[..., None], a, out)
    return out


def _point_tri_dist2(p: Array, a: Array, b: Array, c: Array) -> Array:
    """Squared distance from ``p`` to triangle ``(a, b, c)``, broadcast over ``[...]``."""
    q = _closest_point_on_triangle(p, a, b, c)
    d = p - q
    return _dot(d, d)


def _solid_angle(p: Array, a: Array, b: Array, c: Array) -> Array:
    """Signed solid angle of triangle ``(a, b, c)`` seen from ``p`` (van Oosterom-Strackee).

    Broadcasts over ``[...]``; ``[..., 3]`` inputs -> ``[...]`` output [sr]. With
    outward (CCW-from-outside) winding, the solid angles of a closed mesh sum to
    ``+4 pi`` at interior points and ``0`` at exterior points, so
    ``sum / (4 pi)`` is the generalized winding number (``~1`` inside, ``~0``
    outside).
    """
    av = a - p
    bv = b - p
    cv = c - p
    la = jnp.linalg.norm(av, axis=-1)
    lb = jnp.linalg.norm(bv, axis=-1)
    lc = jnp.linalg.norm(cv, axis=-1)
    numer = _dot(av, jnp.cross(bv, cv))
    denom = la * lb * lc + _dot(av, bv) * lc + _dot(bv, cv) * la + _dot(cv, av) * lb
    return 2.0 * jnp.arctan2(numer, denom)


def _tri_arrays(mesh: TriMesh) -> tuple[Array, Array, Array]:
    tris = mesh.vertices[mesh.faces_array()]  # [F, 3, 3]
    return tris[:, 0], tris[:, 1], tris[:, 2]


# Default cap on ``batch_points * face_batch`` (point-face pairs held live in a
# single kernel), chosen so the ~15 transient ``[pairs, 3]`` float arrays of the
# distance/winding kernels stay well under the dev box's ~1 GB budget. GPU
# deployments can raise it (fewer, larger launches) via the ``max_pairs`` knob.
_MAX_PAIRS = 1 << 20


def _iter_point_chunks(pts: Array, batch_points: int):
    n = int(pts.shape[0])
    bp = max(1, int(batch_points))
    for i in range(0, n, bp):
        yield pts[i : i + bp]


def _iter_face_chunks(a: Array, b: Array, c: Array, face_batch: int):
    f = int(a.shape[0])
    fb = max(1, int(face_batch))
    for i in range(0, f, fb):
        yield a[i : i + fb], b[i : i + fb], c[i : i + fb]


def _face_batch(batch_points: int, n_faces: int, max_pairs: int) -> int:
    return int(np.clip(max(1, max_pairs // max(1, batch_points)), 1, n_faces))


def _dist_and_winding(
    mesh: TriMesh, points: Array, batch_points: int, max_pairs: int
) -> tuple[Array, Array]:
    """Min unsigned distance and winding number of ``mesh`` at ``points``.

    Doubly chunked: an outer Python loop over ``batch_points`` query points and
    an inner Python loop over ``face_batch`` faces (``batch_points * face_batch``
    bounded by ``max_pairs``), accumulating the running min squared distance and
    the running solid-angle sum. Peak memory is therefore ``~ batch_points *
    face_batch`` regardless of the mesh size or grid size -- the property the
    memory-capped dev box needs. The per-block kernel is jitted (compiled once
    for the full block shape, once more for any short trailing block).
    """
    a, b, c = _tri_arrays(mesh)
    fb = _face_batch(batch_points, int(a.shape[0]), max_pairs)

    @jax.jit
    def block(chunk: Array, af: Array, bf: Array, cf: Array) -> tuple[Array, Array]:
        pe = chunk[:, None, :]  # [B, 1, 3]
        d2 = _point_tri_dist2(pe, af[None], bf[None], cf[None])  # [B, fb]
        w = jnp.sum(_solid_angle(pe, af[None], bf[None], cf[None]), axis=1)  # [B]
        return jnp.min(d2, axis=1), w

    dist_parts: list[Array] = []
    wind_parts: list[Array] = []
    for chunk in _iter_point_chunks(points, batch_points):
        d2min = jnp.full((chunk.shape[0],), jnp.inf, dtype=chunk.dtype)
        wsum = jnp.zeros((chunk.shape[0],), dtype=chunk.dtype)
        for af, bf, cf in _iter_face_chunks(a, b, c, fb):
            d2, w = block(chunk, af, bf, cf)
            d2min = jnp.minimum(d2min, d2)
            wsum = wsum + w
        dist_parts.append(jnp.sqrt(d2min))
        wind_parts.append(wsum / (4.0 * jnp.pi))
    return jnp.concatenate(dist_parts), jnp.concatenate(wind_parts)


def winding_number(
    mesh: TriMesh,
    points: ArrayLike,
    *,
    batch_points: int = 4096,
    max_pairs: int = _MAX_PAIRS,
) -> Array:
    """Generalized winding number of ``mesh`` at ``points`` (chunked, pure JAX).

    Sum of the signed solid angles of every triangle divided by ``4 pi``. For a
    watertight, outward-wound mesh it is ``~1`` strictly inside, ``~0`` strictly
    outside, and varies smoothly across the surface -- robust even for very thin
    bodies (rotor blades) where ray-parity tests are unreliable.

    Args:
        mesh: The body :class:`~auraflow.body.mesh.TriMesh`.
        points: Query point(s) [m], shape ``[3]`` (single) or ``[S, 3]``.
        batch_points: Query points per jitted kernel call (memory knob).
        max_pairs: Cap on ``batch_points * face_batch`` point-face pairs held
            live (memory knob; raise on a big GPU). Result is invariant.

    Returns:
        Winding number(s), scalar for a single point or shape ``[S]``.
    """
    pts = jnp.asarray(points, dtype=mesh.vertices.dtype)
    single = pts.ndim == 1
    if single:
        pts = pts[None, :]
    _, w = _dist_and_winding(mesh, pts, batch_points, max_pairs)
    return w[0] if single else w


def sdf_grid_jax(
    mesh: TriMesh,
    box_lo: ArrayLike,
    box_hi: ArrayLike,
    cells: int | tuple[int, int, int],
    *,
    batch_points: int = 4096,
    max_pairs: int = _MAX_PAIRS,
) -> Array:
    """GPU-parallel signed-distance grid via brute-force distance + winding sign.

    Nodes are placed at ``linspace(box_lo[i], box_hi[i], cells_i)`` per axis
    (endpoints inclusive), identical to :func:`sdf_grid`, so :func:`sdf_eval`
    reproduces the grid exactly at the nodes. For each grid point the unsigned
    distance is ``min`` over all faces of the exact point->triangle distance
    (Ericson's robust closest-point-on-triangle), and the sign comes from the
    generalized winding number (:func:`winding_number`; ``w > 1/2`` <=> inside
    <=> negative). The reduction is one vectorized ``[batch_points, face_batch]``
    kernel, jitted once and looped over point/face chunks in Python.

    Args:
        mesh: The body :class:`~auraflow.body.mesh.TriMesh`.
        box_lo: Lower box corner [m], shape ``[3]``.
        box_hi: Upper box corner [m], shape ``[3]``.
        cells: Node counts per axis; an int applies to all three axes.
        batch_points: Query points per kernel call (memory/latency knob).
        max_pairs: Cap on the ``batch_points * face_batch`` point-face pairs held
            live in one kernel; peak device memory scales with it (default keeps
            the dev box under ~0.5 GB). Raise it on a big GPU for fewer, larger
            launches. The result is identical for any value.

    Returns:
        Signed distances [m], shape ``[Nx, Ny, Nz]``, **negative inside** the
        body and positive outside (level-set convention).
    """
    xs, ys, zs = _axes(box_lo, box_hi, cells)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = jnp.asarray(
        np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1), dtype=mesh.vertices.dtype
    )
    dist, w = _dist_and_winding(mesh, pts, batch_points, max_pairs)
    sign = jnp.where(w > 0.5, -1.0, 1.0)
    return jnp.asarray((dist * sign).reshape(gx.shape), dtype=jnp.float64)


def _sdf_grid_trimesh(
    mesh: TriMesh,
    box_lo: ArrayLike,
    box_hi: ArrayLike,
    cells: int | tuple[int, int, int],
    *,
    batch_size: int = 4096,
) -> Array:
    """Legacy exact SDF via ``trimesh.proximity.signed_distance`` (CPU, numpy).

    Single-threaded and slow, kept only for cross-validation against
    :func:`sdf_grid_jax`. Requires the optional ``mesh`` extra. Chunked in
    ``batch_size`` query points because trimesh's closest-point machinery
    allocates candidate matrices proportional to the query count.
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


def sdf_grid(
    mesh: TriMesh,
    box_lo: ArrayLike,
    box_hi: ArrayLike,
    cells: int | tuple[int, int, int],
    *,
    method: str = "jax",
    batch_points: int = 4096,
    batch_size: int = 4096,
) -> Array:
    """Signed-distance grid on a uniform Cartesian box (dispatches on ``method``).

    Nodes are at ``linspace(box_lo[i], box_hi[i], cells_i)`` (endpoints
    inclusive), so :func:`sdf_eval` reproduces the grid exactly at the nodes.
    **Negative inside** the body (level-set convention).

    Args:
        mesh: The body :class:`~auraflow.body.mesh.TriMesh`.
        box_lo: Lower box corner [m], shape ``[3]``.
        box_hi: Upper box corner [m], shape ``[3]``.
        cells: Node counts per axis; an int applies to all three axes.
        method: ``"jax"`` (default) -> :func:`sdf_grid_jax` (GPU brute-force +
            winding number, no trimesh); ``"trimesh"`` -> the exact single-thread
            ``trimesh.proximity`` path (optional ``mesh`` extra), for
            cross-validation.
        batch_points: Query-point chunk for the ``"jax"`` method (memory knob).
        batch_size: Query-point chunk for the ``"trimesh"`` method (memory knob).

    Returns:
        Signed distances [m], shape ``[Nx, Ny, Nz]``.
    """
    if method == "jax":
        return sdf_grid_jax(mesh, box_lo, box_hi, cells, batch_points=batch_points)
    if method == "trimesh":
        return _sdf_grid_trimesh(mesh, box_lo, box_hi, cells, batch_size=batch_size)
    raise ValueError(f"sdf_grid method must be 'jax' or 'trimesh', got {method!r}")


# --- Disk cache ------------------------------------------------------------


def _resolve_cache_dir(cache_dir: str | os.PathLike[str] | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    env = os.environ.get("AURAFLOW_SDF_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "auraflow" / "sdf"


def _sdf_cache_key(
    mesh: TriMesh,
    box_lo: ArrayLike,
    box_hi: ArrayLike,
    cells: int | tuple[int, int, int],
    method: str,
) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(mesh.vertices, dtype=np.float64)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(mesh.faces_array(), dtype=np.int64)).tobytes())
    h.update(np.asarray(box_lo, dtype=np.float64).tobytes())
    h.update(np.asarray(box_hi, dtype=np.float64).tobytes())
    n = (cells, cells, cells) if isinstance(cells, int) else tuple(cells)
    h.update(np.asarray(n, dtype=np.int64).tobytes())
    h.update(f"{method}:{_SDF_METHOD_VERSION}".encode())
    return h.hexdigest()


def cached_sdf_grid(
    mesh: TriMesh,
    box_lo: ArrayLike,
    box_hi: ArrayLike,
    cells: int | tuple[int, int, int],
    *,
    method: str = "jax",
    cache_dir: str | os.PathLike[str] | None = None,
    batch_points: int = 4096,
) -> Array:
    """Build (or load) an SDF grid, memoized on disk by a content hash.

    The cache key is a SHA-256 of the mesh vertices, faces, box corners, cell
    counts and the method (plus an internal method-version tag), so any change
    to geometry or resolution produces a fresh entry and stale numerics never
    survive a version bump. Grids are stored as float32 ``.npz`` (~13 MB at the
    192-scale CFD grid). The default directory is ``~/.cache/auraflow/sdf``,
    overridable by the ``cache_dir`` argument or the ``AURAFLOW_SDF_CACHE``
    environment variable.

    On ephemeral GPU workers (e.g. Kaggle) the cache pays off *within* a job --
    a multi-case sweep that reuses one mesh/box builds the grid once; jobs that
    want to reuse it across runs stage the ``.npz`` into ``results/`` and pull it
    back. Canonical blade SDFs (:mod:`auraflow.body.sdf_compose`) go through this
    same cache.

    Args:
        mesh: The body :class:`~auraflow.body.mesh.TriMesh`.
        box_lo: Lower box corner [m], shape ``[3]``.
        box_hi: Upper box corner [m], shape ``[3]``.
        cells: Node counts per axis; an int applies to all three axes.
        method: SDF build method (see :func:`sdf_grid`).
        cache_dir: Cache directory override (else ``AURAFLOW_SDF_CACHE`` or the
            default ``~/.cache/auraflow/sdf``).
        batch_points: Query-point chunk for the ``"jax"`` build (memory knob).

    Returns:
        Signed distances [m], shape ``[Nx, Ny, Nz]`` (negative inside).
    """
    cdir = _resolve_cache_dir(cache_dir)
    key = _sdf_cache_key(mesh, box_lo, box_hi, cells, method)
    path = cdir / f"{key}.npz"
    if path.exists():
        with np.load(path) as data:
            return jnp.asarray(data["grid"], dtype=jnp.float64)
    grid = sdf_grid(mesh, box_lo, box_hi, cells, method=method, batch_points=batch_points)
    cdir.mkdir(parents=True, exist_ok=True)
    # Write to a unique temp name that still ends in ``.npz`` (np.savez_compressed
    # appends ``.npz`` otherwise), then atomically rename into place.
    tmp = cdir / f"{key}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, grid=np.asarray(grid, dtype=np.float32))
    os.replace(tmp, path)
    return grid


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
