"""Triangle-mesh geometry for general 3D bodies (``auraflow.body``).

See ``docs/architecture.md`` -> "Generalization principle (v2)" and the
``auraflow.body`` module spec. A :class:`TriMesh` is the discretization that
every acoustic adapter consumes: area-weighted, single-point panel quadrature
at each face centroid.

Conventions (``docs/architecture.md`` -> "Library conventions"):

- SI units (m), right-handed world frame, trailing ``xyz`` axis.
- **Winding**: triangles are wound counterclockwise seen from *outside* the
  body, so the per-face normal ``cross(v1-v0, v2-v0)`` points **outward**. The
  parametric primitives guarantee this (verified by ``volume() > 0``).
- **Compactness assumption**: each panel is treated as a single monopole/dipole
  at its centroid, valid only when panels are much smaller than the acoustic
  wavelength. Refine the mesh, not the quadrature.

Static vs traced: ``vertices [V, 3]`` is a traced, differentiable JAX leaf;
``faces [F, 3]`` (connectivity) and ``is_watertight`` are static (topology,
fixed structure). Face count ``F`` and vertex count ``V`` are static.
"""

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

__all__ = ["TriMesh"]

# Golden ratio for the icosahedron seed of the icosphere.
_PHI = (1.0 + np.sqrt(5.0)) / 2.0

# Canonical icosahedron (12 vertices, 20 faces). Faces are re-oriented outward
# at build time, so the raw winding here need not be perfect.
_ICO_VERTS = np.array(
    [
        [-1, _PHI, 0], [1, _PHI, 0], [-1, -_PHI, 0], [1, -_PHI, 0],
        [0, -1, _PHI], [0, 1, _PHI], [0, -1, -_PHI], [0, 1, -_PHI],
        [_PHI, 0, -1], [_PHI, 0, 1], [-_PHI, 0, -1], [-_PHI, 0, 1],
    ],
    dtype=np.float64,
)  # fmt: skip
_ICO_FACES = np.array(
    [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ],
    dtype=np.int64,
)  # fmt: skip


def _is_watertight(faces: np.ndarray) -> bool:
    """True iff every edge is shared by exactly two faces (closed manifold).

    Pure-numpy topological check used at construction time (no geometry, no
    tracing). Robust to winding: edges are compared as unordered vertex pairs.
    """
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2)) and len(faces) > 0


def _orient_outward(vertices: np.ndarray, faces: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Flip faces so every normal points away from ``center`` (numpy, setup only).

    Valid for star-convex bodies about ``center`` (all AuraFlow primitives).
    Returns a new ``faces`` array with outward winding.
    """
    tris = vertices[faces]  # [F, 3, 3]
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    nrm = np.cross(e1, e2)
    cent = tris.mean(axis=1)
    outward = np.einsum("ij,ij->i", cent - center, nrm) >= 0.0
    faces = faces.copy()
    faces[~outward] = faces[~outward][:, ::-1]
    return faces


class TriMesh(eqx.Module):
    """A watertight-or-open triangle mesh with area-weighted panel quadrature.

    Attributes:
        vertices: Vertex positions [m], shape ``[V, 3]`` (traced float64 leaf,
            differentiable).
        faces: Triangle vertex indices ``[F, 3]`` int, static topology with
            outward winding. Stored as a hashable tuple of index triples (so the
            mesh is a valid static argument to ``jax.jit``); use
            :meth:`faces_array` for the ``[F, 3]`` array form.
        is_watertight: Whether the mesh is a closed 2-manifold (static bool).
    """

    vertices: Array
    faces: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    is_watertight: bool = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        faces: ArrayLike | Sequence[Sequence[int]],
        *,
        is_watertight: bool | None = None,
    ):
        """Build a mesh from vertices and triangle connectivity.

        Args:
            vertices: Vertex positions [m], shape ``[V, 3]``. Cast to float64.
            faces: Triangle vertex indices, shape ``[F, 3]`` (0-based).
            is_watertight: Override the closed-manifold flag; if ``None`` it is
                derived from the connectivity (every edge shared by two faces).
        """
        self.vertices = jnp.asarray(vertices, dtype=jnp.float64)
        faces_np = np.asarray(faces, dtype=np.int64)
        if faces_np.ndim != 2 or faces_np.shape[1] != 3:
            raise ValueError(f"faces must have shape [F, 3], got {faces_np.shape}")
        self.faces = tuple(tuple(int(i) for i in tri) for tri in faces_np)
        self.is_watertight = (
            _is_watertight(faces_np) if is_watertight is None else bool(is_watertight)
        )

    def faces_array(self) -> Array:
        """Triangle connectivity as an integer array, shape ``[F, 3]``."""
        return jnp.asarray(self.faces, dtype=jnp.int64)

    @property
    def n_faces(self) -> int:
        """Number of triangular faces ``F`` (static int)."""
        return len(self.faces)

    @property
    def n_vertices(self) -> int:
        """Number of vertices ``V`` (static int)."""
        return int(self.vertices.shape[0])

    def _triangles(self) -> Array:
        """Per-face vertex coordinates, shape ``[F, 3, 3]`` (face, corner, xyz)."""
        return self.vertices[self.faces_array()]

    def centroids(self) -> Array:
        """Per-face centroids (panel quadrature points) [m], shape ``[F, 3]``."""
        return self._triangles().mean(axis=1)

    def _face_cross(self) -> Array:
        """Per-face ``(v1-v0) x (v2-v0)``, shape ``[F, 3]`` (outward, area-scaled)."""
        tris = self._triangles()
        e1 = tris[:, 1] - tris[:, 0]
        e2 = tris[:, 2] - tris[:, 0]
        return jnp.cross(e1, e2)

    def areas(self) -> Array:
        """Per-face areas [m^2], shape ``[F]``."""
        return 0.5 * jnp.linalg.norm(self._face_cross(), axis=-1)

    def normals(self) -> Array:
        """Per-face outward unit normals, shape ``[F, 3]``.

        The direction follows the outward-winding convention; zero-area faces
        would yield NaNs (primitives never produce them).
        """
        cross = self._face_cross()
        return cross / jnp.linalg.norm(cross, axis=-1, keepdims=True)

    def total_area(self) -> Array:
        """Total surface area [m^2], scalar."""
        return jnp.sum(self.areas())

    def volume(self) -> Array:
        """Signed enclosed volume [m^3], scalar (divergence theorem).

        ``V = (1/6) sum_f v0 . (v1 x v2)``. Positive for an outward-wound closed
        mesh; meaningful only when :attr:`is_watertight`.
        """
        tris = self._triangles()
        v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
        return jnp.sum(jnp.sum(v0 * jnp.cross(v1, v2), axis=-1)) / 6.0

    # --- Parametric primitives (outward winding guaranteed) ------------------

    @classmethod
    def sphere(cls, radius: ArrayLike = 1.0, subdivisions: int = 2) -> "TriMesh":
        """Icosphere: a recursively subdivided icosahedron projected to a sphere.

        Args:
            radius: Sphere radius [m], scalar (traced, differentiable).
            subdivisions: Number of 1-to-4 subdivision passes (static int ``>= 0``).
                ``F = 20 * 4**subdivisions``.

        Returns:
            A watertight :class:`TriMesh` centred at the origin with outward
            normals; area -> ``4 pi r^2`` and volume -> ``(4/3) pi r^3`` as the
            subdivision level increases.
        """
        if subdivisions < 0:
            raise ValueError(f"subdivisions must be >= 0, got {subdivisions}")
        verts = _ICO_VERTS / np.linalg.norm(_ICO_VERTS, axis=1, keepdims=True)
        faces = _ICO_FACES
        for _ in range(subdivisions):
            verts, faces = _subdivide(verts, faces)
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        faces = _orient_outward(verts, faces, np.zeros(3))
        return cls(jnp.asarray(radius, dtype=jnp.float64) * jnp.asarray(verts), faces)

    @classmethod
    def box(cls, extents: ArrayLike | tuple[float, float, float] = (1.0, 1.0, 1.0)) -> "TriMesh":
        """Axis-aligned box centred at the origin.

        Args:
            extents: Full side lengths ``(lx, ly, lz)`` [m], shape ``[3]`` or a
                scalar (cube). Traced, differentiable.

        Returns:
            A watertight :class:`TriMesh` (8 vertices, 12 faces, outward normals).
        """
        ext = jnp.broadcast_to(jnp.asarray(extents, dtype=jnp.float64), (3,))
        unit = np.array(
            [
                [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        )  # fmt: skip
        quads = [
            (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
            (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
        ]  # fmt: skip
        faces = np.array(
            [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], dtype=np.int64
        )
        faces = _orient_outward(unit, faces, np.zeros(3))
        return cls(ext * jnp.asarray(unit), faces)

    @classmethod
    def disk(cls, radius: ArrayLike = 1.0, n: int = 32) -> "TriMesh":
        """Flat circular disk in the ``z = 0`` plane, normals along ``+z``.

        Open (single-sided) surface, so it is not watertight. Used e.g. as a
        loudspeaker membrane or a piston baffle patch.

        Args:
            radius: Disk radius [m], scalar (traced).
            n: Number of rim segments (static int ``>= 3``).

        Returns:
            A :class:`TriMesh` with ``n`` triangles fanned from the centre.
        """
        if n < 3:
            raise ValueError(f"n must be >= 3, got {n}")
        theta = np.arange(n) * (2.0 * np.pi / n)
        rim = np.stack([np.cos(theta), np.sin(theta), np.zeros(n)], axis=-1)
        verts = np.concatenate([np.zeros((1, 3)), rim], axis=0)  # centre + rim
        faces = np.array([[0, 1 + k, 1 + (k + 1) % n] for k in range(n)], dtype=np.int64)
        return cls(
            jnp.asarray(radius, dtype=jnp.float64) * jnp.asarray(verts), faces, is_watertight=False
        )

    @classmethod
    def cylinder(cls, radius: ArrayLike = 1.0, height: ArrayLike = 1.0, n: int = 32) -> "TriMesh":
        """Closed right circular cylinder about the ``z`` axis, centred at origin.

        Args:
            radius: Cylinder radius [m], scalar (traced).
            height: Cylinder height along ``z`` [m], scalar (traced).
            n: Number of angular segments (static int ``>= 3``).

        Returns:
            A watertight :class:`TriMesh` (side quads + top/bottom caps) with
            outward normals.
        """
        if n < 3:
            raise ValueError(f"n must be >= 3, got {n}")
        theta = np.arange(n) * (2.0 * np.pi / n)
        ring = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
        bottom = np.concatenate([ring, np.full((n, 1), -0.5)], axis=1)
        top = np.concatenate([ring, np.full((n, 1), 0.5)], axis=1)
        c_bot = np.array([[0.0, 0.0, -0.5]])
        c_top = np.array([[0.0, 0.0, 0.5]])
        verts = np.concatenate([bottom, top, c_bot, c_top], axis=0)
        i_cb, i_ct = 2 * n, 2 * n + 1
        faces = []
        for k in range(n):
            k1 = (k + 1) % n
            b0, b1, t0, t1 = k, k1, n + k, n + k1
            faces += [(b0, b1, t1), (b0, t1, t0)]  # side quad
            faces.append((i_cb, k1, k))  # bottom cap fan
            faces.append((i_ct, n + k, n + k1))  # top cap fan
        faces = np.array(faces, dtype=np.int64)
        faces = _orient_outward(verts, faces, np.zeros(3))
        scale = jnp.stack(
            [
                jnp.asarray(radius, dtype=jnp.float64),
                jnp.asarray(radius, dtype=jnp.float64),
                jnp.asarray(height, dtype=jnp.float64),
            ]
        )
        return cls(scale * jnp.asarray(verts), faces)

    @classmethod
    def flat_plate(cls, chord: ArrayLike = 1.0, span: ArrayLike = 1.0) -> "TriMesh":
        """Flat rectangular plate in ``z = 0`` (chord along x, span along y).

        Open surface (not watertight); the two triangles share the ``+z`` normal.

        Args:
            chord: Streamwise length along x [m], scalar (traced).
            span: Lateral length along y [m], scalar (traced).

        Returns:
            A :class:`TriMesh` with 2 triangles, consistent ``+z`` normals.
        """
        unit = np.array(
            [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        scale = jnp.stack(
            [
                jnp.asarray(chord, dtype=jnp.float64),
                jnp.asarray(span, dtype=jnp.float64),
                jnp.asarray(1.0),
            ]
        )
        return cls(scale * jnp.asarray(unit), faces, is_watertight=False)


def _subdivide(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One 1-to-4 loop subdivision (numpy, setup only), preserving winding.

    Each triangle ``(a, b, c)`` gains edge midpoints and becomes four triangles.
    Midpoints are de-duplicated across shared edges via a cache.
    """
    verts = list(vertices)
    cache: dict[tuple[int, int], int] = {}

    def midpoint(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        if key not in cache:
            cache[key] = len(verts)
            verts.append((vertices[i] + vertices[j]) / 2.0)
        return cache[key]

    new_faces = []
    for a, b, c in faces:
        ab = midpoint(a, b)
        bc = midpoint(b, c)
        ca = midpoint(c, a)
        new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.array(verts, dtype=np.float64), np.array(new_faces, dtype=np.int64)
