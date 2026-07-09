"""3D-model import/export for ``auraflow.body`` (optional ``mesh`` extra).

Loads STL / OBJ / PLY / GLB-GLTF / OFF meshes via `trimesh
<https://trimesh.org>`_ and converts them to a float64 :class:`~auraflow.body.mesh.TriMesh`.
`trimesh` is imported **lazily** so that ``import auraflow.body`` works without
it; only :func:`load_mesh` / :func:`save_mesh` require the ``mesh`` extra
(``pip install 'auraflow[mesh]'``).

At import the mesh is cleaned up: duplicate vertices are merged and (with
``repair=True``, the default) winding/normals are fixed to be consistent and
outward, and watertightness is reported on the resulting :class:`TriMesh`.
numpy is used freely here -- this is the IO boundary, not a differentiated path.
"""

from os import PathLike
from typing import Any

import jax.numpy as jnp
import numpy as np

from auraflow.body.mesh import TriMesh

__all__ = ["load_mesh", "save_mesh"]


def _import_trimesh() -> "Any":
    """Import trimesh, or raise a clear error naming the ``mesh`` extra."""
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "auraflow.body mesh import/export requires the optional 'mesh' extra. "
            "Install it with:  pip install 'auraflow[mesh]'  (adds trimesh>=4)."
        ) from exc
    return trimesh


def load_mesh(path: str | PathLike[str], *, repair: bool = True) -> TriMesh:
    """Load a 3D model file into a float64 :class:`TriMesh`.

    Supports any single-geometry format trimesh handles (STL, OBJ, PLY,
    GLB/GLTF, OFF). Scenes are concatenated into one mesh; duplicate vertices
    are merged (``process=True``). With ``repair=True`` the winding is made
    consistent and normals are flipped outward (``trimesh.repair``), matching
    the outward-winding invariant of :class:`TriMesh`.

    Args:
        path: Path to the model file.
        repair: If ``True`` (default) merge duplicate vertices, fix winding and
            orient normals outward before conversion.

    Returns:
        A :class:`TriMesh` with float64 vertices; :attr:`TriMesh.is_watertight`
        reflects the (possibly repaired) loaded geometry.
    """
    trimesh = _import_trimesh()
    mesh = trimesh.load(str(path), force="mesh", process=True)
    if repair:
        mesh.merge_vertices()
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return TriMesh(jnp.asarray(vertices), faces, is_watertight=bool(mesh.is_watertight))


def save_mesh(mesh: TriMesh, path: str | PathLike[str]) -> None:
    """Write a :class:`TriMesh` to a 3D model file (format from the extension).

    Args:
        mesh: The mesh to export.
        path: Output path; the extension selects the format (``.stl``, ``.obj``,
            ``.ply``, ``.glb``, ``.off``).
    """
    trimesh = _import_trimesh()
    tm = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )
    tm.export(str(path))
