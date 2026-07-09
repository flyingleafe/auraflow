"""General 3D-body core: meshes, model import, motion, SDF, FW-H sources.

``auraflow.body`` generalizes the library from rotors to *any* 3D body or
surface moving in (or vibrating within) a compressible fluid -- imported 3D
models, loudspeaker membranes, airframes. See ``docs/architecture.md`` ->
"Generalization principle (v2)".

The pipeline is

    TriMesh (imported or parametric)
      x Motion (rigid pose(t) [+ optional surface vibration u_n(face, t)])
      -> panel_histories -> FW-H source adapters -> auraflow.fwh -> signals

Public API
----------
- Geometry: :class:`TriMesh`.
- Import/export (optional ``mesh`` extra, lazy trimesh):
  :func:`load_mesh`, :func:`save_mesh`.
- Kinematics: :class:`Motion`, :class:`StaticPose`, :class:`ConstantVelocity`,
  :class:`SpinMotion`, :class:`WaypointMotion`, :class:`ComposedMotion`,
  :class:`SurfaceVibration`, :func:`pose_derivatives`, :func:`panel_histories`,
  :class:`PanelHistories`.
- FW-H sources: :func:`permeable_surface`, :func:`impermeable_sources`,
  :func:`mesh_pressure` (the one-call mesh radiation path).
- Loudspeaker: :class:`Speaker`, :func:`circular_piston`, :func:`select_faces`.
- Signed distance: :func:`sdf_grid` (GPU brute-force + winding number by
  default), :func:`sdf_grid_jax`, :func:`winding_number`, :func:`cached_sdf_grid`
  (disk-memoized), :func:`sdf_eval`.

``import auraflow.body`` works without trimesh installed; only
:func:`load_mesh` / :func:`save_mesh` / ``sdf_grid(method="trimesh")`` require
the ``mesh`` extra (the default ``sdf_grid`` is pure JAX).
"""

from auraflow.body.airfoil_profile import naca0012, naca4_profile
from auraflow.body.blade import blade_mesh, rotor_levelset_case, rotor_mesh
from auraflow.body.io import load_mesh, save_mesh
from auraflow.body.mesh import TriMesh
from auraflow.body.motion import (
    ComposedMotion,
    ConstantVelocity,
    HarmonicTranslation,
    Motion,
    PanelHistories,
    SpinMotion,
    StaticPose,
    SurfaceVibration,
    WaypointMotion,
    panel_histories,
    pose_derivatives,
)
from auraflow.body.sdf import (
    cached_sdf_grid,
    sdf_eval,
    sdf_grid,
    sdf_grid_jax,
    winding_number,
)
from auraflow.body.sources import impermeable_sources, mesh_pressure, permeable_surface
from auraflow.body.speaker import Speaker, circular_piston, select_faces

__all__ = [
    "ComposedMotion",
    "ConstantVelocity",
    "HarmonicTranslation",
    "Motion",
    "PanelHistories",
    "Speaker",
    "SpinMotion",
    "StaticPose",
    "SurfaceVibration",
    "TriMesh",
    "WaypointMotion",
    "blade_mesh",
    "cached_sdf_grid",
    "circular_piston",
    "impermeable_sources",
    "load_mesh",
    "mesh_pressure",
    "naca0012",
    "naca4_profile",
    "panel_histories",
    "permeable_surface",
    "pose_derivatives",
    "rotor_levelset_case",
    "rotor_mesh",
    "save_mesh",
    "sdf_eval",
    "sdf_grid",
    "sdf_grid_jax",
    "select_faces",
    "winding_number",
]
