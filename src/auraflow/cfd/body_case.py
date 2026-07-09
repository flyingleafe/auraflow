"""Level-set FLUID-SOLID bodies and permeable mesh surfaces for the CFD backend.

Bridges :mod:`auraflow.body` (general 3D meshes + motion) to the compressible
JAX-Fluids near-field solver:

- :func:`levelset_body_case` builds a JAX-Fluids **FLUID-SOLID level-set** case
  whose immersed solid is an arbitrary closed :class:`~auraflow.body.mesh.TriMesh`.
  The initial level-set field is the body's signed-distance grid
  (:func:`auraflow.body.sdf.sdf_grid`) sampled at the CFD cell centres and handed
  to the driver, which injects it via ``InitializationManager.initialization(
  user_levelset_init=...)`` -- so *any* mesh works, not just closed-form shapes.
- :func:`permeable_mesh_surface` wraps :func:`auraflow.body.permeable_surface`
  into a :class:`PermeableMeshSurface` that duck-types
  :class:`~auraflow.cfd.sphere.PermeableSphere` (``points``/``normals``/``area``),
  so :func:`auraflow.cfd.run.run_acoustic_case` can sample the flow on *any*
  closed mesh, not only the Fibonacci sphere.

Sign convention (verified against JAX-Fluids 0.2.1 source,
``jaxfluids/levelset/creation/generic_shapes.py`` and ``mask_functions.py``):
JAX-Fluids' level-set is **negative inside the solid** and positive in the fluid
(a circle of radius ``a`` is ``phi = -a + |x - c|``; the *fluid* mask is
``volume_fraction > 0``, i.e. ``phi > 0``). :func:`auraflow.body.sdf.sdf_grid` is
already negative inside the body (the solid), so **no sign flip is applied** --
the SDF grid is the level-set field directly.

Supported envelope (JAX-Fluids v0.2.1, see
``docs/research/jaxfluids-evaluation.md`` and the docstrings below):

- **Static solid** (``StaticPose`` or ``motion=None``): fully supported.
- **Prescribed moving solid** (rigid translation/rotation): supported via
  ``solid_coupling.dynamic = "ONE-WAY"`` with a prescribed solid-velocity field
  ``(x, y, z, t) -> (u, v, w)`` derived from the motion, for the *analytic*
  rigid motions (:class:`~auraflow.body.motion.ConstantVelocity`,
  :class:`~auraflow.body.motion.HarmonicTranslation`, constant-rate
  :class:`~auraflow.body.motion.SpinMotion`) whose world velocity field is
  closed-form. Other motions raise ``NotImplementedError`` with guidance (pass an
  explicit ``solid_velocity`` override, or precompute).
- **Two-way (fluid-driven) rigid-body dynamics** is out of scope (JAX-Fluids
  ``solid_coupling.dynamic = "TWO-WAY"``; a roadmap item).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from auraflow.body.mesh import TriMesh
from auraflow.body.motion import (
    ConstantVelocity,
    HarmonicTranslation,
    Motion,
    SpinMotion,
    StaticPose,
)
from auraflow.body.sources import permeable_surface
from auraflow.cfd.case import (
    BoxDomain,
    CFDCase,
    _material_properties,
    _numerical_setup,
    _outer_bcs,
    _quiescent_primitives,
    _sponge_forcing,
    acoustic_timestep,
)
from auraflow.core.medium import Medium

# A 3-vector accepted as an array or a plain float triple (user convenience),
# matching auraflow.body.motion.Vec3.
Vec3 = ArrayLike | tuple[float, float, float]

__all__ = [
    "LevelsetBodyCase",
    "PermeableMeshSurface",
    "levelset_body_case",
    "permeable_mesh_surface",
]


class PermeableMeshSurface(eqx.Module):
    """Permeable FW-H data surface backed by a closed :class:`TriMesh`.

    The general-mesh counterpart of :class:`~auraflow.cfd.sphere.PermeableSphere`:
    it exposes the identical ``points``/``normals``/``area`` fields (face
    centroids, outward unit normals, per-face areas), so the CFD driver
    (:func:`auraflow.cfd.run.run_acoustic_case`) and the permeable-surface FW-H
    solver consume it interchangeably. Trilinear sampling of the CFD grid onto
    ``points`` uses the same :func:`auraflow.cfd.sphere.sample_primitives` path.

    Attributes:
        points: Face-centroid positions [m], shape ``[F, 3]`` (``y_panels``).
        normals: Outward unit normals, shape ``[F, 3]``.
        area: Per-face area weights [m^2], shape ``[F]``.
        n_points: Number of surface points ``F`` (static int).
    """

    points: Array
    normals: Array
    area: Array
    n_points: int = eqx.field(static=True)

    @classmethod
    def from_mesh(cls, mesh: TriMesh) -> PermeableMeshSurface:
        """Build the permeable surface from a (closed) triangle mesh.

        Args:
            mesh: The permeable-surface :class:`TriMesh` (should be watertight so
                the outward normals/areas form a closed data surface enclosing the
                sources).

        Returns:
            A :class:`PermeableMeshSurface`.
        """
        points, normals, area = permeable_surface(mesh)
        return cls(points=points, normals=normals, area=area, n_points=mesh.n_faces)


def permeable_mesh_surface(mesh: TriMesh) -> PermeableMeshSurface:
    """Permeable CFD data surface from any closed mesh (see :class:`PermeableMeshSurface`)."""
    return PermeableMeshSurface.from_mesh(mesh)


@dataclass(frozen=True)
class LevelsetBodyCase(CFDCase):
    """A JAX-Fluids FLUID-SOLID level-set case for an immersed body.

    Extends :class:`~auraflow.cfd.case.CFDCase` with the initial level-set field
    the driver injects via ``user_levelset_init`` (the body SDF sampled at the
    CFD cell centres; **negative inside the solid**, JAX-Fluids convention) and a
    flag recording whether the solid is prescribed-moving.

    Attributes:
        levelset_init: Initial level-set grid [m], shape ``[nx, ny, nz]`` at the
            interior cell centres (negative inside the body). ``None`` only if the
            case was built without a mesh (not produced by
            :func:`levelset_body_case`).
        is_moving: ``True`` for a prescribed-moving solid (``solid_coupling.dynamic
            = "ONE-WAY"``), ``False`` for a static solid.
    """

    levelset_init: Array | None = None
    is_moving: bool = False


def _cell_center_sdf(mesh: TriMesh, domain: BoxDomain) -> Array:
    """Body SDF sampled at the interior cell centres, shape ``[nx, ny, nz]``.

    The cell centres are ``linspace(lo + dx/2, hi - dx/2, n)`` per axis (matching
    :meth:`BoxDomain.cell_centers`), so passing the shifted corners to
    :func:`auraflow.body.sdf.sdf_grid` places its inclusive-endpoint node grid
    exactly on the cell centres. The result is negative inside the body -- the
    JAX-Fluids level-set field verbatim (no sign flip; see the module docstring).
    """
    # Imported here (not at module top) to avoid an import cycle: auraflow.body.sdf
    # imports auraflow.cfd.sphere, which triggers auraflow.cfd (this package).
    from auraflow.body.sdf import sdf_grid

    lo = np.asarray([domain.x_range[0], domain.y_range[0], domain.z_range[0]])
    hi = np.asarray([domain.x_range[1], domain.y_range[1], domain.z_range[1]])
    n = np.asarray(domain.cells)
    d = (hi - lo) / n
    cc_lo = lo + 0.5 * d
    cc_hi = hi - 0.5 * d
    return sdf_grid(mesh, cc_lo, cc_hi, domain.cells)


def _solid_velocity_field(
    motion: Motion, solid_velocity: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Prescribed solid-velocity ``(x, y, z, t)`` callables for a moving solid.

    Returns ``None`` for a static solid (``StaticPose``). For the analytic rigid
    motions the world velocity field is closed-form and returned as JAX-Fluids
    ``solid_properties.velocity`` entries -- floats (constant) or stringified
    ``jnp`` lambdas of ``(x, y, z, t)``. An explicit ``solid_velocity`` override
    (a ``{"u", "v", "w"}`` dict of floats/lambda-strings) takes precedence and
    supports any prescribed field.

    Raises:
        NotImplementedError: for a moving motion whose velocity field is not
            closed-form here (e.g. :class:`~auraflow.body.motion.WaypointMotion`,
            :class:`~auraflow.body.motion.ComposedMotion`, tabulated
            :class:`~auraflow.body.motion.SpinMotion`) and without an override.
    """
    if solid_velocity is not None:
        return {k: solid_velocity[k] for k in ("u", "v", "w")}
    if isinstance(motion, StaticPose):
        return None
    if isinstance(motion, ConstantVelocity):
        v = np.asarray(motion.v, dtype=float).ravel()
        return {"u": float(v[0]), "v": float(v[1]), "w": float(v[2])}
    if isinstance(motion, HarmonicTranslation):
        d = np.asarray(motion.direction, dtype=float).ravel()
        amp = float(np.asarray(motion.amplitude))
        omega = float(np.asarray(motion.omega))
        c = d * amp * omega  # velocity amplitude per axis: dir * A * omega
        return {
            axis: f"lambda x, y, z, t: {c[i]!r} * jnp.cos({omega!r} * t)"
            for i, axis in enumerate(("u", "v", "w"))
        }
    if isinstance(motion, SpinMotion):
        if motion.omega is None:
            raise NotImplementedError(
                "Tabulated SpinMotion (from_azimuth) is not supported as a "
                "prescribed level-set solid velocity. Use SpinMotion.constant, or "
                "pass an explicit solid_velocity={'u','v','w'} override."
            )
        axis = np.asarray(motion.axis, dtype=float).ravel()
        axis = axis / np.linalg.norm(axis)
        # Plain Python floats: the components are embedded into the lambda
        # strings via repr, and JAX-Fluids evals them with only ``jnp`` in scope,
        # so a numpy scalar would leak "np.float64(...)" (undefined there).
        wv = float(np.asarray(motion.omega)) * axis  # angular velocity vector
        w = [float(x) for x in wv]
        c = [float(x) for x in np.asarray(motion.center, dtype=float).ravel()]
        # Rigid rotation: v(X) = w x (X - center) (steady field for constant rate).
        return {
            "u": f"lambda x, y, z, t: {w[1]!r} * (z - {c[2]!r}) - {w[2]!r} * (y - {c[1]!r})",
            "v": f"lambda x, y, z, t: {w[2]!r} * (x - {c[0]!r}) - {w[0]!r} * (z - {c[2]!r})",
            "w": f"lambda x, y, z, t: {w[0]!r} * (y - {c[1]!r}) - {w[1]!r} * (x - {c[0]!r})",
        }
    raise NotImplementedError(
        f"levelset_body_case does not support prescribed motion of type "
        f"{type(motion).__name__} in JAX-Fluids v0.2.1. Supported analytic motions: "
        "StaticPose, ConstantVelocity, HarmonicTranslation, constant-rate SpinMotion. "
        "For anything else pass an explicit solid_velocity={'u','v','w'} of "
        "(x, y, z, t) jnp-lambda strings, or run the static case."
    )


def levelset_body_case(
    mesh: TriMesh,
    motion: Motion | None = None,
    *,
    box_lo: Vec3,
    box_hi: Vec3,
    cells: tuple[int, int, int],
    medium: Medium | None = None,
    cfl: float = 0.5,
    mach_max: float = 0.0,
    end_time: float | None = None,
    sponge_thickness: float | None = None,
    sponge_sigma: float = 0.5,
    solid_velocity: dict[str, Any] | None = None,
    is_double: bool = False,
    case_name: str = "levelset_body",
) -> LevelsetBodyCase:
    """Build a FLUID-SOLID level-set CFD case with an immersed body mesh.

    The body ``mesh`` becomes a JAX-Fluids level-set solid: its signed-distance
    grid (negative inside the body) is sampled at the CFD cell centres and stored
    on the returned case for the driver to inject as the initial level-set. The
    box is a uniform Cartesian grid ``[box_lo, box_hi]`` with sponge-absorbing
    outer faces (JAX-Fluids has no non-reflecting BC).

    Motion support (JAX-Fluids v0.2.1 -- see the module docstring for the full
    envelope):

    - ``motion=None`` or a :class:`~auraflow.body.motion.StaticPose`: a **static
      solid** (no ``solid_coupling``); fully supported and locally runnable.
    - an analytic rigid motion (:class:`~auraflow.body.motion.ConstantVelocity`,
      :class:`~auraflow.body.motion.HarmonicTranslation`, constant-rate
      :class:`~auraflow.body.motion.SpinMotion`) or an explicit ``solid_velocity``
      override: a **prescribed-moving solid** via ``solid_coupling.dynamic =
      "ONE-WAY"`` with a ``(x, y, z, t)`` solid-velocity field. The level-set is
      advected by that field. This path is resolution-hungry and intended for GPU
      runs; the *case build* is validated but a full local run is not.
    - any other motion raises ``NotImplementedError`` (pass ``solid_velocity``).

    Args:
        mesh: The immersed body :class:`~auraflow.body.mesh.TriMesh` (closed).
        motion: The body :class:`~auraflow.body.motion.Motion` (default: static).
        box_lo: Lower box corner [m], shape ``[3]``.
        box_hi: Upper box corner [m], shape ``[3]``.
        cells: ``(nx, ny, nz)`` cell counts; all must be ``> 1`` (a body is 3-D).
        medium: Ambient :class:`~auraflow.core.medium.Medium` (default ISA).
        cfl: Acoustic CFL for the fixed timestep.
        mach_max: Peak flow Mach number (bounds the timestep; use the solid's peak
            surface Mach for a moving body).
        end_time: Physical end time [s] (default: one box-crossing time). Only used
            by ``InputManager`` validation; the driver marches a fixed step count.
        sponge_thickness: Sponge depth [m] (default ``0.2 *`` min box edge).
        sponge_sigma: Peak sponge strength in ``[0, 1]`` (default 0.5).
        solid_velocity: Explicit prescribed solid-velocity override, a dict
            ``{"u", "v", "w"}`` of floats or ``(x, y, z, t)`` jnp-lambda strings;
            forces the moving path for any motion.
        is_double: Use float64 compute (default float32; FW-H upcasts).
        case_name: JAX-Fluids case name.

    Returns:
        A :class:`LevelsetBodyCase` (a :class:`~auraflow.cfd.case.CFDCase` plus the
        initial level-set grid), ready for :func:`auraflow.cfd.run.run_acoustic_case`.

    Raises:
        ValueError: if any axis has ``<= 1`` cell (a body needs a 3-D box).
        NotImplementedError: for an unsupported moving motion (see above).
    """
    medium = Medium() if medium is None else medium
    lo = np.asarray(box_lo, dtype=float).ravel()
    hi = np.asarray(box_hi, dtype=float).ravel()
    if any(c <= 1 for c in cells):
        raise ValueError(f"levelset bodies need a 3-D box (all cells > 1), got {cells}")
    domain = BoxDomain(
        x_range=(float(lo[0]), float(hi[0])),
        y_range=(float(lo[1]), float(hi[1])),
        z_range=(float(lo[2]), float(hi[2])),
        cells=cells,
    )
    dx = min(domain.spacing())
    dt = acoustic_timestep(dx, float(medium.c0), mach_max=mach_max, cfl=cfl)
    if end_time is None:
        end_time = float(max(hi[i] - lo[i] for i in range(3)) / float(medium.c0))
    if sponge_thickness is None:
        # Plain float: it is embedded into the sponge lambda via repr, so a numpy
        # scalar would leak "np.float64(...)" (undefined in JAX-Fluids' eval scope).
        sponge_thickness = float(0.2 * min(hi[i] - lo[i] for i in range(3)))
    else:
        sponge_thickness = float(sponge_thickness)

    active = domain.active_axes
    velocity_field = _solid_velocity_field(
        StaticPose() if motion is None else motion, solid_velocity
    )
    is_moving = velocity_field is not None

    bcs = _outer_bcs(domain)
    case: dict[str, Any] = {
        "general": {
            "case_name": case_name,
            "end_time": float(end_time),
            "save_dt": float(end_time),
            "save_path": "./results",
        },
        "domain": {
            "x": {"cells": cells[0], "range": list(domain.x_range)},
            "y": {"cells": cells[1], "range": list(domain.y_range)},
            "z": {"cells": cells[2], "range": list(domain.z_range)},
        },
        # With a level-set model JAX-Fluids expects BCs nested per field.
        "boundary_conditions": {"primitives": bcs, "levelset": bcs},
        "initial_condition": {
            "primitives": _quiescent_primitives(medium, active),
            # Overridden at runtime by user_levelset_init (the body SDF grid); a
            # trivial all-fluid placeholder satisfies InputManager validation.
            "levelset": "lambda x, y, z: 0.1 + 0.0 * x",
        },
        "material_properties": _material_properties(medium),
        "forcings": _sponge_forcing(medium, domain, sponge_thickness, sponge_sigma),
    }
    if is_moving:
        assert velocity_field is not None
        case["solid_properties"] = {"velocity": velocity_field}

    numerical = _numerical_setup(dt, is_double=is_double)
    # Level-set needs more conservative halos than geometry halos (interface
    # extension stencil) and the positivity interpolation limiter (sanity checks
    # in jaxfluids/input/numerical_setup/read_levelset.py).
    numerical["conservatives"]["halo_cells"] = 4
    numerical["conservatives"]["positivity"] = {"is_interpolation_limiter": True}
    numerical["active_forcings"] = {"is_sponge_layer_forcing": True}
    levelset_block: dict[str, Any] = {"model": "FLUID-SOLID", "halo_cells": 2}
    if is_moving:
        levelset_block["solid_coupling"] = {"dynamic": "ONE-WAY"}
    numerical["levelset"] = levelset_block

    levelset_init = _cell_center_sdf(mesh, domain)
    return LevelsetBodyCase(
        case=case,
        numerical_setup=numerical,
        domain=domain,
        dt=dt,
        medium=medium,
        levelset_init=levelset_init,
        is_moving=is_moving,
    )
