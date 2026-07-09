"""AuraFlow full-CFD backend: compressible near field + permeable-surface FW-H.

A compressible JAX-Fluids simulation is run in a Cartesian box around the
rotor(s); the flow ``(rho, u, p)`` is sampled on a **static permeable sphere**
enclosing the sources every few steps, and the far field is obtained by feeding
that surface history to the permeable-surface Farassat 1A solver
(:func:`auraflow.fwh.f1a_permeable_static`). See ``docs/architecture.md`` and the
research digests ``docs/research/jaxfluids-evaluation.md`` and
``docs/research/cfd-fwh-reference.md``.

JAX-Fluids is an **optional** dependency (extra ``cfd``, pinned in
``pyproject.toml``). It is imported lazily inside
:func:`~auraflow.cfd.run.run_acoustic_case`, so ``import auraflow.cfd`` and all
geometry/case builders work without the extra; only actually running a CFD case
requires it.

Public API
----------
Case builders and resolution helpers (:mod:`auraflow.cfd.case`):

- :func:`acoustic_box_case`, :func:`rotor_box_case`
- :class:`CFDCase`, :class:`BoxDomain`
- :func:`resolution_for_frequency`, :func:`acoustic_timestep`,
  :func:`points_per_wavelength`

Permeable sphere (:mod:`auraflow.cfd.sphere`):

- :class:`PermeableSphere`, :func:`fibonacci_sphere`
- :func:`trilinear_interpolate`, :func:`sample_primitives`

Driver (:mod:`auraflow.cfd.run`):

- :func:`run_acoustic_case`, :func:`propagate_to_observers`
- :class:`SurfaceHistory`
"""

from auraflow.cfd.body_case import (
    LevelsetBodyCase,
    PermeableMeshSurface,
    levelset_body_case,
    permeable_mesh_surface,
)
from auraflow.cfd.case import (
    BoxDomain,
    CFDCase,
    acoustic_box_case,
    acoustic_timestep,
    points_per_wavelength,
    resolution_for_frequency,
    rotor_box_case,
)
from auraflow.cfd.flyover import (
    quadrotor_surface_flyover,
    synthesize_flyover_wavs,
    tile_surface_history,
)
from auraflow.cfd.run import (
    SurfaceHistory,
    propagate_to_observers,
    run_acoustic_case,
)
from auraflow.cfd.sphere import (
    PermeableSphere,
    fibonacci_sphere,
    sample_primitives,
    trilinear_interpolate,
)

__all__ = [
    "BoxDomain",
    "CFDCase",
    "LevelsetBodyCase",
    "PermeableMeshSurface",
    "PermeableSphere",
    "SurfaceHistory",
    "acoustic_box_case",
    "acoustic_timestep",
    "fibonacci_sphere",
    "levelset_body_case",
    "permeable_mesh_surface",
    "points_per_wavelength",
    "propagate_to_observers",
    "quadrotor_surface_flyover",
    "resolution_for_frequency",
    "rotor_box_case",
    "run_acoustic_case",
    "sample_primitives",
    "synthesize_flyover_wavs",
    "tile_surface_history",
    "trilinear_interpolate",
]
