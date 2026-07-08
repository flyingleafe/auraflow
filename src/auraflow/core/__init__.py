"""AuraFlow core: frames, blade/rotor geometry, airfoil polars, medium.

Shared differentiable building blocks used by all simulation backends
(``auraflow.bemt``, ``auraflow.cona``, ``auraflow.cfd``). See
``docs/architecture.md`` for frame, shape, and unit conventions.
"""

from auraflow.core.airfoil import TablePolar, ThinAirfoilPolar
from auraflow.core.blade import BladeGeometry, Rotor, Vehicle
from auraflow.core.frames import (
    azimuth_at,
    euler_zyx_matrix,
    integrate_azimuth,
    interp1d,
    rot_x,
    rot_y,
    rot_z,
)
from auraflow.core.medium import Medium

__all__ = [
    "BladeGeometry",
    "Medium",
    "Rotor",
    "TablePolar",
    "ThinAirfoilPolar",
    "Vehicle",
    "azimuth_at",
    "euler_zyx_matrix",
    "integrate_azimuth",
    "interp1d",
    "rot_x",
    "rot_y",
    "rot_z",
]
