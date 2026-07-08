"""AuraFlow FW-H acoustic solvers.

Differentiable, vmappable Ffowcs Williams-Hawkings far-field propagation in
JAX (float64), with the volume/quadrupole term dropped (permeable surfaces
enclose the nonlinear region). See ``docs/architecture.md`` and the research
digests under ``docs/research/`` for the governing equations.

Public API
----------
Time / geometry machinery (:mod:`auraflow.fwh.geometry`):

- :func:`radiation_vectors`, :func:`mach_radial`, :func:`doppler_factor`
- :func:`convective_radiation` (Formulation 1C geometry)
- :func:`source_time_derivative`, :func:`arrival_times`, :func:`resample_sum`,
  :func:`default_observer_grid`

Farassat Formulation 1A (:mod:`auraflow.fwh.f1a`):

- :func:`f1a_pressure` -- core moving compact-source kernel (thickness+loading)
- :func:`f1a_loading`, :func:`f1a_thickness` -- convenience wrappers
- :func:`permeable_surface_sources` -- ``(rho, u, p, n, v) -> (Q_n, L_i)``
- :func:`f1a_permeable`, :func:`f1a_permeable_static` -- permeable-surface paths

Formulation 1C (:mod:`auraflow.fwh.f1c`, uniform mean flow ``U0 = M0 c0``):

- :func:`f1c_pressure` -- convective kernel (reduces to F1A at ``M0 = 0``)
- :func:`f1c_windtunnel` -- static-geometry wind-tunnel fast path
"""

from auraflow.fwh.f1a import (
    f1a_loading,
    f1a_permeable,
    f1a_permeable_static,
    f1a_pressure,
    f1a_thickness,
    permeable_surface_sources,
)
from auraflow.fwh.f1c import f1c_pressure, f1c_windtunnel
from auraflow.fwh.geometry import (
    arrival_times,
    convective_radiation,
    default_observer_grid,
    doppler_factor,
    mach_radial,
    radiation_vectors,
    resample_sum,
    source_time_derivative,
)

__all__ = [
    "arrival_times",
    "convective_radiation",
    "default_observer_grid",
    "doppler_factor",
    "f1a_loading",
    "f1a_permeable",
    "f1a_permeable_static",
    "f1a_pressure",
    "f1a_thickness",
    "f1c_pressure",
    "f1c_windtunnel",
    "mach_radial",
    "permeable_surface_sources",
    "radiation_vectors",
    "resample_sum",
    "source_time_derivative",
]
