"""BEMT aerodynamics backend: blade-element momentum loading + compact F1A tonal noise.

This backend fixes the three predecessor bugs documented in
``docs/research/fwh-rotor-sim-audit.md``: (a) BEMT-induced velocity now feeds the
acoustic source loads, (b) blade forces track the time-varying ``Omega(t)``
rather than a frozen mean, and (c) azimuth is the integral ``psi = int Omega dt``
(via :func:`auraflow.core.frames.integrate_azimuth`).

Public API
----------
Steady solver (:mod:`auraflow.bemt.solver`):

- :func:`steady_bemt`, :func:`solve_inflow_angle`, :func:`prandtl_tip_root_loss`
- :class:`AnnulusState`, :class:`RotorLoads`

Inflow models (:mod:`auraflow.bemt.inflow`):

- :func:`glauert_inflow`, :func:`wake_skew_angle`, :func:`pitt_peters_inflow`

Unsteady quasi-steady marching (:mod:`auraflow.bemt.unsteady`):

- :func:`march_bemt`, :class:`SectionState`

Acoustics coupling (:mod:`auraflow.bemt.acoustics`):

- :func:`rotor_tonal_noise`, :func:`section_loading_sources`, :func:`section_thickness_sources`
"""

from auraflow.bemt.acoustics import (
    rotor_tonal_noise,
    section_loading_sources,
    section_thickness_sources,
)
from auraflow.bemt.inflow import glauert_inflow, pitt_peters_inflow, wake_skew_angle
from auraflow.bemt.solver import (
    AnnulusState,
    RotorLoads,
    prandtl_tip_root_loss,
    solve_inflow_angle,
    steady_bemt,
)
from auraflow.bemt.unsteady import SectionState, march_bemt

__all__ = [
    "AnnulusState",
    "RotorLoads",
    "SectionState",
    "glauert_inflow",
    "march_bemt",
    "pitt_peters_inflow",
    "prandtl_tip_root_loss",
    "rotor_tonal_noise",
    "section_loading_sources",
    "section_thickness_sources",
    "solve_inflow_angle",
    "steady_bemt",
    "wake_skew_angle",
]
