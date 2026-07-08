r"""JASA data-generation datasets: NASA 1-Pax quadrotor flyovers via CONA.

Reproduces the data-generation pipeline of the JASA drone-noise paper (Lee, Ko,
Seshadri, Rauleder, JASA 159(4):3418-3435, 2026) with AuraFlow's CONA backend,
and provides the CONA-vs-CFD comparison plumbing. See the digests
``docs/research/jasa-datagen-reference.md`` and
``docs/research/nasa-1pax-vehicle.md``.

Sub-modules
-----------
- :mod:`auraflow.datasets.nasa_1pax` -- the vehicle (single source of truth for
  the blade geometry; delegates mass/inertia/allocation to
  :class:`auraflow.cona.flight.Multirotor`).
- :mod:`auraflow.datasets.jasa` -- :class:`JASAScenario`, :func:`generate_flyover`,
  the microphone array, scenario grids and ``.npz``/WAV saving.
- :mod:`auraflow.datasets.compare` -- CONA vs CFD+FW-H comparison metrics
  (accepts a synthetic :class:`~auraflow.cfd.run.SurfaceHistory`).
- :mod:`auraflow.datasets.dload_io` -- **optional** dload output management
  (lazy import; base import works without the ``data`` extra / credentials).

``import auraflow.datasets`` works without the ``data`` (dload) or ``cfd``
(jaxfluids) extras and without any credentials; only committing to dload or
running a CFD case pulls those in.
"""

from auraflow.datasets.compare import (
    cfd_observer_signals,
    compare_cona_vs_cfd,
    compare_signals,
    signal_metrics,
)
from auraflow.datasets.dload_io import (
    commit_flyovers,
    flyover_sample,
    flyover_samples,
    open_repository,
)
from auraflow.datasets.jasa import (
    JASAScenario,
    generate_flyover,
    generate_scenario_grid,
    jasa_microphone_array,
    save_flyover,
    scenario_id,
)
from auraflow.datasets.nasa_1pax import (
    BPF_HZ,
    GROSS_WEIGHT_KG,
    HOVER_OMEGA,
    HOVER_RPM,
    N_BLADES,
    N_ROTORS,
    ROTOR_RADIUS,
    SOLIDITY,
    nasa_1pax_blade,
    nasa_1pax_hover_collective,
    nasa_1pax_multirotor,
    nasa_1pax_polar,
    nasa_1pax_vehicle,
)

__all__ = [
    "BPF_HZ",
    "GROSS_WEIGHT_KG",
    "HOVER_OMEGA",
    "HOVER_RPM",
    "N_BLADES",
    "N_ROTORS",
    "ROTOR_RADIUS",
    "SOLIDITY",
    "JASAScenario",
    "cfd_observer_signals",
    "commit_flyovers",
    "compare_cona_vs_cfd",
    "compare_signals",
    "flyover_sample",
    "flyover_samples",
    "generate_flyover",
    "generate_scenario_grid",
    "jasa_microphone_array",
    "nasa_1pax_blade",
    "nasa_1pax_hover_collective",
    "nasa_1pax_multirotor",
    "nasa_1pax_polar",
    "nasa_1pax_vehicle",
    "open_repository",
    "save_flyover",
    "scenario_id",
    "signal_metrics",
]
