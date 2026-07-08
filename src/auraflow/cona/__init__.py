"""CONA backend: multirotor flight dynamics, gusts, and noise auralization.

This package implements the CONA framework (Ko et al.,
``docs/research/cona-reference.md``) as a differentiable JAX pipeline. The
flight-dynamics front end lives here; it produces the vehicle and per-rotor
kinematic histories that the downstream aeroacoustic stages consume.

Flight dynamics (:mod:`auraflow.cona.flight`):

- :class:`Multirotor`, :class:`ControllerGains`, :class:`FlightHistory`
- :func:`simulate` -- closed-loop 6-DOF RK4 rollout with a geometric SE(3)
  tracking controller
- :func:`geometric_controller`, :func:`desired_attitude`, :func:`attitude_error`,
  :func:`hat`, :func:`vee`
- Trajectory generators :func:`hover`, :func:`straight_flyover`

Gusts (:mod:`auraflow.cona.gusts`):

- :func:`dryden_gust`, :func:`dryden_parameters`, :class:`DrydenParameters`,
  :data:`W20_PRESETS`

Aerodynamics + tonal acoustics (the HBEM middle stage):

- Prescribed wake (:mod:`auraflow.cona.wake`): :class:`PrescribedWake`,
  :func:`make_prescribed_wake`, :func:`lamb_oseen_swirl`,
  :func:`biot_savart_segment`, :func:`vortex_circulation`, :func:`core_radius`,
  :func:`beddoes_wake_nodes`
- Unsteady aero (:mod:`auraflow.cona.unsteady_aero`): :func:`wagner_function`,
  :func:`deficiency_march`, :func:`effective_aoa`, :func:`unsteady_lift`
- Airloads (:mod:`auraflow.cona.airloads`): :func:`rotor_section_state`,
  :func:`cona_airloads` (produce the BEMT :class:`SectionState` PyTree)
- Tonal noise (:mod:`auraflow.cona.tonal`): :func:`cona_tonal_noise`,
  :func:`mean_flow_frame`
"""

from auraflow.cona.airloads import cona_airloads, rotor_section_state
from auraflow.cona.auralize import (
    auralize_broadband,
    cona_auralize,
    resample_linear,
    synthesize_observer_signal,
)
from auraflow.cona.bpm import (
    BLThickness,
    BPMSpectra,
    a_shape,
    b_shape,
    boundary_layer_thickness,
    bpm_third_octave,
    directivity_high,
    directivity_low,
    k1_amplitude,
    k2_amplitude,
    st1_peak,
    tbl_te_noise,
)
from auraflow.cona.broadband import (
    doppler_rebin,
    rotor_broadband_levels,
    rotor_broadband_spectrogram,
)
from auraflow.cona.flight import (
    ControllerGains,
    FlightHistory,
    Multirotor,
    attitude_error,
    desired_attitude,
    geometric_controller,
    hat,
    hover,
    simulate,
    straight_flyover,
    vee,
)
from auraflow.cona.gusts import (
    W20_PRESETS,
    DrydenParameters,
    dryden_gust,
    dryden_parameters,
)
from auraflow.cona.tonal import cona_tonal_noise, mean_flow_frame
from auraflow.cona.unsteady_aero import (
    deficiency_march,
    effective_aoa,
    unsteady_lift,
    wagner_function,
)
from auraflow.cona.wake import (
    PrescribedWake,
    beddoes_wake_nodes,
    biot_savart_segment,
    core_radius,
    lamb_oseen_swirl,
    make_prescribed_wake,
    vortex_circulation,
)

__all__ = [
    "W20_PRESETS",
    "BLThickness",
    "BPMSpectra",
    "ControllerGains",
    "DrydenParameters",
    "FlightHistory",
    "Multirotor",
    "PrescribedWake",
    "a_shape",
    "attitude_error",
    "auralize_broadband",
    "b_shape",
    "beddoes_wake_nodes",
    "biot_savart_segment",
    "boundary_layer_thickness",
    "bpm_third_octave",
    "cona_airloads",
    "cona_auralize",
    "cona_tonal_noise",
    "core_radius",
    "deficiency_march",
    "desired_attitude",
    "directivity_high",
    "directivity_low",
    "doppler_rebin",
    "dryden_gust",
    "dryden_parameters",
    "effective_aoa",
    "geometric_controller",
    "hat",
    "hover",
    "k1_amplitude",
    "k2_amplitude",
    "lamb_oseen_swirl",
    "make_prescribed_wake",
    "mean_flow_frame",
    "resample_linear",
    "rotor_broadband_levels",
    "rotor_broadband_spectrogram",
    "rotor_section_state",
    "simulate",
    "st1_peak",
    "straight_flyover",
    "synthesize_observer_signal",
    "tbl_te_noise",
    "unsteady_lift",
    "vee",
    "vortex_circulation",
    "wagner_function",
]
