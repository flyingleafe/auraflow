"""DJI Phantom (9450) vehicle reconstruction vs the digest (docs/research/dji-9450-reference.md).

Checks the published/digitized constants; that the digitized blade, at the
nominal hover RPM, hovers on its own (fixed-pitch: 4-rotor thrust ~ weight with a
near-zero trim collective) under BEMT; the BPF/tip-Mach band; the lofted blade
mesh is watertight; the resolved-rotor level-set case validates through the real
JAX-Fluids ``InputManager``; and a tiny CONA flyover produces a clean BPF
harmonic comb. All cases are deliberately small (CPU/float64).
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.bemt.solver import steady_bemt
from auraflow.core.blade import Rotor
from auraflow.core.medium import Medium
from auraflow.datasets.dji_phantom import (
    BPF_HZ,
    HOVER_OMEGA,
    HOVER_RPM,
    MASS,
    N_BLADES,
    N_ROTORS,
    ROTOR_RADIUS,
    TIP_MACH,
    dji_9450_blade,
    dji_9450_blade_mesh,
    dji_phantom_hover_collective,
    dji_phantom_multirotor,
    dji_phantom_polar,
    dji_phantom_vehicle,
)

_G = 9.80665


def test_published_constants_match_digest():
    assert N_ROTORS == 4
    assert N_BLADES == 2  # DJI 9450 is two-bladed
    assert ROTOR_RADIUS == 0.1200  # 9.45 in true diameter / 2
    assert MASS == 1.280  # Phantom 3 Adv./Pro./4K (ships the 9450 prop)
    assert HOVER_RPM == 5400.0  # literature nominal (weight/4 gives 5040-5400)
    # BPF = 2 * 5400 / 60 = 180 Hz.
    assert abs(BPF_HZ - 180.0) < 1e-6
    # HOVER_OMEGA consistent with HOVER_RPM.
    assert abs(HOVER_OMEGA - HOVER_RPM * 2.0 * math.pi / 60.0) < 1e-6


def test_bpf_and_tip_mach_in_drone_band():
    # BPF is an audible consumer-drone tone.
    assert 150.0 < BPF_HZ < 300.0
    # Tip Mach for this rotor class is ~0.20 (digest: 0.19-0.20 for the real
    # 9450). NOTE: the task brief suggested 0.3-0.6, but that is the wrong band
    # for a 0.12 m / 5400 RPM rotor -- the digitized geometry gives Mtip = 0.20.
    assert 0.15 < TIP_MACH < 0.35
    assert abs(TIP_MACH - HOVER_OMEGA * ROTOR_RADIUS / 340.0) < 1e-6


def test_digitized_blade_shape():
    """Chord peaks inboard (~r/R 0.28), twist decreases root-to-tip (Deters Fig. 7)."""
    blade = dji_9450_blade(64)
    r = np.asarray(blade.r)
    c = np.asarray(blade.chord)
    tw = np.asarray(blade.twist)
    # Radial extent: 0.15 R -> R.
    assert abs(r[0] - 0.15 * ROTOR_RADIUS) < 1e-6
    assert abs(r[-1] - ROTOR_RADIUS) < 1e-6
    # Peak chord is inboard (r/R ~ 0.25-0.35), not at root or tip.
    i_peak = int(np.argmax(c))
    assert 0.20 < r[i_peak] / ROTOR_RADIUS < 0.40
    assert c[-1] < c[i_peak]  # tapered tip
    # Twist is nose-up everywhere and falls from root to tip overall.
    assert np.all(tw > 0.0)
    assert tw[0] > tw[-1]
    assert math.degrees(tw.max()) < 25.0  # digitized peak ~20.8 deg


def test_multirotor_delegation_no_duplicate_constants():
    """The geometric vehicle reads hubs/spins back from the Multirotor (single source)."""
    mrotor = dji_phantom_multirotor()
    vehicle = dji_phantom_vehicle(8)
    assert mrotor.n_rotors == N_ROTORS
    assert vehicle.n_rotors == N_ROTORS
    assert float(mrotor.mass) == MASS
    for i, rotor in enumerate(vehicle.rotors):
        assert jnp.allclose(rotor.hub_position, mrotor.rotor_positions[i])
        assert int(rotor.spin_direction) == int(round(float(mrotor.spin_signs[i])))
        assert rotor.n_blades == N_BLADES
    # Diagonal rotors share spin; adjacent alternate (torque balance).
    spins = [int(r.spin_direction) for r in vehicle.rotors]
    assert sum(spins) == 0
    # Published X-layout hub offset: 0.5 * 0.350 m * cos45 = 0.1237 m each axis.
    d = 0.5 * 0.350 * math.cos(math.radians(45.0))
    assert np.allclose(np.abs(np.asarray(mrotor.rotor_positions[:, :2])), d, atol=1e-3)


def test_hover_gate_reaches_weight():
    """The digitized fixed-pitch geometry hovers at 5400 RPM on its own (weight/4)."""
    medium = Medium()
    polar = dji_phantom_polar()
    rotor = Rotor(blade=dji_9450_blade(24), n_blades=N_BLADES)

    # (1) Fixed pitch: at zero trim offset the 4-rotor thrust already ~ weight.
    loads0 = steady_bemt(rotor, medium, HOVER_OMEGA, collective=0.0, polar=polar)
    weight = MASS * _G
    assert abs(4.0 * float(loads0.thrust) - weight) / weight < 0.20

    # (2) The hover-trim offset is therefore tiny (a fixed-pitch prop, not a
    # collective-controlled rotor).
    coll = dji_phantom_hover_collective(24, medium, polar)
    assert abs(coll) < math.radians(3.0)
    loads = steady_bemt(rotor, medium, HOVER_OMEGA, collective=coll, polar=polar)
    target = weight / N_ROTORS
    assert abs(float(loads.thrust) - target) / target < 0.05

    # (3) CT/sigma in a physical band (digest: helicopter CT ~ 0.011-0.014).
    sigma = N_BLADES * float(jnp.mean(rotor.blade.chord)) / (math.pi * ROTOR_RADIUS)
    ct_sigma = float(loads.ct) / sigma
    assert 0.07 < ct_sigma < 0.18


def test_blade_mesh_watertight_and_positive_volume():
    mesh = dji_9450_blade_mesh(n_span=6, n_chord=16)
    assert mesh.is_watertight
    assert float(mesh.volume()) > 0.0


def test_rotor_levelset_case_validates_through_input_manager():
    """The 9450 resolved-rotor level-set case validates through the real InputManager.

    ~1 mm-class dx on a TINY 16^3 grid (InputManager-only; no marching).
    """
    pytest.importorskip("jaxfluids")
    from jaxfluids import InputManager

    from auraflow.body.blade import rotor_levelset_case

    rotor = Rotor(blade=dji_9450_blade(6), n_blades=N_BLADES)
    h = 0.008  # box half-extent [m]; 16 cells -> dx = 1 mm
    case = rotor_levelset_case(
        rotor,
        omega=HOVER_OMEGA,
        box_lo=(-h, -h, -h),
        box_hi=(h, h, h),
        cells=(16, 16, 16),
        n_chord=16,
        hub=True,
        blade_cells=(24, 24, 24),
        sdf_cache=False,
    )
    assert np.asarray(case.levelset_init).shape == (16, 16, 16)
    im = InputManager(case.case, case.numerical_setup)
    assert im.equation_information.levelset_model == "FLUID-SOLID"
    assert im.equation_information.is_moving_levelset is True


def test_cona_flyover_has_bpf_harmonic_comb():
    """A tiny Phantom CONA hover flyover shows a clear peak on the BPF comb."""
    from auraflow.datasets.jasa import JASAScenario, generate_flyover
    from auraflow.signal.spectra import narrowband_spectrum

    n_stations = 6
    polar = dji_phantom_polar()
    collective = dji_phantom_hover_collective(n_stations, polar=polar)
    sc = JASAScenario(
        speed=0.0,
        altitude=5.0,
        duration=0.3,
        fs=2000.0,
        seed=0,
        mics=jnp.asarray([[2.0, 0.0, 0.0]]),
    )
    res = generate_flyover(
        sc,
        polar=polar,
        collective=collective,
        vehicle=dji_phantom_vehicle(n_stations),
        multirotor=dji_phantom_multirotor(),
        bpf_hz=BPF_HZ,
        include_broadband=False,
        n_stations=n_stations,
        n_source_times=400,
        n_frames=8,
        n_fft=128,
        gl_iters=5,
        obs_chunk=1,
        low_memory=True,
    )
    assert abs(res["meta"]["bpf_hz"] - BPF_HZ) < 1e-6
    tonal = np.asarray(res["tonal"][0])
    assert np.all(np.isfinite(tonal))

    freqs, amp = narrowband_spectrum(jnp.asarray(tonal), sc.fs)
    freqs = np.asarray(freqs)
    amp = np.asarray(amp)
    idx_bpf = int(np.argmin(np.abs(freqs - BPF_HZ)))
    # The BPF bin dominates the spectrum median by a wide margin.
    assert amp[idx_bpf] > 8.0 * np.median(amp)
    # The dominant line lies on the BPF harmonic comb (a harmonic may exceed the
    # partially-cancelled fundamental for a symmetric 4-rotor array; that is
    # physical -- same assertion pattern as tests/datasets/test_jasa.py).
    df = float(freqs[1] - freqs[0])
    f_max = float(freqs[int(np.argmax(amp))])
    n_harm = max(round(f_max / BPF_HZ), 1)
    assert abs(f_max - n_harm * BPF_HZ) <= df
