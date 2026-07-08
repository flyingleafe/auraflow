"""NASA 1-Pax vehicle reconstruction vs the digest (docs/research/nasa-1pax-vehicle.md).

Checks the published constants and that the reconstructed blade, at the published
hover RPM, actually trims to the per-rotor hover thrust (weight / 4) at a
physical collective -- i.e. the geometry is self-consistent with BEMT.
"""

import math

import jax.numpy as jnp

from auraflow.bemt.solver import steady_bemt
from auraflow.core.blade import Rotor
from auraflow.core.medium import Medium
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

_G = 9.80665


def test_published_constants_match_digest():
    assert GROSS_WEIGHT_KG == 583.85
    assert N_ROTORS == 4
    assert N_BLADES == 3
    assert ROTOR_RADIUS == 1.951  # 6.4 ft
    assert SOLIDITY == 0.065
    assert HOVER_RPM == 671.0
    # BPF = 3 * 671 / 60 = 33.55 Hz.
    assert abs(BPF_HZ - 33.55) < 1e-6
    # Tip speed 450 ft/s = 137.16 m/s => Omega = tip/R.
    assert abs(HOVER_OMEGA * ROTOR_RADIUS - 137.16) < 1.0


def test_reconstructed_solidity_is_0p065():
    """Thrust-weighted (r^2) solidity of the reconstructed blade == 0.065."""
    blade = nasa_1pax_blade(64)
    r = jnp.asarray(blade.r)  # station radii
    c = jnp.asarray(blade.chord)
    # sigma = Nb * <c>_{r^2} / (pi R); <c>_{r^2} = int c r^2 / int r^2.
    w = r**2
    c_bar = float(jnp.trapezoid(c * w, r) / jnp.trapezoid(w, r))
    sigma = N_BLADES * c_bar / (math.pi * ROTOR_RADIUS)
    assert abs(sigma - SOLIDITY) < 2e-3


def test_multirotor_delegation_no_duplicate_constants():
    """The geometric vehicle reads hubs/spins back from the Multirotor (single source)."""
    mrotor = nasa_1pax_multirotor()
    vehicle = nasa_1pax_vehicle(8)
    assert mrotor.n_rotors == N_ROTORS
    assert vehicle.n_rotors == N_ROTORS
    assert float(mrotor.mass) == GROSS_WEIGHT_KG
    for i, rotor in enumerate(vehicle.rotors):
        assert jnp.allclose(rotor.hub_position, mrotor.rotor_positions[i])
        assert int(rotor.spin_direction) == int(round(float(mrotor.spin_signs[i])))
    # Diagonal rotors share spin; adjacent alternate (torque balance).
    spins = [int(r.spin_direction) for r in vehicle.rotors]
    assert sum(spins) == 0


def test_hover_trim_reaches_weight_within_15pct():
    """At 671 RPM the trimmed collective produces per-rotor thrust ~ weight/4."""
    medium = Medium()
    polar = nasa_1pax_polar()
    coll = nasa_1pax_hover_collective(24, medium, polar)
    # Collective must be inside the trim bracket (not saturated at 0 or 30 deg).
    assert math.radians(1.0) < coll < math.radians(30.0)

    rotor = Rotor(blade=nasa_1pax_blade(24), n_blades=N_BLADES)
    loads = steady_bemt(rotor, medium, HOVER_OMEGA, v_climb=0.0, collective=coll, polar=polar)
    target = GROSS_WEIGHT_KG * _G / N_ROTORS
    assert abs(float(loads.thrust) - target) / target < 0.15
    # CT/sigma should be near the digest's ~0.104 (loose band; reconstruction).
    ct_sigma = float(loads.ct) / SOLIDITY
    assert 0.07 < ct_sigma < 0.14
