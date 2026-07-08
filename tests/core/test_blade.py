"""Tests for auraflow.core.blade."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.core import BladeGeometry, Rotor, Vehicle, rot_z


@pytest.fixture
def blade() -> BladeGeometry:
    return BladeGeometry.linear(
        radius=0.5,
        hub_radius=0.05,
        n_stations=20,
        chord_root=0.06,
        chord_tip=0.02,
        twist_root=jnp.deg2rad(20.0),
        twist_tip=jnp.deg2rad(8.0),
    )


class TestBladeGeometryStations:
    def test_station_radii_span_blade(self, blade):
        assert blade.r.shape == (20,)
        assert float(blade.r[0]) == pytest.approx(0.05)
        assert float(blade.r[-1]) == pytest.approx(0.5)
        np.testing.assert_allclose(jnp.diff(blade.r), jnp.diff(blade.r)[0], atol=1e-15)

    def test_dr_sums_to_span(self, blade):
        assert float(jnp.sum(blade.dr)) == pytest.approx(0.45, abs=1e-14)

    def test_dr_trapezoid_half_panels_at_ends(self, blade):
        h = (0.5 - 0.05) / 19
        np.testing.assert_allclose(blade.dr[0], h / 2, atol=1e-15)
        np.testing.assert_allclose(blade.dr[-1], h / 2, atol=1e-15)
        np.testing.assert_allclose(blade.dr[1:-1], h, atol=1e-15)

    def test_dr_matches_trapezoid_rule(self, blade):
        # sum(f * dr) must equal jnp.trapezoid(f, r) for any integrand.
        f = blade.r**2 + 3.0 * blade.r
        np.testing.assert_allclose(jnp.sum(f * blade.dr), jnp.trapezoid(f, blade.r), rtol=1e-14)

    def test_requires_two_stations(self):
        with pytest.raises(ValueError):
            BladeGeometry(
                radius=1.0, hub_radius=0.0, chord=jnp.array([0.1]), twist=jnp.array([0.0])
            )

    def test_mismatched_arrays_raise(self):
        with pytest.raises(ValueError):
            BladeGeometry(radius=1.0, hub_radius=0.0, chord=jnp.zeros(4), twist=jnp.zeros(5))


class TestChordTwistInterpolation:
    def test_linear_blade_chord_at(self, blade):
        rq = jnp.linspace(0.05, 0.5, 7)
        expected = 0.06 + (0.02 - 0.06) * (rq - 0.05) / 0.45
        np.testing.assert_allclose(blade.chord_at(rq), expected, atol=1e-15)

    def test_linear_blade_twist_at(self, blade):
        rq = jnp.array([0.05, 0.275, 0.5])
        expected = jnp.deg2rad(jnp.array([20.0, 14.0, 8.0]))
        np.testing.assert_allclose(blade.twist_at(rq), expected, atol=1e-14)

    def test_from_arrays_reproduces_uniform_input(self):
        r = jnp.linspace(0.1, 1.0, 10)
        chord = 0.1 - 0.05 * r
        twist = jnp.deg2rad(15.0) * (1.0 - r)
        geom = BladeGeometry.from_arrays(r, chord, twist)
        assert geom.n_stations == 10
        np.testing.assert_allclose(geom.r, r, atol=1e-15)
        np.testing.assert_allclose(geom.chord, chord, atol=1e-15)
        np.testing.assert_allclose(geom.twist, twist, atol=1e-15)

    def test_from_arrays_resamples_nonuniform_input(self):
        # Piecewise-linear data on a non-uniform grid; resampling onto uniform
        # stations must agree with direct linear interpolation.
        r = jnp.array([0.1, 0.2, 0.5, 1.0])
        chord = jnp.array([0.10, 0.09, 0.05, 0.02])
        twist = jnp.array([0.3, 0.25, 0.15, 0.05])
        geom = BladeGeometry.from_arrays(r, chord, twist, n_stations=16)
        assert geom.n_stations == 16
        np.testing.assert_allclose(geom.chord, jnp.interp(geom.r, r, chord), atol=1e-15)
        np.testing.assert_allclose(geom.twist, jnp.interp(geom.r, r, twist), atol=1e-15)
        assert float(geom.hub_radius) == pytest.approx(0.1)
        assert float(geom.radius) == pytest.approx(1.0)


class TestQuarterChordPoints:
    def test_at_zero_azimuth_along_x(self, blade):
        pts = blade.quarter_chord_points(0.0)
        assert pts.shape == (20, 3)
        np.testing.assert_allclose(pts[:, 0], blade.r, atol=1e-15)
        np.testing.assert_allclose(pts[:, 1], 0.0, atol=1e-15)
        np.testing.assert_allclose(pts[:, 2], 0.0, atol=1e-15)

    def test_rotation_consistency(self, blade):
        psi = 1.234
        pts0 = blade.quarter_chord_points(0.0)
        pts = blade.quarter_chord_points(psi)
        expected = jnp.einsum("ij,sj->si", rot_z(psi), pts0)
        np.testing.assert_allclose(pts, expected, atol=1e-14)

    def test_radius_preserved_under_rotation(self, blade):
        pts = blade.quarter_chord_points(2.5)
        np.testing.assert_allclose(jnp.linalg.norm(pts, axis=-1), blade.r, atol=1e-14)

    def test_batched_azimuth_shapes(self, blade):
        assert blade.quarter_chord_points(jnp.zeros(4)).shape == (4, 20, 3)
        assert blade.quarter_chord_points(jnp.zeros((2, 5))).shape == (2, 5, 20, 3)
        psi = jnp.array([0.0, 0.7, -1.4])
        batched = blade.quarter_chord_points(psi)
        for i in range(3):
            np.testing.assert_allclose(batched[i], blade.quarter_chord_points(psi[i]), atol=1e-15)


class TestSectionVelocity:
    def test_cross_product_identity(self, blade):
        psi, omega = 0.9, 150.0
        pts = blade.quarter_chord_points(psi)
        vel = blade.section_velocity(psi, omega)
        expected = jnp.cross(jnp.array([0.0, 0.0, omega]), pts)
        np.testing.assert_allclose(vel, expected, atol=1e-12)

    def test_speed_is_omega_r(self, blade):
        vel = blade.section_velocity(0.3, 100.0)
        np.testing.assert_allclose(jnp.linalg.norm(vel, axis=-1), 100.0 * blade.r, rtol=1e-13)

    def test_velocity_tangential(self, blade):
        psi = 2.2
        pts = blade.quarter_chord_points(psi)
        vel = blade.section_velocity(psi, 80.0)
        np.testing.assert_allclose(jnp.sum(pts * vel, axis=-1), 0.0, atol=1e-10)

    def test_negative_omega_reverses_velocity(self, blade):
        v_pos = blade.section_velocity(0.4, 60.0)
        v_neg = blade.section_velocity(0.4, -60.0)
        np.testing.assert_allclose(v_neg, -v_pos, atol=1e-13)

    def test_batched_shapes(self, blade):
        psi = jnp.zeros(7)
        assert blade.section_velocity(psi, 50.0).shape == (7, 20, 3)
        assert blade.section_velocity(psi, jnp.full(7, 50.0)).shape == (7, 20, 3)


class TestDifferentiability:
    def test_grad_thrust_like_wrt_chord_params(self):
        omega = 200.0

        def thrust_like(chord_root, chord_tip):
            b = BladeGeometry.linear(
                radius=0.5,
                hub_radius=0.05,
                n_stations=30,
                chord_root=chord_root,
                chord_tip=chord_tip,
                twist_root=0.3,
                twist_tip=0.1,
            )
            # ~ 0.5 rho c (Omega r)^2 cl dr summed over stations
            return jnp.sum(0.5 * 1.225 * b.chord * (omega * b.r) ** 2 * b.twist * b.dr)

        grads = jax.grad(thrust_like, argnums=(0, 1))(0.06, 0.02)
        assert all(jnp.isfinite(g) for g in grads)
        assert all(float(g) > 0.0 for g in grads)

    def test_grad_wrt_radius(self):
        def tip_speed(radius):
            b = BladeGeometry.linear(radius, 0.05, 10, 0.05, 0.02, 0.2, 0.1)
            return jnp.linalg.norm(b.section_velocity(0.0, 100.0)[-1])

        g = jax.grad(tip_speed)(0.5)
        assert float(g) == pytest.approx(100.0, rel=1e-10)

    def test_grad_through_quarter_chord_points(self, blade):
        def y_coord(psi):
            return blade.quarter_chord_points(psi)[-1, 1]

        g = jax.grad(y_coord)(0.0)
        assert float(g) == pytest.approx(float(blade.r[-1]), rel=1e-12)


class TestRotor:
    @pytest.fixture
    def rotor(self, blade) -> Rotor:
        return Rotor(
            blade=blade,
            n_blades=4,
            hub_position=jnp.array([1.0, 2.0, 3.0]),
            hub_orientation=rot_z(jnp.pi / 2),
        )

    def test_blade_azimuth_offsets(self, blade):
        rotor = Rotor(blade=blade, n_blades=4)
        np.testing.assert_allclose(
            rotor.blade_azimuths(0.0), 2.0 * jnp.pi * jnp.arange(4) / 4, atol=1e-15
        )

    def test_blade_azimuths_offset_by_psi0(self, blade):
        rotor = Rotor(blade=blade, n_blades=3)
        np.testing.assert_allclose(
            rotor.blade_azimuths(0.5), 0.5 + 2.0 * jnp.pi * jnp.arange(3) / 3, atol=1e-15
        )

    def test_blade_azimuths_batched(self, blade):
        rotor = Rotor(blade=blade, n_blades=2)
        psi0 = jnp.array([0.0, 0.1, 0.2])
        assert rotor.blade_azimuths(psi0).shape == (3, 2)

    def test_spin_direction_reverses_offsets(self, blade):
        ccw = Rotor(blade=blade, n_blades=3, spin_direction=1)
        cw = Rotor(blade=blade, n_blades=3, spin_direction=-1)
        np.testing.assert_allclose(cw.blade_azimuths(0.0), -ccw.blade_azimuths(0.0), atol=1e-15)

    def test_invalid_spin_direction_raises(self, blade):
        with pytest.raises(ValueError):
            Rotor(blade=blade, n_blades=2, spin_direction=0)

    def test_default_placement(self, blade):
        rotor = Rotor(blade=blade, n_blades=2)
        np.testing.assert_allclose(rotor.hub_position, jnp.zeros(3), atol=0)
        np.testing.assert_allclose(rotor.hub_orientation, jnp.eye(3), atol=0)

    def test_to_parent_points(self, rotor):
        # Rotor +x maps to parent +y under rot_z(pi/2), then hub offset applies.
        p = rotor.to_parent_points(jnp.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(p, [1.0, 3.0, 3.0], atol=1e-14)

    def test_to_parent_vectors_no_translation(self, rotor):
        v = rotor.to_parent_vectors(jnp.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-14)

    def test_to_parent_points_batched(self, rotor):
        pts = rotor.blade.quarter_chord_points(jnp.zeros(5))  # [5, S, 3]
        out = rotor.to_parent_points(pts)
        assert out.shape == pts.shape


class TestVehicle:
    def test_minimal_construction(self, blade):
        rotors = (
            Rotor(blade=blade, n_blades=2, hub_position=jnp.array([0.3, 0.0, 0.0])),
            Rotor(
                blade=blade,
                n_blades=2,
                hub_position=jnp.array([-0.3, 0.0, 0.0]),
                spin_direction=-1,
            ),
        )
        vehicle = Vehicle(rotors=rotors, position=jnp.array([0.0, 0.0, 30.0]))
        assert vehicle.n_rotors == 2
        np.testing.assert_allclose(vehicle.attitude, jnp.eye(3), atol=0)

    def test_rotor_in_world_composition(self, blade):
        rotor = Rotor(blade=blade, n_blades=2, hub_position=jnp.array([0.5, 0.0, 0.0]))
        vehicle = Vehicle(
            rotors=(rotor,),
            position=jnp.array([10.0, 0.0, 30.0]),
            attitude=rot_z(jnp.pi / 2),
        )
        world_rotor = vehicle.rotor_in_world(0)
        # Body-frame hub (0.5, 0, 0) yawed by 90 deg -> (0, 0.5, 0) + position.
        np.testing.assert_allclose(world_rotor.hub_position, [10.0, 0.5, 30.0], atol=1e-14)
        np.testing.assert_allclose(world_rotor.hub_orientation, rot_z(jnp.pi / 2), atol=1e-14)
        assert world_rotor.spin_direction == rotor.spin_direction
        assert world_rotor.n_blades == rotor.n_blades

    def test_world_mapping_consistency(self, blade):
        rotor = Rotor(
            blade=blade,
            n_blades=2,
            hub_position=jnp.array([0.3, -0.2, 0.1]),
            hub_orientation=rot_z(0.7),
        )
        vehicle = Vehicle(rotors=(rotor,), position=jnp.array([1.0, 2.0, 3.0]), attitude=rot_z(0.4))
        pts_rotor = blade.quarter_chord_points(0.9)
        # rotor -> body -> world must equal rotor -> world via composed rotor.
        via_body = vehicle.to_world_points(rotor.to_parent_points(pts_rotor))
        via_world_rotor = vehicle.rotor_in_world(0).to_parent_points(pts_rotor)
        np.testing.assert_allclose(via_body, via_world_rotor, atol=1e-13)
