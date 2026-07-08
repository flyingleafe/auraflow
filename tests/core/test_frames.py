"""Tests for auraflow.core.frames."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.core import (
    azimuth_at,
    euler_zyx_matrix,
    integrate_azimuth,
    interp1d,
    rot_x,
    rot_y,
    rot_z,
)

ROTATIONS = [rot_x, rot_y, rot_z]


class TestRotationMatrices:
    @pytest.mark.parametrize("rot", ROTATIONS)
    def test_orthonormal_and_proper(self, rot):
        angles = jnp.array([-2.3, -0.4, 0.0, 0.7, 1.9, 4.0])
        for a in angles:
            mat = rot(a)
            np.testing.assert_allclose(mat @ mat.T, jnp.eye(3), atol=1e-14)
            assert float(jnp.linalg.det(mat)) == pytest.approx(1.0, abs=1e-14)

    @pytest.mark.parametrize("rot", ROTATIONS)
    def test_identity_at_zero(self, rot):
        np.testing.assert_allclose(rot(0.0), jnp.eye(3), atol=1e-15)

    @pytest.mark.parametrize("rot", ROTATIONS)
    def test_composition(self, rot):
        a, b = 0.7, -1.2
        np.testing.assert_allclose(rot(a) @ rot(b), rot(a + b), atol=1e-14)

    @pytest.mark.parametrize("rot", ROTATIONS)
    def test_batched_shapes(self, rot):
        assert rot(0.5).shape == (3, 3)
        assert rot(jnp.zeros(4)).shape == (4, 3, 3)
        assert rot(jnp.zeros((2, 5))).shape == (2, 5, 3, 3)
        # Batched result matches per-element evaluation.
        psi = jnp.array([0.1, 1.0, -2.0])
        batched = rot(psi)
        for i in range(3):
            np.testing.assert_allclose(batched[i], rot(psi[i]), atol=1e-15)

    def test_rot_z_rotates_x_toward_y(self):
        psi = 0.3
        v = rot_z(psi) @ jnp.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(v, [np.cos(psi), np.sin(psi), 0.0], atol=1e-15)

    def test_rot_x_rotates_y_toward_z(self):
        a = 0.3
        v = rot_x(a) @ jnp.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(v, [0.0, np.cos(a), np.sin(a)], atol=1e-15)

    def test_rot_y_rotates_z_toward_x(self):
        a = 0.3
        v = rot_y(a) @ jnp.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(v, [np.sin(a), 0.0, np.cos(a)], atol=1e-15)


class TestEulerZYX:
    def test_zero_angles_identity(self):
        np.testing.assert_allclose(euler_zyx_matrix(0.0, 0.0, 0.0), jnp.eye(3), atol=1e-15)

    def test_matches_factor_product(self):
        roll, pitch, yaw = 0.2, -0.5, 1.1
        expected = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
        np.testing.assert_allclose(euler_zyx_matrix(roll, pitch, yaw), expected, atol=1e-15)

    def test_pure_yaw_maps_body_x(self):
        yaw = 0.8
        mat = euler_zyx_matrix(0.0, 0.0, yaw)
        v_world = mat @ jnp.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(v_world, [np.cos(yaw), np.sin(yaw), 0.0], atol=1e-15)

    def test_batched_broadcast(self):
        roll = jnp.zeros(4)
        mat = euler_zyx_matrix(roll, 0.1, 0.2)
        assert mat.shape == (4, 3, 3)
        np.testing.assert_allclose(mat[0], euler_zyx_matrix(0.0, 0.1, 0.2), atol=1e-15)

    def test_orthonormal(self):
        mat = euler_zyx_matrix(0.3, -0.7, 2.0)
        np.testing.assert_allclose(mat @ mat.T, jnp.eye(3), atol=1e-14)


class TestInterp1d:
    def test_values_at_nodes_and_midpoints(self):
        x = jnp.array([0.0, 1.0, 3.0])
        y = jnp.array([1.0, 2.0, -2.0])
        np.testing.assert_allclose(interp1d(x, x, y), y, atol=1e-15)
        assert float(interp1d(0.5, x, y)) == pytest.approx(1.5)
        assert float(interp1d(2.0, x, y)) == pytest.approx(0.0)

    def test_clamps_outside_range(self):
        x = jnp.array([0.0, 1.0])
        y = jnp.array([3.0, 5.0])
        assert float(interp1d(-1.0, x, y)) == pytest.approx(3.0)
        assert float(interp1d(2.0, x, y)) == pytest.approx(5.0)

    def test_batched_query_shape(self):
        x = jnp.linspace(0.0, 1.0, 5)
        y = x**2
        xq = jnp.zeros((2, 3))
        assert interp1d(xq, x, y).shape == (2, 3)


class TestIntegrateAzimuth:
    def test_constant_omega_exact(self):
        t = jnp.linspace(0.0, 1.0, 11)
        omega = 5.0
        psi = integrate_azimuth(t, omega)
        np.testing.assert_allclose(psi, omega * t, rtol=0, atol=1e-14)

    def test_constant_omega_array_input(self):
        t = jnp.linspace(0.0, 2.0, 21)
        omega = jnp.full_like(t, 30.0)
        np.testing.assert_allclose(integrate_azimuth(t, omega), 30.0 * t, atol=1e-12)

    def test_linear_ramp_matches_analytic(self):
        # Omega(t) = w0 + a t  =>  psi(t) = w0 t + a t^2 / 2 (trapezoid is exact
        # for linear integrands).
        w0, a = 10.0, 4.0
        t = jnp.linspace(0.0, 3.0, 31)
        omega = w0 + a * t
        psi = integrate_azimuth(t, omega)
        np.testing.assert_allclose(psi, w0 * t + 0.5 * a * t**2, rtol=1e-13, atol=1e-12)

    def test_initial_azimuth_offset(self):
        t = jnp.linspace(0.0, 1.0, 5)
        psi = integrate_azimuth(t, 2.0, psi0=1.5)
        assert float(psi[0]) == pytest.approx(1.5)
        np.testing.assert_allclose(psi, 1.5 + 2.0 * t, atol=1e-14)

    def test_nonuniform_grid(self):
        t = jnp.array([0.0, 0.1, 0.4, 1.0])
        psi = integrate_azimuth(t, 7.0)
        np.testing.assert_allclose(psi, 7.0 * t, atol=1e-14)

    def test_quadratic_omega_second_order_accurate(self):
        t = jnp.linspace(0.0, 1.0, 201)
        omega = t**2
        psi = integrate_azimuth(t, omega)
        np.testing.assert_allclose(psi, t**3 / 3.0, atol=1e-5)

    def test_gradient_wrt_omega_is_trapezoid_weights(self):
        t = jnp.linspace(0.0, 1.0, 6)  # dt = 0.2
        omega = jnp.full(6, 3.0)
        grad = jax.grad(lambda om: integrate_azimuth(t, om)[-1])(omega)
        assert bool(jnp.all(jnp.isfinite(grad)))
        # d psi_end / d omega_i are exactly the trapezoid quadrature weights.
        np.testing.assert_allclose(grad, [0.1, 0.2, 0.2, 0.2, 0.2, 0.1], atol=1e-15)


class TestAzimuthAt:
    def test_matches_at_grid_points(self):
        t = jnp.linspace(0.0, 1.0, 21)
        omega = 20.0 + 5.0 * t
        psi = integrate_azimuth(t, omega)
        np.testing.assert_allclose(azimuth_at(t, t, psi), psi, atol=1e-14)

    def test_midpoints_are_neighbor_averages(self):
        t = jnp.linspace(0.0, 1.0, 11)
        psi = integrate_azimuth(t, 10.0 * jnp.sin(t) + 20.0)
        mid = 0.5 * (t[:-1] + t[1:])
        np.testing.assert_allclose(azimuth_at(mid, t, psi), 0.5 * (psi[:-1] + psi[1:]), atol=1e-13)

    def test_batched_tau(self):
        t = jnp.linspace(0.0, 1.0, 11)
        psi = integrate_azimuth(t, 5.0)
        tau = jnp.array([[0.05, 0.15, 0.25], [0.35, 0.45, 0.55]])
        out = azimuth_at(tau, t, psi)
        assert out.shape == (2, 3)
        np.testing.assert_allclose(out, 5.0 * tau, atol=1e-13)

    def test_gradient_wrt_omega_samples_finite(self):
        t = jnp.linspace(0.0, 1.0, 11)

        def psi_at_tau(omega):
            psi = integrate_azimuth(t, omega)
            return azimuth_at(0.37, t, psi)

        grad = jax.grad(psi_at_tau)(jnp.full(11, 25.0))
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.sum(jnp.abs(grad))) > 0.0

    def test_fixes_predecessor_variable_omega_bug(self):
        # For a decelerating rotor, psi(t) != Omega(t) * t; the integral is correct.
        t = jnp.linspace(0.0, 2.0, 401)
        omega = 100.0 - 20.0 * t
        psi = integrate_azimuth(t, omega)
        analytic = 100.0 * t - 10.0 * t**2
        wrong = omega * t
        np.testing.assert_allclose(psi, analytic, rtol=1e-12, atol=1e-10)
        assert float(jnp.max(jnp.abs(wrong - analytic))) > 1.0
