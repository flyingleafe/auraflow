"""Tests for auraflow.fwh.geometry primitives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


class TestRadiationVectors:
    def test_distance_and_unit(self):
        x = jnp.array([3.0, 0.0, 4.0])
        y = jnp.zeros((2, 3))
        r, rhat = radiation_vectors(x, y)
        np.testing.assert_allclose(r, [5.0, 5.0], atol=1e-14)
        np.testing.assert_allclose(rhat[0], [0.6, 0.0, 0.8], atol=1e-14)
        np.testing.assert_allclose(jnp.linalg.norm(rhat, axis=-1), 1.0, atol=1e-14)

    def test_points_from_source_to_observer(self):
        x = jnp.array([1.0, 0.0, 0.0])
        y = jnp.array([[-1.0, 0.0, 0.0]])
        _, rhat = radiation_vectors(x, y)
        np.testing.assert_allclose(rhat[0], [1.0, 0.0, 0.0], atol=1e-14)

    def test_batched_time_axis(self):
        x = jnp.array([0.0, 0.0, 10.0])
        y = jnp.zeros((4, 7, 3))
        r, rhat = radiation_vectors(x, y)
        assert r.shape == (4, 7)
        assert rhat.shape == (4, 7, 3)


class TestMachAndDoppler:
    def test_radial_mach(self):
        v = jnp.array([100.0, 0.0, 0.0])
        rhat = jnp.array([1.0, 0.0, 0.0])
        assert float(mach_radial(v, rhat, 340.0)) == pytest.approx(100.0 / 340.0)

    def test_doppler_is_one_minus_mr(self):
        v = jnp.array([34.0, 0.0, 0.0])
        rhat = jnp.array([1.0, 0.0, 0.0])
        assert float(doppler_factor(v, rhat, 340.0)) == pytest.approx(0.9)

    def test_perpendicular_motion_zero_mr(self):
        v = jnp.array([0.0, 50.0, 0.0])
        rhat = jnp.array([1.0, 0.0, 0.0])
        assert float(mach_radial(v, rhat, 340.0)) == pytest.approx(0.0, abs=1e-15)


class TestConvectiveRadiation:
    def test_reduces_to_ordinary_at_zero_mach(self):
        x = jnp.array([2.0, -1.0, 3.0])
        y = jnp.zeros((5, 3)) + jnp.array([0.3, 0.2, -0.1])
        r, rstar, rt, rts = convective_radiation(x, y, 0.0)
        r_ref, rhat_ref = radiation_vectors(x, y)
        np.testing.assert_allclose(r, r_ref, atol=1e-13)
        np.testing.assert_allclose(rstar, r_ref, atol=1e-13)
        np.testing.assert_allclose(rt, rhat_ref, atol=1e-13)
        np.testing.assert_allclose(rts, rhat_ref, atol=1e-13)

    def test_radiation_vectors_are_gradients(self):
        # R~_i = dR/dx_i and R~*_i = dR*/dx_i by definition.
        y = jnp.array([0.1, -0.2, 0.3])
        m0 = 0.4

        def r_and_rstar(x):
            r, rstar, _, _ = convective_radiation(x, y, m0)
            return r, rstar

        x = jnp.array([1.5, 0.7, -0.9])
        (_, _, rt, rts) = convective_radiation(x, y, m0)
        gr = jax.jacobian(lambda xx: r_and_rstar(xx)[0])(x)
        grs = jax.jacobian(lambda xx: r_and_rstar(xx)[1])(x)
        np.testing.assert_allclose(rt, gr, atol=1e-10)
        np.testing.assert_allclose(rts, grs, atol=1e-10)

    def test_phase_amplitude_relation(self):
        # R = (-M0 d1 + R*) / beta^2
        x = jnp.array([4.0, 1.0, 2.0])
        y = jnp.array([[0.0, 0.0, 0.0]])
        m0 = 0.5
        r, rstar, _, _ = convective_radiation(x, y, m0)
        beta2 = 1 - m0**2
        np.testing.assert_allclose(r, (-m0 * 4.0 + rstar) / beta2, atol=1e-13)


class TestSourceTimeDerivative:
    def test_matches_analytic_derivative(self):
        tau = jnp.linspace(0.0, 1.0, 401)
        dtau = tau[1] - tau[0]
        f = jnp.sin(3.0 * tau)
        d = source_time_derivative(f, dtau)
        # Interior is 2nd order; check away from the ends.
        np.testing.assert_allclose(d[2:-2], 3.0 * jnp.cos(3.0 * tau)[2:-2], atol=2e-4)

    def test_vector_axis(self):
        tau = jnp.linspace(0.0, 1.0, 51)
        dtau = tau[1] - tau[0]
        f = jnp.stack([tau, 2.0 * tau, -tau], axis=-1)[None]  # [1, T, 3]
        d = source_time_derivative(f, dtau, axis=1)
        np.testing.assert_allclose(d[0, 5], [1.0, 2.0, -1.0], atol=1e-12)

    def test_differentiable_wrt_values(self):
        tau = jnp.linspace(0.0, 1.0, 11)
        dtau = tau[1] - tau[0]
        g = jax.grad(lambda f: jnp.sum(source_time_derivative(f, dtau) ** 2))(jnp.sin(tau))
        assert bool(jnp.all(jnp.isfinite(g)))


class TestArrivalAndResample:
    def test_arrival_time(self):
        np.testing.assert_allclose(arrival_times(1.0, 340.0, 340.0), 2.0, atol=1e-14)

    def test_resample_recovers_shifted_signal(self):
        # One source, pure delay: resampling onto a shifted grid recovers values.
        tau = jnp.linspace(0.0, 1.0, 200)
        vals = jnp.sin(2 * jnp.pi * 3 * tau)[None, :]
        delay = 0.2
        arrival = (tau + delay)[None, :]
        t_obs = jnp.linspace(0.3, 0.9, 150)
        out = resample_sum(arrival, vals, t_obs)
        np.testing.assert_allclose(out, jnp.sin(2 * jnp.pi * 3 * (t_obs - delay)), atol=2e-3)

    def test_resample_sums_over_sources(self):
        tau = jnp.linspace(0.0, 1.0, 100)
        arrival = jnp.stack([tau, tau])  # identical, no delay
        vals = jnp.stack([jnp.ones_like(tau), 2.0 * jnp.ones_like(tau)])
        out = resample_sum(arrival, vals, jnp.linspace(0.1, 0.9, 50))
        np.testing.assert_allclose(out, 3.0, atol=1e-12)

    def test_default_observer_grid_within_common_window(self):
        tau = jnp.linspace(0.0, 1.0, 10)
        arrival = jnp.stack([tau + 0.5, tau + 0.2])  # different delays
        grid = default_observer_grid(arrival, 20)
        assert float(grid[0]) == pytest.approx(0.5)  # max of per-source minima
        assert float(grid[-1]) == pytest.approx(1.2)  # min of per-source maxima
