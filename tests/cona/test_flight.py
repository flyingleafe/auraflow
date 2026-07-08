"""6-DOF flight dynamics + geometric SE(3) controller.

Small, standalone cases (short trajectories) exercising hover equilibrium, step
tracking, straight-flyover tracking, the controller math, and end-to-end
differentiability through the lax.scan rollout.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from auraflow.cona.flight import (
    ControllerGains,
    Multirotor,
    attitude_error,
    desired_attitude,
    geometric_controller,
    hover,
    simulate,
    straight_flyover,
)

_G = 9.80665


def _ortho_error(R: jax.Array) -> float:
    """Max over time of ||R^T R - I||_F for an [T, 3, 3] attitude history."""
    gram = jnp.einsum("tij,tik->tjk", R, R)
    return float(jnp.max(jnp.linalg.norm(gram - jnp.eye(3), axis=(1, 2))))


class TestHover:
    def test_equilibrium_stays_put(self):
        mr = Multirotor.nasa_1pax()
        gains = ControllerGains.for_vehicle(mr)
        x0 = jnp.array([0.0, 0.0, 50.0])
        t = jnp.linspace(0.0, 3.0, 601)
        h = simulate(mr, gains, hover(x0), t, x0=x0, v0=jnp.zeros(3))
        drift = np.asarray(jnp.linalg.norm(h.x - x0, axis=1))
        assert drift.max() < 1e-3  # sub-millimetre over seconds

    def test_per_rotor_thrust_is_weight_over_four(self):
        mr = Multirotor.nasa_1pax()
        gains = ControllerGains.for_vehicle(mr)
        x0 = jnp.array([0.0, 0.0, 50.0])
        t = jnp.linspace(0.0, 1.0, 201)
        h = simulate(mr, gains, hover(x0), t, x0=x0, v0=jnp.zeros(3))
        expected = float(mr.mass) * _G / 4.0
        np.testing.assert_allclose(np.asarray(h.rotor_thrusts), expected, rtol=1e-6)


class TestStepResponse:
    def test_converges_no_nan_orthonormal(self):
        mr = Multirotor.nasa_1pax()
        gains = ControllerGains.for_vehicle(mr)
        x0 = jnp.array([0.0, 0.0, 50.0])
        target = jnp.array([0.0, 1.0, 50.0])  # 1 m lateral step
        t = jnp.linspace(0.0, 10.0, 1001)
        h = simulate(mr, gains, hover(target), t, x0=x0, v0=jnp.zeros(3))
        err = np.asarray(jnp.linalg.norm(h.x - target, axis=1))
        assert np.all(np.isfinite(np.asarray(h.x)))
        assert err[-1] < 1e-2  # settled to the commanded offset
        assert err[-1] < 0.05 * err[0]  # substantial convergence from 1 m
        assert _ortho_error(h.R) < 1e-8


class TestFlyover:
    def test_tracks_straight_line(self):
        mr = Multirotor.nasa_1pax()
        gains = ControllerGains.for_vehicle(mr)
        speed, altitude = 10.0, 50.0
        ref = straight_flyover(speed, altitude, heading=0.0, t_pass=5.0)
        t = jnp.linspace(0.0, 10.0, 1001)
        x_start = ref(t[0])[0]
        h = simulate(mr, gains, ref, t, x0=x_start, v0=jnp.array([speed, 0.0, 0.0]))
        # Cross-track = deviation perpendicular to the flight direction (+x): y, z.
        x_des = jnp.stack([ref(tt)[0] for tt in t])
        cross = np.asarray(jnp.linalg.norm((h.x - x_des)[:, 1:], axis=1))
        assert cross[100:].max() < 0.1  # after transient
        # RPM histories smooth: no spikes between consecutive samples.
        speeds = np.asarray(h.rotor_speeds)
        assert np.all(np.isfinite(speeds))
        assert np.abs(np.diff(speeds, axis=0)).max() < 1e-2

    def test_history_shapes(self):
        mr = Multirotor.nasa_1pax()
        gains = ControllerGains.for_vehicle(mr)
        t = jnp.linspace(0.0, 0.5, 51)
        x0 = jnp.array([0.0, 0.0, 30.0])
        h = simulate(mr, gains, hover(x0), t, x0=x0, v0=jnp.zeros(3))
        nt, nr = t.shape[0], mr.n_rotors
        assert h.x.shape == (nt, 3)
        assert h.v.shape == (nt, 3)
        assert h.R.shape == (nt, 3, 3)
        assert h.Omega_body.shape == (nt, 3)
        assert h.rotor_speeds.shape == (nt, nr)
        assert h.rotor_thrusts.shape == (nt, nr)


class TestControllerMath:
    def test_moment_reduces_to_gyroscopic_feedforward(self):
        # With zero errors (R = Rc, Omega = Omega_c) the moment reduces to the
        # gyroscopic feed-forward Omega x J Omega.
        mr = Multirotor.nasa_1pax()
        gains = ControllerGains.for_vehicle(mr)
        x0 = jnp.array([0.0, 0.0, 20.0])
        ref = hover(x0)(jnp.array(0.0))  # hover -> force_des = mg e3, Rc = I
        w = jnp.array([0.05, -0.03, 0.02])
        state = (x0, jnp.zeros(3), jnp.eye(3), w)  # R = I = Rc, Omega = w
        _, M = geometric_controller(state, ref, mr, gains, omega_c=w)
        expected = jnp.cross(w, mr.inertia @ w)
        np.testing.assert_allclose(np.asarray(M), np.asarray(expected), atol=1e-10)

    def test_attitude_error_is_antisymmetric_map(self):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)

        # Two random rotations via matrix exponential of skew matrices.
        def rand_rot(k):
            from auraflow.cona.flight import hat

            w = jax.random.normal(k, (3,))
            return jax.scipy.linalg.expm(hat(w))

        R = rand_rot(k1)
        Rc = rand_rot(k2)
        S = Rc.T @ R - R.T @ Rc
        # The inner matrix is exactly antisymmetric.
        np.testing.assert_allclose(np.asarray(S), -np.asarray(S).T, atol=1e-12)
        # attitude_error is its vee; zero iff R == Rc.
        e = attitude_error(R, Rc)
        assert e.shape == (3,)
        assert float(jnp.linalg.norm(attitude_error(R, R))) < 1e-12

    def test_desired_attitude_is_rotation(self):
        force_des = jnp.array([1.0, -2.0, 15.0])
        Rc = desired_attitude(force_des, jnp.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(np.asarray(Rc.T @ Rc), np.eye(3), atol=1e-12)
        np.testing.assert_allclose(float(jnp.linalg.det(Rc)), 1.0, atol=1e-12)
        # Body +z (thrust axis) aligns with the desired thrust direction.
        np.testing.assert_allclose(
            np.asarray(Rc[:, 2]), np.asarray(force_des / jnp.linalg.norm(force_des)), atol=1e-12
        )


class TestGradient:
    def test_grad_mean_speed_wrt_mass(self):
        t = jnp.linspace(0.0, 1.0, 101)
        x0 = jnp.array([0.0, 0.0, 50.0])

        def mean_speed(mass):
            mr = eqx.tree_at(lambda m: m.mass, Multirotor.nasa_1pax(), mass)
            gains = ControllerGains.for_vehicle(mr)
            h = simulate(mr, gains, hover(x0), t, x0=x0, v0=jnp.zeros(3))
            return jnp.mean(h.rotor_speeds)

        val, grad = jax.value_and_grad(mean_speed)(583.85)
        assert np.isfinite(float(grad))
        # d/dm sqrt(m g /(4 k_f)) = 0.5 * omega_hover / m at hover.
        expected = 0.5 * float(val) / 583.85
        np.testing.assert_allclose(float(grad), expected, rtol=1e-4)

    def test_grad_final_error_wrt_gain(self):
        t = jnp.linspace(0.0, 2.0, 201)
        x0 = jnp.array([0.0, 0.0, 50.0])
        target = jnp.array([0.0, 1.0, 50.0])

        def final_err(k_x):
            mr = Multirotor.nasa_1pax()
            gains = eqx.tree_at(lambda g: g.k_x, ControllerGains.for_vehicle(mr), k_x)
            h = simulate(mr, gains, hover(target), t, x0=x0, v0=jnp.zeros(3))
            return jnp.linalg.norm(h.x[-1] - target)

        _, grad = jax.value_and_grad(final_err)(
            float(ControllerGains.for_vehicle(Multirotor.nasa_1pax()).k_x)
        )
        assert np.isfinite(float(grad))
        assert abs(float(grad)) > 0.0
