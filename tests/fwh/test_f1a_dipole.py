"""Stationary point dipole (compact loading) vs analytic solution.

Analytic point force ``F(t)`` at the origin:
``p'(x, t) = (r_hat . Fdot(t - r/c0)) / (4 pi c0 r)  (far field)``
``          + (r_hat . F(t - r/c0)) / (4 pi r^2)      (near field).``
"""

import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.core.medium import Medium
from auraflow.fwh import f1a_loading


def _pulse(t, t0, sig):
    return jnp.exp(-0.5 * ((t - t0) / sig) ** 2)


def _pulse_dot(t, t0, sig):
    return -((t - t0) / sig**2) * jnp.exp(-0.5 * ((t - t0) / sig) ** 2)


class TestStationaryDipole:
    def _run(self, r0, obs_dir, force_dir):
        med = Medium()
        c0 = float(med.c0)
        n = 400
        tau = jnp.linspace(0.0, 0.02, n)
        t0, sig = 0.01, 8e-4
        obs_dir = jnp.asarray(obs_dir) / jnp.linalg.norm(jnp.asarray(obs_dir))
        force_dir = jnp.asarray(force_dir)
        x_obs = (r0 * obs_dir)[None, :]
        y = jnp.zeros((1, n, 3))
        zero = jnp.zeros((1, n, 3))
        amp = 3.0
        force = amp * _pulse(tau, t0, sig)[None, :, None] * force_dir[None, None, :]
        t_obs = jnp.linspace(tau[5] + r0 / c0, tau[-6] + r0 / c0, n)
        p = f1a_loading(x_obs, y, zero, zero, force, med, tau, t_obs)[0]
        # analytic
        tr = t_obs - r0 / c0
        f_ret = amp * _pulse(tr, t0, sig)
        fdot_ret = amp * _pulse_dot(tr, t0, sig)
        rdotf = float(jnp.dot(obs_dir, force_dir))
        p_ana = rdotf * fdot_ret / (4 * np.pi * c0 * r0) + rdotf * f_ret / (4 * np.pi * r0**2)
        return p, p_ana

    def test_matches_analytic_on_axis(self):
        p, p_ana = self._run(2.0, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        mask = jnp.abs(p_ana) > 0.02 * jnp.max(jnp.abs(p_ana))
        rel = jnp.linalg.norm((p - p_ana)[mask]) / jnp.linalg.norm(p_ana[mask])
        assert float(rel) < 0.01

    def test_matches_analytic_oblique(self):
        p, p_ana = self._run(2.5, [1.0, 1.0, 0.5], [0.0, 1.0, 0.0])
        mask = jnp.abs(p_ana) > 0.02 * jnp.max(jnp.abs(p_ana))
        rel = jnp.linalg.norm((p - p_ana)[mask]) / jnp.linalg.norm(p_ana[mask])
        assert float(rel) < 0.02

    def test_null_perpendicular_to_force(self):
        # A dipole radiates nothing broadside (r_hat perpendicular to F).
        p, _ = self._run(2.0, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        assert float(jnp.max(jnp.abs(p))) == pytest.approx(0.0, abs=1e-10)
