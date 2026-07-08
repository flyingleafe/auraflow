"""Stationary monopole (compact pulsating sphere) vs analytic solution.

Analytic point mass source of volume-flow ``Q(t)``:
``p'(x, t) = rho0 Qdot(t - r/c0) / (4 pi r)`` (thickness / monopole noise).
Exercised through both the core F1A kernel and the permeable static fast path.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.core.medium import Medium
from auraflow.fwh import f1a_permeable_static, f1a_pressure


def _gaussian_pulse(t, t0, sig):
    return jnp.exp(-0.5 * ((t - t0) / sig) ** 2)


def _gaussian_pulse_dot(t, t0, sig):
    return -((t - t0) / sig**2) * jnp.exp(-0.5 * ((t - t0) / sig) ** 2)


class TestStationaryMonopole:
    def _setup(self, r0=2.0, n=400):
        med = Medium()
        c0 = float(med.c0)
        tau = jnp.linspace(0.0, 0.02, n)
        t0, sig = 0.01, 8e-4
        x_obs = jnp.array([[r0, 0.0, 0.0]])
        y = jnp.zeros((1, n, 3))
        zero = jnp.zeros((1, n, 3))
        q_vol = _gaussian_pulse(tau, t0, sig)  # volume flow Q(t) [m^3/s]
        qn = (med.rho0 * q_vol)[None, :]  # Q_n area density (single unit panel)
        t_obs = jnp.linspace(tau[5] + r0 / c0, tau[-6] + r0 / c0, n)
        # analytic p' at the observer times
        qdot = _gaussian_pulse_dot(t_obs - r0 / c0, t0, sig)
        p_ana = med.rho0 * qdot / (4.0 * np.pi * r0)
        return med, tau, x_obs, y, zero, qn, t_obs, p_ana

    def test_kernel_matches_analytic(self):
        med, tau, x_obs, y, zero, qn, t_obs, p_ana = self._setup()
        load = jnp.zeros((1, tau.size, 3))
        pt, pl = f1a_pressure(x_obs, y, zero, zero, qn, load, med, tau, t_obs)
        mask = jnp.abs(p_ana) > 0.02 * jnp.max(jnp.abs(p_ana))
        rel = jnp.linalg.norm((pt[0] - p_ana)[mask]) / jnp.linalg.norm(p_ana[mask])
        assert float(rel) < 0.01
        assert float(jnp.max(jnp.abs(pl))) == pytest.approx(0.0, abs=1e-12)

    def test_permeable_static_path_matches_analytic(self):
        # Same monopole delivered through the permeable-surface static fast path:
        # a single unit panel whose normal points at the observer, carrying a
        # radial mass flux rho0 * u_n = rho0 * Q(t) (area = 1).
        med, tau, x_obs, y, zero, qn, t_obs, p_ana = self._setup()
        n = tau.size
        normal = jnp.array([[1.0, 0.0, 0.0]])
        area = jnp.array([1.0])
        rho = jnp.full((1, n), float(med.rho0))
        q_vol = qn[0] / med.rho0
        u = q_vol[None, :, None] * normal[:, None, :]  # u_n = Q(t)
        p = jnp.full((1, n), float(med.p0))
        pt, pl = f1a_permeable_static(x_obs, y[:, 0], normal, area, rho, u, p, med, tau, t_obs)
        mask = jnp.abs(p_ana) > 0.02 * jnp.max(jnp.abs(p_ana))
        rel = jnp.linalg.norm((pt[0] - p_ana)[mask]) / jnp.linalg.norm(p_ana[mask])
        assert float(rel) < 0.01

    def test_inverse_distance_decay(self):
        # Peak thickness pressure scales as 1/r.
        med, tau, x_obs, y, zero, qn, t_obs, _ = self._setup(r0=2.0)
        c0 = float(med.c0)
        load = jnp.zeros((1, tau.size, 3))
        peak = {}
        for r0 in (2.0, 4.0):
            xo = jnp.array([[r0, 0.0, 0.0]])
            to = jnp.linspace(tau[5] + r0 / c0, tau[-6] + r0 / c0, tau.size)
            pt, _ = f1a_pressure(xo, y, zero, zero, qn, load, med, tau, to)
            peak[r0] = float(jnp.max(jnp.abs(pt)))
        assert peak[2.0] / peak[4.0] == pytest.approx(2.0, rel=0.02)
