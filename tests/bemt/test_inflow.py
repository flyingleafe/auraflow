"""Glauert momentum inflow and Pitt-Peters linear inflow."""

import jax
import jax.numpy as jnp
import numpy as np

from auraflow.bemt.inflow import glauert_inflow, pitt_peters_inflow, wake_skew_angle


class TestGlauert:
    def test_hover_limit_exact(self):
        ct = 0.012
        lam = float(glauert_inflow(ct, mu=0.0, mu_z=0.0))
        np.testing.assert_allclose(lam, np.sqrt(ct / 2.0), rtol=1e-6)

    def test_high_speed_limit(self):
        ct = 0.012
        mu = 0.5  # mu >> lambda
        lam = float(glauert_inflow(ct, mu=mu, mu_z=0.0))
        np.testing.assert_allclose(lam, ct / (2.0 * mu), rtol=2e-3)

    def test_climb_raises_inflow(self):
        ct = 0.012
        lam0 = float(glauert_inflow(ct, mu=0.0, mu_z=0.0))
        lam_climb = float(glauert_inflow(ct, mu=0.0, mu_z=0.05))
        # Total inflow rises with climb, but by less than mu_z (induced part drops).
        assert lam0 < lam_climb < lam0 + 0.05

    def test_residual_satisfied(self):
        ct, mu, mu_z = 0.02, 0.15, 0.01
        lam = float(glauert_inflow(ct, mu=mu, mu_z=mu_z))
        res = lam - mu_z - ct / (2.0 * np.sqrt(mu**2 + lam**2))
        assert abs(res) < 1e-8

    def test_grad_finite_and_matches_fd(self):
        def f(ct):
            return glauert_inflow(ct, mu=0.1, mu_z=0.0)

        g = float(jax.grad(f)(jnp.asarray(0.012)))
        h = 1e-7
        fd = (float(f(jnp.asarray(0.012 + h))) - float(f(jnp.asarray(0.012 - h)))) / (2 * h)
        assert np.isfinite(g)
        np.testing.assert_allclose(g, fd, rtol=1e-4)


class TestWakeSkew:
    def test_hover_zero_skew(self):
        assert float(wake_skew_angle(0.0, 0.05)) == 0.0

    def test_edgewise_approaches_90(self):
        chi = float(wake_skew_angle(1.0, 1e-6))
        np.testing.assert_allclose(chi, np.pi / 2, atol=1e-5)


class TestPittPeters:
    def test_chi_zero_is_uniform(self):
        lam0 = 0.05
        rr = jnp.linspace(0.1, 1.0, 10)
        psi = jnp.linspace(0.0, 2 * np.pi, 10)
        lam = pitt_peters_inflow(lam0, chi=0.0, r_over_R=rr, psi=psi)
        np.testing.assert_allclose(np.asarray(lam), lam0, rtol=1e-12)

    def test_kx_at_90deg_skew(self):
        lam0 = 0.05
        # At r/R = 1, psi = 0: lambda = lam0 (1 + kx). Extract kx.
        lam = float(pitt_peters_inflow(lam0, chi=np.pi / 2, r_over_R=1.0, psi=0.0))
        kx = lam / lam0 - 1.0
        np.testing.assert_allclose(kx, 15.0 * np.pi / 32.0, rtol=1e-6)

    def test_kx_sign_fore_aft(self):
        # kx > 0: more inflow at psi=0 (downstream) than psi=pi (upstream).
        lam0 = 0.05
        aft = float(pitt_peters_inflow(lam0, chi=np.pi / 3, r_over_R=1.0, psi=0.0))
        fore = float(pitt_peters_inflow(lam0, chi=np.pi / 3, r_over_R=1.0, psi=np.pi))
        assert aft > lam0 > fore
