"""Dryden low-altitude turbulence (MIL-F-8785C).

Statistical checks with fixed seeds: the discrete forming filters must reproduce
their target stationary variances, zero wind must give zero gust, and the
generator must be differentiable and scan-safe.
"""

import jax
import jax.numpy as jnp
import numpy as np

from auraflow.cona.gusts import (
    W20_PRESETS,
    dryden_gust,
    dryden_parameters,
)


class TestParameters:
    def test_low_altitude_scale_lengths_and_intensities(self):
        # h = 50 m, moderate. L_w = h; sigma_w = 0.1 * W20.
        p = dryden_parameters(50.0, "moderate")
        assert float(p.L_w) == 50.0
        np.testing.assert_allclose(float(p.sigma_w), 0.1 * W20_PRESETS["moderate"], rtol=1e-12)
        # Lu = Lv > Lw at low altitude; sigma_u = sigma_v > sigma_w.
        assert float(p.L_u) == float(p.L_v) > float(p.L_w)
        assert float(p.sigma_u) == float(p.sigma_v) > float(p.sigma_w)

    def test_presets_ordered(self):
        assert W20_PRESETS["light"] < W20_PRESETS["moderate"] < W20_PRESETS["severe"]


class TestVariance:
    def test_component_variances_match_targets(self):
        # Stationary variance is exact by construction; verify with an ensemble
        # (pooled over seeds + time) so the Monte-Carlo error is small
        # regardless of the (long) turbulence correlation time.
        altitude, v_air, w20, dt, n = 30.0, 15.0, "moderate", 0.01, 400
        p = dryden_parameters(altitude, w20)
        keys = jax.random.split(jax.random.PRNGKey(1), 800)
        g = jax.vmap(lambda k: dryden_gust(k, altitude, v_air, w20, dt, n))(keys)
        g = np.asarray(g)  # [N, T, 3]
        targets = [float(p.sigma_u), float(p.sigma_v), float(p.sigma_w)]
        for i, sigma in enumerate(targets):
            ratio = g[:, :, i].var() / sigma**2
            assert abs(ratio - 1.0) < 0.05, f"component {i}: variance ratio {ratio}"

    def test_zero_wind_gives_zero(self):
        g = dryden_gust(jax.random.PRNGKey(3), 50.0, 20.0, 0.0, 0.01, 100)
        assert float(jnp.max(jnp.abs(g))) == 0.0

    def test_u_autocorrelation_decays_like_scale(self):
        # Sanity on the longitudinal filter: lag-1 autocorrelation ~ exp(-V dt/Lu).
        altitude, v_air, w20, dt, n = 30.0, 15.0, "moderate", 0.01, 200000
        p = dryden_parameters(altitude, w20)
        g = np.asarray(dryden_gust(jax.random.PRNGKey(5), altitude, v_air, w20, dt, n))[:, 0]
        g = g - g.mean()
        rho1 = float(np.mean(g[1:] * g[:-1]) / np.mean(g * g))
        expected = float(np.exp(-v_air * dt / float(p.L_u)))
        assert abs(rho1 - expected) < 0.02


class TestShapesAndGrad:
    def test_shape(self):
        g = dryden_gust(jax.random.PRNGKey(0), 40.0, 12.0, "light", 0.01, 64)
        assert g.shape == (64, 3)
        assert np.all(np.isfinite(np.asarray(g)))

    def test_single_step(self):
        g = dryden_gust(jax.random.PRNGKey(0), 40.0, 12.0, "light", 0.01, 1)
        assert g.shape == (1, 3)

    def test_differentiable_in_intensity(self):
        # d(gust variance)/d(W20) is finite through the whole scan.
        def rms(w20):
            g = dryden_gust(jax.random.PRNGKey(7), 30.0, 15.0, w20, 0.01, 500)
            return jnp.sqrt(jnp.mean(g[:, 2] ** 2))

        val, grad = jax.value_and_grad(rms)(8.0)
        assert np.isfinite(float(val)) and np.isfinite(float(grad))
        assert float(grad) > 0.0  # stronger wind -> larger gust RMS
