"""Differentiability gate: grad of OASPL w.r.t. a source parameter vs finite diff."""

import jax
import jax.numpy as jnp
import numpy as np

from auraflow.core.medium import Medium
from auraflow.fwh import f1a_loading
from auraflow.signal.spectra import oaspl


def _oaspl_of_force_amplitude(amp):
    """OASPL at a static observer from a compact dipole with force amplitude ``amp``."""
    med = Medium()
    c0 = float(med.c0)
    n = 300
    tau = jnp.linspace(0.0, 0.02, n)
    t0, sig = 0.01, 1e-3
    r0 = 2.0
    x_obs = jnp.array([[r0, 0.0, 0.0]])
    y = jnp.zeros((1, n, 3))
    zero = jnp.zeros((1, n, 3))
    shape = jnp.exp(-0.5 * ((tau - t0) / sig) ** 2)
    force = amp * shape[None, :, None] * jnp.array([1.0, 0.0, 0.0])[None, None, :]
    t_obs = jnp.linspace(tau[5] + r0 / c0, tau[-6] + r0 / c0, n)
    p = f1a_loading(x_obs, y, zero, zero, force, med, tau, t_obs)[0]
    return oaspl(p)


class TestGradient:
    def test_grad_finite_and_matches_fd(self):
        amp0 = 4.0
        g = jax.grad(_oaspl_of_force_amplitude)(amp0)
        assert bool(jnp.isfinite(g))
        assert float(jnp.abs(g)) > 0.0
        h = 1e-3
        fd = (_oaspl_of_force_amplitude(amp0 + h) - _oaspl_of_force_amplitude(amp0 - h)) / (2 * h)
        np.testing.assert_allclose(float(g), float(fd), rtol=1e-4)

    def test_oaspl_scales_logarithmically(self):
        # Doubling force amplitude raises OASPL by ~20 log10(2) ~ 6.02 dB.
        lo = _oaspl_of_force_amplitude(2.0)
        hi = _oaspl_of_force_amplitude(4.0)
        np.testing.assert_allclose(float(hi - lo), 20 * np.log10(2.0), atol=0.05)
