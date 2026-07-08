"""Moving compact monopole: Doppler frequency shift of a harmonic source.

A harmonic monopole in uniform rectilinear subsonic motion toward a distant
on-axis observer has its frequency shifted by the convective Doppler factor
``f_obs = f_src / (1 - M cos theta)``; here ``cos theta = 1`` (approach along
the line of sight), so ``f_obs = f_src / (1 - M)``.
"""

import jax.numpy as jnp
import numpy as np

from auraflow.core.medium import Medium
from auraflow.fwh import f1a_thickness
from auraflow.signal.spectra import narrowband_spectrum


def _received_frequency(mach, f_src=500.0):
    med = Medium()
    c0 = float(med.c0)
    v_x = mach * c0
    n = 2048
    # Emit over enough periods; keep the source well away from the observer.
    tau = jnp.linspace(0.0, 0.04, n)
    x_far = 60.0
    y_x = -5.0 + v_x * tau  # moving in +x, toward the observer at +x_far
    y = jnp.stack([y_x, jnp.zeros(n), jnp.zeros(n)], axis=-1)[None]  # [1, n, 3]
    v = jnp.zeros((1, n, 3)).at[..., 0].set(v_x)
    a = jnp.zeros((1, n, 3))
    qn = jnp.sin(2 * np.pi * f_src * tau)[None, :]
    x_obs = jnp.array([[x_far, 0.0, 0.0]])
    # Common arrival window (source approaches, so delay shrinks over time).
    r0 = x_far - float(y_x[0])
    r1 = x_far - float(y_x[-1])
    t_obs = jnp.linspace(tau[3] + r0 / c0, tau[-4] + r1 / c0, n)
    fs = float(1.0 / (t_obs[1] - t_obs[0]))
    p = f1a_thickness(x_obs, y, v, a, qn, med, tau, t_obs)[0]
    freqs, amp = narrowband_spectrum(p, fs)
    return float(freqs[int(jnp.argmax(amp))])


class TestDopplerShift:
    def test_approaching_source_upshift(self):
        mach = 0.2
        f_src = 500.0
        f_obs = _received_frequency(mach, f_src)
        f_expected = f_src / (1.0 - mach)
        assert abs(f_obs - f_expected) / f_expected < 0.005

    def test_faster_source_larger_shift(self):
        f_slow = _received_frequency(0.1)
        f_fast = _received_frequency(0.3)
        assert f_fast > f_slow > 500.0
