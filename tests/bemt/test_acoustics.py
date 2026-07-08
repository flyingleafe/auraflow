"""BEMT -> Farassat-1A tonal noise: BPF periodicity, induced effect, gradients."""

import jax
import jax.numpy as jnp
import numpy as np

from auraflow.bemt.acoustics import rotor_tonal_noise
from auraflow.core.airfoil import ThinAirfoilPolar
from auraflow.core.blade import BladeGeometry, Rotor
from auraflow.core.medium import Medium
from auraflow.signal.spectra import narrowband_spectrum, oaspl


def _small_rotor(n_stations=8, n_blades=2):
    blade = BladeGeometry.linear(
        radius=0.15,
        hub_radius=0.02,
        n_stations=n_stations,
        chord_root=0.018,
        chord_tip=0.012,
        twist_root=np.radians(16.0),
        twist_tip=np.radians(8.0),
    )
    return Rotor(blade=blade, n_blades=n_blades)


# Off-axis observer (on-axis hover loading is steady -> no tone there).
_OBS = jnp.array([[0.75, 0.0, -1.30]])


class TestBladePassingTone:
    def test_fundamental_at_bpf_and_finite(self):
        rotor = _small_rotor(n_blades=2)
        med = Medium()
        omega = 500.0
        n = 600
        t = jnp.linspace(0.0, 0.08, n)
        p_tot, p_th, p_ld, t_obs = rotor_tonal_noise(
            rotor,
            med,
            t,
            omega,
            _OBS,
            collective=np.radians(6.0),
            polar=ThinAirfoilPolar(cd0=0.01),
        )
        p = np.asarray(p_tot[0])
        assert np.all(np.isfinite(p))
        assert np.all(np.isfinite(np.asarray(p_th)))
        assert np.all(np.isfinite(np.asarray(p_ld)))
        fs = float(1.0 / (t_obs[1] - t_obs[0]))
        freqs, amp = narrowband_spectrum(jnp.asarray(p), fs)
        f_peak = float(freqs[int(np.argmax(np.asarray(amp)))])
        bpf = rotor.n_blades * omega / (2.0 * np.pi)
        assert abs(f_peak - bpf) / bpf < 0.1

    def test_loading_dominates_thickness_at_low_mach(self):
        rotor = _small_rotor()
        med = Medium()
        t = jnp.linspace(0.0, 0.08, 600)
        _, p_th, p_ld, _ = rotor_tonal_noise(
            rotor,
            med,
            t,
            500.0,
            _OBS,
            collective=np.radians(6.0),
            polar=ThinAirfoilPolar(cd0=0.01),
        )
        # At modest tip Mach the loading tone exceeds the compact thickness tone.
        assert float(jnp.std(p_ld)) > float(jnp.std(p_th))


class TestInducedEffectOnSound:
    def test_induced_on_off_changes_signal(self):
        rotor = _small_rotor()
        med = Medium()
        t = jnp.linspace(0.0, 0.06, 400)
        args = dict(collective=np.radians(8.0), polar=ThinAirfoilPolar(cd0=0.01), thickness=False)
        _, _, p_on, _ = rotor_tonal_noise(rotor, med, t, 500.0, _OBS, include_induced=True, **args)
        _, _, p_off, _ = rotor_tonal_noise(
            rotor, med, t, 500.0, _OBS, include_induced=False, **args
        )
        rel = float(jnp.linalg.norm(p_on - p_off) / jnp.linalg.norm(p_off))
        assert rel > 0.05


class TestEndToEndGradient:
    def _oaspl_of_collective(self, collective):
        rotor = _small_rotor(n_stations=6)
        med = Medium()
        t = jnp.linspace(0.0, 0.05, 160)
        p_tot = rotor_tonal_noise(
            rotor,
            med,
            t,
            480.0,
            _OBS,
            collective=collective,
            polar=ThinAirfoilPolar(cd0=0.01),
            thickness=True,
        )[0]
        return oaspl(p_tot)[0]

    def test_grad_finite_and_matches_fd(self):
        c0 = jnp.asarray(np.radians(7.0))
        g = float(jax.grad(self._oaspl_of_collective)(c0))
        assert np.isfinite(g)
        h = 1e-4
        fd = (
            float(self._oaspl_of_collective(c0 + h)) - float(self._oaspl_of_collective(c0 - h))
        ) / (2 * h)
        np.testing.assert_allclose(g, fd, rtol=2e-3)
