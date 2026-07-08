"""Tests for auraflow.signal.spectra against scipy and analytic references."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.signal as sps

from auraflow.signal import (
    P_REF,
    a_weighted_oaspl,
    a_weighting,
    harmonic_levels,
    narrowband_spectrum,
    oaspl,
    psd_to_db,
    spl,
    third_octave_bands,
    third_octave_levels,
    third_octave_spectrogram,
    welch_psd,
)

FS = 44100.0


def sine(freq: float, amp: float, fs: float = FS, n: int = 44100) -> jnp.ndarray:
    t = np.arange(n) / fs
    return jnp.asarray(amp * np.sin(2.0 * np.pi * freq * t))


class TestSpl:
    def test_spl_reference(self):
        assert float(spl(P_REF)) == pytest.approx(0.0, abs=1e-12)
        assert float(spl(10 * P_REF)) == pytest.approx(20.0, abs=1e-12)

    def test_oaspl_pure_sine(self):
        amp = 0.2
        x = sine(1000.0, amp)
        expected = 20.0 * np.log10(amp / np.sqrt(2.0) / P_REF)
        assert float(oaspl(x)) == pytest.approx(expected, abs=0.01)

    def test_oaspl_removes_mean_and_respects_axis(self):
        x = sine(1000.0, 0.2, n=8820)
        stacked = jnp.stack([x, x + 5.0])  # [O, T], one signal with DC offset
        levels = oaspl(stacked, axis=-1)
        assert levels.shape == (2,)
        np.testing.assert_allclose(levels[0], levels[1], atol=1e-9)

    def test_psd_to_db(self):
        assert float(psd_to_db(P_REF**2)) == pytest.approx(0.0, abs=1e-12)
        assert float(psd_to_db(100.0 * P_REF**2)) == pytest.approx(20.0, abs=1e-12)


class TestWelch:
    @pytest.mark.parametrize("noverlap", [None, 768])
    def test_matches_scipy_white_noise(self, noverlap):
        rng = np.random.default_rng(0)
        x = rng.normal(size=16384)
        nperseg = 1024
        f_ref, p_ref = sps.welch(x, fs=FS, window="hann", nperseg=nperseg, noverlap=noverlap)
        f, p = welch_psd(jnp.asarray(x), FS, nperseg=nperseg, noverlap=noverlap)
        np.testing.assert_allclose(np.asarray(f), f_ref, rtol=1e-12)
        # DC bin is ~0 after constant detrend; compare it in absolute terms.
        np.testing.assert_allclose(np.asarray(p)[1:], p_ref[1:], rtol=1e-6)
        assert abs(float(p[0]) - p_ref[0]) < 1e-12

    def test_matches_scipy_sine(self):
        x = np.asarray(sine(1000.0, 0.5, n=32768))
        f_ref, p_ref = sps.welch(x, fs=FS, window="hann", nperseg=2048)
        f, p = welch_psd(jnp.asarray(x), FS, nperseg=2048)
        np.testing.assert_allclose(np.asarray(f), f_ref, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(p)[1:], p_ref[1:], rtol=1e-6, atol=1e-25)

    def test_odd_nperseg_matches_scipy(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=8192)
        f_ref, p_ref = sps.welch(x, fs=FS, window="hann", nperseg=511)
        f, p = welch_psd(jnp.asarray(x), FS, nperseg=511)
        np.testing.assert_allclose(np.asarray(f), f_ref, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(p)[1:], p_ref[1:], rtol=1e-6)

    def test_parseval_white_noise(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=65536)
        f, p = welch_psd(jnp.asarray(x), FS, nperseg=1024)
        df = float(f[1] - f[0])
        assert float(jnp.sum(p) * df) == pytest.approx(float(np.var(x)), rel=0.03)

    def test_parseval_sine(self):
        amp = 0.7
        x = sine(1000.0, amp, n=65536)
        f, p = welch_psd(x, FS, nperseg=4096)
        df = float(f[1] - f[0])
        assert float(jnp.sum(p) * df) == pytest.approx(amp**2 / 2.0, rel=0.01)


class TestNarrowband:
    def test_sine_peak_amplitude(self):
        amp = 0.3
        x = sine(1000.0, amp)  # bin-centered: df = 1 Hz
        freqs, mag = narrowband_spectrum(x, FS)
        k = int(jnp.argmax(mag))
        assert float(freqs[k]) == pytest.approx(1000.0, abs=1e-9)
        assert float(mag[k]) == pytest.approx(amp, rel=1e-9)


class TestHarmonicLevels:
    def test_bandlimited_sawtooth(self):
        fs, n, f0, n_harm = 40960.0, 40960, 160.0, 6
        t = np.arange(n) / fs
        x = sum((2.0 / (np.pi * k)) * np.sin(2.0 * np.pi * k * f0 * t) for k in range(1, 13))
        freqs, levels = harmonic_levels(jnp.asarray(x), fs, f0, n_harm)
        k = np.arange(1, n_harm + 1)
        expected = 20.0 * np.log10((2.0 / (np.pi * k)) / np.sqrt(2.0) / P_REF)
        np.testing.assert_allclose(np.asarray(freqs), k * f0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(levels), expected, atol=0.01)

    def test_off_bin_sine(self):
        # Fundamental not on an FFT bin: parabolic refinement must recover
        # frequency and level despite scalloping.
        fs, n, f0, amp = 44100.0, 44100, 163.37, 0.5
        t = np.arange(n) / fs
        x = jnp.asarray(amp * np.sin(2.0 * np.pi * f0 * t))
        freqs, levels = harmonic_levels(x, fs, f0, 1)
        expected = 20.0 * np.log10(amp / np.sqrt(2.0) / P_REF)
        assert float(freqs[0]) == pytest.approx(f0, abs=0.3)
        assert float(levels[0]) == pytest.approx(expected, abs=0.3)


class TestAWeighting:
    # IEC 61672-1 table values (dB, rounded to 0.1). The table is defined at
    # the exact preferred band frequencies, hence 15848.9 Hz for nominal 16 kHz.
    SPOT_VALUES = {
        63.0: -26.2,
        100.0: -19.1,
        125.0: -16.1,
        250.0: -8.6,
        500.0: -3.2,
        1000.0: 0.0,
        2000.0: 1.2,
        2500.0: 1.3,
        4000.0: 1.0,
        8000.0: -1.1,
        15848.9: -6.6,
    }

    def test_zero_at_1khz(self):
        assert float(a_weighting(1000.0)) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize(("freq", "expected"), sorted(SPOT_VALUES.items()))
    def test_spot_values(self, freq, expected):
        assert float(a_weighting(freq)) == pytest.approx(expected, abs=0.1)

    def test_a_weighted_oaspl_1khz_sine(self):
        x = sine(1000.0, 0.2)
        assert float(a_weighted_oaspl(x, FS)) == pytest.approx(float(oaspl(x)), abs=0.01)

    def test_a_weighted_oaspl_100hz_sine(self):
        x = sine(100.0, 0.2)
        expected = float(oaspl(x)) + float(a_weighting(100.0))
        assert float(a_weighted_oaspl(x, FS)) == pytest.approx(expected, abs=0.02)


class TestThirdOctave:
    def test_band_centers_and_edges(self):
        centers, edges = third_octave_bands()
        centers, edges = np.asarray(centers), np.asarray(edges)
        assert centers.shape == (31,)
        assert edges.shape == (32,)
        assert 1000.0 in centers  # exactly, not approximately
        assert centers[0] == pytest.approx(19.9526, abs=1e-3)
        assert centers[-1] == pytest.approx(19952.62, abs=0.1)
        np.testing.assert_allclose(edges[:-1], centers * 10.0**-0.05, rtol=1e-12)
        np.testing.assert_allclose(edges[1:], centers * 10.0**0.05, rtol=1e-12)
        np.testing.assert_allclose(centers[1:] / centers[:-1], 10.0**0.1, rtol=1e-12)

    def test_band_range_selection(self):
        centers, _ = third_octave_bands(100.0, 8000.0)
        centers = np.asarray(centers)
        assert centers[0] == pytest.approx(100.0, rel=1e-12)
        assert centers[-1] == pytest.approx(7943.28, abs=0.01)
        assert centers.shape == (20,)

    def test_sine_energy_concentrated_in_band(self):
        x = sine(1000.0, 1.0)
        centers, levels = third_octave_levels(x, FS, nperseg=4096)
        powers = np.asarray(10.0 ** (levels / 10.0))
        band = int(np.argmin(np.abs(np.asarray(centers) - 1000.0)))
        assert np.asarray(centers)[band] == 1000.0
        assert powers[band] / powers.sum() >= 0.95
        assert float(levels[band]) == pytest.approx(float(oaspl(x)), abs=0.3)

    def test_spectrogram_shape_and_stationarity(self):
        rng = np.random.default_rng(2)
        x = jnp.asarray(rng.normal(size=44100))
        frame_len, hop = 4096, 2048
        centers, levels = third_octave_spectrogram(x, FS, frame_len, hop)
        n_frames = 1 + (44100 - frame_len) // hop
        assert levels.shape == (n_frames, centers.shape[0])
        # Mid bands of stationary white noise: levels steady across frames.
        mid = np.asarray(levels)[:, 15:25]
        assert np.all(np.isfinite(mid))
        assert np.all(mid.std(axis=0) < 3.0)

    def test_spectrogram_consistent_with_welch_levels(self):
        rng = np.random.default_rng(4)
        x = jnp.asarray(rng.normal(size=44100))
        centers, spec = third_octave_spectrogram(x, FS, 4096, 2048)
        _, ref = third_octave_levels(x, FS, nperseg=4096)
        frame_mean = 10.0 * np.log10(np.mean(10.0 ** (np.asarray(spec) / 10.0), axis=0))
        np.testing.assert_allclose(frame_mean[10:], np.asarray(ref)[10:], atol=0.5)


class TestGradients:
    def test_grad_a_weighted_energy_is_finite(self):
        rng = np.random.default_rng(5)
        x = jnp.asarray(rng.normal(size=2048))

        grad = jax.grad(lambda s: a_weighted_oaspl(s, 8000.0))(x)
        assert grad.shape == x.shape
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.max(jnp.abs(grad))) > 0.0

    def test_grad_third_octave_energy_is_finite(self):
        rng = np.random.default_rng(6)
        x = jnp.asarray(rng.normal(size=4096))

        def total_energy(s):
            _, levels = third_octave_levels(s, FS, nperseg=1024, fmin=100.0, fmax=3150.0)
            return jnp.sum(10.0 ** (levels / 10.0))

        grad = jax.grad(total_energy)(x)
        assert bool(jnp.all(jnp.isfinite(grad)))
