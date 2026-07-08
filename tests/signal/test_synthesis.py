"""Tests for auraflow.signal.synthesis: STFT/iSTFT, Griffin-Lim, band spreading."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.signal import (
    griffin_lim,
    istft,
    stft,
    third_octave_bands,
    third_octave_levels,
    third_octave_to_stft_mag,
)

FS = 44100.0


def spectral_convergence(mag_target: jnp.ndarray, y: jnp.ndarray, n_fft: int, hop: int) -> float:
    mag_y = jnp.abs(stft(y, n_fft, hop))
    return float(jnp.linalg.norm(mag_y - mag_target) / jnp.linalg.norm(mag_target))


class TestStftIstft:
    @pytest.mark.parametrize("hop", [256, 128])  # 50% and 75% overlap
    def test_hann_roundtrip(self, hop):
        rng = np.random.default_rng(0)
        n_fft = 512
        x = jnp.asarray(rng.normal(size=4096))
        z = stft(x, n_fft, hop)
        n_frames = 1 + (4096 - n_fft) // hop
        assert z.shape == (n_frames, n_fft // 2 + 1)
        assert jnp.iscomplexobj(z)
        y = istft(z, n_fft, hop)
        length = (n_frames - 1) * hop + n_fft
        assert y.shape == (length,)
        # Sample 0 is lost (periodic Hann window is zero there); the rest is exact.
        np.testing.assert_allclose(np.asarray(y)[1:], np.asarray(x)[1:length], atol=1e-10)

    def test_boxcar_roundtrip_no_overlap(self):
        rng = np.random.default_rng(1)
        n_fft = 256
        x = jnp.asarray(rng.normal(size=2048))
        y = istft(stft(x, n_fft, n_fft, window="boxcar"), n_fft, n_fft, window="boxcar")
        np.testing.assert_allclose(np.asarray(y), np.asarray(x), atol=1e-12)

    def test_stft_matches_manual_rfft(self):
        rng = np.random.default_rng(2)
        n_fft, hop = 128, 64
        x = rng.normal(size=1024)
        z = stft(jnp.asarray(x), n_fft, hop)
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)
        expected = np.fft.rfft(x[2 * hop : 2 * hop + n_fft] * w)
        np.testing.assert_allclose(np.asarray(z[2]), expected, rtol=1e-10, atol=1e-12)


class TestGriffinLim:
    def _sine_mixture_mag(self, n_fft: int, hop: int):
        fs, n = 16000.0, 16000
        t = np.arange(n) / fs
        x = (
            np.sin(2.0 * np.pi * 440.0 * t)
            + 0.5 * np.sin(2.0 * np.pi * 1000.0 * t + 1.0)
            + 0.25 * np.sin(2.0 * np.pi * 3000.0 * t + 2.0)
        )
        return jnp.abs(stft(jnp.asarray(x), n_fft, hop))

    def test_converges_on_sine_mixture(self):
        n_fft, hop = 1024, 256
        mag = self._sine_mixture_mag(n_fft, hop)
        y = griffin_lim(mag, n_fft, hop, n_iters=60, key=jax.random.PRNGKey(1))
        assert y.shape == ((mag.shape[0] - 1) * hop + n_fft,)
        assert spectral_convergence(mag, y, n_fft, hop) < 0.1

    def test_classic_no_momentum_converges(self):
        n_fft, hop = 1024, 256
        mag = self._sine_mixture_mag(n_fft, hop)
        y = griffin_lim(mag, n_fft, hop, n_iters=60, key=jax.random.PRNGKey(1), momentum=0.0)
        assert spectral_convergence(mag, y, n_fft, hop) < 0.2

    def test_deterministic_given_key(self):
        n_fft, hop = 512, 128
        mag = self._sine_mixture_mag(n_fft, hop)
        key = jax.random.PRNGKey(7)
        y1 = griffin_lim(mag, n_fft, hop, n_iters=10, key=key)
        y2 = griffin_lim(mag, n_fft, hop, n_iters=10, key=key)
        np.testing.assert_array_equal(np.asarray(y1), np.asarray(y2))
        # Default key works and is reproducible too.
        y3 = griffin_lim(mag, n_fft, hop, n_iters=10)
        y4 = griffin_lim(mag, n_fft, hop, n_iters=10, key=jax.random.PRNGKey(0))
        np.testing.assert_array_equal(np.asarray(y3), np.asarray(y4))


class TestThirdOctaveToStftMag:
    def test_shape_and_zeros_outside_bands(self):
        n_fft = 2048
        centers, _ = third_octave_bands(200.0, 4000.0)
        levels = jnp.full((centers.shape[0],), 80.0)
        mag = third_octave_to_stft_mag(levels, centers, n_fft, FS)
        assert mag.shape == (n_fft // 2 + 1,)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / FS)
        lo = float(centers[0]) * 10.0**-0.05
        hi = float(centers[-1]) * 10.0**0.05
        outside = (freqs < lo) | (freqs >= hi)
        assert np.all(np.asarray(mag)[outside] == 0.0)
        assert np.all(np.asarray(mag)[~outside] > 0.0)

    def test_spectrogram_input_broadcasts(self):
        n_fft, n_frames = 1024, 7
        centers, _ = third_octave_bands(500.0, 2000.0)
        levels = jnp.broadcast_to(jnp.full((centers.shape[0],), 70.0), (n_frames, centers.shape[0]))
        mag = third_octave_to_stft_mag(levels, centers, n_fft, FS)
        assert mag.shape == (n_frames, n_fft // 2 + 1)

    def test_griffin_lim_band_level_roundtrip(self):
        # Smooth (pink-noise-like, gently tilted) band levels, 100 Hz - 8 kHz:
        # spread to an STFT magnitude, synthesize with Griffin-Lim, re-analyze.
        n_fft, hop, n_frames = 4096, 1024, 24
        centers, _ = third_octave_bands(100.0, 8000.0)
        target = 70.0 - 1.0 * jnp.log2(centers / 1000.0)  # dB, smooth tilt
        levels = jnp.broadcast_to(target, (n_frames, centers.shape[0]))
        mag = third_octave_to_stft_mag(levels, centers, n_fft, FS)
        y = griffin_lim(mag, n_fft, hop, n_iters=60, key=jax.random.PRNGKey(3))
        out_centers, out_levels = third_octave_levels(y, FS, nperseg=n_fft, fmin=100.0, fmax=8000.0)
        np.testing.assert_allclose(np.asarray(out_centers), np.asarray(centers), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(out_levels), np.asarray(target), atol=1.5)
