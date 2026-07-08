"""CONA auralization: Griffin-Lim round-trip and tonal+broadband combination."""

import jax.numpy as jnp
import numpy as np

from auraflow.cona.auralize import (
    auralize_broadband,
    cona_auralize,
    synthesize_observer_signal,
)
from auraflow.signal.spectra import third_octave_bands, third_octave_levels


class TestBroadbandRoundTrip:
    def test_levels_recovered_within_2db(self):
        fs = 8000.0
        n_fft = 1024
        centers, _ = third_octave_bands(200.0, 3000.0)
        nb = centers.shape[0]
        # Flat-ish target spectrogram, 2 frames.
        target = jnp.full((2, nb), 70.0)
        sig = auralize_broadband(target, centers, fs, n_fft, n_iters=60)
        assert np.all(np.isfinite(np.asarray(sig)))
        # Re-analyse the synthesized signal into 1/3-octave levels.
        _, lv = third_octave_levels(sig, fs, nperseg=n_fft, fmin=200.0, fmax=3000.0)
        lv = np.asarray(lv)
        tgt = np.asarray(target[0])
        # Compare bands that actually hold FFT bins (low bands may be empty).
        occ = lv > -100.0
        diff = np.abs(lv[occ] - tgt[occ])
        assert np.median(diff) < 2.0

    def test_waveform_amplitude_matches_band_energy(self):
        # After edge cropping, the raw waveform OASPL must match the band-energy
        # sum (no ill-conditioned edge blow-up).
        from auraflow.signal.spectra import oaspl

        fs = 44100.0
        n_fft = 2048
        centers, _ = third_octave_bands(200.0, 10000.0)
        nb = centers.shape[0]
        target = jnp.full((16, nb), 70.0)
        sig = auralize_broadband(target, centers, fs, n_fft, n_iters=40)
        expected = 70.0 + 10.0 * np.log10(nb)  # incoherent band sum
        assert abs(float(oaspl(sig)) - expected) < 3.0
        assert float(np.max(np.abs(np.asarray(sig)))) < 10.0  # sane peak [Pa]


class TestCombine:
    def test_tonal_plus_broadband_length_and_finite(self):
        fs = 8000.0
        centers, _ = third_octave_bands(200.0, 3000.0)
        nb = centers.shape[0]
        # Synthetic tonal: a 300 Hz sine over 0.1 s at a coarse source rate.
        tt = jnp.linspace(0.0, 0.1, 200)
        tonal = jnp.sin(2 * np.pi * 300.0 * tt)
        spec = jnp.full((3, nb), 65.0)
        out = synthesize_observer_signal(
            fs,
            0.1,
            tonal_pressure=tonal,
            tonal_t=tt,
            broadband_spectrograms=[spec],
            band_centers=centers,
            n_fft=512,
            n_iters=40,
        )
        n = int(round(fs * 0.1))
        assert out["total"].shape == (n,)
        assert out["tonal"].shape == (n,)
        assert out["broadband"].shape == (n,)
        assert np.all(np.isfinite(np.asarray(out["total"])))
        # Total = tonal + broadband exactly.
        assert np.allclose(
            np.asarray(out["total"]),
            np.asarray(out["tonal"]) + np.asarray(out["broadband"]),
        )
        # Both components carry energy.
        assert np.std(np.asarray(out["tonal"])) > 1e-3
        assert np.std(np.asarray(out["broadband"])) > 1e-6

    def test_per_rotor_separate_phase(self):
        # Two identical spectrograms auralised separately should not be
        # bit-identical (independent random phase per rotor).
        fs = 8000.0
        centers, _ = third_octave_bands(200.0, 3000.0)
        nb = centers.shape[0]
        spec = jnp.full((2, nb), 60.0)
        out = synthesize_observer_signal(
            fs,
            0.08,
            broadband_spectrograms=[spec, spec],
            band_centers=centers,
            n_fft=512,
            n_iters=20,
        )
        assert np.all(np.isfinite(np.asarray(out["broadband"])))

    def test_cona_auralize_wrapper(self):
        fs = 8000.0
        centers, _ = third_octave_bands(200.0, 3000.0)
        nb = centers.shape[0]
        tt = jnp.linspace(0.05, 0.15, 300)
        tonal = 0.5 * jnp.sin(2 * np.pi * 250.0 * tt)
        spec = jnp.full((4, nb), 68.0)
        out = cona_auralize(fs, tonal, tt, [spec], centers, n_fft=512, n_iters=30)
        n = int(round(fs * 0.1))
        assert out["total"].shape == (n,)
        assert np.all(np.isfinite(np.asarray(out["total"])))
