"""Smoke + physics-sanity tests for the JASA flyover generation (tiny; CPU/float64).

Every case here is deliberately small (few kHz, 1-2 mics, short duration, coarse
grids) so the whole file runs in a couple of seconds and a few hundred MB -- see
docs/research/jasa-datagen-reference.md for the full-scale (44.1 kHz, 256-mic)
spec that belongs on a GPU.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np

from auraflow.datasets.jasa import (
    JASAScenario,
    generate_flyover,
    jasa_microphone_array,
    scenario_id,
)
from auraflow.datasets.nasa_1pax import BPF_HZ
from auraflow.signal.spectra import harmonic_levels, narrowband_spectrum

# Shared tiny generation knobs. low_memory clears XLA compile caches between
# pipeline stages -- without it a single flyover peaks >1.1 GB host RAM (the
# dev box has ~1.3 GB available) purely from accumulated compiled executables.
_TINY: dict[str, Any] = dict(
    n_stations=6,
    n_source_times=200,
    n_frames=8,
    n_fft=128,
    gl_iters=5,
    obs_chunk=2,
    low_memory=True,
)


def _mics(*pts):
    return jnp.asarray(pts, dtype=float)


def test_microphone_array_is_256_grid():
    mics = jasa_microphone_array()
    assert mics.shape == (256, 3)  # 32 x 8
    assert jnp.allclose(mics[:, 2], 0.0)  # ground level
    assert float(mics[:, 0].min()) == -150.0 and float(mics[:, 0].max()) == 160.0
    assert float(mics[:, 1].min()) == 0.0 and float(mics[:, 1].max()) == 70.0


def test_generate_flyover_smoke_shapes_and_metadata():
    sc = JASAScenario(
        speed=8.0,
        altitude=30.0,
        duration=0.25,
        fs=2000.0,
        seed=0,
        mics=_mics([-30.0, 0.0, 0.0], [40.0, 20.0, 0.0]),
    )
    res = generate_flyover(sc, **_TINY)
    n = int(round(sc.fs * sc.duration))
    for k in ("audio", "tonal", "broadband"):
        assert res[k].shape == (2, n)
        assert np.all(np.isfinite(res[k]))
    # tonal + broadband == total.
    assert np.allclose(res["audio"], res["tonal"] + res["broadband"], atol=1e-9)
    # Broadband actually carries energy (bands occupied).
    assert np.std(res["broadband"]) > 0.0
    assert res["band_centers"].size > 0
    # Metadata complete.
    meta = res["meta"]
    for key in (
        "speed",
        "altitude",
        "duration",
        "fs",
        "seed",
        "bpf_hz",
        "n_mics",
        "n_audio",
        "collective_rad",
        "c0",
        "rho0",
        "include_broadband",
    ):
        assert key in meta
    assert meta["n_mics"] == 2 and meta["n_audio"] == n
    assert abs(meta["bpf_hz"] - BPF_HZ) < 1e-6


def test_tonal_has_bpf_tone():
    """A hovering rotor's tonal signal shows a clear peak at the BPF."""
    sc = JASAScenario(
        speed=0.0, altitude=30.0, duration=0.5, fs=2000.0, seed=0, mics=_mics([20.0, 0.0, 0.0])
    )
    res = generate_flyover(sc, include_broadband=False, **_TINY)
    tonal = res["tonal"][0]
    freqs, amp = narrowband_spectrum(jnp.asarray(tonal), sc.fs)
    freqs = np.asarray(freqs)
    amp = np.asarray(amp)
    idx_bpf = int(np.argmin(np.abs(freqs - BPF_HZ)))
    # The BPF bin dominates the spectrum median by a wide margin.
    assert amp[idx_bpf] > 8.0 * np.median(amp)
    # The dominant line lies on the BPF harmonic comb (with 4 coherently summed
    # rotors and a mic on the symmetry plane, a BPF *harmonic* may exceed the
    # partially cancelled fundamental -- e.g. 2xBPF; that is physical).
    df = float(freqs[1] - freqs[0])
    f_max = float(freqs[int(np.argmax(amp))])
    n_harm = max(round(f_max / BPF_HZ), 1)
    assert abs(f_max - n_harm * BPF_HZ) <= df


def test_doppler_approach_bpf_above_recede():
    """Approaching-half BPF is higher than receding-half BPF (flyover Doppler)."""
    # Low altitude + higher speed so the source really passes over and the
    # line-of-sight radial velocity flips sign at mid-run (t_pass = 0.5 s).
    sc = JASAScenario(
        speed=30.0, altitude=15.0, duration=1.0, fs=4000.0, seed=0, mics=_mics([0.0, 0.0, 0.0])
    )
    res = generate_flyover(sc, include_broadband=False, **_TINY)
    tonal = np.asarray(res["tonal"][0])
    half = tonal.size // 2
    f_app, _ = harmonic_levels(jnp.asarray(tonal[:half]), sc.fs, BPF_HZ, 1)
    f_rec, _ = harmonic_levels(jnp.asarray(tonal[half:]), sc.fs, BPF_HZ, 1)
    f_app, f_rec = float(f_app[0]), float(f_rec[0])
    assert f_app > f_rec
    # Both stay near the BPF (sanity: we located the right line).
    assert abs(f_app - BPF_HZ) < 0.25 * BPF_HZ
    assert abs(f_rec - BPF_HZ) < 0.25 * BPF_HZ


def test_determinism_same_seed_identical_different_seed_differs():
    sc0 = JASAScenario(
        speed=8.0, altitude=30.0, duration=0.25, fs=2000.0, seed=0, mics=_mics([-20.0, 0.0, 0.0])
    )
    sc0b = JASAScenario(
        speed=8.0, altitude=30.0, duration=0.25, fs=2000.0, seed=0, mics=_mics([-20.0, 0.0, 0.0])
    )
    sc1 = JASAScenario(
        speed=8.0, altitude=30.0, duration=0.25, fs=2000.0, seed=1, mics=_mics([-20.0, 0.0, 0.0])
    )
    a0 = generate_flyover(sc0, **_TINY)["audio"]
    a0b = generate_flyover(sc0b, **_TINY)["audio"]
    a1 = generate_flyover(sc1, **_TINY)["audio"]
    assert np.array_equal(a0, a0b)  # bit-identical for the same seed
    assert not np.allclose(a0, a1)  # Griffin-Lim phases reseed with the seed


def test_different_seed_gives_different_gust_realization():
    """With a gust on, the tonal signal (airload-driven) differs between seeds."""
    kw: dict[str, Any] = dict(gust_w20="moderate")
    sc0 = JASAScenario(
        speed=10.0,
        altitude=30.0,
        duration=0.25,
        fs=2000.0,
        seed=0,
        mics=_mics([10.0, 0.0, 0.0]),
        **kw,
    )
    sc1 = JASAScenario(
        speed=10.0,
        altitude=30.0,
        duration=0.25,
        fs=2000.0,
        seed=1,
        mics=_mics([10.0, 0.0, 0.0]),
        **kw,
    )
    t0 = generate_flyover(sc0, include_broadband=False, **_TINY)["tonal"]
    t1 = generate_flyover(sc1, include_broadband=False, **_TINY)["tonal"]
    assert np.all(np.isfinite(t0)) and np.all(np.isfinite(t1))
    assert not np.allclose(t0, t1)


def test_scenario_id_is_stable_and_filesystem_safe():
    sc = JASAScenario(speed=8.0, altitude=30.0, seed=2)
    sid = scenario_id(sc)
    assert sid == scenario_id(JASAScenario(speed=8.0, altitude=30.0, seed=2))
    assert "." not in sid and "/" not in sid
