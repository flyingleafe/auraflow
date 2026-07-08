"""CONA-vs-CFD comparison plumbing (no JAX-Fluids: a synthetic SurfaceHistory).

The comparison utilities are backend-agnostic once two pressure signals are in
hand, so a hand-built breathing-sphere :class:`SurfaceHistory` stands in for a
real CFD run -- exactly what scripts/cona_vs_cfd.py --dry does.
"""

import jax.numpy as jnp
import numpy as np

from auraflow.cfd.run import SurfaceHistory
from auraflow.cfd.sphere import PermeableSphere
from auraflow.core.medium import Medium
from auraflow.datasets.compare import (
    cfd_observer_signals,
    compare_cona_vs_cfd,
    compare_signals,
    signal_metrics,
)


def _breathing_sphere(sphere: PermeableSphere, medium: Medium, n_t: int = 64) -> SurfaceHistory:
    """A smooth radial monopole surface history (finite, non-trivial far field)."""
    s = int(sphere.points.shape[0])
    tau = jnp.linspace(0.0, 0.02, n_t)
    osc = jnp.sin(2.0 * jnp.pi * 300.0 * tau)  # [T]
    p = float(medium.p0) + 3.0 * osc[None, :] * jnp.ones((s, 1))
    rho = float(medium.rho0) + (p - float(medium.p0)) / float(medium.c0) ** 2
    u = 0.02 * sphere.normals[:, None, :] * osc[None, :, None]
    return SurfaceHistory(tau=tau, rho=rho, u=u, p=p)


def test_signal_metrics_structure_and_finiteness():
    rng = np.random.default_rng(0)
    p = rng.standard_normal(2000)
    m = signal_metrics(p, 8000.0, fmin=50.0, fmax=3000.0)
    assert set(m) == {"oaspl", "oaspl_a", "band_centers", "band_levels"}
    assert np.isfinite(m["oaspl"]) and np.isfinite(m["oaspl_a"])
    assert m["band_centers"].shape == m["band_levels"].shape


def test_compare_signals_across_sample_rates():
    rng = np.random.default_rng(1)
    p_ref = rng.standard_normal(4000)
    p_test = rng.standard_normal(1000)
    cmp = compare_signals(p_ref, 8000.0, p_test, 2000.0, fmin=50.0, fmax=3000.0)
    # fmax clamped to 0.5 * min(fs) = 1000 Hz, so both spectra share centres.
    assert cmp["band_centers"].max() <= 1000.0
    for key in ("band_level_diff", "band_level_rmse", "oaspl_diff", "oaspl_a_diff"):
        assert key in cmp
    assert np.isfinite(cmp["band_level_rmse"])
    assert cmp["ref"]["band_levels"].shape == cmp["test"]["band_levels"].shape


def test_cfd_observer_signals_from_synthetic_history():
    medium = Medium()
    sphere = PermeableSphere.fibonacci(24, radius=0.5)
    surf = _breathing_sphere(sphere, medium)
    observers = jnp.asarray([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    p, t_obs, fs = cfd_observer_signals(surf, sphere, observers, medium)
    assert p.shape[0] == 2 and t_obs.shape[0] == 2
    assert np.all(np.isfinite(np.asarray(p)))
    assert np.all(np.asarray(fs) > 0.0)


def test_compare_cona_vs_cfd_end_to_end():
    medium = Medium()
    sphere = PermeableSphere.fibonacci(24, radius=0.5)
    surf = _breathing_sphere(sphere, medium)
    observers = jnp.asarray([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    rng = np.random.default_rng(2)
    cona_audio = rng.standard_normal((2, 4000))  # pretend CONA @ 44.1k-ish
    out = compare_cona_vs_cfd(
        cona_audio, 8000.0, surf, sphere, observers, medium, fmin=50.0, fmax=2000.0
    )
    assert len(out["observers"]) == 2
    for key in ("oaspl_diff_mean", "oaspl_diff_rms", "band_level_rmse_mean"):
        assert np.isfinite(out[key])


def test_compare_cona_vs_cfd_observer_count_mismatch_raises():
    medium = Medium()
    sphere = PermeableSphere.fibonacci(16, radius=0.5)
    surf = _breathing_sphere(sphere, medium)
    observers = jnp.asarray([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    cona_audio = np.zeros((1, 2000))  # only 1 row, 2 observers
    try:
        compare_cona_vs_cfd(cona_audio, 8000.0, surf, sphere, observers, medium)
    except ValueError:
        return
    raise AssertionError("expected ValueError on observer-count mismatch")
