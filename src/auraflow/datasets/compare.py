r"""CONA vs CFD+FW-H comparison utilities for the JASA study.

The JASA data set is generated with the fast CONA backend
(:mod:`auraflow.datasets.jasa`); the point of the *comparison* is to regenerate
the same flyover condition with the full-CFD + permeable-surface FW-H backend
(:mod:`auraflow.cfd`) and quantify how far the cheap surrogate is from the
reference. This module holds the metric plumbing that both
``scripts/cona_vs_cfd.py`` and the tests use.

Everything here is backend-agnostic once the two pressure signals are in hand:

- :func:`signal_metrics` reduces one pressure time series to OASPL, A-weighted
  OASPL and a one-third-octave band spectrum (via :mod:`auraflow.signal`).
- :func:`compare_signals` diffs two signals (which may live at *different*
  sample rates, as CONA runs at 44.1 kHz while a coarse CFD run resolves far
  less) on a common band set and reports level/OASPL/spectrum differences.
- :func:`cfd_observer_signals` turns a :class:`~auraflow.cfd.run.SurfaceHistory`
  (real or **synthetic** -- the tests feed a hand-built one so no JAX-Fluids run
  is needed) into per-observer pressure signals via
  :func:`~auraflow.cfd.run.propagate_to_observers`.
- :func:`compare_cona_vs_cfd` is the top-level entry: a CONA audio block plus a
  surface history in, a structured comparison dict out.

All pressures are Pa, levels dB re 20 uPa, frequencies Hz. NumPy on the way out.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from auraflow.cfd.run import SurfaceHistory, propagate_to_observers
from auraflow.cfd.sphere import PermeableSphere
from auraflow.core.medium import Medium
from auraflow.signal.spectra import (
    a_weighted_oaspl,
    oaspl,
    third_octave_levels,
)

__all__ = [
    "cfd_observer_signals",
    "compare_cona_vs_cfd",
    "compare_signals",
    "signal_metrics",
]


def signal_metrics(
    p: ArrayLike,
    fs: float,
    *,
    fmin: float = 20.0,
    fmax: float = 20000.0,
    nperseg: int | None = None,
) -> dict[str, Any]:
    """Reduce one pressure time series to summary acoustic metrics.

    Args:
        p: Acoustic pressure [Pa], shape ``[T]``.
        fs: Sample rate [Hz].
        fmin: Lowest nominal 1/3-octave band centre [Hz].
        fmax: Highest nominal band centre [Hz] (clamped to ``fs/2`` inside
            :func:`~auraflow.signal.spectra.third_octave_levels`).
        nperseg: Welch segment length; default ``min(T, 4096)``.

    Returns:
        Dict (NumPy scalars/arrays): ``"oaspl"`` [dB], ``"oaspl_a"`` [dB(A)],
        ``"band_centers"`` [Hz] ``[n_bands]`` and ``"band_levels"``
        [dB] ``[n_bands]``.
    """
    p = jnp.asarray(p, dtype=float)
    centers, levels = third_octave_levels(p, fs, nperseg=nperseg, fmin=fmin, fmax=fmax)
    return {
        "oaspl": float(oaspl(p)),
        "oaspl_a": float(a_weighted_oaspl(p, fs)),
        "band_centers": np.asarray(centers),
        "band_levels": np.asarray(levels),
    }


def compare_signals(
    p_ref: ArrayLike,
    fs_ref: float,
    p_test: ArrayLike,
    fs_test: float,
    *,
    fmin: float = 20.0,
    fmax: float = 20000.0,
    nperseg: int | None = None,
) -> dict[str, Any]:
    """Compare a reference and a test pressure signal on a common band set.

    The two signals may have different sample rates (CONA at 44.1 kHz vs a
    coarse CFD run); the shared 1/3-octave band range is clamped to the lower
    Nyquist so both spectra land on the same centres.

    Args:
        p_ref: Reference pressure [Pa], shape ``[T_ref]`` (here: CONA).
        fs_ref: Reference sample rate [Hz].
        p_test: Test pressure [Pa], shape ``[T_test]`` (here: CFD+FW-H).
        fs_test: Test sample rate [Hz].
        fmin, fmax: Requested band range [Hz]; ``fmax`` is additionally clamped
            to ``0.5 * min(fs_ref, fs_test)``.
        nperseg: Welch segment length for both (default per signal length).

    Returns:
        Dict with ``"ref"`` and ``"test"`` sub-dicts (:func:`signal_metrics`
        restricted to the common bands), plus differences ``test - ref``:
        ``"band_centers"`` [Hz], ``"band_level_diff"`` [dB] ``[n_bands]``,
        ``"oaspl_diff"`` [dB], ``"oaspl_a_diff"`` [dB], and
        ``"band_level_rmse"`` [dB] (RMS band-level difference).
    """
    fmax_common = min(float(fmax), 0.5 * min(float(fs_ref), float(fs_test)))
    ref = signal_metrics(p_ref, fs_ref, fmin=fmin, fmax=fmax_common, nperseg=nperseg)
    test = signal_metrics(p_test, fs_test, fmin=fmin, fmax=fmax_common, nperseg=nperseg)
    band_diff = test["band_levels"] - ref["band_levels"]
    # Only score bands that actually carry energy in both signals (avoid the
    # -inf floor of empty low bands dominating the RMSE).
    floor = -120.0
    occupied = (ref["band_levels"] > floor) & (test["band_levels"] > floor)
    scored = band_diff[occupied] if np.any(occupied) else band_diff
    return {
        "ref": ref,
        "test": test,
        "band_centers": ref["band_centers"],
        "band_level_diff": band_diff,
        "band_level_rmse": float(np.sqrt(np.mean(np.square(scored)))),
        "oaspl_diff": test["oaspl"] - ref["oaspl"],
        "oaspl_a_diff": test["oaspl_a"] - ref["oaspl_a"],
    }


def cfd_observer_signals(
    surface_history: SurfaceHistory,
    sphere: PermeableSphere,
    observers: ArrayLike,
    medium: Medium | None = None,
    *,
    n_obs: int | None = None,
) -> tuple[Array, Array, Array]:
    """Propagate a (possibly synthetic) surface history to observer signals.

    Thin wrapper over :func:`~auraflow.cfd.run.propagate_to_observers` that also
    returns the per-observer effective sample rate, so the result drops straight
    into :func:`compare_signals`. No JAX-Fluids is involved: the surface history
    is an ordinary PyTree and may be hand-built (the tests do exactly this).

    Args:
        surface_history: A :class:`~auraflow.cfd.run.SurfaceHistory` on ``sphere``.
        sphere: The static permeable sphere the history was sampled on.
        observers: Observer positions [m], shape ``[O, 3]``.
        medium: Ambient medium (default sea-level ISA).
        n_obs: Observer-time samples per observer (default: ``T`` of the history).

    Returns:
        ``(p, t_obs, fs)``: pressure [Pa] ``[O, T_obs]``, per-observer time grids
        [s] ``[O, T_obs]`` (uniform within each observer), and per-observer
        effective sample rates [Hz] ``[O]``.
    """
    medium = Medium() if medium is None else medium
    observers = jnp.asarray(observers, dtype=float)
    p, t_obs = propagate_to_observers(surface_history, sphere, observers, medium, n_obs=n_obs)
    # Grid is a per-observer linspace, so fs = (T_obs - 1) / (t_hi - t_lo).
    span = t_obs[:, -1] - t_obs[:, 0]
    fs = (t_obs.shape[1] - 1) / jnp.where(span > 0, span, 1.0)
    return p, t_obs, fs


def compare_cona_vs_cfd(
    cona_audio: ArrayLike,
    cona_fs: float,
    surface_history: SurfaceHistory,
    sphere: PermeableSphere,
    observers: ArrayLike,
    medium: Medium | None = None,
    *,
    fmin: float = 20.0,
    fmax: float = 20000.0,
    n_obs: int | None = None,
) -> dict[str, Any]:
    """Compare a CONA audio block against CFD+FW-H at the same observers.

    Args:
        cona_audio: CONA pressures [Pa], shape ``[O, n]`` (from
            :func:`auraflow.datasets.jasa.generate_flyover`; one row per observer).
        cona_fs: CONA sample rate [Hz] (44.1 kHz in the paper).
        surface_history: The CFD :class:`~auraflow.cfd.run.SurfaceHistory`.
        sphere: The permeable sphere the history lives on.
        observers: Observer positions [m], shape ``[O, 3]`` (same order/count as
            ``cona_audio`` rows).
        medium: Ambient medium (default sea-level ISA).
        fmin, fmax: Band range for the comparison [Hz].
        n_obs: CFD observer-time samples (default: history length ``T``).

    Returns:
        Dict with ``"observers"`` (list of per-observer :func:`compare_signals`
        dicts), and aggregate ``"oaspl_diff_mean"``, ``"oaspl_diff_rms"``,
        ``"band_level_rmse_mean"`` [dB] over the observers.
    """
    cona_audio = jnp.asarray(cona_audio, dtype=float)
    observers = jnp.asarray(observers, dtype=float)
    if cona_audio.shape[0] != observers.shape[0]:
        raise ValueError(
            f"observer count mismatch: cona_audio has {cona_audio.shape[0]} rows, "
            f"observers has {observers.shape[0]}"
        )
    p_cfd, _, fs_cfd = cfd_observer_signals(surface_history, sphere, observers, medium, n_obs=n_obs)
    p_cfd = np.asarray(p_cfd)
    fs_cfd = np.asarray(fs_cfd)
    cona_audio_np = np.asarray(cona_audio)

    per_obs: list[dict[str, Any]] = []
    for o in range(observers.shape[0]):
        per_obs.append(
            compare_signals(
                cona_audio_np[o],
                float(cona_fs),
                p_cfd[o],
                float(fs_cfd[o]),
                fmin=fmin,
                fmax=fmax,
            )
        )
    oaspl_diffs = np.array([c["oaspl_diff"] for c in per_obs])
    band_rmses = np.array([c["band_level_rmse"] for c in per_obs])
    return {
        "observers": per_obs,
        "oaspl_diff_mean": float(np.mean(oaspl_diffs)),
        "oaspl_diff_rms": float(np.sqrt(np.mean(np.square(oaspl_diffs)))),
        "band_level_rmse_mean": float(np.mean(band_rmses)),
    }
