"""Spectral analysis of acoustic pressure signals.

Pure-JAX (float64-safe, differentiable) implementations of the standard
acoustic post-processing chain: SPL/OASPL, Welch PSD, narrowband spectra,
harmonic extraction, IEC 61672-1 A-weighting and IEC 61260-1 base-10
third-octave band levels.

Conventions
-----------
- Units are SI: pressures in Pa, frequencies in Hz, sample rates in Hz.
- Levels are dB re ``p_ref`` (default 20 uPa, the standard reference in air).
- Time signals are 1-D ``[T]`` unless stated otherwise; PSDs are one-sided
  power spectral densities in Pa^2/Hz following ``scipy.signal.welch``
  conventions (periodic window, per-segment constant detrend, density
  scaling).
- Structural parameters (``fs``, ``nperseg``, window name, band limits) are
  static Python values; signals and levels are traced JAX arrays.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

__all__ = [
    "P_REF",
    "a_weighted_oaspl",
    "a_weighting",
    "harmonic_levels",
    "narrowband_spectrum",
    "oaspl",
    "psd_to_db",
    "spl",
    "third_octave_bands",
    "third_octave_levels",
    "third_octave_spectrogram",
    "welch_psd",
]

P_REF: float = 20e-6
"""Standard reference pressure in air [Pa]."""

_TINY: float = 1e-300
"""Floor for mean-square values before taking logs (avoids -inf/NaN)."""


def _get_window(window: str, n: int) -> Array:
    """Return a length-``n`` periodic (DFT-even) window, shape ``[n]``.

    Supported: ``"hann"`` (periodic, matching ``scipy.signal.get_window``)
    and ``"boxcar"``/``"rectangular"``.
    """
    if window == "hann":
        return 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * jnp.arange(n) / n)
    if window in ("boxcar", "rectangular"):
        return jnp.ones(n)
    raise ValueError(f"unsupported window: {window!r} (expected 'hann' or 'boxcar')")


def _frame(x: Array, frame_len: int, hop: int) -> Array:
    """Slice a 1-D signal ``[T]`` into overlapping frames ``[n_frames, frame_len]``.

    Frame ``m`` is ``x[m*hop : m*hop + frame_len]``; trailing samples that do
    not fill a complete frame are dropped.
    """
    n = x.shape[-1]
    if frame_len <= 0 or hop <= 0:
        raise ValueError(f"frame_len and hop must be positive, got {frame_len=}, {hop=}")
    if frame_len > n:
        raise ValueError(f"signal too short: {n} samples < frame_len={frame_len}")
    n_frames = 1 + (n - frame_len) // hop
    idx = jnp.arange(n_frames)[:, None] * hop + jnp.arange(frame_len)[None, :]
    return x[idx]


def spl(p_rms: ArrayLike, p_ref: float = P_REF) -> Array:
    """Sound pressure level of an RMS pressure.

    Args:
        p_rms: RMS acoustic pressure [Pa], any shape.
        p_ref: reference pressure [Pa].

    Returns:
        ``20 log10(p_rms / p_ref)`` in dB re ``p_ref``, same shape as ``p_rms``.
    """
    return 20.0 * jnp.log10(jnp.asarray(p_rms) / p_ref)


def oaspl(signal: Array, axis: int = -1, p_ref: float = P_REF) -> Array:
    """Overall sound pressure level of a pressure time series.

    The mean (DC offset) is removed along ``axis`` before forming the RMS, so
    only the fluctuating pressure contributes.

    Args:
        signal: acoustic pressure [Pa]; time along ``axis`` (e.g. ``[O, T]``).
        axis: time axis.
        p_ref: reference pressure [Pa].

    Returns:
        OASPL in dB re ``p_ref``; shape of ``signal`` with ``axis`` removed.
    """
    p = signal - jnp.mean(signal, axis=axis, keepdims=True)
    return spl(jnp.sqrt(jnp.mean(p * p, axis=axis)), p_ref)


def _frame_psds(x: Array, fs: float, nperseg: int, hop: int, window: str) -> tuple[Array, Array]:
    """Per-frame one-sided modified periodograms (Welch segments, no averaging).

    Each frame is constant-detrended, windowed and scaled to a one-sided PSD
    in Pa^2/Hz (``scipy.signal.welch`` density scaling).

    Returns:
        ``(freqs [nperseg//2+1], psds [n_frames, nperseg//2+1])``.
    """
    w = _get_window(window, nperseg)
    frames = _frame(x, nperseg, hop)
    frames = frames - jnp.mean(frames, axis=-1, keepdims=True)
    spec = jnp.fft.rfft(frames * w, axis=-1)
    psds = (spec.real**2 + spec.imag**2) / (fs * jnp.sum(w * w))
    if nperseg % 2 == 0:
        psds = psds.at[..., 1:-1].multiply(2.0)
    else:
        psds = psds.at[..., 1:].multiply(2.0)
    freqs = jnp.fft.rfftfreq(nperseg, 1.0 / fs)
    return freqs, psds


def welch_psd(
    x: Array,
    fs: float,
    nperseg: int,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[Array, Array]:
    """Welch one-sided power spectral density estimate.

    Matches ``scipy.signal.welch(x, fs, window, nperseg, noverlap)`` with its
    defaults: periodic window, per-segment constant detrend, one-sided density
    scaling (Pa^2/Hz), mean average over segments.

    Args:
        x: pressure signal [Pa], shape ``[T]``.
        fs: sample rate [Hz].
        nperseg: segment length in samples (static).
        noverlap: overlap in samples; default ``nperseg // 2`` (50%).
        window: ``"hann"`` (default) or ``"boxcar"``.

    Returns:
        ``(freqs, psd)``: frequencies [Hz] and PSD [Pa^2/Hz], both
        ``[nperseg//2 + 1]``.
    """
    if noverlap is None:
        noverlap = nperseg // 2
    if not 0 <= noverlap < nperseg:
        raise ValueError(f"need 0 <= noverlap < nperseg, got {noverlap=}, {nperseg=}")
    freqs, psds = _frame_psds(x, fs, nperseg, nperseg - noverlap, window)
    return freqs, jnp.mean(psds, axis=0)


def psd_to_db(psd: ArrayLike, p_ref: float = P_REF) -> Array:
    """Convert a PSD [Pa^2/Hz] to spectral level [dB/Hz re ``p_ref^2``].

    Args:
        psd: one-sided PSD [Pa^2/Hz], any shape.
        p_ref: reference pressure [Pa].

    Returns:
        ``10 log10(psd / p_ref^2)``, same shape as ``psd``.
    """
    return 10.0 * jnp.log10(jnp.asarray(psd) / p_ref**2)


def narrowband_spectrum(x: Array, fs: float, window: str = "hann") -> tuple[Array, Array]:
    """One-sided amplitude spectrum from a single windowed rFFT.

    The signal mean is removed, the periodic window applied, and magnitudes
    scaled by the window coherent gain and doubled (except DC and Nyquist), so
    a sine of amplitude ``A`` at an exact bin frequency peaks at ``A``.

    Args:
        x: pressure signal [Pa], shape ``[T]``.
        fs: sample rate [Hz].
        window: ``"hann"`` (default) or ``"boxcar"``.

    Returns:
        ``(freqs, amp)``: frequencies [Hz] and amplitudes [Pa], both
        ``[T//2 + 1]``; frequency resolution is ``fs / T``.
    """
    n = x.shape[-1]
    w = _get_window(window, n)
    xd = x - jnp.mean(x, axis=-1, keepdims=True)
    amp = jnp.abs(jnp.fft.rfft(xd * w)) / jnp.sum(w)
    if n % 2 == 0:
        amp = amp.at[..., 1:-1].multiply(2.0)
    else:
        amp = amp.at[..., 1:].multiply(2.0)
    freqs = jnp.fft.rfftfreq(n, 1.0 / fs)
    return freqs, amp


def harmonic_levels(
    x: Array, fs: float, f0: ArrayLike, n_harmonics: int, p_ref: float = P_REF
) -> tuple[Array, Array]:
    """SPL of the harmonics of a fundamental, from a single periodogram.

    For each harmonic ``k*f0`` (``k = 1..n_harmonics``) the largest bin of the
    Hann-windowed amplitude spectrum within +-``f0/2`` of the nominal harmonic
    frequency is located, then refined by parabolic interpolation of the
    log-magnitude over the peak bin and its two neighbours, yielding
    sub-bin frequency and scalloping-corrected amplitude estimates.

    The signal must contain at least one full period per FFT bin around each
    harmonic, i.e. ``len(x) >= fs / f0``, so the +-``f0/2`` search window
    holds at least one bin. Peak-bin selection is piecewise constant (zero
    gradient); the refined outputs carry gradients from the spectrum values.

    Args:
        x: pressure signal [Pa], shape ``[T]``.
        fs: sample rate [Hz].
        f0: fundamental frequency [Hz] (e.g. blade-passing frequency).
        n_harmonics: number of harmonics to extract (static).
        p_ref: reference pressure [Pa].

    Returns:
        ``(freqs, levels)``, both ``[n_harmonics]``: refined harmonic
        frequencies [Hz] and SPLs [dB re ``p_ref``] (level of each harmonic's
        sine component, i.e. ``spl(amplitude / sqrt(2))``).
    """
    freqs, amp = narrowband_spectrum(x, fs, window="hann")
    df = fs / x.shape[-1]
    amp_db = 20.0 * jnp.log10(amp + _TINY)
    f_target = jnp.arange(1, n_harmonics + 1) * jnp.asarray(f0)
    dist = jnp.abs(freqs[None, :] - f_target[:, None])
    masked = jnp.where(dist <= 0.5 * jnp.asarray(f0), amp_db[None, :], -jnp.inf)
    m = jnp.clip(jnp.argmax(masked, axis=-1), 1, freqs.shape[0] - 2)
    a_lo, a_pk, a_hi = amp_db[m - 1], amp_db[m], amp_db[m + 1]
    denom = a_lo - 2.0 * a_pk + a_hi
    ok = jnp.abs(denom) > 1e-12
    delta = jnp.where(ok, 0.5 * (a_lo - a_hi) / jnp.where(ok, denom, 1.0), 0.0)
    delta = jnp.clip(delta, -0.5, 0.5)
    peak_db = a_pk - 0.25 * (a_lo - a_hi) * delta
    f_peak = (m + delta) * df
    levels = peak_db - 20.0 * math.log10(math.sqrt(2.0) * p_ref)
    return f_peak, levels


# IEC 61672-1 analog A-weighting pole frequencies [Hz].
_A_F1 = 20.598997
_A_F2 = 107.65265
_A_F3 = 737.86223
_A_F4 = 12194.217

# Normalization so that A(1000 Hz) = 0 dB exactly (approx +2.000 dB).
_A_OFFSET_DB = -20.0 * math.log10(
    (_A_F4**2 * 1e12)
    / ((1e6 + _A_F1**2) * math.sqrt((1e6 + _A_F2**2) * (1e6 + _A_F3**2)) * (1e6 + _A_F4**2))
)


def _a_response(f_sq: Array) -> Array:
    """Un-normalized analog A-weight magnitude response R_A(f), from f^2 [Hz^2]."""
    return (_A_F4**2 * f_sq**2) / (
        (f_sq + _A_F1**2) * jnp.sqrt((f_sq + _A_F2**2) * (f_sq + _A_F3**2)) * (f_sq + _A_F4**2)
    )


def a_weighting(freqs: ArrayLike) -> Array:
    """IEC 61672-1 analog A-weighting curve.

    Args:
        freqs: frequencies [Hz], any shape.

    Returns:
        A-weight in dB, same shape (0 dB at 1 kHz; ``-inf`` at 0 Hz).
    """
    f_sq = jnp.square(jnp.asarray(freqs))
    return 20.0 * jnp.log10(_a_response(f_sq)) + _A_OFFSET_DB


def a_weighted_oaspl(x: Array, fs: float, p_ref: float = P_REF) -> Array:
    """A-weighted overall sound pressure level (dBA) of a time signal.

    Computes a full-length boxcar periodogram (exact Parseval split of the
    signal variance), applies the analog A-weighting in power, and integrates.

    Args:
        x: pressure signal [Pa], shape ``[T]``.
        fs: sample rate [Hz].
        p_ref: reference pressure [Pa].

    Returns:
        Scalar A-weighted OASPL [dB(A) re ``p_ref``].
    """
    n = x.shape[-1]
    freqs, psd = welch_psd(x, fs, nperseg=n, window="boxcar")
    weight_pow = _a_response(jnp.square(freqs)) ** 2 * 10.0 ** (_A_OFFSET_DB / 10.0)
    msq = jnp.sum(psd * weight_pow) * (fs / n)
    return spl(jnp.sqrt(msq), p_ref)


def third_octave_bands(fmin: float = 20.0, fmax: float = 20000.0) -> tuple[Array, Array]:
    """IEC 61260-1 base-10 preferred one-third-octave band frequencies.

    Exact centers are ``1000 * 10**(n/10)`` for integer band index ``n``
    (``n = 0`` is exactly 1000 Hz); band edges are ``center * 10**(+-1/20)``,
    so adjacent bands share edges. A band is included when its *nominal*
    center lies in ``[fmin, fmax]``, i.e. ``n = round(10 log10(f/1000))``
    (the default range spans nominal 20 Hz .. 20 kHz, 31 bands).

    Args:
        fmin: lowest nominal center frequency to include [Hz] (static).
        fmax: highest nominal center frequency to include [Hz] (static).

    Returns:
        ``(centers [n_bands], edges [n_bands+1])`` in Hz; ``edges[i]`` and
        ``edges[i+1]`` bound band ``i``.
    """
    n_lo = round(10.0 * math.log10(fmin / 1000.0))
    n_hi = round(10.0 * math.log10(fmax / 1000.0))
    if n_hi < n_lo:
        raise ValueError(f"empty band range: {fmin=}, {fmax=}")
    centers = 1000.0 * 10.0 ** (jnp.arange(n_lo, n_hi + 1) / 10.0)
    edges = 1000.0 * 10.0 ** ((jnp.arange(n_lo, n_hi + 2) - 0.5) / 10.0)
    return centers, edges


def _band_msq_from_psd(psds: Array, freqs: Array, edges: Array) -> Array:
    """Integrate PSDs over frequency bands by rectangle rule on FFT bins.

    Args:
        psds: one-sided PSD(s) [Pa^2/Hz], shape ``[..., n_bins]`` on a uniform
            frequency grid.
        freqs: bin frequencies [Hz], shape ``[n_bins]``.
        edges: band edges [Hz], shape ``[n_bands+1]``.

    Returns:
        Band mean-square pressures [Pa^2], shape ``[..., n_bands]``. Bins are
        assigned to the band with ``edges[i] <= f < edges[i+1]``; bands
        containing no bins integrate to zero.
    """
    df = freqs[1] - freqs[0]
    in_band = (freqs[None, :] >= edges[:-1, None]) & (freqs[None, :] < edges[1:, None])
    return jnp.sum(in_band * psds[..., None, :], axis=-1) * df


def third_octave_levels(
    x: Array,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    fmin: float = 20.0,
    fmax: float = 20000.0,
    p_ref: float = P_REF,
) -> tuple[Array, Array]:
    """One-third-octave band SPLs of a time signal via Welch PSD integration.

    The Welch PSD is integrated over each band's edge frequencies (rectangle
    rule over FFT bins). Bands narrower than the frequency resolution
    ``fs / nperseg`` may contain no bins and clamp to a floor level; choose
    ``nperseg`` accordingly for low bands.

    Args:
        x: pressure signal [Pa], shape ``[T]``.
        fs: sample rate [Hz].
        nperseg: Welch segment length; default ``min(T, 4096)``.
        noverlap: Welch overlap; default 50%.
        window: ``"hann"`` (default) or ``"boxcar"``.
        fmin: lowest nominal band center [Hz].
        fmax: highest nominal band center [Hz] (clamped to ``fs/2``).
        p_ref: reference pressure [Pa].

    Returns:
        ``(centers [n_bands], levels [n_bands])``: band center frequencies
        [Hz] and band SPLs [dB re ``p_ref``].
    """
    if nperseg is None:
        nperseg = min(x.shape[-1], 4096)
    centers, edges = third_octave_bands(fmin, min(fmax, 0.5 * fs))
    freqs, psd = welch_psd(x, fs, nperseg, noverlap, window)
    msq = _band_msq_from_psd(psd, freqs, edges)
    levels = 10.0 * jnp.log10(jnp.maximum(msq, _TINY) / p_ref**2)
    return centers, levels


def third_octave_spectrogram(
    x: Array,
    fs: float,
    frame_len: int,
    hop: int,
    window: str = "hann",
    fmin: float = 20.0,
    fmax: float = 20000.0,
    p_ref: float = P_REF,
) -> tuple[Array, Array]:
    """One-third-octave band SPL spectrogram of a time signal.

    The signal is framed (frame ``m`` covers samples
    ``m*hop : m*hop + frame_len``; ``n_frames = 1 + (T - frame_len) // hop``),
    each frame's windowed one-sided periodogram is integrated over the band
    edges as in :func:`third_octave_levels`.

    Args:
        x: pressure signal [Pa], shape ``[T]``.
        fs: sample rate [Hz].
        frame_len: frame length in samples (static).
        hop: hop between frames in samples (static).
        window: ``"hann"`` (default) or ``"boxcar"``.
        fmin: lowest nominal band center [Hz].
        fmax: highest nominal band center [Hz] (clamped to ``fs/2``).
        p_ref: reference pressure [Pa].

    Returns:
        ``(centers [n_bands], levels [n_frames, n_bands])``: band center
        frequencies [Hz] and per-frame band SPLs [dB re ``p_ref``].
    """
    centers, edges = third_octave_bands(fmin, min(fmax, 0.5 * fs))
    freqs, psds = _frame_psds(x, fs, frame_len, hop, window)
    msq = _band_msq_from_psd(psds, freqs, edges)
    levels = 10.0 * jnp.log10(jnp.maximum(msq, _TINY) / p_ref**2)
    return centers, levels
