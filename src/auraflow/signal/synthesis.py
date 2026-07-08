"""Time-signal synthesis from spectrograms: STFT/iSTFT and Griffin-Lim.

Used by the CONA auralization pipeline: BPM broadband noise is produced as a
one-third-octave band spectrogram, spread onto an STFT magnitude grid
(:func:`third_octave_to_stft_mag`) and phase-reconstructed with
:func:`griffin_lim` into a time signal.

Conventions match :mod:`auraflow.signal.spectra`: SI units (Pa, Hz, s),
periodic windows, spectrograms shaped ``[n_frames, n_bins]`` with
``n_bins = n_fft // 2 + 1``. ``n_fft``, ``hop``, ``n_iters`` and window names
are static; signals and levels are traced JAX arrays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from auraflow.signal.spectra import P_REF, _frame, _get_window

__all__ = [
    "griffin_lim",
    "istft",
    "stft",
    "third_octave_to_stft_mag",
]

_NOLA_EPS: float = 1e-11
"""Window-envelope threshold below which iSTFT output samples are zeroed."""


def stft(x: Array, n_fft: int, hop: int, window: str = "hann") -> Array:
    """Short-time Fourier transform (one-sided, no padding or centering).

    Frame ``m`` is ``x[m*hop : m*hop + n_fft]`` multiplied by the periodic
    window and transformed with an rFFT; ``n_frames = 1 + (T - n_fft) // hop``
    and trailing samples that do not fill a frame are dropped.

    Args:
        x: time signal [Pa], shape ``[T]`` with ``T >= n_fft``.
        n_fft: frame/FFT length in samples (static).
        hop: hop between frames in samples (static).
        window: ``"hann"`` (default) or ``"boxcar"``.

    Returns:
        Complex spectrogram ``[n_frames, n_fft//2 + 1]`` (unnormalized rFFT of
        the windowed frames).
    """
    w = _get_window(window, n_fft)
    return jnp.fft.rfft(_frame(x, n_fft, hop) * w, axis=-1)


def istft(z: Array, n_fft: int, hop: int, window: str = "hann") -> Array:
    """Inverse STFT by windowed overlap-add with least-squares normalization.

    Reconstructs ``y[n] = sum_m w[n - m*hop] * irfft(z[m])[n - m*hop] /
    sum_m w[n - m*hop]^2``, the least-squares signal estimate of Griffin &
    Lim. This inverts :func:`stft` exactly (up to floating point) at every
    sample where the squared-window overlap-add envelope is nonzero, for any
    window/hop satisfying the NOLA condition — in particular the periodic
    Hann window with ``hop = n_fft/2`` (50% overlap, envelope 1.5 in the
    interior) or ``hop = n_fft/4`` (75% overlap, envelope 0.75·n_fft/hop·...
    constant interior). The periodic Hann is zero at its first sample, so
    sample 0 of the output (and any sample with envelope below ``1e-11``) is
    set to zero.

    Args:
        z: complex spectrogram ``[n_frames, n_fft//2 + 1]`` (as from
            :func:`stft` with the same ``n_fft``, ``hop``, ``window``).
        n_fft: frame/FFT length in samples (static).
        hop: hop between frames in samples (static).
        window: ``"hann"`` (default) or ``"boxcar"``.

    Returns:
        Real time signal [Pa], shape ``[(n_frames - 1) * hop + n_fft]``.
    """
    n_frames = z.shape[0]
    length = (n_frames - 1) * hop + n_fft
    w = _get_window(window, n_fft)
    frames = jnp.fft.irfft(z, n=n_fft, axis=-1) * w
    idx = jnp.arange(n_frames)[:, None] * hop + jnp.arange(n_fft)[None, :]
    num = jnp.zeros(length, dtype=frames.dtype).at[idx].add(frames)
    den = jnp.zeros(length, dtype=w.dtype).at[idx].add(jnp.broadcast_to(w * w, idx.shape))
    ok = den > _NOLA_EPS
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), 0.0)


def griffin_lim(
    mag: Array,
    n_fft: int,
    hop: int,
    n_iters: int = 60,
    key: Array | None = None,
    momentum: float = 0.99,
    window: str = "hann",
) -> Array:
    """Griffin-Lim phase reconstruction from an STFT magnitude spectrogram.

    Classic Griffin-Lim alternating projection accelerated with Perraudin's
    momentum (FGLA): starting from uniform random phases, each iteration
    projects onto the set of consistent spectrograms via
    ``stft(istft(mag * angles))`` and extrapolates with momentum
    ``alpha = momentum / (1 + momentum)``; ``momentum = 0`` recovers the
    classic algorithm. Iterations run in a ``lax.scan`` (``n_iters`` static),
    and the result is deterministic for a fixed ``key``.

    Args:
        mag: nonnegative STFT magnitude ``[n_frames, n_bins]`` with
            ``n_bins = n_fft//2 + 1``, in the scaling of :func:`stft`.
        n_fft: frame/FFT length in samples (static).
        hop: hop between frames in samples (static).
        n_iters: number of projection iterations (static).
        key: PRNG key for the initial phases; defaults to
            ``jax.random.PRNGKey(0)``.
        momentum: FGLA momentum in ``[0, 1)``.
        window: ``"hann"`` (default) or ``"boxcar"``.

    Returns:
        Real time signal [Pa], shape ``[(n_frames - 1) * hop + n_fft]``, whose
        STFT magnitude approximates ``mag``.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    phase = jax.random.uniform(key, mag.shape, dtype=mag.dtype, maxval=2.0 * jnp.pi)
    angles = jnp.exp(1j * phase)
    alpha = momentum / (1.0 + momentum)

    def body(carry: tuple[Array, Array], _: None) -> tuple[tuple[Array, Array], None]:
        angles, prev = carry
        rebuilt = stft(istft(mag * angles, n_fft, hop, window), n_fft, hop, window)
        update = rebuilt - alpha * prev
        angles = update / jnp.maximum(jnp.abs(update), 1e-16)
        return (angles, rebuilt), None

    (angles, _), _ = jax.lax.scan(body, (angles, jnp.zeros_like(angles)), None, length=n_iters)
    return istft(mag * angles, n_fft, hop, window)


def third_octave_to_stft_mag(
    band_levels_db: Array,
    band_centers: Array,
    n_fft: int,
    fs: float,
    p_ref: float = P_REF,
    window: str = "hann",
) -> Array:
    """Spread one-third-octave band SPLs onto an STFT magnitude grid.

    Each band's mean-square pressure ``p_ref^2 * 10**(L/10)`` is distributed
    uniformly in power over the rFFT bins whose frequency lies within the band
    edges ``[fc * 10**(-1/20), fc * 10**(1/20))``. Bins outside every band are
    zero; bands narrower than the bin spacing ``fs / n_fft`` contain no bins
    and are silently dropped — choose ``n_fft`` large enough for the lowest
    band. Magnitudes are scaled consistently with :func:`stft` (Welch density
    conventions), so ``third_octave_levels(griffin_lim(result, ...), ...)``
    recovers approximately ``band_levels_db``.

    Args:
        band_levels_db: band SPLs [dB re ``p_ref``], shape ``[..., n_bands]``
            (typically ``[n_frames, n_bands]``, one row per STFT frame).
        band_centers: exact band center frequencies [Hz], ``[n_bands]`` (as
            from :func:`auraflow.signal.spectra.third_octave_bands`).
        n_fft: target frame/FFT length in samples (static).
        fs: sample rate [Hz].
        p_ref: reference pressure [Pa].
        window: analysis window the magnitudes should be consistent with.

    Returns:
        Magnitude spectrogram ``[..., n_fft//2 + 1]`` [Pa, :func:`stft`
        scaling].
    """
    freqs = jnp.fft.rfftfreq(n_fft, 1.0 / fs)
    lo = band_centers * 10.0 ** (-0.05)
    hi = band_centers * 10.0**0.05
    in_band = (freqs[None, :] >= lo[:, None]) & (freqs[None, :] < hi[:, None])
    bins_per_band = jnp.sum(in_band, axis=-1)
    band_msq = p_ref**2 * 10.0 ** (band_levels_db / 10.0)
    per_bin_msq = band_msq / jnp.maximum(bins_per_band, 1)
    bin_msq = jnp.einsum("...b,bk->...k", per_bin_msq, in_band.astype(freqs.dtype))
    w = _get_window(window, n_fft)
    return jnp.sqrt(bin_msq * (n_fft * jnp.sum(w * w) / 2.0))
