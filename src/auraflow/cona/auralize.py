r"""CONA auralization: 1/3-octave spectrogram + tonal pressure -> time signal.

The final stage of the CONA pipeline. Two synthesis paths are combined:

- **Broadband**: a BPM one-third-octave spectrogram (from
  :func:`auraflow.cona.broadband.rotor_broadband_spectrogram`) is spread onto an
  STFT magnitude grid with uniform per-band power
  (:func:`auraflow.signal.third_octave_to_stft_mag`, the NASA rule) and
  phase-reconstructed with Fast Griffin--Lim
  (:func:`auraflow.signal.griffin_lim`, momentum 0.99). Per-rotor spectrograms
  are auralised *separately* (independent random phase) to avoid spurious
  broadband inter-rotor coherence, then summed -- the Ko et al. (JASA 2023)
  practice.
- **Tonal**: the deterministic thickness+loading pressure time history from
  :func:`auraflow.cona.tonal.cona_tonal_noise` is resampled to the audio rate
  and added directly.

Everything is JAX float64 and differentiable. Sample rates are static; JASA
uses 44.1 kHz but the tests use small ``fs``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from auraflow.signal.spectra import P_REF
from auraflow.signal.synthesis import griffin_lim, third_octave_to_stft_mag

__all__ = [
    "auralize_broadband",
    "cona_auralize",
    "resample_linear",
    "synthesize_observer_signal",
]


def resample_linear(signal: Array, t_src: Array, t_dst: Array) -> Array:
    """Linearly resample a time signal onto a new time grid (per row).

    Args:
        signal: Source samples [Pa], shape ``[.., T_src]`` (time last).
        t_src: Source sample times [s], shape ``[T_src]`` (increasing).
        t_dst: Target sample times [s], shape ``[T_dst]``.

    Returns:
        Resampled signal ``[.., T_dst]``; endpoints hold the edge values.
    """
    signal = jnp.asarray(signal, dtype=float)
    t_src = jnp.asarray(t_src, dtype=float)
    t_dst = jnp.asarray(t_dst, dtype=float)
    flat = signal.reshape(-1, signal.shape[-1])
    out = jax.vmap(lambda row: jnp.interp(t_dst, t_src, row))(flat)
    return out.reshape(*signal.shape[:-1], t_dst.shape[0])


def auralize_broadband(
    spectrogram_db: Array,
    band_centers: Array,
    fs: float,
    n_fft: int,
    hop: int | None = None,
    n_iters: int = 60,
    key: Array | None = None,
    momentum: float = 0.99,
    window: str = "hann",
    p_ref: float = P_REF,
) -> Array:
    r"""Auralise one 1/3-octave SPL spectrogram to a broadband time signal.

    Spreads each band's power uniformly over its STFT bins
    (:func:`third_octave_to_stft_mag`) and runs Fast Griffin--Lim to recover a
    consistent time signal whose 1/3-octave levels approximate the input.

    The overlap-add inverse STFT is ill-conditioned at the signal edges (the
    first/last ``n_fft`` samples span fewer than the full set of overlapping
    frames, so the least-squares window-envelope denominator is small and the
    edge samples blow up). To keep the *returned* waveform's true amplitude
    consistent with its band levels, the spectrogram is padded by
    ``ceil(n_fft/hop)`` edge-replicated frames on each side before Griffin--Lim
    and the corresponding samples are cropped afterwards, so the ill-conditioned
    transients fall entirely in the discarded padding. The returned length is
    unchanged: ``(n_frames - 1)*hop + n_fft``.

    Args:
        spectrogram_db: Band SPLs [dB re ``p_ref``], shape ``[n_frames, n_bands]``
            (one row per synthesis frame).
        band_centers: Band centre frequencies [Hz], shape ``[n_bands]``.
        fs: Audio sample rate [Hz].
        n_fft: STFT/FFT length in samples (static). Must resolve the lowest band.
        hop: STFT hop [samples]; default ``n_fft // 4`` (75% overlap).
        n_iters: Griffin--Lim iterations (static).
        key: PRNG key for the initial phases (default ``PRNGKey(0)``).
        momentum: FGLA momentum in ``[0, 1)``.
        window: STFT window name.
        p_ref: Reference pressure [Pa].

    Returns:
        Broadband time signal [Pa], shape ``[(n_frames - 1)*hop + n_fft]``.
    """
    if hop is None:
        hop = n_fft // 4
    spec = jnp.asarray(spectrogram_db, dtype=float)
    pad = -(-n_fft // hop)  # ceil(n_fft/hop): frames to cover one full window
    spec_p = jnp.concatenate(
        [jnp.repeat(spec[:1], pad, axis=0), spec, jnp.repeat(spec[-1:], pad, axis=0)], axis=0
    )
    mag = third_octave_to_stft_mag(
        spec_p,
        jnp.asarray(band_centers, dtype=float),
        n_fft,
        fs,
        p_ref=p_ref,
        window=window,
    )
    sig = griffin_lim(mag, n_fft, hop, n_iters=n_iters, key=key, momentum=momentum, window=window)
    return sig[pad * hop : sig.shape[0] - pad * hop]


def synthesize_observer_signal(
    fs: float,
    duration: float,
    *,
    tonal_pressure: Array | None = None,
    tonal_t: Array | None = None,
    broadband_spectrograms: list[Array] | None = None,
    band_centers: Array | None = None,
    n_fft: int = 1024,
    hop: int | None = None,
    n_iters: int = 60,
    key: Array | None = None,
    momentum: float = 0.99,
    window: str = "hann",
    p_ref: float = P_REF,
) -> dict[str, Array]:
    r"""Combine tonal + broadband noise into one observer audio signal.

    Builds a uniform audio grid of ``round(duration*fs)`` samples. The tonal
    pressure (if given) is linearly resampled onto it; each broadband
    spectrogram is auralised separately (independent phase) and the results are
    summed. All components are trimmed/zero-padded to the common length.

    Args:
        fs: Audio sample rate [Hz].
        duration: Signal duration [s] (sets the output length ``n = round(fs*dur)``).
        tonal_pressure: Deterministic tonal pressure [Pa], shape ``[T_tonal]``
            (one observer); ``None`` to omit.
        tonal_t: Times [s] for ``tonal_pressure``, shape ``[T_tonal]``.
        broadband_spectrograms: List of 1/3-octave SPL spectrograms
            ``[n_frames, n_bands]`` (one per rotor); ``None``/empty to omit.
        band_centers: Band centres [Hz] for the spectrograms, shape ``[n_bands]``.
        n_fft, hop, n_iters, key, momentum, window: Griffin--Lim / STFT settings.
        p_ref: Reference pressure [Pa].

    Returns:
        Dict with ``"tonal"``, ``"broadband"`` and ``"total"`` time signals
        [Pa], each shape ``[n]`` with ``n = round(fs*duration)``.
    """
    n = int(round(fs * duration))
    t_audio = jnp.arange(n) / fs
    if hop is None:
        hop = n_fft // 4

    tonal = jnp.zeros(n)
    if tonal_pressure is not None and tonal_t is not None:
        tp = jnp.asarray(tonal_pressure, dtype=float)
        tt = jnp.asarray(tonal_t, dtype=float)
        # Map the tonal window onto the start of the audio grid.
        tonal = resample_linear(tp, tt - tt[0], t_audio)

    broadband = jnp.zeros(n)
    if broadband_spectrograms:
        if band_centers is None:
            raise ValueError("band_centers required when broadband_spectrograms is given")
        base_key = jax.random.PRNGKey(0) if key is None else key
        for i, spec in enumerate(broadband_spectrograms):
            sub = jax.random.fold_in(base_key, i)
            sig = auralize_broadband(
                spec,
                band_centers,
                fs,
                n_fft,
                hop=hop,
                n_iters=n_iters,
                key=sub,
                momentum=momentum,
                window=window,
                p_ref=p_ref,
            )
            broadband = broadband + _fit_length(sig, n)

    return {"tonal": tonal, "broadband": broadband, "total": tonal + broadband}


def _fit_length(x: Array, n: int) -> Array:
    """Trim or zero-pad a 1-D signal to length ``n``."""
    m = x.shape[0]
    if m >= n:
        return x[:n]
    return jnp.concatenate([x, jnp.zeros(n - m, dtype=x.dtype)])


def cona_auralize(
    fs: float,
    tonal_pressure: Array,
    tonal_t: Array,
    broadband_spectrograms: list[Array],
    band_centers: Array,
    *,
    n_fft: int = 1024,
    hop: int | None = None,
    n_iters: int = 60,
    key: Array | None = None,
    momentum: float = 0.99,
    window: str = "hann",
    p_ref: float = P_REF,
) -> dict[str, Array]:
    r"""Convenience wrapper: auralise one observer over the tonal time window.

    The output duration is taken from the tonal time grid
    (``tonal_t[-1] - tonal_t[0]``). Thin shim over
    :func:`synthesize_observer_signal`.

    Args:
        fs: Audio sample rate [Hz].
        tonal_pressure: Tonal pressure [Pa], shape ``[T_tonal]`` (one observer).
        tonal_t: Times [s] for ``tonal_pressure``, shape ``[T_tonal]``.
        broadband_spectrograms: Per-rotor 1/3-octave SPL spectrograms.
        band_centers: Band centres [Hz], shape ``[n_bands]``.
        n_fft, hop, n_iters, key, momentum, window, p_ref: synthesis settings.

    Returns:
        Dict of ``"tonal"``, ``"broadband"``, ``"total"`` signals [Pa].
    """
    tonal_t = jnp.asarray(tonal_t, dtype=float)
    duration = float(tonal_t[-1] - tonal_t[0])
    return synthesize_observer_signal(
        fs,
        duration,
        tonal_pressure=tonal_pressure,
        tonal_t=tonal_t,
        broadband_spectrograms=broadband_spectrograms,
        band_centers=band_centers,
        n_fft=n_fft,
        hop=hop,
        n_iters=n_iters,
        key=key,
        momentum=momentum,
        window=window,
        p_ref=p_ref,
    )
