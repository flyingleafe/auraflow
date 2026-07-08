r"""Brooks & Burley rotor application of the BPM self-noise model.

Consumes a rotor's :class:`~auraflow.bemt.unsteady.SectionState` (produced
identically by :func:`auraflow.bemt.unsteady.march_bemt` and
:func:`auraflow.cona.airloads.rotor_section_state`) and turns every
``(blade, station, time)`` quasi-steady segment into a BPM one-third-octave
self-noise spectrum, radiated to world-frame observers with retarded TE-frame
directivity and an energy-conserving Doppler band re-binning.

Recipe (``docs/research/bpm-rotor-application.md``, Brooks & Burley
AIAA 2001-2210)
---------------------------------------------------------------------------
- Each disc segment is a quasi-steady isolated airfoil with a local
  trailing-edge frame built from the segment kinematics: ``e_span`` from the
  spanwise gradient of the blade positions, ``e_x`` (streamwise, downstream
  into the wake) from ``-velocity`` orthogonalised against ``e_span``, and
  ``e_z = e_x x e_span`` (section normal).
- The observer is transformed into that (retarded) TE frame; the RP-1218
  ``D_bar_h`` / ``D_bar_l`` directivities are evaluated there (we keep the
  original RP-1218 directivity forms with retarded geometry rather than the
  B&B ``(1 - M cos xi)^-4`` amplification -- a documented choice).
- Doppler ``f0/f = 1/(1 - M_tot cos xi_r)`` (source Mach toward the observer)
  shifts each segment spectrum; :func:`doppler_rebin` re-bins onto the fixed
  one-third-octave grid conserving band energy exactly for a static source.
- Source nulling: tip-vortex noise on the outermost station only; TE-bluntness
  only where the segment Mach ``< 0.5``; LBL-VS off by default (rotor inflow is
  rarely uniform enough).

Two output modes
----------------
- **Time-varying** (primary, JASA): each segment's Doppler-shifted spectrum is
  assigned to observer (arrival) time ``tau + R/c0`` and resampled onto a
  uniform frame grid, giving a ``[O, n_frames, n_bands]`` 1/3-octave
  spectrogram (energy-summed over blades and stations).
- **Rev-averaged** (Brooks & Burley Eq. 26): the emission-time energy sum with
  uniform azimuth dwell, giving ``[O, n_bands]`` band levels.

SI, float64, ``grad``-safe. Small shapes recommended (stations <= 20, few
observer frames) given host memory limits.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from auraflow.bemt.unsteady import SectionState
from auraflow.cona.bpm import bpm_third_octave
from auraflow.core.medium import Medium
from auraflow.signal.spectra import P_REF, third_octave_bands

__all__ = [
    "doppler_rebin",
    "rotor_broadband_levels",
    "rotor_broadband_spectrogram",
]

_TINY = 1e-30


def doppler_rebin(band_msq: Array, doppler: Array) -> Array:
    r"""Re-bin one-third-octave band energy under a Doppler frequency shift.

    A Doppler factor ``D = f/f0`` multiplies every frequency, which on the
    base-10 one-third-octave grid (centres ``10^{k/10}``) is a *uniform* shift
    of ``10 log10(D)`` band indices. Each source band's mean-square energy is
    split linearly between the two nearest shifted target bands. Because the
    shift is constant across bands for a given source element, the operation is
    a convex combination of two integer shifts and therefore conserves total
    energy exactly (up to grid-edge truncation); for ``D = 1`` it is the
    identity.

    Args:
        band_msq: Band mean-square pressures [Pa^2], shape ``[.., n_bands]``.
        doppler: Doppler factor ``D = f/f0`` [-], broadcastable to ``[..]``.

    Returns:
        Re-binned band mean-square pressures [Pa^2], shape ``[.., n_bands]``.
    """
    nb = band_msq.shape[-1]
    doppler = jnp.asarray(doppler, dtype=float)
    shift = 10.0 * jnp.log10(jnp.maximum(doppler, _TINY))  # [..]
    s0 = jnp.floor(shift)
    frac = (shift - s0)[..., None]  # [.., 1]
    tgt = jnp.arange(nb)
    # Source band that lands on target i is (i - shift); nearest integers are
    # (i - s0) and (i - s0 - 1). Same s0 for all i within an element => energy
    # conserving.
    idx_a = (tgt[None, :] if band_msq.ndim > 1 else tgt) - s0[..., None]
    idx_a = idx_a.astype(jnp.int32)
    idx_b = idx_a - 1

    def gather(idx: Array) -> Array:
        valid = (idx >= 0) & (idx < nb)
        safe = jnp.clip(idx, 0, nb - 1)
        return jnp.where(valid, jnp.take_along_axis(band_msq, safe, axis=-1), 0.0)

    return (1.0 - frac) * gather(idx_a) + frac * gather(idx_b)


def _segment_frames(position: Array) -> tuple[Array, Array]:
    """Spanwise and (raw) streamwise unit axes per segment, shape ``[B,S,T,3]``.

    ``e_span`` is the normalised spanwise gradient of the blade quarter-chord
    positions along the station axis; the streamwise axis is refined later from
    the segment velocity. Requires ``S >= 2``.
    """
    span = cast(Array, jnp.gradient(position, axis=1))  # tangent along stations [B,S,T,3]
    e_span = span / jnp.maximum(jnp.linalg.norm(span, axis=-1, keepdims=True), _TINY)
    return e_span, span


def _te_axes(velocity: Array, e_span: Array) -> tuple[Array, Array, Array]:
    """Orthonormal TE frame ``(e_x, e_span, e_z)`` per segment ``[B,S,T,3]``.

    ``e_x`` (streamwise, downstream into the wake) is ``-velocity`` made
    orthogonal to ``e_span``; ``e_z = e_x x e_span`` is the section normal.
    """
    vmag = jnp.maximum(jnp.linalg.norm(velocity, axis=-1, keepdims=True), _TINY)
    e_x0 = -velocity / vmag
    e_x = e_x0 - jnp.sum(e_x0 * e_span, axis=-1, keepdims=True) * e_span
    e_x = e_x / jnp.maximum(jnp.linalg.norm(e_x, axis=-1, keepdims=True), _TINY)
    e_z = jnp.cross(e_x, e_span)
    e_z = e_z / jnp.maximum(jnp.linalg.norm(e_z, axis=-1, keepdims=True), _TINY)
    return e_x, e_span, e_z


def _segment_msq(
    state: SectionState,
    observers: Array,
    medium: Medium,
    bands: Array,
    *,
    tripped: bool,
    include_lbl_vs: bool,
    include_tip: bool,
    include_bluntness: bool,
    h: ArrayLike,
    psi_deg: ArrayLike,
    tip_rounded: bool,
    prandtl_glauert: bool,
) -> tuple[Array, Array]:
    """Per-segment band mean-square at each observer and its arrival time.

    Returns:
        ``(msq, t_arr)`` with ``msq`` shape ``[O, B, S, T, n_bands]`` [Pa^2] and
        ``t_arr`` shape ``[O, B, S, T]`` [s].
    """
    pos = state.position  # [B,S,T,3]
    vel = state.velocity
    n_b, n_s, n_t = state.w.shape
    n_o = observers.shape[0]
    c0 = medium.c0

    e_span, _ = _segment_frames(pos)
    e_x, e_span, e_z = _te_axes(vel, e_span)

    # Observer displacement in world frame [O,B,S,T,3].
    d = observers[:, None, None, None, :] - pos[None]
    r_dist = jnp.maximum(jnp.linalg.norm(d, axis=-1), _TINY)  # [O,B,S,T]
    rhat = d / r_dist[..., None]

    # Source Mach toward observer and Doppler factor.
    m_r = jnp.sum(vel[None] * rhat, axis=-1) / c0  # [O,B,S,T]
    doppler = 1.0 / jnp.clip(1.0 - m_r, 0.05, None)  # f/f0

    # TE-frame observer coordinates and emission angles.
    x_e = jnp.sum(d * e_x[None], axis=-1)
    y_e = jnp.sum(d * e_span[None], axis=-1)
    z_e = jnp.sum(d * e_z[None], axis=-1)
    theta_e = jnp.arccos(jnp.clip(x_e / r_dist, -1.0, 1.0))
    sin2phi = z_e**2 / jnp.maximum(y_e**2 + z_e**2, _TINY)
    phi_e = jnp.arcsin(jnp.clip(jnp.sqrt(jnp.maximum(sin2phi, _TINY)), 0.0, 1.0))

    # Broadcast per-segment scalars to [O,B,S,T].
    def obst(x_bst: Array) -> Array:
        return jnp.broadcast_to(x_bst[None], (n_o, n_b, n_s, n_t))

    w = obst(state.w)
    alpha_deg = obst(jnp.rad2deg(state.alpha))
    reyn = obst(state.reynolds)
    mach = obst(state.mach)
    chord = obst(jnp.broadcast_to(state.chord[None, :, None], (n_b, n_s, n_t)))
    span = obst(jnp.broadcast_to(state.dr[None, :, None], (n_b, n_s, n_t)))

    shp = (n_o, n_b, n_s, n_t)
    flat = int(n_o * n_b * n_s * n_t)

    def one(u, cc, sp, rc, mm, al, th, ph, rr):
        return bpm_third_octave(
            bands,
            u,
            cc,
            sp,
            rc,
            mm,
            medium,
            alpha_deg=al,
            theta_e_deg=jnp.rad2deg(th),
            phi_e_deg=jnp.rad2deg(ph),
            r_e=rr,
            tripped=tripped,
            include_tbl_te=True,
            include_lbl_vs=include_lbl_vs,
            include_tip=include_tip,
            include_bluntness=include_bluntness,
            h=h,
            psi_deg=psi_deg,
            tip_rounded=tip_rounded,
            prandtl_glauert=prandtl_glauert,
        )

    spec = jax.vmap(one)(
        w.reshape(flat),
        chord.reshape(flat),
        span.reshape(flat),
        reyn.reshape(flat),
        mach.reshape(flat),
        alpha_deg.reshape(flat),
        theta_e.reshape(flat),
        phi_e.reshape(flat),
        r_dist.reshape(flat),
    )

    def unflat(x: Array) -> Array:
        return x.reshape(*shp, bands.shape[0])

    tbl = 10.0 ** (unflat(spec.tbl_te) / 10.0) * P_REF**2
    lbl = 10.0 ** (unflat(spec.lbl_vs) / 10.0) * P_REF**2
    tip = 10.0 ** (unflat(spec.tip) / 10.0) * P_REF**2
    blunt = 10.0 ** (unflat(spec.bluntness) / 10.0) * P_REF**2

    # Source nulling (B&B): tip on the outer station only; bluntness for M<0.5.
    if include_tip:
        is_tip = (jnp.arange(n_s) == n_s - 1).astype(float)[None, None, :, None, None]
        tip = tip * is_tip
    else:
        tip = jnp.zeros_like(tip)
    if include_bluntness:
        blunt = blunt * (mach[..., None] < 0.5).astype(float)
    else:
        blunt = jnp.zeros_like(blunt)
    if not include_lbl_vs:
        lbl = jnp.zeros_like(lbl)

    msq = tbl + lbl + tip + blunt  # [O,B,S,T,n_bands]
    msq = doppler_rebin(msq, doppler)
    return msq, r_dist / c0  # msq, propagation delay (arrival = emission t + this)


def rotor_broadband_spectrogram(
    section_state: SectionState,
    observers: ArrayLike,
    medium: Medium,
    t: ArrayLike,
    *,
    bands: Array | None = None,
    fmin: float = 100.0,
    fmax: float = 20000.0,
    n_frames: int | None = None,
    tripped: bool = True,
    include_lbl_vs: bool = False,
    include_tip: bool = True,
    include_bluntness: bool = False,
    h: ArrayLike = 0.0,
    psi_deg: ArrayLike = 14.0,
    tip_rounded: bool = True,
    prandtl_glauert: bool = True,
) -> tuple[Array, Array, Array]:
    r"""Time-varying BPM broadband 1/3-octave spectrogram for one rotor.

    Each ``(blade, station, time)`` segment's Doppler-shifted BPM spectrum is
    assigned to its observer arrival time ``t + R/c0`` and resampled onto a
    uniform frame grid, then energy-summed over blades and stations.

    Args:
        section_state: One rotor's :class:`SectionState` (``[B,S,T]`` leaves).
        observers: World-frame observer positions [m], shape ``[O, 3]``.
        medium: Ambient medium.
        t: Emission-time grid [s], shape ``[T]`` (the flight-history times).
        bands: Optional explicit band centres [Hz]; default IEC bands over
            ``[fmin, fmax]`` clamped to ``c0/2``-ish (uses ``third_octave_bands``).
        fmin, fmax: Band range if ``bands`` is not given [Hz].
        n_frames: Number of observer-time frames (static); default ``T``. The
            frame hop is ``(t_hi - t_lo) / (n_frames - 1)``.
        tripped: Heavily-tripped BL correlations.
        include_lbl_vs: Include LBL-VS (off by default on a rotor).
        include_tip: Include tip-vortex noise (outer station only).
        include_bluntness: Include TE-bluntness (``M < 0.5`` segments only).
        h: TE thickness [m] (bluntness).
        psi_deg: TE solid angle [deg] (bluntness).
        tip_rounded: Rounded vs flat tip.
        prandtl_glauert: Apply ``1/(1-M^2)`` to TBL-TE (rotor convention).

    Returns:
        ``(centers, spectrogram, frame_times)``: band centres [Hz] ``[n_bands]``,
        SPL spectrogram [dB re 20 uPa] ``[O, n_frames, n_bands]`` and the frame
        time grid [s] ``[n_frames]``.
    """
    observers = jnp.asarray(observers, dtype=float)
    t = jnp.asarray(t, dtype=float)
    if bands is None:
        bands, _ = third_octave_bands(fmin, fmax)
    n_t = t.shape[0]
    if n_frames is None:
        n_frames = n_t

    msq, delay = _segment_msq(
        section_state,
        observers,
        medium,
        bands,
        tripped=tripped,
        include_lbl_vs=include_lbl_vs,
        include_tip=include_tip,
        include_bluntness=include_bluntness,
        h=h,
        psi_deg=psi_deg,
        tip_rounded=tip_rounded,
        prandtl_glauert=prandtl_glauert,
    )
    n_o, n_b, n_s, _, n_k = msq.shape
    t_arr = t[None, None, None, :] + delay  # [O,B,S,T]

    # Common arrival window across all segments/observers.
    t_lo = jnp.max(jnp.min(t_arr, axis=-1))
    t_hi = jnp.min(jnp.max(t_arr, axis=-1))
    frame_times = jnp.linspace(t_lo, t_hi, n_frames)

    # Resample each segment's spectrum onto the frame grid (interp in arrival
    # time, per band) and energy-sum over blades+stations.
    seg = int(n_b * n_s)
    t_arr_g = t_arr.reshape(n_o, seg, n_t)
    msq_g = msq.reshape(n_o, seg, n_t, n_k)

    def interp_band(xp: Array, fp: Array) -> Array:  # xp [T], fp [T]
        return jnp.interp(frame_times, xp, fp, left=0.0, right=0.0)

    def interp_seg(xp: Array, fp_tk: Array) -> Array:  # xp [T], fp_tk [T,K]
        return jax.vmap(lambda fp_t: interp_band(xp, fp_t))(fp_tk.T)  # [K,n_frames]

    def interp_obs(xp_gt: Array, fp_gtk: Array) -> Array:
        return jax.vmap(interp_seg)(xp_gt, fp_gtk)  # [seg,K,n_frames]

    resampled = jax.vmap(interp_obs)(t_arr_g, msq_g)  # [O,seg,K,n_frames]
    frame_msq = jnp.sum(resampled, axis=1).transpose(0, 2, 1)  # [O,n_frames,K]
    spectrogram = 10.0 * jnp.log10(jnp.maximum(frame_msq, _TINY) / P_REF**2)
    return bands, spectrogram, frame_times


def rotor_broadband_levels(
    section_state: SectionState,
    observers: ArrayLike,
    medium: Medium,
    t: ArrayLike,
    *,
    bands: Array | None = None,
    fmin: float = 100.0,
    fmax: float = 20000.0,
    tripped: bool = True,
    include_lbl_vs: bool = False,
    include_tip: bool = True,
    include_bluntness: bool = False,
    h: ArrayLike = 0.0,
    psi_deg: ArrayLike = 14.0,
    tip_rounded: bool = True,
    prandtl_glauert: bool = True,
) -> tuple[Array, Array]:
    r"""Revolution-averaged BPM broadband 1/3-octave levels (B&B Eq. 26).

    The emission-time energy sum over all ``(blade, station, time)`` segments
    with uniform azimuth dwell (``Delta psi / 360`` = uniform time weighting)
    and blade-count summation, i.e. the time-mean of the segment-summed band
    mean-square. See :func:`rotor_broadband_spectrogram` for arguments.

    Args:
        section_state: One rotor's :class:`SectionState`.
        observers: World-frame observer positions [m], shape ``[O, 3]``.
        medium: Ambient medium.
        t: Emission-time grid [s], shape ``[T]``.

    Returns:
        ``(centers, levels)``: band centres [Hz] ``[n_bands]`` and rev-averaged
        SPL [dB re 20 uPa] ``[O, n_bands]``.
    """
    observers = jnp.asarray(observers, dtype=float)
    t = jnp.asarray(t, dtype=float)
    if bands is None:
        bands, _ = third_octave_bands(fmin, fmax)

    msq, _ = _segment_msq(
        section_state,
        observers,
        medium,
        bands,
        tripped=tripped,
        include_lbl_vs=include_lbl_vs,
        include_tip=include_tip,
        include_bluntness=include_bluntness,
        h=h,
        psi_deg=psi_deg,
        tip_rounded=tip_rounded,
        prandtl_glauert=prandtl_glauert,
    )
    # Sum energy over blades+stations, average over emission time (dwell).
    seg_sum = jnp.sum(msq, axis=(1, 2))  # [O,T,n_bands]
    levels_msq = jnp.mean(seg_sum, axis=1)  # [O,n_bands]
    levels = 10.0 * jnp.log10(jnp.maximum(levels_msq, _TINY) / P_REF**2)
    return bands, levels
