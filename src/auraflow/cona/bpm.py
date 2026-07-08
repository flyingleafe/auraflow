r"""BPM airfoil self-noise model (NASA RP-1218), differentiable in JAX.

Implements the Brooks--Pope--Marcolini (BPM) semi-empirical broadband
airfoil-self-noise prediction, *Airfoil Self-Noise and Prediction*, NASA
RP-1218 (1989). Equation numbers in the docstrings/comments are the report's.
Everything is digested in ``docs/research/bpm-model-reference.md`` and verified
1:1 against OpenFAST ``AeroAcoustics.f90``.

Five mechanisms are provided, each producing a one-third-octave-band SPL
spectrum [dB re 20 uPa]:

- turbulent boundary-layer trailing-edge (TBL-TE), with the attached/separated
  (stall) angle switch, pressure/suction/separation contributions (Eqs. 24--30);
- laminar boundary-layer vortex-shedding (LBL-VS, Eq. 53), for untripped /
  lightly tripped airfoils only;
- tip-vortex-formation noise (Eq. 61), tip segment only;
- trailing-edge bluntness vortex-shedding (Eqs. 70--82), for finite TE thickness.

plus the RP-1218 boundary-layer-thickness correlations (NACA 0012) and the
Appendix-B convective directivity functions.

Conventions (RP-1218 sec. 0)
----------------------------
- All lengths in **metres** (chord ``c``, segment span ``L``, TE thickness
  ``h``, observer distance ``re``, boundary-layer thicknesses). ``delta*``,
  ``L`` and ``re`` are in *consistent* metres so ``delta* L / re^2`` is
  dimensionless.
- ``U`` section inflow speed [m/s]; ``M = U/c0``; ``Mc ~= 0.8 M``.
- ``Rc = U c / nu`` chord Reynolds number; ``R_dstar_p = U delta*_p / nu``.
- ``alpha*`` effective angle of attack in **degrees** (from the zero-lift line);
  the correlations use ``|alpha*|`` (symmetric NACA 0012).
- ``f`` one-third-octave centre frequency [Hz]; SPL dB re 20 uPa; ``log`` is
  base 10.
- Calibration range: NACA 0012, ``Rc <~ 3e6``, ``M <~ 0.21``, ``alpha* <= 25.2``.

Numerical safety
----------------
Every ``jnp.where`` branch is evaluated on both sides with finite arguments
(``sqrt`` guarded by ``maximum(., 0)``, ``log10`` by ``maximum(., tiny)``), so
the model never propagates NaNs under ``jax.grad``. Piecewise transitions are
not smoothed (not required by the model), but no branch produces a
non-finite value that could contaminate the selected branch's gradient.

The public entry point :func:`bpm_third_octave` is fully ``vmap``-able over
blade sections; per-mechanism helpers are exposed for testing against the
report's anchor values.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from auraflow.core.medium import Medium

__all__ = [
    "BLThickness",
    "BPMSpectra",
    "a_shape",
    "b_shape",
    "boundary_layer_thickness",
    "bpm_third_octave",
    "bluntness_noise",
    "directivity_high",
    "directivity_low",
    "k1_amplitude",
    "k2_amplitude",
    "lbl_vs_noise",
    "st1_peak",
    "tbl_te_noise",
    "te_frame_angles",
    "tip_vortex_noise",
]

_TINY = 1e-30
_SPL_FLOOR = -300.0
"""dB floor for nulled contributions (mean-square ~1e-30, finite gradient)."""


def _log10(x: Array) -> Array:
    """``log10`` with a positive floor (finite value and gradient)."""
    return jnp.log10(jnp.maximum(x, _TINY))


def _pow10(x: Array) -> Array:
    return jnp.power(10.0, x)


def _sqrt(x: Array) -> Array:
    """``sqrt`` with a strictly-positive floor (finite value *and* gradient).

    Flooring by ``_TINY`` (not ``0``) is essential for gradient safety: with a
    ``0`` floor, at ``x < 0`` the ``maximum`` gate has zero gradient while
    ``0.5/sqrt(0) = inf``, so ``0 * inf = NaN`` would flow back through an
    unselected ``jnp.where`` branch. A positive floor keeps that product finite.
    """
    return jnp.sqrt(jnp.maximum(x, _TINY))


# =====================================================================
# Boundary-layer thickness correlations (RP-1218 Eqs. 2--16, NACA 0012)
# =====================================================================


class BLThickness(NamedTuple):
    """Trailing-edge boundary-layer thicknesses [m] for one flow state.

    Attributes:
        delta_p: Pressure-side BL thickness ``delta_p`` [m] (LBL-VS scaling).
        delta_s: Suction-side BL thickness ``delta_s`` [m].
        dstar_p: Pressure-side displacement thickness ``delta*_p`` [m].
        dstar_s: Suction-side displacement thickness ``delta*_s`` [m].
        dstar_avg: Mean displacement thickness ``(delta*_p + delta*_s)/2`` [m].
    """

    delta_p: Array
    delta_s: Array
    dstar_p: Array
    dstar_s: Array
    dstar_avg: Array


def boundary_layer_thickness(
    re_c: ArrayLike, alpha_deg: ArrayLike, chord: ArrayLike, tripped: bool = True
) -> BLThickness:
    r"""NACA 0012 trailing-edge BL thicknesses (RP-1218 Eqs. 2--16).

    Zero-``alpha`` thicknesses come from the tripped (Eqs. 2--4) or untripped
    (Eqs. 5--7) correlations, then are scaled by the pressure-side (Eqs. 8--10)
    and suction-side (Eqs. 11--13 tripped / 14--16 untripped) ``alpha``
    dependence. Thicknesses are normalised by ``chord`` internally and returned
    dimensionally in metres.

    Args:
        re_c: Chord Reynolds number ``Rc = U c / nu`` [-].
        alpha_deg: Effective angle of attack ``alpha*`` [deg]; ``|alpha*|`` is
            used (symmetric section). Clamped to the correlation range 0..25 deg.
        chord: Section chord ``c`` [m].
        tripped: If ``True`` use the heavily-tripped correlations, else the
            untripped ones.

    Returns:
        A :class:`BLThickness` with ``[..]``-shaped leaves in metres.
    """
    re_c = jnp.asarray(re_c, dtype=float)
    chord = jnp.asarray(chord, dtype=float)
    a = jnp.clip(jnp.abs(jnp.asarray(alpha_deg, dtype=float)), 0.0, 25.0)
    logr = _log10(re_c)

    # --- Zero-alpha thickness ratios (delta0/c, delta0*/c) ---
    if tripped:
        d0_c = _pow10(1.892 - 0.9045 * logr + 0.0596 * logr**2)
        dstar0_c = jnp.where(
            re_c <= 3.0e5,
            0.0601 * jnp.power(jnp.maximum(re_c, _TINY), -0.114),
            _pow10(3.411 - 1.5397 * logr + 0.1059 * logr**2),
        )
    else:
        d0_c = _pow10(1.6569 - 0.9045 * logr + 0.0596 * logr**2)
        dstar0_c = _pow10(3.0187 - 1.5397 * logr + 0.1059 * logr**2)

    # --- Pressure side (Eqs. 8--10), both trip states ---
    dp_ratio = _pow10(-0.04175 * a + 0.00106 * a**2)
    dstar_p_ratio = _pow10(-0.0432 * a + 0.00113 * a**2)

    # --- Suction side (Eqs. 11--13 tripped / 14--16 untripped) ---
    if tripped:
        ds_ratio = jnp.where(
            a < 5.0,
            _pow10(0.0311 * a),
            jnp.where(a <= 12.5, 0.3468 * _pow10(0.1231 * a), 5.718 * _pow10(0.0258 * a)),
        )
        dstar_s_ratio = jnp.where(
            a < 5.0,
            _pow10(0.0679 * a),
            jnp.where(a <= 12.5, 0.381 * _pow10(0.1516 * a), 14.296 * _pow10(0.0258 * a)),
        )
    else:
        ds_ratio = jnp.where(
            a < 7.5,
            _pow10(0.03114 * a),
            jnp.where(a <= 12.5, 0.0303 * _pow10(0.2336 * a), 12.0 * _pow10(0.0258 * a)),
        )
        dstar_s_ratio = jnp.where(
            a < 7.5,
            _pow10(0.0679 * a),
            jnp.where(a <= 12.5, 0.0162 * _pow10(0.3066 * a), 52.42 * _pow10(0.0258 * a)),
        )

    delta_p = dp_ratio * d0_c * chord
    delta_s = ds_ratio * d0_c * chord
    dstar_p = dstar_p_ratio * dstar0_c * chord
    dstar_s = dstar_s_ratio * dstar0_c * chord
    return BLThickness(delta_p, delta_s, dstar_p, dstar_s, 0.5 * (dstar_p + dstar_s))


# =====================================================================
# Directivity (RP-1218 Appendix B)
# =====================================================================


def te_frame_angles(observer_te: Array) -> tuple[Array, Array, Array]:
    r"""TE-local emission angles from an observer vector in the TE frame.

    The trailing-edge frame has ``x_e`` downstream along the chordline into the
    wake, ``y_e`` spanwise and ``z_e`` normal. From the observer position vector
    ``(x_e, y_e, z_e)`` relative to the TE:

    - ``Theta_e = acos(x_e / re)`` (angle from the streamwise axis),
    - ``sin^2 Phi_e = z_e^2 / (y_e^2 + z_e^2)``.

    Args:
        observer_te: Observer position(s) in the TE frame [m], shape ``[.., 3]``.

    Returns:
        ``(theta_e, phi_e, re)`` in radians, radians and metres, shape ``[..]``.
    """
    x_e = observer_te[..., 0]
    y_e = observer_te[..., 1]
    z_e = observer_te[..., 2]
    re = jnp.sqrt(x_e**2 + y_e**2 + z_e**2)
    theta_e = jnp.arccos(jnp.clip(x_e / jnp.maximum(re, _TINY), -1.0, 1.0))
    sin2phi = z_e**2 / jnp.maximum(y_e**2 + z_e**2, _TINY)
    phi_e = jnp.arcsin(jnp.clip(_sqrt(sin2phi), 0.0, 1.0))
    return theta_e, phi_e, re


def directivity_high(
    theta_e: ArrayLike, phi_e: ArrayLike, mach: ArrayLike, mach_c: ArrayLike
) -> Array:
    r"""High-frequency convective directivity ``D_bar_h`` (RP-1218 Eq. B1).

    ``D_bar_h = 2 sin^2(Theta_e/2) sin^2(Phi_e) /
    [(1 + M cosTheta_e)(1 + (M - Mc) cosTheta_e)^2]``. Normalised so
    ``D_bar_h = 1`` at ``Theta_e = Phi_e = 90 deg``. Used by every mechanism
    except separated (stalled) TBL-TE.

    Args:
        theta_e: Emission polar angle ``Theta_e`` [rad].
        phi_e: Emission azimuth angle ``Phi_e`` [rad].
        mach: Section Mach number ``M`` [-].
        mach_c: Convective Mach number ``Mc`` [-].

    Returns:
        ``D_bar_h`` [-], broadcast shape.
    """
    theta_e = jnp.asarray(theta_e, dtype=float)
    phi_e = jnp.asarray(phi_e, dtype=float)
    mach = jnp.asarray(mach, dtype=float)
    mach_c = jnp.asarray(mach_c, dtype=float)
    ct = jnp.cos(theta_e)
    num = 2.0 * jnp.sin(0.5 * theta_e) ** 2 * jnp.sin(phi_e) ** 2
    den = (1.0 + mach * ct) * (1.0 + (mach - mach_c) * ct) ** 2
    return num / jnp.where(jnp.abs(den) < _TINY, _TINY, den)


def directivity_low(theta_e: ArrayLike, phi_e: ArrayLike, mach: ArrayLike) -> Array:
    r"""Low-frequency directivity ``D_bar_l`` (RP-1218 Eq. B2).

    ``D_bar_l = sin^2(Theta_e) sin^2(Phi_e) / (1 + M cosTheta_e)^4``. Used for
    the separated (stalled) TBL-TE contribution; a dipole with a null along the
    TE plane (``Theta_e = 0`` or ``pi``).

    Args:
        theta_e: Emission polar angle ``Theta_e`` [rad].
        phi_e: Emission azimuth angle ``Phi_e`` [rad].
        mach: Section Mach number ``M`` [-].

    Returns:
        ``D_bar_l`` [-], broadcast shape.
    """
    theta_e = jnp.asarray(theta_e, dtype=float)
    phi_e = jnp.asarray(phi_e, dtype=float)
    mach = jnp.asarray(mach, dtype=float)
    ct = jnp.cos(theta_e)
    num = jnp.sin(theta_e) ** 2 * jnp.sin(phi_e) ** 2
    den = (1.0 + mach * ct) ** 4
    return num / jnp.where(jnp.abs(den) < _TINY, _TINY, den)


# =====================================================================
# TBL-TE spectral shape functions A and B (RP-1218 sec. "A/B-shape")
# =====================================================================


def _a_min(a: Array) -> Array:
    a = jnp.abs(a)
    sq = _sqrt(67.552 - 886.788 * a**2) - 8.219
    mid = -32.665 * a + 3.981
    hi = -142.795 * a**3 + 103.656 * a**2 - 57.757 * a + 6.006
    return jnp.where(a < 0.204, sq, jnp.where(a <= 0.244, mid, hi))


def _a_max(a: Array) -> Array:
    a = jnp.abs(a)
    sq = _sqrt(67.552 - 886.788 * a**2) - 8.219
    mid = -15.901 * a + 1.098
    hi = -4.669 * a**3 + 3.491 * a**2 - 16.699 * a + 1.149
    return jnp.where(a < 0.13, sq, jnp.where(a <= 0.321, mid, hi))


def _a0_of_rc(re_c: Array) -> Array:
    lo = jnp.asarray(0.57)
    mid = -9.57e-13 * (re_c - 8.57e5) ** 2 + 1.13
    return jnp.where(re_c < 9.52e4, lo, jnp.where(re_c <= 8.57e5, mid, 1.13))


def a_shape(strouhal_ratio: ArrayLike, re_c: ArrayLike) -> Array:
    r"""TBL-TE ``A``-shape spectral function (RP-1218 A-curve).

    ``a = |log10(St/St_peak)|``; ``A(a) = A_min(a) + AR (A_max(a) - A_min(a))``
    with the interpolation factor ``AR = (-20 - A_min(a0)) / (A_max(a0) -
    A_min(a0))`` set by ``a0(Rc)``. By construction ``A(a0) = -20`` and
    ``A(0) = 0`` (the spectral peak).

    Args:
        strouhal_ratio: ``St / St_peak`` [-] (the A-curve argument).
        re_c: Chord Reynolds number, sets the interpolation width ``a0``.

    Returns:
        ``A`` [dB], broadcast shape.
    """
    a = jnp.abs(_log10(jnp.asarray(strouhal_ratio, dtype=float)))
    a0 = _a0_of_rc(jnp.asarray(re_c, dtype=float))
    amin0 = _a_min(a0)
    amax0 = _a_max(a0)
    ar = (-20.0 - amin0) / jnp.where(jnp.abs(amax0 - amin0) < _TINY, _TINY, amax0 - amin0)
    return _a_min(a) + ar * (_a_max(a) - _a_min(a))


def _b_min(b: Array) -> Array:
    b = jnp.abs(b)
    sq = _sqrt(16.888 - 886.788 * b**2) - 4.109
    mid = -83.607 * b + 8.138
    hi = -817.810 * b**3 + 355.210 * b**2 - 135.024 * b + 10.619
    return jnp.where(b < 0.13, sq, jnp.where(b <= 0.145, mid, hi))


def _b_max(b: Array) -> Array:
    b = jnp.abs(b)
    sq = _sqrt(16.888 - 886.788 * b**2) - 4.109
    mid = -31.330 * b + 1.854
    hi = -80.541 * b**3 + 44.174 * b**2 - 39.381 * b + 2.344
    return jnp.where(b < 0.10, sq, jnp.where(b <= 0.187, mid, hi))


def _b0_of_rc(re_c: Array) -> Array:
    lo = jnp.asarray(0.30)
    mid = -4.48e-13 * (re_c - 8.57e5) ** 2 + 0.56
    return jnp.where(re_c < 9.52e4, lo, jnp.where(re_c <= 8.57e5, mid, 0.56))


def b_shape(strouhal_ratio: ArrayLike, re_c: ArrayLike) -> Array:
    r"""TBL-TE ``B``-shape spectral function (RP-1218 B-curve).

    ``b = |log10(Sts/St2)|``; ``B(b) = B_min(b) + BR (B_max(b) - B_min(b))``
    with ``BR = (-20 - B_min(b0)) / (B_max(b0) - B_min(b0))`` from ``b0(Rc)``.
    ``B(b0) = -20``, ``B(0) = 0``.

    Args:
        strouhal_ratio: ``Sts / St2`` [-] (the B-curve argument).
        re_c: Chord Reynolds number, sets ``b0``.

    Returns:
        ``B`` [dB], broadcast shape.
    """
    b = jnp.abs(_log10(jnp.asarray(strouhal_ratio, dtype=float)))
    b0 = _b0_of_rc(jnp.asarray(re_c, dtype=float))
    bmin0 = _b_min(b0)
    bmax0 = _b_max(b0)
    br = (-20.0 - bmin0) / jnp.where(jnp.abs(bmax0 - bmin0) < _TINY, _TINY, bmax0 - bmin0)
    return _b_min(b) + br * (_b_max(b) - _b_min(b))


# =====================================================================
# TBL-TE amplitude constants K1, dK1, K2 (RP-1218 sec. "Amplitudes")
# =====================================================================


def k1_amplitude(re_c: ArrayLike) -> Array:
    r"""TBL-TE amplitude ``K1(Rc)`` (RP-1218).

    ``K1 = -4.31 log Rc + 156.3`` (``Rc < 2.47e5``); ``-9.0 log Rc + 181.6``
    (``2.47e5 <= Rc < 8e5``); ``128.5`` (``Rc >= 8e5``).

    Args:
        re_c: Chord Reynolds number.

    Returns:
        ``K1`` [dB].
    """
    re_c = jnp.asarray(re_c, dtype=float)
    logr = _log10(re_c)
    lo = -4.31 * logr + 156.3
    mid = -9.0 * logr + 181.6
    return jnp.where(re_c < 2.47e5, lo, jnp.where(re_c < 8.0e5, mid, 128.5))


def _delta_k1(alpha_deg: Array, r_dstar_p: Array) -> Array:
    """dK1 pressure-side amplitude adjustment (RP-1218)."""
    val = alpha_deg * (1.43 * _log10(r_dstar_p) - 5.29)
    return jnp.where(r_dstar_p <= 5000.0, val, 0.0)


def k2_amplitude(alpha_deg: ArrayLike, mach: ArrayLike, k1: ArrayLike) -> Array:
    r"""TBL-TE separation amplitude ``K2`` (RP-1218).

    ``K2 = K1 + {-1000 (alpha < gamma0 - gamma);
    sqrt(beta^2 - (beta/gamma)^2 (alpha - gamma0)^2) + beta0
    (|alpha - gamma0| <= gamma); -12 (alpha > gamma0 + gamma)}`` with
    ``gamma = 27.094 M + 3.31``, ``gamma0 = 23.43 M + 4.651``,
    ``beta = 72.65 M + 10.74``, ``beta0 = -34.19 M - 13.82``.

    Args:
        alpha_deg: ``|alpha*|`` [deg].
        mach: Section Mach number.
        k1: The ``K1`` amplitude [dB].

    Returns:
        ``K2`` [dB].
    """
    a = jnp.asarray(alpha_deg, dtype=float)
    m = jnp.asarray(mach, dtype=float)
    gamma = 27.094 * m + 3.31
    gamma0 = 23.43 * m + 4.651
    beta = 72.65 * m + 10.74
    beta0 = -34.19 * m - 13.82
    da = a - gamma0
    mid = _sqrt(beta**2 - (beta / jnp.maximum(gamma, _TINY)) ** 2 * da**2) + beta0
    delta = jnp.where(da < -gamma, -1000.0, jnp.where(da <= gamma, mid, -12.0))
    return jnp.asarray(k1, dtype=float) + delta


# =====================================================================
# Strouhal helpers
# =====================================================================


def st1_peak(mach: ArrayLike) -> Array:
    r"""TBL-TE reference Strouhal number ``St1 = 0.02 M^{-0.6}`` (RP-1218)."""
    m = jnp.maximum(jnp.asarray(mach, dtype=float), _TINY)
    return 0.02 * jnp.power(m, -0.6)


def _st2_ratio(alpha_deg: Array) -> Array:
    """St2/St1 factor vs alpha (RP-1218)."""
    a = jnp.asarray(alpha_deg, dtype=float)
    mid = _pow10(0.0054 * (a - 1.33) ** 2)
    return jnp.where(a < 1.33, 1.0, jnp.where(a <= 12.5, mid, 4.72))


# =====================================================================
# Mechanism: TBL-TE + separation (RP-1218 Eqs. 24--30)
# =====================================================================


def tbl_te_noise(
    freqs: Array,
    U: ArrayLike,
    mach: ArrayLike,
    re_c: ArrayLike,
    dstar_p: ArrayLike,
    dstar_s: ArrayLike,
    span: ArrayLike,
    re: ArrayLike,
    dbar_h: ArrayLike,
    dbar_l: ArrayLike,
    alpha_deg: ArrayLike,
    nu: ArrayLike,
) -> Array:
    r"""Turbulent-BL trailing-edge + separation noise (RP-1218 Eqs. 24--30).

    Sums the pressure-side (Eq. 25), suction-side (Eq. 26) and separation
    (Eq. 27 attached / Eq. 30 stalled) contributions in energy. The
    attached/stalled switch is at ``(alpha*)0 = min(gamma0, 12.5 deg)``; in the
    stalled branch the pressure and suction contributions are nulled and the
    separation term uses ``A'`` (A-curve with ``Rc -> 3 Rc``) and the
    low-frequency directivity ``D_bar_l``.

    Args:
        freqs: One-third-octave centre frequencies [Hz], shape ``[n_bands]``.
        U: Section inflow speed [m/s].
        mach: Section Mach number.
        re_c: Chord Reynolds number.
        dstar_p, dstar_s: Pressure/suction displacement thicknesses [m].
        span: Segment wetted span ``L`` [m].
        re: Observer distance ``re`` [m].
        dbar_h, dbar_l: High/low-frequency directivities.
        alpha_deg: ``|alpha*|`` [deg].
        nu: Kinematic viscosity [m^2/s].

    Returns:
        TBL-TE total SPL [dB], shape ``[n_bands]``.
    """
    m = jnp.asarray(mach, dtype=float)
    a = jnp.abs(jnp.asarray(alpha_deg, dtype=float))
    U = jnp.asarray(U, dtype=float)
    dstar_p = jnp.asarray(dstar_p, dtype=float)
    dstar_s = jnp.asarray(dstar_s, dtype=float)

    st1 = st1_peak(m)
    st2 = st1 * _st2_ratio(a)
    st_bar1 = 0.5 * (st1 + st2)

    stp = freqs * dstar_p / jnp.maximum(U, _TINY)
    sts = freqs * dstar_s / jnp.maximum(U, _TINY)

    k1 = k1_amplitude(re_c)
    r_dstar_p = U * dstar_p / jnp.maximum(jnp.asarray(nu, dtype=float), _TINY)
    dk1 = _delta_k1(a, r_dstar_p)
    k2 = k2_amplitude(a, m, k1)

    log_m5 = 5.0 * _log10(m)
    log_geom = _log10(jnp.asarray(span, dtype=float)) - 2.0 * _log10(jnp.asarray(re, dtype=float))
    base_p = 10.0 * (_log10(dstar_p) + log_m5 + log_geom + _log10(jnp.asarray(dbar_h)))
    base_s = 10.0 * (_log10(dstar_s) + log_m5 + log_geom + _log10(jnp.asarray(dbar_h)))
    base_alpha_l = 10.0 * (_log10(dstar_s) + log_m5 + log_geom + _log10(jnp.asarray(dbar_l)))

    # Attached form.
    spl_p_att = base_p + a_shape(stp / st1, re_c) + (k1 - 3.0) + dk1
    spl_s_att = base_s + a_shape(sts / st_bar1, re_c) + (k1 - 3.0)
    spl_a_att = base_s + b_shape(sts / st2, re_c) + k2

    # Stalled form: p/s nulled, separation on D_bar_l with A'(Rc->3Rc).
    spl_a_stall = base_alpha_l + a_shape(sts / st2, 3.0 * jnp.asarray(re_c, dtype=float)) + k2

    gamma0 = 23.43 * m + 4.651
    alpha0 = jnp.minimum(gamma0, 12.5)
    attached = a <= alpha0

    spl_p = jnp.where(attached, spl_p_att, _SPL_FLOOR)
    spl_s = jnp.where(attached, spl_s_att, _SPL_FLOOR)
    spl_a = jnp.where(attached, spl_a_att, spl_a_stall)

    return 10.0 * _log10(_pow10(spl_p / 10.0) + _pow10(spl_s / 10.0) + _pow10(spl_a / 10.0))


# =====================================================================
# Mechanism: LBL-VS (RP-1218 Eq. 53) -- untripped / lightly tripped only
# =====================================================================


def _g1(e: Array) -> Array:
    le = _log10(e)
    b1 = 39.8 * le - 11.12
    b2 = 98.409 * le + 2.0
    b3 = -5.076 + _sqrt(2.484 - 506.25 * le**2)
    b4 = -98.409 * le + 2.0
    b5 = -39.8 * le - 11.12
    return jnp.where(
        e <= 0.5974,
        b1,
        jnp.where(e <= 0.8545, b2, jnp.where(e <= 1.17, b3, jnp.where(e <= 1.674, b4, b5))),
    )


def _g2(d: Array) -> Array:
    ld = _log10(d)
    b1 = 77.852 * ld + 15.328
    b2 = 65.188 * ld + 9.125
    b3 = -114.052 * ld**2
    b4 = -65.188 * ld + 9.125
    b5 = -77.852 * ld + 15.328
    return jnp.where(
        d <= 0.3237,
        b1,
        jnp.where(d <= 0.5689, b2, jnp.where(d <= 1.7579, b3, jnp.where(d <= 3.0889, b4, b5))),
    )


def lbl_vs_noise(
    freqs: Array,
    U: ArrayLike,
    mach: ArrayLike,
    re_c: ArrayLike,
    delta_p: ArrayLike,
    span: ArrayLike,
    re: ArrayLike,
    dbar_h: ArrayLike,
    alpha_deg: ArrayLike,
) -> Array:
    r"""Laminar-BL vortex-shedding noise (RP-1218 Eq. 53).

    ``SPL = 10 log(delta_p M^5 L D_bar_h / re^2) + G1(St'/St'_peak) +
    G2(Rc/(Rc)0) + G3(alpha*)``. Note this uses the *pressure-side BL
    thickness* ``delta_p`` (not ``delta*``), with ``St' = f delta_p / U``. Only
    physical for untripped / lightly tripped airfoils.

    Args:
        freqs: Band centre frequencies [Hz], shape ``[n_bands]``.
        U: Section inflow speed [m/s].
        mach: Section Mach number.
        re_c: Chord Reynolds number.
        delta_p: Pressure-side BL thickness ``delta_p`` [m].
        span: Segment span ``L`` [m].
        re: Observer distance [m].
        dbar_h: High-frequency directivity.
        alpha_deg: ``|alpha*|`` [deg].

    Returns:
        LBL-VS SPL [dB], shape ``[n_bands]``.
    """
    m = jnp.asarray(mach, dtype=float)
    a = jnp.abs(jnp.asarray(alpha_deg, dtype=float))
    re_c = jnp.asarray(re_c, dtype=float)
    delta_p = jnp.asarray(delta_p, dtype=float)
    U = jnp.maximum(jnp.asarray(U, dtype=float), _TINY)

    st_prime = freqs * delta_p / U
    st1p = jnp.where(
        re_c <= 1.3e5,
        0.18,
        jnp.where(re_c <= 4.0e5, 0.001756 * jnp.power(re_c, 0.3931), 0.28),
    )
    st_peak = st1p * _pow10(-0.04 * a)
    e = st_prime / jnp.maximum(st_peak, _TINY)

    rc0 = jnp.where(a <= 3.0, _pow10(0.215 * a + 4.978), _pow10(0.120 * a + 5.263))
    d = re_c / jnp.maximum(rc0, _TINY)

    g3 = 171.04 - 3.03 * a
    base = 10.0 * (
        _log10(delta_p)
        + 5.0 * _log10(m)
        + _log10(jnp.asarray(span, dtype=float))
        - 2.0 * _log10(jnp.asarray(re, dtype=float))
        + _log10(jnp.asarray(dbar_h))
    )
    return base + _g1(e) + _g2(d) + g3


# =====================================================================
# Mechanism: tip-vortex-formation noise (RP-1218 Eq. 61)
# =====================================================================


def tip_vortex_noise(
    freqs: Array,
    mach: ArrayLike,
    chord: ArrayLike,
    c0: ArrayLike,
    re: ArrayLike,
    dbar_h: ArrayLike,
    tip_alpha_deg: ArrayLike,
    rounded: bool = True,
) -> Array:
    r"""Tip-vortex-formation noise (RP-1218 Eq. 61).

    ``SPL = 10 log(M^2 M_max^3 l^2 D_bar_h / re^2) - 30.5 (log St'' + 0.3)^2
    + 126`` with ``St'' = f l / U_max``, ``M_max/M = 1 + 0.036 alpha'_tip``,
    ``U_max = c0 M_max`` and the spanwise extent ``l`` from the rounded or flat
    tip correlation. (The docs-page ``M_max^2`` is a misprint; report/OpenFAST
    use ``M^2 M_max^3``.)

    Args:
        freqs: Band centre frequencies [Hz], shape ``[n_bands]``.
        mach: Section (tip) Mach number ``M``.
        chord: Tip chord ``c`` [m].
        c0: Speed of sound [m/s].
        re: Observer distance [m].
        dbar_h: High-frequency directivity.
        tip_alpha_deg: Local tip angle of attack ``alpha'_tip`` [deg].
        rounded: ``True`` for a rounded tip, ``False`` for a flat/square tip.

    Returns:
        Tip-noise SPL [dB], shape ``[n_bands]``.
    """
    m = jnp.asarray(mach, dtype=float)
    chord = jnp.asarray(chord, dtype=float)
    c0 = jnp.asarray(c0, dtype=float)
    ap = jnp.abs(jnp.asarray(tip_alpha_deg, dtype=float))

    if rounded:
        l_over_c = 0.008 * ap
    else:
        l_over_c = jnp.where(ap <= 2.0, 0.0230 + 0.0169 * ap, 0.0378 + 0.0095 * ap)
    ell = l_over_c * chord

    m_max = m * (1.0 + 0.036 * ap)
    u_max = c0 * m_max
    st2p = freqs * ell / jnp.maximum(u_max, _TINY)

    base = 10.0 * (
        2.0 * _log10(m)
        + 3.0 * _log10(m_max)
        + 2.0 * _log10(ell)
        - 2.0 * _log10(jnp.asarray(re, dtype=float))
        + _log10(jnp.asarray(dbar_h))
    )
    return base - 30.5 * (_log10(st2p) + 0.3) ** 2 + 126.0


# =====================================================================
# Mechanism: TE bluntness vortex-shedding (RP-1218 Eqs. 70--82)
# =====================================================================


def _g4(h_dstar: Array, psi: Array) -> Array:
    lo = 17.5 * _log10(h_dstar) + 157.5 - 1.114 * psi
    hi = 169.7 - 1.114 * psi
    return jnp.where(h_dstar <= 5.0, lo, hi)


def _g5_psi14(x: Array, eta: Array) -> Array:
    """(G5)_{Psi=14} spectral shape (RP-1218)."""
    mu = jnp.where(
        x < 0.25,
        0.1221,
        jnp.where(
            x <= 0.62,
            -0.2175 * x + 0.1755,
            jnp.where(x <= 1.15, -0.0308 * x + 0.0596, 0.0242),
        ),
    )
    m = jnp.where(
        x <= 0.02,
        0.0,
        jnp.where(
            x <= 0.5,
            68.724 * x - 1.35,
            jnp.where(
                x <= 0.62,
                308.475 * x - 121.23,
                jnp.where(
                    x <= 1.15,
                    224.811 * x - 69.35,
                    jnp.where(x < 1.2, 1583.28 * x - 1631.59, 268.344),
                ),
            ),
        ),
    )
    mu = jnp.maximum(mu, _TINY)
    eta0 = -_sqrt(m**2 * mu**4 / (6.25 + m**2 * mu**2))
    k = 2.5 * _sqrt(1.0 - (eta0 / mu) ** 2) - 2.5 - m * eta0
    b1 = m * eta + k
    b2 = 2.5 * _sqrt(1.0 - (eta / mu) ** 2) - 2.5
    b3 = _sqrt(1.5625 - 1194.99 * eta**2) - 1.25
    b4 = -155.543 * eta + 4.375
    return jnp.where(
        eta < eta0,
        b1,
        jnp.where(eta < 0.0, b2, jnp.where(eta < 0.03616, b3, b4)),
    )


def bluntness_noise(
    freqs: Array,
    U: ArrayLike,
    mach: ArrayLike,
    dstar_avg: ArrayLike,
    span: ArrayLike,
    re: ArrayLike,
    dbar_h: ArrayLike,
    h: ArrayLike,
    psi_deg: ArrayLike = 14.0,
) -> Array:
    r"""Trailing-edge bluntness vortex-shedding noise (RP-1218 Eqs. 70--82).

    ``SPL = 10 log(h M^5.5 L D_bar_h / re^2) + G4(h/dstar_avg, Psi) +
    G5(h/dstar_avg, Psi, St'''/St'''_peak)`` with ``St''' = f h / U`` and
    ``dstar_avg = (delta*_p + delta*_s)/2``. ``G5`` interpolates between the
    ``Psi = 0`` and ``Psi = 14 deg`` shapes (``G5 = (G5)_0 + 0.0714 Psi
    [(G5)_14 - (G5)_0]``), the ``Psi = 0`` shape reusing the ``Psi = 14``
    machinery with ``x -> x' = 6.724 x^2 - 4.019 x + 1.107`` (Eq. 82). The
    OpenFAST caps ``G5 <- min(G5, 0)`` and ``G5 <- min(G5, G5|_{x=0.25})`` are
    applied.

    Args:
        freqs: Band centre frequencies [Hz], shape ``[n_bands]``.
        U: Section inflow speed [m/s].
        mach: Section Mach number.
        dstar_avg: Mean displacement thickness [m].
        span: Segment span ``L`` [m].
        re: Observer distance [m].
        dbar_h: High-frequency directivity.
        h: Trailing-edge thickness ``h`` [m].
        psi_deg: TE solid angle ``Psi`` [deg] (0 flat plate, ~14 NACA 0012).

    Returns:
        Bluntness SPL [dB], shape ``[n_bands]``. Goes to the floor as ``h -> 0``.
    """
    m = jnp.asarray(mach, dtype=float)
    U = jnp.maximum(jnp.asarray(U, dtype=float), _TINY)
    h = jnp.asarray(h, dtype=float)
    psi = jnp.asarray(psi_deg, dtype=float)
    dstar_avg = jnp.maximum(jnp.asarray(dstar_avg, dtype=float), _TINY)

    x = h / dstar_avg  # h/dstar_avg

    st3 = freqs * h / U
    st3_peak = jnp.where(
        x >= 0.2,
        (0.212 - 0.0045 * psi) / (1.0 + 0.235 / x - 0.0132 / x**2),
        0.1 * x + 0.095 - 0.00243 * psi,
    )
    eta = _log10(st3 / jnp.maximum(st3_peak, _TINY))

    def g5_full(xv: Array) -> Array:
        g14 = _g5_psi14(xv, eta)
        xp = 6.724 * xv**2 - 4.019 * xv + 1.107
        g0 = _g5_psi14(xp, eta)
        return g0 + 0.0714 * psi * (g14 - g0)

    g5 = g5_full(x)
    g5_ref = g5_full(jnp.asarray(0.25))
    g5 = jnp.minimum(jnp.minimum(g5, 0.0), g5_ref)

    base = 10.0 * (
        _log10(h)
        + 5.5 * _log10(m)
        + _log10(jnp.asarray(span, dtype=float))
        - 2.0 * _log10(jnp.asarray(re, dtype=float))
        + _log10(jnp.asarray(dbar_h))
    )
    spl = base + _g4(x, psi) + g5
    # Vanish smoothly as h -> 0 (base -> -inf via log h); floor it.
    return jnp.where(h > _TINY, spl, _SPL_FLOOR)


# =====================================================================
# Public assembly
# =====================================================================


class BPMSpectra(NamedTuple):
    """Per-mechanism and total one-third-octave BPM self-noise spectra [dB].

    Each leaf has shape ``[.., n_bands]`` (the leading dims match the broadcast
    of the scalar inputs). Nulled mechanisms sit at the ``-300 dB`` floor.

    Attributes:
        tbl_te: TBL-TE + separation SPL.
        lbl_vs: LBL-VS SPL (floor unless enabled).
        tip: Tip-vortex SPL (floor unless enabled).
        bluntness: TE-bluntness SPL (floor unless enabled).
        total: Energy sum of the active mechanisms.
    """

    tbl_te: Array
    lbl_vs: Array
    tip: Array
    bluntness: Array
    total: Array


def bpm_third_octave(
    freqs: Array,
    U: ArrayLike,
    chord: ArrayLike,
    span: ArrayLike,
    re_c: ArrayLike,
    mach: ArrayLike,
    medium: Medium,
    *,
    alpha_deg: ArrayLike = 0.0,
    theta_e_deg: ArrayLike = 90.0,
    phi_e_deg: ArrayLike = 90.0,
    r_e: ArrayLike = 1.0,
    observer_te: Array | None = None,
    mach_c: ArrayLike | None = None,
    tripped: bool = True,
    include_tbl_te: bool = True,
    include_lbl_vs: bool = False,
    include_tip: bool = False,
    include_bluntness: bool = False,
    h: ArrayLike = 0.0,
    psi_deg: ArrayLike = 14.0,
    tip_rounded: bool = True,
    tip_alpha_deg: ArrayLike | None = None,
    prandtl_glauert: bool = False,
) -> BPMSpectra:
    r"""BPM one-third-octave self-noise spectra for one airfoil section.

    Assembles the requested mechanisms (TBL-TE always by default) into
    per-mechanism and energy-summed total SPL spectra. Fully ``vmap``-able over
    blade sections: pass scalar (0-d) flow/geometry quantities and a 1-D
    ``freqs`` and ``vmap`` over the section axis.

    Observer geometry is given either as TE-frame emission angles
    (``theta_e_deg``, ``phi_e_deg``, ``r_e``) or as a TE-frame observer vector
    ``observer_te`` (which overrides the angles).

    Args:
        freqs: One-third-octave band centres [Hz], shape ``[n_bands]``.
        U: Section inflow speed [m/s].
        chord: Section chord ``c`` [m].
        span: Segment wetted span ``L`` [m].
        re_c: Chord Reynolds number ``Rc`` [-].
        mach: Section Mach number ``M`` [-].
        medium: Ambient medium (uses ``c0`` and ``nu``).
        alpha_deg: Effective angle of attack ``alpha*`` [deg].
        theta_e_deg: Emission polar angle ``Theta_e`` [deg].
        phi_e_deg: Emission azimuth angle ``Phi_e`` [deg].
        r_e: Observer distance ``re`` [m].
        observer_te: Optional TE-frame observer vector ``[.., 3]`` [m]
            (overrides the angle arguments).
        mach_c: Convective Mach number; default ``0.8 M``.
        tripped: Heavily-tripped BL correlations (else untripped).
        include_tbl_te: Include TBL-TE (default ``True``).
        include_lbl_vs: Include LBL-VS (untripped only physically).
        include_tip: Include tip-vortex noise.
        include_bluntness: Include TE-bluntness noise.
        h: Trailing-edge thickness [m] (bluntness).
        psi_deg: TE solid angle ``Psi`` [deg] (bluntness).
        tip_rounded: Rounded (``True``) vs flat tip for tip noise.
        tip_alpha_deg: Local tip AoA [deg]; defaults to ``alpha_deg``.
        prandtl_glauert: Apply the ``1/(1 - M^2)`` compressibility
            amplification to TBL-TE (rotor-application convention; off by
            default so isolated-airfoil anchors match RP-1218).

    Returns:
        A :class:`BPMSpectra` of ``[.., n_bands]`` SPL spectra.
    """
    freqs = jnp.asarray(freqs, dtype=float)
    m = jnp.asarray(mach, dtype=float)
    mc = 0.8 * m if mach_c is None else jnp.asarray(mach_c, dtype=float)
    c0 = medium.c0
    nu = medium.nu

    if observer_te is not None:
        theta_e, phi_e, re = te_frame_angles(jnp.asarray(observer_te, dtype=float))
    else:
        theta_e = jnp.deg2rad(jnp.asarray(theta_e_deg, dtype=float))
        phi_e = jnp.deg2rad(jnp.asarray(phi_e_deg, dtype=float))
        re = jnp.asarray(r_e, dtype=float)

    dbar_h = directivity_high(theta_e, phi_e, m, mc)
    dbar_l = directivity_low(theta_e, phi_e, m)

    bl = boundary_layer_thickness(re_c, alpha_deg, chord, tripped=tripped)

    shape = jnp.broadcast_shapes(jnp.shape(m), jnp.shape(re), freqs.shape)
    floor = jnp.full(shape, _SPL_FLOOR)

    if include_tbl_te:
        tbl = tbl_te_noise(
            freqs, U, m, re_c, bl.dstar_p, bl.dstar_s, span, re, dbar_h, dbar_l, alpha_deg, nu
        )
        if prandtl_glauert:
            tbl = tbl - 10.0 * _log10(1.0 - jnp.minimum(m**2, 0.999999))
    else:
        tbl = floor

    if include_lbl_vs:
        lbl = lbl_vs_noise(freqs, U, m, re_c, bl.delta_p, span, re, dbar_h, alpha_deg)
    else:
        lbl = floor

    if include_tip:
        ta = alpha_deg if tip_alpha_deg is None else tip_alpha_deg
        tip = tip_vortex_noise(freqs, m, chord, c0, re, dbar_h, ta, rounded=tip_rounded)
    else:
        tip = floor

    if include_bluntness:
        blunt = bluntness_noise(freqs, U, m, bl.dstar_avg, span, re, dbar_h, h, psi_deg)
    else:
        blunt = floor

    tbl, lbl, tip, blunt = jnp.broadcast_arrays(tbl, lbl, tip, blunt)
    total = 10.0 * _log10(
        _pow10(tbl / 10.0) + _pow10(lbl / 10.0) + _pow10(tip / 10.0) + _pow10(blunt / 10.0)
    )
    return BPMSpectra(tbl, lbl, tip, blunt, total)
