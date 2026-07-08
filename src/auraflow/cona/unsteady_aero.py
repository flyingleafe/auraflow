r"""Attached-flow unsteady thin-airfoil aerodynamics (CONA deficiency march).

Reconstructs the CONA unsteady-load correction
(``docs/research/cona-reference.md`` "Unsteady corrections" and
``docs/research/cona-external-formulations.md`` §2): the indicial (Wagner)
response to a time-varying three-quarter-chord downwash, marched with the
van der Wall & Leishman deficiency-function recurrence, plus the
non-circulatory apparent-mass term.

Downwash and effective angle
----------------------------
The circulatory lift lags the quasi-steady value because the shed wake takes
time to develop. Marching the *downwash* ``w(s) = V(s) alpha(s)`` (the digest's
`` h_dot = alpha_dot = 0`` variable-velocity simplification) through the
deficiency functions ``X, Y`` gives the effective downwash

.. math::
    w_E(s) = w(s) - X(s) - Y(s),

and the circulatory lift per unit span

.. math::
    L_C = \tfrac12 \rho\, c\, C_{L\alpha}\, V(s)\, \alpha_E(s),
    \qquad \alpha_E = \alpha - (X + Y)/V

(see :func:`unsteady_lift` for the exact grouping and the apparent-mass term).

Mid-point deficiency recurrence (Leishman "Algorithm D", 2nd-order, in
semichords ``s`` with ``Delta s_n = (V_n + V_{n-1}) Delta t / c``):

.. math::
    X_n &= X_{n-1} e^{-b_1 \beta^2 \Delta s_n}
        + A_1 \Delta w_n e^{-b_1 \beta^2 \Delta s_n / 2} \\
    Y_n &= Y_{n-1} e^{-b_2 \beta^2 \Delta s_n}
        + A_2 \Delta w_n e^{-b_2 \beta^2 \Delta s_n / 2}

with Jones coefficients ``A1=0.165, b1=0.0455, A2=0.335, b2=0.3``,
``Delta w_n = w_n - w_{n-1}``, compressibility exponent scaling by
``beta^2 = 1 - M^2``, and ``X_0 = Y_0 = 0``. This is exactly the recurrence the
digest flags for validation against the Wagner step response
``phi(s) = 1 - A1 e^{-b1 s} - A2 e^{-b2 s}`` (see :func:`wagner_function` and
``tests/cona/test_unsteady_aero.py``).

Everything is float64-safe, ``lax.scan`` over time and ``vmap`` over
(blade, station); no gradient-killing clamps.
"""

from typing import cast

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

__all__ = [
    "JONES_COEFFS",
    "deficiency_march",
    "effective_aoa",
    "unsteady_lift",
    "wagner_function",
]

#: Jones two-term approximation of the Wagner function ``(A1, b1, A2, b2)``.
JONES_COEFFS: tuple[float, float, float, float] = (0.165, 0.0455, 0.335, 0.3)


def wagner_function(s: ArrayLike) -> Array:
    r"""Jones two-term approximation of the Wagner indicial function.

    ``phi(s) = 1 - A1 e^{-b1 s} - A2 e^{-b2 s}`` with the Jones coefficients
    :data:`JONES_COEFFS`. ``phi(0) = 1 - A1 - A2 = 0.5`` and ``phi -> 1`` as
    ``s -> inf``: the fraction of the steady circulatory lift developed ``s``
    semichords after a step change in angle of attack.

    Args:
        s: Distance travelled in semichords [-], any shape ``>= 0``.

    Returns:
        ``phi(s)`` [-], same shape as ``s``.
    """
    a1, b1, a2, b2 = JONES_COEFFS
    s = jnp.asarray(s)
    return 1.0 - a1 * jnp.exp(-b1 * s) - a2 * jnp.exp(-b2 * s)


def deficiency_march(w: Array, ds: Array, beta2: ArrayLike = 1.0) -> tuple[Array, Array]:
    r"""March the van der Wall & Leishman deficiency functions ``X, Y``.

    Mid-point ("Algorithm D") recurrence over the time axis (see the module
    docstring). ``X_0 = Y_0 = 0``; the first increment is ``Delta w_0 = w_0``
    (step onto the initial downwash from rest), so a downwash that is constant
    from ``n = 0`` produces the Wagner build-up.

    Args:
        w: Three-quarter-chord downwash history ``w = V alpha`` [m/s], shape
            ``[T]`` (time along axis 0).
        ds: Per-step semichord increments ``Delta s_n`` [-], shape ``[T]``
            (``ds[0]`` is unused for the carry but kept for shape symmetry).
        beta2: Compressibility factor ``beta^2 = 1 - M^2`` [-], scalar or
            broadcastable with ``w``.

    Returns:
        ``(X, Y)`` deficiency-function histories [m/s], each shape ``[T]``.
    """
    a1, b1, a2, b2 = JONES_COEFFS
    w = jnp.asarray(w)
    ds = jnp.asarray(ds)
    beta2 = jnp.broadcast_to(jnp.asarray(beta2, dtype=w.dtype), w.shape)

    dw = jnp.concatenate([w[:1], jnp.diff(w)])  # Delta w_n, with Delta w_0 = w_0

    def step(
        carry: tuple[Array, Array], inp: tuple[Array, Array, Array]
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        x_prev, y_prev = carry
        dw_n, ds_n, beta2_n = inp
        e1 = jnp.exp(-b1 * beta2_n * ds_n)
        e2 = jnp.exp(-b2 * beta2_n * ds_n)
        x = x_prev * e1 + a1 * dw_n * jnp.sqrt(e1)
        y = y_prev * e2 + a2 * dw_n * jnp.sqrt(e2)
        return (x, y), (x, y)

    z = jnp.zeros((), dtype=w.dtype)
    _, (x_hist, y_hist) = jax.lax.scan(step, (z, z), (dw, ds, beta2))
    return x_hist, y_hist


def effective_aoa(
    v: Array,
    alpha: Array,
    dt: ArrayLike,
    chord: ArrayLike,
    mach: Array | None = None,
) -> Array:
    r"""Wagner-lagged effective angle of attack ``alpha_E = alpha - (X+Y)/V``.

    Marches the deficiency functions on the downwash ``w = V alpha`` and returns
    the effective (lagged) angle of attack. Unlike the lift, ``alpha_E`` is
    independent of the lift-curve slope, so a caller can feed ``alpha_E`` back
    into any polar (with its own stall / compressibility model) to obtain the
    unsteady circulatory coefficients -- the route :mod:`auraflow.cona.airloads`
    takes.

    Args:
        v: Resultant section speed history ``V`` [m/s], shape ``[T]``.
        alpha: Quasi-steady angle-of-attack history [rad], shape ``[T]``.
        dt: Time-step [s], scalar.
        chord: Section chord [m], scalar.
        mach: Optional section Mach history [-], shape ``[T]``.

    Returns:
        Effective angle of attack ``alpha_E`` [rad], shape ``[T]``.
    """
    v = jnp.asarray(v)
    alpha = jnp.asarray(alpha)
    w = v * alpha
    v_prev = jnp.concatenate([v[:1], v[:-1]])
    ds = (v + v_prev) * jnp.asarray(dt) / jnp.asarray(chord)
    beta2 = 1.0 if mach is None else jnp.clip(1.0 - jnp.asarray(mach) ** 2, 1.0e-6, 1.0)
    x_hist, y_hist = deficiency_march(w, ds, beta2)
    v_safe = jnp.where(jnp.abs(v) < 1.0e-9, 1.0e-9, v)
    return alpha - (x_hist + y_hist) / v_safe


def unsteady_lift(
    v: Array,
    alpha: Array,
    dt: ArrayLike,
    chord: ArrayLike,
    rho: ArrayLike,
    cl_alpha: ArrayLike = 2.0 * jnp.pi,
    mach: Array | None = None,
) -> tuple[Array, Array, Array]:
    r"""Attached-flow unsteady sectional lift (circulatory + apparent mass).

    Marches the downwash ``w = V alpha`` through :func:`deficiency_march` and
    forms, per ``docs/research/cona-external-formulations.md`` §2 and CONA
    Eq. 9,

    .. math::
        L_C  &= \tfrac12 \rho\, c\, C_{L\alpha}\, V\, w_E / V
             = \tfrac12 \rho\, c\, C_{L\alpha}\, V(s)\,(\alpha - (X+Y)/V), \\
        L_{NC} &= \tfrac12 \rho\, C_{L\alpha}\, c^2\, \dot V\, \alpha,

    i.e. the effective angle of attack is ``alpha_E = alpha - (X + Y)/V`` and
    the circulatory lift uses the Wagner-lagged ``alpha_E`` in place of the
    quasi-steady ``alpha``. The apparent-mass term ``L_NC`` is the
    variable-velocity contribution (the digest's ``h_dot = alpha_dot = 0``
    reduction). ``s`` accumulates as ``Delta s_n = (V_n + V_{n-1}) Delta t /
    c``.

    Args:
        v: Resultant section speed history ``V`` [m/s], shape ``[T]``.
        alpha: Quasi-steady angle-of-attack history [rad], shape ``[T]``.
        dt: Time-step [s], scalar (uniform grid).
        chord: Section chord ``c`` [m], scalar.
        rho: Ambient density [kg/m^3], scalar.
        cl_alpha: Lift-curve slope ``C_{L alpha}`` [1/rad], scalar.
        mach: Optional section Mach history [-], shape ``[T]``; sets the
            compressibility factor ``beta^2 = 1 - M^2`` in the recurrence.

    Returns:
        ``(lift, alpha_eff, lift_circ)``: total lift per span ``L_C + L_NC``
        [N/m], effective angle of attack ``alpha_E`` [rad], and the circulatory
        part ``L_C`` [N/m] -- each shape ``[T]``.
    """
    v = jnp.asarray(v)
    alpha = jnp.asarray(alpha)
    dt = jnp.asarray(dt)
    chord = jnp.asarray(chord)
    rho = jnp.asarray(rho)
    cl_alpha = jnp.asarray(cl_alpha)

    w = v * alpha  # three-quarter-chord downwash
    v_prev = jnp.concatenate([v[:1], v[:-1]])
    ds = (v + v_prev) * dt / chord  # semichord increments
    beta2 = 1.0 if mach is None else jnp.clip(1.0 - jnp.asarray(mach) ** 2, 1.0e-6, 1.0)

    x_hist, y_hist = deficiency_march(w, ds, beta2)
    v_safe = jnp.where(jnp.abs(v) < 1.0e-9, 1.0e-9, v)
    alpha_eff = alpha - (x_hist + y_hist) / v_safe

    lift_circ = 0.5 * rho * chord * cl_alpha * v * alpha_eff
    v_dot = cast(Array, jnp.gradient(v, dt))
    lift_nc = 0.5 * rho * cl_alpha * chord**2 * v_dot * alpha
    return lift_circ + lift_nc, alpha_eff, lift_circ
