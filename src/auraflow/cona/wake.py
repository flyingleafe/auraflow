r"""Beddoes prescribed-wake induced velocity (CONA aerodynamics).

Reconstructs the CONA prescribed-wake sub-model
(``docs/research/cona-reference.md`` "Beddoes prescribed wake" and
``docs/research/cona-external-formulations.md`` §3): a rotor's trailed tip
vortices are laid out on a *prescribed* helical geometry parameterized by the
wake age ``zeta`` (the azimuth swept since the vortex left the blade), given a
Lamb-Oseen swirl distribution with a growing viscous core, and the induced
velocity at arbitrary evaluation points is recovered by summing the
Biot-Savart contributions of the straight filament segments joining the wake
nodes.

Pieces
------
- :func:`vortex_circulation` -- root/bound circulation ``Gamma_v`` from the
  thrust coefficient (``Gamma_v = 2 pi C_T Omega R^2 / b``) with an optional
  first-harmonic azimuthal modulation ``(gamma0, gamma1s, gamma1c)``.
- :func:`core_radius` -- Squire / Bhagwat-Leishman viscous core growth
  ``r_c(zeta) = sqrt(r_c0^2 + 4 alpha_L delta nu zeta / Omega)``.
- :func:`lamb_oseen_swirl` -- the analytic swirl profile
  ``V_theta(r) = Gamma/(2 pi r)(1 - exp(-alpha r^2 / r_c^2))`` (peak at
  ``r = r_c``, ``-> Gamma/(2 pi r)`` far field).
- :func:`biot_savart_segment` -- desingularized induced velocity of a straight
  vortex segment (van Garrel cut-off; no singularity, grad-safe).
- :func:`beddoes_wake_nodes` -- the prescribed helical node positions (rotor
  frame) for every blade over ``n_rev`` revolutions of wake age.
- :class:`PrescribedWake` -- bundles the wake geometry + per-segment
  circulation and evaluates the induced velocity at query points
  (:meth:`PrescribedWake.induced_velocity`) or on an ``(r, psi)`` disk grid
  (:meth:`PrescribedWake.inflow_grid`).

Frames & units follow ``docs/architecture.md``: rotor frame, ``+z`` the thrust
axis, azimuth from ``+x`` toward ``+y``. SI throughout (m, s, rad); everything
is float64-safe and differentiable (piecewise wake branches use ``jnp.where``,
the Biot-Savart core uses a smooth cut-off -- no gradient-killing clamps).

Scaling note
------------
The Biot-Savart sum is ``O(P * b * N_age)`` for ``P`` query points, ``b``
blades and ``N_age`` wake-age segments per blade. Keep ``n_azimuth`` (segments
per revolution) and ``n_rev`` modest -- the default ``n_azimuth=24``,
``n_rev=4`` gives ~96 segments per blade, which resolves the near wake that
dominates the disk inflow without exploding the cost. Prefer the cached
:meth:`PrescribedWake.inflow_grid` (built once, interpolated) inside the
airloads march rather than re-evaluating the full sum at every time step.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from auraflow.core.medium import Medium

__all__ = [
    "PrescribedWake",
    "beddoes_wake_nodes",
    "biot_savart_segment",
    "core_radius",
    "lamb_oseen_swirl",
    "vortex_circulation",
]

# Lamb-Oseen constant: swirl peaks at r = r_c for this value.
_ALPHA_L = 1.25643
_EPS = 1.0e-12


def vortex_circulation(
    ct: ArrayLike,
    omega: ArrayLike,
    radius: ArrayLike,
    n_blades: int,
    psi_release: ArrayLike = 0.0,
    gamma0: ArrayLike = 1.0,
    gamma1s: ArrayLike = 0.0,
    gamma1c: ArrayLike = 0.0,
) -> Array:
    r"""Trailed tip-vortex circulation from the thrust coefficient.

    Per ``docs/research/cona-external-formulations.md`` §3: the mean bound/root
    circulation is ``Gamma_bar0 = 2 pi C_T / b`` (nondimensional), giving the
    dimensional trailed circulation

    .. math::
        \Gamma_v = \frac{2 \pi C_T \Omega R^2}{b}
            \, (\gamma_0 + \gamma_{1s} \sin\psi_v + \gamma_{1c} \cos\psi_v),

    with ``b`` the blade count and the optional first-harmonic modulation
    ``(gamma0, gamma1s, gamma1c)`` (``gamma0 = 1`` nominal) capturing the
    advancing/retreating loading asymmetry in forward flight.

    Args:
        ct: Rotor thrust coefficient ``C_T = T/(rho A (Omega R)^2)`` [-].
        omega: Rotor speed magnitude ``|Omega|`` [rad/s].
        radius: Rotor tip radius ``R`` [m].
        n_blades: Blade count ``b`` (static int).
        psi_release: Release azimuth ``psi_v`` [rad], any shape (for the
            modulation term).
        gamma0: Mean modulation factor [-].
        gamma1s: Sine (lateral) modulation coefficient [-].
        gamma1c: Cosine (longitudinal) modulation coefficient [-].

    Returns:
        Trailed circulation ``Gamma_v`` [m^2/s], broadcasting ``psi_release``.
    """
    ct = jnp.asarray(ct)
    omega = jnp.asarray(omega)
    radius = jnp.asarray(radius)
    psi = jnp.asarray(psi_release)
    gamma_mean = 2.0 * jnp.pi * ct * omega * radius**2 / n_blades
    modulation = gamma0 + gamma1s * jnp.sin(psi) + gamma1c * jnp.cos(psi)
    return gamma_mean * modulation


def core_radius(
    zeta: ArrayLike,
    omega: ArrayLike,
    gamma_v: ArrayLike,
    medium: Medium,
    r_c0: ArrayLike,
    a1: ArrayLike = 2.0e-4,
) -> Array:
    r"""Squire / Bhagwat-Leishman viscous tip-vortex core growth.

    .. math::
        r_c(\zeta) = \sqrt{r_{c0}^2 + 4 \alpha_L \delta \nu \zeta / \Omega},
        \qquad \delta = 1 + a_1 \mathrm{Re}_v, \quad \mathrm{Re}_v = |\Gamma_v|/\nu

    (``docs/research/cona-external-formulations.md`` §3), with ``alpha_L =
    1.25643``. ``zeta/Omega`` is the physical age (time since the element left
    the blade). ``delta`` is the eddy-viscosity amplification (``~3-20`` for
    model rotors); ``a1`` is tunable (literature ``6.5e-5 .. 2e-4``).

    Args:
        zeta: Wake age [rad], any shape.
        omega: Rotor speed magnitude ``|Omega|`` [rad/s], scalar.
        gamma_v: Trailed circulation ``Gamma_v`` [m^2/s] (sets ``Re_v``).
        medium: Ambient medium (supplies ``nu``).
        r_c0: Initial core radius ``r_c0`` [m] (``~0.05c .. 0.25c``).
        a1: Eddy-viscosity constant ``a1`` [-].

    Returns:
        Core radius ``r_c(zeta)`` [m], same broadcast shape as ``zeta``.
    """
    zeta = jnp.asarray(zeta)
    omega = jnp.abs(jnp.asarray(omega)) + _EPS
    re_v = jnp.abs(jnp.asarray(gamma_v)) / medium.nu
    delta = 1.0 + jnp.asarray(a1) * re_v
    growth = 4.0 * _ALPHA_L * delta * medium.nu * jnp.abs(zeta) / omega
    return jnp.sqrt(jnp.asarray(r_c0) ** 2 + growth)


def lamb_oseen_swirl(r: ArrayLike, gamma_v: ArrayLike, r_c: ArrayLike) -> Array:
    r"""Lamb-Oseen swirl velocity at radial distance ``r`` from the vortex axis.

    ``V_theta(r) = Gamma_v/(2 pi r) (1 - exp(-alpha_L r^2 / r_c^2))``
    (``docs/research/cona-external-formulations.md`` §3): a rigid-body core that
    peaks at ``r = r_c`` and relaxes to the potential swirl ``Gamma_v/(2 pi r)``
    far from the core. Evaluated in the ``r -> 0`` limit safely (the series
    ``(1 - e^{-x})/r -> alpha_L r / r_c^2`` is finite).

    Args:
        r: Radial distance from the vortex axis [m], any shape ``>= 0``.
        gamma_v: Circulation ``Gamma_v`` [m^2/s].
        r_c: Core radius ``r_c`` [m].

    Returns:
        Swirl speed ``V_theta`` [m/s], same broadcast shape as ``r``.
    """
    r = jnp.asarray(r)
    r_c = jnp.asarray(r_c)
    r_safe = jnp.hypot(r, _EPS)
    return (
        jnp.asarray(gamma_v) / (2.0 * jnp.pi * r_safe) * (1.0 - jnp.exp(-_ALPHA_L * r**2 / r_c**2))
    )


def biot_savart_segment(p: Array, a: Array, b: Array, gamma: ArrayLike, r_c: ArrayLike) -> Array:
    r"""Desingularized Biot-Savart velocity of a straight vortex segment.

    Induced velocity at point ``p`` from a straight vortex filament of
    circulation ``gamma`` running from ``a`` to ``b``:

    .. math::
        u = \frac{\Gamma}{4\pi}\,
            \frac{(r_1 \times r_2)}{|r_1 \times r_2|^2 + (r_c |r_0|)^2}\,
            \Big(r_0\cdot\big(\tfrac{r_1}{|r_1|} - \tfrac{r_2}{|r_2|}\big)\Big),

    with ``r_1 = p - a``, ``r_2 = p - b``, ``r_0 = b - a`` (van Garrel /
    Bhagwat cut-off core: the ``(r_c |r_0|)^2`` term removes the ``1/d``
    singularity on the filament axis, so the field is smooth and the gradient
    is finite everywhere). For ``d >> r_c`` this reproduces the potential
    result, and a long segment recovers the infinite-line ``Gamma/(2 pi d)``.

    Args:
        p: Evaluation point [m], shape ``[3]``.
        a: Segment start node [m], shape ``[3]``.
        b: Segment end node [m], shape ``[3]``.
        gamma: Segment circulation [m^2/s], scalar.
        r_c: Core radius [m], scalar.

    Returns:
        Induced velocity [m/s], shape ``[3]``.
    """
    gamma = jnp.asarray(gamma)
    r_c = jnp.asarray(r_c)
    r1 = p - a
    r2 = p - b
    r0 = b - a
    cross = jnp.cross(r1, r2)
    cross_sq = jnp.sum(cross**2)
    n1 = jnp.linalg.norm(r1) + _EPS
    n2 = jnp.linalg.norm(r2) + _EPS
    r0_norm_sq = jnp.sum(r0**2)
    scalar = jnp.sum(r0 * (r1 / n1 - r2 / n2))
    denom = cross_sq + r_c**2 * r0_norm_sq + _EPS
    return gamma / (4.0 * jnp.pi) * cross * scalar / denom


def _skew_angle(mu_x: Array, mu_z: Array, lam_i: Array) -> Array:
    """Wake skew angle chi = atan2(mu_x, mu_z + lam_i) (0 in hover)."""
    return jnp.arctan2(jnp.abs(mu_x), jnp.abs(mu_z) + lam_i + _EPS)


def beddoes_wake_nodes(
    psi_blades: Array,
    zeta: Array,
    mu_x: Array,
    mu_z: Array,
    lam_i: Array,
    radius: Array,
    r_v: Array,
    w0: Array,
    ws: Array,
    wc: Array,
) -> Array:
    r"""Prescribed Beddoes tip-vortex node positions (rotor frame, metres).

    For a blade whose tip is currently at azimuth ``psi_b`` and a wake age
    ``zeta`` (radians swept since release), the vortex element was released at
    azimuth ``psi_v = psi_b - zeta`` and, per the parameterization in
    ``docs/research/cona-reference.md`` (nondimensionalized by ``R``):

    .. math::
        \bar x_v &= \bar r_v \cos\psi_v + \mu_x\,\zeta \\
        \bar y_v &= \bar r_v \sin\psi_v \\
        \bar z_v &= -\mu_z\,\zeta + \text{(piecewise settling)}

    with the three-branch axial settling (``w0, ws, wc`` tuning, ``chi`` the
    wake skew):

    - branch 1 (``x_v < -r_v cos psi_v``):
      ``-lam_i [w0 - ws mu_x y_v + wc chi (cos psi_v + mu_x zeta/(2 r_v)
      - |y_v^3|)] zeta``
    - branch 2 (``cos psi_v > 0``):
      ``-2 lam_i (w0 - ws mu_x y_v - wc chi |y_v^3|) zeta``
    - branch 3 (otherwise): the forward-flight far-wake plateau
      ``-2 lam_i (w0 - ws mu_x y_v - wc chi |y_v^3|)/mu_x``.

    Branch 3 diverges as ``mu_x -> 0``; near hover it is replaced by the
    branch-2 (``propto zeta``) form so the descent stays finite (documented
    deviation -- the plateau is a forward-flight construct). All positions are
    returned dimensionally (metres) after multiplying by ``R``.

    Args:
        psi_blades: Current tip azimuths of each blade [rad], shape ``[B]``.
        zeta: Wake-age nodes [rad], shape ``[N]`` (increasing, ``>= 0``).
        mu_x: In-plane advance ratio ``V cos a_p / (Omega R)`` [-], scalar.
        mu_z: Axial advance ratio (climb / disk-normal inflow) [-], scalar.
        lam_i: Mean induced inflow ratio ``lambda_i`` [-], scalar.
        radius: Tip radius ``R`` [m], scalar.
        r_v: Trailed-vortex radius (fraction of ``R``) ``bar r_v`` [-], scalar.
        w0, ws, wc: Beddoes settling parameters [-], scalars.

    Returns:
        Node positions in the rotor frame [m], shape ``[B, N, 3]``.
    """
    psi_v = psi_blades[:, None] - zeta[None, :]  # [B, N]
    cos_v = jnp.cos(psi_v)
    sin_v = jnp.sin(psi_v)
    x_v = r_v * cos_v + mu_x * zeta[None, :]
    y_v = r_v * sin_v
    chi = _skew_angle(mu_x, mu_z, lam_i)

    y3 = jnp.abs(y_v**3)
    mu_x_safe = jnp.where(jnp.abs(mu_x) < 1.0e-3, 1.0e-3, mu_x)
    b1 = (
        -lam_i
        * (w0 - ws * mu_x * y_v + wc * chi * (cos_v + mu_x * zeta[None, :] / (2.0 * r_v) - y3))
        * zeta[None, :]
    )
    b2 = -2.0 * lam_i * (w0 - ws * mu_x * y_v - wc * chi * y3) * zeta[None, :]
    b3 = -2.0 * lam_i * (w0 - ws * mu_x * y_v - wc * chi * y3) / mu_x_safe

    cond1 = x_v < -r_v * cos_v
    cond2 = cos_v > 0.0
    z_fwd = jnp.where(cond1, b1, jnp.where(cond2, b2, b3))
    z_hover = jnp.where(cond1, b1, b2)  # branch-3 -> branch-2 near hover
    z_settle = jnp.where(jnp.abs(mu_x) < 1.0e-3, z_hover, z_fwd)

    z_v = -mu_z * zeta[None, :] + z_settle
    nodes = jnp.stack([x_v, y_v, z_v], axis=-1)  # [B, N, 3] nondim
    return nodes * radius


class PrescribedWake(eqx.Module):
    """Prescribed Beddoes tip-vortex wake and its Biot-Savart induced field.

    Holds the straight-filament segment endpoints (rotor frame), per-segment
    circulation, and core radii; :meth:`induced_velocity` sums the
    desingularized Biot-Savart contributions of every segment at query points.

    Attributes:
        seg_a: Segment start nodes [m], shape ``[F, 3]`` (``F`` filaments).
        seg_b: Segment end nodes [m], shape ``[F, 3]``.
        gamma: Per-segment circulation [m^2/s], shape ``[F]``.
        r_c: Per-segment core radius [m], shape ``[F]``.
        radius: Rotor tip radius [m] (for convenience/grid extents).
        lam0: Reference (mean) induced inflow ratio [-] used to build the wake.
    """

    seg_a: Array
    seg_b: Array
    gamma: Array
    r_c: Array
    radius: Array
    lam0: Array

    def induced_velocity(self, points: ArrayLike) -> Array:
        """Induced velocity at query points from the whole wake.

        Args:
            points: Evaluation points in the rotor frame [m], shape ``[P, 3]``.

        Returns:
            Induced velocity [m/s], shape ``[P, 3]`` (``+z`` is the thrust
            axis; the disk sees a downward, i.e. ``-z``, induced velocity for a
            thrusting rotor).
        """
        points = jnp.asarray(points)

        def at_point(p: Array) -> Array:
            contrib = jax.vmap(biot_savart_segment, in_axes=(None, 0, 0, 0, 0))(
                p, self.seg_a, self.seg_b, self.gamma, self.r_c
            )
            return jnp.sum(contrib, axis=0)

        return jax.vmap(at_point)(points)

    def inflow_grid(self, r_grid: ArrayLike, psi_grid: ArrayLike) -> Array:
        r"""Axial induced velocity ``u_z(r, psi)`` on a disk grid.

        Evaluates the induced velocity at disk points ``(r cos psi, r sin psi,
        0)`` and returns its thrust-axis component ``u_z`` [m/s] (negative =
        downwash below a thrusting disk). Divide by the tip speed ``Omega R``
        for the inflow ratio ``lambda = -u_z/(Omega R)``. Intended to be built
        once and interpolated inside the airloads march.

        Args:
            r_grid: Radial query stations [m], shape ``[Nr]``.
            psi_grid: Azimuth query angles [rad], shape ``[Npsi]``.

        Returns:
            Axial induced velocity ``u_z`` [m/s] at each ``(r, psi)``, shape
            ``[Nr, Npsi]`` (thrust-axis component; negative below the disk).
        """
        r_grid = jnp.asarray(r_grid)
        psi_grid = jnp.asarray(psi_grid)
        rr, pp = jnp.meshgrid(r_grid, psi_grid, indexing="ij")  # [Nr, Npsi]
        pts = jnp.stack([rr * jnp.cos(pp), rr * jnp.sin(pp), jnp.zeros_like(rr)], axis=-1).reshape(
            -1, 3
        )
        uz = self.induced_velocity(pts)[:, 2]
        return uz.reshape(rr.shape)


def make_prescribed_wake(
    ct: ArrayLike,
    omega: ArrayLike,
    radius: ArrayLike,
    n_blades: int,
    medium: Medium,
    mu_x: ArrayLike = 0.0,
    mu_z: ArrayLike = 0.0,
    spin: int = 1,
    psi0: ArrayLike = 0.0,
    n_azimuth: int = 24,
    n_rev: int = 4,
    r_v: ArrayLike = 1.0,
    chord_ref: ArrayLike | None = None,
    r_c0: ArrayLike | None = None,
    a1: ArrayLike = 2.0e-4,
    w0: ArrayLike = 1.0,
    ws: ArrayLike = 0.0,
    wc: ArrayLike = 0.5,
    gamma0: ArrayLike = 1.0,
    gamma1s: ArrayLike = 0.0,
    gamma1c: ArrayLike = 0.0,
) -> PrescribedWake:
    r"""Build a :class:`PrescribedWake` from a rotor operating point.

    Lays ``n_blades`` tip-vortex helices over ``n_rev`` revolutions of wake age
    (``n_azimuth`` segments per revolution), assigns each the Lamb-Oseen
    circulation :func:`vortex_circulation` (with first-harmonic modulation) and
    the growing viscous core :func:`core_radius`, and stores the straight
    segments for Biot-Savart evaluation. The mean induced inflow driving the
    prescribed geometry is the momentum estimate ``lambda_i = sqrt(|C_T|/2)``
    (hover) blended with the axial advance ``mu_z``.

    Args:
        ct: Rotor thrust coefficient ``C_T`` [-].
        omega: Rotor speed magnitude ``|Omega|`` [rad/s].
        radius: Tip radius ``R`` [m].
        n_blades: Blade count ``b`` (static int).
        medium: Ambient medium.
        mu_x: In-plane advance ratio [-].
        mu_z: Axial advance ratio (climb/inflow, ``+`` downward) [-].
        spin: Rotation sense ``+1`` (CCW from ``+z``) or ``-1`` (static int).
        psi0: Reference azimuth of blade 0 [rad].
        n_azimuth: Wake-age segments per revolution (static int).
        n_rev: Revolutions of wake age retained (static int).
        r_v: Trailed-vortex radius as a fraction of ``R`` [-].
        chord_ref: Reference chord [m] to set ``r_c0 = 0.15 chord_ref`` when
            ``r_c0`` is not given. Defaults to ``0.1 R`` if both are ``None``.
        r_c0: Initial core radius [m] (overrides ``chord_ref``).
        a1: Core-growth eddy-viscosity constant [-].
        w0, ws, wc: Beddoes settling parameters [-].
        gamma0, gamma1s, gamma1c: Circulation modulation coefficients [-].

    Returns:
        A :class:`PrescribedWake` ready for induced-velocity evaluation.
    """
    ct = jnp.asarray(ct, dtype=float)
    omega = jnp.asarray(omega, dtype=float)
    radius = jnp.asarray(radius, dtype=float)
    mu_x = jnp.asarray(mu_x, dtype=float)
    mu_z = jnp.asarray(mu_z, dtype=float)
    r_v = jnp.asarray(r_v, dtype=float)

    if r_c0 is None:
        c_ref = 0.1 * radius if chord_ref is None else jnp.asarray(chord_ref, dtype=float)
        r_c0 = 0.15 * c_ref
    r_c0 = jnp.asarray(r_c0, dtype=float)

    lam_i = jnp.sqrt(jnp.abs(ct) / 2.0)  # momentum hover inflow ratio

    # Blade tip azimuths (equally spaced along the rotation sense).
    offsets = spin * 2.0 * jnp.pi * jnp.arange(n_blades) / n_blades
    psi_blades = jnp.asarray(psi0, dtype=float) + offsets  # [B]

    # Wake-age node grid: node midpoints define segment endpoints.
    n_nodes = n_azimuth * n_rev + 1
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi * n_rev, n_nodes)  # [N]

    nodes = beddoes_wake_nodes(
        psi_blades,
        zeta,
        mu_x,
        mu_z,
        lam_i,
        radius,
        r_v,
        jnp.asarray(w0, dtype=float),
        jnp.asarray(ws, dtype=float),
        jnp.asarray(wc, dtype=float),
    )  # [B, N, 3]

    seg_a = nodes[:, :-1, :].reshape(-1, 3)  # [B*(N-1), 3]
    seg_b = nodes[:, 1:, :].reshape(-1, 3)

    # Per-segment circulation from the release azimuth (midpoint of the pair).
    psi_rel = psi_blades[:, None] - 0.5 * (zeta[:-1] + zeta[1:])[None, :]  # [B, N-1]
    gamma = vortex_circulation(
        ct, omega, radius, n_blades, psi_rel, gamma0, gamma1s, gamma1c
    ).reshape(-1)
    # Sign: a thrusting rotor's tip vortex induces downwash at the disk. The
    # sense follows the spin; fold it into the circulation sign.
    gamma = spin * gamma

    zeta_mid = 0.5 * (zeta[:-1] + zeta[1:])  # [N-1]
    r_c_seg = core_radius(
        jnp.broadcast_to(zeta_mid[None, :], (n_blades, zeta_mid.shape[0])).reshape(-1),
        omega,
        jnp.mean(jnp.abs(gamma)) + _EPS,
        medium,
        r_c0,
        a1,
    )

    return PrescribedWake(
        seg_a=seg_a, seg_b=seg_b, gamma=gamma, r_c=r_c_seg, radius=radius, lam0=lam_i
    )


__all__.append("make_prescribed_wake")
