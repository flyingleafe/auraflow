r"""NASA 1-Pax UAM quadrotor: the single source of truth for the JASA vehicle.

Reconstructs the vehicle used by the JASA data-generation paper (Lee, Ko,
Seshadri, Rauleder, JASA 159(4):3418-3435, 2026) from the digests
``docs/research/nasa-1pax-vehicle.md`` and ``docs/research/jasa-datagen-reference.md``.
The vehicle is the rotor-speed-controlled 1-Pax variant of Malpica &
Withrow-Maser (VFS TVF 2020), itself a resize of Johnson/Silva/Solis 2018.

Single source of truth
----------------------
Mass / inertia / rotor-allocation (hub positions, spin senses, ``k_f``,
``c_tauf``) live in :meth:`auraflow.cona.flight.Multirotor.nasa_1pax`; this
module *delegates* to it (:func:`nasa_1pax_multirotor`) and *reads back* the hub
positions and spin senses when it places the geometric rotors
(:func:`nasa_1pax_vehicle`) -- no vehicle constant is written down twice. What
lives *here* is the blade geometry (radius, twist, taper, chord reconstruction,
airfoil polar) that the flight module has no need for.

Published vs reconstructed (digest ``nasa-1pax-vehicle.md`` sect. "Blade geometry")
----------------------------------------------------------------------------------
Published: gross weight 583.85 kg, 4 rotors x 3 blades, tip radius
``R = 1.951 m``, thrust-weighted solidity ``sigma = 0.065``, linear twist
``-12 deg`` root-to-tip, taper ``0.75`` (tip/root chord), hover ``671 RPM``
(``Omega = 70.3 rad/s``, tip speed ``137.16 m/s``, ``BPF = 33.55 Hz``).

Reconstructed (documented assumptions, not published):

- **Chord** ``c(r) = c_root (1 - (1-taper)(r - r_hub)/(R - r_hub))`` with
  ``c_root`` solved so the ``r^2``-weighted (thrust-weighted) mean chord gives
  ``sigma = 0.065``; i.e. the thrust-weighted mean chord is
  ``c_bar = sigma pi R / Nb = 0.1329 m`` and ``c_root = c_bar / <shape>_{r^2}``.
- **Twist** linear ``0 -> -12 deg`` from root to tip; the absolute collective is
  set by hover trim (:func:`nasa_1pax_hover_collective`).
- **Airfoil** a representative thin-airfoil polar (``2 pi`` slope, small profile
  drag); CONA's UAM case used XFOIL polars -- documented sensitivity.
- **Root cutout** ``0.15 R`` (typical; not published).

All lengths [m], angles [rad] internally (degrees only in the named constants).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp

from auraflow.bemt.solver import Polar, steady_bemt
from auraflow.cona.flight import Multirotor
from auraflow.core.airfoil import ThinAirfoilPolar
from auraflow.core.blade import BladeGeometry, Rotor, Vehicle
from auraflow.core.medium import Medium

if TYPE_CHECKING:
    from auraflow.body.mesh import TriMesh

__all__ = [
    "BPF_HZ",
    "GROSS_WEIGHT_KG",
    "HOVER_OMEGA",
    "HOVER_RPM",
    "LINEAR_TWIST_DEG",
    "N_BLADES",
    "N_ROTORS",
    "ROOT_CUTOUT_FRAC",
    "ROTOR_RADIUS",
    "SOLIDITY",
    "TAPER",
    "TIP_SPEED",
    "nasa_1pax_blade",
    "nasa_1pax_blade_mesh",
    "nasa_1pax_hover_collective",
    "nasa_1pax_multirotor",
    "nasa_1pax_polar",
    "nasa_1pax_rotor_mesh",
    "nasa_1pax_vehicle",
    "trim_hover_collective",
]

_G = 9.80665  # standard gravity [m/s^2] (matches auraflow.core.medium)

# --- Published constants (digest nasa-1pax-vehicle.md) ------------------------
GROSS_WEIGHT_KG: float = 583.85
"""Design gross weight [kg]."""
N_ROTORS: int = 4
"""Number of rotors."""
N_BLADES: int = 3
"""Blades per rotor."""
ROTOR_RADIUS: float = 1.951
"""Tip radius ``R`` [m] (= 6.4 ft)."""
SOLIDITY: float = 0.065
"""Thrust-weighted rotor solidity [-]."""
LINEAR_TWIST_DEG: float = -12.0
"""Linear geometric twist, root-to-tip [deg]."""
TAPER: float = 0.75
"""Blade taper ratio (tip chord / root chord) [-]."""
ROOT_CUTOUT_FRAC: float = 0.15
"""Root cutout as a fraction of ``R`` [-] (reconstructed assumption)."""
HOVER_RPM: float = 671.0
"""Hover rotor speed [rev/min]."""
HOVER_OMEGA: float = 70.3
"""Hover rotor speed [rad/s] (= 671 RPM = 450 ft/s tip / R)."""
TIP_SPEED: float = 137.16
"""Hover tip speed [m/s] (= 450 ft/s)."""
BPF_HZ: float = N_BLADES * HOVER_RPM / 60.0
"""Blade-passing frequency at hover [Hz] (= 3 * 671 / 60 = 33.55 Hz)."""


def _reconstructed_root_chord(n_grid: int = 2048) -> float:
    """Root chord [m] giving thrust-weighted solidity ``SOLIDITY``.

    ``c_bar = SOLIDITY * pi * R / Nb`` is the ``r^2``-weighted mean chord; with
    the linear-taper shape ``s(r) = 1 - (1-TAPER)(r-r_hub)/(R-r_hub)`` the root
    chord is ``c_root = c_bar / <s>_{r^2}`` where ``<s>_{r^2}`` is the
    ``r^2``-weighted mean of ``s`` over ``[r_hub, R]`` (trapezoid quadrature).
    """
    r_hub = ROOT_CUTOUT_FRAC * ROTOR_RADIUS
    r = jnp.linspace(r_hub, ROTOR_RADIUS, n_grid)
    shape = 1.0 - (1.0 - TAPER) * (r - r_hub) / (ROTOR_RADIUS - r_hub)
    w = r**2
    mean_shape = float(jnp.trapezoid(shape * w, r) / jnp.trapezoid(w, r))
    c_bar = SOLIDITY * math.pi * ROTOR_RADIUS / N_BLADES
    return c_bar / mean_shape


def nasa_1pax_blade(n_stations: int = 16) -> BladeGeometry:
    """Reconstructed 1-Pax blade geometry (linear taper + linear twist).

    Args:
        n_stations: Number of radial stations ``S`` (static int, ``>= 2``).

    Returns:
        A :class:`~auraflow.core.blade.BladeGeometry` on ``[0.15 R, R]`` with the
        reconstructed chord (thrust-weighted solidity ``0.065``, taper ``0.75``)
        and linear ``0 -> -12 deg`` twist. The absolute collective is added at
        airload time (:func:`nasa_1pax_hover_collective`).
    """
    c_root = _reconstructed_root_chord()
    return BladeGeometry.linear(
        radius=ROTOR_RADIUS,
        hub_radius=ROOT_CUTOUT_FRAC * ROTOR_RADIUS,
        n_stations=n_stations,
        chord_root=c_root,
        chord_tip=TAPER * c_root,
        twist_root=0.0,
        twist_tip=math.radians(LINEAR_TWIST_DEG),
    )


def nasa_1pax_blade_mesh(n_span: int = 16, n_chord: int = 60) -> TriMesh:
    """Watertight NACA-0012 blade :class:`TriMesh` for the reconstructed 1-Pax blade.

    Lofts :func:`nasa_1pax_blade` (the single source of truth for the blade
    chord/twist) with :func:`auraflow.body.blade.blade_mesh` and a symmetric
    NACA 0012 section. Built in the blade section frame (x spanwise, y chordwise
    toward the leading edge, z thrust-normal; quarter chord on the spanwise axis).

    Args:
        n_span: Number of radial stations ``S`` (static int) -- also the lofted
            span discretization.
        n_chord: Chordwise samples per surface for the section (static int).

    Returns:
        A watertight, outward-wound blade :class:`~auraflow.body.mesh.TriMesh`.
    """
    from auraflow.body.blade import blade_mesh

    return blade_mesh(nasa_1pax_blade(n_span), n_chord=n_chord)


def nasa_1pax_rotor_mesh(
    n_span: int = 16, n_chord: int = 60, hub: bool | dict[str, float] = False
) -> TriMesh:
    """Three-blade rotor :class:`TriMesh` for the 1-Pax rotor (rotor frame).

    Places ``N_BLADES`` copies of :func:`nasa_1pax_blade_mesh` at their equal
    azimuths via :func:`auraflow.body.blade.rotor_mesh` (thrust axis ``+z``,
    ``spin_direction = +1``), optionally with a hub cylinder. Geometry is read
    from :func:`nasa_1pax_blade` -- no duplicated constants.

    Args:
        n_span: Radial stations per blade (static int).
        n_chord: Chordwise samples per surface (static int).
        hub: Hub option forwarded to :func:`auraflow.body.blade.rotor_mesh`.

    Returns:
        A rotor :class:`~auraflow.body.mesh.TriMesh` in the rotor frame.
    """
    from auraflow.body.blade import rotor_mesh

    rotor = Rotor(blade=nasa_1pax_blade(n_span), n_blades=N_BLADES)
    return rotor_mesh(rotor, hub=hub, n_chord=n_chord)


def nasa_1pax_polar() -> ThinAirfoilPolar:
    """Representative airfoil polar for the 1-Pax blade (documented assumption).

    A thin-airfoil ``2 pi`` lift slope with a small profile drag and a modest
    induced-drag factor, standing in for the unpublished "modern airfoils"
    (CONA's UAM case used XFOIL polars -- see the module docstring). Smooth and
    differentiable.
    """
    return ThinAirfoilPolar(alpha0=0.0, cl_alpha=2.0 * math.pi, cd0=0.011, k=0.02)


def nasa_1pax_multirotor(drag_coeff: float = 0.0, motor_tau: float | None = None) -> Multirotor:
    """The 1-Pax :class:`~auraflow.cona.flight.Multirotor` (delegates, no dup).

    Thin pass-through to :meth:`auraflow.cona.flight.Multirotor.nasa_1pax` so the
    mass / inertia / hub-placement / motor constants have exactly one home.

    Args:
        drag_coeff: Linear wind-drag coefficient [N.s/m] (the additive-gust
            hook; ``> 0`` couples :mod:`auraflow.cona.gusts` into the flight).
        motor_tau: First-order motor time constant [s] or ``None``.

    Returns:
        The configured :class:`~auraflow.cona.flight.Multirotor`.
    """
    return Multirotor.nasa_1pax(drag_coeff=drag_coeff, motor_tau=motor_tau)


def nasa_1pax_vehicle(n_stations: int = 16) -> Vehicle:
    """Geometric :class:`~auraflow.core.blade.Vehicle` for the 1-Pax quadrotor.

    Hub positions and spin senses are read back from
    :func:`nasa_1pax_multirotor` (the single source of truth), so this never
    re-states them; the blade geometry (:func:`nasa_1pax_blade`) is attached to
    each of the ``N_ROTORS`` rotors. Every rotor's thrust axis is body ``+z``
    (identity ``hub_orientation``); the rear rotors carry the ``+0.683 m`` hub
    height already encoded in the multirotor positions.

    Args:
        n_stations: Radial stations per blade (static int).

    Returns:
        A :class:`~auraflow.core.blade.Vehicle` with ``N_ROTORS`` rotors at the
        1-Pax X-arrangement.
    """
    mrotor = nasa_1pax_multirotor()
    blade = nasa_1pax_blade(n_stations)
    positions = mrotor.rotor_positions  # [Nr, 3], body frame
    spins = mrotor.spin_signs  # [Nr]
    eye = jnp.eye(3)
    rotors = tuple(
        Rotor(
            blade=blade,
            n_blades=N_BLADES,
            hub_position=positions[i],
            hub_orientation=eye,
            spin_direction=int(round(float(spins[i]))),
        )
        for i in range(mrotor.n_rotors)
    )
    return Vehicle(rotors=rotors)


def trim_hover_collective(
    rotor: Rotor,
    medium: Medium,
    omega: float,
    target_thrust: float,
    polar: Polar | None = None,
    *,
    lo: float = 0.0,
    hi: float = math.radians(30.0),
    n_iter: int = 48,
) -> float:
    """Collective [rad] giving ``target_thrust`` at ``omega`` in hover (bisection).

    Runs :func:`auraflow.bemt.solver.steady_bemt` for one rotor and bisects the
    collective pitch until the integrated rotor thrust matches ``target_thrust``.
    Thrust is monotonic in collective below stall, so a bracketing bisection is
    robust; the result is a Python ``float`` (a static number the airload stage
    adds to the blade twist).

    Args:
        rotor: The rotor (blade geometry + blade count).
        medium: Ambient medium.
        omega: Rotor speed magnitude [rad/s].
        target_thrust: Desired rotor thrust [N] (e.g. weight / ``N_ROTORS``).
        polar: Airfoil polar; defaults to :func:`nasa_1pax_polar`.
        lo, hi: Collective search bracket [rad].
        n_iter: Bisection iterations (static; ``2^-48`` rad is ample).

    Returns:
        The trimmed collective pitch [rad].
    """
    if polar is None:
        polar = nasa_1pax_polar()

    def thrust_at(coll: float) -> float:
        loads = steady_bemt(rotor, medium, omega, v_climb=0.0, collective=coll, polar=polar)
        return float(loads.thrust)

    a, b = lo, hi
    for _ in range(n_iter):
        mid = 0.5 * (a + b)
        if thrust_at(mid) < target_thrust:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


def nasa_1pax_hover_collective(
    n_stations: int = 16,
    medium: Medium | None = None,
    polar: Polar | None = None,
    omega: float = HOVER_OMEGA,
) -> float:
    """Hover-trimmed collective [rad] for the 1-Pax rotor at ``omega``.

    Trims one rotor to ``GROSS_WEIGHT_KG * g / N_ROTORS`` thrust at ``omega``
    (default hover ``70.3 rad/s``). Used as the airload collective for the
    slow (1-10 m/s) JASA level-flight cases, where the hover trim is a good
    approximation.

    Args:
        n_stations: Radial stations for the trimming rotor (static int).
        medium: Ambient medium (default sea-level ISA).
        polar: Airfoil polar (default :func:`nasa_1pax_polar`).
        omega: Rotor speed magnitude [rad/s].

    Returns:
        The hover-trim collective pitch [rad].
    """
    medium = Medium() if medium is None else medium
    rotor = Rotor(blade=nasa_1pax_blade(n_stations), n_blades=N_BLADES)
    target = GROSS_WEIGHT_KG * _G / N_ROTORS
    return trim_hover_collective(rotor, medium, omega, target, polar)
