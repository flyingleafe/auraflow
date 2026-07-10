r"""DJI Phantom quadrotor with DJI 9450 propellers: the drone-scale vehicle.

The small-vehicle counterpart of :mod:`auraflow.datasets.nasa_1pax`. Reconstructs
a DJI Phantom 3 (Advanced/Professional/4K, the variant that ships the two-bladed
DJI 9450 9.4 in x 5.0 in prop) from the digest ``docs/research/dji-9450-reference.md``
for (a) CONA flyovers that sound like a consumer drone (BPF ~180 Hz, audible
harmonic comb) and (b) resolved-blade CFD (single-GPU feasible for this rotor
class). Mirrors ``nasa_1pax.py`` exactly: constants, ``dji_9450_blade``,
``*_blade_mesh``, ``dji_phantom_rotor_mesh``, ``dji_phantom_polar``,
``dji_phantom_multirotor``, ``dji_phantom_vehicle``, ``dji_phantom_hover_collective``.

Single source of truth
----------------------
Mass / inertia / rotor-allocation (hub positions, spin senses, ``k_f``,
``c_tauf``) live in :meth:`auraflow.cona.flight.Multirotor.dji_phantom`; this
module *delegates* to it (:func:`dji_phantom_multirotor`) and *reads back* the
hub positions and spin senses when it places the geometric rotors
(:func:`dji_phantom_vehicle`) -- no vehicle constant is written down twice. What
lives *here* is the blade geometry (radius, chord, twist, airfoil polar).

Naming caveat (digest sect. "Naming caveat")
--------------------------------------------
The propeller most acoustics papers (incl. CONA) call "the DJI 9450" is really
the lower-pitch **DJI 9443** ("DJI-CF") measured by Zawodny, Boyd & Burley
(AHS/AIAA 2016, NTRS 20160009054). We adopt the geometry of the **real DJI 9450**
plastic prop, directly measured by Deters, Kleinke & Selig (AIAA 2017-3743, UIUC,
Fig. 7), which matches the "9450" part number and the literature's nominal
``5400 RPM`` hover point. The 9443/DJI-CF hover point sits higher (~6300 RPM,
BPF ~210 Hz) because it has ~14% less pitch -- see the digest.

Published/digitized vs reconstructed (digest ``dji-9450-reference.md``)
----------------------------------------------------------------------
- **DIGITIZED** (Deters Fig. 7, pixel-calibrated laser/optical-scan trace, ~+-0.005
  c/R and +-0.3 deg): the :data:`_R_OVER_R` / :data:`_C_OVER_R` / :data:`_TWIST_DEG`
  chord & twist tables. Peak chord ``c/R ~ 0.254`` at ``r/R ~ 0.28``; twist falls
  from ``~20.8 deg`` (root) to ``~8.5 deg`` (tip).
- **PUBLISHED**: tip radius ``R = 0.1200 m`` (9.45 in true diameter); 2 blades;
  mass ``1.280 kg``; ``0.350 m`` motor-to-motor diagonal; spin pattern.
- **RECONSTRUCTED/ESTIMATED**: nominal hover ``5400 RPM`` (thrust ~ weight/4
  cross-check gives ``5040-5400 RPM``; literature nominal adopted); root cutout
  ``0.15 R``; the airfoil polar (thin cambered plate -- the source used scanned
  sections, **no named airfoil family**; a small negative zero-lift angle stands
  in for the camber).

All lengths [m], angles [rad] internally (degrees only in the named constants).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp

from auraflow.bemt.solver import Polar
from auraflow.cona.flight import Multirotor
from auraflow.core.airfoil import ThinAirfoilPolar
from auraflow.core.blade import BladeGeometry, Rotor, Vehicle
from auraflow.core.medium import Medium
from auraflow.datasets.nasa_1pax import trim_hover_collective

if TYPE_CHECKING:
    from auraflow.body.mesh import TriMesh

__all__ = [
    "BPF_HZ",
    "DIAGONAL_M",
    "HOVER_OMEGA",
    "HOVER_RPM",
    "MASS",
    "N_BLADES",
    "N_ROTORS",
    "ROOT_CUTOUT_FRAC",
    "ROTOR_RADIUS",
    "TIP_MACH",
    "TIP_SPEED",
    "dji_9450_blade",
    "dji_9450_blade_mesh",
    "dji_phantom_hover_collective",
    "dji_phantom_multirotor",
    "dji_phantom_polar",
    "dji_phantom_rotor_mesh",
    "dji_phantom_vehicle",
]

_G = 9.80665  # standard gravity [m/s^2] (matches auraflow.core.medium)
_C0 = 340.0  # reference speed of sound for tip Mach [m/s]

# --- Published / digitized constants (digest dji-9450-reference.md) -----------
MASS: float = 1.280
"""Vehicle mass [kg] (DJI Phantom 3 Adv./Pro./4K, ships the 9450 prop)."""
N_ROTORS: int = 4
"""Number of rotors."""
N_BLADES: int = 2
"""Blades per rotor (DJI 9450 is two-bladed, fixed pitch)."""
ROTOR_RADIUS: float = 0.1200
"""Tip radius ``R`` [m] (= 9.45 in true diameter / 2)."""
ROOT_CUTOUT_FRAC: float = 0.15
"""Root cutout as a fraction of ``R`` [-] (first digitized station, Deters Fig. 7)."""
DIAGONAL_M: float = 0.350
"""Motor-to-motor diagonal wheelbase [m] (published)."""
HOVER_RPM: float = 5400.0
"""Nominal hover rotor speed [rev/min] (literature nominal; weight/4 gives 5040-5400)."""
HOVER_OMEGA: float = HOVER_RPM * 2.0 * math.pi / 60.0
"""Hover rotor speed [rad/s] (= 5400 RPM = 565.49 rad/s)."""
TIP_SPEED: float = HOVER_OMEGA * ROTOR_RADIUS
"""Hover tip speed [m/s] (= 67.9 m/s)."""
TIP_MACH: float = TIP_SPEED / _C0
"""Hover tip Mach number [-] (= 0.20 at a = 340 m/s)."""
BPF_HZ: float = N_BLADES * HOVER_RPM / 60.0
"""Blade-passing frequency at hover [Hz] (= 2 * 5400 / 60 = 180 Hz)."""

# --- Digitized chord/twist distribution (Deters, Kleinke & Selig 2017, Fig. 7) --
# Real DJI 9450 plastic prop; r/R, chord/R, geometric twist (blade pitch) [deg].
# DIGITIZED (pixel-calibrated); see digest table (A). Root cutout 0.15 R.
_R_OVER_R: tuple[float, ...] = (
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
)
_C_OVER_R: tuple[float, ...] = (
    0.180,
    0.219,
    0.246,
    0.254,
    0.240,
    0.223,
    0.204,
    0.191,
    0.179,
    0.165,
    0.156,
    0.144,
    0.133,
    0.125,
    0.115,
    0.105,
    0.096,
    0.057,
)
_TWIST_DEG: tuple[float, ...] = (
    16.8,
    19.0,
    20.7,
    20.8,
    19.9,
    18.6,
    17.0,
    15.8,
    14.8,
    13.9,
    12.8,
    11.8,
    10.9,
    10.6,
    9.9,
    9.7,
    9.6,
    8.5,
)


def dji_9450_blade(n_stations: int = 16) -> BladeGeometry:
    """Digitized DJI 9450 blade geometry (Deters Fig. 7, tabulated chord + twist).

    Builds a :class:`~auraflow.core.blade.BladeGeometry` on ``[0.15 R, R]`` from
    the digitized ``(r/R, c/R, twist)`` table via
    :meth:`~auraflow.core.blade.BladeGeometry.from_arrays` (linear interpolation
    onto ``n_stations`` uniform stations). The twist is the blade's **absolute
    geometric pitch** (a fixed-pitch prop); a small trim offset is added at
    airload time by :func:`dji_phantom_hover_collective`.

    Args:
        n_stations: Number of radial stations ``S`` (static int, ``>= 2``).

    Returns:
        The DJI 9450 :class:`~auraflow.core.blade.BladeGeometry`.
    """
    r = jnp.asarray(_R_OVER_R) * ROTOR_RADIUS
    chord = jnp.asarray(_C_OVER_R) * ROTOR_RADIUS
    twist = jnp.asarray([math.radians(t) for t in _TWIST_DEG])
    return BladeGeometry.from_arrays(r, chord, twist, n_stations=n_stations)


def dji_9450_blade_mesh(n_span: int = 16, n_chord: int = 60) -> TriMesh:
    """Watertight blade :class:`TriMesh` lofted from the DJI 9450 blade.

    Lofts :func:`dji_9450_blade` (the single source of truth for chord/twist)
    with :func:`auraflow.body.blade.blade_mesh` and the default (NACA 0012)
    section -- a symmetric stand-in for the actual thin cambered plate (the
    source used scanned sections; no named airfoil is published). Built in the
    blade section frame (x spanwise, y chordwise toward the LE, z thrust-normal;
    quarter chord on the spanwise axis).

    Args:
        n_span: Radial stations ``S`` (static int) -- also the lofted span
            discretization.
        n_chord: Chordwise samples per surface for the section (static int).

    Returns:
        A watertight, outward-wound blade :class:`~auraflow.body.mesh.TriMesh`.
    """
    from auraflow.body.blade import blade_mesh

    return blade_mesh(dji_9450_blade(n_span), n_chord=n_chord)


def dji_phantom_rotor_mesh(
    n_span: int = 16, n_chord: int = 60, hub: bool | dict[str, float] = False
) -> TriMesh:
    """Two-blade rotor :class:`TriMesh` for the DJI 9450 rotor (rotor frame).

    Places ``N_BLADES`` copies of :func:`dji_9450_blade_mesh` at their equal
    azimuths via :func:`auraflow.body.blade.rotor_mesh` (thrust axis ``+z``,
    ``spin_direction = +1``), optionally with a hub cylinder. Geometry is read
    from :func:`dji_9450_blade` -- no duplicated constants.

    Args:
        n_span: Radial stations per blade (static int).
        n_chord: Chordwise samples per surface (static int).
        hub: Hub option forwarded to :func:`auraflow.body.blade.rotor_mesh`.

    Returns:
        A rotor :class:`~auraflow.body.mesh.TriMesh` in the rotor frame.
    """
    from auraflow.body.blade import rotor_mesh

    rotor = Rotor(blade=dji_9450_blade(n_span), n_blades=N_BLADES)
    return rotor_mesh(rotor, hub=hub, n_chord=n_chord)


def dji_phantom_polar() -> ThinAirfoilPolar:
    """Representative airfoil polar for the DJI 9450 blade (documented assumption).

    A thin **cambered**-plate polar: ``2 pi`` lift slope with a small negative
    zero-lift angle (``alpha0 = -4 deg``) standing in for the injection-molded
    blade's camber, plus a modest profile drag (higher than a clean airfoil, as
    these small blades run at low Reynolds number). The digest notes NASA/UIUC
    worked from scanned cross-sections with **no named airfoil family**; this is
    a smooth, differentiable stand-in for that thin cambered section.
    """
    return ThinAirfoilPolar(alpha0=math.radians(-4.0), cl_alpha=2.0 * math.pi, cd0=0.02, k=0.05)


def dji_phantom_multirotor(drag_coeff: float = 0.0, motor_tau: float | None = None) -> Multirotor:
    """The DJI Phantom :class:`~auraflow.cona.flight.Multirotor` (delegates, no dup).

    Thin pass-through to :meth:`auraflow.cona.flight.Multirotor.dji_phantom` so
    the mass / inertia / hub-placement / motor constants have exactly one home.

    Args:
        drag_coeff: Linear wind-drag coefficient [N.s/m] (the additive-gust
            hook; ``> 0`` couples :mod:`auraflow.cona.gusts` into the flight, so
            the four rotors beat slightly -- realistic drone RPM beating).
        motor_tau: First-order motor time constant [s] or ``None``.

    Returns:
        The configured :class:`~auraflow.cona.flight.Multirotor`.
    """
    return Multirotor.dji_phantom(drag_coeff=drag_coeff, motor_tau=motor_tau)


def dji_phantom_vehicle(n_stations: int = 16) -> Vehicle:
    """Geometric :class:`~auraflow.core.blade.Vehicle` for the DJI Phantom quad.

    Hub positions and spin senses are read back from
    :func:`dji_phantom_multirotor` (the single source of truth), so this never
    re-states them; the blade geometry (:func:`dji_9450_blade`) is attached to
    each of the ``N_ROTORS`` rotors. Every rotor's thrust axis is body ``+z``
    (identity ``hub_orientation``); all four hubs sit in the ``z = 0`` rotor
    plane (rotor-plane height is not published).

    Args:
        n_stations: Radial stations per blade (static int).

    Returns:
        A :class:`~auraflow.core.blade.Vehicle` with ``N_ROTORS`` rotors at the
        DJI Phantom X-arrangement.
    """
    mrotor = dji_phantom_multirotor()
    blade = dji_9450_blade(n_stations)
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


def dji_phantom_hover_collective(
    n_stations: int = 16,
    medium: Medium | None = None,
    polar: Polar | None = None,
    omega: float = HOVER_OMEGA,
) -> float:
    """Hover-trimmed collective [rad] for the DJI 9450 rotor at ``omega``.

    Trims one rotor to ``MASS * g / N_ROTORS`` thrust at ``omega`` (default hover
    ``565.5 rad/s`` = 5400 RPM) with :func:`auraflow.datasets.nasa_1pax.trim_hover_collective`.
    Because the DJI 9450 is a **fixed-pitch** prop whose full geometric pitch is
    already in the blade twist, this trim offset is small (the reconstructed
    geometry very nearly hovers at the nominal RPM on its own); the bracket
    therefore spans a small band around zero collective.

    Args:
        n_stations: Radial stations for the trimming rotor (static int).
        medium: Ambient medium (default sea-level ISA).
        polar: Airfoil polar (default :func:`dji_phantom_polar`).
        omega: Rotor speed magnitude [rad/s].

    Returns:
        The hover-trim collective pitch [rad] (small; ``~0`` for fixed pitch).
    """
    medium = Medium() if medium is None else medium
    polar = dji_phantom_polar() if polar is None else polar
    rotor = Rotor(blade=dji_9450_blade(n_stations), n_blades=N_BLADES)
    target = MASS * _G / N_ROTORS
    return trim_hover_collective(
        rotor,
        medium,
        omega,
        target,
        polar,
        lo=math.radians(-15.0),
        hi=math.radians(15.0),
    )
