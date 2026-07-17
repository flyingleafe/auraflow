r"""DREGON & Michael onboard drone ego-noise auralization (CONA path).

Auralizes the **ego-noise** heard by microphones mounted *on* two real
drone-audition rigs, using the CONA backend:

- **DREGON** -- INRIA's MikroKopter quadrotor (``dregon.inria.fr``), the airframe
  behind the DREGON dataset (cubic 8-MEMS onboard array, 44.1 kHz). Rotors sweep
  a 36--84 rps operating range; hover sits near ``70 rps``
  (Strauss et al. 2018; digested in the harmonic-noise-suppression repo's
  ``data_processing/dregon.py`` and the Gulli et al. 2025 EURASIP paper).
- **Michael (Matrice 100)** -- "Michael's" DJI Matrice 100 X-quad (8-mic
  horizontal ring rig; ``data_processing/michaels.py``). Its DJI flight-controller
  logs put per-motor hover speed near ``4700 rpm`` (~78 rps).

Instead of the few real onboard mics, we place **64 microphones distributed on
spherical shells all around each drone** -- full ``4*pi`` directional coverage at
two ranges (:func:`onboard_mic_array`), a much more diverse observer set than
either real array while remaining "onboard"-scale (mics sit just outside the
rotor disk). This is deliberately *more* diverse than the real cubic/ring arrays,
per the dataset goal.

Pipeline (reuses the JASA CONA path)
------------------------------------
Each drone hovers in place; :func:`auraflow.datasets.jasa.generate_flyover` is
run at ``speed = 0`` so its ``straight_flyover`` reference degenerates to a fixed
hover point. Every mic then records a 44.1 kHz free-field pressure time series =
convective FW-H **tonal** (F1A/F1C) + Griffin-Lim auralized BPM **broadband**,
exactly as in :mod:`auraflow.datasets.jasa`. The same documented deviations apply
(no atmospheric absorption, no ground reflection; the mics are true free-field
onboard observers here, so ground effects are moot).

Vehicle reconstruction (documented assumptions)
-----------------------------------------------
Neither rig publishes a blade scan. Both carry small **two-bladed** props, so we
reuse the digitized DJI 9450 chord/twist *shape*
(:mod:`auraflow.datasets.dji_phantom`) scaled to each rotor radius, and the same
thin-cambered-plate polar -- a documented cross-prop assumption adequate for the
tonal/broadband ego-noise character. Airframe scale (wheelbase, rotor radius,
hover rps, mass) comes from each rig's spec; inertia and ``c_tauf`` are
reconstructed (only the hover flight-hold uses them). See :data:`DRONES`.

Shapes/units mirror :mod:`auraflow.datasets.jasa`: mics ``[64, 3]``, audio
``[64, n]``, SI, float64.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from auraflow.core.airfoil import ThinAirfoilPolar
from auraflow.core.blade import BladeGeometry, Rotor, Vehicle
from auraflow.core.medium import Medium
from auraflow.datasets.dji_phantom import (
    _C_OVER_R,
    _R_OVER_R,
    _TWIST_DEG,
    dji_phantom_polar,
)
from auraflow.datasets.jasa import JASAScenario, generate_flyover
from auraflow.datasets.nasa_1pax import trim_hover_collective

if TYPE_CHECKING:
    from auraflow.cona.flight import Multirotor

__all__ = [
    "DRONES",
    "DroneSpec",
    "egonoise_id",
    "generate_egonoise",
    "onboard_mic_array",
    "save_egonoise",
]

_G = 9.80665  # standard gravity [m/s^2] (matches auraflow.core.medium)
_C0 = 340.0  # reference speed of sound for the reported tip Mach [m/s]
# Diagonally-opposite rotors share spin sense (net-zero yaw); FL, FR, RL, RR.
_SPINS: tuple[int, int, int, int] = (-1, 1, 1, -1)
_N_BLADES = 2


@dataclass(frozen=True)
class DroneSpec:
    """Physical + acoustic reconstruction of one onboard-mic drone rig.

    All lengths [m], masses [kg], speeds [rad/s]. ``arm`` is the centre-to-hub
    distance (so the motor-to-motor diagonal is ``2*arm``); the four hubs sit at
    body ``(+-a, +-a, 0)`` with ``a = arm*cos45`` (X-quad). ``sphere_radii`` are
    the two onboard-array shell radii [m] (see :func:`onboard_mic_array`).

    Attributes:
        name: Short id (dataset key prefix).
        label: Human-readable rig name.
        arm: Centre-to-hub distance [m] (diagonal = ``2*arm``).
        rotor_radius: Blade tip radius ``R`` [m].
        mass: Vehicle mass [kg].
        inertia_diag: Reconstructed body inertia diagonal [kg.m^2].
        hover_rpm: Nominal per-rotor hover speed [rev/min].
        c_tauf: Reconstructed reaction-torque-to-thrust ratio [m].
        real_array: One-line description of the real onboard array (provenance).
        sphere_radii: ``(r_near, r_far)`` onboard-shell radii [m].
    """

    name: str
    label: str
    arm: float
    rotor_radius: float
    mass: float
    inertia_diag: tuple[float, float, float]
    hover_rpm: float
    c_tauf: float
    real_array: str
    sphere_radii: tuple[float, float]

    @property
    def hover_omega(self) -> float:
        """Per-rotor hover speed magnitude [rad/s]."""
        return self.hover_rpm * 2.0 * math.pi / 60.0

    @property
    def bpf_hz(self) -> float:
        """Blade-passing frequency at hover [Hz] (``N_BLADES * rpm / 60``)."""
        return _N_BLADES * self.hover_rpm / 60.0

    @property
    def hub_offset(self) -> float:
        """Per-axis hub offset ``a = arm*cos45`` [m] (hubs at ``(+-a, +-a, 0)``)."""
        return self.arm * math.cos(math.radians(45.0))

    def to_meta(self) -> dict[str, Any]:
        """JSON-serializable spec metadata (provenance for the dataset sample)."""
        return {
            "drone": self.name,
            "label": self.label,
            "arm_m": self.arm,
            "diagonal_m": 2.0 * self.arm,
            "rotor_radius_m": self.rotor_radius,
            "n_blades": _N_BLADES,
            "mass_kg": self.mass,
            "hover_rpm": self.hover_rpm,
            "hover_omega_rad_s": self.hover_omega,
            "bpf_hz": self.bpf_hz,
            "tip_speed_m_s": self.hover_omega * self.rotor_radius,
            "tip_mach": self.hover_omega * self.rotor_radius / _C0,
            "real_array": self.real_array,
            "sphere_radii_m": list(self.sphere_radii),
        }


# --- The two rigs (documented reconstructions; see module docstring) ----------
DRONES: dict[str, DroneSpec] = {
    # MikroKopter quad: motor-to-motor diagonal 0.485 m (dregon.py get_geometry
    # fallback), 10 in (0.254 m dia) two-bladed props, hover ~70 rps. Mass /
    # inertia / c_tauf reconstructed (MikroKopter-class micro quad).
    "dregon": DroneSpec(
        name="dregon",
        label="DREGON MikroKopter quadrotor",
        arm=0.485 / 2.0,
        rotor_radius=0.254 / 2.0,
        mass=1.2,
        inertia_diag=(0.012, 0.012, 0.022),
        hover_rpm=70.0 * 60.0,  # 70 rps -> 4200 rpm
        c_tauf=0.02,
        real_array="cubic 8-MEMS onboard array (DREGON, 44.1 kHz)",
        sphere_radii=(0.45, 0.70),
    ),
    # DJI Matrice 100: motor-to-motor diagonal 0.650 m (michaels.py WHEELBASE),
    # DJI 1345 (13 in / 0.330 m dia) two-bladed props, hover ~4700 rpm from the
    # FLY124/125 flight logs. Mass 2.43 kg (Matrice 100 + TB47D); inertia /
    # c_tauf reconstructed.
    "matrice100": DroneSpec(
        name="matrice100",
        label="Michael's DJI Matrice 100 X-quad",
        arm=0.650 / 2.0,
        rotor_radius=0.330 / 2.0,
        mass=2.43,
        inertia_diag=(0.05, 0.05, 0.09),
        hover_rpm=4700.0,
        c_tauf=0.02,
        real_array="8-mic horizontal ring, 82.5 mm radius (Michael's rig)",
        sphere_radii=(0.60, 0.95),
    ),
}


def _scaled_blade(radius: float, n_stations: int) -> BladeGeometry:
    """Two-bladed prop blade: DJI 9450 chord/twist shape scaled to ``radius``.

    The 9450's digitized ``(r/R, c/R, twist)`` (:mod:`auraflow.datasets.dji_phantom`)
    is a generic small two-bladed prop shape; scaling ``r`` and ``chord`` by the
    target tip radius (twist is scale-free) gives a differentiable stand-in blade
    for rigs with no published scan (documented in the module docstring).
    """
    r = jnp.asarray(_R_OVER_R) * radius
    chord = jnp.asarray(_C_OVER_R) * radius
    twist = jnp.asarray([math.radians(t) for t in _TWIST_DEG])
    return BladeGeometry.from_arrays(r, chord, twist, n_stations=n_stations)


def drone_vehicle(spec: DroneSpec, n_stations: int = 16) -> Vehicle:
    """Geometric :class:`~auraflow.core.blade.Vehicle` for a drone spec.

    Four rotors at the X-quad hubs ``(+-a, +-a, 0)`` (``a = spec.hub_offset``),
    thrust axis body ``+z``, diagonal pairs sharing spin sense
    (:data:`_SPINS`), each carrying :func:`_scaled_blade`.

    Args:
        spec: The drone reconstruction.
        n_stations: Radial stations per blade (static int).

    Returns:
        The X-quad :class:`~auraflow.core.blade.Vehicle`.
    """
    blade = _scaled_blade(spec.rotor_radius, n_stations)
    a = spec.hub_offset
    positions = ((a, a, 0.0), (a, -a, 0.0), (-a, a, 0.0), (-a, -a, 0.0))
    eye = jnp.eye(3)
    rotors = tuple(
        Rotor(
            blade=blade,
            n_blades=_N_BLADES,
            hub_position=jnp.asarray(p),
            hub_orientation=eye,
            spin_direction=s,
        )
        for p, s in zip(positions, _SPINS, strict=True)
    )
    return Vehicle(rotors=rotors)


def drone_multirotor(
    spec: DroneSpec,
    n_stations: int = 16,
    *,
    drag_coeff: float = 0.0,
    motor_tau: float | None = None,
) -> Multirotor:
    """Flight-dynamics :class:`~auraflow.cona.flight.Multirotor` for a drone spec.

    Reads the hub layout / spin senses back from :func:`drone_vehicle` (single
    source of truth), with ``k_f`` calibrated at the hover operating point
    (``m g / 4`` per rotor at ``spec.hover_omega``) and the reconstructed mass /
    inertia / ``c_tauf``.

    Args:
        spec: The drone reconstruction.
        n_stations: Radial stations per blade (must match the vehicle).
        drag_coeff: Linear wind-drag coefficient [N.s/m] (gust coupling hook).
        motor_tau: First-order motor time constant [s] or ``None``.

    Returns:
        The configured :class:`~auraflow.cona.flight.Multirotor`.
    """
    from auraflow.cona.flight import Multirotor

    vehicle = drone_vehicle(spec, n_stations)
    k_f = (spec.mass * _G / 4.0) / spec.hover_omega**2
    return Multirotor.from_vehicle(
        vehicle,
        mass=spec.mass,
        inertia=jnp.diag(jnp.asarray(spec.inertia_diag)),
        k_f=k_f,
        c_tauf=spec.c_tauf,
        drag_coeff=drag_coeff,
        motor_tau=motor_tau,
    )


def drone_hover_collective(
    spec: DroneSpec,
    n_stations: int = 16,
    medium: Medium | None = None,
    polar: ThinAirfoilPolar | None = None,
) -> float:
    """Hover-trimmed collective [rad] for the drone rotor at ``spec.hover_omega``.

    Trims one rotor to ``m g / 4`` thrust at the hover speed with
    :func:`auraflow.datasets.nasa_1pax.trim_hover_collective` (default thin
    cambered-plate polar, :func:`auraflow.datasets.dji_phantom.dji_phantom_polar`).

    Args:
        spec: The drone reconstruction.
        n_stations: Radial stations for the trimming rotor (static int).
        medium: Ambient medium (default sea-level ISA).
        polar: Airfoil polar (default the DJI-9450 thin cambered plate).

    Returns:
        The hover-trim collective pitch [rad].
    """
    medium = Medium() if medium is None else medium
    polar = dji_phantom_polar() if polar is None else polar
    rotor = Rotor(blade=_scaled_blade(spec.rotor_radius, n_stations), n_blades=_N_BLADES)
    target = spec.mass * _G / 4.0
    return trim_hover_collective(
        rotor,
        medium,
        spec.hover_omega,
        target,
        polar,
        lo=math.radians(-15.0),
        hi=math.radians(15.0),
    )


def _fibonacci_sphere(n: int, *, twist: float = 0.0) -> np.ndarray:
    """``n`` near-uniform unit vectors on the sphere (Fibonacci spiral), ``[n, 3]``.

    ``twist`` [rad] rotates the golden-angle azimuth so two shells generated with
    different twists do not share directions.
    """
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n  # polar cosine, evenly spaced in [-1, 1]
    phi = np.arccos(np.clip(z, -1.0, 1.0))
    theta = np.pi * (1.0 + math.sqrt(5.0)) * i + twist  # golden-angle azimuth
    s = np.sin(phi)
    return np.stack([s * np.cos(theta), s * np.sin(theta), z], axis=-1)


def onboard_mic_array(spec: DroneSpec, n_mics: int = 64, *, seed: int = 0) -> Array:
    """64 onboard microphones on two spherical shells around the drone body.

    Splits ``n_mics`` between two concentric Fibonacci shells at
    ``spec.sphere_radii`` (near/far), each near-uniform over the full sphere and
    twisted apart so the two ranges sample distinct directions. The result gives
    ``4*pi`` directional coverage at two source distances -- a much more diverse
    onboard observer set than the real cubic (DREGON) / ring (Matrice) arrays,
    while sitting just outside the rotor disk (mics never coincide with a hub).

    Body frame: X forward, Y left, Z up; origin at the drone body centre (the
    rotor-plane centroid). Returned in the body frame; the hover placement adds
    the world altitude.

    Args:
        spec: The drone reconstruction (provides the two shell radii).
        n_mics: Total microphone count (split near/far; default 64).
        seed: PRNG seed for the per-shell golden-angle twist offsets (keeps the
            layout stable and reproducible per dataset sample).

    Returns:
        Microphone positions in the body frame [m], shape ``[n_mics, 3]``
        (near shell first, then far shell).
    """
    rng = np.random.default_rng(int(seed))
    n_near = n_mics // 2
    n_far = n_mics - n_near
    r_near, r_far = spec.sphere_radii
    near = _fibonacci_sphere(n_near, twist=float(rng.uniform(0.0, 2.0 * np.pi))) * r_near
    far = _fibonacci_sphere(n_far, twist=float(rng.uniform(0.0, 2.0 * np.pi))) * r_far
    return jnp.asarray(np.concatenate([near, far], axis=0))


def egonoise_id(spec: DroneSpec, scenario: JASAScenario, n_mics: int) -> str:
    """Stable, filesystem-safe dataset key for one onboard ego-noise sample.

    ``<drone>_A<alt>_D<dur>_M<n_mics>_s<seed>`` -- the drone plus the hover
    altitude, duration, mic count and seed (speed is always 0 = hover).
    """
    return (
        f"{spec.name}_A{scenario.altitude:04.1f}_D{scenario.duration:04.1f}"
        f"_M{int(n_mics):03d}_s{int(scenario.seed):03d}"
    ).replace(".", "p")


def generate_egonoise(
    drone: str,
    *,
    altitude: float = 10.0,
    duration: float = 1.0,
    fs: float = 44100.0,
    seed: int = 0,
    n_mics: int = 64,
    medium: Medium | None = None,
    n_stations: int = 16,
    include_broadband: bool = True,
    low_memory: bool = False,
    obs_chunk: int = 16,
    **flyover_kwargs: Any,
) -> dict[str, Any]:
    """Generate one drone's onboard ego-noise: 64-mic auralized hover + metadata.

    Builds the drone's CONA vehicle/multirotor/hover-collective, places the
    64-mic onboard sphere (:func:`onboard_mic_array`) around a stationary hover
    at ``altitude``, and runs :func:`auraflow.datasets.jasa.generate_flyover` at
    ``speed = 0`` (hover). The absolute altitude is acoustically irrelevant here
    (free field, no ground/absorption); it only keeps the world frame tidy.

    Args:
        drone: A key in :data:`DRONES` (``"dregon"`` or ``"matrice100"``).
        altitude: Hover altitude (world ``z``) [m].
        duration: Signal duration [s].
        fs: Audio sample rate [Hz].
        seed: PRNG seed (mic-layout twist, Griffin-Lim phases).
        n_mics: Onboard microphone count (default 64).
        medium: Ambient medium (default sea-level ISA).
        n_stations: Radial blade stations (static int).
        include_broadband: Include the BPM broadband component.
        low_memory: Clear XLA caches at stage boundaries (small-box safety).
        obs_chunk: Mics per acoustic-propagation batch.
        **flyover_kwargs: Extra kwargs forwarded to
            :func:`~auraflow.datasets.jasa.generate_flyover` (e.g. ``n_frames``,
            ``n_fft``, ``gl_iters``).

    Returns:
        The :func:`~auraflow.datasets.jasa.generate_flyover` result dict, with
        ``"drone"``/``"spec"`` added and ``"mics_body"`` (body-frame positions),
        and ``meta`` extended with the drone provenance + ``"n_mics"``.
    """
    if drone not in DRONES:
        raise ValueError(f"unknown drone {drone!r}; choose from {sorted(DRONES)}")
    spec = DRONES[drone]
    medium = Medium() if medium is None else medium
    polar = dji_phantom_polar()

    vehicle = drone_vehicle(spec, n_stations)
    multirotor = drone_multirotor(spec, n_stations)
    collective = drone_hover_collective(spec, n_stations, medium, polar)

    mics_body = onboard_mic_array(spec, n_mics, seed=seed)
    mics_world = mics_body + jnp.asarray([0.0, 0.0, altitude])

    scenario = JASAScenario(
        speed=0.0,
        altitude=altitude,
        heading_deg=0.0,
        lateral_offset=0.0,
        duration=duration,
        fs=fs,
        seed=seed,
        mics=mics_world,
    )
    result = generate_flyover(
        scenario,
        medium=medium,
        polar=polar,
        collective=collective,
        vehicle=vehicle,
        multirotor=multirotor,
        bpf_hz=spec.bpf_hz,
        n_stations=n_stations,
        include_broadband=include_broadband,
        obs_chunk=obs_chunk,
        low_memory=low_memory,
        **flyover_kwargs,
    )
    result["drone"] = spec.name
    result["spec"] = spec
    result["mics_body"] = np.asarray(mics_body)
    result["meta"].update(spec.to_meta())
    result["meta"]["n_mics"] = int(n_mics)
    result["key"] = egonoise_id(spec, scenario, n_mics)
    return result


def save_egonoise(result: dict[str, Any], path: str) -> dict[str, str]:
    """Write an ego-noise result to ``<path>.npz`` + per-mic float32 WAVs.

    Like :func:`auraflow.datasets.jasa.save_flyover` but also stores the
    body-frame mic positions (``mics_body``) and the drone key.

    Args:
        result: A :func:`generate_egonoise` result dict.
        path: Output path stem (no extension); parent dirs are created.

    Returns:
        Dict of written paths: ``{"npz": ..., "wav_dir": ...}``.
    """
    import json
    import os

    from scipy.io import wavfile

    stem = os.fspath(path)
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    npz_path = stem + ".npz"
    np.savez_compressed(
        npz_path,
        audio=result["audio"],
        tonal=result["tonal"],
        broadband=result["broadband"],
        t_audio=result["t_audio"],
        mics=result["mics"],
        mics_body=result["mics_body"],
        band_centers=result["band_centers"],
        meta_json=json.dumps(result["meta"]),
    )
    fs = int(round(float(result["meta"]["fs"])))
    wav_dir = stem + "_wav"
    os.makedirs(wav_dir, exist_ok=True)
    audio = np.asarray(result["audio"], dtype=np.float32)
    for o in range(audio.shape[0]):
        wavfile.write(os.path.join(wav_dir, f"mic_{o:03d}.wav"), fs, audio[o])
    return {"npz": npz_path, "wav_dir": wav_dir}


def generate_grid(
    drones: Sequence[str],
    seeds: Sequence[int],
    *,
    altitude: float = 10.0,
    duration: float = 1.0,
    fs: float = 44100.0,
    n_mics: int = 64,
    **gen_kwargs: Any,
) -> list[dict[str, Any]]:
    """Cartesian product of ``drones x seeds`` -> generated ego-noise results.

    Args:
        drones: Drone keys (:data:`DRONES`).
        seeds: PRNG seeds.
        altitude, duration, fs, n_mics: Shared scenario settings.
        **gen_kwargs: Forwarded to :func:`generate_egonoise`.

    Returns:
        One result dict per ``(drone, seed)`` combination (drones outermost).
    """
    out: list[dict[str, Any]] = []
    for d in drones:
        for s in seeds:
            out.append(
                generate_egonoise(
                    d,
                    altitude=altitude,
                    duration=duration,
                    fs=fs,
                    seed=int(s),
                    n_mics=n_mics,
                    **gen_kwargs,
                )
            )
    return out
