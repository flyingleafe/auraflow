"""CONA flyover replay: stream a :class:`FlightHistory` to the live viewer.

The CONA stages are batch (the whole trajectory, airloads, and per-mic signals
are computed up front), so there is nothing to watch *during* the compute the way
there is for the step-by-step CFD driver. Instead this module **replays** a
finished flyover as an animation: it walks the vehicle pose / rotor azimuths over
time and pushes a frame per animation tick, optionally advancing a ring buffer of
selected microphone pressure traces so the frontend can draw a live strip chart.

Everything here is plain NumPy on already-materialised host arrays -- no JAX, no
device compute -- so it is cheap and safe on the dev box.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from auraflow.cona.flight import FlightHistory
    from auraflow.core.blade import Vehicle
    from auraflow.viz.server import VizStreamer

__all__ = ["flight_scene", "stream_flyover"]


def _rotor_layout(vehicle: Vehicle) -> list[dict[str, Any]]:
    """Body-frame rotor layout dicts for the scene message (hub, radius, axis)."""
    rotors: list[dict[str, Any]] = []
    for rotor in vehicle.rotors:
        hub = np.asarray(rotor.hub_position, dtype=float)
        axis = np.asarray(rotor.hub_orientation, dtype=float)[:, 2]  # thrust axis
        rotors.append(
            {
                "hub": [float(v) for v in hub],
                "radius": float(np.asarray(rotor.blade.radius)),
                "n_blades": int(rotor.n_blades),
                "axis": [float(v) for v in axis],
                "spin": int(rotor.spin_direction),
            }
        )
    return rotors


def _azimuths(flight: FlightHistory, spin_signs: np.ndarray) -> np.ndarray:
    """Unwrapped reference-blade azimuth per rotor over time ``[T, Nr]`` [rad].

    Cumulative-trapezoid integral of the signed rotor rate (matches
    :func:`auraflow.core.frames.integrate_azimuth`), done in NumPy for replay.
    """
    t = np.asarray(flight.t, dtype=float)  # [T]
    omega = np.asarray(flight.rotor_speeds, dtype=float) * spin_signs[None, :]  # [T, Nr]
    dt = np.diff(t)[:, None]
    incr = 0.5 * (omega[:-1] + omega[1:]) * dt  # [T-1, Nr]
    psi = np.zeros_like(omega)
    psi[1:] = np.cumsum(incr, axis=0)
    return psi


def flight_scene(
    vehicle: Vehicle,
    flight: FlightHistory,
    mics: Any | None = None,
    *,
    pad: float = 5.0,
    title: str = "CONA flyover",
) -> dict[str, Any]:
    """Assemble :func:`auraflow.viz.stream.encode_scene` kwargs for a flyover.

    The domain box is the axis-aligned bounding box of the whole trajectory (and
    mics, if given), padded by ``pad`` metres so the vehicle never touches the
    edge.

    Args:
        vehicle: The :class:`~auraflow.core.blade.Vehicle` (rotor layout).
        flight: The :class:`~auraflow.cona.flight.FlightHistory` to bound.
        mics: Optional microphone positions ``[M, 3]`` [m].
        pad: Padding added around the bounding box [m].
        title: Scene title shown in the page header.

    Returns:
        A kwargs dict for :meth:`auraflow.viz.server.VizStreamer.init_scene`.
    """
    x = np.asarray(flight.x, dtype=float)  # [T, 3]
    pts = [x]
    if mics is not None:
        pts.append(np.asarray(mics, dtype=float).reshape(-1, 3))
    allp = np.concatenate(pts, axis=0)
    box_min = allp.min(axis=0) - pad
    box_max = allp.max(axis=0) + pad
    # Keep the ground plane (z=0) in view for the mic array.
    box_min[2] = min(box_min[2], -pad)
    return {
        "box_min": [float(v) for v in box_min],
        "box_max": [float(v) for v in box_max],
        "mics": None if mics is None else np.asarray(mics, dtype=np.float32),
        "rotors": _rotor_layout(vehicle),
        "fields": [],
        "title": title,
    }


def stream_flyover(
    streamer: VizStreamer,
    vehicle: Vehicle,
    flight: FlightHistory,
    *,
    mics: Any | None = None,
    mic_signals: Any | None = None,
    mic_t: Any | None = None,
    fps: float = 30.0,
    realtime: bool = True,
    ring: int = 512,
) -> None:
    """Replay a finished flyover as a live animation over ``streamer``.

    Pushes the scene, then one frame per animation tick: vehicle pose, per-rotor
    azimuths, and (if ``mic_signals`` is given) the instantaneous per-mic
    pressure plus a rolling ring buffer of the trace for the frontend strip chart.

    Args:
        streamer: An entered :class:`~auraflow.viz.server.VizStreamer`.
        vehicle: The vehicle whose rotor layout is animated.
        flight: The flight history (pose, rotor speeds) to replay.
        mics: Microphone positions ``[M, 3]`` [m], or ``None``.
        mic_signals: Per-mic pressure ``[M, n]`` [Pa] on grid ``mic_t``, or
            ``None``. Interpolated to each animation time.
        mic_t: Time grid for ``mic_signals`` ``[n]`` [s], or ``None`` (defaults to
            the flight time grid if ``mic_signals`` matches its length).
        fps: Animation frames per second.
        realtime: If ``True``, sleep so playback runs at wall-clock speed
            (flight-time seconds per real second). If ``False``, stream as fast
            as possible.
        ring: Length of the per-mic pressure ring buffer sent each frame.
    """
    t = np.asarray(flight.t, dtype=float)
    x = np.asarray(flight.x, dtype=float)  # [T, 3]
    R = np.asarray(flight.R, dtype=float)  # [T, 3, 3]
    spin_signs = np.array([r.spin_direction for r in vehicle.rotors], dtype=float)
    psi = _azimuths(flight, spin_signs)  # [T, Nr]

    sig = None if mic_signals is None else np.asarray(mic_signals, dtype=float)  # [M, n]
    if sig is not None:
        st = np.asarray(mic_t, dtype=float) if mic_t is not None else t
    else:
        st = None

    streamer.init_scene(**flight_scene(vehicle, flight, mics))

    t0, t1 = float(t[0]), float(t[-1])
    n_ticks = max(int(round((t1 - t0) * fps)), 1)
    tick_times = np.linspace(t0, t1, n_ticks)
    wall_start = time.monotonic()

    for step, tt in enumerate(tick_times):
        pos = np.array([np.interp(tt, t, x[:, k]) for k in range(3)])
        # Interpolate attitude entrywise then re-orthonormalise (small drift).
        Rt = np.array([[np.interp(tt, t, R[:, i, j]) for j in range(3)] for i in range(3)])
        u_, _, vh = np.linalg.svd(Rt)
        Rt = u_ @ vh
        az = np.array([np.interp(tt, t, psi[:, k]) for k in range(psi.shape[1])])

        mic_p = None
        ring_buf = None
        lo = 0
        if sig is not None and st is not None:
            mic_p = np.array([np.interp(tt, st, sig[m]) for m in range(sig.shape[0])])
            lo = max(0, int(np.searchsorted(st, tt)) - ring)
            hi = int(np.searchsorted(st, tt)) + 1
            ring_buf = sig[:, lo:hi]

        extra: dict[str, Any] = {}
        if ring_buf is not None and st is not None:
            extra["mic_ring_t0"] = float(st[lo])
            extra["mic_ring_dt"] = float(st[1] - st[0]) if st.shape[0] > 1 else 0.0
        streamer.push_frame(
            t=float(tt),
            step=step,
            vehicle_pos=pos,
            vehicle_R=Rt.ravel(),
            rotor_azimuths=az,
            mic_p=mic_p,
            mic_ring=ring_buf,
            extra=extra if extra else None,
        )
        if realtime:
            target = wall_start + (float(tt) - t0)
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
