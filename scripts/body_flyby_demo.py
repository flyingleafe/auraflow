#!/usr/bin/env python
"""Fly a rigid body past ground microphones: thickness noise + Doppler.

A primitive mesh (default: a sphere) or an imported model (``--mesh``) is carried
along a smooth :class:`~auraflow.body.WaypointMotion` flyby at altitude over a few
ground mics. Its **thickness** (volume-displacement) noise is radiated with the
mesh FW-H path (:func:`auraflow.body.mesh_pressure`, no surface pressure, so
loading is zero), each mic's OASPL is printed, and the kinematic Doppler factor
``1 / (1 - M_r)`` (radial-Mach line-of-sight) is reported at approach and recede
for the centre mic. With ``--viz`` the moving mesh streams to the browser viewer.

Sized for the low-RAM dev box: coarse mesh, short flight, few mics, few samples.

Example
-------
    systemd-run --user --scope -q -p MemoryMax=1100M -p MemorySwapMax=0 -- \
        uv run python scripts/body_flyby_demo.py --n-times 160
"""

from __future__ import annotations

import argparse

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from auraflow.body import TriMesh, WaypointMotion, load_mesh, mesh_pressure  # noqa: E402
from auraflow.body.motion import pose_derivatives  # noqa: E402
from auraflow.core.medium import Medium  # noqa: E402
from auraflow.signal.spectra import oaspl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mesh", type=str, default=None, help="body mesh; default a sphere")
    p.add_argument("--radius", type=float, default=0.3, help="sphere radius [m] (no --mesh)")
    p.add_argument("--speed", type=float, default=60.0, help="flyby speed [m/s]")
    p.add_argument("--altitude", type=float, default=8.0, help="flyby altitude [m]")
    p.add_argument("--span", type=float, default=40.0, help="track half-length [m]")
    p.add_argument("--duration", type=float, default=None, help="flight time [s]")
    p.add_argument("--n-times", type=int, default=160, help="source-time samples")
    p.add_argument("--viz", action="store_true", help="stream the moving mesh to the browser")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    medium = Medium()
    c0 = float(medium.c0)

    if args.mesh is not None:
        mesh = load_mesh(args.mesh)
        print(f"  Body mesh: {mesh.n_faces} faces, watertight={mesh.is_watertight}")
    else:
        mesh = TriMesh.sphere(args.radius, subdivisions=2)
        print(f"  Sphere r={args.radius} m: {mesh.n_faces} faces")

    duration = args.duration if args.duration is not None else 2.0 * args.span / args.speed
    # Straight flyby along +x at altitude H, passing over the origin at mid-flight.
    x_start, x_end = -args.span, args.span
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * duration
    xs = np.linspace(x_start, x_end, 5)
    waypoints = np.stack([xs, np.zeros(5), np.full(5, args.altitude)], axis=-1)
    motion = WaypointMotion(times, waypoints)

    # Three ground mics under and beside the track.
    mics = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])

    tau = np.linspace(0.0, duration, args.n_times)
    print(
        f"  Flyby: {args.speed} m/s (M={args.speed / c0:.3f}), H={args.altitude} m, "
        f"{duration:.3f} s, {args.n_times} samples"
    )
    print("  Radiating thickness noise (mesh FW-H)…")
    p, t_obs = mesh_pressure(mesh, motion, tau, mics, medium)
    p = np.asarray(p)
    t_obs = np.asarray(t_obs)

    levels = np.asarray(oaspl(jnp.asarray(p), axis=-1))
    print("\n  Mic OASPLs (re 20 uPa):")
    for i, xo in enumerate(mics):
        print(
            f"    mic {i} at {xo.tolist()} m:  {levels[i]:6.2f} dB   "
            f"(peak {np.max(np.abs(p[i])):.3e} Pa)"
        )

    # Kinematic Doppler for the centre mic: f_obs/f_src = 1 / (1 - M_r), with
    # M_r the source Mach projected onto the source->observer line of sight.
    def doppler_ratio(t: float, mic: np.ndarray) -> float:
        r, x, dr, dx, ddr, ddx = pose_derivatives(motion, jnp.asarray(t))
        vel = np.asarray(dx)
        los = mic - np.asarray(x)
        los = los / (np.linalg.norm(los) or 1.0)
        m_r = float(np.dot(vel, los)) / c0  # +ve when approaching the mic
        return 1.0 / (1.0 - m_r)

    mic0 = mics[0]
    approach = doppler_ratio(0.2 * duration, mic0)
    recede = doppler_ratio(0.8 * duration, mic0)
    print(
        f"\n  Doppler at centre mic:  approaching x{approach:.3f}  "
        f"receding x{recede:.3f}  (shift {(approach / recede - 1) * 100:.1f}%)"
    )

    if args.viz:
        import time

        from auraflow.viz import VizStreamer
        from auraflow.viz.body import stream_body

        with VizStreamer(port=args.port) as viz:
            print(f"\n  Open  {viz.http_url}  to watch the flyby.")
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not viz.active:
                time.sleep(0.2)
            try:
                while True:
                    stream_body(
                        viz,
                        mesh,
                        motion,
                        tau,
                        mics=mics,
                        mic_signals=p,
                        mic_t=t_obs,
                        fps=30.0,
                        opacity=0.85,
                        color=(0.85, 0.85, 0.9),
                        title="AuraFlow body flyby",
                    )
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n  Bye.")


if __name__ == "__main__":
    main()
