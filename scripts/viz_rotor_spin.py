#!/usr/bin/env python
"""Live in-browser 3D view of the NASA 1-Pax rotor spinning at hover RPM.

Builds the resolved blade mesh (``auraflow.body.blade``), spins it with
``SpinMotion``, optionally computes the tiny thickness-noise signal at two
microphones for the strip chart, and streams everything to the browser viewer
(``viz-live`` extra). Loops until Ctrl-C.

Run (small dev box safe at the defaults):

    systemd-run --user --scope -q -p MemoryMax=1100M -p MemorySwapMax=0 -- \
        uv run --extra viz-live python scripts/viz_rotor_spin.py

then open the printed URL. Browsing from another machine: pass
``--host 0.0.0.0`` (or ssh -L 8000:localhost:8000 into this box).
"""

from __future__ import annotations

import argparse


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n-span", type=int, default=8, help="blade radial stations (mesh)")
    p.add_argument("--n-chord", type=int, default=10, help="chordwise profile points")
    p.add_argument("--rpm", type=float, default=None, help="rotor speed [RPM]; default hover")
    p.add_argument("--revs", type=float, default=2.0, help="revolutions per replay loop")
    p.add_argument("--n-times", type=int, default=240, help="animation time samples")
    p.add_argument(
        "--no-acoustics", action="store_true", help="skip the mic thickness-noise strip chart"
    )
    p.add_argument("--host", type=str, default="127.0.0.1", help="bind interface")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--check", action="store_true", help="build everything and exit (no server)")
    return p


def main() -> int:
    args = _parser().parse_args()

    import math

    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", True)

    from auraflow.body.motion import SpinMotion
    from auraflow.core.medium import Medium
    from auraflow.datasets.nasa_1pax import HOVER_RPM, nasa_1pax_rotor_mesh

    rpm = HOVER_RPM if args.rpm is None else args.rpm
    omega = rpm * 2.0 * math.pi / 60.0
    mesh = nasa_1pax_rotor_mesh(n_span=args.n_span, n_chord=args.n_chord, hub=True)
    motion = SpinMotion.constant(axis=np.array([0.0, 0.0, 1.0]), omega=omega, center=np.zeros(3))
    duration = args.revs * 2.0 * math.pi / omega
    tau = np.linspace(0.0, duration, args.n_times)
    print(
        f"[rotor-viz] {len(mesh.faces)} faces, {rpm:.0f} RPM "
        f"({args.revs:g} rev / {duration:.3f} s loop)"
    )

    mics = mic_p = mic_t = None
    if not args.no_acoustics:
        from auraflow.body.sources import mesh_pressure

        medium = Medium()
        mics = np.array([[6.0, 0.0, -2.0], [0.0, 6.0, 2.0]])
        p, t_obs = mesh_pressure(
            mesh, motion, jax.numpy.asarray(tau), jax.numpy.asarray(mics), medium
        )
        mic_p, mic_t = np.asarray(p), np.asarray(t_obs)
        rms = np.sqrt(np.mean(mic_p**2, axis=1))
        print(f"[rotor-viz] mic thickness-noise RMS: {rms} Pa")

    if args.check:
        print("[rotor-viz] check OK")
        return 0

    import time

    from auraflow.viz import VizStreamer
    from auraflow.viz.body import stream_body

    with VizStreamer(host=args.host, port=args.port) as viz:
        print(f"[rotor-viz] open  {viz.http_url}  to watch the rotor.")
        try:
            while True:
                stream_body(
                    viz,
                    mesh,
                    motion,
                    tau,
                    mics=mics,
                    mic_signals=mic_p,
                    mic_t=mic_t,
                    fps=30.0,
                    opacity=0.95,
                    color=(0.75, 0.8, 0.95),
                    title="NASA 1-Pax rotor (hover)",
                )
                time.sleep(0.3)
        except KeyboardInterrupt:
            print("\n[rotor-viz] bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
