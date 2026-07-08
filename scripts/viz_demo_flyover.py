#!/usr/bin/env python
"""Live 3-D replay of a CONA quadrotor flyover in the browser.

The CONA pipeline is batch (trajectory, airloads, and per-mic audio are computed
up front), so this **replays** a finished smoke-scale flyover as an animation:
the vehicle flies its straight line over the ground microphone array, rotor disks
spin at the simulated speeds, and a strip chart tracks the pressure at two mics.

Sized for the low-RAM dev box: short duration, low sample rate, few stations,
``low_memory=True``, broadband off by default. Requires the ``viz-live`` extra
(and the base install for the CONA compute).

Example
-------
    systemd-run --user --scope -q -p MemoryMax=1100M -p MemorySwapMax=0 -- \
        uv run --extra viz-live python scripts/viz_demo_flyover.py

Open the printed URL, then use Live / Pause / the scrubber to replay.
"""

from __future__ import annotations

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from auraflow.cona.flight import ControllerGains, simulate, straight_flyover  # noqa: E402
from auraflow.datasets.nasa_1pax import nasa_1pax_multirotor, nasa_1pax_vehicle  # noqa: E402
from auraflow.viz import VizStreamer  # noqa: E402
from auraflow.viz.flyover import stream_flyover  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--speed", type=float, default=8.0, help="ground speed [m/s]")
    p.add_argument("--altitude", type=float, default=30.0)
    p.add_argument("--duration", type=float, default=4.0, help="flight time [s]")
    p.add_argument("--n-times", type=int, default=240, help="flight-sim time samples")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--n-stations", type=int, default=8)
    p.add_argument("--loop", action="store_true", help="replay on a loop")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # A sparse ground mic line under the track (kept small for the demo).
    mics = np.stack([np.linspace(-80.0, 100.0, 19), np.zeros(19), np.zeros(19)], axis=-1)

    vehicle = nasa_1pax_vehicle(args.n_stations)
    mrotor = nasa_1pax_multirotor()
    gains = ControllerGains.for_vehicle(mrotor)
    t = np.linspace(0.0, args.duration, args.n_times)
    ref = straight_flyover(args.speed, args.altitude, 0.0, t_pass=args.duration / 2)
    x0, v0, _, _ = ref(jnp.asarray(0.0))
    print("  Running the 6-DOF flight sim…")
    flight = simulate(mrotor, gains, ref, t, x0, v0)

    # A synthetic per-mic pressure proxy for the strip chart: the flyover
    # amplitude envelope (1/r) modulated at blade-passing frequency. (A full
    # auralization is heavier than the dev box wants; the geometry/animation is
    # the point of this demo.)
    from auraflow.datasets.nasa_1pax import BPF_HZ

    xw = np.asarray(flight.x)
    n = 2000
    st = np.linspace(0.0, args.duration, n)
    xt = np.stack([np.interp(st, t, xw[:, k]) for k in range(3)], axis=-1)
    sig = np.zeros((mics.shape[0], n))
    for m in range(mics.shape[0]):
        r = np.linalg.norm(xt - mics[m], axis=-1)
        sig[m] = np.sin(2 * np.pi * BPF_HZ * st) / np.maximum(r, 1.0) ** 2 * 50.0

    with VizStreamer(host=args.host, port=args.port) as viz:
        print(f"\n  Open  {viz.http_url}  in a browser to watch the flyover.\n")
        print("  Waiting up to 10s for a viewer…")
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not viz.active:
            time.sleep(0.2)
        try:
            while True:
                stream_flyover(
                    viz, vehicle, flight, mics=mics, mic_signals=sig, mic_t=st, fps=args.fps
                )
                if not args.loop:
                    break
            print("  Replay complete. Serving buffer — Ctrl-C to exit.")
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n  Bye.")


if __name__ == "__main__":
    main()
