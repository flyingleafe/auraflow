#!/usr/bin/env python
"""Live in-browser 3-D visualization of a running CFD acoustic case.

Starts the AuraFlow viz hub (HTTP + WebSocket) in-process, then marches a tiny
Gaussian-pressure-pulse box case, streaming a downsampled mid-plane field slice
and the permeable-sphere overpressure to the browser **as the simulation runs**.
Open the printed URL, then watch the pulse expand and hit the sphere live.

Sized to run on the low-RAM dev box: a small 2-D (or tiny 3-D) grid, few steps.
Requires the ``cfd`` (jaxfluids) and ``viz-live`` (websockets) extras.

Example
-------
    systemd-run --user --scope -q -p MemoryMax=1100M -p MemorySwapMax=0 -- \
        uv run --extra cfd --extra viz-live \
        python scripts/viz_demo_cfd.py --cells 64 64 1 --steps 400

Runs until the march finishes, then keeps serving until Ctrl-C so you can scrub
the replay buffer.
"""

from __future__ import annotations

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

from auraflow.cfd.case import acoustic_box_case  # noqa: E402
from auraflow.cfd.run import run_acoustic_case  # noqa: E402
from auraflow.cfd.sphere import PermeableSphere  # noqa: E402
from auraflow.core.medium import Medium  # noqa: E402
from auraflow.viz import VizStreamer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--cells", type=int, nargs=3, default=(64, 64, 1), metavar=("NX", "NY", "NZ"))
    p.add_argument("--half-size", type=float, default=0.5, help="box half-edge [m]")
    p.add_argument("--sphere-radius", type=float, default=0.25)
    p.add_argument("--sphere-points", type=int, default=200)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--sample-every", type=int, default=2)
    p.add_argument("--pulse-width", type=float, default=0.05)
    p.add_argument(
        "--wait", type=float, default=8.0, help="seconds to wait for a browser before marching"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    medium = Medium()
    case = acoustic_box_case(
        medium,
        half_size=args.half_size,
        cells=tuple(args.cells),
        pulse=True,
        pulse_width=args.pulse_width,
    )
    sphere = PermeableSphere.fibonacci(args.sphere_points, args.sphere_radius, (0.0, 0.0, 0.0))

    with VizStreamer(host=args.host, port=args.port) as viz:
        print(f"\n  Open  {viz.http_url}  in a browser to watch the simulation.\n")
        if args.wait > 0:
            print(f"  Waiting up to {args.wait:.0f}s for a viewer to connect…")
            deadline = time.monotonic() + args.wait
            while time.monotonic() < deadline and not viz.active:
                time.sleep(0.2)
            print("  Viewer connected." if viz.active else "  No viewer yet; marching anyway.")

        print(f"  Marching {args.steps} steps (sampling every {args.sample_every})…")
        run_acoustic_case(case, sphere, n_steps=args.steps, sample_every=args.sample_every, viz=viz)
        print("  Simulation complete. Serving replay buffer — Ctrl-C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n  Bye.")


if __name__ == "__main__":
    main()
