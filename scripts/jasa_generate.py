#!/usr/bin/env python
r"""Generate the JASA NASA-1Pax flyover dataset with the CONA backend.

Reproduces the data-generation half of Lee, Ko, Seshadri & Rauleder, JASA
159(4):3418-3435 (2026) -- see ``docs/research/jasa-datagen-reference.md``. A
NASA 1-Pax quadrotor flies a level straight line over a ground microphone array;
each mic records a 44.1 kHz pressure time series (convective FW-H tonal +
Griffin-Lim auralised BPM broadband). One ``.npz`` (+ per-mic WAVs) is written
per flyover under ``--out``.

Scales
------
- ``--smoke``: a **tiny** local run (few kHz, 2 mics, coarse grids, short
  duration). Runs in ~1 s on CPU and a few hundred MB of RAM -- safe on the
  small dev box. Use it to prove the pipeline end-to-end.
- default / full: the paper's 256-mic array at 44.1 kHz x 1 s. This is
  **GPU/omnirun work** (112.9M samples/case); do NOT run it on the dev box.

Full-scale generation (intended omnirun invocation once backends are wired)
---------------------------------------------------------------------------
    omnirun --backend slurm --gpus 1 -- \
        uv run --extra data python scripts/jasa_generate.py \
            --speeds 6 7 8 9 10 --altitudes 30 --seeds 0 1 2 \
            --out results/jasa

Committing to dload (needs the ``data`` extra + R2 creds in ``.env``; may run
in-job or afterwards from the saved outputs):
    # in the same job, right after generation:
    ... scripts/jasa_generate.py ... --commit-dload jasa-flyovers
    # or later, from the saved results directory (no regeneration):
    uv run --extra data python scripts/jasa_generate.py \
        --commit-from results/jasa --commit-dload jasa-flyovers
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--speeds",
        type=float,
        nargs="+",
        default=[6.0, 8.0, 10.0],
        help="Ground speeds V_inf [m/s] (paper GP set: 6-10).",
    )
    p.add_argument(
        "--altitudes", type=float, nargs="+", default=[30.0], help="Flight altitudes [m]."
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="PRNG seeds (gust + Griffin-Lim phase realisations).",
    )
    p.add_argument("--duration", type=float, default=1.0, help="Signal duration [s].")
    p.add_argument("--fs", type=float, default=44100.0, help="Audio sample rate [Hz].")
    p.add_argument(
        "--gust",
        default="0.0",
        help="Dryden wind at 20 ft [m/s] or preset (light/moderate/severe); 0 = calm.",
    )
    p.add_argument("--out", default="results/jasa", help="Output directory.")
    p.add_argument("--no-broadband", action="store_true", help="Tonal only (skip BPM broadband).")
    p.add_argument("--n-stations", type=int, default=16, help="Radial blade stations.")
    p.add_argument("--n-frames", type=int, default=48, help="Broadband spectrogram frames.")
    p.add_argument("--n-fft", type=int, default=2048, help="Griffin-Lim STFT length.")
    p.add_argument("--gl-iters", type=int, default=60, help="Griffin-Lim iterations.")
    p.add_argument("--obs-chunk", type=int, default=16, help="Mics per propagation batch.")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny local run (few kHz, 2 mics, coarse grids) -- safe on a small box.",
    )
    p.add_argument(
        "--low-memory",
        action="store_true",
        help="Clear XLA compile caches between pipeline stages (halves peak RAM, "
        "costs recompiles). Implied by --smoke.",
    )
    p.add_argument("--limit", type=int, default=None, help="Only generate the first N scenarios.")
    p.add_argument(
        "--commit-dload",
        metavar="NAME",
        default=None,
        help="After generating, commit the flyovers to dload dataset NAME "
        "(needs 'data' extra + creds).",
    )
    p.add_argument(
        "--commit-from",
        metavar="DIR",
        default=None,
        help="Skip generation; commit existing .npz flyovers under DIR.",
    )
    p.add_argument(
        "--no-arrays",
        action="store_true",
        help="Omit the lossless float32 arrays field from dload samples.",
    )
    return p


def _gust_arg(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def _smoke_overrides(args: argparse.Namespace) -> None:
    """Shrink everything to a ~1 s CPU / few-hundred-MB run."""
    args.low_memory = True
    args.fs = 4000.0
    args.duration = 0.25
    args.n_stations = 6
    args.n_frames = 12
    args.n_fft = 256
    args.gl_iters = 8
    args.obs_chunk = 2
    if args.speeds == [6.0, 8.0, 10.0]:
        args.speeds = [8.0]
    if args.seeds == [0]:
        args.seeds = [0]


def _smoke_mics():
    import jax.numpy as jnp

    # Two ground mics near the flight path: one ahead, one behind the pass point.
    return jnp.asarray([[-30.0, 0.0, 0.0], [40.0, 20.0, 0.0]])


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # --- Commit-only path (no generation) ------------------------------------
    if args.commit_from is not None:
        if not args.commit_dload:
            print("error: --commit-from requires --commit-dload NAME", file=sys.stderr)
            return 2
        from auraflow.datasets.dload_io import commit_flyovers, results_from_dir

        results = results_from_dir(args.commit_from)
        print(f"committing flyovers from {args.commit_from} -> dload '{args.commit_dload}' ...")
        manifest = commit_flyovers(
            args.commit_dload,
            results,
            meta={"generator": "scripts/jasa_generate.py", "commit_from": args.commit_from},
            recipe=" ".join(sys.argv),
            include_arrays=not args.no_arrays,
        )
        print(f"committed: {manifest}")
        return 0

    if args.smoke:
        _smoke_overrides(args)

    # The CONA pipeline (retarded-time FW-H, Griffin-Lim) is designed for
    # float64; enable x64 before any array is created (matches the test conftest
    # and scripts/cfd_pulse_validation.py). Done here, not at import, so the
    # --help / --commit-from paths stay JAX-free.
    import jax

    jax.config.update("jax_enable_x64", True)

    # Imports here (not at top) so --help is instant and cheap.
    from auraflow.datasets.jasa import generate_flyover, generate_scenario_grid, save_flyover

    mics = _smoke_mics() if args.smoke else None
    scenarios = generate_scenario_grid(
        speeds=args.speeds,
        altitudes=args.altitudes,
        seeds=args.seeds,
        duration=args.duration,
        fs=args.fs,
        gust_w20=_gust_arg(args.gust),
        mics=mics,
    )
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    os.makedirs(args.out, exist_ok=True)
    print(
        f"generating {len(scenarios)} flyover(s) -> {args.out} "
        f"(fs={args.fs:g} Hz, dur={args.duration:g} s, broadband={not args.no_broadband})"
    )

    results = []
    for i, sc in enumerate(scenarios):
        from auraflow.datasets.jasa import scenario_id

        t0 = time.perf_counter()
        result = generate_flyover(
            sc,
            n_stations=args.n_stations,
            n_frames=args.n_frames,
            n_fft=args.n_fft,
            gl_iters=args.gl_iters,
            include_broadband=not args.no_broadband,
            obs_chunk=args.obs_chunk,
            low_memory=args.low_memory,
        )
        stem = os.path.join(args.out, scenario_id(sc))
        paths = save_flyover(result, stem)
        dt = time.perf_counter() - t0
        n = result["audio"].shape
        print(
            f"  [{i + 1}/{len(scenarios)}] {scenario_id(sc)}  audio{tuple(n)}  "
            f"{dt:.2f}s -> {os.path.basename(paths['npz'])}"
        )
        if args.commit_dload:
            results.append(result)

    if args.commit_dload:
        from auraflow.datasets.dload_io import commit_flyovers

        print(f"committing {len(results)} flyover(s) -> dload '{args.commit_dload}' ...")
        manifest = commit_flyovers(
            args.commit_dload,
            results,
            meta={"generator": "scripts/jasa_generate.py", "n_flyovers": len(results)},
            recipe=" ".join(sys.argv),
            include_arrays=not args.no_arrays,
        )
        print(f"committed: {manifest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
