#!/usr/bin/env python
r"""Generate the DREGON + Michael (Matrice 100) onboard drone ego-noise dataset.

Auralizes the ego-noise heard by 64 microphones placed all around each drone
(two spherical shells; :func:`auraflow.datasets.drone_egonoise.onboard_mic_array`)
while it hovers, using the CONA backend -- see
:mod:`auraflow.datasets.drone_egonoise`. One ``.npz`` (+ per-mic WAVs) is written
per (drone, seed) case under ``--out``.

Scales
------
- ``--smoke``: a tiny local run (few kHz, 4 mics, coarse grids, short duration).
  Runs in a few seconds on CPU under a few hundred MB -- safe on the small dev
  box (wrap in a systemd-run MemoryMax cap per CLAUDE.md).
- default / full: 64 mics at 44.1 kHz x 1 s per case. This is GPU/omnirun work;
  do NOT run it on the dev box.

Full-scale generation (omnirun) + commit to dload
-------------------------------------------------
    omnirun --backend slurm --gpus 1 -- \
        uv run --extra data python scripts/egonoise_generate.py \
            --drones dregon matrice100 --seeds 0 1 2 \
            --out results/egonoise --commit-dload drone-egonoise

Committing later from saved outputs (no regeneration):
    uv run --extra data python scripts/egonoise_generate.py \
        --commit-from results/egonoise --commit-dload drone-egonoise
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--drones",
        nargs="+",
        default=["dregon", "matrice100"],
        help="Drone keys to generate (auraflow.datasets.drone_egonoise.DRONES).",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="PRNG seeds.")
    p.add_argument("--altitude", type=float, default=10.0, help="Hover altitude [m].")
    p.add_argument("--duration", type=float, default=1.0, help="Signal duration [s].")
    p.add_argument("--fs", type=float, default=44100.0, help="Audio sample rate [Hz].")
    p.add_argument("--n-mics", type=int, default=64, help="Onboard microphone count.")
    p.add_argument("--out", default="results/egonoise", help="Output directory.")
    p.add_argument("--no-broadband", action="store_true", help="Tonal only (skip BPM broadband).")
    p.add_argument("--n-stations", type=int, default=16, help="Radial blade stations.")
    p.add_argument("--n-frames", type=int, default=48, help="Broadband spectrogram frames.")
    p.add_argument("--n-fft", type=int, default=2048, help="Griffin-Lim STFT length.")
    p.add_argument("--gl-iters", type=int, default=60, help="Griffin-Lim iterations.")
    p.add_argument("--obs-chunk", type=int, default=16, help="Mics per propagation batch.")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny local run (few kHz, 4 mics, coarse grids) -- safe on a small box.",
    )
    p.add_argument(
        "--low-memory",
        action="store_true",
        help="Clear XLA compile caches between pipeline stages (implied by --smoke).",
    )
    p.add_argument(
        "--commit-dload",
        metavar="NAME",
        default=None,
        help="After generating, commit to dload dataset NAME (needs 'data' extra + creds).",
    )
    p.add_argument(
        "--commit-from",
        metavar="DIR",
        default=None,
        help="Skip generation; commit existing .npz ego-noise cases under DIR.",
    )
    p.add_argument(
        "--no-arrays",
        action="store_true",
        help="Omit the lossless float32 arrays field from dload samples.",
    )
    return p


def _smoke_overrides(args: argparse.Namespace) -> None:
    """Shrink everything to a ~few-second CPU / few-hundred-MB run."""
    args.low_memory = True
    args.fs = 4000.0
    args.duration = 0.25
    args.n_mics = 4
    args.n_stations = 6
    args.n_frames = 12
    args.n_fft = 256
    args.gl_iters = 8
    args.obs_chunk = 2


def _load_result_npz(path: str) -> dict:
    """Load a :func:`save_egonoise` ``.npz`` back to a commit-ready result dict."""
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        return {
            "audio": np.asarray(data["audio"]),
            "tonal": np.asarray(data["tonal"]),
            "broadband": np.asarray(data["broadband"]),
            "mics": np.asarray(data["mics"]),
            "mics_body": np.asarray(data["mics_body"]),
            "band_centers": np.asarray(data["band_centers"]),
            "meta": meta,
            "key": os.path.splitext(os.path.basename(path))[0],
        }


def _commit(name: str, results, recipe: str, include_arrays: bool, provenance: dict) -> None:
    from auraflow.datasets.egonoise_io import commit_egonoise

    print(f"committing {provenance} -> dload '{name}' ...")
    manifest = commit_egonoise(
        name, results, meta=provenance, recipe=recipe, include_arrays=include_arrays
    )
    print(f"committed: {manifest}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # --- Commit-only path (no generation) ------------------------------------
    if args.commit_from is not None:
        if not args.commit_dload:
            print("error: --commit-from requires --commit-dload NAME", file=sys.stderr)
            return 2
        paths = sorted(glob.glob(os.path.join(args.commit_from, "*.npz")))
        results = (_load_result_npz(p) for p in paths)
        _commit(
            args.commit_dload,
            results,
            recipe=" ".join(sys.argv),
            include_arrays=not args.no_arrays,
            provenance={
                "generator": "scripts/egonoise_generate.py",
                "commit_from": args.commit_from,
            },
        )
        return 0

    if args.smoke:
        _smoke_overrides(args)

    # CONA is designed for float64; enable x64 before any array is created.
    import jax

    jax.config.update("jax_enable_x64", True)

    from auraflow.datasets.drone_egonoise import generate_egonoise, save_egonoise

    cases = [(d, s) for d in args.drones for s in args.seeds]
    os.makedirs(args.out, exist_ok=True)
    print(
        f"generating {len(cases)} ego-noise case(s) -> {args.out} "
        f"(fs={args.fs:g} Hz, dur={args.duration:g} s, n_mics={args.n_mics}, "
        f"broadband={not args.no_broadband})"
    )

    results = []
    for i, (drone, seed) in enumerate(cases):
        t0 = time.perf_counter()
        result = generate_egonoise(
            drone,
            altitude=args.altitude,
            duration=args.duration,
            fs=args.fs,
            seed=int(seed),
            n_mics=args.n_mics,
            n_stations=args.n_stations,
            include_broadband=not args.no_broadband,
            low_memory=args.low_memory,
            obs_chunk=args.obs_chunk,
            n_frames=args.n_frames,
            n_fft=args.n_fft,
            gl_iters=args.gl_iters,
        )
        stem = os.path.join(args.out, result["key"])
        paths = save_egonoise(result, stem)
        dt = time.perf_counter() - t0
        n = result["audio"].shape
        print(
            f"  [{i + 1}/{len(cases)}] {result['key']}  audio{tuple(n)}  "
            f"{dt:.2f}s -> {os.path.basename(paths['npz'])}"
        )
        if args.commit_dload:
            results.append(result)

    if args.commit_dload:
        _commit(
            args.commit_dload,
            results,
            recipe=" ".join(sys.argv),
            include_arrays=not args.no_arrays,
            provenance={"generator": "scripts/egonoise_generate.py", "n_cases": len(results)},
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
