#!/usr/bin/env python
r"""CONA flyover of the DJI Phantom (9450 rotors) -- a consumer-drone auralization.

The small-vehicle counterpart of ``scripts/jasa_generate.py``: a DJI Phantom
quadrotor (four two-bladed DJI 9450 propellers, ``docs/research/
dji-9450-reference.md``) flies a level straight line over a ground microphone at
drone-realistic parameters (2 s, ~5 m/s, ~15 m altitude), and each mic records a
44.1 kHz pressure time series (convective FW-H tonal + Griffin-Lim auralised BPM
broadband). The tonal comb sits on the blade-passing frequency
(``BPF ~ 180-190 Hz`` at hover) with audible harmonics, so the result actually
sounds like a consumer drone. One ``.npz`` (+ per-mic WAVs) is written to
``--out``.

This reuses the generalised :func:`auraflow.datasets.jasa.generate_flyover`
(vehicle/multirotor/collective/bpf overrides) with the DJI Phantom vehicle from
:mod:`auraflow.datasets.dji_phantom` -- the single source of truth for the blade
geometry, hover trim and hub layout.

Scales
------
- ``--smoke``: a **tiny** local run (few kHz, 2 mics, coarse grids, short
  duration). Runs in ~1 s on CPU and a few hundred MB of RAM -- safe on the small
  dev box (memory-capped). Proves the pipeline end-to-end and prints OASPL + BPF.
- default / full: 44.1 kHz x 2 s at the requested mics. **GPU/omnirun work**; do
  NOT run the full mode on the dev box.

    # tiny local proof (memory-capped, see repo CLAUDE.md):
    systemd-run --user --scope -q -p MemoryMax=1100M -p MemorySwapMax=0 -- \
        uv run python scripts/drone_flyover_demo.py --smoke

    # full auralization on GPU (omnirun):
    omnirun --backend slurm --gpus 1 -- \
        uv run python scripts/drone_flyover_demo.py \
            --speed 5 --altitude 15 --duration 2.0 --gust light --out results/drone
"""

from __future__ import annotations

import argparse
import os
import time


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--speed", type=float, default=5.0, help="Ground speed [m/s] (drone-realistic).")
    p.add_argument("--altitude", type=float, default=15.0, help="Flight altitude [m].")
    p.add_argument("--duration", type=float, default=2.0, help="Signal duration [s].")
    p.add_argument("--fs", type=float, default=44100.0, help="Audio sample rate [Hz].")
    p.add_argument("--lateral-offset", type=float, default=0.0, help="Track lateral offset y [m].")
    p.add_argument("--seed", type=int, default=0, help="PRNG seed (gust + Griffin-Lim phases).")
    p.add_argument(
        "--gust",
        default="0.0",
        help="Per-rotor RPM beating realism: Dryden wind at 20 ft [m/s] or preset "
        "(light/moderate/severe); 0 = calm. 'light' gives audible drone RPM beating.",
    )
    p.add_argument("--out", default="results/drone", help="Output directory.")
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
    args.duration = 0.3
    args.n_stations = 6
    args.n_frames = 12
    args.n_fft = 256
    args.gl_iters = 8
    args.obs_chunk = 2


def _smoke_mics():
    import jax.numpy as jnp

    # Two ground mics near the flight path: one ahead, one abeam the pass point.
    return jnp.asarray([[-15.0, 0.0, 0.0], [10.0, 8.0, 0.0]])


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        _smoke_overrides(args)

    # The CONA pipeline (retarded-time FW-H, Griffin-Lim) is float64 throughout;
    # enable x64 before any array is created (matches the test conftest).
    import jax

    jax.config.update("jax_enable_x64", True)

    from auraflow.datasets.dji_phantom import (
        BPF_HZ,
        dji_phantom_hover_collective,
        dji_phantom_multirotor,
        dji_phantom_polar,
        dji_phantom_vehicle,
    )
    from auraflow.datasets.jasa import JASAScenario, generate_flyover, save_flyover

    medium = None  # ISA sea level
    polar = dji_phantom_polar()
    # A light gust makes each rotor's speed controller work, so the four rotors
    # beat slightly against each other -- the RPM beating that a real quad has.
    drag = 0.0 if args.gust in (0.0, "0.0") else 1.0
    multirotor = dji_phantom_multirotor(drag_coeff=drag)
    vehicle = dji_phantom_vehicle(args.n_stations)

    t0 = time.perf_counter()
    collective = dji_phantom_hover_collective(args.n_stations, polar=polar)
    mics = _smoke_mics() if args.smoke else None
    scenario = JASAScenario(
        speed=args.speed,
        altitude=args.altitude,
        lateral_offset=args.lateral_offset,
        duration=args.duration,
        fs=args.fs,
        seed=args.seed,
        gust_w20=_gust_arg(args.gust),
        mics=mics,
    )

    os.makedirs(args.out, exist_ok=True)
    print(
        f"DJI Phantom flyover: V={args.speed:g} m/s, alt={args.altitude:g} m, "
        f"dur={args.duration:g} s, fs={args.fs:g} Hz, BPF={BPF_HZ:.1f} Hz, "
        f"gust={args.gust} (broadband={not args.no_broadband})"
    )

    result = generate_flyover(
        scenario,
        medium=medium,
        polar=polar,
        collective=collective,
        vehicle=vehicle,
        multirotor=multirotor,
        bpf_hz=BPF_HZ,
        n_stations=args.n_stations,
        n_frames=args.n_frames,
        n_fft=args.n_fft,
        gl_iters=args.gl_iters,
        include_broadband=not args.no_broadband,
        obs_chunk=args.obs_chunk,
        low_memory=args.low_memory,
    )

    stem = os.path.join(args.out, "drone_flyover")
    paths = save_flyover(result, stem)
    dt = time.perf_counter() - t0

    # OASPL per mic + the spectral peak on the BPF comb (mic 0).
    import numpy as np

    audio = np.asarray(result["audio"])
    rms = np.sqrt(np.mean(audio**2, axis=1))
    oaspl = 20.0 * np.log10(np.maximum(rms, 1e-12) / 2e-5)
    p0 = audio[0] - audio[0].mean()
    spec = np.abs(np.fft.rfft(p0))
    freqs = np.fft.rfftfreq(p0.size, 1.0 / float(args.fs))
    f_peak = float(freqs[int(np.argmax(spec))])
    n_harm = max(round(f_peak / BPF_HZ), 1)
    print(
        f"  audio{tuple(audio.shape)}  {dt:.2f}s -> {os.path.basename(paths['npz'])}\n"
        f"  OASPL[dB] per mic: {np.array2string(oaspl, precision=1)}\n"
        f"  mic0 spectral peak {f_peak:.1f} Hz (BPF {BPF_HZ:.1f} Hz, "
        f"harmonic #{n_harm} at {n_harm * BPF_HZ:.1f} Hz)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
