#!/usr/bin/env python
"""Radiate a loudspeaker to a few listeners and record per-listener audio.

Builds a baffled circular piston (default) or loads a cabinet mesh (``--mesh``),
drives its membrane with a generated linear chirp (default) or a provided WAV
(``--wav``), propagates the thickness (monopole) radiation to a handful of
listeners with the mesh FW-H path (:meth:`auraflow.body.Speaker.play`), writes
one WAV per listener to ``results/speaker/``, and prints each listener's OASPL.

Sized for the low-RAM dev box: a short chirp, a coarse membrane, a few
listeners. With ``--viz`` it also streams the speaker mesh + listener levels to
the live browser viewer (needs the ``viz-live`` extra).

Example
-------
    systemd-run --user --scope -q -p MemoryMax=1100M -p MemorySwapMax=0 -- \
        uv run python scripts/speaker_demo.py --fs 8000 --duration 0.03
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from auraflow.body import Speaker, load_mesh  # noqa: E402
from auraflow.core.medium import Medium  # noqa: E402
from auraflow.signal.spectra import oaspl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mesh", type=str, default=None, help="cabinet mesh; default a piston")
    p.add_argument(
        "--membrane-axis",
        default="z",
        choices=["x", "y", "z"],
        help="cabinet face whose +normal side is the membrane (for --mesh)",
    )
    p.add_argument("--wav", type=str, default=None, help="drive with this WAV instead of a chirp")
    p.add_argument("--fs", type=float, default=8000.0, help="chirp sample rate [Hz]")
    p.add_argument("--duration", type=float, default=0.03, help="chirp duration [s]")
    p.add_argument("--f0", type=float, default=300.0, help="chirp start frequency [Hz]")
    p.add_argument("--f1", type=float, default=3000.0, help="chirp end frequency [Hz]")
    p.add_argument("--radius", type=float, default=0.05, help="piston radius [m]")
    p.add_argument("--rings", type=int, default=4, help="piston radial rings (mesh resolution)")
    p.add_argument("--gain", type=float, default=0.05, help="cone-velocity gain [m/s per unit]")
    p.add_argument("--viz", action="store_true", help="stream the mesh + levels to the browser")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--outdir", type=str, default="results/speaker")
    return p.parse_args()


def _chirp(fs: float, duration: float, f0: float, f1: float) -> np.ndarray:
    """Unit-amplitude linear chirp, Hann-tapered to avoid onset transients."""
    n = max(int(round(fs * duration)), 16)
    t = np.arange(n) / fs
    k = (f1 - f0) / max(duration, 1e-9)
    sig = np.sin(2.0 * np.pi * (f0 * t + 0.5 * k * t * t))
    return sig * np.hanning(n)


def _load_wav(path: str) -> tuple[np.ndarray, float]:
    fs, data = wavfile.read(path)
    x = np.asarray(data, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(np.asarray(data).dtype, np.integer):
        x = x / float(np.iinfo(np.asarray(data).dtype).max)
    return x, float(fs)


def _write_wav(path: Path, fs: float, signal: np.ndarray) -> None:
    peak = float(np.max(np.abs(signal))) or 1.0
    pcm = np.clip(signal / peak, -1.0, 1.0)
    wavfile.write(str(path), int(round(fs)), (pcm * 32767.0).astype(np.int16))


def main() -> None:
    args = parse_args()
    medium = Medium()

    if args.wav is not None:
        audio, fs = _load_wav(args.wav)
        print(f"  Loaded {args.wav}: {audio.shape[0]} samples @ {fs:.0f} Hz")
    else:
        fs = args.fs
        audio = _chirp(fs, args.duration, args.f0, args.f1)
        print(f"  Chirp {args.f0:.0f}->{args.f1:.0f} Hz: {audio.shape[0]} samples @ {fs:.0f} Hz")

    if args.mesh is not None:
        mesh = load_mesh(args.mesh)
        ax = {"x": 0, "y": 1, "z": 2}[args.membrane_axis]
        cent = np.asarray(mesh.centroids())
        thresh = float(cent[:, ax].max()) - 1e-6
        speaker = Speaker.from_mesh(mesh, lambda c, a=ax, th=thresh: c[:, a] >= th, baffled=False)
        print(f"  Cabinet mesh: {mesh.n_faces} faces, {len(speaker.membrane_faces)} membrane faces")
    else:
        speaker = Speaker.circular_piston(args.radius, args.rings, baffled=True)
        print(f"  Baffled circular piston r={args.radius} m, {speaker.enclosure.n_faces} faces")

    # A small arc of listeners in front of the membrane (+z), 1 m out.
    listeners = np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [0.0, 0.5, 1.0]])

    print("  Radiating (mesh FW-H, thickness sources)…")
    p, t_obs = speaker.play(audio, fs, listeners, medium, gain=args.gain)
    p = np.asarray(p)
    t_obs = np.asarray(t_obs)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    levels = np.asarray(oaspl(jnp.asarray(p), axis=-1))
    fs_obs = 1.0 / float(np.mean(np.diff(t_obs)))
    print("\n  Listener OASPLs (re 20 uPa):")
    for i, xo in enumerate(listeners):
        _write_wav(outdir / f"listener_{i}.wav", fs_obs, p[i])
        print(
            f"    listener {i} at {xo.tolist()} m:  {levels[i]:6.2f} dB   "
            f"(peak {np.max(np.abs(p[i])):.3e} Pa)"
        )
    print(f"\n  Wrote {listeners.shape[0]} WAVs to {outdir}/")

    if args.viz:
        import time

        from auraflow.body.motion import StaticPose
        from auraflow.viz import VizStreamer
        from auraflow.viz.body import stream_body

        with VizStreamer(port=args.port) as viz:
            print(f"\n  Open  {viz.http_url}  to watch the speaker + listener levels.")
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not viz.active:
                time.sleep(0.2)
            try:
                while True:
                    stream_body(
                        viz,
                        speaker.enclosure,
                        StaticPose(),
                        t_obs,
                        mics=listeners,
                        mic_signals=p,
                        mic_t=t_obs,
                        fps=30.0,
                        opacity=0.7,
                        color=(0.4, 0.7, 1.0),
                        title="AuraFlow speaker",
                    )
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n  Bye.")


if __name__ == "__main__":
    main()
