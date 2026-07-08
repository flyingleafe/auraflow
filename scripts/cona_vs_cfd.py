#!/usr/bin/env python
r"""CONA vs full-CFD+FW-H comparison for a hovering NASA-1Pax rotor.

The JASA dataset (``scripts/jasa_generate.py``) is generated with the fast CONA
backend. This script quantifies how far that surrogate sits from the reference
full-CFD + permeable-surface FW-H backend (:mod:`auraflow.cfd`) at a *matched*
operating condition, and how the agreement tightens as the CFD grid is refined.

Condition
---------
A single isolated 1-Pax rotor (radius ``R``, thrust ``m g / 4``) hovering at the
origin. The CONA side auralises tonal (convective FW-H) + broadband (BPM) noise
at a polar arc of observers; the CFD side runs an **actuator-disk** rotor box
(:func:`auraflow.cfd.rotor_box_case`), samples a static permeable sphere
(:func:`auraflow.cfd.PermeableSphere`), and propagates to the same observers
(:func:`auraflow.cfd.propagate_to_observers`). The two are compared on a shared
1/3-octave band set (:func:`auraflow.datasets.compare.compare_cona_vs_cfd`).

Physics caveat (documented, not silent): the steady actuator disk carries **no
blade-passage tone**, so the BPF line is a CONA-only feature; the comparison is
therefore meaningful for the *broadband* band spectrum and OASPL, not the tonal
peak. Resolving the tone in CFD needs ``method="levelset_blades"`` (a stub in
:func:`rotor_box_case`) and is future GPU work.

Scales
------
- ``--dry``: NO JAX-Fluids. A synthetic :class:`~auraflow.cfd.run.SurfaceHistory`
  stands in for the CFD, so the whole CONA + comparison + plotting path runs
  locally in seconds (exercises the plumbing, not the physics). Use this to
  smoke-test the script on the dev box.
- ``--smoke``: a single 32^3 CFD grid + tiny CONA. Still real JAX-Fluids, so it
  needs the ``cfd`` extra and is slow on CPU -- prefer a GPU.
- default / full: a resolution sweep (48^3, 64^3, ...) at 44.1 kHz CONA. This is
  **GPU/omnirun work**; do NOT run it on the dev box.

Full-scale invocation (intended omnirun once backends are wired)
----------------------------------------------------------------
    omnirun --backend slurm --gpus 1 -- \
        uv run --extra cfd --extra viz python scripts/cona_vs_cfd.py \
            --resolutions 48 64 96 --n-steps 4000 --sample-every 4 \
            --out results/compare

Output: ``results/compare/compare_<N>.npz`` per resolution (band spectra + diffs)
and, with the ``viz`` extra, ``results/compare/cona_vs_cfd.png`` (OASPL error vs
resolution + a band-level overlay).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[48, 64],
        help="CFD cube cell counts N (grid is N^3); the refinement sweep.",
    )
    p.add_argument(
        "--obs-radius", type=float, default=15.0, help="Observer arc radius from the hub [m]."
    )
    p.add_argument(
        "--n-observers", type=int, default=5, help="Observers on the polar arc (elevation sweep)."
    )
    p.add_argument("--box-radii", type=float, default=4.0, help="CFD half box edge in rotor radii.")
    p.add_argument("--n-sphere", type=int, default=200, help="Permeable-sphere sample points S.")
    p.add_argument(
        "--sphere-radii",
        type=float,
        default=1.5,
        help="Permeable-sphere radius in rotor radii (encloses the disk).",
    )
    p.add_argument("--n-steps", type=int, default=2000, help="CFD integration steps.")
    p.add_argument("--sample-every", type=int, default=2, help="Sample the sphere every k steps.")
    p.add_argument("--warmup-steps", type=int, default=0, help="CFD transient steps to discard.")
    p.add_argument("--tip-mach", type=float, default=0.4, help="Tip Mach (bounds the CFD dt).")
    # CONA reference knobs.
    p.add_argument("--fs", type=float, default=44100.0, help="CONA audio sample rate [Hz].")
    p.add_argument("--duration", type=float, default=1.0, help="CONA signal duration [s].")
    p.add_argument("--n-stations", type=int, default=16, help="Radial blade stations.")
    p.add_argument("--n-source-times", type=int, default=400, help="CONA source-time samples.")
    p.add_argument("--n-frames", type=int, default=48, help="Broadband spectrogram frames.")
    p.add_argument("--n-fft", type=int, default=2048, help="Griffin-Lim STFT length.")
    p.add_argument("--gl-iters", type=int, default=60, help="Griffin-Lim iterations.")
    p.add_argument("--fmin", type=float, default=50.0, help="Lowest comparison band centre [Hz].")
    p.add_argument("--fmax", type=float, default=5000.0, help="Highest comparison band centre.")
    p.add_argument("--seed", type=int, default=0, help="PRNG seed (Griffin-Lim phases).")
    p.add_argument("--out", default="results/compare", help="Output directory.")
    p.add_argument(
        "--dry",
        action="store_true",
        help="Use a synthetic SurfaceHistory instead of JAX-Fluids (no cfd extra).",
    )
    p.add_argument(
        "--smoke", action="store_true", help="Tiny single-32^3 run (still real CFD unless --dry)."
    )
    return p


def _smoke_overrides(args: argparse.Namespace) -> None:
    """Shrink to a single 32^3 grid + tiny CONA."""
    args.resolutions = [32]
    args.fs = 8000.0
    args.duration = 0.25
    args.n_stations = 6
    args.n_source_times = 120
    args.n_frames = 12
    args.n_fft = 256
    args.gl_iters = 8
    args.n_observers = 3
    args.n_sphere = 96
    args.n_steps = 200
    args.sample_every = 2


def hover_observers(radius: float, n: int) -> Any:
    """Polar arc of observers around the hub (elevation 15..90 deg in the x-z plane).

    Args:
        radius: Arc radius from the hub [m].
        n: Number of observers.

    Returns:
        Observer positions [m], shape ``[n, 3]`` (world frame, z up).
    """
    import jax.numpy as jnp

    elev = jnp.linspace(jnp.deg2rad(15.0), jnp.deg2rad(90.0), n)
    x = radius * jnp.cos(elev)
    z = -radius * jnp.sin(elev)  # below the rotor (ground side)
    return jnp.stack([x, jnp.zeros_like(x), z], axis=-1)


def cona_hover_audio(args: argparse.Namespace, observers: Any) -> dict[str, Any]:
    """Auralise a single 1-Pax rotor hovering at the origin at ``observers``.

    Mirrors :func:`auraflow.datasets.jasa.generate_flyover` for a *single* rotor
    in hover (a manually-built stationary :class:`~auraflow.cona.flight.FlightHistory`),
    so the source matches the single-disk CFD box.

    Returns:
        Dict with ``"audio"`` [O, n] [Pa], ``"tonal"``, ``"broadband"``, ``"fs"``.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from auraflow.cona.airloads import rotor_section_state
    from auraflow.cona.auralize import synthesize_observer_signal
    from auraflow.cona.broadband import rotor_broadband_spectrogram
    from auraflow.cona.flight import FlightHistory
    from auraflow.cona.tonal import cona_tonal_noise
    from auraflow.core.blade import Rotor, Vehicle
    from auraflow.core.medium import Medium
    from auraflow.datasets.nasa_1pax import (
        HOVER_OMEGA,
        N_BLADES,
        nasa_1pax_blade,
        nasa_1pax_hover_collective,
        nasa_1pax_polar,
    )
    from auraflow.signal.spectra import third_octave_bands

    medium = Medium()
    polar = nasa_1pax_polar()
    blade = nasa_1pax_blade(args.n_stations)
    rotor = Rotor(blade=blade, n_blades=N_BLADES, spin_direction=1)
    vehicle = Vehicle(rotors=(rotor,))
    collective = nasa_1pax_hover_collective(args.n_stations, medium, polar)

    n_t = int(args.n_source_times)
    t = jnp.linspace(0.0, float(args.duration), n_t)
    T3 = (n_t, 3)
    flight = FlightHistory(
        t=t,
        x=jnp.zeros(T3),
        v=jnp.zeros(T3),
        R=jnp.broadcast_to(jnp.eye(3), (n_t, 3, 3)),
        Omega_body=jnp.zeros(T3),
        rotor_speeds=jnp.full((n_t, 1), HOVER_OMEGA),
        rotor_thrusts=jnp.zeros((n_t, 1)),
    )

    fmax_eff = min(float(args.fmax), 0.45 * float(args.fs))
    bands, _ = third_octave_bands(args.fmin, fmax_eff)

    p_tonal, _, _, t_obs = cona_tonal_noise(
        vehicle,
        flight,
        observers,
        medium,
        collective=collective,
        polar=polar,
    )
    state = rotor_section_state(vehicle, flight, 0, medium, collective=collective, polar=polar)
    _, spec, _ = rotor_broadband_spectrogram(
        state,
        observers,
        medium,
        t,
        bands=bands,
        n_frames=args.n_frames,
    )  # [O, n_frames, n_bands]

    n_fft = min(int(args.n_fft), int(round(args.fs * args.duration)))
    key = jax.random.PRNGKey(int(args.seed))
    n_obs = int(observers.shape[0])
    audio = np.zeros((n_obs, int(round(args.fs * args.duration))))
    tonal = np.zeros_like(audio)
    broadband = np.zeros_like(audio)
    for o in range(n_obs):
        out = synthesize_observer_signal(
            float(args.fs),
            float(args.duration),
            tonal_pressure=p_tonal[o],
            tonal_t=t_obs,
            broadband_spectrograms=[spec[o]],
            band_centers=bands,
            n_fft=n_fft,
            n_iters=args.gl_iters,
            key=jax.random.fold_in(key, o),
        )
        audio[o] = np.asarray(out["total"])
        tonal[o] = np.asarray(out["tonal"])
        broadband[o] = np.asarray(out["broadband"])
    return {"audio": audio, "tonal": tonal, "broadband": broadband, "fs": float(args.fs)}


def _synthetic_surface_history(sphere: Any, n_t: int, medium: Any) -> Any:
    """A cheap analytic breathing-sphere surface history (for ``--dry``).

    Not physical rotor noise -- just a smooth radial monopole so
    :func:`propagate_to_observers` produces a finite non-zero far field and the
    comparison plumbing can be exercised without JAX-Fluids.
    """
    import jax.numpy as jnp

    from auraflow.cfd.run import SurfaceHistory

    s = int(sphere.points.shape[0])
    tau = jnp.linspace(0.0, 0.02, n_t)
    osc = jnp.sin(2.0 * jnp.pi * 200.0 * tau)  # [T]
    p = float(medium.p0) + 2.0 * osc[None, :] * jnp.ones((s, 1))  # [S, T]
    rho = float(medium.rho0) + (p - float(medium.p0)) / float(medium.c0) ** 2
    u = 0.01 * sphere.normals[:, None, :] * osc[None, :, None]  # [S, T, 3]
    return SurfaceHistory(tau=tau, rho=rho, u=u, p=p)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        _smoke_overrides(args)

    import jax

    jax.config.update("jax_enable_x64", True)

    import numpy as np

    from auraflow.cfd.run import run_acoustic_case
    from auraflow.cfd.sphere import PermeableSphere
    from auraflow.core.medium import Medium
    from auraflow.datasets.compare import compare_cona_vs_cfd
    from auraflow.datasets.nasa_1pax import GROSS_WEIGHT_KG, N_ROTORS, ROTOR_RADIUS

    medium = Medium()
    observers = hover_observers(args.obs_radius, args.n_observers)
    print(
        f"CONA hover reference: {args.n_observers} observers @ {args.obs_radius:g} m, "
        f"fs={args.fs:g} Hz, dur={args.duration:g} s ..."
    )
    cona = cona_hover_audio(args, observers)
    print(f"  CONA audio {tuple(cona['audio'].shape)} generated.")

    thrust = GROSS_WEIGHT_KG * 9.80665 / N_ROTORS
    sphere = PermeableSphere.fibonacci(
        args.n_sphere, radius=args.sphere_radii * ROTOR_RADIUS, center=(0.0, 0.0, 0.0)
    )

    os.makedirs(args.out, exist_ok=True)
    summary: list[dict[str, Any]] = []
    for n in args.resolutions:
        print(f"[{n}^3] {'synthetic surface (dry)' if args.dry else 'JAX-Fluids rotor box'} ...")
        if args.dry:
            n_samp = max(args.n_steps // args.sample_every, 8)
            surf = _synthetic_surface_history(sphere, n_samp, medium)
        else:
            from auraflow.cfd.case import rotor_box_case

            case = rotor_box_case(
                medium,
                rotor_radius=ROTOR_RADIUS,
                box_radii=args.box_radii,
                cells=(n, n, n),
                thrust=thrust,
                tip_mach=args.tip_mach,
            )
            surf = run_acoustic_case(
                case,
                sphere,
                n_steps=args.n_steps,
                sample_every=args.sample_every,
                warmup_steps=args.warmup_steps,
            )
        cmp = compare_cona_vs_cfd(
            cona["audio"],
            cona["fs"],
            surf,
            sphere,
            observers,
            medium,
            fmin=args.fmin,
            fmax=args.fmax,
        )
        band_centers = cmp["observers"][0]["band_centers"]
        band_diff = np.mean([c["band_level_diff"] for c in cmp["observers"]], axis=0)
        out_npz = os.path.join(args.out, f"compare_{n}.npz")
        np.savez_compressed(
            out_npz,
            cells=n,
            band_centers=band_centers,
            band_level_diff_mean=band_diff,
            oaspl_diff_mean=cmp["oaspl_diff_mean"],
            oaspl_diff_rms=cmp["oaspl_diff_rms"],
            band_level_rmse_mean=cmp["band_level_rmse_mean"],
        )
        print(
            f"  OASPL diff mean {cmp['oaspl_diff_mean']:+.2f} dB, "
            f"band RMSE {cmp['band_level_rmse_mean']:.2f} dB -> {os.path.basename(out_npz)}"
        )
        summary.append(
            {"cells": n, "band_centers": band_centers, "band_diff": band_diff, "cmp": cmp}
        )

    _maybe_plot(summary, args.out)
    return 0


def _maybe_plot(summary: list[dict[str, Any]], out_dir: str) -> None:
    """OASPL-error-vs-resolution + band-level overlay PNG (needs the viz extra)."""
    try:
        import matplotlib  # pyright: ignore[reportMissingImports]

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
    except ImportError:
        print("matplotlib not available (viz extra); skipping PNG.", file=sys.stderr)
        return
    import numpy as np

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    cells = [s["cells"] for s in summary]
    ax0.plot(cells, [s["cmp"]["oaspl_diff_rms"] for s in summary], "o-", label="OASPL RMS")
    ax0.plot(cells, [s["cmp"]["band_level_rmse_mean"] for s in summary], "s--", label="band RMSE")
    ax0.set_xlabel("CFD cells per edge N (N^3)")
    ax0.set_ylabel("CONA vs CFD difference [dB]")
    ax0.set_title("Convergence with CFD resolution")
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    for s in summary:
        ax1.semilogx(
            np.asarray(s["band_centers"]), np.asarray(s["band_diff"]), label=f"{s['cells']}^3"
        )
    ax1.axhline(0.0, color="k", lw=0.8)
    ax1.set_xlabel("1/3-octave band centre [Hz]")
    ax1.set_ylabel("band level, CFD - CONA [dB]")
    ax1.set_title("Mean band-level difference")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    png = os.path.join(out_dir, "cona_vs_cfd.png")
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    raise SystemExit(main())
