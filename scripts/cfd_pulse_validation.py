#!/usr/bin/env python
"""Gaussian-pulse validation of the CFD -> permeable-sphere -> FW-H chain.

Runs a quiescent-air box with an initial Gaussian **pressure pulse** at the
centre, samples the flow on a static permeable sphere, propagates to observers on
the ``+x`` axis with the permeable-surface Farassat 1A solver, and compares the
far field to the exact linear-acoustics solution of a spherically symmetric
pressure pulse with zero initial velocity:

    p'(r, t) = (A / (2 r)) [ (r + c0 t) g(r + c0 t) + (r - c0 t) g(r - c0 t) ],
    g(xi) = exp(-xi^2 / (2 w^2)),

which exhibits the expected ``1/r`` amplitude decay and a fixed pulse shape
arriving at ``t = (r - r0) / c0``. The matched initial density
(``rho' = p'/c0^2``, ``u = 0``) makes the perturbation purely acoustic, so this
analytic form is the exact reference.

Intended for GPU execution via omnirun at 64^3-128^3 (the local test suite only
exercises a 32^3 smoke version). Self-contained: ``argparse`` for all knobs,
saves ``results/cfd_pulse_<N>.npz`` and, if matplotlib is available (``viz``
extra), ``results/cfd_pulse_<N>.png``.

Example
-------
    omnirun -- uv run --extra cfd --extra viz \
        python scripts/cfd_pulse_validation.py --cells 96 --obs-radii 1.5 2.0 3.0

    # local smoke check (small, slow on CPU):
    uv run --extra cfd python scripts/cfd_pulse_validation.py \
        --cells 48 --half-size 0.5 --sphere-radius 0.2 --obs-radii 1.0 2.0
"""

from __future__ import annotations

import argparse
import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from auraflow.cfd.case import acoustic_box_case  # noqa: E402
from auraflow.cfd.run import propagate_to_observers, run_acoustic_case  # noqa: E402
from auraflow.cfd.sphere import PermeableSphere  # noqa: E402
from auraflow.core.medium import Medium  # noqa: E402


def analytic_pulse(r: float, t: np.ndarray, amp: float, width: float, c0: float) -> np.ndarray:
    """Exact linear-acoustics pressure perturbation of a Gaussian pulse.

    Args:
        r: Observer radius [m].
        t: Times [s], shape ``[T]``.
        amp: Pulse peak overpressure ``A`` [Pa].
        width: Gaussian standard deviation ``w`` [m].
        c0: Speed of sound [m/s].

    Returns:
        ``p'(r, t)`` [Pa], shape ``[T]``.
    """
    ct = c0 * t
    g = lambda xi: np.exp(-(xi**2) / (2.0 * width**2))  # noqa: E731
    return amp / (2.0 * r) * ((r + ct) * g(r + ct) + (r - ct) * g(r - ct))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cells", type=int, default=64, help="cells per axis (cube)")
    p.add_argument("--half-size", type=float, default=1.0, help="half box edge [m]")
    p.add_argument("--sphere-radius", type=float, default=0.35, help="permeable sphere radius [m]")
    p.add_argument("--sphere-points", type=int, default=400, help="Fibonacci points on the sphere")
    p.add_argument("--pulse-amplitude", type=float, default=100.0, help="pulse overpressure A [Pa]")
    p.add_argument("--pulse-width", type=float, default=0.08, help="pulse Gaussian width w [m]")
    p.add_argument("--cfl", type=float, default=0.4, help="acoustic CFL")
    p.add_argument(
        "--obs-radii",
        type=float,
        nargs="+",
        default=[1.5, 2.0, 3.0],
        help="observer radii on +x [m]",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=0,
        help="integration steps (0 = auto: wave reaches farthest observer + margin)",
    )
    p.add_argument("--sample-every", type=int, default=2, help="sample the sphere every k steps")
    p.add_argument("--altitude", type=float, default=0.0, help="ISA altitude for the medium [m]")
    p.add_argument("--out-dir", type=str, default="results", help="output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    medium = Medium.standard_atmosphere(args.altitude)
    c0 = float(medium.c0)

    case = acoustic_box_case(
        medium,
        half_size=args.half_size,
        cells=(args.cells, args.cells, args.cells),
        cfl=args.cfl,
        pulse=True,
        pulse_amplitude=args.pulse_amplitude,
        pulse_width=args.pulse_width,
    )
    dt = case.dt
    max_r = max(args.obs_radii)
    if args.steps > 0:
        n_steps = args.steps
    else:
        # Enough source time so every observer's arrival window is populated:
        # the sphere must see the pulse (r_s/c0) and tau must span the observer
        # spread (max_r - min_r)/c0, plus margin.
        t_needed = (
            args.sphere_radius + (max_r - min(args.obs_radii)) + 3.0 * args.pulse_width
        ) / c0
        n_steps = int(np.ceil(t_needed / dt)) + 20
    print(
        f"[cfd_pulse] cells={args.cells}^3 dt={dt:.3e}s steps={n_steps} "
        f"sample_every={args.sample_every} sphere_r={args.sphere_radius}"
    )

    sphere = PermeableSphere.fibonacci(args.sphere_points, radius=args.sphere_radius)
    history = run_acoustic_case(case, sphere, n_steps=n_steps, sample_every=args.sample_every)
    print(
        f"[cfd_pulse] sampled {history.tau.shape[0]} frames; "
        f"surface |p'| max = {float(jnp.max(jnp.abs(history.p - medium.p0))):.3f} Pa"
    )

    observers = jnp.array([[r, 0.0, 0.0] for r in args.obs_radii])
    p_prime, t_obs = propagate_to_observers(history, sphere, observers, medium)
    p_prime = np.asarray(p_prime)
    t_obs = np.asarray(t_obs)

    # Compare each observer to the analytic pulse and the 1/r decay law.
    peaks = np.max(np.abs(p_prime), axis=1)
    analytic = np.stack(
        [
            analytic_pulse(r, t_obs[i], args.pulse_amplitude, args.pulse_width, c0)
            for i, r in enumerate(args.obs_radii)
        ]
    )
    analytic_peaks = np.max(np.abs(analytic), axis=1)
    print("[cfd_pulse] per-observer results:")
    for i, r in enumerate(args.obs_radii):
        num = np.max(np.abs(p_prime[i]))
        ana = analytic_peaks[i]
        # shape correlation over the observer window
        a, b = p_prime[i], analytic[i]
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
        print(
            f"  r={r:5.2f} m: FW-H peak={num:9.4f} Pa  analytic peak={ana:9.4f} Pa  "
            f"ratio={num / ana:5.3f}  shape-corr={corr:5.3f}"
        )

    # 1/r check: peak * r should be roughly constant across observers.
    inv_r_product = peaks * np.asarray(args.obs_radii)
    print(
        f"[cfd_pulse] peak*r spread (1/r law): "
        f"{inv_r_product.min():.4f}..{inv_r_product.max():.4f} "
        f"(rel range {np.ptp(inv_r_product) / np.mean(inv_r_product):.2%})"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.cells}"
    npz_path = os.path.join(args.out_dir, f"cfd_pulse_{tag}.npz")
    np.savez(
        npz_path,
        obs_radii=np.asarray(args.obs_radii),
        t_obs=t_obs,
        p_fwh=p_prime,
        p_analytic=analytic,
        peaks=peaks,
        analytic_peaks=analytic_peaks,
        dt=dt,
        cells=args.cells,
        c0=c0,
    )
    print(f"[cfd_pulse] wrote {npz_path}")

    try:
        import matplotlib  # pyright: ignore[reportMissingImports]

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
    except ImportError:
        print("[cfd_pulse] matplotlib not available (install the 'viz' extra) -- skipping plot")
        return

    n = len(args.obs_radii)
    fig, axes = plt.subplots(n, 1, figsize=(7, 2.2 * n), sharex=False, squeeze=False)
    for i, r in enumerate(args.obs_radii):
        ax = axes[i, 0]
        ax.plot(t_obs[i] * 1e3, p_prime[i], label="CFD + FW-H", lw=1.6)
        ax.plot(t_obs[i] * 1e3, analytic[i], "--", label="analytic 1/r", lw=1.2)
        ax.set_ylabel("p' [Pa]")
        ax.set_title(f"observer r = {r:.2f} m")
        ax.legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel("observer time [ms]")
    fig.suptitle(f"Gaussian pulse: CFD+FW-H vs analytic ({args.cells}^3)")
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, f"cfd_pulse_{tag}.png")
    fig.savefig(png_path, dpi=120)
    print(f"[cfd_pulse] wrote {png_path}")


if __name__ == "__main__":
    main()
