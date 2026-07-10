#!/usr/bin/env python
r"""Hybrid CFD + FW-H JASA flyover synthesis from one resolved-rotor hover run.

Two stages (see ``auraflow.cfd.flyover`` for the physics + approximations):

- **Stage A (GPU)** -- obtain a permeable-surface hover flow history. Either load
  one saved by ``scripts/rotor_resolved_smoke.py --save-surface`` (``--surface
  path.npz``), or run the resolved-rotor level-set CFD inline (reusing that
  script's ``build_resolved_case`` / ``run_resolved_surface``; ``--cells
  --steps ...``) and optionally re-save it (``--save-surface``).
- **Stage B** -- tile the hover history to the flyover duration and, for each
  ``--speeds`` value, fly four phase-staggered copies of the surface along the
  JASA level-flight trajectory (30 m altitude, over the origin at
  ``duration/2``) and radiate to a microphone subset, writing
  ``results/cfd_flyover/V<speed>.npz`` plus per-mic WAVs (float32 Pa, and an
  int16 ``_norm`` set scaled by one shared factor for listening).

Stage A's resolved-rotor case is built through ``rotor_levelset_case`` (default
canonical-blade composition + GPU brute-force SDF with a winding-number sign,
issue #2; the single-blade SDF is disk-cached in ``~/.cache/auraflow/sdf`` /
``$AURAFLOW_SDF_CACHE``), so a run no longer pays the ~1h46m ``trimesh`` SDF build.

Local proof: ``--smoke`` builds a tiny SYNTHETIC breathing-ellipsoid history (no
jaxfluids) and runs the whole Stage-B synthesis end-to-end under the dev-box
memory cap. Everything else (real CFD, 44.1 kHz x 1 s) is GPU/omnirun work.

    # Stage A on GPU (writes the surface into the results npz):
    omnirun submit --backend kaggle --time 6h -y -- \
        uv run --extra gpu --extra cfd --extra mesh python \
        scripts/rotor_resolved_smoke.py --cells 192 --steps 4000 --save-surface \
        --tag jasa
    # Stage B (CPU-ok, but GPU faster) from the saved surface:
    uv run python scripts/cfd_jasa_flyover.py \
        --surface results/rotor_resolved/rotor_jasa.npz --speeds 4 6 8 10 \
        --duration 1.0 --mics row
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any

_RRS: Any = None


def _rrs() -> Any:
    """Load the sibling ``rotor_resolved_smoke`` module by path (scripts/ is not a package).

    Gives access to its Stage-A builders (``build_resolved_case`` /
    ``run_resolved_surface``) and the shared ``_vehicle_module`` / ``VehicleSpec``
    registry (single source of truth for per-vehicle constants + blade geometry).
    """
    global _RRS
    if _RRS is not None:
        return _RRS
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "rotor_resolved_smoke", os.path.join(here, "rotor_resolved_smoke.py")
    )
    assert spec is not None and spec.loader is not None
    rrs = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: @dataclass resolves cls.__module__ through sys.modules
    # during class creation and hard-crashes on an unregistered module.
    sys.modules[spec.name] = rrs
    spec.loader.exec_module(rrs)
    _RRS = rrs
    return rrs


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--vehicle",
        choices=("nasa-1pax", "dji-9450"),
        default="nasa-1pax",
        help="rotor/vehicle whose blade + constants define the source (default nasa-1pax)",
    )
    # Stage A source
    p.add_argument("--surface", type=str, default=None, help="saved surface npz (skip CFD)")
    p.add_argument("--smoke", action="store_true", help="tiny synthetic surface, run locally")
    p.add_argument("--save-surface", action="store_true", help="save the CFD surface npz too")
    # inline CFD (Stage A) knobs -- mirror rotor_resolved_smoke defaults
    p.add_argument("--cells", type=int, default=192)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument(
        "--sample-every",
        type=int,
        default=None,
        help="surface sampling stride in steps (default: derived from --fs-surface)",
    )
    p.add_argument(
        "--fs-surface",
        type=float,
        default=48000.0,
        help="target surface sampling rate [Hz]; the stride is round(1/(fs*dt)). "
        "Audio band is fs/2; Stage-B memory/time scale LINEARLY with fs -- a raw "
        "--sample-every on a fine-dt case (e.g. 145 kHz on the 192-cell DJI grid) "
        "is how the first DJI runs OOMed.",
    )
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--n-stations", type=int, default=24)
    p.add_argument("--n-chord", type=int, default=32)
    p.add_argument("--sphere-sub", type=int, default=4)
    # Stage B knobs
    p.add_argument("--speeds", type=float, nargs="+", default=[4.0, 8.0], help="flyover speeds m/s")
    p.add_argument("--duration", type=float, default=1.0, help="flyover duration [s]")
    p.add_argument("--altitude", type=float, default=30.0, help="flight altitude [m]")
    p.add_argument("--t-pass", type=float, default=None, help="pass-over time [s] (default dur/2)")
    p.add_argument(
        "--mics",
        type=str,
        default="row",
        help="'row' (y=0 row, 32 mics), 'sel' (0,15,31,239), or a comma list of indices",
    )
    p.add_argument("--fs-out", type=float, default=44100.0, help="output audio rate [Hz]")
    p.add_argument("--panel-chunk", type=int, default=512)
    p.add_argument("--obs-chunk", type=int, default=8)
    p.add_argument("--n-obs", type=int, default=None, help="observer-time samples (default: auto)")
    p.add_argument("--out", type=str, default="results/cfd_flyover")
    return p


def _select_mics(sel: str) -> tuple[Any, Any]:
    """Resolve a mic-subset spec against the full JASA 256-mic array."""
    import numpy as np

    from auraflow.datasets.jasa import jasa_microphone_array

    mics = np.asarray(jasa_microphone_array(), dtype=float)  # [256,3]
    if sel == "row":  # y = 0 row (row-major, x fastest -> first 32)
        idx = np.arange(32)
    elif sel == "sel":  # the 4 showcase mics
        idx = np.array([0, 15, 31, 239])
    else:
        idx = np.array([int(x) for x in sel.split(",")])
    return mics[idx], idx


def _smoke_source(spec: Any) -> tuple[dict, dict, float]:
    """A tiny synthetic breathing-ellipsoid history (no jaxfluids), for --smoke."""
    import numpy as np

    from auraflow.body.mesh import TriMesh
    from auraflow.core.medium import Medium

    medium = Medium()
    r = spec.rotor_radius
    mesh = TriMesh.sphere(radius=1.0, subdivisions=1)  # 80 faces
    verts = np.asarray(mesh.vertices) * np.array([1.2 * r, 1.2 * r, 0.6 * r])
    mesh = TriMesh(vertices=verts, faces=mesh.faces)
    surf = {
        "points": np.asarray(mesh.centroids(), dtype=np.float64),
        "normals": np.asarray(mesh.normals(), dtype=np.float64),
        "area": np.asarray(mesh.areas(), dtype=np.float64),
    }
    f_bp = spec.hover_omega * spec.n_blades / (2.0 * np.pi)
    samples_per_period = 20
    dtau = 1.0 / (f_bp * samples_per_period)
    n_in = 3 * samples_per_period + 3  # ~3 blade periods
    tau = np.arange(n_in) * dtau
    n_s = surf["points"].shape[0]
    phase = np.linspace(0.0, 2.0 * np.pi, n_s, endpoint=False)
    # analytic monopole-ish pulsation: radial breathing + a loading fluctuation.
    s1 = np.sin(2 * np.pi * f_bp * tau[None, :] + phase[:, None])
    s2 = 0.3 * np.sin(4 * np.pi * f_bp * tau[None, :] + 2 * phase[:, None])
    u = surf["normals"][:, None, :] * (1.5 * (s1 + s2))[:, :, None]
    rho = float(medium.rho0) + 0.01 * (s1 + s2)
    p = float(medium.p0) + 12.0 * (s1 + 0.5 * s2)
    raw = {"tau": tau, "rho": rho, "u": u, "p": p}
    return surf, raw, spec.hover_omega


def _load_surface(path: str, spec: Any) -> tuple[dict, dict, float]:
    """Load a saved rotor_resolved surface npz -> (geom, raw history, omega)."""
    import numpy as np

    d = np.load(path)
    if "surf_points" not in d:
        raise SystemExit(
            f"{path} has no surface history; regenerate with rotor_resolved_smoke.py --save-surface"
        )
    surf = {
        "points": np.asarray(d["surf_points"], dtype=np.float64),
        "normals": np.asarray(d["surf_normals"], dtype=np.float64),
        "area": np.asarray(d["surf_area"], dtype=np.float64),
    }
    raw = {
        "tau": np.asarray(d["tau"], dtype=np.float64),
        "rho": np.asarray(d["surf_rho"], dtype=np.float64),
        "u": np.asarray(d["surf_u"], dtype=np.float64),
        "p": np.asarray(d["surf_p"], dtype=np.float64),
    }
    omega = float(d["omega"]) if "omega" in d else spec.hover_omega
    return surf, raw, omega


def _run_cfd(args: argparse.Namespace, spec: Any) -> tuple[dict, dict, float]:
    """Run the resolved-rotor CFD inline (Stage A) -> (geom, raw history, omega)."""
    import numpy as np

    rrs = _rrs()
    build_resolved_case = rrs.build_resolved_case
    run_resolved_surface = rrs.run_resolved_surface

    built = build_resolved_case(
        cells=args.cells,
        n_stations=args.n_stations,
        n_chord=args.n_chord,
        sphere_sub=args.sphere_sub,
        vehicle=args.vehicle,
    )
    stride = args.sample_every
    if stride is None:
        stride = max(1, round(1.0 / (args.fs_surface * built.case.dt)))
    fs_actual = 1.0 / (stride * built.case.dt)
    print(
        f"[flyover] surface sampling: every {stride} steps -> {fs_actual / 1e3:.1f} kHz "
        f"(dt={built.case.dt:.3e}s)",
        flush=True,
    )
    hist = run_resolved_surface(built, args.steps, stride, args.warmup)
    surf = {
        "points": np.asarray(built.surface.points, dtype=np.float64),
        "normals": np.asarray(built.surface.normals, dtype=np.float64),
        "area": np.asarray(built.surface.area, dtype=np.float64),
    }
    raw = {
        "tau": np.asarray(hist.tau, dtype=np.float64),
        "rho": np.asarray(hist.rho, dtype=np.float64),
        "u": np.asarray(hist.u, dtype=np.float64),
        "p": np.asarray(hist.p, dtype=np.float64),
    }
    if args.save_surface:
        os.makedirs(args.out, exist_ok=True)
        sp = os.path.join(args.out, "surface.npz")
        np.savez_compressed(
            sp,
            tau=raw["tau"].astype(np.float32),
            surf_rho=raw["rho"].astype(np.float32),
            surf_u=raw["u"].astype(np.float32),
            surf_p=raw["p"].astype(np.float32),
            surf_points=surf["points"].astype(np.float32),
            surf_normals=surf["normals"].astype(np.float32),
            surf_area=surf["area"].astype(np.float32),
            omega=spec.hover_omega,
        )
        print(f"[flyover] wrote surface {sp}")
    return surf, raw, spec.hover_omega


def main() -> int:
    args = _parser().parse_args()

    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", True)

    from scipy.io import wavfile

    from auraflow.cfd.flyover import (
        quadrotor_surface_flyover,
        synthesize_flyover_wavs,
        tile_surface_history,
    )
    from auraflow.core.medium import Medium

    spec = _rrs()._vehicle_module(args.vehicle)
    BPF_HZ = spec.bpf_hz
    N_BLADES = spec.n_blades

    medium = Medium()
    layout = spec.multirotor()  # hub positions + spin signs (single source of truth)

    duration = float(args.duration)
    mics_sel = "0,1" if args.smoke else args.mics
    if args.smoke:
        duration = min(duration, 0.12)
        surf, raw, omega = _smoke_source(spec)
    elif args.surface:
        surf, raw, omega = _load_surface(args.surface, spec)
    else:
        surf, raw, omega = _run_cfd(args, spec)

    mics, mic_idx = _select_mics(mics_sel)
    t_pass = 0.5 * duration if args.t_pass is None else float(args.t_pass)

    tiled = tile_surface_history(raw, omega, N_BLADES, duration=duration)
    print(
        f"[flyover] surface panels={surf['points'].shape[0]} "
        f"tiled T={tiled['tau'].shape[0]} (n_periods={tiled['n_periods']}, "
        f"period_samples={tiled['period_samples']}) mics={mics.shape[0]} BPF={BPF_HZ:.2f} Hz"
    )

    os.makedirs(args.out, exist_ok=True)
    results: dict[float, dict[str, Any]] = {}
    for speed in args.speeds:
        p, t_obs = quadrotor_surface_flyover(
            surf,
            tiled,
            layout,
            speed=float(speed),
            altitude=float(args.altitude),
            t_pass=t_pass,
            observers=mics,
            medium=medium,
            n_obs=args.n_obs,
            panel_chunk=args.panel_chunk,
            obs_chunk=args.obs_chunk,
        )
        p = np.asarray(p)
        t_obs = np.asarray(t_obs)
        wav = synthesize_flyover_wavs(p, t_obs, fs_out=args.fs_out)
        rms = np.sqrt(np.mean(p**2, axis=1))
        oaspl = 20.0 * np.log10(np.maximum(rms, 1e-12) / 2e-5)
        results[speed] = {"p": p, "t_obs": t_obs, "wav": wav, "oaspl": oaspl}
        print(
            f"[flyover] V={speed:g} m/s  T_obs={t_obs.shape[0]}  wav={wav.shape[1]} smp  "
            f"OASPL[dB] min/mean/max={oaspl.min():.1f}/{oaspl.mean():.1f}/{oaspl.max():.1f}"
        )

    # One shared normalization factor across all speeds & mics (preserves relative
    # levels between speeds) for the int16 listening set.
    peak = max(float(np.max(np.abs(r["wav"]))) for r in results.values())
    peak = max(peak, 1e-12)
    fs_out = int(round(float(args.fs_out)))

    for speed, r in results.items():
        tag = f"V{speed:g}".replace(".", "p")
        stem = os.path.join(args.out, tag)
        meta = {
            "speed": float(speed),
            "altitude": float(args.altitude),
            "duration": duration,
            "t_pass": t_pass,
            "fs_out": float(args.fs_out),
            "bpf_hz": float(BPF_HZ),
            "mic_indices": [int(i) for i in mic_idx],
            "norm_factor": peak,
            "n_panels": int(surf["points"].shape[0]),
            "n_periods": int(tiled["n_periods"]),
        }
        np.savez_compressed(
            stem + ".npz",
            p=r["p"].astype(np.float32),
            t_obs=r["t_obs"],
            wav=r["wav"].astype(np.float32),
            mics=mics,
            oaspl=r["oaspl"],
            meta_json=json.dumps(meta),
        )
        wav_dir = stem + "_wav"
        os.makedirs(wav_dir, exist_ok=True)
        wav = r["wav"]
        wav_norm = np.clip(wav / peak, -1.0, 1.0)
        for j in range(wav.shape[0]):
            m = int(mic_idx[j])
            wavfile.write(
                os.path.join(wav_dir, f"mic_{m:03d}.wav"), fs_out, wav[j].astype(np.float32)
            )
            wavfile.write(
                os.path.join(wav_dir, f"mic_{m:03d}_norm.wav"),
                fs_out,
                (wav_norm[j] * 32767.0).astype(np.int16),
            )
        print(f"[flyover] wrote {stem}.npz + {wav.shape[0]} mic WAVs (x2)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
