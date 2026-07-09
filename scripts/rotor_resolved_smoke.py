#!/usr/bin/env python
"""Resolved-blade spinning-rotor CFD smoke run (level-set) + permeable FW-H.

First GPU exercise of the ``levelset_blades`` path: the NASA 1-Pax rotor as a
real lofted blade mesh (``auraflow.body.blade``), spinning at hover Omega inside
a JAX-Fluids FLUID-SOLID level-set box, flow sampled on an enclosing permeable
*ellipsoid* mesh (the box is flat, a sphere would not fit), far field via
permeable FW-H. Saves ``results/rotor_resolved/<tag>.npz`` with the mic signals
and the surface-pressure statistics; prints the BPF and the observed spectral
peak.

With ``--save-surface`` the FULL permeable-surface flow history (``tau, rho, u,
p``) and the surface geometry (``points, normals, area``, float32) are added to
the same ``.npz`` so it can seed the hybrid CFD+FW-H flyover synthesis
(``scripts/cfd_jasa_flyover.py --surface ...``). At the default settings
(``--steps 3000 --sample-every 5 --warmup 500`` -> T=500 samples on a
subdiv-4 icosphere, S=5120 faces) this adds ~50-60 MB per file.

GPU-scale only (see repo CLAUDE.md): do NOT run locally beyond ``--dry``.
Intended first runs (see docs/research/jaxfluids-evaluation.md for the cubic-
cell requirement of the level-set model):

    omnirun submit --backend kaggle --time 6h -y -- \
        uv run --extra gpu --extra cfd --extra mesh python \
        scripts/rotor_resolved_smoke.py --cells 192 --steps 3000 --save-surface
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from auraflow.cfd.body_case import PermeableMeshSurface
    from auraflow.cfd.run import SurfaceHistory
    from auraflow.core.medium import Medium


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cells", type=int, default=192, help="cells along x/y (z gets half)")
    p.add_argument("--box-xy", type=float, default=1.5, help="box half-extent in R units (x,y)")
    p.add_argument("--box-z", type=float, default=0.75, help="box half-extent in R units (z)")
    p.add_argument("--steps", type=int, default=3000, help="integration steps")
    p.add_argument("--sample-every", type=int, default=5, help="surface sampling stride")
    p.add_argument("--warmup", type=int, default=500, help="steps before sampling starts")
    p.add_argument("--n-stations", type=int, default=24, help="blade radial stations")
    p.add_argument("--n-chord", type=int, default=32, help="chordwise profile points")
    p.add_argument("--sphere-sub", type=int, default=4, help="ellipsoid icosphere subdivisions")
    p.add_argument("--n-obs", type=int, default=8, help="observers on a ring")
    p.add_argument("--obs-r", type=float, default=10.0, help="observer ring radius in R units")
    p.add_argument("--obs-elev-deg", type=float, default=-30.0, help="observer elevation [deg]")
    p.add_argument("--out", type=str, default="results/rotor_resolved")
    p.add_argument("--tag", type=str, default=None, help="output filename tag")
    p.add_argument(
        "--save-surface",
        action="store_true",
        help="also store the full SurfaceHistory + surface geometry in the .npz",
    )
    p.add_argument("--dry", action="store_true", help="build the case, print, exit (no march)")
    return p


@dataclass
class ResolvedCase:
    """A built resolved-rotor case: CFD case, permeable surface and geometry meta."""

    case: Any
    surface: PermeableMeshSurface
    medium: Medium
    nxy: int
    nz: int
    dx: float
    box_z: float


def build_resolved_case(
    cells: int = 192,
    box_xy: float = 1.5,
    n_stations: int = 24,
    n_chord: int = 32,
    sphere_sub: int = 4,
    medium: Medium | None = None,
) -> ResolvedCase:
    """Build the resolved-rotor level-set CFD case + enclosing permeable ellipsoid.

    Shared by :func:`main` and the flyover driver (``scripts/cfd_jasa_flyover.py``
    Stage A) so the hover source geometry has exactly one definition.

    Args:
        cells: Cells along x/y (z gets ``cells//2``; cubic cells enforced).
        box_xy: Box half-extent in rotor-radius units for x, y.
        n_stations: Blade radial stations.
        n_chord: Chordwise profile points.
        sphere_sub: Enclosing-ellipsoid icosphere subdivisions.
        medium: Ambient medium (default ISA sea level).

    Returns:
        A :class:`ResolvedCase`.
    """
    import numpy as np

    from auraflow.body.blade import rotor_levelset_case
    from auraflow.body.mesh import TriMesh
    from auraflow.cfd.body_case import permeable_mesh_surface
    from auraflow.core.blade import Rotor
    from auraflow.core.medium import Medium
    from auraflow.datasets.nasa_1pax import HOVER_OMEGA, N_BLADES, ROTOR_RADIUS, nasa_1pax_blade

    medium = Medium() if medium is None else medium
    r = ROTOR_RADIUS
    nxy = int(cells)
    nz = nxy // 2
    # Cubic cells are mandatory for the JAX-Fluids level-set model: keep
    # (2*box_xy*R)/nxy == (2*box_z*R)/nz exactly by deriving box_z from nz.
    dx = 2.0 * box_xy * r / nxy
    box_z = dx * nz / 2.0

    rotor = Rotor(blade=nasa_1pax_blade(n_stations=n_stations), n_blades=N_BLADES)
    case = rotor_levelset_case(
        rotor,
        omega=HOVER_OMEGA,
        box_lo=(-box_xy * r, -box_xy * r, -box_z),
        box_hi=(box_xy * r, box_xy * r, box_z),
        cells=(nxy, nxy, nz),
        n_chord=n_chord,
        hub=True,
        is_double=False,
        medium=medium,
    )

    # Enclosing permeable ellipsoid: xy semi-axis 1.2R (encloses the blades),
    # z semi-axis inside the box with ~20% margin to the sponge.
    ell = TriMesh.sphere(radius=1.0, subdivisions=sphere_sub)
    scale = np.array([1.2 * r, 1.2 * r, 0.8 * box_z])
    ell = TriMesh(vertices=ell.vertices * scale, faces=ell.faces)
    surface = permeable_mesh_surface(ell)
    return ResolvedCase(
        case=case, surface=surface, medium=medium, nxy=nxy, nz=nz, dx=dx, box_z=box_z
    )


def run_resolved_surface(
    built: ResolvedCase, steps: int, sample_every: int, warmup: int
) -> SurfaceHistory:
    """March the resolved-rotor case and sample the permeable surface history."""
    from auraflow.cfd.run import run_acoustic_case

    return run_acoustic_case(
        built.case,
        built.surface,
        n_steps=steps,
        sample_every=sample_every,
        warmup_steps=warmup,
    )


def main() -> int:
    args = _parser().parse_args()

    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", True)

    from auraflow.datasets.nasa_1pax import BPF_HZ, HOVER_OMEGA, ROTOR_RADIUS

    built = build_resolved_case(
        cells=args.cells,
        box_xy=args.box_xy,
        n_stations=args.n_stations,
        n_chord=args.n_chord,
        sphere_sub=args.sphere_sub,
    )
    r = ROTOR_RADIUS
    medium = built.medium
    surface = built.surface
    nxy, nz, dx = built.nxy, built.nz, built.dx

    elev = math.radians(args.obs_elev_deg)
    phis = np.linspace(0.0, 2.0 * math.pi, args.n_obs, endpoint=False)
    obs = np.stack(
        [
            args.obs_r * r * np.cos(phis) * math.cos(elev),
            args.obs_r * r * np.sin(phis) * math.cos(elev),
            np.full_like(phis, args.obs_r * r * math.sin(elev)),
        ],
        axis=-1,
    )

    rev_steps = 2.0 * math.pi / HOVER_OMEGA / built.case.dt
    print(
        f"[rotor] cells=({nxy},{nxy},{nz}) dx={dx * 1e3:.1f}mm dt={built.case.dt:.3e}s "
        f"steps={args.steps} (~{args.steps / rev_steps:.2f} rev) "
        f"surface faces={surface.points.shape[0]} BPF={BPF_HZ:.2f} Hz"
    )
    if args.dry:
        return 0

    from auraflow.cfd.run import propagate_to_observers

    hist = run_resolved_surface(built, args.steps, args.sample_every, args.warmup)
    p, t_obs = propagate_to_observers(hist, surface, jax.numpy.asarray(obs), medium)
    p = np.asarray(p)
    t_obs = np.asarray(t_obs)

    rms = np.sqrt(np.mean(p**2, axis=1))
    oaspl = 20.0 * np.log10(np.maximum(rms, 1e-12) / 2e-5)
    dt_obs = float(t_obs[1] - t_obs[0])
    spec = np.abs(np.fft.rfft(p[0] - p[0].mean()))
    freqs = np.fft.rfftfreq(p.shape[1], dt_obs)
    f_peak = float(freqs[int(np.argmax(spec))])
    print(f"[rotor] OASPL per mic [dB]: {np.array2string(oaspl, precision=1)}")
    print(f"[rotor] mic0 spectral peak at {f_peak:.2f} Hz (BPF {BPF_HZ:.2f} Hz)")

    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or f"c{nxy}_s{args.steps}"
    path = os.path.join(args.out, f"rotor_{tag}.npz")
    fields: dict[str, Any] = dict(
        p=p,
        t_obs=t_obs,
        observers=obs,
        oaspl=oaspl,
        surface_p_rms=np.asarray(np.sqrt(np.mean(hist.p**2, axis=1))),
        tau=np.asarray(hist.tau),
        bpf_hz=BPF_HZ,
        dt=built.case.dt,
        cells=np.array([nxy, nxy, nz]),
    )
    if args.save_surface:
        # Full permeable-surface flow history + geometry (float32) to seed the
        # hybrid flyover synthesis. ~50-60 MB at default settings.
        fields.update(
            surf_rho=np.asarray(hist.rho, dtype=np.float32),
            surf_u=np.asarray(hist.u, dtype=np.float32),
            surf_p=np.asarray(hist.p, dtype=np.float32),
            surf_points=np.asarray(surface.points, dtype=np.float32),
            surf_normals=np.asarray(surface.normals, dtype=np.float32),
            surf_area=np.asarray(surface.area, dtype=np.float32),
            omega=HOVER_OMEGA,
        )
    np.savez_compressed(path, **fields)
    extra = " (+full surface history)" if args.save_surface else ""
    print(f"[rotor] wrote {path}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
