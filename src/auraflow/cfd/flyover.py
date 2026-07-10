r"""Hybrid CFD + FW-H flyover synthesis from a single resolved-rotor hover run.

The full-CFD flyover pipeline (``docs/architecture.md``, CFD backend) is far too
expensive to march a whole moving multirotor through a compressible domain. This
module implements the **hybrid** used to reach the JASA flyover cases from *one*
resolved-blade hover CFD:

1. one resolved-rotor hover CFD (``scripts/rotor_resolved_smoke.py
   --save-surface``) produces a permeable-surface flow history
   :class:`~auraflow.cfd.run.SurfaceHistory` on a static ellipsoid enclosing the
   blades (``tau, rho, u, p`` + the surface ``points/normals/area``);
2. :func:`tile_surface_history` trims that short sampled window to an integer
   number of blade-passing periods and tiles it (with a crossfade) up to the
   flyover duration, giving a steady periodic hover source of arbitrary length;
3. :func:`quadrotor_surface_flyover` places four phase-staggered copies of the
   surface at the vehicle's hub layout, mirrors the counter-rotating rotors,
   rigidly flies the whole vehicle along the JASA trajectory, and radiates every
   panel with the **moving** permeable Farassat-1A kernel
   (:func:`auraflow.fwh.f1a_permeable`) to the ground microphones;
4. :func:`synthesize_flyover_wavs` upsamples the (CFD-rate) mic pressures to an
   audio sample rate.

Physics approximations (all documented at the point of use, valid for the JASA
level-flight cases at advance ratio ``mu = V_inf/(Omega R) <= 0.06``):

- **Quasi-hover source.** The blades are only ever solved in *hover*; the tiled
  hover near field is flown rigidly. The edgewise crossflow that a real forward
  flight would impose on the blade loading is neglected (``mu <= 0.06`` -- the
  1PAX tip speed is ``137 m/s``, so ``10 m/s`` forward flight is a 7% perturbation
  on the tip and much less in RMS over the disk).
- **Galilean boost of the surface fluid velocity.** The hover CFD is solved with
  quiescent air at infinity in the *vehicle* frame; to radiate to the still-air
  *ground* frame the panel fluid velocity is boosted ``u_lab = u_hover + V_inf``
  (``boost_u=True``). The steady ``rho0 * v_n`` thickness of the translating
  closed surface integrates to zero exactly, but its DISCRETE panel sum does
  not: ``|V_inf|`` exceeds the physical fluctuations by ~42 dB on the DJI case
  and the quadrature residual dominated the received signal as low-frequency
  pseudo-sound. The term is therefore dropped symbolically
  (``include_steady_vn=False`` in the FW-H kernel), which is exact for this
  closed rigidly-translating surface.
- **Level attitude.** The small nose-down trim pitch at ``<= 10 m/s`` is neglected
  (the surface is flown with a fixed, level orientation; only its position
  advances).
- **Mirror = exact only for a symmetric blade.** A counter-rotating rotor is
  synthesised by mirroring the hover field about the rotor ``xz``-plane
  (``y -> -y``); this is exact for the symmetric (NACA-0012) reconstructed blade
  and leaves a residual handedness error to the extent the real blade is
  cambered/twisted with a sense.

No jaxfluids import: this module is pure numpy (tiling / trajectory setup, an IO
boundary) + JAX (the differentiable FW-H propagation).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from auraflow.core.medium import Medium
from auraflow.fwh.f1a import f1a_permeable

__all__ = [
    "quadrotor_surface_flyover",
    "synthesize_flyover_wavs",
    "tile_surface_history",
]


# --- history tiling -----------------------------------------------------------


def _as_hist(hist_arrays: Mapping[str, Any] | Any) -> dict[str, np.ndarray]:
    """Extract ``tau, rho, u, p`` as numpy arrays from a mapping or eqx history."""
    if isinstance(hist_arrays, Mapping):
        get = hist_arrays.__getitem__
    else:  # e.g. auraflow.cfd.run.SurfaceHistory (attribute access)
        get = lambda k: getattr(hist_arrays, k)  # noqa: E731
    return {k: np.asarray(get(k)) for k in ("tau", "rho", "u", "p")}


def _tile_axis1(seg: np.ndarray, xfade: int, n_out: int) -> np.ndarray:
    """Periodically tile ``seg`` along axis 1 to length ``n_out`` with a crossfade.

    Overlap-adds copies of ``seg`` (time on axis 1) at hop ``L = seg.shape[1]``,
    so the fundamental period is exactly ``L`` (an integer number of blade
    periods) and no phase drift accumulates. Each copy is extended by its own
    wrapped head (``xfade`` samples) and weighted by a window that ramps up over
    the first ``xfade`` and down over the wrapped tail; complementary linear
    ramps sum to 1 in every overlap, so the join is C0-continuous. For exactly
    periodic ``seg`` the overlap reproduces ``seg`` bit-for-bit; for imperfectly
    periodic CFD data it smooths the residual seam mismatch. The first ``xfade``
    and last ``xfade`` samples of the whole output carry a mild fade in/out (the
    flyover windows the ends anyway).
    """
    seg = np.asarray(seg)
    length = seg.shape[1]
    xfade = int(max(1, min(xfade, length)))
    if n_out <= length:
        return seg[:, :n_out, ...]
    seg_ext = np.concatenate([seg, seg[:, :xfade, ...]], axis=1)  # [S, L+xf, ...]
    ramp = np.linspace(0.0, 1.0, xfade)
    win = np.ones(length + xfade)
    win[:xfade] = ramp
    win[length:] = ramp[::-1]
    wshape = [1] * seg.ndim
    wshape[1] = length + xfade
    seg_w = seg_ext * win.reshape(wshape)
    n_copies = int(np.ceil(n_out / length)) + 1
    buf = np.zeros((seg.shape[0], n_copies * length + xfade) + seg.shape[2:], dtype=seg.dtype)
    for i in range(n_copies):
        s = i * length
        buf[:, s : s + length + xfade, ...] += seg_w
    return buf[:, :n_out, ...]


def _tile_chunk(seg_chunk: np.ndarray, xfade: int, n_out: int, shift: int) -> np.ndarray:
    """Tile a panel-CHUNK of a periodic segment, then phase-roll it, lazily.

    Reproduces ``np.roll(_tile_axis1(seg, xfade, n_out), shift, axis=1)[ps:pe]``
    for the chunk rows ``ps:pe`` -- but only ever allocating the ``[c, n_out, ...]``
    chunk array, never the full ``[S, n_out, ...]`` tile. This is bit-exact with
    the eager path because :func:`_tile_axis1` and :func:`numpy.roll` both act
    independently per row (axis 0), so slicing rows commutes with them.
    """
    tiled = _tile_axis1(seg_chunk, xfade, n_out)  # [c, n_out, ...]
    if shift % n_out != 0:
        tiled = np.roll(tiled, shift, axis=1)
    return tiled


def tile_surface_history(
    hist_arrays: Mapping[str, Any] | Any,
    omega: float,
    n_blades: int,
    duration: float,
    *,
    crossfade_frac: float = 0.125,
    lazy: bool = False,
) -> dict[str, Any]:
    """Tile a short hover surface history to a periodic flyover-length source.

    The rotor near field is periodic with the blade-passing period
    ``T_bp = 2*pi / (omega * n_blades)``. The sampled window is trimmed to the
    nearest integer number of ``T_bp`` (never more than the data covers) and that
    segment is tiled up to ``duration`` with a short crossfade at each seam
    (:func:`_tile_axis1`).

    The **time-mean of ``p``, ``rho`` AND ``u`` is removed per panel** before
    tiling: the hover CFD carries large steady offsets -- DC loading/density,
    and a steady downwash *through* the permeable surface (measured 3x the
    fluctuating part on the DJI case) -- whose FW-H terms on a *translating*
    surface produce a dominating non-radiating low-frequency pedestal
    (hydrodynamic pseudo-sound: on the warmed DJI run 92% of the received
    energy sat below 30 Hz with the mean ``u`` kept). Steady subsonic sources
    in uniform motion radiate nothing physically; only the fluctuations carry
    sound, so all three fields are reduced to fluctuations consistently.
    :func:`quadrotor_surface_flyover` restores the ambient ``p0``/``rho0``
    before the FW-H kernel (which expects absolute ``p`` and ``rho``).

    Args:
        hist_arrays: A :class:`~auraflow.cfd.run.SurfaceHistory` or a mapping with
            ``tau [T]``, ``rho [S, T]``, ``u [S, T, 3]``, ``p [S, T]`` (uniform
            ``tau``).
        omega: Rotor speed magnitude [rad/s] (sets ``T_bp``).
        n_blades: Blades per rotor (sets ``T_bp``).
        duration: Target flyover source duration [s].
        crossfade_frac: Crossfade length as a fraction of one blade-passing
            period (default ``1/8``).
        lazy: If ``True``, do NOT materialize the full tiled ``[S, T_out]``
            arrays. Return instead the small trimmed/de-meaned periodic
            **segment** (``seg_rho [S, seg]``, ``seg_u [S, seg, 3]``,
            ``seg_p [S, seg]``) plus the tiling recipe (``xfade``, ``n_out``);
            :func:`quadrotor_surface_flyover` then tiles each panel-chunk to
            ``T_out`` on the fly (bit-exact with the eager tile -- see
            :func:`_tile_chunk`), so Stage B never holds a full-length surface
            history. Use this for real-scale flyovers (the eager arrays are tens
            of GB at 48 kHz x 5120 panels); the eager default stays for the unit
            tests and direct inspection. The returned dict carries ``lazy: True``.

    Returns:
        Eager (default): a dict with tiled ``tau [T_out]``, ``rho [S, T_out]``,
        ``u [S, T_out, 3]``, ``p [S, T_out]`` (``p``/``rho`` gauge/zero-mean)
        plus bookkeeping ``period_samples`` (samples in one ``T_bp``),
        ``n_periods`` (periods in the trimmed segment) and ``xfade`` (crossfade
        samples). Lazy: ``lazy: True`` plus ``tau [T_out]``, the segment fields
        ``seg_rho``/``seg_u``/``seg_p``, ``period_samples``, ``n_periods``,
        ``xfade`` and ``n_out`` -- a drop-in argument for
        :func:`quadrotor_surface_flyover`.
    """
    h = _as_hist(hist_arrays)
    tau = h["tau"].astype(np.float64)
    rho = h["rho"].astype(np.float64)
    u = h["u"].astype(np.float64)
    p = h["p"].astype(np.float64)
    n_in = tau.shape[0]
    if n_in < 2:
        raise ValueError("surface history needs >= 2 time samples to tile")
    dtau = float(tau[1] - tau[0])

    t_bp = 2.0 * np.pi / (abs(float(omega)) * int(n_blades))
    period_samples = int(round(t_bp / dtau))
    if period_samples < 1:
        raise ValueError(
            f"blade-passing period T_bp={t_bp:.3e}s is below one sample dtau={dtau:.3e}s; "
            "sample the CFD more finely"
        )
    window_span = (n_in - 1) * dtau
    n_periods = int(round(window_span / t_bp))
    n_periods = max(1, n_periods)
    seg_samples = n_periods * period_samples
    while seg_samples > n_in and n_periods > 1:  # never exceed the sampled data
        n_periods -= 1
        seg_samples = n_periods * period_samples
    seg_samples = min(seg_samples, n_in)

    # Remove the per-panel time-mean of p, rho and u (the hover DC pedestal and
    # the steady surface throughflow -- see docstring).
    seg_rho = rho[:, :seg_samples]
    seg_p = p[:, :seg_samples]
    seg_u = u[:, :seg_samples]
    seg_rho = seg_rho - seg_rho.mean(axis=1, keepdims=True)
    seg_p = seg_p - seg_p.mean(axis=1, keepdims=True)
    seg_u = seg_u - seg_u.mean(axis=1, keepdims=True)

    xfade = int(max(1, round(crossfade_frac * period_samples)))
    n_out = int(round(duration / dtau))
    n_out = max(n_out, seg_samples)
    tau_t = tau[0] + np.arange(n_out) * dtau

    if lazy:
        # Keep only the small periodic segment; tiling happens per panel-chunk
        # inside quadrotor_surface_flyover (bit-exact -- see _tile_chunk).
        return {
            "lazy": True,
            "tau": tau_t,
            "seg_rho": seg_rho,
            "seg_u": seg_u,
            "seg_p": seg_p,
            "period_samples": period_samples,
            "n_periods": n_periods,
            "xfade": xfade,
            "n_out": n_out,
        }

    rho_t = _tile_axis1(seg_rho, xfade, n_out)
    p_t = _tile_axis1(seg_p, xfade, n_out)
    u_t = _tile_axis1(seg_u, xfade, n_out)

    return {
        "tau": tau_t,
        "rho": rho_t,
        "u": u_t,
        "p": p_t,
        "period_samples": period_samples,
        "n_periods": n_periods,
        "xfade": xfade,
    }


# --- geometry / layout duck-typing --------------------------------------------


def _geom(surface_geom: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points, normals, area) from a mesh-surface object or a mapping/tuple."""
    if hasattr(surface_geom, "points"):
        pts, nrm, area = surface_geom.points, surface_geom.normals, surface_geom.area
    elif isinstance(surface_geom, Mapping):
        pts, nrm, area = surface_geom["points"], surface_geom["normals"], surface_geom["area"]
    else:
        pts, nrm, area = surface_geom
    return (
        np.asarray(pts, dtype=np.float64),
        np.asarray(nrm, dtype=np.float64),
        np.asarray(area, dtype=np.float64),
    )


def _layout(vehicle_layout: Any) -> tuple[np.ndarray, np.ndarray]:
    """(hub_positions [Nr,3], spin_signs [Nr]) from a Multirotor-like or a tuple."""
    if hasattr(vehicle_layout, "rotor_positions"):
        pos, spins = vehicle_layout.rotor_positions, vehicle_layout.spin_signs
    elif isinstance(vehicle_layout, Mapping):
        pos, spins = vehicle_layout["positions"], vehicle_layout["spins"]
    else:
        pos, spins = vehicle_layout
    return np.asarray(pos, dtype=np.float64), np.asarray(spins, dtype=np.float64).ravel()


# --- moving-surface flyover ---------------------------------------------------

_MIRROR_Y = np.array([1.0, -1.0, 1.0])


def quadrotor_surface_flyover(
    surface_geom: Any,
    tiled_history: Mapping[str, Any],
    vehicle_layout: Any,
    speed: float,
    altitude: float,
    t_pass: float,
    observers: ArrayLike,
    medium: Medium,
    *,
    flight_dir: Sequence[float] = (1.0, 0.0, 0.0),
    phase_offsets: Sequence[float] | None = None,
    boost_u: bool = True,
    n_obs: int | None = None,
    panel_chunk: int = 512,
    obs_chunk: int = 8,
) -> tuple[Array, Array]:
    r"""Radiate a tiled hover surface flown along the JASA trajectory to mics.

    Four copies of ``surface_geom`` are placed at the vehicle hub layout, each fed
    a phase-shifted copy of ``tiled_history`` (staggering the rotors), the
    counter-rotating rotors mirrored about their ``xz``-plane, and the whole
    vehicle rigidly translated along ``flight_dir`` at ``speed`` (level, at
    ``altitude``), passing over the origin at ``t_pass``. Every panel radiates via
    the **moving** permeable Farassat-1A kernel
    (:func:`auraflow.fwh.f1a_permeable`); contributions from all rotors, panel
    chunks and observer chunks are summed onto one shared observer-time grid.

    Panel kinematics: position ``y(tau) = hub_i + local + V_inf*(tau - t_pass)*dir
    + altitude*z``; velocity ``v = V_inf*dir`` (constant); acceleration ``0``.
    Fluid velocity fed to the kernel is boosted ``u + V_inf*dir`` when ``boost_u``
    (module docstring). ``p``/``rho`` from ``tiled_history`` are gauge fluctuations;
    the ambient ``p0``/``rho0`` are restored here.

    Args:
        surface_geom: The permeable surface (``points/normals/area``; a
            :class:`~auraflow.cfd.body_case.PermeableMeshSurface`,
            :class:`~auraflow.cfd.sphere.PermeableSphere`, mapping, or tuple),
            in the rotor frame centred at the hub.
        tiled_history: Output of :func:`tile_surface_history` (gauge ``p``/``rho``).
        vehicle_layout: Hub positions ``[Nr, 3]`` and spin signs ``[Nr]`` -- a
            :class:`~auraflow.cona.flight.Multirotor` (read back, no duplicated
            constants), a mapping ``{"positions", "spins"}``, or a tuple.
        speed: Ground speed ``V_inf`` [m/s].
        altitude: Constant flight altitude [m] (world ``z``).
        t_pass: Time the vehicle passes over the world origin [s].
        observers: Microphone positions [m], shape ``[O, 3]``.
        medium: Ambient :class:`~auraflow.core.medium.Medium`.
        flight_dir: Flight direction (unit-normalised internally); default ``+x``.
        phase_offsets: Per-rotor blade-phase offsets as fractions of one
            blade-passing period; default ``i / Nr`` (evenly staggers the rotors).
        boost_u: Galilean-boost the panel fluid velocity into the ground frame.
        n_obs: Observer-time samples (default: one per source ``dtau`` across the
            arrival window, preserving the CFD bandwidth).
        panel_chunk: Panels per FW-H batch (bounds memory).
        obs_chunk: Observers per FW-H batch (bounds memory).

    Returns:
        ``(p, t_obs)``: total acoustic pressure ``p`` [Pa] shape ``[O, T_obs]``
        (thickness + loading, all rotors) and the shared observer-time grid
        ``t_obs`` [s] shape ``[T_obs]``.
    """
    points, normals, area = _geom(surface_geom)  # [S,3],[S,3],[S]
    positions, spins = _layout(vehicle_layout)  # [Nr,3],[Nr]
    n_rotors = positions.shape[0]
    n_panels = points.shape[0]

    lazy = bool(tiled_history.get("lazy", False))
    tau = np.asarray(tiled_history["tau"], dtype=np.float64)
    n_time = tau.shape[0]
    # Pre-bind the per-path locals (exactly one branch below fills each set).
    seg_rho = seg_u = seg_p = np.empty(0)
    rho_g = u_g = p_g = np.empty(0)
    xfade = 0
    n_out = n_time
    if lazy:
        # Small trimmed/de-meaned periodic segment; tiled per panel-chunk below.
        seg_rho = np.asarray(tiled_history["seg_rho"], dtype=np.float64)  # [S,seg]
        seg_u = np.asarray(tiled_history["seg_u"], dtype=np.float64)  # [S,seg,3]
        seg_p = np.asarray(tiled_history["seg_p"], dtype=np.float64)  # [S,seg]
        xfade = int(tiled_history["xfade"])
        n_out = int(tiled_history["n_out"])
        period_samples = int(tiled_history.get("period_samples", n_time))
    else:
        rho_g = np.asarray(tiled_history["rho"], dtype=np.float64)  # [S,T] gauge
        u_g = np.asarray(tiled_history["u"], dtype=np.float64)  # [S,T,3]
        p_g = np.asarray(tiled_history["p"], dtype=np.float64)  # [S,T] gauge
        period_samples = int(tiled_history.get("period_samples", rho_g.shape[1]))

    obs = np.asarray(observers, dtype=np.float64).reshape(-1, 3)
    n_o = obs.shape[0]
    c0 = float(medium.c0)
    rho0 = float(medium.rho0)
    p0 = float(medium.p0)

    direction = np.asarray(flight_dir, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    v_inf = float(speed) * direction  # [3]
    up = np.array([0.0, 0.0, 1.0])
    # Vehicle-translation term of the world position, per source time. [T,3]
    traj = v_inf[None, :] * (tau - float(t_pass))[:, None] + altitude * up[None, :]

    if phase_offsets is None:
        phase_offsets = [i / n_rotors for i in range(n_rotors)]
    shifts = [int(round(float(f) * period_samples)) % n_time for f in phase_offsets]

    # Per-rotor static local geometry (mirrored for counter-rotating rotors).
    # Eager: also build the rolled/ambient/boosted full histories now. Lazy: keep
    # only the per-rotor mirror flag; the [c,T] history chunks are tiled in the
    # panel loop (no full-length per-rotor arrays ever exist).
    rotor_local_pts: list[np.ndarray] = []
    rotor_local_nrm: list[np.ndarray] = []
    rotor_mirror: list[bool] = []
    rotor_rho: list[np.ndarray] = []
    rotor_u: list[np.ndarray] = []
    rotor_p: list[np.ndarray] = []
    for i in range(n_rotors):
        pts_i = points.copy()
        nrm_i = normals.copy()
        mirror_i = bool(spins[i] < 0)  # counter-rotating: mirror about xz-plane
        if mirror_i:
            pts_i = pts_i * _MIRROR_Y
            nrm_i = nrm_i * _MIRROR_Y
        # world-static part of the panel position (hub + local); the vehicle
        # translation (traj) is added per source time below.
        rotor_local_pts.append(positions[i][None, :] + pts_i)  # [S,3]
        rotor_local_nrm.append(nrm_i)
        rotor_mirror.append(mirror_i)
        if not lazy:
            rho_i = np.roll(rho_g, shifts[i], axis=1)
            p_i = np.roll(p_g, shifts[i], axis=1)
            u_i = np.roll(u_g, shifts[i], axis=1)
            if mirror_i:
                u_i = u_i * _MIRROR_Y
            rotor_rho.append(rho0 + rho_i)  # restore ambient (kernel wants absolute)
            rotor_p.append(p0 + p_i)
            u_feed = u_i + v_inf[None, None, :] if boost_u else u_i
            rotor_u.append(u_feed)

    # Shared observer-time grid over the arrival window (AABB-corner bound).
    all_static = np.concatenate(rotor_local_pts, axis=0)  # [Nr*S,3]
    lo = all_static.min(axis=0)
    hi = all_static.max(axis=0)
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )  # fmt: skip  [8,3]
    ends = np.stack([corners + traj[0], corners + traj[-1]], axis=0).reshape(-1, 3)  # [16,3]
    d = np.linalg.norm(obs[:, None, :] - ends[None, :, :], axis=-1)  # [O,16]
    t_lo = float(tau[0] + d.min() / c0)
    t_hi = float(tau[-1] + d.max() / c0)
    if n_obs is None:
        dtau = float(tau[1] - tau[0])
        n_obs = max(2, int(round((t_hi - t_lo) / dtau)) + 1)
    t_obs = jnp.linspace(t_lo, t_hi, n_obs)

    area_j = jnp.asarray(area)
    tau_j = jnp.asarray(tau)
    p_total = jnp.zeros((n_o, n_obs), dtype=jnp.float64)

    # Bound host/device memory regardless of the tiled length: the per-chunk
    # kernel materializes ~10 [chunk, T, 3] float64 temporaries, so cap
    # chunk*T at ~6.3M elements (~150 MB per [c,T,3] array). A finely sampled
    # surface (small CFD dt) otherwise turns panel_chunk=512 into tens of GB
    # per chunk -- the failure mode that OOM-killed the 145 kHz DJI runs.
    n_t = int(tau_j.shape[0])
    panel_chunk = max(16, min(int(panel_chunk), (6 << 20) // max(n_t, 1)))

    traj_j = jnp.asarray(traj)  # [T,3]
    rho_full = p_full = u_full = jnp.zeros(0)  # eager-only; filled per rotor below
    for i in range(n_rotors):
        local_pts = jnp.asarray(rotor_local_pts[i])  # [S,3]
        nrm = jnp.asarray(rotor_local_nrm[i])  # [S,3]
        shift_i = shifts[i]
        mirror_i = rotor_mirror[i]
        if not lazy:
            rho_full = jnp.asarray(rotor_rho[i])  # [S,T]
            p_full = jnp.asarray(rotor_p[i])  # [S,T]
            u_full = jnp.asarray(rotor_u[i])  # [S,T,3]
        for ps in range(0, n_panels, panel_chunk):
            pe = min(ps + panel_chunk, n_panels)
            if lazy:
                # Tile this panel-chunk to full length on the fly, then apply the
                # ambient restore / mirror / boost that the eager path applied to
                # the full arrays (bit-exact: tiling, roll and the y-mirror all act
                # independently per panel-row and per velocity component).
                seg_u_ch = seg_u[ps:pe] * _MIRROR_Y if mirror_i else seg_u[ps:pe]
                rho_ch = rho0 + _tile_chunk(seg_rho[ps:pe], xfade, n_out, shift_i)
                p_ch = p0 + _tile_chunk(seg_p[ps:pe], xfade, n_out, shift_i)
                u_ch = _tile_chunk(seg_u_ch, xfade, n_out, shift_i)
                if boost_u:
                    u_ch = u_ch + v_inf[None, None, :]
                rho_c = jnp.asarray(rho_ch)  # [c,T]
                p_c = jnp.asarray(p_ch)  # [c,T]
                u_c = jnp.asarray(u_ch)  # [c,T,3]
            else:
                rho_c = rho_full[ps:pe]
                p_c = p_full[ps:pe]
                u_c = u_full[ps:pe]
            # world position per source time: static local + vehicle translation.
            y = local_pts[ps:pe, None, :] + traj_j[None, :, :]  # [c,T,3]
            v = jnp.broadcast_to(jnp.asarray(v_inf), y.shape)  # [c,T,3]
            a = jnp.zeros_like(y)
            for os_ in range(0, n_o, obs_chunk):
                oe = min(os_ + obs_chunk, n_o)
                pt, pl = f1a_permeable(
                    jnp.asarray(obs[os_:oe]),
                    y,
                    v,
                    a,
                    rho_c,
                    u_c,
                    p_c,
                    nrm[ps:pe],
                    area_j[ps:pe],
                    medium,
                    tau_j,
                    t_obs,
                    include_steady_vn=False,
                )
                p_total = p_total.at[os_:oe].add(pt + pl)

    return p_total, t_obs


# --- audio synthesis ----------------------------------------------------------


def synthesize_flyover_wavs(
    p: ArrayLike,
    t_obs: ArrayLike,
    fs_out: float = 44100.0,
) -> np.ndarray:
    """Upsample CFD-rate mic pressures to an audio sample rate.

    The source content is band-limited by the CFD surface-sampling rate: the
    effective bandwidth is ``fs_src / 2 = 1 / (2*dt_obs)`` with ``dt_obs`` the
    ``t_obs`` spacing (for the default 1PAX hover run, ``dt_obs ~ 2e-4 s`` ->
    ``~2.2 kHz``, comfortably covering the first ~60 BPF harmonics). Because the
    signal already has no energy above ``fs_src/2 << fs_out/2``, linear
    interpolation onto the ``fs_out`` grid introduces no audible aliasing
    (window-safe upsampling); no anti-image low-pass is needed.

    Args:
        p: Mic pressures [Pa], shape ``[O, T_obs]`` on the uniform grid ``t_obs``.
        t_obs: Uniform observer-time grid [s], shape ``[T_obs]``.
        fs_out: Output audio sample rate [Hz].

    Returns:
        Upsampled pressures [Pa], shape ``[O, n]`` (float64), ``n`` spanning the
        same ``[t_obs[0], t_obs[-1]]`` window at ``fs_out``.
    """
    p = np.asarray(p, dtype=np.float64)
    t_obs = np.asarray(t_obs, dtype=np.float64).ravel()
    t0, t1 = float(t_obs[0]), float(t_obs[-1])
    n = int(round((t1 - t0) * float(fs_out))) + 1
    t_new = t0 + np.arange(n) / float(fs_out)
    out = np.empty((p.shape[0], n), dtype=np.float64)
    for o in range(p.shape[0]):
        out[o] = np.interp(t_new, t_obs, p[o])
    return out
