"""Rotor-blade and rotor triangle meshes lofted from :class:`BladeGeometry`.

Bridges the *rotor world* (:mod:`auraflow.core.blade`) to the *general-body
world* (:mod:`auraflow.body`): a :class:`~auraflow.core.blade.BladeGeometry`
(radial stations of chord + twist) plus a 2D section profile
(:mod:`auraflow.body.airfoil_profile`) is lofted into a watertight, outward-wound
:class:`~auraflow.body.mesh.TriMesh`, and a :class:`~auraflow.core.blade.Rotor`
into a multi-blade mesh. Those feed resolved-blade level-set CFD
(:func:`rotor_levelset_case`) and mesh-based rotor acoustics
(:func:`auraflow.body.sources.mesh_pressure`).

Frames and conventions (``docs/architecture.md`` -> "Library conventions"):

- **Blade section frame** (the frame :func:`blade_mesh` builds in): x spanwise
  outward, y chordwise toward the leading edge (in the direction of rotation),
  z thrust-normal. For a blade at azimuth ``psi = 0`` this frame coincides with
  the rotor frame, so :func:`rotor_mesh` places blade ``b`` by rotating the
  section-frame mesh by ``rot_z(psi_b)`` about the thrust axis.
- **Pitch axis**: each section is placed with its **quarter-chord point at
  ``y = z = 0``** (on the spanwise axis) -- exactly the compact-source location
  used by :meth:`auraflow.core.blade.BladeGeometry.quarter_chord_points`, so the
  resolved mesh and the compact-blade acoustics share one pitch axis.
- **Twist / pitch**: the section is pitched by ``twist`` about the spanwise
  (``+x``) axis; positive twist is nose-up (leading edge, at ``+y``, rotates
  toward ``+z``), matching :attr:`BladeGeometry.twist`.
- **Winding**: the profile loops (counterclockwise in ``(xi, eta)``, see
  :mod:`auraflow.body.airfoil_profile`) are lofted root-to-tip with a winding
  that makes every face normal point outward (verified by ``volume() > 0``).

Static vs traced: station/chord counts and connectivity are static; vertices
(from chord/twist/radius) are traced and differentiable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from auraflow.body.mesh import TriMesh
from auraflow.body.motion import SpinMotion
from auraflow.core.blade import BladeGeometry, Rotor
from auraflow.core.frames import rot_z

__all__ = [
    "blade_mesh",
    "rotor_levelset_case",
    "rotor_mesh",
]

# A profile is a callable n_points -> [N, 2] section loop (unit chord); the
# default is auraflow.body.airfoil_profile.naca0012.
ProfileFn = Callable[[int], Array]

# Chordwise fraction of the pitch axis (quarter chord). The section origin
# (y = 0) sits here, matching BladeGeometry.quarter_chord_points.
_PITCH_AXIS_FRAC = 0.25


def _default_profile(n_chord: int) -> Array:
    # Imported lazily so ``import auraflow.body.blade`` does not require the
    # airfoil module at module-load (keeps the base import surface minimal).
    from auraflow.body.airfoil_profile import naca0012

    return naca0012(n_chord)


def _section_vertices(loop2d: Array, geometry: BladeGeometry) -> Array:
    """Per-station section loops in the blade section frame, shape ``[S, N, 3]``.

    Each unit-chord section point ``(xi, eta)`` becomes a 3D vertex:
    ``x = r`` (spanwise station), and the in-plane ``(y, z)`` are the
    chordwise/thickness offsets from the quarter-chord pitch axis scaled by the
    local chord and pitched by the local twist about the spanwise axis:

    - ``y0 = (0.25 - xi) * chord`` (leading edge ``xi = 0`` -> ``+y``),
    - ``z0 = eta * chord`` (thickness, ``+z`` upper surface),
    - pitch by ``twist`` about ``+x``: ``y = y0 cos t - z0 sin t``,
      ``z = y0 sin t + z0 cos t`` (nose-up for ``twist > 0``).
    """
    xi = loop2d[:, 0]  # [N]
    eta = loop2d[:, 1]  # [N]
    chord = geometry.chord[:, None]  # [S, 1]
    twist = geometry.twist[:, None]  # [S, 1]
    r = geometry.r[:, None]  # [S, 1]

    y0 = (_PITCH_AXIS_FRAC - xi)[None, :] * chord  # [S, N]
    z0 = eta[None, :] * chord  # [S, N]
    ct, st = jnp.cos(twist), jnp.sin(twist)
    y = y0 * ct - z0 * st
    z = y0 * st + z0 * ct
    x = jnp.broadcast_to(r, y.shape)
    return jnp.stack([x, y, z], axis=-1)  # [S, N, 3]


def _loft_faces(n_stations: int, n_loop: int, root_cap: bool, tip_cap: bool) -> np.ndarray:
    """Static triangle connectivity for the lofted blade (outward winding).

    Vertices are laid out as ``S`` section rings of ``N`` points each
    (``idx(i, k) = i * N + k``), optionally followed by a root-centre vertex and
    a tip-centre vertex used to fan-cap the end rings. The side and cap winding
    below is fixed (topological) and yields ``volume() > 0`` for the CCW section
    loops of :mod:`auraflow.body.airfoil_profile`.
    """
    n, s = n_loop, n_stations
    faces: list[tuple[int, int, int]] = []

    def idx(i: int, k: int) -> int:
        return i * n + (k % n)

    # Side surface: quads between consecutive rings, split into two triangles.
    for i in range(s - 1):
        for k in range(n):
            a = idx(i, k)
            b = idx(i, k + 1)
            c = idx(i + 1, k + 1)
            d = idx(i + 1, k)
            faces.append((a, d, c))
            faces.append((a, c, b))

    base = s * n
    if root_cap:
        cr = base  # root-centre vertex index
        for k in range(n):
            faces.append((cr, idx(0, k), idx(0, k + 1)))
    if tip_cap:
        ct = base + (1 if root_cap else 0)  # tip-centre vertex index
        for k in range(n):
            faces.append((ct, idx(s - 1, k + 1), idx(s - 1, k)))
    return np.asarray(faces, dtype=np.int64)


def blade_mesh(
    geometry: BladeGeometry,
    *,
    profile: ProfileFn = _default_profile,
    n_chord: int = 60,
    root_cap: bool = True,
    tip_cap: bool = True,
) -> TriMesh:
    """Loft a :class:`BladeGeometry` into a watertight blade :class:`TriMesh`.

    The 2D section ``profile`` (unit chord) is placed at every radial station,
    scaled by the station chord, pitched by the station twist, and stacked along
    the span; consecutive sections are connected by triangles and the root/tip
    rings are fan-capped. The result is built in the **blade section frame**
    (x spanwise, y chordwise toward the leading edge, z thrust-normal) with the
    quarter chord on the spanwise axis (see the module docstring), and is
    outward-wound (``volume() > 0``).

    Args:
        geometry: The blade geometry (radial stations, chord, twist).
        profile: Section factory ``n_points -> [N, 2]`` returning a closed CCW
            unit-chord loop (default :func:`auraflow.body.airfoil_profile.naca0012`).
        n_chord: Chordwise samples per surface handed to ``profile`` (static
            int). The section loop then has ``N = 2 * n_chord - 2`` vertices.
        root_cap: Close the root ring with a fan cap (default ``True``).
        tip_cap: Close the tip ring with a fan cap (default ``True``).

    Returns:
        A blade :class:`~auraflow.body.mesh.TriMesh` in the blade section frame;
        watertight when both caps are on. Differentiable through
        ``geometry.chord``/``twist``/``radius``.
    """
    loop2d = jnp.asarray(profile(n_chord), dtype=jnp.float64)  # [N, 2]
    if loop2d.ndim != 2 or loop2d.shape[1] != 2:
        raise ValueError(f"profile must return an [N, 2] loop, got {loop2d.shape}")
    n_loop = int(loop2d.shape[0])
    s = geometry.n_stations

    rings = _section_vertices(loop2d, geometry)  # [S, N, 3]
    verts = rings.reshape(s * n_loop, 3)

    extra: list[Array] = []
    if root_cap:
        extra.append(jnp.mean(rings[0], axis=0))  # root-ring centroid
    if tip_cap:
        extra.append(jnp.mean(rings[-1], axis=0))  # tip-ring centroid
    if extra:
        verts = jnp.concatenate([verts, jnp.stack(extra, axis=0)], axis=0)

    faces = _loft_faces(s, n_loop, root_cap, tip_cap)
    return TriMesh(verts, faces, is_watertight=bool(root_cap and tip_cap))


def rotor_mesh(
    rotor: Rotor,
    *,
    azimuths: Array | None = None,
    hub: bool | dict[str, Any] = False,
    profile: ProfileFn = _default_profile,
    n_chord: int = 60,
    root_cap: bool = True,
    tip_cap: bool = True,
) -> TriMesh:
    """Loft every blade of a :class:`Rotor` into a single rotor-frame mesh.

    One :func:`blade_mesh` is built (in the section frame) and placed at each
    blade azimuth by ``rot_z(psi)`` about the thrust axis, then all blades (and
    an optional hub cylinder) are merged with :meth:`TriMesh.merge`. The mesh is
    expressed in the **rotor frame** (origin at the hub, ``+z`` the thrust axis).

    Args:
        rotor: The rotor (shared blade geometry + blade count + spin sense).
        azimuths: Blade azimuths [rad], shape ``[B]`` (default
            :meth:`Rotor.blade_azimuths` at ``psi0 = 0``, which already encodes
            the equal spacing and spin direction).
        hub: ``False`` for no hub, ``True`` for a default hub cylinder (radius =
            root cutout, height = ``0.5 * root cutout``), or a dict of
            :meth:`TriMesh.cylinder` params (``radius``/``height``/``n``).
        profile, n_chord, root_cap, tip_cap: Passed through to :func:`blade_mesh`.

    Returns:
        A rotor :class:`~auraflow.body.mesh.TriMesh` in the rotor frame,
        watertight when the blades are (each blade + hub is its own closed
        component; :meth:`TriMesh.volume` sums over them).
    """
    blade = blade_mesh(
        rotor.blade, profile=profile, n_chord=n_chord, root_cap=root_cap, tip_cap=tip_cap
    )
    if azimuths is None:
        azimuths = rotor.blade_azimuths(0.0)
    azimuths = jnp.asarray(azimuths, dtype=jnp.float64)

    parts: list[TriMesh] = []
    for b in range(int(azimuths.shape[0])):
        rmat = rot_z(azimuths[b])  # [3, 3]
        placed = jnp.einsum("ij,vj->vi", rmat, blade.vertices)
        parts.append(TriMesh(placed, blade.faces, is_watertight=blade.is_watertight))

    if hub:
        params = {} if hub is True else dict(hub)
        r_hub = float(jnp.asarray(rotor.blade.hub_radius))
        radius = params.pop("radius", r_hub)
        height = params.pop("height", 0.5 * r_hub)
        parts.append(TriMesh.cylinder(radius=radius, height=height, **params))

    return TriMesh.merge(parts)


def _resolve_omega(omega: Any, omega_times: Any) -> tuple[float | tuple[Any, Any], float]:
    """Normalize an ``omega`` spec to (case-omega, peak |Omega|).

    Returns either a float (constant rate) or a ``(times, omegas)`` table, plus
    the peak absolute rate used for the tip-Mach timestep bound. A callable is
    sampled onto ``omega_times`` (required for the callable form) to build the
    table; a table is passed through; a scalar stays constant.
    """
    if callable(omega):
        if omega_times is None:
            raise ValueError(
                "A callable omega(t) needs omega_times=<time grid [s]> so it can be "
                "tabulated into the (differentiable) prescribed solid velocity."
            )
        times = np.asarray(omega_times, dtype=float).ravel()
        omega_fn: Any = omega  # Any (callable() would narrow the return to object)
        omegas = np.asarray([float(omega_fn(t)) for t in times.tolist()], dtype=float)
        return (times, omegas), float(np.max(np.abs(omegas)))
    if isinstance(omega, tuple):
        times, omegas = omega
        omegas = np.asarray(omegas, dtype=float).ravel()
        return (np.asarray(times, dtype=float).ravel(), omegas), float(np.max(np.abs(omegas)))
    return float(omega), abs(float(omega))


def _cell_center_points(box_lo: Any, box_hi: Any, cells: tuple[int, int, int]) -> Array:
    """Interior cell-centre coordinates as an ``[nx*ny*nz, 3]`` point cloud.

    Matches :func:`auraflow.cfd.body_case._cell_center_sdf`: centres are
    ``linspace(lo + d/2, hi - d/2, n)`` per axis (``ij`` meshgrid order).
    """
    lo = np.asarray(box_lo, dtype=float).ravel()
    hi = np.asarray(box_hi, dtype=float).ravel()
    n = np.asarray(cells)
    d = (hi - lo) / n
    cc_lo = lo + 0.5 * d
    cc_hi = hi - 0.5 * d
    xs = np.linspace(cc_lo[0], cc_hi[0], n[0])
    ys = np.linspace(cc_lo[1], cc_hi[1], n[1])
    zs = np.linspace(cc_lo[2], cc_hi[2], n[2])
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return jnp.asarray(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1))


def _compose_rotor_levelset(
    rotor: Rotor,
    *,
    box_lo: Any,
    box_hi: Any,
    cells: tuple[int, int, int],
    axis: Any,
    center: Any,
    hub: bool | dict[str, Any],
    profile: ProfileFn,
    n_chord: int,
    initial_azimuth: float,
    blade_cells: int | tuple[int, int, int] | None,
    blade_padding: float | None,
    cache: bool,
    cache_dir: str | None,
    batch_points: int,
) -> Array:
    """Build the initial level-set grid by composing ONE canonical blade SDF.

    Lofts a single blade, builds its :class:`~auraflow.body.sdf_compose.CanonicalSDF`
    in a tight box (disk-cached), then evaluates the whole-rotor SDF
    (:func:`~auraflow.body.sdf_compose.rotor_sdf`: the one blade at every azimuth
    plus an analytic hub cylinder) at the CFD cell centres. The blade grid is
    sized so its spacing matches the CFD spacing.
    """
    from auraflow.body.sdf_compose import CanonicalSDF, rotor_sdf

    lo = np.asarray(box_lo, dtype=float).ravel()
    hi = np.asarray(box_hi, dtype=float).ravel()
    dx = float(np.min((hi - lo) / np.asarray(cells)))
    pad = 4.0 * dx if blade_padding is None else float(blade_padding)

    blade = blade_mesh(rotor.blade, profile=profile, n_chord=n_chord)
    if blade_cells is None:
        verts = np.asarray(blade.vertices)
        extent = (verts.max(axis=0) - verts.min(axis=0)) + 2.0 * pad
        e = [int(np.clip(np.ceil(v / dx), 8, 160)) for v in extent]
        bcells: tuple[int, int, int] = (e[0], e[1], e[2])
    elif isinstance(blade_cells, int):
        bcells = (blade_cells, blade_cells, blade_cells)
    else:
        bc = tuple(blade_cells)
        bcells = (bc[0], bc[1], bc[2])

    blade_sdf = CanonicalSDF.from_mesh(
        blade,
        padding=pad,
        cells=bcells,
        cache=cache,
        cache_dir=cache_dir,
        batch_points=batch_points,
    )

    hub_params: dict[str, Any] | None = None
    if hub:
        params = {} if hub is True else dict(hub)
        r_hub = float(jnp.asarray(rotor.blade.hub_radius))
        radius = float(params.get("radius", r_hub))
        height = float(params.get("height", 0.5 * r_hub))
        hub_params = {"radius": radius, "half_height": 0.5 * height}

    sdf_fn = rotor_sdf(
        blade_sdf,
        n_blades=rotor.n_blades,
        azimuth=initial_azimuth,
        axis=axis,
        center=center,
        spin_direction=rotor.spin_direction,
        hub=hub_params,
    )
    pts = _cell_center_points(box_lo, box_hi, cells)
    return sdf_fn(pts).reshape(cells)


def rotor_levelset_case(
    rotor_or_mesh: Rotor | TriMesh,
    omega: Any,
    *,
    box_lo: Any,
    box_hi: Any,
    cells: tuple[int, int, int],
    axis: Any = (0.0, 0.0, 1.0),
    center: Any = (0.0, 0.0, 0.0),
    medium: Any = None,
    hub: bool | dict[str, Any] = False,
    profile: ProfileFn = _default_profile,
    n_chord: int = 60,
    mach_max: float | None = None,
    method: str = "compose",
    initial_azimuth: float = 0.0,
    omega_times: Any = None,
    blade_cells: int | tuple[int, int, int] | None = None,
    blade_padding: float | None = None,
    sdf_cache: bool = True,
    sdf_cache_dir: str | None = None,
    sdf_batch_points: int = 4096,
    **case_kwargs: Any,
) -> Any:
    """Resolved spinning-blade FLUID-SOLID level-set CFD case for a rotor.

    Immerses a rotor in a JAX-Fluids level-set box as a **prescribed spinning
    solid** and returns a :func:`auraflow.cfd.body_case.levelset_body_case`. This
    is the real backing for :func:`auraflow.cfd.case.rotor_box_case`
    ``method="levelset_blades"``.

    **Initial level-set (``method``)**:

    - ``"compose"`` (default, the RPM/azimuth-reuse core of issue #2): build the
      SDF of ONE blade once (a small, cheap grid in the blade's own tight box,
      disk-cached), then assemble the whole-rotor level-set by evaluating that
      single canonical blade at every azimuth (plus an analytic hub cylinder) on
      the CFD cell centres -- fast trilinear lookups, **no per-configuration mesh
      SDF build**. So one blade SDF serves every blade count, every RPM and every
      ``initial_azimuth``. Requires a :class:`~auraflow.core.blade.Rotor`.
    - ``"mesh"``: loft the full rotor mesh and build its SDF directly with the
      GPU brute-force kernel (:func:`auraflow.body.sdf.sdf_grid_jax`) -- the
      escape hatch / cross-validation path. Works for a ready ``TriMesh`` too.
    - ``"trimesh"``: as ``"mesh"`` but via the legacy exact ``trimesh`` SDF.

    **RPM (``omega``)**: the initial level-set is **independent of RPM** (only
    ``initial_azimuth`` enters it); JAX-Fluids advects it thereafter with the
    prescribed solid velocity ``v = Omega(t) * (axis x (X - center))``. ``omega``
    may be:

    - a float -> constant rate (a :func:`auraflow.body.motion.SpinMotion.constant`);
    - a ``(times, omegas)`` table -> a time-varying rate advected via a
      ``jnp.interp`` closure (differentiable, constant-extrapolated);
    - a callable ``Omega(t)`` -> sampled onto ``omega_times`` into such a table.

    Args:
        rotor_or_mesh: A :class:`~auraflow.core.blade.Rotor` (required for
            ``method="compose"``; lofted here otherwise) or a ready rotor
            :class:`TriMesh` (rotor frame; forces a mesh SDF path).
        omega: Rotor angular rate [rad/s]: ``float`` | ``(times, omegas)`` table
            | callable ``Omega(t)`` (see above). Signed, right-handed about ``axis``.
        box_lo: Lower box corner [m], shape ``[3]``.
        box_hi: Upper box corner [m], shape ``[3]``.
        cells: ``(nx, ny, nz)`` cell counts (all ``> 1``).
        axis: Rotor thrust/rotation axis [m], shape ``[3]`` (default ``+z``).
        center: A point on the rotation axis (the hub) [m], shape ``[3]``.
        medium: Ambient :class:`~auraflow.core.medium.Medium` (default ISA).
        hub: Hub option (``True``/dict). For ``"compose"`` an analytic capped
            cylinder (radius = root cutout, height = ``0.5 *`` root cutout by
            default); for ``"mesh"`` forwarded to :func:`rotor_mesh`.
        profile, n_chord: Blade lofting options (see :func:`rotor_mesh`).
        mach_max: Peak surface Mach bounding the timestep; default is the tip
            Mach ``max|Omega| * R / c0``.
        method: Initial-level-set build method (see above).
        initial_azimuth: Reference-blade azimuth [rad] of the *initial* level-set.
        omega_times: Time grid [s] to tabulate a callable ``omega`` (required
            only for the callable form).
        blade_cells: Canonical-blade SDF grid size (``"compose"`` only; default
            sized to the CFD spacing).
        blade_padding: Canonical-blade box padding [m] (``"compose"`` only;
            default ``4 *`` CFD spacing).
        sdf_cache, sdf_cache_dir, sdf_batch_points: Canonical-blade SDF disk-cache
            + memory knobs (``"compose"`` only).
        **case_kwargs: Extra keyword args for
            :func:`auraflow.cfd.body_case.levelset_body_case` (e.g. ``cfl``,
            ``end_time``, ``sponge_thickness``, ``is_double``, ``case_name``).

    Returns:
        A :class:`~auraflow.cfd.body_case.LevelsetBodyCase` (prescribed-moving
        solid), ready for :func:`auraflow.cfd.run.run_acoustic_case` on GPU.
    """
    # Imported here (not at module top) to avoid an import cycle: cfd.body_case
    # -> cfd.case -> (lazily) body.blade.
    from auraflow.cfd.body_case import levelset_body_case, spin_solid_velocity
    from auraflow.core.medium import Medium

    medium = Medium() if medium is None else medium
    omega_spec, peak_omega = _resolve_omega(omega, omega_times)

    is_rotor = isinstance(rotor_or_mesh, Rotor)
    if is_rotor:
        rotor = rotor_or_mesh
        tip_r = float(jnp.asarray(rotor.blade.radius))
    else:
        rotor = None
        if method == "compose":
            method = "mesh"  # a raw mesh cannot be decomposed into one blade
        xy = np.asarray(rotor_or_mesh.vertices)[:, :2]
        tip_r = float(np.max(np.linalg.norm(xy, axis=-1)))

    if mach_max is None:
        mach_max = peak_omega * tip_r / float(medium.c0)

    # Initial level-set field + the mesh handed to levelset_body_case.
    mesh: TriMesh | None
    levelset_init: Array | None
    sdf_method = "jax"
    if method == "compose":
        assert rotor is not None
        levelset_init = _compose_rotor_levelset(
            rotor,
            box_lo=box_lo,
            box_hi=box_hi,
            cells=cells,
            axis=axis,
            center=center,
            hub=hub,
            profile=profile,
            n_chord=n_chord,
            initial_azimuth=initial_azimuth,
            blade_cells=blade_cells,
            blade_padding=blade_padding,
            cache=sdf_cache,
            cache_dir=sdf_cache_dir,
            batch_points=sdf_batch_points,
        )
        mesh = None
    elif method in ("mesh", "trimesh"):
        mesh = (
            rotor_mesh(rotor_or_mesh, hub=hub, profile=profile, n_chord=n_chord)
            if isinstance(rotor_or_mesh, Rotor)
            else rotor_or_mesh
        )
        levelset_init = None
        sdf_method = "jax" if method == "mesh" else "trimesh"
    else:
        raise ValueError(
            f"rotor_levelset_case method must be 'compose'/'mesh'/'trimesh', got {method!r}"
        )

    # Prescribed solid velocity: constant rate -> SpinMotion path; time-varying
    # rate -> an explicit interp-closure override (the RPM is not baked into the
    # level-set, only advection).
    if isinstance(omega_spec, tuple):
        motion = None
        solid_velocity = spin_solid_velocity(axis, center, omega_spec)
    else:
        motion = SpinMotion.constant(axis=axis, omega=omega_spec, center=center)
        solid_velocity = None

    return levelset_body_case(
        mesh,
        motion,
        box_lo=box_lo,
        box_hi=box_hi,
        cells=cells,
        medium=medium,
        mach_max=mach_max,
        solid_velocity=solid_velocity,
        levelset_init=levelset_init,
        sdf_method=sdf_method,
        **case_kwargs,
    )
