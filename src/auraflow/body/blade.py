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


def rotor_levelset_case(
    rotor_or_mesh: Rotor | TriMesh,
    omega: float,
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
    **case_kwargs: Any,
) -> Any:
    """Resolved spinning-blade FLUID-SOLID level-set CFD case for a rotor.

    Builds (or accepts) a rotor :class:`~auraflow.body.mesh.TriMesh` and immerses
    it in a JAX-Fluids level-set box as a **prescribed constant-rate spinning
    solid**: :func:`auraflow.cfd.body_case.levelset_body_case` with a
    :func:`auraflow.body.motion.SpinMotion.constant` about ``axis`` through
    ``center`` at rate ``omega``. This is the real backing for
    :func:`auraflow.cfd.case.rotor_box_case` ``method="levelset_blades"``.

    Args:
        rotor_or_mesh: A :class:`~auraflow.core.blade.Rotor` (lofted here via
            :func:`rotor_mesh`) or a ready rotor :class:`TriMesh` (rotor frame).
        omega: Constant rotor angular rate [rad/s] (signed, right-handed about
            ``axis``).
        box_lo: Lower box corner [m], shape ``[3]``.
        box_hi: Upper box corner [m], shape ``[3]``.
        cells: ``(nx, ny, nz)`` cell counts (all ``> 1``).
        axis: Rotor thrust/rotation axis [m], shape ``[3]`` (default ``+z``).
        center: A point on the rotation axis (the hub) [m], shape ``[3]``.
        medium: Ambient :class:`~auraflow.core.medium.Medium` (default ISA).
        hub: Hub option forwarded to :func:`rotor_mesh` (ignored if a mesh is
            passed).
        profile, n_chord: Forwarded to :func:`rotor_mesh` (ignored for a mesh).
        mach_max: Peak surface Mach bounding the timestep; default is the tip
            Mach ``|omega| * R / c0`` from the rotor/mesh extent.
        **case_kwargs: Extra keyword args for
            :func:`auraflow.cfd.body_case.levelset_body_case` (e.g. ``cfl``,
            ``end_time``, ``sponge_thickness``, ``is_double``, ``case_name``).

    Returns:
        A :class:`~auraflow.cfd.body_case.LevelsetBodyCase` (prescribed-moving
        solid), ready for :func:`auraflow.cfd.run.run_acoustic_case` on GPU.
    """
    # Imported here (not at module top) to avoid an import cycle: cfd.body_case
    # -> cfd.case -> (lazily) body.blade.
    from auraflow.cfd.body_case import levelset_body_case
    from auraflow.core.medium import Medium

    if isinstance(rotor_or_mesh, Rotor):
        mesh = rotor_mesh(rotor_or_mesh, hub=hub, profile=profile, n_chord=n_chord)
        tip_r = float(jnp.asarray(rotor_or_mesh.blade.radius))
    else:
        mesh = rotor_or_mesh
        # Rotor frame: radius ~ max in-plane (xy) distance from the axis.
        xy = np.asarray(mesh.vertices)[:, :2]
        tip_r = float(np.max(np.linalg.norm(xy, axis=-1)))

    medium = Medium() if medium is None else medium
    if mach_max is None:
        mach_max = abs(float(omega)) * tip_r / float(medium.c0)

    motion = SpinMotion.constant(axis=axis, omega=omega, center=center)
    return levelset_body_case(
        mesh,
        motion,
        box_lo=box_lo,
        box_hi=box_hi,
        cells=cells,
        medium=medium,
        mach_max=mach_max,
        **case_kwargs,
    )
