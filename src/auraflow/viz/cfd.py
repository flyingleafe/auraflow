"""Turn CFD driver state into live-viz scene/frame payloads.

Helpers the CFD driver (:func:`auraflow.cfd.run.run_acoustic_case`) calls when a
:class:`~auraflow.viz.server.VizStreamer` is attached: build the static scene
(domain box, permeable-sphere point cloud, field-slice plane) once, then per
sample step reduce the interior primitive buffer to a downsampled mid-plane
slice plus the acoustic overpressure ``p' = p - p0`` on the sphere points.

Kept out of :mod:`auraflow.cfd` so importing the CFD backend never pulls the
``viz-live`` extra. Inputs are ordinary NumPy arrays (the driver does a single
``device_get``); no JAX here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from auraflow.viz.stream import downsample_slice

if TYPE_CHECKING:  # pragma: no cover - typing only
    from auraflow.cfd.case import BoxDomain
    from auraflow.cfd.sphere import PermeableSphere

__all__ = ["cfd_frame_kwargs", "cfd_scene_kwargs"]

# Primitive-variable channel order in the interior buffer [5, Nx, Ny, Nz].
_FIELD_INDEX = {"rho": 0, "u": 1, "v": 2, "w": 3, "p": 4}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _plane_ranges(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, slice_axis: str
) -> tuple[list[float], list[float]]:
    """In-plane ``(u_range, v_range)`` [m] of the mid-plane normal to ``slice_axis``."""
    coords = {"x": x, "y": y, "z": z}
    plane = [a for a in ("x", "y", "z") if a != slice_axis]
    out = []
    for a in plane:
        c = coords[a]
        out.append([float(c[0]), float(c[-1])])
    return out[0], out[1]


def cfd_scene_kwargs(
    domain: BoxDomain,
    sphere: PermeableSphere,
    *,
    field: str = "p",
    slice_axis: str = "z",
    max_size: int = 64,
    title: str = "CFD acoustic case",
) -> dict[str, Any]:
    """Assemble :meth:`VizStreamer.init_scene` kwargs for a CFD case.

    Args:
        domain: The :class:`~auraflow.cfd.case.BoxDomain` being simulated.
        sphere: The permeable :class:`~auraflow.cfd.sphere.PermeableSphere`.
        field: Primitive field shown on the slice (``"rho"`` or ``"p"``).
        slice_axis: Axis the field slice is taken normal to (``"x"``/``"y"``/``"z"``).
        max_size: Max slice samples per axis (frontend texture resolution).
        title: Scene title.

    Returns:
        Kwargs dict for :meth:`auraflow.viz.server.VizStreamer.init_scene`.
    """
    x, y, z = (np.asarray(c, dtype=float) for c in domain.cell_centers())
    u_range, v_range = _plane_ranges(x, y, z, slice_axis)
    coord = {"x": x, "y": y, "z": z}[slice_axis]
    return {
        "box_min": [float(x[0]), float(y[0]), float(z[0])],
        "box_max": [float(x[-1]), float(y[-1]), float(z[-1])],
        "sphere_points": np.asarray(sphere.points, dtype=np.float32),
        "fields": [field],
        "slice_plane": {
            "axis": slice_axis,
            "coord": float(coord[len(coord) // 2]),
            "u_axis": [a for a in ("x", "y", "z") if a != slice_axis][0],
            "v_axis": [a for a in ("x", "y", "z") if a != slice_axis][1],
            "u_range": u_range,
            "v_range": v_range,
        },
        "title": title,
    }


def cfd_frame_kwargs(
    interior: Any,
    x: Any,
    y: Any,
    z: Any,
    sphere_p: Any,
    p0: float,
    t: float,
    step: int,
    *,
    field: str = "p",
    slice_axis: str = "z",
    max_size: int = 64,
) -> dict[str, Any]:
    """Reduce one CFD sample to :meth:`VizStreamer.push_frame` kwargs.

    Args:
        interior: Interior primitives ``[5, Nx, Ny, Nz]`` (NumPy, halos stripped),
            ordered ``(rho, u, v, w, p)``.
        x, y, z: Cell-centre coordinate arrays (only shapes are used here).
        sphere_p: Pressure at the sphere points ``[S]`` [Pa].
        p0: Ambient pressure [Pa]; subtracted to give overpressure ``p'``.
        t: Physical time of this sample [s].
        step: Sample index.
        field: Which primitive to slice (``"rho"``/``"p"``).
        slice_axis: Axis the slice is normal to.
        max_size: Max slice samples per axis (block-mean downsampled).

    Returns:
        Kwargs dict for :meth:`auraflow.viz.server.VizStreamer.push_frame`.
    """
    prim = np.asarray(interior)
    field3d = prim[_FIELD_INDEX[field]]  # [Nx, Ny, Nz]
    axis = _AXIS_INDEX[slice_axis]
    mid = field3d.shape[axis] // 2
    plane = np.take(field3d, mid, axis=axis)  # 2-D
    if field == "p":
        plane = plane - p0
    slice_ds = downsample_slice(plane, max_size)
    sp = np.asarray(sphere_p, dtype=float) - p0
    lo = float(min(slice_ds.min(), sp.min()))
    hi = float(max(slice_ds.max(), sp.max()))
    return {
        "t": float(t),
        "step": int(step),
        "field_slice": slice_ds,
        "slice_range": (lo, hi),
        "sphere_p": sp.astype(np.float32),
    }
