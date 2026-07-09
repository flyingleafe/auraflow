"""Stream a general body (``auraflow.body``) to the live viewer.

Replays a :class:`~auraflow.body.mesh.TriMesh` carried by a
:class:`~auraflow.body.motion.Motion` as an animation: the rest (body-frame)
mesh is published once in the scene message and the per-frame **pose** channel
(``mesh_poses``, protocol v2) moves it, exactly the way
:mod:`auraflow.viz.flyover` animates the rotor disks. Like that module this is
plain NumPy on already-materialised host arrays (the motion poses are evaluated
on CPU) -- no CFD, no device compute, so it is cheap and safe on the dev box.

An optional per-face surface scalar (e.g. a loudspeaker membrane's velocity or a
CFD surface pressure ``p'``) is aggregated to per-vertex colours and shipped with
the scene mesh, shading the body by that field.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from auraflow.body.mesh import TriMesh
    from auraflow.body.motion import Motion
    from auraflow.viz.server import VizStreamer

__all__ = ["body_scene", "stream_body", "vertex_colors_from_face_scalar"]


def _pose_np(motion: Motion, t: float) -> tuple[np.ndarray, np.ndarray]:
    """World-from-body pose ``(R, x)`` at time ``t`` as host NumPy arrays."""
    r, x = motion.pose(float(t))
    return np.asarray(r, dtype=float), np.asarray(x, dtype=float)


def _diverging(u: np.ndarray) -> np.ndarray:
    """Blue-white-red diverging map of ``u`` in ``[0, 1]`` -> ``[N, 3]`` RGB."""
    u = np.clip(u, 0.0, 1.0)
    x = 2.0 * u - 1.0  # [-1, 1]
    r = np.where(x < 0, 1.0 + x, 1.0)
    g = np.where(x < 0, 1.0 + x, 1.0 - x)
    b = np.where(x < 0, 1.0, 1.0 - x)
    return np.stack([r, g, b], axis=-1)


def vertex_colors_from_face_scalar(
    mesh: TriMesh, face_scalar: Any, *, symmetric: bool | None = None
) -> np.ndarray:
    """Per-vertex RGB colours from a per-face scalar (area-agnostic averaging).

    The face scalar is scattered to each face's three vertices, averaged per
    vertex, normalised, and mapped through a blue-white-red diverging colormap.

    Args:
        mesh: The :class:`~auraflow.body.mesh.TriMesh`.
        face_scalar: Per-face scalar, shape ``[F]`` (e.g. RMS surface pressure).
        symmetric: If ``True`` centre the colour scale on zero (signed field); if
            ``None`` it is inferred from whether the scalar changes sign.

    Returns:
        Per-vertex colours, shape ``[V, 3]`` (0..1 RGB), for ``encode_scene``.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    fs = np.asarray(face_scalar, dtype=float).reshape(-1)
    n_v = int(np.asarray(mesh.vertices).shape[0])
    acc = np.zeros(n_v)
    cnt = np.zeros(n_v)
    for k in range(3):
        np.add.at(acc, faces[:, k], fs)
        np.add.at(cnt, faces[:, k], 1.0)
    vs = acc / np.maximum(cnt, 1.0)
    if symmetric is None:
        symmetric = bool(vs.min() < 0.0 < vs.max())
    if symmetric:
        amax = float(np.max(np.abs(vs))) or 1.0
        u = 0.5 + 0.5 * vs / amax
    else:
        lo, hi = float(vs.min()), float(vs.max())
        u = (vs - lo) / ((hi - lo) or 1.0)
    return _diverging(u)


def _swept_bounds(
    mesh: TriMesh, motion: Motion | None, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds of the mesh swept over the pose samples ``times``."""
    verts = np.asarray(mesh.vertices, dtype=float)  # [V, 3]
    if motion is None:
        return verts.min(axis=0), verts.max(axis=0)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for t in times:
        r, x = _pose_np(motion, float(t))
        world = verts @ r.T + x
        lo = np.minimum(lo, world.min(axis=0))
        hi = np.maximum(hi, world.max(axis=0))
    return lo, hi


def body_scene(
    mesh: TriMesh,
    motion: Motion | None = None,
    times: Any | None = None,
    *,
    mics: Any | None = None,
    pad: float = 1.0,
    opacity: float = 0.6,
    color: Any | None = None,
    face_scalar: Any | None = None,
    name: str = "body",
    title: str = "AuraFlow body",
) -> dict[str, Any]:
    """Assemble :func:`auraflow.viz.stream.encode_scene` kwargs for a body replay.

    The scene mesh is the **rest** (body-frame) geometry; the per-frame pose
    channel (:func:`stream_body`) animates it. The domain box bounds the mesh
    swept over ``times`` (and the mics), padded by ``pad`` metres.

    Args:
        mesh: The body :class:`~auraflow.body.mesh.TriMesh`.
        motion: The :class:`~auraflow.body.motion.Motion` (for the swept bounds);
            ``None`` bounds the rest mesh only.
        times: Pose sample times [s] used to size the box; ``None`` uses ``t=0``.
        mics: Optional microphone/listener positions ``[M, 3]`` [m].
        pad: Padding added around the swept bounding box [m].
        opacity: Mesh opacity in ``[0, 1]`` (``< 1`` renders translucent).
        color: Base RGB ``[3]`` in ``[0, 1]``; ``None`` uses the frontend default.
        face_scalar: Optional per-face scalar ``[F]`` shading the mesh by colour
            (see :func:`vertex_colors_from_face_scalar`).
        name: Mesh label.
        title: Scene title shown in the page header.

    Returns:
        A kwargs dict for :meth:`auraflow.viz.server.VizStreamer.init_scene`.
    """
    ts = np.asarray([0.0]) if times is None else np.asarray(times, dtype=float)
    lo, hi = _swept_bounds(mesh, motion, ts)
    pts = [lo[None, :], hi[None, :]]
    if mics is not None:
        pts.append(np.asarray(mics, dtype=float).reshape(-1, 3))
    allp = np.concatenate(pts, axis=0)
    box_min = allp.min(axis=0) - pad
    box_max = allp.max(axis=0) + pad
    box_min[2] = min(box_min[2], -pad)  # keep the ground plane in view

    mesh_dict: dict[str, Any] = {
        "name": name,
        "vertices": np.asarray(mesh.vertices, dtype=np.float32),
        "faces": np.asarray(mesh.faces, dtype=np.uint32),
        "opacity": float(opacity),
    }
    if color is not None:
        mesh_dict["color"] = [float(v) for v in np.asarray(color).ravel()[:3]]
    if face_scalar is not None:
        mesh_dict["colors"] = vertex_colors_from_face_scalar(mesh, face_scalar)
    return {
        "box_min": [float(v) for v in box_min],
        "box_max": [float(v) for v in box_max],
        "mics": None if mics is None else np.asarray(mics, dtype=np.float32),
        "meshes": [mesh_dict],
        "fields": [],
        "title": title,
    }


def stream_body(
    streamer: VizStreamer,
    mesh: TriMesh,
    motion: Motion,
    times: Any,
    *,
    mics: Any | None = None,
    mic_signals: Any | None = None,
    mic_t: Any | None = None,
    face_scalar: Any | None = None,
    opacity: float = 0.6,
    color: Any | None = None,
    fps: float = 30.0,
    realtime: bool = True,
    ring: int = 512,
    title: str = "AuraFlow body",
) -> None:
    """Replay a moving :class:`TriMesh` as a live animation over ``streamer``.

    Publishes the scene (rest mesh + optional scalar colouring), then one frame
    per animation tick carrying the mesh world pose (position + rotation) sampled
    from ``motion`` and, if given, the instantaneous per-mic pressure plus a
    rolling ring buffer for the frontend strip chart.

    Args:
        streamer: An entered :class:`~auraflow.viz.server.VizStreamer`.
        mesh: The body :class:`TriMesh` (rest geometry).
        motion: The :class:`~auraflow.body.motion.Motion` to replay.
        times: Source-time grid [s] spanning the animation, shape ``[T]``.
        mics: Microphone/listener positions ``[M, 3]`` [m], or ``None``.
        mic_signals: Per-mic pressure ``[M, n]`` [Pa] on grid ``mic_t``, or
            ``None``. Interpolated to each animation time.
        mic_t: Time grid for ``mic_signals`` ``[n]`` [s] (defaults to ``times``).
        face_scalar: Optional per-face scalar ``[F]`` colouring the mesh.
        opacity: Mesh opacity in ``[0, 1]``.
        color: Base RGB ``[3]`` in ``[0, 1]``.
        fps: Animation frames per second.
        realtime: If ``True``, sleep so playback runs at wall-clock speed.
        ring: Length of the per-mic pressure ring buffer sent each frame.
        title: Scene title.
    """
    ts = np.asarray(times, dtype=float)
    streamer.init_scene(
        **body_scene(
            mesh,
            motion,
            ts,
            mics=mics,
            opacity=opacity,
            color=color,
            face_scalar=face_scalar,
            title=title,
        )
    )

    sig = None if mic_signals is None else np.asarray(mic_signals, dtype=float)  # [M, n]
    st = (np.asarray(mic_t, dtype=float) if mic_t is not None else ts) if sig is not None else None

    t0, t1 = float(ts[0]), float(ts[-1])
    n_ticks = max(int(round((t1 - t0) * fps)), 1)
    tick_times = np.linspace(t0, t1, n_ticks)
    wall_start = time.monotonic()

    for step, tt in enumerate(tick_times):
        r, x = _pose_np(motion, float(tt))
        pose_row = np.concatenate([x.ravel(), r.ravel()])  # [12]

        mic_p = None
        ring_buf = None
        extra: dict[str, Any] = {}
        if sig is not None and st is not None:
            mic_p = np.array([np.interp(tt, st, sig[m]) for m in range(sig.shape[0])])
            lo = max(0, int(np.searchsorted(st, tt)) - ring)
            hi = int(np.searchsorted(st, tt)) + 1
            ring_buf = sig[:, lo:hi]
            extra["mic_ring_t0"] = float(st[lo])
            extra["mic_ring_dt"] = float(st[1] - st[0]) if st.shape[0] > 1 else 0.0

        streamer.push_frame(
            t=float(tt),
            step=step,
            mesh_poses=pose_row[None, :],
            mic_p=mic_p,
            mic_ring=ring_buf,
            extra=extra if extra else None,
        )
        if realtime:
            target = wall_start + (float(tt) - t0)
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
