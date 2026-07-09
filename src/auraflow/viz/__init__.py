"""Live in-browser 3-D visualization of running AuraFlow simulations.

Two halves:

- a **WebSocket streaming hub** (:mod:`auraflow.viz.server`) that a simulation
  loop feeds via the non-blocking :class:`~auraflow.viz.server.VizStreamer`
  context manager, serving a self-contained **three.js frontend** (the packaged
  ``static/index.html`` + ``static/app.js``) over HTTP on the same port;
- a compact **binary wire protocol** (:mod:`auraflow.viz.stream`): a versioned
  JSON header + concatenated float32 array payload, with sim-side downsamplers
  for field slices/bricks.

Backend adapters build the scene/frame payloads: :mod:`auraflow.viz.cfd` for the
step-by-step CFD driver (:func:`auraflow.cfd.run.run_acoustic_case` takes a
``viz=`` argument), :mod:`auraflow.viz.flyover` for replaying a CONA
:class:`~auraflow.cona.flight.FlightHistory`, and :mod:`auraflow.viz.body` for
streaming a general :class:`~auraflow.body.mesh.TriMesh` + motion replay
(protocol v2 mesh + per-mesh pose channel).

The wire protocol (:mod:`auraflow.viz.stream`) is pure NumPy/stdlib and imports
with the base install. The server needs the ``viz-live`` extra (``websockets``),
imported lazily -- ``import auraflow`` and ``import auraflow.viz.stream`` never
require it.
"""

from typing import TYPE_CHECKING

from auraflow.viz.stream import (
    PROTOCOL_VERSION,
    decode_message,
    downsample_brick,
    downsample_slice,
    encode_frame,
    encode_message,
    encode_scene,
)

if TYPE_CHECKING:  # pragma: no cover - typing only (runtime access via __getattr__)
    from auraflow.viz.server import VizStreamer, serve

__all__ = [
    "PROTOCOL_VERSION",
    "VizStreamer",
    "decode_message",
    "downsample_brick",
    "downsample_slice",
    "encode_frame",
    "encode_message",
    "encode_scene",
    "serve",
]


def __getattr__(name: str) -> object:
    """Lazily expose the websockets-dependent server API without importing it."""
    if name in ("VizStreamer", "serve"):
        from auraflow.viz import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
