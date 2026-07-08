"""WebSocket streaming hub + :class:`VizStreamer` for live simulation viz.

Architecture
------------
A tiny broadcast *hub* fans out binary protocol frames (:mod:`auraflow.viz.stream`)
to any number of subscribed browser clients, and serves the self-contained
three.js frontend (``static/index.html`` + ``static/app.js``) over HTTP on the
**same port** (WebSocket upgrades and plain GETs are distinguished by the
``Upgrade`` header). The simulation is the *producer*:

- **Embedded** (default, used by the demo scripts): :class:`VizStreamer` starts
  the hub in a background asyncio thread inside the sim process and pushes frames
  straight into the fan-out -- no loopback socket, lowest overhead.
- **Remote**: :class:`VizStreamer` given a ``url`` connects to a standalone hub
  (``python -m auraflow.viz.server``) as a producer over ``ws://.../produce``;
  the hub rebroadcasts to its browser consumers.

The sim-facing API is non-blocking and best-effort: :meth:`VizStreamer.push_frame`
schedules a send on the hub's event loop and returns immediately; slow clients
get frames dropped from their bounded queues rather than stalling the sim. When
the streamer is disabled or no client is connected, ``push_frame`` returns almost
instantly (it does not even encode), so instrumented sim loops pay ~nothing when
nobody is watching.

``websockets`` is an **optional** dependency (extra ``viz-live``); it is imported
lazily so ``import auraflow`` and ``import auraflow.viz.stream`` work without it.
"""

from __future__ import annotations

import asyncio
import contextlib
import mimetypes
import threading
from importlib import resources
from typing import TYPE_CHECKING, Any

from auraflow.viz.stream import decode_header, encode_frame, encode_scene

if TYPE_CHECKING:  # pragma: no cover - typing only
    from websockets.asyncio.server import ServerConnection

__all__ = ["VizStreamer", "serve", "serve_forever"]

# Per-consumer send queue depth. Small: the frontend only ever wants the newest
# frames, so when a client falls behind we drop old frames rather than buffer.
_QUEUE_MAXSIZE = 4

_CONSUMER_PATHS = frozenset({"/", "/ws"})
_PRODUCER_PATH = "/produce"


def _require_websockets() -> Any:
    """Import the ``websockets`` asyncio server module or raise a helpful error."""
    try:
        import websockets.asyncio.server as ws_server  # noqa: F401

        return ws_server
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "auraflow.viz.server requires the 'viz-live' extra (websockets). "
            "Install with `uv sync --extra viz-live` or "
            "`pip install 'auraflow[viz-live]'`."
        ) from exc


def _static_bytes(name: str) -> bytes | None:
    """Read a file from the packaged ``auraflow.viz.static`` dir, or ``None``."""
    try:
        resource = resources.files("auraflow.viz").joinpath("static", name)
        if not resource.is_file():
            return None
        return resource.read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return None


class _Hub:
    """Fan-out of protocol frames to subscribed consumers, with a cached scene.

    Lives entirely on one asyncio event loop; all mutation happens on that loop
    (producers hop onto it via ``loop.call_soon_threadsafe``).
    """

    def __init__(self) -> None:
        self._consumers: set[asyncio.Queue[bytes]] = set()
        self._scene: bytes | None = None

    @property
    def consumer_count(self) -> int:
        """Number of currently connected consumers (best-effort, cross-thread)."""
        return len(self._consumers)

    def _put(self, queue: asyncio.Queue[bytes], msg: bytes) -> None:
        """Enqueue ``msg``, dropping the oldest frame if the queue is full."""
        try:
            queue.put_nowait(msg)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty, asyncio.QueueFull):
                queue.get_nowait()
                queue.put_nowait(msg)

    def broadcast(self, msg: bytes) -> None:
        """Send ``msg`` to every connected consumer (runs on the loop thread)."""
        for queue in self._consumers:
            self._put(queue, msg)

    def set_scene(self, msg: bytes) -> None:
        """Cache the scene (replayed to future consumers) and broadcast it now."""
        self._scene = msg
        self.broadcast(msg)

    def ingest(self, msg: bytes) -> None:
        """Handle a message from a producer: cache scenes, rebroadcast frames."""
        try:
            is_scene = decode_header(msg).get("type") == "scene"
        except (ValueError, KeyError, IndexError):
            is_scene = False
        if is_scene:
            self.set_scene(msg)
        else:
            self.broadcast(msg)

    async def serve_consumer(self, connection: ServerConnection) -> None:
        """Stream cached scene + subsequent frames to one browser consumer."""
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._consumers.add(queue)
        try:
            if self._scene is not None:
                await connection.send(self._scene)
            while True:
                await connection.send(await queue.get())
        finally:
            self._consumers.discard(queue)

    async def serve_producer(self, connection: ServerConnection) -> None:
        """Ingest frames pushed by a remote producer over ``/produce``."""
        async for msg in connection:
            self.ingest(msg if isinstance(msg, bytes) else msg.encode("utf-8"))


class VizStreamer:
    """Non-blocking live-visualization streamer for simulation loops.

    Use as a context manager around a sim loop::

        with VizStreamer(port=8000) as viz:
            viz.init_scene(box_min=..., box_max=..., sphere_points=...)
            for step in range(n):
                ...  # advance the simulation
                viz.push_frame(t=t, step=step, field_slice=sl, sphere_p=p)

    In the default *embedded* mode ``__enter__`` starts the hub (HTTP + WebSocket)
    in a background thread and blocks until it is bound and ready; ``__exit__``
    stops it cleanly. Pass ``enabled=False`` for a zero-overhead no-op streamer
    (every method returns immediately -- handy for a ``--viz/--no-viz`` flag).
    Pass ``url="ws://host:port"`` to instead push to a standalone hub.

    Args:
        host: Interface to bind (embedded mode). Default loopback only.
        port: TCP port for HTTP + WebSocket (embedded mode).
        enabled: If ``False``, all methods are no-ops (nothing is started).
        url: If given, connect to this hub as a remote producer instead of
            embedding one (``ws://host:port``; ``/produce`` is appended).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        enabled: bool = True,
        url: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.enabled = enabled
        self.url = url
        self._remote = url is not None
        self._hub = _Hub()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._producer: Any = None  # remote-mode producer ServerConnection/ClientConnection
        self._started = False

    # -- lifecycle --------------------------------------------------------- #
    def __enter__(self) -> VizStreamer:
        if self.enabled:
            self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def active(self) -> bool:
        """Whether a :meth:`push_frame` would actually send anything.

        ``True`` only when enabled, started, and (embedded mode) at least one
        browser is connected. Sim loops can gate expensive per-frame work
        (``device_get``, downsampling) on this to pay nothing when unwatched.
        """
        if not self.enabled or not self._started:
            return False
        return self._remote or self._hub.consumer_count > 0

    @property
    def http_url(self) -> str:
        """The ``http://host:port`` URL a browser opens (embedded mode)."""
        host = "localhost" if self.host in ("0.0.0.0", "127.0.0.1", "") else self.host
        return f"http://{host}:{self.port}"

    def start(self) -> None:
        """Start the background thread (hub server, or remote producer link)."""
        if self._started:
            return
        _require_websockets()
        self._thread = threading.Thread(target=self._run, name="auraflow-viz", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            raise RuntimeError("viz server did not become ready within 10 s")
        self._started = True

    def close(self) -> None:
        """Stop the background thread and release the port."""
        if not self._started:
            return
        loop, stop = self._loop, self._stop
        if loop is not None and stop is not None:
            loop.call_soon_threadsafe(stop.set)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._started = False

    # -- producer API ------------------------------------------------------ #
    def init_scene(self, **kwargs: Any) -> None:
        """Publish the static scene (see :func:`auraflow.viz.stream.encode_scene`).

        Cached by the hub and replayed to every client on connect, so it is safe
        to call once before the loop regardless of when browsers attach.
        """
        if not self.enabled or not self._started:
            return
        self._send(encode_scene(**kwargs))

    def push_frame(self, **kwargs: Any) -> None:
        """Publish one frame (see :func:`auraflow.viz.stream.encode_frame`).

        Non-blocking and best-effort. Returns immediately -- without encoding --
        when disabled, not started, or (embedded mode) no browser is connected.
        """
        if not self.enabled or not self._started:
            return
        if not self._remote and self._hub.consumer_count == 0:
            return
        self._send(encode_frame(**kwargs))

    def _send(self, msg: bytes) -> None:
        """Schedule ``msg`` onto the hub loop (thread-safe, fire-and-forget)."""
        loop = self._loop
        if loop is None:
            return
        if self._remote:
            loop.call_soon_threadsafe(self._send_remote, msg)
        elif decode_header(msg).get("type") == "scene":
            loop.call_soon_threadsafe(self._hub.set_scene, msg)
        else:
            loop.call_soon_threadsafe(self._hub.broadcast, msg)

    def _send_remote(self, msg: bytes) -> None:
        """Send ``msg`` over the remote producer link (on the loop thread)."""
        if self._producer is not None:
            asyncio.ensure_future(self._producer.send(msg))

    # -- background thread ------------------------------------------------- #
    def _run(self) -> None:
        try:
            asyncio.run(self._amain())
        except Exception:  # pragma: no cover - surfaced via _ready timeout
            self._ready.set()
            raise

    async def _amain(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        if self._remote:
            await self._run_remote()
        else:
            await self._run_embedded()

    async def _run_embedded(self) -> None:
        assert self._stop is not None
        async with _serve_hub(self._hub, self.host, self.port):
            self._ready.set()
            await self._stop.wait()

    async def _run_remote(self) -> None:
        assert self._stop is not None and self.url is not None
        from websockets.asyncio.client import connect

        url = self.url.rstrip("/") + _PRODUCER_PATH
        async with connect(url) as producer:
            self._producer = producer
            self._ready.set()
            await self._stop.wait()


def _http_response(path: str) -> Any:
    """Build an HTTP :class:`~websockets.http11.Response` for a static GET."""
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    name = "index.html" if path in ("/", "") else path.lstrip("/")
    # Only serve flat basenames from the packaged static dir (no path traversal).
    body = None if ("/" in name or "\\" in name or name.startswith(".")) else _static_bytes(name)
    if body is None:
        msg = b"404 Not Found"
        headers = Headers()
        headers["Content-Type"] = "text/plain"
        headers["Content-Length"] = str(len(msg))
        return Response(404, "Not Found", headers, msg)
    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
    headers = Headers()
    headers["Content-Type"] = ctype
    headers["Content-Length"] = str(len(body))
    return Response(200, "OK", headers, body)


def _serve_hub(hub: _Hub, host: str, port: int) -> Any:
    """Return the ``websockets`` async server context manager bound to ``hub``."""
    ws_server = _require_websockets()

    async def process_request(connection: Any, request: Any) -> Any:
        upgrade = request.headers.get("Upgrade", "")
        if upgrade and upgrade.lower() == "websocket":
            return None  # let the WebSocket handshake proceed
        return _http_response(request.path)

    async def handler(connection: ServerConnection) -> None:
        path = connection.request.path if connection.request is not None else "/"
        # Ignore query strings when routing.
        path = path.split("?", 1)[0]
        if path == _PRODUCER_PATH:
            await hub.serve_producer(connection)
        elif path in _CONSUMER_PATHS:
            await hub.serve_consumer(connection)

    return ws_server.serve(handler, host, port, process_request=process_request)


async def serve_forever(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run a standalone hub until cancelled (async entry point).

    Serves the frontend over HTTP and accepts both browser consumers (``/``) and
    remote producers (``/produce``). A simulation elsewhere pushes frames with
    ``VizStreamer(url="ws://host:port")``.
    """
    hub = _Hub()
    async with _serve_hub(hub, host, port):
        await asyncio.Event().wait()  # run until the process is interrupted


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Blocking standalone-hub entry point (see :func:`serve_forever`)."""
    print(f"auraflow viz hub serving on http://{host}:{port}  (Ctrl-C to stop)")
    try:
        asyncio.run(serve_forever(host, port))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass


def _main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m auraflow.viz.server [--host H] [--port P]``."""
    import argparse

    parser = argparse.ArgumentParser(description="AuraFlow live-visualization hub.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8000, help="TCP port")
    args = parser.parse_args(argv)
    serve(args.host, args.port)


if __name__ == "__main__":  # pragma: no cover - CLI
    _main()
