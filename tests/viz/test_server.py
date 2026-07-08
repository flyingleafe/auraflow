"""Server smoke test (localhost, no JAX/GPU) + streamer no-op path.

Starts an embedded :class:`VizStreamer`, connects a Python WebSocket client as a
consumer, pushes a scene + 2 frames, and asserts the client receives and decodes
them. Uses ``asyncio.run`` inside sync tests (no pytest-asyncio dependency).
"""

import asyncio
import socket
import time
from typing import Any, cast

import numpy as np
import pytest

from auraflow.viz.server import VizStreamer
from auraflow.viz.stream import decode_message

pytest.importorskip("websockets")

from websockets.asyncio.client import connect  # noqa: E402


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _wait_active(viz: VizStreamer, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not viz.active:
        time.sleep(0.02)


class TestServerSmoke:
    def test_scene_and_frames_delivered(self):
        port = _free_port()

        async def run() -> list[Any]:
            with VizStreamer(host="127.0.0.1", port=port) as viz:
                async with connect(f"ws://127.0.0.1:{port}/ws") as client:
                    # Wait until the consumer is registered so pushes aren't dropped.
                    await asyncio.get_running_loop().run_in_executor(None, _wait_active, viz)
                    assert viz.active
                    viz.init_scene(
                        box_min=[0, 0, 0],
                        box_max=[1, 1, 1],
                        sphere_points=np.zeros((3, 3), dtype=np.float32),
                    )
                    viz.push_frame(t=0.0, step=0, sphere_p=np.ones(3))
                    viz.push_frame(t=0.1, step=1, sphere_p=2 * np.ones(3))
                    return [await asyncio.wait_for(client.recv(), timeout=3.0) for _ in range(3)]

        msgs = asyncio.run(run())
        headers = [decode_message(m)[0] for m in msgs]
        assert headers[0]["type"] == "scene"
        assert headers[1]["type"] == "frame" and headers[1]["step"] == 0
        assert headers[2]["type"] == "frame" and headers[2]["step"] == 1
        _, arr2 = decode_message(msgs[2])
        np.testing.assert_allclose(arr2["sphere_p"], 2 * np.ones(3), atol=1e-6)

    def test_late_consumer_gets_cached_scene(self):
        port = _free_port()

        async def run() -> dict[str, Any]:
            with VizStreamer(host="127.0.0.1", port=port) as viz:
                # Publish the scene before any client connects; it must be cached.
                viz.init_scene(box_min=[0, 0, 0], box_max=[2, 2, 2], title="cached")
                async with connect(f"ws://127.0.0.1:{port}/ws") as client:
                    msg = await asyncio.wait_for(client.recv(), timeout=3.0)
                return decode_message(cast(bytes, msg))[0]

        header = asyncio.run(run())
        assert header["type"] == "scene" and header["title"] == "cached"


class TestNoOpPath:
    def test_disabled_streamer_is_instant_noop(self):
        viz = VizStreamer(enabled=False)
        # Never entered/started: every call returns immediately, nothing raised.
        assert viz.active is False
        t0 = time.perf_counter()
        for i in range(1000):
            viz.push_frame(t=float(i), step=i, sphere_p=np.ones(64))
        viz.init_scene(box_min=[0, 0, 0], box_max=[1, 1, 1])
        assert time.perf_counter() - t0 < 0.5  # ~no work per call

    def test_disabled_streamer_context_manager(self):
        with VizStreamer(enabled=False) as viz:
            assert viz.active is False
            viz.push_frame(t=0.0, step=0)  # no-op, no server started

    def test_enabled_but_no_consumer_drops_frames(self):
        port = _free_port()
        with VizStreamer(host="127.0.0.1", port=port) as viz:
            # Started, but nobody is connected -> active False, push is a no-op.
            assert viz.active is False
            viz.push_frame(t=0.0, step=0, sphere_p=np.ones(3))
