"""Wire-protocol round-trip and downsampler correctness (pure NumPy, no JAX)."""

from typing import Any

import numpy as np
import pytest

from auraflow.viz.stream import (
    PROTOCOL_VERSION,
    decode_header,
    decode_message,
    downsample_brick,
    downsample_slice,
    encode_frame,
    encode_message,
    encode_scene,
)


class TestEncodeDecodeRoundTrip:
    def test_header_and_arrays_round_trip(self):
        arrays: dict[str, Any] = {
            "a": np.arange(6, dtype=np.float32).reshape(2, 3),
            "b": np.array([1, 2, 3], dtype=np.int32),
        }
        header: dict[str, Any] = {"type": "frame", "t": 0.5, "step": 3}
        msg = encode_message(header, arrays)
        h, a = decode_message(msg)
        assert h["type"] == "frame"
        assert h["t"] == 0.5 and h["step"] == 3
        assert h["v"] == PROTOCOL_VERSION
        np.testing.assert_array_equal(a["a"], arrays["a"])
        np.testing.assert_array_equal(a["b"], arrays["b"])
        assert a["a"].dtype == np.float32 and a["b"].dtype == np.int32

    def test_float64_downcast_to_float32(self):
        arr = np.linspace(0, 1, 5, dtype=np.float64)
        h, a = decode_message(encode_message({"type": "frame"}, {"x": arr}))
        assert a["x"].dtype == np.float32
        np.testing.assert_allclose(a["x"], arr, atol=1e-6)

    def test_header_only_message(self):
        h, a = decode_message(encode_message({"type": "frame", "step": 1}))
        assert a == {} and h["step"] == 1

    def test_decode_header_is_cheap_and_consistent(self):
        msg = encode_message({"type": "scene", "title": "x"}, {"p": np.zeros(4, np.float32)})
        assert decode_header(msg)["type"] == "scene"

    def test_bad_version_rejected(self):
        msg = bytearray(encode_message({"type": "frame"}))
        # Corrupt the version field by re-encoding a header with wrong "v".
        import json
        import struct

        bad = json.dumps({"v": 999, "type": "frame", "arrays": []}).encode()
        msg = struct.pack(">I", len(bad)) + bad
        with pytest.raises(ValueError):
            decode_message(msg)


class TestSceneFrameBuilders:
    def test_scene_message(self):
        msg = encode_scene(
            box_min=[-1, -1, -1],
            box_max=[1, 1, 1],
            sphere_points=np.zeros((5, 3)),
            mics=np.ones((4, 3)),
            rotors=[{"hub": [0, 0, 0], "radius": 0.5, "n_blades": 2, "axis": [0, 0, 1]}],
            fields=["p"],
            title="demo",
        )
        h, a = decode_message(msg)
        assert h["type"] == "scene" and h["title"] == "demo"
        assert h["box_min"] == [-1.0, -1.0, -1.0]
        assert h["rotors"][0]["n_blades"] == 2
        assert a["sphere_points"].shape == (5, 3)
        assert a["mics"].shape == (4, 3)

    def test_frame_message(self):
        msg = encode_frame(
            t=0.25,
            step=7,
            field_slice=np.zeros((8, 8)),
            slice_range=(-1.0, 1.0),
            sphere_p=np.arange(5),
            vehicle_pos=[1, 2, 3],
            vehicle_R=np.eye(3).ravel(),
            rotor_azimuths=[0.1, 0.2],
        )
        h, a = decode_message(msg)
        assert h["type"] == "frame" and h["step"] == 7
        assert h["slice_range"] == [-1.0, 1.0]
        assert h["vehicle_pos"] == [1.0, 2.0, 3.0]
        assert len(h["vehicle_R"]) == 9
        assert h["rotor_azimuths"] == [0.1, 0.2]
        assert a["field_slice"].shape == (8, 8)
        assert a["sphere_p"].shape == (5,)


class TestDownsampling:
    def test_slice_shape_and_mean_preserved_on_linear_field(self):
        i, j = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
        field = (2.0 * i + 3.0 * j + 1.0).astype(np.float64)  # linear
        ds = downsample_slice(field, max_size=32)
        assert ds.shape == (32, 32)
        # factor 2 divides 64 exactly -> block mean preserves the global mean.
        assert ds.mean() == pytest.approx(field.mean(), rel=1e-6)

    def test_slice_noop_when_small(self):
        field = np.ones((10, 12), dtype=np.float32)
        ds = downsample_slice(field, max_size=64)
        assert ds.shape == (10, 12)

    def test_slice_respects_max_size_when_not_divisible(self):
        field = np.random.default_rng(0).standard_normal((100, 70))
        ds = downsample_slice(field, max_size=64)
        assert ds.shape[0] <= 64 and ds.shape[1] <= 64

    def test_brick_shape_and_mean_preserved(self):
        i, j, k = np.meshgrid(np.arange(32), np.arange(32), np.arange(32), indexing="ij")
        field = (i + 2 * j + 3 * k).astype(np.float64)
        ds = downsample_brick(field, max_size=16)
        assert ds.shape == (16, 16, 16)
        assert ds.mean() == pytest.approx(field.mean(), rel=1e-6)

    def test_brick_noop_when_small(self):
        field = np.ones((8, 8, 8), dtype=np.float32)
        assert downsample_brick(field, max_size=32).shape == (8, 8, 8)
