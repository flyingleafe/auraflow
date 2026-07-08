"""Wire protocol and downsampling for the live-visualization stream.

Pure NumPy / stdlib -- no JAX, no async, no third-party deps -- so it imports
with the base install and is trivially unit-testable. The websocket transport
(:mod:`auraflow.viz.server`) and the browser frontend
(:mod:`auraflow.viz` ``static/``) both speak this protocol.

Message framing
---------------
Every message is a single binary WebSocket frame::

    +--------------------+-------------------------+----------------------+
    | 4 bytes            | ``H`` bytes             | payload bytes        |
    | uint32 big-endian  | UTF-8 JSON header       | raw array data       |
    | header length ``H``|                         | (arrays concatenated)|
    +--------------------+-------------------------+----------------------+

The JSON header always carries ``"v"`` (protocol version, currently
:data:`PROTOCOL_VERSION`) and ``"type"`` (``"scene"`` or ``"frame"``), plus an
``"arrays"`` list describing each binary array (``name``, ``dtype``, ``shape``,
byte ``offset``, ``nbytes``). Scalars, poses, and metadata live directly in the
JSON header; only bulk numeric arrays go in the binary payload. Arrays are cast
to compact dtypes (``float32`` for fields, positions, and pressures) on the
simulation side so the payload is half the size of native float64.

Downsampling
------------
:func:`downsample_slice` / :func:`downsample_brick` block-mean-reduce a 2-D
field slice / 3-D brick to at most ``max_size`` samples per axis *before*
encoding, so the sim only ships a handful of kilobytes per frame regardless of
the CFD grid resolution. Block averaging preserves the field mean exactly when
the reduction factor divides the axis length (see the functions' notes).

Units follow ``docs/architecture.md`` (SI: m, s, Pa, kg/m^3).
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = [
    "PROTOCOL_VERSION",
    "decode_header",
    "decode_message",
    "downsample_brick",
    "downsample_slice",
    "encode_frame",
    "encode_message",
    "encode_scene",
]

# Bump when the header schema changes incompatibly; the frontend checks it.
PROTOCOL_VERSION = 1

# dtypes allowed in the binary payload (compact, JSON-nameable, cross-language).
_ALLOWED_DTYPES = ("float32", "int32", "uint8")


def _coerce_array(a: Any) -> np.ndarray:
    """Cast an array-like to a C-contiguous, payload-legal NumPy array.

    float64 -> float32 (fields/positions/pressures ship single precision);
    integer kinds -> int32; everything else must already be an allowed dtype.
    """
    arr = np.ascontiguousarray(a)
    if arr.dtype == np.float64 or arr.dtype == np.float16:
        arr = arr.astype(np.float32)
    elif np.issubdtype(arr.dtype, np.integer) and arr.dtype.name != "int32":
        arr = arr.astype(np.int32)
    if arr.dtype.name not in _ALLOWED_DTYPES:
        raise ValueError(f"array dtype {arr.dtype} not supported; use one of {_ALLOWED_DTYPES}")
    return np.ascontiguousarray(arr)


def encode_message(header: Mapping[str, Any], arrays: Mapping[str, Any] | None = None) -> bytes:
    """Encode a header dict + named arrays into one binary protocol frame.

    Args:
        header: JSON-serializable metadata. ``"v"`` is injected/overwritten with
            :data:`PROTOCOL_VERSION`; an ``"arrays"`` key is added describing the
            payload (any pre-existing ``"arrays"`` is replaced).
        arrays: Mapping of name -> array-like. Each is coerced to a compact
            dtype (:func:`_coerce_array`) and appended to the binary payload in
            iteration order. ``None`` or empty means a header-only message.

    Returns:
        The framed message bytes (4-byte length prefix + JSON header + payload).
    """
    arrays = arrays or {}
    specs: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    offset = 0
    for name, value in arrays.items():
        arr = _coerce_array(value)
        raw = arr.tobytes()
        specs.append(
            {
                "name": name,
                "dtype": arr.dtype.name,
                "shape": list(arr.shape),
                "offset": offset,
                "nbytes": len(raw),
            }
        )
        chunks.append(raw)
        offset += len(raw)

    full_header = dict(header)
    full_header["v"] = PROTOCOL_VERSION
    full_header["arrays"] = specs
    header_bytes = json.dumps(full_header, separators=(",", ":")).encode("utf-8")
    payload = b"".join(chunks)
    return struct.pack(">I", len(header_bytes)) + header_bytes + payload


def decode_header(data: bytes) -> dict[str, Any]:
    """Parse only the JSON header of a protocol frame (cheap; skips the payload).

    Args:
        data: A frame produced by :func:`encode_message`.

    Returns:
        The decoded header dict (includes ``"v"``, ``"type"``, ``"arrays"``).
    """
    (header_len,) = struct.unpack(">I", data[:4])
    return json.loads(data[4 : 4 + header_len].decode("utf-8"))


def decode_message(data: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Decode a protocol frame back into ``(header, arrays)``.

    Inverse of :func:`encode_message`. Arrays are returned as NumPy views/copies
    with their original (post-coercion) dtype and shape.

    Args:
        data: A frame produced by :func:`encode_message`.

    Returns:
        ``(header, arrays)`` where ``arrays`` maps name -> :class:`numpy.ndarray`.

    Raises:
        ValueError: if the protocol version is unknown.
    """
    (header_len,) = struct.unpack(">I", data[:4])
    header = json.loads(data[4 : 4 + header_len].decode("utf-8"))
    if header.get("v") != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version {header.get('v')!r}")
    payload = memoryview(data)[4 + header_len :]
    arrays: dict[str, np.ndarray] = {}
    for spec in header.get("arrays", []):
        off = int(spec["offset"])
        nbytes = int(spec["nbytes"])
        arr = np.frombuffer(payload[off : off + nbytes], dtype=np.dtype(spec["dtype"]))
        arrays[spec["name"]] = arr.reshape(tuple(spec["shape"]))
    return header, arrays


# --------------------------------------------------------------------------- #
# Downsampling (block-mean reduction)
# --------------------------------------------------------------------------- #
def _pool_factor(n: int, max_size: int) -> int:
    """Smallest integer factor bringing ``n`` down to ``<= max_size`` samples."""
    if max_size < 1:
        raise ValueError("max_size must be >= 1")
    return max(1, -(-n // max_size))  # ceil(n / max_size)


def _block_mean(a: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    """Non-overlapping block mean-pool of ``a`` by ``factors`` per axis.

    Each axis is trimmed to a whole multiple of its factor (trailing remainder
    dropped), reshaped into ``(n_blocks, factor)`` pairs, and averaged over the
    ``factor`` sub-axes. When every factor divides its axis length exactly the
    global mean is preserved; otherwise the dropped edge cells perturb it
    slightly (documented; the frame is a preview, not a quantitative field).
    """
    trimmed = a[tuple(slice(0, (s // f) * f) for s, f in zip(a.shape, factors, strict=True))]
    new_shape: list[int] = []
    for s, f in zip(trimmed.shape, factors, strict=True):
        new_shape += [s // f, f]
    reshaped = trimmed.reshape(new_shape)
    mean_axes = tuple(range(1, reshaped.ndim, 2))
    return reshaped.mean(axis=mean_axes)


def downsample_slice(field: Any, max_size: int = 64) -> np.ndarray:
    """Block-mean a 2-D field slice to ``<= max_size`` samples per axis.

    Args:
        field: 2-D array-like ``[H, W]`` (a scalar-field slice, e.g. p' or rho).
        max_size: Maximum samples per axis after reduction.

    Returns:
        Reduced float32 array ``[<=max_size, <=max_size]``. The mean is preserved
        exactly when ``H`` and ``W`` are each divisible by their pool factor.
    """
    arr = np.asarray(field, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"downsample_slice expects a 2-D field, got shape {arr.shape}")
    factors = (_pool_factor(arr.shape[0], max_size), _pool_factor(arr.shape[1], max_size))
    if factors == (1, 1):
        return arr
    return _block_mean(arr, factors).astype(np.float32)


def downsample_brick(field: Any, max_size: int = 32) -> np.ndarray:
    """Block-mean a 3-D field brick to ``<= max_size`` samples per axis.

    Args:
        field: 3-D array-like ``[Nx, Ny, Nz]``.
        max_size: Maximum samples per axis after reduction.

    Returns:
        Reduced float32 array with each axis ``<= max_size``. The mean is
        preserved exactly when each axis is divisible by its pool factor.
    """
    arr = np.asarray(field, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"downsample_brick expects a 3-D field, got shape {arr.shape}")
    factors = tuple(_pool_factor(n, max_size) for n in arr.shape)
    if factors == (1, 1, 1):
        return arr
    return _block_mean(arr, factors).astype(np.float32)


# --------------------------------------------------------------------------- #
# Message builders
# --------------------------------------------------------------------------- #
def encode_scene(
    *,
    box_min: Any,
    box_max: Any,
    sphere_points: Any | None = None,
    mics: Any | None = None,
    rotors: list[dict[str, Any]] | None = None,
    slice_plane: dict[str, Any] | None = None,
    fields: list[str] | None = None,
    dt: float | None = None,
    title: str = "",
    extra: Mapping[str, Any] | None = None,
) -> bytes:
    """Build a ``scene`` init message (static geometry + stream metadata).

    Sent once per client on connect; describes everything the frontend needs to
    lay out the 3-D scene before frames arrive.

    Args:
        box_min: Domain lower corner ``[3]`` [m].
        box_max: Domain upper corner ``[3]`` [m].
        sphere_points: Permeable-surface points ``[S, 3]`` [m], or ``None``.
        mics: Microphone/observer positions ``[M, 3]`` [m], or ``None``.
        rotors: Per-rotor layout dicts, each ``{"hub": [3], "radius": r,
            "n_blades": B, "axis": [3], "arm": [3] | None}`` (world frame, m).
        slice_plane: Field-slice geometry for the textured plane, e.g.
            ``{"axis": "y", "coord": 0.0, "u_range": [lo, hi], "v_range": [lo,
            hi], "shape": [H, W]}``; ``None`` if no slice is streamed.
        fields: Names of the scalar fields carried per frame (e.g. ``["p"]``).
        dt: Nominal simulation time between pushed frames [s], if known.
        title: Human-readable scene title (shown in the page header).
        extra: Any additional JSON-serializable metadata to merge into the header.

    Returns:
        Encoded scene frame bytes.
    """
    header: dict[str, Any] = {
        "type": "scene",
        "box_min": [float(v) for v in np.asarray(box_min).ravel()],
        "box_max": [float(v) for v in np.asarray(box_max).ravel()],
        "rotors": rotors or [],
        "fields": fields or [],
        "title": title,
    }
    if slice_plane is not None:
        header["slice_plane"] = slice_plane
    if dt is not None:
        header["dt"] = float(dt)
    if extra:
        header.update(dict(extra))

    arrays: dict[str, Any] = {}
    if sphere_points is not None:
        arrays["sphere_points"] = np.asarray(sphere_points, dtype=np.float32)
    if mics is not None:
        arrays["mics"] = np.asarray(mics, dtype=np.float32)
    return encode_message(header, arrays)


def encode_frame(
    *,
    t: float,
    step: int,
    field_slice: Any | None = None,
    brick: Any | None = None,
    slice_range: tuple[float, float] | None = None,
    sphere_p: Any | None = None,
    mic_p: Any | None = None,
    mic_ring: Any | None = None,
    vehicle_pos: Any | None = None,
    vehicle_R: Any | None = None,
    rotor_azimuths: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> bytes:
    """Build a per-step ``frame`` message.

    All array inputs are optional; only what a given backend has is sent.
    Fields should already be downsampled (:func:`downsample_slice` /
    :func:`downsample_brick`) on the simulation side.

    Args:
        t: Physical simulation time of this frame [s].
        step: Integer step/sample index.
        field_slice: 2-D scalar-field slice ``[H, W]`` (already downsampled).
        brick: 3-D scalar-field brick ``[nx, ny, nz]`` (already downsampled).
        slice_range: ``(lo, hi)`` colour-scale range for the slice/brick; if
            ``None`` the frontend autoscales.
        sphere_p: Acoustic pressure p' at the permeable-surface points ``[S]`` [Pa].
        mic_p: Instantaneous pressure at each mic ``[M]`` [Pa].
        mic_ring: Recent per-mic pressure history ``[M, L]`` [Pa] for the
            frontend strip chart (a rolling window ending at ``t``).
        vehicle_pos: Vehicle position, world frame ``[3]`` [m].
        vehicle_R: Vehicle attitude (world<-body) flattened ``[9]`` (row-major).
        rotor_azimuths: Reference-blade azimuth per rotor ``[Nr]`` [rad].
        extra: Additional JSON-serializable header metadata.

    Returns:
        Encoded frame bytes.
    """
    header: dict[str, Any] = {"type": "frame", "t": float(t), "step": int(step)}
    if slice_range is not None:
        header["slice_range"] = [float(slice_range[0]), float(slice_range[1])]
    if vehicle_pos is not None:
        header["vehicle_pos"] = [float(v) for v in np.asarray(vehicle_pos).ravel()]
    if vehicle_R is not None:
        header["vehicle_R"] = [float(v) for v in np.asarray(vehicle_R).ravel()]
    if rotor_azimuths is not None:
        header["rotor_azimuths"] = [float(v) for v in np.asarray(rotor_azimuths).ravel()]
    if extra:
        header.update(dict(extra))

    arrays: dict[str, Any] = {}
    if field_slice is not None:
        arrays["field_slice"] = np.asarray(field_slice, dtype=np.float32)
    if brick is not None:
        arrays["brick"] = np.asarray(brick, dtype=np.float32)
    if sphere_p is not None:
        arrays["sphere_p"] = np.asarray(sphere_p, dtype=np.float32)
    if mic_p is not None:
        arrays["mic_p"] = np.asarray(mic_p, dtype=np.float32)
    if mic_ring is not None:
        arrays["mic_ring"] = np.asarray(mic_ring, dtype=np.float32)
    return encode_message(header, arrays)
