r"""Optional dload (``dload-ml``) output management for JASA flyovers.

dload is the user's dataset library (PyPI ``dload-ml``, import ``dload``): one
S3-compatible bucket is the single source of truth, samples are ``(key,
{field: bytes})`` packed into content-addressed shards. This module turns
:func:`auraflow.datasets.jasa.generate_flyover` results into that sample stream
and (optionally) commits them.

**Lazy / optional by design.** Nothing here imports :mod:`dload` at module load
time -- ``import auraflow.datasets`` and dataset *generation* work without the
``data`` extra and without credentials. Only :func:`commit_flyovers` (and
:func:`open_repository`) touch :mod:`dload`, and only then the remote. Building
the sample iterator (:func:`flyover_samples`) is pure-Python byte packing and is
what the tests exercise (with a mocked ``Repository`` -- no network).

Sample layout (one sample per flyover)
--------------------------------------
- **key**: :func:`auraflow.datasets.jasa.scenario_id` (stable, physics-encoded).
- **fields**:

  - ``"wav"``: multichannel 16-bit PCM WAV (one channel per microphone), for
    listening; scaled by a per-sample peak stored in the metadata so it can be
    de-normalised back to Pa.
  - ``"meta"``: compact JSON (scenario + generation numerics + ``wav_peak_pa``
    and ``n_mics``), via ``dload.codecs.json_bytes`` when available (a local
    JSON fallback keeps this import-light).
  - ``"arrays"`` (optional, on by default): a ``.npz`` with the exact float32
    ``audio``/``tonal``/``broadband`` blocks, ``mics`` and ``band_centers`` --
    the lossless record the WAV can't carry.
"""

from __future__ import annotations

import io
import json
import os
import wave
from collections.abc import Iterable, Iterator
from glob import glob
from typing import Any

import numpy as np

from auraflow.datasets.jasa import JASAScenario, scenario_id

__all__ = [
    "commit_flyovers",
    "flyover_sample",
    "flyover_samples",
    "load_flyover_npz",
    "open_repository",
    "results_from_dir",
]


def _json_bytes(obj: Any) -> bytes:
    """Compact UTF-8 JSON bytes (``dload.codecs.json_bytes`` if dload present)."""
    try:
        from dload import codecs  # type: ignore[import-not-found]

        return codecs.json_bytes(obj)
    except Exception:
        return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def _pcm16_wav_bytes(audio: np.ndarray, fs: int) -> tuple[bytes, float]:
    """Multichannel 16-bit PCM WAV bytes + the peak [Pa] used to normalise.

    Args:
        audio: Pressures [Pa], shape ``[M, n]`` (channel-major, one row per mic).
        fs: Sample rate [Hz].

    Returns:
        ``(wav_bytes, peak_pa)`` where samples were scaled by ``1/peak_pa``
        before quantising to int16 (``peak_pa = max|audio|``, or ``1.0`` if the
        block is all-zero). Recover Pa as ``pcm/32767 * peak_pa``.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[None, :]
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    peak = peak if peak > 0.0 else 1.0
    pcm = np.clip(np.round(audio / peak * 32767.0), -32768, 32767).astype("<i2")
    interleaved = pcm.T.copy(order="C")  # [n, M] frame-major for WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(int(audio.shape[0]))
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(interleaved.tobytes())
    return buf.getvalue(), peak


def _npz_bytes(result: dict[str, Any]) -> bytes:
    """Lossless float32 ``.npz`` of the flyover arrays."""
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        audio=np.asarray(result["audio"], dtype=np.float32),
        tonal=np.asarray(result["tonal"], dtype=np.float32),
        broadband=np.asarray(result["broadband"], dtype=np.float32),
        mics=np.asarray(result["mics"], dtype=np.float32),
        band_centers=np.asarray(result["band_centers"], dtype=np.float32),
    )
    return buf.getvalue()


def flyover_sample(
    result: dict[str, Any], *, include_arrays: bool = True
) -> tuple[str, dict[str, bytes]]:
    """Turn one :func:`~auraflow.datasets.jasa.generate_flyover` result into a sample.

    Args:
        result: A generate_flyover result dict (needs ``audio``, ``mics``,
            ``band_centers``, ``meta``, ``scenario``; ``tonal``/``broadband`` for
            the lossless arrays).
        include_arrays: Include the lossless ``"arrays"`` ``.npz`` field.

    Returns:
        ``(key, fields)`` with ``key`` = :func:`scenario_id` and ``fields`` the
        ``wav``/``meta`` (+ optional ``arrays``) byte payloads.
    """
    fs = int(round(float(result["meta"]["fs"])))
    wav_bytes, peak = _pcm16_wav_bytes(np.asarray(result["audio"]), fs)
    meta = dict(result["meta"])
    meta["wav_peak_pa"] = peak
    meta["wav_channels"] = int(np.asarray(result["audio"]).shape[0])
    fields: dict[str, bytes] = {"wav": wav_bytes, "meta": _json_bytes(meta)}
    if include_arrays:
        fields["arrays"] = _npz_bytes(result)
    key = scenario_id(result["scenario"])
    return key, fields


def flyover_samples(
    results: Iterable[dict[str, Any]], *, include_arrays: bool = True
) -> Iterator[tuple[str, dict[str, bytes]]]:
    """Lazily map generate_flyover results to a dload sample stream.

    Args:
        results: Iterable of generate_flyover result dicts.
        include_arrays: Pass through to :func:`flyover_sample`.

    Yields:
        ``(key, {field: bytes})`` samples for :meth:`dload.Repository.commit`.
    """
    for result in results:
        yield flyover_sample(result, include_arrays=include_arrays)


def _scenario_from_meta(meta: dict[str, Any]) -> JASAScenario:
    """Reconstruct a :class:`~auraflow.datasets.jasa.JASAScenario` from saved meta."""
    return JASAScenario(
        speed=float(meta["speed"]),
        altitude=float(meta["altitude"]),
        heading_deg=float(meta.get("heading_deg", 0.0)),
        lateral_offset=float(meta.get("lateral_offset", 0.0)),
        duration=float(meta["duration"]),
        fs=float(meta["fs"]),
        seed=int(meta["seed"]),
        gust_w20=meta.get("gust_w20", 0.0),
        t_pass=meta.get("t_pass"),
    )


def load_flyover_npz(path: str) -> dict[str, Any]:
    """Load a :func:`~auraflow.datasets.jasa.save_flyover` ``.npz`` back to a result dict.

    Reconstructs the ``result`` shape :func:`flyover_sample` consumes (including
    a ``scenario`` rebuilt from the embedded metadata), so a full-scale run's
    saved outputs can be committed later without regenerating.

    Args:
        path: Path to a flyover ``.npz`` written by ``save_flyover``.

    Returns:
        A result dict with ``audio``/``tonal``/``broadband``/``mics``/
        ``band_centers``/``meta``/``scenario``.
    """
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        return {
            "audio": np.asarray(data["audio"]),
            "tonal": np.asarray(data["tonal"]),
            "broadband": np.asarray(data["broadband"]),
            "mics": np.asarray(data["mics"]),
            "band_centers": np.asarray(data["band_centers"]),
            "meta": meta,
            "scenario": _scenario_from_meta(meta),
        }


def results_from_dir(directory: str) -> Iterator[dict[str, Any]]:
    """Lazily yield flyover result dicts from every ``*.npz`` under a directory.

    Args:
        directory: Directory holding ``save_flyover`` ``.npz`` files.

    Yields:
        Result dicts (one per ``.npz``), in sorted filename order.
    """
    for path in sorted(glob(os.path.join(directory, "*.npz"))):
        yield load_flyover_npz(path)


def open_repository(config: Any = None) -> Any:
    """Open a :class:`dload.Repository` (lazy import; reads env / ``./dload.toml``).

    Args:
        config: Optional ``dload.Config``; ``None`` uses the standard resolution
            (env vars, then ``./dload.toml``, then user config).

    Returns:
        An open :class:`dload.Repository`.

    Raises:
        ImportError: if the ``data`` extra (``dload-ml``) is not installed.
    """
    try:
        import dload  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "auraflow.datasets.dload_io requires the 'data' extra (dload-ml). "
            "Install with `uv sync --extra data` or `pip install 'auraflow[data]'`."
        ) from exc
    return dload.Repository.open(config)


def commit_flyovers(
    name: str,
    results: Iterable[dict[str, Any]],
    *,
    repo: Any = None,
    meta: dict[str, Any] | None = None,
    recipe: str | None = None,
    include_arrays: bool = True,
) -> Any:
    """Commit generated flyovers to a dload dataset (touches the remote).

    Lazy: :mod:`dload` is imported only here. The sample stream is built by
    :func:`flyover_samples`, so committing is streaming (no full-dataset buffer).

    Args:
        name: Dataset name in the bucket (e.g. ``"jasa-flyovers"``).
        results: Iterable of generate_flyover result dicts.
        repo: An open :class:`dload.Repository`, or ``None`` to
            :func:`open_repository` one from the ambient config/credentials.
        meta: Dataset-level metadata dict (merged onto a small provenance stub).
        recipe: The generation script/command text to embed in the manifest.
        include_arrays: Include the lossless ``arrays`` field per sample.

    Returns:
        The ``dload.Manifest`` returned by :meth:`dload.Repository.commit`.
    """
    repo = open_repository() if repo is None else repo
    ds_meta = {"source": "auraflow.datasets.jasa", "vehicle": "nasa_1pax"}
    if meta:
        ds_meta.update(meta)
    samples = flyover_samples(results, include_arrays=include_arrays)
    return repo.commit(name, samples, meta=ds_meta, recipe=recipe)
