r"""dload (``dload-ml``) output management for drone onboard ego-noise.

The onboard counterpart of :mod:`auraflow.datasets.dload_io` (which serves the
JASA ground-array flyovers). Turns :func:`auraflow.datasets.drone_egonoise.
generate_egonoise` results into a dload sample stream and (optionally) commits
them to one dataset holding **both** drones, keyed per-drone.

Lazy / optional by design (mirrors :mod:`dload_io`): nothing here imports
:mod:`dload` at module load; only :func:`commit_egonoise` (and
:func:`open_repository`) touch the remote. The byte-packing helpers are reused
from :mod:`auraflow.datasets.dload_io`.

Sample layout (one sample per drone-hover case)
-----------------------------------------------
- **key**: :func:`auraflow.datasets.drone_egonoise.egonoise_id` (drone-prefixed,
  physics-encoded) -- so DREGON and Matrice samples never collide in one dataset.
- **fields**:

  - ``"wav"``: 64-channel 16-bit PCM WAV (one channel per onboard mic), scaled
    by a per-sample peak stored in the metadata (recover Pa as
    ``pcm/32767 * wav_peak_pa``).
  - ``"meta"``: compact JSON (scenario + generation numerics + drone provenance
    + ``wav_peak_pa``/``wav_channels``).
  - ``"arrays"`` (optional, on by default): a ``.npz`` with the float32
    ``audio``/``tonal``/``broadband`` blocks, world + body mic positions and
    ``band_centers`` -- the lossless record the WAV can't carry.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from auraflow.datasets.dload_io import (
    _json_bytes,
    _pcm16_wav_bytes,
    open_repository,
)

__all__ = [
    "commit_egonoise",
    "egonoise_sample",
    "egonoise_samples",
    "open_repository",
]


def _npz_bytes(result: dict[str, Any]) -> bytes:
    """Lossless float32 ``.npz`` of the ego-noise arrays (incl. body-frame mics)."""
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        audio=np.asarray(result["audio"], dtype=np.float32),
        tonal=np.asarray(result["tonal"], dtype=np.float32),
        broadband=np.asarray(result["broadband"], dtype=np.float32),
        mics=np.asarray(result["mics"], dtype=np.float32),
        mics_body=np.asarray(result["mics_body"], dtype=np.float32),
        band_centers=np.asarray(result["band_centers"], dtype=np.float32),
    )
    return buf.getvalue()


def egonoise_sample(
    result: dict[str, Any], *, include_arrays: bool = True
) -> tuple[str, dict[str, bytes]]:
    """Turn one :func:`generate_egonoise` result into a dload sample.

    Args:
        result: A generate_egonoise result dict (needs ``key``, ``audio``,
            ``mics``, ``mics_body``, ``band_centers``, ``meta``;
            ``tonal``/``broadband`` for the lossless arrays).
        include_arrays: Include the lossless ``"arrays"`` ``.npz`` field.

    Returns:
        ``(key, fields)`` with ``key`` = the drone-prefixed
        :func:`~auraflow.datasets.drone_egonoise.egonoise_id`.
    """
    fs = int(round(float(result["meta"]["fs"])))
    wav_bytes, peak = _pcm16_wav_bytes(np.asarray(result["audio"]), fs)
    meta = dict(result["meta"])
    meta["wav_peak_pa"] = peak
    meta["wav_channels"] = int(np.asarray(result["audio"]).shape[0])
    fields: dict[str, bytes] = {"wav": wav_bytes, "meta": _json_bytes(meta)}
    if include_arrays:
        fields["arrays"] = _npz_bytes(result)
    return str(result["key"]), fields


def egonoise_samples(
    results: Iterable[dict[str, Any]], *, include_arrays: bool = True
) -> Iterator[tuple[str, dict[str, bytes]]]:
    """Lazily map generate_egonoise results to a dload sample stream."""
    for result in results:
        yield egonoise_sample(result, include_arrays=include_arrays)


def commit_egonoise(
    name: str,
    results: Iterable[dict[str, Any]],
    *,
    repo: Any = None,
    meta: dict[str, Any] | None = None,
    recipe: str | None = None,
    include_arrays: bool = True,
) -> Any:
    """Commit generated drone ego-noise to a dload dataset (touches the remote).

    Streaming (no full-dataset buffer): the sample stream is built by
    :func:`egonoise_samples`. Both drones can share one dataset (keys are
    drone-prefixed).

    Args:
        name: Dataset name in the bucket (e.g. ``"drone-egonoise"``).
        results: Iterable of generate_egonoise result dicts.
        repo: An open :class:`dload.Repository`, or ``None`` to open one.
        meta: Dataset-level metadata (merged onto a provenance stub).
        recipe: The generation command text to embed in the manifest.
        include_arrays: Include the lossless ``arrays`` field per sample.

    Returns:
        The ``dload.Manifest`` returned by :meth:`dload.Repository.commit`.
    """
    repo = open_repository() if repo is None else repo
    ds_meta = {
        "source": "auraflow.datasets.drone_egonoise",
        "kind": "onboard-egonoise",
        "vehicles": ["dregon", "matrice100"],
    }
    if meta:
        ds_meta.update(meta)
    samples = egonoise_samples(results, include_arrays=include_arrays)
    return repo.commit(name, samples, meta=ds_meta, recipe=recipe)
