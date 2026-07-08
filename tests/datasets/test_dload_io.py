"""dload sample-iterator packing for JASA flyovers (mocked Repository; no network).

Exercises the pure-Python byte packing (:func:`flyover_sample` /
:func:`flyover_samples`) and the commit path with a fake ``Repository`` that
records what it was handed -- nothing here imports :mod:`dload` or touches R2.
"""

import io
import json
import wave

import numpy as np

from auraflow.datasets.dload_io import (
    commit_flyovers,
    flyover_sample,
    flyover_samples,
)
from auraflow.datasets.jasa import JASAScenario, scenario_id


def _result(n_mics: int = 2, n: int = 200, fs: float = 2000.0) -> dict:
    rng = np.random.default_rng(0)
    scenario = JASAScenario(
        speed=8.0, altitude=30.0, seed=3, fs=fs, duration=n / fs, mics=np.zeros((n_mics, 3))
    )
    audio = rng.standard_normal((n_mics, n)) * 0.5
    return {
        "audio": audio,
        "tonal": audio * 0.6,
        "broadband": audio * 0.4,
        "mics": np.zeros((n_mics, 3)),
        "band_centers": np.array([100.0, 125.0, 160.0]),
        "meta": scenario.to_meta() | {"fs": fs},
        "scenario": scenario,
    }


def test_flyover_sample_key_and_fields():
    res = _result(n_mics=3)
    key, fields = flyover_sample(res)
    assert key == scenario_id(res["scenario"])
    assert set(fields) == {"wav", "meta", "arrays"}
    assert all(isinstance(v, bytes) for v in fields.values())

    # WAV parses: one channel per mic, right sample rate.
    with wave.open(io.BytesIO(fields["wav"]), "rb") as w:
        assert w.getnchannels() == 3
        assert w.getframerate() == int(round(res["meta"]["fs"]))
        assert w.getsampwidth() == 2

    # Meta JSON round-trips and carries the WAV de-normalisation peak.
    meta = json.loads(fields["meta"].decode("utf-8"))
    assert meta["wav_channels"] == 3
    assert meta["wav_peak_pa"] > 0.0

    # Lossless arrays npz has the exact blocks.
    with np.load(io.BytesIO(fields["arrays"])) as npz:
        assert npz["audio"].shape == (3, 200)
        assert set(npz.files) >= {"audio", "tonal", "broadband", "mics", "band_centers"}


def test_flyover_sample_wav_recovers_pressure_scale():
    res = _result(n_mics=1)
    _, fields = flyover_sample(res)
    meta = json.loads(fields["meta"].decode("utf-8"))
    peak = meta["wav_peak_pa"]
    with wave.open(io.BytesIO(fields["wav"]), "rb") as w:
        frames = w.readframes(w.getnframes())
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float64)
    recovered = pcm / 32767.0 * peak
    # int16 quantisation of a peak-normalised block: within a couple LSB in Pa.
    assert np.max(np.abs(recovered - res["audio"][0])) < 3.0 * peak / 32767.0


def test_include_arrays_false_omits_arrays():
    _, fields = flyover_sample(_result(), include_arrays=False)
    assert set(fields) == {"wav", "meta"}


def test_flyover_samples_is_lazy_iterator():
    results = [_result(), _result()]
    stream = flyover_samples(results)
    first = next(stream)
    assert isinstance(first, tuple) and isinstance(first[0], str)
    assert isinstance(first[1], dict)


class _FakeRepo:
    """Stand-in for dload.Repository that records the commit without any network."""

    def __init__(self):
        self.calls = []

    def commit(self, name, samples, *, meta=None, recipe=None):
        # Materialise the streamed samples so we can assert on them.
        collected = [(k, dict(f)) for k, f in samples]
        self.calls.append({"name": name, "samples": collected, "meta": meta, "recipe": recipe})
        return {"name": name, "num_samples": len(collected)}


def test_commit_flyovers_streams_correct_samples_to_repo():
    repo = _FakeRepo()
    results = [_result(), _result(n_mics=1)]
    manifest = commit_flyovers(
        "jasa-flyovers",
        results,
        repo=repo,
        meta={"run": "unit"},
        recipe="pytest",
    )
    assert manifest["num_samples"] == 2
    (call,) = repo.calls
    assert call["name"] == "jasa-flyovers"
    assert call["recipe"] == "pytest"
    # Dataset-level provenance stub is merged with the caller's meta.
    assert call["meta"]["source"] == "auraflow.datasets.jasa"
    assert call["meta"]["run"] == "unit"
    # Every sample has the packed fields; keys are scenario ids.
    for key, fields in call["samples"]:
        assert key.startswith("V")
        assert set(fields) == {"wav", "meta", "arrays"}
