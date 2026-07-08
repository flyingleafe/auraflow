r"""JASA flyover scenario -> per-microphone auralized noise (CONA path).

Wires the CONA backend into the JASA paper's data-generation recipe
(``docs/research/jasa-datagen-reference.md``): a NASA 1-Pax quadrotor flies a
level straight line over a ground microphone array; each mic records a
44.1 kHz pressure time series that is the sum of the time-domain convective
FW-H **tonal** noise and the Griffin--Lim auralization of the per-rotor **BPM
broadband** 1/3-octave spectrograms.

Pipeline (digest steps 1-6)
---------------------------
1. Trajectory: :func:`auraflow.cona.flight.straight_flyover` (level edgewise
   flight along ``+x``, passing over the array at ``t_pass``).
2. Flight sim: :func:`auraflow.cona.flight.simulate` (6-DOF, geometric
   controller) -> per-rotor speed/thrust histories.
3. Airloads: :func:`auraflow.cona.airloads.rotor_section_state` (Beddoes
   prescribed wake + Wagner unsteady) with a hover-trimmed collective.
4. Tonal: :func:`auraflow.cona.tonal.cona_tonal_noise` (F1A/F1C).
5. Broadband: :func:`auraflow.cona.broadband.rotor_broadband_spectrogram` (BPM).
6. Auralization: :func:`auraflow.cona.auralize.synthesize_observer_signal`
   (Fast Griffin--Lim per rotor, summed with the tonal signal).

Deviations from the digest (documented, not silent)
---------------------------------------------------
- **Atmospheric absorption (ISO 9613-1) is NOT applied.** The digest's
  generation pipeline (steps 1-6) contains no air-absorption stage; the only
  propagation-distance effect the paper uses is de-Dopplerization, which is a
  *downstream GP preprocessing* step, not part of the synthesized signal. We
  therefore emit free-field pressure and leave absorption/ground effects to a
  post-process. (Hook: apply a per-band ISO 9613-1 attenuation to the broadband
  spectrogram and a distance-dependent low-pass to the tonal signal if a future
  digest revision says the paper did.)
- **Ground reflection is NOT modeled** (mics sit at ``z = 0`` but no rigid-ground
  image source); the digest states mic placement but not a ground model.
- **Trim** uses a single hover-trimmed collective for all rotors rather than the
  paper's per-rotor fore/aft RPM trim along the mission; valid at the slow
  (1-10 m/s) level-flight speeds. The fore/aft *thrust* split still emerges from
  the closed-loop flight sim (the controller trims each rotor's speed).

Shapes: microphones ``[O, 3]``; audio ``[O, n]`` with ``n = round(fs*duration)``.
SI, float64. Observer-chunked so the (potentially 256-mic) grid fits host memory.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from auraflow.cona.airloads import rotor_section_state
from auraflow.cona.auralize import synthesize_observer_signal
from auraflow.cona.broadband import rotor_broadband_spectrogram
from auraflow.cona.flight import ControllerGains, simulate, straight_flyover
from auraflow.cona.gusts import dryden_gust
from auraflow.cona.tonal import cona_tonal_noise
from auraflow.core.medium import Medium
from auraflow.datasets.nasa_1pax import (
    BPF_HZ,
    nasa_1pax_hover_collective,
    nasa_1pax_multirotor,
    nasa_1pax_polar,
    nasa_1pax_vehicle,
)
from auraflow.signal.spectra import third_octave_bands

__all__ = [
    "JASAScenario",
    "generate_flyover",
    "generate_scenario_grid",
    "jasa_microphone_array",
    "save_flyover",
    "scenario_id",
]


def jasa_microphone_array(
    x_range: tuple[float, float] = (-150.0, 160.0),
    y_range: tuple[float, float] = (0.0, 70.0),
    step: float = 10.0,
    z: float = 0.0,
) -> Array:
    """The JASA ground microphone array (paper default: 256 mics, 32x8).

    A regular ground-level grid (``docs/research/jasa-datagen-reference.md``):
    ``x in [-150, 160]`` and ``y in [0, 70]`` on a ``10 m`` grid gives ``32 x 8
    = 256`` mics (``y >= 0`` by lateral symmetry). Row-major with ``x`` fastest.

    Args:
        x_range: Inclusive ``(x_min, x_max)`` streamwise extent [m].
        y_range: Inclusive ``(y_min, y_max)`` lateral extent [m].
        step: Grid spacing [m].
        z: Microphone height [m] (0 = ground).

    Returns:
        Microphone positions [m], shape ``[O, 3]`` (world frame, z up).
    """
    nx = int(round((x_range[1] - x_range[0]) / step)) + 1
    ny = int(round((y_range[1] - y_range[0]) / step)) + 1
    xs = jnp.linspace(x_range[0], x_range[1], nx)
    ys = jnp.linspace(y_range[0], y_range[1], ny)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    zz = jnp.full_like(xx, z)
    return jnp.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)


@dataclass
class JASAScenario:
    """One JASA flyover case (physical parameters only; numerics live in ``generate_flyover``).

    Defaults follow the digest's nominal case (``docs/research/
    jasa-datagen-reference.md``): level edgewise flight along ``+x`` at 30 m,
    passing over the array origin at mid-run, 1 s at 44.1 kHz.

    Attributes:
        speed: Ground speed ``V_inf`` [m/s] (digest sweeps 1-10, GP uses 6-10).
        altitude: Constant flight altitude [m] (world ``z``).
        heading_deg: Flight heading about world ``+z`` [deg] (0 = along ``+x``).
        lateral_offset: World ``y`` the track passes over at ``t_pass`` [m]
            (0 = over the array centerline).
        duration: Signal duration [s].
        fs: Audio sample rate [Hz].
        seed: PRNG seed (drives the gust realization and Griffin--Lim phases;
            fixes determinism).
        gust_w20: Dryden mean wind at 20 ft [m/s] or preset
            (``"light"``/``"moderate"``/``"severe"``); ``0`` = calm (digest:
            "no wind stated", so the default is calm).
        t_pass: Time the track passes over ``(0, lateral_offset)`` [s]; default
            ``duration / 2`` (digest: passes over origin at mid-run).
        mics: Explicit microphone positions ``[O, 3]`` (array-like) or ``None``
            for the paper's 256-mic array (:func:`jasa_microphone_array`).
    """

    speed: float = 8.0
    altitude: float = 30.0
    heading_deg: float = 0.0
    lateral_offset: float = 0.0
    duration: float = 1.0
    fs: float = 44100.0
    seed: int = 0
    gust_w20: float | str = 0.0
    t_pass: float | None = None
    mics: Any = field(default=None)

    def pass_time(self) -> float:
        """Resolved pass-over time [s] (``duration / 2`` if ``t_pass`` is None)."""
        return 0.5 * self.duration if self.t_pass is None else float(self.t_pass)

    def microphones(self) -> Array:
        """Resolved microphone array ``[O, 3]`` (paper default if ``mics`` is None)."""
        if self.mics is None:
            return jasa_microphone_array()
        return jnp.asarray(self.mics, dtype=float)

    def to_meta(self) -> dict[str, Any]:
        """JSON-serializable physical metadata for this scenario."""
        d = asdict(self)
        d.pop("mics")
        d["t_pass"] = self.pass_time()
        d["heading_rad"] = math.radians(self.heading_deg)
        d["bpf_hz"] = BPF_HZ
        return d


def _world_gust(key: Array, scenario: JASAScenario, dt: float, n_steps: int) -> Array | None:
    """World-frame Dryden gust series ``[T, 3]`` (or ``None`` if calm).

    :func:`auraflow.cona.gusts.dryden_gust` returns body ``(u, v, w-down)``; we
    rotate ``u`` along the heading, ``v`` laterally and map the down component to
    world ``+z`` (up) as ``-w``. Returned as the free-stream perturbation the
    airloads/tonal stages add to the wind.
    """
    if scenario.gust_w20 == 0.0:
        return None
    body = dryden_gust(
        key,
        altitude=scenario.altitude,
        airspeed=max(scenario.speed, 1.0),
        w20=scenario.gust_w20,
        dt=dt,
        n_steps=n_steps,
    )  # [T, 3] = (u, v, w_down)
    th = math.radians(scenario.heading_deg)
    fwd = jnp.array([math.cos(th), math.sin(th), 0.0])
    lat = jnp.array([-math.sin(th), math.cos(th), 0.0])
    up = jnp.array([0.0, 0.0, 1.0])
    return body[:, 0:1] * fwd + body[:, 1:2] * lat + (-body[:, 2:3]) * up


def _chunks(n: int, size: int) -> list[tuple[int, int]]:
    """Half-open ``[lo, hi)`` observer-chunk ranges covering ``range(n)``."""
    return [(i, min(i + size, n)) for i in range(0, n, size)]


def _maybe_clear(low_memory: bool) -> None:
    """Drop XLA compile caches at a stage boundary (see ``low_memory``)."""
    if low_memory:
        jax.clear_caches()


def generate_flyover(
    scenario: JASAScenario,
    *,
    medium: Medium | None = None,
    polar: Any = None,
    collective: float | None = None,
    n_stations: int = 16,
    n_source_times: int | None = None,
    n_frames: int = 48,
    n_fft: int = 2048,
    gl_iters: int = 60,
    fmin: float = 100.0,
    fmax: float = 20000.0,
    include_broadband: bool = True,
    obs_chunk: int = 16,
    low_memory: bool = False,
    wake_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    r"""Generate one JASA flyover: scenario -> per-mic auralized audio + metadata.

    Args:
        scenario: The :class:`JASAScenario` (physical parameters).
        medium: Ambient medium (default sea-level ISA; altitude effect on
            ``c0``/``rho0`` over 30 m is negligible).
        polar: Airfoil polar (default :func:`nasa_1pax_polar`).
        collective: Airload collective pitch [rad]; default hover-trimmed
            (:func:`nasa_1pax_hover_collective`).
        n_stations: Radial blade stations (static int).
        n_source_times: Flight/source time samples ``T``; default
            ``max(round(400*duration), 200)`` -- resolves tonal harmonics to a
            few hundred Hz (the digest's tonal energy band).
        n_frames: Broadband spectrogram frames (static int).
        n_fft: Griffin--Lim STFT length (static int; must resolve ``fmin`` and
            satisfy ``n_fft <= n``).
        gl_iters: Griffin--Lim iterations (static int).
        fmin: Lowest 1/3-octave band centre [Hz].
        fmax: Highest 1/3-octave band centre [Hz]; clamped to ``0.45*fs``.
        include_broadband: Include the BPM broadband component.
        obs_chunk: Microphones processed per acoustic-propagation batch (bounds
            peak host memory; the full 256-mic grid otherwise blows up).
        low_memory: Call :func:`jax.clear_caches` at stage boundaries. The XLA
            compile caches of the successive pipeline stages (trim, flight sim,
            airloads, tonal, broadband, Griffin--Lim) otherwise accumulate to
            >1 GB host RAM even for tiny cases; clearing caps the peak at
            ~750 MB at the cost of recompiling per call/chunk. Use on small
            machines (the dev box); leave off on GPU boxes where the caches
            make repeated scenarios/chunks much faster.
        wake_kwargs: Extra kwargs for
            :func:`~auraflow.cona.airloads.rotor_section_state` (e.g.
            ``n_wake_azimuth``, ``include_induced``).

    Returns:
        A dict with (all NumPy arrays unless noted):

        - ``"audio"`` ``[O, n]`` total pressure [Pa], ``n = round(fs*duration)``;
        - ``"tonal"``, ``"broadband"`` ``[O, n]`` the components;
        - ``"t_audio"`` ``[n]`` audio time grid [s];
        - ``"mics"`` ``[O, 3]`` microphone positions [m];
        - ``"band_centers"`` ``[n_bands]`` 1/3-octave centres [Hz];
        - ``"meta"`` the scenario metadata dict (+ generation numerics);
        - ``"scenario"`` the :class:`JASAScenario`.
    """
    medium = Medium() if medium is None else medium
    polar = nasa_1pax_polar() if polar is None else polar
    wake_kwargs = {} if wake_kwargs is None else dict(wake_kwargs)

    fs = float(scenario.fs)
    duration = float(scenario.duration)
    fmax_eff = min(float(fmax), 0.45 * fs)
    bands, _ = third_octave_bands(fmin, fmax_eff)
    n_audio = int(round(fs * duration))
    n_fft = min(n_fft, n_audio)

    if n_source_times is None:
        n_source_times = max(int(round(400.0 * duration)), 200)
    t = jnp.linspace(0.0, duration, n_source_times)
    dt = float(t[1] - t[0])

    if collective is None:
        collective = nasa_1pax_hover_collective(n_stations, medium, polar)
        _maybe_clear(low_memory)

    vehicle = nasa_1pax_vehicle(n_stations)
    # Gust couples into the flight dynamics only if drag_coeff > 0; here we treat
    # the gust as a free-stream perturbation on the airloads (documented), so the
    # multirotor keeps drag_coeff = 0 and the flight sim tracks the nominal line.
    mrotor = nasa_1pax_multirotor()
    gains = ControllerGains.for_vehicle(mrotor)

    key = jax.random.PRNGKey(int(scenario.seed))
    gust_key, phase_key = jax.random.split(key)
    gust = _world_gust(gust_key, scenario, dt, n_source_times)

    # --- Flight (once) -------------------------------------------------------
    heading = math.radians(scenario.heading_deg)
    ref = straight_flyover(
        scenario.speed,
        scenario.altitude,
        heading,
        t_pass=scenario.pass_time(),
        origin_xy=(0.0, scenario.lateral_offset),
    )
    x0, v0, _, _ = ref(jnp.asarray(0.0))
    flight = simulate(mrotor, gains, ref, t, x0, v0)
    _maybe_clear(low_memory)

    # --- Airloads (once; observer-independent) -------------------------------
    states = None
    if include_broadband:
        states = [
            rotor_section_state(
                vehicle,
                flight,
                i,
                medium,
                collective=collective,
                polar=polar,
                gust=gust,
                **wake_kwargs,
            )
            for i in range(vehicle.n_rotors)
        ]
        _maybe_clear(low_memory)

    mics = scenario.microphones()
    n_obs = int(mics.shape[0])
    audio = np.zeros((n_obs, n_audio))
    tonal = np.zeros((n_obs, n_audio))
    broadband = np.zeros((n_obs, n_audio))

    for lo, hi in _chunks(n_obs, obs_chunk):
        obs = mics[lo:hi]
        p_tonal, _, _, t_obs = cona_tonal_noise(
            vehicle,
            flight,
            obs,
            medium,
            collective=collective,
            polar=polar,
            gust=gust,
            **wake_kwargs,
        )  # [c, T_obs], [T_obs]
        specs_chunk: list[Array] = []
        if include_broadband and states is not None:
            for st in states:
                _, spec, _ = rotor_broadband_spectrogram(
                    st,
                    obs,
                    medium,
                    t,
                    bands=bands,
                    n_frames=n_frames,
                )  # [c, n_frames, n_bands]
                specs_chunk.append(spec)

        for j in range(hi - lo):
            o = lo + j
            bb_specs = [spec[j] for spec in specs_chunk] if specs_chunk else None
            out = synthesize_observer_signal(
                fs,
                duration,
                tonal_pressure=p_tonal[j],
                tonal_t=t_obs,
                broadband_spectrograms=bb_specs,
                band_centers=bands,
                n_fft=n_fft,
                n_iters=gl_iters,
                key=jax.random.fold_in(phase_key, o),
            )
            audio[o] = np.asarray(out["total"])
            tonal[o] = np.asarray(out["tonal"])
            broadband[o] = np.asarray(out["broadband"])
        _maybe_clear(low_memory)

    meta = scenario.to_meta()
    meta.update(
        n_source_times=n_source_times,
        n_frames=n_frames,
        n_fft=n_fft,
        gl_iters=gl_iters,
        n_stations=n_stations,
        fmin=fmin,
        fmax=fmax_eff,
        collective_rad=float(collective),
        n_mics=n_obs,
        n_audio=n_audio,
        include_broadband=include_broadband,
        c0=float(medium.c0),
        rho0=float(medium.rho0),
    )
    return {
        "audio": audio,
        "tonal": tonal,
        "broadband": broadband,
        "t_audio": np.asarray(jnp.arange(n_audio) / fs),
        "mics": np.asarray(mics),
        "band_centers": np.asarray(bands),
        "meta": meta,
        "scenario": scenario,
    }


def scenario_id(scenario: JASAScenario) -> str:
    """Stable, filesystem-safe id for a scenario (used as the dataset sample key).

    Encodes the physical knobs: ``V<speed>_A<alt>_H<heading>_Y<offset>_s<seed>``
    (speeds/angles rounded to 0.1). Two scenarios with the same physics share an
    id (and therefore overwrite / dedup).
    """
    return (
        f"V{scenario.speed:04.1f}_A{scenario.altitude:04.1f}"
        f"_H{scenario.heading_deg:05.1f}_Y{scenario.lateral_offset:+05.1f}"
        f"_s{int(scenario.seed):03d}"
    ).replace(".", "p")


def save_flyover(result: dict[str, Any], path: str) -> dict[str, str]:
    """Write a flyover result to ``<path>.npz`` and per-mic float32 WAV files.

    The ``.npz`` bundles every array plus the metadata (as a JSON string); WAVs
    (float32, preserving absolute Pa) go under ``<path>/mic_XXX.wav`` so they can
    also feed the dload commit (:mod:`auraflow.datasets.dload_io`).

    Args:
        result: A :func:`generate_flyover` result dict.
        path: Output path stem (no extension); parent dirs are created.

    Returns:
        Dict of written paths: ``{"npz": ..., "wav_dir": ...}``.
    """
    import json

    from scipy.io import wavfile

    stem = os.fspath(path)
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    npz_path = stem + ".npz"
    np.savez_compressed(
        npz_path,
        audio=result["audio"],
        tonal=result["tonal"],
        broadband=result["broadband"],
        t_audio=result["t_audio"],
        mics=result["mics"],
        band_centers=result["band_centers"],
        meta_json=json.dumps(result["meta"]),
    )
    fs = int(round(float(result["meta"]["fs"])))
    wav_dir = stem + "_wav"
    os.makedirs(wav_dir, exist_ok=True)
    audio = np.asarray(result["audio"], dtype=np.float32)
    for o in range(audio.shape[0]):
        wavfile.write(os.path.join(wav_dir, f"mic_{o:03d}.wav"), fs, audio[o])
    return {"npz": npz_path, "wav_dir": wav_dir}


def generate_scenario_grid(
    speeds: Sequence[float],
    altitudes: Sequence[float],
    seeds: Sequence[int],
    *,
    headings_deg: Sequence[float] = (0.0,),
    lateral_offsets: Sequence[float] = (0.0,),
    duration: float = 1.0,
    fs: float = 44100.0,
    gust_w20: float | str = 0.0,
    mics: ArrayLike | None = None,
) -> list[JASAScenario]:
    """Cartesian product of scenario knobs -> a list of :class:`JASAScenario`.

    Args:
        speeds: Ground speeds [m/s].
        altitudes: Flight altitudes [m].
        seeds: PRNG seeds.
        headings_deg: Headings [deg].
        lateral_offsets: Lateral track offsets [m].
        duration, fs, gust_w20: Shared scenario settings.
        mics: Shared explicit mic array, or ``None`` for the paper array.

    Returns:
        One :class:`JASAScenario` per combination (speeds outermost).
    """
    out: list[JASAScenario] = []
    for v in speeds:
        for a in altitudes:
            for h in headings_deg:
                for y in lateral_offsets:
                    for s in seeds:
                        out.append(
                            JASAScenario(
                                speed=float(v),
                                altitude=float(a),
                                heading_deg=float(h),
                                lateral_offset=float(y),
                                duration=duration,
                                fs=fs,
                                seed=int(s),
                                gust_w20=gust_w20,
                                mics=mics,
                            )
                        )
    return out
