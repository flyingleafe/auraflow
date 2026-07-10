"""Hybrid CFD+FW-H flyover synthesis (no jaxfluids): tiling + moving-surface FW-H.

All cases use tiny SYNTHETIC surface histories on a coarse sphere (<= 320 panels,
<= 600 time samples, <= 4 mics) so they run under the dev-box memory cap.
"""

import numpy as np
import pytest

from auraflow.body.mesh import TriMesh
from auraflow.cfd.flyover import (
    quadrotor_surface_flyover,
    synthesize_flyover_wavs,
    tile_surface_history,
)
from auraflow.core.medium import Medium
from auraflow.fwh.f1a import f1a_permeable_static

OMEGA = 70.3
N_BLADES = 3
T_BP = 2.0 * np.pi / (OMEGA * N_BLADES)  # blade-passing period [s]


def _small_surface(radius=0.15, subdivisions=1):
    """A coarse breathing-sphere permeable surface as a {points,normals,area} dict."""
    mesh = TriMesh.sphere(radius=radius, subdivisions=subdivisions)
    return {
        "points": np.asarray(mesh.centroids(), dtype=np.float64),
        "normals": np.asarray(mesh.normals(), dtype=np.float64),
        "area": np.asarray(mesh.areas(), dtype=np.float64),
    }


def _breathing_history(surf, f0, tau, amp_u=2.0, amp_p=0.0, amp_rho=0.0, phase=None):
    """Synthetic gauge history: radial breathing at f0, optional p/rho fluctuation."""
    n_s = surf["points"].shape[0]
    n_t = tau.shape[0]
    phase = np.zeros(n_s) if phase is None else np.asarray(phase)
    s = np.sin(2.0 * np.pi * f0 * tau[None, :] + phase[:, None])  # [S,T]
    u = surf["normals"][:, None, :] * (amp_u * s)[:, :, None]  # [S,T,3] radial breathing
    p = amp_p * s
    rho = amp_rho * s
    return {"tau": tau, "rho": rho, "u": u, "p": p, "period_samples": n_t}


# --------------------------------------------------------------------------- #
# tile_surface_history                                                        #
# --------------------------------------------------------------------------- #


class TestTileSurfaceHistory:
    def _raw(self, n_periods_avail=4.3, samples_per_period=40):
        dtau = T_BP / samples_per_period
        n_in = int(round(n_periods_avail * samples_per_period))
        tau = np.arange(n_in) * dtau
        s_pt = np.arange(6)[:, None]  # 6 panels, distinct phases
        phase = 0.3 * s_pt
        # DC pedestal + fundamental + a 2nd harmonic (tests crossfade with harmonics)
        base = 5.0 + s_pt.astype(float)
        sig = np.sin(2 * np.pi * (OMEGA * N_BLADES / (2 * np.pi)) * tau[None, :] + phase)
        sig2 = 0.4 * np.sin(2 * 2 * np.pi * (OMEGA * N_BLADES / (2 * np.pi)) * tau[None, :] + phase)
        p = base + sig + sig2
        rho = 1.2 + 0.01 * (base + sig)
        u = np.stack([sig, sig2, np.zeros_like(sig)], axis=-1)
        return {"tau": tau, "rho": rho, "u": u, "p": p}, dtau

    def test_integer_period_trim(self):
        raw, _ = self._raw(n_periods_avail=4.3, samples_per_period=40)
        out = tile_surface_history(raw, OMEGA, N_BLADES, duration=0.4)
        assert out["n_periods"] == 4  # nearest integer <= 4.3
        assert out["period_samples"] == 40

    def test_trim_never_exceeds_data(self):
        raw, _ = self._raw(n_periods_avail=2.9, samples_per_period=30)
        out = tile_surface_history(raw, OMEGA, N_BLADES, duration=0.4)
        # 2.9 rounds to 3 but only 2.9 periods of data -> trimmed down to fit.
        assert out["n_periods"] * out["period_samples"] <= raw["tau"].shape[0]

    def test_mean_removed(self):
        raw, _ = self._raw()
        out = tile_surface_history(raw, OMEGA, N_BLADES, duration=0.5)
        # The tiling is exactly periodic (period = n_periods*period_samples =: L),
        # so one interior period [L:2L] reproduces the (mean-removed) segment; its
        # per-panel time-mean is machine-zero. (The truncated final partial period
        # and the ramped ends carry a tiny residual, expected.)
        seg_len = out["n_periods"] * out["period_samples"]
        interior = out["p"][:, seg_len : 2 * seg_len]
        assert np.allclose(interior.mean(axis=1), 0.0, atol=1e-9)
        interior_rho = out["rho"][:, seg_len : 2 * seg_len]
        assert np.allclose(interior_rho.mean(axis=1), 0.0, atol=1e-9)

    def test_output_length_and_shapes(self):
        raw, dtau = self._raw()
        duration = 0.5
        out = tile_surface_history(raw, OMEGA, N_BLADES, duration=duration)
        n_out = int(round(duration / dtau))
        assert out["p"].shape == (6, n_out)
        assert out["u"].shape == (6, n_out, 3)
        assert out["tau"].shape == (n_out,)

    def test_periodic_continuity_at_seam(self):
        raw, _ = self._raw()
        out = tile_surface_history(raw, OMEGA, N_BLADES, duration=0.6)
        p = out["p"]
        # within one clean period, the max sample-to-sample change:
        per = out["period_samples"]
        within = np.abs(np.diff(p[:, :per], axis=1)).max()
        # ignore the ramped fade-in/out ends; check every interior seam.
        xf = out["xfade"]
        seam = np.abs(np.diff(p[:, xf : p.shape[1] - xf], axis=1)).max()
        # exact-period tiling: the seam jump matches the within-period jump.
        assert seam < 1.5 * within


# --------------------------------------------------------------------------- #
# quadrotor_surface_flyover physics                                           #
# --------------------------------------------------------------------------- #


class TestStaticReduction:
    def test_speed_zero_matches_static_kernel(self):
        medium = Medium()
        surf = _small_surface(radius=0.2, subdivisions=1)
        tau = np.linspace(0.0, 0.2, 200)
        hist = _breathing_history(surf, f0=60.0, tau=tau, amp_u=1.5, amp_p=3.0, amp_rho=0.02)
        layout = (np.array([[0.0, 0.0, 0.0]]), np.array([1.0]))
        obs = np.array([[3.0, 1.0, -2.0], [-2.0, 4.0, 1.5]])

        p_move, t_obs = quadrotor_surface_flyover(
            surf, hist, layout, speed=0.0, altitude=0.0, t_pass=0.0,
            observers=obs, medium=medium, phase_offsets=[0.0],
        )  # fmt: skip

        # Reference: static permeable kernel on the same (ambient-restored) data.
        rho_abs = float(medium.rho0) + hist["rho"]
        p_abs = float(medium.p0) + hist["p"]
        pt, pl = f1a_permeable_static(
            obs, surf["points"], surf["normals"], surf["area"],
            rho_abs, hist["u"], p_abs, medium, tau, np.asarray(t_obs),
        )  # fmt: skip
        p_ref = np.asarray(pt + pl)
        p_move = np.asarray(p_move)

        err = np.linalg.norm(p_move - p_ref) / (np.linalg.norm(p_ref) + 1e-30)
        assert err < 1e-6


class TestDoppler:
    def test_approaching_tone_shift(self):
        medium = Medium()
        c0 = float(medium.c0)
        speed = 30.0
        f0 = 150.0
        surf = _small_surface(radius=0.1, subdivisions=1)
        duration = 0.3
        tau = np.linspace(0.0, duration, 480)
        hist = _breathing_history(surf, f0=f0, tau=tau, amp_u=1.0)
        layout = (np.array([[0.0, 0.0, 0.0]]), np.array([1.0]))
        # observer far downstream (+x): the source approaches ~head-on, cos(theta)~1
        obs = np.array([[300.0, 0.0, 0.0]])

        p, t_obs = quadrotor_surface_flyover(
            surf, hist, layout, speed=speed, altitude=0.0, t_pass=duration / 2,
            observers=obs, medium=medium, phase_offsets=[0.0],
        )  # fmt: skip
        p = np.asarray(p)[0]
        t_obs = np.asarray(t_obs)
        dt = float(t_obs[1] - t_obs[0])

        sig = p - p.mean()
        win = np.hanning(sig.size)
        spec = np.abs(np.fft.rfft(sig * win))
        freqs = np.fft.rfftfreq(sig.size, dt)
        k = int(np.argmax(spec[1:]) + 1)
        # parabolic interpolation for sub-bin peak frequency
        a, b, cc = spec[k - 1], spec[k], spec[k + 1]
        delta = 0.5 * (a - cc) / (a - 2 * b + cc)
        f_peak = freqs[k] + delta * (freqs[1] - freqs[0])

        mach = speed / c0
        f_expected = f0 / (1.0 - mach)  # cos(theta) ~ 1 (head-on)
        assert abs(f_peak - f_expected) / f_expected < 0.01


class TestMirroring:
    def test_counter_rotating_mirrors_field(self):
        medium = Medium()
        surf = _small_surface(radius=0.2, subdivisions=1)
        tau = np.linspace(0.0, 0.2, 200)
        # y-asymmetric history: fluctuation amplitude depends on panel y-coordinate.
        pts_y = surf["points"][:, 1]
        phase = 2.0 * pts_y
        hist = _breathing_history(surf, f0=70.0, tau=tau, amp_u=1.5, amp_p=2.0, phase=phase)
        # extra y-asymmetry in p:
        hist["p"] = hist["p"] * (1.0 + 0.5 * pts_y[:, None])

        # mirror pair of observers (y -> -y)
        obs = np.array([[10.0, 5.0, -3.0], [10.0, -5.0, -3.0]])
        hub = np.array([[0.0, 0.0, 0.0]])

        def _fly(spin):
            p, _ = quadrotor_surface_flyover(
                surf, hist, (hub, np.array([spin])), speed=8.0, altitude=3.0,
                t_pass=0.1, observers=obs, medium=medium, phase_offsets=[0.0],
            )  # fmt: skip
            return np.asarray(p)

        p_plus = _fly(1.0)  # co-rotating (original)
        p_minus = _fly(-1.0)  # counter-rotating (mirrored)

        # non-trivial and genuinely asymmetric (mirror test would be vacuous otherwise)
        assert np.linalg.norm(p_plus) > 1e-6
        assert np.linalg.norm(p_plus[0] - p_plus[1]) > 1e-3 * np.linalg.norm(p_plus)
        # mirrored rotor at (+y) == original rotor at (-y), and vice versa.
        scale = np.linalg.norm(p_plus)
        assert np.linalg.norm(p_minus[0] - p_plus[1]) < 1e-9 * scale
        assert np.linalg.norm(p_minus[1] - p_plus[0]) < 1e-9 * scale


class TestFourRotorAndWav:
    def test_four_rotor_sum_and_wav_synthesis(self):
        from auraflow.cona.flight import Multirotor

        medium = Medium()
        surf = _small_surface(radius=0.2, subdivisions=1)
        dtau = T_BP / 20.0
        n_in = 60
        tau = np.arange(n_in) * dtau
        raw = _breathing_history(surf, f0=OMEGA * N_BLADES / (2 * np.pi), tau=tau, amp_u=1.5,
                                 amp_p=2.0, amp_rho=0.01)  # fmt: skip
        tiled = tile_surface_history(raw, OMEGA, N_BLADES, duration=0.2)

        layout = Multirotor.nasa_1pax()  # Multirotor duck-typed (positions + spins)
        obs = np.array([[-40.0, 0.0, 0.0], [0.0, 0.0, 0.0], [40.0, 0.0, 0.0], [0.0, 30.0, 0.0]])
        p, t_obs = quadrotor_surface_flyover(
            surf, tiled, layout, speed=8.0, altitude=30.0, t_pass=0.1,
            observers=obs, medium=medium,
        )  # fmt: skip
        p = np.asarray(p)
        t_obs = np.asarray(t_obs)
        assert p.shape[0] == 4
        assert p.shape[1] == t_obs.shape[0]
        assert np.all(np.isfinite(p))

        fs_out = 44100.0
        wav = synthesize_flyover_wavs(p, t_obs, fs_out=fs_out)
        span = float(t_obs[-1] - t_obs[0])
        n_expected = int(round(span * fs_out)) + 1
        assert wav.shape == (4, n_expected)
        assert np.all(np.isfinite(wav))
        # effective output rate matches fs_out
        assert abs((wav.shape[1] - 1) / span - fs_out) / fs_out < 1e-3


class TestLazyTilingEquivalence:
    """The lazy per-chunk tiling path must reproduce the eager full-tile path
    bit-for-bit (GH #3): same crossfade, same phase-roll, same mirror."""

    def _case(self):
        from auraflow.cona.flight import Multirotor

        surf = _small_surface(radius=0.2, subdivisions=1)
        dtau = T_BP / 20.0
        n_in = 73  # not an integer # of periods -> trimming + crossfade exercised
        tau = np.arange(n_in) * dtau
        # y-asymmetric history so the xz-mirror on counter-rotating rotors matters.
        pts_y = surf["points"][:, 1]
        f0 = OMEGA * N_BLADES / (2 * np.pi)
        raw = _breathing_history(surf, f0=f0, tau=tau, amp_u=1.5, amp_p=2.0,
                                 amp_rho=0.01, phase=2.0 * pts_y)  # fmt: skip
        raw["p"] = raw["p"] * (1.0 + 0.5 * pts_y[:, None])  # extra p asymmetry
        layout = Multirotor.nasa_1pax()  # 4 rotors, mixed spin signs
        obs = np.array([[12.0, 5.0, -3.0], [-8.0, -4.0, 2.0], [0.0, 0.0, -6.0]])
        return surf, raw, layout, obs

    def test_lazy_matches_eager_all_rotors(self):
        medium = Medium()
        surf, raw, layout, obs = self._case()
        duration = 0.3  # >> the trimmed segment: many tiling seams
        # non-default, uneven phase offsets to exercise the per-rotor roll.
        phase_offsets = [0.0, 0.37, 0.61, 0.85]

        eager = tile_surface_history(raw, OMEGA, N_BLADES, duration=duration)
        lazy = tile_surface_history(raw, OMEGA, N_BLADES, duration=duration, lazy=True)
        assert lazy["lazy"] is True
        assert "rho" not in lazy  # the full arrays are NOT materialized
        assert lazy["tau"].shape == eager["tau"].shape

        def _fly(tiled):
            p, t = quadrotor_surface_flyover(
                surf, tiled, layout, speed=9.0, altitude=25.0, t_pass=0.12,
                observers=obs, medium=medium, phase_offsets=phase_offsets,
                panel_chunk=17, obs_chunk=2,
            )  # fmt: skip
            return np.asarray(p), np.asarray(t)

        p_eager, t_e = _fly(eager)
        p_lazy, t_l = _fly(lazy)
        assert np.allclose(t_e, t_l, atol=1e-12)
        # non-trivial signal (equivalence would be vacuous otherwise)
        assert np.linalg.norm(p_eager) > 1e-6
        assert np.allclose(p_lazy, p_eager, atol=1e-12, rtol=0.0)


class TestNoLowFrequencyPedestal:
    def test_near_mic_not_pedestal_dominated_with_wide_mic_array(self):
        """A shared observer grid serving mics at very different ranges must not
        inject a low-frequency pedestal at the near mic.

        ``quadrotor_surface_flyover`` builds ONE ``t_obs`` sized to the farthest
        mic (global ``d.max``). For a near (under-path) mic that grid runs far
        past the source's last arrival; the resample must leave that tail SILENT.
        The old clamped extrapolation froze each panel's endpoint integrand into
        a DC plateau over the tail -- a spurious sub-30 Hz pedestal that buried
        the blade tone (>90% of the received energy below 30 Hz on the DJI case).
        """
        medium = Medium()
        surf = _small_surface(radius=0.15, subdivisions=1)
        f0 = 200.0  # clean tone well above the 30 Hz pedestal band
        dur = 0.25
        tau = np.linspace(0.0, dur, int(round(dur / (T_BP / 40))))
        hist = _breathing_history(surf, f0=f0, tau=tau, amp_u=1.5, amp_p=4.0)
        hist["period_samples"] = int(round((1.0 / f0) / (tau[1] - tau[0])))
        layout = (np.array([[0.0, 0.0, 0.0]]), np.array([1.0]))
        # near under-path mic (x=0) + a far mic (x=200 m): the shared grid is
        # sized by the far mic and overruns the near mic's arrivals by ~0.5 s.
        obs = np.array([[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]])

        p, t_obs = quadrotor_surface_flyover(
            surf, hist, layout, speed=8.0, altitude=30.0, t_pass=dur / 2,
            observers=obs, medium=medium, phase_offsets=[0.0],
        )  # fmt: skip
        p = np.asarray(p)
        t_obs = np.asarray(t_obs)
        near = p[0]
        dt = float(t_obs[1] - t_obs[0])
        n = near.size
        spec = np.abs(np.fft.rfft(near)) ** 2
        freqs = np.fft.rfftfreq(n, dt)
        etot = spec.sum() + 1e-30
        frac_low = spec[freqs < 30.0].sum() / etot  # pedestal band
        frac_tone = spec[(freqs > f0 - 40) & (freqs < f0 + 40)].sum() / etot
        # The pedestal is gone: sub-30 Hz is a small fraction and the tone leads.
        assert frac_low < 0.15, f"low-frequency pedestal present: {frac_low:.2%} < 30 Hz"
        assert frac_tone > 0.4, f"blade tone suppressed: only {frac_tone:.2%} near f0"
        # The out-of-window tail is silent (zero-filled), not a frozen DC plateau.
        tail = near[int(0.9 * n) :]
        assert np.allclose(tail, 0.0), "near-mic tail should be silent, not a DC plateau"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
