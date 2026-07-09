"""Analytic validation gates for mesh FW-H radiation (``docs/architecture.md``).

Each test compares a mesh radiated by the body-FW-H path against a classical
closed-form solution:

- **Pulsating (breathing) sphere** vs the exact monopole field
  ``p(r,t) = (rho0 c0 U0 a^2 k / r) cos(k(r-a) - omega t + phi) / sqrt(1+(ka)^2)``
  (Kinsler & Frey, *Fundamentals of Acoustics*, pulsating sphere).
- **Oscillating rigid sphere** vs the analytic dipole (cos-theta directivity and
  the compact volume-displacement amplitude; see the class docstring for the
  thickness-only added-mass caveat).
- **Baffled circular piston** vs the Rayleigh on-axis pressure and the
  ``2 J1(ka sin th)/(ka sin th)`` far-field directivity.
- **Imported-vs-parametric** mesh equivalence, and a translating-source
  **Doppler** shift.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy.special import j1

from auraflow.body import (
    ConstantVelocity,
    HarmonicTranslation,
    StaticPose,
    SurfaceVibration,
    TriMesh,
    circular_piston,
    load_mesh,
    mesh_pressure,
    save_mesh,
)
from auraflow.core.medium import Medium
from auraflow.signal.spectra import narrowband_spectrum

MED = Medium()
C0 = float(MED.c0)
RHO0 = float(MED.rho0)


def _breathing_vibration(
    mesh: TriMesh, tau: jnp.ndarray, u0: float, omega: float
) -> SurfaceVibration:
    """Uniform radial breathing ``u_n = u0 sin(omega t)`` on every face."""
    un = u0 * jnp.sin(omega * tau)
    return SurfaceVibration(
        tuple(range(mesh.n_faces)), tau, jnp.broadcast_to(un[None, :], (mesh.n_faces, tau.size))
    )


def _steady_peak(signal: ArrayLike) -> np.ndarray:
    """Peak absolute (mean-removed) amplitude over the steady portion of a signal."""
    p = np.asarray(signal)
    n = p.shape[-1]
    seg = p[..., int(0.45 * n) : int(0.95 * n)]
    return np.max(np.abs(seg - seg.mean(axis=-1, keepdims=True)), axis=-1)


class TestPulsatingSphere:
    """Breathing icosphere vs the exact monopole solution (ka = 0.3)."""

    def _run(self, r_obs: float, subdiv: int = 2, periods: int = 7, T: int = 512):
        a, ka = 0.1, 0.3
        k = ka / a
        omega = k * C0
        u0 = 0.01
        mesh = TriMesh.sphere(radius=a, subdivisions=subdiv)
        tau = jnp.linspace(0.0, periods * 2 * np.pi / omega, T)
        vib = _breathing_vibration(mesh, tau, u0, omega)
        obs = jnp.array([[r_obs, 0.0, 0.0]])
        p, t_obs = mesh_pressure(mesh, StaticPose(), tau, obs, MED, vibration=vib)
        # Exact pulsating-sphere pressure phasor (surface velocity u0 sin(omega t)
        # <-> complex amplitude i*u0 with the e^{-i omega t} convention).
        phasor = (1j * u0) * (k * a / (k * a + 1j)) * np.exp(1j * k * (r_obs - a))
        amp = RHO0 * C0 * a * phasor / r_obs
        p_ana = np.real(amp * np.exp(-1j * omega * np.asarray(t_obs)))
        return np.asarray(p[0]), p_ana

    def test_matches_exact_monopole(self):
        p, p_ana = self._run(r_obs=1.5)
        n = p.size
        sl = slice(int(0.45 * n), int(0.95 * n))
        rel = np.linalg.norm(p[sl] - p_ana[sl]) / np.linalg.norm(p_ana[sl])
        assert rel < 0.02

    def test_inverse_distance_scaling(self):
        # Monopole pressure amplitude is exactly 1/r; peak(r)/peak(2r) -> 2.
        p1, _ = self._run(r_obs=1.5)
        p2, _ = self._run(r_obs=3.0)
        ratio = float(_steady_peak(p1) / _steady_peak(p2))
        assert ratio == pytest.approx(2.0, rel=0.02)


class TestOscillatingSphere:
    """Rigid harmonic translation vs the analytic dipole (ka = 0.3).

    Thickness-only FW-H of a translating rigid sphere reproduces the
    *volume-displacement* dipole ``|p| = rho0 c0 U0 k^2 a^3 cos(th) / (3 r)``
    (velocity amplitude ``U0``). The exact rigid-sphere dipole is
    ``/(2 r)`` -- the extra factor 3/2 is the added-mass / scattering
    contribution carried by the surface loading term, which a thickness-only
    (no surface-pressure) model omits by construction. Both the exact-solution
    ratio (~2/3) and the strict cos-theta directivity are asserted here.
    """

    def _run(self):
        a, ka = 0.1, 0.3
        k = ka / a
        omega = k * C0
        d0 = 1e-4
        u0 = d0 * omega  # velocity amplitude
        mesh = TriMesh.sphere(radius=a, subdivisions=2)
        T, periods = 640, 8
        tau = jnp.linspace(0.0, periods * 2 * np.pi / omega, T)
        motion = HarmonicTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), d0, omega)
        r_obs = 1.5
        angs = np.array([0.0, 20.0, 40.0, 60.0]) * np.pi / 180.0
        obs = jnp.array([[r_obs * np.sin(t), 0.0, r_obs * np.cos(t)] for t in angs])
        p, _ = mesh_pressure(mesh, motion, tau, obs, MED)
        return _steady_peak(p), angs, k, a, u0, r_obs

    def test_cos_theta_directivity(self):
        amp, angs, *_ = self._run()
        np.testing.assert_allclose(amp / amp[0], np.cos(angs), rtol=0.01, atol=0.005)

    def test_on_axis_volume_displacement_amplitude(self):
        amp, angs, k, a, u0, r = self._run()
        p_voldisp = RHO0 * C0 * u0 * k**2 * a**3 * np.cos(angs) / (3.0 * r)
        assert amp[0] == pytest.approx(p_voldisp[0], rel=0.03)

    def test_two_thirds_of_exact_rigid_sphere(self):
        amp, angs, k, a, u0, r = self._run()
        x = k * a
        h1p = np.exp(1j * x) * (-1j / x + 2 / x**2 + 2j / x**3)
        p_exact = RHO0 * C0 * u0 * np.abs(np.cos(angs[0])) / (k * r * np.abs(h1p))
        assert float(amp[0] / p_exact) == pytest.approx(2.0 / 3.0, rel=0.03)


class TestBaffledPiston:
    """Baffled circular piston vs Rayleigh on-axis and far-field directivity."""

    def _piston(self, ka: float = 2.5):
        a = 0.1
        k = ka / a
        omega = k * C0
        u0 = 0.01
        spk = circular_piston(radius=a, n=8, baffled=True)
        return spk, a, k, omega, u0

    def test_on_axis_rayleigh(self):
        spk, a, k, omega, u0 = self._piston()
        T, periods = 768, 10
        tau = jnp.linspace(0.0, periods * 2 * np.pi / omega, T)
        un = u0 * jnp.sin(omega * tau)
        zs = np.array([0.5, 0.7, 0.9])
        obs = jnp.array([[0.0, 0.0, z] for z in zs])
        p, _ = spk.radiate(un, tau, obs, MED)
        amp = _steady_peak(p)
        # Rayleigh baffled-piston on-axis magnitude (Kinsler & Frey Eq. 7.5.x).
        p_ray = 2 * RHO0 * C0 * u0 * np.abs(np.sin(0.5 * k * (np.sqrt(zs**2 + a**2) - zs)))
        np.testing.assert_allclose(amp, p_ray, rtol=0.05)

    def test_far_field_directivity(self):
        spk, a, k, omega, u0 = self._piston()
        ka = k * a
        T, periods = 768, 6
        tau = jnp.linspace(0.0, periods * 2 * np.pi / omega, T)
        un = u0 * jnp.sin(omega * tau)
        r_obs = 3.0
        angs = np.array([10.0, 20.0, 30.0, 40.0]) * np.pi / 180.0
        obs = jnp.array([[r_obs * np.sin(t), 0.0, r_obs * np.cos(t)] for t in angs])
        p, _ = spk.radiate(un, tau, obs, MED)
        amp = _steady_peak(p)
        directivity = np.abs(2 * j1(ka * np.sin(angs)) / (ka * np.sin(angs)))
        np.testing.assert_allclose(amp / amp[0], directivity / directivity[0], rtol=0.05)


class TestImportedEquivalence:
    """A saved-and-reloaded icosphere radiates identically to the primitive."""

    def test_imported_matches_parametric(self, tmp_path):
        pytest.importorskip("trimesh")  # pyright: ignore[reportMissingImports]
        a, ka = 0.1, 0.3
        k = ka / a
        omega = k * C0
        mesh = TriMesh.sphere(radius=a, subdivisions=2)
        # OFF is ASCII and preserves float64 vertices to ~1e-11 on round-trip
        # (binary STL would downcast to float32); the panel set is identical up
        # to reordering, which does not affect the summed field.
        path = tmp_path / "sphere.off"
        save_mesh(mesh, str(path))
        reloaded = load_mesh(str(path))
        T = 384
        tau = jnp.linspace(0.0, 6 * 2 * np.pi / omega, T)
        obs = jnp.array([[1.2, 0.0, 0.0]])
        vib0 = _breathing_vibration(mesh, tau, 0.01, omega)
        vib1 = _breathing_vibration(reloaded, tau, 0.01, omega)
        p0, _ = mesh_pressure(mesh, StaticPose(), tau, obs, MED, vibration=vib0)
        p1, _ = mesh_pressure(reloaded, StaticPose(), tau, obs, MED, vibration=vib1)
        # Fields agree to the mesh-serialization precision (not the physics).
        np.testing.assert_allclose(np.asarray(p1), np.asarray(p0), rtol=1e-6, atol=1e-9)


class TestDoppler:
    """Translating breathing sphere: received-frequency shift 1/(1 - M cos th)."""

    def test_approaching_source_upshift(self):
        a = 0.02
        f_src = 500.0
        omega = 2 * np.pi * f_src
        mach = 0.2
        vx = mach * C0
        mesh = TriMesh.sphere(radius=a, subdivisions=2)
        T = 2048
        tau = jnp.linspace(0.0, 0.04, T)
        vib = _breathing_vibration(mesh, tau, 0.01, omega)
        motion = ConstantVelocity(x0=(-5.0, 0.0, 0.0), v=(vx, 0.0, 0.0))
        obs = jnp.array([[60.0, 0.0, 0.0]])  # on-axis ahead (cos theta = 1)
        p, t_obs = mesh_pressure(mesh, motion, tau, obs, MED, vibration=vib)
        fs = float(1.0 / (t_obs[1] - t_obs[0]))
        freqs, amp = narrowband_spectrum(p[0], fs)
        f_obs = float(freqs[int(jnp.argmax(amp))])
        f_expected = f_src / (1.0 - mach)
        assert abs(f_obs - f_expected) / f_expected < 0.01
