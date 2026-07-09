"""Lofted blade / rotor meshes: watertightness, volume, chord, twist, acoustics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.body.airfoil_profile import naca0012
from auraflow.body.blade import blade_mesh, rotor_mesh
from auraflow.body.motion import SpinMotion
from auraflow.body.sources import mesh_pressure
from auraflow.core.blade import BladeGeometry, Rotor
from auraflow.core.medium import Medium


def _blade(n_stations: int = 12, twist_tip: float = -0.3) -> BladeGeometry:
    return BladeGeometry.linear(
        radius=1.0,
        hub_radius=0.15,
        n_stations=n_stations,
        chord_root=0.2,
        chord_tip=0.15,
        twist_root=0.0,
        twist_tip=twist_tip,
    )


def _unit_section_area(n_chord: int) -> float:
    loop = np.asarray(naca0012(n_chord))
    x, y = loop[:, 0], loop[:, 1]
    return abs(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


class TestBladeMesh:
    def test_watertight_and_positive_volume(self):
        m = blade_mesh(_blade(), n_chord=30)
        assert m.is_watertight
        assert float(m.volume()) > 0.0  # outward-wound

    def test_open_when_caps_off(self):
        m = blade_mesh(_blade(), n_chord=20, root_cap=False, tip_cap=False)
        assert not m.is_watertight

    def test_volume_matches_section_integral(self):
        # V ~ integral A_section(r) dr with A_section = area_unit * chord(r)^2.
        n_chord = 40
        g = _blade(n_stations=24)
        m = blade_mesh(g, n_chord=n_chord)
        area_unit = _unit_section_area(n_chord)
        c, dr = np.asarray(g.chord), np.asarray(g.dr)
        v_expected = float(np.sum(area_unit * c**2 * dr))
        assert float(m.volume()) == pytest.approx(v_expected, rel=0.03)

    def test_chord_extent_at_station(self):
        n_chord = 40
        g = _blade(n_stations=16)
        m = blade_mesh(g, n_chord=n_chord)
        n_loop = 2 * n_chord - 2
        v = np.asarray(m.vertices)
        # Interior station 8: its ring is vertices [8*N : 9*N]; the LE->TE span
        # in the (y, z) section plane is the local chord (pitch preserves it).
        i = 8
        ring = v[i * n_loop : (i + 1) * n_loop, 1:]
        extent = float(np.max(np.linalg.norm(ring[:, None, :] - ring[None, :, :], axis=-1)))
        assert extent == pytest.approx(float(g.chord[i]), rel=0.02)

    def test_twist_of_each_section(self):
        # The chord line (LE node - TE node) makes angle == twist with +y.
        n_chord = 24
        g = _blade(n_stations=10, twist_tip=-0.35)
        m = blade_mesh(g, n_chord=n_chord)
        n_loop = 2 * n_chord - 2
        v = np.asarray(m.vertices)
        for i in range(g.n_stations):
            le = v[i * n_loop + 0]
            te = v[i * n_loop + (n_chord - 1)]
            vy, vz = le[1] - te[1], le[2] - te[2]
            angle = float(np.arctan2(vz, vy))
            assert angle == pytest.approx(float(g.twist[i]), abs=1e-6)

    def test_volume_grad_through_chord_scale(self):
        g = _blade(n_stations=12)

        def vol(scale):
            g2 = BladeGeometry(g.radius, g.hub_radius, g.chord * scale, g.twist)
            return blade_mesh(g2, n_chord=20).volume()

        grad = jax.grad(vol)(1.0)
        assert np.isfinite(float(grad))
        assert float(grad) > 0.0  # volume ~ chord^2, increasing


class TestRotorMesh:
    def test_n_blade_volume(self):
        g = _blade(n_stations=10)
        blade = blade_mesh(g, n_chord=20)
        rotor = Rotor(blade=g, n_blades=3)
        rm = rotor_mesh(rotor, n_chord=20)
        assert rm.is_watertight  # merge preserves watertightness
        assert float(rm.volume()) == pytest.approx(3.0 * float(blade.volume()), rel=1e-6)

    def test_hub_adds_volume(self):
        g = _blade(n_stations=8)
        rotor = Rotor(blade=g, n_blades=2)
        no_hub = rotor_mesh(rotor, n_chord=16)
        with_hub = rotor_mesh(rotor, n_chord=16, hub=True)
        r_hub = 0.15
        hub_vol = np.pi * r_hub**2 * (0.5 * r_hub)
        assert float(with_hub.volume()) == pytest.approx(float(no_hub.volume()) + hub_vol, rel=0.02)

    def test_blades_at_correct_azimuths(self):
        g = _blade(n_stations=8)
        blade = blade_mesh(g, n_chord=16)
        vb = blade.n_vertices
        rotor = Rotor(blade=g, n_blades=2)  # azimuths [0, pi]
        rm = rotor_mesh(rotor, n_chord=16)
        v = np.asarray(rm.vertices)
        expected = np.asarray(rotor.blade_azimuths(0.0))
        # Each blade's centroid is rot_z(psi_b) @ c0 (c0 the section-frame blade
        # centroid, which has a small y-offset from the quarter-chord pitch axis),
        # so relative centroid azimuths equal the relative blade azimuths.
        az = []
        for b in range(2):
            cen = v[b * vb : (b + 1) * vb].mean(axis=0)
            az.append(np.arctan2(cen[1], cen[0]))
        rel = np.angle(np.exp(1j * ((az[1] - az[0]) - (expected[1] - expected[0]))))
        assert abs(rel) < 1e-6


class TestRotorAcoustics:
    def test_thickness_radiation_periodic_at_bpf(self):
        # Tiny spinning 2-blade rotor, thickness-only (no surface pressure): the
        # radiated pressure at one observer is finite and peaks at the BPF
        # (N_b * Omega / 2pi) in its spectrum.
        med = Medium()
        g = _blade(n_stations=5)
        rotor = Rotor(blade=g, n_blades=2)
        rm = rotor_mesh(rotor, n_chord=8)
        f_rot = 40.0
        omega = 2.0 * np.pi * f_rot
        bpf = rotor.n_blades * f_rot
        motion = SpinMotion.constant(axis=(0.0, 0.0, 1.0), omega=omega)
        n_rev = 8
        tau = jnp.linspace(0.0, n_rev / f_rot, 640)
        # Off-axis, in-plane-ish observer so blade passage modulates the retarded
        # distance at the BPF (an on-axis observer sees a nearly azimuth-constant
        # distance and hence no BPF tone).
        obs = jnp.array([[3.0, 0.0, 0.5]])
        p, t_obs = mesh_pressure(rm, motion, tau, obs, med)
        p = np.asarray(p[0])
        assert np.all(np.isfinite(p))
        assert float(np.max(np.abs(p))) > 0.0
        # The signal is periodic at the BPF (two identical blades -> the rotor
        # repeats every half revolution = 1 / BPF), so its spectrum is a comb of
        # BPF harmonics. Assert (a) the dominant AC tone lands on a BPF harmonic,
        # and (b) the BPF fundamental itself carries real energy.
        dt = float(t_obs[1] - t_obs[0])
        p_ac = p - p.mean()
        spec = np.abs(np.fft.rfft(p_ac))
        freqs = np.fft.rfftfreq(p_ac.size, dt)
        bin_hz = 1.0 / (p_ac.size * dt)
        f_peak = freqs[1 + np.argmax(spec[1:])]
        k = round(f_peak / bpf)
        assert k >= 1
        assert abs(f_peak - k * bpf) <= 2.0 * bin_hz  # dominant tone is a BPF harmonic
        # BPF fundamental is a substantial spectral component (not floor-level).
        i_bpf = int(round(bpf / bin_hz))
        assert spec[i_bpf] > 0.1 * spec.max()
