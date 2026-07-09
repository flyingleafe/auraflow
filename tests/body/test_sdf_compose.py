"""Canonical-SDF composition + disk cache (the RPM/azimuth reuse core).

Checks that a whole-rotor SDF assembled from ONE canonical blade SDF agrees with
a directly-built rotor mesh SDF, that rotating the query equals re-building at a
shifted azimuth (exactly), and that :func:`cached_sdf_grid` memoizes by content.
Everything is capped (small blade, coarse grids, small ``batch_points``).
"""

import jax.numpy as jnp
import numpy as np
import pytest

import auraflow.body.sdf as sdf_mod
from auraflow.body.blade import blade_mesh, rotor_mesh
from auraflow.body.mesh import TriMesh
from auraflow.body.motion import axis_angle_matrix
from auraflow.body.sdf import cached_sdf_grid, sdf_eval, sdf_grid_jax
from auraflow.body.sdf_compose import CanonicalSDF, capped_cylinder_sdf, rotor_sdf
from auraflow.core.blade import BladeGeometry, Rotor

_BP = 256


def _small_rotor(n_blades=2):
    g = BladeGeometry.linear(
        radius=1.0, hub_radius=0.15, n_stations=6,
        chord_root=0.3, chord_tip=0.25, twist_root=0.0, twist_tip=0.0,
    )  # fmt: skip
    return Rotor(blade=g, n_blades=n_blades), g


class TestRotorComposition:
    def test_matches_direct_rotor_mesh(self):
        rotor, g = _small_rotor(2)
        blade = blade_mesh(g, n_chord=10)
        cb = CanonicalSDF.from_mesh(
            blade, padding=0.15, cells=(48, 28, 18), cache=False, batch_points=_BP
        )
        rs = rotor_sdf(cb, n_blades=2, azimuth=0.0, spin_direction=1)

        rm = rotor_mesh(rotor, n_chord=10)
        lo = np.array([-1.1, -1.1, -0.15])
        hi = np.array([1.1, 1.1, 0.15])
        n = (22, 22, 8)
        gd = sdf_grid_jax(rm, lo, hi, n, batch_points=_BP)
        xs = np.linspace(lo[0], hi[0], n[0])
        ys = np.linspace(lo[1], hi[1], n[1])
        zs = np.linspace(lo[2], hi[2], n[2])
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = jnp.asarray(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1))
        direct = np.asarray(sdf_eval(gd, lo, hi, pts))
        comp = np.asarray(rs(pts))
        # Two independent grid interpolations of the same geometry: agree within
        # a grid diagonal, and never disagree on sign far from the surface.
        diag = float(np.linalg.norm((hi - lo) / (np.asarray(n) - 1)))
        assert np.abs(direct - comp).max() < 1.5 * diag
        mism = np.sign(direct) != np.sign(comp)
        assert np.all(np.minimum(np.abs(direct), np.abs(comp))[mism] < diag)

    def test_azimuth_sweep_is_exact_rotation(self):
        _, g = _small_rotor(2)
        blade = blade_mesh(g, n_chord=8)
        cb = CanonicalSDF.from_mesh(
            blade, padding=0.15, cells=(40, 24, 16), cache=False, batch_points=_BP
        )
        dpsi = 0.7
        rs0 = rotor_sdf(cb, n_blades=2, azimuth=0.0, spin_direction=1)
        rs1 = rotor_sdf(cb, n_blades=2, azimuth=dpsi, spin_direction=1)
        rng = np.random.default_rng(0)
        pts = jnp.asarray(rng.uniform(-1.0, 1.0, size=(64, 3)))
        # Rotating the query by -dpsi about z (R_axis(-dpsi) x) then evaluating the
        # azimuth-0 rotor equals evaluating the azimuth-dpsi rotor at x.
        r = axis_angle_matrix(jnp.array([0.0, 0.0, 1.0]), -dpsi)
        rotated = pts @ r.T
        a = np.asarray(rs0(rotated))
        b = np.asarray(rs1(pts))
        assert np.abs(a - b).max() < 1e-12


class TestCappedCylinder:
    def test_sign_and_distance(self):
        # Unit-radius, half-height 0.5 cylinder about +z at the origin.
        pts = jnp.array([
            [0.0, 0.0, 0.0],   # centre: inside
            [2.0, 0.0, 0.0],   # radial outside by 1.0
            [0.0, 0.0, 1.5],   # axial outside by 1.0
        ])  # fmt: skip
        d = np.asarray(capped_cylinder_sdf(pts, radius=1.0, half_height=0.5))
        assert d[0] < 0.0
        assert d[1] == pytest.approx(1.0, abs=1e-9)
        assert d[2] == pytest.approx(1.0, abs=1e-9)


class TestDiskCache:
    def test_second_call_hits_cache(self, tmp_path, monkeypatch):
        mesh = TriMesh.sphere(0.5, 1)
        lo = np.array([-1.0, -1.0, -1.0])
        hi = np.array([1.0, 1.0, 1.0])
        calls = {"n": 0}
        real = sdf_mod.sdf_grid

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(sdf_mod, "sdf_grid", counting)

        g1 = cached_sdf_grid(mesh, lo, hi, 8, cache_dir=str(tmp_path), batch_points=_BP)
        g2 = cached_sdf_grid(mesh, lo, hi, 8, cache_dir=str(tmp_path), batch_points=_BP)
        assert calls["n"] == 1  # second call served from disk, no recompute
        assert np.allclose(np.asarray(g1), np.asarray(g2))

    def test_key_changes_with_vertices(self, tmp_path, monkeypatch):
        lo = np.array([-1.0, -1.0, -1.0])
        hi = np.array([1.0, 1.0, 1.0])
        calls = {"n": 0}
        real = sdf_mod.sdf_grid

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(sdf_mod, "sdf_grid", counting)

        m1 = TriMesh.sphere(0.5, 1)
        m2 = TriMesh(vertices=m1.vertices * 1.1, faces=m1.faces)  # perturbed geometry
        cached_sdf_grid(m1, lo, hi, 8, cache_dir=str(tmp_path), batch_points=_BP)
        cached_sdf_grid(m2, lo, hi, 8, cache_dir=str(tmp_path), batch_points=_BP)
        assert calls["n"] == 2  # different content -> different key -> rebuilt
        assert len(list(tmp_path.glob("*.npz"))) == 2
