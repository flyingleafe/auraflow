"""Mesh -> FW-H source adapters: ``impermeable_sources`` / ``mesh_pressure``.

Checks the source-density formulas and sign conventions, the one-call radiation
path, and the compact-limit cross-check (a thin plate carrying a uniform surface
pressure vs a single equivalent compact Farassat-1A dipole).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.body import (
    StaticPose,
    SurfaceVibration,
    TriMesh,
    impermeable_sources,
    mesh_pressure,
    permeable_surface,
)
from auraflow.core.medium import Medium
from auraflow.fwh import f1a_pressure


class TestImpermeableSources:
    def test_thickness_density_and_loading_sign(self):
        # A single flat plate (+z normal): breathing at u_n and gauge pressure ps.
        med = Medium()
        plate = TriMesh.flat_plate(chord=0.1, span=0.1)
        f = plate.n_faces
        tau = jnp.linspace(0.0, 0.01, 8)
        u0 = 0.3
        vib = SurfaceVibration(tuple(range(f)), tau, jnp.full((f, tau.size), u0))
        ps = jnp.full((f, tau.size), 2.5)  # gauge pressure [Pa]
        y, v, a, load, qn = impermeable_sources(
            plate, StaticPose(), tau, med, p_surface=ps, vibration=vib
        )
        # Q_n = rho0 (v . n); with a static pose the only normal velocity is u_n.
        np.testing.assert_allclose(np.asarray(qn), float(med.rho0) * u0, rtol=1e-12)
        # L = p_surface * n; the plate normal is +z, so load points +z with mag ps.
        assert np.allclose(np.asarray(load)[..., 2], 2.5, rtol=1e-12)
        assert np.allclose(np.asarray(load)[..., :2], 0.0, atol=1e-12)
        assert y.shape == v.shape == a.shape == load.shape

    def test_no_pressure_gives_zero_loading(self):
        med = Medium()
        mesh = TriMesh.sphere(0.1, 1)
        tau = jnp.linspace(0.0, 0.01, 6)
        _, _, _, load, qn = impermeable_sources(mesh, StaticPose(), tau, med)
        assert float(jnp.max(jnp.abs(load))) == 0.0
        # No motion, no vibration -> no thickness source either.
        assert float(jnp.max(jnp.abs(qn))) == pytest.approx(0.0, abs=1e-12)

    def test_permeable_surface_matches_mesh_geometry(self):
        mesh = TriMesh.sphere(0.7, 2)
        points, normals, areas = permeable_surface(mesh)
        np.testing.assert_allclose(np.asarray(points), np.asarray(mesh.centroids()))
        np.testing.assert_allclose(np.asarray(normals), np.asarray(mesh.normals()))
        np.testing.assert_allclose(np.asarray(areas), np.asarray(mesh.areas()))


class TestMeshPressure:
    def test_returns_thickness_plus_loading(self):
        # mesh_pressure must equal f1a_pressure(thickness) + f1a_pressure(loading).
        med = Medium()
        mesh = TriMesh.sphere(0.1, 1)
        f = mesh.n_faces
        tau = jnp.linspace(0.0, 0.006, 128)
        omega = 2 * np.pi * 900.0
        un = 0.02 * jnp.sin(omega * tau)
        vib = SurfaceVibration(tuple(range(f)), tau, jnp.broadcast_to(un[None, :], (f, tau.size)))
        ps = 3.0 * jnp.sin(omega * tau)[None, :] * jnp.ones((f, 1))
        obs = jnp.array([[0.5, 0.2, 0.1]])
        p, t_obs = mesh_pressure(mesh, StaticPose(), tau, obs, med, p_surface=ps, vibration=vib)
        # Rebuild sources and call the kernel directly with the same t_obs.
        y, v, a, load, qn = impermeable_sources(
            mesh, StaticPose(), tau, med, p_surface=ps, vibration=vib
        )
        pt, pl = f1a_pressure(obs, y, v, a, qn, load, med, tau, t_obs, mesh.areas())
        np.testing.assert_allclose(np.asarray(p), np.asarray(pt + pl), rtol=1e-12, atol=1e-14)

    def test_grad_through_vertices_is_finite(self):
        # d(radiated energy)/d(vertices) exists and is finite on a small case.
        med = Medium()
        mesh = TriMesh.sphere(0.05, 1)
        tau = jnp.linspace(0.0, 0.004, 96)
        omega = 2 * np.pi * 1500.0
        un = 0.01 * jnp.sin(omega * tau)

        def energy(verts):
            m = TriMesh(verts, mesh.faces, is_watertight=True)
            vib = SurfaceVibration(
                tuple(range(m.n_faces)), tau, jnp.broadcast_to(un[None, :], (m.n_faces, tau.size))
            )
            obs = jnp.array([[0.4, 0.0, 0.0]])
            p, _ = mesh_pressure(m, StaticPose(), tau, obs, med, vibration=vib)
            return jnp.sum(p[0] ** 2)

        g = jax.grad(energy)(mesh.vertices)
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.linalg.norm(g)) > 0.0


class TestCompactLimit:
    def test_thin_plate_loading_matches_compact_dipole(self):
        # A plate much smaller than the wavelength, much closer than the observer,
        # carrying a uniform gauge pressure p(t), radiates like a single compact
        # F1A dipole with force F = p(t) * area * n at the plate centre.
        med = Medium()
        plate = TriMesh.flat_plate(chord=0.02, span=0.02)  # 2 cm plate
        a_tot = float(plate.total_area())
        n = 400
        tau = jnp.linspace(0.0, 0.02, n)
        t0, sig = 0.01, 1e-3
        ps = jnp.exp(-0.5 * ((tau - t0) / sig) ** 2)  # gauge pressure [Pa]
        p_surface = jnp.broadcast_to(ps[None, :], (plate.n_faces, n))
        r0 = 3.0
        obs = jnp.array([[r0 * np.sin(0.5), 0.0, r0 * np.cos(0.5)]])  # oblique observer
        p_mesh, t_obs = mesh_pressure(plate, StaticPose(), tau, obs, med, p_surface=p_surface)
        # Equivalent compact point dipole (force on the fluid = p * area * zhat).
        y = jnp.zeros((1, n, 3))
        zero = jnp.zeros((1, n, 3))
        force = (ps * a_tot)[None, :, None] * jnp.array([0.0, 0.0, 1.0])[None, None, :]
        qn0 = jnp.zeros((1, n))
        _, p_dip = f1a_pressure(obs, y, zero, zero, qn0, force, med, tau, t_obs)
        pm = np.asarray(p_mesh[0])
        pd = np.asarray(p_dip[0])
        mask = np.abs(pd) > 0.02 * np.max(np.abs(pd))
        rel = np.linalg.norm((pm - pd)[mask]) / np.linalg.norm(pd[mask])
        assert rel < 0.02
