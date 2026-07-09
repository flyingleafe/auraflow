"""Motion kinematics: exact velocities/accelerations via autodiff of the pose."""

import jax
import jax.numpy as jnp
import pytest

from auraflow.body import (
    ComposedMotion,
    ConstantVelocity,
    SpinMotion,
    StaticPose,
    SurfaceVibration,
    TriMesh,
    WaypointMotion,
    panel_histories,
    pose_derivatives,
)


class TestConstantVelocity:
    def test_velocity_exact_accel_zero(self):
        v = jnp.array([0.5, -1.0, 2.0])
        m = ConstantVelocity(x0=(1.0, 2.0, 3.0), v=v)
        _, x, _, dx, _, ddx = pose_derivatives(m, 0.7)
        assert jnp.allclose(x, jnp.array([1.0, 2.0, 3.0]) + v * 0.7, atol=1e-12)
        assert jnp.allclose(dx, v, atol=1e-12)
        assert jnp.allclose(ddx, 0.0, atol=1e-12)


class TestSpinMotion:
    def test_centripetal_exact(self):
        omega = 3.0
        m = SpinMotion.constant(axis=(0.0, 0.0, 1.0), omega=omega)
        r_body = jnp.array([2.0, 0.0, 0.5])  # r_perp = 2
        r_perp = 2.0
        R, x, dR, dx, ddR, ddx = pose_derivatives(m, 0.31)
        y = R @ r_body + x  # current world position
        v = dR @ r_body + dx
        a = ddR @ r_body + ddx
        assert float(jnp.linalg.norm(v)) == pytest.approx(omega * r_perp, abs=1e-10)
        assert float(jnp.linalg.norm(a)) == pytest.approx(omega**2 * r_perp, abs=1e-10)
        # Centripetal: a = -omega^2 * (perpendicular position vector from axis).
        perp = y.at[2].set(0.0)  # axis is +z through origin
        assert jnp.allclose(a, -(omega**2) * perp, atol=1e-9)
        # Velocity is perpendicular to the radial position (tangential).
        assert abs(float(jnp.sum(v * perp))) < 1e-9

    def test_from_azimuth_matches_constant(self):
        from auraflow.core.frames import integrate_azimuth

        t = jnp.linspace(0.0, 1.0, 65)
        omega = 2.0
        psi = integrate_azimuth(t, omega)
        m = SpinMotion.from_azimuth(axis=(0.0, 0.0, 1.0), t_grid=t, psi_grid=psi)
        r_body = jnp.array([1.5, 0.0, 0.0])
        _, _, dR, dx, _, _ = pose_derivatives(m, 0.5)
        v = dR @ r_body + dx
        assert float(jnp.linalg.norm(v)) == pytest.approx(omega * 1.5, rel=1e-6)


class TestComposed:
    def test_velocity_superposition_matches_jvp(self):
        base = ConstantVelocity(x0=(0.0, 0.0, 0.0), v=(1.0, 0.0, 0.0))
        spin = SpinMotion.constant(axis=(0.0, 0.0, 1.0), omega=4.0)
        comp = ComposedMotion(parent=base, child=spin)
        r_body = jnp.array([2.0, 0.0, 0.5])
        _, _, dR, dx, _, _ = pose_derivatives(comp, 0.4)
        v_kin = dR @ r_body + dx

        def world_pt(t):
            R, x = comp.pose(t)
            return R @ r_body + x

        _, v_jvp = jax.jvp(world_pt, (0.4,), (1.0,))
        assert jnp.allclose(v_kin, v_jvp, atol=1e-12)


class TestWaypoint:
    def test_passes_through_and_velocity_finite(self):
        times = jnp.array([0.0, 1.0, 2.0, 3.0])
        pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 1.0], [3.0, 2.0, 2.0]])
        m = WaypointMotion(times, pos)
        for j in range(4):
            _, x = m.pose(times[j])
            assert jnp.allclose(x, pos[j], atol=1e-12)
        # Velocity finite at every knot.
        for tj in times:
            _, _, _, dx, _, _ = pose_derivatives(m, tj)
            assert bool(jnp.all(jnp.isfinite(dx)))

    def test_headings_yaw(self):
        times = jnp.array([0.0, 1.0])
        pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        m = WaypointMotion(times, pos, headings=jnp.array([0.0, jnp.pi / 2]))
        R, _ = m.pose(1.0)
        # Yaw pi/2 maps body +x to world +y.
        assert jnp.allclose(R @ jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]), atol=1e-9)


class TestSurfaceVibration:
    def test_only_selected_faces_move(self):
        mesh = TriMesh.sphere(1.0, 1)
        tau = jnp.linspace(0.0, 1.0, 32)
        u_n = jnp.stack([jnp.sin(6.0 * tau), 0.5 * jnp.cos(3.0 * tau)])
        vib = SurfaceVibration(face_ids=[0, 5], t_grid=tau, u_n=u_n)
        ph = panel_histories(mesh, StaticPose(), tau, vibration=vib)
        # Static enclosure: only faces 0 and 5 have nonzero velocity.
        moving = jnp.max(jnp.abs(ph.v), axis=(1, 2))  # [F]
        assert float(moving[0]) > 0.1
        assert float(moving[5]) > 0.1
        others = jnp.concatenate([moving[1:5], moving[6:]])
        assert float(jnp.max(others)) < 1e-12
        # The velocity on face 0 is along its (world) normal.
        n0 = ph.n[0, 5]
        v0 = ph.v[0, 5]
        assert jnp.allclose(v0 / jnp.linalg.norm(v0), n0 / jnp.linalg.norm(n0), atol=1e-9)


class TestPanelHistories:
    def test_shapes_no_nan_and_grad(self):
        mesh = TriMesh.sphere(1.0, 1)
        motion = SpinMotion.constant(axis=(0.0, 0.0, 1.0), omega=2.0)
        tau = jnp.linspace(0.0, 1.0, 24)
        ph = panel_histories(mesh, motion, tau)
        f = mesh.n_faces
        for arr in (ph.y, ph.v, ph.a, ph.n):
            assert arr.shape == (f, 24, 3)
            assert not bool(jnp.any(jnp.isnan(arr)))
        assert ph.area.shape == (f,)
        # World normals stay unit.
        assert jnp.allclose(jnp.linalg.norm(ph.n, axis=-1), 1.0, atol=1e-9)

        faces = mesh.faces

        def scalar(verts):
            m = TriMesh(verts, faces, is_watertight=True)
            p = panel_histories(m, motion, tau)
            return jnp.sum(p.v**2) + jnp.sum(p.a**2) + jnp.sum(p.y**2)

        g = jax.grad(scalar)(mesh.vertices)
        assert g.shape == mesh.vertices.shape
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.linalg.norm(g)) > 0.0
