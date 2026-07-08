"""Permeable-sphere geometry and grid interpolation (no jaxfluids needed)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.cfd.sphere import (
    PermeableSphere,
    fibonacci_sphere,
    sample_primitives,
    trilinear_interpolate,
)


class TestFibonacciSphere:
    def test_points_lie_on_sphere(self):
        radius, center = 2.0, jnp.array([1.0, -0.5, 0.3])
        points, normals, area = fibonacci_sphere(500, radius, center)
        r = jnp.linalg.norm(points - center, axis=-1)
        assert jnp.allclose(r, radius, atol=1e-6)
        # normals are unit and outward (aligned with the radial direction).
        assert jnp.allclose(jnp.linalg.norm(normals, axis=-1), 1.0, atol=1e-6)
        radial = (points - center) / radius
        assert jnp.all(jnp.sum(normals * radial, axis=-1) > 0.999)

    def test_area_sums_to_sphere_area(self):
        radius = 1.7
        _, _, area = fibonacci_sphere(1000, radius)
        assert float(jnp.sum(area)) == pytest.approx(4.0 * np.pi * radius**2, rel=1e-6)

    def test_centroid_near_center(self):
        center = jnp.array([0.4, 0.0, -1.2])
        points, _, _ = fibonacci_sphere(2000, 1.0, center)
        centroid = jnp.mean(points, axis=0)
        assert jnp.allclose(centroid, center, atol=2e-2)

    def test_module_roundtrip(self):
        sph = PermeableSphere.fibonacci(128, radius=0.5, center=(0.0, 0.0, 0.0))
        assert sph.n_points == 128
        assert sph.points.shape == (128, 3)
        assert sph.normals.shape == (128, 3)
        assert sph.area.shape == (128,)
        assert float(jnp.sum(sph.area)) == pytest.approx(4.0 * np.pi * 0.25, rel=1e-6)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            fibonacci_sphere(0)


class TestTrilinearInterpolation:
    def _grid(self, n=17, lo=-1.0, hi=1.0):
        ax = jnp.linspace(lo, hi, n)
        return ax

    def test_exact_for_linear_field(self):
        x = self._grid()
        y = self._grid()
        z = self._grid()
        gx, gy, gz = jnp.meshgrid(x, y, z, indexing="ij")
        a, b, c, d = 0.7, -1.3, 2.1, 0.4
        field = a * gx + b * gy + c * gz + d
        key = jax.random.PRNGKey(0)
        pts = jax.random.uniform(key, (50, 3), minval=-0.9, maxval=0.9)
        got = trilinear_interpolate(field, x, y, z, pts)
        want = a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d
        assert jnp.allclose(got, want, atol=1e-6)

    def test_multichannel(self):
        x = self._grid()
        gx, gy, gz = jnp.meshgrid(x, x, x, indexing="ij")
        field = jnp.stack([gx, gy, gz, gx + gz], axis=-1)  # [N,N,N,4]
        pts = jnp.array([[0.1, 0.2, -0.3], [-0.5, 0.5, 0.0]])
        got = trilinear_interpolate(field, x, x, x, pts)
        assert got.shape == (2, 4)
        assert jnp.allclose(got[:, 0], pts[:, 0], atol=1e-6)
        assert jnp.allclose(got[:, 3], pts[:, 0] + pts[:, 2], atol=1e-6)

    def test_gradient_finite(self):
        x = self._grid()
        gx, gy, gz = jnp.meshgrid(x, x, x, indexing="ij")
        field = jnp.sin(gx) * jnp.cos(gy) + gz

        def scalar(pts):
            return jnp.sum(trilinear_interpolate(field, x, x, x, pts))

        pts = jnp.array([[0.11, -0.22, 0.33], [0.4, 0.1, -0.2]])
        g = jax.grad(scalar)(pts)
        assert g.shape == pts.shape
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) > 0.0

    def test_sample_primitives_splits_channels(self):
        x = self._grid()
        gx, gy, gz = jnp.meshgrid(x, x, x, indexing="ij")
        ones = jnp.ones_like(gx)
        # primitives ordered (rho, u, v, w, p)
        primitives = jnp.stack([1.2 * ones, gx, gy, gz, 5.0 * ones], axis=0)
        pts = jnp.array([[0.1, -0.2, 0.3]])
        rho, u, p = sample_primitives(primitives, x, x, x, pts)
        assert rho.shape == (1,)
        assert u.shape == (1, 3)
        assert p.shape == (1,)
        assert float(rho[0]) == pytest.approx(1.2, abs=1e-6)
        assert float(p[0]) == pytest.approx(5.0, abs=1e-6)
        assert jnp.allclose(u[0], jnp.array([0.1, -0.2, 0.3]), atol=1e-6)
