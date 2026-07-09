"""Signed-distance field: sign convention, node-exactness, unit gradient.

Grids are kept small (<= 17^3, icosphere subdivisions <= 2) so each test fits
the dev box's memory cap once the differentiable ``map_coordinates`` gather is
traced.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.body import TriMesh, sdf_eval

pytest.importorskip("trimesh")
pytest.importorskip("rtree")

from auraflow.body import sdf_grid  # noqa: E402

_LO = jnp.array([-2.0, -2.0, -2.0])
_HI = jnp.array([2.0, 2.0, 2.0])
_N = 17  # linspace(-2, 2, 17) has step 0.25, so 1.0 and 2.0 are exact nodes.


class TestSphereSDF:
    def test_sign_convention_and_surface(self):
        radius = 1.0
        mesh = TriMesh.sphere(radius, 2)
        grid = sdf_grid(mesh, _LO, _HI, _N)
        # Negative inside: at the centre distance ~ -r.
        center = float(sdf_eval(grid, _LO, _HI, jnp.array([0.0, 0.0, 0.0])))
        assert center == pytest.approx(-radius, abs=0.15)
        # Positive outside: at 2r the distance ~ +r.
        outside = float(sdf_eval(grid, _LO, _HI, jnp.array([2.0, 0.0, 0.0])))
        assert outside == pytest.approx(radius, abs=0.15)
        # Near zero on the surface.
        surf = float(sdf_eval(grid, _LO, _HI, jnp.array([radius, 0.0, 0.0])))
        assert abs(surf) < 0.15

    def test_matches_grid_at_nodes(self):
        mesh = TriMesh.sphere(1.0, 2)
        grid = sdf_grid(mesh, _LO, _HI, _N)
        xs = np.linspace(-2.0, 2.0, _N)
        for i, j, k in [(3, 8, 12), (0, 0, 0), (16, 16, 16), (10, 4, 7)]:
            node = jnp.array([xs[i], xs[j], xs[k]])
            assert float(sdf_eval(grid, _LO, _HI, node)) == pytest.approx(
                float(grid[i, j, k]), abs=1e-9
            )

    def test_gradient_unit_outward(self):
        mesh = TriMesh.sphere(1.0, 2)
        grid = sdf_grid(mesh, _LO, _HI, _N)
        p = jnp.array([1.5, 0.0, 0.0])  # well outside the surface, interior node cell
        g = jax.grad(lambda pt: sdf_eval(grid, _LO, _HI, pt))(p)
        assert bool(jnp.all(jnp.isfinite(g)))
        # |grad SDF| ~ 1 and points radially outward (+x here).
        assert float(jnp.linalg.norm(g)) == pytest.approx(1.0, abs=0.1)
        assert jnp.allclose(g / jnp.linalg.norm(g), jnp.array([1.0, 0.0, 0.0]), atol=0.15)

    def test_batched_points(self):
        mesh = TriMesh.sphere(1.0, 2)
        grid = sdf_grid(mesh, _LO, _HI, _N)
        pts = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        vals = sdf_eval(grid, _LO, _HI, pts)
        assert vals.shape == (2,)
        assert float(vals[0]) < 0.0 < float(vals[1])
