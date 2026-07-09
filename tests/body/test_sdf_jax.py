"""GPU brute-force SDF (``sdf_grid_jax``) + generalized winding number.

Cross-validates the pure-JAX signed-distance build against the exact
``trimesh.proximity`` path, and exercises the winding-number sign on the cases
that motivate it (a thin flat plate, a nonconvex two-sphere union). Grids and
meshes are kept tiny and ``batch_points`` small so each test fits the dev box's
memory cap.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.body import TriMesh, sdf_eval, sdf_grid_jax, winding_number

pytest.importorskip("trimesh")
pytest.importorskip("rtree")

from auraflow.body import sdf_grid  # noqa: E402

_LO = np.array([-2.0, -2.0, -2.0])
_HI = np.array([2.0, 2.0, 2.0])
_N = 13
_BP = 256  # small point-chunk for the dev box


def _cell_diag(lo, hi, n):
    return float(np.linalg.norm((hi - lo) / (n - 1)))


class TestVsTrimesh:
    @pytest.mark.parametrize(
        "mesh",
        [TriMesh.sphere(1.0, 2), TriMesh.box((1.5, 1.0, 2.0))],
        ids=["sphere", "box"],
    )
    def test_matches_trimesh(self, mesh):
        gj = np.asarray(sdf_grid_jax(mesh, _LO, _HI, _N, batch_points=_BP))
        gt = np.asarray(sdf_grid(mesh, _LO, _HI, _N, method="trimesh"))
        # Both are the *exact* Euclidean distance to the same triangulation, so
        # the magnitudes agree to machine precision.
        assert np.abs(np.abs(gj) - np.abs(gt)).max() < 1e-9
        # The sign is identical everywhere except possibly within one cell of the
        # surface (where winding-number and ray-parity can disagree on a node
        # that straddles a facet).
        mism = np.sign(gj) != np.sign(gt)
        assert np.all(np.abs(gt)[mism] < _cell_diag(_LO, _HI, _N))

    def test_negative_inside(self):
        mesh = TriMesh.sphere(1.0, 2)
        g = sdf_grid_jax(mesh, _LO, _HI, _N, batch_points=_BP)
        assert float(sdf_eval(g, _LO, _HI, jnp.array([0.0, 0.0, 0.0]))) < 0.0
        assert float(sdf_eval(g, _LO, _HI, jnp.array([2.0, 0.0, 0.0]))) > 0.0


class TestWindingNumber:
    def test_inside_outside_icosphere(self):
        mesh = TriMesh.sphere(1.0, 1)  # coarse (80 faces)
        w_in = float(winding_number(mesh, jnp.array([0.0, 0.0, 0.0]), batch_points=_BP))
        w_out = float(winding_number(mesh, jnp.array([5.0, 0.0, 0.0]), batch_points=_BP))
        assert w_in == pytest.approx(1.0, abs=1e-6)
        assert w_out == pytest.approx(0.0, abs=1e-6)

    def test_thin_flat_plate_sign_discrimination(self):
        # A single-sheet flat plate (z=0, +z normal): ray-parity is hopeless, but
        # the generalized winding number is cleanly antisymmetric across the
        # sheet -- opposite signs just above vs just below, straddling zero. This
        # is exactly the thin-body discrimination the winding sign buys us for
        # (watertight but razor-thin) rotor blades.
        plate = TriMesh.flat_plate(chord=1.0, span=1.0)
        w_above = float(winding_number(plate, jnp.array([0.0, 0.0, 0.05]), batch_points=64))
        w_below = float(winding_number(plate, jnp.array([0.0, 0.0, -0.05]), batch_points=64))
        assert w_above < 0.0 < w_below
        assert w_above == pytest.approx(-w_below, abs=1e-9)
        # The unsigned distance to the sheet is still exact (~|z|).
        lo = np.array([-1.0, -1.0, -0.5])
        hi = np.array([1.0, 1.0, 0.5])
        g = sdf_grid_jax(plate, lo, hi, (9, 9, 9), batch_points=64)
        d = float(abs(sdf_eval(g, lo, hi, jnp.array([0.0, 0.0, 0.25]))))
        assert d == pytest.approx(0.25, abs=1e-6)


class TestNonconvex:
    def test_two_disjoint_spheres(self):
        # Union of two separated spheres (merge): a point inside either sphere is
        # inside the body; a point in the gap between them is outside. A convex
        # or single-winding assumption would get the gap wrong.
        s1 = TriMesh.sphere(0.5, 1)
        s2 = TriMesh(vertices=s1.vertices + jnp.array([3.0, 0.0, 0.0]), faces=s1.faces)
        two = TriMesh.merge([s1, s2])
        lo = np.array([-1.0, -1.0, -1.0])
        hi = np.array([4.0, 1.0, 1.0])
        g = sdf_grid_jax(two, lo, hi, (22, 8, 8), batch_points=128)
        assert float(sdf_eval(g, lo, hi, jnp.array([0.0, 0.0, 0.0]))) < 0.0  # in s1
        assert float(sdf_eval(g, lo, hi, jnp.array([3.0, 0.0, 0.0]))) < 0.0  # in s2
        assert float(sdf_eval(g, lo, hi, jnp.array([1.5, 0.0, 0.0]))) > 0.0  # gap


class TestGradient:
    def test_unit_radial_gradient(self):
        mesh = TriMesh.sphere(1.0, 2)
        # linspace(-2, 2, 17) has step 0.25, so 1.5 is an exact node (clean
        # radial gradient); a coarser grid smears it toward neighbouring cells.
        g = sdf_grid_jax(mesh, _LO, _HI, 17, batch_points=_BP)
        p = jnp.array([1.5, 0.0, 0.0])
        grad = jax.grad(lambda pt: sdf_eval(g, _LO, _HI, pt))(p)
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.linalg.norm(grad)) == pytest.approx(1.0, abs=0.1)
        assert jnp.allclose(grad / jnp.linalg.norm(grad), jnp.array([1.0, 0.0, 0.0]), atol=0.15)
