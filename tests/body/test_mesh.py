"""TriMesh primitives: area/volume/normal invariants and convergence."""

import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.body import TriMesh


class TestSphere:
    def test_area_volume_converge(self):
        radius = 1.3
        exact_area = 4.0 * np.pi * radius**2
        exact_vol = 4.0 / 3.0 * np.pi * radius**3
        errs_a, errs_v = [], []
        for sd in (1, 2, 3):
            m = TriMesh.sphere(radius, sd)
            assert m.n_faces == 20 * 4**sd
            assert m.is_watertight
            errs_a.append(abs(float(m.total_area()) - exact_area))
            errs_v.append(abs(float(m.volume()) - exact_vol))
        # Converging with refinement, and close at the finest level.
        assert errs_a[0] > errs_a[1] > errs_a[2]
        assert errs_v[0] > errs_v[1] > errs_v[2]
        assert errs_a[-1] / exact_area < 0.02
        assert errs_v[-1] / exact_vol < 0.02

    def test_normals_outward(self):
        m = TriMesh.sphere(2.0, 2)
        # Outward: face normal aligns with centroid direction from the centre.
        dots = jnp.sum(m.normals() * m.centroids(), axis=-1)
        assert float(jnp.min(dots)) > 0.0
        assert jnp.allclose(jnp.linalg.norm(m.normals(), axis=-1), 1.0, atol=1e-12)

    def test_radius_differentiable(self):
        import jax

        g = jax.grad(lambda r: TriMesh.sphere(r, 1).total_area())(1.0)
        assert np.isfinite(float(g)) and float(g) > 0.0


class TestBox:
    def test_exact_area_and_volume(self):
        lx, ly, lz = 2.0, 3.0, 4.0
        m = TriMesh.box((lx, ly, lz))
        assert m.n_faces == 12
        assert m.is_watertight
        exact_area = 2.0 * (lx * ly + ly * lz + lx * lz)
        assert float(m.total_area()) == pytest.approx(exact_area, rel=1e-12)
        assert float(m.volume()) == pytest.approx(lx * ly * lz, rel=1e-12)

    def test_normals_outward(self):
        m = TriMesh.box((1.0, 1.0, 1.0))
        dots = jnp.sum(m.normals() * m.centroids(), axis=-1)
        assert float(jnp.min(dots)) > 0.0


class TestOpenPrimitives:
    def test_disk_area(self):
        r, n = 1.5, 128
        m = TriMesh.disk(r, n)
        assert not m.is_watertight
        # Inscribed-polygon area underestimates pi r^2; tightens with n.
        assert float(m.total_area()) == pytest.approx(np.pi * r**2, rel=2e-3)
        # All normals along +z (consistent single-sided surface).
        assert jnp.allclose(m.normals(), jnp.array([0.0, 0.0, 1.0]), atol=1e-12)

    def test_flat_plate_normals_consistent(self):
        m = TriMesh.flat_plate(2.0, 3.0)
        assert m.n_faces == 2
        assert not m.is_watertight
        assert float(m.total_area()) == pytest.approx(6.0, rel=1e-12)
        assert jnp.allclose(m.normals(), jnp.array([0.0, 0.0, 1.0]), atol=1e-12)


class TestCylinder:
    def test_area_volume_and_winding(self):
        r, h, n = 1.0, 2.0, 256
        m = TriMesh.cylinder(r, h, n)
        assert m.is_watertight
        exact_area = 2.0 * np.pi * r * h + 2.0 * np.pi * r**2
        assert float(m.total_area()) == pytest.approx(exact_area, rel=2e-3)
        assert float(m.volume()) == pytest.approx(np.pi * r**2 * h, rel=2e-3)
        # Outward winding => positive signed volume already checked; normals out.
        dots = jnp.sum(m.normals() * m.centroids(), axis=-1)
        assert float(jnp.min(dots)) > 0.0


def test_faces_are_static_hashable():
    # The mesh must be a valid static argument to jax.jit (hashable treedef).
    import jax

    m = TriMesh.box((1.0, 1.0, 1.0))
    assert isinstance(hash(jax.tree_util.tree_structure(m)), int)
