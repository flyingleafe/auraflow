"""Mesh import/export round-trips and winding repair (needs the ``mesh`` extra)."""

import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.body import TriMesh, load_mesh, save_mesh

trimesh = pytest.importorskip("trimesh")  # pyright: ignore[reportMissingImports]


class TestRoundTrip:
    def test_save_load_preserves_geometry(self, tmp_path):
        mesh = TriMesh.sphere(1.3, 2)
        path = tmp_path / "sphere.stl"
        save_mesh(mesh, str(path))
        loaded = load_mesh(str(path))
        assert loaded.n_faces == mesh.n_faces
        assert loaded.n_vertices == mesh.n_vertices
        assert float(loaded.total_area()) == pytest.approx(float(mesh.total_area()), rel=1e-6)
        # Vertex sets coincide (STL reorders/merges, so compare as point sets).
        v0 = np.sort(np.asarray(mesh.vertices), axis=0)
        v1 = np.sort(np.asarray(loaded.vertices), axis=0)
        assert np.allclose(v0, v1, atol=1e-6)

    def test_obj_roundtrip(self, tmp_path):
        mesh = TriMesh.box((1.0, 2.0, 3.0))
        path = tmp_path / "box.obj"
        save_mesh(mesh, str(path))
        loaded = load_mesh(str(path))
        assert float(loaded.volume()) == pytest.approx(6.0, rel=1e-6)


class TestImportMatchesPrimitive:
    def test_trimesh_sphere_matches_primitive(self, tmp_path):
        radius = 1.0
        tm = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        path = tmp_path / "tm_sphere.stl"
        tm.export(str(path))
        loaded = load_mesh(str(path))
        assert loaded.is_watertight
        assert float(loaded.total_area()) == pytest.approx(4.0 * np.pi * radius**2, rel=0.01)
        assert float(loaded.volume()) == pytest.approx(4.0 / 3.0 * np.pi * radius**3, rel=0.02)


class TestWindingRepair:
    def test_flipped_face_is_repaired(self, tmp_path):
        tm = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        faces = tm.faces.copy()
        faces[0] = faces[0][::-1]  # flip one triangle's winding
        bad = trimesh.Trimesh(vertices=tm.vertices, faces=faces, process=False)
        path = tmp_path / "bad.stl"
        bad.export(str(path))
        repaired = load_mesh(str(path), repair=True)
        # Consistent outward winding => positive signed volume.
        assert float(repaired.volume()) > 0.0
        assert repaired.is_watertight
        # Outward normals everywhere.
        dots = jnp.sum(repaired.normals() * repaired.centroids(), axis=-1)
        assert float(jnp.min(dots)) > 0.0
