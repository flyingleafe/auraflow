"""Protocol v2 mesh streaming: scene meshes + per-frame pose animation."""

import numpy as np
import pytest

from auraflow.body.mesh import TriMesh
from auraflow.viz.body import body_scene, vertex_colors_from_face_scalar
from auraflow.viz.stream import (
    PROTOCOL_VERSION,
    decode_message,
    encode_frame,
    encode_scene,
)


class TestSceneMeshRoundTrip:
    def test_mesh_vertices_and_faces_round_trip(self):
        mesh = TriMesh.box((1.0, 2.0, 0.5))
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        msg = encode_scene(
            box_min=[-1, -1, -1],
            box_max=[1, 1, 1],
            meshes=[{"name": "cab", "vertices": verts, "faces": faces, "opacity": 0.6}],
        )
        h, a = decode_message(msg)
        assert h["v"] == PROTOCOL_VERSION == 2
        assert h["meshes"][0]["name"] == "cab"
        assert h["meshes"][0]["n_vertices"] == verts.shape[0]
        assert h["meshes"][0]["n_faces"] == faces.shape[0]
        assert h["meshes"][0]["opacity"] == pytest.approx(0.6)
        # Vertices float32, faces uint32 flattened (three.js index buffer).
        np.testing.assert_allclose(a["mesh0_vertices"], verts, atol=1e-6)
        assert a["mesh0_vertices"].shape == (verts.shape[0], 3)
        assert a["mesh0_faces"].dtype == np.uint32
        np.testing.assert_array_equal(a["mesh0_faces"], faces.reshape(-1))

    def test_mesh_vertex_colors_round_trip(self):
        mesh = TriMesh.sphere(1.0, subdivisions=1)
        colors = np.tile([0.2, 0.4, 0.6], (mesh.n_vertices, 1))
        msg = encode_scene(
            box_min=[-1, -1, -1],
            box_max=[1, 1, 1],
            meshes=[
                {
                    "vertices": np.asarray(mesh.vertices),
                    "faces": np.asarray(mesh.faces),
                    "colors": colors,
                }
            ],
        )
        h, a = decode_message(msg)
        assert h["meshes"][0]["has_colors"] is True
        assert a["mesh0_colors"].shape == (mesh.n_vertices, 3)
        np.testing.assert_allclose(a["mesh0_colors"], colors, atol=1e-6)

    def test_multiple_meshes(self):
        m0 = TriMesh.box()
        m1 = TriMesh.sphere(0.5, subdivisions=0)
        msg = encode_scene(
            box_min=[-1, -1, -1],
            box_max=[1, 1, 1],
            meshes=[
                {"vertices": np.asarray(m0.vertices), "faces": np.asarray(m0.faces)},
                {"vertices": np.asarray(m1.vertices), "faces": np.asarray(m1.faces)},
            ],
        )
        h, a = decode_message(msg)
        assert len(h["meshes"]) == 2
        assert a["mesh0_faces"].shape[0] == m0.n_faces * 3
        assert a["mesh1_faces"].shape[0] == m1.n_faces * 3


class TestFramePoseAnimation:
    def test_mesh_poses_round_trip(self):
        R = np.eye(3)
        pose = np.concatenate([[1.0, 2.0, 3.0], R.ravel()])
        msg = encode_frame(t=0.1, step=2, mesh_poses=pose[None, :])
        h, _ = decode_message(msg)
        assert len(h["mesh_poses"]) == 1
        assert h["mesh_poses"][0][:3] == [1.0, 2.0, 3.0]
        assert h["mesh_poses"][0][3:] == list(R.ravel())

    def test_multiple_mesh_poses(self):
        poses = np.zeros((3, 12))
        poses[:, 0] = [1.0, 2.0, 3.0]  # distinct x positions
        for k in range(3):
            poses[k, 3:] = np.eye(3).ravel()
        h, _ = decode_message(encode_frame(t=0.0, step=0, mesh_poses=poses))
        assert len(h["mesh_poses"]) == 3
        assert [row[0] for row in h["mesh_poses"]] == [1.0, 2.0, 3.0]


class TestBodySceneAdapter:
    def test_body_scene_bounds_and_mesh(self):
        from auraflow.body.motion import ConstantVelocity

        mesh = TriMesh.sphere(0.3, subdivisions=1)
        motion = ConstantVelocity((-5.0, 0.0, 2.0), (10.0, 0.0, 0.0))
        times = np.linspace(0.0, 1.0, 5)
        kwargs = encode_scene(**body_scene(mesh, motion, times, pad=1.0, title="fly"))
        h, a = decode_message(kwargs)
        assert h["title"] == "fly"
        assert len(h["meshes"]) == 1
        # Box brackets the swept trajectory: x spans roughly [-5, 5] +/- pad.
        assert h["box_min"][0] < -5.0 and h["box_max"][0] > 5.0
        assert a["mesh0_vertices"].shape[0] == mesh.n_vertices

    def test_vertex_colors_from_face_scalar_shape(self):
        mesh = TriMesh.sphere(1.0, subdivisions=1)
        face_scalar = np.linspace(0.0, 1.0, mesh.n_faces)
        colors = vertex_colors_from_face_scalar(mesh, face_scalar)
        assert colors.shape == (mesh.n_vertices, 3)
        assert colors.min() >= 0.0 and colors.max() <= 1.0
