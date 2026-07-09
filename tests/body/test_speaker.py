"""Loudspeaker model API: construction, membrane selection, radiate/play, grad.

Physics gates for the baffled piston live in ``test_gates.py``; this file
exercises the :class:`Speaker` mechanics on small meshes.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.body import (
    Speaker,
    TriMesh,
    circular_piston,
    select_faces,
)
from auraflow.body.speaker import _disk_mesh
from auraflow.core.medium import Medium
from auraflow.signal.spectra import oaspl

MED = Medium()


class TestConstruction:
    def test_circular_piston_membrane_is_whole_disk(self):
        spk = circular_piston(radius=0.1, n=4)
        assert spk.baffled is True
        assert len(spk.membrane_faces) == spk.enclosure.n_faces
        # disk lies in z = 0 with +z normals
        np.testing.assert_allclose(np.asarray(spk.enclosure.normals())[:, 2], 1.0, atol=1e-12)

    def test_disk_mesh_area_converges(self):
        # Concentric-ring disk area -> pi r^2.
        disk = _disk_mesh(1.0, 12, 48)
        assert float(disk.total_area()) == pytest.approx(np.pi, rel=1e-2)

    def test_select_faces_predicate(self):
        box = TriMesh.box((1.0, 1.0, 1.0))
        # +x cabinet face: centroids at x ~ +0.5.
        ids = select_faces(box, lambda c: c[:, 0] > 0.4)
        assert len(ids) == 2  # two triangles per box face
        cents = np.asarray(box.centroids())[list(ids)]
        assert np.all(cents[:, 0] > 0.4)

    def test_from_mesh_with_predicate_and_ids(self):
        box = TriMesh.box((1.0, 1.0, 1.0))
        spk_pred = Speaker.from_mesh(box, lambda c: c[:, 2] > 0.4, baffled=True)
        ids = select_faces(box, lambda c: c[:, 2] > 0.4)
        spk_ids = Speaker.from_mesh(box, ids, baffled=True)
        assert spk_pred.membrane_faces == spk_ids.membrane_faces
        assert spk_pred.baffled is True


class TestRadiate:
    def _small_piston(self):
        return circular_piston(radius=0.05, n=4, baffled=False)

    def test_radiate_shapes_and_broadcast(self):
        spk = self._small_piston()
        T = 128
        omega = 2 * np.pi * 1500.0
        tau = jnp.linspace(0.0, 0.004, T)
        un = 0.01 * jnp.sin(omega * tau)
        obs = jnp.array([[0.0, 0.0, 0.3], [0.1, 0.0, 0.3]])
        p, t_obs = spk.radiate(un, tau, obs, MED)
        assert p.shape == (2, T)
        assert t_obs.shape == (T,)
        # Per-face signal [Fm, T] must give the same field as the shared [T] one.
        un_2d = jnp.broadcast_to(un[None, :], (len(spk.membrane_faces), T))
        p2, _ = spk.radiate(un_2d, tau, obs, MED)
        np.testing.assert_allclose(np.asarray(p2), np.asarray(p), rtol=1e-12)

    def test_baffled_doubles_pressure(self):
        base = circular_piston(radius=0.05, n=4, baffled=False)
        baff = Speaker(base.enclosure, base.membrane_faces, baffled=True)
        T = 128
        omega = 2 * np.pi * 1500.0
        tau = jnp.linspace(0.0, 0.004, T)
        un = 0.01 * jnp.sin(omega * tau)
        obs = jnp.array([[0.0, 0.0, 0.3]])
        p_free, _ = base.radiate(un, tau, obs, MED)
        p_baff, _ = baff.radiate(un, tau, obs, MED)
        np.testing.assert_allclose(np.asarray(p_baff), 2.0 * np.asarray(p_free), rtol=1e-12)


class TestPlay:
    def test_play_maps_audio_to_velocity(self):
        # play(audio, fs, gain) must equal radiate(gain*audio) on tau = n/fs.
        spk = circular_piston(radius=0.05, n=4, baffled=True)
        fs = 30000.0
        T = 128
        f = 1500.0
        audio = jnp.sin(2 * np.pi * f * jnp.arange(T) / fs) * 0.01
        obs = jnp.array([[0.0, 0.0, 0.3]])
        gain = 1.7
        p_play, _ = spk.play(audio, fs, obs, MED, gain=gain)
        tau = jnp.arange(T) / fs
        p_rad, _ = spk.radiate(gain * audio, tau, obs, MED)
        np.testing.assert_allclose(np.asarray(p_play), np.asarray(p_rad), rtol=1e-12)

    def test_grad_oaspl_wrt_gain_matches_fd(self):
        spk = circular_piston(radius=0.05, n=4, baffled=True)
        fs = 30000.0
        T = 128
        f = 1500.0
        audio = jnp.sin(2 * np.pi * f * jnp.arange(T) / fs) * 0.01
        obs = jnp.array([[0.0, 0.0, 0.6]])

        def oaspl_of_gain(g):
            p, _ = spk.play(audio, fs, obs, MED, gain=g)
            return oaspl(p[0])

        g0 = 1.5
        grad = jax.grad(oaspl_of_gain)(g0)
        assert bool(jnp.isfinite(grad))
        h = 1e-3
        fd = (oaspl_of_gain(g0 + h) - oaspl_of_gain(g0 - h)) / (2 * h)
        np.testing.assert_allclose(float(grad), float(fd), rtol=1e-3)
