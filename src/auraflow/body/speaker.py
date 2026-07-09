"""Loudspeaker model: a vibrating membrane on a static rigid enclosure.

A :class:`Speaker` is the degenerate-motion FW-H body (``docs/architecture.md``
-> "Generalization principle"): a static :class:`StaticPose` enclosure whose
selected *membrane* faces carry a prescribed normal-velocity signal ``u_n(t)``
(e.g. an audio waveform). The membrane radiates as thickness (monopole) sources
through :func:`auraflow.body.mesh_pressure`; rigid-enclosure scattering is
neglected -- exact in the free-field (no enclosure) and baffled limits used by
the analytic gates.

Two idealizations, both documented on the methods that make them:

- **Baffled piston** (``baffled=True``): an infinite rigid baffle in the
  membrane plane is modelled by an image source, i.e. simply doubling the
  radiated pressure. Valid for listeners on the source side of the baffle plane.
- **Audio playback** (:meth:`Speaker.play`): the audio waveform is mapped
  directly to membrane normal velocity ``u_n = gain * audio`` on the speaker's
  own sample grid. There is no electroacoustic transducer / enclosure-compliance
  model -- the cone velocity *is* the input signal.

Frames/units: SI, world frame, trailing ``xyz`` axis. Membrane normal velocity
is positive along the outward face normal.
"""

from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from auraflow.body.mesh import TriMesh
from auraflow.body.motion import StaticPose, SurfaceVibration
from auraflow.body.sources import mesh_pressure
from auraflow.core.medium import Medium

__all__ = ["Speaker", "circular_piston", "select_faces"]


def _disk_mesh(radius: ArrayLike, n_rings: int, n_theta: int) -> TriMesh:
    """Concentric-ring triangulation of a flat disk in ``z = 0`` (``+z`` normals).

    Unlike the single-ring fan of :meth:`TriMesh.disk`, this resolves the radial
    coordinate (``n_rings`` rings x ``n_theta`` sectors) so each panel stays much
    smaller than a wavelength -- required for the piston directivity/Rayleigh
    gates, whose accuracy depends on the phase variation across the membrane.

    Args:
        radius: Disk radius [m], scalar (traced, differentiable).
        n_rings: Number of radial rings (static int ``>= 1``).
        n_theta: Number of angular sectors (static int ``>= 3``).

    Returns:
        An open (not watertight) :class:`TriMesh` with consistent ``+z`` normals.
    """
    if n_rings < 1:
        raise ValueError(f"n_rings must be >= 1, got {n_rings}")
    if n_theta < 3:
        raise ValueError(f"n_theta must be >= 3, got {n_theta}")
    theta = np.arange(n_theta) * (2.0 * np.pi / n_theta)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    verts = [np.zeros(3)]
    for i in range(1, n_rings + 1):
        rr = i / n_rings
        for j in range(n_theta):
            verts.append(np.array([rr * cos_t[j], rr * sin_t[j], 0.0]))
    faces: list[tuple[int, int, int]] = []
    # Centre fan to the first ring (indices 1 .. n_theta).
    for j in range(n_theta):
        faces.append((0, 1 + j, 1 + (j + 1) % n_theta))
    # Quad strips between consecutive rings, split into two CCW triangles.
    for i in range(1, n_rings):
        b0 = 1 + (i - 1) * n_theta
        b1 = 1 + i * n_theta
        for j in range(n_theta):
            j1 = (j + 1) % n_theta
            faces.append((b0 + j, b1 + j, b1 + j1))
            faces.append((b0 + j, b1 + j1, b0 + j1))
    verts_arr = jnp.asarray(radius, dtype=jnp.float64) * jnp.asarray(np.stack(verts))
    return TriMesh(verts_arr, np.asarray(faces, dtype=np.int64), is_watertight=False)


def select_faces(mesh: TriMesh, predicate: Callable[[Array], Array]) -> tuple[int, ...]:
    """Face ids whose centroid satisfies ``predicate`` (membrane selection helper).

    Args:
        mesh: The :class:`TriMesh`.
        predicate: Maps face centroids ``[F, 3]`` -> boolean mask ``[F]`` (e.g.
            ``lambda c: c[:, 0] > 0.09`` for the ``+x`` cabinet face). Evaluated
            once at setup; may use jnp on the (concrete) centroid array.

    Returns:
        A tuple of selected face indices (static, for :class:`SurfaceVibration`).
    """
    mask = np.asarray(predicate(mesh.centroids()))
    return tuple(int(i) for i in np.nonzero(mask)[0])


class Speaker(eqx.Module):
    """A loudspeaker: prescribed-velocity membrane faces on a static enclosure.

    Attributes:
        enclosure: The cabinet/membrane :class:`TriMesh` (the full radiating
            surface; for a bare piston this is just the disk).
        membrane_faces: Indices of the vibrating membrane faces (static topology,
            stored as a hashable tuple).
        baffled: If ``True``, model an infinite rigid baffle in the membrane
            plane by doubling the radiated pressure (image source).
    """

    enclosure: TriMesh
    membrane_faces: tuple[int, ...] = eqx.field(static=True)
    baffled: bool = eqx.field(static=True)

    def __init__(
        self,
        enclosure: TriMesh,
        membrane_faces: Sequence[int] | ArrayLike,
        *,
        baffled: bool = False,
    ):
        """Args:
        enclosure: Radiating surface :class:`TriMesh`.
        membrane_faces: Vibrating face indices.
        baffled: Rigid-baffle image doubling (default ``False``).
        """
        self.enclosure = enclosure
        self.membrane_faces = tuple(int(i) for i in np.asarray(membrane_faces).ravel())
        self.baffled = bool(baffled)

    @classmethod
    def circular_piston(
        cls,
        radius: ArrayLike = 0.1,
        n: int = 8,
        *,
        baffled: bool = True,
        n_theta: int | None = None,
    ) -> "Speaker":
        """A flat circular piston (disk membrane, no enclosure) -- the piston gate.

        The whole disk is the membrane, so every face vibrates in phase with the
        prescribed ``u_n(t)``. Defaults to ``baffled=True`` (the canonical
        Rayleigh baffled-piston object).

        Args:
            radius: Piston radius [m], scalar (traced).
            n: Radial ring count of the disk triangulation (static int).
            baffled: Rigid-baffle image doubling (default ``True``).
            n_theta: Angular sector count (default ``4 * n``).

        Returns:
            A :class:`Speaker` whose enclosure is the disk and whose membrane is
            every face of it.
        """
        n_theta = 4 * n if n_theta is None else n_theta
        disk = _disk_mesh(radius, n, n_theta)
        return cls(disk, tuple(range(disk.n_faces)), baffled=baffled)

    @classmethod
    def from_mesh(
        cls,
        mesh: TriMesh,
        membrane_face_ids: Sequence[int] | ArrayLike | Callable[[Array], Array],
        *,
        baffled: bool = False,
    ) -> "Speaker":
        """Build a speaker from an imported cabinet mesh.

        Args:
            mesh: The enclosure :class:`TriMesh` (e.g. an imported cabinet STL).
            membrane_face_ids: Either explicit face indices, or a predicate on
                face centroids ``[F, 3] -> [F]`` bool (passed to
                :func:`select_faces`, e.g. all faces with a ``+x`` centroid).
            baffled: Rigid-baffle image doubling (default ``False``).

        Returns:
            A :class:`Speaker` with the selected membrane faces.
        """
        if callable(membrane_face_ids):
            ids = select_faces(mesh, membrane_face_ids)
        else:
            ids = tuple(int(i) for i in np.asarray(membrane_face_ids).ravel())
        return cls(mesh, ids, baffled=baffled)

    def radiate(
        self,
        u_n: ArrayLike,
        tau: ArrayLike,
        listeners: ArrayLike,
        medium: Medium,
    ) -> tuple[Array, Array]:
        """Radiate a prescribed membrane normal velocity to the listeners.

        Assembles a :class:`SurfaceVibration` on the membrane faces over a
        :class:`StaticPose` enclosure and calls :func:`mesh_pressure` for
        thickness-only radiation. When :attr:`baffled` the pressure is doubled
        (rigid-baffle image; valid for listeners on the source side of the
        membrane plane).

        Args:
            u_n: Membrane outward normal velocity [m/s]. Shape ``[T]`` (one signal
                shared by every membrane face) or ``[Fm, T]`` (per-face signals),
                sampled on ``tau``.
            tau: Uniform source-time grid [s], shape ``[T]``.
            listeners: Listener positions [m], shape ``[O, 3]``.
            medium: Ambient :class:`Medium`.

        Returns:
            ``(p, t_obs)``: radiated pressure [Pa], shape ``[O, T_obs]``, and the
            observer-time grid [s], shape ``[T_obs]``.
        """
        tau = jnp.asarray(tau, dtype=jnp.float64)
        u_n = jnp.asarray(u_n, dtype=jnp.float64)
        n_membrane = len(self.membrane_faces)
        if u_n.ndim == 1:
            u_n = jnp.broadcast_to(u_n[None, :], (n_membrane, tau.shape[0]))
        vibration = SurfaceVibration(self.membrane_faces, tau, u_n)
        p, t_obs = mesh_pressure(
            self.enclosure,
            StaticPose(),
            tau,
            jnp.asarray(listeners, dtype=jnp.float64),
            medium,
            vibration=vibration,
        )
        if self.baffled:
            p = 2.0 * p
        return p, t_obs

    def play(
        self,
        audio: ArrayLike,
        fs: float,
        listeners: ArrayLike,
        medium: Medium,
        *,
        gain: float = 1.0,
    ) -> tuple[Array, Array]:
        """Play an audio waveform as membrane velocity and record it at listeners.

        Convenience wrapper mapping a sampled waveform directly to membrane
        normal velocity ``u_n = gain * audio`` on the speaker's own sample grid
        ``tau = arange(len(audio)) / fs``, then :meth:`radiate`. This is an
        idealization: the waveform *is* the cone velocity (units m/s after the
        gain); there is no electroacoustic transducer or enclosure-compliance
        model between the electrical signal and the cone.

        Args:
            audio: Waveform samples, shape ``[T_a]`` (interpreted as cone
                velocity in m/s after ``gain``).
            fs: Sample rate [Hz].
            listeners: Listener positions [m], shape ``[O, 3]``.
            medium: Ambient :class:`Medium`.
            gain: Scalar velocity gain [m/s per unit sample] (default ``1.0``).

        Returns:
            ``(p, t_obs)``: radiated pressure [Pa], shape ``[O, T_obs]``, and the
            observer-time grid [s], shape ``[T_obs]``.
        """
        audio = jnp.asarray(audio, dtype=jnp.float64)
        tau = jnp.arange(audio.shape[0], dtype=jnp.float64) / fs
        return self.radiate(gain * audio, tau, listeners, medium)


def circular_piston(radius: ArrayLike = 0.1, n: int = 8, *, baffled: bool = True) -> Speaker:
    """Free function alias for :meth:`Speaker.circular_piston` (validation object)."""
    return Speaker.circular_piston(radius, n, baffled=baffled)
