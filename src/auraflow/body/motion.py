"""Rigid + deforming kinematics for general 3D bodies (``auraflow.body``).

See ``docs/architecture.md`` -> ``auraflow.body`` / ``motion.py``. A
:class:`Motion` maps source time to a world-from-body pose ``(R, x)`` such that
a body-frame point ``r`` sits at world position ``R @ r + x``. Velocities and
accelerations are obtained by **automatic differentiation** of the pose
(:func:`pose_derivatives`, nested :func:`jax.jvp`) -- never finite differences.

:func:`panel_histories` is the single kinematic entry point every acoustic
adapter consumes: it turns ``(mesh, motion, tau)`` into per-panel position /
velocity / acceleration / world-normal histories on a uniform source-time grid,
optionally superimposing a prescribed surface vibration (loudspeaker membrane).

Conventions (``docs/architecture.md``): SI units, right-handed world frame,
trailing ``xyz`` axis; angles in radians; ``R`` is a proper rotation
(body-to-world). Discretization counts are static; poses/velocities are traced.
"""

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from auraflow.body.mesh import TriMesh
from auraflow.core.frames import interp1d, rot_z

# A 3-vector accepted as an array or a plain float triple (user convenience).
Vec3 = ArrayLike | tuple[float, float, float]

__all__ = [
    "ComposedMotion",
    "ConstantVelocity",
    "Motion",
    "PanelHistories",
    "SpinMotion",
    "StaticPose",
    "SurfaceVibration",
    "WaypointMotion",
    "panel_histories",
    "pose_derivatives",
]


class Motion(eqx.Module):
    """Base class for world-from-body rigid poses.

    Subclasses implement :meth:`pose`. All parameters are traced (differentiable)
    equinox fields; any discretization structure is static.
    """

    def pose(self, t: ArrayLike) -> tuple[Array, Array]:
        """World-from-body pose at source time ``t``.

        Args:
            t: Source time [s], scalar.

        Returns:
            ``(R, x)`` with rotation ``R`` shape ``[3, 3]`` (body-to-world) and
            translation ``x`` shape ``[3]`` [m], so a body point ``r`` maps to
            world ``R @ r + x``.
        """
        raise NotImplementedError


def pose_derivatives(
    motion: Motion, t: ArrayLike
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Pose and its first two source-time derivatives via nested ``jax.jvp``.

    Exact autodiff of ``motion.pose`` -- no finite differences. Differentiable
    with respect to the motion parameters and ``t``.

    Args:
        motion: A :class:`Motion`.
        t: Source time [s], scalar.

    Returns:
        ``(R, x, dR, dx, ddR, ddx)`` where ``R, dR, ddR`` are ``[3, 3]`` (pose
        rotation and its d/dt, d2/dt2) and ``x, dx, ddx`` are ``[3]``
        (translation [m], velocity [m/s], acceleration [m/s^2]).
    """
    t = jnp.asarray(t, dtype=jnp.float64)

    def first(tt: Array) -> tuple[Array, Array, Array, Array]:
        (r, x), (dr, dx) = jax.jvp(motion.pose, (tt,), (jnp.ones_like(tt),))
        return r, x, dr, dx

    (r, x, dr, dx), (_, _, ddr, ddx) = jax.jvp(first, (t,), (jnp.ones_like(t),))
    return r, x, dr, dx, ddr, ddx


def _skew(a: Array) -> Array:
    """Skew-symmetric matrix ``K`` with ``K @ v == cross(a, v)``, shape ``[3, 3]``."""
    ax, ay, az = a[0], a[1], a[2]
    zero = jnp.zeros_like(ax)
    return jnp.stack(
        [
            jnp.stack([zero, -az, ay]),
            jnp.stack([az, zero, -ax]),
            jnp.stack([-ay, ax, zero]),
        ]
    )


def axis_angle_matrix(axis: ArrayLike, angle: ArrayLike) -> Array:
    """Rodrigues rotation about a (not necessarily unit) ``axis`` by ``angle``.

    Args:
        axis: Rotation axis, shape ``[3]`` (normalized internally).
        angle: Rotation angle [rad], scalar.

    Returns:
        Proper rotation matrix ``R`` shape ``[3, 3]``, active (right-handed
        about ``axis``). Differentiable in both arguments.
    """
    a = jnp.asarray(axis, dtype=jnp.float64)
    a = a / jnp.linalg.norm(a)
    k = _skew(a)
    angle = jnp.asarray(angle, dtype=jnp.float64)
    return jnp.eye(3) + jnp.sin(angle) * k + (1.0 - jnp.cos(angle)) * (k @ k)


class StaticPose(Motion):
    """A fixed pose (no motion): velocities and accelerations are zero.

    Attributes:
        R: Body-to-world rotation, shape ``[3, 3]``.
        x: World position of the body origin [m], shape ``[3]``.
    """

    R: Array
    x: Array

    def __init__(self, R: ArrayLike | None = None, x: Vec3 | None = None):
        """Args:
        R: Body-to-world rotation, shape ``[3, 3]`` (default identity).
        x: World origin position [m], shape ``[3]`` (default zero).
        """
        self.R = jnp.eye(3) if R is None else jnp.asarray(R, dtype=jnp.float64)
        self.x = jnp.zeros(3) if x is None else jnp.asarray(x, dtype=jnp.float64)

    def pose(self, t: ArrayLike) -> tuple[Array, Array]:  # noqa: ARG002
        return self.R, self.x


class ConstantVelocity(Motion):
    """Rigid translation at constant velocity with a fixed orientation.

    ``x(t) = x0 + v t``; ``R`` constant. Velocity is ``v`` exactly and
    acceleration is zero (used e.g. for a Doppler flyover of a rigid body).

    Attributes:
        x0: World position at ``t = 0`` [m], shape ``[3]``.
        v: World velocity [m/s], shape ``[3]``.
        R: Body-to-world rotation, shape ``[3, 3]``.
    """

    x0: Array
    v: Array
    R: Array

    def __init__(self, x0: Vec3, v: Vec3, R: ArrayLike | None = None):
        """Args:
        x0: World position at ``t = 0`` [m], shape ``[3]``.
        v: World velocity [m/s], shape ``[3]``.
        R: Body-to-world rotation, shape ``[3, 3]`` (default identity).
        """
        self.x0 = jnp.asarray(x0, dtype=jnp.float64)
        self.v = jnp.asarray(v, dtype=jnp.float64)
        self.R = jnp.eye(3) if R is None else jnp.asarray(R, dtype=jnp.float64)

    def pose(self, t: ArrayLike) -> tuple[Array, Array]:
        t = jnp.asarray(t, dtype=jnp.float64)
        return self.R, self.x0 + self.v * t


class SpinMotion(Motion):
    """Rigid rotation about a fixed world axis line through ``center``.

    The pose is ``R(t) = Q(psi(t)) @ R0`` and ``x(t) = (I - Q(psi(t))) @ center``,
    where ``Q`` is the Rodrigues rotation about ``axis``. A body point ``r`` then
    maps to ``center + Q(psi) @ (R0 @ r - center)``: it orbits the axis line, so
    a point at perpendicular distance ``d`` from the axis has speed ``|Omega| d``
    and (for constant ``Omega``) centripetal acceleration ``Omega^2 d``. This
    subsumes rotor spin (``center`` = hub, ``axis`` = thrust axis).

    Two construction modes (via classmethods):

    - :meth:`constant`: constant angular rate ``Omega``; ``psi = Omega t`` exact.
    - :meth:`from_azimuth`: precomputed ``psi(t)`` table (e.g. from
      :func:`auraflow.core.frames.integrate_azimuth` for time-varying ``Omega``),
      linearly interpolated -- the same pattern the rotor backends use.

    Attributes:
        axis: Rotation axis, shape ``[3]`` (normalized internally).
        center: A point on the axis line [m], shape ``[3]``.
        R0: Reference body-to-world rotation at ``psi = 0``, shape ``[3, 3]``.
        omega: Constant angular rate [rad/s], scalar, or ``None`` if tabulated.
        t_grid: Time grid [s], shape ``[T]``, or ``None`` for the constant case.
        psi_grid: Azimuth samples [rad], shape ``[T]``, or ``None``.
    """

    axis: Array
    center: Array
    R0: Array
    omega: Array | None
    t_grid: Array | None
    psi_grid: Array | None

    def __init__(
        self,
        axis: Vec3,
        center: Vec3 | None,
        R0: ArrayLike | None,
        omega: ArrayLike | None,
        t_grid: ArrayLike | None,
        psi_grid: ArrayLike | None,
    ):
        self.axis = jnp.asarray(axis, dtype=jnp.float64)
        self.center = jnp.zeros(3) if center is None else jnp.asarray(center, dtype=jnp.float64)
        self.R0 = jnp.eye(3) if R0 is None else jnp.asarray(R0, dtype=jnp.float64)
        self.omega = None if omega is None else jnp.asarray(omega, dtype=jnp.float64)
        self.t_grid = None if t_grid is None else jnp.asarray(t_grid, dtype=jnp.float64)
        self.psi_grid = None if psi_grid is None else jnp.asarray(psi_grid, dtype=jnp.float64)

    @classmethod
    def constant(
        cls,
        axis: Vec3,
        omega: ArrayLike,
        center: Vec3 | None = None,
        R0: ArrayLike | None = None,
    ) -> "SpinMotion":
        """Constant angular rate: ``psi(t) = Omega t`` (exact).

        Args:
            axis: Rotation axis, shape ``[3]``.
            omega: Angular rate [rad/s], scalar (signed, right-handed about axis).
            center: A point on the axis [m], shape ``[3]`` (default origin).
            R0: Reference rotation at ``t = 0``, shape ``[3, 3]`` (default I).
        """
        return cls(axis, center, R0, omega, None, None)

    @classmethod
    def from_azimuth(
        cls,
        axis: Vec3,
        t_grid: ArrayLike,
        psi_grid: ArrayLike,
        center: Vec3 | None = None,
        R0: ArrayLike | None = None,
    ) -> "SpinMotion":
        """Time-varying rate from a precomputed azimuth history.

        Args:
            axis: Rotation axis, shape ``[3]``.
            t_grid: Time grid [s], shape ``[T]``, strictly increasing.
            psi_grid: Azimuth samples [rad], shape ``[T]`` (e.g. from
                :func:`auraflow.core.frames.integrate_azimuth`).
            center: A point on the axis [m], shape ``[3]`` (default origin).
            R0: Reference rotation at ``psi = 0``, shape ``[3, 3]`` (default I).
        """
        return cls(axis, center, R0, None, t_grid, psi_grid)

    def _psi(self, t: Array) -> Array:
        if self.omega is not None:
            return self.omega * t
        assert self.t_grid is not None and self.psi_grid is not None
        return interp1d(t, self.t_grid, self.psi_grid)

    def pose(self, t: ArrayLike) -> tuple[Array, Array]:
        t = jnp.asarray(t, dtype=jnp.float64)
        q = axis_angle_matrix(self.axis, self._psi(t))
        r = q @ self.R0
        x = self.center - q @ self.center
        return r, x


class WaypointMotion(Motion):
    """Smooth trajectory through position waypoints (Catmull-Rom spline).

    Position is a C1 centripetal-uniform Catmull-Rom interpolation of the
    waypoints (passes through every waypoint, continuous velocity). Optional
    per-waypoint heading (yaw about world ``+z``) is linearly interpolated into
    ``R = rot_z(yaw)``; without headings the orientation is identity.

    Attributes:
        times: Waypoint times [s], shape ``[K]``, strictly increasing (static
            values, stored traced but treated as knots).
        positions: Waypoint world positions [m], shape ``[K, 3]``.
        headings: Waypoint yaw angles [rad], shape ``[K]``, or ``None``.
    """

    times: Array
    positions: Array
    headings: Array | None

    def __init__(self, times: ArrayLike, positions: ArrayLike, headings: ArrayLike | None = None):
        """Args:
        times: Waypoint times [s], shape ``[K]`` (K >= 2), strictly increasing.
        positions: Waypoint world positions [m], shape ``[K, 3]``.
        headings: Optional waypoint yaw angles [rad], shape ``[K]``.
        """
        self.times = jnp.asarray(times, dtype=jnp.float64)
        self.positions = jnp.asarray(positions, dtype=jnp.float64)
        self.headings = None if headings is None else jnp.asarray(headings, dtype=jnp.float64)
        if self.times.shape[0] < 2:
            raise ValueError("WaypointMotion needs at least 2 waypoints")

    def pose(self, t: ArrayLike) -> tuple[Array, Array]:
        t = jnp.asarray(t, dtype=jnp.float64)
        times = self.times
        pos = self.positions
        k = times.shape[0]
        # Segment index i in [0, K-2] with t in [times[i], times[i+1]].
        i = jnp.clip(jnp.searchsorted(times, t, side="right") - 1, 0, k - 2)
        t0 = times[i]
        t1 = times[i + 1]
        s = jnp.clip((t - t0) / (t1 - t0), 0.0, 1.0)
        p1 = pos[i]
        p2 = pos[i + 1]
        p0 = pos[jnp.clip(i - 1, 0, k - 1)]
        p3 = pos[jnp.clip(i + 2, 0, k - 1)]
        # Uniform Catmull-Rom basis.
        s2 = s * s
        s3 = s2 * s
        x = 0.5 * (
            2.0 * p1
            + (-p0 + p2) * s
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * s2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * s3
        )
        if self.headings is None:
            r = jnp.eye(3)
        else:
            yaw = interp1d(t, times, self.headings)
            r = rot_z(yaw)
        return r, x


class ComposedMotion(Motion):
    """Composition of a child motion inside a parent motion.

    ``pose = parent.pose ∘ child.pose``: with parent ``(Rp, xp)`` and child
    ``(Rc, xc)``, the world pose is ``R = Rp @ Rc`` and ``x = Rp @ xc + xp``.
    Velocity superposition follows automatically from
    :func:`pose_derivatives`. Use for a blade in a rotor in a vehicle, or a
    spinning body carried by a translating base.

    Attributes:
        parent: Outer (world-from-parent) :class:`Motion`.
        child: Inner (parent-from-child) :class:`Motion`.
    """

    parent: Motion
    child: Motion

    def pose(self, t: ArrayLike) -> tuple[Array, Array]:
        rp, xp = self.parent.pose(t)
        rc, xc = self.child.pose(t)
        return rp @ rc, rp @ xc + xp


class SurfaceVibration(eqx.Module):
    """Prescribed normal velocity on selected faces (loudspeaker membrane).

    A tabulated normal velocity ``u_n(face, t)`` on the source-time grid,
    superimposed on the rigid motion in :func:`panel_histories`. Off-grid times
    are linearly interpolated; the time derivative (for the acceleration term)
    is taken by autodiff of that interpolation.

    Attributes:
        face_ids: Indices of vibrating faces, ``[Fm]`` int (static topology,
            stored as a hashable tuple so the module is jit-static-safe).
        t_grid: Source-time grid [s], shape ``[T]``, strictly increasing.
        u_n: Prescribed outward normal velocity [m/s], shape ``[Fm, T]``
            (positive = along the outward normal).
    """

    face_ids: tuple[int, ...] = eqx.field(static=True)
    t_grid: Array
    u_n: Array

    def __init__(self, face_ids: ArrayLike | Sequence[int], t_grid: ArrayLike, u_n: ArrayLike):
        """Args:
        face_ids: Vibrating face indices, shape ``[Fm]``.
        t_grid: Source-time grid [s], shape ``[T]``.
        u_n: Normal velocity table [m/s], shape ``[Fm, T]``.
        """
        self.face_ids = tuple(int(i) for i in np.asarray(face_ids).ravel())
        self.t_grid = jnp.asarray(t_grid, dtype=jnp.float64)
        self.u_n = jnp.asarray(u_n, dtype=jnp.float64)

    def normal_velocity(self, t: ArrayLike) -> tuple[Array, Array]:
        """Normal velocity and its time derivative at ``t`` on the vibrating faces.

        Args:
            t: Source time [s], scalar.

        Returns:
            ``(u_n, du_n)`` each shape ``[Fm]`` -- normal velocity [m/s] and its
            time derivative [m/s^2].
        """
        t = jnp.asarray(t, dtype=jnp.float64)

        def eval_un(tt: Array) -> Array:
            return jax.vmap(lambda row: interp1d(tt, self.t_grid, row))(self.u_n)

        return jax.jvp(eval_un, (t,), (jnp.ones_like(t),))


class PanelHistories(eqx.Module):
    """Per-panel kinematic histories on a uniform source-time grid.

    The single contract every ``auraflow.body`` acoustic adapter consumes. All
    quantities are in the world frame, SI units, with the trailing ``xyz`` axis.

    Attributes:
        y: Panel centroid positions [m], shape ``[F, T, 3]``.
        v: Panel centroid velocities [m/s], shape ``[F, T, 3]`` (rigid-body
            point velocity plus any prescribed surface vibration ``u_n * n``).
        a: Panel centroid accelerations [m/s^2], shape ``[F, T, 3]``.
        n: Panel outward unit normals in world frame, shape ``[F, T, 3]``.
        area: Panel areas [m^2], shape ``[F]`` (time-invariant; rigid panels).
    """

    y: Array
    v: Array
    a: Array
    n: Array
    area: Array


def panel_histories(
    mesh: TriMesh,
    motion: Motion,
    tau: ArrayLike,
    vibration: SurfaceVibration | None = None,
) -> PanelHistories:
    """Kinematic histories of every panel over the source-time grid ``tau``.

    Rigid-body point kinematics of each face centroid ``r`` (body frame):
    ``y = R r + x``, ``v = dR r + dx``, ``a = ddR r + ddx``, with pose
    derivatives from :func:`pose_derivatives` (exact autodiff). Outward normals
    are rotated by ``R`` (unit-preserving). A :class:`SurfaceVibration` adds
    ``u_n * n`` to ``v`` and ``du_n * n`` to ``a`` on its selected faces only.

    Vectorized (``vmap``) over faces and times; differentiable through the mesh
    vertices and the motion parameters.

    Args:
        mesh: The body surface :class:`~auraflow.body.mesh.TriMesh`.
        motion: The rigid :class:`Motion`.
        tau: Uniform source-time grid [s], shape ``[T]``.
        vibration: Optional prescribed surface vibration.

    Returns:
        A :class:`PanelHistories` with ``y, v, a, n`` of shape ``[F, T, 3]`` and
        ``area`` of shape ``[F]``.
    """
    tau = jnp.asarray(tau, dtype=jnp.float64)
    r_body = mesh.centroids()  # [F, 3]
    n_body = mesh.normals()  # [F, 3]
    area = mesh.areas()  # [F]
    n_faces = r_body.shape[0]
    fids = None if vibration is None else jnp.asarray(vibration.face_ids)

    def at_time(t: Array) -> tuple[Array, Array, Array, Array]:
        r, x, dr, dx, ddr, ddx = pose_derivatives(motion, t)
        y = r_body @ r.T + x  # [F, 3]
        v = r_body @ dr.T + dx
        a = r_body @ ddr.T + ddx
        n = n_body @ r.T  # rotate normals (R orthonormal -> stays unit)
        if vibration is not None:
            un, dun = vibration.normal_velocity(t)  # [Fm]
            un_full = jnp.zeros(n_faces).at[fids].set(un)
            dun_full = jnp.zeros(n_faces).at[fids].set(dun)
            v = v + un_full[:, None] * n
            a = a + dun_full[:, None] * n
        return y, v, a, n

    ys, vs, as_, ns = jax.vmap(at_time)(tau)  # each [T, F, 3]
    swap = lambda arr: jnp.swapaxes(arr, 0, 1)  # noqa: E731  -> [F, T, 3]
    return PanelHistories(y=swap(ys), v=swap(vs), a=swap(as_), n=swap(ns), area=area)
