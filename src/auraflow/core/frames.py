"""Reference frames, rotation matrices, and azimuth kinematics.

Conventions (see ``docs/architecture.md``):

- Right-handed frames, angles in radians, SI units throughout.
- *World frame*: z up. *Rotor frame*: origin at hub, z along thrust axis,
  blade azimuth ``psi`` measured from +x toward +y (counterclockwise seen
  from +z).
- Rotation matrices are *active* rotations: ``rot_z(psi) @ v`` rotates the
  vector ``v`` by ``psi`` about +z. Equivalently, ``rot_z(psi)`` maps
  coordinates from a frame rotated by ``psi`` back to the base frame.

All functions accept scalar or batched angle arrays of shape ``[...]`` and
return matrices of shape ``[..., 3, 3]``. Everything is differentiable and
safe to use inside ``jit``/``grad``.
"""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

__all__ = [
    "azimuth_at",
    "euler_zyx_matrix",
    "integrate_azimuth",
    "interp1d",
    "rot_x",
    "rot_y",
    "rot_z",
]


def rot_x(angle: ArrayLike) -> Array:
    """Rotation matrix about the +x axis.

    Args:
        angle: Rotation angle(s) [rad], shape ``[...]`` (scalar allowed).

    Returns:
        Rotation matrices, shape ``[..., 3, 3]``:
        ``[[1, 0, 0], [0, c, -s], [0, s, c]]``.
    """
    a = jnp.asarray(angle)
    c, s = jnp.cos(a), jnp.sin(a)
    z, o = jnp.zeros_like(c), jnp.ones_like(c)
    return jnp.stack(
        [
            jnp.stack([o, z, z], axis=-1),
            jnp.stack([z, c, -s], axis=-1),
            jnp.stack([z, s, c], axis=-1),
        ],
        axis=-2,
    )


def rot_y(angle: ArrayLike) -> Array:
    """Rotation matrix about the +y axis.

    Args:
        angle: Rotation angle(s) [rad], shape ``[...]`` (scalar allowed).

    Returns:
        Rotation matrices, shape ``[..., 3, 3]``:
        ``[[c, 0, s], [0, 1, 0], [-s, 0, c]]``.
    """
    a = jnp.asarray(angle)
    c, s = jnp.cos(a), jnp.sin(a)
    z, o = jnp.zeros_like(c), jnp.ones_like(c)
    return jnp.stack(
        [
            jnp.stack([c, z, s], axis=-1),
            jnp.stack([z, o, z], axis=-1),
            jnp.stack([-s, z, c], axis=-1),
        ],
        axis=-2,
    )


def rot_z(psi: ArrayLike) -> Array:
    """Rotation matrix about the +z axis.

    ``rot_z(psi) @ [1, 0, 0] == [cos(psi), sin(psi), 0]``: a positive angle
    rotates +x toward +y (counterclockwise seen from +z), matching the blade
    azimuth convention.

    Args:
        psi: Rotation angle(s) [rad], shape ``[...]`` (scalar allowed).

    Returns:
        Rotation matrices, shape ``[..., 3, 3]``:
        ``[[c, -s, 0], [s, c, 0], [0, 0, 1]]``.
    """
    a = jnp.asarray(psi)
    c, s = jnp.cos(a), jnp.sin(a)
    z, o = jnp.zeros_like(c), jnp.ones_like(c)
    return jnp.stack(
        [
            jnp.stack([c, -s, z], axis=-1),
            jnp.stack([s, c, z], axis=-1),
            jnp.stack([z, z, o], axis=-1),
        ],
        axis=-2,
    )


def euler_zyx_matrix(roll: ArrayLike, pitch: ArrayLike, yaw: ArrayLike) -> Array:
    """Body-to-world rotation matrix from intrinsic z-y'-x'' (yaw-pitch-roll) Euler angles.

    The body frame is obtained from the world frame by yawing about z, then
    pitching about the new y, then rolling about the new x. The returned
    matrix ``R`` maps body-frame coordinates to world-frame coordinates:
    ``v_world = R @ v_body``, with ``R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)``.

    Args:
        roll: Roll angle(s) about body x [rad], shape broadcastable with the others.
        pitch: Pitch angle(s) about body y [rad].
        yaw: Yaw angle(s) about body z [rad].

    Returns:
        Rotation matrices, shape ``[..., 3, 3]`` where ``[...]`` is the
        broadcast shape of the three angle arrays.
    """
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def interp1d(xq: ArrayLike, x: ArrayLike, y: ArrayLike) -> Array:
    """Differentiable piecewise-linear interpolation on a 1-D grid.

    Thin wrapper over :func:`jnp.interp` with the semantics used throughout
    AuraFlow:

    - ``x`` must be strictly increasing; ``y`` are the sample values.
    - Queries outside ``[x[0], x[-1]]`` are clamped to the end values
      (constant extrapolation). **Gradient dead zone**: outside the grid the
      derivative with respect to ``xq`` is exactly zero (the value is
      constant there); gradients with respect to ``y`` remain well defined
      everywhere.
    - Differentiable with respect to both ``xq`` and ``y`` (not ``x``).

    Args:
        xq: Query points, any shape ``[...]``. Same units as ``x``.
        x: Grid nodes, shape ``[N]``, strictly increasing.
        y: Sample values at the nodes, shape ``[N]``.

    Returns:
        Interpolated values, shape ``[...]`` (same shape as ``xq``).
    """
    return jnp.interp(jnp.asarray(xq), jnp.asarray(x), jnp.asarray(y))


def integrate_azimuth(t: ArrayLike, omega: ArrayLike, psi0: ArrayLike = 0.0) -> Array:
    """Integrate an angular-rate history into azimuth samples, ``psi(t) = psi0 + int_0^t Omega dt``.

    Uses a cumulative trapezoid rule, which is *exact* for constant and
    linearly varying ``Omega(t)`` and second-order accurate otherwise. The
    result is differentiable with respect to ``omega`` (and ``t``, ``psi0``).

    This replaces the predecessor's incorrect ``psi = Omega(t) * t`` shortcut
    (see ``docs/research/fwh-rotor-sim-audit.md``, bug #1), which is wrong for
    any time-varying rotor speed.

    Args:
        t: Monotonically increasing time grid [s], shape ``[T]``.
        omega: Angular rate samples at the grid times [rad/s], shape ``[T]``
            or scalar (broadcast to ``[T]``). Signed: positive means rotation
            from +x toward +y (counterclockwise seen from +z).
        psi0: Initial azimuth at ``t[0]`` [rad], scalar.

    Returns:
        Azimuth samples ``psi(t)`` [rad], shape ``[T]``, with
        ``psi[0] == psi0``. Not wrapped to ``[0, 2 pi)`` — the unwrapped phase
        is what downstream interpolation needs.
    """
    t = jnp.asarray(t)
    om = jnp.broadcast_to(jnp.asarray(omega), t.shape)
    increments = 0.5 * (om[:-1] + om[1:]) * jnp.diff(t)
    psi_rel = jnp.concatenate([jnp.zeros_like(t[:1]), jnp.cumsum(increments)])
    return jnp.asarray(psi0) + psi_rel


def azimuth_at(tau: ArrayLike, t_grid: ArrayLike, psi_grid: ArrayLike) -> Array:
    """Evaluate an integrated azimuth history at arbitrary (source) times.

    Linear interpolation of the samples produced by :func:`integrate_azimuth`.
    Intended for retarded-/source-time evaluation, where ``tau`` is generally
    batched (e.g. one emission time per source per observer time).

    Args:
        tau: Evaluation times [s], any shape ``[...]``. Times outside
            ``[t_grid[0], t_grid[-1]]`` are clamped (constant extrapolation;
            see :func:`interp1d` for the gradient implications).
        t_grid: Time grid the azimuth was integrated on [s], shape ``[T]``,
            strictly increasing.
        psi_grid: Azimuth samples at ``t_grid`` [rad], shape ``[T]``
            (unwrapped phase).

    Returns:
        Azimuth ``psi(tau)`` [rad], shape ``[...]`` (same shape as ``tau``).
    """
    return interp1d(tau, t_grid, psi_grid)
