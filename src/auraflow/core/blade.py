"""Blade, rotor, and vehicle geometry as equinox modules (pytrees).

Frame conventions (see ``docs/architecture.md``):

- *Blade section frame*: x spanwise outward, y chordwise toward the leading
  edge in the direction of rotation, z thrust-normal.
- *Rotor frame*: origin at hub, z along thrust axis; blade azimuth ``psi``
  measured from +x toward +y. A blade at azimuth ``psi`` has its span along
  ``rot_z(psi) @ [1, 0, 0]``.
- *World frame*: z up.

Static vs traced: station/blade counts are static Python ints; everything
physical (radius, chord, twist, positions, orientations) is stored as JAX
arrays and is differentiable.
"""

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from auraflow.core.frames import interp1d, rot_z

__all__ = ["BladeGeometry", "Rotor", "Vehicle"]


class BladeGeometry(eqx.Module):
    """Parametric rotor-blade geometry discretized at radial stations.

    The blade is described by chord and twist distributions sampled at
    ``n_stations`` uniformly spaced radial stations from ``hub_radius`` to
    ``radius`` (inclusive). Radial integrals over the blade use the
    trapezoid-consistent station widths :attr:`dr` (half panels at the ends),
    so ``sum(f(r) * dr)`` equals the trapezoid rule for ``int f dr``.

    Attributes:
        radius: Tip radius [m], scalar array (traced, differentiable).
        hub_radius: Root cutout radius [m], scalar array.
        chord: Chord at each station [m], shape ``[S]``.
        twist: Geometric twist (pitch) at each station [rad], shape ``[S]``,
            positive nose-up (leading edge toward +z thrust direction).
        n_stations: Number of radial stations ``S`` (static int).
    """

    radius: Array
    hub_radius: Array
    chord: Array
    twist: Array
    n_stations: int = eqx.field(static=True)

    def __init__(
        self,
        radius: ArrayLike,
        hub_radius: ArrayLike,
        chord: ArrayLike,
        twist: ArrayLike,
    ):
        """Construct directly from per-station arrays.

        Args:
            radius: Tip radius [m], scalar.
            hub_radius: Root cutout radius [m], scalar, ``0 <= hub_radius < radius``.
            chord: Chord at each station [m], shape ``[S]`` with ``S >= 2``.
            twist: Twist at each station [rad], shape ``[S]``.
        """
        chord_arr = jnp.asarray(chord)
        twist_arr = jnp.asarray(twist)
        if chord_arr.ndim != 1 or twist_arr.shape != chord_arr.shape:
            raise ValueError(
                f"chord and twist must be 1-D arrays of equal length, got shapes "
                f"{chord_arr.shape} and {twist_arr.shape}"
            )
        if chord_arr.shape[0] < 2:
            raise ValueError("BladeGeometry needs at least 2 radial stations")
        self.radius = jnp.asarray(radius)
        self.hub_radius = jnp.asarray(hub_radius)
        self.chord = chord_arr
        self.twist = twist_arr
        self.n_stations = int(chord_arr.shape[0])

    @classmethod
    def from_arrays(
        cls,
        r: ArrayLike,
        chord: ArrayLike,
        twist: ArrayLike,
        n_stations: int | None = None,
    ) -> "BladeGeometry":
        """Build a blade from tabulated ``(r, chord, twist)`` data.

        The input radial samples need not be uniformly spaced; chord and
        twist are linearly interpolated onto ``n_stations`` uniform,
        trapezoid-consistent stations spanning ``[r[0], r[-1]]``.

        Args:
            r: Radial sample locations [m], shape ``[N]``, strictly increasing.
                ``r[0]`` becomes the hub radius, ``r[-1]`` the tip radius.
            chord: Chord at the samples [m], shape ``[N]``.
            twist: Twist at the samples [rad], shape ``[N]``.
            n_stations: Number of output stations ``S`` (static int).
                Defaults to ``N``.

        Returns:
            A :class:`BladeGeometry` sampled on the uniform station grid.
        """
        r_arr = jnp.asarray(r)
        if n_stations is None:
            n_stations = int(r_arr.shape[0])
        r_new = jnp.linspace(r_arr[0], r_arr[-1], n_stations)
        return cls(
            radius=r_arr[-1],
            hub_radius=r_arr[0],
            chord=interp1d(r_new, r_arr, chord),
            twist=interp1d(r_new, r_arr, twist),
        )

    @classmethod
    def linear(
        cls,
        radius: ArrayLike,
        hub_radius: ArrayLike,
        n_stations: int,
        chord_root: ArrayLike,
        chord_tip: ArrayLike,
        twist_root: ArrayLike,
        twist_tip: ArrayLike,
    ) -> "BladeGeometry":
        """Blade with linear chord taper and linear twist distribution.

        Args:
            radius: Tip radius [m], scalar.
            hub_radius: Root cutout radius [m], scalar.
            n_stations: Number of radial stations ``S`` (static int, >= 2).
            chord_root: Chord at ``r = hub_radius`` [m].
            chord_tip: Chord at ``r = radius`` [m].
            twist_root: Twist at ``r = hub_radius`` [rad].
            twist_tip: Twist at ``r = radius`` [rad].

        Returns:
            A :class:`BladeGeometry` with linearly varying chord and twist.
        """
        s = jnp.linspace(0.0, 1.0, n_stations)
        chord_root = jnp.asarray(chord_root)
        twist_root = jnp.asarray(twist_root)
        return cls(
            radius=radius,
            hub_radius=hub_radius,
            chord=chord_root + (jnp.asarray(chord_tip) - chord_root) * s,
            twist=twist_root + (jnp.asarray(twist_tip) - twist_root) * s,
        )

    @property
    def r(self) -> Array:
        """Station radii [m], shape ``[S]``: uniform from hub_radius to radius."""
        return jnp.linspace(self.hub_radius, self.radius, self.n_stations)

    @property
    def dr(self) -> Array:
        """Trapezoid-consistent station widths [m], shape ``[S]``.

        Interior stations get the full spacing ``h``; the two end stations get
        ``h / 2``, so that ``sum(f(r) * dr)`` is the trapezoid rule for
        ``int_{hub}^{tip} f(r) dr`` and ``sum(dr) == radius - hub_radius``.
        """
        h = (self.radius - self.hub_radius) / (self.n_stations - 1)
        weights = jnp.ones(self.n_stations, dtype=h.dtype).at[0].set(0.5).at[-1].set(0.5)
        return h * weights

    def chord_at(self, rq: ArrayLike) -> Array:
        """Chord at arbitrary radii by differentiable linear interpolation.

        Args:
            rq: Query radii [m], any shape ``[...]``. Clamped to
                ``[hub_radius, radius]`` (constant extrapolation outside).

        Returns:
            Chord [m], shape ``[...]``.
        """
        return interp1d(rq, self.r, self.chord)

    def twist_at(self, rq: ArrayLike) -> Array:
        """Twist at arbitrary radii by differentiable linear interpolation.

        Args:
            rq: Query radii [m], any shape ``[...]``. Clamped to
                ``[hub_radius, radius]`` (constant extrapolation outside).

        Returns:
            Twist [rad], shape ``[...]``.
        """
        return interp1d(rq, self.r, self.twist)

    def quarter_chord_points(self, psi: ArrayLike) -> Array:
        """Compact-source (quarter-chord / pitch-axis) positions in the rotor frame.

        Chordwise-compact approximation: each radial station is represented by
        a point on the blade pitch axis, at ``(r_s, 0, 0)`` in the blade
        section frame. At azimuth ``psi`` the blade span lies along
        ``rot_z(psi) @ [1, 0, 0]``, so station ``s`` sits at
        ``(r_s cos(psi), r_s sin(psi), 0)``.

        Args:
            psi: Blade azimuth angle(s) [rad], shape ``[...]`` (scalar allowed).

        Returns:
            Source positions in the rotor frame [m], shape ``[..., S, 3]``.
        """
        base = jnp.stack(
            [self.r, jnp.zeros(self.n_stations), jnp.zeros(self.n_stations)], axis=-1
        )  # [S, 3]
        return jnp.einsum("...ij,sj->...si", rot_z(psi), base)

    def section_velocity(self, psi: ArrayLike, omega: ArrayLike) -> Array:
        """Rigid-rotation velocity of each compact source in the rotor frame.

        ``v = Omega z_hat x y`` for a point ``y`` on the rotor at azimuth
        ``psi``, i.e. ``v = Omega * (-y_y, y_x, 0)``. The speed of station
        ``s`` is ``|Omega| r_s`` and the velocity is tangential.

        Args:
            psi: Blade azimuth angle(s) [rad], shape ``[...]``.
            omega: Signed angular rate [rad/s], scalar or broadcastable with
                ``psi``. Positive rotates +x toward +y.

        Returns:
            Velocities in the rotor frame [m/s], shape ``[..., S, 3]``.
        """
        pts = self.quarter_chord_points(psi)  # [..., S, 3]
        om = jnp.asarray(omega)[..., None]  # broadcast over stations
        vx = -om * pts[..., 1]
        vy = om * pts[..., 0]
        return jnp.stack([vx, vy, jnp.zeros_like(vx)], axis=-1)


class Rotor(eqx.Module):
    """Multi-blade rotor: one blade geometry replicated at ``n_blades`` azimuths.

    Attributes:
        blade: The (shared) blade geometry.
        n_blades: Number of blades ``B`` (static int).
        hub_position: Hub origin in the parent frame [m], shape ``[3]``.
            The parent frame is the world frame for a standalone rotor, or the
            vehicle body frame when the rotor belongs to a :class:`Vehicle`.
        hub_orientation: Rotation matrix (parent <- rotor), shape ``[3, 3]``:
            maps rotor-frame coordinates to the parent frame. Its third column
            is the thrust axis expressed in the parent frame.
        spin_direction: ``+1`` for counterclockwise rotation seen from the +z
            thrust axis, ``-1`` for clockwise (static int).
    """

    blade: BladeGeometry
    hub_position: Array
    hub_orientation: Array
    n_blades: int = eqx.field(static=True)
    spin_direction: int = eqx.field(static=True)

    def __init__(
        self,
        blade: BladeGeometry,
        n_blades: int,
        hub_position: ArrayLike | None = None,
        hub_orientation: ArrayLike | None = None,
        spin_direction: int = 1,
    ):
        """Construct a rotor.

        Args:
            blade: Blade geometry shared by all blades.
            n_blades: Number of blades ``B`` (static int, >= 1).
            hub_position: Hub origin in the parent frame [m], shape ``[3]``.
                Defaults to the origin.
            hub_orientation: Rotation matrix (parent <- rotor), shape
                ``[3, 3]``. Defaults to identity (thrust along parent +z).
            spin_direction: ``+1`` (CCW seen from +z) or ``-1`` (CW).
        """
        if n_blades < 1:
            raise ValueError("n_blades must be >= 1")
        if spin_direction not in (1, -1):
            raise ValueError("spin_direction must be +1 or -1")
        self.blade = blade
        self.n_blades = int(n_blades)
        self.spin_direction = int(spin_direction)
        self.hub_position = jnp.zeros(3) if hub_position is None else jnp.asarray(hub_position)
        self.hub_orientation = (
            jnp.eye(3) if hub_orientation is None else jnp.asarray(hub_orientation)
        )

    def blade_azimuths(self, psi0: ArrayLike = 0.0) -> Array:
        """Azimuth of every blade given the reference blade's azimuth.

        Blade ``b`` sits at ``psi0 + spin_direction * 2 pi b / B`` so that the
        blades are equally spaced and ordered along the direction of rotation.
        (For azimuth *evolution*, integrate a signed rate, e.g.
        ``omega = spin_direction * |Omega|``, with
        :func:`auraflow.core.frames.integrate_azimuth`.)

        Args:
            psi0: Azimuth of blade 0 [rad], shape ``[...]`` (scalar allowed).

        Returns:
            Azimuths [rad], shape ``[..., B]``.
        """
        offsets = self.spin_direction * 2.0 * jnp.pi * jnp.arange(self.n_blades) / self.n_blades
        return jnp.asarray(psi0)[..., None] + offsets

    def to_parent_points(self, points: ArrayLike) -> Array:
        """Map rotor-frame points to the parent (world or vehicle body) frame.

        Args:
            points: Rotor-frame positions [m], shape ``[..., 3]``.

        Returns:
            Parent-frame positions [m], shape ``[..., 3]``:
            ``hub_position + hub_orientation @ points``.
        """
        return self.hub_position + self.to_parent_vectors(points)

    def to_parent_vectors(self, vectors: ArrayLike) -> Array:
        """Map rotor-frame vectors (velocities, forces) to the parent frame.

        Rotation only — no translation.

        Args:
            vectors: Rotor-frame vectors, shape ``[..., 3]``.

        Returns:
            Parent-frame vectors, shape ``[..., 3]``.
        """
        return jnp.einsum("ij,...j->...i", self.hub_orientation, jnp.asarray(vectors))


class Vehicle(eqx.Module):
    """A collection of rotors placed relative to a vehicle reference frame.

    Minimal placement container: enough to express several rotors in the world
    frame. Full 6-DOF state (velocities, mass properties, control) lives in
    higher-level modules.

    Attributes:
        rotors: The rotors, with their ``hub_position``/``hub_orientation``
            expressed in the vehicle body frame.
        position: Vehicle reference position in the world frame [m], shape ``[3]``.
        attitude: Rotation matrix (world <- body), shape ``[3, 3]``, e.g. from
            :func:`auraflow.core.frames.euler_zyx_matrix`.
    """

    rotors: tuple[Rotor, ...]
    position: Array
    attitude: Array

    def __init__(
        self,
        rotors: tuple[Rotor, ...] | list[Rotor],
        position: ArrayLike | None = None,
        attitude: ArrayLike | None = None,
    ):
        """Construct a vehicle from rotors and a reference pose.

        Args:
            rotors: Rotors with body-frame hub placements.
            position: World-frame reference position [m], shape ``[3]``.
                Defaults to the origin.
            attitude: Rotation matrix (world <- body), shape ``[3, 3]``.
                Defaults to identity.
        """
        self.rotors = tuple(rotors)
        self.position = jnp.zeros(3) if position is None else jnp.asarray(position)
        self.attitude = jnp.eye(3) if attitude is None else jnp.asarray(attitude)

    @property
    def n_rotors(self) -> int:
        """Number of rotors (static int)."""
        return len(self.rotors)

    def to_world_points(self, points: ArrayLike) -> Array:
        """Map body-frame points to the world frame.

        Args:
            points: Body-frame positions [m], shape ``[..., 3]``.

        Returns:
            World-frame positions [m], shape ``[..., 3]``:
            ``position + attitude @ points``.
        """
        return self.position + self.to_world_vectors(points)

    def to_world_vectors(self, vectors: ArrayLike) -> Array:
        """Map body-frame vectors to the world frame (rotation only).

        Args:
            vectors: Body-frame vectors, shape ``[..., 3]``.

        Returns:
            World-frame vectors, shape ``[..., 3]``.
        """
        return jnp.einsum("ij,...j->...i", self.attitude, jnp.asarray(vectors))

    def rotor_in_world(self, index: int) -> Rotor:
        """Rotor ``index`` with its hub placement composed into the world frame.

        Args:
            index: Which rotor (static int).

        Returns:
            A new :class:`Rotor` whose ``hub_position``/``hub_orientation``
            are expressed in the world frame, so its ``to_parent_*`` helpers
            map rotor-frame quantities directly to world coordinates.
        """
        rotor = self.rotors[index]
        return Rotor(
            blade=rotor.blade,
            n_blades=rotor.n_blades,
            hub_position=self.to_world_points(rotor.hub_position),
            hub_orientation=self.attitude @ rotor.hub_orientation,
            spin_direction=rotor.spin_direction,
        )
