r"""6-DOF rigid-body multirotor dynamics + geometric SE(3) tracking controller.

This is the flight-dynamics front end of the CONA backend
(``docs/research/cona-reference.md`` module 1): it produces the vehicle and
per-rotor kinematic histories that the aeroacoustic stage consumes -- vehicle
position/attitude, body rates, and per-rotor speed/thrust time histories along a
commanded trajectory.

Model (see ``docs/research/cona-external-formulations.md`` sect. 5, geometric
controller of Lee, Leok & McClamroch, arXiv:1003.2005 -- our documented open
substitute for CONA's paywalled Zuo 2010 controller):

- **State** ``(x, v, R, Omega)``: world position ``x`` [m], world velocity ``v``
  [m/s], attitude ``R`` (world <- body, in ``SO(3)``), body angular rate
  ``Omega`` [rad/s].
- **Dynamics** (world **z-up**, this library's convention; the digest is written
  e3-down/NED and is converted here -- see below):

  .. math::
     \dot x = v, \quad
     m \dot v = f\,R e_3 - m g e_3 + F_\text{drag}, \quad
     \dot R = R \hat\Omega, \quad
     J \dot\Omega + \Omega\times J\Omega = M

  where ``e3 = [0, 0, 1]`` is world up, ``f`` [N] is the (positive) collective
  thrust along the body ``+z`` (thrust) axis, ``M`` [N.m] the body moment, and
  ``F_drag`` an optional linear wind-drag disturbance (the additive-gust hook,
  :mod:`auraflow.cona.gusts`).
- **Controller**: geometric SE(3) tracking. Translational errors
  ``e_x = x - x_d``, ``e_v = v - v_d`` build a desired thrust vector; the desired
  attitude ``Rc`` is reconstructed from that vector and a commanded heading
  ``b1d``; rotational errors ``e_R``, ``e_Omega`` (with optional angular-velocity
  feed-forward) give the moment.
- **Allocation / motor model**: a rotor mixing matrix built from the vehicle rotor
  positions and spin directions maps ``(f, M)`` to per-rotor thrusts ``f_i``;
  each rotor speed follows the calibrated law ``f_i = k_f Omega_i^2`` with an
  optional first-order motor lag.
- **Integrator**: fixed-step RK4 inside :func:`jax.lax.scan`; the wrench is held
  constant across the step (zero-order hold at the control rate). ``R`` is kept on
  ``SO(3)`` by a per-step polar (SVD) re-orthonormalization.

**Frame-convention note.** The digest is written with ``e3`` pointing *down*
(NED), gravity ``+m g e3`` and thrust ``-f R e3``. AuraFlow's world frame is
**z-up** (``docs/architecture.md``). We therefore expose and integrate the z-up
form throughout: gravity is ``-m g e3`` and thrust ``+f R e3`` with ``f > 0``.
The *rotational* controller is identical in both conventions; only the
translational sign of gravity / desired-thrust vector flips. Everything the
aeroacoustic stage sees (:class:`FlightHistory`) is z-up, SI.

Everything is float64-safe, ``vmap``/``grad``/``scan``-friendly, and contains no
Python loop over time steps.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from auraflow.core.blade import Vehicle

__all__ = [
    "ControllerGains",
    "FlightHistory",
    "Multirotor",
    "Reference",
    "attitude_error",
    "desired_attitude",
    "geometric_controller",
    "hat",
    "hover",
    "simulate",
    "straight_flyover",
    "vee",
]

# Standard gravity [m/s^2] (matches auraflow.core.medium).
_G = 9.80665
_E3 = jnp.array([0.0, 0.0, 1.0])

# A reference is a function of time returning (x_d [3], v_d [3], a_d [3], b1d [3]).
Reference = Callable[[Array], tuple[Array, Array, Array, Array]]


# --------------------------------------------------------------------------- #
# SO(3) helpers
# --------------------------------------------------------------------------- #
def hat(w: ArrayLike) -> Array:
    r"""Skew-symmetric ``hat`` map ``R^3 -> so(3)``.

    ``hat(w) @ u == cross(w, u)`` for any ``u``.

    Args:
        w: 3-vector, shape ``[3]``.

    Returns:
        Skew matrix ``[[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]``, shape ``[3, 3]``.
    """
    w = jnp.asarray(w)
    wx, wy, wz = w[0], w[1], w[2]
    zero = jnp.zeros_like(wx)
    return jnp.stack(
        [
            jnp.stack([zero, -wz, wy]),
            jnp.stack([wz, zero, -wx]),
            jnp.stack([-wy, wx, zero]),
        ]
    )


def vee(S: ArrayLike) -> Array:
    r"""Inverse ``vee`` map ``so(3) -> R^3`` (``vee(hat(w)) == w``).

    Uses the antisymmetric part, so ``vee`` of a nearly-skew matrix is the
    least-squares axis vector.

    Args:
        S: (Approximately) skew-symmetric matrix, shape ``[3, 3]``.

    Returns:
        3-vector, shape ``[3]``: ``0.5 * [S21 - S12, S02 - S20, S10 - S01]``.
    """
    S = jnp.asarray(S)
    return 0.5 * jnp.array([S[2, 1] - S[1, 2], S[0, 2] - S[2, 0], S[1, 0] - S[0, 1]])


def _project_so3(R: Array) -> Array:
    """Nearest rotation matrix to ``R`` via the Higham polar iteration.

    Two Newton steps ``Y <- 0.5 (Y + Y^{-T})`` of the polar decomposition, which
    converge quadratically to the orthogonal polar factor. Since ``R`` is already
    near-orthonormal after one small RK4 step, this restores
    ``||R^T R - I|| ~ machine epsilon`` while staying fully differentiable
    (unlike an SVD projection, whose gradient is singular for a rotation matrix's
    degenerate unit singular values).
    """
    Y = R
    for _ in range(2):
        Y = 0.5 * (Y + jnp.linalg.inv(Y).T)
    return Y


# --------------------------------------------------------------------------- #
# Vehicle / gains modules
# --------------------------------------------------------------------------- #
class Multirotor(eqx.Module):
    """Mass, inertia and rotor-allocation properties of a multirotor.

    The rotor geometry (per-rotor body position and spin direction) is what the
    mixing matrix needs; it is taken from a :class:`auraflow.core.blade.Vehicle`
    via :meth:`from_vehicle`, or supplied directly. All physical fields are JAX
    arrays and differentiable (e.g. for sensitivity of the flight to vehicle
    mass).

    Attributes:
        mass: Vehicle mass ``m`` [kg], scalar.
        inertia: Body inertia tensor ``J`` [kg.m^2], shape ``[3, 3]``
            (about the CG, body axes).
        rotor_positions: Rotor hub positions in the body frame [m], shape
            ``[Nr, 3]``. Only the in-plane ``(x, y)`` components enter the mixing
            (a body-``z`` thrust produces no moment about ``z``-offset).
        spin_signs: Per-rotor spin sense, shape ``[Nr]``, ``+1`` for CCW seen
            from ``+z`` (body up), ``-1`` for CW. Sets the reaction-torque sign.
        k_f: Thrust coefficient in ``f_i = k_f * Omega_i^2`` [N.s^2], scalar.
            Calibrate at hover: ``k_f = f_hover / Omega_hover^2``.
        c_tauf: Reaction-torque-to-thrust ratio ``Q_i = c_tauf * f_i`` [m],
            scalar.
        drag_coeff: Linear translational wind-drag coefficient [N.s/m], scalar.
            Adds ``F_drag = drag_coeff * (v_wind - v)`` to the translational
            dynamics -- the additive-gust hook. Defaults to ``0`` (pure
            rigid body); set positive to couple :mod:`auraflow.cona.gusts`.
        motor_tau: First-order motor time constant [s] or ``None``. If ``None``
            (default) rotor speed tracks the command instantly; otherwise the
            rotor speed relaxes toward the command with this time constant.
        n_rotors: Number of rotors ``Nr`` (static int).
    """

    mass: Array
    inertia: Array
    rotor_positions: Array
    spin_signs: Array
    k_f: Array
    c_tauf: Array
    drag_coeff: Array
    motor_tau: float | None = eqx.field(static=True)
    n_rotors: int = eqx.field(static=True)

    def __init__(
        self,
        mass: ArrayLike,
        inertia: ArrayLike,
        rotor_positions: ArrayLike,
        spin_signs: ArrayLike,
        k_f: ArrayLike,
        c_tauf: ArrayLike,
        drag_coeff: ArrayLike = 0.0,
        motor_tau: float | None = None,
    ):
        pos = jnp.asarray(rotor_positions, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"rotor_positions must have shape [Nr, 3], got {pos.shape}")
        inertia_arr = jnp.asarray(inertia, dtype=float)
        if inertia_arr.shape != (3, 3):
            raise ValueError(f"inertia must have shape [3, 3], got {inertia_arr.shape}")
        self.mass = jnp.asarray(mass, dtype=float)
        self.inertia = inertia_arr
        self.rotor_positions = pos
        self.spin_signs = jnp.asarray(spin_signs, dtype=float)
        self.k_f = jnp.asarray(k_f, dtype=float)
        self.c_tauf = jnp.asarray(c_tauf, dtype=float)
        self.drag_coeff = jnp.asarray(drag_coeff, dtype=float)
        self.motor_tau = None if motor_tau is None else float(motor_tau)
        self.n_rotors = int(pos.shape[0])

    @classmethod
    def from_vehicle(
        cls,
        vehicle: Vehicle,
        mass: ArrayLike,
        inertia: ArrayLike,
        k_f: ArrayLike,
        c_tauf: ArrayLike,
        drag_coeff: ArrayLike = 0.0,
        motor_tau: float | None = None,
    ) -> "Multirotor":
        """Build from a core :class:`~auraflow.core.blade.Vehicle`.

        Rotor body positions and spin directions are read from the vehicle's
        rotors; mass/inertia are supplied separately (the core ``Vehicle`` is a
        geometry container without mass properties).

        Args:
            vehicle: Vehicle whose rotors carry body-frame ``hub_position`` and
                ``spin_direction``.
            mass: Vehicle mass [kg].
            inertia: Body inertia tensor [kg.m^2], shape ``[3, 3]``.
            k_f: Thrust coefficient [N.s^2].
            c_tauf: Torque-to-thrust ratio [m].
            drag_coeff: Linear wind-drag coefficient [N.s/m].
            motor_tau: Motor time constant [s] or ``None``.

        Returns:
            The corresponding :class:`Multirotor`.
        """
        positions = jnp.stack([r.hub_position for r in vehicle.rotors])
        spins = jnp.array([float(r.spin_direction) for r in vehicle.rotors])
        return cls(mass, inertia, positions, spins, k_f, c_tauf, drag_coeff, motor_tau)

    @classmethod
    def nasa_1pax(cls, drag_coeff: ArrayLike = 0.0, motor_tau: float | None = None) -> "Multirotor":
        """Defaults for the NASA 1-Pax UAM quadrotor (``docs/research/nasa-1pax-vehicle.md``).

        Mass 583.85 kg; body inertia diag ``(997.7, 1089.1, 1317.8)`` kg.m^2;
        X-arrangement hubs at ``(+-2.63, +-2.63)`` m (rear rotors 0.683 m higher);
        diagonal rotors share spin sense. ``k_f`` is calibrated from the hover
        operating point (per-rotor thrust ``m g / 4`` at ``Omega = 70.3`` rad/s);
        ``c_tauf = 0.154`` m from the hover torque/thrust ratio (20.8 hp, 1431 N).

        Args:
            drag_coeff: Linear wind-drag coefficient [N.s/m].
            motor_tau: Motor time constant [s] or ``None``.

        Returns:
            A :class:`Multirotor` for the NASA 1-Pax vehicle.
        """
        mass = 583.85
        inertia = jnp.diag(jnp.array([997.7, 1089.1, 1317.8]))
        d = 2.63
        z_rear = 0.683
        # Front rotors at +x (z=0), rear at -x (z=+z_rear); columns +-y.
        positions = jnp.array(
            [
                [d, d, 0.0],  # front-left
                [d, -d, 0.0],  # front-right
                [-d, d, z_rear],  # rear-left
                [-d, -d, z_rear],  # rear-right
            ]
        )
        # Diagonally opposite rotors share spin; adjacent alternate (torque balance).
        spins = jnp.array([1.0, -1.0, -1.0, 1.0])
        omega_hover = 70.3
        thrust_hover = mass * _G / 4.0
        k_f = thrust_hover / omega_hover**2
        c_tauf = 0.154
        return cls(mass, inertia, positions, spins, k_f, c_tauf, drag_coeff, motor_tau)

    @classmethod
    def dji_phantom(
        cls, drag_coeff: ArrayLike = 0.0, motor_tau: float | None = None
    ) -> "Multirotor":
        """Defaults for the DJI Phantom quadrotor (``docs/research/dji-9450-reference.md``).

        Drone-scale counterpart of :meth:`nasa_1pax`. Mass **1.280 kg** (Phantom 3
        Adv./Pro./4K, the variant that ships the DJI 9450 prop; PUBLISHED);
        X-arrangement hubs at ``(+-0.1237, +-0.1237)`` m (from the published
        ``0.350 m`` motor-to-motor diagonal via ``0.5*d*cos45``; rotor plane
        ``z = 0`` -- the rotor-plane height is not published); spin pattern
        front-left CW / front-right CCW / rear-left CCW / rear-right CW (diagonal
        pairs share sense; PUBLISHED community docs). ``k_f`` is calibrated from
        the hover operating point (per-rotor thrust ``m g / 4`` at the nominal
        ``5400 RPM`` = ``565.5 rad/s``). Body inertia and ``c_tauf`` are
        reconstructed (Phantom inertia is not published) -- estimated from the
        mass/arm distribution and a hover figure-of-merit ``~0.55``; documented
        assumptions, adequate for the slow level-flight flyover scenarios.

        Args:
            drag_coeff: Linear wind-drag coefficient [N.s/m].
            motor_tau: Motor time constant [s] or ``None``.

        Returns:
            A :class:`Multirotor` for the DJI Phantom vehicle.
        """
        mass = 1.280
        # Reconstructed body inertia [kg.m^2] (not published): from the motor/prop
        # masses on the +-0.1237 m arms plus a compact battery/shell body.
        inertia = jnp.diag(jnp.array([0.008, 0.008, 0.015]))
        d = 0.1237  # 0.5 * 0.350 m diagonal * cos 45deg
        # Front rotors at +x, rear at -x; columns +-y (rotor plane z = 0).
        positions = jnp.array(
            [
                [d, d, 0.0],  # front-left
                [d, -d, 0.0],  # front-right
                [-d, d, 0.0],  # rear-left
                [-d, -d, 0.0],  # rear-right
            ]
        )
        # front-left CW, front-right CCW, rear-left CCW, rear-right CW
        # (+1 = CCW from +z, -1 = CW). Diagonal pairs share sense; net yaw zero.
        spins = jnp.array([-1.0, 1.0, 1.0, -1.0])
        omega_hover = 565.5  # 5400 RPM
        thrust_hover = mass * _G / 4.0
        k_f = thrust_hover / omega_hover**2
        c_tauf = 0.017  # reconstructed torque/thrust ratio [m] (FM ~0.55 hover)
        return cls(mass, inertia, positions, spins, k_f, c_tauf, drag_coeff, motor_tau)

    def mixing_matrix(self) -> Array:
        r"""Rotor mixing matrix ``B`` mapping per-rotor thrusts to the wrench.

        ``[f, Mx, My, Mz]^T = B @ [f_1, ..., f_Nr]^T`` with, for a rotor at body
        position ``(x_i, y_i, z_i)`` producing thrust ``f_i`` along body ``+z``:

        - total thrust ``f = sum f_i``;
        - roll moment ``Mx = sum f_i y_i``;
        - pitch moment ``My = -sum f_i x_i``;
        - yaw moment ``Mz = sum (-spin_i) c_tauf f_i`` (reaction torque: a CCW
          rotor drags the airframe CW).

        Returns:
            Mixing matrix, shape ``[4, Nr]``.
        """
        x = self.rotor_positions[:, 0]
        y = self.rotor_positions[:, 1]
        return jnp.stack(
            [
                jnp.ones_like(x),
                y,
                -x,
                -self.spin_signs * self.c_tauf,
            ]
        )

    def hover_omega(self) -> Array:
        """Per-rotor speed magnitude at hover [rad/s]: ``sqrt(m g / (Nr k_f))``."""
        return jnp.sqrt(self.mass * _G / (self.n_rotors * self.k_f))


class ControllerGains(eqx.Module):
    """Gains for the geometric SE(3) tracking controller.

    Attributes:
        k_x: Position-error gain [N/m], scalar.
        k_v: Velocity-error gain [N.s/m], scalar.
        k_R: Attitude-error gain [N.m], scalar.
        k_Omega: Angular-rate-error gain [N.m.s], scalar.
    """

    k_x: Array
    k_v: Array
    k_R: Array
    k_Omega: Array

    def __init__(
        self,
        k_x: ArrayLike = 1.0,
        k_v: ArrayLike = 2.0,
        k_R: ArrayLike = 1.0,
        k_Omega: ArrayLike = 2.0,
    ):
        self.k_x = jnp.asarray(k_x, dtype=float)
        self.k_v = jnp.asarray(k_v, dtype=float)
        self.k_R = jnp.asarray(k_R, dtype=float)
        self.k_Omega = jnp.asarray(k_Omega, dtype=float)

    @classmethod
    def for_vehicle(
        cls,
        mrotor: Multirotor,
        omega_n_pos: float = 1.0,
        zeta_pos: float = 1.0,
        omega_n_att: float = 4.0,
        zeta_att: float = 1.0,
    ) -> "ControllerGains":
        r"""Physically-scaled gains for a given vehicle (2nd-order closed loops).

        The translational error dynamics ``m e_x'' + k_v e_x' + k_x e_x = 0`` are
        placed at natural frequency ``omega_n_pos`` and damping ``zeta_pos``, so
        ``k_x = m omega_n^2``, ``k_v = 2 m zeta omega_n``. The attitude loop is
        placed similarly using a representative inertia (the largest diagonal
        moment), at the (faster) ``omega_n_att``. Defaults give a critically
        damped ~1 rad/s translational and ~4 rad/s attitude response, adequate
        for the heavy NASA 1-Pax vehicle and the level-flight scenarios.

        Args:
            mrotor: The vehicle (for mass and inertia).
            omega_n_pos: Translational natural frequency [rad/s].
            zeta_pos: Translational damping ratio [-].
            omega_n_att: Attitude natural frequency [rad/s].
            zeta_att: Attitude damping ratio [-].

        Returns:
            Tuned :class:`ControllerGains`.
        """
        m = mrotor.mass
        j = jnp.max(jnp.diag(mrotor.inertia))
        return cls(
            k_x=m * omega_n_pos**2,
            k_v=2.0 * m * zeta_pos * omega_n_pos,
            k_R=j * omega_n_att**2,
            k_Omega=2.0 * j * zeta_att * omega_n_att,
        )


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
def desired_attitude(force_des: Array, b1d: Array) -> Array:
    r"""Desired attitude ``Rc`` from a desired thrust vector and heading.

    The body ``+z`` (thrust) axis is aligned with the desired thrust direction;
    the body ``+x`` axis is placed in the vertical plane of the commanded heading
    ``b1d`` (world frame). ``Rc = [b1c, b2c, b3c]`` (columns), with
    ``b3c = force_des / |force_des|``, ``b2c = b3c x b1d / |.|``, ``b1c = b2c x b3c``.

    Args:
        force_des: Desired thrust vector in the world frame [N], shape ``[3]``
            (must be non-zero; points along body ``+z`` at the solution).
        b1d: Commanded heading direction in the world frame, shape ``[3]``
            (need not be unit or orthogonal to ``force_des``).

    Returns:
        Desired rotation matrix ``Rc`` (world <- body), shape ``[3, 3]``.
    """
    b3c = force_des / jnp.linalg.norm(force_des)
    b2c = jnp.cross(b3c, b1d)
    b2c = b2c / jnp.linalg.norm(b2c)
    b1c = jnp.cross(b2c, b3c)
    return jnp.stack([b1c, b2c, b3c], axis=1)


def attitude_error(R: Array, Rc: Array) -> Array:
    r"""Attitude error vector ``e_R = 0.5 (Rc^T R - R^T Rc)^\vee``.

    Args:
        R: Current attitude (world <- body), shape ``[3, 3]``.
        Rc: Desired attitude (world <- body), shape ``[3, 3]``.

    Returns:
        Attitude error, shape ``[3]`` (zero iff ``R == Rc``).
    """
    return vee(Rc.T @ R - R.T @ Rc)


def geometric_controller(
    state: tuple[Array, Array, Array, Array],
    ref: tuple[Array, Array, Array, Array],
    mrotor: Multirotor,
    gains: ControllerGains,
    omega_c: Array | None = None,
    omega_dot_c: Array | None = None,
) -> tuple[Array, Array]:
    r"""Geometric SE(3) tracking control law (z-up form).

    Computes the collective thrust ``f`` and body moment ``M`` (Lee/Leok/
    McClamroch, arXiv:1003.2005; z-up conversion documented in the module
    docstring):

    .. math::
       F_\text{des} &= -k_x e_x - k_v e_v + m g e_3 + m \ddot x_d \\
       f &= F_\text{des} \cdot (R e_3) \\
       M &= -k_R e_R - k_\Omega e_\Omega + \Omega\times J\Omega
             - J\big(\hat\Omega R^T R_c \Omega_c - R^T R_c \dot\Omega_c\big)

    with ``e_x = x - x_d``, ``e_v = v - v_d``, ``e_Omega = Omega - R^T Rc Omega_c``.

    Args:
        state: ``(x, v, R, Omega)`` -- world position [m], world velocity [m/s],
            attitude ``[3, 3]``, body rate [rad/s].
        ref: ``(x_d, v_d, a_d, b1d)`` -- desired world position, velocity,
            acceleration [m/s^2] and commanded heading direction, each ``[3]``.
        mrotor: Vehicle (mass, inertia).
        gains: Controller gains.
        omega_c: Feed-forward desired body rate [rad/s], shape ``[3]``.
            Defaults to zero (exact for the constant-attitude hover/level-flight
            setpoints used here).
        omega_dot_c: Feed-forward desired body angular acceleration [rad/s^2],
            shape ``[3]``. Defaults to zero.

    Returns:
        ``(f, M)`` -- collective thrust [N] (scalar) and body moment [N.m]
        (shape ``[3]``).
    """
    x, v, R, Omega = state
    x_d, v_d, a_d, b1d = ref
    m = mrotor.mass
    J = mrotor.inertia
    omega_c = jnp.zeros(3) if omega_c is None else omega_c
    omega_dot_c = jnp.zeros(3) if omega_dot_c is None else omega_dot_c

    e_x = x - x_d
    e_v = v - v_d
    force_des = -gains.k_x * e_x - gains.k_v * e_v + m * _G * _E3 + m * a_d
    b3 = R @ _E3
    f = jnp.dot(force_des, b3)

    Rc = desired_attitude(force_des, b1d)
    e_R = attitude_error(R, Rc)
    RtRc = R.T @ Rc
    e_Omega = Omega - RtRc @ omega_c
    JOmega = J @ Omega
    ffwd = J @ (hat(Omega) @ (RtRc @ omega_c) - RtRc @ omega_dot_c)
    M = -gains.k_R * e_R - gains.k_Omega * e_Omega + jnp.cross(Omega, JOmega) - ffwd
    return f, M


# --------------------------------------------------------------------------- #
# Allocation / motor model
# --------------------------------------------------------------------------- #
def allocate_thrusts(f: Array, M: Array, mrotor: Multirotor) -> Array:
    """Solve the mixing for per-rotor thrusts, clamped non-negative.

    Args:
        f: Collective thrust [N], scalar.
        M: Body moment [N.m], shape ``[3]``.
        mrotor: Vehicle (mixing matrix).

    Returns:
        Per-rotor thrusts [N], shape ``[Nr]``, clamped to ``>= 0`` (a rotor
        cannot push with negative thrust). The clamp is a gradient dead zone
        only where a commanded thrust would be negative.
    """
    B = mrotor.mixing_matrix()
    wrench = jnp.concatenate([jnp.atleast_1d(f), M])
    f_i = jnp.linalg.pinv(B) @ wrench
    return jnp.maximum(f_i, 0.0)


def _thrust_to_speed(f_i: Array, mrotor: Multirotor) -> Array:
    """Per-rotor speed magnitude [rad/s] from thrust via ``f_i = k_f Omega^2``."""
    return jnp.sqrt(jnp.maximum(f_i, 0.0) / mrotor.k_f)


# --------------------------------------------------------------------------- #
# Dynamics + integrator
# --------------------------------------------------------------------------- #
def _deriv(
    state: tuple[Array, Array, Array, Array],
    f: Array,
    M: Array,
    mrotor: Multirotor,
    wind: Array,
) -> tuple[Array, Array, Array, Array]:
    """Rigid-body state derivative for a constant wrench ``(f, M)`` and wind."""
    x, v, R, Omega = state
    m = mrotor.mass
    J = mrotor.inertia
    thrust_world = f * (R @ _E3)
    grav = -m * _G * _E3
    f_drag = mrotor.drag_coeff * (wind - v)
    a = (thrust_world + grav + f_drag) / m
    R_dot = R @ hat(Omega)
    Omega_dot = jnp.linalg.solve(J, M - jnp.cross(Omega, J @ Omega))
    return v, a, R_dot, Omega_dot


def _axpy(
    state: tuple[Array, Array, Array, Array],
    k: tuple[Array, Array, Array, Array],
    h: float,
) -> tuple[Array, Array, Array, Array]:
    """Return ``state + h * k`` element-wise over the state tuple."""
    return tuple(s + h * ki for s, ki in zip(state, k, strict=True))  # type: ignore[return-value]


def _rk4_step(
    state: tuple[Array, Array, Array, Array],
    f: Array,
    M: Array,
    mrotor: Multirotor,
    wind: Array,
    dt: float,
) -> tuple[Array, Array, Array, Array]:
    """One fixed-step RK4 update with the wrench held constant; ``R`` re-projected."""
    k1 = _deriv(state, f, M, mrotor, wind)
    k2 = _deriv(_axpy(state, k1, dt / 2), f, M, mrotor, wind)
    k3 = _deriv(_axpy(state, k2, dt / 2), f, M, mrotor, wind)
    k4 = _deriv(_axpy(state, k3, dt), f, M, mrotor, wind)
    x, v, R, Omega = state
    x = x + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    v = v + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    R = R + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    Omega = Omega + dt / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
    return x, v, _project_so3(R), Omega


class FlightHistory(eqx.Module):
    """Vehicle + per-rotor kinematic history -- the hand-off to the acoustics stage.

    This is **the** documented interface consumed by the CONA aeroacoustic
    modules (``docs/research/cona-reference.md`` module 1 output). All quantities
    are SI in the **world z-up** frame unless stated as body-frame.

    Attributes:
        t: Sample times [s], shape ``[T]`` (uniform grid; entry ``n`` is the
            state at ``t[n]``, with the rotor speeds/thrusts applied over the
            following step).
        x: Vehicle CG position, world frame [m], shape ``[T, 3]``.
        v: Vehicle CG velocity, world frame [m/s], shape ``[T, 3]``.
        R: Attitude (world <- body) [-], shape ``[T, 3, 3]``; column ``k`` is body
            axis ``k`` in world coordinates (column 2 is the thrust axis).
        Omega_body: Body angular rate, **body frame** [rad/s], shape ``[T, 3]``.
        rotor_speeds: Per-rotor speed **magnitude** [rad/s], shape ``[T, Nr]``,
            always ``>= 0``. The signed rotor rate is
            ``spin_signs[i] * rotor_speeds[:, i]``; ``spin_signs`` and hub
            placements come from the vehicle (:class:`Multirotor` /
            :class:`~auraflow.core.blade.Vehicle`). This is what the aeroacoustic
            stage integrates into blade azimuth via
            :func:`auraflow.core.frames.integrate_azimuth`.
        rotor_thrusts: Per-rotor thrust [N], shape ``[T, Nr]`` (``= k_f *
            rotor_speeds^2``), body ``+z`` direction.
    """

    t: Array
    x: Array
    v: Array
    R: Array
    Omega_body: Array
    rotor_speeds: Array
    rotor_thrusts: Array


def simulate(
    mrotor: Multirotor,
    gains: ControllerGains,
    reference: Reference,
    t: ArrayLike,
    x0: ArrayLike,
    v0: ArrayLike,
    R0: ArrayLike | None = None,
    Omega0: ArrayLike | None = None,
    wind: ArrayLike | None = None,
) -> FlightHistory:
    r"""Closed-loop 6-DOF simulation over a uniform time grid.

    Marches the rigid-body dynamics under the geometric controller with a
    fixed-step RK4 integrator inside :func:`jax.lax.scan`. At each step the
    controller is evaluated at the current state and the reference sampled at
    ``t[n]``; the resulting wrench is allocated to per-rotor thrusts (clamped
    ``>= 0``), mapped to rotor speeds (with optional motor lag), the *achieved*
    wrench recomputed from those thrusts, and the state advanced by one step
    with that wrench held constant.

    Args:
        mrotor: Vehicle mass/inertia/allocation properties.
        gains: Controller gains.
        reference: Trajectory generator ``t -> (x_d, v_d, a_d, b1d)`` (see
            :func:`hover`, :func:`straight_flyover`).
        t: Uniform time grid [s], shape ``[T]`` (``T >= 2``).
        x0: Initial world position [m], shape ``[3]``.
        v0: Initial world velocity [m/s], shape ``[3]``.
        R0: Initial attitude (world <- body), shape ``[3, 3]``; defaults to
            identity (level).
        Omega0: Initial body rate [rad/s], shape ``[3]``; defaults to zero.
        wind: World-frame wind velocity series [m/s], shape ``[T, 3]`` or
            ``None``. Only affects the dynamics when ``mrotor.drag_coeff > 0``
            (the additive-gust hook). Held constant (ZOH) over each step.

    Returns:
        A :class:`FlightHistory` with ``[T, ...]`` leaves; entry ``n`` is the
        state at ``t[n]`` and the rotor speeds/thrusts applied over
        ``[t[n], t[n+1]]``.
    """
    t = jnp.asarray(t, dtype=float)
    n_t = t.shape[0]
    dt = float(t[1] - t[0])
    x0 = jnp.asarray(x0, dtype=float)
    v0 = jnp.asarray(v0, dtype=float)
    R0 = jnp.eye(3) if R0 is None else jnp.asarray(R0, dtype=float)
    Omega0 = jnp.zeros(3) if Omega0 is None else jnp.asarray(Omega0, dtype=float)
    if wind is None:
        wind_series = jnp.zeros((n_t, 3))
    else:
        wind_series = jnp.broadcast_to(jnp.asarray(wind, dtype=float), (n_t, 3))

    speed0 = mrotor.hover_omega() * jnp.ones(mrotor.n_rotors)
    tau = mrotor.motor_tau
    lag = 0.0 if tau is None else float(jnp.exp(-dt / tau))

    def step(
        carry: tuple[tuple[Array, Array, Array, Array], Array],
        inp: tuple[Array, Array],
    ):
        state, speed_prev = carry
        t_n, wind_n = inp
        ref_n = reference(t_n)
        f_cmd, M_cmd = geometric_controller(state, ref_n, mrotor, gains)
        f_i_cmd = allocate_thrusts(f_cmd, M_cmd, mrotor)
        speed_cmd = _thrust_to_speed(f_i_cmd, mrotor)
        # First-order motor lag (exact ZOH of dOmega/dt = (cmd - Omega)/tau).
        speed = speed_cmd if tau is None else speed_cmd + (speed_prev - speed_cmd) * lag
        thrust = mrotor.k_f * speed**2
        # Achieved wrench from the actual per-rotor thrusts.
        wrench = mrotor.mixing_matrix() @ thrust
        f_act, M_act = wrench[0], wrench[1:]
        new_state = _rk4_step(state, f_act, M_act, mrotor, wind_n, dt)
        x, v, R, Omega = state
        out = (x, v, R, Omega, speed, thrust)
        return (new_state, speed), out

    init = ((x0, v0, R0, Omega0), speed0)
    _, outs = jax.lax.scan(step, init, (t, wind_series))
    x_h, v_h, R_h, Omega_h, speed_h, thrust_h = outs
    return FlightHistory(
        t=t,
        x=x_h,
        v=v_h,
        R=R_h,
        Omega_body=Omega_h,
        rotor_speeds=speed_h,
        rotor_thrusts=thrust_h,
    )


# --------------------------------------------------------------------------- #
# Trajectory generators
# --------------------------------------------------------------------------- #
def hover(x0: ArrayLike, heading: float = 0.0) -> Reference:
    """Stationary-hover reference at a fixed point.

    Args:
        x0: Hover position, world frame [m], shape ``[3]``.
        heading: Constant heading angle [rad] about world ``+z`` (yaw); sets the
            commanded body ``+x`` direction ``b1d = [cos, sin, 0]``.

    Returns:
        A reference ``t -> (x_d, v_d, a_d, b1d)`` with constant position, zero
        velocity/acceleration.
    """
    x0 = jnp.asarray(x0, dtype=float)
    b1d = jnp.array([jnp.cos(heading), jnp.sin(heading), 0.0])
    zero = jnp.zeros(3)

    def ref(t: Array) -> tuple[Array, Array, Array, Array]:
        del t
        return x0, zero, zero, b1d

    return ref


def straight_flyover(
    speed: ArrayLike,
    altitude: ArrayLike,
    heading: ArrayLike = 0.0,
    t_pass: ArrayLike = 0.0,
    origin_xy: ArrayLike | tuple[float, float] = (0.0, 0.0),
) -> Reference:
    """Constant-velocity level straight-line flyover (the JASA scenario).

    The vehicle flies a straight, constant-altitude, constant-speed line and
    passes over ``origin_xy`` at time ``t_pass`` (``docs/research/
    jasa-datagen-reference.md``: level edgewise flight over a ground mic array).

    Args:
        speed: Ground speed [m/s], scalar.
        altitude: Constant flight altitude (world ``z``) [m], scalar.
        heading: Flight heading [rad] about world ``+z`` (0 = along ``+x``).
        t_pass: Time at which the track passes over ``origin_xy`` [s].
        origin_xy: World ``(x, y)`` the track passes over at ``t_pass`` [m].

    Returns:
        A reference ``t -> (x_d, v_d, a_d, b1d)`` with straight-line position,
        constant velocity, zero acceleration, heading along the velocity.
    """
    speed = jnp.asarray(speed, dtype=float)
    altitude = jnp.asarray(altitude, dtype=float)
    heading = jnp.asarray(heading, dtype=float)
    t_pass = jnp.asarray(t_pass, dtype=float)
    ox, oy = jnp.asarray(origin_xy, dtype=float)
    dir_xy = jnp.array([jnp.cos(heading), jnp.sin(heading), 0.0])
    v_d = speed * dir_xy
    a_d = jnp.zeros(3)
    p_pass = jnp.array([ox, oy, altitude])

    def ref(t: Array) -> tuple[Array, Array, Array, Array]:
        x_d = p_pass + v_d * (t - t_pass)
        return x_d, v_d, a_d, dir_xy

    return ref
