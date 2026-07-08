"""Acoustic medium properties.

Defines the :class:`Medium` equinox module holding the ambient-fluid quantities
that every acoustic backend needs (density, sound speed, ambient pressure,
kinematic viscosity), plus an ISA standard-atmosphere constructor.

All quantities are SI: kg/m^3, m/s, Pa, m^2/s.
"""

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

# --- ISA / air constants (SI) -------------------------------------------------
_T0 = 288.15  # sea-level temperature [K]
_P0 = 101325.0  # sea-level pressure [Pa]
_LAPSE = 0.0065  # tropospheric temperature lapse rate [K/m]
_G = 9.80665  # gravitational acceleration [m/s^2]
_R_AIR = 287.05287  # specific gas constant of dry air [J/(kg K)]
_GAMMA = 1.4  # ratio of specific heats of air [-]
# Sutherland's law for dry air:
_MU_REF = 1.716e-5  # dynamic viscosity at T_ref [Pa s]
_T_REF = 273.15  # Sutherland reference temperature [K]
_S_SUTH = 110.4  # Sutherland constant [K]


class Medium(eqx.Module):
    """Ambient acoustic medium (quiescent fluid properties).

    All fields are JAX arrays (scalars) so they can be traced and
    differentiated through, e.g. for sensitivity of noise metrics to
    atmospheric conditions.

    Attributes:
        rho0: Ambient density [kg/m^3].
        c0: Speed of sound [m/s].
        p0: Ambient (static) pressure [Pa].
        nu: Kinematic viscosity [m^2/s].
    """

    rho0: Array
    c0: Array
    p0: Array
    nu: Array

    def __init__(
        self,
        rho0: ArrayLike = 1.225,
        c0: ArrayLike = 340.294,
        p0: ArrayLike = 101325.0,
        nu: ArrayLike = 1.4607e-5,
    ):
        """Construct a medium from explicit properties.

        Defaults correspond to the ISA standard atmosphere at sea level.

        Args:
            rho0: Ambient density [kg/m^3], scalar.
            c0: Speed of sound [m/s], scalar.
            p0: Ambient pressure [Pa], scalar.
            nu: Kinematic viscosity [m^2/s], scalar.
        """
        self.rho0 = jnp.asarray(rho0)
        self.c0 = jnp.asarray(c0)
        self.p0 = jnp.asarray(p0)
        self.nu = jnp.asarray(nu)

    @classmethod
    def standard_atmosphere(cls, altitude_m: ArrayLike = 0.0) -> "Medium":
        """ISA standard atmosphere at a given geopotential altitude.

        Uses the International Standard Atmosphere troposphere model
        (valid for ``0 <= altitude_m <= 11000``):

        - ``T = T0 - L h`` with ``T0 = 288.15 K``, ``L = 6.5 K/km``;
        - ``p = p0 (T/T0)^(g/(R L))`` with ``p0 = 101325 Pa``;
        - ``rho = p / (R T)`` (ideal gas, ``R = 287.05287 J/(kg K)``);
        - ``c = sqrt(gamma R T)`` with ``gamma = 1.4``;
        - dynamic viscosity from Sutherland's law
          ``mu = mu_ref (T/T_ref)^{3/2} (T_ref + S)/(T + S)``
          with ``mu_ref = 1.716e-5 Pa s``, ``T_ref = 273.15 K``, ``S = 110.4 K``;
          ``nu = mu / rho``.

        The mapping is smooth and differentiable with respect to ``altitude_m``.

        Args:
            altitude_m: Geopotential altitude above mean sea level [m], scalar.
                Values above the tropopause (11 km) are extrapolated with the
                same lapse rate and are physically inaccurate.

        Returns:
            A :class:`Medium` with ISA properties at that altitude.
        """
        h = jnp.asarray(altitude_m)
        temp = _T0 - _LAPSE * h
        p = _P0 * (temp / _T0) ** (_G / (_R_AIR * _LAPSE))
        rho = p / (_R_AIR * temp)
        c = jnp.sqrt(_GAMMA * _R_AIR * temp)
        mu = _MU_REF * (temp / _T_REF) ** 1.5 * (_T_REF + _S_SUTH) / (temp + _S_SUTH)
        return cls(rho0=rho, c0=c, p0=p, nu=mu / rho)
