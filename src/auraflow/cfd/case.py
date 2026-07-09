"""Programmatic JAX-Fluids case and numerical-setup builders for acoustics.

JAX-Fluids is configured by two nested Python dicts -- a *case setup* (domain,
boundary conditions, initial condition, material, forcings) and a *numerical
setup* (fluxes, time integration, precision). Its ``InputManager`` accepts these
dicts directly (no JSON files on disk); string-valued fields are stringified
lambdas that JAX-Fluids ``eval``s with ``jax.numpy`` (``jnp``) in scope, so the
lambda bodies here use ``jnp`` and take exactly the *active* spatial axes as
arguments (plus ``t`` for forcings).

This module builds acoustics-oriented cases:

- :func:`acoustic_box_case` -- a quiescent-air cube with **sponge-layer**
  absorbing boundaries (JAX-Fluids has no non-reflecting/NSCBC BC -- see
  ``docs/research/jaxfluids-evaluation.md``), optionally seeded with a Gaussian
  pressure pulse for validating the CFD -> FW-H chain against linear acoustics.
- :func:`rotor_box_case` -- a box sized for a rotor of radius ``R``. The
  **actuator-disk momentum source** (a ``custom_forcing`` body force that only
  needs ``(x, y, z, t)``) is the implemented primary path per the evaluation
  digest; **resolved moving-levelset blades** are documented but raise
  ``NotImplementedError`` (see the function docstring for what is required).

Plus resolution / timestep helpers with acoustic points-per-wavelength and CFL
guidance (:func:`resolution_for_frequency`, :func:`acoustic_timestep`,
:func:`points_per_wavelength`).

All quantities are SI. The default JAX-Fluids nondimensionalization (all
reference values = 1) is used, so every value passed in and every primitive read
back is dimensional SI. Frames follow ``docs/architecture.md`` (world frame,
``z`` up).
"""

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from jax import Array

from auraflow.core.medium import Medium

__all__ = [
    "BoxDomain",
    "CFDCase",
    "acoustic_box_case",
    "acoustic_timestep",
    "points_per_wavelength",
    "resolution_for_frequency",
    "rotor_box_case",
]

# Ratio of specific heats and derived gas constant for the ideal-gas EOS. R is
# fixed by (c0, rho0, p0) via c0^2 = gamma p0 / rho0 and p0 = rho0 R T so that
# JAX-Fluids' ideal gas reproduces the requested ambient sound speed exactly.
_GAMMA = 1.4


def points_per_wavelength(dx: float, frequency: float, c0: float) -> float:
    """Acoustic spatial resolution in points per wavelength.

    Args:
        dx: Grid spacing [m].
        frequency: Target acoustic frequency [Hz].
        c0: Speed of sound [m/s].

    Returns:
        ``c0 / (frequency * dx)`` -- cells spanning one wavelength. Low-dispersion
        WENO transport of acoustics typically wants ``>= 8``-``16``.
    """
    return c0 / (frequency * dx)


def resolution_for_frequency(frequency: float, c0: float, ppw: float = 12.0) -> float:
    """Grid spacing that resolves a target frequency at ``ppw`` points/wavelength.

    Args:
        frequency: Highest acoustic frequency to resolve [Hz].
        c0: Speed of sound [m/s].
        ppw: Desired points per wavelength (default 12).

    Returns:
        Cell size ``dx = c0 / (frequency * ppw)`` [m].
    """
    return c0 / (frequency * ppw)


def acoustic_timestep(dx: float, c0: float, mach_max: float = 0.0, cfl: float = 0.5) -> float:
    """Fixed acoustic-CFL timestep for a uniform grid.

    Compressible JAX-Fluids is limited by the *acoustic* CFL: the fastest signal
    speed is ``c0 (1 + mach_max)``. A fixed timestep (no adaptivity) is required
    for the differentiable rollout (``docs/research/jaxfluids-evaluation.md``).

    Args:
        dx: Grid spacing [m].
        c0: Speed of sound [m/s].
        mach_max: Peak flow Mach number in the domain (0 for quiescent).
        cfl: CFL number (default 0.5; keep conservative for accuracy).

    Returns:
        Timestep ``dt = cfl * dx / (c0 (1 + mach_max))`` [s].
    """
    return cfl * dx / (c0 * (1.0 + mach_max))


@dataclass(frozen=True)
class BoxDomain:
    """Uniform Cartesian box for a JAX-Fluids run.

    Axes with a single cell are *inactive* (JAX-Fluids treats the run as 2-D/1-D
    and the initial-condition/forcing lambdas take only the active axes).

    Attributes:
        x_range: ``(x_min, x_max)`` [m].
        y_range: ``(y_min, y_max)`` [m].
        z_range: ``(z_min, z_max)`` [m].
        cells: Cell counts ``(nx, ny, nz)`` (static ints).
    """

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    cells: tuple[int, int, int]

    @property
    def active_axes(self) -> tuple[str, ...]:
        """Names of the axes with more than one cell, in ``x, y, z`` order."""
        names = ("x", "y", "z")
        return tuple(n for n, c in zip(names, self.cells, strict=True) if c > 1)

    def spacing(self) -> tuple[float, float, float]:
        """Cell sizes ``(dx, dy, dz)`` [m]."""
        ranges = (self.x_range, self.y_range, self.z_range)
        return tuple(  # type: ignore[return-value]
            (hi - lo) / n for (lo, hi), n in zip(ranges, self.cells, strict=True)
        )

    def cell_centers(self) -> tuple[Array, Array, Array]:
        """Cell-centre coordinate arrays ``(x, y, z)``, shapes ``(nx, ny, nz)``.

        These match the interior primitive array returned by the driver and are
        the grids passed to :func:`auraflow.cfd.sphere.sample_primitives`.
        """
        ranges = (self.x_range, self.y_range, self.z_range)
        axes = []
        for (lo, hi), n in zip(ranges, self.cells, strict=True):
            d = (hi - lo) / n
            axes.append(jnp.linspace(lo + 0.5 * d, hi - 0.5 * d, n))
        return axes[0], axes[1], axes[2]


@dataclass(frozen=True)
class CFDCase:
    """A ready-to-run JAX-Fluids acoustics case.

    Bundles the two JAX-Fluids setup dicts with the metadata the driver
    (:mod:`auraflow.cfd.run`) needs to sample the permeable sphere and convert to
    the FW-H time grid.

    Attributes:
        case: JAX-Fluids *case setup* dict.
        numerical_setup: JAX-Fluids *numerical setup* dict.
        domain: The :class:`BoxDomain` (for reconstructing the sampling grid).
        dt: Fixed timestep [s].
        medium: Ambient :class:`~auraflow.core.medium.Medium`.
    """

    case: dict[str, Any]
    numerical_setup: dict[str, Any]
    domain: BoxDomain
    dt: float
    medium: Medium = field(default_factory=Medium)


def _gas_constant(medium: Medium) -> float:
    """Specific gas constant ``R`` consistent with ``(c0, rho0)`` at ``gamma``.

    From ``c0^2 = gamma p / rho`` and ``p = rho R T`` we get ``R T = c0^2/gamma``;
    JAX-Fluids' ideal gas only needs ``(gamma, R)`` plus the initial ``(rho, p)``,
    so we return ``R`` implied by ``p0 = rho0 R T0`` with ``R T0 = c0^2/gamma``,
    i.e. ``R = p0 / (rho0 T0)`` where ``T0 = c0^2 / (gamma R)`` -- eliminating
    ``T0`` gives ``R`` free, so we fix ``T0 = 288.15`` and return the matching R.
    """
    # Ideal gas: T0 chosen at ISA 288.15 K, R follows from p0 = rho0 R T0.
    t0 = 288.15
    return float(medium.p0) / (float(medium.rho0) * t0)


def _numerical_setup(dt: float, is_double: bool = False) -> dict[str, Any]:
    """WENO5-Z / HLLC / RK3 inviscid compressible numerical setup.

    Fixed-timestep, single-precision-compute by default (converted to float64 at
    the FW-H boundary). Inviscid: dissipative fluxes are inactive, so the whole
    ``dissipative_fluxes`` block is omitted.
    """
    return {
        "conservatives": {
            "halo_cells": 3,
            "time_integration": {
                "integrator": "RK3",
                "CFL": 0.9,
                "fixed_timestep": float(dt),
            },
            "convective_fluxes": {
                "convective_solver": "GODUNOV",
                "godunov": {
                    "riemann_solver": "HLLC",
                    "signal_speed": "EINFELDT",
                    "reconstruction_stencil": "WENO5-Z",
                    "reconstruction_variable": "PRIMITIVE",
                },
            },
        },
        "active_physics": {
            "is_convective_flux": True,
            "is_viscous_flux": False,
            "is_heat_flux": False,
        },
        "precision": {
            "is_double_precision_compute": bool(is_double),
            "is_double_precision_output": True,
        },
        "output": {"is_active": False, "is_xdmf": False},
    }


def _material_properties(medium: Medium) -> dict[str, Any]:
    """Ideal-gas material block reproducing the medium's ``(c0, rho0, p0)``."""
    return {
        "equation_of_state": {
            "model": "IdealGas",
            "specific_heat_ratio": _GAMMA,
            "specific_gas_constant": _gas_constant(medium),
        },
        "transport": {
            "dynamic_viscosity": {"model": "CUSTOM", "value": 0.0},
            "bulk_viscosity": 0.0,
            "thermal_conductivity": {"model": "CUSTOM", "value": 0.0},
        },
    }


def _quiescent_primitives(medium: Medium, active_axes: tuple[str, ...]) -> dict[str, Any]:
    """Uniform ambient initial condition (rho0, 0, 0, 0, p0) as float literals."""
    return {
        "rho": float(medium.rho0),
        "u": 0.0,
        "v": 0.0,
        "w": 0.0,
        "p": float(medium.p0),
    }


def _pulse_primitives(
    medium: Medium,
    amplitude: float,
    width: float,
    center: tuple[float, float, float],
    active_axes: tuple[str, ...],
) -> dict[str, Any]:
    """Gaussian pressure-pulse initial condition as stringified ``jnp`` lambdas.

    Isentropic pulse: ``p = p0 + A g``, ``rho = rho0 + A g / c0^2``, ``u = 0``,
    where ``g = exp(-|x - x_c|^2 / (2 w^2))`` over the active axes. Matching the
    density keeps the perturbation acoustic (no entropy spot), so the far field
    is the clean linear-acoustics pulse used by the validation script.
    """
    args = ", ".join(active_axes)
    axis_index = {"x": 0, "y": 1, "z": 2}
    sq = " + ".join(f"({a} - {center[axis_index[a]]!r})**2" for a in active_axes)
    gauss = f"jnp.exp(-({sq}) / (2.0 * {float(width)!r}**2))"
    rho0 = float(medium.rho0)
    p0 = float(medium.p0)
    c0 = float(medium.c0)
    amp = float(amplitude)
    return {
        "rho": f"lambda {args}: {rho0!r} + {amp!r} * {gauss} / {c0!r}**2",
        "u": 0.0,
        "v": 0.0,
        "w": 0.0,
        "p": f"lambda {args}: {p0!r} + {amp!r} * {gauss}",
    }


def _sponge_strength_lambda(domain: BoxDomain, thickness: float, sigma_max: float) -> str:
    """Sponge strength ``sigma(x[,y[,z]], t)`` ramping to ``sigma_max`` near faces.

    Quadratic ramp over ``thickness`` metres inward from every active outer face,
    combined across axes by ``jnp.maximum``. Returned as a stringified lambda
    whose argument names are exactly ``(*active_axes, "t")`` (JAX-Fluids requires
    the lambda's varnames to match).

    JAX-Fluids applies the sponge as ``dq/dt = -(strength/dt) (q - q_ref)``, i.e.
    ``strength`` is the **dimensionless fraction** of the local error relaxed per
    timestep. It must stay in ``[0, 1]`` (RK stages apply it repeatedly, so keep
    ``sigma_max`` well below 1 for stability); this is enforced by the final
    ``jnp.clip``.
    """
    active = domain.active_axes
    args = ", ".join(active + ("t",))
    ranges = {"x": domain.x_range, "y": domain.y_range, "z": domain.z_range}
    terms = []
    for a in active:
        lo, hi = ranges[a]
        lo_edge = lo + thickness
        hi_edge = hi - thickness
        lo_ramp = f"jnp.where({a} < {lo_edge!r}, (({lo_edge!r} - {a}) / {thickness!r})**2, 0.0)"
        hi_ramp = f"jnp.where({a} > {hi_edge!r}, (({a} - {hi_edge!r}) / {thickness!r})**2, 0.0)"
        terms.append(f"jnp.maximum({lo_ramp}, {hi_ramp})")
    combined = terms[0]
    for t in terms[1:]:
        combined = f"jnp.maximum({combined}, {t})"
    return f"lambda {args}: {sigma_max!r} * jnp.clip({combined}, 0.0, 1.0)"


def _sponge_forcing(
    medium: Medium, domain: BoxDomain, thickness: float, sigma_max: float
) -> dict[str, Any]:
    """Sponge-layer forcing block: relax primitives to ambient near the faces."""
    return {
        "sponge_layer": {
            "primitives": {
                "rho": float(medium.rho0),
                "u": 0.0,
                "v": 0.0,
                "w": 0.0,
                "p": float(medium.p0),
            },
            "strength": _sponge_strength_lambda(domain, thickness, sigma_max),
        }
    }


def _general(case_name: str, end_time: float) -> dict[str, Any]:
    """General block. We drive our own loop, but InputManager validates these."""
    return {
        "case_name": case_name,
        "end_time": float(end_time),
        "save_dt": float(end_time),  # single dump at end if simulate() is ever used
        "save_path": "./results",
    }


def _outer_bcs(domain: BoxDomain) -> dict[str, Any]:
    """Boundary conditions: zero-gradient (outflow) on active-axis faces.

    Faces on an inactive axis (single-cell, 2-D/1-D runs) must be ``INACTIVE``
    per JAX-Fluids; the sponge layer absorbs reflections at the active faces.
    """
    face_axis = {
        "east": "x",
        "west": "x",
        "north": "y",
        "south": "y",
        "top": "z",
        "bottom": "z",
    }
    active = domain.active_axes
    return {
        face: {"type": "ZEROGRADIENT" if axis in active else "INACTIVE"}
        for face, axis in face_axis.items()
    }


def acoustic_box_case(
    medium: Medium | None = None,
    *,
    half_size: float = 1.0,
    cells: tuple[int, int, int] = (48, 48, 48),
    cfl: float = 0.5,
    end_time: float | None = None,
    sponge_thickness: float | None = None,
    sponge_sigma: float | None = None,
    pulse: bool = False,
    pulse_amplitude: float = 10.0,
    pulse_width: float = 0.05,
    pulse_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    is_double: bool = False,
    case_name: str = "acoustic_box",
) -> CFDCase:
    """Quiescent-air box with sponge boundaries and an optional pressure pulse.

    The box is centred on the origin, spanning ``[-half_size, half_size]`` on each
    active axis. Any axis whose cell count is 1 is inactive (2-D/1-D run). The
    outer faces are zero-gradient and a sponge layer relaxes the solution to
    ambient near them (JAX-Fluids has no non-reflecting BC).

    Args:
        medium: Ambient medium (default sea-level ISA). Supplies ``rho0``, ``p0``,
            ``c0`` for the ideal-gas material and sponge/initial states.
        half_size: Half box edge length [m].
        cells: ``(nx, ny, nz)`` cell counts. Use e.g. ``(48, 48, 48)`` locally,
            ``(96, 1, 96)`` for a cheap 2-D pulse test.
        cfl: Acoustic CFL for the fixed timestep.
        end_time: Physical end time [s] (default: time for sound to cross the box,
            ``2 half_size / c0``). Only used by ``InputManager`` validation and
            the optional ``simulate()`` path; the driver marches a fixed step
            count.
        sponge_thickness: Sponge layer depth [m] (default ``0.25 half_size``).
        sponge_sigma: Peak sponge strength -- the dimensionless fraction of the
            local error relaxed per timestep at the outer edge, in ``[0, 1]``
            (default ``0.5``).
        pulse: If True, seed a Gaussian pressure pulse (for CFD->FW-H validation).
        pulse_amplitude: Pulse peak overpressure ``A`` [Pa].
        pulse_width: Gaussian standard deviation ``w`` [m].
        pulse_center: Pulse centre [m], shape ``[3]``.
        is_double: Use float64 compute (default float32; FW-H upcasts anyway).
        case_name: JAX-Fluids case name.

    Returns:
        A :class:`CFDCase`.
    """
    medium = Medium() if medium is None else medium
    domain = BoxDomain(
        x_range=(-half_size, half_size),
        y_range=(-half_size, half_size),
        z_range=(-half_size, half_size),
        cells=cells,
    )
    dx = min(domain.spacing()[i] for i, c in enumerate(cells) if c > 1)
    dt = acoustic_timestep(dx, float(medium.c0), mach_max=0.0, cfl=cfl)
    if end_time is None:
        end_time = 2.0 * half_size / float(medium.c0)
    if sponge_thickness is None:
        sponge_thickness = 0.25 * half_size
    if sponge_sigma is None:
        sponge_sigma = 0.5

    active = domain.active_axes
    ic = (
        _pulse_primitives(medium, pulse_amplitude, pulse_width, pulse_center, active)
        if pulse
        else _quiescent_primitives(medium, active)
    )
    case = {
        "general": _general(case_name, end_time),
        "domain": {
            "x": {"cells": cells[0], "range": list(domain.x_range)},
            "y": {"cells": cells[1], "range": list(domain.y_range)},
            "z": {"cells": cells[2], "range": list(domain.z_range)},
        },
        "boundary_conditions": _outer_bcs(domain),
        "initial_condition": {"primitives": ic},
        "material_properties": _material_properties(medium),
        "forcings": _sponge_forcing(medium, domain, sponge_thickness, sponge_sigma),
    }
    numerical = _numerical_setup(dt, is_double=is_double)
    numerical["active_forcings"] = {"is_sponge_layer_forcing": True}
    return CFDCase(case=case, numerical_setup=numerical, domain=domain, dt=dt, medium=medium)


def _actuator_disk_forcing_lambda(
    thrust_per_area: float,
    hub_center: tuple[float, float, float],
    radius: float,
    disk_thickness: float,
    axis: str,
    active_axes: tuple[str, ...],
) -> str:
    """Stringified momentum-source lambda for a steady actuator disk.

    Adds a body force (momentum source, ``d(rho u_axis)/dt``) of magnitude
    ``thrust_per_area`` distributed over a thin disk of radius ``radius`` and
    thickness ``disk_thickness`` centred at ``hub_center``, oriented along
    ``axis`` (the thrust axis, ``'z'`` by convention). Only ``(x, y, z, t)`` are
    needed, so this is expressible as a JAX-Fluids ``custom_forcing`` per the
    evaluation digest.
    """
    args = ", ".join(active_axes + ("t",))
    axis_index = {"x": 0, "y": 1, "z": 2}
    in_plane = [a for a in active_axes if a != axis]
    r2 = " + ".join(f"({a} - {hub_center[axis_index[a]]!r})**2" for a in in_plane) or "0.0"
    ax = hub_center[axis_index[axis]]
    disk = (
        f"jnp.where(({r2}) <= {radius!r}**2, "
        f"jnp.where(jnp.abs({axis} - {ax!r}) <= {0.5 * disk_thickness!r}, "
        f"{thrust_per_area / disk_thickness!r}, 0.0), 0.0)"
    )
    return f"lambda {args}: {disk}"


def rotor_box_case(
    medium: Medium | None = None,
    *,
    rotor_radius: float,
    box_radii: float = 4.0,
    cells: tuple[int, int, int] = (48, 48, 48),
    hub_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    thrust: float = 1.0,
    disk_thickness: float | None = None,
    thrust_axis: str = "z",
    cfl: float = 0.5,
    tip_mach: float = 0.3,
    end_time: float | None = None,
    method: str = "actuator_disk",
    is_double: bool = False,
    case_name: str = "rotor_box",
) -> CFDCase:
    """Box sized for a rotor, with an actuator-disk momentum source.

    The domain is a cube of half-edge ``box_radii * rotor_radius`` centred on the
    hub, with sponge-absorbing boundaries. Two rotor representations are
    considered (``docs/research/jaxfluids-evaluation.md``):

    - ``method="actuator_disk"`` (**implemented**): a steady body-force disk added
      as a JAX-Fluids ``custom_forcing``. Custom-forcing callables receive only
      ``(x, y, z, t)`` (not the flow state), which is exactly enough for a
      prescribed-thrust disk. This is the primary path for JASA-style rotor runs.
    - ``method="levelset_blades"`` (**deferred**): resolved rotating blades via a
      moving FLUID-SOLID level set. The level-set machinery now exists
      (:func:`auraflow.cfd.body_case.levelset_body_case` builds a FLUID-SOLID case
      for any body :class:`~auraflow.body.mesh.TriMesh`, including a constant-rate
      :class:`~auraflow.body.motion.SpinMotion` prescribed solid velocity); what
      is missing is generating watertight blade meshes from a
      :class:`~auraflow.core.blade.BladeGeometry`. Until that exists this raises
      ``NotImplementedError`` pointing at :func:`levelset_body_case`.

    Args:
        medium: Ambient medium (default sea-level ISA).
        rotor_radius: Rotor tip radius ``R`` [m].
        box_radii: Half box edge in units of ``R`` (default 4 -- keep the sphere
            enclosing the tip vortices well inside the sponge-free core).
        cells: ``(nx, ny, nz)`` cell counts.
        hub_center: Hub position [m], shape ``[3]``.
        thrust: Total disk thrust [N] (spread over the disk area).
        disk_thickness: Actuator-disk axial thickness [m] (default 4 cells).
        thrust_axis: Thrust axis, one of ``"x"``, ``"y"``, ``"z"`` (default z).
        cfl: Acoustic CFL for the fixed timestep.
        tip_mach: Expected tip Mach number, used only to bound the timestep.
        end_time: Physical end time [s] (default: several box-crossing times).
        method: ``"actuator_disk"`` (implemented) or ``"levelset_blades"`` (stub).
        is_double: Use float64 compute.
        case_name: JAX-Fluids case name.

    Returns:
        A :class:`CFDCase` for ``method="actuator_disk"``.

    Raises:
        NotImplementedError: for ``method="levelset_blades"``.
        ValueError: for an unknown ``method`` or ``thrust_axis``.
    """
    medium = Medium() if medium is None else medium
    if thrust_axis not in ("x", "y", "z"):
        raise ValueError(f"thrust_axis must be x/y/z, got {thrust_axis!r}")
    half = box_radii * rotor_radius
    cx, cy, cz = hub_center
    domain = BoxDomain(
        x_range=(cx - half, cx + half),
        y_range=(cy - half, cy + half),
        z_range=(cz - half, cz + half),
        cells=cells,
    )
    dx = min(domain.spacing()[i] for i, c in enumerate(cells) if c > 1)
    dt = acoustic_timestep(dx, float(medium.c0), mach_max=tip_mach, cfl=cfl)
    if end_time is None:
        end_time = 4.0 * half / float(medium.c0)
    if disk_thickness is None:
        disk_thickness = 4.0 * dx

    if method == "levelset_blades":
        raise NotImplementedError(
            "Resolved moving-levelset blades are deferred. The FLUID-SOLID "
            "level-set machinery exists -- use "
            "auraflow.cfd.body_case.levelset_body_case(blade_mesh, "
            "SpinMotion.constant(axis, omega, center=hub), box_lo=..., box_hi=..., "
            "cells=...) to immerse a rotating blade body (it builds the level-set "
            "field, ONE-WAY solid coupling, and the prescribed rotational solid "
            "velocity). What is still missing is generating watertight blade "
            "TriMeshes from a core.blade.BladeGeometry (future work); this is "
            "resolution-hungry and intended for GPU runs. Use "
            "method='actuator_disk' for the primary rotor-noise path."
        )
    if method != "actuator_disk":
        raise ValueError(f"unknown method {method!r}")

    active = domain.active_axes
    disk_area = jnp.pi * rotor_radius**2
    thrust_per_area = float(thrust) / float(disk_area)
    forcing_lambda = _actuator_disk_forcing_lambda(
        thrust_per_area, hub_center, rotor_radius, disk_thickness, thrust_axis, active
    )
    momentum = {"x": "u", "y": "v", "z": "w"}[thrust_axis]
    custom: dict[str, Any] = {"rho": 0.0, "u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0}
    custom[momentum] = forcing_lambda

    sponge_thickness = 0.5 * rotor_radius
    sponge_sigma = 0.5
    case = {
        "general": _general(case_name, end_time),
        "domain": {
            "x": {"cells": cells[0], "range": list(domain.x_range)},
            "y": {"cells": cells[1], "range": list(domain.y_range)},
            "z": {"cells": cells[2], "range": list(domain.z_range)},
        },
        "boundary_conditions": _outer_bcs(domain),
        "initial_condition": {"primitives": _quiescent_primitives(medium, active)},
        "material_properties": _material_properties(medium),
        "forcings": {
            "custom_forcing": custom,
            **_sponge_forcing(medium, domain, sponge_thickness, sponge_sigma),
        },
    }
    numerical = _numerical_setup(dt, is_double=is_double)
    numerical["active_forcings"] = {
        "is_custom_forcing": True,
        "is_sponge_layer_forcing": True,
    }
    return CFDCase(case=case, numerical_setup=numerical, domain=domain, dt=dt, medium=medium)
