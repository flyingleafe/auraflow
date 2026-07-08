"""Steady per-annulus BEMT: momentum-theory gate, Prandtl loss, gradients.

Imports the submodule directly so the whole ``auraflow.bemt`` package need not
be importable to run this file in isolation (per the low-RAM one-file-at-a-time
test policy).
"""

import jax
import jax.numpy as jnp
import numpy as np

from auraflow.bemt.solver import (
    prandtl_tip_root_loss,
    steady_bemt,
)
from auraflow.core.airfoil import TablePolar, ThinAirfoilPolar
from auraflow.core.blade import BladeGeometry, Rotor
from auraflow.core.medium import Medium


def _linear_polar():
    """Exact ``cl = 2 pi alpha``, ``cd = 0`` polar (no Mach/Re dependence)."""
    a = jnp.linspace(-0.4, 0.4, 81)
    return TablePolar(alpha_grid=a, cl_table=2.0 * jnp.pi * a, cd_table=jnp.zeros_like(a))


def _ideal_hover_rotor(chord=0.03, twist_deg=6.0, n_stations=24):
    blade = BladeGeometry.linear(
        radius=1.0,
        hub_radius=0.1,
        n_stations=n_stations,
        chord_root=chord,
        chord_tip=chord,
        twist_root=np.radians(twist_deg),
        twist_tip=np.radians(twist_deg),
    )
    return Rotor(blade=blade, n_blades=2)


def _closed_form_inflow(rotor, twist, a=2.0 * np.pi):
    """Leishman swirl-free combined BEMT hover inflow ratio lambda(r/R)."""
    R = float(rotor.blade.radius)
    c = float(rotor.blade.chord[0])
    B = rotor.n_blades
    sigma = B * c / (np.pi * R)  # constant-chord rotor solidity
    rho = np.asarray(rotor.blade.r) / R
    lam = (sigma * a / 16.0) * (np.sqrt(1.0 + 32.0 * twist * rho / (sigma * a)) - 1.0)
    return rho, lam


class TestMomentumGate:
    def test_induced_velocity_matches_closed_form(self):
        rotor = _ideal_hover_rotor()
        med = Medium()
        omega = 120.0
        twist = float(rotor.blade.twist[0])
        loads = steady_bemt(
            rotor,
            med,
            omega,
            v_climb=0.0,
            collective=0.0,
            polar=_linear_polar(),
            tip_loss=False,
            root_loss=False,
        )
        rho, lam_ref = _closed_form_inflow(rotor, twist)
        vtip = omega * float(rotor.blade.radius)
        lam_bemt = np.asarray(loads.annulus.inflow_axial) / vtip
        # Compare interior stations (edges carry the largest swirl/curvature gap).
        interior = (rho > 0.2) & (rho < 0.98)
        rel = np.abs(lam_bemt[interior] - lam_ref[interior]) / lam_ref[interior]
        assert float(rel.max()) < 5e-3

    def test_ct_matches_closed_form(self):
        rotor = _ideal_hover_rotor()
        med = Medium()
        omega = 120.0
        twist = float(rotor.blade.twist[0])
        loads = steady_bemt(
            rotor,
            med,
            omega,
            polar=_linear_polar(),
            tip_loss=False,
            root_loss=False,
        )
        rho, lam_ref = _closed_form_inflow(rotor, twist)
        # CT = integral 4 lambda^2 rho drho over the blade span.
        drho = np.asarray(rotor.blade.dr) / float(rotor.blade.radius)
        ct_ref = float(np.sum(4.0 * lam_ref**2 * rho * drho))
        assert abs(float(loads.ct) - ct_ref) / ct_ref < 5e-3


class TestPrandtlLoss:
    def test_limits_and_monotonicity(self):
        R, Rhub, B = 1.0, 0.1, 3
        r = jnp.linspace(0.11, 1.0 - 1e-4, 50)
        sinphi = jnp.full_like(r, np.sin(np.radians(5.0)))
        F = prandtl_tip_root_loss(r, R, Rhub, sinphi, B)
        F = np.asarray(F)
        # Inboard (mid-span) approaches 1.
        assert F[len(F) // 2] > 0.98
        # Approaches 0 at the tip.
        assert F[-1] < 0.05
        # Monotone decreasing over the outer 30% (tip region).
        outer = F[r > 0.7]
        assert np.all(np.diff(outer) < 1e-9)

    def test_tip_and_root_both_drop(self):
        R, Rhub, B = 1.0, 0.15, 2
        r = jnp.linspace(0.16, 0.999, 40)
        sinphi = jnp.full_like(r, np.sin(np.radians(6.0)))
        F = np.asarray(prandtl_tip_root_loss(r, R, Rhub, sinphi, B))
        assert F[0] < F[len(F) // 2]  # root loss pulls the innermost station down
        assert F[-1] < F[len(F) // 2]  # tip loss pulls the outermost down


class TestGradients:
    def _thrust_of_collective(self, collective):
        rotor = _ideal_hover_rotor(twist_deg=8.0)
        med = Medium()
        polar = ThinAirfoilPolar(cd0=0.01, k=0.02)
        return steady_bemt(rotor, med, 120.0, collective=collective, polar=polar).thrust

    def _thrust_of_chord_scale(self, scale):
        base = _ideal_hover_rotor(twist_deg=8.0)
        blade = BladeGeometry(
            radius=base.blade.radius,
            hub_radius=base.blade.hub_radius,
            chord=base.blade.chord * scale,
            twist=base.blade.twist,
        )
        rotor = Rotor(blade=blade, n_blades=base.n_blades)
        med = Medium()
        return steady_bemt(rotor, med, 120.0, polar=ThinAirfoilPolar(cd0=0.01, k=0.02)).thrust

    def test_dthrust_dcollective_matches_fd(self):
        g = jax.grad(self._thrust_of_collective)(jnp.asarray(0.05))
        h = 1e-5
        fd = (
            self._thrust_of_collective(jnp.asarray(0.05 + h))
            - self._thrust_of_collective(jnp.asarray(0.05 - h))
        ) / (2 * h)
        assert bool(jnp.isfinite(g))
        np.testing.assert_allclose(float(g), float(fd), rtol=1e-4)

    def test_dthrust_dchord_matches_fd(self):
        g = jax.grad(self._thrust_of_chord_scale)(jnp.asarray(1.0))
        h = 1e-5
        fd = (
            self._thrust_of_chord_scale(jnp.asarray(1.0 + h))
            - self._thrust_of_chord_scale(jnp.asarray(1.0 - h))
        ) / (2 * h)
        assert bool(jnp.isfinite(g))
        np.testing.assert_allclose(float(g), float(fd), rtol=1e-4)


class TestClimbSanity:
    def test_climb_reduces_thrust_at_fixed_pitch(self):
        rotor = _ideal_hover_rotor(twist_deg=8.0)
        med = Medium()
        polar = ThinAirfoilPolar(cd0=0.008, k=0.0)
        hover = steady_bemt(rotor, med, 120.0, v_climb=0.0, polar=polar)
        climb = steady_bemt(rotor, med, 120.0, v_climb=5.0, polar=polar)
        # Climb raises the inflow angle, lowers alpha, lowers thrust.
        assert float(climb.thrust) < float(hover.thrust)
        assert float(hover.thrust) > 0.0
