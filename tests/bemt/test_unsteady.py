"""Quasi-steady marching: shapes and regression on the predecessor bugs.

Bugs fixed (docs/research/fwh-rotor-sim-audit.md):
- (b) forces track time-varying Omega(t), not a frozen mean;
- (c) azimuth = integral of Omega, not Omega(t)*t;
- (a) BEMT-induced velocity enters the loads.
"""

import jax.numpy as jnp
import numpy as np

from auraflow.bemt.unsteady import march_bemt
from auraflow.core.airfoil import ThinAirfoilPolar
from auraflow.core.blade import BladeGeometry, Rotor
from auraflow.core.medium import Medium


def _rotor(n_stations=8, n_blades=2, twist_deg=10.0):
    blade = BladeGeometry.linear(
        radius=0.6,
        hub_radius=0.08,
        n_stations=n_stations,
        chord_root=0.05,
        chord_tip=0.035,
        twist_root=np.radians(twist_deg),
        twist_tip=np.radians(twist_deg - 4),
    )
    return Rotor(blade=blade, n_blades=n_blades)


class TestShapes:
    def test_leaf_shapes(self):
        rotor = _rotor()
        med = Medium()
        t = jnp.linspace(0.0, 0.05, 48)
        state = march_bemt(rotor, med, t, omega=300.0)
        b, s, tt = rotor.n_blades, rotor.blade.n_stations, t.shape[0]
        assert state.phi.shape == (b, s, tt)
        assert state.force_on_fluid.shape == (b, s, tt, 3)
        assert state.position.shape == (b, s, tt, 3)
        assert state.psi.shape == (b, tt)
        assert np.all(np.isfinite(np.asarray(state.force_on_fluid)))


class TestTimeVaryingOmega:
    def test_forces_track_instantaneous_omega(self):
        # Bug (b): with a ramping Omega, the thrust-direction force must vary in
        # time (~Omega^2), not stay frozen at the mean.
        rotor = _rotor()
        med = Medium()
        t = jnp.linspace(0.0, 0.1, 80)
        omega = jnp.linspace(200.0, 400.0, 80)
        state = march_bemt(rotor, med, t, omega=omega, polar=ThinAirfoilPolar(cd0=0.01))
        # Axial (thrust) force at a mid station, blade 0, over time.
        fz = np.asarray(-state.force_on_fluid[0, rotor.blade.n_stations // 2, :, 2])
        assert fz.std() / abs(fz.mean()) > 0.3  # strongly time-varying
        # Monotone-ish increase with omega: end force >> start force.
        assert fz[-1] > 2.5 * fz[0]

    def test_differs_from_frozen_mean(self):
        rotor = _rotor()
        med = Medium()
        t = jnp.linspace(0.0, 0.1, 80)
        omega = jnp.linspace(200.0, 400.0, 80)
        var = march_bemt(rotor, med, t, omega=omega)
        frozen = march_bemt(rotor, med, t, omega=float(jnp.mean(omega)))
        f_var = np.asarray(var.force_on_fluid[0, :, :, 2])
        f_frozen = np.asarray(frozen.force_on_fluid[0, :, :, 2])
        assert np.linalg.norm(f_var - f_frozen) / np.linalg.norm(f_frozen) > 0.1


class TestInducedVelocity:
    def test_induced_reduces_alpha_and_changes_loads(self):
        # Bug (a): induced velocity raises the inflow angle, lowers alpha and the
        # sectional lift, versus the no-induction case.
        rotor = _rotor()
        med = Medium()
        t = jnp.linspace(0.0, 0.03, 32)
        on = march_bemt(rotor, med, t, omega=300.0, include_induced=True)
        off = march_bemt(rotor, med, t, omega=300.0, include_induced=False)
        assert float(jnp.mean(on.alpha)) < float(jnp.mean(off.alpha))
        assert float(jnp.mean(on.v_axial)) > 0.0
        assert float(jnp.mean(off.v_axial)) == 0.0
        # Loads differ materially.
        rel = float(
            jnp.linalg.norm(on.lift_per_span - off.lift_per_span)
            / jnp.linalg.norm(off.lift_per_span)
        )
        assert rel > 0.1


class TestAzimuth:
    def test_azimuth_is_integral_not_product(self):
        # Bug (c): for a ramping Omega, psi(t) = int Omega dt, not Omega(t)*t.
        rotor = _rotor()
        med = Medium()
        t = jnp.linspace(0.0, 0.1, 50)
        omega = jnp.linspace(200.0, 400.0, 50)
        state = march_bemt(rotor, med, t, omega=omega)
        psi0 = np.asarray(state.psi[0])
        # Reference cumulative-trapezoid integral of Omega.
        ref = np.concatenate(
            [
                [0.0],
                np.cumsum(
                    0.5 * (np.asarray(omega)[1:] + np.asarray(omega)[:-1]) * np.diff(np.asarray(t))
                ),
            ]
        )
        np.testing.assert_allclose(psi0, ref, atol=1e-9)
        wrong = np.asarray(omega) * np.asarray(t)  # the predecessor's bug
        assert np.abs(psi0[-1] - ref[-1]) < 1e-6
        assert np.abs(psi0[-1] - wrong[-1]) > 1.0  # clearly different from Omega*t
