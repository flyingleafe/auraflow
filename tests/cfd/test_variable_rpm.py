"""Variable-RPM resolved-rotor level-set cases.

The initial level-set is RPM-independent (only ``initial_azimuth`` enters it);
the rate enters the *prescribed solid velocity* ``v = Omega(t) (axis x (X -
center))``. These tests check the constant-rate regression, the time-varying
table (a ``jnp.interp`` closure) both numerically and through the real
``InputManager``, and the callable-omega tabulation path. Cases are tiny/coarse
so they fit the dev box memory cap.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.body.blade import rotor_levelset_case
from auraflow.cfd.body_case import LevelsetBodyCase, spin_solid_velocity
from auraflow.core.blade import BladeGeometry, Rotor

importorskip = pytest.importorskip

# JAX-Fluids' level-set model requires cubic cells: dx == dy == dz, so
# (2.4 / 12) == (1.2 / 6) == 0.2.
_BOX_LO = (-1.2, -1.2, -0.6)
_BOX_HI = (1.2, 1.2, 0.6)
_CELLS = (12, 12, 6)


def _rotor(n_blades=2):
    g = BladeGeometry.linear(
        radius=1.0, hub_radius=0.15, n_stations=6,
        chord_root=0.3, chord_tip=0.25, twist_root=0.0, twist_tip=0.0,
    )  # fmt: skip
    return Rotor(blade=g, n_blades=n_blades)


def _case(omega, **kw):
    return rotor_levelset_case(
        _rotor(),
        omega=omega,
        box_lo=_BOX_LO,
        box_hi=_BOX_HI,
        cells=_CELLS,
        n_chord=8,
        sdf_cache=False,
        sdf_batch_points=256,
        **kw,
    )


def _eval_velocity(vel: dict[str, str], x, y, z, t):
    scope = {"jnp": jnp}
    return np.array([float(eval(vel[k], scope)(x, y, z, t)) for k in ("u", "v", "w")])


class TestSpinSolidVelocity:
    def test_constant_matches_omega_r(self):
        vel = spin_solid_velocity((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 50.0)
        # A point at radius r off the +z axis moves at |omega| * r.
        v = _eval_velocity(vel, 0.4, 0.0, 0.0, 0.0)
        assert np.linalg.norm(v) == pytest.approx(50.0 * 0.4, abs=1e-9)

    def test_table_is_linear_interp(self):
        table = ([0.0, 0.01, 0.02], [40.0, 55.0, 60.0])
        vel = spin_solid_velocity((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), table)
        assert "jnp.interp" in vel["u"] and "np.float64" not in vel["u"]
        r = 0.5
        v0 = _eval_velocity(vel, r, 0.0, 0.0, 0.0)
        v_mid = _eval_velocity(vel, r, 0.0, 0.0, 0.005)  # halfway 40->55 = 47.5
        assert np.linalg.norm(v0) == pytest.approx(40.0 * r, abs=1e-9)
        assert np.linalg.norm(v_mid) == pytest.approx(47.5 * r, abs=1e-9)


class TestVariableRpmCase:
    def test_constant_regression(self):
        case = _case(50.0)
        assert isinstance(case, LevelsetBodyCase)
        assert case.is_moving is True
        assert np.asarray(case.levelset_init).shape == _CELLS
        u = case.case["solid_properties"]["velocity"]["u"]
        assert "np.float64" not in u
        assert "jnp.interp" not in u  # constant path, not a table

    def test_table_case_shape_and_lambda(self):
        table = ([0.0, 0.05, 0.1], [40.0, 60.0, 55.0])
        case = _case(table, initial_azimuth=0.3)
        assert case.is_moving is True
        u = case.case["solid_properties"]["velocity"]["u"]
        assert "jnp.interp" in u and "np.float64" not in u

    def test_callable_tabulated(self):
        case = _case(lambda t: 30.0 + 500.0 * t, omega_times=[0.0, 0.05, 0.1])
        assert case.is_moving is True
        u = case.case["solid_properties"]["velocity"]["u"]
        assert "jnp.interp" in u

    def test_callable_needs_time_grid(self):
        with pytest.raises(ValueError, match="omega_times"):
            _case(lambda t: 30.0)

    def test_initial_levelset_rpm_independent(self):
        # The initial level-set depends only on initial_azimuth, not on the rate.
        table = ([0.0, 0.1], [10.0, 90.0])
        a = np.asarray(_case(20.0, initial_azimuth=0.0).levelset_init)
        b = np.asarray(_case(table, initial_azimuth=0.0).levelset_init)
        assert np.allclose(a, b)


class TestInputManagerValidation:
    def test_table_case_validates(self):
        importorskip("jaxfluids")
        from jaxfluids import InputManager

        table = ([0.0, 0.05, 0.1], [40.0, 60.0, 55.0])
        case = _case(table)
        im = InputManager(case.case, case.numerical_setup)
        assert im.equation_information.levelset_model == "FLUID-SOLID"
        assert im.equation_information.is_moving_levelset is True
