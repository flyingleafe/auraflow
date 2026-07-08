"""Tests for auraflow.core.medium."""

import jax
import jax.numpy as jnp
import pytest

from auraflow.core import Medium


class TestStandardAtmosphere:
    def test_sea_level_values(self):
        m = Medium.standard_atmosphere(0.0)
        assert m.p0 == pytest.approx(101325.0)
        assert float(m.rho0) == pytest.approx(1.225, abs=1e-3)
        assert float(m.c0) == pytest.approx(340.294, abs=0.05)
        # Standard sea-level kinematic viscosity ~1.461e-5 m^2/s.
        assert float(m.nu) == pytest.approx(1.461e-5, rel=1e-2)

    def test_default_constructor_matches_sea_level(self):
        m_default = Medium()
        m_isa = Medium.standard_atmosphere(0.0)
        assert float(m_default.rho0) == pytest.approx(float(m_isa.rho0), rel=1e-3)
        assert float(m_default.c0) == pytest.approx(float(m_isa.c0), rel=1e-3)
        assert float(m_default.p0) == pytest.approx(float(m_isa.p0), rel=1e-6)
        assert float(m_default.nu) == pytest.approx(float(m_isa.nu), rel=1e-2)

    def test_tropopause_values(self):
        # Known ISA values at 11 km: T = 216.65 K, p = 22632 Pa, rho = 0.3639 kg/m^3.
        m = Medium.standard_atmosphere(11000.0)
        assert float(m.p0) == pytest.approx(22632.0, rel=1e-3)
        assert float(m.rho0) == pytest.approx(0.3639, rel=1e-3)
        # c = sqrt(1.4 * 287.05 * 216.65) = 295.07 m/s
        assert float(m.c0) == pytest.approx(295.07, abs=0.05)

    def test_monotonic_with_altitude(self):
        altitudes = jnp.linspace(0.0, 10000.0, 11)
        media = [Medium.standard_atmosphere(h) for h in altitudes]
        rhos = jnp.array([m.rho0 for m in media])
        ps = jnp.array([m.p0 for m in media])
        cs = jnp.array([m.c0 for m in media])
        nus = jnp.array([m.nu for m in media])
        assert jnp.all(jnp.diff(rhos) < 0)
        assert jnp.all(jnp.diff(ps) < 0)
        assert jnp.all(jnp.diff(cs) < 0)
        # Kinematic viscosity increases with altitude (rho falls faster than mu).
        assert jnp.all(jnp.diff(nus) > 0)

    def test_differentiable_wrt_altitude(self):
        drho_dh = jax.grad(lambda h: Medium.standard_atmosphere(h).rho0)(500.0)
        assert jnp.isfinite(drho_dh)
        assert drho_dh < 0.0
        # ISA density gradient near sea level is about -1.1e-4 kg/m^3 per m.
        assert float(drho_dh) == pytest.approx(-1.1e-4, rel=0.1)


class TestMediumModule:
    def test_is_pytree_with_traced_fields(self):
        m = Medium(rho0=1.2, c0=343.0, p0=1.0e5, nu=1.5e-5)
        leaves = jax.tree_util.tree_leaves(m)
        assert len(leaves) == 4
        assert all(jnp.isfinite(leaf) for leaf in leaves)

    def test_fields_are_arrays(self):
        m = Medium(rho0=1.2)
        assert isinstance(m.rho0, jax.Array)
        assert isinstance(m.nu, jax.Array)
