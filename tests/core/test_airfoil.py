"""Tests for auraflow.core.airfoil."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.core import TablePolar, ThinAirfoilPolar


class TestThinAirfoilPolar:
    def test_linear_range_matches_thin_airfoil_theory(self):
        polar = ThinAirfoilPolar(alpha0=0.0, cd0=0.01, k=0.0)
        alpha = jnp.deg2rad(jnp.linspace(-5.0, 5.0, 21))
        cl, cd = polar(alpha)
        np.testing.assert_allclose(cl, 2.0 * jnp.pi * alpha, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(cd, 0.01, atol=1e-15)

    def test_zero_lift_angle_shift(self):
        alpha0 = jnp.deg2rad(-2.0)
        polar = ThinAirfoilPolar(alpha0=alpha0)
        cl_at_alpha0, _ = polar(alpha0)
        assert float(cl_at_alpha0) == pytest.approx(0.0, abs=1e-12)
        cl, _ = polar(jnp.deg2rad(3.0))
        assert float(cl) == pytest.approx(float(2 * jnp.pi * (jnp.deg2rad(3.0) - alpha0)), rel=1e-3)

    def test_custom_lift_slope(self):
        polar = ThinAirfoilPolar(cl_alpha=5.7)
        cl, _ = polar(jnp.deg2rad(4.0))
        assert float(cl) == pytest.approx(5.7 * float(jnp.deg2rad(4.0)), rel=1e-3)

    def test_quadratic_drag(self):
        polar = ThinAirfoilPolar(cd0=0.008, k=0.05)
        alpha = jnp.deg2rad(6.0)
        cl, cd = polar(alpha)
        assert float(cd) == pytest.approx(0.008 + 0.05 * float(cl) ** 2, rel=1e-12)

    def test_saturates_smoothly_beyond_stall(self):
        polar = ThinAirfoilPolar()
        cl_max = 2.0 * jnp.pi * float(jnp.deg2rad(20.0))
        cl60, _ = polar(jnp.deg2rad(60.0))
        cl90, _ = polar(jnp.deg2rad(90.0))
        assert float(cl60) == pytest.approx(cl_max, rel=1e-3)
        assert float(cl90) == pytest.approx(cl_max, rel=1e-6)
        # No overshoot, and cl remains monotonic nondecreasing across stall
        # (up to float64 rounding on the saturated plateau).
        alpha = jnp.deg2rad(jnp.linspace(-90.0, 90.0, 721))
        cl, _ = polar(alpha)
        assert float(jnp.max(jnp.abs(cl))) <= cl_max + 1e-9
        assert bool(jnp.all(jnp.diff(cl) >= -1e-12))

    def test_antisymmetric_saturation(self):
        polar = ThinAirfoilPolar()
        alpha = jnp.deg2rad(jnp.array([10.0, 25.0, 50.0]))
        cl_pos, _ = polar(alpha)
        cl_neg, _ = polar(-alpha)
        np.testing.assert_allclose(cl_neg, -cl_pos, atol=1e-12)

    def test_gradient_nonzero_everywhere(self):
        polar = ThinAirfoilPolar()
        dcl = jax.vmap(jax.grad(lambda a: polar(a)[0]))
        alpha = jnp.deg2rad(jnp.array([-80.0, -30.0, -20.0, 0.0, 15.0, 20.0, 25.0, 45.0, 80.0]))
        grads = dcl(alpha)
        assert bool(jnp.all(jnp.isfinite(grads)))
        assert bool(jnp.all(grads > 0.0))

    def test_gradient_matches_slope_in_linear_range(self):
        polar = ThinAirfoilPolar()
        g = jax.grad(lambda a: polar(a)[0])(jnp.deg2rad(2.0))
        assert float(g) == pytest.approx(2.0 * jnp.pi, rel=1e-3)

    def test_prandtl_glauert_mach_correction(self):
        polar = ThinAirfoilPolar()
        alpha = jnp.deg2rad(4.0)
        cl_incomp, _ = polar(alpha)
        cl_m, _ = polar(alpha, mach=0.5)
        assert float(cl_m) == pytest.approx(float(cl_incomp) / np.sqrt(1.0 - 0.25), rel=1e-12)

    def test_reynolds_ignored(self):
        polar = ThinAirfoilPolar()
        cl_a, cd_a = polar(0.1)
        cl_b, cd_b = polar(0.1, reynolds=1e6)
        assert float(cl_a) == float(cl_b)
        assert float(cd_a) == float(cd_b)

    def test_differentiable_wrt_polar_parameters(self):
        def loss(cl_alpha, cd0):
            polar = ThinAirfoilPolar(cl_alpha=cl_alpha, cd0=cd0, k=0.02)
            cl, cd = polar(jnp.deg2rad(5.0))
            return cl**2 + cd

        grads = jax.grad(loss, argnums=(0, 1))(2 * jnp.pi, 0.01)
        assert all(jnp.isfinite(g) for g in grads)
        assert all(float(g) != 0.0 for g in grads)


class TestTablePolar1D:
    @pytest.fixture
    def polar(self) -> TablePolar:
        alpha = jnp.deg2rad(jnp.linspace(-10.0, 10.0, 9))
        cl = 6.0 * alpha + 0.1
        cd = 0.01 + 0.3 * alpha**2
        return TablePolar(alpha_grid=alpha, cl_table=cl, cd_table=cd)

    def test_reproduces_node_values(self, polar):
        cl, cd = polar(polar.alpha_grid)
        np.testing.assert_allclose(cl, polar.cl_table, atol=1e-14)
        np.testing.assert_allclose(cd, polar.cd_table, atol=1e-14)

    def test_midpoints_match_manual_lerp(self, polar):
        mid = 0.5 * (polar.alpha_grid[:-1] + polar.alpha_grid[1:])
        cl, cd = polar(mid)
        np.testing.assert_allclose(cl, 0.5 * (polar.cl_table[:-1] + polar.cl_table[1:]), atol=1e-14)
        np.testing.assert_allclose(cd, 0.5 * (polar.cd_table[:-1] + polar.cd_table[1:]), atol=1e-14)

    def test_clamps_out_of_range(self, polar):
        cl_low, _ = polar(jnp.deg2rad(-50.0))
        cl_high, _ = polar(jnp.deg2rad(50.0))
        assert float(cl_low) == pytest.approx(float(polar.cl_table[0]))
        assert float(cl_high) == pytest.approx(float(polar.cl_table[-1]))

    def test_batched_query_shape(self, polar):
        cl, cd = polar(jnp.zeros((3, 4)))
        assert cl.shape == (3, 4)
        assert cd.shape == (3, 4)

    def test_gradients_finite(self, polar):
        g_alpha = jax.grad(lambda a: polar(a)[0])(0.01)
        assert jnp.isfinite(g_alpha)
        assert float(g_alpha) == pytest.approx(6.0, rel=1e-10)

        def wrt_table(cl_table):
            p = TablePolar(polar.alpha_grid, cl_table, polar.cd_table)
            return p(0.01)[0]

        g_table = jax.grad(wrt_table)(polar.cl_table)
        assert bool(jnp.all(jnp.isfinite(g_table)))
        assert float(jnp.sum(g_table)) == pytest.approx(1.0)  # lerp weights sum to 1


class TestTablePolar2D:
    @pytest.fixture
    def polar(self) -> TablePolar:
        alpha = jnp.linspace(-0.2, 0.2, 5)
        mach = jnp.array([0.0, 0.3, 0.6])
        cl = 6.0 * alpha[:, None] / jnp.sqrt(1.0 - mach[None, :] ** 2)
        cd = 0.01 + 0.02 * mach[None, :] + 0.0 * alpha[:, None]
        return TablePolar(alpha_grid=alpha, cl_table=cl, cd_table=cd, mach_grid=mach)

    def test_reproduces_node_values(self, polar):
        for i, a in enumerate(polar.alpha_grid):
            for j, m in enumerate(polar.mach_grid):
                cl, cd = polar(a, mach=m)
                assert float(cl) == pytest.approx(float(polar.cl_table[i, j]), abs=1e-14)
                assert float(cd) == pytest.approx(float(polar.cd_table[i, j]), abs=1e-14)

    def test_cell_center_matches_manual_bilinear(self, polar):
        a = 0.5 * (polar.alpha_grid[1] + polar.alpha_grid[2])
        m = 0.5 * (polar.mach_grid[0] + polar.mach_grid[1])
        cl, _ = polar(a, mach=m)
        manual = 0.25 * (
            polar.cl_table[1, 0]
            + polar.cl_table[1, 1]
            + polar.cl_table[2, 0]
            + polar.cl_table[2, 1]
        )
        assert float(cl) == pytest.approx(float(manual), abs=1e-14)

    def test_general_point_matches_manual_bilinear(self, polar):
        a, m = 0.07, 0.2
        i, j = 2, 0  # brackets: alpha[2]=0.0..alpha[3]=0.1, mach[0]=0.0..mach[1]=0.3
        fa = (a - float(polar.alpha_grid[i])) / float(polar.alpha_grid[i + 1] - polar.alpha_grid[i])
        fm = (m - float(polar.mach_grid[j])) / float(polar.mach_grid[j + 1] - polar.mach_grid[j])
        tbl = polar.cl_table
        manual = (
            (1 - fa) * (1 - fm) * tbl[i, j]
            + fa * (1 - fm) * tbl[i + 1, j]
            + (1 - fa) * fm * tbl[i, j + 1]
            + fa * fm * tbl[i + 1, j + 1]
        )
        cl, _ = polar(a, mach=m)
        assert float(cl) == pytest.approx(float(manual), abs=1e-14)

    def test_missing_mach_raises(self, polar):
        with pytest.raises(ValueError, match="mach"):
            polar(0.1)

    def test_broadcasting_alpha_and_mach(self, polar):
        alpha = jnp.zeros((4, 1))
        mach = jnp.array([0.1, 0.2, 0.5])
        cl, cd = polar(alpha, mach=mach)
        assert cl.shape == (4, 3)
        assert cd.shape == (4, 3)

    def test_gradients_finite_wrt_both_queries(self, polar):
        grads = jax.grad(lambda a, m: polar(a, mach=m)[0], argnums=(0, 1))(0.05, 0.2)
        assert all(jnp.isfinite(g) for g in grads)
        assert float(grads[0]) > 0.0  # cl increases with alpha


class TestTablePolar3D:
    @pytest.fixture
    def polar(self) -> TablePolar:
        key = jax.random.PRNGKey(0)
        alpha = jnp.linspace(-0.3, 0.3, 4)
        mach = jnp.linspace(0.0, 0.6, 3)
        re = jnp.array([1e5, 1e6])
        k1, k2 = jax.random.split(key)
        cl = jax.random.normal(k1, (4, 3, 2))
        cd = 0.01 + 0.1 * jax.random.uniform(k2, (4, 3, 2))
        return TablePolar(alpha, cl, cd, mach_grid=mach, reynolds_grid=re)

    def test_reproduces_node_values(self, polar):
        cl, cd = polar(
            polar.alpha_grid[1], mach=polar.mach_grid[2], reynolds=polar.reynolds_grid[0]
        )
        assert float(cl) == pytest.approx(float(polar.cl_table[1, 2, 0]), abs=1e-12)
        assert float(cd) == pytest.approx(float(polar.cd_table[1, 2, 0]), abs=1e-12)

    def test_cell_center_matches_corner_average(self, polar):
        a = 0.5 * (polar.alpha_grid[0] + polar.alpha_grid[1])
        m = 0.5 * (polar.mach_grid[1] + polar.mach_grid[2])
        re = 0.5 * (polar.reynolds_grid[0] + polar.reynolds_grid[1])
        cl, _ = polar(a, mach=m, reynolds=re)
        corners = [polar.cl_table[i, 1 + j, k] for i, j, k in itertools.product((0, 1), repeat=3)]
        assert float(cl) == pytest.approx(float(sum(corners)) / 8.0, abs=1e-12)

    def test_missing_reynolds_raises(self, polar):
        with pytest.raises(ValueError, match="reynolds"):
            polar(0.1, mach=0.3)

    def test_gradients_finite(self, polar):
        g = jax.grad(lambda a: polar(a, mach=0.25, reynolds=5e5)[0])(0.02)
        assert jnp.isfinite(g)


class TestTablePolarValidation:
    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            TablePolar(jnp.linspace(0, 1, 5), jnp.zeros(4), jnp.zeros(5))

    def test_too_few_nodes_raises(self):
        with pytest.raises(ValueError, match="2 nodes"):
            TablePolar(jnp.array([0.0]), jnp.zeros(1), jnp.zeros(1))
