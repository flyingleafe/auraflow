"""NACA 4-digit section profiles: closed loop, thickness, camber, gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.body.airfoil_profile import naca0012, naca4_profile


def _shoelace(loop: np.ndarray) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


class TestClosedLoop:
    def test_shape_and_no_duplicate_vertices(self):
        n = 60
        loop = np.asarray(naca4_profile(0.02, 0.4, 0.12, n))
        # N = 2 n - 2 (shared LE and TE vertices are not duplicated).
        assert loop.shape == (2 * n - 2, 2)
        # No two consecutive vertices coincide (a valid non-degenerate loop).
        d = np.linalg.norm(np.diff(loop, axis=0, append=loop[:1]), axis=1)
        assert float(d.min()) > 1e-6

    def test_positive_shoelace_no_self_intersection(self):
        # Counterclockwise (positive area) and convex-hull-free crossing check:
        # the upper surface stays above the lower surface at every chord station.
        for m, p, tc in [(0.0, 0.0, 0.12), (0.02, 0.4, 0.12), (0.04, 0.5, 0.2)]:
            n = 80
            loop = np.asarray(naca4_profile(m, p, tc, n))
            assert _shoelace(loop) > 0.0
            lower = loop[:n]  # LE -> TE
            upper = np.concatenate([loop[:1], loop[: n - 2 : -1]])  # LE + TE->LE reversed
            # Match lengths and compare y at the shared chordwise samples.
            yl = lower[:, 1]
            yu = np.concatenate([upper[:1, 1], upper[1:, 1][::-1]])
            assert np.all(yu[1:-1] >= yl[1:-1] - 1e-12)


class TestThicknessCamber:
    def test_max_thickness_within_1pct(self):
        for tc in (0.08, 0.12, 0.2):
            loop = np.asarray(naca0012.func(0.0, 0.0, tc, 200))
            # Symmetric section: peak upper-surface eta ~ tc/2; full thickness ~ tc.
            full_thickness = float(loop[:, 1].max() - loop[:, 1].min())
            assert full_thickness == pytest.approx(tc, rel=1e-2)

    def test_camber_line_endpoints(self):
        # Camber-line endpoints: the LE node (loop index 0, xi = 0) is (0, 0)
        # and the TE node (index n-1, xi = 1) is (1, 0). (The extreme-x surface
        # vertices differ slightly due to the NACA leading-edge droop.)
        n = 100
        loop = np.asarray(naca4_profile(0.04, 0.4, 0.12, n))
        le = loop[0]
        te = loop[n - 1]
        assert le[0] == pytest.approx(0.0, abs=1e-9)
        assert le[1] == pytest.approx(0.0, abs=1e-9)
        assert te[0] == pytest.approx(1.0, abs=1e-9)
        assert te[1] == pytest.approx(0.0, abs=1e-6)  # closed TE

    def test_symmetric_has_zero_camber(self):
        loop = np.asarray(naca0012(120))
        # Symmetric about eta = 0: max upper == -min lower.
        assert loop[:, 1].max() == pytest.approx(-loop[:, 1].min(), rel=1e-6)


class TestDifferentiable:
    def test_area_grad_in_tc_is_finite_positive(self):
        def area(tc):
            loop = naca4_profile(0.0, 0.0, tc, 60)
            x, y = loop[:, 0], loop[:, 1]
            return 0.5 * jnp.sum(x * jnp.roll(y, -1) - jnp.roll(x, -1) * y)

        g = jax.grad(area)(0.12)
        assert np.isfinite(float(g))
        assert float(g) > 0.0  # thicker section -> larger enclosed area

    def test_grad_in_camber_finite(self):
        def area(m):
            loop = naca4_profile(m, 0.4, 0.12, 60)
            x, y = loop[:, 0], loop[:, 1]
            return 0.5 * jnp.sum(x * jnp.roll(y, -1) - jnp.roll(x, -1) * y)

        g = jax.grad(area)(0.03)
        assert np.isfinite(float(g))
