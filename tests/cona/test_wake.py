"""Prescribed Beddoes wake: Lamb-Oseen, Biot-Savart, and hover sanity gates."""

import jax
import jax.numpy as jnp
import numpy as np

from auraflow.cona.wake import (
    biot_savart_segment,
    core_radius,
    lamb_oseen_swirl,
    make_prescribed_wake,
    vortex_circulation,
)
from auraflow.core.medium import Medium


class TestLambOseen:
    def test_peak_swirl_at_core_radius(self):
        gamma = 5.0
        r_c = 0.03
        r = jnp.linspace(1e-4, 0.3, 4000)
        vth = np.asarray(lamb_oseen_swirl(r, gamma, r_c))
        r_peak = float(r[int(np.argmax(vth))])
        assert abs(r_peak - r_c) / r_c < 0.02

    def test_far_field_potential_swirl(self):
        gamma = 5.0
        r_c = 0.03
        r_far = 1.0
        vth = float(lamb_oseen_swirl(r_far, gamma, r_c))
        assert abs(vth - gamma / (2.0 * np.pi * r_far)) / abs(vth) < 1e-6

    def test_core_growth_monotone(self):
        med = Medium()
        zeta = jnp.linspace(0.0, 8.0 * jnp.pi, 50)
        rc = np.asarray(core_radius(zeta, omega=300.0, gamma_v=2.0, medium=med, r_c0=0.01))
        assert np.all(np.diff(rc) >= -1e-15)
        assert rc[-1] > rc[0]


class TestBiotSavart:
    def test_long_filament_matches_infinite_line(self):
        # A long straight filament along z, evaluated at distance d on +x, must
        # give ~ Gamma/(2 pi d) (perpendicular, tangential swirl direction).
        gamma = 3.0
        d = 0.5
        length = 400.0
        a = jnp.array([0.0, 0.0, -length])
        b = jnp.array([0.0, 0.0, length])
        p = jnp.array([d, 0.0, 0.0])
        v = np.asarray(biot_savart_segment(p, a, b, gamma, r_c=1e-4))
        speed = np.linalg.norm(v)
        assert abs(speed - gamma / (2.0 * np.pi * d)) / (gamma / (2.0 * np.pi * d)) < 1e-3
        # Swirl is perpendicular to both the filament (z) and the radial (x).
        assert abs(v[0]) < 1e-6 and abs(v[2]) < 1e-6

    def test_no_nan_on_axis(self):
        gamma = 3.0
        a = jnp.array([0.0, 0.0, -1.0])
        b = jnp.array([0.0, 0.0, 1.0])
        p = jnp.array([0.0, 0.0, 0.0])  # on the filament axis
        v = np.asarray(biot_savart_segment(p, a, b, gamma, r_c=0.05))
        assert np.all(np.isfinite(v))

    def test_grad_finite_near_core(self):
        a = jnp.array([0.0, 0.0, -1.0])
        b = jnp.array([0.0, 0.0, 1.0])

        def scalar(d):
            p = jnp.array([d, 0.0, 0.0])
            return jnp.sum(biot_savart_segment(p, a, b, 3.0, 0.05) ** 2)

        g = float(jax.grad(scalar)(1e-4))
        assert np.isfinite(g)


class TestHoverWake:
    def test_disk_inflow_is_downward_and_order_of_momentum(self):
        med = Medium()
        radius = 0.6
        omega = 300.0
        ct = 0.012
        wake = make_prescribed_wake(
            ct,
            omega,
            radius,
            n_blades=2,
            medium=med,
            mu_x=0.0,
            mu_z=0.0,
            n_azimuth=24,
            n_rev=4,
            chord_ref=0.045,
        )
        # Evaluate axial induced velocity across the disk (mid radii).
        r_grid = jnp.linspace(0.2 * radius, 0.9 * radius, 6)
        psi_grid = jnp.linspace(0.0, 2.0 * jnp.pi, 12, endpoint=False)
        uz = np.asarray(wake.inflow_grid(r_grid, psi_grid))
        # Downwash below a thrusting disk: u_z < 0 on average.
        assert uz.mean() < 0.0
        # Magnitude within a factor ~2-3 of the momentum hover inflow
        # v_i = lambda0 * Omega R, lambda0 = sqrt(CT/2).
        lam0 = float(jnp.sqrt(ct / 2.0))
        v_mom = lam0 * omega * radius
        ratio = abs(uz.mean()) / v_mom
        assert 0.3 < ratio < 3.0, ratio

    def test_circulation_scales_with_ct(self):
        g1 = float(vortex_circulation(0.01, 300.0, 0.6, 2))
        g2 = float(vortex_circulation(0.02, 300.0, 0.6, 2))
        assert abs(g2 / g1 - 2.0) < 1e-6

    def test_grad_through_inflow_wrt_rc0(self):
        med = Medium()

        def scalar(r_c0):
            wake = make_prescribed_wake(
                0.012,
                300.0,
                0.6,
                2,
                med,
                n_azimuth=16,
                n_rev=3,
                r_c0=r_c0,
            )
            uz = wake.inflow_grid(jnp.linspace(0.2, 0.5, 4), jnp.linspace(0.0, 6.0, 6))
            return jnp.sum(uz)

        g = float(jax.grad(scalar)(0.01))
        assert np.isfinite(g)
