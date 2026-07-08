"""Attached-flow unsteady aero: the Wagner step-response validation gate.

The digest flags the reconstructed deficiency recurrence as needing exactly the
Wagner check: a step in angle of attack (constant speed) must make the
normalized circulatory lift follow phi(s) = 1 - A1 e^{-b1 s} - A2 e^{-b2 s}.
"""

import jax
import jax.numpy as jnp
import numpy as np

from auraflow.cona.unsteady_aero import (
    deficiency_march,
    unsteady_lift,
    wagner_function,
)


class TestWagnerStepResponse:
    def test_normalized_circulatory_lift_matches_wagner(self):
        # Constant speed, step in alpha at n=0 -> circulatory lift / steady
        # circulatory lift must equal the Wagner function to < 1% over
        # s in [0.1, 20].
        v_const = 50.0
        chord = 0.1
        alpha_step = np.radians(1.0)
        dt = chord / v_const * 0.02  # ds = 2 * 0.02 = 0.04 semichords/step
        n = 3000
        v = jnp.full((n,), v_const)
        alpha = jnp.full((n,), alpha_step)
        lift, alpha_eff, lift_circ = unsteady_lift(v, alpha, dt, chord, rho=1.225)

        # Cumulative semichord distance.
        ds = 2.0 * v_const * dt / chord
        s = ds * jnp.arange(n)
        # Steady circulatory lift (alpha_eff -> alpha).
        steady = 0.5 * 1.225 * chord * (2.0 * jnp.pi) * v_const * alpha_step
        ratio = np.asarray(lift_circ / steady)
        phi = np.asarray(wagner_function(s))

        mask = (np.asarray(s) >= 0.1) & (np.asarray(s) <= 20.0)
        err = np.abs(ratio[mask] - phi[mask])
        assert err.max() < 0.01, err.max()

    def test_wagner_endpoints(self):
        assert abs(float(wagner_function(0.0)) - 0.5) < 1e-12
        assert abs(float(wagner_function(1e4)) - 1.0) < 1e-6

    def test_deficiency_zero_for_zero_downwash(self):
        w = jnp.zeros(50)
        ds = jnp.full(50, 0.1)
        x, y = deficiency_march(w, ds)
        assert float(jnp.max(jnp.abs(x))) == 0.0
        assert float(jnp.max(jnp.abs(y))) == 0.0


class TestApparentMass:
    def test_accelerating_flow_adds_noncirculatory_lift(self):
        # A ramping speed at fixed alpha gives a non-zero apparent-mass term,
        # so total lift differs from the pure circulatory lift.
        n = 400
        dt = 1e-4
        v = jnp.linspace(40.0, 80.0, n)
        alpha = jnp.full((n,), np.radians(3.0))
        lift, _, lift_circ = unsteady_lift(v, alpha, dt, chord=0.1, rho=1.225)
        assert float(jnp.linalg.norm(lift - lift_circ)) > 0.0


class TestGradient:
    def test_grad_through_lift_wrt_clalpha(self):
        v = jnp.full((100,), 50.0)
        alpha = jnp.full((100,), 0.05)

        def scalar(cla):
            lift, _, _ = unsteady_lift(v, alpha, 1e-4, 0.1, 1.225, cl_alpha=cla)
            return jnp.sum(lift)

        g = jax.grad(scalar)(2.0 * jnp.pi)
        assert np.isfinite(float(g))
        assert float(g) > 0.0
