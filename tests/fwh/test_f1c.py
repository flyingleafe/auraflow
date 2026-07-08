"""Formulation 1C cross-checks: F1A reduction and the wind-tunnel special case."""

import jax
import jax.numpy as jnp
import numpy as np

from auraflow.core.medium import Medium
from auraflow.fwh import f1a_pressure, f1c_pressure, f1c_windtunnel


def _random_moving_scenario(key, n_src=3, n_time=160):
    med = Medium()
    tau = jnp.linspace(0.0, 0.03, n_time)
    ks = jax.random.split(key, 8)
    y0 = jax.random.normal(ks[0], (n_src, 3)) * 0.5
    vel = jax.random.normal(ks[1], (n_src, 3)) * 20.0  # ~0.06 Mach
    acc = jax.random.normal(ks[2], (n_src, 3)) * 8.0
    y = (
        y0[:, None, :]
        + vel[:, None, :] * tau[None, :, None]
        + 0.5 * acc[:, None, :] * (tau[None, :, None] ** 2)
    )
    v = vel[:, None, :] + acc[:, None, :] * tau[None, :, None]
    a = jnp.broadcast_to(acc[:, None, :], (n_src, n_time, 3))
    qn = jax.random.normal(ks[3], (n_src, n_time)) * 0.1 + jnp.sin(2 * np.pi * 300 * tau)[None, :]
    load = (
        jax.random.normal(ks[4], (n_src, n_time, 3)) * 0.1
        + jnp.cos(2 * np.pi * 250 * tau)[None, :, None]
    )
    x_obs = jnp.array([[5.0, 3.0, -2.0], [-4.0, 1.0, 6.0]])
    t_obs = jnp.linspace(float(tau[5]) + 0.02, float(tau[-6]) + 0.01, n_time)
    return med, y, v, a, qn, load, tau, t_obs, x_obs


class TestF1CReducesToF1A:
    def test_thickness_and_loading_match_at_zero_mach(self):
        med, y, v, a, qn, load, tau, t_obs, x_obs = _random_moving_scenario(jax.random.PRNGKey(0))
        pt_a, pl_a = f1a_pressure(x_obs, y, v, a, qn, load, med, tau, t_obs)
        pt_c, pl_c = f1c_pressure(x_obs, y, v, a, qn, load, med, 0.0, tau, t_obs)
        assert float(jnp.linalg.norm(pt_a - pt_c) / jnp.linalg.norm(pt_a)) < 1e-10
        assert float(jnp.linalg.norm(pl_a - pl_c) / jnp.linalg.norm(pl_a)) < 1e-10

    def test_reduction_holds_for_several_seeds(self):
        for seed in (1, 2, 3):
            med, y, v, a, qn, load, tau, t_obs, x_obs = _random_moving_scenario(
                jax.random.PRNGKey(seed)
            )
            pt_a, pl_a = f1a_pressure(x_obs, y, v, a, qn, load, med, tau, t_obs)
            pt_c, pl_c = f1c_pressure(x_obs, y, v, a, qn, load, med, 0.0, tau, t_obs)
            assert float(jnp.linalg.norm(pt_a - pt_c) / jnp.linalg.norm(pt_a)) < 1e-9
            assert float(jnp.linalg.norm(pl_a - pl_c) / jnp.linalg.norm(pl_a)) < 1e-9


class TestWindTunnelSpecialCase:
    def test_matches_full_1c_static_geometry(self):
        # Static source + observer in uniform flow: the wind-tunnel fast path
        # must equal the general 1C kernel evaluated with v = a = 0.
        med = Medium()
        key = jax.random.PRNGKey(7)
        ks = jax.random.split(key, 4)
        n_src, n_time = 4, 200
        tau = jnp.linspace(0.0, 0.02, n_time)
        y_panels = jax.random.normal(ks[0], (n_src, 3))
        normal = jax.random.normal(ks[1], (n_src, 3))
        normal = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
        area = jnp.abs(jax.random.normal(ks[2], (n_src,))) + 0.5
        qn = jnp.sin(2 * np.pi * 200 * tau)[None, :] * jax.random.uniform(ks[3], (n_src, 1))
        load = jnp.cos(2 * np.pi * 150 * tau)[None, :, None] * jax.random.normal(
            ks[3], (n_src, 1, 3)
        )
        x_obs = jnp.array([[8.0, 2.0, -3.0], [-6.0, 4.0, 1.0]])
        mach0 = 0.3
        t_obs = jnp.linspace(float(tau[5]) + 0.03, float(tau[-6]) + 0.02, n_time)

        pt_w, pl_w = f1c_windtunnel(x_obs, y_panels, normal, area, qn, load, med, mach0, tau, t_obs)

        y_static = jnp.broadcast_to(y_panels[:, None, :], (n_src, n_time, 3))
        zero = jnp.zeros((n_src, n_time, 3))
        pt_g, pl_g = f1c_pressure(
            x_obs, y_static, zero, zero, qn, load, med, mach0, tau, t_obs, area=area
        )
        assert float(jnp.linalg.norm(pt_w - pt_g) / jnp.linalg.norm(pt_g)) < 1e-9
        assert float(jnp.linalg.norm(pl_w - pl_g) / jnp.linalg.norm(pl_g)) < 1e-9
