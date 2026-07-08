"""CONA tonal noise: BPF presence, forward-flight f1c, and low-speed consistency."""

import jax
import jax.numpy as jnp
import numpy as np

from auraflow.cona.flight import FlightHistory
from auraflow.cona.tonal import cona_tonal_noise
from auraflow.core.airfoil import ThinAirfoilPolar
from auraflow.core.blade import BladeGeometry, Rotor, Vehicle
from auraflow.core.medium import Medium

_G = 9.80665


def _rotor(n_stations=8, n_blades=2, spin=1, hub=(0.0, 0.0, 0.0)):
    blade = BladeGeometry.linear(
        radius=0.15,
        hub_radius=0.02,
        n_stations=n_stations,
        chord_root=0.02,
        chord_tip=0.014,
        twist_root=np.radians(18.0),
        twist_tip=np.radians(8.0),
    )
    return Rotor(blade=blade, n_blades=n_blades, hub_position=jnp.asarray(hub), spin_direction=spin)


def _history(vehicle, omega, v_world, n_t=200, t_end=0.06, alt=2.0):
    t = jnp.linspace(0.0, t_end, n_t)
    v_world = jnp.asarray(v_world, dtype=float)
    x0 = jnp.array([0.0, 0.0, alt])
    x = x0[None, :] + v_world[None, :] * t[:, None]
    v = jnp.broadcast_to(v_world, (n_t, 3))
    R = jnp.broadcast_to(jnp.eye(3), (n_t, 3, 3))
    Omega = jnp.zeros((n_t, 3))
    nr = vehicle.n_rotors
    speeds = jnp.full((n_t, nr), omega)
    thrusts = jnp.full((n_t, nr), 2.0)
    return FlightHistory(
        t=t, x=x, v=v, R=R, Omega_body=Omega, rotor_speeds=speeds, rotor_thrusts=thrusts
    )


class TestHoverBPF:
    def test_bpf_fundamental_present(self):
        veh = Vehicle(rotors=(_rotor(n_blades=2),))
        med = Medium()
        omega = 900.0  # rad/s
        hist = _history(veh, omega, v_world=(0.0, 0.0, 0.0), n_t=600, t_end=0.15)
        # Off-axis observer (below and to the side): blade-passage tones are
        # strong here (on-axis the axisymmetric hover signal is nearly steady).
        obs = jnp.array([[1.2, 0.0, -1.0]])
        p, _, _, t_obs = cona_tonal_noise(
            veh,
            hist,
            obs,
            med,
            collective=np.radians(8.0),
            polar=ThinAirfoilPolar(cd0=0.02),
        )
        sig = np.asarray(p[0])
        # Drop the initial transient (unsteady-aero build-up from rest).
        sig = sig[sig.size // 4 :]
        sig = sig - sig.mean()
        dt = float(t_obs[1] - t_obs[0])
        freqs = np.fft.rfftfreq(sig.size, dt)
        mag = np.abs(np.fft.rfft(sig * np.hanning(sig.size)))
        bpf = 2 * omega / (2 * np.pi)  # 2 blades
        peak_f = freqs[np.argmax(mag[1:]) + 1]
        assert abs(peak_f - bpf) / bpf < 0.25, (peak_f, bpf)


class TestForwardFlight:
    def test_f1c_runs_finite(self):
        veh = Vehicle(rotors=(_rotor(n_blades=2),))
        med = Medium()
        hist = _history(veh, omega=900.0, v_world=(10.0, 0.0, 0.0), n_t=200)
        obs = jnp.array([[3.0, 0.0, -1.5], [0.0, 3.0, -1.5]])
        p, pt, pl, _ = cona_tonal_noise(
            veh,
            hist,
            obs,
            med,
            collective=np.radians(8.0),
            flow_model="f1c",
        )
        assert np.all(np.isfinite(np.asarray(p)))
        assert float(jnp.max(jnp.abs(p))) > 0.0


class TestLowSpeedConsistency:
    def test_f1c_approaches_f1a_at_low_speed(self):
        veh = Vehicle(rotors=(_rotor(n_blades=2),))
        med = Medium()
        # mu ~ 0.01: Vtip = 900*0.15 = 135 m/s, V = 1.35 m/s -> mach0 ~ 0.004.
        hist = _history(veh, omega=900.0, v_world=(1.35, 0.0, 0.0), n_t=200)
        obs = jnp.array([[0.0, 0.0, -1.5]])
        common_t = jnp.linspace(0.02, 0.05, 200)
        p_a, _, _, _ = cona_tonal_noise(
            veh,
            hist,
            obs,
            med,
            collective=np.radians(8.0),
            flow_model="f1a",
            t_obs=common_t,
        )
        p_c, _, _, _ = cona_tonal_noise(
            veh,
            hist,
            obs,
            med,
            collective=np.radians(8.0),
            flow_model="f1c",
            t_obs=common_t,
        )
        a = np.asarray(p_a[0])
        c = np.asarray(p_c[0])
        rel_l2 = np.linalg.norm(a - c) / np.linalg.norm(a)
        assert rel_l2 < 0.2, rel_l2


class TestGradient:
    def test_grad_oaspl_wrt_collective(self):
        veh = Vehicle(rotors=(_rotor(n_blades=2, n_stations=6),))
        med = Medium()
        hist = _history(veh, omega=900.0, v_world=(0.0, 0.0, 0.0), n_t=120, t_end=0.05)
        obs = jnp.array([[0.0, 0.0, -1.5]])

        def oaspl(coll):
            p, _, _, _ = cona_tonal_noise(veh, hist, obs, med, collective=coll)
            return 10.0 * jnp.log10(jnp.mean(p[0] ** 2) / (20e-6) ** 2 + 1e-30)

        g = float(jax.grad(oaspl)(np.radians(8.0)))
        assert np.isfinite(g)
