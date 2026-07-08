"""CONA HBEM airloads: SectionState shapes, hover thrust, and regressions."""

import jax
import jax.numpy as jnp
import numpy as np

from auraflow.bemt.solver import steady_bemt
from auraflow.cona.airloads import cona_airloads, rotor_section_state
from auraflow.cona.flight import FlightHistory
from auraflow.core.airfoil import ThinAirfoilPolar
from auraflow.core.blade import BladeGeometry, Rotor, Vehicle
from auraflow.core.medium import Medium

_G = 9.80665


def _single_rotor_vehicle(n_stations=10, n_blades=2):
    blade = BladeGeometry.linear(
        radius=0.6,
        hub_radius=0.08,
        n_stations=n_stations,
        chord_root=0.05,
        chord_tip=0.035,
        twist_root=np.radians(16.0),
        twist_tip=np.radians(8.0),
    )
    rotor = Rotor(blade=blade, n_blades=n_blades)
    return Vehicle(rotors=(rotor,))


def _hover_history(vehicle, omega, mass, n_t=64, t_end=0.06):
    """A synthetic hover FlightHistory: fixed pose, constant rotor speed."""
    t = jnp.linspace(0.0, t_end, n_t)
    x = jnp.broadcast_to(jnp.array([0.0, 0.0, 5.0]), (n_t, 3))
    v = jnp.zeros((n_t, 3))
    R = jnp.broadcast_to(jnp.eye(3), (n_t, 3, 3))
    Omega = jnp.zeros((n_t, 3))
    speeds = jnp.full((n_t, 1), omega)
    thrusts = jnp.full((n_t, 1), mass * _G)  # single rotor carries the weight
    return FlightHistory(
        t=t,
        x=x,
        v=v,
        R=R,
        Omega_body=Omega,
        rotor_speeds=speeds,
        rotor_thrusts=thrusts,
    )


class TestShapes:
    def test_leaf_shapes_and_finite(self):
        veh = _single_rotor_vehicle()
        med = Medium()
        hist = _hover_history(veh, omega=320.0, mass=6.0)
        state = rotor_section_state(veh, hist, 0, med, collective=np.radians(6.0))
        b, s, tt = 2, 10, 64
        assert state.phi.shape == (b, s, tt)
        assert state.force_on_fluid.shape == (b, s, tt, 3)
        assert state.position.shape == (b, s, tt, 3)
        assert np.all(np.isfinite(np.asarray(state.force_on_fluid)))
        assert np.all(np.isfinite(np.asarray(state.alpha)))


class TestHoverThrust:
    def test_thrust_integral_near_weight(self):
        # Choose a collective so steady BEMT gives ~ weight, then check the
        # airloads thrust integral is the same order (prescribed-wake inflow
        # differs from momentum inflow, so tolerance is generous).
        veh = _single_rotor_vehicle()
        med = Medium()
        rotor = veh.rotors[0]
        omega = 320.0
        polar = ThinAirfoilPolar(cd0=0.012)
        coll = np.radians(6.0)
        loads = steady_bemt(rotor, med, omega, collective=coll, polar=polar)
        weight = float(loads.thrust)  # define "weight" as the steady thrust
        mass = weight / _G

        hist = _hover_history(veh, omega=omega, mass=mass)
        state = rotor_section_state(veh, hist, 0, med, collective=coll, polar=polar)
        # Thrust = sum over blades, stations of fn * dr; fn = -force_on_fluid_z.
        fn = -np.asarray(state.force_on_fluid[..., 2])  # [B,S,T]
        dr = np.asarray(state.dr)
        thrust_t = np.einsum("bst,s->t", fn, dr)  # [T]
        thrust_mean = thrust_t.mean()
        # Within a factor of 2 of the momentum-based steady thrust.
        assert 0.5 * weight < thrust_mean < 2.0 * weight, (thrust_mean, weight)


class TestTimeVarying:
    def test_ramping_omega_changes_forces(self):
        veh = _single_rotor_vehicle()
        med = Medium()
        t = jnp.linspace(0.0, 0.1, 80)
        n_t = 80
        speeds = jnp.linspace(220.0, 380.0, n_t)[:, None]
        hist = FlightHistory(
            t=t,
            x=jnp.broadcast_to(jnp.array([0.0, 0.0, 5.0]), (n_t, 3)),
            v=jnp.zeros((n_t, 3)),
            R=jnp.broadcast_to(jnp.eye(3), (n_t, 3, 3)),
            Omega_body=jnp.zeros((n_t, 3)),
            rotor_speeds=speeds,
            rotor_thrusts=jnp.full((n_t, 1), 60.0),
        )
        state = rotor_section_state(veh, hist, 0, med, collective=np.radians(6.0))
        fz = -np.asarray(state.force_on_fluid[0, 5, :, 2])
        assert fz.std() / abs(fz.mean()) > 0.2
        assert fz[-1] > 2.0 * fz[0]

    def test_induced_reduces_alpha(self):
        veh = _single_rotor_vehicle()
        med = Medium()
        hist = _hover_history(veh, omega=320.0, mass=6.0)
        on = rotor_section_state(
            veh, hist, 0, med, collective=np.radians(6.0), include_induced=True
        )
        off = rotor_section_state(
            veh, hist, 0, med, collective=np.radians(6.0), include_induced=False
        )
        assert float(jnp.mean(on.alpha)) < float(jnp.mean(off.alpha))
        assert float(jnp.mean(on.v_axial)) > 0.0


class TestMultiRotor:
    def test_cona_airloads_returns_one_state_per_rotor(self):
        blade = BladeGeometry.linear(
            radius=0.5,
            hub_radius=0.06,
            n_stations=6,
            chord_root=0.04,
            chord_tip=0.03,
            twist_root=np.radians(14.0),
            twist_tip=np.radians(8.0),
        )
        r1 = Rotor(
            blade=blade, n_blades=2, hub_position=jnp.array([0.5, 0.0, 0.0]), spin_direction=1
        )
        r2 = Rotor(
            blade=blade, n_blades=2, hub_position=jnp.array([-0.5, 0.0, 0.0]), spin_direction=-1
        )
        veh = Vehicle(rotors=(r1, r2))
        med = Medium()
        n_t = 40
        t = jnp.linspace(0.0, 0.04, n_t)
        hist = FlightHistory(
            t=t,
            x=jnp.broadcast_to(jnp.array([0.0, 0.0, 5.0]), (n_t, 3)),
            v=jnp.zeros((n_t, 3)),
            R=jnp.broadcast_to(jnp.eye(3), (n_t, 3, 3)),
            Omega_body=jnp.zeros((n_t, 3)),
            rotor_speeds=jnp.full((n_t, 2), 340.0),
            rotor_thrusts=jnp.full((n_t, 2), 30.0),
        )
        states = cona_airloads(veh, hist, med, collective=np.radians(6.0))
        assert len(states) == 2
        assert np.all(np.isfinite(np.asarray(states[0].force_on_fluid)))
        assert np.all(np.isfinite(np.asarray(states[1].force_on_fluid)))


class TestGradient:
    def test_grad_thrust_wrt_collective(self):
        veh = _single_rotor_vehicle(n_stations=6)
        med = Medium()
        hist = _hover_history(veh, omega=320.0, mass=6.0, n_t=32, t_end=0.03)

        def mean_thrust(coll):
            state = rotor_section_state(veh, hist, 0, med, collective=coll)
            fn = -state.force_on_fluid[..., 2]
            return jnp.mean(jnp.einsum("bst,s->t", fn, state.dr))

        g = float(jax.grad(mean_thrust)(np.radians(6.0)))
        assert np.isfinite(g)
        assert g > 0.0  # more collective -> more thrust
