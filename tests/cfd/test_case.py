"""JAX-Fluids case-dict builders accepted by InputManager (setup-only, tiny grid).

Skipped cleanly when the ``cfd`` extra (jaxfluids) is not installed.
"""

import jax.numpy as jnp
import pytest

pytest.importorskip("jaxfluids")

from jaxfluids import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    InitializationManager,
    InputManager,
)

from auraflow.cfd.case import (  # noqa: E402
    BoxDomain,
    acoustic_box_case,
    acoustic_timestep,
    points_per_wavelength,
    resolution_for_frequency,
    rotor_box_case,
)
from auraflow.core.medium import Medium  # noqa: E402


class TestResolutionHelpers:
    def test_ppw_and_inverse_consistent(self):
        c0 = 340.0
        dx = resolution_for_frequency(1000.0, c0, ppw=10.0)
        assert points_per_wavelength(dx, 1000.0, c0) == pytest.approx(10.0)

    def test_acoustic_timestep_scales_with_dx_and_mach(self):
        dt0 = acoustic_timestep(0.01, 340.0, mach_max=0.0, cfl=0.5)
        assert dt0 == pytest.approx(0.5 * 0.01 / 340.0)
        # higher Mach -> smaller stable step
        assert acoustic_timestep(0.01, 340.0, mach_max=0.5) < dt0


class TestBoxDomain:
    def test_active_axes_and_spacing(self):
        dom = BoxDomain((-1, 1), (-1, 1), (0, 0.1), (16, 16, 1))
        assert dom.active_axes == ("x", "y")
        dx, dy, dz = dom.spacing()
        assert dx == pytest.approx(2.0 / 16)
        x, y, z = dom.cell_centers()
        assert x.shape == (16,) and z.shape == (1,)
        # cell centres are offset half a cell from the box edge
        assert float(x[0]) == pytest.approx(-1.0 + 0.5 * dx)


class TestAcousticCaseBuildsAndInitializes:
    def _init(self, case):
        im = InputManager(case.case, case.numerical_setup)
        jxf = InitializationManager(im).initialization()
        return im, jxf

    def test_quiescent_case_initializes(self):
        case = acoustic_box_case(Medium(), half_size=0.5, cells=(16, 16, 16))
        im, jxf = self._init(case)
        sl = tuple(im.domain_information.domain_slices_conservatives)
        interior = jxf.simulation_buffers.material_fields.primitives[(slice(None), *sl)]
        assert interior.shape == (5, 16, 16, 16)
        # uniform ambient
        assert jnp.allclose(interior[0], float(case.medium.rho0))
        assert jnp.allclose(interior[4], float(case.medium.p0))

    def test_pulse_case_seeds_overpressure(self):
        case = acoustic_box_case(
            Medium(),
            half_size=0.5,
            cells=(16, 16, 16),
            pulse=True,
            pulse_amplitude=200.0,
            pulse_width=0.12,
        )
        im, jxf = self._init(case)
        sl = tuple(im.domain_information.domain_slices_conservatives)
        p = jxf.simulation_buffers.material_fields.primitives[(slice(None), *sl)][4]
        p0 = float(case.medium.p0)
        assert float(jnp.max(p)) > p0 + 10.0  # pulse present
        assert float(jnp.min(p)) == pytest.approx(p0, abs=1.0)  # ambient far from centre

    def test_two_dimensional_case_initializes(self):
        case = acoustic_box_case(Medium(), half_size=0.5, cells=(24, 24, 1), pulse=True)
        im, jxf = self._init(case)
        assert im.domain_information.dim == 2
        sl = tuple(im.domain_information.domain_slices_conservatives)
        interior = jxf.simulation_buffers.material_fields.primitives[(slice(None), *sl)]
        assert interior.shape == (5, 24, 24, 1)


class TestRotorCase:
    def test_actuator_disk_case_initializes(self):
        case = rotor_box_case(
            Medium(),
            rotor_radius=0.1,
            box_radii=3.0,
            cells=(16, 16, 16),
            thrust=2.0,
            method="actuator_disk",
        )
        im = InputManager(case.case, case.numerical_setup)
        jxf = InitializationManager(im).initialization()
        # custom forcing is active and initial field is quiescent
        sl = tuple(im.domain_information.domain_slices_conservatives)
        interior = jxf.simulation_buffers.material_fields.primitives[(slice(None), *sl)]
        assert interior.shape == (5, 16, 16, 16)
        assert jnp.allclose(interior[0], float(case.medium.rho0))

    def test_levelset_blades_not_implemented(self):
        with pytest.raises(NotImplementedError):
            rotor_box_case(rotor_radius=0.1, method="levelset_blades")

    def test_unknown_method_and_axis_raise(self):
        with pytest.raises(ValueError):
            rotor_box_case(rotor_radius=0.1, method="bogus")
        with pytest.raises(ValueError):
            rotor_box_case(rotor_radius=0.1, thrust_axis="q")
