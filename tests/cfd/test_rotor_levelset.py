"""Resolved-blade level-set rotor CFD cases (``method='levelset_blades'``).

The case-validation checks push the programmatic JAX-Fluids dicts through the
real ``InputManager`` (and the initialized level-set buffer), so they
``importorskip`` jaxfluids. The SDF-sign check needs trimesh (the ``mesh`` extra).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.body.blade import blade_mesh, rotor_levelset_case, rotor_mesh
from auraflow.cfd.body_case import LevelsetBodyCase
from auraflow.cfd.case import rotor_box_case
from auraflow.core.blade import BladeGeometry, Rotor

importorskip = pytest.importorskip

# Small box enclosing the (real-scale) NASA 1-Pax rotor at coarse resolution.
# JAX-Fluids' level-set model requires **cubic** cells (unity aspect ratio) and
# at least as many cells per axis as halo cells (4), so dz == dx == dy and
# nz >= 4: span 4.4 / 20 == 1.32 / 6 == 0.22 m.
_BOX_LO = (-2.2, -2.2, -0.66)
_BOX_HI = (2.2, 2.2, 0.66)
_CELLS = (20, 20, 6)


def _coarse_nasa_rotor() -> Rotor:
    from auraflow.datasets.nasa_1pax import N_BLADES, nasa_1pax_blade

    return Rotor(blade=nasa_1pax_blade(n_stations=5), n_blades=N_BLADES)


class TestRotorLevelsetCase:
    def _input_manager(self, case: LevelsetBodyCase):
        importorskip("jaxfluids")
        from jaxfluids import InputManager

        return InputManager(case.case, case.numerical_setup)

    def test_nasa_rotor_spinning_case_validates(self):
        from auraflow.datasets.nasa_1pax import HOVER_OMEGA

        rotor = _coarse_nasa_rotor()
        case = rotor_levelset_case(
            rotor,
            omega=HOVER_OMEGA,
            box_lo=_BOX_LO,
            box_hi=_BOX_HI,
            cells=_CELLS,
            n_chord=8,
        )
        assert isinstance(case, LevelsetBodyCase)
        assert case.is_moving is True
        assert np.asarray(case.levelset_init).shape == _CELLS
        # A prescribed rotational solid velocity was emitted (clean float lambda).
        u_lambda = case.case["solid_properties"]["velocity"]["u"]
        assert "np.float64" not in u_lambda  # JAX-Fluids evals with jnp only
        im = self._input_manager(case)
        assert im.equation_information.levelset_model == "FLUID-SOLID"
        assert im.equation_information.is_moving_levelset is True

    def test_rotor_box_case_wiring_validates(self):
        # cfd.case.rotor_box_case(method='levelset_blades', rotor=...) delegates to
        # rotor_levelset_case and returns a validating LevelsetBodyCase.
        rotor = _coarse_nasa_rotor()
        # rotor_box_case makes a cube, so unity-aspect cells need nx == ny == nz.
        case = rotor_box_case(
            rotor_radius=1.951,
            box_radii=1.2,
            cells=(12, 12, 12),
            method="levelset_blades",
            rotor=rotor,
            n_chord=8,
            tip_mach=0.4,
        )
        assert isinstance(case, LevelsetBodyCase)
        assert case.is_moving is True
        im = self._input_manager(case)
        assert im.equation_information.is_moving_levelset is True

    def test_levelset_blades_requires_rotor(self):
        with pytest.raises(NotImplementedError, match="rotor="):
            rotor_box_case(rotor_radius=1.0, method="levelset_blades")

    def test_levelset_field_negative_inside_when_resolved(self):
        importorskip("trimesh")
        # A fat, well-resolved single-blade rotor: rotor_levelset_case's CFD-facing
        # level-set field (the body SDF at cell centres) must be negative inside
        # the blade and positive in the fluid (JAX-Fluids' negative-inside
        # convention). Coarse real-blade grids under-resolve the thin section, so
        # a fat blade + fine box is used. (JAX-Fluids' own preservation of this
        # sign on ingest is covered generically by the sphere test in
        # tests/cfd/test_body_case.py.)
        g = BladeGeometry.linear(
            radius=1.0, hub_radius=0.15, n_stations=12,
            chord_root=0.5, chord_tip=0.4, twist_root=0.0, twist_tip=0.0,
        )  # fmt: skip
        rotor = Rotor(blade=g, n_blades=1)
        case = rotor_levelset_case(
            rotor,
            omega=50.0,
            box_lo=(0.1, -0.45, -0.12),
            box_hi=(1.05, 0.2, 0.12),
            cells=(32, 24, 16),
            n_chord=32,
        )
        ls = np.asarray(case.levelset_init)
        assert float(ls.min()) < 0.0  # cells inside the blade
        assert float(ls.max()) > 0.0  # fluid cells outside


class TestBladeSDFSign:
    def test_sdf_sign_inside_vs_outside_blade(self):
        importorskip("trimesh")
        from auraflow.body.sdf import sdf_eval, sdf_grid

        # Fat, untwisted blade with a tight z-box so the coarse grid resolves the
        # section thickness (a thin blade would fall below one z-cell and the
        # trilinear SDF would smear to positive everywhere).
        g = BladeGeometry.linear(
            radius=1.0,
            hub_radius=0.15,
            n_stations=12,
            chord_root=0.5,
            chord_tip=0.4,
            twist_root=0.0,
            twist_tip=0.0,
        )
        mesh = blade_mesh(g, n_chord=24)
        lo = np.array([0.1, -0.9, -0.1])
        hi = np.array([1.05, 0.5, 0.1])
        grid = sdf_grid(mesh, lo, hi, (24, 24, 20))
        # On the pitch axis at mid span (r=0.5, y=z=0) the point sits inside the
        # airfoil (quarter-chord, mid-thickness); far out in z it is outside.
        inside = jnp.array([0.5, 0.0, 0.0])
        outside = jnp.array([0.5, 0.0, 0.09])
        assert float(sdf_eval(grid, lo, hi, inside)) < 0.0
        assert float(sdf_eval(grid, lo, hi, outside)) > 0.0


def test_blade_and_rotor_mesh_are_watertight():
    # Cheap sanity (no jaxfluids/trimesh): the meshes fed to CFD are closed.
    g = BladeGeometry.linear(
        radius=1.0, hub_radius=0.15, n_stations=8,
        chord_root=0.2, chord_tip=0.15, twist_root=0.0, twist_tip=-0.2,
    )  # fmt: skip
    assert blade_mesh(g, n_chord=16).is_watertight
    assert rotor_mesh(Rotor(blade=g, n_blades=2), n_chord=16).is_watertight
