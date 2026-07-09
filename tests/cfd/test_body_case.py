"""Level-set FLUID-SOLID bodies + permeable mesh surfaces.

The permeable-mesh checks are pure JAX. The level-set case checks validate the
programmatic JAX-Fluids case dict through the real ``InputManager`` (and, for the
sign check, the initialized level-set buffer), so they ``importorskip`` jaxfluids.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from auraflow.body.mesh import TriMesh
from auraflow.body.motion import ConstantVelocity, StaticPose, WaypointMotion
from auraflow.cfd.body_case import (
    LevelsetBodyCase,
    PermeableMeshSurface,
    levelset_body_case,
    permeable_mesh_surface,
)
from auraflow.cfd.sphere import PermeableSphere, sample_primitives


class TestPermeableMeshSurface:
    def test_duck_types_permeable_sphere(self):
        surf = permeable_mesh_surface(TriMesh.sphere(1.0, subdivisions=2))
        assert isinstance(surf, PermeableMeshSurface)
        # Same field names/shapes as PermeableSphere (points/normals/area).
        assert surf.points.shape == (surf.n_points, 3)
        assert surf.normals.shape == (surf.n_points, 3)
        assert surf.area.shape == (surf.n_points,)

    def test_areas_and_normals_match_sphere(self):
        radius = 1.3
        mesh = TriMesh.sphere(radius, subdivisions=4)
        surf = permeable_mesh_surface(mesh)
        # Total area ~ 4 pi r^2, same target the Fibonacci sphere sums to exactly.
        sph = PermeableSphere.fibonacci(surf.n_points, radius=radius)
        assert float(jnp.sum(surf.area)) == pytest.approx(float(jnp.sum(sph.area)), rel=2e-3)
        # Outward normals: aligned with the radial direction on a centred sphere.
        radial = surf.points / jnp.linalg.norm(surf.points, axis=-1, keepdims=True)
        assert float(jnp.min(jnp.sum(surf.normals * radial, axis=-1))) > 0.9

    def test_sampling_shape_on_mesh_points(self):
        surf = permeable_mesh_surface(TriMesh.sphere(0.5, subdivisions=2))
        # Constant primitive field -> sampled values equal the constant.
        n = 8
        x = jnp.linspace(-1, 1, n)
        prim = jnp.stack([jnp.full((n, n, n), v) for v in (1.2, 0.0, 0.0, 0.0, 5.0)])
        rho, u, p = sample_primitives(prim, x, x, x, surf.points)
        assert rho.shape == (surf.n_points,)
        assert u.shape == (surf.n_points, 3)
        assert p.shape == (surf.n_points,)
        assert float(jnp.mean(rho)) == pytest.approx(1.2, rel=1e-5)
        assert float(jnp.mean(p)) == pytest.approx(5.0, rel=1e-5)


class TestLevelsetInitField:
    """SDF sign convention -- pure array checks (no jaxfluids)."""

    def test_static_case_levelset_negative_inside_positive_outside(self):
        mesh = TriMesh.sphere(0.2, subdivisions=2)
        case = levelset_body_case(
            mesh, box_lo=(-0.5, -0.5, -0.5), box_hi=(0.5, 0.5, 0.5), cells=(16, 16, 16)
        )
        assert isinstance(case, LevelsetBodyCase)
        assert case.is_moving is False
        ls = np.asarray(case.levelset_init)
        assert ls.shape == (16, 16, 16)
        # JAX-Fluids convention: negative inside the solid, positive in the fluid.
        # Centre cells sit inside the r=0.2 sphere; corner cells are outside.
        assert ls[8, 8, 8] < 0.0
        assert ls[0, 0, 0] > 0.0
        # |SDF| ~ distance: the near-surface interior magnitude is < the radius.
        assert abs(ls[8, 8, 8]) < 0.2


importorskip = pytest.importorskip


class TestLevelsetCaseValidation:
    """The programmatic case dicts must validate through the real InputManager."""

    def _input_manager(self, case: LevelsetBodyCase):
        importorskip("jaxfluids")
        from jaxfluids import InputManager

        return InputManager(case.case, case.numerical_setup)

    def test_static_sphere_validates(self):
        mesh = TriMesh.sphere(0.2, subdivisions=2)
        case = levelset_body_case(
            mesh,
            StaticPose(),
            box_lo=(-0.5, -0.5, -0.5),
            box_hi=(0.5, 0.5, 0.5),
            cells=(16, 16, 16),
        )
        im = self._input_manager(case)
        assert im.equation_information.levelset_model == "FLUID-SOLID"
        assert im.equation_information.is_moving_levelset is False

    def test_moving_constant_velocity_validates(self):
        mesh = TriMesh.sphere(0.2, subdivisions=2)
        case = levelset_body_case(
            mesh,
            ConstantVelocity((0.0, 0.0, 0.0), (30.0, 0.0, 0.0)),
            box_lo=(-0.5, -0.5, -0.5),
            box_hi=(0.5, 0.5, 0.5),
            cells=(16, 16, 16),
            mach_max=0.1,
        )
        assert case.is_moving is True
        assert case.case["solid_properties"]["velocity"]["u"] == pytest.approx(30.0)
        im = self._input_manager(case)
        assert im.equation_information.is_moving_levelset is True

    def test_levelset_sign_matches_jaxfluids_buffer(self):
        importorskip("jaxfluids")
        from jaxfluids import InitializationManager

        mesh = TriMesh.sphere(0.2, subdivisions=2)
        case = levelset_body_case(
            mesh, box_lo=(-0.5, -0.5, -0.5), box_hi=(0.5, 0.5, 0.5), cells=(16, 16, 16)
        )
        im = self._input_manager(case)
        init = InitializationManager(im)
        buf = init.initialization(user_levelset_init=jnp.asarray(case.levelset_init))
        lsf = np.asarray(buf.simulation_buffers.levelset_fields.levelset)
        # Interior centre cell is inside the solid: the ingested level-set is < 0
        # there (JAX-Fluids stores negative-inside, our SDF convention verbatim).
        c = lsf.shape[0] // 2
        assert lsf[c, c, c] < 0.0

    def test_unsupported_motion_raises(self):
        mesh = TriMesh.sphere(0.2, subdivisions=1)
        waypoint = WaypointMotion(
            np.array([0.0, 1.0]), np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        with pytest.raises(NotImplementedError, match="solid_velocity"):
            levelset_body_case(
                mesh,
                waypoint,
                box_lo=(-0.5, -0.5, -0.5),
                box_hi=(0.5, 0.5, 0.5),
                cells=(16, 16, 16),
            )

    def test_requires_3d_box(self):
        mesh = TriMesh.sphere(0.2, subdivisions=1)
        with pytest.raises(ValueError, match="3-D box"):
            levelset_body_case(
                mesh,
                box_lo=(-0.5, -0.5, -0.5),
                box_hi=(0.5, 0.5, 0.5),
                cells=(16, 16, 1),
            )
