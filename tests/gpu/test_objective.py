from pathlib import Path

import jax
import numpy as np
import pytest
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import CurveLength, SurfaceRZFourier, create_equally_spaced_curves
from simsopt.gpu import (
    fourier_basis_set,
    minimal_coil_objective,
    normalized_flux,
    quadratic_flux,
    symmetry_transforms,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux

TEST_FILE = (
    Path(__file__).parent / ".." / "test_files" / "input.LandremanPaul2021_QA"
).resolve()


def test_quadratic_flux_matches_definition():
    rng = np.random.default_rng(81)
    field = rng.normal(size=(17, 3))
    normal = rng.normal(size=(17, 3))
    target = rng.normal(size=17)
    unit_normal = normal / np.linalg.norm(normal, axis=1)[:, None]
    expected = 0.5 * np.mean(
        (np.sum(field * unit_normal, axis=1) - target) ** 2
        * np.linalg.norm(normal, axis=1)
    )
    np.testing.assert_allclose(
        quadratic_flux(field, normal, target), expected, rtol=2e-15, atol=2e-15
    )

    denominator = np.mean(
        np.sum(field * field, axis=1) * np.linalg.norm(normal, axis=1)
    )
    np.testing.assert_allclose(
        normalized_flux(field, normal, target),
        expected / denominator,
        rtol=2e-15,
        atol=2e-15,
    )


@pytest.mark.parametrize("vjp_mode", ["autodiff", "custom"])
def test_compiled_minimal_objective_and_gradients_match_simsopt(vjp_mode):
    nbase = 2
    order = 3
    nquad = 35
    nfp = 2
    stellsym = True
    length_target = 4.0
    length_weight = 0.03

    surface = SurfaceRZFourier.from_vmec_input(
        TEST_FILE, range="half period", nphi=7, ntheta=8
    )
    base_curves = create_equally_spaced_curves(
        nbase,
        nfp,
        stellsym,
        R0=1.0,
        R1=0.5,
        order=order,
        numquadpoints=nquad,
    )
    rng = np.random.default_rng(123)
    for curve in base_curves:
        curve.x = curve.x + rng.normal(scale=2e-3, size=curve.dof_size)
    base_current_objects = [Current(1.0e5), Current(-0.8e5)]
    coils = coils_via_symmetries(base_curves, base_current_objects, nfp, stellsym)
    bs = BiotSavart(coils)
    cpu_flux = SquaredFlux(surface, bs)
    cpu_objective = cpu_flux + length_weight * QuadraticPenalty(
        sum(CurveLength(curve) for curve in base_curves),
        length_target,
        "max",
    )

    curve_dofs = np.stack([curve.get_dofs() for curve in base_curves])
    base_currents = np.asarray(
        [current.get_value() for current in base_current_objects]
    )
    bases = fourier_basis_set(base_curves[0].quadpoints, order)
    transforms, current_signs = symmetry_transforms(nfp, stellsym)
    surface_points = surface.gamma().reshape((-1, 3))
    surface_normal = surface.normal().reshape((-1, 3))
    target = np.zeros(surface_points.shape[0])

    def objective(dofs, currents):
        return minimal_coil_objective(
            dofs,
            currents,
            bases,
            transforms,
            current_signs,
            surface_points,
            surface_normal,
            target,
            length_target=length_target,
            length_weight=length_weight,
            target_tile_size=9,
            source_tile_size=17,
            vjp_mode=vjp_mode,
        )

    value, (curve_gradient, current_gradient) = jax.jit(
        jax.value_and_grad(objective, argnums=(0, 1))
    )(curve_dofs, base_currents)
    cpu_derivative = cpu_objective.dJ(partials=True)

    np.testing.assert_allclose(value, cpu_objective.J(), rtol=2e-11, atol=2e-11)
    for index, curve in enumerate(base_curves):
        np.testing.assert_allclose(
            curve_gradient[index],
            cpu_derivative(curve),
            rtol=2e-9,
            atol=2e-9,
        )
    for index, current in enumerate(base_current_objects):
        np.testing.assert_allclose(
            current_gradient[index],
            cpu_derivative(current),
            rtol=2e-9,
            atol=2e-12,
        )
