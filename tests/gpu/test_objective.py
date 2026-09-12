from pathlib import Path

import jax
import numpy as np
import pytest
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    ArclengthVariation,
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    LpCurveCurvature,
    MeanSquaredCurvature,
    SurfaceRZFourier,
    create_equally_spaced_curves,
)
from simsopt.gpu import (
    curve_pair_indices,
    fourier_basis_set,
    minimal_coil_augmented_lagrangian_terms,
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
    curvature_threshold = 0.7
    curvature_weight = 2e-4
    msc_threshold = 1.0
    msc_weight = 3e-4
    arclength_weight = 5e-3

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
    cpu_objective += curvature_weight * sum(
        LpCurveCurvature(curve, 2.0, curvature_threshold) for curve in base_curves
    )
    cpu_objective += msc_weight * sum(
        QuadraticPenalty(MeanSquaredCurvature(curve), msc_threshold, "max")
        for curve in base_curves
    )
    cpu_objective += arclength_weight * sum(
        ArclengthVariation(curve, nintervals="full") for curve in base_curves
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
            curvature_threshold=curvature_threshold,
            curvature_weight=curvature_weight,
            mean_squared_curvature_threshold=msc_threshold,
            mean_squared_curvature_weight=msc_weight,
            arclength_variation_weight=arclength_weight,
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


@pytest.mark.parametrize("vjp_mode", ["autodiff", "custom"])
def test_full_engineering_objective_and_gradients_match_simsopt(vjp_mode):
    nbase, order, nquad, nfp = 2, 3, 27, 2
    surface = SurfaceRZFourier.from_vmec_input(
        TEST_FILE, range="half period", nphi=5, ntheta=6
    )
    base_curves = create_equally_spaced_curves(
        nbase,
        nfp,
        True,
        R0=1.0,
        R1=0.5,
        order=order,
        numquadpoints=nquad,
    )
    base_current_objects = [Current(1.0e5), Current(-0.8e5)]
    coils = coils_via_symmetries(base_curves, base_current_objects, nfp, stellsym=True)
    physical_curves = [coil.curve for coil in coils]
    field = BiotSavart(coils)
    length_weight = 1e-6
    cc_weight, cc_threshold = 1000.0, 0.7
    cs_weight, cs_threshold = 10.0, 0.7
    curvature_weight, curvature_threshold = 1e-6, 5.0
    msc_weight, msc_threshold = 1e-6, 5.0

    cpu_flux = SquaredFlux(surface, field)
    cpu_length = sum(CurveLength(curve) for curve in base_curves)
    cpu_coil_coil = CurveCurveDistance(
        physical_curves, cc_threshold, num_basecurves=nbase
    )
    cpu_coil_surface = CurveSurfaceDistance(physical_curves, surface, cs_threshold)
    cpu_curvature = sum(
        LpCurveCurvature(curve, 2.0, curvature_threshold) for curve in base_curves
    )
    cpu_msc = sum(
        QuadraticPenalty(MeanSquaredCurvature(curve), msc_threshold, "max")
        for curve in base_curves
    )
    cpu_objective = cpu_flux + length_weight * cpu_length
    cpu_objective += cc_weight * cpu_coil_coil
    cpu_objective += cs_weight * cpu_coil_surface
    cpu_objective += curvature_weight * cpu_curvature
    cpu_objective += msc_weight * cpu_msc

    curve_dofs = np.stack([curve.get_dofs() for curve in base_curves])
    base_currents = np.asarray(
        [current.get_value() for current in base_current_objects]
    )
    bases = fourier_basis_set(base_curves[0].quadpoints, order)
    transforms, current_signs = symmetry_transforms(nfp, True)
    surface_points = surface.gamma().reshape((-1, 3))
    surface_normal = surface.normal().reshape((-1, 3))
    target = np.zeros(surface_points.shape[0])
    pairs = curve_pair_indices(len(physical_curves), nbase)

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
            length_target=None,
            length_weight=length_weight,
            curvature_threshold=curvature_threshold,
            curvature_weight=curvature_weight,
            mean_squared_curvature_threshold=msc_threshold,
            mean_squared_curvature_weight=msc_weight,
            coil_coil_pair_indices=pairs,
            coil_coil_distance_threshold=cc_threshold,
            coil_coil_distance_weight=cc_weight,
            coil_surface_distance_threshold=cs_threshold,
            coil_surface_distance_weight=cs_weight,
            target_tile_size=7,
            source_tile_size=13,
            vjp_mode=vjp_mode,
        )

    value, (curve_gradient, current_gradient) = jax.jit(
        jax.value_and_grad(objective, argnums=(0, 1))
    )(curve_dofs, base_currents)
    base_value, constraint_values = jax.jit(
        lambda dofs, currents: minimal_coil_augmented_lagrangian_terms(
            dofs,
            currents,
            bases,
            transforms,
            current_signs,
            surface_points,
            surface_normal,
            target,
            length_weight=length_weight,
            curvature_threshold=curvature_threshold,
            mean_squared_curvature_threshold=msc_threshold,
            coil_coil_pair_indices=pairs,
            coil_coil_distance_threshold=cc_threshold,
            coil_surface_distance_threshold=cs_threshold,
            target_tile_size=7,
            source_tile_size=13,
            vjp_mode=vjp_mode,
        )
    )(curve_dofs, base_currents)
    cpu_derivative = cpu_objective.dJ(partials=True)

    np.testing.assert_allclose(value, cpu_objective.J(), rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(
        base_value, cpu_flux.J() + length_weight * cpu_length.J(), rtol=3e-11
    )
    np.testing.assert_allclose(
        constraint_values,
        [
            cpu_coil_coil.J(),
            cpu_coil_surface.J(),
            cpu_curvature.J(),
            cpu_msc.J(),
        ],
        rtol=3e-11,
        atol=3e-11,
    )
    for index, curve in enumerate(base_curves):
        np.testing.assert_allclose(
            curve_gradient[index], cpu_derivative(curve), rtol=4e-9, atol=4e-9
        )
    for index, current in enumerate(base_current_objects):
        np.testing.assert_allclose(
            current_gradient[index], cpu_derivative(current), rtol=2e-9, atol=2e-12
        )
