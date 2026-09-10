import numpy as np
import pytest
from simsopt.field import apply_symmetries_to_curves
from simsopt.geo import CurveXYZFourier
from simsopt.gpu import (
    coefficients_to_dofs,
    curve_lengths,
    dofs_to_coefficients,
    evaluate_cartesian_fourier_derivatives,
    expand_by_symmetry,
    fourier_basis_set,
    symmetry_transforms,
)


@pytest.mark.parametrize("order", [0, 1, 3, 6])
def test_fourier_derivatives_match_curvexyzfourier(order):
    rng = np.random.default_rng(20260910 + order)
    quadpoints = np.linspace(0.0, 1.0, 37, endpoint=False)
    curve = CurveXYZFourier(quadpoints, order)
    dofs = rng.normal(scale=0.2, size=curve.dof_size)
    curve.x = dofs

    bases = fourier_basis_set(quadpoints, order, max_derivative=3)
    actual = np.asarray(
        evaluate_cartesian_fourier_derivatives(dofs_to_coefficients(dofs), bases)
    )
    expected = np.stack(
        [
            curve.gamma(),
            curve.gammadash(),
            curve.gammadashdash(),
            curve.gammadashdashdash(),
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_dof_conversion_round_trip_and_validation():
    dofs = np.arange(45.0).reshape(3, 15)
    np.testing.assert_array_equal(
        np.asarray(coefficients_to_dofs(dofs_to_coefficients(dofs))), dofs
    )
    with pytest.raises(ValueError, match="divisible by three"):
        dofs_to_coefficients(np.zeros(10))
    with pytest.raises(ValueError, match="2\\*order\\+1"):
        dofs_to_coefficients(np.zeros(12))


@pytest.mark.parametrize("nfp,stellsym", [(1, False), (2, False), (3, True)])
def test_symmetry_expansion_matches_simsopt(nfp, stellsym):
    order = 3
    nquad = 29
    curves = [CurveXYZFourier(nquad, order), CurveXYZFourier(nquad, order)]
    rng = np.random.default_rng(42)
    for curve in curves:
        curve.x = rng.normal(size=curve.dof_size)
    base_currents = np.asarray([1.2e5, -0.7e5])

    base_gamma = np.stack([curve.gamma() for curve in curves])
    base_gammadash = np.stack([curve.gammadash() for curve in curves])
    transforms, signs = symmetry_transforms(nfp, stellsym, dtype=np.float64)
    gamma, gammadash, currents = expand_by_symmetry(
        base_gamma, base_gammadash, base_currents, transforms, signs
    )

    expected_curves = apply_symmetries_to_curves(curves, nfp, stellsym)
    expected_gamma = np.stack([curve.gamma() for curve in expected_curves])
    expected_gammadash = np.stack([curve.gammadash() for curve in expected_curves])
    expected_currents = np.tile(
        np.concatenate(
            [base_currents, -base_currents] if stellsym else [base_currents]
        ),
        nfp,
    )

    np.testing.assert_allclose(gamma, expected_gamma, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(gammadash, expected_gammadash, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(currents, expected_currents)
    np.testing.assert_allclose(
        curve_lengths(base_gammadash),
        [np.mean(curve.incremental_arclength()) for curve in curves],
        rtol=2e-14,
        atol=2e-14,
    )
