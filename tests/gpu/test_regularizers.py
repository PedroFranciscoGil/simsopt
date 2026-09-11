import jax
import numpy as np
import pytest
from simsopt.geo import (
    ArclengthVariation,
    CurveXYZFourier,
    LpCurveCurvature,
    MeanSquaredCurvature,
)
from simsopt.gpu import (
    arclength_variation,
    curve_curvatures,
    dofs_to_coefficients,
    evaluate_cartesian_fourier_derivatives,
    fourier_basis_set,
    lp_curve_curvature_penalty,
    mean_squared_curvature,
)


def perturbed_curves():
    rng = np.random.default_rng(20260911)
    curves = [CurveXYZFourier(41, 4), CurveXYZFourier(41, 4)]
    for index, curve in enumerate(curves):
        curve.set("xc(0)", 1.0)
        curve.set("xc(1)", 0.45 + 0.04 * index)
        curve.set("ys(1)", 0.45 + 0.04 * index)
        curve.set("zs(1)", 0.06)
        curve.x = curve.x + rng.normal(scale=3e-3, size=curve.dof_size)
    return curves


def curve_derivatives(curves):
    dofs = np.stack([curve.get_dofs() for curve in curves])
    bases = fourier_basis_set(curves[0].quadpoints, curves[0].order)
    derivatives = evaluate_cartesian_fourier_derivatives(
        dofs_to_coefficients(dofs), bases[:3]
    )
    return dofs, derivatives[1], derivatives[2]


def test_curve_local_regularizer_values_match_simsopt():
    curves = perturbed_curves()
    _, gammadash, gammadashdash = curve_derivatives(curves)

    np.testing.assert_allclose(
        curve_curvatures(gammadash, gammadashdash),
        np.stack([curve.kappa() for curve in curves]),
        rtol=3e-13,
        atol=3e-13,
    )
    np.testing.assert_allclose(
        lp_curve_curvature_penalty(
            gammadash, gammadashdash, p=2.0, threshold=0.7
        ),
        [LpCurveCurvature(curve, 2.0, 0.7).J() for curve in curves],
        rtol=3e-13,
        atol=3e-13,
    )
    np.testing.assert_allclose(
        mean_squared_curvature(gammadash, gammadashdash),
        [MeanSquaredCurvature(curve).J() for curve in curves],
        rtol=3e-13,
        atol=3e-13,
    )
    np.testing.assert_allclose(
        arclength_variation(gammadash),
        [ArclengthVariation(curve, nintervals="full").J() for curve in curves],
        rtol=3e-13,
        atol=3e-13,
    )


def test_curve_local_regularizer_combined_gradient_matches_simsopt():
    curves = perturbed_curves()
    dofs, _, _ = curve_derivatives(curves)
    bases = fourier_basis_set(curves[0].quadpoints, curves[0].order)

    def device_objective(curve_dofs):
        derivatives = evaluate_cartesian_fourier_derivatives(
            dofs_to_coefficients(curve_dofs), bases[:3]
        )
        gammadash, gammadashdash = derivatives[1], derivatives[2]
        curvature = jax.numpy.sum(
            lp_curve_curvature_penalty(
                gammadash, gammadashdash, p=2.0, threshold=0.7
            )
        )
        msc = jax.numpy.sum(mean_squared_curvature(gammadash, gammadashdash))
        arclength = jax.numpy.sum(arclength_variation(gammadash))
        return 0.4 * curvature + 0.2 * msc + 0.1 * arclength

    device_gradient = jax.jit(jax.grad(device_objective))(dofs)
    cpu_objective = sum(
        0.4 * LpCurveCurvature(curve, 2.0, 0.7)
        + 0.2 * MeanSquaredCurvature(curve)
        + 0.1 * ArclengthVariation(curve, nintervals="full")
        for curve in curves
    )
    cpu_derivative = cpu_objective.dJ(partials=True)
    for index, curve in enumerate(curves):
        np.testing.assert_allclose(
            device_gradient[index],
            cpu_derivative(curve),
            rtol=2e-10,
            atol=2e-10,
        )


def test_curve_local_regularizer_validation():
    with pytest.raises(ValueError, match="gammadash"):
        curve_curvatures(np.zeros((4, 2)), np.zeros((4, 2)))
    with pytest.raises(ValueError, match="match"):
        curve_curvatures(np.zeros((4, 3)), np.zeros((5, 3)))
    with pytest.raises(ValueError, match="positive"):
        lp_curve_curvature_penalty(
            np.ones((4, 3)), np.ones((4, 3)), p=0.0
        )


def test_curvature_threshold_activation_is_exact():
    points = np.linspace(0.0, 1.0, 32, endpoint=False)
    angle = 2 * np.pi * points
    gammadash = 2 * np.pi * np.stack(
        (-np.sin(angle), np.cos(angle), np.zeros_like(angle)), axis=-1
    )
    gammadashdash = (2 * np.pi) ** 2 * np.stack(
        (-np.cos(angle), -np.sin(angle), np.zeros_like(angle)), axis=-1
    )

    assert lp_curve_curvature_penalty(
        gammadash, gammadashdash, p=2.0, threshold=1.0
    ) == pytest.approx(0.0, abs=1e-28)
    assert lp_curve_curvature_penalty(
        gammadash, gammadashdash, p=2.0, threshold=1.0 + 1e-8
    ) == pytest.approx(0.0, abs=1e-28)
    assert lp_curve_curvature_penalty(
        gammadash, gammadashdash, p=2.0, threshold=1.0 - 1e-8
    ) > 0.0
