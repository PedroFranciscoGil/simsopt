import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import CurveXYZFourier
from simsopt.gpu import (
    biot_savart_field,
    biot_savart_field_custom_vjp,
    biot_savart_field_reference,
)


def _coil_data():
    rng = np.random.default_rng(17)
    order = 3
    curves = [CurveXYZFourier(31, order), CurveXYZFourier(31, order)]
    for index, curve in enumerate(curves):
        dofs = rng.normal(scale=0.05, size=curve.dof_size)
        dofs[0] += 1.0 + 0.25 * index
        dofs[2 * order + 2] += 0.25
        dofs[4 * order + 2] += 0.15
        curve.x = dofs
    currents = np.asarray([1.1e5, -0.6e5])
    gamma = np.stack([curve.gamma() for curve in curves])
    gammadash = np.stack([curve.gammadash() for curve in curves])
    points = rng.normal(scale=0.2, size=(23, 3))
    return curves, currents, gamma, gammadash, points


@pytest.mark.parametrize(
    "target_tile_size,source_tile_size", [(5, 7), (23, 62), (32, 128)]
)
def test_tiled_field_matches_full_tensor_and_cpp(target_tile_size, source_tile_size):
    curves, currents, gamma, gammadash, points = _coil_data()
    actual = np.asarray(
        biot_savart_field(
            points,
            gamma,
            gammadash,
            currents,
            target_tile_size=target_tile_size,
            source_tile_size=source_tile_size,
        )
    )
    tensor_reference = np.asarray(
        biot_savart_field_reference(points, gamma, gammadash, currents)
    )

    bs = BiotSavart(
        [Coil(curve, Current(current)) for curve, current in zip(curves, currents)]
    )
    bs.set_points(points)
    cpp_reference = bs.B()

    np.testing.assert_allclose(actual, tensor_reference, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual, cpp_reference, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize(
    "field_function", [biot_savart_field, biot_savart_field_custom_vjp]
)
def test_tiled_field_vjp_matches_full_tensor(field_function):
    _, currents, gamma, gammadash, points = _coil_data()
    cotangent = np.arange(points.size, dtype=float).reshape(points.shape) / 100

    def tiled(targets, g, dg, current):
        return field_function(
            targets,
            g,
            dg,
            current,
            target_tile_size=6,
            source_tile_size=11,
        )

    def full(targets, g, dg, current):
        return biot_savart_field_reference(targets, g, dg, current)

    tiled_pullback = jax.vjp(tiled, points, gamma, gammadash, currents)[1](cotangent)
    full_pullback = jax.vjp(full, points, gamma, gammadash, currents)[1](cotangent)
    for actual, expected in zip(tiled_pullback, full_pullback):
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)


def test_custom_vjp_field_matches_primal_with_padding():
    _, currents, gamma, gammadash, points = _coil_data()
    expected = biot_savart_field(
        points,
        gamma,
        gammadash,
        currents,
        target_tile_size=6,
        source_tile_size=11,
    )
    actual = biot_savart_field_custom_vjp(
        points,
        gamma,
        gammadash,
        currents,
        target_tile_size=6,
        source_tile_size=11,
    )
    np.testing.assert_array_equal(actual, expected)


def test_custom_vjp_centered_directional_derivative():
    _, currents, gamma, gammadash, points = _coil_data()
    rng = np.random.default_rng(91)
    cotangent = rng.normal(size=points.shape)
    inputs = (points, gamma, gammadash, currents)
    directions = tuple(
        direction / np.linalg.norm(direction)
        for direction in (rng.normal(size=value.shape) for value in inputs)
    )

    def contracted_field(targets, positions, tangents, current_values):
        field = biot_savart_field_custom_vjp(
            targets,
            positions,
            tangents,
            current_values,
            target_tile_size=6,
            source_tile_size=11,
        )
        return np.asarray(jnp.vdot(cotangent, field))

    def field(*values):
        return biot_savart_field_custom_vjp(
            *values, target_tile_size=6, source_tile_size=11
        )

    pullback = jax.vjp(field, *inputs)[1](cotangent)
    analytic = sum(
        np.vdot(np.asarray(gradient), direction)
        for gradient, direction in zip(pullback, directions)
    )
    step = 1e-5
    plus = contracted_field(
        *(value + step * direction for value, direction in zip(inputs, directions))
    )
    minus = contracted_field(
        *(value - step * direction for value, direction in zip(inputs, directions))
    )
    centered = (plus - minus) / (2 * step)
    np.testing.assert_allclose(analytic, centered, rtol=2e-8, atol=2e-10)


def test_biot_savart_rejects_invalid_shapes_and_tiles():
    points = np.zeros((2, 3))
    gamma = np.zeros((1, 4, 3))
    gammadash = np.zeros_like(gamma)
    currents = np.ones(1)
    with pytest.raises(ValueError, match="tile sizes"):
        biot_savart_field(points, gamma, gammadash, currents, target_tile_size=0)
    with pytest.raises(ValueError, match="gammadash"):
        biot_savart_field(points, gamma, np.zeros((1, 3, 3)), currents)
    with pytest.raises(ValueError, match="currents"):
        biot_savart_field(points, gamma, gammadash, np.ones(2))


def test_biot_savart_promotes_inputs_to_a_common_dtype():
    _, currents, gamma, gammadash, points = _coil_data()
    field = biot_savart_field(
        points.astype(np.float32),
        gamma,
        gammadash,
        currents,
        target_tile_size=8,
        source_tile_size=19,
    )
    assert field.dtype == np.dtype("float64")
