import jax
import numpy as np
import pytest
from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import CurveXYZFourier
from simsopt.gpu import biot_savart_field, biot_savart_field_reference


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


def test_tiled_field_vjp_matches_full_tensor():
    _, currents, gamma, gammadash, points = _coil_data()
    cotangent = np.arange(points.size, dtype=float).reshape(points.shape) / 100

    def tiled(g, dg, current):
        return biot_savart_field(
            points,
            g,
            dg,
            current,
            target_tile_size=6,
            source_tile_size=11,
        )

    def full(g, dg, current):
        return biot_savart_field_reference(points, g, dg, current)

    tiled_pullback = jax.vjp(tiled, gamma, gammadash, currents)[1](cotangent)
    full_pullback = jax.vjp(full, gamma, gammadash, currents)[1](cotangent)
    for actual, expected in zip(tiled_pullback, full_pullback):
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)


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
