import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt.geo.curveobjectives import cc_distance_pure, cs_distance_pure
from simsopt.gpu import (
    coil_coil_distance,
    coil_surface_distance,
    curve_pair_indices,
)


def test_curve_pair_indices_match_simsopt_symmetry_rule():
    pairs = curve_pair_indices(8, 2)
    expected = [(i, j) for i in range(8) for j in range(i) if j < 2]
    np.testing.assert_array_equal(pairs, expected)
    assert curve_pair_indices(0).shape == (0, 2)
    with pytest.raises(ValueError, match="num_basecurves"):
        curve_pair_indices(4, 5)


def test_coil_coil_all_pairs_value_and_gradient_match_reference():
    rng = np.random.default_rng(104)
    gamma = rng.normal(size=(4, 9, 3))
    gammadash = rng.normal(size=gamma.shape)
    pairs = curve_pair_indices(4, 2)
    threshold = 2.4

    def reference(points, tangents):
        return sum(
            cc_distance_pure(points[i], tangents[i], points[j], tangents[j], threshold)
            for i, j in pairs
        )

    reference_value, reference_gradient = jax.value_and_grad(reference, argnums=(0, 1))(
        jnp.asarray(gamma), jnp.asarray(gammadash)
    )
    value, gradient = jax.jit(jax.value_and_grad(coil_coil_distance, argnums=(0, 1)))(
        gamma, gammadash, pairs, threshold
    )

    np.testing.assert_allclose(value, reference_value, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        gradient[0], reference_gradient[0], rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        gradient[1], reference_gradient[1], rtol=2e-13, atol=2e-13
    )


def test_tiled_coil_surface_value_and_gradient_match_reference():
    rng = np.random.default_rng(105)
    gamma = rng.normal(size=(3, 8, 3))
    gammadash = rng.normal(size=gamma.shape)
    surface_points = rng.normal(size=(17, 3))
    surface_normal = rng.normal(size=(17, 3))
    threshold = 2.2

    def reference(points, tangents):
        return sum(
            cs_distance_pure(
                points[i],
                tangents[i],
                surface_points,
                surface_normal,
                threshold,
            )
            for i in range(points.shape[0])
        )

    def tiled(points, tangents):
        return coil_surface_distance(
            points,
            tangents,
            surface_points,
            surface_normal,
            threshold,
            target_tile_size=5,
        )

    reference_value, reference_gradient = jax.value_and_grad(reference, argnums=(0, 1))(
        jnp.asarray(gamma), jnp.asarray(gammadash)
    )
    value, gradient = jax.jit(jax.value_and_grad(tiled, argnums=(0, 1)))(
        gamma, gammadash
    )

    np.testing.assert_allclose(value, reference_value, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        gradient[0], reference_gradient[0], rtol=3e-13, atol=3e-13
    )
    np.testing.assert_allclose(
        gradient[1], reference_gradient[1], rtol=3e-13, atol=3e-13
    )
