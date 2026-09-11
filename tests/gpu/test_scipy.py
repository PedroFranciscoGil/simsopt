import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize
from simsopt.gpu import GpuConfig, ScipyCoilObjectiveBridge, ScipyObjectiveBridge


def test_scipy_bridge_compiles_and_optimizes():
    target = np.asarray([1.5, -2.0, 0.25])

    def objective(x):
        residual = x - target
        return 0.5 * residual @ residual

    initial_x = np.asarray([4.0, 3.0, -1.0])
    bridge = ScipyObjectiveBridge(objective, initial_x).compile(initial_x)
    assert bridge.is_compiled
    assert bridge.evaluations == 0

    result = minimize(bridge, initial_x, jac=True, method="L-BFGS-B")
    assert result.success
    np.testing.assert_allclose(result.x, target, rtol=0.0, atol=1e-12)
    assert bridge.evaluations == result.nfev


def test_scipy_bridge_validates_flat_shape_and_dtype():
    with pytest.raises(ValueError, match="one-dimensional"):
        ScipyObjectiveBridge(lambda x: x.sum(), np.zeros((2, 2)))
    with pytest.raises(TypeError, match="floating-point"):
        ScipyObjectiveBridge(lambda x: x.sum(), np.ones(2, dtype=int))

    bridge = ScipyObjectiveBridge(lambda x: (x * x).sum(), np.ones(2))
    with pytest.raises(ValueError, match="shape"):
        bridge(np.ones(3))


class ExampleCoilData:
    base_currents = np.asarray([10.0, 20.0, 30.0])
    curve_dofs = np.asarray([[1.0, 2.0], [3.0, 4.0]])

    def objective(self, curve_dofs, currents, *, targets, config):
        del config
        variables = jnp.concatenate((currents[jnp.asarray([0, 2])], curve_dofs.ravel()))
        residual = variables - jnp.asarray(targets)
        return 0.5 * residual @ residual


def test_coil_scipy_bridge_preserves_fixed_currents_and_simsopt_order():
    data = ExampleCoilData()
    targets = np.asarray([10.0, -2.0, 0.5, -0.5, 1.5, -1.5])
    bridge = ScipyCoilObjectiveBridge(
        data,
        free_current_indices=[2],
        objective_kwargs={"targets": targets},
        config=GpuConfig(),
    )
    np.testing.assert_array_equal(bridge.initial_x, [30.0, 1.0, 2.0, 3.0, 4.0])
    curve_dofs, currents = bridge.unpack(bridge.initial_x)
    np.testing.assert_array_equal(curve_dofs, data.curve_dofs)
    np.testing.assert_array_equal(currents, data.base_currents)

    bridge.compile()
    result = minimize(bridge, bridge.initial_x, jac=True, method="L-BFGS-B")
    assert result.success
    np.testing.assert_allclose(result.x, targets[1:], rtol=0.0, atol=1e-11)
    _, final_currents = bridge.unpack(result.x)
    assert final_currents[0] == 10.0
    assert final_currents[1] == 20.0


def test_coil_scipy_bridge_applies_current_coordinate_scale_and_chain_rule():
    data = ExampleCoilData()
    bridge = ScipyCoilObjectiveBridge(
        data,
        free_current_indices=[2],
        objective_kwargs={"targets": np.zeros(6)},
        current_scale=10.0,
    )

    np.testing.assert_array_equal(bridge.initial_x, [3.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(bridge.coordinate_scales, [10.0, 1, 1, 1, 1])
    np.testing.assert_array_equal(
        bridge.to_physical_variables(bridge.initial_x), bridge.physical_initial_x
    )
    _, currents = bridge.unpack(bridge.initial_x)
    np.testing.assert_array_equal(currents, data.base_currents)
    np.testing.assert_array_equal(
        bridge.pullback_gradient(np.ones(5)), [10.0, 1, 1, 1, 1]
    )
    value, gradient = bridge(bridge.initial_x)
    assert value == 515.0
    np.testing.assert_array_equal(gradient, [300.0, 1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("indices", [[1, 1], [-1], [3]])
def test_coil_scipy_bridge_validates_free_current_indices(indices):
    with pytest.raises(ValueError, match="free_current_indices"):
        ScipyCoilObjectiveBridge(
            ExampleCoilData(),
            free_current_indices=indices,
            objective_kwargs={"targets": np.zeros(6)},
        )


@pytest.mark.parametrize("scale", [0.0, -1.0, np.inf, np.nan])
def test_coil_scipy_bridge_validates_current_scale(scale):
    with pytest.raises(ValueError, match="current_scale"):
        ScipyCoilObjectiveBridge(
            ExampleCoilData(),
            free_current_indices=[2],
            objective_kwargs={"targets": np.zeros(6)},
            current_scale=scale,
        )
