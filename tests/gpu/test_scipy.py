import numpy as np
import pytest
from scipy.optimize import minimize
from simsopt.gpu import ScipyObjectiveBridge


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
