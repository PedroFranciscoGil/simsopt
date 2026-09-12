import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt.gpu import (
    ScipyAugmentedLagrangianBridge,
    equality_augmented_lagrangian,
    minimize_equality_augmented_lagrangian,
)


def test_equality_augmented_lagrangian_value_and_gradient():
    lagrange_multipliers = jnp.asarray([2.0, -1.0])
    penalties = jnp.asarray([4.0, 3.0])

    def objective(x):
        base = 0.5 * jnp.vdot(x, x)
        constraints = jnp.asarray([x[0] - 1.0, x[1] + 2.0])
        return equality_augmented_lagrangian(
            base, constraints, lagrange_multipliers, penalties
        )

    x = jnp.asarray([3.0, -1.0])
    value, gradient = jax.value_and_grad(objective)(x)
    expected_constraints = np.asarray([2.0, 1.0])
    expected_value = (
        0.5 * np.dot(x, x)
        - np.dot(lagrange_multipliers, expected_constraints)
        + 0.5 * np.dot(penalties, expected_constraints**2)
    )
    expected_gradient = np.asarray(x) - np.asarray(lagrange_multipliers)
    expected_gradient += np.asarray(penalties) * expected_constraints
    np.testing.assert_allclose(value, expected_value)
    np.testing.assert_allclose(gradient, expected_gradient)


def test_dynamic_state_bridge_reuses_compilation():
    def terms(x):
        return 0.5 * jnp.vdot(x, x), jnp.asarray([x[0] - 1.0])

    initial_x = np.asarray([2.0, -1.0])
    bridge = ScipyAugmentedLagrangianBridge(terms, initial_x, 1)
    bridge.set_state([0.0], [10.0]).compile(initial_x)
    compiled = bridge._compiled
    first_value, first_gradient = bridge(initial_x)
    bridge.set_state([-3.0], [20.0])
    second_value, second_gradient = bridge(initial_x)

    assert bridge._compiled is compiled
    assert second_value != first_value
    assert not np.array_equal(second_gradient, first_gradient)
    base, constraints = bridge.evaluate_terms(initial_x)
    assert base == 2.5
    np.testing.assert_array_equal(constraints, [1.0])


def test_augmented_lagrangian_solver_reaches_equality_solution():
    def terms(x):
        base = 0.5 * (x[0] - 3.0) ** 2 + 0.5 * x[1] ** 2
        # The production constraints are likewise nonnegative penalty
        # objectives whose zero level set represents feasibility.
        return base, jnp.asarray([0.5 * (x[0] - 1.0) ** 2])

    initial_x = np.asarray([4.0, 2.0])
    bridge = ScipyAugmentedLagrangianBridge(terms, initial_x, 1).compile(initial_x)
    result = minimize_equality_augmented_lagrangian(
        bridge,
        initial_x,
        mu_init=10.0,
        tau=10.0,
        max_outer_iterations=12,
        max_inner_iterations=50,
        gradient_tolerance=1e-6,
        constraint_tolerance=1e-7,
        mu_max=1e12,
    )

    assert result.success
    np.testing.assert_allclose(result.x, [1.0, 0.0], atol=2e-4)
    assert abs(result.constraints[0]) <= 1e-7
    assert result.total_evaluations == bridge.evaluations
    assert len(result.history) == result.outer_iterations
    assert result.history[0]["penalties_after"] == [100.0]


def test_augmented_lagrangian_validates_penalties():
    def terms(x):
        return jnp.vdot(x, x), jnp.asarray([x[0]])

    initial_x = np.asarray([1.0])
    bridge = ScipyAugmentedLagrangianBridge(terms, initial_x, 1)
    with pytest.raises(ValueError, match="mu_init"):
        minimize_equality_augmented_lagrangian(bridge, initial_x, mu_init=1.0)
