import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt.gpu import (
    ScipyAugmentedLagrangianBridge,
    equality_augmented_lagrangian,
    minimize_equality_augmented_lagrangian,
    smooth_abs_constraints,
    smooth_sqrt_constraints,
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


def test_dynamic_state_bridge_can_pin_cpu_platform():
    def terms(x):
        return jnp.vdot(x, x), jnp.asarray([x[0]])

    initial_x = np.asarray([2.0])
    bridge = ScipyAugmentedLagrangianBridge(
        terms, initial_x, 1, platform="cpu"
    ).set_state([0.0], [10.0])

    value, gradient = bridge(initial_x)

    assert bridge.device_platform == "cpu"
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))


def test_dynamic_state_bridge_scales_only_augmented_coordinates():
    def terms(x):
        return 0.5 * jnp.vdot(x, x), jnp.asarray([x[0] - 1.0])

    initial_x = np.asarray([3.0])
    bridge = ScipyAugmentedLagrangianBridge(
        terms, initial_x, 1, constraint_scales=[2.0]
    )
    bridge.set_state([1.5], [4.0]).compile(initial_x)

    base, raw_constraints = bridge.evaluate_terms(initial_x)
    scaled_constraints = bridge.scale_constraints(raw_constraints)
    value, gradient = bridge(initial_x)

    assert base == 4.5
    np.testing.assert_array_equal(raw_constraints, [2.0])
    np.testing.assert_array_equal(scaled_constraints, [1.0])
    assert value == pytest.approx(5.0)
    np.testing.assert_allclose(gradient, [4.25])
    np.testing.assert_array_equal(bridge.constraint_scales, [2.0])


def test_smooth_sqrt_constraint_mapping_value_and_gradient():
    epsilon = 0.25
    constraints = jnp.asarray([0.0, 4.0])
    values = smooth_sqrt_constraints(constraints, epsilon)
    gradient = jax.grad(lambda value: smooth_sqrt_constraints(value[None], epsilon)[0])(
        jnp.asarray(4.0)
    )

    np.testing.assert_allclose(values, np.sqrt([epsilon**2, 4 + epsilon**2]) - epsilon)
    assert values[0] == 0
    assert gradient == pytest.approx(0.5 / np.sqrt(4 + epsilon**2))


def test_dynamic_state_bridge_maps_smooth_residual_coordinates():
    def terms(x):
        return 0.5 * jnp.vdot(x, x), jnp.asarray([x[0] ** 2])

    initial_x = np.asarray([2.0])
    bridge = ScipyAugmentedLagrangianBridge(
        terms,
        initial_x,
        1,
        constraint_scales=[2.0],
        constraint_transform="smooth_sqrt",
        transform_epsilon=0.25,
    ).set_state([0.0], [4.0])
    value, gradient = bridge(initial_x)
    mapped = (np.sqrt(4.0 + 0.25**2) - 0.25) / 2.0

    np.testing.assert_allclose(bridge.map_constraints([4.0]), [mapped])
    assert value == pytest.approx(2.0 + 2.0 * mapped**2)
    expected_gradient = 2.0 + 4.0 * mapped / np.sqrt(4.0 + 0.25**2)
    np.testing.assert_allclose(gradient, [expected_gradient])


def test_smooth_abs_mapping_is_zero_preserving_and_asymptotically_linear():
    epsilon = 0.1
    constraints = jnp.asarray([0.0, 0.2, 10.0])
    values = smooth_abs_constraints(constraints, epsilon)
    zero_gradient = jax.grad(
        lambda value: smooth_abs_constraints(value[None], epsilon)[0]
    )(jnp.asarray(0.0))

    np.testing.assert_allclose(
        values, np.sqrt(np.asarray(constraints) ** 2 + epsilon**2) - epsilon
    )
    assert values[0] == 0.0
    assert zero_gradient == 0.0
    assert values[-1] == pytest.approx(10.0 - epsilon, rel=1e-4)


def test_dynamic_constraint_scales_reuse_compilation():
    def terms(x):
        return 0.5 * jnp.vdot(x, x), jnp.asarray([x[0]])

    initial_x = np.asarray([2.0])
    bridge = ScipyAugmentedLagrangianBridge(
        terms, initial_x, 1, constraint_scales=[2.0]
    ).set_state([0.0], [10.0])
    bridge.compile(initial_x)
    compiled = bridge._compiled
    first_value, _ = bridge(initial_x)
    bridge.set_constraint_scales([1.0])
    second_value, _ = bridge(initial_x)

    assert bridge._compiled is compiled
    assert second_value > first_value
    np.testing.assert_array_equal(bridge.constraint_scales, [1.0])


@pytest.mark.parametrize("scales", ([1.0, 2.0], [0.0], [np.nan]))
def test_dynamic_state_bridge_validates_constraint_scales(scales):
    def terms(x):
        return jnp.vdot(x, x), jnp.asarray([x[0]])

    with pytest.raises(ValueError, match="constraint_scales"):
        ScipyAugmentedLagrangianBridge(
            terms, np.asarray([1.0]), 1, constraint_scales=scales
        )


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
    np.testing.assert_array_equal(result.scaled_constraints, result.constraints)
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


def test_inner_stationarity_safeguard_prevents_outer_update():
    def terms(x):
        return 0.5 * jnp.vdot(x, x), jnp.asarray([1.0])

    initial_x = np.asarray([10.0])
    bridge = ScipyAugmentedLagrangianBridge(terms, initial_x, 1)
    result = minimize_equality_augmented_lagrangian(
        bridge,
        initial_x,
        mu_init=10.0,
        max_outer_iterations=4,
        max_inner_iterations=1,
        require_inner_stationarity=True,
    )

    assert not result.success
    assert result.terminated_by_inner_safeguard
    assert result.outer_iterations == 1
    np.testing.assert_array_equal(result.penalties, [10.0])
    np.testing.assert_array_equal(result.lagrange_multipliers, [0.0])
    assert not result.history[0]["inner_stationary"]
    assert not result.history[0]["inner_stage_accepted"]
    assert not result.history[0]["outer_update_applied"]
    assert "safeguard" in result.message


def test_inner_stationarity_factor_validation():
    def terms(x):
        return jnp.vdot(x, x), jnp.asarray([x[0] ** 2])

    bridge = ScipyAugmentedLagrangianBridge(terms, np.asarray([1.0]), 1)
    with pytest.raises(ValueError, match="inner_stationarity_factor"):
        minimize_equality_augmented_lagrangian(
            bridge, np.asarray([1.0]), inner_stationarity_factor=0.5
        )


def test_compact_history_summarizes_large_constraint_state():
    def terms(x):
        return 0.5 * jnp.vdot(x, x), jnp.asarray([x[0] ** 2, x[0] ** 2])

    initial_x = np.asarray([1.0])
    bridge = ScipyAugmentedLagrangianBridge(terms, initial_x, 2)
    result = minimize_equality_augmented_lagrangian(
        bridge,
        initial_x,
        max_outer_iterations=1,
        max_inner_iterations=2,
        history_vector_mode="summary",
    )

    record = result.history[0]
    assert "constraints" not in record
    assert record["vector_summaries"]["constraints"]["size"] == 2
    assert record["vector_summaries"]["penalties_before"]["minimum"] == 10.0


def test_global_penalty_update_preserves_uniform_penalties():
    def terms(x):
        return 0.5 * jnp.vdot(x, x), jnp.asarray([1.0, 0.0])

    initial_x = np.asarray([1.0])
    bridge = ScipyAugmentedLagrangianBridge(terms, initial_x, 2)
    result = minimize_equality_augmented_lagrangian(
        bridge,
        initial_x,
        max_outer_iterations=1,
        max_inner_iterations=1,
        constraint_tolerance=1e-16,
        penalty_update_mode="global",
    )

    np.testing.assert_array_equal(result.penalties, [100.0, 100.0])


def test_penalty_update_mode_validation():
    def terms(x):
        return jnp.vdot(x, x), jnp.asarray([x[0]])

    bridge = ScipyAugmentedLagrangianBridge(terms, np.asarray([1.0]), 1)
    with pytest.raises(ValueError, match="penalty_update_mode"):
        minimize_equality_augmented_lagrangian(
            bridge, np.asarray([1.0]), penalty_update_mode="invalid"
        )


def test_stationarity_can_use_relative_stage_gradient_reduction():
    def terms(x):
        return 0.5 * (x[0] - 3.0) ** 2, jnp.asarray([1.0])

    initial_x = np.asarray([100.0])
    bridge = ScipyAugmentedLagrangianBridge(terms, initial_x, 1)
    result = minimize_equality_augmented_lagrangian(
        bridge,
        initial_x,
        max_outer_iterations=1,
        max_inner_iterations=1,
        require_inner_stationarity=True,
        inner_stationarity_relative_tolerance=0.999,
    )

    record = result.history[0]
    assert record["gradient_norm_infinity"] > record["absolute_stationarity_threshold"]
    assert record["gradient_infinity_reduction_ratio"] < 0.999
    assert record["inner_stationary"]
    assert record["outer_update_applied"]
    assert not result.terminated_by_inner_safeguard


def test_scale_continuation_reduces_scale_and_preserves_linear_coefficient():
    def terms(x):
        return 0.5 * jnp.vdot(x, x), jnp.asarray([1.0])

    initial_x = np.asarray([1.0])
    bridge = ScipyAugmentedLagrangianBridge(
        terms, initial_x, 1, constraint_scales=[4.0]
    )
    result = minimize_equality_augmented_lagrangian(
        bridge,
        initial_x,
        max_outer_iterations=1,
        constraint_scale_reduction_factor=0.5,
        minimum_constraint_scales=[1.0],
    )

    record = result.history[0]
    np.testing.assert_array_equal(result.constraint_scales, [2.0])
    np.testing.assert_array_equal(record["constraint_scales_before"], [4.0])
    np.testing.assert_array_equal(record["constraint_scales_after"], [2.0])
    old_linear_coefficient = record["multipliers_after"][0] / 2.0
    unrescaled_coefficient = -100.0 * 0.25 / 4.0
    assert old_linear_coefficient == pytest.approx(unrescaled_coefficient)
