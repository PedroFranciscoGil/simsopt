import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt.gpu import (
    DeviceAugmentedLagrangian,
    DeviceAugmentedLagrangianConfig,
    ScipyAugmentedLagrangianBridge,
    minimize_equality_augmented_lagrangian,
)

jax.config.update("jax_enable_x64", True)


def equality_terms(x):
    return 0.5 * jnp.square(x[0] - 2.0), jnp.asarray([x[0] - 1.0])


def test_device_augmented_lagrangian_matches_host_reference():
    initial = np.asarray([3.0])
    config = DeviceAugmentedLagrangianConfig(
        mu_init=10.0,
        tau=2.0,
        max_outer_iterations=20,
        max_inner_iterations=50,
        history_size=5,
        gradient_tolerance=1e-8,
        constraint_tolerance=1e-6,
        require_inner_stationarity=False,
        penalty_update_mode="global",
        constraint_transform="identity",
        constraint_scale_reduction_factor=1.0,
    )
    device_solver = DeviceAugmentedLagrangian(
        equality_terms,
        initial,
        1,
        config=config,
        platform="cpu",
    )
    device_result = device_solver.run(initial)

    bridge = ScipyAugmentedLagrangianBridge(
        equality_terms,
        initial,
        1,
        constraint_transform="identity",
        platform="cpu",
    )
    host_result = minimize_equality_augmented_lagrangian(
        bridge,
        initial,
        mu_init=10.0,
        tau=2.0,
        max_outer_iterations=20,
        max_inner_iterations=50,
        gradient_tolerance=1e-8,
        constraint_tolerance=1e-6,
        maxcor=5,
        require_inner_stationarity=False,
        penalty_update_mode="global",
    )

    assert device_result.device_resident
    assert device_result.host_callbacks == 0
    assert device_result.outer_iterations <= 20
    assert device_result.total_inner_iterations > 0
    assert len(device_result.history) == device_result.outer_iterations
    assert abs(device_result.constraints[0]) <= 2e-6
    assert device_result.x[0] == pytest.approx(1.0, abs=2e-6)
    assert device_result.x[0] == pytest.approx(host_result.x[0], abs=2e-6)


def test_device_augmented_lagrangian_compile_is_reusable():
    initial = np.asarray([3.0])
    config = DeviceAugmentedLagrangianConfig(
        max_outer_iterations=2,
        max_inner_iterations=3,
        history_size=2,
        require_inner_stationarity=False,
        constraint_transform="identity",
        constraint_scale_reduction_factor=1.0,
    )
    solver = DeviceAugmentedLagrangian(
        equality_terms, initial, 1, config=config, platform="cpu"
    ).compile(initial)

    first = solver.run(initial)
    second = solver.run(initial)

    assert solver.is_compiled
    np.testing.assert_allclose(first.x, second.x, rtol=0.0, atol=0.0)
    assert first.total_evaluations == second.total_evaluations


def test_device_augmented_lagrangian_validates_static_contract():
    initial = np.asarray([3.0])
    with pytest.raises(ValueError, match="mu_init"):
        DeviceAugmentedLagrangianConfig(mu_init=1.0)
    with pytest.raises(ValueError, match="constraint_transform"):
        DeviceAugmentedLagrangianConfig(constraint_transform="bad")
    with pytest.raises(ValueError, match="must not exceed"):
        DeviceAugmentedLagrangian(
            equality_terms,
            initial,
            1,
            constraint_scales=np.asarray([1.0]),
            minimum_constraint_scales=np.asarray([2.0]),
        )
