import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize
from simsopt.gpu import (
    DeviceLBFGS,
    DeviceLBFGSConfig,
    TargetAwareCheckpointRecorder,
    TargetAwareConfig,
)

jax.config.update("jax_enable_x64", True)


def test_target_aware_recorder_stops_after_target_stagnation():
    policy = TargetAwareConfig(
        target_flux=0.5,
        checkpoint_patience=2,
        infeasible_patience=5,
        minimum_relative_improvement=1e-3,
    )
    recorder = TargetAwareCheckpointRecorder(
        lambda x: (float(np.vdot(x, x)), 2.0 * x),
        lambda x: (float(x[0] ** 2), np.zeros(1)),
        policy,
    ).initialize(np.asarray([2.0]))

    recorder.callback(np.asarray([0.7]))
    recorder.callback(np.asarray([0.7]))
    with pytest.raises(StopIteration):
        recorder.callback(np.asarray([0.7]))

    state, selection = recorder.selection()
    np.testing.assert_array_equal(state, [0.7])
    assert selection["stop_reason"] == ("target_reached_then_feasible_stagnation")
    assert selection["selected"]["flux_target_satisfied"]


def test_scipy_catches_target_aware_stop_iteration():
    initial = np.asarray([2.0])
    policy = TargetAwareConfig(
        target_flux=10.0,
        checkpoint_patience=1,
        infeasible_patience=5,
        minimum_relative_improvement=0.99,
    )
    recorder = TargetAwareCheckpointRecorder(
        lambda x: (float(0.5 * np.vdot(x, x)), x.copy()),
        lambda x: (float(0.5 * x[0] ** 2), np.zeros(1)),
        policy,
    ).initialize(initial)

    result = minimize(
        recorder,
        initial,
        jac=True,
        method="L-BFGS-B",
        callback=recorder.callback,
        options={"maxiter": 20, "gtol": 0.0, "ftol": 0.0},
    )

    assert result.status == 99
    assert recorder.stop_reason == "target_reached_then_feasible_stagnation"


def test_target_aware_recorder_stops_after_sustained_infeasibility():
    policy = TargetAwareConfig(
        target_flux=1.0,
        checkpoint_patience=10,
        infeasible_patience=2,
    )
    recorder = TargetAwareCheckpointRecorder(
        lambda x: (float(np.vdot(x, x)), 2.0 * x),
        lambda x: (float(x[0] ** 2), np.asarray([max(0.0, x[1])])),
        policy,
    ).initialize(np.asarray([0.5, 0.0]))

    recorder.callback(np.asarray([0.4, 1.0]))
    with pytest.raises(StopIteration):
        recorder.callback(np.asarray([0.3, 1.0]))

    _, selection = recorder.selection()
    assert selection["stop_reason"] == ("target_reached_then_sustained_infeasibility")
    assert selection["selected_index"] == 0


def test_device_lbfgs_solves_quadratic_in_one_compiled_call():
    initial = np.asarray([3.0, -4.0], dtype=float)
    target = TargetAwareConfig(
        target_flux=1e-10,
        checkpoint_patience=50,
        infeasible_patience=50,
    )
    config = DeviceLBFGSConfig(
        history_size=5,
        max_iterations=50,
        gradient_tolerance=1e-10,
        target=target,
    )

    def objective(x):
        return 0.5 * jnp.vdot(x, x)

    def quality(x):
        return objective(x), jnp.zeros((1,), dtype=x.dtype)

    solver = DeviceLBFGS(
        objective, quality, initial, config=config, platform="cpu"
    ).compile(initial)
    result = solver.run(initial)

    assert solver.device_platform == "cpu"
    assert result.success
    assert result.status == 1
    assert result.iterations <= 2
    assert result.evaluations >= 2
    assert result.found_feasible_checkpoint
    assert result.target_found
    np.testing.assert_allclose(result.x, np.zeros(2), atol=1e-12)
    assert result.quadratic_flux <= target.target_flux
    assert result.objective_history.shape == (result.iterations + 1,)


def test_device_lbfgs_infeasible_fallback_breaks_residual_ties_by_flux():
    initial = np.asarray([1.0], dtype=float)
    config = DeviceLBFGSConfig(max_iterations=5, gradient_tolerance=1e-12)

    def objective(x):
        return 0.5 * jnp.square(x[0] + 2.0)

    def quality(x):
        return jnp.square(x[0]), jnp.ones((1,), dtype=x.dtype)

    result = DeviceLBFGS(
        objective, quality, initial, config=config, platform="cpu"
    ).run(initial)

    assert not result.found_feasible_checkpoint
    assert result.selected_iteration == 1
    np.testing.assert_allclose(result.x, np.zeros(1), atol=1e-15)
    assert result.quadratic_flux == pytest.approx(0.0, abs=1e-30)


def test_device_lbfgs_validates_configuration_and_shape():
    with pytest.raises(ValueError, match="history_size"):
        DeviceLBFGSConfig(history_size=0)
    with pytest.raises(ValueError, match="backtracking_factor"):
        DeviceLBFGSConfig(backtracking_factor=1.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        DeviceLBFGS(
            lambda x: jnp.sum(x * x),
            lambda x: (jnp.sum(x * x), jnp.zeros(1)),
            np.ones((2, 2)),
        )

    solver = DeviceLBFGS(
        lambda x: jnp.sum(x * x),
        lambda x: (jnp.sum(x * x), jnp.zeros(1)),
        np.ones(2),
        platform="cpu",
    )
    with pytest.raises(ValueError, match="shape"):
        solver.run(np.ones(3))
