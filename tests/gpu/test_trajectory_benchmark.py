import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


def load_trajectory_module():
    benchmark_dir = Path(__file__).parents[2] / "benchmarks" / "gpu"
    path = benchmark_dir / "compare_scipy_trajectories.py"
    spec = spec_from_file_location("gpu_trajectory_benchmark", path)
    module = module_from_spec(spec)
    with patch.object(sys, "path", [str(benchmark_dir), *sys.path]):
        spec.loader.exec_module(module)
    return module


def test_trajectory_recorder_uses_evaluated_callback_state():
    module = load_trajectory_module()

    def objective(x):
        return np.sum(x * x), 2.0 * x

    recorder = module.TrajectoryRecorder(objective)
    state = np.asarray([1.0, -2.0])
    value, gradient = recorder(state)
    recorder.callback(state)

    assert value == 5.0
    np.testing.assert_array_equal(gradient, [2.0, -4.0])
    assert recorder.evaluations[0]["gradient_norm"] == pytest.approx(np.sqrt(20.0))
    assert recorder.iterations == [
        {
            "iteration": 1,
            "objective": 5.0,
            "gradient_norm": pytest.approx(np.sqrt(20.0)),
            "step_norm": 0.0,
        }
    ]
    with pytest.raises(RuntimeError, match="not an evaluated point"):
        recorder.callback(np.zeros(2))


def test_trajectory_comparison_separates_currents_and_curves():
    module = load_trajectory_module()
    cpu_recorder = module.TrajectoryRecorder(lambda x: (0.0, x))
    gpu_recorder = module.TrajectoryRecorder(lambda x: (0.0, x))
    cpu_state = np.asarray([10.0, 1.0, 2.0])
    gpu_state = np.asarray([10.0, 1.0 + 1e-8, 2.0])
    cpu_recorder(cpu_state)
    cpu_recorder.callback(cpu_state)
    gpu_recorder(gpu_state)
    gpu_recorder.callback(gpu_state)
    cpu_result = SimpleNamespace(status=1, nit=1, nfev=1, fun=0.0, x=cpu_state)
    gpu_result = SimpleNamespace(status=1, nit=1, nfev=1, fun=0.0, x=gpu_state)

    comparison = module.compare_trajectories(
        cpu_result, gpu_result, cpu_recorder, gpu_recorder, nfree=1
    )

    assert comparison["same_status"]
    assert comparison["same_iteration_count"]
    assert comparison["same_evaluation_count"]
    assert comparison["final_current_relative_l2_error"] == 0.0
    assert comparison["final_curve_relative_l2_error"] > 0.0
