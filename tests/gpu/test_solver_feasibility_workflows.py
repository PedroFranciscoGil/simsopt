import argparse
import math
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

BENCHMARK_DIR = Path(__file__).parents[2] / "benchmarks" / "gpu"


def load_module(filename, name):
    spec = spec_from_file_location(name, BENCHMARK_DIR / filename)
    module = module_from_spec(spec)
    with patch.object(sys, "path", [str(BENCHMARK_DIR), *sys.path]):
        spec.loader.exec_module(module)
    return module


def synthetic_result(cpu_ratios, gpu_ratios, gradient=0.1, rms=0.01):
    names = ("a", "b", "c", "d")
    comparisons = {
        backend: {
            name: {
                "violation": ratio,
                "tolerance": 1.0,
                "passed": ratio <= 1.0,
            }
            for name, ratio in zip(names, ratios, strict=True)
        }
        for backend, ratios in (("cpu", cpu_ratios), ("gpu", gpu_ratios))
    }
    metrics = {
        "objective": 2.0,
        "normalized_normal_field": {"root_mean_square": rms},
    }
    optimization = {"final_gradient_norm": gradient, "evaluations": 12}
    return {
        "gates": {"absolute_engineering_feasibility": {"measured": comparisons}},
        "cpu": {"optimization": optimization, "final_metrics": metrics},
        "gpu": {
            "optimization": optimization,
            "final_metrics_from_cpu_oracle": metrics,
        },
    }


def test_penalty_list_parsers_and_restart_ranking():
    module = load_module("run_penalty_continuation.py", "gpu_penalty_continuation")

    assert module.positive_float_list("1,10,100") == [1.0, 10.0, 100.0]
    assert module.positive_int_list("20,40") == [20, 40]
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        module.positive_float_list("1,0")
    with pytest.raises(argparse.ArgumentTypeError, match="comma-separated"):
        module.positive_int_list("1,bad")

    result = synthetic_result([0.1] * 4, [0.2] * 4)
    assert module.choose_seed_backend(result, "best") == "cpu"
    assert module.choose_seed_backend(result, "gpu") == "gpu"


def test_solver_candidate_rank_prioritizes_absolute_feasibility():
    module = load_module("sweep_solver_feasibility.py", "gpu_solver_sweep")
    feasible = synthetic_result([0.9] * 4, [0.8] * 4, gradient=10.0, rms=1.0)
    infeasible = synthetic_result([0.1] * 4, [1.01, 0.1, 0.1, 0.1])

    assert module.candidate_rank(feasible) < module.candidate_rank(infeasible)
    assert module.positive_int_list("10,30,100") == [10, 30, 100]
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        module.positive_int_list("10,-1")


def test_continuation_command_carries_solver_and_restart_settings(tmp_path):
    module = load_module("run_penalty_continuation.py", "gpu_continuation_command")
    args = SimpleNamespace(
        problem="stress",
        maxcor=30,
        maxls=50,
        current_scale=100000.0,
        target_tile_size=1024,
        source_tile_size=4320,
        distance_feasibility_tolerance=1e-4,
        curvature_feasibility_tolerance=1e-3,
        mean_squared_curvature_feasibility_tolerance=1e-3,
        ftol=None,
        gtol=None,
        output_dir=tmp_path,
        allow_non_gpu=False,
    )
    restart = tmp_path / "restart.json"
    output = tmp_path / "result.json"

    command = module.comparison_command(
        args, 10.0, 100, output, restart, final_stage=True
    )

    assert command[command.index("--maxcor") + 1] == "30"
    assert command[command.index("--maxls") + 1] == "50"
    assert command[command.index("--constraint-weight-multiplier") + 1] == "10.0"
    assert command[command.index("--initial-variables") + 1] == str(restart)
    assert "--visualization-dir" in command
    assert "--no-visualization" not in command
    assert math.isfinite(
        module.backend_rank(synthetic_result([0.1] * 4, [0.2] * 4), "cpu")[0]
    )
