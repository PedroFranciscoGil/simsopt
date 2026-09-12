import argparse
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import pytest


def load_benchmark_module(filename, name):
    benchmark_dir = Path(__file__).parents[2] / "benchmarks" / "gpu"
    spec = spec_from_file_location(name, benchmark_dir / filename)
    module = module_from_spec(spec)
    with patch.object(sys, "path", [str(benchmark_dir), *sys.path]):
        spec.loader.exec_module(module)
    return module


def test_constraint_scale_parser():
    module = load_benchmark_module(
        "benchmark_augmented_lagrangian.py", "gpu_al_benchmark"
    )
    assert module.positive_float_vector("1,2,3,4").tolist() == [1.0, 2.0, 3.0, 4.0]
    for text in ("1,2,3", "1,2,3,0", "1,2,3,nan"):
        with pytest.raises(argparse.ArgumentTypeError):
            module.positive_float_vector(text)


def test_conditioning_command_records_configuration_and_visualization(tmp_path):
    module = load_benchmark_module(
        "sweep_augmented_lagrangian_conditioning.py", "gpu_al_conditioning"
    )
    args = argparse.Namespace(
        gradient_tolerance=1e-8,
        constraint_tolerance=1e-8,
        maxcor=100,
        maxls=50,
        current_scale=1e5,
        target_tile_size=1024,
        source_tile_size=4320,
        distance_feasibility_tolerance=1e-4,
        curvature_feasibility_tolerance=1e-3,
        mean_squared_curvature_feasibility_tolerance=1e-3,
        allow_non_gpu=False,
    )
    candidate = module.CANDIDATES[-1]
    screen = module.benchmark_command(
        args, "engineering", candidate, tmp_path / "screen.json", final=False
    )
    final = module.benchmark_command(
        args, "stress", candidate, tmp_path / "final.json", final=True
    )

    scale_index = screen.index("--constraint-scales") + 1
    assert (
        screen[scale_index]
        == "0.00050000000000000001,0.02,0.00059999999999999995,0.00069999999999999999"
    )
    assert "--no-visualization" in screen
    assert "--no-visualization" not in final


def synthetic_result(*, failed_ratio, success=True):
    violations = {
        "coil_coil_distance_deficit": failed_ratio * 1e-4,
        "coil_surface_distance_deficit": 0.0,
        "maximum_curvature_excess": 0.0,
        "maximum_mean_squared_curvature_excess": 0.0,
    }
    tolerances = {
        "coil_coil_distance_deficit": 1e-4,
        "coil_surface_distance_deficit": 1e-4,
        "maximum_curvature_excess": 1e-3,
        "maximum_mean_squared_curvature_excess": 1e-3,
    }
    backend = {
        "optimization": {
            "success": success,
            "final_gradient_norm": 1e-7,
            "total_evaluations": 10,
        },
        "final_metrics": {
            "coil_constraints": {"violations": violations},
            "normalized_normal_field": {"root_mean_square": 0.01},
        },
    }
    return {
        "acceptance_gates": {
            "gpu_backend": {"passed": True},
            "initial_float64_parity": {"passed": True},
        },
        "feasibility_tolerances": tolerances,
        "cpu": backend,
        "gpu": backend,
    }


def test_candidate_rank_prioritizes_physical_feasibility():
    module = load_benchmark_module(
        "sweep_augmented_lagrangian_conditioning.py", "gpu_al_conditioning_rank"
    )
    feasible_but_nonstationary = synthetic_result(failed_ratio=0.5, success=False)
    stationary_but_infeasible = synthetic_result(failed_ratio=2.0, success=True)
    assert module.candidate_rank(feasible_but_nonstationary) < module.candidate_rank(
        stationary_but_infeasible
    )
