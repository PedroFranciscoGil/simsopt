import copy
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import pytest


def load_production_module():
    benchmark_dir = Path(__file__).parents[2] / "benchmarks" / "gpu"
    path = benchmark_dir / "production_benchmark.py"
    spec = spec_from_file_location("gpu_production_benchmark", path)
    module = module_from_spec(spec)
    with patch.object(sys, "path", [str(benchmark_dir), *sys.path]):
        spec.loader.exec_module(module)
    return module


def example_results():
    problem = {
        "name": "stress",
        "description": "stress fixture",
        "ncoils": 6,
        "order": 12,
        "nquad": 180,
        "nphi": 128,
        "ntheta": 128,
        "regularized": True,
    }
    winner = {
        "vjp_mode": "custom",
        "target_tile_size": 1024,
        "source_tile_size": 2048,
        "speedup_over_cpu_baseline": 3.2,
    }
    objective = {
        "scope": "gpu_native_local_engineering",
        "terms": [
            "quadratic_flux",
            "curve_length_penalty",
            "lp_curvature",
            "mean_squared_curvature_penalty",
            "arclength_variation",
        ],
        "deferred_terms": [
            "coil_coil_distance",
            "coil_surface_distance",
        ],
        "length_target": 18.0,
        "length_weight": 1.0,
    }
    sweep = {
        "problem": problem,
        "objective": objective,
        "dimensions": {"surface_points": 16384, "source_points": 4320},
        "protocol": {
            "parity_tolerances": {
                "value_absolute_error": 1e-9,
                "gradient_relative_l2_error": 1e-7,
            }
        },
        "winner": winner,
    }
    profile = {
        "problem": problem,
        "objective": objective,
        "vjp_mode": "custom",
        "tiles": {"target": 1024, "source": 2048},
        "environment": {"jax_backend": "gpu"},
        "compilation_seconds": 2.0,
        "steady_state": {
            "median_seconds": 0.2,
            "coefficient_of_variation": 0.01,
        },
        "device_memory": {
            "peak_bytes_in_use": 2_000_000_000,
            "bytes_limit": 10_000_000_000,
        },
        "parity": {
            "value_absolute_error": 1e-12,
            "curve_gradient_relative_l2_error": 2e-10,
            "current_gradient_relative_l2_error": 3e-10,
        },
    }
    return sweep, profile


def test_production_summary_evaluates_gates_and_scope():
    module = load_production_module()
    sweep, profile = example_results()

    summary = module.production_summary(sweep, profile)

    assert summary["all_gates_passed"]
    assert summary["objective_scope"] == "gpu_native_local_engineering"
    assert summary["gates"]["speedup_over_one_thread_cpu"]["passed"]
    assert summary["gates"]["device_memory_fraction"]["measured"] == 0.2
    assert summary["deferred_objective_terms"] == [
        "coil_coil_distance",
        "coil_surface_distance",
    ]


def test_production_summary_rejects_mismatched_profile():
    module = load_production_module()
    sweep, profile = example_results()
    mismatched = copy.deepcopy(profile)
    mismatched["tiles"]["target"] = 512

    with pytest.raises(ValueError, match="tile sizes"):
        module.production_summary(sweep, mismatched)


def test_memory_fraction_handles_unavailable_statistics():
    module = load_production_module()

    assert module.memory_fraction(None) is None
    assert module.memory_fraction({"peak_bytes_in_use": 1}) is None
