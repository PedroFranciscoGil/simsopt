import argparse
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import numpy as np
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


def test_gradient_balancing_is_attenuation_only_and_caps_contributions():
    module = load_benchmark_module(
        "benchmark_augmented_lagrangian.py", "gpu_al_gradient_balance"
    )
    scales, diagnostics = module.gradient_balanced_constraint_scales(
        np.asarray([4.0, 0.0, 1.0, 9.0]),
        np.asarray([2.0, 0.0]),
        np.asarray([[3.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 4.0]]),
        mu_init=4.0,
        constraint_transform="identity",
        transform_epsilon=1e-4,
        maximum_ratio=1.0,
    )

    np.testing.assert_allclose(scales, np.sqrt([24.0, 1.0, 4.0, 72.0]))
    assert np.all(scales >= 1.0)
    balanced = diagnostics["balanced_initial_penalty_gradient_norm_estimates"]
    assert max(balanced.values()) == pytest.approx(diagnostics["base_gradient_norm"])


def test_smooth_sqrt_gradient_balancing_rejects_negative_penalties():
    module = load_benchmark_module(
        "benchmark_augmented_lagrangian.py", "gpu_al_gradient_balance_validation"
    )
    with pytest.raises(ValueError, match="nonnegative"):
        module.gradient_balanced_constraint_scales(
            np.asarray([-1.0, 0.0, 0.0, 0.0]),
            np.asarray([1.0]),
            np.ones((4, 1)),
            mu_init=4.0,
            constraint_transform="smooth_sqrt",
            transform_epsilon=1e-4,
            maximum_ratio=1.0,
        )


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
        "minimum_coil_coil_distance": failed_ratio * 1e-4,
        "minimum_coil_surface_distance": 0.0,
        "maximum_curvature": 0.0,
        "maximum_mean_squared_curvature": 0.0,
    }
    tolerances = {
        "minimum_coil_coil_distance": 1e-4,
        "minimum_coil_surface_distance": 1e-4,
        "maximum_curvature": 1e-3,
        "maximum_mean_squared_curvature": 1e-3,
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


def test_safeguard_qualification_requires_every_scientific_gate():
    module = load_benchmark_module(
        "sweep_augmented_lagrangian_safeguards.py", "gpu_al_safeguard_gates"
    )
    result = {
        "acceptance_gates": {
            name: {"passed": True} for name in module.QUALIFICATION_GATES
        }
    }
    assert module.qualifies(result)
    result["acceptance_gates"]["stationary_convergence"]["passed"] = False
    assert not module.qualifies(result)


def test_safeguard_command_enables_guard_and_keeps_visualizations(tmp_path):
    module = load_benchmark_module(
        "sweep_augmented_lagrangian_safeguards.py", "gpu_al_safeguard_command"
    )
    args = argparse.Namespace(
        max_outer_iterations=10,
        max_inner_iterations=200,
        mu_init=4.0,
        tau=2.0,
        mu_max=1e6,
        inner_stationarity_factor=1.0,
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
    candidate = module.CANDIDATES[1]
    command = module.benchmark_command(
        args, "engineering", candidate, tmp_path / "screen.json"
    )

    assert "--require-inner-stationarity" in command
    assert "--automatic-constraint-scaling" in command
    assert "--no-visualization" not in command
    assert command[command.index("--constraint-transform") + 1] == "smooth_sqrt"


def test_safeguard_notebook_resolves_only_vtk_metadata_against_artifact_root():
    notebook_path = (
        Path(__file__).parents[2]
        / "benchmarks"
        / "gpu"
        / "colab_augmented_lagrangian_safeguards.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    validation_source = "".join(
        next(
            cell["source"]
            for cell in notebook["cells"]
            if any("def validate_result" in line for line in cell.get("source", []))
        )
    )

    assert '("surface_vts", "coils_vtu")' in validation_source
    assert "artifact_root / Path(final_design[artifact_key]).name" in validation_source
    assert "for artifact in final_design.values()" not in validation_source
