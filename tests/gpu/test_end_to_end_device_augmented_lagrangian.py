import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import pytest


def load_module(filename, name):
    benchmark_dir = Path(__file__).parents[2] / "benchmarks" / "gpu"
    specification = spec_from_file_location(name, benchmark_dir / filename)
    module = module_from_spec(specification)
    with patch.object(sys, "path", [str(benchmark_dir), *sys.path]):
        specification.loader.exec_module(module)
    return module


def final_metrics():
    return {
        "objective": 1e-4,
        "quadratic_flux": 1e-6,
        "normalized_normal_field": {
            "mean_absolute": 1e-3,
            "root_mean_square": 2e-3,
            "maximum_absolute": 5e-3,
        },
        "coil_constraints": {
            "measurements": {},
            "limits": {
                "coil_coil_distance_threshold": 0.1,
                "coil_surface_distance_threshold": 0.3,
                "curvature_threshold": 5.0,
                "mean_squared_curvature_threshold": 5.0,
            },
        },
    }


def optimization():
    return {
        "seconds": 1.0,
        "base_objective": 1e-4,
        "outer_iterations": 1,
        "total_evaluations": 3,
        "history": [
            {
                "outer_iteration": 1,
                "base_objective": 1e-4,
                "constraint_norm_infinity": 1e-6,
                "optimizer_variables": [0.0],
            }
        ],
    }


def result_contract():
    validation = {"passed": True}
    return {
        "schema_version": 5,
        "workflow": "end_to_end_device_augmented_lagrangian",
        "method": {
            "refinement_performed": False,
            "gpu_outer_loop_device_resident": True,
            "matched_design_envelope_bounds": True,
            "inner_stationarity_is_diagnostic": True,
            "target_aware_outer_checkpoint_selection": True,
            "minimum_continuation_depth_enforced": True,
            "engineering_residual_zero_set_matches_validation_envelope": True,
        },
        "solver": {"minimum_outer_iterations": 8, "max_outer_iterations": 8},
        "design_envelope": {"curve_coefficient_bound_radius_m": 0.25},
        "residual_contract": {
            "term_settings": {
                "coil_coil_distance_threshold": 0.0901,
                "coil_surface_distance_threshold": 0.2701,
                "curvature_threshold": 5.499,
                "mean_squared_curvature_threshold": 5.499,
            },
            "transition_widths": {
                "distance": 1e-4,
                "curvature": 1e-3,
                "mean_squared_curvature": 1e-3,
            },
            "optimizer_zero_boundaries": {
                "minimum_coil_coil_distance": 0.09,
                "minimum_coil_surface_distance": 0.27,
                "maximum_curvature": 5.5,
                "maximum_mean_squared_curvature": 5.5,
            },
        },
        "validation_policy": {"target_relative_tolerance": 0.1},
        "cpu": {
            "execution_platform": "cpu",
            "compilation_seconds": 0.5,
            "optimization": optimization(),
            "final_metrics": final_metrics(),
            "scientific_validation": validation,
            "checkpoint_selection": {
                "candidate_count": 1,
                "candidates": [{"source": "outer_1"}],
                "selected": {"source": "outer_1"},
            },
        },
        "gpu_native": {
            "execution_platform": "gpu",
            "device_resident": True,
            "host_callbacks": 0,
            "compilation_seconds": 2.0,
            "execution_samples_seconds": [0.1, 0.11],
            "warm_median_seconds": 0.105,
            "optimization": optimization(),
            "final_metrics": final_metrics(),
            "scientific_validation": validation,
            "checkpoint_selection": {
                "candidate_count": 1,
                "candidates": [{"source": "outer_1"}],
                "selected": {"source": "outer_1"},
            },
        },
        "scientifically_validated": True,
    }


def test_benchmark_defaults_match_production_comparison():
    module = load_module(
        "benchmark_end_to_end_device_augmented_lagrangian.py",
        "end_to_end_device_al_benchmark",
    )
    with patch.object(sys, "argv", ["benchmark"]):
        args = module.parse_args()

    assert args.max_outer_iterations == 8
    assert args.minimum_outer_iterations == 8
    assert args.max_inner_iterations == 300
    assert args.history_size == 20
    assert args.cpu_maxcor == 100
    assert args.gpu_warm_repeats == 3
    assert args.minimum_current_ratio == 0.5
    assert args.maximum_current_ratio == 1.5
    assert args.curve_coefficient_bound_radius == 0.25
    assert args.inner_acceptance_mode == "budgeted"


def test_benchmark_requires_the_complete_outer_continuation():
    module = load_module(
        "benchmark_end_to_end_device_augmented_lagrangian.py",
        "end_to_end_device_al_benchmark_invalid_depth",
    )
    with patch.object(
        sys,
        "argv",
        [
            "benchmark",
            "--max-outer-iterations",
            "8",
            "--minimum-outer-iterations",
            "4",
        ],
    ), pytest.raises(SystemExit, match="2"):
        module.parse_args()


def test_benchmark_aligns_al_residuals_with_validation_envelope():
    module = load_module(
        "benchmark_end_to_end_device_augmented_lagrangian.py",
        "end_to_end_device_al_benchmark_target_envelope",
    )
    settings = {
        "length_weight": 1e-6,
        "coil_coil_distance_threshold": 0.1,
        "coil_surface_distance_threshold": 0.3,
        "curvature_threshold": 5.0,
        "mean_squared_curvature_threshold": 5.0,
    }

    kwargs = module.target_al_residual_kwargs(settings, 0.1)

    assert kwargs["length_weight"] == 1e-6
    assert kwargs["coil_coil_distance_threshold"] - 1e-4 == pytest.approx(0.09)
    assert kwargs["coil_surface_distance_threshold"] - 1e-4 == pytest.approx(0.27)
    assert kwargs["curvature_threshold"] + 1e-3 == pytest.approx(5.5)
    assert kwargs["mean_squared_curvature_threshold"] + 1e-3 == pytest.approx(
        5.5
    )


def test_analyzer_validates_device_placement_and_no_refinement():
    module = load_module(
        "analyze_end_to_end_device_augmented_lagrangian.py",
        "end_to_end_device_al_analyzer",
    )
    result = result_contract()
    assert module.validate_result(result) is result

    result["method"]["refinement_performed"] = True
    with pytest.raises(ValueError, match="refinement"):
        module.validate_result(result)


def test_analyzer_rejects_a_misaligned_optimizer_envelope():
    module = load_module(
        "analyze_end_to_end_device_augmented_lagrangian.py",
        "end_to_end_device_al_analyzer_bad_envelope",
    )
    result = result_contract()
    result["residual_contract"]["optimizer_zero_boundaries"][
        "minimum_coil_surface_distance"
    ] = 0.28

    with pytest.raises(ValueError, match="not target-aligned"):
        module.validate_result(result)


def test_colab_runs_matched_no_refinement_comparison_and_downloads_archive():
    notebook = json.loads(
        (
            Path(__file__).parents[2]
            / "benchmarks"
            / "gpu"
            / "colab_end_to_end_device_augmented_lagrangian.ipynb"
        ).read_text()
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert "benchmark_end_to_end_device_augmented_lagrangian.py" in source
    assert 'jax.default_backend() == "gpu"' in source
    assert '"--max-outer-iterations", "8"' in source
    assert '"--minimum-outer-iterations", "8"' in source
    assert '"--max-inner-iterations", "300"' in source
    assert '"--history-size", "20"' in source
    assert '"--curve-coefficient-bound-radius", "0.25"' in source
    assert '"--inner-acceptance-mode", "budgeted"' in source
    assert 'result["method"]["refinement_performed"] is False' in source
    assert 'result["method"]["matched_design_envelope_bounds"]' in source
    assert 'result["method"]["inner_stationarity_is_diagnostic"]' in source
    assert 'result["method"]["minimum_continuation_depth_enforced"]' in source
    assert (
        'result["method"]'
        '["engineering_residual_zero_set_matches_validation_envelope"]' in source
    )
    assert 'result["schema_version"] == 5' in source
    assert 'result["residual_contract"]["optimizer_zero_boundaries"]' in source
    assert 'result["gpu_native"]["host_callbacks"] == 0' in source
    assert "path.is_absolute()" in source
    assert "files.download(archive)" in source
