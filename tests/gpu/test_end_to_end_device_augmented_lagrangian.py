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
        "coil_constraints": {"measurements": {}},
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
            }
        ],
    }


def result_contract():
    validation = {"passed": True}
    return {
        "schema_version": 1,
        "workflow": "end_to_end_device_augmented_lagrangian",
        "method": {
            "refinement_performed": False,
            "gpu_outer_loop_device_resident": True,
        },
        "cpu": {
            "execution_platform": "cpu",
            "compilation_seconds": 0.5,
            "optimization": optimization(),
            "final_metrics": final_metrics(),
            "scientific_validation": validation,
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
    assert args.max_inner_iterations == 300
    assert args.history_size == 20
    assert args.cpu_maxcor == 100
    assert args.gpu_warm_repeats == 3


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
    assert '"--max-inner-iterations", "300"' in source
    assert '"--history-size", "20"' in source
    assert 'result["method"]["refinement_performed"] is False' in source
    assert 'result["gpu_native"]["host_callbacks"] == 0' in source
    assert "path.is_absolute()" in source
    assert "files.download(archive)" in source
