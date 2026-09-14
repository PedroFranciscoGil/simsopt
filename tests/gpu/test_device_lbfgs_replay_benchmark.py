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


def test_committed_analysis_contains_replayable_qualified_warm_start():
    root = Path(__file__).parents[2]
    path = (
        root
        / "docs"
        / "gpu_native"
        / "figures"
        / "device_lbfgs_comparison_analysis_summary.json"
    )
    module = load_module(
        "benchmark_device_lbfgs_replay.py", "gpu_device_lbfgs_replay_benchmark"
    )

    baseline = module.load_baseline(path)

    assert baseline["problem"]["name"] == "engineering"
    assert baseline["qualified_warm_start"]["source_backend"] == "cpu"
    assert len(baseline["qualified_warm_start"]["physical_variables"]) == 207


def test_baseline_loader_rejects_unqualified_or_incomplete_data(tmp_path):
    module = load_module(
        "benchmark_device_lbfgs_replay.py", "gpu_device_lbfgs_replay_validation"
    )
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "workflow": "local_residual_augmented_lagrangian_analysis",
                "scientifically_validated": False,
                "qualified_warm_start": {"physical_variables": []},
            }
        )
    )

    with pytest.raises(ValueError, match="scientifically validated"):
        module.load_baseline(path)

    path.write_text(
        json.dumps(
            {
                "workflow": "local_residual_augmented_lagrangian_analysis",
                "scientifically_validated": True,
                "qualified_warm_start": {},
            }
        )
    )
    with pytest.raises(TypeError, match="qualified physical warm start"):
        module.load_baseline(path)


def test_candidate_names_are_stable_and_sortable():
    module = load_module(
        "benchmark_device_lbfgs_replay.py", "gpu_device_lbfgs_replay_names"
    )

    assert module.candidate_name(10, 15) == "history-010-patience-015"
    assert module.candidate_name(100, 25) == "history-100-patience-025"


def test_replay_analyzer_reproduces_grid_and_winner_contract():
    module = load_module(
        "analyze_device_lbfgs_replay.py", "gpu_device_lbfgs_replay_analyzer"
    )
    checkpoint = {
        "index": 0,
        "quadratic_flux": 1e-6,
        "target_envelope_residual_norm_infinity": 0.0,
        "target_feasible": True,
    }
    candidate = {
        "name": "history-010-patience-015",
        "history_size": 10,
        "checkpoint_patience": 15,
        "compilation_seconds": 1.0,
        "optimization": {
            "device_resident": True,
            "host_callbacks": 0,
            "seconds": 0.5,
            "evaluations": 1,
            "checkpoint_selection": {
                "selected_index": 0,
                "selected": checkpoint,
                "checkpoints": [checkpoint],
            },
        },
        "final_metrics": {"quadratic_flux": 1e-6},
        "engineering_validation": {"passed": True},
        "quadratic_flux_validation": {"passed": True},
        "scientifically_validated": True,
    }
    result = {
        "schema_version": 1,
        "workflow": "device_lbfgs_replay_sweep",
        "execution_platform": "gpu",
        "sweep": {
            "history_sizes": [10],
            "checkpoint_patiences": [15],
            "candidate_count": 1,
        },
        "candidates": [candidate],
        "qualified_indices": [0],
        "winner_index": 0,
        "winner": candidate,
        "scientifically_validated": True,
    }

    assert module.validate_result(result) is result
    result["execution_platform"] = "cpu"
    with pytest.raises(ValueError, match="did not execute on a GPU"):
        module.validate_result(result)


def test_colab_runs_eight_candidate_gpu_replay_and_downloads_archive():
    notebook = json.loads(
        (
            Path(__file__).parents[2]
            / "benchmarks"
            / "gpu"
            / "colab_device_lbfgs_replay.ipynb"
        ).read_text()
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert "benchmark_device_lbfgs_replay.py" in source
    assert 'jax.default_backend() == "gpu"' in source
    assert '"--history-sizes", "10", "20", "50", "100"' in source
    assert '"--checkpoint-patiences", "15", "25"' in source
    assert 'result["sweep"]["candidate_count"] == 8' in source
    assert 'result["execution_platform"] == "gpu"' in source
    assert "files.download(archive)" in source
