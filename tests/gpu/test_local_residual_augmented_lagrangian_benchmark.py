import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


def load_benchmark_module():
    benchmark_dir = Path(__file__).parents[2] / "benchmarks" / "gpu"
    specification = spec_from_file_location(
        "gpu_local_residual_augmented_lagrangian_benchmark",
        benchmark_dir / "benchmark_local_residual_augmented_lagrangian.py",
    )
    module = module_from_spec(specification)
    with patch.object(sys, "path", [str(benchmark_dir), *sys.path]):
        specification.loader.exec_module(module)
    return module


def test_family_l2_scaling_is_attenuation_only_and_controls_each_family():
    module = load_benchmark_module()
    residuals = np.asarray([3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    layout = {
        "active": {"start": 0, "stop": 2},
        "inactive": {"start": 2, "stop": 6},
    }

    scales, diagnostics = module.residual_scales(residuals, layout, "family_l2")

    np.testing.assert_array_equal(scales, [5.0, 5.0, 2.0, 2.0, 2.0, 2.0])
    assert diagnostics["active"]["mapped_initial_l2_norm"] == 1.0
    assert diagnostics["inactive"]["applied_scale"] == 2.0
    assert np.all(scales >= 1.0)


def test_residual_scaling_rejects_incomplete_layout():
    module = load_benchmark_module()

    with pytest.raises(ValueError, match="cover"):
        module.residual_scales(
            np.ones(3), {"first": {"start": 0, "stop": 2}}, "family_l2"
        )


def test_family_summaries_preserve_named_slices_without_full_vectors():
    module = load_benchmark_module()
    summaries = module.family_summaries(
        np.asarray([0.0, 2.0, -3.0]),
        {
            "residuals": {"start": 0, "stop": 2},
            "multipliers": {"start": 2, "stop": 3},
        },
    )

    assert summaries["residuals"]["active_count"] == 1
    assert summaries["residuals"]["maximum"] == 2.0
    assert summaries["multipliers"]["minimum"] == -3.0


def test_colab_runs_gpu_workflow_and_resolves_only_artifact_fields():
    notebook_path = (
        Path(__file__).parents[2]
        / "benchmarks"
        / "gpu"
        / "colab_local_residual_augmented_lagrangian.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert "benchmark_local_residual_augmented_lagrangian.py" in source
    assert 'jax.default_backend() == "gpu"' in source
    assert "path = artifact_root / artifact" in source
    assert 'metrics = result[backend]["final_metrics"]' in source
    assert "files.download(archive)" in source
