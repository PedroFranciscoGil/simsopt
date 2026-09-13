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


def test_minimum_residual_scales_use_family_count_floor():
    module = load_benchmark_module()
    layout = {
        "first": {"start": 0, "stop": 4},
        "second": {"start": 4, "stop": 13},
    }

    scales = module.minimum_residual_scales(layout, 13, "sqrt_count")

    np.testing.assert_array_equal(scales[:4], 2.0)
    np.testing.assert_array_equal(scales[4:], 3.0)


def _final_metrics(
    *,
    objective=2.0,
    normal=(0.01, 0.02, 0.03),
    coil_coil=0.1,
    coil_surface=0.3,
    curvature=5.0,
    mean_squared_curvature=5.0,
    length=12.0,
):
    return {
        "objective": objective,
        "normalized_normal_field": {
            "mean_absolute": normal[0],
            "root_mean_square": normal[1],
            "maximum_absolute": normal[2],
        },
        "coil_constraints": {
            "measurements": {
                "minimum_coil_coil_distance": coil_coil,
                "minimum_coil_surface_distance": coil_surface,
                "maximum_curvature": curvature,
                "maximum_mean_squared_curvature": mean_squared_curvature,
                "total_base_coil_length": length,
            },
            "limits": {
                "coil_coil_distance_threshold": 0.1,
                "coil_surface_distance_threshold": 0.3,
                "curvature_threshold": 5.0,
                "mean_squared_curvature_threshold": 5.0,
            },
        },
    }


def test_engineering_validation_uses_one_sided_ten_percent_target_envelope():
    module = load_benchmark_module()
    boundary_metrics = _final_metrics(
        coil_coil=0.09,
        coil_surface=0.27,
        curvature=5.5,
        mean_squared_curvature=5.5,
    )

    validation = module.engineering_target_validation(boundary_metrics, 0.10)

    assert validation["passed"]
    assert all(item["passed"] for item in validation["comparisons"].values())
    failed = module.engineering_target_validation(
        _final_metrics(coil_surface=0.269), 0.10
    )
    assert not failed["passed"]
    assert not failed["comparisons"]["minimum_coil_surface_distance"]["passed"]


def test_qoi_agreement_retains_objective_field_and_coil_measurements():
    module = load_benchmark_module()
    cpu = _final_metrics()
    gpu = _final_metrics(
        objective=2.2,
        normal=(0.011, 0.022, 0.033),
        coil_coil=0.11,
        coil_surface=0.33,
        curvature=5.5,
        mean_squared_curvature=5.5,
        length=13.2,
    )

    validation = module.qoi_backend_agreement(gpu, cpu, 0.10)

    assert validation["passed"]
    assert set(validation["comparisons"]) == {
        "objective",
        "normalized_normal_field_mean",
        "normalized_normal_field_rms",
        "normalized_normal_field_maximum",
        "minimum_coil_coil_distance",
        "minimum_coil_surface_distance",
        "maximum_curvature",
        "maximum_mean_squared_curvature",
        "total_base_coil_length",
    }
    assert not module.qoi_backend_agreement(
        _final_metrics(objective=2.3), cpu, 0.10
    )["passed"]


def test_json_default_converts_numpy_scalars():
    module = load_benchmark_module()

    rendered = json.dumps(
        {"passed": np.bool_(True), "value": np.float64(1.25)},
        default=module._json_default,
    )

    assert json.loads(rendered) == {"passed": True, "value": 1.25}


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
