import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
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


def test_device_checkpoint_patience_has_independent_measured_default():
    module = load_benchmark_module()
    with patch.object(sys, "argv", ["benchmark"]):
        args = module.parse_args()

    assert args.target_checkpoint_patience == 25
    assert args.device_lbfgs_checkpoint_patience == 15


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
    quadratic_flux=1e-5,
    normal=(0.01, 0.02, 0.03),
    coil_coil=0.1,
    coil_surface=0.3,
    curvature=5.0,
    mean_squared_curvature=5.0,
    length=12.0,
):
    return {
        "objective": objective,
        "quadratic_flux": quadratic_flux,
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


def test_target_envelope_settings_move_penalty_boundaries_inward():
    module = load_benchmark_module()
    settings = {
        "coil_coil_distance_threshold": 0.1,
        "coil_surface_distance_threshold": 0.3,
        "curvature_threshold": 5.0,
        "mean_squared_curvature_threshold": 4.0,
        "coil_coil_distance_weight": 1000.0,
        "coil_surface_distance_weight": 10.0,
        "curvature_weight": 1e-6,
        "mean_squared_curvature_weight": 2e-6,
    }

    refined = module.target_envelope_objective_settings(settings, 0.1, 3.0)

    assert refined["coil_coil_distance_threshold"] == pytest.approx(0.09)
    assert refined["coil_surface_distance_threshold"] == pytest.approx(0.27)
    assert refined["curvature_threshold"] == pytest.approx(5.5)
    assert refined["mean_squared_curvature_threshold"] == pytest.approx(4.4)
    assert refined["coil_coil_distance_weight"] == 3000.0
    assert refined["curvature_weight"] == 3e-6


def test_target_envelope_residuals_have_exact_physical_zero_boundaries():
    module = load_benchmark_module()
    settings = {
        "coil_coil_distance_threshold": 0.1,
        "coil_surface_distance_threshold": 0.3,
        "curvature_threshold": 5.0,
        "mean_squared_curvature_threshold": 4.0,
    }
    tolerances = {
        "distance": 1e-4,
        "curvature": 1e-3,
        "mean_squared_curvature": 2e-3,
    }

    kwargs = module.target_envelope_residual_kwargs(
        settings, 0.1, tolerances
    )

    assert kwargs["coil_coil_distance_threshold"] - 1e-4 == pytest.approx(0.09)
    assert kwargs["coil_surface_distance_threshold"] - 1e-4 == pytest.approx(0.27)
    assert kwargs["curvature_threshold"] + 1e-3 == pytest.approx(5.5)
    assert kwargs["mean_squared_curvature_threshold"] + 2e-3 == pytest.approx(
        4.4
    )


def test_flux_checkpoint_selects_lowest_flux_target_feasible_iterate():
    module = load_benchmark_module()

    class QualityBridge:
        def evaluate_terms(self, x):
            flux = float(x[0])
            residuals = np.asarray([max(0.0, 2.0 - float(x[1]))])
            return flux, residuals

    recorder = SimpleNamespace(
        iteration_states=[np.asarray([4.0, 1.0]), np.asarray([3.0, 2.0])]
    )
    result = SimpleNamespace(x=np.asarray([1.0, 1.5]))

    state, selection = module.select_flux_checkpoint(
        np.asarray([5.0, 2.0]), result, recorder, QualityBridge()
    )

    np.testing.assert_array_equal(state, [3.0, 2.0])
    assert selection["feasible_candidate_count"] == 2
    assert selection["selected"]["target_feasible"]
    assert selection["selected"]["quadratic_flux"] == 3.0


def test_common_warm_start_prefers_lowest_flux_feasible_al_endpoint():
    module = load_benchmark_module()

    class QualityBridge:
        def evaluate_terms(self, x):
            return float(x[0]), np.asarray([max(0.0, 2.0 - float(x[1]))])

    state, selection = module.select_common_flux_warm_start(
        {
            "cpu": np.asarray([4.0, 2.0]),
            "gpu": np.asarray([3.0, 2.0]),
            "infeasible": np.asarray([1.0, 1.0]),
        },
        QualityBridge(),
    )

    np.testing.assert_array_equal(state, [3.0, 2.0])
    assert selection["selected_backend"] == "gpu"
    assert selection["selected"]["target_feasible"]


def test_quadratic_flux_target_uses_one_sided_ten_percent_allowance():
    module = load_benchmark_module()

    assert module.quadratic_flux_target_validation(
        _final_metrics(quadratic_flux=1.1e-5), 1e-5, 0.1
    )["passed"]
    assert not module.quadratic_flux_target_validation(
        _final_metrics(quadratic_flux=1.1001e-5), 1e-5, 0.1
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
    assert '"--quadratic-flux-target", "1e-5"' in source
    assert 'result["schema_version"] == 4' in source
    assert 'result["device_lbfgs"]["device_resident"]' in source
    assert 'result["device_gpu"]["execution_platform"] == "gpu"' in source
    assert '"--target-checkpoint-patience", "25"' in source
    assert '"--device-lbfgs-history-size", "20"' in source
    assert '"--device-lbfgs-checkpoint-patience", "15"' in source
    assert 'tests/gpu/test_lbfgs.py' in source
    assert "math.isclose" in source
    assert 'target_allowed_boundary"] == 1.1e-5' not in source
    assert "path = artifact_root / artifact" in source
    assert 'metrics = result[backend]["final_metrics"]' in source
    assert "files.download(archive)" in source
