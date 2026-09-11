import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


def load_metrics_module():
    benchmark_dir = Path(__file__).parents[2] / "benchmarks" / "gpu"
    path = benchmark_dir / "optimization_metrics.py"
    spec = spec_from_file_location("gpu_optimization_metrics", path)
    module = module_from_spec(spec)
    with patch.object(sys, "path", [str(benchmark_dir), *sys.path]):
        spec.loader.exec_module(module)
    return module


def test_normalized_normal_field_metrics_records_mean_rms_max_and_zero_count():
    metrics = load_metrics_module()
    field = np.asarray([[[1.0, 0, 0], [1.0, 1.0, 0], [0.0, 0, 0]]])
    normals = np.asarray([[[1.0, 0, 0], [0.0, 1.0, 0], [1.0, 0, 0]]])

    result = metrics.normalized_normal_field_metrics(field, normals)

    assert result["mean_absolute"] == pytest.approx((1 + 1 / np.sqrt(2)) / 2)
    assert result["root_mean_square"] == pytest.approx(np.sqrt(0.75))
    assert result["maximum_absolute"] == 1.0
    assert result["valid_surface_points"] == 2
    assert result["zero_field_surface_points"] == 1


class ExampleCurve:
    def __init__(self, arclength, curvature):
        self._arclength = np.asarray(arclength)
        self._curvature = np.asarray(curvature)

    def incremental_arclength(self):
        return self._arclength

    def kappa(self):
        return self._curvature


class ExampleDistance:
    def __init__(self, distance):
        self.distance = distance

    def shortest_distance(self):
        return self.distance


def test_coil_constraint_metrics_records_measurements_margins_and_violations():
    metrics = load_metrics_module()
    curves = [ExampleCurve([1, 1], [2, 4]), ExampleCurve([2, 2], [1, 1])]
    components = {
        "coil_coil_distance": ExampleDistance(0.2),
        "coil_surface_distance": ExampleDistance(0.25),
    }
    thresholds = {
        "length_target": 2.5,
        "coil_coil_distance_threshold": 0.1,
        "coil_surface_distance_threshold": 0.3,
        "curvature_threshold": 5.0,
        "mean_squared_curvature_threshold": 5.0,
    }

    result = metrics.coil_constraint_metrics(curves, components, thresholds)

    assert result["measurements"]["base_coil_lengths"] == [1.0, 2.0]
    assert result["measurements"]["maximum_curvature"] == 4.0
    assert result["measurements"]["maximum_mean_squared_curvature"] == 10.0
    assert result["margins"]["minimum_coil_coil_distance"] == pytest.approx(0.1)
    assert result["violations"]["minimum_coil_surface_distance"] == pytest.approx(0.05)
    assert result["violations"]["maximum_mean_squared_curvature"] == 5.0


def test_flatten_numeric_metrics_preserves_nested_numeric_arrays_only():
    metrics = load_metrics_module()

    result = metrics.flatten_numeric_metrics(
        {"label": "ignored", "missing": None, "group": {"values": [1, 2]}}
    )

    np.testing.assert_array_equal(result["group.values"], [1, 2])
    assert set(result) == {"group.values"}


def test_upper_bound_quality_applies_relative_and_absolute_slack():
    metrics = load_metrics_module()

    result = metrics.upper_bound_quality(
        {"finite": 10.5, "zero": 5e-9},
        {"finite": 10.0, "zero": 0.0},
    )

    assert result["passed"]
    assert result["comparisons"]["finite"]["allowed"] == 10.5
    assert result["comparisons"]["zero"]["allowed"] == 1e-8


def test_upper_bound_quality_rejects_material_regression():
    metrics = load_metrics_module()

    result = metrics.upper_bound_quality({"metric": 1.051}, {"metric": 1.0})

    assert not result["passed"]
    assert not result["comparisons"]["metric"]["passed"]


class ExampleObjective:
    x = None


class ExampleSurface:
    def __init__(self):
        self.extra_data = None

    def gamma(self):
        return np.zeros((1, 2, 3))

    def unitnormal(self):
        return np.asarray([[[1.0, 0, 0], [0.0, 1.0, 0]]])

    def to_vtk(self, filename, extra_data=None):
        self.filename = Path(filename)
        self.extra_data = extra_data


class ExampleCoil:
    def __init__(self, curve):
        self.curve = curve


class ExampleField:
    def __init__(self):
        self.coils = [ExampleCoil("curve-a"), ExampleCoil("curve-b")]

    def B(self):
        return np.asarray([[2.0, 0, 0], [0.0, -4.0, 0]])


def test_export_final_design_visualization_writes_vts_vtu_and_surface_data():
    metrics = load_metrics_module()
    objective = ExampleObjective()
    surface = ExampleSurface()
    field = ExampleField()

    with patch("simsopt.field.coil.coils_to_vtk") as coils_to_vtk:
        result = metrics.export_final_design_visualization(
            objective, field, surface, np.asarray([1.0, 2.0]), Path("final")
        )

    np.testing.assert_array_equal(objective.x, [1.0, 2.0])
    np.testing.assert_array_equal(
        surface.extra_data["B_dot_n_over_abs_B"].ravel(), [1.0, -1.0]
    )
    np.testing.assert_array_equal(
        surface.extra_data["abs_B_dot_n_over_abs_B"].ravel(), [1.0, 1.0]
    )
    coils_to_vtk.assert_called_once_with(field.coils, Path("final_coils"), close=True)
    assert result["surface_vts"] == "final_surface.vts"
    assert result["coils_vtu"] == "final_coils.vtu"
    assert result["coil_point_data"] == ["idx", "I"]
