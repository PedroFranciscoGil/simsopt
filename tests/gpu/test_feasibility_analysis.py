import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pytest


def load_analysis_module():
    benchmark_dir = Path(__file__).parents[2] / "benchmarks" / "gpu"
    path = benchmark_dir / "analyze_feasibility_study.py"
    spec = spec_from_file_location("gpu_feasibility_analysis", path)
    module = module_from_spec(spec)
    with patch.object(sys, "path", [str(benchmark_dir), *sys.path]):
        spec.loader.exec_module(module)
    return module


def test_normalized_constraint_ratios_follow_named_tolerances():
    module = load_analysis_module()
    metrics = {
        "coil_constraints": {
            "violations": {
                "maximum_curvature": 0.002,
                "maximum_mean_squared_curvature": 0.0,
                "minimum_coil_coil_distance": 0.00005,
                "minimum_coil_surface_distance": 0.0002,
            }
        }
    }
    configuration = {
        "curvature_feasibility_tolerance": 0.001,
        "mean_squared_curvature_feasibility_tolerance": 0.0,
        "distance_feasibility_tolerance": 0.0001,
    }

    assert module.normalized_constraint_ratios(metrics, configuration) == {
        "maximum_curvature": 2.0,
        "maximum_mean_squared_curvature": 0.0,
        "minimum_coil_coil_distance": 0.5,
        "minimum_coil_surface_distance": 2.0,
    }


def test_feasibility_archive_rejects_missing_members(tmp_path):
    module = load_analysis_module()
    archive = tmp_path / "empty.zip"
    with ZipFile(archive, "w"):
        pass

    with pytest.raises(ValueError, match="missing required files"):
        module.read_archive(archive)
