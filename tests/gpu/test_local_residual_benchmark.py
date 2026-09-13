import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import numpy as np


def load_benchmark_module():
    benchmark_dir = Path(__file__).parents[2] / "benchmarks" / "gpu"
    specification = spec_from_file_location(
        "gpu_local_residual_benchmark",
        benchmark_dir / "benchmark_local_residual_parity.py",
    )
    module = module_from_spec(specification)
    with patch.object(sys, "path", [str(benchmark_dir), *sys.path]):
        specification.loader.exec_module(module)
    return module


def test_cpu_residual_oracle_uses_physical_allowances_and_fixed_order():
    module = load_benchmark_module()
    measurements = {
        "coil_coil_distance": np.array([[0.89, 0.91]]),
        "coil_surface_distance": np.array([[1.88, 1.92]]),
        "curvature": np.array([[3.3, 3.1]]),
        "mean_squared_curvature": np.array([4.4]),
    }
    settings = {
        "coil_coil_distance_threshold": 1.0,
        "coil_surface_distance_threshold": 2.0,
        "curvature_threshold": 3.0,
        "mean_squared_curvature_threshold": 4.0,
        "distance_feasibility_tolerance": 0.1,
        "curvature_feasibility_tolerance": 0.2,
        "mean_squared_curvature_feasibility_tolerance": 0.25,
    }

    families, residuals = module.residuals_from_measurements(measurements, settings)

    np.testing.assert_allclose(families["coil_coil_distance"], [[0.1, 0.0]])
    np.testing.assert_allclose(families["coil_surface_distance"], [[0.2, 0.0]])
    np.testing.assert_allclose(families["curvature"], [[0.5, 0.0]])
    np.testing.assert_allclose(families["mean_squared_curvature"], [0.6])
    np.testing.assert_allclose(residuals, [0.1, 0.0, 0.2, 0.0, 0.5, 0.0, 0.6])


def test_forced_settings_activate_constant_and_varying_families_off_boundary():
    module = load_benchmark_module()
    measurements = {
        "coil_coil_distance": np.array([[0.2, 0.4, 0.6]]),
        "coil_surface_distance": np.array([[0.3, 0.5, 0.7]]),
        "curvature": np.array([[2.0, 2.0, 2.0]]),
        "mean_squared_curvature": np.array([4.0, 4.0]),
    }
    settings = {
        "coil_coil_distance_threshold": 0.1,
        "coil_surface_distance_threshold": 0.3,
        "curvature_threshold": 5.0,
        "mean_squared_curvature_threshold": 5.0,
        "distance_feasibility_tolerance": 1e-4,
        "curvature_feasibility_tolerance": 1e-3,
        "mean_squared_curvature_feasibility_tolerance": 1e-3,
    }

    forced = module._forced_settings(measurements, settings)
    families, _ = module.residuals_from_measurements(measurements, forced)

    assert all(np.count_nonzero(values > 0.0) > 0 for values in families.values())
    np.testing.assert_allclose(families["curvature"], 10.0)
    np.testing.assert_allclose(families["mean_squared_curvature"], 10.0)
