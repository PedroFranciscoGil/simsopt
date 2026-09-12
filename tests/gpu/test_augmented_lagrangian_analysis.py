import json
import struct
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pytest


def load_analysis_module():
    benchmark_dir = Path(__file__).parents[2] / "benchmarks" / "gpu"
    spec = spec_from_file_location(
        "gpu_augmented_lagrangian_analysis",
        benchmark_dir / "analyze_augmented_lagrangian.py",
    )
    module = module_from_spec(spec)
    with patch.object(sys, "path", [str(benchmark_dir), *sys.path]):
        spec.loader.exec_module(module)
    return module


def example_record():
    return {
        "outer_iteration": 1,
        "inner_seconds": 2.0,
        "inner_evaluations": 3,
        "augmented_lagrangian": 0.5,
        "base_objective": 0.4,
        "gradient_norm": 0.1,
        "constraint_norm_infinity": 0.01,
        "constraints": [0.01, 0.0, 0.0, 0.0],
        "multipliers_before": [0.0] * 4,
        "multipliers_after": [-1.0, 0.0, 0.0, 0.0],
        "penalties_before": [10.0] * 4,
        "penalties_after": [100.0, 10.0, 10.0, 10.0],
    }


def example_result(module):
    record = example_record()
    return {
        "schema_version": 5,
        "method": {
            "name": "equality_zero_penalty_augmented_lagrangian",
            "constraint_names": list(module.CONSTRAINT_NAMES),
        },
        "solver": {"constraint_tolerance": 1e-8},
        "comparison": {"optimization_speedup": 2.0, "amortized_speedup": 1.5},
        "gpu_configuration": {"compilation_seconds": 1.0},
        "environment": {"simsopt_revision": "abc", "jax_backend": "gpu"},
        "initial_parity": {},
        "acceptance_gates": {},
        "cpu": {
            "optimization": {
                "outer_iterations": 1,
                "outer_history": [record],
            },
            "final_metrics": {},
        },
        "gpu": {
            "optimization": {
                "outer_iterations": 1,
                "outer_history": [{**record, "inner_seconds": 1.0}],
            },
            "final_metrics": {},
        },
    }


def vtk_surface(values):
    values = np.asarray(values, dtype="<f8")
    header = (
        b'<VTKFile><StructuredGrid WholeExtent="0 1 0 1 0 0">'
        b'<DataArray Name="B_dot_n_over_abs_B" type="Float64" offset="0"/>'
        b'</StructuredGrid><AppendedData encoding="raw">\n_'
    )
    return (
        header
        + struct.pack("<Q", values.nbytes)
        + values.tobytes()
        + b"</AppendedData>"
    )


def write_archive(path, module, result):
    with ZipFile(path, "w") as archive:
        archive.writestr(module.RESULT_NAME, json.dumps(result))
        for index, name in enumerate(module.SURFACE_NAMES):
            archive.writestr(name, vtk_surface(np.arange(4) + index))
        for name in module.COIL_NAMES:
            archive.writestr(name, b"nonempty")


def test_augmented_lagrangian_archive_validation_and_plots(tmp_path):
    module = load_analysis_module()
    archive = tmp_path / "result.zip"
    result = example_result(module)
    write_archive(archive, module, result)

    loaded, surfaces = module.read_archive(archive)
    assert loaded["schema_version"] == 5
    cpu, gpu = module.surface_fields(surfaces)
    np.testing.assert_array_equal(cpu, [[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_array_equal(gpu, cpu + 1.0)

    convergence = tmp_path / "convergence.png"
    performance = tmp_path / "performance.png"
    surface = tmp_path / "surface.png"
    module.plot_convergence(loaded, convergence)
    module.plot_performance(loaded, performance)
    module.plot_surface_fields(surfaces, surface)
    assert all(path.stat().st_size > 0 for path in (convergence, performance, surface))


def test_augmented_lagrangian_archive_rejects_wrong_schema(tmp_path):
    module = load_analysis_module()
    archive = tmp_path / "wrong.zip"
    result = example_result(module)
    result["schema_version"] = 4
    write_archive(archive, module, result)

    with pytest.raises(ValueError, match="schema version 5 or 6"):
        module.read_archive(archive)


def test_augmented_lagrangian_archive_accepts_scaled_schema(tmp_path):
    module = load_analysis_module()
    archive = tmp_path / "scaled.zip"
    result = example_result(module)
    result["schema_version"] = 6
    result["constraint_scaling"] = {
        "names": list(module.CONSTRAINT_NAMES),
        "scales": [1e-3, 1e-2, 1e-3, 1e-3],
    }
    for backend in ("cpu", "gpu"):
        record = result[backend]["optimization"]["outer_history"][0]
        record["scaled_constraint_norm_infinity"] = 1.0
        record["scaled_constraints"] = [1.0, 0.0, 0.0, 0.0]
    write_archive(archive, module, result)

    loaded, _ = module.read_archive(archive)
    assert loaded["schema_version"] == 6
