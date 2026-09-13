import struct
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import numpy as np


def load_analysis_module():
    benchmark_dir = Path(__file__).parents[2] / "benchmarks" / "gpu"
    spec = spec_from_file_location(
        "gpu_al_conditioning_analysis",
        benchmark_dir / "analyze_augmented_lagrangian_conditioning.py",
    )
    module = module_from_spec(spec)
    with patch.object(sys, "path", [str(benchmark_dir), *sys.path]):
        spec.loader.exec_module(module)
    return module


def load_safeguard_analysis_module():
    benchmark_dir = Path(__file__).parents[2] / "benchmarks" / "gpu"
    spec = spec_from_file_location(
        "gpu_al_safeguard_analysis",
        benchmark_dir / "analyze_augmented_lagrangian_safeguards.py",
    )
    module = module_from_spec(spec)
    with patch.object(sys, "path", [str(benchmark_dir), *sys.path]):
        spec.loader.exec_module(module)
    return module


def synthetic_result(failed_ratio, *, success=True):
    violations = {
        "minimum_coil_coil_distance": failed_ratio * 1e-4,
        "minimum_coil_surface_distance": 0.0,
        "maximum_curvature": 0.0,
        "maximum_mean_squared_curvature": 0.0,
    }
    tolerances = {
        "minimum_coil_coil_distance": 1e-4,
        "minimum_coil_surface_distance": 1e-4,
        "maximum_curvature": 1e-3,
        "maximum_mean_squared_curvature": 1e-3,
    }
    backend = {
        "optimization": {
            "success": success,
            "final_gradient_norm": 1e-7,
            "total_evaluations": 10,
        },
        "final_metrics": {
            "coil_constraints": {"violations": violations},
            "normalized_normal_field": {"root_mean_square": 0.01},
        },
    }
    return {
        "acceptance_gates": {
            "gpu_backend": {"passed": True},
            "initial_float64_parity": {"passed": True},
        },
        "feasibility_tolerances": tolerances,
        "cpu": backend,
        "gpu": backend,
    }


def vtk_surface(signed, absolute=None):
    signed = np.asarray(signed, dtype="<f8")
    if absolute is None:
        absolute = np.abs(signed)
    absolute = np.asarray(absolute, dtype="<f8")
    first = signed.tobytes()
    second = absolute.tobytes()
    second_offset = 8 + len(first)
    header = f'''<VTKFile byte_order="LittleEndian" header_type="UInt64">
<StructuredGrid WholeExtent="0 1 0 1 0 0">
<DataArray Name="B_dot_n_over_abs_B" type="Float64" offset="0"/>
<DataArray Name="abs_B_dot_n_over_abs_B" type="Float64" offset="{second_offset}"/>
</StructuredGrid><AppendedData encoding="raw">\n_'''.encode()
    return (
        header
        + struct.pack("<Q", len(first))
        + first
        + struct.pack("<Q", len(second))
        + second
        + b"</AppendedData></VTKFile>"
    )


def test_analysis_rank_reproduces_feasibility_first_policy():
    module = load_analysis_module()
    feasible_but_nonstationary = synthetic_result(0.5, success=False)
    stationary_but_infeasible = synthetic_result(2.0, success=True)
    assert module.candidate_rank(feasible_but_nonstationary) < module.candidate_rank(
        stationary_but_infeasible
    )


def test_analysis_reads_cpu_gpu_surface_fields():
    module = load_analysis_module()
    cpu = vtk_surface([1.0, -2.0, 3.0, -4.0])
    gpu = vtk_surface([2.0, -1.0, 4.0, -3.0])

    fields = module.surface_fields({"cpu": cpu, "gpu": gpu})

    assert fields["cpu"].shape == (2, 2)
    np.testing.assert_array_equal(fields["gpu"] - fields["cpu"], 1.0)


def test_safeguard_analysis_reads_cpu_gpu_surface_fields():
    module = load_safeguard_analysis_module()
    cpu = vtk_surface([1.0, -2.0, 3.0, -4.0])
    gpu = vtk_surface([2.0, -1.0, 4.0, -3.0])

    fields = module.surface_fields({"cpu": cpu, "gpu": gpu})

    assert fields["cpu"].shape == (2, 2)
    np.testing.assert_array_equal(fields["gpu"] - fields["cpu"], 1.0)
