import struct
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest


def load_analysis_module():
    path = (
        Path(__file__).parents[2]
        / "benchmarks"
        / "gpu"
        / "analyze_extended_convergence.py"
    )
    spec = spec_from_file_location("gpu_extended_analysis", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vtk_payload(values):
    header = b"""<?xml version="1.0"?>
<VTKFile type="StructuredGrid" byte_order="LittleEndian" header_type="UInt64">
<StructuredGrid WholeExtent="0 0 0 1 0 1">
<DataArray Name="B_dot_n_over_abs_B" NumberOfComponents="1" type="Float64"
 format="appended" offset="0"/>
</StructuredGrid><AppendedData encoding="raw">
_"""
    data = np.asarray(values, dtype="<f8").tobytes()
    return header + struct.pack("<Q", len(data)) + data + b"</AppendedData></VTKFile>"


def test_raw_appended_vtk_array_and_surface_shape():
    module = load_analysis_module()
    payload = vtk_payload([1.0, -2.0, 3.0, -4.0])

    np.testing.assert_array_equal(
        module.appended_vtk_array(payload, "B_dot_n_over_abs_B"),
        [1.0, -2.0, 3.0, -4.0],
    )
    assert module.surface_shape(payload) == (1, 2, 2)


def test_raw_appended_vtk_array_rejects_missing_field():
    module = load_analysis_module()
    payload = vtk_payload([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="was not found"):
        module.appended_vtk_array(payload, "missing")
