from pathlib import Path

import pytest
from simsopt.field import Current
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from simsopt.gpu import GpuConfig, minimal_coil_data

TEST_FILE = (
    Path(__file__).parent / ".." / "test_files" / "input.LandremanPaul2021_QA"
).resolve()


def test_minimal_coil_adapter_shapes_and_metadata():
    surface = SurfaceRZFourier.from_vmec_input(
        TEST_FILE, range="half period", nphi=5, ntheta=6
    )
    curves = create_equally_spaced_curves(
        3, surface.nfp, surface.stellsym, order=4, numquadpoints=27
    )
    currents = [Current(1e5), Current(0.8e5), Current(-0.6e5)]
    data = minimal_coil_data(
        surface,
        curves,
        currents,
        surface.nfp,
        surface.stellsym,
        config=GpuConfig(target_tile_size=7, source_tile_size=13),
    )

    assert data.curve_dofs.shape == (3, 27)
    assert data.base_currents.shape == (3,)
    assert data.bases.shape == (4, 27, 9)
    assert data.transforms.shape == (2 * surface.nfp, 3, 3)
    assert data.surface_points.shape == (30, 3)
    assert data.surface_normal.shape == (30, 3)
    assert data.target_normal_field.shape == (30,)
    assert data.order == 4


def test_minimal_coil_adapter_rejects_inconsistent_inputs():
    surface = SurfaceRZFourier.from_vmec_input(TEST_FILE, nphi=3, ntheta=4)
    curves = create_equally_spaced_curves(
        2, surface.nfp, surface.stellsym, order=2, numquadpoints=15
    )
    currents = [Current(1e5), Current(0.8e5)]
    with pytest.raises(ValueError, match="equal length"):
        minimal_coil_data(surface, curves, currents[:1], 2, True)
    curves[1] = create_equally_spaced_curves(
        1, surface.nfp, surface.stellsym, order=2, numquadpoints=16
    )[0]
    with pytest.raises(ValueError, match="identical quadrature"):
        minimal_coil_data(surface, curves, currents, 2, True)


def test_gpu_config_rejects_unknown_vjp_mode():
    assert GpuConfig().vjp_mode == "custom"
    with pytest.raises(ValueError, match="vjp_mode"):
        GpuConfig(vjp_mode="unknown")
