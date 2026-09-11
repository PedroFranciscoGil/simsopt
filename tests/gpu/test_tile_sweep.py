import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

import pytest


def load_tile_sweep_module():
    benchmark_dir = Path(__file__).parents[2] / "benchmarks" / "gpu"
    path = benchmark_dir / "sweep_gpu_tiles.py"
    spec = spec_from_file_location("gpu_tile_sweep", path)
    module = module_from_spec(spec)
    with patch.object(sys, "path", [str(benchmark_dir), *sys.path]):
        spec.loader.exec_module(module)
    return module


def test_tile_size_parsing_and_dimension_clipping():
    module = load_tile_sweep_module()

    assert module.positive_size_list("64, 128,64,256") == (64, 128, 256)
    assert module.resolve_sizes((64, 128, 2048), 1200) == (64, 128, 1200)
    assert module.resolve_sizes((1024, 2048), 512) == (512,)
    assert module.vjp_mode_list("autodiff,custom,autodiff") == (
        "autodiff",
        "custom",
    )

    with pytest.raises(module.argparse.ArgumentTypeError, match="positive"):
        module.positive_size_list("32,0")
    with pytest.raises(
        module.argparse.ArgumentTypeError, match="comma-separated integers"
    ):
        module.positive_size_list("32,invalid")
    with pytest.raises(module.argparse.ArgumentTypeError, match="vjp modes"):
        module.vjp_mode_list("automatic")


def test_timing_summary_and_parity_gate():
    module = load_tile_sweep_module()

    summary = module.timing_summary([1.0, 2.0, 3.0])
    assert summary["median_seconds"] == 2.0
    assert summary["mean_seconds"] == 2.0
    assert summary["minimum_seconds"] == 1.0
    assert summary["maximum_seconds"] == 3.0
    assert summary["coefficient_of_variation"] == pytest.approx((2 / 3) ** 0.5 / 2)

    tolerances = {
        "value_absolute_error": 1e-9,
        "gradient_relative_l2_error": 1e-7,
    }
    passing = {
        "value_absolute_error": 1e-12,
        "curve_gradient_relative_l2_error": 1e-10,
        "current_gradient_relative_l2_error": 1e-10,
    }
    assert module.passes_parity(passing, tolerances)
    passing["curve_gradient_relative_l2_error"] = 2e-7
    assert not module.passes_parity(passing, tolerances)
