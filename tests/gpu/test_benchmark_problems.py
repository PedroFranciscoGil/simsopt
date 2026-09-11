from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def test_benchmark_problem_matrix():
    path = (Path(__file__).parents[2] / "benchmarks" / "gpu" / "problems.py").resolve()
    spec = spec_from_file_location("gpu_benchmark_problems", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    assert set(module.PROBLEMS) == {"minimal", "engineering", "stress"}
    for name, problem in module.PROBLEMS.items():
        assert problem.name == name
        assert problem.ncoils > 0
        assert problem.order > 0
        assert problem.nquad >= 2 * problem.order + 1
        assert problem.nphi > 0
        assert problem.ntheta > 0

    with pytest.raises(ValueError, match="unknown problem"):
        module.get_problem("not-a-problem")

    engineering = module.core_objective_metadata(module.PROBLEMS["engineering"])
    assert engineering["scope"] == "gpu_native_flux_plus_length_core"
    assert engineering["terms"] == ["quadratic_flux", "curve_length_penalty"]
    assert "lp_curvature" in engineering["deferred_terms"]
    assert (
        module.core_objective_metadata(module.PROBLEMS["minimal"])["deferred_terms"]
        == []
    )

    local = module.objective_metadata(
        module.PROBLEMS["engineering"], "local-engineering"
    )
    assert local["scope"] == "gpu_native_local_engineering"
    assert "lp_curvature" in local["terms"]
    assert local["deferred_terms"] == [
        "coil_coil_distance",
        "coil_surface_distance",
    ]
    full = module.objective_metadata(module.PROBLEMS["engineering"], "full-engineering")
    assert full["scope"] == "gpu_native_full_engineering"
    assert full["length_target"] is None
    assert "coil_coil_distance" in full["terms"]
    assert "coil_surface_distance" in full["terms"]
    assert full["deferred_terms"] == []
    with pytest.raises(ValueError, match="objective scope"):
        module.objective_metadata(module.PROBLEMS["minimal"], "complete")


def test_scaled_engineering_weights_preserves_thresholds_and_length():
    path = (Path(__file__).parents[2] / "benchmarks" / "gpu" / "problems.py").resolve()
    spec = spec_from_file_location("gpu_benchmark_weight_scaling", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = module.objective_metadata(
        module.PROBLEMS["engineering"], "full-engineering"
    )

    scaled = module.scaled_engineering_weights(metadata, 10.0)

    for name in module.ENGINEERING_WEIGHT_NAMES:
        assert scaled[name] == pytest.approx(10.0 * metadata[name])
    assert scaled["length_weight"] == metadata["length_weight"]
    assert scaled["curvature_threshold"] == metadata["curvature_threshold"]
    assert scaled["engineering_weight_multiplier"] == 10.0
    assert "engineering_weight_multiplier" not in metadata


@pytest.mark.parametrize("multiplier", [0.0, -1.0, float("inf")])
def test_scaled_engineering_weights_rejects_invalid_multiplier(multiplier):
    path = (Path(__file__).parents[2] / "benchmarks" / "gpu" / "problems.py").resolve()
    spec = spec_from_file_location("gpu_benchmark_invalid_weight_scaling", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="finite and positive"):
        module.scaled_engineering_weights({}, multiplier)
