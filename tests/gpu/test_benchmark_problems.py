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
