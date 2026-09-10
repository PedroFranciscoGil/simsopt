"""Canonical problem sizes for GPU-native coil benchmarks."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    """Static dimensions and objective selection for a benchmark problem."""

    name: str
    description: str
    ncoils: int
    order: int
    nquad: int
    nphi: int
    ntheta: int
    regularized: bool

    def as_dict(self):
        return asdict(self)


PROBLEMS = {
    "minimal": BenchmarkSpec(
        name="minimal",
        description="Documented minimal stage-two objective",
        ncoils=4,
        order=5,
        nquad=75,
        nphi=32,
        ntheta=32,
        regularized=False,
    ),
    "engineering": BenchmarkSpec(
        name="engineering",
        description="Stage-two objective with engineering regularizers",
        ncoils=4,
        order=8,
        nquad=120,
        nphi=64,
        ntheta=64,
        regularized=True,
    ),
    "stress": BenchmarkSpec(
        name="stress",
        description="High-resolution scaling and memory stress case",
        ncoils=6,
        order=12,
        nquad=180,
        nphi=128,
        ntheta=128,
        regularized=True,
    ),
}


def get_problem(name: str) -> BenchmarkSpec:
    """Return a benchmark specification by name."""
    try:
        return PROBLEMS[name]
    except KeyError as error:
        choices = ", ".join(sorted(PROBLEMS))
        raise ValueError(
            f"unknown problem {name!r}; choose one of {choices}"
        ) from error
