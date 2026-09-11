"""Canonical problem sizes for GPU-native coil benchmarks."""

from dataclasses import asdict, dataclass

OBJECTIVE_SCOPES = ("core", "local-engineering")


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


def objective_metadata(spec: BenchmarkSpec, scope: str = "core"):
    """Describe one GPU benchmark objective without overstating its scope."""
    if scope not in OBJECTIVE_SCOPES:
        choices = ", ".join(OBJECTIVE_SCOPES)
        raise ValueError(f"unknown objective scope {scope!r}; choose one of {choices}")
    scope_name = (
        "gpu_native_flux_plus_length_core"
        if scope == "core"
        else "gpu_native_local_engineering"
    )
    metadata = {
        "scope": scope_name,
        "terms": ["quadratic_flux", "curve_length_penalty"],
        "deferred_terms": [],
        "length_target": 18.0,
        "length_weight": 1.0,
    }
    if scope == "local-engineering":
        metadata.update(
            {
                "terms": [
                    *metadata["terms"],
                    "lp_curvature",
                    "mean_squared_curvature_penalty",
                    "arclength_variation",
                ],
                "curvature_threshold": 5.0,
                "curvature_weight": 1e-6,
                "mean_squared_curvature_threshold": 5.0,
                "mean_squared_curvature_weight": 1e-6,
                "arclength_variation_weight": 1e-9,
            }
        )
    if spec.regularized:
        metadata["deferred_terms"] = [
            "coil_coil_distance",
            "coil_surface_distance",
        ]
        if scope == "core":
            metadata["deferred_terms"].extend(
                [
                    "lp_curvature",
                    "mean_squared_curvature_penalty",
                    "arclength_variation",
                ]
            )
    return metadata


def objective_call_kwargs(metadata):
    """Select objective-call keywords from benchmark metadata."""
    names = (
        "length_target",
        "length_weight",
        "curvature_threshold",
        "curvature_weight",
        "mean_squared_curvature_threshold",
        "mean_squared_curvature_weight",
        "arclength_variation_weight",
    )
    return {name: metadata[name] for name in names if name in metadata}


def core_objective_metadata(spec: BenchmarkSpec):
    """Return metadata for the historical flux-plus-length benchmark."""
    return objective_metadata(spec, "core")
