"""Canonical problem sizes for GPU-native coil benchmarks."""

from dataclasses import asdict, dataclass

OBJECTIVE_SCOPES = ("core", "local-engineering", "full-engineering")


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
    scope_names = {
        "core": "gpu_native_flux_plus_length_core",
        "local-engineering": "gpu_native_local_engineering",
        "full-engineering": "gpu_native_full_engineering",
    }
    metadata = {
        "scope": scope_names[scope],
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
    elif scope == "full-engineering":
        metadata.update(
            {
                "terms": [
                    "quadratic_flux",
                    "curve_length",
                    "coil_coil_distance",
                    "coil_surface_distance",
                    "lp_curvature",
                    "mean_squared_curvature_penalty",
                ],
                "length_target": None,
                "length_weight": 1e-6,
                "coil_coil_distance_threshold": 0.1,
                "coil_coil_distance_weight": 1000.0,
                "coil_surface_distance_threshold": 0.3,
                "coil_surface_distance_weight": 10.0,
                "curvature_threshold": 5.0,
                "curvature_weight": 1e-6,
                "mean_squared_curvature_threshold": 5.0,
                "mean_squared_curvature_weight": 1e-6,
            }
        )
    if spec.regularized and scope != "full-engineering":
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
        "coil_coil_distance_threshold",
        "coil_coil_distance_weight",
        "coil_surface_distance_threshold",
        "coil_surface_distance_weight",
    )
    return {name: metadata[name] for name in names if name in metadata}


def core_objective_metadata(spec: BenchmarkSpec):
    """Return metadata for the historical flux-plus-length benchmark."""
    return objective_metadata(spec, "core")
