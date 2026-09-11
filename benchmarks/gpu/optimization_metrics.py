"""Physical final-state metrics shared by coil optimization benchmarks."""

import numpy as np


def normalized_normal_field_metrics(field_values, unit_normals):
    """Summarize ``|B dot n| / |B|`` over target-surface quadrature points."""
    field_values = np.asarray(field_values, dtype=float)
    unit_normals = np.asarray(unit_normals, dtype=float)
    if field_values.shape != unit_normals.shape or field_values.ndim < 2:
        raise ValueError(
            "field_values and unit_normals must have matching (..., 3) shapes"
        )
    if field_values.shape[-1] != 3:
        raise ValueError(
            "field_values and unit_normals must have matching (..., 3) shapes"
        )
    field_values = field_values.reshape((-1, 3))
    unit_normals = unit_normals.reshape((-1, 3))

    field_magnitude = np.linalg.norm(field_values, axis=1)
    valid = field_magnitude > 0.0
    if not np.any(valid):
        raise ValueError(
            "normalized normal field is undefined when |B| is zero everywhere"
        )
    normal_field = np.sum(field_values * unit_normals, axis=1)
    ratio = np.abs(normal_field[valid]) / field_magnitude[valid]
    return {
        "definition": "abs(B dot n) / norm(B)",
        "mean_absolute": float(np.mean(ratio)),
        "root_mean_square": float(np.sqrt(np.mean(ratio**2))),
        "maximum_absolute": float(np.max(ratio)),
        "valid_surface_points": int(np.count_nonzero(valid)),
        "zero_field_surface_points": int(np.count_nonzero(~valid)),
    }


def coil_constraint_metrics(base_curves, components, thresholds):
    """Return physical coil measurements, limits, margins, and violations."""
    lengths = []
    maximum_curvatures = []
    mean_squared_curvatures = []
    for curve in base_curves:
        arclength = np.asarray(curve.incremental_arclength(), dtype=float)
        curvature = np.asarray(curve.kappa(), dtype=float)
        lengths.append(float(np.mean(arclength)))
        maximum_curvatures.append(float(np.max(curvature)))
        mean_squared_curvatures.append(
            float(np.sum(curvature**2 * arclength) / np.sum(arclength))
        )

    measurements = {
        "base_coil_lengths": lengths,
        "total_base_coil_length": float(np.sum(lengths)),
        "base_coil_maximum_curvatures": maximum_curvatures,
        "maximum_curvature": float(np.max(maximum_curvatures)),
        "base_coil_mean_squared_curvatures": mean_squared_curvatures,
        "maximum_mean_squared_curvature": float(np.max(mean_squared_curvatures)),
    }
    if "coil_coil_distance" in components:
        measurements["minimum_coil_coil_distance"] = float(
            components["coil_coil_distance"].shortest_distance()
        )
    if "coil_surface_distance" in components:
        measurements["minimum_coil_surface_distance"] = float(
            components["coil_surface_distance"].shortest_distance()
        )

    limits = {
        name: thresholds[name]
        for name in (
            "length_target",
            "coil_coil_distance_threshold",
            "coil_surface_distance_threshold",
            "curvature_threshold",
            "mean_squared_curvature_threshold",
        )
        if thresholds.get(name) is not None
    }
    margin_specs = {
        "total_base_coil_length": ("length_target", -1.0),
        "minimum_coil_coil_distance": ("coil_coil_distance_threshold", 1.0),
        "minimum_coil_surface_distance": (
            "coil_surface_distance_threshold",
            1.0,
        ),
        "maximum_curvature": ("curvature_threshold", -1.0),
        "maximum_mean_squared_curvature": (
            "mean_squared_curvature_threshold",
            -1.0,
        ),
    }
    margins = {}
    violations = {}
    for measurement_name, (limit_name, direction) in margin_specs.items():
        if measurement_name not in measurements or limit_name not in limits:
            continue
        measured = measurements[measurement_name]
        limit = limits[limit_name]
        margin = direction * (measured - limit)
        margins[measurement_name] = float(margin)
        violations[measurement_name] = float(max(0.0, -margin))

    return {
        "measurements": measurements,
        "limits": limits,
        "margins": margins,
        "violations": violations,
    }


def final_coil_metrics(
    objective,
    components,
    base_curves,
    field,
    surface,
    x,
    thresholds,
):
    """Evaluate one final design entirely through the SIMSOPT CPU oracle."""
    objective.x = np.asarray(x)
    field_values = field.B().reshape(surface.gamma().shape)
    unit_normals = surface.unitnormal()
    normal_field = np.sum(field_values * unit_normals, axis=-1)
    coil_coil = components.get("coil_coil_distance")
    coil_surface = components.get("coil_surface_distance")
    metrics = {
        "objective": float(objective.J()),
        "gradient_norm": float(np.linalg.norm(objective.dJ())),
        "quadratic_flux": float(components["quadratic_flux"].J()),
        "curve_length_sum": float(components["curve_length_sum"].J()),
        "curvature_penalty": (
            float(components["curvature"].J()) if "curvature" in components else 0.0
        ),
        "mean_squared_curvature_penalty": (
            float(components["mean_squared_curvature_penalty"].J())
            if "mean_squared_curvature_penalty" in components
            else 0.0
        ),
        "mean_absolute_normal_field": float(np.mean(np.abs(normal_field))),
        "maximum_absolute_normal_field": float(np.max(np.abs(normal_field))),
        "normalized_normal_field": normalized_normal_field_metrics(
            field_values, unit_normals
        ),
        "coil_constraints": coil_constraint_metrics(
            base_curves, components, thresholds
        ),
        "base_current_values_amperes": [
            float(field.coils[index].current.get_value())
            for index in range(len(base_curves))
        ],
    }
    if coil_coil is not None:
        metrics["coil_coil_distance_penalty"] = float(coil_coil.J())
        metrics["minimum_coil_coil_distance"] = float(coil_coil.shortest_distance())
    if coil_surface is not None:
        metrics["coil_surface_distance_penalty"] = float(coil_surface.J())
        metrics["minimum_coil_surface_distance"] = float(
            coil_surface.shortest_distance()
        )
    metrics["maximum_curvature"] = metrics["coil_constraints"]["measurements"][
        "maximum_curvature"
    ]
    return metrics


def flatten_numeric_metrics(metrics, prefix=""):
    """Flatten numeric JSON leaves while ignoring labels and null values."""
    flattened = {}
    for name, value in metrics.items():
        qualified_name = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            flattened.update(flatten_numeric_metrics(value, qualified_name))
            continue
        if value is None or isinstance(value, str):
            continue
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number):
            flattened[qualified_name] = array
    return flattened
