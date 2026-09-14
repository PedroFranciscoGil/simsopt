"""Compare matched CPU/GPU AL solves using local engineering residuals."""

import argparse
import json
import math
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
from benchmark_objective import build_problem, environment
from optimization_metrics import (
    absolute_feasibility,
    export_final_design_visualization,
    final_coil_metrics,
    flatten_numeric_metrics,
    upper_bound_quality,
)
from problems import PROBLEMS, get_problem, objective_call_kwargs, objective_metadata
from scipy.optimize import minimize
from simsopt.gpu import (
    GpuConfig,
    ScipyAugmentedLagrangianBridge,
    ScipyCoilObjectiveBridge,
    minimal_coil_data,
    minimize_equality_augmented_lagrangian,
)

SCALING_POLICIES = ("family_l2", "sqrt_count", "identity")
MINIMUM_SCALING_POLICIES = ("sqrt_count", "identity")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), default="engineering")
    parser.add_argument("--max-outer-iterations", type=int, default=8)
    parser.add_argument("--max-inner-iterations", type=int, default=200)
    parser.add_argument("--mu-init", type=float, default=10.0)
    parser.add_argument("--mu-max", type=float, default=1e12)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-8)
    parser.add_argument("--constraint-tolerance", type=float, default=1e-6)
    parser.add_argument("--target-relative-tolerance", type=float, default=0.10)
    parser.add_argument("--inner-stationarity-factor", type=float, default=1.0)
    parser.add_argument("--inner-stationarity-relative-tolerance", type=float)
    parser.add_argument("--maxcor", type=int, default=100)
    parser.add_argument("--maxls", type=int, default=50)
    parser.add_argument(
        "--residual-scaling-policy", choices=SCALING_POLICIES, default="family_l2"
    )
    parser.add_argument(
        "--minimum-residual-scaling-policy",
        choices=MINIMUM_SCALING_POLICIES,
        default="sqrt_count",
    )
    parser.add_argument("--constraint-scale-reduction-factor", type=float, default=1.0)
    parser.add_argument(
        "--constraint-transform",
        choices=("identity", "smooth_abs"),
        default="identity",
    )
    parser.add_argument("--constraint-transform-epsilon", type=float, default=1e-3)
    parser.add_argument("--quadratic-flux-target", type=float, default=1e-5)
    parser.add_argument("--max-refinement-iterations", type=int, default=600)
    parser.add_argument("--max-refinement-evaluations", type=int, default=1500)
    parser.add_argument(
        "--refinement-constraint-weight-multiplier", type=float, default=1.0
    )
    parser.add_argument("--target-tile-size", type=int, default=1024)
    parser.add_argument("--source-tile-size", type=int, default=4320)
    parser.add_argument("--vjp-mode", choices=("autodiff", "custom"), default="custom")
    parser.add_argument("--current-scale", type=float, default=1e5)
    parser.add_argument("--distance-feasibility-tolerance", type=float, default=1e-4)
    parser.add_argument("--curvature-feasibility-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--mean-squared-curvature-feasibility-tolerance",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--initial-variables", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--allow-non-gpu", action="store_true")
    args = parser.parse_args()
    if args.max_outer_iterations < 1 or args.max_inner_iterations < 1:
        parser.error("iteration limits must be positive")
    if args.max_refinement_iterations < 1 or args.max_refinement_evaluations < 1:
        parser.error("refinement limits must be positive")
    if args.maxcor < 1 or args.maxls < 1:
        parser.error("maxcor and maxls must be positive")
    if args.target_tile_size < 1 or args.source_tile_size < 1:
        parser.error("tile sizes must be positive")
    for name in (
        "mu_init",
        "mu_max",
        "tau",
        "gradient_tolerance",
        "constraint_tolerance",
        "target_relative_tolerance",
        "inner_stationarity_factor",
        "constraint_scale_reduction_factor",
        "constraint_transform_epsilon",
        "quadratic_flux_target",
        "refinement_constraint_weight_multiplier",
        "current_scale",
        "distance_feasibility_tolerance",
        "curvature_feasibility_tolerance",
        "mean_squared_curvature_feasibility_tolerance",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"{name.replace('_', '-')} must be finite and positive")
    if args.mu_init <= 1 or args.tau <= 1 or args.mu_max < args.mu_init:
        parser.error("require mu-init > 1, tau > 1, and mu-max >= mu-init")
    if args.inner_stationarity_factor < 1:
        parser.error("inner-stationarity-factor must be at least one")
    if args.target_relative_tolerance >= 1:
        parser.error("target-relative-tolerance must be below one")
    if not 0 < args.constraint_scale_reduction_factor <= 1:
        parser.error("constraint-scale-reduction-factor must be in (0, 1]")
    if args.inner_stationarity_relative_tolerance is not None and (
        not np.isfinite(args.inner_stationarity_relative_tolerance)
        or not 0 < args.inner_stationarity_relative_tolerance < 1
    ):
        parser.error("inner-stationarity-relative-tolerance must be in (0, 1)")
    if args.no_visualization and args.visualization_dir is not None:
        parser.error("no-visualization and visualization-dir are mutually exclusive")
    return args


def residual_scales(initial_residuals, layout, policy):
    """Return one attenuation-only scale per local residual.

    ``family_l2`` uses ``max(sqrt(n), ||r_0||_2)`` for every member of a
    family. This bounds an initially active family's mapped L2 norm by one and
    prevents an inactive family with many entries from dominating when it
    first activates at residuals of order one.
    """
    residuals = np.asarray(initial_residuals, dtype=float)
    if residuals.ndim != 1 or not np.all(np.isfinite(residuals)):
        raise ValueError("initial_residuals must be a finite vector")
    if policy not in SCALING_POLICIES:
        raise ValueError(f"unknown residual scaling policy {policy!r}")
    scales = np.empty_like(residuals)
    diagnostics = {}
    expected_start = 0
    for name, item in layout.items():
        start, stop = int(item["start"]), int(item["stop"])
        if start != expected_start or stop <= start or stop > residuals.size:
            raise ValueError("residual layout must be contiguous and nonempty")
        values = residuals[start:stop]
        count_scale = math.sqrt(values.size)
        if policy == "family_l2":
            scale = max(1.0, count_scale, float(np.linalg.norm(values)))
        elif policy == "sqrt_count":
            scale = max(1.0, count_scale)
        else:
            scale = 1.0
        scales[start:stop] = scale
        diagnostics[name] = {
            "start": start,
            "stop": stop,
            "count": int(values.size),
            "initial_active_count": int(np.count_nonzero(values > 0.0)),
            "initial_maximum": float(np.max(values, initial=0.0)),
            "initial_l2_norm": float(np.linalg.norm(values)),
            "applied_scale": scale,
            "mapped_initial_l2_norm": float(np.linalg.norm(values / scale)),
        }
        expected_start = stop
    if expected_start != residuals.size:
        raise ValueError("residual layout does not cover the complete vector")
    return scales, diagnostics


def minimum_residual_scales(layout, residual_count, policy):
    """Return the lower bound used by staged family-scale continuation."""
    if policy not in MINIMUM_SCALING_POLICIES:
        raise ValueError(f"unknown minimum residual scaling policy {policy!r}")
    scales = np.empty(residual_count, dtype=float)
    offset = 0
    for item in layout.values():
        start, stop = int(item["start"]), int(item["stop"])
        if start != offset or stop <= start or stop > residual_count:
            raise ValueError("residual layout must be contiguous and nonempty")
        scale = math.sqrt(stop - start) if policy == "sqrt_count" else 1.0
        scales[start:stop] = scale
        offset = stop
    if offset != residual_count:
        raise ValueError("residual layout does not cover the complete vector")
    return scales


def _vector_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "active_count": int(np.count_nonzero(values > 0.0)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "l2_norm": float(np.linalg.norm(values)),
        "norm_infinity": float(np.linalg.norm(values, ord=np.inf)),
    }


def family_summaries(values, layout):
    values = np.asarray(values, dtype=float)
    return {
        name: _vector_summary(values[item["start"] : item["stop"]])
        for name, item in layout.items()
    }


def relative_error(actual, expected, floor=1e-30):
    return float(
        np.linalg.norm(np.asarray(actual) - np.asarray(expected))
        / max(float(np.linalg.norm(expected)), floor)
    )


def gate(measured, threshold, passed):
    return {"measured": measured, "threshold": threshold, "passed": bool(passed)}


class RefinementRecorder:
    """Retain accepted refinement iterates without timing oracle diagnostics."""

    def __init__(self, function):
        self.function = function
        self.evaluations = 0
        self.iteration_states = []

    def __call__(self, x):
        self.evaluations += 1
        return self.function(x)

    def callback(self, x):
        self.iteration_states.append(np.asarray(x, dtype=float).copy())


def target_envelope_objective_settings(settings, relative_tolerance, multiplier):
    """Return direct-penalty settings whose zero set is the target envelope."""
    refined = dict(settings)
    for name in ("coil_coil_distance_threshold", "coil_surface_distance_threshold"):
        refined[name] = settings[name] * (1.0 - relative_tolerance)
    for name in ("curvature_threshold", "mean_squared_curvature_threshold"):
        refined[name] = settings[name] * (1.0 + relative_tolerance)
    for name in (
        "coil_coil_distance_weight",
        "coil_surface_distance_weight",
        "curvature_weight",
        "mean_squared_curvature_weight",
    ):
        refined[name] = settings[name] * multiplier
    refined["target_relative_tolerance"] = relative_tolerance
    refined["constraint_weight_multiplier"] = multiplier
    return refined


def target_envelope_residual_kwargs(settings, relative_tolerance, tolerances):
    """Parameterize local residuals so zero means inside the target envelope."""
    distance_tolerance = tolerances["distance"]
    curvature_tolerance = tolerances["curvature"]
    mean_squared_tolerance = tolerances["mean_squared_curvature"]
    return {
        "length_weight": 0.0,
        "coil_coil_distance_threshold": (
            settings["coil_coil_distance_threshold"]
            * (1.0 - relative_tolerance)
            + distance_tolerance
        ),
        "coil_surface_distance_threshold": (
            settings["coil_surface_distance_threshold"]
            * (1.0 - relative_tolerance)
            + distance_tolerance
        ),
        "curvature_threshold": (
            settings["curvature_threshold"] * (1.0 + relative_tolerance)
            - curvature_tolerance
        ),
        "mean_squared_curvature_threshold": (
            settings["mean_squared_curvature_threshold"]
            * (1.0 + relative_tolerance)
            - mean_squared_tolerance
        ),
        "distance_feasibility_tolerance": distance_tolerance,
        "curvature_feasibility_tolerance": curvature_tolerance,
        "mean_squared_curvature_feasibility_tolerance": mean_squared_tolerance,
    }


def select_flux_checkpoint(initial_x, result, recorder, quality_bridge):
    """Select minimum flux among target-feasible accepted iterates."""
    states = [np.asarray(initial_x, dtype=float)]
    states.extend(recorder.iteration_states)
    states.append(np.asarray(result.x, dtype=float))
    unique_states = []
    for state in states:
        if not unique_states or not np.array_equal(state, unique_states[-1]):
            unique_states.append(state)
    checkpoints = []
    for index, state in enumerate(unique_states):
        flux, residuals = quality_bridge.evaluate_terms(state)
        residual_norm = float(np.linalg.norm(residuals, ord=np.inf))
        feasible = residual_norm <= 1e-12
        checkpoints.append(
            {
                "index": index,
                "quadratic_flux": float(flux),
                "target_envelope_residual_norm_infinity": residual_norm,
                "target_feasible": bool(feasible),
            }
        )
    selected_index = min(
        range(len(checkpoints)),
        key=lambda index: (
            not checkpoints[index]["target_feasible"],
            checkpoints[index]["quadratic_flux"]
            if checkpoints[index]["target_feasible"]
            else checkpoints[index]["target_envelope_residual_norm_infinity"],
            checkpoints[index]["quadratic_flux"],
        ),
    )
    return unique_states[selected_index].copy(), {
        "candidate_count": len(checkpoints),
        "feasible_candidate_count": sum(
            item["target_feasible"] for item in checkpoints
        ),
        "selected_index": selected_index,
        "selected": checkpoints[selected_index],
        "checkpoints": checkpoints,
    }


def select_common_flux_warm_start(states, quality_bridge):
    """Choose one AL warm start for a matched CPU/GPU refinement."""
    candidates = {}
    for name, state in states.items():
        flux, residuals = quality_bridge.evaluate_terms(state)
        residual_norm = float(np.linalg.norm(residuals, ord=np.inf))
        candidates[name] = {
            "quadratic_flux": float(flux),
            "target_envelope_residual_norm_infinity": residual_norm,
            "target_feasible": bool(residual_norm <= 1e-12),
        }
    selected = min(
        candidates,
        key=lambda name: (
            not candidates[name]["target_feasible"],
            candidates[name]["quadratic_flux"]
            if candidates[name]["target_feasible"]
            else candidates[name]["target_envelope_residual_norm_infinity"],
            candidates[name]["quadratic_flux"],
            name,
        ),
    )
    return np.asarray(states[selected], dtype=float).copy(), {
        "selected_backend": selected,
        "selected": candidates[selected],
        "candidates": candidates,
    }


def refinement_summary(result, recorder, seconds, selection):
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "seconds": seconds,
        "iterations": int(result.nit),
        "evaluations": int(result.nfev),
        "gradient_evaluations": int(result.njev),
        "recorded_evaluations": recorder.evaluations,
        "terminal_objective": float(result.fun),
        "terminal_gradient_norm_infinity": float(
            np.linalg.norm(result.jac, ord=np.inf)
        ),
        "checkpoint_selection": selection,
    }


def quadratic_flux_target_validation(metrics, target, relative_tolerance):
    """Apply a one-sided relative allowance to the explicit flux target."""
    allowed = target * (1.0 + relative_tolerance)
    measured = float(metrics["quadratic_flux"])
    return {
        "measured": measured,
        "target": target,
        "allowed_boundary": allowed,
        "passed": bool(measured <= allowed),
    }


def engineering_target_validation(metrics, relative_tolerance):
    """Validate physical measurements against one-sided engineering targets."""
    if not np.isfinite(relative_tolerance) or not 0 <= relative_tolerance < 1:
        raise ValueError("relative_tolerance must be in [0, 1)")
    constraints = metrics["coil_constraints"]
    measurements = constraints["measurements"]
    limits = constraints["limits"]
    specifications = {
        "minimum_coil_coil_distance": (
            "coil_coil_distance_threshold",
            "minimum",
        ),
        "minimum_coil_surface_distance": (
            "coil_surface_distance_threshold",
            "minimum",
        ),
        "maximum_curvature": ("curvature_threshold", "maximum"),
        "maximum_mean_squared_curvature": (
            "mean_squared_curvature_threshold",
            "maximum",
        ),
    }
    comparisons = {}
    for measurement_name, (limit_name, direction) in specifications.items():
        measured = float(measurements[measurement_name])
        target = float(limits[limit_name])
        if not np.isfinite(measured) or not np.isfinite(target) or target <= 0:
            raise ValueError(
                f"{measurement_name} and its engineering target must be finite, "
                "with a positive target"
            )
        if direction == "minimum":
            allowed_boundary = target * (1.0 - relative_tolerance)
            fractional_target_deviation = max(0.0, target - measured) / target
            passed = measured >= allowed_boundary or np.isclose(
                measured, allowed_boundary, rtol=1e-14, atol=0.0
            )
        else:
            allowed_boundary = target * (1.0 + relative_tolerance)
            fractional_target_deviation = max(0.0, measured - target) / target
            passed = measured <= allowed_boundary or np.isclose(
                measured, allowed_boundary, rtol=1e-14, atol=0.0
            )
        comparisons[measurement_name] = {
            "measured": measured,
            "target": target,
            "direction": direction,
            "allowed_boundary": allowed_boundary,
            "fractional_target_deviation": fractional_target_deviation,
            "passed": bool(passed),
        }
    return {
        "relative_tolerance": relative_tolerance,
        "comparisons": comparisons,
        "passed": all(item["passed"] for item in comparisons.values()),
    }


def qoi_backend_agreement(gpu_metrics, cpu_metrics, relative_tolerance):
    """Compare the explicitly retained final quantities of interest."""
    cpu_measurements = cpu_metrics["coil_constraints"]["measurements"]
    gpu_measurements = gpu_metrics["coil_constraints"]["measurements"]
    values = {
        "objective": (cpu_metrics["objective"], gpu_metrics["objective"]),
        "normalized_normal_field_mean": (
            cpu_metrics["normalized_normal_field"]["mean_absolute"],
            gpu_metrics["normalized_normal_field"]["mean_absolute"],
        ),
        "normalized_normal_field_rms": (
            cpu_metrics["normalized_normal_field"]["root_mean_square"],
            gpu_metrics["normalized_normal_field"]["root_mean_square"],
        ),
        "normalized_normal_field_maximum": (
            cpu_metrics["normalized_normal_field"]["maximum_absolute"],
            gpu_metrics["normalized_normal_field"]["maximum_absolute"],
        ),
        **{
            name: (cpu_measurements[name], gpu_measurements[name])
            for name in (
                "minimum_coil_coil_distance",
                "minimum_coil_surface_distance",
                "maximum_curvature",
                "maximum_mean_squared_curvature",
                "total_base_coil_length",
            )
        },
    }
    comparisons = {}
    for name, (cpu_value, gpu_value) in values.items():
        cpu_value = float(cpu_value)
        gpu_value = float(gpu_value)
        relative_difference = abs(gpu_value - cpu_value) / max(
            abs(cpu_value), 1e-14
        )
        passed = relative_difference <= relative_tolerance or np.isclose(
            relative_difference, relative_tolerance, rtol=1e-14, atol=0.0
        )
        comparisons[name] = {
            "cpu": cpu_value,
            "gpu": gpu_value,
            "relative_difference": relative_difference,
            "reference": "cpu",
            "passed": bool(passed),
        }
    return {
        "relative_tolerance": relative_tolerance,
        "comparisons": comparisons,
        "passed": all(item["passed"] for item in comparisons.values()),
    }


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def metric_relative_errors(candidate, reference):
    candidate_values = flatten_numeric_metrics(candidate)
    reference_values = flatten_numeric_metrics(reference)
    return {
        name: relative_error(
            candidate_values[name], reference_values[name], floor=1e-12
        )
        for name in reference_values
    }


def optimization_summary(result, coordinate_bridge, layout):
    return {
        "success": result.success,
        "message": result.message,
        "seconds": result.seconds,
        "outer_iterations": result.outer_iterations,
        "total_inner_iterations": result.total_inner_iterations,
        "total_evaluations": result.total_evaluations,
        "final_augmented_lagrangian": result.fun,
        "final_base_objective": result.base_objective,
        "final_gradient_norm": float(np.linalg.norm(result.jac)),
        "final_gradient_norm_infinity": float(np.linalg.norm(result.jac, ord=np.inf)),
        "final_raw_constraint_norm_infinity": float(
            np.linalg.norm(result.constraints, ord=np.inf)
        ),
        "final_al_constraint_norm_infinity": float(
            np.linalg.norm(result.scaled_constraints, ord=np.inf)
        ),
        "terminated_by_inner_safeguard": result.terminated_by_inner_safeguard,
        "final_residual_families": family_summaries(result.constraints, layout),
        "final_scaled_residual_families": family_summaries(
            result.scaled_constraints, layout
        ),
        "final_multiplier_families": family_summaries(
            result.lagrange_multipliers, layout
        ),
        "final_penalty_families": family_summaries(result.penalties, layout),
        "final_constraint_scale_families": family_summaries(
            result.constraint_scales, layout
        ),
        "final_physical_variables": coordinate_bridge.to_physical_variables(
            result.x
        ).tolist(),
        "outer_history": list(result.history),
    }


def nvidia_smi():
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader",
    ]
    try:
        return subprocess.check_output(command, text=True).strip().splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def main():
    args = parse_args()
    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_non_gpu:
        raise RuntimeError(
            f"JAX selected {backend!r}, not 'gpu'. Select an NVIDIA GPU runtime."
        )

    spec = get_problem(args.problem)
    settings = objective_metadata(spec, "full-engineering")
    surface, base_curves, field, components, cpu_penalty_objective = build_problem(
        spec, regularized=True, objective_settings=settings
    )
    base_current_objects = [field.coils[index].current for index in range(spec.ncoils)]
    free_current_indices = np.asarray(
        [index for index, current in enumerate(base_current_objects) if current.x.size],
        dtype=np.int32,
    )
    ntarget = surface.gamma().size // 3
    nsource = len(field.coils) * spec.nquad
    config = GpuConfig(
        target_tile_size=min(args.target_tile_size, ntarget),
        source_tile_size=min(args.source_tile_size, nsource),
        vjp_mode=args.vjp_mode,
    )
    device_data = minimal_coil_data(
        surface,
        base_curves,
        base_current_objects,
        surface.nfp,
        stellsym=True,
        config=config,
    )
    # Keep constants backend-neutral so separate CPU and GPU executables can be
    # lowered from the same numerical program in one process.
    data = replace(
        device_data,
        bases=np.asarray(device_data.bases),
        transforms=np.asarray(device_data.transforms),
        current_signs=np.asarray(device_data.current_signs),
    )
    coordinate_bridge = ScipyCoilObjectiveBridge(
        data,
        free_current_indices=free_current_indices,
        objective_kwargs={
            "length_target": None,
            "length_weight": settings["length_weight"],
        },
        current_scale=args.current_scale,
        config=config,
    )
    if args.initial_variables is None:
        physical_initial_x = cpu_penalty_objective.x.copy()
    else:
        physical_initial_x = np.asarray(
            json.loads(args.initial_variables.read_text()), dtype=float
        )
    if physical_initial_x.shape != coordinate_bridge.physical_initial_x.shape:
        raise ValueError(
            "initial physical variables must have shape "
            f"{coordinate_bridge.physical_initial_x.shape}, got {physical_initial_x.shape}"
        )
    initial_x = physical_initial_x / coordinate_bridge.coordinate_scales
    term_kwargs = {
        "length_weight": settings["length_weight"],
        "curvature_threshold": settings["curvature_threshold"],
        "curvature_feasibility_tolerance": args.curvature_feasibility_tolerance,
        "mean_squared_curvature_threshold": settings[
            "mean_squared_curvature_threshold"
        ],
        "mean_squared_curvature_feasibility_tolerance": (
            args.mean_squared_curvature_feasibility_tolerance
        ),
        "coil_coil_distance_threshold": settings["coil_coil_distance_threshold"],
        "coil_surface_distance_threshold": settings["coil_surface_distance_threshold"],
        "distance_feasibility_tolerance": args.distance_feasibility_tolerance,
    }

    def terms(x):
        curve_dofs, currents = coordinate_bridge.unpack(x)
        return data.local_residual_terms(
            curve_dofs, currents, **term_kwargs, config=config
        )

    layout = data.local_residual_layout()
    residual_count = int(next(reversed(layout.values()))["stop"])
    bootstrap = ScipyAugmentedLagrangianBridge(
        terms, initial_x, residual_count, platform="cpu"
    )
    _, initial_residuals = bootstrap.evaluate_terms(initial_x)
    scales, scale_diagnostics = residual_scales(
        initial_residuals, layout, args.residual_scaling_policy
    )
    minimum_scales = minimum_residual_scales(
        layout, residual_count, args.minimum_residual_scaling_policy
    )
    if np.any(minimum_scales > scales):
        raise ValueError(
            "minimum residual scales exceed initial scales; choose a compatible "
            "initial/minimum scaling policy"
        )
    accelerator_platform = "gpu" if backend == "gpu" else "cpu"
    cpu_bridge = ScipyAugmentedLagrangianBridge(
        terms,
        initial_x,
        residual_count,
        constraint_scales=scales,
        constraint_transform=args.constraint_transform,
        transform_epsilon=args.constraint_transform_epsilon,
        platform="cpu",
    )
    gpu_bridge = ScipyAugmentedLagrangianBridge(
        terms,
        initial_x,
        residual_count,
        constraint_scales=scales,
        constraint_transform=args.constraint_transform,
        transform_epsilon=args.constraint_transform_epsilon,
        platform=accelerator_platform,
    )
    initial_multipliers = np.zeros(residual_count)
    initial_penalties = np.full(residual_count, args.mu_init)
    compile_times = {}
    for name, bridge in (("cpu", cpu_bridge), ("gpu", gpu_bridge)):
        bridge.set_state(initial_multipliers, initial_penalties)
        start = time.perf_counter()
        bridge.compile(initial_x)
        compile_times[name] = time.perf_counter() - start

    cpu_base, cpu_residuals = cpu_bridge.evaluate_terms(initial_x)
    gpu_base, gpu_residuals = gpu_bridge.evaluate_terms(initial_x)
    cpu_al, cpu_gradient = cpu_bridge(initial_x)
    gpu_al, gpu_gradient = gpu_bridge(initial_x)
    initial_parity = {
        "base_objective_absolute_error": abs(gpu_base - cpu_base),
        "residual_relative_l2_error": relative_error(gpu_residuals, cpu_residuals),
        "residual_maximum_absolute_error": float(
            np.max(np.abs(gpu_residuals - cpu_residuals), initial=0.0)
        ),
        "augmented_lagrangian_absolute_error": abs(gpu_al - cpu_al),
        "gradient_relative_l2_error": relative_error(gpu_gradient, cpu_gradient),
        "cpu_residual_families": family_summaries(cpu_residuals, layout),
        "gpu_residual_families": family_summaries(gpu_residuals, layout),
    }
    solve_kwargs = {
        "mu_init": args.mu_init,
        "lagrange_multiplier_init": None,
        "tau": args.tau,
        "max_outer_iterations": args.max_outer_iterations,
        "max_inner_iterations": args.max_inner_iterations,
        "gradient_tolerance": args.gradient_tolerance,
        "constraint_tolerance": args.constraint_tolerance,
        "maxcor": args.maxcor,
        "maxls": args.maxls,
        "mu_max": args.mu_max,
        "require_inner_stationarity": True,
        "inner_stationarity_factor": args.inner_stationarity_factor,
        "inner_stationarity_relative_tolerance": (
            args.inner_stationarity_relative_tolerance
        ),
        "history_vector_mode": "summary",
        "penalty_update_mode": "global",
        "constraint_scale_reduction_factor": (args.constraint_scale_reduction_factor),
        "minimum_constraint_scales": minimum_scales,
    }
    cpu_bridge.evaluations = 0
    gpu_bridge.evaluations = 0
    cpu_result = minimize_equality_augmented_lagrangian(
        cpu_bridge, initial_x, **solve_kwargs
    )
    gpu_result = minimize_equality_augmented_lagrangian(
        gpu_bridge, initial_x, **solve_kwargs
    )

    # The AL establishes a feasible warm start.  A second, explicitly pinned
    # CPU/GPU phase then minimizes the full direct objective with its hinge
    # boundaries moved to the agreed 10% physical target envelope.  Accepted
    # iterates are retained so an aggressive L-BFGS-B step cannot replace a
    # lower-flux feasible design with an infeasible terminal state.
    refinement_settings = target_envelope_objective_settings(
        settings,
        args.target_relative_tolerance,
        args.refinement_constraint_weight_multiplier,
    )
    refinement_bridges = {
        name: ScipyCoilObjectiveBridge(
            data,
            free_current_indices=free_current_indices,
            objective_kwargs=objective_call_kwargs(refinement_settings),
            current_scale=args.current_scale,
            config=config,
            platform=platform,
        )
        for name, platform in (("cpu", "cpu"), ("gpu", accelerator_platform))
    }
    refinement_compile_times = {}
    for name, bridge in refinement_bridges.items():
        start = time.perf_counter()
        bridge.compile()
        refinement_compile_times[name] = time.perf_counter() - start

    quality_kwargs = target_envelope_residual_kwargs(
        settings,
        args.target_relative_tolerance,
        {
            "distance": args.distance_feasibility_tolerance,
            "curvature": args.curvature_feasibility_tolerance,
            "mean_squared_curvature": (
                args.mean_squared_curvature_feasibility_tolerance
            ),
        },
    )

    def quality_terms(x):
        curve_dofs, currents = coordinate_bridge.unpack(x)
        return data.local_residual_terms(
            curve_dofs, currents, **quality_kwargs, config=config
        )

    quality_bridge = ScipyAugmentedLagrangianBridge(
        quality_terms,
        initial_x,
        residual_count,
        platform=accelerator_platform,
    )
    start = time.perf_counter()
    quality_bridge.compile(initial_x)
    quality_compile_seconds = time.perf_counter() - start

    common_warm_x, common_warm_start = select_common_flux_warm_start(
        {"cpu": cpu_result.x, "gpu": gpu_result.x}, quality_bridge
    )

    refinement_options = {
        "maxiter": args.max_refinement_iterations,
        "maxfun": args.max_refinement_evaluations,
        "maxcor": args.maxcor,
        "maxls": args.maxls,
        "ftol": 0.0,
        "gtol": args.gradient_tolerance,
    }
    refinement_results = {}
    refinement_recorders = {}
    refinement_seconds = {}
    for name in ("cpu", "gpu"):
        recorder = RefinementRecorder(refinement_bridges[name])
        start = time.perf_counter()
        result = minimize(
            recorder,
            common_warm_x,
            method="L-BFGS-B",
            jac=True,
            callback=recorder.callback,
            options=refinement_options,
        )
        refinement_seconds[name] = time.perf_counter() - start
        refinement_results[name] = result
        refinement_recorders[name] = recorder

    selected_states = {}
    refinement_selections = {}
    for name in ("cpu", "gpu"):
        selected_states[name], refinement_selections[name] = select_flux_checkpoint(
            common_warm_x,
            refinement_results[name],
            refinement_recorders[name],
            quality_bridge,
        )

    cpu_physical_x = coordinate_bridge.to_physical_variables(selected_states["cpu"])
    gpu_physical_x = coordinate_bridge.to_physical_variables(selected_states["gpu"])
    cpu_metrics = final_coil_metrics(
        cpu_penalty_objective,
        components,
        base_curves,
        field,
        surface,
        cpu_physical_x,
        settings,
    )
    gpu_metrics = final_coil_metrics(
        cpu_penalty_objective,
        components,
        base_curves,
        field,
        surface,
        gpu_physical_x,
        settings,
    )

    visualization_dir = None if args.no_visualization else args.visualization_dir
    visualization_stem = ""
    if (
        visualization_dir is None
        and args.output is not None
        and not args.no_visualization
    ):
        visualization_dir = args.output.parent
        visualization_stem = args.output.stem + "-"
    visualizations = None
    if visualization_dir is not None:
        visualization_dir.mkdir(parents=True, exist_ok=True)
        visualizations = {
            "cpu_final": export_final_design_visualization(
                cpu_penalty_objective,
                field,
                surface,
                cpu_physical_x,
                visualization_dir / f"{visualization_stem}cpu_final",
            ),
            "gpu_final": export_final_design_visualization(
                cpu_penalty_objective,
                field,
                surface,
                gpu_physical_x,
                visualization_dir / f"{visualization_stem}gpu_final",
            ),
        }

    normal_names = ("mean_absolute", "root_mean_square", "maximum_absolute")
    normal_quality = upper_bound_quality(
        {name: gpu_metrics["normalized_normal_field"][name] for name in normal_names},
        {name: cpu_metrics["normalized_normal_field"][name] for name in normal_names},
    )
    constraint_quality = upper_bound_quality(
        gpu_metrics["coil_constraints"]["violations"],
        cpu_metrics["coil_constraints"]["violations"],
    )
    feasibility_tolerances = {
        "maximum_curvature": args.curvature_feasibility_tolerance,
        "maximum_mean_squared_curvature": (
            args.mean_squared_curvature_feasibility_tolerance
        ),
        "minimum_coil_coil_distance": args.distance_feasibility_tolerance,
        "minimum_coil_surface_distance": args.distance_feasibility_tolerance,
    }
    cpu_feasibility = absolute_feasibility(
        cpu_metrics["coil_constraints"]["violations"], feasibility_tolerances
    )
    gpu_feasibility = absolute_feasibility(
        gpu_metrics["coil_constraints"]["violations"], feasibility_tolerances
    )
    total_cpu_seconds = cpu_result.seconds + refinement_seconds["cpu"]
    total_gpu_seconds = gpu_result.seconds + refinement_seconds["gpu"]
    speedup = total_cpu_seconds / total_gpu_seconds
    metric_errors = metric_relative_errors(gpu_metrics, cpu_metrics)
    parity_passed = (
        initial_parity["base_objective_absolute_error"] <= 1e-9
        and initial_parity["residual_relative_l2_error"] <= 1e-9
        and initial_parity["gradient_relative_l2_error"] <= 1e-7
    )
    cpu_target_validation = engineering_target_validation(
        cpu_metrics, args.target_relative_tolerance
    )
    gpu_target_validation = engineering_target_validation(
        gpu_metrics, args.target_relative_tolerance
    )
    cpu_flux_validation = quadratic_flux_target_validation(
        cpu_metrics, args.quadratic_flux_target, args.target_relative_tolerance
    )
    gpu_flux_validation = quadratic_flux_target_validation(
        gpu_metrics, args.quadratic_flux_target, args.target_relative_tolerance
    )
    backend_qoi_agreement = qoi_backend_agreement(
        gpu_metrics, cpu_metrics, args.target_relative_tolerance
    )
    scientific_validation = {
        "target_relative_tolerance": args.target_relative_tolerance,
        "cpu_engineering_targets": cpu_target_validation,
        "gpu_engineering_targets": gpu_target_validation,
        "cpu_quadratic_flux_target": cpu_flux_validation,
        "gpu_quadratic_flux_target": gpu_flux_validation,
        "passed": bool(
            cpu_target_validation["passed"]
            and gpu_target_validation["passed"]
            and cpu_flux_validation["passed"]
            and gpu_flux_validation["passed"]
        ),
    }
    acceptance_gates = {
        "gpu_backend": gate(backend, "gpu", backend == "gpu"),
        "initial_float64_parity": gate(
            initial_parity,
            {
                "base_objective_absolute_error": 1e-9,
                "residual_relative_l2_error": 1e-9,
                "gradient_relative_l2_error": 1e-7,
            },
            parity_passed,
        ),
        "engineering_target_envelope": gate(
            {
                "cpu": cpu_target_validation["comparisons"],
                "gpu": gpu_target_validation["comparisons"],
            },
            (
                "both backends satisfy every one-sided engineering target "
                f"within {args.target_relative_tolerance:.0%}"
            ),
            cpu_target_validation["passed"] and gpu_target_validation["passed"],
        ),
        "quadratic_flux_target": gate(
            {
                "cpu": cpu_flux_validation,
                "gpu": gpu_flux_validation,
            },
            (
                f"both backends have quadratic flux <= {args.quadratic_flux_target:g} "
                f"with {args.target_relative_tolerance:.0%} allowance"
            ),
            cpu_flux_validation["passed"] and gpu_flux_validation["passed"],
        ),
        "cpu_gpu_quantity_of_interest_agreement": gate(
            backend_qoi_agreement["comparisons"],
            f"relative difference <= {args.target_relative_tolerance:.0%}",
            backend_qoi_agreement["passed"],
        ),
        "end_to_end_optimization_speedup": gate(speedup, 3.0, speedup >= 3.0),
    }
    diagnostic_checks = {
        "tight_local_residual_feasibility": gate(
            {
                "cpu": float(np.linalg.norm(cpu_result.constraints, ord=np.inf)),
                "gpu": float(np.linalg.norm(gpu_result.constraints, ord=np.inf)),
            },
            args.constraint_tolerance,
            np.linalg.norm(cpu_result.constraints, ord=np.inf)
            <= args.constraint_tolerance
            and np.linalg.norm(gpu_result.constraints, ord=np.inf)
            <= args.constraint_tolerance,
        ),
        "legacy_five_percent_normal_field_quality": gate(
            normal_quality["comparisons"],
            "GPU <= max(1.05 * CPU, CPU + 1e-8)",
            normal_quality["passed"],
        ),
        "legacy_five_percent_constraint_quality": gate(
            constraint_quality["comparisons"],
            "GPU <= max(1.05 * CPU, CPU + 1e-8)",
            constraint_quality["passed"],
        ),
        "tight_absolute_engineering_feasibility": gate(
            {
                "cpu": cpu_feasibility["comparisons"],
                "gpu": gpu_feasibility["comparisons"],
            },
            "both backends satisfy every physical violation allowance",
            cpu_feasibility["passed"] and gpu_feasibility["passed"],
        ),
        "al_first_order_convergence": gate(
            {
                "cpu_success": cpu_result.success,
                "gpu_success": gpu_result.success,
                "cpu_gradient_norm": float(np.linalg.norm(cpu_result.jac)),
                "gpu_gradient_norm": float(np.linalg.norm(gpu_result.jac)),
                "cpu_final_gradient_reduction_ratio": cpu_result.history[-1][
                    "gradient_infinity_reduction_ratio"
                ],
                "gpu_final_gradient_reduction_ratio": gpu_result.history[-1][
                    "gradient_infinity_reduction_ratio"
                ],
            },
            {
                "both_success": True,
                "absolute_tolerance": args.gradient_tolerance,
                "relative_stage_tolerance": (
                    args.inner_stationarity_relative_tolerance
                ),
            },
            cpu_result.success and gpu_result.success,
        ),
        "al_physics_evaluation_budget": gate(
            gpu_result.total_evaluations,
            f"<= 1.1 * {cpu_result.total_evaluations}",
            gpu_result.total_evaluations
            <= math.ceil(1.1 * cpu_result.total_evaluations),
        ),
    }
    output = {
        "schema_version": 3,
        "workflow": "local_residual_augmented_lagrangian",
        "method": {
            "name": "local_residual_equality_augmented_lagrangian",
            "formula": "f - lambda^T (q(r) / s) + 0.5 sum(mu_i (q(r_i) / s_i)^2)",
            "constraint_semantics": (
                "nonnegative local hinge residuals normalized by physical "
                "feasibility allowances; raw zero means feasible"
            ),
            "jacobian_representation": "matrix-free reverse-mode VJP",
            "constraint_transform": args.constraint_transform,
            "constraint_transform_epsilon": args.constraint_transform_epsilon,
            "refinement": (
                "full direct-penalty objective with target-envelope hinge "
                "boundaries and best feasible accepted-iterate checkpointing"
            ),
        },
        "problem": spec.as_dict(),
        "objective": settings,
        "solver": {
            "inner_method": "L-BFGS-B",
            **{
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in solve_kwargs.items()
                if key != "minimum_constraint_scales"
            },
            "minimum_constraint_scales": "see residual_scaling.continuation",
        },
        "dimensions": {
            "optimization_variables": initial_x.size,
            "free_currents": free_current_indices.size,
            "curve_dofs": data.curve_dofs.size,
            "surface_points": ntarget,
            "source_points": nsource,
            "local_residuals": residual_count,
        },
        "gpu_configuration": {
            "target_tile_size": config.target_tile_size,
            "source_tile_size": config.source_tile_size,
            "vjp_mode": config.vjp_mode,
            "cpu_compilation_seconds": compile_times["cpu"],
            "gpu_compilation_seconds": compile_times["gpu"],
            "cpu_refinement_compilation_seconds": (
                refinement_compile_times["cpu"]
            ),
            "gpu_refinement_compilation_seconds": (
                refinement_compile_times["gpu"]
            ),
            "quality_compilation_seconds": quality_compile_seconds,
            "accelerator_platform": accelerator_platform,
        },
        "coordinate_scaling": {
            "current_scale_amperes": args.current_scale,
            "curve_scale": 1.0,
        },
        "residual_contract": {
            "family_order": list(layout),
            "layout": layout,
            "physical_feasibility_tolerances": feasibility_tolerances,
            "term_settings": term_kwargs,
        },
        "residual_scaling": {
            "policy": args.residual_scaling_policy,
            "formula": (
                "family_l2: max(sqrt(family_count), initial_family_l2); "
                "one shared attenuation scale per family"
            ),
            "family_diagnostics": scale_diagnostics,
            "continuation": {
                "reduction_factor": args.constraint_scale_reduction_factor,
                "minimum_policy": args.minimum_residual_scaling_policy,
                "minimum_family_scales": {
                    name: float(minimum_scales[item["start"]])
                    for name, item in layout.items()
                },
                "update_condition": (
                    "inner first-order acceptance and scaled AL progress"
                ),
                "multiplier_rescaling": (
                    "lambda_new = lambda_old * scale_new / scale_old"
                ),
            },
        },
        "flux_refinement": {
            "method": "L-BFGS-B",
            "quadratic_flux_target": args.quadratic_flux_target,
            "target_allowed_boundary": (
                args.quadratic_flux_target * (1.0 + args.target_relative_tolerance)
            ),
            "target_envelope_objective": refinement_settings,
            "target_envelope_residual_kwargs": quality_kwargs,
            "options": refinement_options,
            "checkpoint_policy": (
                "minimum quadratic flux among accepted iterates satisfying all "
                "engineering target-envelope residuals; otherwise minimum "
                "residual infinity norm then flux"
            ),
            "common_warm_start": common_warm_start,
        },
        "initial_state": {
            "source": "canonical_problem"
            if args.initial_variables is None
            else str(args.initial_variables),
            "physical_variables": physical_initial_x.tolist(),
        },
        "initial_parity": initial_parity,
        "validation_policy": {
            "name": "physical_quantity_target_envelope",
            "target_relative_tolerance": args.target_relative_tolerance,
            "scientific_validation_rule": (
                "scientifically_validated is determined exclusively by the "
                "physical engineering target envelope and explicit quadratic-"
                "flux target, not by raw gradient magnitude or CPU/GPU path "
                "agreement"
            ),
            "engineering_target_semantics": (
                "minimum-distance measurements may be at most the stated "
                "fraction below target; maximum curvature measurements may be "
                "at most the stated fraction above target"
            ),
            "normalized_normal_field_semantics": (
                "B dot n over abs(B) has ideal target zero, for which percentage "
                "error is undefined; retain mean, RMS, and maximum absolute values "
                "and report CPU/GPU agreement as an independent technical gate"
            ),
            "quadratic_flux_semantics": (
                "quadratic flux is a one-sided upper target with the same relative "
                "allowance as the engineering targets"
            ),
            "gradient_semantics": (
                "large coil-optimization gradients are expected and are recorded "
                "as optimizer diagnostics, not scientific validation vetoes"
            ),
            "strict_residual_semantics": (
                "strict local-AL residual and feasibility tolerances diagnose "
                "solver convergence but do not override physical target validation"
            ),
        },
        "scientific_validation": scientific_validation,
        "scientifically_validated": scientific_validation["passed"],
        "cpu": {
            "execution_platform": cpu_bridge.device_platform,
            "optimization": optimization_summary(cpu_result, coordinate_bridge, layout),
            "flux_refinement": refinement_summary(
                refinement_results["cpu"],
                refinement_recorders["cpu"],
                refinement_seconds["cpu"],
                refinement_selections["cpu"],
            ),
            "final_metrics": cpu_metrics,
        },
        "gpu": {
            "execution_platform": gpu_bridge.device_platform,
            "optimization": optimization_summary(gpu_result, coordinate_bridge, layout),
            "flux_refinement": refinement_summary(
                refinement_results["gpu"],
                refinement_recorders["gpu"],
                refinement_seconds["gpu"],
                refinement_selections["gpu"],
            ),
            "final_metrics": gpu_metrics,
        },
        "comparison": {
            "optimization_speedup": speedup,
            "al_speedup": cpu_result.seconds / gpu_result.seconds,
            "refinement_speedup": (
                refinement_seconds["cpu"] / refinement_seconds["gpu"]
            ),
            "cpu_total_optimization_seconds": total_cpu_seconds,
            "gpu_total_optimization_seconds": total_gpu_seconds,
            "amortized_speedup": total_cpu_seconds
            / (
                total_gpu_seconds
                + compile_times["gpu"]
                + refinement_compile_times["gpu"]
                + quality_compile_seconds
            ),
            "maximum_oracle_metric_relative_error": max(metric_errors.values()),
            "oracle_metric_relative_errors": metric_errors,
        },
        "acceptance_gates": acceptance_gates,
        "diagnostic_checks": diagnostic_checks,
        "all_gates_passed": all(
            item["passed"] for item in acceptance_gates.values()
        ),
        "visualizations": visualizations,
        "environment": environment(),
        "nvidia_smi": nvidia_smi(),
    }
    rendered = json.dumps(output, indent=2, default=_json_default)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
