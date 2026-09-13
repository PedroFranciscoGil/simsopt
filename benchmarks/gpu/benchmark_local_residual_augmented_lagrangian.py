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
from problems import PROBLEMS, get_problem, objective_metadata
from simsopt.gpu import (
    GpuConfig,
    ScipyAugmentedLagrangianBridge,
    ScipyCoilObjectiveBridge,
    minimal_coil_data,
    minimize_equality_augmented_lagrangian,
)

SCALING_POLICIES = ("family_l2", "sqrt_count", "identity")


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
    parser.add_argument("--inner-stationarity-factor", type=float, default=1.0)
    parser.add_argument("--maxcor", type=int, default=100)
    parser.add_argument("--maxls", type=int, default=50)
    parser.add_argument(
        "--residual-scaling-policy", choices=SCALING_POLICIES, default="family_l2"
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
        "inner_stationarity_factor",
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
    accelerator_platform = "gpu" if backend == "gpu" else "cpu"
    cpu_bridge = ScipyAugmentedLagrangianBridge(
        terms,
        initial_x,
        residual_count,
        constraint_scales=scales,
        platform="cpu",
    )
    gpu_bridge = ScipyAugmentedLagrangianBridge(
        terms,
        initial_x,
        residual_count,
        constraint_scales=scales,
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
        "mu_init": initial_penalties,
        "lagrange_multiplier_init": initial_multipliers,
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
        "history_vector_mode": "summary",
        "penalty_update_mode": "global",
    }
    cpu_bridge.evaluations = 0
    gpu_bridge.evaluations = 0
    cpu_result = minimize_equality_augmented_lagrangian(
        cpu_bridge, initial_x, **solve_kwargs
    )
    cpu_physical_x = coordinate_bridge.to_physical_variables(cpu_result.x)
    cpu_metrics = final_coil_metrics(
        cpu_penalty_objective,
        components,
        base_curves,
        field,
        surface,
        cpu_physical_x,
        settings,
    )
    gpu_result = minimize_equality_augmented_lagrangian(
        gpu_bridge, initial_x, **solve_kwargs
    )
    gpu_physical_x = coordinate_bridge.to_physical_variables(gpu_result.x)
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
    speedup = cpu_result.seconds / gpu_result.seconds
    metric_errors = metric_relative_errors(gpu_metrics, cpu_metrics)
    parity_passed = (
        initial_parity["base_objective_absolute_error"] <= 1e-9
        and initial_parity["residual_relative_l2_error"] <= 1e-9
        and initial_parity["gradient_relative_l2_error"] <= 1e-7
    )
    gates = {
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
        "local_residual_feasibility": gate(
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
        "final_normal_field_quality": gate(
            normal_quality["comparisons"],
            "GPU <= max(1.05 * CPU, CPU + 1e-8)",
            normal_quality["passed"],
        ),
        "final_constraint_quality": gate(
            constraint_quality["comparisons"],
            "GPU <= max(1.05 * CPU, CPU + 1e-8)",
            constraint_quality["passed"],
        ),
        "absolute_engineering_feasibility": gate(
            {
                "cpu": cpu_feasibility["comparisons"],
                "gpu": gpu_feasibility["comparisons"],
            },
            "both backends satisfy every physical violation allowance",
            cpu_feasibility["passed"] and gpu_feasibility["passed"],
        ),
        "stationary_convergence": gate(
            {
                "cpu_success": cpu_result.success,
                "gpu_success": gpu_result.success,
                "cpu_gradient_norm": float(np.linalg.norm(cpu_result.jac)),
                "gpu_gradient_norm": float(np.linalg.norm(gpu_result.jac)),
            },
            {"both_success": True, "maximum_gradient_norm": args.gradient_tolerance},
            cpu_result.success and gpu_result.success,
        ),
        "physics_evaluation_budget": gate(
            gpu_result.total_evaluations,
            f"<= 1.1 * {cpu_result.total_evaluations}",
            gpu_result.total_evaluations
            <= math.ceil(1.1 * cpu_result.total_evaluations),
        ),
        "optimization_speedup": gate(speedup, 3.0, speedup >= 3.0),
    }
    output = {
        "schema_version": 1,
        "workflow": "local_residual_augmented_lagrangian",
        "method": {
            "name": "local_residual_equality_augmented_lagrangian",
            "formula": "f - lambda^T (r / s) + 0.5 sum(mu_i (r_i / s_i)^2)",
            "constraint_semantics": (
                "nonnegative local hinge residuals normalized by physical "
                "feasibility allowances; raw zero means feasible"
            ),
            "jacobian_representation": "matrix-free reverse-mode VJP",
        },
        "problem": spec.as_dict(),
        "objective": settings,
        "solver": {
            "inner_method": "L-BFGS-B",
            **{
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in solve_kwargs.items()
            },
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
        },
        "initial_state": {
            "source": "canonical_problem"
            if args.initial_variables is None
            else str(args.initial_variables),
            "physical_variables": physical_initial_x.tolist(),
        },
        "initial_parity": initial_parity,
        "cpu": {
            "execution_platform": cpu_bridge.device_platform,
            "optimization": optimization_summary(cpu_result, coordinate_bridge, layout),
            "final_metrics": cpu_metrics,
        },
        "gpu": {
            "execution_platform": gpu_bridge.device_platform,
            "optimization": optimization_summary(gpu_result, coordinate_bridge, layout),
            "final_metrics": gpu_metrics,
        },
        "comparison": {
            "optimization_speedup": speedup,
            "amortized_speedup": cpu_result.seconds
            / (gpu_result.seconds + compile_times["gpu"]),
            "maximum_oracle_metric_relative_error": max(metric_errors.values()),
            "oracle_metric_relative_errors": metric_errors,
        },
        "acceptance_gates": gates,
        "all_gates_passed": all(item["passed"] for item in gates.values()),
        "visualizations": visualizations,
        "environment": environment(),
        "nvidia_smi": nvidia_smi(),
    }
    rendered = json.dumps(output, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
