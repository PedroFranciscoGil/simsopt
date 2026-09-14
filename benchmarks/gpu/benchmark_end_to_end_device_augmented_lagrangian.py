"""Compare original CPU and fully device-resident augmented Lagrangians."""

import argparse
import json
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
from benchmark_local_residual_augmented_lagrangian import (
    _json_default,
    engineering_target_validation,
    family_summaries,
    minimum_residual_scales,
    nvidia_smi,
    qoi_backend_agreement,
    quadratic_flux_target_validation,
    residual_scales,
)
from benchmark_objective import build_problem, environment
from optimization_metrics import export_final_design_visualization, final_coil_metrics
from problems import PROBLEMS, get_problem, objective_metadata
from simsopt.gpu import (
    DeviceAugmentedLagrangian,
    DeviceAugmentedLagrangianConfig,
    GpuConfig,
    ScipyAugmentedLagrangianBridge,
    ScipyCoilObjectiveBridge,
    minimal_coil_data,
    minimize_equality_augmented_lagrangian,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), default="engineering")
    parser.add_argument("--max-outer-iterations", type=int, default=8)
    parser.add_argument("--max-inner-iterations", type=int, default=300)
    parser.add_argument("--mu-init", type=float, default=10.0)
    parser.add_argument("--mu-max", type=float, default=1e12)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-8)
    parser.add_argument("--constraint-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--inner-stationarity-relative-tolerance", type=float, default=0.01
    )
    parser.add_argument("--history-size", type=int, default=20)
    parser.add_argument("--cpu-maxcor", type=int, default=100)
    parser.add_argument("--max-line-search-iterations", type=int, default=50)
    parser.add_argument("--constraint-transform-epsilon", type=float, default=0.1)
    parser.add_argument("--constraint-scale-reduction-factor", type=float, default=0.5)
    parser.add_argument("--target-relative-tolerance", type=float, default=0.10)
    parser.add_argument("--quadratic-flux-target", type=float, default=1e-5)
    parser.add_argument("--current-scale", type=float, default=1e5)
    parser.add_argument("--target-tile-size", type=int, default=1024)
    parser.add_argument("--source-tile-size", type=int, default=4320)
    parser.add_argument("--gpu-warm-repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--allow-non-gpu", action="store_true")
    args = parser.parse_args()
    for name in (
        "max_outer_iterations",
        "max_inner_iterations",
        "history_size",
        "cpu_maxcor",
        "max_line_search_iterations",
        "target_tile_size",
        "source_tile_size",
        "gpu_warm_repeats",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    for name in (
        "mu_init",
        "mu_max",
        "tau",
        "gradient_tolerance",
        "constraint_tolerance",
        "inner_stationarity_relative_tolerance",
        "constraint_transform_epsilon",
        "constraint_scale_reduction_factor",
        "target_relative_tolerance",
        "quadratic_flux_target",
        "current_scale",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name.replace('_', '-')} must be finite and positive")
    if args.mu_init <= 1 or args.mu_init > args.mu_max or args.tau <= 1:
        parser.error("penalty configuration is invalid")
    if not 0 < args.inner_stationarity_relative_tolerance < 1:
        parser.error("inner stationarity relative tolerance must be below one")
    if not 0 < args.constraint_scale_reduction_factor <= 1:
        parser.error("constraint scale reduction factor must be at most one")
    if not 0 < args.target_relative_tolerance < 1:
        parser.error("target relative tolerance must be below one")
    if args.no_visualization and args.visualization_dir is not None:
        parser.error("no-visualization and visualization-dir are mutually exclusive")
    return args


def optimization_summary(result):
    return {
        "success": bool(result.success),
        "message": result.message,
        "outer_iterations": int(result.outer_iterations),
        "total_inner_iterations": int(result.total_inner_iterations),
        "total_evaluations": int(result.total_evaluations),
        "seconds": float(result.seconds),
        "base_objective": float(result.base_objective),
        "final_augmented_lagrangian": float(result.fun),
        "final_gradient_norm_infinity": float(np.linalg.norm(result.jac, ord=np.inf)),
        "final_constraint_norm_infinity": float(
            np.linalg.norm(result.constraints, ord=np.inf)
        ),
        "final_scaled_constraint_norm_infinity": float(
            np.linalg.norm(result.scaled_constraints, ord=np.inf)
        ),
        "terminated_by_inner_safeguard": bool(
            result.terminated_by_inner_safeguard
        ),
        "history": list(result.history),
    }


def target_validation(metrics, target, tolerance):
    engineering = engineering_target_validation(metrics, tolerance)
    flux = quadratic_flux_target_validation(metrics, target, tolerance)
    return {
        "engineering": engineering,
        "quadratic_flux": flux,
        "passed": bool(engineering["passed"] and flux["passed"]),
    }


def main():
    args = parse_args()
    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_non_gpu:
        raise RuntimeError(
            f"JAX selected {backend!r}, not 'gpu'. Select an NVIDIA GPU runtime."
        )
    accelerator_platform = "gpu" if backend == "gpu" else backend
    spec = get_problem(args.problem)
    settings = objective_metadata(spec, "full-engineering")
    surface, base_curves, field, components, cpu_objective = build_problem(
        spec, regularized=True, objective_settings=settings
    )
    base_currents = [field.coils[index].current for index in range(spec.ncoils)]
    free_current_indices = np.asarray(
        [index for index, current in enumerate(base_currents) if current.x.size],
        dtype=np.int32,
    )
    ntarget = surface.gamma().size // 3
    nsource = len(field.coils) * spec.nquad
    gpu_config = GpuConfig(
        target_tile_size=min(args.target_tile_size, ntarget),
        source_tile_size=min(args.source_tile_size, nsource),
        vjp_mode="custom",
    )
    raw_data = minimal_coil_data(
        surface,
        base_curves,
        base_currents,
        surface.nfp,
        stellsym=True,
        config=gpu_config,
    )
    data = replace(
        raw_data,
        bases=np.asarray(raw_data.bases),
        transforms=np.asarray(raw_data.transforms),
        current_signs=np.asarray(raw_data.current_signs),
    )
    coordinate_bridge = ScipyCoilObjectiveBridge(
        data,
        free_current_indices=free_current_indices,
        objective_kwargs={
            "length_target": None,
            "length_weight": settings["length_weight"],
        },
        current_scale=args.current_scale,
        config=gpu_config,
    )
    physical_initial_x = cpu_objective.x.copy()
    initial_x = physical_initial_x / coordinate_bridge.coordinate_scales
    term_kwargs = {
        "length_weight": settings["length_weight"],
        "curvature_threshold": settings["curvature_threshold"],
        "curvature_feasibility_tolerance": 1e-3,
        "mean_squared_curvature_threshold": settings[
            "mean_squared_curvature_threshold"
        ],
        "mean_squared_curvature_feasibility_tolerance": 1e-3,
        "coil_coil_distance_threshold": settings["coil_coil_distance_threshold"],
        "coil_surface_distance_threshold": settings[
            "coil_surface_distance_threshold"
        ],
        "distance_feasibility_tolerance": 1e-4,
    }

    def terms(x):
        curve_dofs, currents = coordinate_bridge.unpack(x)
        return data.local_residual_terms(
            curve_dofs, currents, **term_kwargs, config=gpu_config
        )

    layout = data.local_residual_layout()
    residual_count = int(next(reversed(layout.values()))["stop"])
    bootstrap = ScipyAugmentedLagrangianBridge(
        terms, initial_x, residual_count, platform="cpu"
    )
    initial_base_objective, initial_constraints = bootstrap.evaluate_terms(initial_x)
    scales, scale_diagnostics = residual_scales(
        initial_constraints, layout, "family_l2"
    )
    minimum_scales = minimum_residual_scales(layout, residual_count, "sqrt_count")
    if np.any(minimum_scales > scales):
        raise ValueError("minimum residual scales exceed initial scales")

    cpu_bridge = ScipyAugmentedLagrangianBridge(
        terms,
        initial_x,
        residual_count,
        constraint_scales=scales,
        constraint_transform="smooth_abs",
        transform_epsilon=args.constraint_transform_epsilon,
        platform="cpu",
    ).set_state(np.zeros(residual_count), np.full(residual_count, args.mu_init))
    start = time.perf_counter()
    cpu_bridge.compile(initial_x)
    cpu_compile_seconds = time.perf_counter() - start
    cpu_bridge.evaluations = 0
    cpu_result = minimize_equality_augmented_lagrangian(
        cpu_bridge,
        initial_x,
        mu_init=args.mu_init,
        tau=args.tau,
        max_outer_iterations=args.max_outer_iterations,
        max_inner_iterations=args.max_inner_iterations,
        gradient_tolerance=args.gradient_tolerance,
        constraint_tolerance=args.constraint_tolerance,
        maxcor=args.cpu_maxcor,
        maxls=args.max_line_search_iterations,
        mu_max=args.mu_max,
        require_inner_stationarity=True,
        inner_stationarity_factor=1.0,
        inner_stationarity_relative_tolerance=(
            args.inner_stationarity_relative_tolerance
        ),
        history_vector_mode="summary",
        penalty_update_mode="global",
        constraint_scale_reduction_factor=(
            args.constraint_scale_reduction_factor
        ),
        minimum_constraint_scales=minimum_scales,
    )

    device_config = DeviceAugmentedLagrangianConfig(
        mu_init=args.mu_init,
        tau=args.tau,
        mu_max=args.mu_max,
        max_outer_iterations=args.max_outer_iterations,
        max_inner_iterations=args.max_inner_iterations,
        history_size=args.history_size,
        max_line_search_iterations=args.max_line_search_iterations,
        gradient_tolerance=args.gradient_tolerance,
        constraint_tolerance=args.constraint_tolerance,
        require_inner_stationarity=True,
        inner_stationarity_factor=1.0,
        inner_stationarity_relative_tolerance=(
            args.inner_stationarity_relative_tolerance
        ),
        penalty_update_mode="global",
        constraint_transform="smooth_abs",
        transform_epsilon=args.constraint_transform_epsilon,
        constraint_scale_reduction_factor=(
            args.constraint_scale_reduction_factor
        ),
    )
    device_solver = DeviceAugmentedLagrangian(
        terms,
        initial_x,
        residual_count,
        constraint_scales=scales,
        minimum_constraint_scales=minimum_scales,
        config=device_config,
        platform=accelerator_platform,
    )
    start = time.perf_counter()
    device_solver.compile(initial_x)
    device_compile_seconds = time.perf_counter() - start
    device_results = []
    execution_samples = []
    for _ in range(args.gpu_warm_repeats):
        result = device_solver.run(initial_x)
        device_results.append(result)
        execution_samples.append(result.seconds)
    device_result = device_results[0]
    device_warm_median = statistics.median(execution_samples)

    cpu_physical_x = coordinate_bridge.to_physical_variables(cpu_result.x)
    device_physical_x = coordinate_bridge.to_physical_variables(device_result.x)
    cpu_metrics = final_coil_metrics(
        cpu_objective,
        components,
        base_curves,
        field,
        surface,
        cpu_physical_x,
        settings,
    )
    device_metrics = final_coil_metrics(
        cpu_objective,
        components,
        base_curves,
        field,
        surface,
        device_physical_x,
        settings,
    )
    cpu_validation = target_validation(
        cpu_metrics, args.quadratic_flux_target, args.target_relative_tolerance
    )
    device_validation = target_validation(
        device_metrics, args.quadratic_flux_target, args.target_relative_tolerance
    )
    agreement = qoi_backend_agreement(
        device_metrics, cpu_metrics, args.target_relative_tolerance
    )

    visualization_dir = None if args.no_visualization else args.visualization_dir
    if visualization_dir is None and args.output is not None and not args.no_visualization:
        visualization_dir = args.output.parent
    visualizations = None
    if visualization_dir is not None:
        visualization_dir.mkdir(parents=True, exist_ok=True)
        visualizations = {
            "cpu_final": export_final_design_visualization(
                cpu_objective,
                field,
                surface,
                cpu_physical_x,
                visualization_dir / "end-to-end-al-cpu",
            ),
            "gpu_native_final": export_final_design_visualization(
                cpu_objective,
                field,
                surface,
                device_physical_x,
                visualization_dir / "end-to-end-al-gpu-native",
            ),
        }

    cpu_end_to_end = cpu_compile_seconds + cpu_result.seconds
    gpu_cold_end_to_end = device_compile_seconds + execution_samples[0]
    output = {
        "schema_version": 1,
        "workflow": "end_to_end_device_augmented_lagrangian",
        "method": {
            "name": "local_residual_equality_augmented_lagrangian",
            "refinement_performed": False,
            "matched_outer_update": True,
            "cpu_inner_solver": "SciPy L-BFGS-B",
            "gpu_inner_solver": "compiled device-resident L-BFGS",
            "gpu_outer_loop_device_resident": True,
            "gpu_host_callbacks": 0,
        },
        "problem": spec.as_dict(),
        "solver": {
            "max_outer_iterations": args.max_outer_iterations,
            "max_inner_iterations": args.max_inner_iterations,
            "mu_init": args.mu_init,
            "mu_max": args.mu_max,
            "tau": args.tau,
            "gradient_tolerance": args.gradient_tolerance,
            "constraint_tolerance": args.constraint_tolerance,
            "inner_stationarity_relative_tolerance": (
                args.inner_stationarity_relative_tolerance
            ),
            "history_size": args.history_size,
            "cpu_maxcor": args.cpu_maxcor,
            "max_line_search_iterations": args.max_line_search_iterations,
            "constraint_transform": "smooth_abs",
            "constraint_transform_epsilon": args.constraint_transform_epsilon,
            "penalty_update_mode": "global",
            "constraint_scale_reduction_factor": (
                args.constraint_scale_reduction_factor
            ),
            "residual_scaling_policy": "family_l2",
            "minimum_residual_scaling_policy": "sqrt_count",
        },
        "gpu_configuration": {
            "target_tile_size": gpu_config.target_tile_size,
            "source_tile_size": gpu_config.source_tile_size,
            "vjp_mode": gpu_config.vjp_mode,
            "accelerator_platform": accelerator_platform,
        },
        "initial_state": {
            "source": "canonical_equally_spaced_circular_coils",
            "physical_variables": physical_initial_x.tolist(),
            "base_objective": initial_base_objective,
            "constraint_norm_infinity": float(
                np.linalg.norm(initial_constraints, ord=np.inf)
            ),
            "constraint_families": family_summaries(initial_constraints, layout),
            "residual_scale_diagnostics": scale_diagnostics,
        },
        "validation_policy": {
            "target_relative_tolerance": args.target_relative_tolerance,
            "quadratic_flux_target": args.quadratic_flux_target,
            "quadratic_flux_allowed_boundary": (
                args.quadratic_flux_target
                * (1.0 + args.target_relative_tolerance)
            ),
            "scientific_rule": (
                "each endpoint independently passes quadratic flux and all "
                "engineering target envelopes; gradient size is diagnostic"
            ),
            "trajectory_agreement_is_diagnostic": True,
        },
        "cpu": {
            "execution_platform": cpu_bridge.device_platform,
            "implementation": "Python AL outer loop with SciPy L-BFGS-B inner solves",
            "compilation_seconds": cpu_compile_seconds,
            "end_to_end_seconds": cpu_end_to_end,
            "optimization": optimization_summary(cpu_result),
            "final_metrics": cpu_metrics,
            "scientific_validation": cpu_validation,
            "physical_variables": cpu_physical_x.tolist(),
        },
        "gpu_native": {
            "execution_platform": device_solver.device_platform,
            "implementation": "single compiled AL outer/inner executable",
            "device_resident": True,
            "host_callbacks": 0,
            "compilation_seconds": device_compile_seconds,
            "execution_samples_seconds": execution_samples,
            "warm_median_seconds": device_warm_median,
            "cold_end_to_end_seconds": gpu_cold_end_to_end,
            "optimization": optimization_summary(device_result),
            "final_metrics": device_metrics,
            "scientific_validation": device_validation,
            "physical_variables": device_physical_x.tolist(),
        },
        "comparison": {
            "warm_optimization_speedup": cpu_result.seconds / device_warm_median,
            "cold_end_to_end_speedup": cpu_end_to_end / gpu_cold_end_to_end,
            "cpu_to_gpu_evaluation_ratio": (
                cpu_result.total_evaluations
                / max(device_result.total_evaluations, 1)
            ),
            "qoi_backend_agreement": agreement,
        },
        "scientifically_validated": bool(
            cpu_validation["passed"] and device_validation["passed"]
        ),
        "technical_trajectory_agreement": bool(agreement["passed"]),
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
