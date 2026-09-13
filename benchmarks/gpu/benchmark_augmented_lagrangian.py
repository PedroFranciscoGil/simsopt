"""Compare CPU and GPU equality augmented-Lagrangian coil optimization."""

import argparse
import json
import math
import subprocess
import time
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

CONSTRAINT_NAMES = (
    "coil_coil_distance_penalty",
    "coil_surface_distance_penalty",
    "curvature_penalty",
    "mean_squared_curvature_penalty",
)


def positive_float_vector(text):
    """Parse four finite positive comma-separated constraint scales."""
    try:
        values = np.asarray([float(value) for value in text.split(",")])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "constraint scales must be comma-separated numbers"
        ) from error
    if values.shape != (len(CONSTRAINT_NAMES),):
        raise argparse.ArgumentTypeError(
            f"exactly {len(CONSTRAINT_NAMES)} constraint scales are required"
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise argparse.ArgumentTypeError(
            "constraint scales must be finite and positive"
        )
    return values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), default="engineering")
    parser.add_argument("--max-outer-iterations", type=int, default=8)
    parser.add_argument("--max-inner-iterations", type=int, default=50)
    parser.add_argument("--mu-init", type=float, default=10.0)
    parser.add_argument("--mu-max", type=float, default=1e12)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-8)
    parser.add_argument("--constraint-tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--constraint-scales",
        type=positive_float_vector,
        default=positive_float_vector("1,1,1,1"),
        help=(
            "Four positive scales in coil-coil, coil-surface, curvature, MSC "
            "order; the AL operates on q(c_i) / scale_i."
        ),
    )
    parser.add_argument(
        "--constraint-transform",
        choices=("identity", "smooth_sqrt"),
        default="identity",
        help="Coordinate transform applied before constraint scaling.",
    )
    parser.add_argument("--constraint-transform-epsilon", type=float, default=1e-4)
    parser.add_argument("--automatic-constraint-scaling", action="store_true")
    parser.add_argument(
        "--maximum-initial-constraint-gradient-ratio", type=float, default=1.0
    )
    parser.add_argument("--require-inner-stationarity", action="store_true")
    parser.add_argument("--inner-stationarity-factor", type=float, default=1.0)
    parser.add_argument("--maxcor", type=int, default=100)
    parser.add_argument("--maxls", type=int, default=20)
    parser.add_argument("--target-tile-size", type=int, default=1024)
    parser.add_argument("--source-tile-size", type=int, default=4320)
    parser.add_argument("--vjp-mode", choices=("autodiff", "custom"), default="custom")
    parser.add_argument("--current-scale", type=float, default=1.0)
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
    if (
        not np.isfinite(args.mu_init)
        or not np.isfinite(args.tau)
        or np.isnan(args.mu_max)
        or args.mu_init <= 1
        or args.tau <= 1
        or args.mu_max < args.mu_init
    ):
        parser.error("require mu-init > 1, tau > 1, and mu-max >= mu-init")
    if args.maxcor < 1 or args.maxls < 1:
        parser.error("maxcor and maxls must be positive")
    if args.target_tile_size < 1 or args.source_tile_size < 1:
        parser.error("tile sizes must be positive")
    if not np.isfinite(args.current_scale) or args.current_scale <= 0:
        parser.error("current-scale must be finite and positive")
    if (
        not np.isfinite(args.constraint_transform_epsilon)
        or args.constraint_transform_epsilon <= 0
    ):
        parser.error("constraint-transform-epsilon must be finite and positive")
    if (
        not np.isfinite(args.maximum_initial_constraint_gradient_ratio)
        or args.maximum_initial_constraint_gradient_ratio <= 0
    ):
        parser.error(
            "maximum-initial-constraint-gradient-ratio must be finite and positive"
        )
    if (
        not np.isfinite(args.inner_stationarity_factor)
        or args.inner_stationarity_factor < 1
    ):
        parser.error("inner-stationarity-factor must be finite and at least one")
    if args.automatic_constraint_scaling and not np.array_equal(
        args.constraint_scales, np.ones(len(CONSTRAINT_NAMES))
    ):
        parser.error(
            "automatic-constraint-scaling cannot be combined with manual scales"
        )
    for name in (
        "gradient_tolerance",
        "constraint_tolerance",
        "distance_feasibility_tolerance",
        "curvature_feasibility_tolerance",
        "mean_squared_curvature_feasibility_tolerance",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f"{name.replace('_', '-')} must be finite and nonnegative")
    if args.gradient_tolerance == 0 or args.constraint_tolerance == 0:
        parser.error("solver tolerances must be positive")
    if args.no_visualization and args.visualization_dir is not None:
        parser.error("no-visualization and visualization-dir are mutually exclusive")
    return args


class CpuAugmentedLagrangianBridge:
    """SIMSOPT CPU oracle using the same AL scalar and coordinate scaling."""

    def __init__(
        self,
        base_objective,
        constraints,
        coordinate_bridge,
        derivative_objects,
        constraint_scales,
        constraint_transform="identity",
        transform_epsilon=1e-4,
    ):
        self.base_objective = base_objective
        self.constraints = tuple(constraints)
        self.coordinate_bridge = coordinate_bridge
        self.derivative_objects = tuple(derivative_objects)
        self.constraint_scales = np.asarray(constraint_scales, dtype=float)
        self.constraint_transform = constraint_transform
        self.transform_epsilon = float(transform_epsilon)
        if self.constraint_scales.shape != (len(self.constraints),):
            raise ValueError("constraint_scales must match constraints")
        if not np.all(np.isfinite(self.constraint_scales)) or np.any(
            self.constraint_scales <= 0
        ):
            raise ValueError("constraint_scales must be finite and positive")
        if self.constraint_transform not in ("identity", "smooth_sqrt"):
            raise ValueError("unsupported constraint_transform")
        if not np.isfinite(self.transform_epsilon) or self.transform_epsilon <= 0:
            raise ValueError("transform_epsilon must be finite and positive")
        self.lagrange_multipliers = np.zeros(len(self.constraints))
        self.penalties = np.ones(len(self.constraints))
        self.evaluations = 0

    def set_state(self, lagrange_multipliers, penalties):
        self.lagrange_multipliers = np.asarray(lagrange_multipliers, dtype=float).copy()
        self.penalties = np.asarray(penalties, dtype=float).copy()
        return self

    def _set_x(self, x):
        self.base_objective.x = self.coordinate_bridge.to_physical_variables(x)

    def evaluate_terms(self, x):
        self._set_x(x)
        return float(self.base_objective.J()), np.asarray(
            [constraint.J() for constraint in self.constraints], dtype=float
        )

    def map_constraints(self, constraints):
        constraints = np.asarray(constraints, dtype=float)
        if constraints.shape != self.constraint_scales.shape:
            raise ValueError("constraints must match constraint_scales")
        if self.constraint_transform == "smooth_sqrt":
            if np.any(constraints < 0):
                raise ValueError("smooth_sqrt constraints must be nonnegative")
            constraints = (
                np.sqrt(constraints + self.transform_epsilon**2)
                - self.transform_epsilon
            )
        return constraints / self.constraint_scales

    def scale_constraints(self, constraints):
        return self.map_constraints(constraints)

    def constraint_map_derivative(self, constraints):
        constraints = np.asarray(constraints, dtype=float)
        if self.constraint_transform == "identity":
            derivative = np.ones_like(constraints)
        else:
            derivative = 0.5 / np.sqrt(constraints + self.transform_epsilon**2)
        return derivative / self.constraint_scales

    def evaluate_term_derivatives(self, x):
        """Return base gradient and constraint Jacobian in optimizer coordinates."""
        self._set_x(x)
        base_gradient = self.coordinate_bridge.pullback_gradient(
            self._flat_gradient(self.base_objective)
        )
        constraint_jacobian = np.stack(
            [
                self.coordinate_bridge.pullback_gradient(
                    self._flat_gradient(constraint)
                )
                for constraint in self.constraints
            ]
        )
        return base_gradient, constraint_jacobian

    def __call__(self, x):
        base_value, raw_constraint_values = self.evaluate_terms(x)
        constraint_values = self.map_constraints(raw_constraint_values)
        value = (
            base_value
            - np.dot(self.lagrange_multipliers, constraint_values)
            + 0.5 * np.dot(self.penalties, constraint_values**2)
        )
        gradient = self._flat_gradient(self.base_objective)
        coefficients = -self.lagrange_multipliers + self.penalties * constraint_values
        map_derivatives = self.constraint_map_derivative(raw_constraint_values)
        for coefficient, constraint, map_derivative in zip(
            coefficients,
            self.constraints,
            map_derivatives,
            strict=True,
        ):
            gradient += coefficient * map_derivative * self._flat_gradient(constraint)
        self.evaluations += 1
        return float(value), self.coordinate_bridge.pullback_gradient(gradient)

    def _flat_gradient(self, objective):
        derivative = objective.dJ(partials=True)
        gradient = np.concatenate(
            [
                np.asarray(derivative(optimizable), dtype=float).reshape((-1,))
                for optimizable in self.derivative_objects
            ]
        )
        expected = self.coordinate_bridge.physical_initial_x.shape
        if gradient.shape != expected:
            raise RuntimeError(
                f"CPU derivative has shape {gradient.shape}, expected {expected}"
            )
        return gradient


def relative_error(actual, expected, floor=1e-30):
    return float(
        np.linalg.norm(np.asarray(actual) - np.asarray(expected))
        / max(float(np.linalg.norm(expected)), floor)
    )


def gradient_balanced_constraint_scales(
    raw_constraints,
    base_gradient,
    constraint_jacobian,
    *,
    mu_init,
    constraint_transform,
    transform_epsilon,
    maximum_ratio,
):
    """Choose attenuation-only scales from initial AL gradient contributions.

    The returned scales are never below one.  Consequently automatic scaling
    cannot amplify a constraint relative to the unscaled formulation, which
    prevents the ``1 / s**2`` explosion observed in the value-scaled study.
    """
    raw_constraints = np.asarray(raw_constraints, dtype=float)
    base_gradient = np.asarray(base_gradient, dtype=float)
    constraint_jacobian = np.asarray(constraint_jacobian, dtype=float)
    if raw_constraints.ndim != 1 or base_gradient.ndim != 1:
        raise ValueError("constraints and base_gradient must be one-dimensional")
    if constraint_jacobian.shape != (raw_constraints.size, base_gradient.size):
        raise ValueError("constraint_jacobian shape does not match gradients")
    if not (
        np.all(np.isfinite(raw_constraints))
        and np.all(np.isfinite(base_gradient))
        and np.all(np.isfinite(constraint_jacobian))
    ):
        raise ValueError("calibration inputs must be finite")
    if constraint_transform == "smooth_sqrt" and np.any(raw_constraints < 0):
        raise ValueError("smooth_sqrt constraints must be nonnegative")
    if not np.isfinite(transform_epsilon) or transform_epsilon <= 0:
        raise ValueError("transform_epsilon must be finite and positive")
    if not np.isfinite(mu_init) or mu_init <= 0:
        raise ValueError("mu_init must be finite and positive")
    if not np.isfinite(maximum_ratio) or maximum_ratio <= 0:
        raise ValueError("maximum_ratio must be finite and positive")
    if constraint_transform == "identity":
        mapped = raw_constraints
        map_derivative = np.ones_like(raw_constraints)
    elif constraint_transform == "smooth_sqrt":
        mapped = np.sqrt(raw_constraints + transform_epsilon**2) - transform_epsilon
        map_derivative = 0.5 / np.sqrt(raw_constraints + transform_epsilon**2)
    else:
        raise ValueError("unsupported constraint_transform")
    base_gradient_norm = float(np.linalg.norm(base_gradient))
    denominator = maximum_ratio * max(base_gradient_norm, np.finfo(float).tiny)
    row_norms = np.linalg.norm(constraint_jacobian, axis=1)
    unscaled_contributions = (
        float(mu_init) * np.abs(mapped) * np.abs(map_derivative) * row_norms
    )
    scales = np.sqrt(np.maximum(unscaled_contributions / denominator, 1.0))
    diagnostics = {
        "base_gradient_norm": base_gradient_norm,
        "constraint_jacobian_row_norms": dict(
            zip(CONSTRAINT_NAMES, row_norms.tolist())
        ),
        "unscaled_initial_penalty_gradient_norm_estimates": dict(
            zip(CONSTRAINT_NAMES, unscaled_contributions.tolist())
        ),
        "maximum_ratio_to_base_gradient": maximum_ratio,
        "selected_scales": dict(zip(CONSTRAINT_NAMES, scales.tolist())),
        "balanced_initial_penalty_gradient_norm_estimates": dict(
            zip(CONSTRAINT_NAMES, (unscaled_contributions / scales**2).tolist())
        ),
    }
    return scales, diagnostics


def gate(measured, threshold, passed):
    return {"measured": measured, "threshold": threshold, "passed": bool(passed)}


def metric_relative_errors(gpu_metrics, cpu_metrics):
    gpu_values = flatten_numeric_metrics(gpu_metrics)
    cpu_values = flatten_numeric_metrics(cpu_metrics)
    return {
        name: relative_error(gpu_values[name], cpu_values[name], floor=1e-12)
        for name in cpu_values
    }


def result_summary(result, coordinate_bridge):
    raw_constraints = np.asarray(result.constraints)
    al_constraints = np.asarray(result.scaled_constraints)
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
        "terminated_by_inner_safeguard": result.terminated_by_inner_safeguard,
        "final_al_constraint_norm_infinity": float(
            np.linalg.norm(al_constraints, ord=np.inf)
        ),
        "final_al_constraints": dict(zip(CONSTRAINT_NAMES, al_constraints.tolist())),
        # Retained for schema-6 consumers.  In schema 7 these are the complete
        # transformed-and-scaled AL coordinates, not necessarily c / s alone.
        "final_scaled_constraint_norm_infinity": float(
            np.linalg.norm(al_constraints, ord=np.inf)
        ),
        "final_raw_constraint_norm_infinity": float(
            np.linalg.norm(raw_constraints, ord=np.inf)
        ),
        "final_constraints": dict(zip(CONSTRAINT_NAMES, raw_constraints.tolist())),
        "final_scaled_constraints": dict(
            zip(CONSTRAINT_NAMES, al_constraints.tolist())
        ),
        "final_lagrange_multipliers": dict(
            zip(CONSTRAINT_NAMES, result.lagrange_multipliers.tolist())
        ),
        "final_penalties": dict(zip(CONSTRAINT_NAMES, result.penalties.tolist())),
        "final_physical_variables": coordinate_bridge.to_physical_variables(
            result.x
        ).tolist(),
        "outer_history": list(result.history),
    }


def nvidia_smi():
    """Return accelerator identity without making it a benchmark dependency."""
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
    base_objective = (
        components["quadratic_flux"]
        + settings["length_weight"] * components["curve_length_sum"]
    )
    cpu_constraints = (
        components["coil_coil_distance"],
        components["coil_surface_distance"],
        components["curvature"],
        components["mean_squared_curvature_penalty"],
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
    data = minimal_coil_data(
        surface,
        base_curves,
        base_current_objects,
        surface.nfp,
        stellsym=True,
        config=config,
    )
    # Reuse the established SIMSOPT-coordinate adapter; its scalar objective is
    # not compiled or timed in this workflow.
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
        "mean_squared_curvature_threshold": settings[
            "mean_squared_curvature_threshold"
        ],
        "coil_coil_distance_threshold": settings["coil_coil_distance_threshold"],
        "coil_surface_distance_threshold": settings["coil_surface_distance_threshold"],
    }

    def gpu_terms(x):
        curve_dofs, currents = coordinate_bridge.unpack(x)
        return data.augmented_lagrangian_terms(
            curve_dofs, currents, **term_kwargs, config=config
        )

    derivative_objects = [
        *[current for current in base_current_objects if current.x.size],
        *base_curves,
    ]
    calibration_bridge = CpuAugmentedLagrangianBridge(
        base_objective,
        cpu_constraints,
        coordinate_bridge,
        derivative_objects,
        np.ones(len(CONSTRAINT_NAMES)),
        constraint_transform=args.constraint_transform,
        transform_epsilon=args.constraint_transform_epsilon,
    )
    cpu_base, cpu_constraint_values = calibration_bridge.evaluate_terms(initial_x)
    base_gradient, constraint_jacobian = calibration_bridge.evaluate_term_derivatives(
        initial_x
    )
    recommended_scales, gradient_balance = gradient_balanced_constraint_scales(
        cpu_constraint_values,
        base_gradient,
        constraint_jacobian,
        mu_init=args.mu_init,
        constraint_transform=args.constraint_transform,
        transform_epsilon=args.constraint_transform_epsilon,
        maximum_ratio=args.maximum_initial_constraint_gradient_ratio,
    )
    constraint_scales = (
        recommended_scales
        if args.automatic_constraint_scaling
        else np.asarray(args.constraint_scales, dtype=float)
    )
    unscaled_estimates = np.asarray(
        list(
            gradient_balance[
                "unscaled_initial_penalty_gradient_norm_estimates"
            ].values()
        )
    )
    gradient_balance["automatic_scaling_enabled"] = bool(
        args.automatic_constraint_scaling
    )
    gradient_balance["recommended_attenuation_scales"] = gradient_balance.pop(
        "selected_scales"
    )
    gradient_balance["applied_scales"] = dict(
        zip(CONSTRAINT_NAMES, constraint_scales.tolist())
    )
    gradient_balance["applied_initial_penalty_gradient_norm_estimates"] = dict(
        zip(CONSTRAINT_NAMES, (unscaled_estimates / constraint_scales**2).tolist())
    )

    gpu_bridge = ScipyAugmentedLagrangianBridge(
        gpu_terms,
        initial_x,
        len(CONSTRAINT_NAMES),
        constraint_scales=constraint_scales,
        constraint_transform=args.constraint_transform,
        transform_epsilon=args.constraint_transform_epsilon,
    )
    gpu_bridge.set_state(np.zeros(4), np.full(4, args.mu_init))
    compilation_start = time.perf_counter()
    gpu_bridge.compile(initial_x)
    compilation_seconds = time.perf_counter() - compilation_start
    cpu_bridge = CpuAugmentedLagrangianBridge(
        base_objective,
        cpu_constraints,
        coordinate_bridge,
        derivative_objects,
        constraint_scales,
        constraint_transform=args.constraint_transform,
        transform_epsilon=args.constraint_transform_epsilon,
    ).set_state(np.zeros(4), np.full(4, args.mu_init))

    cpu_base, cpu_constraint_values = cpu_bridge.evaluate_terms(initial_x)
    gpu_base, gpu_constraint_values = gpu_bridge.evaluate_terms(initial_x)
    cpu_value, cpu_gradient = cpu_bridge(initial_x)
    gpu_value, gpu_gradient = gpu_bridge(initial_x)
    initial_parity = {
        "base_objective_absolute_error": abs(gpu_base - cpu_base),
        "constraint_relative_l2_error": relative_error(
            gpu_constraint_values, cpu_constraint_values
        ),
        "augmented_lagrangian_absolute_error": abs(gpu_value - cpu_value),
        "gradient_relative_l2_error": relative_error(gpu_gradient, cpu_gradient),
        "cpu_constraints": dict(zip(CONSTRAINT_NAMES, cpu_constraint_values.tolist())),
        "gpu_constraints": dict(zip(CONSTRAINT_NAMES, gpu_constraint_values.tolist())),
        "cpu_al_constraints": dict(
            zip(
                CONSTRAINT_NAMES,
                cpu_bridge.map_constraints(cpu_constraint_values).tolist(),
            )
        ),
        "gpu_al_constraints": dict(
            zip(
                CONSTRAINT_NAMES,
                gpu_bridge.map_constraints(gpu_constraint_values).tolist(),
            )
        ),
    }
    # Schema-6 aliases retained for downstream readers during migration.
    initial_parity["cpu_scaled_constraints"] = initial_parity["cpu_al_constraints"]
    initial_parity["gpu_scaled_constraints"] = initial_parity["gpu_al_constraints"]

    solve_kwargs = {
        "mu_init": args.mu_init,
        "lagrange_multiplier_init": np.zeros(4),
        "tau": args.tau,
        "max_outer_iterations": args.max_outer_iterations,
        "max_inner_iterations": args.max_inner_iterations,
        "gradient_tolerance": args.gradient_tolerance,
        "constraint_tolerance": args.constraint_tolerance,
        "maxcor": args.maxcor,
        "maxls": args.maxls,
        "mu_max": args.mu_max,
        "require_inner_stationarity": args.require_inner_stationarity,
        "inner_stationarity_factor": args.inner_stationarity_factor,
    }
    # Exclude parity calls from optimization evaluation accounting.
    cpu_bridge.evaluations = 0
    gpu_bridge.evaluations = 0
    cpu_result = minimize_equality_augmented_lagrangian(
        cpu_bridge, initial_x, **solve_kwargs
    )
    cpu_metrics = final_coil_metrics(
        cpu_penalty_objective,
        components,
        base_curves,
        field,
        surface,
        coordinate_bridge.to_physical_variables(cpu_result.x),
        settings,
    )
    gpu_result = minimize_equality_augmented_lagrangian(
        gpu_bridge, initial_x, **solve_kwargs
    )
    gpu_metrics = final_coil_metrics(
        cpu_penalty_objective,
        components,
        base_curves,
        field,
        surface,
        coordinate_bridge.to_physical_variables(gpu_result.x),
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
                coordinate_bridge.to_physical_variables(cpu_result.x),
                visualization_dir / f"{visualization_stem}cpu_final",
            ),
            "gpu_final": export_final_design_visualization(
                cpu_penalty_objective,
                field,
                surface,
                coordinate_bridge.to_physical_variables(gpu_result.x),
                visualization_dir / f"{visualization_stem}gpu_final",
            ),
        }

    metric_errors = metric_relative_errors(gpu_metrics, cpu_metrics)
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
    gates = {
        "gpu_backend": gate(backend, "gpu", backend == "gpu"),
        "initial_float64_parity": gate(
            initial_parity,
            {
                "base_objective_absolute_error": 1e-9,
                "constraint_relative_l2_error": 1e-7,
                "gradient_relative_l2_error": 1e-7,
            },
            initial_parity["base_objective_absolute_error"] <= 1e-9
            and initial_parity["constraint_relative_l2_error"] <= 1e-7
            and initial_parity["gradient_relative_l2_error"] <= 1e-7,
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
            "both backends satisfy all physical violation tolerances",
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
        "schema_version": 7,
        "method": {
            "name": "equality_zero_penalty_augmented_lagrangian",
            "formula": "f - lambda^T c + 0.5 sum(mu_i c_i^2)",
            "constraint_names": list(CONSTRAINT_NAMES),
            "constraint_semantics": (
                "nonnegative SIMSOPT hinge-penalty objectives; zero means the "
                "underlying engineering inequality is feasible"
            ),
            "al_coordinate_formula": (
                "c_hat_i = q(c_i) / scale_i; f - lambda^T c_hat + "
                "0.5 sum(mu_i c_hat_i^2)"
            ),
            "reference_adaptation": (
                "attached auglag_qa.py convention, with flux plus length retained "
                "as f and deterministic zero initial multipliers"
            ),
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
        },
        "gpu_configuration": {
            "target_tile_size": config.target_tile_size,
            "source_tile_size": config.source_tile_size,
            "vjp_mode": config.vjp_mode,
            "compilation_seconds": compilation_seconds,
        },
        "coordinate_scaling": {
            "current_scale_amperes": args.current_scale,
            "curve_scale": 1.0,
        },
        "constraint_scaling": {
            "names": list(CONSTRAINT_NAMES),
            "scales": constraint_scales.tolist(),
            "convention": "al_constraint_i = q(raw_constraint_i) / scale_i",
        },
        "constraint_mapping": {
            "transform": args.constraint_transform,
            "transform_epsilon": args.constraint_transform_epsilon,
            "automatic_gradient_balancing": args.automatic_constraint_scaling,
            "gradient_balance_calibration": gradient_balance,
        },
        "initial_state": {
            "source": "canonical_problem"
            if args.initial_variables is None
            else str(args.initial_variables),
            "physical_variables": physical_initial_x.tolist(),
        },
        "initial_parity": initial_parity,
        "cpu": {
            "optimization": result_summary(cpu_result, coordinate_bridge),
            "final_metrics": cpu_metrics,
        },
        "gpu": {
            "optimization": result_summary(gpu_result, coordinate_bridge),
            "final_metrics": gpu_metrics,
        },
        "comparison": {
            "optimization_speedup": speedup,
            "amortized_speedup": cpu_result.seconds
            / (gpu_result.seconds + compilation_seconds),
            "maximum_oracle_metric_relative_error": max(metric_errors.values()),
            "oracle_metric_relative_errors": metric_errors,
        },
        "feasibility_tolerances": feasibility_tolerances,
        "acceptance_gates": gates,
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
