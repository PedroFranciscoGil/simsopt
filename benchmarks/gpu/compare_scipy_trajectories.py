"""Compare CPU and GPU L-BFGS-B trajectories for one engineering objective."""

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from benchmark_objective import build_problem, environment
from problems import PROBLEMS, get_problem, objective_call_kwargs, objective_metadata
from scipy.optimize import minimize
from simsopt.gpu import GpuConfig, ScipyCoilObjectiveBridge, minimal_coil_data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), default="stress")
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--maxcor", type=int, default=300)
    parser.add_argument("--target-tile-size", type=int, default=1024)
    parser.add_argument("--source-tile-size", type=int, default=4320)
    parser.add_argument("--vjp-mode", choices=("autodiff", "custom"), default="custom")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-non-gpu",
        action="store_true",
        help="Permit CPU execution for local harness validation only.",
    )
    args = parser.parse_args()
    if args.maxiter < 1 or args.maxcor < 1:
        parser.error("maxiter and maxcor must be positive")
    if args.target_tile_size < 1 or args.source_tile_size < 1:
        parser.error("tile sizes must be positive")
    return args


def relative_error(actual, expected, floor=1e-30):
    scale = max(float(np.linalg.norm(expected)), floor)
    return float(np.linalg.norm(actual - expected) / scale)


class TrajectoryRecorder:
    """Record timed evaluations and accepted L-BFGS-B iterates."""

    def __init__(self, function):
        self.function = function
        self.evaluations = []
        self._evaluation_states = []
        self.iterations = []
        self.iteration_states = []

    def __call__(self, x):
        x = np.asarray(x).copy()
        start = time.perf_counter()
        value, gradient = self.function(x)
        seconds = time.perf_counter() - start
        gradient = np.asarray(gradient)
        record = {
            "evaluation": len(self.evaluations) + 1,
            "seconds": seconds,
            "objective": float(value),
            "gradient_norm": float(np.linalg.norm(gradient)),
        }
        self.evaluations.append(record)
        self._evaluation_states.append((x, float(value), gradient.copy()))
        return float(value), gradient

    def callback(self, x):
        x = np.asarray(x).copy()
        match = next(
            (
                state
                for state in reversed(self._evaluation_states)
                if np.array_equal(state[0], x)
            ),
            None,
        )
        if match is None:
            raise RuntimeError("SciPy callback state was not an evaluated point")
        previous = self.iteration_states[-1] if self.iteration_states else None
        step_norm = 0.0 if previous is None else float(np.linalg.norm(x - previous))
        self.iteration_states.append(x)
        self.iterations.append(
            {
                "iteration": len(self.iterations) + 1,
                "objective": match[1],
                "gradient_norm": float(np.linalg.norm(match[2])),
                "step_norm": step_norm,
            }
        )


def optimization_summary(result, recorder, seconds):
    samples = [record["seconds"] for record in recorder.evaluations]
    return {
        "seconds": seconds,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "evaluations": int(result.nfev),
        "gradient_evaluations": int(result.njev),
        "final_objective": float(result.fun),
        "final_gradient_norm": float(np.linalg.norm(result.jac)),
        "evaluation_timing": {
            "median_seconds": statistics.median(samples),
            "minimum_seconds": min(samples),
            "maximum_seconds": max(samples),
            "samples_seconds": samples,
        },
        "evaluation_history": recorder.evaluations,
        "iteration_history": recorder.iterations,
    }


def oracle_metrics(objective, components, base_curves, field, surface, x):
    """Evaluate physical metrics through the existing SIMSOPT implementation."""
    objective.x = np.asarray(x)
    field_values = field.B().reshape(surface.gamma().shape)
    normal_field = np.sum(field_values * surface.unitnormal(), axis=-1)
    coil_coil = components["coil_coil_distance"]
    coil_surface = components["coil_surface_distance"]
    return {
        "objective": float(objective.J()),
        "gradient_norm": float(np.linalg.norm(objective.dJ())),
        "quadratic_flux": float(components["quadratic_flux"].J()),
        "curve_length_sum": float(components["curve_length_sum"].J()),
        "coil_coil_distance_penalty": float(coil_coil.J()),
        "coil_surface_distance_penalty": float(coil_surface.J()),
        "curvature_penalty": float(components["curvature"].J()),
        "mean_squared_curvature_penalty": float(
            components["mean_squared_curvature_penalty"].J()
        ),
        "mean_absolute_normal_field": float(np.mean(np.abs(normal_field))),
        "maximum_absolute_normal_field": float(np.max(np.abs(normal_field))),
        "minimum_coil_coil_distance": float(coil_coil.shortest_distance()),
        "minimum_coil_surface_distance": float(coil_surface.shortest_distance()),
        "maximum_curvature": float(
            max(np.max(curve.kappa()) for curve in base_curves)
        ),
    }


def compare_trajectories(cpu_result, gpu_result, cpu_recorder, gpu_recorder, nfree):
    common = min(len(cpu_recorder.iterations), len(gpu_recorder.iterations))
    objective_errors = []
    current_errors = []
    curve_errors = []
    for index in range(common):
        cpu_state = cpu_recorder.iteration_states[index]
        gpu_state = gpu_recorder.iteration_states[index]
        objective_errors.append(
            abs(
                cpu_recorder.iterations[index]["objective"]
                - gpu_recorder.iterations[index]["objective"]
            )
        )
        current_errors.append(
            relative_error(gpu_state[:nfree], cpu_state[:nfree], floor=1.0)
        )
        curve_errors.append(
            relative_error(gpu_state[nfree:], cpu_state[nfree:], floor=1.0)
        )
    return {
        "common_iterations": common,
        "same_status": int(cpu_result.status) == int(gpu_result.status),
        "same_iteration_count": int(cpu_result.nit) == int(gpu_result.nit),
        "same_evaluation_count": int(cpu_result.nfev) == int(gpu_result.nfev),
        "maximum_iteration_objective_absolute_error": max(
            objective_errors, default=0.0
        ),
        "maximum_iteration_current_relative_l2_error": max(
            current_errors, default=0.0
        ),
        "maximum_iteration_curve_relative_l2_error": max(curve_errors, default=0.0),
        "final_objective_absolute_error": abs(
            float(cpu_result.fun) - float(gpu_result.fun)
        ),
        "final_current_relative_l2_error": relative_error(
            gpu_result.x[:nfree], cpu_result.x[:nfree], floor=1.0
        ),
        "final_curve_relative_l2_error": relative_error(
            gpu_result.x[nfree:], cpu_result.x[nfree:], floor=1.0
        ),
    }


def metric_relative_errors(gpu_metrics, cpu_metrics):
    return {
        name: abs(gpu_metrics[name] - cpu_metrics[name])
        / max(abs(cpu_metrics[name]), 1e-12)
        for name in cpu_metrics
    }


def gate(measured, threshold, passed):
    return {"measured": measured, "threshold": threshold, "passed": bool(passed)}


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
    objective_settings = objective_metadata(spec, "full-engineering")
    surface, base_curves, field, components, cpu_objective = build_problem(
        spec, regularized=True
    )
    base_current_objects = [field.coils[index].current for index in range(spec.ncoils)]
    free_current_indices = np.asarray(
        [
            index
            for index, current in enumerate(base_current_objects)
            if current.x.size
        ],
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
    gpu_bridge = ScipyCoilObjectiveBridge(
        data,
        free_current_indices=free_current_indices,
        objective_kwargs=objective_call_kwargs(objective_settings),
        config=config,
    )
    initial_x = cpu_objective.x.copy()
    if initial_x.shape != gpu_bridge.initial_x.shape or not np.array_equal(
        initial_x, gpu_bridge.initial_x
    ):
        raise RuntimeError("GPU flat variables do not match SIMSOPT's dof ordering")

    compile_start = time.perf_counter()
    gpu_bridge.compile()
    compilation_seconds = time.perf_counter() - compile_start

    cpu_objective.x = initial_x
    initial_cpu_value = float(cpu_objective.J())
    initial_cpu_gradient = np.asarray(cpu_objective.dJ())
    initial_gpu_value, initial_gpu_gradient = gpu_bridge(initial_x)
    initial_parity = {
        "value_absolute_error": abs(initial_gpu_value - initial_cpu_value),
        "gradient_relative_l2_error": relative_error(
            initial_gpu_gradient, initial_cpu_gradient
        ),
    }

    def cpu_function(x):
        cpu_objective.x = x
        return cpu_objective.J(), cpu_objective.dJ()

    options = {"maxiter": args.maxiter, "maxcor": args.maxcor}
    cpu_recorder = TrajectoryRecorder(cpu_function)
    cpu_start = time.perf_counter()
    cpu_result = minimize(
        cpu_recorder,
        initial_x,
        jac=True,
        method="L-BFGS-B",
        options=options,
        tol=1e-15,
        callback=cpu_recorder.callback,
    )
    cpu_seconds = time.perf_counter() - cpu_start
    cpu_metrics = oracle_metrics(
        cpu_objective, components, base_curves, field, surface, cpu_result.x
    )

    gpu_recorder = TrajectoryRecorder(gpu_bridge)
    gpu_start = time.perf_counter()
    gpu_result = minimize(
        gpu_recorder,
        initial_x,
        jac=True,
        method="L-BFGS-B",
        options=options,
        tol=1e-15,
        callback=gpu_recorder.callback,
    )
    gpu_seconds = time.perf_counter() - gpu_start
    gpu_metrics = oracle_metrics(
        cpu_objective, components, base_curves, field, surface, gpu_result.x
    )

    comparison = compare_trajectories(
        cpu_result,
        gpu_result,
        cpu_recorder,
        gpu_recorder,
        free_current_indices.size,
    )
    errors = metric_relative_errors(gpu_metrics, cpu_metrics)
    comparison["oracle_metric_relative_errors"] = errors
    comparison["maximum_oracle_metric_relative_error"] = max(errors.values())
    optimization_speedup = cpu_seconds / gpu_seconds
    amortized_speedup = cpu_seconds / (gpu_seconds + compilation_seconds)

    gates = {
        "gpu_backend": gate(backend, "gpu", backend == "gpu"),
        "initial_float64_parity": gate(
            initial_parity,
            {"value_absolute_error": 1e-9, "gradient_relative_l2_error": 1e-7},
            initial_parity["value_absolute_error"] <= 1e-9
            and initial_parity["gradient_relative_l2_error"] <= 1e-7,
        ),
        "accepted_trajectory_parity": gate(
            {
                "objective_absolute_error": comparison[
                    "maximum_iteration_objective_absolute_error"
                ],
                "curve_relative_l2_error": comparison[
                    "maximum_iteration_curve_relative_l2_error"
                ],
            },
            {"objective_absolute_error": 1e-8, "curve_relative_l2_error": 1e-6},
            comparison["maximum_iteration_objective_absolute_error"] <= 1e-8
            and comparison["maximum_iteration_curve_relative_l2_error"] <= 1e-6,
        ),
        "final_engineering_metrics": gate(
            comparison["maximum_oracle_metric_relative_error"],
            1e-5,
            comparison["maximum_oracle_metric_relative_error"] <= 1e-5,
        ),
        "physics_evaluation_budget": gate(
            int(gpu_result.nfev),
            f"<= 1.1 * {int(cpu_result.nfev)}",
            gpu_result.nfev <= math.ceil(1.1 * cpu_result.nfev),
        ),
        "optimization_speedup": gate(
            optimization_speedup, 3.0, optimization_speedup >= 3.0
        ),
    }

    result = {
        "schema_version": 1,
        "problem": spec.as_dict(),
        "objective": objective_settings,
        "solver": {
            "method": "L-BFGS-B",
            "maxiter": args.maxiter,
            "maxcor": args.maxcor,
            "tol": 1e-15,
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
        "initial_parity": initial_parity,
        "cpu": {
            "optimization": optimization_summary(
                cpu_result, cpu_recorder, cpu_seconds
            ),
            "final_metrics": cpu_metrics,
        },
        "gpu": {
            "optimization": optimization_summary(
                gpu_result, gpu_recorder, gpu_seconds
            ),
            "final_metrics_from_cpu_oracle": gpu_metrics,
        },
        "comparison": {
            **comparison,
            "optimization_speedup_excluding_compilation": optimization_speedup,
            "optimization_speedup_including_compilation": amortized_speedup,
        },
        "gates": gates,
        "all_gates_passed": all(item["passed"] for item in gates.values()),
        "environment": environment(),
        "nvidia_smi": nvidia_smi(),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
