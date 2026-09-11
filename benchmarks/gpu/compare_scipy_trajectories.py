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
from optimization_metrics import final_coil_metrics, flatten_numeric_metrics
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
    parser.add_argument(
        "--current-scale",
        type=float,
        default=1.0,
        help="Physical amperes represented by one free-current coordinate unit.",
    )
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
    if not np.isfinite(args.current_scale) or args.current_scale <= 0:
        parser.error("current-scale must be finite and positive")
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
        "maximum_iteration_current_relative_l2_error": max(current_errors, default=0.0),
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
    gpu_values = flatten_numeric_metrics(gpu_metrics)
    cpu_values = flatten_numeric_metrics(cpu_metrics)
    if gpu_values.keys() != cpu_values.keys():
        raise ValueError("CPU and GPU final metric schemas do not match")
    errors = {}
    for name, expected in cpu_values.items():
        actual = gpu_values[name]
        errors[name] = relative_error(actual, expected, floor=1e-12)
    return errors


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
    gpu_bridge = ScipyCoilObjectiveBridge(
        data,
        free_current_indices=free_current_indices,
        objective_kwargs=objective_call_kwargs(objective_settings),
        current_scale=args.current_scale,
        config=config,
    )
    physical_initial_x = cpu_objective.x.copy()
    initial_x = gpu_bridge.initial_x.copy()
    if physical_initial_x.shape != initial_x.shape or not np.allclose(
        physical_initial_x,
        gpu_bridge.to_physical_variables(initial_x),
        rtol=1e-14,
        atol=0.0,
    ):
        raise RuntimeError("GPU flat variables do not match SIMSOPT's dof ordering")

    compile_start = time.perf_counter()
    gpu_bridge.compile()
    compilation_seconds = time.perf_counter() - compile_start

    cpu_objective.x = physical_initial_x
    initial_cpu_value = float(cpu_objective.J())
    initial_cpu_gradient = gpu_bridge.pullback_gradient(cpu_objective.dJ())
    initial_gpu_value, initial_gpu_gradient = gpu_bridge(initial_x)
    initial_parity = {
        "value_absolute_error": abs(initial_gpu_value - initial_cpu_value),
        "gradient_relative_l2_error": relative_error(
            initial_gpu_gradient, initial_cpu_gradient
        ),
    }

    def cpu_function(x):
        cpu_objective.x = gpu_bridge.to_physical_variables(x)
        return cpu_objective.J(), gpu_bridge.pullback_gradient(cpu_objective.dJ())

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
    cpu_metrics = final_coil_metrics(
        cpu_objective,
        components,
        base_curves,
        field,
        surface,
        gpu_bridge.to_physical_variables(cpu_result.x),
        objective_settings,
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
    gpu_metrics = final_coil_metrics(
        cpu_objective,
        components,
        base_curves,
        field,
        surface,
        gpu_bridge.to_physical_variables(gpu_result.x),
        objective_settings,
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
        "stationary_convergence": gate(
            {
                "cpu_success": bool(cpu_result.success),
                "gpu_success": bool(gpu_result.success),
                "cpu_gradient_norm": float(np.linalg.norm(cpu_result.jac)),
                "gpu_gradient_norm": float(np.linalg.norm(gpu_result.jac)),
            },
            {"both_success": True, "maximum_gradient_norm": 1e-5},
            cpu_result.success
            and gpu_result.success
            and np.linalg.norm(cpu_result.jac) <= 1e-5
            and np.linalg.norm(gpu_result.jac) <= 1e-5,
        ),
        "optimization_speedup": gate(
            optimization_speedup, 3.0, optimization_speedup >= 3.0
        ),
    }

    result = {
        "schema_version": 2,
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
        "coordinate_scaling": {
            "current_scale_amperes": args.current_scale,
            "curve_scale": 1.0,
            "convention": "physical_variables = optimizer_variables * scales",
        },
        "initial_parity": initial_parity,
        "cpu": {
            "optimization": optimization_summary(cpu_result, cpu_recorder, cpu_seconds),
            "final_metrics": cpu_metrics,
        },
        "gpu": {
            "optimization": optimization_summary(gpu_result, gpu_recorder, gpu_seconds),
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
