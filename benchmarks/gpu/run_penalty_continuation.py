"""Run synchronized CPU/GPU engineering-penalty continuation stages."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from problems import PROBLEMS

ROOT = Path(__file__).resolve().parents[2]
COMPARISON = ROOT / "benchmarks" / "gpu" / "compare_scipy_trajectories.py"


def positive_float_list(value):
    try:
        values = [float(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated floating-point values"
        ) from error
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be finite and positive")
    return values


def positive_int_list(value):
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integer values"
        ) from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), default="stress")
    parser.add_argument(
        "--penalty-multipliers", type=positive_float_list, default=[1.0, 10.0, 100.0]
    )
    parser.add_argument(
        "--stage-maxiters", type=positive_int_list, default=[100, 100, 200]
    )
    parser.add_argument("--maxcor", type=int, default=30)
    parser.add_argument("--maxls", type=int, default=50)
    parser.add_argument("--ftol", type=float)
    parser.add_argument("--gtol", type=float)
    parser.add_argument("--current-scale", type=float, default=100000.0)
    parser.add_argument("--target-tile-size", type=int, default=1024)
    parser.add_argument("--source-tile-size", type=int, default=4320)
    parser.add_argument(
        "--seed-backend",
        choices=("best", "cpu", "gpu"),
        default="best",
        help="Common state used to start the next stage.",
    )
    parser.add_argument("--distance-feasibility-tolerance", type=float, default=1e-4)
    parser.add_argument("--curvature-feasibility-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--mean-squared-curvature-feasibility-tolerance",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-non-gpu", action="store_true")
    args = parser.parse_args()
    if len(args.stage_maxiters) == 1:
        args.stage_maxiters *= len(args.penalty_multipliers)
    if len(args.stage_maxiters) != len(args.penalty_multipliers):
        parser.error("stage-maxiters must contain one value or one per penalty stage")
    if args.maxcor < 1 or args.maxls < 1:
        parser.error("maxcor and maxls must be positive")
    if args.target_tile_size < 1 or args.source_tile_size < 1:
        parser.error("tile sizes must be positive")
    if not math.isfinite(args.current_scale) or args.current_scale <= 0:
        parser.error("current-scale must be finite and positive")
    for name in (
        "distance_feasibility_tolerance",
        "curvature_feasibility_tolerance",
        "mean_squared_curvature_feasibility_tolerance",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"{name.replace('_', '-')} must be finite and nonnegative")
    for name in ("ftol", "gtol"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"{name} must be finite and nonnegative")
    args.output_dir = args.output_dir.resolve()
    return args


def backend_metrics(result, backend):
    if backend == "cpu":
        return result["cpu"]["final_metrics"]
    return result["gpu"]["final_metrics_from_cpu_oracle"]


def backend_rank(result, backend):
    """Rank a restart state by absolute feasibility, then field quality/objective."""
    comparisons = result["gates"]["absolute_engineering_feasibility"]["measured"][
        backend
    ]
    ratios = []
    for comparison in comparisons.values():
        tolerance = comparison["tolerance"]
        violation = comparison["violation"]
        ratios.append(
            violation / tolerance
            if tolerance
            else 0.0
            if violation == 0.0
            else math.inf
        )
    metrics = backend_metrics(result, backend)
    return (
        sum(ratio > 1.0 for ratio in ratios),
        max(ratios, default=0.0),
        sum(ratios),
        metrics["normalized_normal_field"]["root_mean_square"],
        metrics["objective"],
    )


def choose_seed_backend(result, policy):
    if policy != "best":
        return policy
    return min(("cpu", "gpu"), key=lambda backend: backend_rank(result, backend))


def comparison_command(args, multiplier, maxiter, output, initial, final_stage):
    command = [
        sys.executable,
        str(COMPARISON),
        "--problem",
        args.problem,
        "--maxiter",
        str(maxiter),
        "--trajectory-parity-iterations",
        str(min(25, maxiter)),
        "--maxcor",
        str(args.maxcor),
        "--maxls",
        str(args.maxls),
        "--current-scale",
        str(args.current_scale),
        "--target-tile-size",
        str(args.target_tile_size),
        "--source-tile-size",
        str(args.source_tile_size),
        "--constraint-weight-multiplier",
        str(multiplier),
        "--distance-feasibility-tolerance",
        str(args.distance_feasibility_tolerance),
        "--curvature-feasibility-tolerance",
        str(args.curvature_feasibility_tolerance),
        "--mean-squared-curvature-feasibility-tolerance",
        str(args.mean_squared_curvature_feasibility_tolerance),
        "--output",
        str(output),
    ]
    if args.ftol is not None:
        command.extend(("--ftol", str(args.ftol)))
    if args.gtol is not None:
        command.extend(("--gtol", str(args.gtol)))
    if initial is not None:
        command.extend(("--initial-variables", str(initial)))
    if final_stage:
        command.extend(("--visualization-dir", str(args.output_dir)))
    else:
        command.append("--no-visualization")
    if args.allow_non_gpu:
        command.append("--allow-non-gpu")
    return command


def stage_summary(result, result_file, multiplier, selected_seed):
    return {
        "penalty_multiplier": multiplier,
        "result_file": result_file.name,
        "selected_seed_backend": selected_seed,
        "cpu": {
            "optimization": {
                name: result["cpu"]["optimization"][name]
                for name in (
                    "seconds",
                    "success",
                    "status",
                    "message",
                    "iterations",
                    "evaluations",
                    "final_objective",
                    "final_gradient_norm",
                )
            },
            "final_metrics": result["cpu"]["final_metrics"],
        },
        "gpu": {
            "optimization": {
                name: result["gpu"]["optimization"][name]
                for name in (
                    "seconds",
                    "success",
                    "status",
                    "message",
                    "iterations",
                    "evaluations",
                    "final_objective",
                    "final_gradient_norm",
                )
            },
            "final_metrics_from_cpu_oracle": result["gpu"][
                "final_metrics_from_cpu_oracle"
            ],
        },
        "seed_ranks": {
            backend: list(backend_rank(result, backend)) for backend in ("cpu", "gpu")
        },
        "gates": result["gates"],
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stages = []
    initial_file = None
    cpu_seconds = 0.0
    gpu_seconds = 0.0
    compilation_seconds = 0.0
    cpu_evaluations = 0
    gpu_evaluations = 0
    final_result = None

    for index, (multiplier, maxiter) in enumerate(
        zip(args.penalty_multipliers, args.stage_maxiters, strict=True), start=1
    ):
        result_file = args.output_dir / f"stage-{index:02d}-comparison.json"
        command = comparison_command(
            args,
            multiplier,
            maxiter,
            result_file,
            initial_file,
            index == len(args.penalty_multipliers),
        )
        print(
            f"Running continuation stage {index}/{len(args.penalty_multipliers)} "
            f"(weight={multiplier:g}, maxiter={maxiter})...",
            flush=True,
        )
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        result = json.loads(result_file.read_text())
        if result.get("schema_version") != 4:
            raise RuntimeError("comparison runner did not produce schema version 4")
        selected_seed = choose_seed_backend(result, args.seed_backend)
        print(
            f"Stage {index} complete; selected {selected_seed} restart, "
            f"absolute feasibility={result['gates']['absolute_engineering_feasibility']['passed']}",
            flush=True,
        )
        stages.append(stage_summary(result, result_file, multiplier, selected_seed))
        cpu_seconds += result["cpu"]["optimization"]["seconds"]
        gpu_seconds += result["gpu"]["optimization"]["seconds"]
        compilation_seconds += result["gpu_configuration"]["compilation_seconds"]
        cpu_evaluations += result["cpu"]["optimization"]["evaluations"]
        gpu_evaluations += result["gpu"]["optimization"]["evaluations"]
        final_result = result
        if index < len(args.penalty_multipliers):
            initial_file = args.output_dir / f"stage-{index:02d}-restart.json"
            initial_file.write_text(
                json.dumps(result[selected_seed]["final_physical_variables"]) + "\n"
            )

    summary = {
        "schema_version": 1,
        "workflow": "synchronized_engineering_penalty_continuation",
        "configuration": {
            "problem": args.problem,
            "penalty_multipliers": args.penalty_multipliers,
            "stage_maxiters": args.stage_maxiters,
            "seed_backend_policy": args.seed_backend,
            "maxcor": args.maxcor,
            "maxls": args.maxls,
            "ftol": args.ftol,
            "gtol": args.gtol,
            "current_scale": args.current_scale,
            "distance_feasibility_tolerance": args.distance_feasibility_tolerance,
            "curvature_feasibility_tolerance": (args.curvature_feasibility_tolerance),
            "mean_squared_curvature_feasibility_tolerance": (
                args.mean_squared_curvature_feasibility_tolerance
            ),
        },
        "totals": {
            "cpu_seconds": cpu_seconds,
            "gpu_seconds": gpu_seconds,
            "gpu_compilation_seconds": compilation_seconds,
            "cpu_evaluations": cpu_evaluations,
            "gpu_evaluations": gpu_evaluations,
            "speedup_excluding_compilation": cpu_seconds / gpu_seconds,
            "speedup_including_compilation": cpu_seconds
            / (gpu_seconds + compilation_seconds),
        },
        "stages": stages,
        "final_result_file": stages[-1]["result_file"],
        "final_visualization": final_result["visualization"],
        "acceptance": {
            "all_stage_gpu_backend": all(
                stage["gates"]["gpu_backend"]["passed"] for stage in stages
            ),
            "all_stage_initial_parity": all(
                stage["gates"]["initial_float64_parity"]["passed"] for stage in stages
            ),
            "all_stage_early_trajectory_parity": all(
                stage["gates"]["early_trajectory_parity"]["passed"] for stage in stages
            ),
            "final_absolute_engineering_feasibility": final_result["gates"][
                "absolute_engineering_feasibility"
            ]["passed"],
            "final_stationary_convergence": final_result["gates"][
                "stationary_convergence"
            ]["passed"],
            "final_normal_field_quality": final_result["gates"][
                "final_normal_field_quality"
            ]["passed"],
            "final_physics_evaluation_budget": final_result["gates"][
                "physics_evaluation_budget"
            ]["passed"],
            "final_optimization_speedup": final_result["gates"]["optimization_speedup"][
                "passed"
            ],
            "total_optimization_speedup": cpu_seconds / gpu_seconds >= 3.0,
        },
        "environment": final_result["environment"],
        "nvidia_smi": final_result["nvidia_smi"],
    }
    summary["accepted"] = all(summary["acceptance"].values())
    summary_file = args.output_dir / "penalty-continuation-summary.json"
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    summary_file.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
