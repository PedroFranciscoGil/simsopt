"""Sweep SciPy L-BFGS-B memory and line-search settings on both backends."""

import argparse
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

from problems import PROBLEMS

ROOT = Path(__file__).resolve().parents[2]
COMPARISON = ROOT / "benchmarks" / "gpu" / "compare_scipy_trajectories.py"


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
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument(
        "--maxcor-values", type=positive_int_list, default=[10, 30, 100, 300]
    )
    parser.add_argument("--maxls-values", type=positive_int_list, default=[20, 50])
    parser.add_argument("--ftol", type=float)
    parser.add_argument("--gtol", type=float)
    parser.add_argument("--current-scale", type=float, default=100000.0)
    parser.add_argument("--constraint-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--target-tile-size", type=int, default=1024)
    parser.add_argument("--source-tile-size", type=int, default=4320)
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
    if args.maxiter < 1:
        parser.error("maxiter must be positive")
    if args.target_tile_size < 1 or args.source_tile_size < 1:
        parser.error("tile sizes must be positive")
    for name in ("current_scale", "constraint_weight_multiplier"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name.replace('_', '-')} must be finite and positive")
    for name in (
        "distance_feasibility_tolerance",
        "curvature_feasibility_tolerance",
        "mean_squared_curvature_feasibility_tolerance",
        "ftol",
        "gtol",
    ):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"{name.replace('_', '-')} must be finite and nonnegative")
    args.output_dir = args.output_dir.resolve()
    return args


def feasibility_ratios(result):
    comparisons = result["gates"]["absolute_engineering_feasibility"]["measured"]
    ratios = []
    for backend in ("cpu", "gpu"):
        for comparison in comparisons[backend].values():
            tolerance = comparison["tolerance"]
            violation = comparison["violation"]
            ratios.append(
                violation / tolerance
                if tolerance
                else 0.0
                if violation == 0.0
                else math.inf
            )
    return ratios


def candidate_rank(result):
    """Lexicographic engineering/solver rank; smaller is better."""
    ratios = feasibility_ratios(result)
    failed = sum(ratio > 1.0 for ratio in ratios)
    cpu_metrics = result["cpu"]["final_metrics"]
    gpu_metrics = result["gpu"]["final_metrics_from_cpu_oracle"]
    gradients = (
        result["cpu"]["optimization"]["final_gradient_norm"],
        result["gpu"]["optimization"]["final_gradient_norm"],
    )
    normal_fields = (
        cpu_metrics["normalized_normal_field"]["root_mean_square"],
        gpu_metrics["normalized_normal_field"]["root_mean_square"],
    )
    evaluations = (
        result["cpu"]["optimization"]["evaluations"]
        + result["gpu"]["optimization"]["evaluations"]
    )
    return (
        failed,
        max(ratios, default=0.0),
        sum(ratios),
        max(gradients),
        max(normal_fields),
        evaluations,
    )


def comparison_command(args, maxcor, maxls, output):
    command = [
        sys.executable,
        str(COMPARISON),
        "--problem",
        args.problem,
        "--maxiter",
        str(args.maxiter),
        "--trajectory-parity-iterations",
        str(min(25, args.maxiter)),
        "--maxcor",
        str(maxcor),
        "--maxls",
        str(maxls),
        "--current-scale",
        str(args.current_scale),
        "--constraint-weight-multiplier",
        str(args.constraint_weight_multiplier),
        "--target-tile-size",
        str(args.target_tile_size),
        "--source-tile-size",
        str(args.source_tile_size),
        "--distance-feasibility-tolerance",
        str(args.distance_feasibility_tolerance),
        "--curvature-feasibility-tolerance",
        str(args.curvature_feasibility_tolerance),
        "--mean-squared-curvature-feasibility-tolerance",
        str(args.mean_squared_curvature_feasibility_tolerance),
        "--no-visualization",
        "--output",
        str(output),
    ]
    if args.ftol is not None:
        command.extend(("--ftol", str(args.ftol)))
    if args.gtol is not None:
        command.extend(("--gtol", str(args.gtol)))
    if args.allow_non_gpu:
        command.append("--allow-non-gpu")
    return command


def compact_candidate(result, result_file, maxcor, maxls):
    optimization_names = (
        "seconds",
        "success",
        "status",
        "message",
        "iterations",
        "evaluations",
        "final_objective",
        "final_gradient_norm",
    )
    return {
        "maxcor": maxcor,
        "maxls": maxls,
        "result_file": result_file.name,
        "rank": list(candidate_rank(result)),
        "initial_parity_passed": result["gates"]["initial_float64_parity"]["passed"],
        "early_trajectory_parity_passed": result["gates"]["early_trajectory_parity"][
            "passed"
        ],
        "absolute_engineering_feasibility_passed": result["gates"][
            "absolute_engineering_feasibility"
        ]["passed"],
        "stationary_convergence_passed": result["gates"]["stationary_convergence"][
            "passed"
        ],
        "cpu": {
            "optimization": {
                name: result["cpu"]["optimization"][name] for name in optimization_names
            },
            "final_metrics": result["cpu"]["final_metrics"],
        },
        "gpu": {
            "optimization": {
                name: result["gpu"]["optimization"][name] for name in optimization_names
            },
            "final_metrics_from_cpu_oracle": result["gpu"][
                "final_metrics_from_cpu_oracle"
            ],
        },
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    results = []
    for maxcor, maxls in itertools.product(args.maxcor_values, args.maxls_values):
        result_file = args.output_dir / f"maxcor-{maxcor}-maxls-{maxls}.json"
        command = comparison_command(args, maxcor, maxls, result_file)
        print(
            f"Running solver candidate maxcor={maxcor}, maxls={maxls}...",
            flush=True,
        )
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        result = json.loads(result_file.read_text())
        if result.get("schema_version") != 4:
            raise RuntimeError("comparison runner did not produce schema version 4")
        candidate = compact_candidate(result, result_file, maxcor, maxls)
        print(
            f"Candidate complete; rank={candidate['rank']}, "
            f"parity={candidate['initial_parity_passed'] and candidate['early_trajectory_parity_passed']}",
            flush=True,
        )
        candidates.append(candidate)
        results.append(result)

    eligible = [
        index
        for index, candidate in enumerate(candidates)
        if candidate["initial_parity_passed"]
        and candidate["early_trajectory_parity_passed"]
    ]
    if not eligible:
        raise RuntimeError("no solver candidate passed the backend parity gates")
    winner_index = min(eligible, key=lambda index: candidate_rank(results[index]))
    winner = candidates[winner_index]
    summary = {
        "schema_version": 1,
        "workflow": "scipy_lbfgsb_feasibility_sweep",
        "configuration": {
            "problem": args.problem,
            "maxiter": args.maxiter,
            "maxcor_values": args.maxcor_values,
            "maxls_values": args.maxls_values,
            "ftol": args.ftol,
            "gtol": args.gtol,
            "current_scale": args.current_scale,
            "constraint_weight_multiplier": args.constraint_weight_multiplier,
            "distance_feasibility_tolerance": args.distance_feasibility_tolerance,
            "curvature_feasibility_tolerance": args.curvature_feasibility_tolerance,
            "mean_squared_curvature_feasibility_tolerance": (
                args.mean_squared_curvature_feasibility_tolerance
            ),
        },
        "ranking": [
            "failed_absolute_constraints",
            "maximum_normalized_violation",
            "sum_normalized_violation",
            "maximum_final_gradient_norm",
            "maximum_final_normal_field_rms",
            "combined_function_evaluations",
        ],
        "winner": {
            "index": winner_index,
            "maxcor": winner["maxcor"],
            "maxls": winner["maxls"],
            "result_file": winner["result_file"],
            "rank": winner["rank"],
        },
        "candidates": candidates,
        "environment": results[winner_index]["environment"],
        "nvidia_smi": results[winner_index]["nvidia_smi"],
    }
    summary_file = args.output_dir / "solver-feasibility-sweep.json"
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    summary_file.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
