"""Screen AL conditioning choices and run the winner at production scale."""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from problems import PROBLEMS

CONSTRAINT_NAMES = (
    "coil_coil_distance_penalty",
    "coil_surface_distance_penalty",
    "curvature_penalty",
    "mean_squared_curvature_penalty",
)

# The measured scales are rounded maxima observed across the first production
# AL history.  They make a representative active value O(1); they are recorded
# as data-derived candidates, not silently adopted as universal constants.
MEASURED_STRESS_SCALES = (5e-4, 2e-2, 6e-4, 7e-4)

CANDIDATES = (
    {
        "name": "baseline",
        "max_outer_iterations": 6,
        "max_inner_iterations": 50,
        "mu_init": 10.0,
        "tau": 10.0,
        "mu_max": 1e12,
        "constraint_scales": (1.0, 1.0, 1.0, 1.0),
        "isolates": "short-inner aggressive-growth reference",
    },
    {
        "name": "long-inner",
        "max_outer_iterations": 6,
        "max_inner_iterations": 150,
        "mu_init": 10.0,
        "tau": 10.0,
        "mu_max": 1e12,
        "constraint_scales": (1.0, 1.0, 1.0, 1.0),
        "isolates": "inner iteration budget",
    },
    {
        "name": "gentle-growth",
        "max_outer_iterations": 8,
        "max_inner_iterations": 150,
        "mu_init": 4.0,
        "tau": 2.0,
        "mu_max": 1e6,
        "constraint_scales": (1.0, 1.0, 1.0, 1.0),
        "isolates": "slower penalty growth and lower cap",
    },
    {
        "name": "scaled-gentle-growth",
        "max_outer_iterations": 8,
        "max_inner_iterations": 150,
        "mu_init": 4.0,
        "tau": 2.0,
        "mu_max": 1e6,
        "constraint_scales": MEASURED_STRESS_SCALES,
        "isolates": "family scaling relative to gentle-growth",
    },
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen-problem", choices=sorted(PROBLEMS), default="engineering"
    )
    parser.add_argument("--final-problem", choices=sorted(PROBLEMS), default="stress")
    parser.add_argument("--maxcor", type=int, default=100)
    parser.add_argument("--maxls", type=int, default=50)
    parser.add_argument("--current-scale", type=float, default=100000.0)
    parser.add_argument("--target-tile-size", type=int, default=1024)
    parser.add_argument("--source-tile-size", type=int, default=4320)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-8)
    parser.add_argument("--constraint-tolerance", type=float, default=1e-8)
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
    if args.screen_problem == args.final_problem:
        parser.error("screen-problem and final-problem must differ")
    if args.maxcor < 1 or args.maxls < 1:
        parser.error("maxcor and maxls must be positive")
    for name in (
        "current_scale",
        "gradient_tolerance",
        "constraint_tolerance",
        "distance_feasibility_tolerance",
        "curvature_feasibility_tolerance",
        "mean_squared_curvature_feasibility_tolerance",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name.replace('_', '-')} must be finite and positive")
    if args.target_tile_size < 1 or args.source_tile_size < 1:
        parser.error("tile sizes must be positive")
    return args


def constraint_scale_argument(scales):
    return ",".join(f"{value:.17g}" for value in scales)


def benchmark_command(args, problem, candidate, output, *, final):
    command = [
        sys.executable,
        str(Path(__file__).with_name("benchmark_augmented_lagrangian.py")),
        "--problem",
        problem,
        "--max-outer-iterations",
        str(candidate["max_outer_iterations"]),
        "--max-inner-iterations",
        str(candidate["max_inner_iterations"]),
        "--mu-init",
        str(candidate["mu_init"]),
        "--tau",
        str(candidate["tau"]),
        "--mu-max",
        str(candidate["mu_max"]),
        "--constraint-scales",
        constraint_scale_argument(candidate["constraint_scales"]),
        "--gradient-tolerance",
        str(args.gradient_tolerance),
        "--constraint-tolerance",
        str(args.constraint_tolerance),
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
        "--distance-feasibility-tolerance",
        str(args.distance_feasibility_tolerance),
        "--curvature-feasibility-tolerance",
        str(args.curvature_feasibility_tolerance),
        "--mean-squared-curvature-feasibility-tolerance",
        str(args.mean_squared_curvature_feasibility_tolerance),
        "--output",
        str(output),
    ]
    if not final:
        command.append("--no-visualization")
    if args.allow_non_gpu:
        command.append("--allow-non-gpu")
    return command


def feasibility_ratios(result, backend):
    violations = result[backend]["final_metrics"]["coil_constraints"]["violations"]
    tolerances = result["feasibility_tolerances"]
    return {name: violations[name] / tolerances[name] for name in tolerances}


def candidate_rank(result):
    """Rank physical feasibility first, then stationarity and field quality."""
    gates = result["acceptance_gates"]
    if not (
        gates["gpu_backend"]["passed"] and gates["initial_float64_parity"]["passed"]
    ):
        return (math.inf,) * 7
    ratios = {
        backend: feasibility_ratios(result, backend) for backend in ("cpu", "gpu")
    }
    all_ratios = [value for backend in ratios.values() for value in backend.values()]
    failed_constraints = sum(value > 1.0 for value in all_ratios)
    stationary_failures = sum(
        not result[backend]["optimization"]["success"] for backend in ("cpu", "gpu")
    )
    maximum_gradient = max(
        result[backend]["optimization"]["final_gradient_norm"]
        for backend in ("cpu", "gpu")
    )
    mean_rms_field = (
        sum(
            result[backend]["final_metrics"]["normalized_normal_field"][
                "root_mean_square"
            ]
            for backend in ("cpu", "gpu")
        )
        / 2.0
    )
    evaluations = sum(
        result[backend]["optimization"]["total_evaluations"]
        for backend in ("cpu", "gpu")
    )
    return (
        failed_constraints,
        max(all_ratios),
        sum(all_ratios),
        stationary_failures,
        maximum_gradient,
        mean_rms_field,
        evaluations,
    )


def compact_candidate(index, configuration, result, result_file):
    def optimization_snapshot(backend):
        optimization = result[backend]["optimization"]
        return {
            key: optimization[key]
            for key in (
                "success",
                "seconds",
                "outer_iterations",
                "total_inner_iterations",
                "total_evaluations",
                "final_base_objective",
                "final_gradient_norm",
                "final_raw_constraint_norm_infinity",
                "final_scaled_constraint_norm_infinity",
                "final_constraints",
                "final_scaled_constraints",
                "final_penalties",
            )
        }

    return {
        "index": index,
        "configuration": {
            key: list(value) if key == "constraint_scales" else value
            for key, value in configuration.items()
        },
        "result_file": result_file.name,
        "rank": list(candidate_rank(result)),
        "initial_parity_passed": result["acceptance_gates"]["initial_float64_parity"][
            "passed"
        ],
        "absolute_feasibility_passed": result["acceptance_gates"][
            "absolute_engineering_feasibility"
        ]["passed"],
        "stationarity_passed": result["acceptance_gates"]["stationary_convergence"][
            "passed"
        ],
        "speedup": result["comparison"]["optimization_speedup"],
        "cpu_final_metrics": result["cpu"]["final_metrics"],
        "gpu_final_metrics": result["gpu"]["final_metrics"],
        "cpu_final_optimization": optimization_snapshot("cpu"),
        "gpu_final_optimization": optimization_snapshot("gpu"),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    candidate_summaries = []
    candidate_results = []
    for index, candidate in enumerate(CANDIDATES):
        result_file = args.output_dir / f"screen-{index:02d}-{candidate['name']}.json"
        subprocess.run(
            benchmark_command(
                args,
                args.screen_problem,
                candidate,
                result_file,
                final=False,
            ),
            check=True,
            env=environment,
        )
        result = json.loads(result_file.read_text())
        if result.get("schema_version") != 6:
            raise ValueError("conditioning screen requires comparison schema 6")
        candidate_results.append(result)
        candidate_summaries.append(
            compact_candidate(index, candidate, result, result_file)
        )

    ordering = sorted(
        range(len(candidate_results)),
        key=lambda index: candidate_rank(candidate_results[index]),
    )
    winner_index = ordering[0]
    winner = CANDIDATES[winner_index]
    final_file = args.output_dir / f"{args.final_problem}-conditioning-winner.json"
    subprocess.run(
        benchmark_command(args, args.final_problem, winner, final_file, final=True),
        check=True,
        env=environment,
    )
    final_result = json.loads(final_file.read_text())
    if final_result.get("schema_version") != 6:
        raise ValueError("production confirmation requires comparison schema 6")

    summary = {
        "schema_version": 1,
        "workflow": "augmented_lagrangian_conditioning_study",
        "screen_problem": args.screen_problem,
        "final_problem": args.final_problem,
        "candidate_design": {
            "principle": (
                "baseline, then an isolated inner-budget change, a grouped "
                "growth/cap change, and an isolated scaling change"
            ),
            "measured_scale_source": (
                "rounded per-family maximum raw constraints in archived c7d522f6 "
                "stress AL history"
            ),
            "constraint_order": list(CONSTRAINT_NAMES),
        },
        "candidates": candidate_summaries,
        "ranking": ordering,
        "winner": {
            "index": winner_index,
            "configuration": {
                key: list(value) if key == "constraint_scales" else value
                for key, value in winner.items()
            },
        },
        "final_result_file": final_file.name,
        "final_acceptance_gates": final_result["acceptance_gates"],
        "accepted": all(
            gate["passed"] for gate in final_result["acceptance_gates"].values()
        ),
        "environment": final_result["environment"],
        "nvidia_smi": final_result["nvidia_smi"],
    }
    summary_file = args.output_dir / "conditioning-study-summary.json"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
