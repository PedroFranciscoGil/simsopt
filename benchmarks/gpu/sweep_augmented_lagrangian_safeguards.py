"""Qualify safeguarded AL conditioning choices before production execution."""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from problems import PROBLEMS

QUALIFICATION_GATES = (
    "gpu_backend",
    "initial_float64_parity",
    "absolute_engineering_feasibility",
    "stationary_convergence",
    "final_normal_field_quality",
    "final_constraint_quality",
)

CANDIDATES = (
    {
        "name": "identity-strict",
        "constraint_transform": "identity",
        "transform_epsilon": 1e-4,
        "automatic_scaling": False,
        "maximum_gradient_ratio": 1.0,
        "isolates": "strict inner-stationarity safeguard reference",
    },
    {
        "name": "smooth-sqrt-1e-3",
        "constraint_transform": "smooth_sqrt",
        "transform_epsilon": 1e-3,
        "automatic_scaling": True,
        "maximum_gradient_ratio": 1.0,
        "isolates": "residual-like AL coordinates with automatic attenuation",
    },
    {
        "name": "smooth-sqrt-1e-4",
        "constraint_transform": "smooth_sqrt",
        "transform_epsilon": 1e-4,
        "automatic_scaling": True,
        "maximum_gradient_ratio": 1.0,
        "isolates": "smaller smoothing region",
    },
    {
        "name": "smooth-sqrt-1e-4-quarter-gradient",
        "constraint_transform": "smooth_sqrt",
        "transform_epsilon": 1e-4,
        "automatic_scaling": True,
        "maximum_gradient_ratio": 0.25,
        "isolates": "stronger initial constraint-gradient attenuation",
    },
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen-problem", choices=sorted(PROBLEMS), default="engineering"
    )
    parser.add_argument("--final-problem", choices=sorted(PROBLEMS), default="stress")
    parser.add_argument("--max-outer-iterations", type=int, default=10)
    parser.add_argument("--max-inner-iterations", type=int, default=200)
    parser.add_argument("--mu-init", type=float, default=4.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--mu-max", type=float, default=1e6)
    parser.add_argument("--maxcor", type=int, default=100)
    parser.add_argument("--maxls", type=int, default=50)
    parser.add_argument("--current-scale", type=float, default=100000.0)
    parser.add_argument("--target-tile-size", type=int, default=1024)
    parser.add_argument("--source-tile-size", type=int, default=4320)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-8)
    parser.add_argument("--constraint-tolerance", type=float, default=1e-8)
    parser.add_argument("--inner-stationarity-factor", type=float, default=1.0)
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
    for name in (
        "max_outer_iterations",
        "max_inner_iterations",
        "maxcor",
        "maxls",
        "target_tile_size",
        "source_tile_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    for name in (
        "mu_init",
        "tau",
        "mu_max",
        "current_scale",
        "gradient_tolerance",
        "constraint_tolerance",
        "inner_stationarity_factor",
        "distance_feasibility_tolerance",
        "curvature_feasibility_tolerance",
        "mean_squared_curvature_feasibility_tolerance",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name.replace('_', '-')} must be finite and positive")
    if args.mu_init <= 1 or args.tau <= 1 or args.mu_max < args.mu_init:
        parser.error("require mu-init > 1, tau > 1, and mu-max >= mu-init")
    if args.inner_stationarity_factor < 1:
        parser.error("inner-stationarity-factor must be at least one")
    return args


def benchmark_command(args, problem, candidate, output):
    command = [
        sys.executable,
        str(Path(__file__).with_name("benchmark_augmented_lagrangian.py")),
        "--problem",
        problem,
        "--max-outer-iterations",
        str(args.max_outer_iterations),
        "--max-inner-iterations",
        str(args.max_inner_iterations),
        "--mu-init",
        str(args.mu_init),
        "--tau",
        str(args.tau),
        "--mu-max",
        str(args.mu_max),
        "--constraint-transform",
        candidate["constraint_transform"],
        "--constraint-transform-epsilon",
        str(candidate["transform_epsilon"]),
        "--maximum-initial-constraint-gradient-ratio",
        str(candidate["maximum_gradient_ratio"]),
        "--require-inner-stationarity",
        "--inner-stationarity-factor",
        str(args.inner_stationarity_factor),
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
    if candidate["automatic_scaling"]:
        command.append("--automatic-constraint-scaling")
    if args.allow_non_gpu:
        command.append("--allow-non-gpu")
    return command


def qualifies(result):
    """Return true only when all scientific promotion gates pass."""
    gates = result["acceptance_gates"]
    return all(gates[name]["passed"] for name in QUALIFICATION_GATES)


def feasibility_ratios(result):
    tolerances = result["feasibility_tolerances"]
    return {
        backend: {
            name: value / tolerances[name]
            for name, value in result[backend]["final_metrics"]["coil_constraints"][
                "violations"
            ].items()
        }
        for backend in ("cpu", "gpu")
    }


def diagnostic_rank(result):
    """Order diagnostics without weakening the binary qualification rule."""
    ratios = feasibility_ratios(result)
    values = [value for backend in ratios.values() for value in backend.values()]
    failed_gates = sum(
        not result["acceptance_gates"][name]["passed"] for name in QUALIFICATION_GATES
    )
    safeguards = sum(
        result[backend]["optimization"]["terminated_by_inner_safeguard"]
        for backend in ("cpu", "gpu")
    )
    maximum_gradient = max(
        result[backend]["optimization"]["final_gradient_norm"]
        for backend in ("cpu", "gpu")
    )
    mean_rms = (
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
        failed_gates,
        sum(value > 1.0 for value in values),
        max(values),
        safeguards,
        maximum_gradient,
        mean_rms,
        evaluations,
    )


def compact_candidate(index, candidate, result, result_file):
    def optimization_snapshot(backend):
        optimization = result[backend]["optimization"]
        return {
            key: optimization[key]
            for key in (
                "success",
                "message",
                "seconds",
                "outer_iterations",
                "total_inner_iterations",
                "total_evaluations",
                "final_base_objective",
                "final_gradient_norm",
                "terminated_by_inner_safeguard",
                "final_raw_constraint_norm_infinity",
                "final_al_constraint_norm_infinity",
                "final_constraints",
                "final_al_constraints",
                "final_penalties",
            )
        }

    return {
        "index": index,
        "configuration": candidate,
        "result_file": result_file.name,
        "qualified": qualifies(result),
        "qualification_gates": {
            name: result["acceptance_gates"][name]["passed"]
            for name in QUALIFICATION_GATES
        },
        "diagnostic_rank": list(diagnostic_rank(result)),
        "feasibility_ratios": feasibility_ratios(result),
        "constraint_mapping": result["constraint_mapping"],
        "cpu_optimization": optimization_snapshot("cpu"),
        "gpu_optimization": optimization_snapshot("gpu"),
        "cpu_final_metrics": result["cpu"]["final_metrics"],
        "gpu_final_metrics": result["gpu"]["final_metrics"],
        "visualizations": result["visualizations"],
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    results = []
    candidates = []
    for index, candidate in enumerate(CANDIDATES):
        result_file = args.output_dir / f"screen-{index:02d}-{candidate['name']}.json"
        subprocess.run(
            benchmark_command(args, args.screen_problem, candidate, result_file),
            check=True,
            env=environment,
        )
        result = json.loads(result_file.read_text())
        if result.get("schema_version") != 7:
            raise ValueError("safeguard screen requires comparison schema 7")
        results.append(result)
        candidates.append(compact_candidate(index, candidate, result, result_file))

    qualified = [index for index, result in enumerate(results) if qualifies(result)]
    diagnostic_ordering = sorted(
        range(len(results)), key=lambda index: diagnostic_rank(results[index])
    )
    winner_index = (
        min(qualified, key=lambda index: diagnostic_rank(results[index]))
        if qualified
        else None
    )
    final_file = None
    final_result = None
    if winner_index is not None:
        final_file = args.output_dir / f"{args.final_problem}-qualified-winner.json"
        subprocess.run(
            benchmark_command(
                args, args.final_problem, CANDIDATES[winner_index], final_file
            ),
            check=True,
            env=environment,
        )
        final_result = json.loads(final_file.read_text())
        if final_result.get("schema_version") != 7:
            raise ValueError("production confirmation requires comparison schema 7")

    summary = {
        "schema_version": 1,
        "workflow": "augmented_lagrangian_safeguard_qualification",
        "screen_problem": args.screen_problem,
        "final_problem": args.final_problem,
        "policy": {
            "qualification_gates": list(QUALIFICATION_GATES),
            "production_requires_qualified_screen": True,
            "screen_visualizations_exported": True,
            "diagnostic_ranking_cannot_promote_an_unqualified_candidate": True,
        },
        "candidates": candidates,
        "qualified_indices": qualified,
        "diagnostic_ordering": diagnostic_ordering,
        "winner": None
        if winner_index is None
        else {"index": winner_index, "configuration": CANDIDATES[winner_index]},
        "production": {
            "executed": final_result is not None,
            "result_file": None if final_file is None else final_file.name,
            "accepted": False if final_result is None else qualifies(final_result),
            "acceptance_gates": None
            if final_result is None
            else final_result["acceptance_gates"],
            "skip_reason": (
                "no screening candidate passed every scientific qualification gate"
                if final_result is None
                else None
            ),
        },
        "environment": results[0]["environment"],
        "nvidia_smi": results[0]["nvidia_smi"],
    }
    summary_file = args.output_dir / "safeguard-qualification-summary.json"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
