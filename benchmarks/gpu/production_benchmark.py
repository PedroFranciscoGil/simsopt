"""Run and assess the production-scale GPU-native core-objective benchmark."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from problems import PROBLEMS

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), default="stress")
    parser.add_argument(
        "--target-tile-sizes", default="256,512,1024,2048,4096"
    )
    parser.add_argument(
        "--source-tile-sizes", default="512,1024,2048,4096,8192"
    )
    parser.add_argument("--screening-warmup", type=int, default=2)
    parser.add_argument("--screening-repeats", type=int, default=5)
    parser.add_argument("--confirmation-warmup", type=int, default=3)
    parser.add_argument("--confirmation-repeats", type=int, default=10)
    parser.add_argument("--cpu-warmup", type=int, default=2)
    parser.add_argument("--cpu-repeats", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--trace-steps", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("simsopt-production-profile")
    )
    parser.add_argument(
        "--allow-non-gpu",
        action="store_true",
        help="Permit CPU execution for local harness validation only.",
    )
    args = parser.parse_args()
    nonnegative = (
        args.screening_warmup,
        args.confirmation_warmup,
        args.cpu_warmup,
    )
    positive = (
        args.screening_repeats,
        args.confirmation_repeats,
        args.cpu_repeats,
        args.top_k,
        args.trace_steps,
    )
    if any(value < 0 for value in nonnegative):
        parser.error("warm-up counts must be non-negative")
    if any(value < 1 for value in positive):
        parser.error("repeat, top-k, and trace-step counts must be positive")
    return args


def _gate(measured, threshold, passed):
    return {
        "measured": measured,
        "threshold": threshold,
        "passed": passed,
    }


def memory_fraction(memory_stats):
    """Return peak allocation as a fraction of the device memory limit."""
    if not memory_stats:
        return None
    peak = memory_stats.get("peak_bytes_in_use")
    limit = memory_stats.get("bytes_limit")
    if peak is None or not limit:
        return None
    return peak / limit


def production_summary(sweep, profile):
    """Combine sweep and profile results and evaluate production gates."""
    winner = sweep["winner"]
    if profile["problem"]["name"] != sweep["problem"]["name"]:
        raise ValueError("sweep and profile refer to different benchmark problems")
    if profile["objective"] != sweep["objective"]:
        raise ValueError("sweep and profile refer to different objectives")
    if profile["vjp_mode"] != winner["vjp_mode"]:
        raise ValueError("profile reverse-pass mode does not match the sweep winner")
    if (
        profile["tiles"]["target"] != winner["target_tile_size"]
        or profile["tiles"]["source"] != winner["source_tile_size"]
    ):
        raise ValueError("profile tile sizes do not match the sweep winner")

    tolerances = sweep["protocol"]["parity_tolerances"]
    parity = profile["parity"]
    parity_measured = max(
        parity["curve_gradient_relative_l2_error"],
        parity["current_gradient_relative_l2_error"],
    )
    parity_passed = (
        parity["value_absolute_error"]
        <= tolerances["value_absolute_error"]
        and parity_measured <= tolerances["gradient_relative_l2_error"]
    )
    speedup = winner["speedup_over_cpu_baseline"]
    variation = profile["steady_state"]["coefficient_of_variation"]
    allocated_fraction = memory_fraction(profile["device_memory"])
    backend = profile["environment"]["jax_backend"]

    gates = {
        "gpu_backend": _gate(backend, "gpu", backend == "gpu"),
        "float64_parity": _gate(
            {
                "value_absolute_error": parity["value_absolute_error"],
                "maximum_gradient_relative_l2_error": parity_measured,
            },
            tolerances,
            parity_passed,
        ),
        "steady_state_variation": _gate(variation, 0.05, variation <= 0.05),
        "speedup_over_one_thread_cpu": _gate(
            speedup,
            3.0,
            speedup is not None and speedup >= 3.0,
        ),
        "device_memory_fraction": _gate(
            allocated_fraction,
            0.8,
            allocated_fraction is not None and allocated_fraction <= 0.8,
        ),
    }
    return {
        "schema_version": 1,
        "problem": sweep["problem"],
        "dimensions": sweep["dimensions"],
        "objective_scope": sweep["objective"]["scope"],
        "deferred_objective_terms": sweep["objective"]["deferred_terms"],
        "winner": winner,
        "profile": {
            "compilation_seconds": profile["compilation_seconds"],
            "steady_state": profile["steady_state"],
            "device_memory": profile["device_memory"],
            "parity": parity,
        },
        "gates": gates,
        "all_gates_passed": all(gate["passed"] for gate in gates.values()),
    }


def run(command, env):
    """Run one benchmark subprocess from the repository root."""
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_file = output_dir / f"{args.problem}-tile-sweep.json"
    profile_file = output_dir / f"{args.problem}-gpu.json"
    summary_file = output_dir / f"{args.problem}-summary.json"
    trace_dir = output_dir / "trace"
    memory_file = output_dir / "device-memory.prof"

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    common_gpu_flag = ["--allow-non-gpu"] if args.allow_non_gpu else []
    run(
        [
            sys.executable,
            "benchmarks/gpu/sweep_gpu_tiles.py",
            "--problem",
            args.problem,
            "--target-tile-sizes",
            args.target_tile_sizes,
            "--source-tile-sizes",
            args.source_tile_sizes,
            "--vjp-modes",
            "custom",
            "--warmup",
            str(args.screening_warmup),
            "--repeats",
            str(args.screening_repeats),
            "--cpu-warmup",
            str(args.cpu_warmup),
            "--cpu-repeats",
            str(args.cpu_repeats),
            "--top-k",
            str(args.top_k),
            "--confirmation-warmup",
            str(args.confirmation_warmup),
            "--confirmation-repeats",
            str(args.confirmation_repeats),
            "--output",
            str(sweep_file),
            "--quiet",
            *common_gpu_flag,
        ],
        env,
    )
    sweep = json.loads(sweep_file.read_text())
    winner = sweep["winner"]
    run(
        [
            sys.executable,
            "benchmarks/gpu/profile_gpu_objective.py",
            "--problem",
            args.problem,
            "--warmup",
            str(args.confirmation_warmup),
            "--repeats",
            str(args.confirmation_repeats),
            "--target-tile-size",
            str(winner["target_tile_size"]),
            "--source-tile-size",
            str(winner["source_tile_size"]),
            "--vjp-mode",
            winner["vjp_mode"],
            "--trace-dir",
            str(trace_dir),
            "--trace-steps",
            str(args.trace_steps),
            "--memory-profile",
            str(memory_file),
            "--output",
            str(profile_file),
            *common_gpu_flag,
        ],
        env,
    )
    profile = json.loads(profile_file.read_text())
    summary = production_summary(sweep, profile)
    summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
