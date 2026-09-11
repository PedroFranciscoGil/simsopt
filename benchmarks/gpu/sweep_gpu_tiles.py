"""Autotune tile sizes and reverse passes for the GPU-native coil objective."""

import argparse
import gc
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_objective import build_problem, environment, measure
from problems import (
    OBJECTIVE_SCOPES,
    PROBLEMS,
    get_problem,
    objective_call_kwargs,
    objective_metadata,
)
from simsopt.gpu import GpuConfig, minimal_coil_data


def positive_size_list(value):
    """Parse a comma-separated, duplicate-free list of positive integers."""
    try:
        sizes = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "tile sizes must be comma-separated integers"
        ) from error
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("tile sizes must be positive")
    return sizes


def resolve_sizes(requested, dimension):
    """Clip tiles to a dimension and preserve order while removing duplicates."""
    return tuple(dict.fromkeys(min(size, dimension) for size in requested))


def vjp_mode_list(value):
    """Parse a comma-separated list of supported reverse-pass implementations."""
    modes = tuple(dict.fromkeys(item.strip() for item in value.split(",")))
    allowed = {"autodiff", "custom"}
    if not modes or any(mode not in allowed for mode in modes):
        raise argparse.ArgumentTypeError("vjp modes must be 'autodiff' and/or 'custom'")
    return modes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), default="minimal")
    parser.add_argument("--objective-scope", choices=OBJECTIVE_SCOPES, default="core")
    parser.add_argument(
        "--target-tile-sizes",
        type=positive_size_list,
        default=positive_size_list("64,128,256,512,1024"),
    )
    parser.add_argument(
        "--source-tile-sizes",
        type=positive_size_list,
        default=positive_size_list("128,256,512,1024,2048"),
    )
    parser.add_argument(
        "--vjp-modes",
        type=vjp_mode_list,
        default=vjp_mode_list("autodiff,custom"),
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--cpu-warmup",
        type=int,
        help="CPU baseline warm-ups; defaults to --warmup.",
    )
    parser.add_argument(
        "--cpu-repeats",
        type=int,
        help="CPU baseline repetitions; defaults to --repeats.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--confirmation-warmup", type=int, default=3)
    parser.add_argument("--confirmation-repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-cpu-baseline",
        action="store_true",
        help="Skip the synchronized C++/OpenMP objective-and-gradient timing.",
    )
    parser.add_argument(
        "--allow-non-gpu",
        action="store_true",
        help="Permit CPU execution for local harness validation only.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Do not print the final JSON to stdout."
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.confirmation_warmup < 0:
        parser.error("warm-up counts must be non-negative")
    if args.repeats < 1 or args.confirmation_repeats < 1 or args.top_k < 1:
        parser.error("repeat counts and top-k must be positive")
    if args.cpu_warmup is not None and args.cpu_warmup < 0:
        parser.error("cpu-warmup must be non-negative")
    if args.cpu_repeats is not None and args.cpu_repeats < 1:
        parser.error("cpu-repeats must be positive")
    return args


def block_until_ready(tree):
    return jax.tree.map(lambda leaf: leaf.block_until_ready(), tree)


def timing_summary(samples):
    mean = statistics.fmean(samples)
    deviation = statistics.pstdev(samples)
    return {
        "median_seconds": statistics.median(samples),
        "mean_seconds": mean,
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "coefficient_of_variation": deviation / mean if mean else 0.0,
        "samples_seconds": samples,
    }


def relative_error(actual, expected):
    scale = max(float(np.linalg.norm(expected)), 1e-30)
    return float(np.linalg.norm(actual - expected) / scale)


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


def device_memory_stats(device):
    stats = device.memory_stats()
    if stats is None:
        return None
    return {
        key: value.item() if hasattr(value, "item") else value
        for key, value in stats.items()
    }


def measure_candidate(
    data,
    device_dofs,
    device_currents,
    target_tile_size,
    source_tile_size,
    vjp_mode,
    warmup,
    repeats,
    objective_kwargs,
):
    """Compile and synchronously measure one tile and reverse-pass candidate."""
    config = GpuConfig(
        target_tile_size=target_tile_size,
        source_tile_size=source_tile_size,
        vjp_mode=vjp_mode,
    )

    def objective(curve_dofs, currents):
        return data.objective(
            curve_dofs,
            currents,
            **objective_kwargs,
            config=config,
        )

    value_and_gradient = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))
    compile_start = time.perf_counter()
    compiled = value_and_gradient.lower(device_dofs, device_currents).compile()
    compilation_seconds = time.perf_counter() - compile_start

    for _ in range(warmup):
        block_until_ready(compiled(device_dofs, device_currents))

    samples = []
    latest = None
    for _ in range(repeats):
        start = time.perf_counter()
        latest = block_until_ready(compiled(device_dofs, device_currents))
        samples.append(time.perf_counter() - start)

    value, (curve_gradient, current_gradient) = latest
    return {
        "compilation_seconds": compilation_seconds,
        "timing": timing_summary(samples),
        "value": float(value),
        "curve_gradient": np.asarray(curve_gradient),
        "current_gradient": np.asarray(current_gradient),
    }


def parity(measurement, cpu_value, cpu_curve_gradient, cpu_current_gradient):
    return {
        "value_absolute_error": abs(measurement["value"] - cpu_value),
        "curve_gradient_max_absolute_error": float(
            np.max(np.abs(measurement["curve_gradient"] - cpu_curve_gradient))
        ),
        "curve_gradient_relative_l2_error": relative_error(
            measurement["curve_gradient"], cpu_curve_gradient
        ),
        "current_gradient_max_absolute_error": float(
            np.max(np.abs(measurement["current_gradient"] - cpu_current_gradient))
        ),
        "current_gradient_relative_l2_error": relative_error(
            measurement["current_gradient"], cpu_current_gradient
        ),
    }


def passes_parity(result, tolerances):
    return (
        result["value_absolute_error"] <= tolerances["value_absolute_error"]
        and result["curve_gradient_relative_l2_error"]
        <= tolerances["gradient_relative_l2_error"]
        and result["current_gradient_relative_l2_error"]
        <= tolerances["gradient_relative_l2_error"]
    )


def candidate_id(target_tile_size, source_tile_size, vjp_mode):
    return f"{vjp_mode}-target-{target_tile_size}-source-{source_tile_size}"


def run_candidate(
    data,
    device_dofs,
    device_currents,
    cpu_value,
    cpu_curve_gradient,
    cpu_current_gradient,
    target_tile_size,
    source_tile_size,
    vjp_mode,
    warmup,
    repeats,
    tolerances,
    objective_kwargs,
):
    try:
        measurement = measure_candidate(
            data,
            device_dofs,
            device_currents,
            target_tile_size,
            source_tile_size,
            vjp_mode,
            warmup,
            repeats,
            objective_kwargs,
        )
        parity_result = parity(
            measurement,
            cpu_value,
            cpu_curve_gradient,
            cpu_current_gradient,
        )
        return {
            "status": "ok",
            "eligible": passes_parity(parity_result, tolerances),
            "compilation_seconds": measurement["compilation_seconds"],
            "timing": measurement["timing"],
            "parity": parity_result,
        }
    except (RuntimeError, ValueError, MemoryError, FloatingPointError) as error:
        return {
            "status": "error",
            "eligible": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    finally:
        gc.collect()
        jax.clear_caches()


def main():
    args = parse_args()
    sweep_start = time.perf_counter()
    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_non_gpu:
        raise RuntimeError(
            f"JAX selected {backend!r}, not 'gpu'. In Colab select "
            "Runtime > Change runtime type > GPU, then restart and rerun."
        )
    gpu_inventory = nvidia_smi()
    if backend == "gpu" and not gpu_inventory:
        raise RuntimeError("JAX reports a GPU but nvidia-smi found no NVIDIA device")

    spec = get_problem(args.problem)
    objective_settings = objective_metadata(spec, args.objective_scope)
    objective_kwargs = objective_call_kwargs(objective_settings)
    surface, base_curves, field, _, cpu_objective = build_problem(
        spec,
        regularized=args.objective_scope == "full-engineering",
        local_engineering=args.objective_scope == "local-engineering",
    )
    base_current_objects = [field.coils[index].current for index in range(spec.ncoils)]
    data = minimal_coil_data(
        surface,
        base_curves,
        base_current_objects,
        surface.nfp,
        stellsym=True,
    )
    device_dofs = jnp.asarray(data.curve_dofs)
    device_currents = jnp.asarray(data.base_currents)

    initial_x = cpu_objective.x.copy()
    cpu_device = jax.devices("cpu")[0]

    def cpu_combined_value_gradient():
        with jax.default_device(cpu_device):
            cpu_objective.x = initial_x
            return cpu_objective.J(), cpu_objective.dJ()

    cpu_timing = None
    if not args.skip_cpu_baseline:
        cpu_warmup = args.warmup if args.cpu_warmup is None else args.cpu_warmup
        cpu_repeats = args.repeats if args.cpu_repeats is None else args.cpu_repeats
        cpu_timing = measure(
            cpu_combined_value_gradient,
            cpu_warmup,
            cpu_repeats,
        )
    with jax.default_device(cpu_device):
        cpu_objective.x = initial_x
        cpu_value = float(cpu_objective.J())
        cpu_derivative = cpu_objective.dJ(partials=True)
    cpu_curve_gradient = np.stack([cpu_derivative.data[curve] for curve in base_curves])
    cpu_current_gradient = np.asarray(
        [cpu_derivative.data[current][0] for current in base_current_objects]
    )

    ntarget = data.surface_points.shape[0]
    nsource = len(field.coils) * data.bases.shape[1]
    target_sizes = resolve_sizes(args.target_tile_sizes, ntarget)
    source_sizes = resolve_sizes(args.source_tile_sizes, nsource)
    candidates_to_screen = [
        (target_size, source_size, vjp_mode)
        for target_size in target_sizes
        for source_size in source_sizes
        for vjp_mode in args.vjp_modes
    ]
    tolerances = {
        "value_absolute_error": 1e-9,
        "gradient_relative_l2_error": 1e-7,
    }

    candidates = []
    for index, (target_size, source_size, vjp_mode) in enumerate(
        candidates_to_screen, start=1
    ):
        print(
            f"Screening {index}/{len(candidates_to_screen)}: "
            f"mode={vjp_mode}, target={target_size}, source={source_size}",
            flush=True,
        )
        screening = run_candidate(
            data,
            device_dofs,
            device_currents,
            cpu_value,
            cpu_curve_gradient,
            cpu_current_gradient,
            target_size,
            source_size,
            vjp_mode,
            args.warmup,
            args.repeats,
            tolerances,
            objective_kwargs,
        )
        candidate = {
            "id": candidate_id(target_size, source_size, vjp_mode),
            "vjp_mode": vjp_mode,
            "target_tile_size": target_size,
            "source_tile_size": source_size,
            "target_blocks": math.ceil(ntarget / target_size),
            "source_blocks": math.ceil(nsource / source_size),
            "tile_iterations": math.ceil(ntarget / target_size)
            * math.ceil(nsource / source_size),
            "screening": screening,
            "confirmation": None,
        }
        candidates.append(candidate)

    eligible = sorted(
        (candidate for candidate in candidates if candidate["screening"]["eligible"]),
        key=lambda candidate: candidate["screening"]["timing"]["median_seconds"],
    )
    for rank, candidate in enumerate(eligible, start=1):
        candidate["screening_rank"] = rank
    if not eligible:
        raise RuntimeError("no tile-size candidate completed and passed parity")

    confirmation_candidates = eligible[: args.top_k]
    default_specification = (min(128, ntarget), min(256, nsource), "autodiff")
    default_candidate = next(
        (
            candidate
            for candidate in candidates
            if (
                candidate["target_tile_size"],
                candidate["source_tile_size"],
                candidate["vjp_mode"],
            )
            == default_specification
        ),
        None,
    )
    if (
        default_candidate is not None
        and default_candidate not in confirmation_candidates
    ):
        confirmation_candidates.append(default_candidate)
    best_autodiff_candidate = next(
        (candidate for candidate in eligible if candidate["vjp_mode"] == "autodiff"),
        None,
    )
    if (
        best_autodiff_candidate is not None
        and best_autodiff_candidate not in confirmation_candidates
    ):
        confirmation_candidates.append(best_autodiff_candidate)

    for index, candidate in enumerate(confirmation_candidates, start=1):
        print(
            f"Confirming {index}/{len(confirmation_candidates)}: "
            f"mode={candidate['vjp_mode']}, target={candidate['target_tile_size']}, "
            f"source={candidate['source_tile_size']}",
            flush=True,
        )
        candidate["confirmation"] = run_candidate(
            data,
            device_dofs,
            device_currents,
            cpu_value,
            cpu_curve_gradient,
            cpu_current_gradient,
            candidate["target_tile_size"],
            candidate["source_tile_size"],
            candidate["vjp_mode"],
            args.confirmation_warmup,
            args.confirmation_repeats,
            tolerances,
            objective_kwargs,
        )

    confirmed = sorted(
        (
            candidate
            for candidate in confirmation_candidates
            if candidate["confirmation"]["eligible"]
        ),
        key=lambda candidate: candidate["confirmation"]["timing"]["median_seconds"],
    )
    if not confirmed:
        raise RuntimeError("no confirmation candidate completed and passed parity")
    for rank, candidate in enumerate(confirmed, start=1):
        candidate["confirmation_rank"] = rank
    winner = confirmed[0]
    winner_median = winner["confirmation"]["timing"]["median_seconds"]

    default_confirmed_median = None
    if (
        default_candidate is not None
        and default_candidate["confirmation"] is not None
        and default_candidate["confirmation"]["eligible"]
    ):
        default_confirmed_median = default_candidate["confirmation"]["timing"][
            "median_seconds"
        ]
    best_autodiff_confirmed_median = None
    if (
        best_autodiff_candidate is not None
        and best_autodiff_candidate["confirmation"] is not None
        and best_autodiff_candidate["confirmation"]["eligible"]
    ):
        best_autodiff_confirmed_median = best_autodiff_candidate["confirmation"][
            "timing"
        ]["median_seconds"]

    result = {
        "schema_version": 3,
        "problem": spec.as_dict(),
        "objective": objective_settings,
        "dimensions": {
            "base_coils": len(base_curves),
            "physical_coils": len(field.coils),
            "surface_points": ntarget,
            "curve_quadrature_points": data.bases.shape[1],
            "source_points": nsource,
            "target_source_interactions": ntarget * nsource,
            "differentiated_variables": int(
                data.curve_dofs.size + data.base_currents.size
            ),
        },
        "environment": environment(),
        "nvidia_smi": gpu_inventory,
        "protocol": {
            "screening_warmup": args.warmup,
            "screening_repeats": args.repeats,
            "cpu_warmup": (args.warmup if args.cpu_warmup is None else args.cpu_warmup),
            "cpu_repeats": (
                args.repeats if args.cpu_repeats is None else args.cpu_repeats
            ),
            "confirmation_top_k": args.top_k,
            "confirmation_warmup": args.confirmation_warmup,
            "confirmation_repeats": args.confirmation_repeats,
            "requested_target_tile_sizes": args.target_tile_sizes,
            "requested_source_tile_sizes": args.source_tile_sizes,
            "vjp_modes": args.vjp_modes,
            "resolved_target_tile_sizes": target_sizes,
            "resolved_source_tile_sizes": source_sizes,
            "parity_tolerances": tolerances,
        },
        "cpu_baseline": cpu_timing,
        "cpu_baseline_device": str(cpu_device),
        "candidates": candidates,
        "winner": {
            "id": winner["id"],
            "vjp_mode": winner["vjp_mode"],
            "target_tile_size": winner["target_tile_size"],
            "source_tile_size": winner["source_tile_size"],
            "target_blocks": winner["target_blocks"],
            "source_blocks": winner["source_blocks"],
            "tile_iterations": winner["tile_iterations"],
            "confirmation_timing": winner["confirmation"]["timing"],
            "speedup_over_default_tiles": (
                default_confirmed_median / winner_median
                if default_confirmed_median is not None
                else None
            ),
            "speedup_over_best_autodiff": (
                best_autodiff_confirmed_median / winner_median
                if best_autodiff_confirmed_median is not None
                else None
            ),
            "speedup_over_cpu_baseline": (
                cpu_timing["median_seconds"] / winner_median
                if cpu_timing is not None
                else None
            ),
        },
        "process_device_memory_after_sweep": device_memory_stats(jax.devices()[0]),
        "total_sweep_seconds": time.perf_counter() - sweep_start,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    if not args.quiet:
        print(encoded)
    else:
        print(json.dumps(result["winner"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
