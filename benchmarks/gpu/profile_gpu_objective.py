"""Profile the compiled minimal coil objective on an NVIDIA GPU."""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_objective import build_problem, environment
from problems import PROBLEMS, get_problem
from simsopt.gpu import GpuConfig, minimal_coil_data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), default="minimal")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--target-tile-size", type=int, default=128)
    parser.add_argument("--source-tile-size", type=int, default=256)
    parser.add_argument(
        "--vjp-mode", choices=("autodiff", "custom"), default="custom"
    )
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--trace-steps", type=int, default=3)
    parser.add_argument("--memory-profile", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-non-gpu",
        action="store_true",
        help="Permit CPU execution for local harness validation only.",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.trace_steps < 1:
        parser.error(
            "repeats and trace-steps must be positive; warmup must be non-negative"
        )
    if args.target_tile_size < 1 or args.source_tile_size < 1:
        parser.error("tile sizes must be positive")
    return args


def block_until_ready(tree):
    return jax.tree.map(lambda leaf: leaf.block_until_ready(), tree)


def relative_error(actual, expected):
    scale = max(float(np.linalg.norm(expected)), 1e-30)
    return float(np.linalg.norm(actual - expected) / scale)


def device_memory_stats(device):
    stats = device.memory_stats()
    if stats is None:
        return None
    return {
        key: value.item() if hasattr(value, "item") else value
        for key, value in stats.items()
    }


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
            f"JAX selected {backend!r}, not 'gpu'. In Colab select "
            "Runtime > Change runtime type > GPU, then restart and rerun."
        )
    gpu_inventory = nvidia_smi()
    if backend == "gpu" and not gpu_inventory:
        raise RuntimeError("JAX reports a GPU but nvidia-smi found no NVIDIA device")

    spec = get_problem(args.problem)
    surface, base_curves, field, _, cpu_objective = build_problem(
        spec, regularized=False
    )
    base_current_objects = [field.coils[index].current for index in range(spec.ncoils)]
    config = GpuConfig(
        target_tile_size=args.target_tile_size,
        source_tile_size=args.source_tile_size,
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

    def objective(curve_dofs, currents):
        return data.objective(
            curve_dofs,
            currents,
            length_target=18.0,
            length_weight=1.0,
            config=config,
        )

    value_and_gradient = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))
    device_dofs = jnp.asarray(data.curve_dofs)
    device_currents = jnp.asarray(data.base_currents)

    compile_start = time.perf_counter()
    compiled = value_and_gradient.lower(device_dofs, device_currents).compile()
    compilation_seconds = time.perf_counter() - compile_start

    for _ in range(args.warmup):
        block_until_ready(compiled(device_dofs, device_currents))

    samples = []
    latest = None
    for _ in range(args.repeats):
        start = time.perf_counter()
        latest = block_until_ready(compiled(device_dofs, device_currents))
        samples.append(time.perf_counter() - start)

    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(
            str(args.trace_dir),
            create_perfetto_trace=True,
        ):
            for step in range(args.trace_steps):
                with jax.profiler.StepTraceAnnotation(
                    "minimal_objective_value_and_gradient", step_num=step
                ):
                    block_until_ready(compiled(device_dofs, device_currents))

    if args.memory_profile:
        args.memory_profile.parent.mkdir(parents=True, exist_ok=True)
        jax.profiler.save_device_memory_profile(str(args.memory_profile))

    gpu_value, (gpu_curve_gradient, gpu_current_gradient) = latest
    gpu_value = float(gpu_value)
    gpu_curve_gradient = np.asarray(gpu_curve_gradient)
    gpu_current_gradient = np.asarray(gpu_current_gradient)

    cpu_value = float(cpu_objective.J())
    cpu_derivative = cpu_objective.dJ(partials=True)
    cpu_curve_gradient = np.stack([cpu_derivative.data[curve] for curve in base_curves])
    cpu_current_gradient = np.asarray(
        [cpu_derivative.data[current][0] for current in base_current_objects]
    )

    device = jax.devices()[0]
    result = {
        "schema_version": 2,
        "problem": spec.as_dict(),
        "objective": {
            "terms": ["quadratic_flux", "curve_length_penalty"],
            "length_target": 18.0,
            "length_weight": 1.0,
        },
        "tiles": {
            "target": args.target_tile_size,
            "source": args.source_tile_size,
        },
        "vjp_mode": args.vjp_mode,
        "dimensions": {
            "base_coils": len(base_curves),
            "physical_coils": len(field.coils),
            "surface_points": data.surface_points.shape[0],
            "curve_quadrature_points": data.bases.shape[1],
            "differentiated_variables": int(
                data.curve_dofs.size + data.base_currents.size
            ),
        },
        "environment": environment(),
        "nvidia_smi": gpu_inventory,
        "device_memory": device_memory_stats(device),
        "compilation_seconds": compilation_seconds,
        "steady_state": {
            "median_seconds": statistics.median(samples),
            "minimum_seconds": min(samples),
            "maximum_seconds": max(samples),
            "samples_seconds": samples,
        },
        "parity": {
            "gpu_value": gpu_value,
            "cpu_value": cpu_value,
            "value_absolute_error": abs(gpu_value - cpu_value),
            "curve_gradient_max_absolute_error": float(
                np.max(np.abs(gpu_curve_gradient - cpu_curve_gradient))
            ),
            "curve_gradient_relative_l2_error": relative_error(
                gpu_curve_gradient, cpu_curve_gradient
            ),
            "current_gradient_max_absolute_error": float(
                np.max(np.abs(gpu_current_gradient - cpu_current_gradient))
            ),
            "current_gradient_relative_l2_error": relative_error(
                gpu_current_gradient, cpu_current_gradient
            ),
        },
        "artifacts": {
            "trace_dir": str(args.trace_dir) if args.trace_dir else None,
            "memory_profile": (
                str(args.memory_profile) if args.memory_profile else None
            ),
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
