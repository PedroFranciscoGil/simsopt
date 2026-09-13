"""Qualify local engineering-residual values and directional Jacobians."""

import argparse
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_objective import build_problem, environment
from problems import PROBLEMS, get_problem, objective_metadata
from simsopt.gpu import GpuConfig, ScipyCoilObjectiveBridge, minimal_coil_data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), default="engineering")
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument("--finite-difference-step", type=float, default=1e-7)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--target-tile-size", type=int, default=1024)
    parser.add_argument("--source-tile-size", type=int, default=4320)
    parser.add_argument("--vjp-mode", choices=("autodiff", "custom"), default="custom")
    parser.add_argument("--current-scale", type=float, default=1e5)
    parser.add_argument("--distance-feasibility-tolerance", type=float, default=1e-4)
    parser.add_argument("--curvature-feasibility-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--mean-squared-curvature-feasibility-tolerance",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-non-gpu", action="store_true")
    args = parser.parse_args()
    for name in (
        "finite_difference_step",
        "current_scale",
        "distance_feasibility_tolerance",
        "curvature_feasibility_tolerance",
        "mean_squared_curvature_feasibility_tolerance",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"{name.replace('_', '-')} must be finite and positive")
    if args.directions < 1 or args.repeats < 1:
        parser.error("directions and repeats must be positive")
    if args.target_tile_size < 1 or args.source_tile_size < 1:
        parser.error("tile sizes must be positive")
    return args


def _nearest_measurements(base_curves, physical_curves, surface_points, pairs):
    """Evaluate the sampled physical quantities defining every residual."""
    physical_gamma = np.stack([np.asarray(curve.gamma()) for curve in physical_curves])
    coil_coil = []
    for first, second in np.asarray(pairs, dtype=int):
        differences = (
            physical_gamma[first, :, None, :] - physical_gamma[second, None, :, :]
        )
        coil_coil.append(np.min(np.linalg.norm(differences, axis=-1), axis=1))
    coil_surface = []
    for gamma in physical_gamma:
        differences = gamma[:, None, :] - surface_points[None, :, :]
        coil_surface.append(np.min(np.linalg.norm(differences, axis=-1), axis=1))
    curvature = np.stack([np.asarray(curve.kappa()) for curve in base_curves])
    speeds = np.stack(
        [np.asarray(curve.incremental_arclength()) for curve in base_curves]
    )
    mean_squared_curvature = np.sum(curvature**2 * speeds, axis=1) / np.sum(
        speeds, axis=1
    )
    return {
        "coil_coil_distance": np.asarray(coil_coil),
        "coil_surface_distance": np.asarray(coil_surface),
        "curvature": curvature,
        "mean_squared_curvature": mean_squared_curvature,
    }


def residuals_from_measurements(measurements, settings):
    """Apply the documented physical allowances and nondimensionalization."""
    distance_scale = settings["distance_feasibility_tolerance"]
    curvature_scale = settings["curvature_feasibility_tolerance"]
    mean_scale = settings["mean_squared_curvature_feasibility_tolerance"]
    families = {
        "coil_coil_distance": np.maximum(
            settings["coil_coil_distance_threshold"]
            - distance_scale
            - measurements["coil_coil_distance"],
            0.0,
        )
        / distance_scale,
        "coil_surface_distance": np.maximum(
            settings["coil_surface_distance_threshold"]
            - distance_scale
            - measurements["coil_surface_distance"],
            0.0,
        )
        / distance_scale,
        "curvature": np.maximum(
            measurements["curvature"]
            - settings["curvature_threshold"]
            - curvature_scale,
            0.0,
        )
        / curvature_scale,
        "mean_squared_curvature": np.maximum(
            measurements["mean_squared_curvature"]
            - settings["mean_squared_curvature_threshold"]
            - mean_scale,
            0.0,
        )
        / mean_scale,
    }
    return families, np.concatenate([value.reshape(-1) for value in families.values()])


def _family_summary(values):
    values = np.asarray(values)
    return {
        "count": int(values.size),
        "active_count": int(np.count_nonzero(values > 0.0)),
        "maximum": float(np.max(values, initial=0.0)),
        "l2_norm": float(np.linalg.norm(values)),
    }


def _parity(reference, candidate, floor=1e-14):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    difference = candidate - reference
    return {
        "maximum_absolute_error": float(np.max(np.abs(difference), initial=0.0)),
        "relative_l2_error": float(
            np.linalg.norm(difference) / max(np.linalg.norm(reference), floor)
        ),
    }


def _forced_settings(measurements, settings):
    """Construct a diagnostic point at which every residual family is active."""

    def interior_cut(values, fraction, fallback_offset):
        flat = np.sort(np.asarray(values).reshape(-1))
        cluster_tolerance = 1e-10 * max(1.0, float(np.max(np.abs(flat))))
        unique = [float(flat[0])]
        for value in flat[1:]:
            if value - unique[-1] > cluster_tolerance:
                unique.append(float(value))
        unique = np.asarray(unique)
        if unique.size == 1:
            return float(unique[0] - fallback_offset)
        lower = min(int(fraction * (unique.size - 1)), unique.size - 2)
        return float(0.5 * (unique[lower] + unique[lower + 1]))

    forced = dict(settings)
    distance_scale = settings["distance_feasibility_tolerance"]
    curvature_scale = settings["curvature_feasibility_tolerance"]
    mean_scale = settings["mean_squared_curvature_feasibility_tolerance"]
    forced["coil_coil_distance_threshold"] = float(
        interior_cut(measurements["coil_coil_distance"], 0.35, 10.0 * distance_scale)
        + distance_scale
    )
    forced["coil_surface_distance_threshold"] = float(
        interior_cut(measurements["coil_surface_distance"], 0.35, 10.0 * distance_scale)
        + distance_scale
    )
    forced["curvature_threshold"] = float(
        interior_cut(measurements["curvature"], 0.65, 10.0 * curvature_scale)
        - curvature_scale
    )
    forced["mean_squared_curvature_threshold"] = float(
        interior_cut(measurements["mean_squared_curvature"], 0.50, 10.0 * mean_scale)
        - mean_scale
    )
    return forced


def _measure(operation, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    return {
        "median_seconds": float(statistics.median(samples)),
        "minimum_seconds": float(min(samples)),
        "maximum_seconds": float(max(samples)),
        "samples_seconds": samples,
    }


def _block_tree(tree):
    for leaf in jax.tree.leaves(tree):
        leaf.block_until_ready()


def main():
    args = parse_args()
    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_non_gpu:
        raise RuntimeError(
            f"JAX selected {backend!r}, not 'gpu'. Select an NVIDIA GPU runtime."
        )

    spec = get_problem(args.problem)
    metadata = objective_metadata(spec, "full-engineering")
    surface, base_curves, field, components, cpu_penalty_objective = build_problem(
        spec, regularized=True, objective_settings=metadata
    )
    physical_curves = [coil.curve for coil in field.coils]
    base_objective = (
        components["quadratic_flux"]
        + metadata["length_weight"] * components["curve_length_sum"]
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
    coordinate_bridge = ScipyCoilObjectiveBridge(
        data,
        free_current_indices=free_current_indices,
        objective_kwargs={
            "length_target": None,
            "length_weight": metadata["length_weight"],
        },
        current_scale=args.current_scale,
        config=config,
    )
    initial_x = coordinate_bridge.initial_x.copy()
    surface_points = np.asarray(data.surface_points)
    tolerances = {
        "distance_feasibility_tolerance": args.distance_feasibility_tolerance,
        "curvature_feasibility_tolerance": args.curvature_feasibility_tolerance,
        "mean_squared_curvature_feasibility_tolerance": (
            args.mean_squared_curvature_feasibility_tolerance
        ),
    }
    production_settings = {
        **tolerances,
        "coil_coil_distance_threshold": metadata["coil_coil_distance_threshold"],
        "coil_surface_distance_threshold": metadata["coil_surface_distance_threshold"],
        "curvature_threshold": metadata["curvature_threshold"],
        "mean_squared_curvature_threshold": metadata[
            "mean_squared_curvature_threshold"
        ],
    }

    def cpu_terms(x, settings):
        cpu_penalty_objective.x = coordinate_bridge.to_physical_variables(x)
        measurements = _nearest_measurements(
            base_curves,
            physical_curves,
            surface_points,
            data.coil_coil_pair_indices,
        )
        families, residuals = residuals_from_measurements(measurements, settings)
        return float(base_objective.J()), families, residuals, measurements

    def gpu_term_function(settings):
        kwargs = {
            "length_weight": metadata["length_weight"],
            **settings,
        }

        def terms(x):
            curve_dofs, currents = coordinate_bridge.unpack(x)
            return data.local_residual_terms(
                curve_dofs, currents, **kwargs, config=config
            )

        return terms

    cpu_base, cpu_families, cpu_residuals, _ = cpu_terms(initial_x, production_settings)
    probe_rng = np.random.default_rng(20260912)
    parity_probe_offset = np.zeros_like(initial_x)
    geometry_offset = probe_rng.normal(size=initial_x.size - free_current_indices.size)
    geometry_offset *= 1e-4 / np.max(np.abs(geometry_offset))
    parity_probe_offset[free_current_indices.size :] = geometry_offset
    parity_x = initial_x + parity_probe_offset
    _, _, _, parity_measurements = cpu_terms(parity_x, production_settings)
    forced_settings = _forced_settings(parity_measurements, production_settings)

    compile_start = time.perf_counter()
    production_executable = (
        jax.jit(gpu_term_function(production_settings))
        .lower(jnp.asarray(initial_x))
        .compile()
    )
    production_compile_seconds = time.perf_counter() - compile_start
    gpu_base, gpu_residuals = production_executable(jnp.asarray(initial_x))
    _block_tree((gpu_base, gpu_residuals))
    gpu_residuals = np.asarray(gpu_residuals)
    layout = data.local_residual_layout()
    gpu_families = {
        name: gpu_residuals[item["start"] : item["stop"]]
        for name, item in layout.items()
    }

    forced_gpu_function = gpu_term_function(forced_settings)
    forced_executable = (
        jax.jit(forced_gpu_function).lower(jnp.asarray(initial_x)).compile()
    )
    forced_gpu_base, forced_gpu_residuals = forced_executable(jnp.asarray(parity_x))
    _block_tree((forced_gpu_base, forced_gpu_residuals))
    _, forced_cpu_families, forced_cpu_residuals, _ = cpu_terms(
        parity_x, forced_settings
    )
    forced_gpu_residuals = np.asarray(forced_gpu_residuals)
    forced_gpu_families = {
        name: forced_gpu_residuals[item["start"] : item["stop"]]
        for name, item in layout.items()
    }

    def residual_jvp(x, direction):
        return jax.jvp(
            lambda variables: forced_gpu_function(variables)[1],
            (x,),
            (direction,),
        )[1]

    jvp_executable = (
        jax.jit(residual_jvp)
        .lower(jnp.asarray(initial_x), jnp.asarray(initial_x))
        .compile()
    )
    rng = np.random.default_rng(20260913)
    directional_results = []
    for index in range(args.directions):
        direction = rng.normal(size=parity_x.size)
        direction /= np.linalg.norm(direction)
        gpu_tangent = jvp_executable(jnp.asarray(parity_x), jnp.asarray(direction))
        gpu_tangent.block_until_ready()
        step = args.finite_difference_step
        plus = cpu_terms(parity_x + step * direction, forced_settings)[2]
        minus = cpu_terms(parity_x - step * direction, forced_settings)[2]
        cpu_tangent = (plus - minus) / (2.0 * step)
        gpu_plus = forced_executable(jnp.asarray(parity_x + step * direction))[1]
        gpu_minus = forced_executable(jnp.asarray(parity_x - step * direction))[1]
        _block_tree((gpu_plus, gpu_minus))
        gpu_finite_difference = (np.asarray(gpu_plus) - np.asarray(gpu_minus)) / (
            2.0 * step
        )
        family_parity = {}
        for name, item in layout.items():
            family_slice = slice(item["start"], item["stop"])
            family_parity[name] = _parity(
                cpu_tangent[family_slice],
                np.asarray(gpu_tangent)[family_slice],
                floor=1e-12,
            )
        directional_results.append(
            {
                "index": index,
                "direction_seed": 20260913,
                "finite_difference_step": step,
                **_parity(cpu_tangent, np.asarray(gpu_tangent), floor=1e-12),
                "gpu_finite_difference_vs_jvp": _parity(
                    gpu_finite_difference, np.asarray(gpu_tangent), floor=1e-12
                ),
                "cpu_vs_gpu_finite_difference": _parity(
                    cpu_tangent, gpu_finite_difference, floor=1e-12
                ),
                "family_parity": family_parity,
                "cpu_tangent_l2_norm": float(np.linalg.norm(cpu_tangent)),
                "gpu_tangent_l2_norm": float(np.linalg.norm(gpu_tangent)),
            }
        )

    def timed_cpu():
        cpu_terms(initial_x, production_settings)

    device_initial_x = jnp.asarray(initial_x)

    def timed_gpu():
        values = production_executable(device_initial_x)
        _block_tree(values)

    cpu_timing = _measure(timed_cpu, args.repeats)
    gpu_timing = _measure(timed_gpu, args.repeats)
    value_parity = _parity(cpu_residuals, gpu_residuals)
    forced_value_parity = _parity(forced_cpu_residuals, forced_gpu_residuals)
    base_absolute_error = abs(float(gpu_base) - cpu_base)
    base_relative_error = base_absolute_error / max(abs(cpu_base), 1e-14)
    all_forced_active = all(
        np.count_nonzero(values > 0.0) > 0 for values in forced_cpu_families.values()
    )
    maximum_jacobian_relative_error = max(
        item["relative_l2_error"] for item in directional_results
    )
    gates = {
        "gpu_backend": {"passed": backend == "gpu", "observed": backend},
        "float64_enabled": {
            "passed": bool(jax.config.jax_enable_x64),
            "observed": bool(jax.config.jax_enable_x64),
        },
        "base_objective_value_parity": {
            "passed": base_relative_error <= 5e-10 or base_absolute_error <= 1e-11,
            "relative_error": base_relative_error,
            "absolute_error": base_absolute_error,
        },
        "production_residual_value_parity": {
            "passed": value_parity["relative_l2_error"] <= 5e-10
            or value_parity["maximum_absolute_error"] <= 1e-11,
            **value_parity,
        },
        "forced_family_activation": {
            "passed": all_forced_active,
            "active_counts": {
                name: int(np.count_nonzero(values > 0.0))
                for name, values in forced_cpu_families.items()
            },
        },
        "forced_residual_value_parity": {
            "passed": forced_value_parity["relative_l2_error"] <= 5e-10,
            **forced_value_parity,
        },
        "directional_jacobian_parity": {
            "passed": maximum_jacobian_relative_error <= 5e-4,
            "maximum_relative_l2_error": maximum_jacobian_relative_error,
            "tolerance": 5e-4,
        },
        "residual_dimension": {
            "passed": cpu_residuals.size <= 100_000,
            "observed": int(cpu_residuals.size),
            "maximum": 100_000,
        },
    }
    result = {
        "schema_version": 1,
        "workflow": "local_engineering_residual_parity",
        "problem": spec.as_dict(),
        "environment": environment(),
        "precision": "float64",
        "configuration": {
            "target_tile_size": config.target_tile_size,
            "source_tile_size": config.source_tile_size,
            "vjp_mode": config.vjp_mode,
            "current_scale_amperes": args.current_scale,
            "finite_difference_directions": args.directions,
            "finite_difference_step": args.finite_difference_step,
        },
        "residual_contract": {
            "family_order": list(layout),
            "layout": layout,
            "definition": (
                "positive physical deficit or excess divided by its feasibility "
                "allowance"
            ),
            "sampling": (
                "nearest sampled point for distances; pointwise base-curve "
                "curvature; one MSC residual per base curve"
            ),
            "production_settings": production_settings,
            "forced_activation_settings": forced_settings,
            "jacobian_probe": {
                "geometry_offset_seed": 20260912,
                "geometry_offset_infinity_norm": float(
                    np.max(np.abs(parity_probe_offset))
                ),
                "purpose": (
                    "break exact nearest-neighbor and curvature ties before "
                    "testing the classical directional derivative"
                ),
            },
        },
        "production": {
            "cpu_base_objective": cpu_base,
            "gpu_base_objective": float(gpu_base),
            "base_objective_parity": {
                "absolute_error": base_absolute_error,
                "relative_error": base_relative_error,
            },
            "residual_parity": value_parity,
            "cpu_families": {
                name: _family_summary(values) for name, values in cpu_families.items()
            },
            "gpu_families": {
                name: _family_summary(values) for name, values in gpu_families.items()
            },
        },
        "forced_activation": {
            "residual_parity": forced_value_parity,
            "cpu_families": {
                name: _family_summary(values)
                for name, values in forced_cpu_families.items()
            },
            "gpu_families": {
                name: _family_summary(values)
                for name, values in forced_gpu_families.items()
            },
            "directional_jacobian_parity": directional_results,
        },
        "timing": {
            "production_gpu_compile_seconds": production_compile_seconds,
            "cpu_terms": cpu_timing,
            "gpu_compiled_terms": gpu_timing,
            "warm_speedup": cpu_timing["median_seconds"] / gpu_timing["median_seconds"],
        },
        "acceptance_gates": gates,
        "all_gates_passed": all(gate["passed"] for gate in gates.values()),
    }
    text = json.dumps(result, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(args.output)


if __name__ == "__main__":
    main()
