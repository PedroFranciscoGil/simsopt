"""Replay a qualified AL warm start across device-resident L-BFGS settings."""

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
from benchmark_local_residual_augmented_lagrangian import (
    device_refinement_summary,
    engineering_target_validation,
    nvidia_smi,
    qoi_backend_agreement,
    quadratic_flux_target_validation,
    target_envelope_objective_settings,
    target_envelope_residual_kwargs,
)
from benchmark_objective import build_problem, environment
from optimization_metrics import export_final_design_visualization, final_coil_metrics
from problems import get_problem, objective_call_kwargs, objective_metadata
from simsopt.gpu import (
    DeviceLBFGS,
    DeviceLBFGSConfig,
    GpuConfig,
    ScipyCoilObjectiveBridge,
    TargetAwareConfig,
    minimal_coil_data,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = (
    ROOT
    / "docs"
    / "gpu_native"
    / "figures"
    / "device_lbfgs_comparison_analysis_summary.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-analysis", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--history-sizes", type=int, nargs="+", default=[10, 20, 50, 100]
    )
    parser.add_argument("--checkpoint-patiences", type=int, nargs="+", default=[15, 25])
    parser.add_argument("--infeasible-patience", type=int, default=15)
    parser.add_argument("--minimum-relative-improvement", type=float, default=1e-3)
    parser.add_argument("--max-iterations", type=int, default=150)
    parser.add_argument("--max-line-search-iterations", type=int, default=30)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-8)
    parser.add_argument("--target-relative-tolerance", type=float, default=0.10)
    parser.add_argument("--quadratic-flux-target", type=float, default=1e-5)
    parser.add_argument("--current-scale", type=float, default=1e5)
    parser.add_argument("--target-tile-size", type=int, default=1024)
    parser.add_argument("--source-tile-size", type=int, default=4320)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--allow-non-gpu", action="store_true")
    args = parser.parse_args()
    if not args.history_sizes or any(value < 1 for value in args.history_sizes):
        parser.error("history sizes must be positive")
    if not args.checkpoint_patiences or any(
        value < 1 for value in args.checkpoint_patiences
    ):
        parser.error("checkpoint patiences must be positive")
    if args.infeasible_patience < 1:
        parser.error("infeasible-patience must be positive")
    if args.max_iterations < 1 or args.max_line_search_iterations < 1:
        parser.error("iteration limits must be positive")
    for name in (
        "gradient_tolerance",
        "target_relative_tolerance",
        "quadratic_flux_target",
        "current_scale",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name.replace('_', '-')} must be finite and positive")
    if not 0 <= args.minimum_relative_improvement < 1:
        parser.error("minimum-relative-improvement must be in [0, 1)")
    if args.target_relative_tolerance >= 1:
        parser.error("target-relative-tolerance must be below one")
    if args.target_tile_size < 1 or args.source_tile_size < 1:
        parser.error("tile sizes must be positive")
    if args.no_visualization and args.visualization_dir is not None:
        parser.error("no-visualization and visualization-dir are mutually exclusive")
    return args


def load_baseline(path):
    baseline = json.loads(path.read_text())
    if baseline.get("workflow") != "local_residual_augmented_lagrangian_analysis":
        raise ValueError("baseline is not a local-residual AL analysis")
    warm = baseline.get("qualified_warm_start")
    if not isinstance(warm, dict) or not isinstance(
        warm.get("physical_variables"), list
    ):
        raise TypeError("baseline does not contain a qualified physical warm start")
    if baseline.get("scientifically_validated") is not True:
        raise ValueError("baseline must be scientifically validated")
    return baseline


def candidate_name(history_size, checkpoint_patience):
    return f"history-{history_size:03d}-patience-{checkpoint_patience:03d}"


def main():
    args = parse_args()
    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_non_gpu:
        raise RuntimeError(
            f"JAX selected {backend!r}, not 'gpu'. Select an NVIDIA GPU runtime."
        )
    accelerator_platform = "gpu" if backend == "gpu" else backend
    baseline = load_baseline(args.baseline_analysis)
    problem = baseline.get("problem") or {}
    if problem.get("name") != "engineering":
        raise ValueError("the replay baseline must use the engineering problem")

    spec = get_problem("engineering")
    settings = objective_metadata(spec, "full-engineering")
    surface, base_curves, field, components, cpu_objective = build_problem(
        spec, regularized=True, objective_settings=settings
    )
    base_currents = [field.coils[index].current for index in range(spec.ncoils)]
    free_current_indices = np.asarray(
        [index for index, current in enumerate(base_currents) if current.x.size],
        dtype=np.int32,
    )
    ntarget = surface.gamma().size // 3
    nsource = len(field.coils) * spec.nquad
    gpu_config = GpuConfig(
        target_tile_size=min(args.target_tile_size, ntarget),
        source_tile_size=min(args.source_tile_size, nsource),
        vjp_mode="custom",
    )
    raw_data = minimal_coil_data(
        surface,
        base_curves,
        base_currents,
        surface.nfp,
        stellsym=True,
        config=gpu_config,
    )
    data = replace(
        raw_data,
        bases=np.asarray(raw_data.bases),
        transforms=np.asarray(raw_data.transforms),
        current_signs=np.asarray(raw_data.current_signs),
    )
    bridge = ScipyCoilObjectiveBridge(
        data,
        free_current_indices=free_current_indices,
        objective_kwargs={
            "length_target": None,
            "length_weight": settings["length_weight"],
        },
        current_scale=args.current_scale,
        config=gpu_config,
    )
    physical_warm = np.asarray(
        baseline["qualified_warm_start"]["physical_variables"], dtype=float
    )
    if physical_warm.shape != bridge.physical_initial_x.shape:
        raise ValueError(
            f"warm start must have shape {bridge.physical_initial_x.shape}, "
            f"got {physical_warm.shape}"
        )
    warm_x = physical_warm / bridge.coordinate_scales

    refinement_settings = target_envelope_objective_settings(
        settings, args.target_relative_tolerance, 1.0
    )
    refinement_kwargs = objective_call_kwargs(refinement_settings)
    quality_kwargs = target_envelope_residual_kwargs(
        settings,
        args.target_relative_tolerance,
        {
            "distance": 1e-4,
            "curvature": 1e-3,
            "mean_squared_curvature": 1e-3,
        },
    )

    def objective(x):
        curve_dofs, currents = bridge.unpack(x)
        return data.objective(
            curve_dofs, currents, **refinement_kwargs, config=gpu_config
        )

    def quality(x):
        curve_dofs, currents = bridge.unpack(x)
        return data.local_residual_terms(
            curve_dofs, currents, **quality_kwargs, config=gpu_config
        )

    reference_metrics = baseline["final_metrics"]
    target_flux = args.quadratic_flux_target * (1.0 + args.target_relative_tolerance)
    candidates = []
    states = {}
    for history_size in sorted(set(args.history_sizes)):
        for checkpoint_patience in sorted(set(args.checkpoint_patiences)):
            name = candidate_name(history_size, checkpoint_patience)
            target = TargetAwareConfig(
                target_flux=target_flux,
                feasibility_tolerance=1e-12,
                checkpoint_patience=checkpoint_patience,
                infeasible_patience=args.infeasible_patience,
                minimum_relative_improvement=args.minimum_relative_improvement,
            )
            solver_config = DeviceLBFGSConfig(
                history_size=history_size,
                max_iterations=args.max_iterations,
                max_line_search_iterations=args.max_line_search_iterations,
                gradient_tolerance=args.gradient_tolerance,
                target=target,
            )
            solver = DeviceLBFGS(
                objective,
                quality,
                warm_x,
                config=solver_config,
                platform=accelerator_platform,
            )
            start = time.perf_counter()
            solver.compile(warm_x)
            compilation_seconds = time.perf_counter() - start
            start = time.perf_counter()
            result = solver.run(warm_x)
            solve_seconds = time.perf_counter() - start
            physical_x = bridge.to_physical_variables(result.x)
            metrics = final_coil_metrics(
                cpu_objective,
                components,
                base_curves,
                field,
                surface,
                physical_x,
                settings,
            )
            engineering = engineering_target_validation(
                metrics, args.target_relative_tolerance
            )
            flux = quadratic_flux_target_validation(
                metrics,
                args.quadratic_flux_target,
                args.target_relative_tolerance,
            )
            qoi_agreement = qoi_backend_agreement(
                metrics,
                reference_metrics["cpu"],
                args.target_relative_tolerance,
            )
            objective_agreement = qoi_agreement["comparisons"]["objective"]
            scientific_passed = engineering["passed"] and flux["passed"]
            candidates.append(
                {
                    "name": name,
                    "history_size": history_size,
                    "checkpoint_patience": checkpoint_patience,
                    "compilation_seconds": compilation_seconds,
                    "optimization": device_refinement_summary(result, solve_seconds),
                    "final_metrics": metrics,
                    "engineering_validation": engineering,
                    "quadratic_flux_validation": flux,
                    "cpu_qoi_agreement": qoi_agreement,
                    "objective_agreement": objective_agreement,
                    "scientifically_validated": bool(scientific_passed),
                    "physical_variables": physical_x.tolist(),
                }
            )
            states[name] = physical_x

    qualified = [
        index
        for index, candidate in enumerate(candidates)
        if candidate["scientifically_validated"]
    ]
    if not qualified:
        winner_index = None
    else:
        winner_index = min(
            qualified,
            key=lambda index: (
                not candidates[index]["objective_agreement"]["passed"],
                candidates[index]["final_metrics"]["quadratic_flux"],
                candidates[index]["optimization"]["seconds"],
                candidates[index]["name"],
            ),
        )
    winner = None if winner_index is None else candidates[winner_index]

    visualization = None
    visualization_dir = None if args.no_visualization else args.visualization_dir
    if (
        visualization_dir is None
        and args.output is not None
        and not args.no_visualization
    ):
        visualization_dir = args.output.parent
    if winner is not None and visualization_dir is not None:
        visualization_dir.mkdir(parents=True, exist_ok=True)
        visualization = export_final_design_visualization(
            cpu_objective,
            field,
            surface,
            states[winner["name"]],
            visualization_dir / f"device-lbfgs-replay-{winner['name']}",
        )

    output = {
        "schema_version": 1,
        "workflow": "device_lbfgs_replay_sweep",
        "problem": spec.as_dict(),
        "baseline": {
            "analysis_file": args.baseline_analysis.name,
            "archive_sha256": baseline["archive_sha256"],
            "revision": baseline["revision"],
            "warm_start_source_backend": baseline["qualified_warm_start"][
                "source_backend"
            ],
            "reference_metrics": reference_metrics,
        },
        "sweep": {
            "history_sizes": sorted(set(args.history_sizes)),
            "checkpoint_patiences": sorted(set(args.checkpoint_patiences)),
            "infeasible_patience": args.infeasible_patience,
            "minimum_relative_improvement": args.minimum_relative_improvement,
            "max_iterations": args.max_iterations,
            "max_line_search_iterations": args.max_line_search_iterations,
            "candidate_count": len(candidates),
        },
        "validation_policy": {
            "target_relative_tolerance": args.target_relative_tolerance,
            "quadratic_flux_target": args.quadratic_flux_target,
            "quadratic_flux_allowed_boundary": target_flux,
            "scientific_rule": "quadratic flux and all engineering targets pass",
            "ranking": (
                "prefer scientific candidates with CPU-objective agreement; "
                "then minimum quadratic flux, solve time, and name"
            ),
        },
        "gpu_configuration": {
            "target_tile_size": gpu_config.target_tile_size,
            "source_tile_size": gpu_config.source_tile_size,
            "vjp_mode": gpu_config.vjp_mode,
            "accelerator_platform": accelerator_platform,
        },
        "execution_platform": accelerator_platform,
        "candidates": candidates,
        "qualified_indices": qualified,
        "winner_index": winner_index,
        "winner": winner,
        "scientifically_validated": bool(
            winner is not None and winner["scientifically_validated"]
        ),
        "all_gates_passed": bool(
            backend == "gpu"
            and winner is not None
            and winner["scientifically_validated"]
            and winner["objective_agreement"]["passed"]
        ),
        "visualization": visualization,
        "environment": environment(),
        "nvidia_smi": nvidia_smi(),
    }
    rendered = json.dumps(output, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
