"""Validate a feasibility-study archive and generate report-ready figures."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_extended_convergence import appended_vtk_array, surface_shape

SWEEP_SUMMARY = "solver-sweep/solver-feasibility-sweep.json"
CONTINUATION_SUMMARY = "penalty-continuation/penalty-continuation-summary.json"
SURFACE_NAMES = (
    "penalty-continuation/cpu_final_surface.vts",
    "penalty-continuation/gpu_final_surface.vts",
)
COIL_NAMES = (
    "penalty-continuation/cpu_final_coils.vtu",
    "penalty-continuation/gpu_final_coils.vtu",
)
COLORS = {"cpu": "#2456a6", "gpu": "#d46715"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def read_archive(archive):
    """Read and validate all summaries, stage results, and final VTK payloads."""
    with ZipFile(archive) as artifact:
        members = set(artifact.namelist())
        required = {SWEEP_SUMMARY, CONTINUATION_SUMMARY, *SURFACE_NAMES, *COIL_NAMES}
        missing = required - members
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"archive is missing required files: {names}")
        sweep = json.loads(artifact.read(SWEEP_SUMMARY))
        continuation = json.loads(artifact.read(CONTINUATION_SUMMARY))
        stage_names = [
            f"penalty-continuation/{stage['result_file']}"
            for stage in continuation.get("stages", [])
        ]
        missing_stages = set(stage_names) - members
        if missing_stages:
            names = ", ".join(sorted(missing_stages))
            raise ValueError(f"archive is missing stage results: {names}")
        stages = [json.loads(artifact.read(name)) for name in stage_names]
        surfaces = {name: artifact.read(name) for name in SURFACE_NAMES}
        coils = {name: artifact.read(name) for name in COIL_NAMES}

    if sweep.get("schema_version") != 1 or sweep.get("workflow") != (
        "scipy_lbfgsb_feasibility_sweep"
    ):
        raise ValueError("solver-sweep schema version 1 is required")
    if continuation.get("schema_version") != 1 or continuation.get("workflow") != (
        "synchronized_engineering_penalty_continuation"
    ):
        raise ValueError("penalty-continuation schema version 1 is required")
    multipliers = continuation["configuration"]["penalty_multipliers"]
    if len(stages) != len(multipliers) or not stages:
        raise ValueError("continuation stages do not match configured multipliers")
    for result, multiplier in zip(stages, multipliers, strict=True):
        if result.get("schema_version") != 4:
            raise ValueError("comparison schema version 4 is required")
        if result["objective"]["engineering_weight_multiplier"] != multiplier:
            raise ValueError("stage penalty multiplier does not match its summary")
    if any(not payload for payload in (*surfaces.values(), *coils.values())):
        raise ValueError("final VTK payloads must be nonempty")
    return sweep, continuation, stages, surfaces


def feasibility_tolerances(configuration):
    return {
        "maximum_curvature": configuration["curvature_feasibility_tolerance"],
        "maximum_mean_squared_curvature": configuration[
            "mean_squared_curvature_feasibility_tolerance"
        ],
        "minimum_coil_coil_distance": configuration["distance_feasibility_tolerance"],
        "minimum_coil_surface_distance": configuration[
            "distance_feasibility_tolerance"
        ],
    }


def normalized_constraint_ratios(metrics, configuration):
    violations = metrics["coil_constraints"]["violations"]
    tolerances = feasibility_tolerances(configuration)
    ratios = {}
    for name, tolerance in tolerances.items():
        violation = violations[name]
        ratios[name] = (
            violation / tolerance
            if tolerance
            else 0.0
            if violation == 0.0
            else math.inf
        )
    return ratios


def backend_metrics(container, backend):
    if backend == "cpu":
        return container["cpu"]["final_metrics"]
    return container["gpu"]["final_metrics_from_cpu_oracle"]


def plot_solver_sweep(sweep, output):
    candidates = sweep["candidates"]
    labels = [
        f"m={candidate['maxcor']}\nls={candidate['maxls']}" for candidate in candidates
    ]
    x = np.arange(len(candidates))
    speedups = [
        candidate["cpu"]["optimization"]["seconds"]
        / candidate["gpu"]["optimization"]["seconds"]
        for candidate in candidates
    ]
    ratios = {
        backend: [
            max(
                normalized_constraint_ratios(
                    backend_metrics(candidate, backend), sweep["configuration"]
                ).values()
            )
            for candidate in candidates
        ]
        for backend in ("cpu", "gpu")
    }

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.35), constrained_layout=True)
    bars = axes[0].bar(x, speedups, color="#4a9d62", width=0.66)
    winner = sweep["winner"]["index"]
    bars[winner].set_edgecolor("black")
    bars[winner].set_linewidth(1.8)
    axes[0].axhline(3.0, color="#7b2f2f", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Warm optimization speedup")
    axes[0].set_title("Solver-setting performance")
    axes[0].set_ylim(0.0, max(speedups) * 1.18)

    width = 0.38
    axes[1].bar(
        x - width / 2,
        ratios["cpu"],
        width,
        color=COLORS["cpu"],
        label="CPU",
    )
    axes[1].bar(
        x + width / 2,
        ratios["gpu"],
        width,
        color=COLORS["gpu"],
        label="GPU",
    )
    axes[1].axhline(1.0, color="#7b2f2f", linestyle="--", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Worst violation / tolerance")
    axes[1].set_title("Absolute feasibility after 25 iterations")
    axes[1].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("L-BFGS-B configuration")
        axis.grid(True, axis="y", which="both", color="#d8d8d8", linewidth=0.5)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_continuation(continuation, stages, output):
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), constrained_layout=True)
    multipliers = continuation["configuration"]["penalty_multipliers"]
    offsets = np.cumsum(
        [
            0,
            *[
                len(stage["cpu"]["optimization"]["iteration_history"])
                for stage in stages[:-1]
            ],
        ]
    )
    for backend in ("cpu", "gpu"):
        for stage_index, (stage, offset) in enumerate(
            zip(stages, offsets, strict=True)
        ):
            history = stage[backend]["optimization"]["iteration_history"]
            iterations = offset + np.arange(1, len(history) + 1)
            values = [entry["objective"] for entry in history]
            axes[0, 0].semilogy(
                iterations,
                values,
                color=COLORS[backend],
                label=backend.upper() if stage_index == 0 else None,
            )
    for boundary in offsets[1:]:
        axes[0, 0].axvline(boundary, color="#777777", linestyle=":", linewidth=0.9)
    axes[0, 0].set_title("Accepted-iterate objective histories")
    axes[0, 0].set_xlabel("Cumulative iteration (objective changes by stage)")
    axes[0, 0].set_ylabel("Stage objective")
    axes[0, 0].legend(frameon=False, fontsize=8)

    x = np.arange(len(stages))
    labels = [f"{multiplier:g}x" for multiplier in multipliers]
    width = 0.38
    maximum_ratios = {"cpu": [], "gpu": []}
    normal_fields = {"cpu": [], "gpu": []}
    for stage in stages:
        for backend in ("cpu", "gpu"):
            metrics = backend_metrics(stage, backend)
            maximum_ratios[backend].append(
                max(
                    normalized_constraint_ratios(
                        metrics, continuation["configuration"]
                    ).values()
                )
            )
            normal_fields[backend].append(
                metrics["normalized_normal_field"]["root_mean_square"]
            )
    for backend, shift in (("cpu", -width / 2), ("gpu", width / 2)):
        axes[0, 1].bar(
            x + shift,
            maximum_ratios[backend],
            width,
            color=COLORS[backend],
            label=backend.upper(),
        )
        axes[1, 0].plot(
            x,
            normal_fields[backend],
            marker="o",
            color=COLORS[backend],
            label=backend.upper(),
        )
    axes[0, 1].set_yscale("log")
    axes[0, 1].axhline(1.0, color="#7b2f2f", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title("Worst absolute constraint violation")
    axes[0, 1].set_ylabel("Violation / tolerance")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Normalized normal-field quality")
    axes[1, 0].set_ylabel(r"RMS $|B\cdot n|/|B|$")
    axes[1, 0].legend(frameon=False, fontsize=8)

    stage_speedups = [
        stage["cpu"]["optimization"]["seconds"]
        / stage["gpu"]["optimization"]["seconds"]
        for stage in stages
    ]
    total_speedup = continuation["totals"]["speedup_excluding_compilation"]
    speed_labels = [*labels, "total"]
    axes[1, 1].bar(
        np.arange(len(speed_labels)),
        [*stage_speedups, total_speedup],
        color=["#4a9d62"] * len(labels) + ["#235d3a"],
        width=0.66,
    )
    axes[1, 1].axhline(3.0, color="#7b2f2f", linestyle="--", linewidth=1.0)
    axes[1, 1].set_xticks(np.arange(len(speed_labels)), speed_labels)
    axes[1, 1].set_title("Warm optimization speedup")
    axes[1, 1].set_ylabel("CPU time / GPU time")

    for axis in (axes[0, 1], axes[1, 0]):
        axis.set_xticks(x, labels)
        axis.set_xlabel("Engineering penalty multiplier")
    for axis in axes.flat:
        axis.grid(True, which="both", color="#d8d8d8", linewidth=0.5)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def surface_fields(surfaces):
    fields = []
    for name in SURFACE_NAMES:
        payload = surfaces[name]
        values = appended_vtk_array(payload, "B_dot_n_over_abs_B")
        shape = surface_shape(payload)
        fields.append(values.reshape(shape, order="F").squeeze().T)
    return fields


def plot_surface_fields(surfaces, output):
    cpu, gpu = surface_fields(surfaces)
    difference = gpu - cpu
    field_limit = max(float(np.max(np.abs(cpu))), float(np.max(np.abs(gpu))))
    difference_limit = float(np.max(np.abs(difference)))
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), constrained_layout=True)
    images = []
    for axis, values, title in zip(
        axes,
        (cpu, gpu, difference),
        ("CPU final", "GPU final", "GPU minus CPU"),
        strict=True,
    ):
        limit = difference_limit if title == "GPU minus CPU" else field_limit
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        images.append(image)
        axis.set_title(title)
        axis.set_xlabel(r"poloidal index $\theta$")
    axes[0].set_ylabel(r"toroidal index $\phi$")
    figure.colorbar(images[1], ax=axes[:2], shrink=0.86, label=r"$(B\cdot n)/|B|$")
    figure.colorbar(images[2], ax=axes[2], shrink=0.86, label="difference")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def analysis_summary(archive, sweep, continuation, stages, surfaces):
    cpu_surface, gpu_surface = surface_fields(surfaces)
    cpu_final = backend_metrics(stages[-1], "cpu")
    gpu_final = backend_metrics(stages[-1], "gpu")
    return {
        "schema_version": 1,
        "workflow": "feasibility_study_analysis",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "revision": continuation["environment"]["simsopt_revision"],
        "gpu": continuation["nvidia_smi"],
        "solver_sweep_winner": sweep["winner"],
        "continuation_accepted": continuation["accepted"],
        "acceptance": continuation["acceptance"],
        "totals": continuation["totals"],
        "stage_worst_normalized_violations": {
            backend: [
                max(
                    normalized_constraint_ratios(
                        backend_metrics(stage, backend),
                        continuation["configuration"],
                    ).values()
                )
                for stage in stages
            ]
            for backend in ("cpu", "gpu")
        },
        "final_metrics": {"cpu": cpu_final, "gpu": gpu_final},
        "surface_comparison": {
            "signed_correlation": float(
                np.corrcoef(cpu_surface.ravel(), gpu_surface.ravel())[0, 1]
            ),
            "relative_l2_difference": float(
                np.linalg.norm(gpu_surface - cpu_surface) / np.linalg.norm(cpu_surface)
            ),
            "gpu_smaller_absolute_fraction": float(
                np.mean(np.abs(gpu_surface) < np.abs(cpu_surface))
            ),
        },
    }


def main():
    args = parse_args()
    sweep, continuation, stages, surfaces = read_archive(args.archive)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_solver_sweep(sweep, args.output_dir / "feasibility_solver_sweep.png")
    plot_continuation(
        continuation, stages, args.output_dir / "feasibility_continuation.png"
    )
    plot_surface_fields(surfaces, args.output_dir / "feasibility_surface_field.png")
    summary = analysis_summary(args.archive, sweep, continuation, stages, surfaces)
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    (args.output_dir / "feasibility_analysis_summary.json").write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
