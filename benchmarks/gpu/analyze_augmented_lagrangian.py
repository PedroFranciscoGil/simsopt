"""Validate an augmented-Lagrangian archive and generate report figures."""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_extended_convergence import appended_vtk_array, surface_shape

RESULT_NAME = "stress-augmented-lagrangian.json"
SURFACE_NAMES = (
    "stress-augmented-lagrangian-cpu_final_surface.vts",
    "stress-augmented-lagrangian-gpu_final_surface.vts",
)
COIL_NAMES = (
    "stress-augmented-lagrangian-cpu_final_coils.vtu",
    "stress-augmented-lagrangian-gpu_final_coils.vtu",
)
CONSTRAINT_NAMES = (
    "coil_coil_distance_penalty",
    "coil_surface_distance_penalty",
    "curvature_penalty",
    "mean_squared_curvature_penalty",
)
COLORS = {"cpu": "#2456a6", "gpu": "#d46715"}
CONSTRAINT_COLORS = ("#5b4b9a", "#2c8c78", "#bc5a45", "#8a7426")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def _finite(value, label):
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")


def read_archive(archive):
    """Read and structurally validate the result and visualization payloads."""
    with ZipFile(archive) as artifact:
        names = artifact.namelist()
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate member names")
        if any(
            PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            for name in names
        ):
            raise ValueError("archive contains an unsafe member path")
        members = set(names)
        required = {RESULT_NAME, *SURFACE_NAMES, *COIL_NAMES}
        missing = required - members
        if missing:
            raise ValueError(
                "archive is missing required files: " + ", ".join(sorted(missing))
            )
        result = json.loads(artifact.read(RESULT_NAME))
        surfaces = {name: artifact.read(name) for name in SURFACE_NAMES}
        coils = {name: artifact.read(name) for name in COIL_NAMES}

    if result.get("schema_version") != 5:
        raise ValueError("augmented-Lagrangian schema version 5 is required")
    if result.get("method", {}).get("name") != (
        "equality_zero_penalty_augmented_lagrangian"
    ):
        raise ValueError("archive contains the wrong optimization method")
    if tuple(result["method"]["constraint_names"]) != CONSTRAINT_NAMES:
        raise ValueError("constraint name/order contract does not match")
    if any(not payload for payload in (*surfaces.values(), *coils.values())):
        raise ValueError("final VTK payloads must be nonempty")
    for backend in ("cpu", "gpu"):
        optimization = result[backend]["optimization"]
        history = optimization["outer_history"]
        if len(history) != optimization["outer_iterations"] or not history:
            raise ValueError(f"{backend} outer history is incomplete")
        for expected_iteration, record in enumerate(history, start=1):
            if record["outer_iteration"] != expected_iteration:
                raise ValueError(f"{backend} outer iterations are not contiguous")
            for key in (
                "augmented_lagrangian",
                "base_objective",
                "gradient_norm",
                "constraint_norm_infinity",
                "inner_seconds",
            ):
                _finite(record[key], f"{backend} {key}")
            for key in (
                "constraints",
                "multipliers_before",
                "multipliers_after",
                "penalties_before",
                "penalties_after",
            ):
                values = np.asarray(record[key], dtype=float)
                if values.shape != (len(CONSTRAINT_NAMES),) or not np.all(
                    np.isfinite(values)
                ):
                    raise ValueError(f"{backend} {key} has invalid values")
    return result, surfaces


def _positive(values, floor=None):
    values = np.asarray(values, dtype=float)
    if floor is None:
        floor = np.finfo(float).tiny
    return np.maximum(values, floor)


def plot_convergence(result, output):
    """Plot outer objectives, residuals, individual constraints, and penalties."""
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 6.3), constrained_layout=True)
    for backend in ("cpu", "gpu"):
        history = result[backend]["optimization"]["outer_history"]
        outer = np.arange(1, len(history) + 1)
        color = COLORS[backend]
        axes[0, 0].semilogy(
            outer,
            _positive([record["base_objective"] for record in history]),
            marker="o",
            color=color,
            label=f"{backend.upper()} base $f$",
        )
        axes[0, 0].semilogy(
            outer,
            _positive([record["augmented_lagrangian"] for record in history]),
            linestyle="--",
            color=color,
            label=f"{backend.upper()} $\\mathcal{{L}}_A$",
        )
        axes[0, 1].semilogy(
            outer,
            _positive([record["gradient_norm"] for record in history]),
            marker="o",
            color=color,
            label=f"{backend.upper()} $\\|\\nabla\\mathcal{{L}}_A\\|_2$",
        )
        axes[0, 1].semilogy(
            outer,
            _positive([record["constraint_norm_infinity"] for record in history]),
            linestyle="--",
            color=color,
            label=f"{backend.upper()} $\\|c\\|_\\infty$",
        )
        linestyle = "-" if backend == "cpu" else "--"
        for index, (name, constraint_color) in enumerate(
            zip(CONSTRAINT_NAMES, CONSTRAINT_COLORS, strict=True)
        ):
            axes[1, 0].semilogy(
                outer,
                _positive([record["constraints"][index] for record in history], 1e-12),
                linestyle=linestyle,
                color=constraint_color,
                label=(
                    name.replace("_penalty", "").replace("_", " ")
                    if backend == "cpu"
                    else None
                ),
            )
            axes[1, 1].semilogy(
                outer,
                [record["penalties_after"][index] for record in history],
                linestyle=linestyle,
                color=constraint_color,
                label=(
                    name.replace("_penalty", "").replace("_", " ")
                    if backend == "cpu"
                    else None
                ),
            )

    axes[0, 0].set_title("Outer objective endpoints")
    axes[0, 0].set_ylabel("Objective value")
    axes[0, 1].set_title("Stationarity versus aggregate constraints")
    axes[0, 1].set_ylabel("Norm")
    axes[1, 0].set_title("Zero-target penalty constraints")
    axes[1, 0].set_ylabel("Constraint value")
    axes[1, 0].axhline(
        result["solver"]["constraint_tolerance"],
        color="#333333",
        linestyle=":",
        linewidth=1.0,
    )
    axes[1, 1].set_title("Componentwise penalty escalation")
    axes[1, 1].set_ylabel(r"Penalty $\mu_i$")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    axes[0, 1].legend(frameon=False, fontsize=7)
    axes[1, 0].legend(frameon=False, fontsize=6)
    axes[1, 1].legend(frameon=False, fontsize=6)
    axes[1, 0].text(
        0.02,
        0.04,
        "solid: CPU; dashed: GPU",
        transform=axes[1, 0].transAxes,
        fontsize=7,
    )
    axes[1, 1].text(
        0.02,
        0.94,
        "solid: CPU; dashed: GPU",
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize=7,
    )
    for axis in axes.flat:
        axis.set_xlabel("Augmented-Lagrangian outer iteration")
        axis.set_xticks(np.arange(1, 9))
        axis.grid(True, which="both", color="#d8d8d8", linewidth=0.5)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_performance(result, output):
    """Plot outer-loop wall times/evaluations and per-stage speedup."""
    cpu = result["cpu"]["optimization"]["outer_history"]
    gpu = result["gpu"]["optimization"]["outer_history"]
    outer = np.arange(1, len(cpu) + 1)
    width = 0.36
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.25), constrained_layout=True)
    for backend, history, shift in (
        ("cpu", cpu, -width / 2),
        ("gpu", gpu, width / 2),
    ):
        axes[0].bar(
            outer + shift,
            [record["inner_seconds"] for record in history],
            width,
            color=COLORS[backend],
            label=backend.upper(),
        )
        axes[1].bar(
            outer + shift,
            [record["inner_evaluations"] for record in history],
            width,
            color=COLORS[backend],
            label=backend.upper(),
        )
    speedups = [
        cpu_record["inner_seconds"] / gpu_record["inner_seconds"]
        for cpu_record, gpu_record in zip(cpu, gpu, strict=True)
    ]
    axes[2].bar(outer, speedups, color="#4a9d62", width=0.64)
    axes[2].axhline(3.0, color="#7b2f2f", linestyle="--", linewidth=1.0)
    axes[2].axhline(
        result["comparison"]["optimization_speedup"],
        color="#235d3a",
        linestyle=":",
        linewidth=1.2,
        label=f"total {result['comparison']['optimization_speedup']:.2f}x",
    )
    axes[0].set_title("Inner-solve wall time")
    axes[0].set_ylabel("Seconds")
    axes[1].set_title("Physics evaluations")
    axes[1].set_ylabel("Evaluations")
    axes[2].set_title("Warm speedup by outer step")
    axes[2].set_ylabel("CPU time / GPU time")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xlabel("Outer iteration")
        axis.set_xticks(outer)
        axis.grid(True, axis="y", color="#d8d8d8", linewidth=0.5)
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


def analysis_summary(archive, result, surfaces):
    cpu_surface, gpu_surface = surface_fields(surfaces)
    histories = {
        backend: result[backend]["optimization"]["outer_history"]
        for backend in ("cpu", "gpu")
    }
    return {
        "schema_version": 1,
        "workflow": "augmented_lagrangian_analysis",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "revision": result["environment"]["simsopt_revision"],
        "jax_backend": result["environment"]["jax_backend"],
        "initial_parity": result["initial_parity"],
        "acceptance_gates": result["acceptance_gates"],
        "performance": {
            "compilation_seconds": result["gpu_configuration"]["compilation_seconds"],
            "warm_speedup": result["comparison"]["optimization_speedup"],
            "amortized_speedup": result["comparison"]["amortized_speedup"],
            "stage_speedups": [
                cpu_record["inner_seconds"] / gpu_record["inner_seconds"]
                for cpu_record, gpu_record in zip(
                    histories["cpu"], histories["gpu"], strict=True
                )
            ],
        },
        "final_metrics": {
            backend: result[backend]["final_metrics"] for backend in ("cpu", "gpu")
        },
        "final_optimization": {
            backend: {
                key: result[backend]["optimization"][key]
                for key in (
                    "success",
                    "seconds",
                    "outer_iterations",
                    "total_inner_iterations",
                    "total_evaluations",
                    "final_base_objective",
                    "final_gradient_norm",
                    "final_constraint_norm_infinity",
                    "final_constraints",
                    "final_lagrange_multipliers",
                    "final_penalties",
                )
            }
            for backend in ("cpu", "gpu")
        },
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
            "maximum_absolute_difference": float(
                np.max(np.abs(gpu_surface - cpu_surface))
            ),
        },
    }


def main():
    args = parse_args()
    result, surfaces = read_archive(args.archive)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_convergence(result, args.output_dir / "augmented_lagrangian_convergence.png")
    plot_performance(result, args.output_dir / "augmented_lagrangian_performance.png")
    plot_surface_fields(
        surfaces, args.output_dir / "augmented_lagrangian_surface_field.png"
    )
    summary = analysis_summary(args.archive, result, surfaces)
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    (args.output_dir / "augmented_lagrangian_analysis_summary.json").write_text(
        encoded + "\n"
    )
    print(encoded)


if __name__ == "__main__":
    main()
