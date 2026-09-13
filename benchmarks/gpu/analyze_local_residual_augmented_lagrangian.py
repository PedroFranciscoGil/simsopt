"""Validate and visualize a local-residual AL result archive."""

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

RESULT_NAME = "local-residual-augmented-lagrangian.json"
FAMILIES = (
    "coil_coil_distance",
    "coil_surface_distance",
    "curvature",
    "mean_squared_curvature",
)
COLORS = {"cpu": "#2456a6", "gpu": "#d46715"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def _finite(value, label):
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")


def read_archive(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with ZipFile(path) as artifact:
        names = artifact.namelist()
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate member names")
        if any(
            PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            for name in names
        ):
            raise ValueError("archive contains an unsafe member path")
        matches = [name for name in names if PurePosixPath(name).name == RESULT_NAME]
        if len(matches) != 1:
            raise ValueError(f"archive must contain exactly one {RESULT_NAME}")
        result = json.loads(artifact.read(matches[0]))
        visualizations = result.get("visualizations")
        if not isinstance(visualizations, dict):
            raise TypeError("final visualization contract is missing")
        payloads = {}
        for backend in ("cpu_final", "gpu_final"):
            files = visualizations.get(backend, {})
            if set(files) < {"surface_vts", "coils_vtu"}:
                raise ValueError(f"{backend} visualization contract is incomplete")
            for kind in ("surface_vts", "coils_vtu"):
                basename = PurePosixPath(files[kind]).name
                candidates = [
                    name for name in names if PurePosixPath(name).name == basename
                ]
                if len(candidates) != 1:
                    raise ValueError(f"archive must contain exactly one {basename}")
                payload = artifact.read(candidates[0])
                if not payload:
                    raise ValueError(f"{basename} must be nonempty")
                payloads[(backend, kind)] = payload
    return result, payloads, digest


def validate_result(result):
    if result.get("schema_version") not in (1, 2) or result.get("workflow") != (
        "local_residual_augmented_lagrangian"
    ):
        raise ValueError("local-residual AL result contract does not match")
    if result.get("method", {}).get("name") != (
        "local_residual_equality_augmented_lagrangian"
    ):
        raise ValueError("optimization method does not match")
    layout = result.get("residual_contract", {}).get("layout", {})
    if tuple(layout) != FAMILIES:
        raise ValueError("residual family order differs from the contract")
    offset = 0
    for family in FAMILIES:
        item = layout[family]
        if item.get("start") != offset or item.get("stop", 0) <= offset:
            raise ValueError(f"invalid residual layout for {family}")
        offset = item["stop"]
    if offset != result["dimensions"]["local_residuals"]:
        raise ValueError("residual dimension differs from layout")
    solver = result.get("solver", {})
    if (
        solver.get("require_inner_stationarity") is not True
        or solver.get("history_vector_mode") != "summary"
        or solver.get("penalty_update_mode") != "global"
    ):
        raise ValueError("required safeguard or large-vector policy is missing")
    if result["schema_version"] >= 2:
        if result.get("method", {}).get("constraint_transform") not in (
            "identity",
            "smooth_abs",
        ):
            raise ValueError("schema-2 local transform is invalid")
        continuation = result.get("residual_scaling", {}).get("continuation", {})
        reduction_factor = continuation.get("reduction_factor")
        if reduction_factor is None or not 0 < float(reduction_factor) <= 1:
            raise ValueError("schema-2 scale continuation contract is invalid")
        relative_tolerance = solver.get("inner_stationarity_relative_tolerance")
        if relative_tolerance is not None and not 0 < float(relative_tolerance) < 1:
            raise ValueError("schema-2 relative first-order contract is invalid")
        scientific_validation = result.get("scientific_validation", {})
        scientific_components = (
            scientific_validation.get("cpu_engineering_targets", {}),
            scientific_validation.get("gpu_engineering_targets", {}),
            scientific_validation.get("cpu_gpu_quantity_of_interest_agreement", {}),
        )
        reproduced_validation = all(
            component.get("passed") is True for component in scientific_components
        )
        if (
            result.get("scientifically_validated") is not reproduced_validation
            or scientific_validation.get("passed") is not reproduced_validation
        ):
            raise ValueError("scientific-validation summary does not reproduce")
        validation_policy = result.get("validation_policy", {})
        target_tolerance = validation_policy.get("target_relative_tolerance")
        if (
            validation_policy.get("name") != "physical_quantity_target_envelope"
            or target_tolerance is None
            or not 0 < float(target_tolerance) < 1
        ):
            raise ValueError("schema-2 physical target validation policy is invalid")
        if not isinstance(result.get("diagnostic_checks"), dict):
            raise ValueError("schema-2 optimizer diagnostics are missing")
    for backend in ("cpu", "gpu"):
        optimization = result[backend]["optimization"]
        history = optimization.get("outer_history", [])
        if not history or len(history) != optimization["outer_iterations"]:
            raise ValueError(f"{backend} outer history is incomplete")
        for expected, record in enumerate(history, start=1):
            if record.get("outer_iteration") != expected:
                raise ValueError(f"{backend} outer history is not contiguous")
            for key in (
                "base_objective",
                "augmented_lagrangian",
                "gradient_norm",
                "constraint_norm_infinity",
                "scaled_constraint_norm_infinity",
                "inner_seconds",
            ):
                _finite(record.get(key), f"{backend} {key}")
            if "constraints" in record or "penalties_after" in record:
                raise ValueError("large history vectors were not compacted")
            summaries = record.get("vector_summaries", {})
            for name in (
                "constraints",
                "scaled_constraints",
                "multipliers_before",
                "multipliers_after",
                "penalties_before",
                "penalties_after",
            ):
                if summaries.get(name, {}).get("size") != offset:
                    raise ValueError(f"{backend} {name} summary has wrong size")
        final_metrics = result[backend].get("final_metrics", {})
        if not {
            "objective",
            "normalized_normal_field",
            "coil_constraints",
        } <= set(final_metrics):
            raise ValueError(f"{backend} final metric contract is incomplete")
        for summary_name in (
            "final_residual_families",
            "final_scaled_residual_families",
            "final_multiplier_families",
            "final_penalty_families",
        ):
            if tuple(optimization.get(summary_name, {})) != FAMILIES:
                raise ValueError(f"{backend} {summary_name} is incomplete")
        if (
            result["schema_version"] >= 2
            and tuple(optimization.get("final_constraint_scale_families", {}))
            != FAMILIES
        ):
            raise ValueError(f"{backend} final constraint scales are incomplete")
    gates = result.get("acceptance_gates", {})
    if result.get("all_gates_passed") is not all(
        item.get("passed") is True for item in gates.values()
    ):
        raise ValueError("all-gates summary does not reproduce")
    return result


def surface_fields(payloads):
    fields = []
    for backend in ("cpu_final", "gpu_final"):
        payload = payloads[(backend, "surface_vts")]
        values = appended_vtk_array(payload, "B_dot_n_over_abs_B")
        fields.append(values.reshape(surface_shape(payload), order="F").squeeze().T)
    return fields


def plot_optimization(result, output):
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), constrained_layout=True)
    for backend in ("cpu", "gpu"):
        history = result[backend]["optimization"]["outer_history"]
        outer = np.arange(1, len(history) + 1)
        color = COLORS[backend]
        axes[0, 0].semilogy(
            outer,
            np.maximum([item["base_objective"] for item in history], 1e-30),
            "o-",
            color=color,
            label=backend.upper(),
        )
        axes[0, 1].semilogy(
            outer,
            np.maximum([item["gradient_norm_infinity"] for item in history], 1e-30),
            "o-",
            color=color,
            label=f"{backend.upper()} gradient",
        )
        axes[0, 1].semilogy(
            outer,
            np.maximum([item["constraint_norm_infinity"] for item in history], 1e-30),
            "--",
            color=color,
            label=f"{backend.upper()} residual",
        )
        axes[1, 0].bar(
            outer + (-0.18 if backend == "cpu" else 0.18),
            [item["inner_seconds"] for item in history],
            0.36,
            color=color,
            label=backend.upper(),
        )
    labels = ("coil--coil", "coil--surface", "curvature", "MSC")
    x = np.arange(len(FAMILIES))
    width = 0.36
    for backend, shift in (("cpu", -width / 2), ("gpu", width / 2)):
        summaries = result[backend]["optimization"]["final_residual_families"]
        axes[1, 1].bar(
            x + shift,
            [summaries[name]["maximum"] for name in FAMILIES],
            width,
            color=COLORS[backend],
            label=backend.upper(),
        )
    axes[0, 0].set_title("Base objective")
    axes[0, 0].set_ylabel("flux + weighted length")
    axes[0, 1].set_title("Stationarity and physical residual")
    axes[0, 1].set_ylabel("infinity norm")
    axes[1, 0].set_title("Inner-solve wall time")
    axes[1, 0].set_ylabel("seconds")
    axes[1, 1].set_title("Final residual maxima by family")
    axes[1, 1].set_ylabel("units of feasibility allowance")
    axes[1, 1].set_yscale("symlog", linthresh=1e-12)
    axes[1, 1].set_xticks(x, labels, rotation=18)
    for axis in axes.flat:
        axis.grid(True, axis="y", color="#d8d8d8", linewidth=0.5)
        axis.legend(frameon=False, fontsize=7)
    axes[0, 0].set_xlabel("outer iteration")
    axes[0, 1].set_xlabel("outer iteration")
    axes[1, 0].set_xlabel("outer iteration")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_performance(result, output):
    cpu = result["cpu"]["optimization"]
    gpu = result["gpu"]["optimization"]
    compile_cpu = result["gpu_configuration"]["cpu_compilation_seconds"]
    compile_gpu = result["gpu_configuration"]["gpu_compilation_seconds"]
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    axes[0].bar(
        ("CPU solve", "GPU solve", "CPU compile", "GPU compile"),
        (cpu["seconds"], gpu["seconds"], compile_cpu, compile_gpu),
        color=(COLORS["cpu"], COLORS["gpu"], "#8aa7d3", "#e2a477"),
    )
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("seconds")
    axes[0].set_title("Whole-run timing")
    axes[1].bar(
        ("warm solve", "amortized"),
        (
            result["comparison"]["optimization_speedup"],
            result["comparison"]["amortized_speedup"],
        ),
        color=("#4a9d62", "#8d6bb1"),
    )
    axes[1].axhline(3.0, color="#7b2f2f", linestyle="--", label="3x gate")
    axes[1].set_ylabel("CPU time / GPU time")
    axes[1].set_title("Optimization speedup")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(True, axis="y", color="#d8d8d8", linewidth=0.5)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_surface(payloads, output):
    cpu, gpu = surface_fields(payloads)
    difference = gpu - cpu
    field_limit = max(float(np.max(np.abs(cpu))), float(np.max(np.abs(gpu))), 1e-30)
    difference_limit = max(float(np.max(np.abs(difference))), 1e-30)
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), constrained_layout=True)
    images = []
    for axis, values, title in zip(
        axes,
        (cpu, gpu, difference),
        ("CPU final", "GPU final", "GPU minus CPU"),
        strict=True,
    ):
        limit = difference_limit if title == "GPU minus CPU" else field_limit
        images.append(
            axis.imshow(
                values,
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
        )
        axis.set_title(title)
        axis.set_xlabel(r"poloidal index $\theta$")
    axes[0].set_ylabel(r"toroidal index $\phi$")
    figure.colorbar(images[1], ax=axes[:2], shrink=0.86, label=r"$(B\cdot n)/|B|$")
    figure.colorbar(images[2], ax=axes[2], shrink=0.86, label="difference")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main():
    args = parse_args()
    result, payloads, digest = read_archive(args.archive)
    validate_result(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_optimization(result, args.output_dir / "local_residual_al_convergence.png")
    plot_performance(result, args.output_dir / "local_residual_al_performance.png")
    plot_surface(payloads, args.output_dir / "local_residual_al_surface.png")
    cpu_surface, gpu_surface = surface_fields(payloads)
    summary = {
        "schema_version": 1,
        "workflow": "local_residual_augmented_lagrangian_analysis",
        "archive_sha256": digest,
        "revision": result["environment"]["simsopt_revision"],
        "environment": result["environment"],
        "residual_scaling": result["residual_scaling"],
        "initial_parity": result["initial_parity"],
        "validation_policy": result.get("validation_policy"),
        "scientific_validation": result.get("scientific_validation"),
        "scientifically_validated": result.get("scientifically_validated"),
        "acceptance_gates": result["acceptance_gates"],
        "diagnostic_checks": result.get("diagnostic_checks", {}),
        "all_gates_passed": result["all_gates_passed"],
        "comparison": result["comparison"],
        "final_metrics": {
            backend: result[backend]["final_metrics"] for backend in ("cpu", "gpu")
        },
        "final_optimization": {
            backend: {
                key: value
                for key, value in result[backend]["optimization"].items()
                if key != "outer_history" and key != "final_physical_variables"
            }
            for backend in ("cpu", "gpu")
        },
        "surface_comparison": {
            "signed_correlation": float(
                np.corrcoef(cpu_surface.ravel(), gpu_surface.ravel())[0, 1]
            ),
            "relative_l2_difference": float(
                np.linalg.norm(gpu_surface - cpu_surface)
                / max(np.linalg.norm(cpu_surface), 1e-30)
            ),
            "maximum_absolute_difference": float(
                np.max(np.abs(gpu_surface - cpu_surface))
            ),
        },
        "figures": [
            "local_residual_al_convergence.png",
            "local_residual_al_performance.png",
            "local_residual_al_surface.png",
        ],
    }
    summary_path = args.output_dir / "local_residual_al_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
