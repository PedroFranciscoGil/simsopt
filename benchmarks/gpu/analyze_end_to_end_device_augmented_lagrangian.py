"""Validate and visualize an end-to-end CPU/device AL comparison archive."""

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

RESULT_NAME = "end-to-end-device-augmented-lagrangian.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def read_archive(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate members")
        if any(
            PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            for name in names
        ):
            raise ValueError("archive contains an unsafe member path")
        matches = [name for name in names if PurePosixPath(name).name == RESULT_NAME]
        if len(matches) != 1:
            raise ValueError(f"archive must contain exactly one {RESULT_NAME}")
        result = json.loads(archive.read(matches[0]))
        payload_names = []
        visualizations = result.get("visualizations")
        if visualizations is not None:
            for backend in ("cpu_final", "gpu_native_final"):
                if backend not in visualizations:
                    raise ValueError(f"missing {backend} visualization")
                for kind in ("surface_vts", "coils_vtu"):
                    basename = PurePosixPath(visualizations[backend][kind]).name
                    found = [
                        name for name in names if PurePosixPath(name).name == basename
                    ]
                    if len(found) != 1 or not archive.read(found[0]):
                        raise ValueError(f"missing or empty visualization {basename}")
                    payload_names.append(basename)
    return result, sorted(payload_names), digest


def validate_result(result):
    if result.get("schema_version") not in (1, 2, 3) or result.get("workflow") != (
        "end_to_end_device_augmented_lagrangian"
    ):
        raise ValueError("end-to-end AL result contract does not match")
    method = result.get("method", {})
    if method.get("refinement_performed") is not False:
        raise ValueError("comparison unexpectedly contains a refinement phase")
    if result["schema_version"] >= 2 and (
        method.get("matched_design_envelope_bounds") is not True
        or not result.get("design_envelope")
    ):
        raise ValueError("matched design-envelope bounds are missing")
    if result["schema_version"] >= 3 and (
        method.get("inner_stationarity_is_diagnostic") is not True
        or method.get("target_aware_outer_checkpoint_selection") is not True
    ):
        raise ValueError("budgeted checkpoint-selection policy is missing")
    if (
        result.get("cpu", {}).get("execution_platform") != "cpu"
        or result.get("gpu_native", {}).get("execution_platform") != "gpu"
        or result["gpu_native"].get("device_resident") is not True
        or result["gpu_native"].get("host_callbacks") != 0
        or method.get("gpu_outer_loop_device_resident") is not True
    ):
        raise ValueError("CPU/device execution placement contract failed")
    required_metrics = {
        "objective",
        "quadratic_flux",
        "normalized_normal_field",
        "coil_constraints",
    }
    for backend in ("cpu", "gpu_native"):
        record = result[backend]
        if not required_metrics <= record.get("final_metrics", {}).keys():
            raise ValueError(f"{backend} final metrics are incomplete")
        optimization = record.get("optimization", {})
        history = optimization.get("history", [])
        if len(history) != optimization.get("outer_iterations"):
            raise ValueError(f"{backend} outer history is incomplete")
        if result["schema_version"] >= 3 and any(
            not item.get("optimizer_variables") for item in history
        ):
            raise ValueError(f"{backend} outer designs are missing")
        for value in (
            optimization.get("seconds"),
            optimization.get("base_objective"),
            record.get("compilation_seconds"),
        ):
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"{backend} contains a non-finite scalar")
        if result["schema_version"] >= 3:
            selection = record.get("checkpoint_selection", {})
            if (
                selection.get("candidate_count") != len(selection.get("candidates", []))
                or not selection.get("selected")
            ):
                raise ValueError(f"{backend} checkpoint selection is incomplete")
    if not result["gpu_native"].get("execution_samples_seconds"):
        raise ValueError("GPU execution samples are missing")
    reproduced = bool(
        result["cpu"]["scientific_validation"]["passed"]
        and result["gpu_native"]["scientific_validation"]["passed"]
    )
    if result.get("scientifically_validated") is not reproduced:
        raise ValueError("scientific validation does not reproduce")
    return result


def plot_performance(result, output):
    cpu = result["cpu"]
    gpu = result["gpu_native"]
    compile_times = [cpu["compilation_seconds"], gpu["compilation_seconds"]]
    execution_times = [
        cpu["optimization"]["seconds"],
        gpu["execution_samples_seconds"][0],
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    positions = np.arange(2)
    axes[0].bar(positions, compile_times, label="compile", color="#7794b8")
    axes[0].bar(
        positions,
        execution_times,
        bottom=compile_times,
        label="optimization",
        color="#d4825b",
    )
    axes[0].set_xticks(positions, ["CPU SciPy AL", "GPU-native AL"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("seconds (log scale)")
    axes[0].set_title("Cold solver end-to-end time")
    axes[0].legend(frameon=False)
    evaluations = [
        cpu["optimization"]["total_evaluations"],
        gpu["optimization"]["total_evaluations"],
    ]
    axes[1].bar(positions, evaluations, color=["#4c78a8", "#f58518"])
    axes[1].set_xticks(positions, ["CPU SciPy AL", "GPU-native AL"])
    axes[1].set_ylabel("objective/gradient evaluations")
    axes[1].set_title("Complete AL work")
    for index, value in enumerate(evaluations):
        axes[1].text(index, value, str(value), ha="center", va="bottom")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_convergence(result, output):
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    for backend, label, color in (
        ("cpu", "CPU SciPy AL", "#4c78a8"),
        ("gpu_native", "GPU-native AL", "#f58518"),
    ):
        history = result[backend]["optimization"]["history"]
        outer = np.asarray([item["outer_iteration"] for item in history])
        base = np.asarray([item["base_objective"] for item in history])
        constraints = np.asarray(
            [item["constraint_norm_infinity"] for item in history]
        )
        axes[0].semilogy(outer, np.maximum(base, 1e-30), "o-", label=label, color=color)
        axes[1].semilogy(
            outer,
            np.maximum(constraints, 1e-30),
            "o-",
            label=label,
            color=color,
        )
    axes[0].set_title("Base objective")
    axes[1].set_title("Raw constraint infinity norm")
    for axis in axes:
        axis.set_xlabel("AL outer iteration")
        axis.grid(True, axis="y", linewidth=0.5, color="#d8d8d8")
        axis.legend(frameon=False)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_final_metrics(result, output):
    names = ("quadratic flux", "normal mean", "normal RMS", "normal max")

    def values(backend):
        metrics = result[backend]["final_metrics"]
        normal = metrics["normalized_normal_field"]
        return np.asarray(
            [
                metrics["quadratic_flux"],
                normal["mean_absolute"],
                normal["root_mean_square"],
                normal["maximum_absolute"],
            ]
        )

    cpu = values("cpu")
    gpu = values("gpu_native")
    positions = np.arange(len(names))
    width = 0.38
    figure, axis = plt.subplots(figsize=(8.4, 4.5), constrained_layout=True)
    axis.bar(positions - width / 2, cpu, width, label="CPU SciPy AL")
    axis.bar(positions + width / 2, gpu, width, label="GPU-native AL")
    axis.set_xticks(positions, names, rotation=15, ha="right")
    axis.set_yscale("log")
    axis.set_ylabel("final value (log scale)")
    axis.set_title("No-refinement final magnetic metrics")
    axis.legend(frameon=False)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main():
    args = parse_args()
    result, payload_names, digest = read_archive(args.archive)
    validate_result(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        "end_to_end_al_performance.png",
        "end_to_end_al_convergence.png",
        "end_to_end_al_final_metrics.png",
    ]
    plot_performance(result, args.output_dir / figures[0])
    plot_convergence(result, args.output_dir / figures[1])
    plot_final_metrics(result, args.output_dir / figures[2])
    summary = {
        "schema_version": 1,
        "workflow": "end_to_end_device_augmented_lagrangian_analysis",
        "archive_sha256": digest,
        "environment": result["environment"],
        "method": result["method"],
        "problem": result["problem"],
        "solver": result["solver"],
        "design_envelope": result.get("design_envelope"),
        "validation_policy": result["validation_policy"],
        "cpu": {
            key: value
            for key, value in result["cpu"].items()
            if key != "physical_variables"
        },
        "gpu_native": {
            key: value
            for key, value in result["gpu_native"].items()
            if key != "physical_variables"
        },
        "comparison": result["comparison"],
        "scientifically_validated": result["scientifically_validated"],
        "technical_trajectory_agreement": result[
            "technical_trajectory_agreement"
        ],
        "visualization_files": payload_names,
        "figures": figures,
    }
    output = args.output_dir / "end_to_end_al_analysis_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
