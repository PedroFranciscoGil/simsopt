"""Validate and visualize an augmented-Lagrangian conditioning archive."""

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

SUMMARY_NAME = "conditioning-study-summary.json"
EXPECTED_CANDIDATES = (
    "baseline",
    "long-inner",
    "gentle-growth",
    "scaled-gentle-growth",
)
CONSTRAINT_NAMES = (
    "coil_coil_distance_penalty",
    "coil_surface_distance_penalty",
    "curvature_penalty",
    "mean_squared_curvature_penalty",
)
BACKEND_COLORS = {"cpu": "#2456a6", "gpu": "#d46715"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def _finite(value, label):
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")


def _validate_result(result, label, *, require_visualizations=False):
    if result.get("schema_version") not in (6, 7):
        raise ValueError(f"{label} must use augmented-Lagrangian schema 6 or 7")
    method = result.get("method", {})
    if method.get("name") != "equality_zero_penalty_augmented_lagrangian":
        raise ValueError(f"{label} contains the wrong optimization method")
    if tuple(method.get("constraint_names", ())) != CONSTRAINT_NAMES:
        raise ValueError(f"{label} constraint order does not match")
    scaling = result.get("constraint_scaling", {})
    scales = np.asarray(scaling.get("scales", ()), dtype=float)
    if (
        tuple(scaling.get("names", ())) != CONSTRAINT_NAMES
        or scales.shape != (len(CONSTRAINT_NAMES),)
        or not np.all(np.isfinite(scales))
        or np.any(scales <= 0)
    ):
        raise ValueError(f"{label} has invalid constraint scaling")
    if result.get("environment", {}).get("jax_backend") != "gpu":
        raise ValueError(f"{label} did not use a GPU backend")
    if not result.get("nvidia_smi"):
        raise ValueError(f"{label} lacks NVIDIA provenance")

    for backend in ("cpu", "gpu"):
        optimization = result[backend]["optimization"]
        history = optimization["outer_history"]
        if not history or len(history) != optimization["outer_iterations"]:
            raise ValueError(f"{label} {backend} history is incomplete")
        for expected_iteration, record in enumerate(history, start=1):
            if record["outer_iteration"] != expected_iteration:
                raise ValueError(f"{label} {backend} history is not contiguous")
            for key in (
                "inner_seconds",
                "augmented_lagrangian",
                "base_objective",
                "gradient_norm",
                "constraint_norm_infinity",
                "scaled_constraint_norm_infinity",
            ):
                _finite(record[key], f"{label} {backend} {key}")
            for key in (
                "constraints",
                "scaled_constraints",
                "multipliers_before",
                "multipliers_after",
                "penalties_before",
                "penalties_after",
            ):
                values = np.asarray(record[key], dtype=float)
                if values.shape != (len(CONSTRAINT_NAMES),) or not np.all(
                    np.isfinite(values)
                ):
                    raise ValueError(f"{label} {backend} {key} is invalid")

        for key in (
            "final_base_objective",
            "final_gradient_norm",
            "final_raw_constraint_norm_infinity",
            "final_scaled_constraint_norm_infinity",
            "seconds",
        ):
            _finite(optimization[key], f"{label} {backend} {key}")
        metrics = result[backend]["final_metrics"]
        _finite(metrics["objective"], f"{label} {backend} objective")
        for key in ("mean_absolute", "root_mean_square", "maximum_absolute"):
            _finite(
                metrics["normalized_normal_field"][key],
                f"{label} {backend} normalized field {key}",
            )
        violations = metrics["coil_constraints"]["violations"]
        if violations.keys() != result["feasibility_tolerances"].keys():
            raise ValueError(f"{label} {backend} feasibility schema differs")
        for name, value in violations.items():
            _finite(value, f"{label} {backend} {name}")
            if value < 0:
                raise ValueError(f"{label} {backend} violations must be nonnegative")

    if require_visualizations:
        visualizations = result.get("visualizations")
        if not visualizations or set(visualizations) != {"cpu_final", "gpu_final"}:
            raise ValueError(f"{label} lacks CPU/GPU visualization metadata")


def feasibility_ratios(result, backend):
    violations = result[backend]["final_metrics"]["coil_constraints"]["violations"]
    tolerances = result["feasibility_tolerances"]
    return {name: violations[name] / tolerances[name] for name in tolerances}


def candidate_rank(result):
    if not (
        result["acceptance_gates"]["gpu_backend"]["passed"]
        and result["acceptance_gates"]["initial_float64_parity"]["passed"]
    ):
        return (math.inf,) * 7
    ratios = {
        backend: feasibility_ratios(result, backend) for backend in ("cpu", "gpu")
    }
    all_ratios = [value for backend in ratios.values() for value in backend.values()]
    stationary_failures = sum(
        not result[backend]["optimization"]["success"] for backend in ("cpu", "gpu")
    )
    return (
        sum(value > 1.0 for value in all_ratios),
        max(all_ratios),
        sum(all_ratios),
        stationary_failures,
        max(
            result[backend]["optimization"]["final_gradient_norm"]
            for backend in ("cpu", "gpu")
        ),
        sum(
            result[backend]["final_metrics"]["normalized_normal_field"][
                "root_mean_square"
            ]
            for backend in ("cpu", "gpu")
        )
        / 2.0,
        sum(
            result[backend]["optimization"]["total_evaluations"]
            for backend in ("cpu", "gpu")
        ),
    )


def _validate_configuration(result, configuration, label):
    solver = result["solver"]
    for key in (
        "max_outer_iterations",
        "max_inner_iterations",
        "mu_init",
        "tau",
        "mu_max",
    ):
        if solver[key] != configuration[key]:
            raise ValueError(f"{label} {key} differs from its declared candidate")
    if not np.array_equal(
        np.asarray(result["constraint_scaling"]["scales"]),
        np.asarray(configuration["constraint_scales"]),
    ):
        raise ValueError(f"{label} constraint scales differ from its candidate")


def read_archive(archive):
    """Read and cross-check the complete conditioning-study archive."""
    with ZipFile(archive) as artifact:
        names = artifact.namelist()
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate member names")
        if any(
            PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            for name in names
        ):
            raise ValueError("archive contains an unsafe member path")
        if SUMMARY_NAME not in names:
            raise ValueError(f"archive is missing {SUMMARY_NAME}")
        summary = json.loads(artifact.read(SUMMARY_NAME))
        if summary.get("schema_version") != 1 or summary.get("workflow") != (
            "augmented_lagrangian_conditioning_study"
        ):
            raise ValueError("conditioning summary contract does not match")
        candidates = summary.get("candidates", [])
        if len(candidates) != len(EXPECTED_CANDIDATES):
            raise ValueError("conditioning summary must contain four candidates")

        results = []
        required = {SUMMARY_NAME, summary["final_result_file"]}
        for index, candidate in enumerate(candidates):
            if candidate["index"] != index:
                raise ValueError("candidate indices must be contiguous")
            if candidate["configuration"]["name"] != EXPECTED_CANDIDATES[index]:
                raise ValueError("candidate names/order do not match")
            required.add(candidate["result_file"])
        missing = required - set(names)
        if missing:
            raise ValueError(
                "archive is missing result files: " + ", ".join(sorted(missing))
            )

        for index, candidate in enumerate(candidates):
            result = json.loads(artifact.read(candidate["result_file"]))
            _validate_result(result, candidate["result_file"])
            _validate_configuration(
                result, candidate["configuration"], candidate["result_file"]
            )
            if result["problem"]["name"] != summary["screen_problem"]:
                raise ValueError("candidate result uses the wrong screening problem")
            expected_rank = np.asarray(candidate_rank(result), dtype=float)
            reported_rank = np.asarray(candidate["rank"], dtype=float)
            if expected_rank.shape != reported_rank.shape or not np.allclose(
                expected_rank, reported_rank, rtol=1e-12, atol=1e-15
            ):
                raise ValueError(f"candidate {index} rank does not reproduce")
            results.append(result)

        expected_order = sorted(
            range(len(results)), key=lambda index: candidate_rank(results[index])
        )
        if summary["ranking"] != expected_order:
            raise ValueError("reported candidate ordering does not reproduce")
        winner_index = summary["winner"]["index"]
        if winner_index != expected_order[0]:
            raise ValueError("reported winner is not the first ranked candidate")

        final_result = json.loads(artifact.read(summary["final_result_file"]))
        _validate_result(
            final_result, summary["final_result_file"], require_visualizations=True
        )
        _validate_configuration(
            final_result,
            summary["winner"]["configuration"],
            summary["final_result_file"],
        )
        if final_result["problem"]["name"] != summary["final_problem"]:
            raise ValueError("production result uses the wrong final problem")
        if summary["accepted"] != all(
            gate["passed"] for gate in final_result["acceptance_gates"].values()
        ):
            raise ValueError("production acceptance summary does not match its gates")

        surfaces = {}
        for backend in ("cpu", "gpu"):
            metadata = final_result["visualizations"][f"{backend}_final"]
            surface_name = metadata["surface_vts"]
            coil_name = metadata["coils_vtu"]
            if surface_name not in names or coil_name not in names:
                raise ValueError(f"archive lacks final {backend} VTK artifacts")
            surfaces[backend] = artifact.read(surface_name)
            coil_payload = artifact.read(coil_name)
            if not surfaces[backend] or not coil_payload:
                raise ValueError(f"final {backend} VTK artifacts must be nonempty")
            signed = appended_vtk_array(surfaces[backend], "B_dot_n_over_abs_B")
            absolute = appended_vtk_array(surfaces[backend], "abs_B_dot_n_over_abs_B")
            if signed.shape != absolute.shape or not np.allclose(
                np.abs(signed), absolute, rtol=1e-12, atol=1e-14
            ):
                raise ValueError(f"final {backend} surface field arrays disagree")

    return summary, results, final_result, surfaces


def surface_fields(surfaces):
    fields = {}
    for backend, payload in surfaces.items():
        values = appended_vtk_array(payload, "B_dot_n_over_abs_B")
        shape = surface_shape(payload)
        fields[backend] = values.reshape(shape, order="F").squeeze().T
    if fields["cpu"].shape != fields["gpu"].shape:
        raise ValueError("CPU/GPU final surface shapes differ")
    return fields


def plot_screen(summary, results, output):
    """Compare feasibility, stationarity, field quality, and timings."""
    labels = [candidate["configuration"]["name"] for candidate in summary["candidates"]]
    x = np.arange(len(labels))
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 6.4), constrained_layout=True)
    for backend, shift in (("cpu", -width / 2), ("gpu", width / 2)):
        color = BACKEND_COLORS[backend]
        axes[0, 0].bar(
            x + shift,
            [max(feasibility_ratios(result, backend).values()) for result in results],
            width,
            color=color,
            label=backend.upper(),
        )
        axes[0, 1].bar(
            x + shift,
            [
                result[backend]["optimization"]["final_gradient_norm"]
                for result in results
            ],
            width,
            color=color,
            label=backend.upper(),
        )
        axes[1, 0].bar(
            x + shift,
            [
                result[backend]["final_metrics"]["normalized_normal_field"][
                    "root_mean_square"
                ]
                for result in results
            ],
            width,
            color=color,
            label=backend.upper(),
        )
    axes[1, 1].bar(
        x,
        [result["comparison"]["optimization_speedup"] for result in results],
        0.62,
        color="#4a9d62",
    )
    axes[0, 0].axhline(1.0, color="#7b2f2f", linestyle="--", linewidth=1.0)
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 1].axhline(3.0, color="#7b2f2f", linestyle="--", linewidth=1.0)
    axes[0, 0].set_title("Worst direct violation / tolerance")
    axes[0, 1].set_title(r"Final $\|\nabla\mathcal{L}_A\|_2$")
    axes[1, 0].set_title(r"Final RMS $|B\cdot n|/|B|$")
    axes[1, 1].set_title("Reported warm speedup")
    axes[1, 1].set_ylabel("CPU time / GPU time")
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=18, ha="right", fontsize=8)
        axis.grid(True, axis="y", which="both", color="#d8d8d8", linewidth=0.5)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_winner_convergence(result, output):
    """Expose raw/scaled feasibility and premature inner termination."""
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 6.2), constrained_layout=True)
    for backend in ("cpu", "gpu"):
        history = result[backend]["optimization"]["outer_history"]
        outer = np.arange(1, len(history) + 1)
        color = BACKEND_COLORS[backend]
        axes[0, 0].semilogy(
            outer,
            np.maximum(
                [record["constraint_norm_infinity"] for record in history], 1e-14
            ),
            marker="o",
            color=color,
            label=f"{backend.upper()} raw",
        )
        axes[0, 0].semilogy(
            outer,
            np.maximum(
                [record["scaled_constraint_norm_infinity"] for record in history],
                1e-14,
            ),
            linestyle="--",
            color=color,
            label=f"{backend.upper()} scaled",
        )
        axes[0, 1].semilogy(
            outer,
            [record["gradient_norm"] for record in history],
            marker="o",
            color=color,
            label=backend.upper(),
        )
        axes[1, 0].plot(
            outer,
            [record["inner_iterations"] for record in history],
            marker="o",
            color=color,
            label=backend.upper(),
        )
        axes[1, 1].semilogy(
            outer,
            [max(record["penalties_after"]) for record in history],
            marker="o",
            color=color,
            label=backend.upper(),
        )
    axes[0, 0].axhline(
        result["solver"]["constraint_tolerance"],
        color="#333333",
        linestyle=":",
        linewidth=1.0,
    )
    axes[0, 1].axhline(
        result["solver"]["gradient_tolerance"],
        color="#333333",
        linestyle=":",
        linewidth=1.0,
    )
    axes[0, 0].set_title("Raw and scaled aggregate constraints")
    axes[0, 1].set_title("AL stationarity")
    axes[1, 0].set_title("Inner L-BFGS-B iterations")
    axes[1, 1].set_title("Largest component penalty")
    for axis in axes.flat:
        axis.set_xlabel("Outer iteration")
        axis.set_xticks(np.arange(1, 9))
        axis.grid(True, which="both", color="#d8d8d8", linewidth=0.5)
        axis.legend(frameon=False, fontsize=8)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_surface_fields(surfaces, output):
    fields = surface_fields(surfaces)
    cpu = fields["cpu"]
    gpu = fields["gpu"]
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


def analysis_summary(archive, workflow, results, final_result, surfaces):
    fields = surface_fields(surfaces)
    cpu_surface = fields["cpu"]
    gpu_surface = fields["gpu"]
    cpu_seconds = final_result["cpu"]["optimization"]["seconds"]
    gpu_seconds = final_result["gpu"]["optimization"]["seconds"]
    cpu_evaluations = final_result["cpu"]["optimization"]["total_evaluations"]
    gpu_evaluations = final_result["gpu"]["optimization"]["total_evaluations"]
    candidates = []
    for metadata, result in zip(workflow["candidates"], results, strict=True):
        candidates.append(
            {
                "index": metadata["index"],
                "name": metadata["configuration"]["name"],
                "rank": metadata["rank"],
                "failed_physical_checks": int(candidate_rank(result)[0]),
                "worst_physical_violation_ratio": candidate_rank(result)[1],
                "warm_speedup": result["comparison"]["optimization_speedup"],
                "cpu": {
                    "final_gradient_norm": result["cpu"]["optimization"][
                        "final_gradient_norm"
                    ],
                    "normalized_field_rms": result["cpu"]["final_metrics"][
                        "normalized_normal_field"
                    ]["root_mean_square"],
                    "physical_violation_ratios": feasibility_ratios(result, "cpu"),
                },
                "gpu": {
                    "final_gradient_norm": result["gpu"]["optimization"][
                        "final_gradient_norm"
                    ],
                    "normalized_field_rms": result["gpu"]["final_metrics"][
                        "normalized_normal_field"
                    ]["root_mean_square"],
                    "physical_violation_ratios": feasibility_ratios(result, "gpu"),
                },
            }
        )
    return {
        "schema_version": 1,
        "workflow": "augmented_lagrangian_conditioning_analysis",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "revision": final_result["environment"]["simsopt_revision"],
        "nvidia_smi": final_result["nvidia_smi"],
        "candidate_ranking": workflow["ranking"],
        "candidates": candidates,
        "mechanical_winner": workflow["winner"],
        "production_accepted": workflow["accepted"],
        "production_acceptance_gates": final_result["acceptance_gates"],
        "production_performance": {
            "cpu_seconds": cpu_seconds,
            "gpu_seconds": gpu_seconds,
            "compilation_seconds": final_result["gpu_configuration"][
                "compilation_seconds"
            ],
            "warm_speedup": final_result["comparison"]["optimization_speedup"],
            "amortized_speedup": final_result["comparison"]["amortized_speedup"],
            "cpu_evaluations": cpu_evaluations,
            "gpu_evaluations": gpu_evaluations,
            "per_evaluation_throughput_ratio": (
                (cpu_seconds / cpu_evaluations) / (gpu_seconds / gpu_evaluations)
            ),
            "comparison_warning": (
                "warm/amortized whole-run speedups compare non-equivalent work "
                "because the GPU trajectory stopped moving early"
            ),
        },
        "production_final": {
            backend: {
                "optimization": {
                    key: final_result[backend]["optimization"][key]
                    for key in (
                        "success",
                        "final_base_objective",
                        "final_gradient_norm",
                        "final_raw_constraint_norm_infinity",
                        "final_scaled_constraint_norm_infinity",
                        "final_constraints",
                        "final_scaled_constraints",
                        "final_penalties",
                    )
                },
                "metrics": final_result[backend]["final_metrics"],
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
    workflow, results, final_result, surfaces = read_archive(args.archive)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_screen(workflow, results, args.output_dir / "al_conditioning_screen.png")
    plot_winner_convergence(
        final_result, args.output_dir / "al_conditioning_winner_convergence.png"
    )
    plot_surface_fields(
        surfaces, args.output_dir / "al_conditioning_winner_surface_field.png"
    )
    summary = analysis_summary(args.archive, workflow, results, final_result, surfaces)
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    (args.output_dir / "al_conditioning_analysis_summary.json").write_text(
        encoded + "\n"
    )
    print(encoded)


if __name__ == "__main__":
    main()
