"""Validate and visualize a safeguarded AL qualification archive."""

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
from analyze_augmented_lagrangian_conditioning import (
    BACKEND_COLORS,
    CONSTRAINT_NAMES,
    _validate_result,
)
from analyze_extended_convergence import appended_vtk_array, surface_shape
from sweep_augmented_lagrangian_safeguards import (
    CANDIDATES,
    QUALIFICATION_GATES,
    diagnostic_rank,
    qualifies,
)

SUMMARY_NAME = "safeguard-qualification-summary.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def _member_name(value):
    return PurePosixPath(value).name


def _validate_schema7_contract(result, label):
    if result.get("schema_version") != 7:
        raise ValueError(f"{label} must use augmented-Lagrangian schema 7")
    mapping = result.get("constraint_mapping", {})
    if mapping.get("transform") not in ("identity", "smooth_sqrt"):
        raise ValueError(f"{label} has an invalid constraint transform")
    epsilon = mapping.get("transform_epsilon")
    if not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError(f"{label} has an invalid transform epsilon")
    calibration = mapping.get("gradient_balance_calibration", {})
    base_gradient_norm = calibration.get("base_gradient_norm", math.nan)
    maximum_ratio = calibration.get("maximum_ratio_to_base_gradient", math.nan)
    if (
        not math.isfinite(float(base_gradient_norm))
        or base_gradient_norm < 0
        or not math.isfinite(float(maximum_ratio))
        or maximum_ratio <= 0
    ):
        raise ValueError(f"{label} has invalid scalar gradient calibration")
    if calibration.get("automatic_scaling_enabled") is not mapping.get(
        "automatic_gradient_balancing"
    ):
        raise ValueError(f"{label} has inconsistent automatic-scaling metadata")
    for key in (
        "constraint_jacobian_row_norms",
        "unscaled_initial_penalty_gradient_norm_estimates",
        "recommended_attenuation_scales",
        "applied_scales",
        "applied_initial_penalty_gradient_norm_estimates",
    ):
        values = calibration.get(key, {})
        if tuple(values) != CONSTRAINT_NAMES or any(
            not math.isfinite(float(value)) or value < 0 for value in values.values()
        ):
            raise ValueError(f"{label} has invalid gradient calibration {key}")
    applied_scales = np.asarray(
        list(calibration["applied_scales"].values()), dtype=float
    )
    if np.any(applied_scales <= 0) or not np.array_equal(
        applied_scales, np.asarray(result["constraint_scaling"]["scales"])
    ):
        raise ValueError(f"{label} applied scales differ from the AL scales")
    if result.get("solver", {}).get("require_inner_stationarity") is not True:
        raise ValueError(f"{label} did not enable the inner safeguard")
    for backend in ("cpu", "gpu"):
        optimization = result[backend]["optimization"]
        if not isinstance(optimization.get("terminated_by_inner_safeguard"), bool):
            raise TypeError(f"{label} {backend} lacks safeguard status")
        for record in optimization["outer_history"]:
            for key in (
                "gradient_norm_infinity",
                "requested_inner_gradient_tolerance",
                "step_norm_infinity",
            ):
                if not math.isfinite(float(record.get(key, math.nan))):
                    raise ValueError(f"{label} {backend} lacks finite {key}")
            for key in (
                "inner_stationary",
                "inner_stage_accepted",
                "outer_update_applied",
            ):
                if not isinstance(record.get(key), bool):
                    raise TypeError(f"{label} {backend} lacks boolean {key}")
            if not record["inner_stage_accepted"] and record["outer_update_applied"]:
                raise ValueError(f"{label} {backend} updated a rejected inner stage")


def _validate_candidate_configuration(result, candidate, label):
    mapping = result["constraint_mapping"]
    if mapping["transform"] != candidate["constraint_transform"]:
        raise ValueError(f"{label} transform differs from its candidate")
    if not math.isclose(
        mapping["transform_epsilon"], candidate["transform_epsilon"], rel_tol=1e-14
    ):
        raise ValueError(f"{label} epsilon differs from its candidate")
    if mapping["automatic_gradient_balancing"] is not candidate["automatic_scaling"]:
        raise ValueError(f"{label} automatic-scaling flag differs")
    ratio = mapping["gradient_balance_calibration"]["maximum_ratio_to_base_gradient"]
    if not math.isclose(ratio, candidate["maximum_gradient_ratio"], rel_tol=1e-14):
        raise ValueError(f"{label} gradient ratio differs from its candidate")


def _read_visualizations(artifact, names, result, label):
    surfaces = {}
    for backend in ("cpu", "gpu"):
        metadata = result["visualizations"][f"{backend}_final"]
        surface_name = _member_name(metadata["surface_vts"])
        coil_name = _member_name(metadata["coils_vtu"])
        if surface_name not in names or coil_name not in names:
            raise ValueError(f"archive lacks {label} {backend} VTK artifacts")
        surface = artifact.read(surface_name)
        coil = artifact.read(coil_name)
        if not surface or not coil:
            raise ValueError(f"{label} {backend} VTK artifacts must be nonempty")
        signed = appended_vtk_array(surface, "B_dot_n_over_abs_B")
        absolute = appended_vtk_array(surface, "abs_B_dot_n_over_abs_B")
        if signed.shape != absolute.shape or not np.allclose(
            np.abs(signed), absolute, rtol=1e-12, atol=1e-14
        ):
            raise ValueError(f"{label} {backend} surface arrays disagree")
        surfaces[backend] = surface
    return surfaces


def read_archive(archive):
    """Read the workflow, reproduce qualification, and validate every VTK file."""
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
            "augmented_lagrangian_safeguard_qualification"
        ):
            raise ValueError("safeguard summary contract does not match")
        if tuple(summary.get("policy", {}).get("qualification_gates", ())) != (
            QUALIFICATION_GATES
        ):
            raise ValueError("qualification gate contract differs")
        metadata = summary.get("candidates", [])
        if len(metadata) != len(CANDIDATES):
            raise ValueError("safeguard summary must contain four candidates")

        results = []
        screen_surfaces = []
        for index, (reported, candidate) in enumerate(
            zip(metadata, CANDIDATES, strict=True)
        ):
            if (
                reported.get("index") != index
                or reported.get("configuration") != candidate
            ):
                raise ValueError("candidate identity/order does not match")
            result_name = reported["result_file"]
            if result_name not in names:
                raise ValueError(f"archive is missing {result_name}")
            result = json.loads(artifact.read(result_name))
            _validate_result(result, result_name, require_visualizations=True)
            _validate_schema7_contract(result, result_name)
            _validate_candidate_configuration(result, candidate, result_name)
            if result["problem"]["name"] != summary["screen_problem"]:
                raise ValueError("candidate uses the wrong screening problem")
            if reported["qualified"] is not qualifies(result):
                raise ValueError("reported qualification does not reproduce")
            expected_rank = np.asarray(diagnostic_rank(result), dtype=float)
            if not np.allclose(
                expected_rank,
                np.asarray(reported["diagnostic_rank"], dtype=float),
                rtol=1e-12,
                atol=1e-15,
            ):
                raise ValueError("reported diagnostic rank does not reproduce")
            results.append(result)
            screen_surfaces.append(
                _read_visualizations(artifact, names, result, result_name)
            )

        expected_qualified = [
            index for index, result in enumerate(results) if qualifies(result)
        ]
        expected_order = sorted(
            range(len(results)), key=lambda index: diagnostic_rank(results[index])
        )
        if summary["qualified_indices"] != expected_qualified:
            raise ValueError("qualified candidate list does not reproduce")
        if summary["diagnostic_ordering"] != expected_order:
            raise ValueError("diagnostic ordering does not reproduce")
        winner = summary["winner"]
        production = summary["production"]
        final_result = None
        final_surfaces = None
        if expected_qualified:
            expected_winner = min(
                expected_qualified, key=lambda index: diagnostic_rank(results[index])
            )
            if winner is None or winner["index"] != expected_winner:
                raise ValueError("qualified winner does not reproduce")
            if not production["executed"] or not production["result_file"]:
                raise ValueError("qualified screen did not execute production")
            if production["result_file"] not in names:
                raise ValueError("archive lacks the production result")
            final_result = json.loads(artifact.read(production["result_file"]))
            _validate_result(
                final_result, production["result_file"], require_visualizations=True
            )
            _validate_schema7_contract(final_result, production["result_file"])
            _validate_candidate_configuration(
                final_result, CANDIDATES[expected_winner], production["result_file"]
            )
            if final_result["problem"]["name"] != summary["final_problem"]:
                raise ValueError("production uses the wrong problem")
            if production["accepted"] is not qualifies(final_result):
                raise ValueError("production acceptance does not reproduce")
            final_surfaces = _read_visualizations(
                artifact, names, final_result, production["result_file"]
            )
        else:
            if winner is not None or production["executed"]:
                raise ValueError("unqualified workflow improperly promoted a candidate")
            if not production.get("skip_reason"):
                raise ValueError("skipped production lacks a reason")

    return summary, results, screen_surfaces, final_result, final_surfaces


def feasibility_ratios(result, backend):
    tolerances = result["feasibility_tolerances"]
    violations = result[backend]["final_metrics"]["coil_constraints"]["violations"]
    return {name: violations[name] / tolerances[name] for name in tolerances}


def plot_screen(summary, results, output):
    labels = [candidate["configuration"]["name"] for candidate in summary["candidates"]]
    x = np.arange(len(labels))
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), constrained_layout=True)
    axes[0, 0].bar(
        x,
        [
            sum(
                not result["acceptance_gates"][gate]["passed"]
                for gate in QUALIFICATION_GATES
            )
            for result in results
        ],
        color="#7b2f2f",
    )
    for backend, shift in (("cpu", -width / 2), ("gpu", width / 2)):
        color = BACKEND_COLORS[backend]
        axes[0, 1].bar(
            x + shift,
            [max(feasibility_ratios(result, backend).values()) for result in results],
            width,
            color=color,
            label=backend.upper(),
        )
        axes[1, 0].bar(
            x + shift,
            [
                result[backend]["optimization"]["final_gradient_norm"]
                for result in results
            ],
            width,
            color=color,
            label=backend.upper(),
        )
        axes[1, 1].bar(
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
    axes[0, 0].set_title("Failed qualification gates")
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 1].set_title("Worst physical violation / tolerance")
    axes[0, 1].axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[1, 0].set_title(r"Final $\|\nabla\mathcal{L}_A\|_2$")
    axes[1, 1].set_title(r"Final RMS $|B\cdot n|/|B|$")
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.set_yscale("log")
        axis.legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=18, ha="right", fontsize=8)
        axis.grid(True, axis="y", which="both", color="#d8d8d8", linewidth=0.5)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_calibration(summary, results, output):
    labels = [candidate["configuration"]["name"] for candidate in summary["candidates"]]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.18
    for family, name in enumerate(CONSTRAINT_NAMES):
        shifts = (family - 1.5) * width
        axes[0].bar(
            x + shifts,
            [result["constraint_scaling"]["scales"][family] for result in results],
            width,
            label=name.replace("_penalty", "").replace("_", " "),
        )
        axes[1].bar(
            x + shifts,
            [
                result["constraint_mapping"]["gradient_balance_calibration"][
                    "applied_initial_penalty_gradient_norm_estimates"
                ][name]
                for result in results
            ],
            width,
        )
    axes[0].set_title("Applied AL-coordinate scales")
    axes[1].set_title("Estimated initial penalty-gradient norms")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=18, ha="right", fontsize=8)
        axis.grid(True, axis="y", which="both", color="#d8d8d8", linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=7)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_convergence(result, output):
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), constrained_layout=True)
    for backend in ("cpu", "gpu"):
        history = result[backend]["optimization"]["outer_history"]
        outer = np.arange(1, len(history) + 1)
        color = BACKEND_COLORS[backend]
        axes[0].semilogy(
            outer,
            np.maximum(
                [record["constraint_norm_infinity"] for record in history], 1e-16
            ),
            marker="o",
            color=color,
            label=backend.upper(),
        )
        axes[1].semilogy(
            outer,
            [record["gradient_norm_infinity"] for record in history],
            marker="o",
            color=color,
            label=backend.upper(),
        )
        axes[1].semilogy(
            outer,
            [record["requested_inner_gradient_tolerance"] for record in history],
            linestyle="--",
            color=color,
        )
        axes[2].plot(
            outer,
            [record["inner_iterations"] for record in history],
            marker="o",
            color=color,
            label=backend.upper(),
        )
    axes[0].set_title("Raw constraint infinity norm")
    axes[1].set_title("Inner gradient (solid) and request (dashed)")
    axes[2].set_title("Inner iterations")
    for axis in axes:
        axis.set_xlabel("Outer iteration")
        axis.grid(True, which="both", color="#d8d8d8", linewidth=0.5)
        axis.legend(frameon=False, fontsize=8)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def surface_fields(surfaces):
    fields = {}
    for backend, payload in surfaces.items():
        values = appended_vtk_array(payload, "B_dot_n_over_abs_B")
        fields[backend] = values.reshape(surface_shape(payload), order="F").squeeze().T
    if fields["cpu"].shape != fields["gpu"].shape:
        raise ValueError("CPU/GPU surface shapes differ")
    return fields


def plot_surface(surfaces, output):
    fields = surface_fields(surfaces)
    difference = fields["gpu"] - fields["cpu"]
    common = max(np.max(np.abs(fields["cpu"])), np.max(np.abs(fields["gpu"])))
    diff_limit = np.max(np.abs(difference))
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.2), constrained_layout=True)
    images = []
    for axis, values, title in zip(
        axes,
        (fields["cpu"], fields["gpu"], difference),
        ("CPU final", "GPU final", "GPU minus CPU"),
        strict=True,
    ):
        limit = diff_limit if title == "GPU minus CPU" else common
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


def build_analysis(archive, workflow, results, chosen_index, chosen_result, surfaces):
    def optimization_snapshot(result, backend):
        optimization = result[backend]["optimization"]
        return {
            key: optimization[key]
            for key in (
                "success",
                "message",
                "seconds",
                "outer_iterations",
                "total_inner_iterations",
                "total_evaluations",
                "final_base_objective",
                "final_gradient_norm",
                "terminated_by_inner_safeguard",
                "final_constraints",
                "final_al_constraints",
                "final_penalties",
            )
        }

    def candidate_analysis(index, result):
        cpu_optimization = result["cpu"]["optimization"]
        gpu_optimization = result["gpu"]["optimization"]
        cpu_last = cpu_optimization["outer_history"][-1]
        gpu_last = gpu_optimization["outer_history"][-1]
        return {
            "index": index,
            "name": CANDIDATES[index]["name"],
            "qualified": qualifies(result),
            "diagnostic_rank": list(diagnostic_rank(result)),
            "acceptance_gates": result["acceptance_gates"],
            "constraint_mapping": result["constraint_mapping"],
            "performance": {
                "warm_speedup": result["comparison"]["optimization_speedup"],
                "amortized_speedup": result["comparison"]["amortized_speedup"],
                "per_evaluation_throughput_ratio": (
                    (
                        cpu_optimization["seconds"]
                        / cpu_optimization["total_evaluations"]
                    )
                    / (
                        gpu_optimization["seconds"]
                        / gpu_optimization["total_evaluations"]
                    )
                ),
            },
            "safeguard_stop": {
                "cpu_gradient_to_request_ratio": (
                    cpu_last["gradient_norm_infinity"]
                    / cpu_last["requested_inner_gradient_tolerance"]
                ),
                "gpu_gradient_to_request_ratio": (
                    gpu_last["gradient_norm_infinity"]
                    / gpu_last["requested_inner_gradient_tolerance"]
                ),
                "cpu_outer_update_applied": cpu_last["outer_update_applied"],
                "gpu_outer_update_applied": gpu_last["outer_update_applied"],
            },
            "physical_violation_ratios": {
                backend: feasibility_ratios(result, backend)
                for backend in ("cpu", "gpu")
            },
            "cpu_final_metrics": result["cpu"]["final_metrics"],
            "gpu_final_metrics": result["gpu"]["final_metrics"],
            "cpu_optimization": optimization_snapshot(result, "cpu"),
            "gpu_optimization": optimization_snapshot(result, "gpu"),
        }

    fields = surface_fields(surfaces)
    cpu = fields["cpu"]
    gpu = fields["gpu"]
    return {
        "schema_version": 1,
        "workflow": "augmented_lagrangian_safeguard_analysis",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "revision": results[0]["environment"]["simsopt_revision"],
        "nvidia_smi": results[0]["nvidia_smi"],
        "qualified_indices": workflow["qualified_indices"],
        "winner": workflow["winner"],
        "production": workflow["production"],
        "visualized_result": {
            "kind": "production"
            if workflow["production"]["executed"]
            else "best_screen_diagnostic",
            "screen_index": chosen_index,
            "problem": chosen_result["problem"]["name"],
        },
        "candidates": [
            candidate_analysis(index, result) for index, result in enumerate(results)
        ],
        "surface_comparison": {
            "signed_correlation": float(np.corrcoef(cpu.ravel(), gpu.ravel())[0, 1]),
            "relative_l2_difference": float(
                np.linalg.norm(gpu - cpu)
                / max(np.linalg.norm(cpu), np.finfo(float).tiny)
            ),
            "gpu_smaller_absolute_fraction": float(np.mean(np.abs(gpu) < np.abs(cpu))),
        },
    }


def main():
    args = parse_args()
    workflow, results, screen_surfaces, final_result, final_surfaces = read_archive(
        args.archive
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_screen(workflow, results, args.output_dir / "al_safeguard_screen.png")
    plot_calibration(
        workflow, results, args.output_dir / "al_safeguard_calibration.png"
    )
    if final_result is None:
        chosen_index = workflow["diagnostic_ordering"][0]
        chosen_result = results[chosen_index]
        chosen_surfaces = screen_surfaces[chosen_index]
    else:
        chosen_index = workflow["winner"]["index"]
        chosen_result = final_result
        chosen_surfaces = final_surfaces
    plot_convergence(chosen_result, args.output_dir / "al_safeguard_convergence.png")
    plot_surface(chosen_surfaces, args.output_dir / "al_safeguard_surface.png")
    analysis = build_analysis(
        args.archive,
        workflow,
        results,
        chosen_index,
        chosen_result,
        chosen_surfaces,
    )
    output = args.output_dir / "al_safeguard_analysis_summary.json"
    output.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
