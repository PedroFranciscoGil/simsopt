"""Validate and visualize a device-resident L-BFGS replay sweep archive."""

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

RESULT_NAME = "device-lbfgs-replay-sweep.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


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
        payloads = {}
        visualization = result.get("visualization")
        if visualization is not None:
            if set(visualization) < {"surface_vts", "coils_vtu"}:
                raise ValueError("winner visualization contract is incomplete")
            for kind in ("surface_vts", "coils_vtu"):
                basename = PurePosixPath(visualization[kind]).name
                matches = [
                    name for name in names if PurePosixPath(name).name == basename
                ]
                if len(matches) != 1:
                    raise ValueError(f"archive must contain exactly one {basename}")
                payloads[kind] = artifact.read(matches[0])
                if not payloads[kind]:
                    raise ValueError(f"{basename} must be nonempty")
    return result, payloads, digest


def validate_result(result):
    if result.get("schema_version") != 1 or result.get("workflow") != (
        "device_lbfgs_replay_sweep"
    ):
        raise ValueError("device-LBFGS replay result contract does not match")
    if result.get("execution_platform") != "gpu":
        raise ValueError("replay sweep did not execute on a GPU")
    sweep = result.get("sweep", {})
    histories = sweep.get("history_sizes", [])
    patiences = sweep.get("checkpoint_patiences", [])
    candidates = result.get("candidates", [])
    if (
        not histories
        or not patiences
        or len(candidates) != len(histories) * len(patiences)
        or sweep.get("candidate_count") != len(candidates)
    ):
        raise ValueError("replay sweep grid is incomplete")
    expected = {(history, patience) for history in histories for patience in patiences}
    observed = set()
    for candidate in candidates:
        key = (candidate.get("history_size"), candidate.get("checkpoint_patience"))
        observed.add(key)
        optimization = candidate.get("optimization", {})
        selection = optimization.get("checkpoint_selection", {})
        checkpoints = selection.get("checkpoints", [])
        selected = selection.get("selected_index")
        if (
            optimization.get("device_resident") is not True
            or optimization.get("host_callbacks") != 0
            or not checkpoints
            or selected not in range(len(checkpoints))
            or selection.get("selected") != checkpoints[selected]
        ):
            raise ValueError(f"invalid device optimization for {candidate.get('name')}")
        for metric in (
            candidate.get("final_metrics", {}).get("quadratic_flux"),
            optimization.get("seconds"),
            candidate.get("compilation_seconds"),
        ):
            if metric is None or not math.isfinite(float(metric)):
                raise ValueError(
                    f"non-finite candidate metric for {candidate.get('name')}"
                )
        reproduced = bool(
            candidate.get("engineering_validation", {}).get("passed")
            and candidate.get("quadratic_flux_validation", {}).get("passed")
        )
        if candidate.get("scientifically_validated") is not reproduced:
            raise ValueError("candidate scientific validation does not reproduce")
    if observed != expected:
        raise ValueError("replay sweep grid coordinates differ from request")
    qualified = [
        index
        for index, candidate in enumerate(candidates)
        if candidate["scientifically_validated"]
    ]
    if result.get("qualified_indices") != qualified:
        raise ValueError("qualified candidate indices do not reproduce")
    winner_index = result.get("winner_index")
    if winner_index is None:
        if result.get("winner") is not None or result.get("scientifically_validated"):
            raise ValueError("empty winner contract is inconsistent")
    elif (
        winner_index not in qualified
        or result.get("winner") != candidates[winner_index]
        or result.get("scientifically_validated") is not True
    ):
        raise ValueError("winner contract is inconsistent")
    return result


def grid(result, field):
    histories = result["sweep"]["history_sizes"]
    patiences = result["sweep"]["checkpoint_patiences"]
    values = np.full((len(histories), len(patiences)), np.nan)
    for candidate in result["candidates"]:
        row = histories.index(candidate["history_size"])
        column = patiences.index(candidate["checkpoint_patience"])
        values[row, column] = field(candidate)
    return histories, patiences, values


def plot_grid(result, output):
    fields = (
        ("quadratic flux", lambda item: item["final_metrics"]["quadratic_flux"]),
        ("warm seconds", lambda item: item["optimization"]["seconds"]),
        ("physics evaluations", lambda item: item["optimization"]["evaluations"]),
        (
            "CPU objective relative difference",
            lambda item: item["objective_agreement"]["relative_difference"],
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), constrained_layout=True)
    for axis, (title, field) in zip(axes.flat, fields, strict=True):
        histories, patiences, values = grid(result, field)
        image = axis.imshow(values, cmap="viridis", aspect="auto")
        axis.set_xticks(np.arange(len(patiences)), patiences)
        axis.set_yticks(np.arange(len(histories)), histories)
        axis.set_xlabel("checkpoint patience")
        axis.set_ylabel("history size")
        axis.set_title(title)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                label = f"{value:.3g}"
                axis.text(column, row, label, ha="center", va="center", color="white")
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_trajectories(result, output):
    figure, axis = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
    target = result["validation_policy"]["quadratic_flux_allowed_boundary"]
    for candidate in result["candidates"]:
        checkpoints = candidate["optimization"]["checkpoint_selection"]["checkpoints"]
        flux = np.asarray([checkpoint["quadratic_flux"] for checkpoint in checkpoints])
        selected = candidate["optimization"]["checkpoint_selection"]["selected_index"]
        (line,) = axis.semilogy(
            np.arange(len(flux)),
            np.maximum(flux, 1e-30),
            linewidth=1.1,
            alpha=0.75,
            label=f"m={candidate['history_size']}, p={candidate['checkpoint_patience']}",
        )
        axis.scatter(selected, flux[selected], marker="*", s=55, color=line.get_color())
    axis.axhline(target, color="#7b2f2f", linestyle="--", label="flux boundary")
    axis.set_xlabel("accepted checkpoint")
    axis.set_ylabel("quadratic flux")
    axis.set_title("Device-LBFGS replay trajectories")
    axis.grid(True, axis="y", color="#d8d8d8", linewidth=0.5)
    axis.legend(frameon=False, fontsize=7, ncols=3)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main():
    args = parse_args()
    result, payloads, digest = read_archive(args.archive)
    validate_result(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = ["device_lbfgs_replay_grid.png", "device_lbfgs_replay_trajectories.png"]
    plot_grid(result, args.output_dir / figures[0])
    plot_trajectories(result, args.output_dir / figures[1])
    compact_candidates = []
    for candidate in result["candidates"]:
        compact_candidates.append(
            {
                key: value
                for key, value in candidate.items()
                if key not in ("physical_variables",)
            }
        )
        compact_candidates[-1]["optimization"] = {
            key: value
            for key, value in candidate["optimization"].items()
            if not key.endswith("_history") and key != "checkpoint_selection"
        }
        compact_candidates[-1]["checkpoint_selection"] = {
            key: value
            for key, value in candidate["optimization"]["checkpoint_selection"].items()
            if key != "checkpoints"
        }
    summary = {
        "schema_version": 1,
        "workflow": "device_lbfgs_replay_sweep_analysis",
        "archive_sha256": digest,
        "environment": result["environment"],
        "baseline": result["baseline"],
        "sweep": result["sweep"],
        "validation_policy": result["validation_policy"],
        "qualified_indices": result["qualified_indices"],
        "winner_index": result["winner_index"],
        "scientifically_validated": result["scientifically_validated"],
        "all_gates_passed": result["all_gates_passed"],
        "candidates": compact_candidates,
        "visualization_files": sorted(payloads),
        "figures": figures,
    }
    path = args.output_dir / "device_lbfgs_replay_analysis_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
