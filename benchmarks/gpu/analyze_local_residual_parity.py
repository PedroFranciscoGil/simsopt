"""Validate and visualize a local-residual parity result or Colab archive."""

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

RESULT_NAME = "local-residual-parity.json"
FAMILIES = (
    "coil_coil_distance",
    "coil_surface_distance",
    "curvature",
    "mean_squared_curvature",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def read_result(path):
    """Read a JSON result or the named JSON member from a safe ZIP archive."""
    if path.suffix.lower() != ".zip":
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text()), digest
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
        return json.loads(artifact.read(matches[0])), digest


def validate_result(result):
    """Validate schema, family layout, parity diagnostics, and gate consistency."""
    if result.get("schema_version") != 1 or result.get("workflow") != (
        "local_engineering_residual_parity"
    ):
        raise ValueError("local-residual result contract does not match")
    contract = result.get("residual_contract", {})
    if tuple(contract.get("family_order", ())) != FAMILIES:
        raise ValueError("residual family order differs from the contract")
    layout = contract.get("layout", {})
    offset = 0
    for family in FAMILIES:
        item = layout.get(family, {})
        if item.get("start") != offset or item.get("stop", -1) <= offset:
            raise ValueError(f"invalid contiguous layout for {family}")
        offset = item["stop"]
    for section_name in ("production", "forced_activation"):
        section = result.get(section_name, {})
        for backend_name in ("cpu_families", "gpu_families"):
            families = section.get(backend_name, {})
            if tuple(families) != FAMILIES:
                raise ValueError(f"{section_name} lacks ordered {backend_name}")
            for family, metrics in families.items():
                for key in ("count", "active_count", "maximum", "l2_norm"):
                    value = metrics.get(key)
                    if value is None or not math.isfinite(float(value)) or value < 0:
                        raise ValueError(f"invalid {section_name} {family} {key}")
                expected = layout[family]["stop"] - layout[family]["start"]
                if metrics["count"] != expected:
                    raise ValueError(
                        f"{section_name} {family} count differs from layout"
                    )
                if metrics["active_count"] > metrics["count"]:
                    raise ValueError(
                        f"{section_name} {family} active count exceeds its count"
                    )
        for family in FAMILIES:
            cpu = section["cpu_families"][family]
            gpu = section["gpu_families"][family]
            if cpu["active_count"] != gpu["active_count"]:
                raise ValueError(
                    f"{section_name} {family} CPU/GPU active sets disagree"
                )
    directions = result["forced_activation"].get("directional_jacobian_parity", [])
    expected_directions = result["configuration"]["finite_difference_directions"]
    if len(directions) != expected_directions:
        raise ValueError("directional-Jacobian record count differs")
    for direction in directions:
        for key in ("maximum_absolute_error", "relative_l2_error"):
            if not math.isfinite(float(direction.get(key, math.nan))):
                raise ValueError(f"directional Jacobian has invalid {key}")
        if tuple(direction.get("family_parity", {})) != FAMILIES:
            raise ValueError("directional Jacobian lacks family parity")
    gates = result.get("acceptance_gates", {})
    required_gates = (
        "gpu_backend",
        "float64_enabled",
        "base_objective_value_parity",
        "production_residual_value_parity",
        "forced_family_activation",
        "forced_residual_value_parity",
        "directional_jacobian_parity",
        "residual_dimension",
    )
    if tuple(gates) != required_gates:
        raise ValueError("acceptance gate contract differs")
    base_parity = result["production"]["base_objective_parity"]
    production_parity = result["production"]["residual_parity"]
    forced_parity = result["forced_activation"]["residual_parity"]
    maximum_directional_error = max(
        direction["relative_l2_error"] for direction in directions
    )
    directional_gate = gates["directional_jacobian_parity"]
    reproduced_gates = {
        "gpu_backend": result["environment"]["jax_backend"] == "gpu",
        "float64_enabled": result["precision"] == "float64"
        and gates["float64_enabled"].get("observed") is True,
        "base_objective_value_parity": (
            base_parity["relative_error"] <= 5e-10
            or base_parity["absolute_error"] <= 1e-11
        ),
        "production_residual_value_parity": (
            production_parity["relative_l2_error"] <= 5e-10
            or production_parity["maximum_absolute_error"] <= 1e-11
        ),
        "forced_family_activation": all(
            result["forced_activation"]["cpu_families"][family]["active_count"] > 0
            for family in FAMILIES
        ),
        "forced_residual_value_parity": (forced_parity["relative_l2_error"] <= 5e-10),
        "directional_jacobian_parity": (
            maximum_directional_error <= directional_gate["tolerance"]
        ),
        "residual_dimension": offset <= gates["residual_dimension"]["maximum"],
    }
    for name, reproduced_passed in reproduced_gates.items():
        if gates[name].get("passed") is not reproduced_passed:
            raise ValueError(f"acceptance gate {name} does not reproduce")
    if not math.isclose(
        directional_gate["maximum_relative_l2_error"],
        maximum_directional_error,
        rel_tol=1e-14,
    ):
        raise ValueError("directional-Jacobian gate maximum does not reproduce")
    if gates["residual_dimension"].get("observed") != offset:
        raise ValueError("residual-dimension gate differs from the layout")
    reproduced = all(gate.get("passed") is True for gate in gates.values())
    if result.get("all_gates_passed") is not reproduced:
        raise ValueError("all-gates summary does not reproduce")
    return result


def make_figure(result, output):
    """Plot activation, residual magnitude, derivative error, and timing."""
    labels = ["coil--coil", "coil--surface", "curvature", "MSC"]
    x = np.arange(len(FAMILIES))
    production = result["production"]
    forced = result["forced_activation"]
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))

    width = 0.36
    for offset, (name, section) in zip(
        (-width / 2, width / 2),
        (("production", production), ("forced probe", forced)),
        strict=True,
    ):
        counts = [
            section["cpu_families"][family]["active_count"] for family in FAMILIES
        ]
        axes[0, 0].bar(x + offset, counts, width, label=name)
    axes[0, 0].set_xticks(x, labels, rotation=18)
    axes[0, 0].set_yscale("symlog", linthresh=0.5)
    axes[0, 0].set_ylabel("active residuals")
    axes[0, 0].set_title("Residual activation")
    axes[0, 0].legend()

    cpu_maxima = [forced["cpu_families"][family]["maximum"] for family in FAMILIES]
    gpu_maxima = [forced["gpu_families"][family]["maximum"] for family in FAMILIES]
    axes[0, 1].bar(x - width / 2, cpu_maxima, width, label="CPU")
    axes[0, 1].bar(x + width / 2, gpu_maxima, width, label="GPU")
    axes[0, 1].set_xticks(x, labels, rotation=18)
    axes[0, 1].set_yscale("symlog", linthresh=1e-12)
    axes[0, 1].set_ylabel("maximum dimensionless residual")
    axes[0, 1].set_title("Forced-probe value parity")
    axes[0, 1].legend()

    directions = forced["directional_jacobian_parity"]
    direction_x = np.arange(1, len(directions) + 1)
    errors = [item["relative_l2_error"] for item in directions]
    tolerance = result["acceptance_gates"]["directional_jacobian_parity"]["tolerance"]
    axes[1, 0].semilogy(direction_x, errors, "o-", label="CPU FD vs GPU JVP")
    axes[1, 0].axhline(tolerance, color="tab:red", linestyle="--", label="gate")
    axes[1, 0].set_xlabel("deterministic direction")
    axes[1, 0].set_ylabel("relative $L^2$ error")
    axes[1, 0].set_title("Directional Jacobian parity")
    axes[1, 0].legend()

    timing = result["timing"]
    timing_values = [
        timing["cpu_terms"]["median_seconds"],
        timing["gpu_compiled_terms"]["median_seconds"],
    ]
    axes[1, 1].bar(("CPU oracle", "GPU compiled"), timing_values)
    axes[1, 1].set_ylabel("median seconds")
    axes[1, 1].set_title(f"Warm evaluation ({timing['warm_speedup']:.2f}x)")

    figure.suptitle(
        f"Local residual qualification: {result['problem']['name']}"
        f" ({result['environment']['jax_backend']})"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    result, digest = read_result(args.input)
    validate_result(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "local_residual_parity.png"
    make_figure(result, figure_path)
    summary = {
        "schema_version": 1,
        "workflow": "local_engineering_residual_parity_analysis",
        "input_sha256": digest,
        "problem": result["problem"]["name"],
        "backend": result["environment"]["jax_backend"],
        "environment": result["environment"],
        "configuration": result["configuration"],
        "residual_contract": result["residual_contract"],
        "production": result["production"],
        "forced_activation": result["forced_activation"],
        "timing": result["timing"],
        "all_gates_passed": result["all_gates_passed"],
        "acceptance_gates": result["acceptance_gates"],
        "warm_speedup": result["timing"]["warm_speedup"],
        "figure": figure_path.name,
    }
    summary_path = args.output_dir / "local_residual_parity_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
