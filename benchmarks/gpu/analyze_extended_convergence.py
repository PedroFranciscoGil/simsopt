"""Generate report figures from an extended-convergence artifact archive."""

import argparse
import json
import re
import struct
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULT_NAME = "stress-extended-convergence.json"
SURFACE_NAMES = ("cpu_final_surface.vts", "gpu_final_surface.vts")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def read_archive(archive):
    """Return the benchmark result and final surface payloads from a zip."""
    with ZipFile(archive) as artifact:
        members = set(artifact.namelist())
        required = {RESULT_NAME, *SURFACE_NAMES}
        missing = required - members
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"archive is missing required files: {names}")
        result = json.loads(artifact.read(RESULT_NAME))
        surfaces = {name: artifact.read(name) for name in SURFACE_NAMES}
    if result.get("schema_version") != 3:
        raise ValueError("extended-convergence schema version 3 is required")
    return result, surfaces


def appended_vtk_array(payload, name):
    """Read one little-endian raw-appended array from a pyevtk XML payload."""
    appended = re.search(br'<AppendedData encoding="raw">\s*_', payload)
    if appended is None:
        raise ValueError("VTK file has no raw appended-data section")
    encoded_name = re.escape(name.encode())
    declaration = re.search(
        rb'<DataArray Name="'
        + encoded_name
        + rb'"[^>]*type="([^"]+)"[^>]*offset="([0-9]+)"',
        payload[: appended.start()],
    )
    if declaration is None:
        raise ValueError(f"VTK array {name!r} was not found")
    dtype_names = {b"Float32": "<f4", b"Float64": "<f8"}
    try:
        dtype = np.dtype(dtype_names[declaration.group(1)])
    except KeyError as error:
        raise ValueError("only floating-point VTK arrays are supported") from error
    start = appended.end() + int(declaration.group(2))
    byte_count = struct.unpack_from("<Q", payload, start)[0]
    if byte_count % dtype.itemsize:
        raise ValueError(f"invalid byte count for VTK array {name!r}")
    return np.frombuffer(
        payload,
        dtype=dtype,
        count=byte_count // dtype.itemsize,
        offset=start + 8,
    )


def surface_shape(payload):
    extent = re.search(
        br'WholeExtent="([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+) '
        br'([0-9]+) ([0-9]+)"',
        payload,
    )
    if extent is None:
        raise ValueError("structured-grid extent was not found")
    values = [int(value) for value in extent.groups()]
    return tuple(values[index + 1] - values[index] + 1 for index in (0, 2, 4))


def plot_convergence(result, output):
    cpu = result["cpu"]["optimization"]["iteration_history"]
    gpu = result["gpu"]["optimization"]["iteration_history"]
    iterations = np.arange(1, min(len(cpu), len(gpu)) + 1)
    cpu_objective = np.asarray([entry["objective"] for entry in cpu])
    gpu_objective = np.asarray([entry["objective"] for entry in gpu])
    cpu_gradient = np.asarray([entry["gradient_norm"] for entry in cpu])
    gpu_gradient = np.asarray([entry["gradient_norm"] for entry in gpu])

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.35), constrained_layout=True)
    for axis, cpu_values, gpu_values, ylabel in (
        (axes[0], cpu_objective, gpu_objective, "Objective"),
        (axes[1], cpu_gradient, gpu_gradient, r"Gradient norm $\|\nabla J\|_2$"),
    ):
        axis.semilogy(iterations, cpu_values, color="#2456a6", label="CPU")
        axis.semilogy(iterations, gpu_values, color="#d46715", label="GPU")
        axis.axvspan(1, 25, color="#4a9d62", alpha=0.10, label="parity window")
        axis.set_xlabel("Accepted L-BFGS-B iteration")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", color="#d8d8d8", linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_surface_fields(surfaces, output):
    fields = []
    for name in SURFACE_NAMES:
        payload = surfaces[name]
        values = appended_vtk_array(payload, "B_dot_n_over_abs_B")
        shape = surface_shape(payload)
        fields.append(values.reshape(shape, order="F").squeeze().T)
    cpu, gpu = fields
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


def main():
    args = parse_args()
    result, surfaces = read_archive(args.archive)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_convergence(result, args.output_dir / "extended_convergence.png")
    plot_surface_fields(surfaces, args.output_dir / "extended_surface_field.png")


if __name__ == "__main__":
    main()
