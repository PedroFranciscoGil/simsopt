"""Measure the pinned SIMSOPT CPU oracle for coil optimization."""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import jax
import numpy as np
import scipy
import simsopt
from problems import PROBLEMS, get_problem
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    LpCurveCurvature,
    MeanSquaredCurvature,
    SurfaceRZFourier,
    create_equally_spaced_curves,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux

ROOT = Path(__file__).resolve().parents[2]
SURFACE_FILE = ROOT / "tests" / "test_files" / "input.LandremanPaul2021_QA"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), default="minimal")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--optimization-iterations", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup must be non-negative")
    if args.optimization_iterations < 0:
        parser.error("optimization-iterations must be non-negative")
    return args


def measure(operation, warmup, repeats):
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    return {
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "samples_seconds": samples,
    }


def build_problem(spec, regularized=None):
    if regularized is None:
        regularized = spec.regularized
    surface = SurfaceRZFourier.from_vmec_input(
        SURFACE_FILE,
        range="half period",
        nphi=spec.nphi,
        ntheta=spec.ntheta,
    )
    base_curves = create_equally_spaced_curves(
        spec.ncoils,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.5,
        order=spec.order,
        numquadpoints=spec.nquad,
    )
    base_currents = [Current(1e5) for _ in range(spec.ncoils)]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, stellsym=True)
    field = BiotSavart(coils)
    points = np.ascontiguousarray(surface.gamma().reshape((-1, 3)))
    field.set_points(points)

    flux = SquaredFlux(surface, field)
    lengths = [CurveLength(curve) for curve in base_curves]
    components = {"quadratic_flux": flux, "curve_length_sum": sum(lengths)}
    if regularized:
        all_curves = [coil.curve for coil in coils]
        components.update(
            {
                "coil_coil_distance": CurveCurveDistance(
                    all_curves, 0.1, num_basecurves=spec.ncoils
                ),
                "coil_surface_distance": CurveSurfaceDistance(all_curves, surface, 0.3),
                "curvature": sum(
                    LpCurveCurvature(curve, 2, 5.0) for curve in base_curves
                ),
                "mean_squared_curvature_penalty": sum(
                    QuadraticPenalty(MeanSquaredCurvature(curve), 5.0, "max")
                    for curve in base_curves
                ),
            }
        )
        objective = (
            flux
            + 1e-6 * components["curve_length_sum"]
            + 1000.0 * components["coil_coil_distance"]
            + 10.0 * components["coil_surface_distance"]
            + 1e-6 * components["curvature"]
            + 1e-6 * components["mean_squared_curvature_penalty"]
        )
    else:
        objective = flux + QuadraticPenalty(components["curve_length_sum"], 18.0, "max")
    return surface, base_curves, field, components, objective


def revision():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def environment():
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "jax": jax.__version__,
        "simsopt": simsopt.__version__,
        "simsopt_revision": revision(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def main():
    args = parse_args()
    spec = get_problem(args.problem)
    surface, curves, field, components, objective = build_problem(spec)
    initial_x = objective.x.copy()
    cotangent = np.ones((surface.gamma().size // 3, 3))

    def curve_evaluation():
        for curve in curves:
            curve.invalidate_cache()
            curve.gamma()
            curve.gammadash()

    def field_evaluation():
        field.clear_cached_properties()
        field.B()

    def field_vjp():
        field.clear_cached_properties()
        field.B_vjp(cotangent)

    def combined_value_gradient():
        objective.x = initial_x
        return objective.J(), objective.dJ()

    timings = {
        "curve_evaluation": measure(curve_evaluation, args.warmup, args.repeats),
        "biot_savart_field": measure(field_evaluation, args.warmup, args.repeats),
        "biot_savart_vjp": measure(field_vjp, args.warmup, args.repeats),
        "combined_value_gradient": measure(
            combined_value_gradient, args.warmup, args.repeats
        ),
        "components": {
            name: measure(component.J, args.warmup, args.repeats)
            for name, component in components.items()
        },
    }

    optimization = None
    if args.optimization_iterations:
        objective.x = initial_x

        def fun(x):
            objective.x = x
            return objective.J(), objective.dJ()

        start = time.perf_counter()
        result = minimize(
            fun,
            initial_x,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": args.optimization_iterations, "maxcor": 300},
            tol=1e-15,
        )
        optimization = {
            "seconds": time.perf_counter() - start,
            "iterations": int(result.nit),
            "evaluations": int(result.nfev),
            "success": bool(result.success),
            "status": int(result.status),
            "final_objective": float(result.fun),
        }

    objective.x = initial_x
    result = {
        "schema_version": 1,
        "problem": spec.as_dict(),
        "dimensions": {
            "physical_coils": len(field.coils),
            "surface_points": surface.gamma().size // 3,
            "free_variables": initial_x.size,
        },
        "objective_components": sorted(components),
        "reference": {
            "initial_objective": float(objective.J()),
            "initial_gradient_norm": float(np.linalg.norm(objective.dJ())),
        },
        "environment": environment(),
        "timings": timings,
        "optimization": optimization,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
