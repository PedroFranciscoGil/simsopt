#!/usr/bin/env python

"""
qss_script_jax_qi.py — QSS Optimisation for Quasi-Isodynamic (QI) configs (JAX variant)
========================================================================================

Same architecture as qss_script_jax.py (AL1 + AL2 augmented-Lagrangian
with a fully JAX-differentiable squared-flux objective) but targeting
quasi-isodynamic (QI) equilibria instead of quasi-axisymmetric (QA).

Surface targets (from qi_goodman/simsopt_driver.py):
- QIBResidual2  — main QI omnigenity residual (objective, returns vector)
- AspectRatioPen — aspect ratio penalty, zero when aspect <= threshold
- MirrorRatioPen — mirror ratio penalty, zero when mirror <= threshold
- MaxElongationPen — elongation penalty, zero when elongation <= threshold

Initial condition: warm_start v20260504_v1
(NFP=3, stellsym, vacuum, R0=1.0)
NOTE: coil parameters below are inherited from the W7-X scaling
(auglag_w7x.py) and will need retuning for this smaller device.
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
from simsopt import make_optimizable
from simsopt._core.util import ObjectiveFailure
from simsopt.mhd import Vmec
from simsopt.objectives import SquaredFluxJaxFull
from simsopt.util import MpiPartition, proc0_print
from simsopt.solve import (
    least_squares_mpi_solve, least_squares_mpi_solve_qss, auglag_solver,
)
from simsopt.geo import create_equally_spaced_curves, curves_to_vtk
from simsopt.field import (
    regularization_circ, coils_via_symmetries, Current, BiotSavart,
)
from simsopt.field.force import LpCurveForce

from qss_helpers import (
    Objective, Constraint,
    build_al_subproblem, update_dual, al_diagnostics,
    diagnose_al2_multipliers,
    build_coil_objectives, fixed_surface, compute_bn_modb,
    surface_vtk_point_data, reinit_coils_to_surface_scaled_circular,
    print_coil_properties, alm_reset_decision_from_surface_change,
    update_alm_warmstart_state, make_ls_callbacks, ArclengthGaugeFix,
)

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
import optimization_vtk_animation

# Import QI-specific targets from qi_goodman/Targets.py
# (QIBResidual2 lives here; the auglag_for_surface copy does not have it)
_qi_targets_dir = os.path.join(_script_dir, 'qi_goodman')
if _qi_targets_dir not in sys.path:
    sys.path.insert(0, _qi_targets_dir)
from Targets import QIBResidual2, AspectRatioPen, MirrorRatioPen, MaxElongationPen

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
_qss_argparser = argparse.ArgumentParser(
    description="QSS Optimisation for QI configurations with JAX pipeline.",
)

_qss_argparser.add_argument(
    "--no-coils", dest="enable_coils", action="store_false", default=True,
    help="Disable coil constraints (equilibrium-only optimisation). Default: coils enabled.",
)

_qss_argparser.add_argument(
    "--run-dir", type=str, default=None,
    help="If set, all outputs are written to this directory (created if missing).",
)

_qss_argparser.add_argument(
    "--demo-jax", action="store_true", default=False,
    help="Run the JAX diagnostics demo (J, grad, Hessian timing) before optimisation.",
)

_qss_argparser.add_argument(
    "--hessian-method", type=str, default="fd", choices=["fd", "jax"],
    help="Method for the squared-flux term of the AL2 Lagrangian Hessian that "
         "Eq.(18) inverts: 'fd' (finite differences, the validated production "
         "default) or 'jax' (analytic autodiff, diagnostics/A-B only). "
         "Production runs with FD-estimated Hessians; default: fd.",
)

_qss_argparser.add_argument(
    "--track-hessian", action="store_true", default=False,
    help="Always compute FD gradients alongside JAX for comparison tracking. "
         "Expensive: O(n_surf) extra evaluations per AL1 call.",
)

_qss_argparser.add_argument(
    "--n-continuation-steps", type=int, default=7,
    help="Number of continuation steps (default: 7). Max modes go 2..N+1.",
)

_qss_argparser.add_argument(
    "--vmec-input", type=str, default=None,
    help="Path to VMEC input file. Default: <script_dir>/qi_goodman/zenodo_goodman/"
         "configurations/warm_start/input.v20260504_v1",
)

_qss_argparser.add_argument(
    "--skip-surface-opt", action="store_true", default=False,
    help="Skip the AL1+AL2 surface continuation loop and run only the final "
         "coil optimisation on the input surface as given.",
)

_qss_args = _qss_argparser.parse_args()

ENABLE_COILS = bool(_qss_args.enable_coils)
DEMO_JAX = bool(_qss_args.demo_jax)
HESSIAN_METHOD = _qss_args.hessian_method
TRACK_HESSIAN = bool(_qss_args.track_hessian)
N_CONTINUATION_STEPS = _qss_args.n_continuation_steps
SKIP_SURFACE_OPT = bool(_qss_args.skip_surface_opt)


# ---------------------------------------------------------------------------
# AL1 hyper-parameters
# ---------------------------------------------------------------------------

AL_OUTER_ITER  = 5      # dual updates per continuation step
RHO_INIT       = 1.0    # initial penalty coefficient
RHO_SCALE      = 2.0    # multiplicative growth for rho each AL1 iter
RHO_MAX        = 1e4    # upper bound on rho
CONSTRAINT_TOL = 1e-4   # skip dual update if |c_i| already below this
LAM_MAX        = 1e3    # cap |lambda_i| to prevent multiplier explosion

# Coil geometry constants (scaled for W7-X from auglag_w7x.py)
COIL_RADIUS_FACTOR   = 3.6
COIL_QUADPOINTS      = 256    # higher resolution for W7-X coils
ARC_LENGTH_SAMPLES   = 2048
HESSIAN_SIZE_CUTOFF  = 500

# Eq.(18) implicit-solve regularisation (Option B / B1). Truncated-SVD solve
# of H^{-1}(dc/dy) that floors |singular values| below EQ18_IMPLICIT_RCOND *
# sigma_max, bounding near-null amplification at 1/(rcond*sigma_max). Lets a
# tight, active squared-flux constraint keep a usable Eq.(18) gradient on the
# flat coil-optimum valley. For the FD Hessian (--hessian-method fd), whose
# near-null eigenvalues are noisier/indefinite, 1e-5 is a safer floor than the
# 1e-6 library default; raise toward 1e-4 if the Jf gradient still spikes,
# lower toward 1e-6 if it looks over-damped. Set EQ18_IMPLICIT_REGULARIZE
# False to restore the exact (un-regularised) solve for A/B comparison.
EQ18_IMPLICIT_REGULARIZE = True
EQ18_IMPLICIT_RCOND      = 1e-5

# AL2-objective gauge fix. Adds w * sum ArclengthVariation(coil) to the inner
# coil objective, which fixes the dominant curve-REPARAMETRISATION null
# direction of the coil optimum (orthogonal to the flux gradient, so it lifts
# the degeneracy WITHOUT fighting the objective). This improves the Eq.(18)
# coil-Hessian conditioning (null_frac ~0.22 -> ~0.02-0.05 at full resolution),
# making the squared-flux adjoint better-posed. 0 = off. Validation showed 3e-1
# lifts more; 1e-1 is a moderate default with less coil bias. NOTE: this lifts
# the gauge degeneracy but NOT the residual indefinite-subproblem
# non-convergence (see results_outline.md) — gradients are improved, not
# cleanly validated at production resolution.
EQ18_ARCLENGTH_GAUGE     = 1e-1

# QI residual calculation parameters (from qi_goodman/simsopt_driver.py)
QI_NPHI    = 301   # toroidal grid points for B along each well
QI_NALPHA  = 75    # number of field lines
QI_NBJ     = 51    # bounce-point grid
QI_MPOL    = 20    # poloidal modes in Boozer transformation
QI_NTOR    = 20    # toroidal modes in Boozer transformation
QI_SARR    = [1/51, 10/51, 20/51, 30/51, 40/51, 50/51]
QI_WEIGHTS = 1e-6 / np.array([4.1e-8, 5.9e-7, 9.6e-7, 1.2e-6, 1.48e-6, 2.38e-6])

# VMEC resolution (held fixed throughout continuation, as in qi_goodman)
VMEC_MPOL = 10
VMEC_NTOR = 10


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

SEP = "=" * 60
SEP_THIN = "-" * 60

def main():
    proc0_print("")
    proc0_print(SEP)
    proc0_print("  QSS (Quasi-Single-Stage) QI Optimization -- AL1 [JAX pipeline, warm_start]")
    proc0_print(SEP)
    proc0_print("")

    # ---------------------------------------------------------------------------
    # MPI / VMEC setup
    # ---------------------------------------------------------------------------

    mpi = MpiPartition()
    mpi.write()

    _default_vmec_input = os.path.join(
        _script_dir, 'qi_goodman', 'zenodo_goodman', 'configurations',
        'warm_start', 'input.v20260504_v1')
    filename = _qss_args.vmec_input or os.path.abspath(_default_vmec_input)
    vmec = Vmec(filename, mpi=mpi, verbose=False,
                surf_type='JaxSurfaceRZFourier', range_surface='full torus', nphi = 72, ntheta = 72)
    surf = vmec.boundary

    # ---------------------------------------------------------------------------
    # QI equilibrium targets
    # ---------------------------------------------------------------------------
    ASPECT_THRESHOLD     = 10   # penalise aspect ratio above this
    MIRROR_THRESHOLD     = 0.3   # penalise mirror ratio above this
    ELONGATION_THRESHOLD = 5    # penalise max elongation above this
    WELL_TARGET          = 0.015    # penalise vacuum well W below this (W>0 is stable)

    # QI omnigenity residual (main objective — returns a vector of residuals)
    # QIBResidual2 calls vmec.run() internally.
    qi_opt = make_optimizable(
        QIBResidual2, vmec, QI_SARR,
        nphi=QI_NPHI, nalpha=QI_NALPHA, nBj=QI_NBJ,
        mpol=QI_MPOL, ntor=QI_NTOR, weights=QI_WEIGHTS,
    )

    # Penalty targets (return scalar, zero when satisfied)
    # AspectRatioPen and MirrorRatioPen call vmec.run() internally.
    aspect_opt = make_optimizable(AspectRatioPen, vmec, t=ASPECT_THRESHOLD)
    mirror_opt = make_optimizable(MirrorRatioPen, vmec, t=MIRROR_THRESHOLD)

    # MaxElongationPen does NOT call vmec.run(); wrap it.
    def _max_elongation_con(v):
        v.run()
        return MaxElongationPen(v, t=ELONGATION_THRESHOLD)

    elong_opt = make_optimizable(_max_elongation_con, vmec)

    # Vacuum-well penalty: zero when W >= WELL_TARGET, else (WELL_TARGET - W)**2.
    # vmec.vacuum_well() calls vmec.run() internally.
    def _vacuum_well_con(v):
        W = v.vacuum_well()
        return max(0.0, WELL_TARGET - W) ** 2

    well_opt = make_optimizable(_vacuum_well_con, vmec)

    # ---------------------------------------------------------------------------
    # Coil optimisation parameters (AL2 subproblem)
    # Scaled for W7-X geometry (R0≈5.5m), reference: auglag_w7x.py
    # ---------------------------------------------------------------------------
    LENGTH_TARGET        = 2.4    # per-coil max length [m] (scaled from QI: 3.5 * R0_w7x/R0_qi)
    FLUX_THRESHOLD       = 1e-6  # squared-flux penalty threshold
    CC_THRESHOLD         = 0.10   # coil-coil min distance [m] (from auglag_w7x.py)
    CS_THRESHOLD         = 0.10    # coil-surface min distance [m] (from auglag_w7x.py)
    MSC_THRESHOLD        = 15    # mean-squared-curvature threshold (from auglag_w7x.py)
    CURVATURE_THRESHOLD  = 6   # max curvature [1/m] (from auglag_w7x.py)
    FORCE_THRESHOLD      = 100    # coil force threshold [MN/m] (from auglag_w7x.py)
    ncoils = 5
    order  = 4                    # higher Fourier order for W7-X coil complexity

    a      = 0.15                 # regularisation cross-section radius [m] (from auglag_w7x.py)
    R0     = surf.get_rc(0, 0)
    R1     = surf.get_rc(1, 0) * 3   # W7-X coil init convention (from auglag_w7x.py)
    # Total current: 15 kA × 108 turns × 5 coil types ≈ 8.1 MA (W7-X scale)
    total_current = 8.1e6

    base_curves = create_equally_spaced_curves(
        ncoils, surf.nfp, stellsym=surf.stellsym, R0=R0, R1=R1, order=order,
        numquadpoints=COIL_QUADPOINTS)
    base_currents = [Current(total_current / ncoils * 1e-5) * 1e5 for _ in range(ncoils-1)]
    total_current_obj = Current(total_current)
    total_current_obj.fix_all()
    base_currents += [total_current_obj - sum(base_currents)]
    regularizations = [regularization_circ(a) for _ in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, surf.stellsym, regularizations=regularizations)
    base_coils_full = coils[:ncoils]

    curves = [c.curve for c in coils]
    base_curves_full = curves[:ncoils]

    # Place coils along the magnetic axis from the start, matching the
    # reinit pathway used during AL2 and final coil optimisation.
    reinit_coils_to_surface_scaled_circular(
        surf, base_curves_full, base_currents, ncoils, order,
        total_current, COIL_RADIUS_FACTOR, ARC_LENGTH_SAMPLES,
    )

    # ---------------------------------------------------------------------------
    # BiotSavart field
    # ---------------------------------------------------------------------------
    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))

    # ---- AL2 objective & constraints (coil subproblem) ----

    _coil_objs = build_coil_objectives(
        surf, bs, base_curves_full, base_coils_full, curves, ncoils, LENGTH_TARGET,
        hessian_method=HESSIAN_METHOD, flux_threshold=FLUX_THRESHOLD,
        cc_threshold=CC_THRESHOLD, cs_threshold=CS_THRESHOLD,
        curvature_threshold=CURVATURE_THRESHOLD, msc_threshold=MSC_THRESHOLD)
    Jccdist = _coil_objs["Jccdist"]
    Jcsdist = _coil_objs["Jcsdist"]
    Jlink = _coil_objs["Jlink"]
    Jmscs = _coil_objs["Jmscs"]

    # Coil force constraint (Lorentz force on base coils from all coils)
    Jforce = LpCurveForce(base_coils_full, coils, p=2.0,
                          threshold=FORCE_THRESHOLD, downsample=2)

    # Initial coil quantities (post-reinit, pre-optimisation).
    proc0_print("")
    proc0_print(SEP_THIN)
    proc0_print("  Initial coil quantities")
    proc0_print(SEP_THIN)
    print_coil_properties(surf, bs, Jccdist, Jcsdist, base_curves_full, Jmscs, Jlink)
    proc0_print(f"  Initial Jforce penalty: {Jforce.J():.6e}")
    proc0_print(f"  Initial currents: "
                f"{[float(c.current.get_value()) for c in base_coils_full]}")
    proc0_print("")

    # AL2 objective: the arclength gauge fix (or None when weight==0). Built
    # against bs.dof_names so the per-curve DOF mapping is layout-proof
    # (currents come first in the coil vector, then curves).
    f_al2 = (ArclengthGaugeFix(base_curves_full, list(bs.dof_names),
                               EQ18_ARCLENGTH_GAUGE)
             if EQ18_ARCLENGTH_GAUGE > 0 else None)

    def _sync_gauge(ref_obj):
        """Sync the gauge-fix .x / reference to the coil-DOF vector, INSIDE a
        surface-fixed context (where ref_obj.x is coil-only). Required before
        each AL2 solve: augmented_lagrangian_method reads f.x as its initial
        guess."""
        if f_al2 is not None:
            cx = np.asarray(ref_obj.x, dtype=np.float64).copy()
            f_al2.ensure_ref(cx)
            f_al2.x = cx

    al2_c_list = _coil_objs["c_list"] + [Jforce]
    AL2_CONSTRAINT_NAMES = ["Jf", "Jccdist", "Jcsdist", "Jmscs", "Jls", "J_kappa", "Jlink", "Jforce"]

    # ======================================================================
    # COIL CONSTRAINTS promoted to AL1
    # ======================================================================
    AL1_COIL_CONSTRAINTS = {
        "Jf":  0,
        "Jls": 4,
        "J_kappa": 5,
    }
    AL1_COIL_CONSTRAINT_NAMES   = list(AL1_COIL_CONSTRAINTS.keys())
    AL1_COIL_CONSTRAINT_INDICES = list(AL1_COIL_CONSTRAINTS.values())
    n_coil_con   = len(AL1_COIL_CONSTRAINTS)
    coil_targets = np.zeros(n_coil_con)

    # ---------------------------------------------------------------------------
    # PROBLEM SPECIFICATION for AL1
    # ---------------------------------------------------------------------------

    OBJECTIVES = [
        Objective(fn=qi_opt.J, target=0.0, label="QI"),
    ]

    EQ_CONSTRAINTS = [
        Constraint(fn=aspect_opt.J,  target=0.0, label="aspect_pen"),
        Constraint(fn=mirror_opt.J,  target=0.0, label="mirror_pen"),
        Constraint(fn=elong_opt.J,   target=0.0, label="elongation_pen"),
        Constraint(fn=well_opt.J,    target=0.0, label="well_pen"),
    ]

    # ---------------------------------------------------------------------------
    # Configuration — output directory
    # ---------------------------------------------------------------------------

    _timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _coil_tag = "coils" if ENABLE_COILS else "eqonly"
    _constraint_str = (
        f"L{LENGTH_TARGET}_cc{CC_THRESHOLD}_cs{CS_THRESHOLD}_msc{MSC_THRESHOLD}"
        f"_curv{CURVATURE_THRESHOLD}_{_coil_tag}"
    )
    if _qss_args.run_dir:
        OUT_DIR = os.path.abspath(_qss_args.run_dir)
        if not OUT_DIR.endswith(os.sep):
            OUT_DIR += os.sep
    else:
        OUT_DIR = os.path.join(_script_dir, "output", f"qss_jax_qi_warmstart_al1_{_constraint_str}_{_timestamp}", "")
    os.makedirs(OUT_DIR, exist_ok=True)

    EXPORT_VTK_ANIMATION = True
    vtk_exporter = (optimization_vtk_animation.OptimizationVTKExporter(OUT_DIR, export_initial=True)
                    if EXPORT_VTK_ANIMATION else None)

    proc0_print("  Setup")
    proc0_print(SEP_THIN)
    proc0_print(f"  Hessian method: {HESSIAN_METHOD}")
    proc0_print(f"  Eq.18 implicit reg: {EQ18_IMPLICIT_REGULARIZE} "
                f"(rcond={EQ18_IMPLICIT_RCOND:.1e})")
    proc0_print(f"  Eq.18 arclength gauge fix: {EQ18_ARCLENGTH_GAUGE:.1e} "
                f"({'ON' if f_al2 is not None else 'OFF'})")
    proc0_print(f"  Surface DOFs: {surf.x.shape}")
    proc0_print(f"  VMEC DOFs:   {vmec.x.shape}")
    proc0_print(f"  NFP:         {surf.nfp}")
    proc0_print(f"  Output dir:   {OUT_DIR}")
    _targets_str = ", ".join(f"{n}=0" for n in AL1_COIL_CONSTRAINT_NAMES)
    proc0_print(f"  AL1 coil constraints ({n_coil_con}): {_targets_str}")
    proc0_print(f"  QI params: nphi={QI_NPHI}, nalpha={QI_NALPHA}, nBj={QI_NBJ}, "
                f"mpol={QI_MPOL}, ntor={QI_NTOR}, n_surfaces={len(QI_SARR)}")
    proc0_print(f"  Thresholds: aspect<={ASPECT_THRESHOLD}, mirror<={MIRROR_THRESHOLD}, "
                f"elongation<={ELONGATION_THRESHOLD}, well>={WELL_TARGET}")
    proc0_print("")
    surf.to_vtk(OUT_DIR + "surf_init_qss", extra_data=surface_vtk_point_data(surf, bs))
    if vtk_exporter is not None and mpi.proc0_world:
        vtk_exporter.export_frame(
            surf, curves, vtk_exporter.next_frame(),
            surface_extra_data=surface_vtk_point_data(surf, bs),
            only_on_proc0=True, is_proc0=mpi.proc0_world,
        )

    # Run VMEC on the initial equilibrium and copy its wout into OUT_DIR.
    vmec.files_to_delete = []
    _orig_cwd_init = os.getcwd()
    try:
        os.chdir(OUT_DIR)
        vmec.run()
    finally:
        os.chdir(_orig_cwd_init)
    if mpi.proc0_world and vmec.output_file and os.path.exists(vmec.output_file):
        _initial_wout = os.path.join(OUT_DIR, "wout_initial.nc")
        shutil.copy2(vmec.output_file, _initial_wout)
        proc0_print(f"  Saved initial wout to {_initial_wout}")

    # ---------------------------------------------------------------------------
    # Initial equilibrium metrics — print as a table + save to JSON.
    # Mirrors the final-metrics block so before/after are directly comparable.
    # ---------------------------------------------------------------------------
    qi_residuals_init = qi_opt.J()
    qi_total_init      = float(np.sum(qi_residuals_init ** 2))
    aspect_pen_init    = float(aspect_opt.J())
    mirror_pen_init    = float(mirror_opt.J())
    elong_pen_init     = float(elong_opt.J())
    well_pen_init      = float(well_opt.J())
    aspect_init        = float(vmec.aspect())
    mean_iota_init     = float(vmec.mean_iota())
    vacuum_well_init   = float(vmec.vacuum_well())
    area_init          = float(surf.area())
    volume_init        = float(surf.volume())

    proc0_print("")
    proc0_print(SEP)
    proc0_print("  Initial equilibrium metrics")
    proc0_print(SEP)
    proc0_print(f"  {'Aspect ratio':<20s}{aspect_init:>20.6f}")
    proc0_print(f"  {'Mean iota':<20s}{mean_iota_init:>20.6f}")
    proc0_print(f"  {'QI residual':<20s}{qi_total_init:>20.6e}")
    proc0_print(f"  {'Mirror ratio pen':<20s}{mirror_pen_init:>20.6e}")
    proc0_print(f"  {'Aspect ratio pen':<20s}{aspect_pen_init:>20.6e}")
    proc0_print(f"  {'Elongation pen':<20s}{elong_pen_init:>20.6e}")
    proc0_print(f"  {'Vacuum well pen':<20s}{well_pen_init:>20.6e}")
    proc0_print(f"  {'Magnetic well':<20s}{vacuum_well_init:>20.6e}")
    proc0_print(f"  {'Surface area':<20s}{area_init:>20.6f}")
    proc0_print(f"  {'Surface volume':<20s}{volume_init:>20.6f}")
    proc0_print(SEP)
    proc0_print("")

    if mpi.proc0_world:
        _initial_metrics = {
            "aspect_ratio":         aspect_init,
            "mean_iota":            mean_iota_init,
            "qi_residual_sum_sq":   qi_total_init,
            "mirror_ratio_penalty": mirror_pen_init,
            "aspect_ratio_penalty": aspect_pen_init,
            "elongation_penalty":   elong_pen_init,
            "vacuum_well_penalty":  well_pen_init,
            "vacuum_well":          vacuum_well_init,
            "area":                 area_init,
            "volume":               volume_init,
            "nfp":                  int(surf.nfp),
            "stellsym":             bool(surf.stellsym),
            "vmec_input":           filename,
        }
        _initial_metrics_path = os.path.join(OUT_DIR, "initial_metrics.json")
        with open(_initial_metrics_path, "w") as _imf:
            json.dump(_initial_metrics, _imf, indent=2)
        proc0_print(f"  Saved initial metrics to {_initial_metrics_path}")

    # ---------------------------------------------------------------------------
    # (Optional) Demonstrate: J, gradient, Hessian via pure JAX
    # ---------------------------------------------------------------------------
    if DEMO_JAX:
        Jf_jax_full = SquaredFluxJaxFull(
            surface=surf,
            base_curves=base_curves_full,
            base_currents=[c.current for c in base_coils_full],
            nfp=surf.nfp,
            stellsym=surf.stellsym,
            target=None,
            definition="normalized",
        )

        proc0_print(f"  SquaredFluxJaxFull created:")
        proc0_print(f"    n_surf_dofs  = {Jf_jax_full.n_surf_dofs}")
        proc0_print(f"    n_curve_dofs = {Jf_jax_full.n_curve_dofs_total}")
        proc0_print(f"    n_current    = {Jf_jax_full.n_current_dofs}")
        proc0_print(f"    n_dofs_total = {Jf_jax_full.n_dofs}")
        proc0_print("")

        proc0_print(SEP_THIN)
        proc0_print("  Demonstrating fully JAX-traced squared flux:")
        proc0_print(SEP_THIN)

        t0 = time.time()
        J_val = Jf_jax_full.J()
        t1 = time.time()
        proc0_print(f"  J  = {J_val:.10e}  ({t1-t0:.3f}s, includes JIT compilation)")

        t0 = time.time()
        J_val2 = Jf_jax_full.J()
        t1 = time.time()
        proc0_print(f"  J  = {J_val2:.10e}  ({t1-t0:.3f}s, cached/warm)")

        t0 = time.time()
        g = Jf_jax_full.dJ()
        t1 = time.time()
        proc0_print(f"  ||grad||     = {np.linalg.norm(g):.10e}  ({t1-t0:.3f}s, includes JIT)")
        proc0_print(f"    grad_surf  = {np.linalg.norm(g[Jf_jax_full.surf_slice]):.6e}")
        proc0_print(f"    grad_curve = {np.linalg.norm(g[Jf_jax_full.curve_slice]):.6e}")
        proc0_print(f"    grad_curr  = {np.linalg.norm(g[Jf_jax_full.current_slice]):.6e}")

        t0 = time.time()
        g2 = Jf_jax_full.dJ()
        t1 = time.time()
        proc0_print(f"  ||grad|| (warm) = {np.linalg.norm(g2):.10e}  ({t1-t0:.3f}s)")

        v = np.random.default_rng(42).standard_normal(Jf_jax_full.n_dofs)
        v /= np.linalg.norm(v)
        t0 = time.time()
        Hv = Jf_jax_full.hvp(v)
        t1 = time.time()
        proc0_print(f"  ||H @ v||   = {np.linalg.norm(Hv):.10e}  ({t1-t0:.3f}s, HVP, includes JIT)")

        if Jf_jax_full.n_dofs <= HESSIAN_SIZE_CUTOFF:
            t0 = time.time()
            H = Jf_jax_full.d2J()
            t1 = time.time()
            proc0_print(f"  Hessian shape = {H.shape}  ({t1-t0:.3f}s, via d2J())")
            proc0_print(f"    ||H||_F    = {np.linalg.norm(H):.6e}")
            proc0_print(f"    symmetry   = {np.linalg.norm(H - H.T):.6e} (should be ~0)")

            H_ss = Jf_jax_full.d2J_surf_surf(H)
            H_cc = Jf_jax_full.d2J_coil_coil(H)
            H_sc = Jf_jax_full.d2J_surf_coil(H)
            proc0_print(f"    ||H_ss||   = {np.linalg.norm(H_ss):.6e}  (surface-surface)")
            proc0_print(f"    ||H_cc||   = {np.linalg.norm(H_cc):.6e}  (coil-coil)")
            proc0_print(f"    ||H_sc||   = {np.linalg.norm(H_sc):.6e}  (surface-coil cross)")

            Hv_direct = H @ v
            proc0_print(f"    ||Hv_hvp - Hv_direct|| = {np.linalg.norm(Hv - Hv_direct):.6e}")

            eigvals = np.linalg.eigvalsh(H)
            proc0_print(f"    eigenvalues: min={eigvals[0]:.6e}, max={eigvals[-1]:.6e}")
            n_neg = np.sum(eigvals < 0)
            proc0_print(f"    negative eigenvalues: {n_neg}/{len(eigvals)}")
        else:
            proc0_print(f"  Skipping full Hessian (n_dofs={Jf_jax_full.n_dofs} > {HESSIAN_SIZE_CUTOFF})")
            proc0_print(f"  Use hvp() for Hessian-vector products instead.")

        proc0_print("")

    # AL2 solver parameters
    al2_maxiter        = 100
    al2_max_iter_auglag = 6
    al2_tau            = 5
    al2_grad_tol       = 1e-6
    al2_c_tol          = 1e-6
    al2_mu_initial     = 10

    # Surface optimisation parameters.
    # Forward FD: halves the Jacobian cost vs centered (nparams+1 evals
    # instead of 2*nparams) and keeps the per-evaluation reduce buffer under
    # the MPI 2 GiB message limit for the large QI residual vector. Truncation
    # error is O(h) instead of O(h^2), but the VMEC residual noise floor
    # (~1e-10..1e-8) dominates anyway, so step ~1e-5 (per Fu et al. (2025)
    # Taylor tests, ~1e-3 * DOF-scale) stays the practical sweet spot;
    # rel_step picks that up for heterogeneous Fourier coefficients.
    diff_method              = "forward"
    finite_difference_abs_step = 1e-5
    finite_difference_rel_step = 1e-5
    max_nfev                 = 20
    REINIT_COILS_EACH_ALM    = True
    SURFACE_DOF_RESET_THRESHOLD = 1e-15

    # History
    J1_J2_history = []
    _last_J2_holder: list = [None]
    alm_warmstart_state = {"prev_surface_x": None, "prev_lag_mul": None, "prev_mu": None}

    # Hessian tracking
    hessian_tracking = []
    _eq18_eval_counter = [0]

    # AL2 multiplier history (per Fu et al. App. B / Fig. 16 — detect
    # 'dropped-active' inner constraints whose lambda collapses to ~0).
    al2_multiplier_history = {name: [] for name in AL2_CONSTRAINT_NAMES}


    # ---------------------------------------------------------------------------
    # Main continuation + AL1 loop
    # ---------------------------------------------------------------------------

    # Initialise AL1 dual variables
    n_eq_con = len(EQ_CONSTRAINTS)

    lam_eq = np.zeros(n_eq_con)
    lam_coil = np.zeros(n_coil_con)
    rho = RHO_INIT

    if mpi.proc0_world:
        rng = np.random.default_rng(42)
        lam_eq[:] = rng.uniform(0.0, 1.0, size=n_eq_con)
        lam_coil[:] = rng.uniform(0.0, 1.0, size=n_coil_con)
    mpi.comm_world.Bcast(lam_eq, root=0)
    mpi.comm_world.Bcast(lam_coil, root=0)

    proc0_print(f"  Initial lam_eq   = {lam_eq}")
    proc0_print(f"  Initial lam_coil = {lam_coil}")
    proc0_print(f"  Initial rho      = {rho}")
    proc0_print("")

    _last_coil_metric_values = np.full(n_coil_con, np.nan)
    # Surface DOFs at which _last_coil_metric_values was computed. The LS
    # solver also evaluates rejected trial points, so before the dual update
    # we must check the metrics correspond to the ACCEPTED iterate.
    _last_coil_metrics_x = [None]

    if SKIP_SURFACE_OPT:
        proc0_print("")
        proc0_print(SEP)
        proc0_print("  --skip-surface-opt set: bypassing AL1+AL2 surface continuation.")
        proc0_print("  Going directly to final coil optimisation on the input surface.")
        proc0_print(SEP)
        proc0_print("")

    # Fallback surface for the graceful-stop path: the last full Fourier
    # vector at which VMEC converged. local_full_x is parametrization-only, so
    # its size is constant across continuation steps (max_mode only changes the
    # fix mask, not the surface resolution) and it is safe to restore later.
    _continuation_aborted = False
    last_good_full_x = np.asarray(surf.local_full_x, dtype=np.float64).copy()

    for step in range(0 if SKIP_SURFACE_OPT else N_CONTINUATION_STEPS):
        # QI continuation: modes 2, 3, 4, ... (matching qi_goodman)
        max_mode = step + 2

        # Fixed VMEC resolution throughout (QI needs higher resolution)
        vmec.indata.mpol = VMEC_MPOL
        vmec.indata.ntor = VMEC_NTOR

        proc0_print("")
        proc0_print(SEP)
        proc0_print(f"  Continuation step {step+1}/{N_CONTINUATION_STEPS} -- max_mode={max_mode}, "
                    f"mpol=ntor={vmec.indata.mpol}")
        proc0_print(f"  lam_eq={lam_eq}, lam_coil={lam_coil}, rho={rho:.4g}")
        proc0_print(f"  Previous VMEC iteration: {vmec.iter}")
        proc0_print(SEP)
        proc0_print("")

        # ------------------------------------------------------------------
        # Sync surf state across ALL ranks before the next FD jacobian.
        #
        # Order is critical: fix the mask FIRST so every rank has the same
        # len(surf.x), then Bcast the free DOFs. Doing Bcast before the
        # mask reset (as the previous version did) means the buffer sizes
        # can still disagree across leaders, which Open MPI may surface as
        # MPI_ERR_TRUNCATE or — depending on transport — as a silent hang.
        #
        # `local_full_x` is parametrization-only (size depends on mpol/
        # ntor/stellsym), so it's always the same size on every rank.
        # `surf.x` after fix_all + fixed_range is also parametrization-
        # determined, identical everywhere. Broadcasting both ensures the
        # full Fourier vector AND the fix-mask-determined free DOFs agree.
        # ------------------------------------------------------------------
        # Barrier 1 — pin a hang here (rather than inside Bcast) if any
        # rank is somehow behind from the previous step.
        mpi.comm_world.Barrier()

        # (1) Sync the FULL Fourier vector — same size on every rank
        # regardless of fix state.
        _full = np.asarray(surf.local_full_x, dtype=np.float64).copy()
        mpi.comm_world.Bcast(_full, root=0)
        surf.local_full_x = _full

        # (2) Reset the fix mask deterministically on every rank.
        surf.fix_all()
        surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
        surf.fix("rc(0,0)")

        # (3) Sync the free DOFs (sizes now guaranteed equal) so the FD
        # finite-difference Jacobian sees matching x0 on every leader.
        _x_buf = np.asarray(surf.x, dtype=np.float64).copy()
        mpi.comm_world.Bcast(_x_buf, root=0)
        surf.x = _x_buf

        # Make per-rank surf.x size visible so a future size mismatch is
        # diagnosed by a print, not by MPI_ERR_TRUNCATE.
        _local_n = len(surf.x)
        _sizes = mpi.comm_world.allgather(_local_n)
        proc0_print(f"  surf free DOFs per rank after sync: {_sizes}")
        if len(set(_sizes)) != 1:
            raise RuntimeError(
                f"surf.x size differs across ranks: {_sizes}. "
                f"DOF sync is broken; aborting before MPI Bcast truncates.")
        # Barrier 2 — fence off the sync block from the next AL1 iter.
        mpi.comm_world.Barrier()
        if len(set(_sizes)) != 1:
            raise RuntimeError(
                f"surf.x size differs across ranks: {_sizes}. "
                f"DOF sync is broken; aborting before MPI Bcast truncates.")

        # ------------------------------------------------------------------
        # AL1 outer loop
        # ------------------------------------------------------------------
        for al_iter in range(AL_OUTER_ITER):

            proc0_print(SEP_THIN)
            proc0_print(f"  AL1 outer iter {al_iter+1}/{AL_OUTER_ITER}  "
                         f"(max_mode={max_mode}, rho={rho:.4g})")
            proc0_print(SEP_THIN)

            # Checkpoint the AL1 state at the START of this outer iter so
            # verify_eq18.py can reproduce the configuration even if the run
            # crashes/stops mid-way. Two files: a step-tagged snapshot for
            # post-hoc analysis, and an "al1_checkpoint_latest.json" rolling
            # pointer so the verifier can be aimed at the most recent state
            # without grepping filenames.
            if mpi.proc0_world:
                _ckpt = {
                    "step":       int(step),
                    "al_iter":    int(al_iter),
                    "max_mode":   int(max_mode),
                    "rho":        float(rho),
                    "lam_eq":     lam_eq.tolist(),
                    "lam_coil":   lam_coil.tolist(),
                    "surface_dofs": np.asarray(surf.x).tolist(),
                    "vmec_dofs":  np.asarray(vmec.x).tolist(),
                    "vmec_input": filename,
                }
                _ckpt_path = os.path.join(
                    OUT_DIR,
                    f"al1_checkpoint_step{step:02d}_iter{al_iter:02d}.json")
                with open(_ckpt_path, "w") as _cf:
                    json.dump(_ckpt, _cf, indent=2)
                with open(os.path.join(OUT_DIR, "al1_checkpoint_latest.json"),
                          "w") as _cf:
                    json.dump(_ckpt, _cf, indent=2)

            # 1. Build AL1 subproblem (QI objectives + penalty constraints)
            al_prob = build_al_subproblem(OBJECTIVES, EQ_CONSTRAINTS, lam_eq, rho)

            if not ENABLE_COILS:
                # ---- Equilibrium-only (no coil constraints) ----
                iter_cb, jac_cb = make_ls_callbacks(
                    is_qss=False, max_mode=max_mode,
                    surf=surf, curves=curves, out_dir=OUT_DIR, bs=bs,
                    history=J1_J2_history,
                    step=step, al_iter=al_iter, al_outer_iter=AL_OUTER_ITER,
                    n_continuation_steps=N_CONTINUATION_STEPS,
                    vtk_exporter=vtk_exporter, is_proc0=mpi.proc0_world,
                    j2_holder=None,
                )
                least_squares_mpi_solve(
                    al_prob, mpi, grad=True,
                    abs_step=finite_difference_abs_step,
                    rel_step=finite_difference_rel_step,
                    diff_method=diff_method,
                    max_nfev=max_nfev,
                    iteration_callback=iter_cb,
                    jac_callback=jac_cb,
                )
            else:
                # ---- QSS: AL1 with coil constraints via eq.18 ----
                _alm_run = [0]

                _rho_closure = float(rho)
                _lam_coil_closure = lam_coil.copy()
                _coil_targets_closure = coil_targets.copy()
                _al_iter_closure = int(al_iter)
                _max_mode_closure = int(max_mode)
                _step_closure = int(step)

                def compute_J2_and_grad(_rho_c=_rho_closure,
                                        _lam_c=_lam_coil_closure,
                                        _targets_c=_coil_targets_closure,
                                        _al_iter=_al_iter_closure,
                                        _max_mode=_max_mode_closure,
                                        _step=_step_closure,
                                        compute_grad=True):
                    use_initial, current_surface_x, delta = \
                        alm_reset_decision_from_surface_change(
                            surf, alm_warmstart_state, SURFACE_DOF_RESET_THRESHOLD)
                    if use_initial:
                        msg = ("initial coils/multipliers (first iter)." if delta is None
                               else f"reset (max |dsurf|={delta:.3e} > {SURFACE_DOF_RESET_THRESHOLD:.3e}).")
                        proc0_print(f"  AL2 init: {msg}")
                    else:
                        proc0_print(f"  AL2 init: warm-start (max |dsurf|={delta:.3e}).")

                    if use_initial and REINIT_COILS_EACH_ALM:
                        reinit_coils_to_surface_scaled_circular(
                            surf, base_curves_full, base_currents, ncoils, order,
                            total_current, COIL_RADIUS_FACTOR, ARC_LENGTH_SAMPLES,
                        )
                    lag_mul_init = (None if use_initial
                                    else alm_warmstart_state.get("prev_lag_mul"))
                    mu_init = (al2_mu_initial if use_initial
                               else alm_warmstart_state.get("prev_mu", al2_mu_initial))

                    _alm_run[0] += 1
                    run_id = _alm_run[0]
                    if mpi.proc0_world:
                        surf.to_vtk(
                            OUT_DIR + f"surf_al1_{_al_iter}_al2_{run_id}_mode_{_max_mode}",
                            extra_data=surface_vtk_point_data(surf, bs))
                        curves_to_vtk(
                            curves,
                            OUT_DIR + f"coils_init_al1_{_al_iter}_al2_{run_id}_mode_{_max_mode}")

                    with fixed_surface(surf, _max_mode):
                        _sync_gauge(al2_c_list[0])   # coil-DOF sync (surf fixed)
                        solver = auglag_solver(
                            surface=surf, maxiter=al2_maxiter,
                            max_iter_auglag=al2_max_iter_auglag,
                            tau=al2_tau, f=f_al2, constraints=al2_c_list,
                            grad_tol=al2_grad_tol, c_tol=al2_c_tol,
                            max_mode=_max_mode,
                            constraint_names=AL2_CONSTRAINT_NAMES,
                            mu_initial=mu_init, lag_mul_initial=lag_mul_init,
                        )
                        x_coils, fnc, al2_lag_mul, al2_mu = solver.solve()
                        update_alm_warmstart_state(
                            alm_warmstart_state, current_surface_x,
                            al2_lag_mul, al2_mu)

                        if mpi.proc0_world:
                            curves_to_vtk(
                                curves,
                                OUT_DIR + f"coils_opt_al1_{_al_iter}_al2_{run_id}_mode_{_max_mode}")

                    all_al2_vals = solver.constraint_values()
                    coil_vals = np.array(
                        [all_al2_vals[i] for i in AL1_COIL_CONSTRAINT_INDICES])
                    _last_coil_metric_values[:] = coil_vals
                    _last_coil_metrics_x[0] = np.asarray(
                        surf.x, dtype=np.float64).copy()

                    # AL2 multiplier diagnosis (Fu et al. App. B). The history
                    # dict is also written to JSON at end-of-run for post-hoc
                    # Fig.16-style inspection.
                    _al2_diag = diagnose_al2_multipliers(
                        al2_lag_mul, all_al2_vals, AL2_CONSTRAINT_NAMES,
                        al2_multiplier_history,
                        mult_tol=1e-3, value_tol=CONSTRAINT_TOL,
                    )
                    if mpi.proc0_world:
                        proc0_print("  AL2 multiplier diagnosis (Fu et al. App. B):")
                        _flagged = False
                        for _name, _flag, _lam, _val in _al2_diag:
                            _marker = "" if _flag == "OK" else f"  <-- {_flag}"
                            proc0_print(
                                f"    {_name:10s}  lambda={_lam:+.3e}  "
                                f"c={_val:+.3e}{_marker}")
                            if _flag in ("DROPPED_ACTIVE", "SPORADIC"):
                                _flagged = True
                        if _flagged:
                            proc0_print(
                                "  [WARNING] One or more AL2 constraints flagged "
                                "as DROPPED_ACTIVE/SPORADIC — Eq. (18) gradient "
                                "may be unreliable this iter.")

                    print_coil_properties(
                        surf, bs, Jccdist, Jcsdist, base_curves_full, Jmscs, Jlink)

                    if not np.all(np.isfinite(coil_vals)):
                        proc0_print(
                            f"  [WARNING] Coil metric values contain NaN/inf: "
                            f"{coil_vals}. Returning J2=0, zero grad.")
                        alm_warmstart_state["prev_surface_x"] = None
                        alm_warmstart_state["prev_lag_mul"] = None
                        alm_warmstart_state["prev_mu"] = None
                        n_surf = len(surf.x)
                        return 0.0, np.zeros(n_surf)

                    if not compute_grad:
                        # Metrics-only refresh (used before the AL1 dual
                        # update): J2 at the current surface, no Eq.18 solve.
                        shifted = coil_vals - _targets_c + _lam_c / _rho_c
                        J2 = 0.5 * _rho_c * float(np.sum(shifted ** 2))
                        _last_J2_holder[0] = float(J2)
                        proc0_print(
                            f"  AL1 coil penalty J2 = {J2:.6e}  "
                            f"(metrics-only refresh, gradient skipped)")
                        return J2, np.zeros(len(surf.x))

                    # Eq. (18) / (19) of Fu et al. (2025) demands that the
                    # Hessian inverted is d2 L_final / d x'^2, where L_final
                    # is the *inner* AL Lagrangian at its final iteration —
                    # with the FULL inner constraint list, the converged
                    # inner multipliers (al2_lag_mul), and the converged
                    # inner penalty (al2_mu). Previously the solver here was
                    # built with only the 3 AL1-promoted constraints and the
                    # AL1 outer multipliers/rho, which is mathematically a
                    # different Lagrangian. The adjoint is then sliced to the
                    # AL1-promoted rows for J2 / grad_J2 below.
                    eq18_solver = auglag_solver(
                        surface=surf, maxiter=0, max_iter_auglag=0,
                        tau=1, f=f_al2,
                        constraints=al2_c_list,
                        max_mode=_max_mode,
                        constraint_names=AL2_CONSTRAINT_NAMES,
                        implicit_regularize=EQ18_IMPLICIT_REGULARIZE,
                        implicit_rcond=EQ18_IMPLICIT_RCOND,
                    )

                    eq18_ok = False
                    coil_grads_jax = None
                    try:
                        all_constraint_grads = eq18_solver.grad_individual(
                            x_coils, al2_lag_mul, al2_mu)
                        coil_grads_jax = np.asarray(all_constraint_grads)[
                            AL1_COIL_CONSTRAINT_INDICES]
                        if np.all(np.isfinite(coil_grads_jax)):
                            eq18_ok = True
                            coil_grads = coil_grads_jax
                        else:
                            proc0_print(
                                "  [WARNING] Eq.18 produced non-finite gradients; "
                                "falling back to FD.")
                    except Exception as e:
                        proc0_print(
                            f"  [WARNING] Eq.18 gradient computation failed: {e}")
                        proc0_print(
                            "  Falling back to finite-difference approximation.")

                    n_surf = len(surf.x)
                    coil_grads_fd = None
                    if not eq18_ok or TRACK_HESSIAN:
                        # NOTE: this FD is the PARTIAL derivative ∂c/∂y at
                        # fixed coils, not the IFT total dc/dy. It is useful
                        # only as (i) a crash-safety fallback and (ii) a
                        # qualitative sanity check; do NOT interpret JAX-vs-FD
                        # rel_err as a correctness measure of the adjoint. A
                        # true validation needs AL2 re-solved per perturbation
                        # (see --validate-eq18, TODO).
                        coil_grads_fd = np.zeros((n_coil_con, n_surf), dtype=np.float64)
                        fd_eps = finite_difference_abs_step
                        proc0_print(
                            f"  FD eq.18 ({n_surf} surface DOFs, eps={fd_eps:.1e}, central)...")
                        surf_x_orig = np.copy(surf.x)

                        def _coil_vals_at(surf_x):
                            surf.x = surf_x
                            bs.set_points(surf.gamma().reshape((-1, 3)))
                            return np.array(
                                [al2_c_list[i].J()
                                 for i in AL1_COIL_CONSTRAINT_INDICES])

                        for j in range(n_surf):
                            x_fwd = surf_x_orig.copy(); x_fwd[j] += fd_eps
                            vals_fwd = _coil_vals_at(x_fwd)
                            x_bwd = surf_x_orig.copy(); x_bwd[j] -= fd_eps
                            vals_bwd = _coil_vals_at(x_bwd)
                            coil_grads_fd[:, j] = (vals_fwd - vals_bwd) / (2.0 * fd_eps)
                        surf.x = surf_x_orig
                        bs.set_points(surf.gamma().reshape((-1, 3)))

                    if not eq18_ok:
                        coil_grads = coil_grads_fd
                        proc0_print("  Using FD gradients (JAX failed).")

                    if TRACK_HESSIAN and coil_grads_fd is not None:
                        _eq18_eval_counter[0] += 1
                        norm_fd = float(np.linalg.norm(coil_grads_fd))
                        tracking_entry = {
                            "eval": _eq18_eval_counter[0],
                            "step": _step + 1,
                            "al_iter": _al_iter + 1,
                            "max_mode": _max_mode,
                            "n_surf_dofs": n_surf,
                            "norm_fd": norm_fd,
                        }
                        if coil_grads_jax is not None and eq18_ok:
                            norm_jax = float(np.linalg.norm(coil_grads_jax))
                            norm_diff = float(np.linalg.norm(coil_grads_jax - coil_grads_fd))
                            rel_err = norm_diff / (norm_fd + 1e-30)
                            tracking_entry["norm_jax"] = norm_jax
                            tracking_entry["norm_diff"] = norm_diff
                            tracking_entry["rel_err"] = rel_err
                            proc0_print(
                                f"  Hessian tracking #{_eq18_eval_counter[0]}: "
                                f"||JAX||_F={norm_jax:.6e}, ||FD||_F={norm_fd:.6e}, "
                                f"||JAX-FD||_F={norm_diff:.6e}, rel_err={rel_err:.6e}")
                        else:
                            tracking_entry["norm_jax"] = None
                            tracking_entry["norm_diff"] = None
                            tracking_entry["rel_err"] = None
                            proc0_print(
                                f"  Hessian tracking #{_eq18_eval_counter[0]}: "
                                f"JAX failed, ||FD||_F={norm_fd:.6e}")
                        hessian_tracking.append(tracking_entry)

                    if mpi.proc0_world:
                        proc0_print("")
                        proc0_print("  Eq.18 gradient norms (coil constraints w.r.t. surface DOFs):")
                        for k, idx in enumerate(AL1_COIL_CONSTRAINT_INDICES):
                            name = AL1_COIL_CONSTRAINT_NAMES[k]
                            proc0_print(
                                f"    {name:12s}  ||df/dy|| = "
                                f"{np.linalg.norm(coil_grads[k]):.6e}  "
                                f"value = {coil_vals[k]:.6e}  "
                                f"target = {_targets_c[k]:.6e}")
                        proc0_print("")

                    shifted = coil_vals - _targets_c + _lam_c / _rho_c
                    J2 = 0.5 * _rho_c * float(np.sum(shifted ** 2))
                    grad_J2 = np.zeros_like(coil_grads[0])
                    for k in range(n_coil_con):
                        grad_J2 += _rho_c * shifted[k] * coil_grads[k]

                    if not np.isfinite(J2) or not np.all(np.isfinite(grad_J2)):
                        proc0_print(
                            f"  [WARNING] Non-finite J2={J2} or grad_J2; "
                            f"zeroing out.")
                        J2 = 0.0
                        grad_J2 = np.zeros_like(grad_J2)

                    proc0_print(
                        f"  AL1 coil penalty J2 = {J2:.6e}  "
                        f"||grad J2|| = {np.linalg.norm(grad_J2):.6e}")

                    _last_J2_holder[0] = float(J2)

                    return J2, grad_J2

                iter_cb, jac_cb = make_ls_callbacks(
                    is_qss=True, max_mode=max_mode,
                    surf=surf, curves=curves, out_dir=OUT_DIR, bs=bs,
                    history=J1_J2_history,
                    step=step, al_iter=al_iter, al_outer_iter=AL_OUTER_ITER,
                    n_continuation_steps=N_CONTINUATION_STEPS,
                    vtk_exporter=vtk_exporter, is_proc0=mpi.proc0_world,
                    j2_holder=_last_J2_holder,
                )
                least_squares_mpi_solve_qss(
                    prob_equilibrium=al_prob,
                    compute_J2_and_grad=compute_J2_and_grad,
                    mpi=mpi,
                    abs_step=finite_difference_abs_step,
                    rel_step=finite_difference_rel_step,
                    diff_method=diff_method,
                    iteration_callback=iter_cb,
                    jac_callback=jac_cb,
                    max_nfev=max_nfev,
                )

                # Refresh coil metrics at the ACCEPTED iterate. The last
                # compute_J2_and_grad call inside the LS solve may have been
                # at a rejected trial point; the dual update below must use
                # c(x_accepted). surf.x is the accepted optimum here (the
                # solver broadcasts and re-sets it on exit). Skip the AL2
                # re-solve when the last evaluation already was at this x.
                if mpi.proc0_world:
                    _x_accepted = np.asarray(surf.x, dtype=np.float64)
                    if (_last_coil_metrics_x[0] is None
                            or not np.array_equal(_last_coil_metrics_x[0],
                                                  _x_accepted)):
                        proc0_print(
                            "  Last coil metrics were from a non-accepted "
                            "trial point; re-evaluating at the accepted "
                            "iterate for the dual update.")
                        compute_J2_and_grad(compute_grad=False)

            # 2. Dual update for equilibrium constraints
            #    update_dual reads constraint values via con.fn(), which on
            #    each rank invokes VMEC in its own MPI group. Different
            #    groups can return values that drift in the last few digits,
            #    which would let different ranks decide the break check
            #    below differently — and deadlock inside the next AL1 iter's
            #    MPIFiniteDifference. So compute on proc0 only and Bcast.
            #    update_dual re-runs VMEC at the accepted surface. At high mode
            #    counts VMEC can fail to converge (ierr=2); a raw
            #    ObjectiveFailure here would propagate to the top and
            #    MPI_ABORT the whole job, discarding all prior work. Catch it
            #    on proc0, broadcast the outcome so every rank branches
            #    identically (no deadlock), then restore the last converged
            #    surface and stop the continuation gracefully — the final coil
            #    optimisation + metrics still run.
            if mpi.proc0_world:
                try:
                    lam_eq, c_eq = update_dual(
                        lam_eq, rho, EQ_CONSTRAINTS, CONSTRAINT_TOL, LAM_MAX)
                    vmec_failed = False
                except ObjectiveFailure as _e:
                    proc0_print(
                        f"  [WARNING] VMEC failed during the AL1 dual update: {_e}")
                    proc0_print(
                        "  Graceful stop: restoring the last converged surface "
                        "and proceeding to the final coil optimisation.")
                    vmec_failed = True
                    c_eq = np.full(n_eq_con, np.nan)
            else:
                vmec_failed = None
            vmec_failed = mpi.comm_world.bcast(vmec_failed, root=0)
            if vmec_failed:
                surf.local_full_x = np.asarray(
                    last_good_full_x, dtype=np.float64).copy()
                _continuation_aborted = True
                break
            mpi.comm_world.Bcast(lam_eq, root=0)
            if not mpi.proc0_world:
                c_eq = np.empty(n_eq_con, dtype=np.float64)
            else:
                c_eq = np.asarray(c_eq, dtype=np.float64)
            mpi.comm_world.Bcast(c_eq, root=0)

            # 3. Dual update for coil constraints
            #    _last_coil_metric_values is set inside compute_J2_and_grad,
            #    which only fires on proc0 (inside the LS callback). So on
            #    workers it's NaN/stale. Compute the dual update on proc0
            #    and Bcast.
            if ENABLE_COILS:
                if mpi.proc0_world:
                    if np.all(np.isfinite(_last_coil_metric_values)):
                        lam_coil, c_coil = update_dual(
                            lam_coil, rho,
                            _last_coil_metric_values - coil_targets,
                            CONSTRAINT_TOL, LAM_MAX)
                        coil_ok = True
                    else:
                        proc0_print("  [WARNING] Coil metrics NaN — skipping coil dual update.")
                        c_coil = np.full(n_coil_con, np.nan)
                        coil_ok = False
                else:
                    coil_ok = False
                coil_ok = mpi.comm_world.bcast(coil_ok, root=0)
                mpi.comm_world.Bcast(lam_coil, root=0)
                if not mpi.proc0_world:
                    c_coil = (np.empty(n_coil_con, dtype=np.float64)
                              if coil_ok else np.full(n_coil_con, np.nan))
                else:
                    c_coil = np.asarray(c_coil, dtype=np.float64)
                if coil_ok:
                    mpi.comm_world.Bcast(c_coil, root=0)
            else:
                c_coil = np.zeros(n_coil_con)

            # 4. Diagnostics (proc0 prints; obj.fn() is called on every rank
            #    inside, but the result is only consumed by the print).
            al_diagnostics(al_iter, AL_OUTER_ITER, c_eq, EQ_CONSTRAINTS,
                            c_coil, AL1_COIL_CONSTRAINT_NAMES,
                            OBJECTIVES, lam_eq, lam_coil, rho)

            # 5. Grow penalty
            rho = min(rho * RHO_SCALE, RHO_MAX)

            # The equilibrium dual update above evaluated VMEC without failure,
            # so the current surface is converged — record it as the fallback
            # for the graceful-stop path. surf.local_full_x is identical across
            # ranks here (the LS solver broadcasts the accepted iterate on exit).
            last_good_full_x = np.asarray(
                surf.local_full_x, dtype=np.float64).copy()

            # 6. Early exit if all constraints are tight.
            #    Decide on proc0 and broadcast so every rank takes the same
            #    branch — otherwise one rank `break`s while another enters
            #    the next iter's MPIFiniteDifference and deadlocks waiting
            #    for workers that already left.
            if mpi.proc0_world:
                all_violations = np.concatenate([c_eq, c_coil])
                finite_violations = all_violations[np.isfinite(all_violations)]
                should_break = (
                    len(finite_violations) == len(all_violations)
                    and np.linalg.norm(finite_violations) < CONSTRAINT_TOL)
                if should_break:
                    proc0_print(
                        f"  All constraints satisfied to {CONSTRAINT_TOL:.1e} -- "
                        f"exiting AL1 loop early.")
            else:
                should_break = None
            should_break = mpi.comm_world.bcast(should_break, root=0)
            if should_break:
                break

        # Graceful stop: a VMEC failure in the dual update above restored the
        # last converged surface and set the abort flag. Leave the continuation
        # loop and go straight to the final coil optimisation + metrics.
        if _continuation_aborted:
            proc0_print("")
            proc0_print(SEP)
            proc0_print(f"  Continuation halted at step {step+1}/{N_CONTINUATION_STEPS} "
                        f"(max_mode={max_mode}) after a VMEC convergence failure.")
            proc0_print("  Using the last converged surface for the final stage.")
            proc0_print(SEP)
            proc0_print("")
            break

        # Preserve VMEC output files
        vmec.files_to_delete = []
        proc0_print("")
        proc0_print(f"  Completed max_mode = {max_mode}.  "
                    f"Final VMEC iteration: {vmec.iter}")
        proc0_print("")


    # ---------------------------------------------------------------------------
    # Final coil optimisation (for metrics)
    # ---------------------------------------------------------------------------

    if mpi.proc0_world:
        _intermediate = {
            "surface_dofs": np.asarray(surf.x).tolist(),
            "vmec_dofs": np.asarray(vmec.x).tolist(),
            "lam_eq": lam_eq.tolist(),
            "lam_coil": lam_coil.tolist(),
            "rho": float(rho),
        }
        with open(os.path.join(OUT_DIR, "intermediate_state.json"), "w") as _isf:
            json.dump(_intermediate, _isf, indent=2)
        proc0_print(f"  Saved intermediate state to {OUT_DIR}intermediate_state.json")

    # Guard the final run: even after restoring the last converged surface,
    # VMEC could fail here. Every rank runs the same surface, so the failure
    # (if any) is identical across ranks and they all return together — no
    # MPI_ABORT, and intermediate_state.json is already saved.
    _orig_cwd = os.getcwd()
    _final_vmec_ok = True
    try:
        os.chdir(OUT_DIR)
        vmec.run()
    except ObjectiveFailure as _e:
        _final_vmec_ok = False
        proc0_print(f"  [WARNING] VMEC failed at the final surface: {_e}")
        proc0_print("  Intermediate state was saved above; skipping the final "
                    "coil optimisation and metrics. Re-run with fewer "
                    "continuation steps or a coarser VMEC resolution.")
    finally:
        os.chdir(_orig_cwd)
    if not _final_vmec_ok:
        return
    if mpi.proc0_world and vmec.output_file and os.path.exists(vmec.output_file):
        _final_wout = os.path.join(OUT_DIR, "wout_optimized.nc")
        shutil.copy2(vmec.output_file, _final_wout)
        proc0_print(f"  Saved optimized wout to {_final_wout}")
    surf.to_vtk(OUT_DIR + "surf_final_qss",
                extra_data=surface_vtk_point_data(surf, bs))

    proc0_print("")
    proc0_print(SEP)
    proc0_print("  Final coil optimisation (for metrics)")
    proc0_print(SEP)
    proc0_print("")

    LENGTH_TARGET_FINAL = 1.1 * LENGTH_TARGET
    _final_objs = build_coil_objectives(
        surf, bs, base_curves_full, base_coils_full, curves, ncoils,
        LENGTH_TARGET_FINAL,
        hessian_method=HESSIAN_METHOD, flux_threshold=FLUX_THRESHOLD,
        cc_threshold=CC_THRESHOLD, cs_threshold=CS_THRESHOLD,
        curvature_threshold=CURVATURE_THRESHOLD, msc_threshold=MSC_THRESHOLD)
    Jf_final_obj = _final_objs["Jf"]
    Jls_final = _final_objs["Jls"]
    Jccdist_final = _final_objs["Jccdist"]
    Jcsdist_final = _final_objs["Jcsdist"]
    Jlink_final = _final_objs["Jlink"]
    Jmscs_final = _final_objs["Jmscs"]
    Jforce_final = LpCurveForce(base_coils_full, coils, p=2.0,
                                threshold=FORCE_THRESHOLD, downsample=2)

    surf.fix_all()
    if REINIT_COILS_EACH_ALM:
        reinit_coils_to_surface_scaled_circular(
            surf, base_curves_full, base_currents, ncoils, order,
            total_current, COIL_RADIUS_FACTOR, ARC_LENGTH_SAMPLES,
        )
    # Snapshot the coils that the final auglag actually starts from, so we can
    # check in ParaView whether they already overlap/puncture the plasma.
    if mpi.proc0_world:
        curves_to_vtk(curves, OUT_DIR + "coils_init_final_stage")
        surf.to_vtk(OUT_DIR + "surf_init_final_stage",
                    extra_data=surface_vtk_point_data(surf, bs))
    # Sync the gauge fix to the (freshly reinit'd) final-stage coils. surf is
    # fixed (surf.fix_all() above), so the constraint's .x is coil-only. Reset
    # the gauge reference to these coils (force a fresh capture for this stage).
    if f_al2 is not None:
        f_al2.ref = None
        _sync_gauge(_final_objs["c_list"][0])
    _final_solver = auglag_solver(
        surface=surf, maxiter=al2_maxiter,
        max_iter_auglag=al2_max_iter_auglag,
        tau=al2_tau, f=f_al2, constraints=_final_objs["c_list"] + [Jforce_final],
        grad_tol=al2_grad_tol, c_tol=al2_c_tol,
        constraint_names=AL2_CONSTRAINT_NAMES,
    )
    _final_solver.solve()

    # ---- VTK export (proc0 only) ----
    if mpi.proc0_world:
        curves_to_vtk(curves, OUT_DIR + "optimized_coils_qss_final")
    if vtk_exporter is not None:
        frame_idx = vtk_exporter.next_frame()
        vtk_exporter.export_frame(
            surf, curves, frame_idx,
            surface_extra_data=surface_vtk_point_data(surf, bs),
            only_on_proc0=True, is_proc0=mpi.proc0_world,
        )
        vtk_exporter.write_pvd(only_on_proc0=True, is_proc0=mpi.proc0_world)

    # ---- Collect metrics (all ranks compute, proc0 prints/saves) ----
    Bn_over_B_final, _ = compute_bn_modb(surf, bs)
    avg_Bn = float(np.mean(np.abs(Bn_over_B_final)))
    max_Bn = float(np.max(np.abs(Bn_over_B_final)))
    max_curvature = float(max(np.max(c.kappa()) for c in base_curves_full))
    mean_sq_curv = sum(J.J() for J in Jmscs_final) / len(Jmscs_final)
    individual_lengths = [float(J.J()) for J in Jls_final]
    individual_msc = [float(J.J()) for J in Jmscs_final]
    individual_curvatures = [float(np.max(c.kappa())) for c in base_curves_full]

    # Compute final QI metrics (triggers vmec.run() via cached state)
    qi_residuals_final = qi_opt.J()
    qi_total_final = float(np.sum(qi_residuals_final**2))
    aspect_pen_final = float(aspect_opt.J())
    mirror_pen_final = float(mirror_opt.J())
    elong_pen_final = float(elong_opt.J())
    well_pen_final = float(well_opt.J())

    proc0_print("")
    proc0_print(SEP)
    proc0_print("  Final equilibrium metrics")
    proc0_print(SEP)
    proc0_print(f"  Aspect ratio:    {vmec.aspect():.6f}")
    proc0_print(f"  Mean iota:       {vmec.mean_iota():.6f}")
    proc0_print(f"  QI residual:     {qi_total_final:.6e}")
    proc0_print(f"  Mirror ratio pen:{mirror_pen_final:.6e}")
    proc0_print(f"  Aspect ratio pen:{aspect_pen_final:.6e}")
    proc0_print(f"  Elongation pen:  {elong_pen_final:.6e}")
    proc0_print(f"  Vacuum well pen: {well_pen_final:.6e}")
    proc0_print(f"  Magnetic well:   {vmec.vacuum_well():.6e}")
    proc0_print(f"  Surface area:    {surf.area():.6f}")
    proc0_print(f"  Surface volume:  {surf.volume():.6f}")
    proc0_print("")

    proc0_print(SEP)
    proc0_print("  Final coil metrics")
    proc0_print(SEP)
    proc0_print(f"  Squared flux (Jf):           {Jf_final_obj.J():.6e}")
    proc0_print(f"  Total coil length:           {sum(individual_lengths):.6f}")
    proc0_print(f"  Coil-coil distance:          {Jccdist_final.J():.6e}")
    proc0_print(f"  Min CC distance:             {Jccdist_final.shortest_distance():.6f}")
    proc0_print(f"  Coil-surface distance:       {Jcsdist_final.J():.6e}")
    proc0_print(f"  Min CS distance:             {Jcsdist_final.shortest_distance():.6f}")
    proc0_print(f"  Max curvature:               {max_curvature:.6f}")
    proc0_print(f"  Mean squared curvature:      {mean_sq_curv:.6f}")
    proc0_print(f"  Linking number:              {Jlink_final.J():.6e}")
    proc0_print(f"  Coil force penalty:          {Jforce_final.J():.6e}")
    proc0_print(f"  Avg |B.n/|B||:              {avg_Bn:.6e}")
    proc0_print(f"  Max |B.n/|B||:              {max_Bn:.6e}")
    proc0_print("")

    if mpi.proc0_world:
        bs.save(os.path.join(OUT_DIR, "biot_savart_final.json"))

    # ---- Final JAX Hessian analysis ----
    proc0_print(SEP)
    proc0_print("  Final JAX Hessian analysis (SquaredFluxJaxFull)")
    proc0_print(SEP)

    Jf_final_full = SquaredFluxJaxFull(
        surface=surf,
        base_curves=base_curves_full,
        base_currents=[c.current for c in base_coils_full],
        nfp=surf.nfp, stellsym=surf.stellsym,
        target=None, definition="normalized",
    )
    g_final = Jf_final_full.dJ()
    proc0_print(f"  J  = {Jf_final_full.J():.10e}")
    proc0_print(f"  ||grad|| = {np.linalg.norm(g_final):.10e}")

    if Jf_final_full.n_dofs <= HESSIAN_SIZE_CUTOFF:
        t0 = time.time()
        H_final = Jf_final_full.d2J()
        t1 = time.time()
        proc0_print(f"  Hessian computed in {t1-t0:.3f}s")
        proc0_print(f"  ||H||_F = {np.linalg.norm(H_final):.6e}")
        eigvals = np.linalg.eigvalsh(H_final)
        proc0_print(f"  eigenvalues: min={eigvals[0]:.6e}, max={eigvals[-1]:.6e}")
        proc0_print(f"  condition number: {eigvals[-1] / (abs(eigvals[0]) + 1e-30):.6e}")
        if mpi.proc0_world:
            np.save(os.path.join(OUT_DIR, "hessian_squared_flux.npy"), H_final)
            np.save(os.path.join(OUT_DIR, "grad_squared_flux.npy"), g_final)
            proc0_print(f"  Saved Hessian and gradient to {OUT_DIR}")

    # ---- AL1 state summary ----
    proc0_print("")
    proc0_print(f"  AL1 final state:")
    proc0_print(f"    lam_eq   = [{', '.join(f'{v:+.8g}' for v in lam_eq)}]")
    proc0_print(f"    lam_coil = [{', '.join(f'{v:+.8g}' for v in lam_coil)}]")
    proc0_print(f"    rho      = {rho:.8g}")

    # ---- Save all final metrics to JSON (proc0 only) ----
    if mpi.proc0_world:
        final_metrics = {
            "surface": {
                "aspect_ratio": float(vmec.aspect()),
                "mean_iota": float(vmec.mean_iota()),
                "qi_residual_sum_sq": qi_total_final,
                "mirror_ratio_penalty": mirror_pen_final,
                "aspect_ratio_penalty": aspect_pen_final,
                "elongation_penalty": elong_pen_final,
                "vacuum_well_penalty": well_pen_final,
                "vacuum_well": float(vmec.vacuum_well()),
                "area": float(surf.area()),
                "volume": float(surf.volume()),
                "n_dofs_free": int(len(surf.x)),
                "nfp": int(surf.nfp),
            },
            "coils": {
                "squared_flux_Jf": float(Jf_final_obj.J()),
                "avg_Bn_over_B": avg_Bn,
                "max_Bn_over_B": max_Bn,
                "total_length": float(sum(individual_lengths)),
                "individual_lengths": individual_lengths,
                "cc_distance_penalty": float(Jccdist_final.J()),
                "min_cc_distance": float(Jccdist_final.shortest_distance()),
                "cs_distance_penalty": float(Jcsdist_final.J()),
                "min_cs_distance": float(Jcsdist_final.shortest_distance()),
                "max_curvature": max_curvature,
                "individual_max_curvatures": individual_curvatures,
                "mean_squared_curvature_avg": float(mean_sq_curv),
                "individual_msc": individual_msc,
                "linking_number": float(Jlink_final.J()),
                "force_penalty": float(Jforce_final.J()),
                "n_coils_base": ncoils,
                "n_coils_total": len(curves),
                "fourier_order": order,
            },
            "optimization": {
                "hessian_method": HESSIAN_METHOD,
                "n_continuation_steps": N_CONTINUATION_STEPS,
                "al1_outer_iters": AL_OUTER_ITER,
                "rho_final": float(rho),
                "lam_eq_final": lam_eq.tolist(),
                "lam_coil_final": lam_coil.tolist(),
                "flux_threshold": FLUX_THRESHOLD,
                "length_target": LENGTH_TARGET,
                "cc_threshold": CC_THRESHOLD,
                "cs_threshold": CS_THRESHOLD,
                "msc_threshold": MSC_THRESHOLD,
                "curvature_threshold": CURVATURE_THRESHOLD,
                "force_threshold": FORCE_THRESHOLD,
                "aspect_threshold": ASPECT_THRESHOLD,
                "mirror_threshold": MIRROR_THRESHOLD,
                "elongation_threshold": ELONGATION_THRESHOLD,
                "well_target": WELL_TARGET,
            },
            "qi_parameters": {
                "nphi": QI_NPHI,
                "nalpha": QI_NALPHA,
                "nBj": QI_NBJ,
                "mpol": QI_MPOL,
                "ntor": QI_NTOR,
                "sarr": QI_SARR,
                "weights": QI_WEIGHTS.tolist(),
            },
        }
        _metrics_path = os.path.join(OUT_DIR, "final_metrics.json")
        with open(_metrics_path, "w") as _mf:
            json.dump(final_metrics, _mf, indent=2)
        proc0_print(f"  Saved final metrics to {_metrics_path}")

        # ------------------------------------------------------------------
        # Save AL2 multiplier history (Fu et al. Fig. 16 analog)
        # ------------------------------------------------------------------
        _al2_mult_path = os.path.join(OUT_DIR, "al2_multiplier_history.json")
        with open(_al2_mult_path, "w") as _mf:
            json.dump(al2_multiplier_history, _mf, indent=2)
        proc0_print(f"  Saved AL2 multiplier history to {_al2_mult_path}")

        # ------------------------------------------------------------------
        # Save J1 / J2 history (raw + plot)
        # ------------------------------------------------------------------
        if len(J1_J2_history) > 0:
            _hist_path = os.path.join(OUT_DIR, "j1_j2_history.json")
            with open(_hist_path, "w") as _hf:
                json.dump(J1_J2_history, _hf, indent=2)
            proc0_print(f"  Saved J1/J2 history to {_hist_path}")

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                iters = np.arange(1, len(J1_J2_history) + 1)
                J1_vals = np.array([h["J1"] for h in J1_J2_history], dtype=float)
                J2_vals = np.array([h.get("J2") if h.get("J2") is not None else np.nan
                                    for h in J1_J2_history], dtype=float)

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(iters, J1_vals, marker="o", label="J1 (QI equilibrium)",
                        color="tab:blue")
                if ENABLE_COILS and np.any(np.isfinite(J2_vals) & (J2_vals > 0)):
                    ax.plot(iters, J2_vals, marker="s",
                            label="J2 (coil penalty)", color="tab:red")
                ax.set_xlabel("Surface iteration (cumulative)")
                ax.set_ylabel("Objective value")
                ax.set_yscale("log")
                ax.set_title("J1 (QI) / J2 (coil) vs surface iteration")
                ax.grid(True, which="both", alpha=0.3)
                ax.legend()

                _prev_step = J1_J2_history[0].get("step")
                _prev_al = J1_J2_history[0].get("al_iter")
                for i, h in enumerate(J1_J2_history):
                    if h.get("step") != _prev_step or h.get("al_iter") != _prev_al:
                        ax.axvline(i + 1, color="grey", linestyle=":", alpha=0.4)
                        _prev_step = h.get("step")
                        _prev_al = h.get("al_iter")

                fig.tight_layout()
                _plot_path = os.path.join(OUT_DIR, "j1_j2_history.png")
                fig.savefig(_plot_path, dpi=150)
                plt.close(fig)
                proc0_print(f"  Saved J1/J2 plot to {_plot_path}")
            except Exception as _plot_err:
                proc0_print(f"  [WARNING] Could not generate J1/J2 plot: {_plot_err}")



if __name__ == "__main__":
    # If any rank crashes, tear down the whole MPI job instead of letting the
    # other ranks spin on collectives that will never return. Without this,
    # an exception on one rank leaves the rest deadlocked inside Bcast/Bcast
    # and the user sees an endless trickle of stale output.
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc()
        try:
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
        except Exception:
            sys.exit(1)
