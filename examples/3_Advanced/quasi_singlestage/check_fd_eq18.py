#!/usr/bin/env python
"""
check_fd_eq18.py — Does FD of the QSS coil sub-objective produce meaningful,
non-zero gradients w.r.t. the surface DOFs?
============================================================================

This is a clean-room rewrite that emulates ONE AL1 iteration of
qss_script_jax_qi.py with INTEGRATED coil optimisation, and checks — using
ONLY finite differences, no JAX — that the total derivative

        d c_i / d x  =  d/dx  c_i( x, y*(x) )

is finite, non-zero, and well-behaved. Here:

  x        surface (boundary) Fourier DOFs   (the OUTER variable)
  y*(x)    coil DOFs at the AL2 inner optimum (INTEGRATED coil optimisation)
  c_i      the AL1-promoted coil constraints: Jf, Jls, J_kappa.

WHY THIS EXISTS
---------------
The QSS gradient (Fu et al. 2025, Eq. 18) is the TOTAL derivative of a coil
metric through the re-optimised coils. A finite-difference reference for it
MUST re-solve the coils at every perturbed surface — otherwise it measures
only the partial ∂c/∂x at fixed coils, which is a different object. This
script builds that correct FD reference and sanity-checks it three ways:

  1. FD Jacobian (total): central FD of c_i over surface DOFs, RE-SOLVING the
     coils at each perturbation. Report ||J||, per-row norms, zero-row count,
     finite fraction. The headline question "are the matrices non-zero?" is
     answered here.

  2. FD Jacobian (partial): same but WITHOUT re-solving coils. The difference
     (total − partial) is exactly the implicit / IFT contribution; we print
     its size so you can see how much the coil re-optimisation matters.

  3. Taylor test along a random surface direction d: with
       g·d := (directional total derivative from the finest central FD),
     plot
       T1(ε) = | c(x+εd) − c(x) |                  (slope 1: c is smooth)
       T2(ε) = | c(x+εd) − c(x) − ε (g·d) |        (slope 2: g·d is correct)
     A clean slope-2 T2 certifies that y*(x) varies smoothly with x and that
     the FD directional derivative is the genuine total derivative.

SOLVE STRATEGY (why it is no longer erratic)
--------------------------------------------
Each constraint evaluation re-solves the AL2 coil problem WARM-STARTED from
the baseline optimum (y*_0, λ_0, μ_0) and run to a tolerance it can actually
reach within maxiter. The previous flakiness came from grad_tol=1e-10 with
maxiter=30: the solver stopped after 30 steps WITHOUT converging, landing on
a path-dependent, non-stationary y* (erratic). Here grad_tol is achievable
in the iteration budget, so every solve reaches a true stationary point and
y*(x) is a well-defined, reproducible function of x.

USAGE
-----
    python check_fd_eq18.py \
        --vmec-input qi_goodman/zenodo_goodman/configurations/warm_start/input.v20260504_v1 \
        --max-mode 1 --out-dir output/check_fd --fast

Single-process is fine (no MPI needed). --fast uses coarse grids and short ε
sweeps for a quick read; drop it for a precise run.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simsopt.mhd import Vmec
from simsopt.util import proc0_print
from simsopt.geo import create_equally_spaced_curves
from simsopt.field import (
    regularization_circ, coils_via_symmetries, Current, BiotSavart,
)
from simsopt.field.force import LpCurveForce
from simsopt.solve import auglag_solver
from simsopt.objectives import SquaredFluxJaxAnalytic, SquaredFluxJax

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from qss_helpers import (
    build_coil_objectives, fixed_surface,
    reinit_coils_to_surface_scaled_circular, CoilRidge,
    ArclengthGaugeFix, CombinedF,
)


# ---------------------------------------------------------------------------
# Configuration (mirrors qss_script_jax_qi.py)
# ---------------------------------------------------------------------------

CONFIG = dict(
    # Hessian formulation for the squared-flux term of the AL2 Lagrangian
    # Hessian that Eq.(18) inverts: "jax" (analytic autodiff,
    # SquaredFluxJaxAnalytic) or "fd" (finite differences, SquaredFluxJax).
    # MUST match the production --hessian-method to validate what you ship.
    # Overridden by the --hessian-method CLI flag.
    hessian_method        = "jax",
    # B2 ridge weight on the AL2 objective (Tikhonov 0.5*w*||y_coil-ref||^2).
    # Lifts the flat coil-optimum eigenvalues so y*(x) is unique and the solve
    # is stable. 0 = off (default). Overridden by --ridge.
    ridge_weight          = 0.0,
    # Arclength gauge-fix weight on the AL2 objective: w * sum ArclengthVariation.
    # Fixes ONLY the curve-reparametrisation null direction (orthogonal to the
    # flux gradient), so it lifts the flat directions without fighting the
    # objective. 0 = off. Overridden by --arclength-gauge.
    arclength_weight      = 0.0,
    # AL2 inner solver — tolerances are ACHIEVABLE within maxiter so every
    # solve reaches a true stationary point (the fix for the erratic y*).
    al2_tau               = 5,
    al2_mu_initial        = 10,
    # Cap on the AL2 penalty mu. mu-runaway (mu -> ~1e5 at full resolution)
    # inflates the coil-Hessian scale and the -lambda*grad^2 negative curvature,
    # stalling the inner L-BFGS solve (gradL ~ 1e-2, ~40 negative eigenvalues).
    # inf = uncapped (original behaviour). Overridden by --mu-max.
    al2_mu_max            = np.inf,
    # Inner-subproblem minimiser. 'L-BFGS-B' (default) stalls on the indefinite
    # AL2 coil subproblem (~50 negative-curvature directions); 'trust-ncg' /
    # 'trust-krylov' / 'Newton-CG' use a FD Hessian-vector product to descend
    # through negative curvature to the stationary point. Overridden by
    # --inner-solver.
    al2_minimize_method   = "L-BFGS-B",
    # Baseline (cold) solve. NOTE: do NOT crank max_iter_auglag to "converge
    # harder" — each outer iter multiplies mu by tau for still-violated
    # constraints, and Jf (squared flux) is infeasible here (~3e-6 vs 1e-7
    # threshold), so mu RUNS AWAY: 20 outer iters drove mu~1e9, Hessian
    # lambda_max~1e10, and ||grad_y L|| EXPLODED to 1e5 (worse, not better).
    # 8 outer iters sits near the conditioning sweet spot.
    al2_maxiter_base      = 1000,
    al2_max_iter_auglag_b = 8,
    al2_grad_tol_base     = 1e-8,
    al2_c_tol_base        = 1e-8,
    # Per-perturbation (warm) solve: cheaper, but still converges.
    al2_maxiter_warm      = 500,
    al2_max_iter_auglag_w = 8,
    al2_grad_tol_warm     = 1e-9,
    al2_c_tol_warm        = 1e-9,
    # Coil geometry / constraints
    LENGTH_TARGET         = 4.5,
    # FLUX_THRESHOLD relaxed 1e-7 -> 5e-6: the squared flux floor reachable by
    # these coils at max_mode=1 is ~3e-6, so 1e-7 was INFEASIBLE and drove
    # mu_Jf runaway (Hessian conditioning collapse, Jf Eq.18 row ill-posed).
    # 5e-6 sits just above the achievable floor so Jf becomes satisfiable.
    FLUX_THRESHOLD        = 1e-16,
    CC_THRESHOLD          = 0.10,
    CS_THRESHOLD          = 0.10,
    MSC_THRESHOLD         = 10,
    CURVATURE_THRESHOLD   = 10,
    FORCE_THRESHOLD       = 0.1,
    COIL_RADIUS_FACTOR    = 3.6,
    COIL_QUADPOINTS       = 256,
    ARC_LENGTH_SAMPLES    = 2048,
    a_reg                 = 0.15,
    ncoils                = 4,
    order                 = 6,
    total_current         = 3e5,
    VMEC_MPOL             = 10,
    VMEC_NTOR             = 10,
    nphi_surf             = 128,
    ntheta_surf           = 128,
)

FAST_OVERRIDES = dict(
    al2_maxiter_base      = 120,
    al2_max_iter_auglag_b = 6,
    al2_grad_tol_base     = 1e-7,
    al2_c_tol_base        = 1e-7,
    al2_maxiter_warm      = 60,
    al2_max_iter_auglag_w = 4,
    al2_grad_tol_warm     = 1e-7,
    al2_c_tol_warm        = 1e-7,
    COIL_QUADPOINTS       = 64,
    ARC_LENGTH_SAMPLES    = 512,
    nphi_surf             = 32,
    ntheta_surf           = 32,
)

# ε grids: keep ε >= 1e-4 (below that the inner solve precision floor swamps
# the signal) and ε <= ~5e-2 (above that higher-order Taylor terms dominate).
EPS_GRID_TAYLOR      = np.logspace(-4, -1.3, 10)
EPS_GRID_SWEEP       = np.logspace(-4, -1.3, 7)
FAST_EPS_GRID_TAYLOR = np.logspace(-4, -1.3, 6)
FAST_EPS_GRID_SWEEP  = np.logspace(-4, -1.3, 4)

# AL2 constraint ordering — MUST match qss_helpers.build_coil_objectives
# (c_list) + Jforce appended, exactly as qss_script_jax_qi.py does.
AL2_CONSTRAINT_NAMES        = ["Jf", "Jccdist", "Jcsdist", "Jmscs", "Jls",
                               "J_kappa", "Jlink", "Jforce"]
AL1_COIL_CONSTRAINT_INDICES = [0, 4, 5]            # Jf, Jls, J_kappa
AL1_COIL_CONSTRAINT_NAMES   = ["Jf", "Jls", "J_kappa"]
N_AL1 = len(AL1_COIL_CONSTRAINT_INDICES)


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------

def build_state(vmec_input, max_mode):
    """Build surf, coils, BiotSavart and the AL2 constraint list, with the
    surface free-DOF mask set to the continuation step `max_mode` exactly as
    qss_script_jax_qi.py does (mmin=0..max_mode, n in [-max_mode, max_mode],
    rc(0,0) fixed)."""
    proc0_print(f"  Loading VMEC: {vmec_input}")
    vmec = Vmec(vmec_input, verbose=False,
                surf_type='JaxSurfaceRZFourier', range_surface='full torus',
                nphi=CONFIG["nphi_surf"], ntheta=CONFIG["ntheta_surf"])
    surf = vmec.boundary
    vmec.indata.mpol = CONFIG["VMEC_MPOL"]
    vmec.indata.ntor = CONFIG["VMEC_NTOR"]

    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode,
                     fixed=False)
    surf.fix("rc(0,0)")
    proc0_print(f"  Surface free DOFs at max_mode={max_mode}: {surf.x.size}")

    ncoils, order = CONFIG["ncoils"], CONFIG["order"]
    R0 = surf.get_rc(0, 0)
    R1 = surf.get_rc(1, 0) * 3
    base_curves = create_equally_spaced_curves(
        ncoils, surf.nfp, stellsym=surf.stellsym, R0=R0, R1=R1, order=order,
        numquadpoints=CONFIG["COIL_QUADPOINTS"])
    base_currents = [Current(CONFIG["total_current"] / ncoils * 1e-5) * 1e5
                     for _ in range(ncoils - 1)]
    total_cur = Current(CONFIG["total_current"]); total_cur.fix_all()
    base_currents += [total_cur - sum(base_currents)]
    regs = [regularization_circ(CONFIG["a_reg"]) for _ in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, surf.nfp,
                                 surf.stellsym, regularizations=regs)
    base_coils_full = coils[:ncoils]
    curves = [c.curve for c in coils]
    base_curves_full = curves[:ncoils]

    reinit_coils_to_surface_scaled_circular(
        surf, base_curves_full, base_currents, ncoils, order,
        CONFIG["total_current"], CONFIG["COIL_RADIUS_FACTOR"],
        CONFIG["ARC_LENGTH_SAMPLES"])

    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))

    hmethod = CONFIG["hessian_method"]
    coil_objs = build_coil_objectives(
        surf, bs, base_curves_full, base_coils_full, curves, ncoils,
        CONFIG["LENGTH_TARGET"], hessian_method=hmethod,
        flux_threshold=CONFIG["FLUX_THRESHOLD"],
        cc_threshold=CONFIG["CC_THRESHOLD"], cs_threshold=CONFIG["CS_THRESHOLD"],
        curvature_threshold=CONFIG["CURVATURE_THRESHOLD"],
        msc_threshold=CONFIG["MSC_THRESHOLD"])
    Jforce = LpCurveForce(base_coils_full, coils, p=2.0,
                          threshold=CONFIG["FORCE_THRESHOLD"], downsample=2)
    al2_c_list = coil_objs["c_list"] + [Jforce]
    assert len(al2_c_list) == len(AL2_CONSTRAINT_NAMES)

    # RAW (unthresholded) squared flux — the smooth coil-field-match metric the
    # paper differentiates. threshold=0 => max(0, flux-0) = flux, so this is the
    # genuine squared flux, NOT the clamped AL1 penalty (which sits at its kink
    # when satisfied and is non-smooth / signal-free for a Taylor test). Built
    # with the SAME Hessian formulation as the AL2 Jf term so the raw-flux
    # adjoint exercises exactly the production (--hessian-method) path.
    if hmethod == "jax":
        raw_flux = SquaredFluxJaxAnalytic(
            surf, bs, base_curves=base_curves_full,
            base_currents=[c.current for c in base_coils_full],
            nfp=surf.nfp, stellsym=surf.stellsym,
            definition="normalized", threshold=0.0, fixed_surface=False)
    else:
        raw_flux = SquaredFluxJax(
            surf, bs, definition="normalized", threshold=0.0,
            fixed_surface=False)

    # B2 AL2-objective regularisers. n_coil = total free coil DOFs (curve shapes
    # + free currents), surface-mask-independent, FIRST in the [coil, surf]
    # ordering. f_al2 = None (=> identical to f=None) when both weights are 0.
    n_coil = (sum(int(np.asarray(c.x).size) for c in base_curves_full)
              + sum(int(np.asarray(cur.x).size)
                    for cur in base_currents[:ncoils - 1]))
    _f_terms = []
    if CONFIG["ridge_weight"] > 0:
        _f_terms.append(CoilRidge(n_coil, CONFIG["ridge_weight"]))
    if CONFIG["arclength_weight"] > 0:
        # Reference = BiotSavart.dof_names: the exact coil-vector order
        # (currents, then curves), surface-independent. Name-matching makes
        # the per-curve DOF mapping layout-proof.
        _f_terms.append(ArclengthGaugeFix(
            base_curves_full, list(bs.dof_names), CONFIG["arclength_weight"]))
    if not _f_terms:
        f_al2 = None
    elif len(_f_terms) == 1:
        f_al2 = _f_terms[0]
    else:
        f_al2 = CombinedF(_f_terms)

    return dict(vmec=vmec, surf=surf, bs=bs, al2_c_list=al2_c_list,
                base_curves_full=base_curves_full, base_coils_full=base_coils_full,
                base_currents=base_currents, raw_flux=raw_flux,
                f_al2=f_al2, n_coil=n_coil,
                ncoils=ncoils, order=order, max_mode=max_mode)


# ---------------------------------------------------------------------------
# Coil solve + evaluation (the "integrated coil optimisation")
# ---------------------------------------------------------------------------

_solve_count = [0]
_solve_time  = [0.0]


def _run_solver(state, solver):
    t0 = time.time()
    with fixed_surface(state["surf"], state["max_mode"]):
        # Sync the B2 ridge to the coil-DOF vector INSIDE fixed_surface (where
        # al2_c_list[0].x is coil-only), and capture its fixed reference once.
        fr = state.get("f_al2")
        if fr is not None:
            cx = np.asarray(state["al2_c_list"][0].x, np.float64).copy()
            fr.ensure_ref(cx)
            fr.x = cx
        x_coils, _fnc, lag_mul, mu = solver.solve()
    state["bs"].set_points(state["surf"].gamma().reshape((-1, 3)))
    _solve_count[0] += 1
    _solve_time[0]  += time.time() - t0
    return x_coils, lag_mul, mu


def solve_baseline(state):
    """Cold AL2 solve from the circular-axis init — this is the 'one AL1
    iteration' coil optimisation. Returns (x_coils, lag_mul, mu)."""
    state["bs"].set_points(state["surf"].gamma().reshape((-1, 3)))
    reinit_coils_to_surface_scaled_circular(
        state["surf"], state["base_curves_full"], state["base_currents"],
        state["ncoils"], state["order"], CONFIG["total_current"],
        CONFIG["COIL_RADIUS_FACTOR"], CONFIG["ARC_LENGTH_SAMPLES"])
    state["bs"].set_points(state["surf"].gamma().reshape((-1, 3)))
    solver = auglag_solver(
        surface=state["surf"], maxiter=CONFIG["al2_maxiter_base"],
        max_iter_auglag=CONFIG["al2_max_iter_auglag_b"], tau=CONFIG["al2_tau"],
        f=state.get("f_al2"), constraints=state["al2_c_list"],
        grad_tol=CONFIG["al2_grad_tol_base"], c_tol=CONFIG["al2_c_tol_base"],
        max_mode=state["max_mode"], constraint_names=AL2_CONSTRAINT_NAMES,
        mu_initial=CONFIG["al2_mu_initial"], lag_mul_initial=None,
        mu_max=CONFIG["al2_mu_max"],
        minimize_method=CONFIG["al2_minimize_method"])
    return _run_solver(state, solver)


def snapshot_coils(state):
    """Capture base curve shapes + free currents so warm re-solves all start
    from the identical baseline (no path coupling between perturbations)."""
    return dict(
        curve=[np.asarray(c.x, np.float64).copy()
               for c in state["base_curves_full"]],
        current=[np.asarray(c.x, np.float64).copy()
                 for c in state["base_currents"][:state["ncoils"] - 1]])


def restore_coils(state, snap):
    for c, d in zip(state["base_curves_full"], snap["curve"]):
        c.x = d
    for cur, d in zip(state["base_currents"][:state["ncoils"] - 1],
                      snap["current"]):
        cur.x = d
    state["bs"].set_points(state["surf"].gamma().reshape((-1, 3)))


def solve_warm(state, baseline):
    """Warm AL2 re-solve from the baseline (coils, λ, μ). Independent of any
    other call. Returns (x_coils, lag_mul, mu).

    The .copy() on mu0/lag0 is defensive: augmented_lagrangian_method used to
    alias the caller's mu array and mutate it in place (mu_k *= tau), which
    silently inflated the shared baseline mu across solves — the source of
    the irreproducible 'floor' in earlier runs. Fixed in the library too, but
    the verifier must not depend on the library version."""
    snap, lag0, mu0 = baseline
    restore_coils(state, snap)
    solver = auglag_solver(
        surface=state["surf"], maxiter=CONFIG["al2_maxiter_warm"],
        max_iter_auglag=CONFIG["al2_max_iter_auglag_w"], tau=CONFIG["al2_tau"],
        f=state.get("f_al2"), constraints=state["al2_c_list"],
        grad_tol=CONFIG["al2_grad_tol_warm"], c_tol=CONFIG["al2_c_tol_warm"],
        max_mode=state["max_mode"], constraint_names=AL2_CONSTRAINT_NAMES,
        mu_initial=np.asarray(mu0, np.float64).copy(),
        lag_mul_initial=np.asarray(lag0, np.float64).copy(),
        mu_max=CONFIG["al2_mu_max"],
        minimize_method=CONFIG["al2_minimize_method"])
    return _run_solver(state, solver)


def eval_c(state):
    """AL1-promoted constraint values at the current (surf, coils) state."""
    state["bs"].set_points(state["surf"].gamma().reshape((-1, 3)))
    vals = np.array([state["al2_c_list"][i].J()
                     for i in AL1_COIL_CONSTRAINT_INDICES], dtype=np.float64)
    return vals


def _set_surf_dof(surf, x0, j, eps):
    xp = np.asarray(x0, np.float64).copy()
    xp[j] += eps
    surf.x = xp


# ---------------------------------------------------------------------------
# Finite differences
# ---------------------------------------------------------------------------

def fd_total_column(state, j, eps, baseline):
    """Central FD of c_i w.r.t. surf.x[j], RE-SOLVING coils (warm) at each
    perturbation. NaN-trapped: a diverged solve yields NaN for THIS column
    only and the coils are scrubbed back to baseline so nothing leaks."""
    surf = state["surf"]
    x0 = np.asarray(surf.x, np.float64).copy()
    snap = baseline[0]

    _set_surf_dof(surf, x0, j, +eps)
    solve_warm(state, baseline)
    v_fwd = eval_c(state)

    _set_surf_dof(surf, x0, j, -eps)
    solve_warm(state, baseline)
    v_bwd = eval_c(state)

    surf.x = x0
    restore_coils(state, snap)

    if not (np.all(np.isfinite(v_fwd)) and np.all(np.isfinite(v_bwd))):
        return np.full(N_AL1, np.nan)
    return (v_fwd - v_bwd) / (2.0 * eps)


def fd_partial_column(state, j, eps):
    """Central FD of c_i w.r.t. surf.x[j] at FIXED coils (no re-solve)."""
    surf = state["surf"]
    x0 = np.asarray(surf.x, np.float64).copy()

    _set_surf_dof(surf, x0, j, +eps)
    v_fwd = eval_c(state)
    _set_surf_dof(surf, x0, j, -eps)
    v_bwd = eval_c(state)

    surf.x = x0
    state["bs"].set_points(surf.gamma().reshape((-1, 3)))
    if not (np.all(np.isfinite(v_fwd)) and np.all(np.isfinite(v_bwd))):
        return np.full(N_AL1, np.nan)
    return (v_fwd - v_bwd) / (2.0 * eps)


def fd_jacobian(state, baseline, eps, cols):
    """Full (or sub-sampled) FD total + partial Jacobians over `cols`.
    Returns (J_total, J_partial), each shape (N_AL1, len(cols))."""
    J_tot = np.zeros((N_AL1, len(cols)))
    J_par = np.zeros((N_AL1, len(cols)))
    for k, j in enumerate(cols):
        J_tot[:, k] = fd_total_column(state, int(j), eps, baseline)
        J_par[:, k] = fd_partial_column(state, int(j), eps)
        proc0_print(f"    col {j:3d}: "
                    f"tot={J_tot[:, k]}  par={J_par[:, k]}")
    return J_tot, J_par


# ---------------------------------------------------------------------------
# Coil DOF vector (for the sensitivity test)
# ---------------------------------------------------------------------------

def coil_dof_vector(state):
    """Flatten the AL2 coil DOFs into a single vector, in a fixed order:
    [base curve shapes ..., free base currents ...]. Returns (vec, n_curve)
    where n_curve is the length of the curve-shape block (the rest is the
    current block) so callers can split the displacement by kind."""
    curve_parts = [np.asarray(c.x, np.float64).ravel()
                   for c in state["base_curves_full"]]
    curr_parts = [np.asarray(cur.x, np.float64).ravel()
                  for cur in state["base_currents"][:state["ncoils"] - 1]]
    n_curve = int(sum(p.size for p in curve_parts))
    vec = np.concatenate(curve_parts + curr_parts) if (curve_parts or curr_parts) \
        else np.zeros(0)
    return vec, n_curve


def coil_geometry(state):
    """Physical coil geometry: the 3-D points gamma() of each base curve.

    Returns (G, per_coil) where
      G        : (ncoils, nquad, 3) array of curve points, and
      per_coil : list of the per-coil (nquad, 3) arrays (views into G).

    This is the gauge-relevant quantity the sensitivity test should use:
    coil Fourier DOFs are NON-UNIQUE at a flat AL2 optimum (gauge / near-null
    directions), so ||Δ(DOF vector)|| is dominated by gauge drift. The curve
    points in space are physical. (Caveat: gamma() at FIXED quadrature points
    is still sensitive to curve REPARAMETRISATION — a point sliding along the
    same geometric curve — so we also offer a nearest-point distance below.)"""
    G = np.stack([np.asarray(c.gamma(), np.float64)
                  for c in state["base_curves_full"]], axis=0)
    return G, [G[i] for i in range(G.shape[0])]


def coil_geo_distance(G1, G2, robust=False):
    """Geometric distance between two coil-point sets G1, G2 of shape
    (ncoils, nquad, 3). Returns (total, per_coil).

    robust=False : pointwise RMS  sqrt(mean_k ||g1_k - g2_k||^2)  per coil,
                   then root-sum-square over coils. Fast, but reparam-sensitive.
    robust=True  : symmetric nearest-point RMS (discrete Hausdorff-lite),
                   invariant to reparametrisation of each curve."""
    ncoils = G1.shape[0]
    per = np.zeros(ncoils)
    for i in range(ncoils):
        a, b = G1[i], G2[i]
        if not robust:
            per[i] = np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))
        else:
            # nearest neighbour each way, symmetric RMS
            d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)  # (nq, nq)
            ab = np.sqrt(np.mean(np.min(d2, axis=1)))
            ba = np.sqrt(np.mean(np.min(d2, axis=0)))
            per[i] = 0.5 * (ab + ba)
    total = float(np.linalg.norm(per))
    return total, per


def eval_c_fixed(state, x_new, snap):
    """c_i( x_new, y_base ) at FIXED coils (no AL2 solve). Restores baseline
    coils, sets the surface, evaluates. Leaves surf at x_new (caller resets)."""
    restore_coils(state, snap)
    state["surf"].x = np.asarray(x_new, np.float64).copy()
    return eval_c(state)


# ---------------------------------------------------------------------------
# TEST A — partial Taylor test at FIXED coils (no re-optimisation)
# ---------------------------------------------------------------------------

def partial_taylor_test(state, snap, eps_grid, seed=0):
    """Holding coils at the baseline y_base, treat c(x) = c_i(x, y_base) as a
    plain smooth function of the surface DOFs and run a Taylor test.

    There is NO inner solve here, so there is NO solver noise floor: this
    validates the FD plumbing + surface parametrisation + constraint
    evaluation in isolation. Expect T1 slope 1 and T2 slope 2 for any
    constraint with a genuine direct surface dependence (in practice Jf;
    Jls/J_kappa have ZERO partial because coil length/curvature do not depend
    on the surface, so their T1 sits at ~0)."""
    rng = np.random.default_rng(seed)
    x0 = np.asarray(state["surf"].x, np.float64).copy()
    d = rng.standard_normal(x0.size); d /= np.linalg.norm(d)

    c0 = eval_c_fixed(state, x0, snap)

    eps_ref = float(np.min(eps_grid))
    v_p = eval_c_fixed(state, x0 + eps_ref * d, snap)
    v_m = eval_c_fixed(state, x0 - eps_ref * d, snap)
    g_dot_d = (v_p - v_m) / (2.0 * eps_ref)
    proc0_print(f"  [A] partial g·d (fixed coils, ε={eps_ref:.1e}) = {g_dot_d}")

    T1 = {n: [] for n in AL1_COIL_CONSTRAINT_NAMES}
    T2 = {n: [] for n in AL1_COIL_CONSTRAINT_NAMES}
    for eps in eps_grid:
        v = eval_c_fixed(state, x0 + eps * d, snap)
        delta = v - c0
        for k, n in enumerate(AL1_COIL_CONSTRAINT_NAMES):
            T1[n].append(abs(delta[k]))
            T2[n].append(abs(delta[k] - eps * g_dot_d[k]))

    state["surf"].x = x0
    restore_coils(state, snap)
    return dict(eps=np.asarray(eps_grid), d=d, g_dot_d=g_dot_d, c0=c0,
                T1={k: np.asarray(v) for k, v in T1.items()},
                T2={k: np.asarray(v) for k, v in T2.items()})


# ---------------------------------------------------------------------------
# TEST B — coil sensitivity: how far do the AL2 coils move when x moves by ε?
# ---------------------------------------------------------------------------

def coil_sensitivity_test(state, baseline, eps_grid, seed=0, robust=False):
    """Answer: "if I change the surface by ε along d, how do the auglag coils
    move PHYSICALLY?" For each ε we warm-solve at x0±εd (both from the SAME
    baseline, so the comparison isolates the surface change) and measure the
    coil-GEOMETRY displacement from the curve points gamma() — NOT the DOF
    vector, which is gauge-contaminated at a flat AL2 optimum:

        Δγ(ε) = geo_distance( γ*(x0+εd), γ*(x0-εd) ) / 2     (central, metres)
        sens(ε) = Δγ(ε) / ε                                  (m per unit x)

    A differentiable implicit map gives Δγ ∝ ε (slope 1) and sens -> a plateau
    (= physical coil motion per unit surface move). The raw DOF norm ||Δy|| is
    ALSO recorded as a secondary contrast: if ||Δy|| is erratic while ||Δγ||
    is clean, that proves the DOF noise is pure gauge drift.

    A same-x reproducibility floor (solve twice at x0) is measured in BOTH
    metrics; physical signal below the gamma floor is solver noise."""
    rng = np.random.default_rng(seed + 1)
    x0 = np.asarray(state["surf"].x, np.float64).copy()
    d = rng.standard_normal(x0.size); d /= np.linalg.norm(d)
    snap = baseline[0]

    # Reproducibility floor (same x0, two independent warm solves).
    state["surf"].x = x0
    solve_warm(state, baseline)
    G_a, _ = coil_geometry(state); y_a, n_curve = coil_dof_vector(state)
    state["surf"].x = x0
    solve_warm(state, baseline)
    G_b, _ = coil_geometry(state); y_b, _ = coil_dof_vector(state)
    floor_geo, floor_geo_per = coil_geo_distance(G_a, G_b, robust=robust)
    floor_dof = float(np.linalg.norm(y_a - y_b))
    ncoils = G_a.shape[0]
    proc0_print(f"  [B] repro floor:  ||Δγ||={floor_geo:.3e} m   "
                f"||Δy_dof||={floor_dof:.3e}   (ncoils={ncoils}, "
                f"nquad={G_a.shape[1]}, robust={robust})")

    dgeo, sens, dgeo_per, ddof, fin = [], [], [], [], []
    for eps in eps_grid:
        state["surf"].x = x0 + eps * d
        solve_warm(state, baseline)
        G_p, _ = coil_geometry(state); y_p, _ = coil_dof_vector(state)
        state["surf"].x = x0 - eps * d
        solve_warm(state, baseline)
        G_m, _ = coil_geometry(state); y_m, _ = coil_dof_vector(state)

        ok = (np.all(np.isfinite(G_p)) and np.all(np.isfinite(G_m))
              and np.all(np.isfinite(y_p)) and np.all(np.isfinite(y_m)))
        fin.append(bool(ok))
        if not ok:
            dgeo.append(np.nan); sens.append(np.nan)
            dgeo_per.append(np.full(ncoils, np.nan)); ddof.append(np.nan)
            continue
        # central geometric displacement: distance between the two solved
        # geometries, halved (matches a central FD magnitude).
        g_tot, g_per = coil_geo_distance(G_p, G_m, robust=robust)
        g_tot *= 0.5; g_per = g_per * 0.5
        dof = float(np.linalg.norm(y_p - y_m) / 2.0)
        dgeo.append(g_tot); sens.append(g_tot / eps)
        dgeo_per.append(g_per); ddof.append(dof)
        proc0_print(f"    ε={eps:.2e}  ||Δγ||={g_tot:.3e} m  "
                    f"sens={g_tot/eps:.3e}  ||Δy_dof||={dof:.3e}")

    state["surf"].x = x0
    restore_coils(state, snap)
    return dict(eps=np.asarray(eps_grid), d=d,
                floor_geo=floor_geo, floor_dof=floor_dof,
                floor_geo_per=floor_geo_per,
                ncoils=int(ncoils), nquad=int(G_a.shape[1]),
                n_coil_dofs=int(y_a.size), robust=bool(robust),
                dgeo=np.asarray(dgeo), sens=np.asarray(sens),
                dgeo_per=np.asarray(dgeo_per), ddof=np.asarray(ddof),
                finite=np.asarray(fin, dtype=bool))


# ---------------------------------------------------------------------------
# TEST C — Hessian spectrum: is the AL2 degeneracy gauge or physical?
# ---------------------------------------------------------------------------

def _classify_null_vectors(evecs_null, ref_dof_names, order):
    """Classify each near-null coil-Hessian eigenvector by DOF composition,
    to decide which regulariser the residual degeneracy needs.

    For each near-null unit eigenvector v (coil-DOF space, ordered by
    ref_dof_names) report the fraction of its norm in:
      - CURRENTS    (names containing 'Current')         -> current null-space
      - HIGH-order Fourier curve modes (mode k > order/2) -> flux-insensitive
                                                             coil wiggles
      - LOW-order   curve modes (the rest)               -> fundamental / gauge

    A high CURRENT fraction => add a current ridge; high HIGH-order fraction
    => a smoothness (high-mode) ridge that is ~orthogonal to the flux; LOW-order
    dominance => reparametrisation/fundamental flatness (arclength gauge)."""
    import re
    names = list(ref_dof_names)
    curr_mask = np.array(["Current" in n for n in names])
    hi_mask = np.zeros(len(names), dtype=bool)
    for i, n in enumerate(names):
        if "Current" in n:
            continue
        m = re.search(r"\((\d+)\)", n)        # e.g. CurveXYZFourier1:xc(3)
        if m and int(m.group(1)) > max(1, order // 2):
            hi_mask[i] = True
    lo_mask = (~curr_mask) & (~hi_mask)
    out = []
    for j in range(evecs_null.shape[1]):
        v = evecs_null[:, j]
        nv = np.linalg.norm(v) or 1.0
        out.append(dict(
            current_frac=float(np.linalg.norm(v[curr_mask]) / nv),
            highorder_frac=float(np.linalg.norm(v[hi_mask]) / nv),
            loworder_frac=float(np.linalg.norm(v[lo_mask]) / nv)))
    return out


def hessian_spectrum_test(state, x_coils, lag_mul, mu, svtol=1e-6, extra=None,
                          classify_ref=None):
    """Eigendecompose the coil-coil block of the AL2 Lagrangian Hessian at
    the baseline optimum — the exact operator Eq.(18) inverts — and measure
    how much of each AL1 constraint's coil-gradient lives in the near-null
    eigenspace.

    Motivation (from the gamma_fixed run): the solver is fully deterministic
    (repro floor = 0) yet y*(x) jumps by ~1-5 cm under arbitrarily small
    surface changes. Both endpoints satisfy ||grad L|| < 1e-9, so the Hessian
    must have near-null directions with lambda ~ grad_tol / jump ~ 1e-7.
    Eq.(18) is still well-posed IF the constraint gradients dc/dy are
    orthogonal to those directions (i.e. the flat valley is GAUGE — curve
    reparametrisation / current null-combos that don't change physics). This
    test measures that overlap directly:

        null_frac_i = || P_null (dc_i/dy) || / || dc_i/dy ||

    where P_null projects onto eigenvectors with |lambda| < svtol*lambda_max
    (the same truncation the qss.py SVD preconditioner applies).

    Verdict per constraint:
        GAUGE_SAFE   null_frac < 1e-2  : Eq.(18)+svtol-truncation well-posed
        MARGINAL     1e-2..0.3         : partial contamination
        ILL_POSED    > 0.3             : the flat directions are physical
    """
    from simsopt.solve.augmented_lagrangian_fast import (
        hessian_augmented_lagrangian, grad_f, set_combined_dof_state)

    surf = state["surf"]
    mm = state["max_mode"]
    x_flat = np.asarray(surf.x, np.float64).ravel()
    y_flat = np.asarray(x_coils, np.float64).ravel()
    n_coil = y_flat.size
    c_list = state["al2_c_list"]

    proc0_print(f"  [C] building AL2 Lagrangian Hessian "
                f"({n_coil}x{n_coil} coil-coil block) ...")
    t0 = time.time()
    set_combined_dof_state(surf, c_list, x_flat, y_flat)
    H_terms = np.asarray(hessian_augmented_lagrangian(
        surface_dofs=x_flat, coil_dofs=y_flat, f=None,
        equality_constraints=c_list, lag_mul=lag_mul, mu=mu))
    if not np.all(np.isfinite(H_terms)):
        n_bad = int(np.sum(~np.isfinite(H_terms)))
        proc0_print(f"  [WARNING] Hessian has {n_bad} non-finite entries; "
                    f"zeroing them.")
        H_terms = np.where(np.isfinite(H_terms), H_terms, 0.0)
    H = np.sum(H_terms, axis=0)[:n_coil, :n_coil]
    H = 0.5 * (H + H.T)
    proc0_print(f"      Hessian built in {time.time()-t0:.1f}s   "
                f"||H||_F = {np.linalg.norm(H):.3e}")

    evals, evecs = np.linalg.eigh(H)          # ascending
    lam_max = float(np.max(np.abs(evals)))
    cutoff = svtol * lam_max
    near_null = np.abs(evals) < cutoff
    n_null = int(near_null.sum())
    n_neg = int(np.sum(evals < -cutoff))
    proc0_print(f"      eigenvalues: min={evals[0]:.3e}  max={evals[-1]:.3e}")
    proc0_print(f"      |lambda| < svtol*max ({cutoff:.3e}): {n_null}/{n_coil}"
                f"   significantly negative: {n_neg}")

    # Constraint gradients w.r.t. coil DOFs (same fix-mask dance as
    # df_dx_all_terms in qss.py).
    grads = {}
    surf.fix_all()
    try:
        for name, idx in zip(AL1_COIL_CONSTRAINT_NAMES,
                             AL1_COIL_CONSTRAINT_INDICES):
            g = np.asarray(grad_f(c_list[idx], x_flat, y_flat, wrt_x=False),
                           dtype=np.float64)
            grads[name] = g
        # Extra metrics (e.g. the raw squared flux) — same coil-gradient and
        # overlap analysis, against the SAME AL2 Hessian eigenbasis.
        if extra:
            for name, obj in extra.items():
                grads[name] = np.asarray(
                    grad_f(obj, x_flat, y_flat, wrt_x=False), dtype=np.float64)
    finally:
        surf.fixed_range(mmin=0, mmax=mm, nmin=-mm, nmax=mm, fixed=False)
        surf.fix("rc(0,0)")

    # Stationarity sanity: ||grad_y L|| should be ~grad_tol at the optimum.
    cvals = np.array([c.J() for c in c_list], dtype=np.float64)
    gradL = np.zeros(n_coil)
    surf.fix_all()
    try:
        for i, c in enumerate(c_list):
            gci = np.asarray(grad_f(c, x_flat, y_flat, wrt_x=False),
                             dtype=np.float64)
            gradL += (-lag_mul[i] + np.atleast_1d(mu)[min(i, np.atleast_1d(mu).size-1)]
                      * cvals[i]) * gci
    finally:
        surf.fixed_range(mmin=0, mmax=mm, nmin=-mm, nmax=mm, fixed=False)
        surf.fix("rc(0,0)")
    proc0_print(f"      stationarity ||grad_y L|| = {np.linalg.norm(gradL):.3e}")

    # Overlaps: coefficients of each unit gradient in the eigenbasis.
    overlaps = {}     # |v_k . g_hat| per eigenvector
    null_frac = {}
    verdict = {}
    for name, g in grads.items():
        gn = float(np.linalg.norm(g))
        if gn < 1e-300:
            overlaps[name] = np.zeros(n_coil)
            null_frac[name] = float("nan")
            verdict[name] = "ZERO_GRADIENT"
            continue
        coef = np.abs(evecs.T @ (g / gn))      # (n_coil,)
        overlaps[name] = coef
        nf = float(np.linalg.norm(coef[near_null]))
        null_frac[name] = nf
        verdict[name] = ("GAUGE_SAFE" if nf < 1e-2
                         else "MARGINAL" if nf < 0.3 else "ILL_POSED")
        proc0_print(f"      {name:9s}: ||dc/dy||={gn:.3e}   "
                    f"null_frac={nf:.3e}   -> {verdict[name]}")

    # Classify the near-null eigenvectors (which regulariser the residual
    # degeneracy needs) when a reference DOF-name list is supplied.
    null_classes = None
    if classify_ref is not None and n_null > 0:
        order = state.get("order", 4)
        null_classes = _classify_null_vectors(
            evecs[:, near_null], classify_ref, order)
        proc0_print(f"      near-null eigenvector composition "
                    f"(n_null={n_null}):")
        cur = np.mean([c["current_frac"] for c in null_classes])
        hi = np.mean([c["highorder_frac"] for c in null_classes])
        lo = np.mean([c["loworder_frac"] for c in null_classes])
        proc0_print(f"        mean fractions  current={cur:.2f}  "
                    f"high-order-Fourier={hi:.2f}  low-order-curve={lo:.2f}")
        for j, c in enumerate(null_classes):
            proc0_print(f"        v{j}: curr={c['current_frac']:.2f}  "
                        f"hi={c['highorder_frac']:.2f}  "
                        f"lo={c['loworder_frac']:.2f}")

    return dict(evals=evals, evecs_skipped=True, svtol=svtol,
                cutoff=cutoff, lam_max=lam_max, n_null=n_null, n_neg=n_neg,
                n_coil=n_coil, grad_norms={k: float(np.linalg.norm(v))
                                           for k, v in grads.items()},
                overlaps=overlaps, null_frac=null_frac, verdict=verdict,
                grad_L_norm=float(np.linalg.norm(gradL)),
                null_classes=null_classes)


def plot_spectrum(spec, out_path):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    evals = spec["evals"]; n = len(evals)
    idx = np.arange(n)

    # A. eigenvalue magnitude spectrum (sign-coloured)
    a = ax[0]
    pos = evals > 0
    a.semilogy(idx[pos], evals[pos], ".", color="tab:blue", label="λ > 0")
    a.semilogy(idx[~pos], np.abs(evals[~pos]), "x", color="tab:red",
               label="λ < 0 (|λ| shown)")
    a.axhline(spec["cutoff"], color="grey", ls=":",
              label=f"svtol·λmax = {spec['cutoff']:.1e}")
    a.set_xlabel("eigenvalue index (ascending |λ| order not guaranteed)")
    a.set_ylabel("|λ|")
    a.set_title(f"TEST C: AL2 coil-Hessian spectrum\n"
                f"{spec['n_null']}/{spec['n_coil']} below svtol cutoff, "
                f"{spec['n_neg']} negative")
    a.grid(which="both", alpha=0.3); a.legend(fontsize=8)

    # B. per-constraint overlap with each eigenvector
    b = ax[1]
    for name, coef in spec["overlaps"].items():
        b.semilogy(np.abs(evals) + 1e-300, coef + 1e-300, ".", ms=4,
                   alpha=0.6, label=name)
    b.axvline(spec["cutoff"], color="grey", ls=":")
    b.set_xscale("log")
    b.set_xlabel("|λ| of eigenvector")
    b.set_ylabel(r"$|v_k \cdot \hat{g}_i|$")
    b.set_title("TEST C: where does dc/dy live in the eigenbasis?\n"
                "(mass left of the dotted line = inverted catastrophically)")
    b.grid(which="both", alpha=0.3); b.legend(fontsize=8)

    # C. cumulative gradient mass below threshold
    c = ax[2]
    ths = np.logspace(np.log10(max(np.min(np.abs(evals)), 1e-16)),
                      np.log10(spec["lam_max"]), 60)
    for name, coef in spec["overlaps"].items():
        mass = [float(np.linalg.norm(coef[np.abs(evals) < t])) for t in ths]
        c.loglog(ths, np.asarray(mass) + 1e-300, label=name)
    c.axvline(spec["cutoff"], color="grey", ls=":",
              label="svtol·λmax")
    c.set_xlabel("eigenvalue threshold t")
    c.set_ylabel(r"$\|P_{|\lambda|<t}\, \hat{g}_i\|$")
    c.set_title("TEST C: cumulative gradient mass vs threshold\n"
                "(flat ≈0 left of cutoff ⇒ GAUGE_SAFE)")
    c.grid(which="both", alpha=0.3); c.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# TEST D — Eq.(18) ADJOINT Taylor test (the paper-relevant validation)
# ---------------------------------------------------------------------------

def c_total_at(state, x_new, baseline):
    """AL1 constraint values at the bilevel point ( x_new, y*(x_new) ):
    set the surface, WARM re-solve the coils, evaluate. Restores baseline
    coils + the original surface afterwards so each call is independent."""
    surf = state["surf"]
    x0 = np.asarray(surf.x, np.float64).copy()
    snap = baseline[0]
    surf.x = np.asarray(x_new, np.float64).copy()
    solve_warm(state, baseline)
    v = eval_c(state)
    surf.x = x0
    restore_coils(state, snap)
    return v


def adjoint_directional(state, x_coils, lag_mul, mu, d, regularize, rcond):
    """Eq.(18) adjoint directional derivative (dc_i/dx . d) for the AL1
    constraints, built with the given B1 regularisation setting. No inner
    re-solve: grad_individual evaluates the analytic Eq.(18) total derivative
    at the AL2 optimum (x_coils, lag_mul, mu)."""
    eq18 = auglag_solver(
        surface=state["surf"], maxiter=0, max_iter_auglag=0, tau=1,
        f=state.get("f_al2"),
        constraints=state["al2_c_list"], max_mode=state["max_mode"],
        constraint_names=AL2_CONSTRAINT_NAMES,
        implicit_regularize=regularize, implicit_rcond=rcond)
    G = np.asarray(eq18.grad_individual(x_coils, lag_mul, mu))   # (8, n_surf)
    G_al1 = G[AL1_COIL_CONSTRAINT_INDICES]                       # (3, n_surf)
    return G_al1 @ np.asarray(d, np.float64)                     # (3,)


def eval_raw_flux(state):
    """Raw (smooth, unthresholded) squared flux at the current coil state."""
    state["bs"].set_points(state["surf"].gamma().reshape((-1, 3)))
    return np.atleast_1d(float(state["raw_flux"].J()))


def adjoint_directional_raw(state, x_coils, lag_mul, mu, d, regularize, rcond):
    """Eq.(18) adjoint directional derivative of the RAW squared flux.

    The raw-flux objective is APPENDED to the AL2 constraint list with
    lambda=0, mu=0, so it contributes nothing to the AL2 Lagrangian Hessian or
    cross-Hessian (those stay exactly the inner problem's). grad_individual
    then returns its row using the shared H/cross and the raw-flux gradients
    d(flux)/dx, d(flux)/dy — i.e. the correct Eq.(18) total derivative of the
    smooth flux through the re-optimised coils."""
    c_ext = state["al2_c_list"] + [state["raw_flux"]]
    names_ext = AL2_CONSTRAINT_NAMES + ["Jf_raw"]
    lag_ext = np.concatenate([np.atleast_1d(np.asarray(lag_mul, np.float64)),
                              [0.0]])
    mu_ext = np.concatenate([np.atleast_1d(np.asarray(mu, np.float64)), [0.0]])
    eq18 = auglag_solver(
        surface=state["surf"], maxiter=0, max_iter_auglag=0, tau=1,
        f=state.get("f_al2"),
        constraints=c_ext, max_mode=state["max_mode"],
        constraint_names=names_ext,
        implicit_regularize=regularize, implicit_rcond=rcond)
    G = np.asarray(eq18.grad_individual(x_coils, lag_ext, mu_ext))  # (9, n_surf)
    return np.atleast_1d(G[-1] @ np.asarray(d, np.float64))         # raw-flux row


def taylor_adjoint_test(state, baseline, x_coils, lag_mul, mu,
                        eps_grid, seed=0, rcond=1e-5, raw_flux=False):
    """The paper validation. Along a random surface direction d, compare the
    Eq.(18) adjoint directional derivative g.d to finite differences of the
    BILEVEL metric c(x, y*(x)) (coils re-solved at each step):

        T1(eps) = | c(x+eps d) - c(x) |                 -> slope 1
        T2(eps) = | c(x+eps d) - c(x) - eps (g.d) |     -> slope 2 iff g correct

    raw_flux=False : c = the 3 clamped AL1 constraints (Jf penalty, Jls, J_kappa).
    raw_flux=True  : c = the single RAW squared flux (smooth, unthresholded) —
                     the signal-rich, paper-aligned quantity.

    Both the B1-on (production) and B1-off (exact) adjoints are evaluated; T2
    uses B1-on (the gradient the optimiser actually uses), and the central FD
    directional derivative D_FD(eps) is recorded so the reader sees FD converge
    to the adjoint value. Baseline c0 uses the SAME warm path as the
    perturbations (no cold/warm offset)."""
    if raw_flux:
        names = ["Jf_raw"]
        eval_fn = eval_raw_flux
        def gd_fn(dd, reg):
            return adjoint_directional_raw(state, x_coils, lag_mul, mu, dd,
                                           regularize=reg, rcond=rcond)
    else:
        names = list(AL1_COIL_CONSTRAINT_NAMES)
        eval_fn = eval_c
        def gd_fn(dd, reg):
            return adjoint_directional(state, x_coils, lag_mul, mu, dd,
                                       regularize=reg, rcond=rcond)

    rng = np.random.default_rng(seed)
    x0 = np.asarray(state["surf"].x, np.float64).copy()
    d = rng.standard_normal(x0.size); d /= np.linalg.norm(d)

    def _bilevel(x_new):
        """c(x_new, y*(x_new)) via warm re-solve; restores baseline after."""
        snap = baseline[0]
        state["surf"].x = np.asarray(x_new, np.float64).copy()
        solve_warm(state, baseline)
        v = np.atleast_1d(eval_fn(state))
        state["surf"].x = x0
        restore_coils(state, snap)
        return v

    state["surf"].x = x0
    gd_b1 = np.atleast_1d(gd_fn(d, True))
    state["surf"].x = x0
    gd_exact = np.atleast_1d(gd_fn(d, False))
    state["surf"].x = x0
    for k, n in enumerate(names):
        proc0_print(f"  [D] {n:9s} adjoint g.d:  B1={gd_b1[k]:+.6e}   "
                    f"exact={gd_exact[k]:+.6e}")

    c0 = _bilevel(x0)
    proc0_print(f"  [D] baseline c (warm) = {c0}")

    T1 = {n: [] for n in names}
    T2 = {n: [] for n in names}
    Dfd = {n: [] for n in names}
    fin = []
    for eps in eps_grid:
        c_plus = _bilevel(x0 + eps * d)
        c_minus = _bilevel(x0 - eps * d)
        ok = np.all(np.isfinite(c_plus)) and np.all(np.isfinite(c_minus))
        fin.append(bool(ok))
        d_fd = (c_plus - c_minus) / (2.0 * eps)
        for k, n in enumerate(names):
            T1[n].append(abs(c_plus[k] - c0[k]))
            T2[n].append(abs(c_plus[k] - c0[k] - eps * gd_b1[k]))
            Dfd[n].append(float(d_fd[k]))
        proc0_print(f"    eps={eps:.2e}  D_FD[{names[0]}]={d_fd[0]:+.4e}  "
                    f"(adjoint B1={gd_b1[0]:+.4e})")

    state["surf"].x = x0
    restore_coils(state, baseline[0])
    return dict(eps=np.asarray(eps_grid), d=d, c0=c0, names=names,
                gd_b1=np.asarray(gd_b1), gd_exact=np.asarray(gd_exact),
                T1={k: np.asarray(v) for k, v in T1.items()},
                T2={k: np.asarray(v) for k, v in T2.items()},
                Dfd={k: np.asarray(v) for k, v in Dfd.items()},
                finite=np.asarray(fin, dtype=bool), rcond=rcond)


def plot_taylor_adjoint(tay, spec, out_path, focus="Jf"):
    """Two-panel paper figure for `focus` (default Jf):
      (a) Taylor test: T1 (slope 1) + T2 (slope 2) of the bilevel constraint
          vs the Eq.(18) adjoint; reference slopes; fitted T2 slope.
      (b) Coil-Hessian spectrum + where dc/dy lives, with the B1 rcond cutoff
          — explains the regularisation and the T2 floor."""
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # ---- (a) Taylor test ----
    a = ax[0]
    e = tay["eps"]
    T1 = tay["T1"][focus]; T2 = tay["T2"][focus]
    a.loglog(e, T1 + 1e-300, "o", color="tab:blue", ms=6,
             label=r"$T_1=|c(x{+}\epsilon d)-c(x)|$")
    a.loglog(e, T2 + 1e-300, "s", color="tab:red", ms=6,
             label=r"$T_2=|c(x{+}\epsilon d)-c(x)-\epsilon\,\nabla c_{\rm adj}\!\cdot d|$")
    # reference slopes anchored to the first finite point of each curve
    t1a = T1[np.isfinite(T1) & (T1 > 0)]
    t2a = T2[np.isfinite(T2) & (T2 > 0)]
    if t1a.size:
        a.loglog(e, t1a[0] * (e / e[0]), "--", color="tab:blue", alpha=0.5,
                 label="slope 1 (ref)")
    if t2a.size:
        a.loglog(e, t2a[0] * (e / e[0]) ** 2, "--", color="tab:red", alpha=0.5,
                 label="slope 2 (ref)")
    # fit T2 slope over the resolvable window (above the noise floor)
    floor = 3.0 * (spec.get("taylor_floor", 0.0) if spec else 0.0)
    mask = np.isfinite(T2) & (T2 > max(floor, 0.0))
    s2 = fit_slope(e, T2, mask=mask)
    a.set_xlabel(r"perturbation size $\epsilon$", fontsize=12)
    a.set_ylabel("Taylor residual", fontsize=12)
    a.set_title(f"(a) Eq.(18) adjoint Taylor test — squared flux $J_f$\n"
                rf"fitted $T_2$ slope $\approx$ {s2:.2f} (2 $\Rightarrow$ adjoint correct)",
                fontsize=12)
    a.grid(which="both", alpha=0.3)
    a.legend(fontsize=9, loc="upper left")

    # ---- (b) spectrum + gradient overlap for focus constraint ----
    b = ax[1]
    if spec is not None:
        evals = np.asarray(spec["evals"])
        coef = np.asarray(spec["overlaps"][focus])
        cutoff = spec["cutoff"]
        b.scatter(np.abs(evals) + 1e-300, coef + 1e-300, s=22, alpha=0.6,
                  color="tab:purple",
                  label=rf"$|v_k\cdot\widehat{{\partial {focus}/\partial y}}|$")
        b.axvline(cutoff, color="grey", ls=":",
                  label=rf"B1 cutoff $r_{{\rm cond}}\lambda_{{\max}}$")
        b.set_xscale("log"); b.set_yscale("log")
        nf = spec["null_frac"].get(focus, float("nan"))
        nf_str = "nan" if (nf != nf) else f"{nf:.2e}"
        b.set_title(f"(b) where $\\partial {focus}/\\partial y$ lives in the "
                    f"coil-Hessian eigenbasis\n"
                    f"null-space fraction = {nf_str}  "
                    f"({spec['n_null']}/{spec['n_coil']} dirs below cutoff)",
                    fontsize=12)
        b.set_xlabel(r"eigenvalue magnitude $|\lambda_k|$", fontsize=12)
        b.set_ylabel(r"gradient overlap $|v_k\cdot\hat{g}|$", fontsize=12)
        b.grid(which="both", alpha=0.3); b.legend(fontsize=9)
    else:
        b.text(0.5, 0.5, "spectrum not computed", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.replace(".png", ".pdf"))   # vector for the paper
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting / plots
# ---------------------------------------------------------------------------

def fit_slope(eps, y, mask=None):
    y = np.asarray(y, float)
    good = np.isfinite(y) & (y > 0)
    if mask is not None:
        good &= mask
    if good.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(eps[good]), np.log(y[good]), 1)[0])


def make_plots(ptaylor, sens, out_path):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # A. Partial Taylor (fixed coils)
    a = ax[0, 0]
    e = ptaylor["eps"]
    for n in AL1_COIL_CONSTRAINT_NAMES:
        a.loglog(e, ptaylor["T1"][n] + 1e-300, "--o", ms=3, alpha=0.6,
                 label=f"{n}: T1")
        a.loglog(e, ptaylor["T2"][n] + 1e-300, "-s", ms=3, label=f"{n}: T2")
    a.loglog(e, (e / e[0]) * (ptaylor["T1"][AL1_COIL_CONSTRAINT_NAMES[0]][0] + 1e-300),
             "k:", alpha=0.4, label="slope 1")
    a.loglog(e, (e / e[0]) ** 2 * (ptaylor["T2"][AL1_COIL_CONSTRAINT_NAMES[0]][0] + 1e-300),
             "k-.", alpha=0.4, label="slope 2")
    a.set_xlabel(r"$\varepsilon$"); a.set_ylabel("Taylor remainder")
    a.set_title("TEST A: partial Taylor at FIXED coils (no re-solve)\n"
                "validates FD plumbing — T2 slope 2 expected (Jf)")
    a.grid(which="both", alpha=0.3); a.legend(fontsize=7)

    # B. PHYSICAL coil displacement ||Δγ|| vs ε (gamma), with DOF contrast
    b = ax[0, 1]
    es = sens["eps"]
    b.loglog(es, sens["dgeo"] + 1e-300, "-o", color="tab:blue",
             label=r"$\|\Delta\gamma\|$ (coil geometry, m)")
    b.loglog(es, sens["ddof"] + 1e-300, "-^", color="tab:orange", alpha=0.7,
             label=r"$\|\Delta y_{\rm dof}\|$ (gauge-contaminated)")
    if sens["floor_geo"] > 0:
        b.axhline(sens["floor_geo"], color="tab:blue", ls=":",
                  label=f"γ repro floor {sens['floor_geo']:.1e}")
    if sens["floor_dof"] > 0:
        b.axhline(sens["floor_dof"], color="tab:orange", ls=":", alpha=0.6,
                  label=f"DOF repro floor {sens['floor_dof']:.1e}")
    fdg = sens["dgeo"][np.isfinite(sens["dgeo"])]
    if fdg.size:
        b.loglog(es, (es / es[0]) * fdg[0], "k:", alpha=0.4, label="slope 1")
    b.set_xlabel(r"$\varepsilon$ (surface step)")
    b.set_ylabel(r"coil displacement")
    b.set_title("TEST B: how far do auglag coils move PHYSICALLY?\n"
                "γ clean + slope-1 while DOF erratic ⇒ DOF noise is gauge")
    b.grid(which="both", alpha=0.3); b.legend(fontsize=7)

    # C. Directional sensitivity ||Δγ||/ε vs ε — should plateau
    c = ax[1, 0]
    c.semilogx(es, sens["sens"], "-o", color="tab:red")
    if sens["floor_geo"] > 0:
        c.axhline(sens["floor_geo"] / es[0], color="grey", ls=":", alpha=0.5,
                  label="floor/ε (noise ceiling at smallest ε)")
        c.legend(fontsize=7)
    c.set_xlabel(r"$\varepsilon$")
    c.set_ylabel(r"$\|\Delta\gamma\|/\varepsilon$  (m per unit surface move)")
    c.set_title("TEST B: physical coil sensitivity coefficient\n"
                "(plateau = metres of coil motion per unit surface change)")
    c.grid(which="both", alpha=0.3)

    # D. per-coil geometric displacement (which coil moves most)
    dd = ax[1, 1]
    per = sens["dgeo_per"]            # (n_eps, ncoils)
    if per.ndim == 2 and per.shape[1] > 0:
        for i in range(per.shape[1]):
            dd.loglog(es, per[:, i] + 1e-300, "-o", ms=3, label=f"coil {i}")
    dd.set_xlabel(r"$\varepsilon$")
    dd.set_ylabel(r"$\|\Delta\gamma\|$ per coil (m)")
    dd.set_title("TEST B: per-coil geometric displacement\n"
                 "(which coils respond to the surface move)")
    dd.grid(which="both", alpha=0.3); dd.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vmec-input", required=True)
    ap.add_argument("--max-mode", type=int, default=1)
    ap.add_argument("--out-dir", default="output/check_fd")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--hessian-method", choices=["jax", "fd"], default="jax",
                    help="Formulation of the squared-flux AL2-Hessian term "
                         "that Eq.(18) inverts: 'jax' (analytic) or 'fd' "
                         "(finite differences). Set 'fd' to validate exactly "
                         "the production --hessian-method fd path.")
    ap.add_argument("--robust", action="store_true",
                    help="Use reparametrisation-invariant (nearest-point) "
                         "coil-geometry distance instead of pointwise. "
                         "Slower (O(nquad^2)) but immune to curve reparam.")
    ap.add_argument("--hessian-spectrum", action="store_true",
                    help="Run ONLY the Hessian-spectrum diagnostic (Test C): "
                         "one cold baseline solve, then eigendecompose the "
                         "AL2 coil Hessian and measure the overlap of each "
                         "AL1 constraint gradient with the near-null "
                         "eigenspace. Answers whether the flat valley is "
                         "gauge (Eq.18 well-posed) or physical (ill-posed). "
                         "No warm re-solves — much cheaper than Tests A+B.")
    ap.add_argument("--svtol", type=float, default=1e-6,
                    help="Near-null cutoff |lambda| < svtol*lambda_max for "
                         "Test C (match qss.py preconditioner svtol). "
                         "Default 1e-6.")
    ap.add_argument("--mu-max", type=float, default=float("inf"),
                    help="Cap on the AL2 penalty parameter mu. Prevents "
                         "mu-runaway (mu -> ~1e5 at full resolution) that "
                         "inflates the coil-Hessian and stalls the inner "
                         "L-BFGS solve. Try 1e2-1e3. inf = uncapped (default).")
    ap.add_argument("--inner-solver", type=str, default="L-BFGS-B",
                    choices=["L-BFGS-B", "trust-ncg", "trust-krylov",
                             "Newton-CG"],
                    help="AL2 inner-subproblem minimiser. L-BFGS-B stalls on "
                         "the indefinite coil subproblem (~50 negative-curvature "
                         "directions); the trust-region / Newton-CG methods use "
                         "a FD Hessian-vector product to descend through "
                         "negative curvature to the stationary point. Try "
                         "'trust-ncg'.")
    ap.add_argument("--classify-null", action="store_true",
                    help="With --hessian-spectrum: classify each near-null "
                         "coil-Hessian eigenvector by DOF composition "
                         "(currents / high-order Fourier / low-order curve) to "
                         "decide which regulariser the residual degeneracy "
                         "needs. Diagnostic for the full-res convergence fix.")
    ap.add_argument("--taylor-adjoint", action="store_true",
                    help="Produce the PAPER figure (Test D): Taylor test of "
                         "the Eq.(18) adjoint vs bilevel finite differences "
                         "for squared flux Jf, alongside the coil-Hessian "
                         "spectrum/overlap that justifies the B1 "
                         "regularisation. Outputs taylor_adjoint.{png,pdf}.")
    ap.add_argument("--eq18-rcond", type=float, default=1e-5,
                    help="B1 truncated-SVD rcond for the Eq.(18) adjoint solve "
                         "in Test D (matches EQ18_IMPLICIT_RCOND in the main "
                         "script). Default 1e-5.")
    ap.add_argument("--raw-flux", action="store_true",
                    help="In Test D, differentiate the RAW (smooth, "
                         "unthresholded) squared flux instead of the clamped "
                         "AL1 Jf penalty. The smooth flux is the signal-rich, "
                         "paper-aligned quantity (the clamped penalty sits at "
                         "its kink when satisfied and has no Taylor signal).")
    ap.add_argument("--perturb-surface", type=float, default=0.0,
                    help="Before the baseline solve, nudge the surface by this "
                         "magnitude along a random (seeded) direction, moving "
                         "to a FLUX-ACTIVE operating point where dJf/dx is "
                         "large enough to resolve. 0 = warm-start as-is.")
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="B2 ridge weight on the AL2 objective "
                         "(0.5*w*||y_coil-ref||^2). Lifts the flat coil-optimum "
                         "eigenvalues so y*(x) is unique and the solve is "
                         "stable. Sweep (1e-4, 1e-3, 1e-2) until "
                         "null_frac[Jf_raw] < 1e-2. 0 = off.")
    ap.add_argument("--arclength-gauge", type=float, default=0.0,
                    help="Arclength gauge-fix weight on the AL2 objective "
                         "(w * sum ArclengthVariation). Fixes ONLY the curve "
                         "reparametrisation null direction (orthogonal to flux), "
                         "so it lifts the flat directions WITHOUT fighting the "
                         "objective — letting the solve actually converge. "
                         "Sweep (1e-3, 1e-2, 1e-1). Combinable with --ridge.")
    args = ap.parse_args()

    if args.fast:
        CONFIG.update(FAST_OVERRIDES)
        eps_taylor = FAST_EPS_GRID_TAYLOR
        eps_sens   = FAST_EPS_GRID_SWEEP
    else:
        eps_taylor = EPS_GRID_TAYLOR
        eps_sens   = EPS_GRID_SWEEP
    CONFIG["hessian_method"]  = args.hessian_method   # after --fast, so it wins
    CONFIG["ridge_weight"]    = args.ridge
    CONFIG["arclength_weight"] = args.arclength_gauge
    CONFIG["al2_mu_max"]      = args.mu_max
    CONFIG["al2_minimize_method"] = args.inner_solver

    # Partial Taylor (fixed coils) has no solver noise, so probe a wider,
    # finer ε range than the (solve-bound) sensitivity test.
    eps_taylor_A = np.logspace(-7, -1.3, 16)

    os.makedirs(args.out_dir, exist_ok=True)
    proc0_print("=" * 64)
    proc0_print("  check_fd_eq18 — partial Taylor + coil-sensitivity tests")
    proc0_print("=" * 64)
    proc0_print(f"  fast={args.fast}  max_mode={args.max_mode}  "
                f"hessian_method={CONFIG['hessian_method']}  "
                f"ridge={CONFIG['ridge_weight']:.1e}  "
                f"arclength={CONFIG['arclength_weight']:.1e}  "
                f"mu_max={CONFIG['al2_mu_max']:.1e}  "
                f"inner_solver={CONFIG['al2_minimize_method']}  "
                f"LENGTH_TARGET={CONFIG['LENGTH_TARGET']}")

    state = build_state(args.vmec_input, args.max_mode)

    # ---- Optional: move to a flux-active operating point ----
    if args.perturb_surface > 0:
        rng = np.random.default_rng(args.seed + 99)
        x0 = np.asarray(state["surf"].x, np.float64).copy()
        dd = rng.standard_normal(x0.size); dd /= np.linalg.norm(dd)
        state["surf"].x = x0 + args.perturb_surface * dd
        state["bs"].set_points(state["surf"].gamma().reshape((-1, 3)))
        proc0_print(f"  Perturbed surface by {args.perturb_surface:.3e} along a "
                    f"random direction (flux-active operating point).")

    # ---- One AL1 iteration: integrated coil optimisation ----
    proc0_print("\n  [baseline] cold AL2 coil optimisation ...")
    t0 = time.time()
    x_coils0, lag0, mu0 = solve_baseline(state)
    snap0 = snapshot_coils(state)
    baseline = (snap0, lag0, mu0)
    c0 = eval_c(state)
    proc0_print(f"  baseline done in {time.time()-t0:.1f}s")
    proc0_print(f"    ||lag_mul|| = {np.linalg.norm(lag0):.3e}   "
                f"mu(mean) = {float(np.mean(mu0)):.3e}")
    for k, n in enumerate(AL1_COIL_CONSTRAINT_NAMES):
        proc0_print(f"    baseline c[{n}] = {c0[k]:.6e}")

    # ---- TEST D only: paper figure (adjoint Taylor + spectrum), then exit ----
    if args.taylor_adjoint:
        focus = "Jf_raw" if args.raw_flux else "Jf"
        proc0_print(f"\n  [TEST D] Eq.(18) adjoint Taylor test (paper figure, "
                    f"focus={focus}) ...")
        tay = taylor_adjoint_test(state, baseline, x_coils0, lag0, mu0,
                                  eps_taylor, seed=args.seed,
                                  rcond=args.eq18_rcond, raw_flux=args.raw_flux)
        proc0_print("\n  [TEST D companion] coil-Hessian spectrum ...")
        extra = {"Jf_raw": state["raw_flux"]} if args.raw_flux else None
        spec = hessian_spectrum_test(state, x_coils0, lag0, mu0,
                                     svtol=args.svtol, extra=extra)

        names = tay["names"]
        # Floor estimate for the T2 slope fit: the smallest finite T2 across
        # the probed metrics is the noise floor; pass it through so the fit
        # ignores the floored tail.
        spec["taylor_floor"] = float(np.nanmin(
            [np.nanmin(tay["T2"][n][np.isfinite(tay["T2"][n])])
             for n in names if np.isfinite(tay["T2"][n]).any()] or [0.0]))

        plot_path = os.path.join(args.out_dir, "taylor_adjoint.png")
        plot_taylor_adjoint(tay, spec, plot_path, focus=focus)
        proc0_print(f"\n  Saved PAPER figure to {plot_path} (+ .pdf)")

        # Per-metric fitted T2 slopes (resolvable window).
        slopes = {}
        for n in names:
            T2 = tay["T2"][n]
            m = np.isfinite(T2) & (T2 > 3.0 * spec["taylor_floor"])
            slopes[n] = fit_slope(tay["eps"], T2, mask=m)

        raw = dict(
            max_mode=args.max_mode, n_surf=int(state["surf"].x.size),
            fast=bool(args.fast), hessian_method=CONFIG["hessian_method"],
            ridge_weight=CONFIG["ridge_weight"],
            arclength_weight=CONFIG["arclength_weight"],
            eq18_rcond=args.eq18_rcond, svtol=args.svtol,
            raw_flux=bool(args.raw_flux),
            perturb_surface=float(args.perturb_surface),
            baseline_c={n: float(c0[k])
                        for k, n in enumerate(AL1_COIL_CONSTRAINT_NAMES)},
            taylor=dict(
                names=names, eps=tay["eps"].tolist(),
                gd_b1={n: float(tay["gd_b1"][k]) for k, n in enumerate(names)},
                gd_exact={n: float(tay["gd_exact"][k]) for k, n in enumerate(names)},
                T1={n: tay["T1"][n].tolist() for n in names},
                T2={n: tay["T2"][n].tolist() for n in names},
                Dfd={n: tay["Dfd"][n].tolist() for n in names},
                finite=tay["finite"].tolist(),
                T2_slope=slopes, taylor_floor=spec["taylor_floor"]),
            spectrum=dict(
                eigenvalues=spec["evals"].tolist(),
                lam_max=spec["lam_max"], cutoff=spec["cutoff"],
                n_null=spec["n_null"], n_neg=spec["n_neg"],
                n_coil=spec["n_coil"], grad_L_norm=spec["grad_L_norm"],
                null_frac=spec["null_frac"], verdict=spec["verdict"]),
            solves=dict(count=_solve_count[0], total_s=_solve_time[0]),
        )
        with open(os.path.join(args.out_dir, "taylor_adjoint.json"), "w") as f:
            json.dump(raw, f, indent=2)
        proc0_print(f"  Saved data to {os.path.join(args.out_dir, 'taylor_adjoint.json')}")

        proc0_print("\n" + "=" * 64)
        proc0_print("  TEST D: fitted T2 slopes (2 => Eq.18 adjoint validated):")
        for k, n in enumerate(names):
            proc0_print(f"    {n:9s}: T2 slope = {slopes[n]:+.2f}   "
                        f"adjoint g.d (B1) = {tay['gd_b1'][k]:+.3e}   "
                        f"exact = {tay['gd_exact'][k]:+.3e}")
        proc0_print("=" * 64)
        return

    # ---- TEST C only: Hessian spectrum, then exit ----
    if args.hessian_spectrum:
        proc0_print("\n  [TEST C] AL2 Hessian spectrum + gradient overlap ...")
        extra = {"Jf_raw": state["raw_flux"]} if args.raw_flux else None
        classify_ref = list(state["bs"].dof_names) if args.classify_null else None
        spec = hessian_spectrum_test(state, x_coils0, lag0, mu0,
                                     svtol=args.svtol, extra=extra,
                                     classify_ref=classify_ref)
        # Cheap calibration readout: this is the ~1-solve proxy for whether the
        # regularisation weights will give a clean Taylor run. Want
        # null_frac[Jf_raw] -> 0, n_null -> 0, gradL small.
        proc0_print(f"\n  CALIBRATION (ridge={CONFIG['ridge_weight']:.1e}, "
                    f"arclength={CONFIG['arclength_weight']:.1e}):")
        proc0_print(f"    lam_max={spec['lam_max']:.3e}  cutoff={spec['cutoff']:.3e}  "
                    f"n_null={spec['n_null']}  n_neg={spec['n_neg']}  "
                    f"gradL={spec['grad_L_norm']:.3e}")
        for n in spec["null_frac"]:
            nf = spec["null_frac"][n]
            proc0_print(f"    null_frac[{n:9s}] = "
                        f"{'nan' if nf != nf else f'{nf:.3e}'}  "
                        f"-> {spec['verdict'][n]}")
        plot_path = os.path.join(args.out_dir, "hessian_spectrum.png")
        plot_spectrum(spec, plot_path)
        proc0_print(f"\n  Saved spectrum plot to {plot_path}")

        raw = dict(
            max_mode=args.max_mode, n_surf=int(state["surf"].x.size),
            fast=bool(args.fast), svtol=args.svtol,
            baseline_c={n: float(c0[k])
                        for k, n in enumerate(AL1_COIL_CONSTRAINT_NAMES)},
            eigenvalues=spec["evals"].tolist(),
            lam_max=spec["lam_max"], cutoff=spec["cutoff"],
            n_null=spec["n_null"], n_neg=spec["n_neg"],
            n_coil=spec["n_coil"],
            grad_L_norm=spec["grad_L_norm"],
            grad_norms=spec["grad_norms"],
            overlaps={k: v.tolist() for k, v in spec["overlaps"].items()},
            null_frac=spec["null_frac"],
            verdict=spec["verdict"],
            null_classes=spec.get("null_classes"),
            ridge_weight=CONFIG["ridge_weight"],
            arclength_weight=CONFIG["arclength_weight"],
            solves=dict(count=_solve_count[0], total_s=_solve_time[0]),
        )
        json_path = os.path.join(args.out_dir, "hessian_spectrum.json")
        with open(json_path, "w") as f:
            json.dump(raw, f, indent=2)
        proc0_print(f"  Saved spectrum data to {json_path}")

        proc0_print("\n" + "=" * 64)
        proc0_print("  TEST C verdicts (Eq.18 well-posedness):")
        for n, v in spec["verdict"].items():
            nf = spec["null_frac"][n]
            nf_str = "nan" if np.isnan(nf) else f"{nf:.3e}"
            proc0_print(f"    {n:9s}: {v:11s} (null_frac={nf_str})")
        proc0_print("=" * 64)
        return

    # ---- TEST A: partial Taylor at fixed coils ----
    proc0_print("\n  [TEST A] partial Taylor at FIXED coils (no solves) ...")
    ptaylor = partial_taylor_test(state, snap0, eps_taylor_A, seed=args.seed)

    # ---- TEST B: coil sensitivity ----
    proc0_print("\n  [TEST B] coil-movement sensitivity (warm re-solves) ...")
    sens = coil_sensitivity_test(state, baseline, eps_sens, seed=args.seed,
                                 robust=args.robust)

    # ---- Slope diagnostics ----
    proc0_print("\n  Diagnostics:")
    proc0_print("   TEST A (fixed-coil partial Taylor):")
    A_verdict = {}
    for n in AL1_COIL_CONSTRAINT_NAMES:
        s1 = fit_slope(ptaylor["eps"], ptaylor["T1"][n])
        s2 = fit_slope(ptaylor["eps"], ptaylor["T2"][n])
        direct = abs(float(ptaylor["g_dot_d"][AL1_COIL_CONSTRAINT_NAMES.index(n)]))
        has_dep = direct > 1e-12
        A_verdict[n] = dict(slope_T1=s1, slope_T2=s2, g_dot_d=direct,
                            has_direct_dep=bool(has_dep))
        tag = "" if has_dep else "  (no direct surface dep: g·d≈0, expected)"
        proc0_print(f"    {n:9s}: T1≈{s1:+.2f}  T2≈{s2:+.2f}  "
                    f"|g·d|={direct:.2e}{tag}")

    proc0_print("   TEST B (physical coil sensitivity, gamma metric):")
    fin = sens["finite"]
    above = np.isfinite(sens["dgeo"]) & (sens["dgeo"] > 3 * sens["floor_geo"])
    s_geo = fit_slope(sens["eps"], sens["dgeo"], mask=above)
    s_dof = fit_slope(sens["eps"], sens["ddof"])
    plateau = (float(np.nanmedian(sens["sens"][above]))
               if above.any() else float("nan"))
    proc0_print(f"    finite solves            : {int(fin.sum())}/{len(fin)}")
    proc0_print(f"    γ  repro floor ||Δγ||    : {sens['floor_geo']:.3e} m")
    proc0_print(f"    DOF repro floor ||Δy||   : {sens['floor_dof']:.3e}")
    proc0_print(f"    slope of ||Δγ|| vs ε     : {s_geo:+.2f}  (≈1 ⇒ smooth physical map)")
    proc0_print(f"    slope of ||Δy_dof|| vs ε : {s_dof:+.2f}  (erratic ⇒ gauge noise)")
    proc0_print(f"    γ sensitivity plateau    : {plateau:.3e} m  "
                f"(coil motion per unit surface move)")

    # ---- Plots + JSON ----
    plot_path = os.path.join(args.out_dir, "check_fd_eq18.png")
    make_plots(ptaylor, sens, plot_path)
    proc0_print(f"\n  Saved plot to {plot_path}")

    n_solves, t_solve = _solve_count[0], _solve_time[0]
    t_avg = t_solve / max(n_solves, 1)
    raw = dict(
        max_mode=args.max_mode, n_surf=int(state["surf"].x.size),
        fast=bool(args.fast),
        baseline_c={n: float(c0[k])
                    for k, n in enumerate(AL1_COIL_CONSTRAINT_NAMES)},
        test_A_partial_taylor=dict(
            eps=ptaylor["eps"].tolist(),
            g_dot_d=[float(v) for v in ptaylor["g_dot_d"]],
            T1={n: ptaylor["T1"][n].tolist() for n in AL1_COIL_CONSTRAINT_NAMES},
            T2={n: ptaylor["T2"][n].tolist() for n in AL1_COIL_CONSTRAINT_NAMES},
            verdict=A_verdict),
        test_B_coil_sensitivity=dict(
            eps=sens["eps"].tolist(),
            robust=sens["robust"],
            floor_geo=sens["floor_geo"], floor_dof=sens["floor_dof"],
            floor_geo_per=np.asarray(sens["floor_geo_per"]).tolist(),
            ncoils=sens["ncoils"], nquad=sens["nquad"],
            n_coil_dofs=sens["n_coil_dofs"],
            dgeo=sens["dgeo"].tolist(),
            sens=sens["sens"].tolist(),
            ddof=sens["ddof"].tolist(),
            dgeo_per=sens["dgeo_per"].tolist(),
            finite=sens["finite"].tolist(),
            slope_geo=s_geo, slope_dof=s_dof, sensitivity_plateau=plateau),
        solves=dict(count=n_solves, total_s=t_solve, avg_s=t_avg),
    )
    json_path = os.path.join(args.out_dir, "check_fd_eq18.json")
    with open(json_path, "w") as f:
        json.dump(raw, f, indent=2)
    proc0_print(f"  Saved data to {json_path}")
    proc0_print(f"\n  AL2 solves: {n_solves}  total {t_solve:.0f}s  "
                f"avg {t_avg:.1f}s/solve")
    proc0_print("=" * 64)


if __name__ == "__main__":
    main()
