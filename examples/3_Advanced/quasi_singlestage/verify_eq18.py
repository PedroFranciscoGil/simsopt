#!/usr/bin/env python
"""
verify_eq18.py — Diagnose Eq. (18) JAX adjoint vs FD gradients.
==============================================================

Re-creates the AL1-iteration state of a finished or in-progress
qss_script_jax_qi.py run, then computes three gradient references at a
single (x, y*(x), λ_AL2, μ_AL2) configuration:

  (a) JAX  — Eq.(18) total derivative via auglag_solver.grad_individual.
  (b) FD partial — central FD of c_i w.r.t. surface DOFs at FIXED coils
                   (matches what --track-hessian reports in-loop).
  (c) FD total — central FD of c_i w.r.t. surface DOFs, RE-SOLVING AL2
                 (warm-started from y*) at every perturbation. This is the
                 gold-standard reference for Eq.(18).

Three diagnostic tests are run and plotted:

  Test 1 (Taylor): pick random direction d; sweep ε in log-space; plot
                   the first-order Taylor remainder
                   |c(x+εd, y*(x+εd)) - c(x) - ε·(∇c·d)| vs ε.
                   Slope ≈ 2 → JAX adjoint is correct. Slope ≈ 1 → wrong.
  Test 2 (step sweep): on one column j, sweep ε; plot ||FD_total(ε) - JAX||
                       vs ε. Expect a U-curve bottoming near zero.
  Test 3 (scatter): entry-by-entry scatter of JAX vs FD_total (and vs
                    FD_partial) over a subsample of columns.

Usage:

    mpirun -n 2 python verify_eq18.py --run-dir <path/to/output/qss_jax_qi_.../>

The run dir must contain intermediate_state.json + initial_metrics.json +
final_metrics.json (or be a finished run dir). Output (plots + JSON) is
written to <run-dir>/verify_eq18/.

If you change AL2 hyperparameters or constraint thresholds in
qss_script_jax_qi.py, mirror the change in CONFIG below or override via
CLI.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Headless matplotlib (script may run under MPI on a compute node).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simsopt import make_optimizable
from simsopt.mhd import Vmec
from simsopt.util import MpiPartition, proc0_print
from simsopt.geo import create_equally_spaced_curves
from simsopt.field import (
    regularization_circ, coils_via_symmetries, Current, BiotSavart,
)
from simsopt.field.force import LpCurveForce
from simsopt.solve import auglag_solver

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from qss_helpers import (
    build_coil_objectives, fixed_surface,
    reinit_coils_to_surface_scaled_circular,
)


# ---------------------------------------------------------------------------
# Default config — must mirror qss_script_jax_qi.py
# ---------------------------------------------------------------------------

CONFIG = dict(
    # AL2 inner solver (production tolerances from qss_script_jax_qi.py)
    al2_maxiter           = 400,
    al2_max_iter_auglag   = 6,
    al2_tau               = 5,
    al2_grad_tol          = 1e-8,
    al2_c_tol             = 1e-8,
    al2_mu_initial        = 10,
    # Verifier AL2 tolerances. The FD reference must match what Eq.(18)/(19)
    # actually linearizes: y*(x) = argmin_y L_final(y; λ_0, μ_0) at the
    # baseline multipliers held CONSTANT (paper Eq. 19: "we treat the
    # multipliers μ and λ as constants"). So:
    #   - max_iter_auglag_v = 2: exactly ONE inner minimization at fixed
    #     (λ_0, μ_0). NOTE the off-by-one in augmented_lagrangian_method:
    #     it starts k=1 and loops `while ... and k < MAXITER_lag`, so
    #     MAXITER_lag=1 runs ZERO minimizations (returns the warm start
    #     unchanged → coils never move → fd_total == fd_partial, a silent
    #     no-op). MAXITER_lag=2 runs the body once (k=1<2) then exits, which
    #     is the single fixed-multiplier minimization we want. Re-running the
    #     outer loop (>2) would grow μ by τ each iter, and the count that
    #     runs before c_tol differs between +ε/-ε solves → erratic FD.
    #   - grad_tol = 1e-8 (tight): precise y* so the constraint-value noise
    #     (~grad_tol) doesn't flood FD = Δc/ε. Opens the useful ε window down
    #     to ~1e-4.
    #   - maxiter large: enough L-BFGS steps to actually REACH grad_tol from
    #     a perturbed warm start (a low cap stops mid-flight at a
    #     non-stationary, perturbation-dependent point → erratic).
    al2_maxiter_verify    = 800,
    al2_max_iter_auglag_v = 2,
    al2_grad_tol_verify   = 1e-8,
    al2_c_tol_verify      = 1e-8,
    # Coil geometry / constraints (QI defaults)
    LENGTH_TARGET         = 3,
    FLUX_THRESHOLD        = 1e-7,
    CC_THRESHOLD          = 0.10,
    CS_THRESHOLD          = 0.10,
    MSC_THRESHOLD         = 15,
    CURVATURE_THRESHOLD   = 6,
    FORCE_THRESHOLD       = 100,
    COIL_RADIUS_FACTOR    = 3.6,
    COIL_QUADPOINTS       = 256,
    ARC_LENGTH_SAMPLES    = 2048,
    a_reg                 = 0.15,
    ncoils                = 5,
    order                 = 4,
    total_current         = 8.1e6,
    # VMEC geometry
    VMEC_MPOL             = 10,
    VMEC_NTOR             = 10,
    nphi_surf             = 72,
    ntheta_surf           = 72,
)

# Overrides applied when --fast is passed. SMOKE TEST ONLY: trades adjoint
# precision for wall-clock time (loose grad_tol, coarse grids). Keeps the
# single-inner-solve (max_iter_auglag_v=1) so the loose-tol FD isn't ALSO
# corrupted by μ-growth across outer iters. Use the full (non-fast) config
# for the actual validation.
FAST_OVERRIDES = dict(
    al2_maxiter_verify    = 60,
    al2_max_iter_auglag_v = 2,   # =2 ⇒ ONE inner minimization (off-by-one)
    al2_grad_tol_verify   = 1e-3,
    al2_c_tol_verify      = 1e-3,
    COIL_QUADPOINTS       = 64,
    ARC_LENGTH_SAMPLES    = 512,
    nphi_surf             = 32,
    ntheta_surf           = 32,
)
# Trimmed ε grids: drop ε < 1e-4 (below inner-solver precision floor, no
# signal there) and ε > ~0.05 (higher-order Taylor terms dominate).
FAST_EPS_GRID_TAYLOR = np.logspace(-4, -1.3, 8)
FAST_EPS_GRID_SWEEP  = np.logspace(-4, -1.3, 6)
FAST_N_COLS_SCATTER  = 4

# AL2 c_list ordering — must match qss_script_jax_qi.py exactly
AL2_CONSTRAINT_NAMES        = ["Jf", "Jccdist", "Jcsdist", "Jmscs", "Jls",
                               "J_kappa", "Jlink", "Jforce"]
AL1_COIL_CONSTRAINT_INDICES = [0, 4, 5]                  # Jf, Jls, J_kappa
AL1_COIL_CONSTRAINT_NAMES   = ["Jf", "Jls", "J_kappa"]


# ---------------------------------------------------------------------------
# State reconstruction
# ---------------------------------------------------------------------------

def _resolve_source(run_dir, checkpoint, vmec_input, max_mode):
    """Three input modes:

      A. --checkpoint <path>     : explicit AL1 checkpoint JSON; max_mode is
                                   read FROM the file.
      B. --run-dir <path>        : auto-pick the best snapshot:
                                   prefer al1_checkpoint_latest.json, then
                                   intermediate_state.json. max_mode is
                                   read from the file (latest checkpoint)
                                   or must be supplied (intermediate state).
      C. --vmec-input <path>     : standalone "initial state" mode, no AL1
                                   state needed. max_mode required on CLI.

    Returns (snapshot_dict, vmec_input_path, max_mode_resolved).
    snapshot_dict contains surface_dofs and (optionally) lam_eq, lam_coil,
    rho — keys absent if no AL1 state.
    """
    if checkpoint:
        with open(checkpoint) as f:
            snap = json.load(f)
        vinput = snap.get("vmec_input")
        if not vinput:
            raise RuntimeError(
                f"{checkpoint} has no 'vmec_input' key — pass --vmec-input.")
        mm = max_mode if max_mode is not None else snap.get("max_mode")
        if mm is None:
            raise RuntimeError(
                f"{checkpoint} has no 'max_mode' key — pass --max-mode.")
        return snap, vinput, int(mm)

    if vmec_input:
        # Initial-state mode: no AL1 state at all.
        if max_mode is None:
            raise RuntimeError("--vmec-input requires --max-mode.")
        return {}, vmec_input, int(max_mode)

    if run_dir:
        # Prefer the rolling checkpoint, fall back to intermediate_state.
        latest = os.path.join(run_dir, "al1_checkpoint_latest.json")
        inter  = os.path.join(run_dir, "intermediate_state.json")
        init   = os.path.join(run_dir, "initial_metrics.json")
        if os.path.exists(latest):
            with open(latest) as f:
                snap = json.load(f)
            vinput = snap.get("vmec_input") or _read_vmec_input(init)
            mm = max_mode if max_mode is not None else snap.get("max_mode")
            if mm is None:
                raise RuntimeError(
                    f"{latest} has no 'max_mode' — pass --max-mode.")
            return snap, vinput, int(mm)
        if os.path.exists(inter):
            with open(inter) as f:
                snap = json.load(f)
            vinput = _read_vmec_input(init)
            if max_mode is None:
                raise RuntimeError(
                    "intermediate_state.json has no max_mode — pass "
                    "--max-mode (= N_CONTINUATION_STEPS + 1 for that run).")
            return snap, vinput, int(max_mode)
        raise RuntimeError(
            f"No checkpoint found in {run_dir}. Looked for "
            "al1_checkpoint_latest.json and intermediate_state.json. "
            "Either run the main script long enough for at least one AL1 "
            "iter (it now writes al1_checkpoint_*.json at the start of every "
            "outer iter), or use --vmec-input for an initial-state check.")

    raise RuntimeError(
        "Need one of --checkpoint, --run-dir, or --vmec-input.")


def _read_vmec_input(init_metrics_path):
    if not os.path.exists(init_metrics_path):
        raise RuntimeError(
            f"Cannot find {init_metrics_path}; pass --vmec-input explicitly.")
    with open(init_metrics_path) as f:
        return json.load(f)["vmec_input"]


def build_state(snap, vmec_input, max_mode, mpi):
    """Rebuild surf, bs, al2_c_list, and matching solver setup.

    `snap` is either {} (initial-state mode) or a dict with at least
    'surface_dofs'. The surface mask is set to
    (mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode), matching
    qss_script_jax_qi.py at continuation step with that max_mode.
    """
    proc0_print(f"  Loading VMEC from {vmec_input}")
    vmec = Vmec(vmec_input, mpi=mpi, verbose=False,
                surf_type='JaxSurfaceRZFourier', range_surface='full torus',
                nphi=CONFIG["nphi_surf"], ntheta=CONFIG["ntheta_surf"])
    surf = vmec.boundary
    vmec.indata.mpol = CONFIG["VMEC_MPOL"]
    vmec.indata.ntor = CONFIG["VMEC_NTOR"]

    # Restore surface mask matching the continuation step.
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode,
                     fixed=False)
    surf.fix("rc(0,0)")

    # Restore surface DOFs from snapshot if present.
    saved_dofs = snap.get("surface_dofs")
    if saved_dofs is not None:
        saved = np.asarray(saved_dofs, dtype=np.float64)
        if saved.size != surf.x.size:
            raise RuntimeError(
                f"surf.x size mismatch: saved={saved.size}, "
                f"current={surf.x.size}. Pass --max-mode matching the "
                f"snapshot that was written.")
        surf.x = saved
        proc0_print(f"  Loaded {saved.size} surface DOFs from snapshot.")
    else:
        proc0_print(f"  Initial-state mode: using VMEC input surface "
                    f"({surf.x.size} free DOFs at max_mode={max_mode}).")

    # Coils — circular axis layout as the main script does pre-AL2 reinit.
    ncoils = CONFIG["ncoils"]
    order = CONFIG["order"]
    R0 = surf.get_rc(0, 0)
    R1 = surf.get_rc(1, 0) * 3
    base_curves = create_equally_spaced_curves(
        ncoils, surf.nfp, stellsym=surf.stellsym,
        R0=R0, R1=R1, order=order,
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
        CONFIG["ARC_LENGTH_SAMPLES"],
    )

    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))

    coil_objs = build_coil_objectives(
        surf, bs, base_curves_full, base_coils_full, curves, ncoils,
        CONFIG["LENGTH_TARGET"],
        hessian_method="jax",
        flux_threshold=CONFIG["FLUX_THRESHOLD"],
        cc_threshold=CONFIG["CC_THRESHOLD"],
        cs_threshold=CONFIG["CS_THRESHOLD"],
        curvature_threshold=CONFIG["CURVATURE_THRESHOLD"],
        msc_threshold=CONFIG["MSC_THRESHOLD"],
    )
    Jforce = LpCurveForce(base_coils_full, coils, p=2.0,
                          threshold=CONFIG["FORCE_THRESHOLD"], downsample=2)
    al2_c_list = coil_objs["c_list"] + [Jforce]

    return dict(
        vmec=vmec, surf=surf, bs=bs, al2_c_list=al2_c_list,
        base_curves_full=base_curves_full, base_currents=base_currents,
        curves=curves, ncoils=ncoils, order=order,
        coil_objs=coil_objs,
    )


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def make_solver(state, max_mode):
    """Construct an auglag_solver with VERIFIER tolerances (tighter than
    production) so y*(x) is precise enough to make Eq.(18) meaningful."""
    return auglag_solver(
        surface=state["surf"],
        maxiter=CONFIG["al2_maxiter_verify"],
        max_iter_auglag=CONFIG["al2_max_iter_auglag_v"],
        tau=CONFIG["al2_tau"],
        f=None,
        constraints=state["al2_c_list"],
        grad_tol=CONFIG["al2_grad_tol_verify"],
        c_tol=CONFIG["al2_c_tol_verify"],
        max_mode=max_mode,
        constraint_names=AL2_CONSTRAINT_NAMES,
        mu_initial=CONFIG["al2_mu_initial"],
        lag_mul_initial=None,
    )


def cold_start_coils(state):
    """Reset coils to the circular-axis layout. Used ONCE to seed the
    baseline; subsequent perturbations warm-start from the baseline so that
    y*(x) is the IFT-relevant smooth function (no x-dependent reinit step
    folded into y*)."""
    reinit_coils_to_surface_scaled_circular(
        state["surf"], state["base_curves_full"], state["base_currents"],
        state["ncoils"], state["order"],
        CONFIG["total_current"], CONFIG["COIL_RADIUS_FACTOR"],
        CONFIG["ARC_LENGTH_SAMPLES"],
    )
    state["bs"].set_points(state["surf"].gamma().reshape((-1, 3)))


_solve_count = [0]
_solve_time  = [0.0]


def _solve_al2(state, max_mode, solver):
    """Internal: time-stamp + run an already-built solver. Updates global
    solve_stats counters."""
    t0 = time.time()
    with fixed_surface(state["surf"], max_mode):
        x_coils, _fnc, lag_mul, mu = solver.solve()
    state["bs"].set_points(state["surf"].gamma().reshape((-1, 3)))
    _solve_count[0] += 1
    _solve_time[0] += time.time() - t0
    return x_coils, lag_mul, mu


def solve_y_star_cold(state, max_mode):
    """COLD AL2 solve from a circular-axis init. Used ONCE for the
    baseline y*(x_0)."""
    state["bs"].set_points(state["surf"].gamma().reshape((-1, 3)))
    cold_start_coils(state)
    return _solve_al2(state, max_mode, make_solver(state, max_mode))


def capture_baseline(state):
    """Snapshot baseline coil DOFs (curve shape + free currents) AFTER a
    cold AL2 solve so we can restore them before each warm re-solve."""
    return {
        "curve_dofs":   [np.asarray(c.x, dtype=np.float64).copy()
                         for c in state["base_curves_full"]],
        "current_dofs": [np.asarray(c.x, dtype=np.float64).copy()
                         for c in state["base_currents"][:state["ncoils"]-1]],
    }


def restore_baseline_coils(state, snapshot):
    """Reset base curve shapes and base currents to the baseline snapshot.
    The last current is derived as (total - sum others), so it updates
    automatically when the others are restored."""
    for c, dofs in zip(state["base_curves_full"], snapshot["curve_dofs"]):
        c.x = dofs
    for cur, dofs in zip(state["base_currents"][:state["ncoils"]-1],
                          snapshot["current_dofs"]):
        cur.x = dofs
    state["bs"].set_points(state["surf"].gamma().reshape((-1, 3)))


def solve_y_star_warm(state, max_mode, baseline_coils, baseline_lag_mul,
                       baseline_mu):
    """WARM AL2 solve from the baseline (y*_0, λ_0, μ_0). With tight
    grad_tol the inner solver actually iterates a few steps to re-find
    stationarity at the perturbed x; this is the y*(x) the IFT linearizes.
    Each call is independent — no state propagates between calls."""
    restore_baseline_coils(state, baseline_coils)
    solver = auglag_solver(
        surface=state["surf"],
        maxiter=CONFIG["al2_maxiter_verify"],
        max_iter_auglag=CONFIG["al2_max_iter_auglag_v"],
        tau=CONFIG["al2_tau"],
        f=None,
        constraints=state["al2_c_list"],
        grad_tol=CONFIG["al2_grad_tol_verify"],
        c_tol=CONFIG["al2_c_tol_verify"],
        max_mode=max_mode,
        constraint_names=AL2_CONSTRAINT_NAMES,
        mu_initial=baseline_mu,
        lag_mul_initial=baseline_lag_mul,
    )
    return _solve_al2(state, max_mode, solver)


def solve_stats():
    n = _solve_count[0]; t = _solve_time[0]
    return n, t, (t / max(n, 1))


def coil_vals(state):
    """Evaluate AL2 constraints at the current (surf, bs) state."""
    state["bs"].set_points(state["surf"].gamma().reshape((-1, 3)))
    return np.array([c.J() for c in state["al2_c_list"]], dtype=np.float64)


def jax_adjoint(state, max_mode, x_coils, lag_mul, mu):
    """Eq.(18) gradient via auglag_solver.grad_individual.
    Returns (n_AL1, n_surf) — rows in the order AL1_COIL_CONSTRAINT_NAMES."""
    eq18_solver = auglag_solver(
        surface=state["surf"], maxiter=0, max_iter_auglag=0,
        tau=1, f=None, constraints=state["al2_c_list"],
        max_mode=max_mode, constraint_names=AL2_CONSTRAINT_NAMES,
    )
    all_grads = eq18_solver.grad_individual(x_coils, lag_mul, mu)
    return np.asarray(all_grads)[AL1_COIL_CONSTRAINT_INDICES]


def _perturb_e_j(x0, j, eps):
    """Return x0 with x0[j] += eps as a fresh array (do NOT mutate x0)."""
    xp = np.asarray(x0, dtype=np.float64).copy()
    xp[j] += eps
    return xp


def _warm_eval(state, x_new, max_mode, baseline):
    """Set surf.x = x_new, WARM-START AL2 from baseline, evaluate the
    AL1-promoted constraints.

    Fix A (NaN trap): if the warm solve diverges (non-finite constraint
    values), restore the baseline coils so the NaN can't propagate to the
    next evaluation, and return all-NaN to flag THIS point as invalid.
    Returns (vals, ok)."""
    coils_snap, lag_mul_0, mu_0 = baseline
    state["surf"].x = x_new
    solve_y_star_warm(state, max_mode, coils_snap, lag_mul_0, mu_0)
    vals = coil_vals(state)[AL1_COIL_CONSTRAINT_INDICES]
    if not np.all(np.isfinite(vals)):
        # Scrub the diverged coil state back to the known-good baseline.
        restore_baseline_coils(state, coils_snap)
        return np.full(len(AL1_COIL_CONSTRAINT_INDICES), np.nan), False
    return vals, True


def fd_partial_column(state, j, eps, baseline):
    """Central FD of AL1-promoted constraints w.r.t. surf.x[j] at FIXED
    coils, held at the baseline optimum y*_0.

    Restores baseline coils first so the 'fixed' coils are y*_0 (not
    whatever a previous fd_total left behind). Note: must build the
    perturbed vector first, then assign via the .x setter in a single
    call — `surf.x[j] += eps` does NOT work because the surf.x getter
    returns a fresh copy of the DOFs."""
    coils_snap, _, _ = baseline
    restore_baseline_coils(state, coils_snap)
    x0 = np.asarray(state["surf"].x, dtype=np.float64).copy()

    state["surf"].x = _perturb_e_j(x0, j, +eps)
    v_fwd = coil_vals(state)[AL1_COIL_CONSTRAINT_INDICES]

    state["surf"].x = _perturb_e_j(x0, j, -eps)
    v_bwd = coil_vals(state)[AL1_COIL_CONSTRAINT_INDICES]

    state["surf"].x = x0
    state["bs"].set_points(state["surf"].gamma().reshape((-1, 3)))
    return (v_fwd - v_bwd) / (2.0 * eps)


def fd_total_column(state, j, eps, max_mode, baseline):
    """Central FD of AL1-promoted constraints w.r.t. surf.x[j].

    For each perturbation we WARM-START AL2 from the baseline (y*_0, λ_0,
    μ_0). Both fwd and bwd start from the SAME baseline — no propagation
    between calls. If either solve diverges (Fix A trap), the whole column
    is returned as NaN and the coil state is scrubbed back to baseline so
    downstream columns stay clean."""
    coils_snap, lag_mul_0, mu_0 = baseline
    x0 = np.asarray(state["surf"].x, dtype=np.float64).copy()

    v_fwd, ok_fwd = _warm_eval(state, _perturb_e_j(x0, j, +eps),
                               max_mode, baseline)
    v_bwd, ok_bwd = _warm_eval(state, _perturb_e_j(x0, j, -eps),
                               max_mode, baseline)

    # Leave the coils at the clean baseline regardless of outcome, so the
    # subsequent fd_partial / next column sees a defined state.
    state["surf"].x = x0
    restore_baseline_coils(state, coils_snap)

    if not (ok_fwd and ok_bwd):
        return np.full(len(AL1_COIL_CONSTRAINT_INDICES), np.nan)
    return (v_fwd - v_bwd) / (2.0 * eps)


def coil_vals_at_x(state, max_mode, x_new, baseline):
    """Set surf.x = x_new, warm-start AL2 from baseline, return c values.
    NaN-trapped (returns all-NaN on a diverged solve)."""
    vals, _ok = _warm_eval(state, x_new, max_mode, baseline)
    return vals


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def taylor_test(state, max_mode, baseline, *,
                eps_grid, seed=0):
    """Test 1. Pick a random unit direction d in surface-DOF space and
    measure how c(x_0+εd, y*(x_0+εd)) varies with ε:

        T1(ε) = |c(x_0+εd, y*(x_0+εd)) - c(x_0)|.

    Slope ≈ 1 ⇒ y*(x) is a smooth function of x (the FD chord is meaningful).
    Flat (slope ≈ 0) ⇒ the inner AL re-solve isn't actually moving y*.
    """
    rng = np.random.default_rng(seed)
    x0 = np.copy(state["surf"].x)
    d = rng.standard_normal(x0.size); d /= np.linalg.norm(d)

    state["surf"].x = x0
    vals_0 = coil_vals(state)[AL1_COIL_CONSTRAINT_INDICES]
    proc0_print(f"    baseline c = {vals_0}")

    T1 = {n: [] for n in AL1_COIL_CONSTRAINT_NAMES}

    for eps in eps_grid:
        x_new = x0 + eps * d
        vals_eps = coil_vals_at_x(state, max_mode, x_new, baseline)
        delta = vals_eps - vals_0
        for k, name in enumerate(AL1_COIL_CONSTRAINT_NAMES):
            T1[name].append(abs(delta[k]))

    # Sanity check: T1 should NOT be flat in ε. If it is, the AL2 solver
    # isn't moving and the test is uninformative.
    for name in AL1_COIL_CONSTRAINT_NAMES:
        t1 = np.asarray(T1[name])
        if t1.size > 3:
            stdrel = t1.std() / max(t1.mean(), 1e-30)
            if stdrel < 1e-3:
                proc0_print(
                    f"    [WARNING] T1[{name}] is nearly constant in ε "
                    f"(rel std {stdrel:.1e}). AL2 may not be re-solving "
                    f"properly; check al2_grad_tol_verify and warm-start "
                    f"restoration.")

    state["surf"].x = x0
    restore_baseline_coils(state, baseline[0])
    return dict(eps=np.asarray(eps_grid),
                T1={k: np.asarray(v) for k, v in T1.items()},
                direction=d, vals_0=vals_0)


def step_size_sweep(state, max_mode, baseline, *,
                    eps_grid, seed=0):
    """Test 2. Pick one random column j; compute FD_total(ε) at each ε.
    Each ε is independent (warm-start from baseline).

    Key sanity check: does FD_total converge to a stable value as ε
    shrinks, or does it diverge/oscillate? Stable convergence ⇒ the FD
    implementation is producing a well-defined total derivative."""
    rng = np.random.default_rng(seed)
    n_surf = state["surf"].x.size
    j = int(rng.integers(0, n_surf))
    proc0_print(f"    sweeping column j={j}")

    fd_vals = {n: [] for n in AL1_COIL_CONSTRAINT_NAMES}

    for eps in eps_grid:
        fd_col = fd_total_column(state, j, eps, max_mode, baseline)
        for k, name in enumerate(AL1_COIL_CONSTRAINT_NAMES):
            fd_vals[name].append(float(fd_col[k]))
    return dict(eps=np.asarray(eps_grid),
                j=j,
                fd_vals={k: np.asarray(v) for k, v in fd_vals.items()})


def entry_scatter(state, max_mode, baseline, *,
                  eps_total, n_cols=None, seed=0):
    """Test 3. Build FD_total matrix on a random subset of columns at a
    single ε (warm-from-baseline AL2 per column); also build FD_partial.
    Key check: FD_total has no NaNs, no extreme values, and varies with
    column index — i.e., is producing real gradient estimates."""
    rng = np.random.default_rng(seed)
    n_surf = state["surf"].x.size
    if n_cols is None or n_cols >= n_surf:
        cols = np.arange(n_surf)
    else:
        cols = np.sort(rng.choice(n_surf, size=n_cols, replace=False))

    n_al1 = len(AL1_COIL_CONSTRAINT_INDICES)
    fd_total_M   = np.zeros((n_al1, cols.size), dtype=np.float64)
    fd_partial_M = np.zeros((n_al1, cols.size), dtype=np.float64)

    for idx, j in enumerate(cols):
        fd_total_M[:, idx]   = fd_total_column(state, int(j), eps_total,
                                                max_mode, baseline)
        fd_partial_M[:, idx] = fd_partial_column(state, int(j), eps_total,
                                                 baseline)
    return dict(cols=cols,
                fd_total=fd_total_M, fd_partial=fd_partial_M,
                eps_total=eps_total)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_results(taylor, sweep, scatter, out_path):
    """Diagnostic plot focused on whether FD_total is producing well-defined,
    smooth-in-ε estimates of the total derivative dc/dx through y*(x).
    This run is NOT validating the JAX adjoint — it is validating that the
    FD reference itself works."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    e = taylor["eps"]

    # A. Taylor T1: |c(x+εd, y*(x+εd)) - c(x_0)| vs ε. If the FD chord is
    # well-defined, T1 ~ ε (slope 1) in the linear regime. Constant T1 ⇒
    # y*(x) isn't actually responding to surface perturbations.
    ax = axes[0, 0]
    for name in AL1_COIL_CONSTRAINT_NAMES:
        ax.loglog(e, taylor["T1"][name] + 1e-30,
                  marker="o", label=f"{name}")
    ref1 = e * (taylor["T1"][AL1_COIL_CONSTRAINT_NAMES[0]][0] / max(e[0], 1e-30))
    ax.loglog(e, ref1, "k:", alpha=0.4, label="ref slope 1")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$|\,c(x_0+\varepsilon d,\,y^*(x_0+\varepsilon d)) - c(x_0)\,|$")
    ax.set_title("Test 1: c-change vs ε along random direction d\n"
                 "(slope 1 ⇒ y*(x) varies smoothly with x; FD is meaningful)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # B. Step-size sweep: FD_total(ε) for one column j across many ε. If FD
    # is working it should converge to a stable value as ε ↓ (until noise
    # floor); divergent or oscillating curves ⇒ basin switching.
    ax = axes[0, 1]
    e_sweep = sweep["eps"]
    for name in AL1_COIL_CONSTRAINT_NAMES:
        vals = sweep["fd_vals"][name]
        ax.semilogx(e_sweep, vals, marker="o", label=name)
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$FD_{\rm total}\,[c_i,\,j]\,(\varepsilon)$")
    ax.set_title(f"Test 2: FD_total convergence at column j={sweep['j']}\n"
                 "(curves should plateau as ε ↓; oscillation = basin switching)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # C. FD_total entries vs column index (line plot). Shows the FD-derived
    # gradient is a real vector with no NaN, no extreme spikes, varies
    # smoothly with column j.
    ax = axes[1, 0]
    cols = scatter["cols"]
    for k, name in enumerate(AL1_COIL_CONSTRAINT_NAMES):
        ax.plot(cols, scatter["fd_total"][k], marker="o",
                label=f"{name}: FD_total")
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("surface DOF index j")
    ax.set_ylabel(r"$FD_{\rm total}\,[c_i,\,j]$")
    ax.set_title(f"Test 3: FD_total entries vs column  (ε={scatter['eps_total']:.0e})\n"
                 "(finite, varies with j ⇒ FD methodology works)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # D. FD_total vs FD_partial (per-entry scatter). FD_partial ignores how
    # coils respond to surface moves; FD_total includes it. Their
    # disagreement is the IFT cross-Hessian term made visible. Both should
    # be nonzero and bounded — that is the "FD is alive" signal.
    ax = axes[1, 1]
    for k, name in enumerate(AL1_COIL_CONSTRAINT_NAMES):
        ax.scatter(scatter["fd_partial"][k].ravel(),
                   scatter["fd_total"][k].ravel(),
                   s=24, alpha=0.7, label=name)
    lo = float(np.nanmin([scatter["fd_partial"].min(), scatter["fd_total"].min()]))
    hi = float(np.nanmax([scatter["fd_partial"].max(), scatter["fd_total"].max()]))
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
    ax.set_xlabel(r"$FD_{\rm partial}$  (coils fixed)")
    ax.set_ylabel(r"$FD_{\rm total}$  (coils re-solved)")
    ax.set_title("Test 4: FD_total vs FD_partial\n"
                 "(off-diagonal scatter = IFT cross-Hessian term, "
                 "which is what Eq.18 captures)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_verification(state, max_mode, out_dir, *, eps_grid_taylor,
                     eps_grid_sweep, eps_total_scatter, n_cols_scatter,
                     seed):
    os.makedirs(out_dir, exist_ok=True)

    proc0_print("")
    proc0_print("=" * 60)
    proc0_print("  Eq.(18) verification")
    proc0_print("=" * 60)
    proc0_print(f"  n_surf DOFs:    {state['surf'].x.size}")
    proc0_print(f"  AL2 constraints: {len(state['al2_c_list'])}")
    proc0_print(f"  AL1-promoted   : {AL1_COIL_CONSTRAINT_NAMES}")

    # ---- Baseline ----
    # ONE cold solve at x_0 to seed y*_0, λ_0, μ_0. All subsequent
    # perturbations warm-start from this snapshot with tight grad_tol so
    # that y*(x_0 + δx) is the IFT-relevant smooth function (no
    # x-dependent reinit folded in).
    t0 = time.time()
    x_coils_0, lag_mul_0, mu_0 = solve_y_star_cold(state, max_mode)
    proc0_print(f"  Baseline cold AL2 solve done in {time.time()-t0:.1f}s")
    proc0_print(f"    ||lag_mul|| = {np.linalg.norm(lag_mul_0):.3e}")
    proc0_print(f"    mu (mean)   = {float(np.mean(mu_0)):.3e}")
    proc0_print(f"    baseline c  = {coil_vals(state)}")

    coils_snap = capture_baseline(state)
    baseline = (coils_snap, lag_mul_0, mu_0)
    y0 = np.concatenate(coils_snap["curve_dofs"]
                        + coils_snap["current_dofs"])

    # ---- Sanity probe: confirm the warm solve ACTUALLY MOVES the coils ----
    # A silent no-op (e.g. the MAXITER_lag off-by-one) makes y*(x+δx) == y*_0,
    # which collapses fd_total onto fd_partial and zeros the coil-only
    # constraints. Catch it here in seconds rather than after a full run.
    x0_probe = np.asarray(state["surf"].x, dtype=np.float64).copy()
    probe_eps = 1e-3
    state["surf"].x = _perturb_e_j(x0_probe, 0, probe_eps)
    solve_y_star_warm(state, max_mode, coils_snap, lag_mul_0, mu_0)
    y_probe = np.concatenate(
        [np.asarray(c.x, dtype=np.float64).ravel()
         for c in state["base_curves_full"]]
        + [np.asarray(c.x, dtype=np.float64).ravel()
           for c in state["base_currents"][:state["ncoils"]-1]])
    dy = float(np.linalg.norm(y_probe - y0))
    state["surf"].x = x0_probe
    restore_baseline_coils(state, coils_snap)
    proc0_print(f"  Coil-motion probe (ε={probe_eps:.0e} on DOF 0): "
                f"||Δy*|| = {dy:.3e}")
    if dy < 1e-12:
        proc0_print(
            "  [ERROR] Warm solve did NOT move the coils — fd_total will "
            "collapse onto fd_partial. Check al2_max_iter_auglag_v (must be "
            ">=2 due to the MAXITER_lag off-by-one) and grad_tol/maxiter.")

    # NOTE: JAX adjoint is NOT computed in this run. We are exclusively
    # validating that the FD implementation of dc/dx through y*(x) is
    # producing well-defined, finite, smooth-in-ε gradient estimates.

    # ---- Test 1: Taylor (smoothness of y*(x) via T1 scaling) ----
    t0 = time.time()
    proc0_print("\n  [Test 1] Taylor T1: does c(x+εd, y*(x+εd)) - c(x_0) "
                "scale linearly in ε?")
    taylor = taylor_test(state, max_mode, baseline,
                         eps_grid=eps_grid_taylor, seed=seed)
    proc0_print(f"  ...done in {time.time()-t0:.1f}s")

    # ---- Test 2: Step-size sweep (FD_total convergence) ----
    t0 = time.time()
    proc0_print("\n  [Test 2] Step-size sweep: does FD_total(ε) converge "
                "to a stable value as ε shrinks?")
    sweep = step_size_sweep(state, max_mode, baseline,
                            eps_grid=eps_grid_sweep, seed=seed)
    proc0_print(f"  ...done in {time.time()-t0:.1f}s")

    # ---- Test 3: Entry-level FD_total values ----
    t0 = time.time()
    proc0_print("\n  [Test 3] FD_total entries across columns "
                "(check for NaN/extreme values)...")
    scatter = entry_scatter(state, max_mode, baseline,
                            eps_total=eps_total_scatter,
                            n_cols=n_cols_scatter, seed=seed)
    proc0_print(f"  ...done in {time.time()-t0:.1f}s")

    # ---- Health check on FD_total ----
    fd_total_arr = scatter["fd_total"]
    fd_partial_arr = scatter["fd_partial"]
    n_nan_total = int(np.sum(~np.isfinite(fd_total_arr)))
    n_zero_total = int(np.sum(fd_total_arr == 0.0))
    n_entries = fd_total_arr.size
    n_valid = n_entries - n_nan_total
    # Per-ε NaN counts for the step sweep (which ε's the inner solve survived).
    sweep_nan_by_eps = {
        name: int(np.sum(~np.isfinite(sweep["fd_vals"][name])))
        for name in AL1_COIL_CONSTRAINT_NAMES
    }
    proc0_print("")
    proc0_print(f"  FD_total scatter entries: {n_entries} total, "
                f"{n_valid} finite, {n_nan_total} NaN/inf, "
                f"{n_zero_total} exactly zero.")
    if n_valid > 0:
        proc0_print(f"    finite range: [{np.nanmin(fd_total_arr):+.3e}, "
                    f"{np.nanmax(fd_total_arr):+.3e}]")
    else:
        proc0_print("    [WARNING] all FD_total scatter entries NaN — "
                    "inner solve diverging at this ε across all columns.")
    proc0_print(f"  Step-sweep NaN counts per constraint "
                f"(of {sweep['eps'].size} ε's): {sweep_nan_by_eps}")

    # Plots
    plot_path = os.path.join(out_dir, "verify_eq18.png")
    plot_results(taylor, sweep, scatter, plot_path)
    proc0_print(f"\n  Saved diagnostic plot to {plot_path}")

    # Save raw numerics for downstream analysis.
    raw = {
        "max_mode": int(max_mode),
        "n_surf": int(state["surf"].x.size),
        "AL1_constraints": AL1_COIL_CONSTRAINT_NAMES,
        "AL2_constraints": AL2_CONSTRAINT_NAMES,
        "taylor": {
            "eps": taylor["eps"].tolist(),
            "T1":  {n: taylor["T1"][n].tolist()
                    for n in AL1_COIL_CONSTRAINT_NAMES},
        },
        "step_sweep": {
            "eps": sweep["eps"].tolist(),
            "j":   int(sweep["j"]),
            "fd_vals": {n: sweep["fd_vals"][n].tolist()
                        for n in AL1_COIL_CONSTRAINT_NAMES},
        },
        "scatter": {
            "eps_total": float(scatter["eps_total"]),
            "cols": scatter["cols"].tolist(),
            "fd_total":   scatter["fd_total"].tolist(),
            "fd_partial": scatter["fd_partial"].tolist(),
        },
    }
    raw_path = os.path.join(out_dir, "verify_eq18.json")
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)
    proc0_print(f"  Saved raw numbers to {raw_path}")

    # ---- Slope diagnostic on T1 — actionable summary ----
    # T1 = |c(x+εd, y*(x+εd)) - c(x_0)|. We're not validating JAX in this
    # run; we're checking that y*(x) is a well-defined smooth function so
    # the FD chord is meaningful. T1 slope ≈ 1 over the linear regime ⇒
    # FD methodology works. Slope ≈ 0 (flat) ⇒ y* isn't moving with x.
    proc0_print("\n  Slope diagnostic (Test 1, T1 vs ε):")
    proc0_print("    Slope near 1 ⇒ y*(x) varies smoothly with x; FD is meaningful.")
    proc0_print("    Slope near 0 (flat) ⇒ AL2 re-solve isn't actually moving y*.")
    eps = taylor["eps"]
    for name in AL1_COIL_CONSTRAINT_NAMES:
        y = taylor["T1"][name]
        good = (y > 0) & np.isfinite(y)
        if good.sum() >= 2:
            mid = np.where(good)[0][: max(2, good.sum() // 2)]
            slope = np.polyfit(np.log(eps[mid]), np.log(y[mid]), 1)[0]
            proc0_print(f"    {name:10s}  T1 slope ≈ {slope:+.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify Eq.(18) JAX adjoint via Taylor + FD tests.",
        epilog="Provide ONE of --checkpoint, --run-dir, or --vmec-input. "
               "If you give --vmec-input you can verify without ever having "
               "run the main script.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", default=None,
                     help="Path to a specific al1_checkpoint_*.json "
                          "(or intermediate_state.json). max_mode is read "
                          "from the file unless overridden by --max-mode.")
    src.add_argument("--run-dir", default=None,
                     help="qss_script_jax_qi.py output directory. The "
                          "verifier will pick the most recent state "
                          "automatically (al1_checkpoint_latest.json first, "
                          "then intermediate_state.json).")
    src.add_argument("--vmec-input", default=None,
                     help="VMEC input file. Standalone 'initial-state' mode: "
                          "no AL1 state needed. Requires --max-mode.")

    parser.add_argument("--max-mode", type=int, default=None,
                        help="Continuation max_mode. Required with "
                             "--vmec-input; optional otherwise (read from "
                             "checkpoint).")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write plots and JSON. "
                             "Default: <run-dir>/verify_eq18/ or "
                             "./verify_eq18/.")
    parser.add_argument("--n-cols-scatter", type=int, default=12,
                        help="Number of surface DOF columns sampled for "
                             "the FD-total scatter (each costs 2 AL2 solves). "
                             "Default 12. Overridden to "
                             f"{FAST_N_COLS_SCATTER} by --fast.")
    parser.add_argument("--eps-total-scatter", type=float, default=1e-3,
                        help="ε used for FD_total in the scatter test "
                             "(per paper Taylor tests, ~1e-3).")
    parser.add_argument("--fast", action="store_true",
                        help="Trade adjoint precision for wall-clock time: "
                             "loose AL2 tolerances, coarser coil/surface "
                             "grids, shorter ε grids, fewer scatter cols. "
                             "Targets a Taylor-slope sanity check in O(1h).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.fast:
        CONFIG.update(FAST_OVERRIDES)
        eps_grid_taylor = FAST_EPS_GRID_TAYLOR
        eps_grid_sweep  = FAST_EPS_GRID_SWEEP
        n_cols_scatter  = (args.n_cols_scatter
                           if args.n_cols_scatter != 12
                           else FAST_N_COLS_SCATTER)
        proc0_print("  [--fast] using loose tolerances + small grids "
                    f"(grad_tol={CONFIG['al2_grad_tol_verify']:.0e}, "
                    f"coil_quad={CONFIG['COIL_QUADPOINTS']}, "
                    f"surf_grid={CONFIG['nphi_surf']}²).")
    else:
        # Trimmed: ε in [1e-4, ~0.05]. Below 1e-4 is the inner-solver
        # precision floor (no signal); above ~0.05 higher-order Taylor
        # terms dominate.
        eps_grid_taylor = np.logspace(-4, -1.3, 14)
        eps_grid_sweep  = np.logspace(-4, -1.3, 10)
        n_cols_scatter  = args.n_cols_scatter

    snap, vmec_input, max_mode = _resolve_source(
        args.run_dir, args.checkpoint, args.vmec_input, args.max_mode)

    if args.out_dir:
        out_dir = args.out_dir
    elif args.run_dir:
        out_dir = os.path.join(args.run_dir, "verify_eq18")
    elif args.checkpoint:
        out_dir = os.path.join(os.path.dirname(args.checkpoint),
                               "verify_eq18")
    else:
        out_dir = "verify_eq18"

    mpi = MpiPartition()
    proc0_print(f"  MPI ranks: {mpi.comm_world.Get_size()}")
    proc0_print(f"  max_mode  : {max_mode}")
    if "step" in snap:
        proc0_print(f"  Snapshot  : step {snap['step']}, "
                    f"AL1 iter {snap.get('al_iter')}, "
                    f"rho={snap.get('rho')}")

    state = build_state(snap, vmec_input, max_mode, mpi)

    run_verification(
        state, max_mode, out_dir,
        eps_grid_taylor=eps_grid_taylor,
        eps_grid_sweep=eps_grid_sweep,
        eps_total_scatter=args.eps_total_scatter,
        n_cols_scatter=n_cols_scatter,
        seed=args.seed,
    )

    n_solves, t_total, t_avg = solve_stats()
    proc0_print("")
    proc0_print(f"  Total cold AL2 solves: {n_solves}")
    proc0_print(f"  Total AL2 wall time : {t_total/60:.1f} min")
    proc0_print(f"  Mean per-solve time : {t_avg:.1f} s")


if __name__ == "__main__":
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
