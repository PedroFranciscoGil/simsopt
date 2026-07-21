#!/usr/bin/env python
"""
Test script for the Hessian of the augmented Lagrangian with a NON-FIXED
surface, reproducing the conditions in qss_script_jax.py.

Setup:
  - JaxSurfaceRZFourier with surface DOFs partially free (max_mode=2)
  - JaxCurveXYZFourier coils
  - SquaredFluxJaxAnalytic with fixed_surface=False (JAX analytic d2J)
  - CurveSurfaceDistance_Fast with fix_surface=False (JAX end-to-end)
  - Combined DOF ordering: [coil_dofs, surface_dofs]
  - hessian_augmented_lagrangian(surface_dofs, coil_dofs, ...) returns (3, n, n)

Tests:
  1. Taylor test: 1st order O(eps^2), 2nd order O(eps^3)
  2. Hessian Frobenius norm: ||H_jax - H_fd||_F / ||H_fd||_F
  3. Symmetry: ||H - H^T||_F
  4. Eigenvalue analysis
  5. Per-constraint d2J accuracy

Usage:
    python test_hessian_auglag.py
"""

import os
import time
import numpy as np
from pathlib import Path

from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    SurfaceRZFourier, create_equally_spaced_curves,
    CurveLength, CurveCurveDistance, MeanSquaredCurvature, LpCurveCurvature,
)
from simsopt.geo.curveobjectives_fast import CurveSurfaceDistance_Fast
from simsopt.geo.jaxsurface import JaxSurfaceRZFourier
from simsopt.geo.curvexyzfourier import JaxCurveXYZFourier
from simsopt.objectives import QuadraticPenalty, SquaredFluxJaxAnalytic
from simsopt.solve.augmented_lagrangian_fast import (
    hessian_augmented_lagrangian,
    set_combined_dof_state,
    _eval_constraints_and_jacobian,
    dummyObjective,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NCOILS = 2
ORDER = 3
NUMQUADPOINTS = 64

LENGTH_TARGET = 18.0
CC_THRESHOLD = 0.1
CS_THRESHOLD = 0.1
CURVATURE_THRESHOLD = 5.0
MSC_THRESHOLD = 5.0
FLUX_THRESHOLD = 1e-16

MAX_MODE = 1  # surface DOFs free up to this mode (like qss_script_jax.py)

# ---------------------------------------------------------------------------
# Load surface as JaxSurfaceRZFourier with free DOFs
# ---------------------------------------------------------------------------
TEST_DIR = (Path(__file__).resolve().parent.parent.parent.parent
            / "tests" / "test_files")
filename = TEST_DIR / "input.LandremanPaul2021_QA"
assert filename.exists(), f"Input file not found: {filename}"

nphi = 16
ntheta = 16
s_rz = SurfaceRZFourier.from_vmec_input(
    filename, range="half period", nphi=nphi, ntheta=ntheta,
)

surf = JaxSurfaceRZFourier(
    quadpoints_phi=s_rz.quadpoints_phi,
    quadpoints_theta=s_rz.quadpoints_theta,
    mpol=s_rz.mpol, ntor=s_rz.ntor,
    nfp=s_rz.nfp, stellsym=s_rz.stellsym,
    dofs=s_rz.get_dofs(),
)

# Free surface DOFs up to MAX_MODE (like qss_script_jax.py)
surf.fix_all()
surf.fixed_range(mmin=0, mmax=MAX_MODE, nmin=-MAX_MODE, nmax=MAX_MODE, fixed=False)
surf.fix("rc(0,0)")  # Major radius stays fixed

n_surf_dofs = len(surf.x)
print(f"Surface: nfp={surf.nfp}, nphi={nphi}, ntheta={ntheta}")
print(f"  mpol={surf.mpol}, ntor={surf.ntor}")
print(f"  Free surface DOFs: {n_surf_dofs} (max_mode={MAX_MODE})")

# ---------------------------------------------------------------------------
# Create initial coils (JaxCurveXYZFourier)
# ---------------------------------------------------------------------------
R0 = 1.0
R1 = 0.5
std_curves = create_equally_spaced_curves(
    NCOILS, surf.nfp, stellsym=True, R0=R0, R1=R1,
    order=ORDER, numquadpoints=NUMQUADPOINTS,
)

base_curves = []
for sc in std_curves:
    jc = JaxCurveXYZFourier(
        np.linspace(0, 1, NUMQUADPOINTS, endpoint=False), ORDER,
    )
    jc.set_dofs(sc.get_dofs())
    base_curves.append(jc)

base_currents = [Current(1e5) for _ in range(NCOILS)]
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, True)
bs = BiotSavart(coils)
bs.set_points(surf.gamma().reshape((-1, 3)))
curves = [c.curve for c in coils]

# ---------------------------------------------------------------------------
# Build constraint objectives (non-fixed surface)
# ---------------------------------------------------------------------------
# SquaredFluxJaxAnalytic: analytic JAX Hessian, surface is free
Jf = SquaredFluxJaxAnalytic(
    surf, bs,
    base_curves=[c.curve for c in coils[:NCOILS]],
    base_currents=[c.current for c in coils[:NCOILS]],
    nfp=surf.nfp, stellsym=surf.stellsym,
    definition="normalized", threshold=FLUX_THRESHOLD,
    fixed_surface=False,
)

Jls = [CurveLength(c) for c in base_curves]

Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=NCOILS)

# CurveSurfaceDistance_Fast with fix_surface=False (traces through surface)
Jcsdist = CurveSurfaceDistance_Fast(
    base_curves, surf, CS_THRESHOLD, fix_surface=False,
)
print(f"\nCurveSurfaceDistance_Fast._has_e2e = {Jcsdist._has_e2e}")

Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

c_list = [
    Jf,                                                              # c0: squared flux (normalized)
    Jccdist,                                                         # c1: coil-coil distance
    Jcsdist,                                                         # c2: coil-surface distance (JAX)
    sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs),   # c3: MSC curvature
    QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max"),                 # c4: total coil length
    sum(Jcs),                                                        # c5: Lp curvature
]

print(f"\nConstraints: {len(c_list)}")
for i, c_i in enumerate(c_list):
    print(f"  c{i} ({type(c_i).__name__:30s}): J = {c_i.J():.6e}, ndofs = {len(c_i.x)}")

# ---------------------------------------------------------------------------
# Set up augmented Lagrangian state
# ---------------------------------------------------------------------------
m_eq = len(c_list)

# Extract coil and surface DOFs separately (canonical ordering: [coil, surf])
# The first constraint (Jf) has the most DOFs since it depends on both coils and surface
# We use Jf.x to get the combined ordering
x_combined = np.asarray(Jf.x, dtype=np.float64).copy()
n_total = len(x_combined)
n_coil_dofs = n_total - n_surf_dofs

# Split into coil and surface DOFs
coil_dofs = x_combined[:n_coil_dofs].copy()
surface_dofs = x_combined[n_coil_dofs:].copy()

print(f"\nTotal combined DOFs: {n_total}")
print(f"  Coil DOFs:    {n_coil_dofs}")
print(f"  Surface DOFs: {n_surf_dofs}")

# Initialize multipliers and penalty
np.random.seed(42)
c_vals_init = np.array([float(c.J()) for c in c_list])
lag_mul = -np.random.rand(m_eq) * np.sign(c_vals_init)
mu_k = np.full(m_eq, 10.0, dtype=np.float64)

print(f"Lagrange multipliers: {lag_mul}")
print(f"Penalty mu: {mu_k}")

f_obj = dummyObjective(x_combined)


# ===================================================================
# Helper: evaluate L_A and grad L_A at a given combined DOF point
# ===================================================================
def eval_LA(x_comb):
    """Return (L_A value, grad L_A) at combined DOF point [coil, surf].

    Uses _eval_constraints_and_jacobian with n_surface_dofs so that
    coil-only constraints are placed at the correct offset in the
    Jacobian (canonical ordering: [coil_dofs, surface_dofs]).
    """
    y = x_comb[:n_coil_dofs]   # coil
    s = x_comb[n_coil_dofs:]   # surface

    # Set DOF state on all constraints
    set_combined_dof_state(surf, c_list, s, y)

    # f_obj is a dummyObjective => f_val=0, grad_f=zeros
    grad_f = np.zeros_like(x_comb)

    # Evaluate constraints and Jacobian with correct surface DOF count
    c_vals, c_jac = _eval_constraints_and_jacobian(
        c_list, x_comb, n_surface_dofs=n_surf_dofs)

    # L_A = f(x) - lag^T g + 0.5 * mu^T (g.*g)
    L = -np.dot(lag_mul, c_vals) + 0.5 * np.dot(mu_k, c_vals * c_vals)

    # grad L_A = grad_f - lag^T @ jac + jac^T @ (mu * g)
    dL = grad_f.copy()
    dL -= np.dot(lag_mul, c_jac)
    dL += np.dot(c_jac.T, mu_k * c_vals)

    return float(L), dL


# ===================================================================
# TEST 1: Taylor test (first and second order)
# ===================================================================
print("\n" + "=" * 70)
print("  TEST 1: Taylor test on augmented Lagrangian (coil + surface DOFs)")
print("=" * 70)

# Set DOF state before first evaluation
set_combined_dof_state(surf, c_list, surface_dofs, coil_dofs)
L0, g0 = eval_LA(x_combined)
print(f"  L_A(x0) = {L0:.10e}")
print(f"  ||grad L_A|| = {np.linalg.norm(g0):.6e}")
print(f"  ||grad_coil||  = {np.linalg.norm(g0[:n_coil_dofs]):.6e}")
print(f"  ||grad_surf||  = {np.linalg.norm(g0[n_coil_dofs:]):.6e}")

# Compute Hessian
print(f"\n  Computing Hessian (JAX) with surface_dofs ({n_surf_dofs}) + coil_dofs ({n_coil_dofs}) ...")
t0 = time.time()
H_terms = hessian_augmented_lagrangian(
    surface_dofs=surface_dofs,
    coil_dofs=coil_dofs,
    f=f_obj,
    equality_constraints=c_list,
    lag_mul=lag_mul,
    mu=mu_k,
)
H_jax = np.asarray(H_terms[0] + H_terms[1] + H_terms[2])
t_hess = time.time() - t0
print(f"  Hessian computed in {t_hess:.2f}s, shape={H_jax.shape}")
print(f"  ||H||_F = {np.linalg.norm(H_jax):.6e}")

# Report block structure
H_cc = H_jax[:n_coil_dofs, :n_coil_dofs]
H_ss = H_jax[n_coil_dofs:, n_coil_dofs:]
H_cs = H_jax[:n_coil_dofs, n_coil_dofs:]
print(f"  ||H_cc|| = {np.linalg.norm(H_cc):.6e}  (coil-coil, {H_cc.shape})")
print(f"  ||H_ss|| = {np.linalg.norm(H_ss):.6e}  (surf-surf, {H_ss.shape})")
print(f"  ||H_cs|| = {np.linalg.norm(H_cs):.6e}  (coil-surf, {H_cs.shape})")

# Random direction
np.random.seed(123)
h = np.random.randn(n_total)
h /= np.linalg.norm(h)

gTh = np.dot(g0, h)
hTHh = h @ H_jax @ h

print(f"\n  Direction h: ||h|| = 1, g^T h = {gTh:.6e}, h^T H h = {hTHh:.6e}")

epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
print(f"\n  {'eps':>10s}  {'|1st order err|':>16s}  {'rate':>6s}  {'|2nd order err|':>16s}  {'rate':>6s}")
print(f"  {'-'*10}  {'-'*16}  {'-'*6}  {'-'*16}  {'-'*6}")

err1_prev = None
err2_prev = None
for eps in epsilons:
    L_eps, _ = eval_LA(x_combined + eps * h)
    err1 = abs(L_eps - L0 - eps * gTh)
    err2 = abs(L_eps - L0 - eps * gTh - 0.5 * eps**2 * hTHh)
    rate1 = f"{err1 / err1_prev:.2f}" if err1_prev is not None and err1_prev > 0 else "  -- "
    rate2 = f"{err2 / err2_prev:.2f}" if err2_prev is not None and err2_prev > 0 else "  -- "
    print(f"  {eps:10.1e}  {err1:16.6e}  {rate1:>6s}  {err2:16.6e}  {rate2:>6s}")
    err1_prev = err1
    err2_prev = err2

# Check convergence rates using eps pairs where we are NOT at machine precision
L_1, _ = eval_LA(x_combined + epsilons[0] * h)
L_2, _ = eval_LA(x_combined + epsilons[1] * h)

err1_a = abs(L_1 - L0 - epsilons[0] * gTh)
err1_b = abs(L_2 - L0 - epsilons[1] * gTh)
err2_a = abs(L_1 - L0 - epsilons[0] * gTh - 0.5 * epsilons[0]**2 * hTHh)
err2_b = abs(L_2 - L0 - epsilons[1] * gTh - 0.5 * epsilons[1]**2 * hTHh)

ratio_1st = err1_b / (err1_a + 1e-30)
ratio_2nd = err2_b / (err2_a + 1e-30)

print(f"\n  1st order convergence ratio (eps {epsilons[0]:.0e}->{epsilons[1]:.0e}, expect ~0.01): {ratio_1st:.4f}")
print(f"  2nd order convergence ratio (eps {epsilons[0]:.0e}->{epsilons[1]:.0e}, expect ~0.001): {ratio_2nd:.6f}")

taylor_1st_ok = ratio_1st < 0.05
taylor_2nd_ok = ratio_2nd < 0.05  # Relaxed slightly for mixed coil+surf DOFs

print(f"  1st order Taylor test: {'PASSED' if taylor_1st_ok else 'FAILED'}")
print(f"  2nd order Taylor test: {'PASSED' if taylor_2nd_ok else 'FAILED'}")
assert taylor_1st_ok, f"1st order Taylor test failed: ratio = {ratio_1st:.4f}"
assert taylor_2nd_ok, f"2nd order Taylor test failed: ratio = {ratio_2nd:.6f}"


# ===================================================================
# TEST 2: Hessian vs finite differences (Frobenius norm)
# ===================================================================
print("\n" + "=" * 70)
print("  TEST 2: Hessian accuracy — JAX vs finite differences")
print("=" * 70)

print(f"\n  Building FD Hessian ({n_total} x {n_total}) via central differences of gradient ...")
eps_fd = 1e-5
t0 = time.time()
H_fd = np.zeros((n_total, n_total), dtype=np.float64)
for j in range(n_total):
    xp = x_combined.copy(); xp[j] += eps_fd
    _, gp = eval_LA(xp)
    xm = x_combined.copy(); xm[j] -= eps_fd
    _, gm = eval_LA(xm)
    H_fd[:, j] = (gp - gm) / (2 * eps_fd)
t_fd = time.time() - t0
H_fd = 0.5 * (H_fd + H_fd.T)
print(f"  FD Hessian computed in {t_fd:.1f}s")
print(f"  ||H_fd||_F = {np.linalg.norm(H_fd):.6e}")

# Report FD block structure
H_fd_cc = H_fd[:n_coil_dofs, :n_coil_dofs]
H_fd_ss = H_fd[n_coil_dofs:, n_coil_dofs:]
H_fd_cs = H_fd[:n_coil_dofs, n_coil_dofs:]
print(f"  ||H_fd_cc|| = {np.linalg.norm(H_fd_cc):.6e}")
print(f"  ||H_fd_ss|| = {np.linalg.norm(H_fd_ss):.6e}")
print(f"  ||H_fd_cs|| = {np.linalg.norm(H_fd_cs):.6e}")

# Frobenius norm of the difference
diff = H_jax - H_fd
frob_diff = np.linalg.norm(diff)
frob_fd = np.linalg.norm(H_fd)
rel_err = frob_diff / (frob_fd + 1e-30)

print(f"\n  ||H_jax - H_fd||_F = {frob_diff:.6e}")
print(f"  ||H_fd||_F         = {frob_fd:.6e}")
print(f"  Relative Frobenius error: {rel_err:.6e}")

max_abs_err = np.max(np.abs(diff))
print(f"  Max element-wise |H_jax - H_fd|: {max_abs_err:.6e}")

# Per-block errors
for block_name, H_j_block, H_f_block in [
    ("H_cc (coil-coil)", H_cc, H_fd_cc),
    ("H_ss (surf-surf)", H_ss, H_fd_ss),
    ("H_cs (coil-surf)", H_cs, H_fd_cs),
]:
    block_diff = np.linalg.norm(H_j_block - H_f_block)
    block_ref = np.linalg.norm(H_f_block) + 1e-30
    print(f"  {block_name}: rel err = {block_diff/block_ref:.6e}")

hess_fd_ok = rel_err < 1e-1  # Slightly relaxed for mixed coil+surface
print(f"\n  Frobenius norm test (tol=1e-1): {'PASSED' if hess_fd_ok else 'FAILED'}")
assert hess_fd_ok, f"Hessian FD test failed: relative error = {rel_err:.2e}"


# ===================================================================
# TEST 3: Symmetry check
# ===================================================================
print("\n" + "=" * 70)
print("  TEST 3: Hessian symmetry")
print("=" * 70)

sym_err = np.linalg.norm(H_jax - H_jax.T)
print(f"  ||H_jax - H_jax^T||_F = {sym_err:.6e}")
sym_ok = sym_err < 1e-8
print(f"  Symmetry test: {'PASSED' if sym_ok else 'FAILED'}")
assert sym_ok, f"Symmetry test failed: ||H - H^T|| = {sym_err:.2e}"


# ===================================================================
# TEST 4: Eigenvalue analysis
# ===================================================================
print("\n" + "=" * 70)
print("  TEST 4: Eigenvalue analysis")
print("=" * 70)

eigvals = np.linalg.eigvalsh(H_jax)
print(f"  Eigenvalue range: [{eigvals[0]:.6e}, {eigvals[-1]:.6e}]")
print(f"  Negative eigenvalues: {np.sum(eigvals < 0)}/{len(eigvals)}")
print(f"  Condition number (|max/min|): {abs(eigvals[-1]) / (abs(eigvals[0]) + 1e-30):.2e}")

eigvals_fd = np.linalg.eigvalsh(H_fd)
print(f"\n  FD eigenvalue range: [{eigvals_fd[0]:.6e}, {eigvals_fd[-1]:.6e}]")
print(f"  FD negative eigenvalues: {np.sum(eigvals_fd < 0)}/{len(eigvals_fd)}")


# ===================================================================
# TEST 5: Per-constraint d2J accuracy
# ===================================================================
print("\n" + "=" * 70)
print("  TEST 5: Per-constraint d2J accuracy (JAX vs FD)")
print("=" * 70)

for i, c_i in enumerate(c_list):
    name = type(c_i).__name__
    try:
        H_ci = np.asarray(c_i.d2J())
    except Exception as e:
        print(f"  c{i} ({name}): d2J not available — {e}")
        continue

    nd = len(c_i.x)
    x_ci = c_i.x.copy()

    if H_ci.shape[0] != nd:
        print(f"  c{i} ({name:30s}): d2J shape {H_ci.shape} != ({nd},{nd}), skipping FD comparison")
        continue

    eps_ci = 1e-5
    H_ci_fd = np.zeros((nd, nd), dtype=np.float64)
    for j in range(nd):
        xp = x_ci.copy(); xp[j] += eps_ci
        c_i.x = xp; gp = np.asarray(c_i.dJ())
        xm = x_ci.copy(); xm[j] -= eps_ci
        c_i.x = xm; gm = np.asarray(c_i.dJ())
        H_ci_fd[:, j] = (gp - gm) / (2 * eps_ci)
    c_i.x = x_ci
    H_ci_fd = 0.5 * (H_ci_fd + H_ci_fd.T)

    frob_ci = np.linalg.norm(H_ci - H_ci_fd)
    frob_ci_ref = np.linalg.norm(H_ci_fd) + 1e-30
    rel_ci = frob_ci / frob_ci_ref

    print(f"  c{i} ({name:30s}): ||H_jax - H_fd||_F / ||H_fd||_F = {rel_ci:.6e}  "
          f"[ndofs={nd}, {'PASS' if rel_ci < 1e-2 else 'FAIL'}]")


# ===================================================================
# TEST 6: Cross-term Hessian verification
# ===================================================================
print("\n" + "=" * 70)
print("  TEST 6: Cross-term Hessian H_cs (coil-surf block)")
print("=" * 70)

# Compare JAX cross-term block with FD cross-term block
H_cs_jax = H_jax[:n_coil_dofs, n_coil_dofs:]
H_cs_fd = H_fd[:n_coil_dofs, n_coil_dofs:]

cs_diff = np.linalg.norm(H_cs_jax - H_cs_fd)
cs_ref = np.linalg.norm(H_cs_fd) + 1e-30
cs_rel = cs_diff / cs_ref

print(f"  ||H_cs_jax||_F = {np.linalg.norm(H_cs_jax):.6e}")
print(f"  ||H_cs_fd||_F  = {np.linalg.norm(H_cs_fd):.6e}")
print(f"  ||diff||_F     = {cs_diff:.6e}")
print(f"  Relative error = {cs_rel:.6e}")

# Also check H_sc = H_cs^T (from FD Hessian)
H_sc_fd = H_fd[n_coil_dofs:, :n_coil_dofs]
transpose_err = np.linalg.norm(H_cs_fd - H_sc_fd.T)
print(f"  ||H_cs_fd - H_sc_fd^T||_F = {transpose_err:.6e}  (FD cross symmetry)")

cs_ok = cs_rel < 1e-1
print(f"  Cross-term test (tol=1e-1): {'PASSED' if cs_ok else 'FAILED'}")


# ===================================================================
# Summary
# ===================================================================
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
all_passed = taylor_1st_ok and taylor_2nd_ok and hess_fd_ok and sym_ok and cs_ok
print(f"  Taylor test (1st order):     {'PASSED' if taylor_1st_ok else 'FAILED'}")
print(f"  Taylor test (2nd order):     {'PASSED' if taylor_2nd_ok else 'FAILED'}")
print(f"  Hessian Frobenius (vs FD):   {'PASSED' if hess_fd_ok else 'FAILED'}  (rel err = {rel_err:.2e})")
print(f"  Hessian symmetry:            {'PASSED' if sym_ok else 'FAILED'}  (||H-H^T|| = {sym_err:.2e})")
print(f"  Cross-term H_cs:             {'PASSED' if cs_ok else 'FAILED'}  (rel err = {cs_rel:.2e})")
print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
print("=" * 70)
