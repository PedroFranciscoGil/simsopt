#!/usr/bin/env python3
"""
Squared-flux convergence comparison: Fourier vs B-spline (GL panels vs uniform).

Builds a set of stellarator-symmetric coils for the Landreman & Paul QA
configuration and sweeps the number of curve quadrature points.  For each
count the script evaluates on the target plasma surface:

  * Normalised squared flux  ∫(B·n)² dA / ∫B² dA
  * Average |B·n|/|B|

Three curves are compared:
  1. Fourier (CurveXYZFourier) — trapezoidal rule, spectral convergence
  2. B-spline + GL panels      — composite Gauss-Legendre, high-order convergence
  3. B-spline + uniform        — trapezoidal rule, O(N^{-2}) convergence

Outputs:
  - convergence_data.npz       raw arrays
  - sqflux_convergence.png     normalised squared flux vs nquad
  - bdotn_convergence.png      average |B·n|/|B| vs nquad
  - relative_error.png         relative error from converged value (semilog)
"""

import os
import time
from math import cos, sin, pi
from pathlib import Path

import numpy as np

from simsopt.geo import SurfaceRZFourier
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curvebspline import JaxCurveBSpline
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.objectives import SquaredFlux

# ── Configuration ─────────────────────────────────────────────────────
TEST_DIR = Path(__file__).resolve().parent.parent.parent / 'tests' / 'test_files'
VMEC_INPUT = TEST_DIR / 'input.LandremanPaul2021_QA_lowres'

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output_sqflux')
os.makedirs(OUT_DIR, exist_ok=True)

NCOILS = 4
NFP = 2
STELLSYM = True
TOTAL_CURRENT = 3e5
FOURIER_ORDER = 6
BSPLINE_NCP = 60
BSPLINE_DEGREE = 3

NPHI_SURF = 32
NTHETA_SURF = 32

QUADPOINT_SWEEP = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]


# ── Surface setup (done once) ────────────────────────────────────────
print(f"Loading surface from {VMEC_INPUT} ...")
s = SurfaceRZFourier.from_vmec_input(
    VMEC_INPUT, range='half period', nphi=NPHI_SURF, ntheta=NTHETA_SURF,
)
R0 = float(s.get_rc(0, 0))
R1 = 0.5 * R0


# ── Coil builders ────────────────────────────────────────────────────
def _make_currents():
    base_currents = [
        Current(TOTAL_CURRENT / NCOILS * 1e-5) * 1e5
        for _ in range(NCOILS - 1)
    ]
    total = Current(TOTAL_CURRENT)
    total.fix_all()
    base_currents.append(total - sum(base_currents))
    return base_currents


def make_fourier_coils(numquadpoints):
    curves = []
    for i in range(NCOILS):
        curve = CurveXYZFourier(numquadpoints, FOURIER_ORDER)
        angle = (i + 0.5) * (2 * pi) / ((1 + int(STELLSYM)) * NFP * NCOILS)
        curve.set('xc(0)', cos(angle) * R0)
        curve.set('xc(1)', cos(angle) * R1)
        curve.set('yc(0)', sin(angle) * R0)
        curve.set('yc(1)', sin(angle) * R1)
        curve.set('zs(1)', -R1)
        curve.x = curve.x
        curves.append(curve)
    return coils_via_symmetries(curves, _make_currents(), NFP, STELLSYM)


def make_bspline_gl_coils(numquadpoints):
    """B-spline with Gauss-Legendre panel weights."""
    curves = []
    for i in range(NCOILS):
        angle = (i + 0.5) * (2 * pi) / ((1 + int(STELLSYM)) * NFP * NCOILS)
        curve = JaxCurveBSpline(numquadpoints, BSPLINE_NCP, degree=BSPLINE_DEGREE)
        dofs = np.zeros(3 * BSPLINE_NCP)
        ca, sa = cos(angle), sin(angle)
        for j in range(BSPLINE_NCP):
            theta = 2 * pi * j / BSPLINE_NCP
            ct, st = cos(theta), sin(theta)
            dofs[3 * j + 0] = ca * R0 + ca * R1 * ct
            dofs[3 * j + 1] = sa * R0 + sa * R1 * ct
            dofs[3 * j + 2] = -R1 * st
        curve.set_dofs(dofs)
        curve.x = curve.x
        curves.append(curve)
    return coils_via_symmetries(curves, _make_currents(), NFP, STELLSYM)


def make_bspline_uniform_coils(numquadpoints):
    """B-spline with uniform trapezoidal weights (old behaviour)."""
    quadpoints = np.linspace(0, 1, numquadpoints, endpoint=False)
    curves = []
    for i in range(NCOILS):
        angle = (i + 0.5) * (2 * pi) / ((1 + int(STELLSYM)) * NFP * NCOILS)
        curve = JaxCurveBSpline(quadpoints, BSPLINE_NCP, degree=BSPLINE_DEGREE)
        dofs = np.zeros(3 * BSPLINE_NCP)
        ca, sa = cos(angle), sin(angle)
        for j in range(BSPLINE_NCP):
            theta = 2 * pi * j / BSPLINE_NCP
            ct, st = cos(theta), sin(theta)
            dofs[3 * j + 0] = ca * R0 + ca * R1 * ct
            dofs[3 * j + 1] = sa * R0 + sa * R1 * ct
            dofs[3 * j + 2] = -R1 * st
        curve.set_dofs(dofs)
        curve.x = curve.x
        curves.append(curve)
    return coils_via_symmetries(curves, _make_currents(), NFP, STELLSYM)


# ── Metric evaluation ────────────────────────────────────────────────
def evaluate_metrics(coils):
    bs = BiotSavart(coils)
    bs.set_points(s.gamma().reshape((-1, 3)))

    Jf = SquaredFlux(s, bs, definition='normalized')
    sqflux = float(Jf.J())

    B = bs.B().reshape((NPHI_SURF, NTHETA_SURF, 3))
    n_hat = s.unitnormal()
    BdotN = np.sum(B * n_hat, axis=2)
    absB = np.linalg.norm(B, axis=2)
    avg_BdotN_over_B = float(np.mean(np.abs(BdotN)) / np.mean(absB))

    return sqflux, avg_BdotN_over_B


# ── Main sweep ───────────────────────────────────────────────────────
def run():
    print('=' * 76)
    print('SQUARED-FLUX CONVERGENCE BENCHMARK')
    print(f'  Surface : {VMEC_INPUT.name}')
    print(f'  Fourier : order={FOURIER_ORDER}  ({3*(2*FOURIER_ORDER+1)} DOFs/coil)')
    print(f'  B-spline: n_cp={BSPLINE_NCP}, deg={BSPLINE_DEGREE}  '
          f'({3*BSPLINE_NCP} DOFs/coil)')
    print(f'  Coils   : {NCOILS} base × NFP={NFP} × stellsym')
    print(f'  Quadpts : {QUADPOINT_SWEEP}')
    print('=' * 76)

    fourier_sqflux, fourier_bdotn = [], []
    gl_sqflux, gl_bdotn = [], []
    uni_sqflux, uni_bdotn = [], []
    actual_nq_gl = []

    for nq in QUADPOINT_SWEEP:
        print(f'\n--- numquadpoints = {nq} ---')

        t0 = time.time()
        coils_f = make_fourier_coils(nq)
        sf_f, bn_f = evaluate_metrics(coils_f)
        dt_f = time.time() - t0
        fourier_sqflux.append(sf_f)
        fourier_bdotn.append(bn_f)

        t0 = time.time()
        coils_gl = make_bspline_gl_coils(nq)
        nq_actual = coils_gl[0].curve.quadpoints.shape[0]
        actual_nq_gl.append(nq_actual)
        sf_gl, bn_gl = evaluate_metrics(coils_gl)
        dt_gl = time.time() - t0
        gl_sqflux.append(sf_gl)
        gl_bdotn.append(bn_gl)

        t0 = time.time()
        coils_uni = make_bspline_uniform_coils(nq)
        sf_uni, bn_uni = evaluate_metrics(coils_uni)
        dt_uni = time.time() - t0
        uni_sqflux.append(sf_uni)
        uni_bdotn.append(bn_uni)

        print(f'  Fourier        : sqflux={sf_f:.6e}  <|B·n|>/<|B|>={bn_f:.6e}  ({dt_f:.1f}s)')
        print(f'  BSpline GL(→{nq_actual:3d}): sqflux={sf_gl:.6e}  <|B·n|>/<|B|>={bn_gl:.6e}  ({dt_gl:.1f}s)')
        print(f'  BSpline uni    : sqflux={sf_uni:.6e}  <|B·n|>/<|B|>={bn_uni:.6e}  ({dt_uni:.1f}s)')

    nq_arr = np.array(QUADPOINT_SWEEP)
    fourier_sqflux = np.array(fourier_sqflux)
    fourier_bdotn = np.array(fourier_bdotn)
    gl_sqflux = np.array(gl_sqflux)
    gl_bdotn = np.array(gl_bdotn)
    uni_sqflux = np.array(uni_sqflux)
    uni_bdotn = np.array(uni_bdotn)

    np.savez(
        os.path.join(OUT_DIR, 'convergence_data.npz'),
        quadpoints=nq_arr, nq_gl=actual_nq_gl,
        fourier_sqflux=fourier_sqflux, fourier_bdotn=fourier_bdotn,
        gl_sqflux=gl_sqflux, gl_bdotn=gl_bdotn,
        uni_sqflux=uni_sqflux, uni_bdotn=uni_bdotn,
    )
    print(f'\nRaw data saved to {OUT_DIR}/convergence_data.npz')

    # ── Plots ─────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    # --- Plot 1: normalised squared flux vs nquad ---
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    ax1.semilogy(nq_arr, fourier_sqflux, 'bo-', lw=2,
                 label='Fourier (trapezoidal)')
    ax1.semilogy(nq_arr, gl_sqflux, 'gs-', lw=2,
                 label='B-spline + GL panels')
    ax1.semilogy(nq_arr, uni_sqflux, 'r^--', lw=2,
                 label='B-spline + uniform (trapezoidal)')
    ax1.set_xlabel('Number of quadrature points per coil', fontsize=13)
    ax1.set_ylabel(r'Normalised squared flux $\int (B \cdot n)^2 / \int B^2$',
                   fontsize=13)
    ax1.set_title('Squared flux convergence — Landreman-Paul QA', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUT_DIR, 'sqflux_convergence.png'), dpi=150)
    print(f"  → {OUT_DIR}/sqflux_convergence.png")

    # --- Plot 2: average |B·n|/|B| vs nquad ---
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.semilogy(nq_arr, fourier_bdotn, 'bo-', lw=2,
                 label='Fourier (trapezoidal)')
    ax2.semilogy(nq_arr, gl_bdotn, 'gs-', lw=2,
                 label='B-spline + GL panels')
    ax2.semilogy(nq_arr, uni_bdotn, 'r^--', lw=2,
                 label='B-spline + uniform (trapezoidal)')
    ax2.set_xlabel('Number of quadrature points per coil', fontsize=13)
    ax2.set_ylabel(r'$\langle |B \cdot n| \rangle / \langle |B| \rangle$',
                   fontsize=13)
    ax2.set_title(r'$\langle |B \cdot n| / |B| \rangle$ convergence — Landreman-Paul QA',
                  fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, 'bdotn_convergence.png'), dpi=150)
    print(f"  → {OUT_DIR}/bdotn_convergence.png")

    # --- Plot 3: relative error from converged value (semilog) ---
    ref_f_sq = fourier_sqflux[-1]
    ref_gl_sq = gl_sqflux[-1]
    ref_uni_sq = uni_sqflux[-1]

    ref_f_bn = fourier_bdotn[-1]
    ref_gl_bn = gl_bdotn[-1]
    ref_uni_bn = uni_bdotn[-1]

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

    def _plot_rel_err(ax, nq, vals, ref, label, fmt, **kw):
        err = np.abs(vals - ref) / np.abs(ref)
        mask = err > 0
        if np.any(mask):
            ax.semilogy(nq[mask], err[mask], fmt, label=label, **kw)

    _plot_rel_err(ax3a, nq_arr, fourier_sqflux, ref_f_sq,
                  'Fourier', 'bo-', lw=2)
    _plot_rel_err(ax3a, nq_arr, gl_sqflux, ref_gl_sq,
                  'B-spline + GL', 'gs-', lw=2)
    _plot_rel_err(ax3a, nq_arr, uni_sqflux, ref_uni_sq,
                  'B-spline + uniform', 'r^--', lw=2)
    ax3a.set_xlabel('Number of quadrature points', fontsize=12)
    ax3a.set_ylabel('Relative error in squared flux', fontsize=12)
    ax3a.set_title('Squared flux convergence rate', fontsize=13)
    ax3a.legend(fontsize=10)
    ax3a.grid(True, alpha=0.3)

    _plot_rel_err(ax3b, nq_arr, fourier_bdotn, ref_f_bn,
                  'Fourier', 'bo-', lw=2)
    _plot_rel_err(ax3b, nq_arr, gl_bdotn, ref_gl_bn,
                  'B-spline + GL', 'gs-', lw=2)
    _plot_rel_err(ax3b, nq_arr, uni_bdotn, ref_uni_bn,
                  'B-spline + uniform', 'r^--', lw=2)
    ax3b.set_xlabel('Number of quadrature points', fontsize=12)
    ax3b.set_ylabel(r'Relative error in $\langle|B \cdot n|/|B|\rangle$',
                    fontsize=12)
    ax3b.set_title(r'$|B \cdot n|/|B|$ convergence rate', fontsize=13)
    ax3b.legend(fontsize=10)
    ax3b.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(os.path.join(OUT_DIR, 'relative_error.png'), dpi=150)
    print(f"  → {OUT_DIR}/relative_error.png")

    # ── Summary ───────────────────────────────────────────────────────
    print('\n' + '=' * 76)
    print('SUMMARY (nq=128)')
    idx = QUADPOINT_SWEEP.index(128)
    print(f'  Fourier        : sqflux = {fourier_sqflux[idx]:.8e}   '
          f'<|B·n|>/<|B|> = {fourier_bdotn[idx]:.8e}')
    print(f'  BSpline GL     : sqflux = {gl_sqflux[idx]:.8e}   '
          f'<|B·n|>/<|B|> = {gl_bdotn[idx]:.8e}')
    print(f'  BSpline uniform: sqflux = {uni_sqflux[idx]:.8e}   '
          f'<|B·n|>/<|B|> = {uni_bdotn[idx]:.8e}')

    err_gl  = abs(gl_sqflux[idx] - ref_gl_sq) / abs(ref_gl_sq) if ref_gl_sq != 0 else 0
    err_uni = abs(uni_sqflux[idx] - ref_uni_sq) / abs(ref_uni_sq) if ref_uni_sq != 0 else 0
    print(f'\n  Sqflux relative error at nq=128:')
    print(f'    GL panels : {err_gl:.2e}')
    print(f'    Uniform   : {err_uni:.2e}')
    if err_gl > 0:
        print(f'    GL improvement factor: {err_uni / err_gl:.1f}x')
    print('=' * 76)


if __name__ == '__main__':
    run()
