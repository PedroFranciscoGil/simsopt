#!/usr/bin/env python3
"""
Convergence test for composite Gauss-Legendre panel quadrature vs uniform
trapezoidal rule in the Biot-Savart kernel.

Both Fourier and B-spline coils represent the same circular coil of radius R
centred at the origin.  We sweep the number of quadrature points and measure
the B field at a test point, comparing against a high-resolution reference.

Expected results:
  - Fourier (CurveXYZFourier): spectral convergence (machine precision fast)
  - B-spline + uniform weights (old): O(N^{-2}) convergence floor from C^1 gammadash
  - B-spline + GL panel weights (new): high-order polynomial convergence
"""

import os
import numpy as np
from simsopt.geo import CurveXYZFourier
from simsopt.geo.curvebspline import JaxCurveBSpline
from simsopt.field import BiotSavart, Current, Coil

R = 1.0
I = 1e5
TEST_POINT = np.array([[0.3, 0.2, 0.4]])
N_CTRL = 24

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output_convergence')
os.makedirs(OUT_DIR, exist_ok=True)


def make_fourier_coil(nquad):
    c = CurveXYZFourier(nquad, 3)
    dofs = np.zeros(c.num_dofs())
    dofs[2] = R   # xc(1) = R  -> x = R cos(2pi t)
    dofs[8] = R   # ys(1) = R  -> y = R sin(2pi t)
    c.set_dofs(dofs)
    return Coil(c, Current(I))


def make_bspline_gl_coil(nquad):
    """B-spline with Gauss-Legendre panel weights (the new default)."""
    c = JaxCurveBSpline(nquad, N_CTRL, degree=3)
    theta = np.linspace(0, 2 * np.pi, N_CTRL, endpoint=False)
    dofs = np.zeros(3 * N_CTRL)
    for i in range(N_CTRL):
        dofs[3 * i] = R * np.cos(theta[i])
        dofs[3 * i + 1] = R * np.sin(theta[i])
    c.set_dofs(dofs)
    return Coil(c, Current(I))


def make_bspline_uniform_coil(nquad):
    """B-spline with uniform trapezoidal weights (the old behaviour)."""
    quadpoints = np.linspace(0, 1, nquad, endpoint=False)
    c = JaxCurveBSpline(quadpoints, N_CTRL, degree=3)
    theta = np.linspace(0, 2 * np.pi, N_CTRL, endpoint=False)
    dofs = np.zeros(3 * N_CTRL)
    for i in range(N_CTRL):
        dofs[3 * i] = R * np.cos(theta[i])
        dofs[3 * i + 1] = R * np.sin(theta[i])
    c.set_dofs(dofs)
    return Coil(c, Current(I))


def eval_B(coil):
    bs = BiotSavart([coil])
    bs.set_points(TEST_POINT)
    return bs.B().copy().flatten()


def run():
    nq_sweep = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]

    B_ref_fourier = eval_B(make_fourier_coil(2048))
    B_ref_bspline = eval_B(make_bspline_gl_coil(2048))

    print(f"Reference B (Fourier,  nq=2048): {B_ref_fourier}")
    print(f"Reference B (BSpline GL, nq→2048): {B_ref_bspline}")
    print(f"|B_ref_fourier| = {np.linalg.norm(B_ref_fourier):.12e}")
    print()

    fourier_err = []
    bspline_gl_err = []
    bspline_uni_err = []
    actual_nq_gl = []

    header = (f"{'nq':>6s}  {'nq_gl':>6s}  {'Fourier err':>14s}  "
              f"{'BSpline GL err':>14s}  {'BSpline uni err':>14s}")
    print(header)
    print("-" * len(header))

    for nq in nq_sweep:
        # Fourier
        B_f = eval_B(make_fourier_coil(nq))
        err_f = np.linalg.norm(B_f - B_ref_fourier) / np.linalg.norm(B_ref_fourier)
        fourier_err.append(err_f)

        # B-spline with GL panels
        coil_gl = make_bspline_gl_coil(nq)
        nq_gl = coil_gl.curve.quadpoints.shape[0]
        actual_nq_gl.append(nq_gl)
        B_gl = eval_B(coil_gl)
        err_gl = np.linalg.norm(B_gl - B_ref_bspline) / np.linalg.norm(B_ref_bspline)
        bspline_gl_err.append(err_gl)

        # B-spline with uniform weights (old trapezoidal rule)
        B_uni = eval_B(make_bspline_uniform_coil(nq))
        err_uni = np.linalg.norm(B_uni - B_ref_bspline) / np.linalg.norm(B_ref_bspline)
        bspline_uni_err.append(err_uni)

        print(f"{nq:6d}  {nq_gl:6d}  {err_f:14.6e}  {err_gl:14.6e}  {err_uni:14.6e}")

    fourier_err = np.array(fourier_err)
    bspline_gl_err = np.array(bspline_gl_err)
    bspline_uni_err = np.array(bspline_uni_err)
    nq_arr = np.array(nq_sweep)

    np.savez(os.path.join(OUT_DIR, 'convergence_data.npz'),
             nq=nq_arr, nq_gl=actual_nq_gl,
             fourier_err=fourier_err,
             bspline_gl_err=bspline_gl_err,
             bspline_uni_err=bspline_uni_err)

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        mask_f = fourier_err > 0
        mask_gl = bspline_gl_err > 0
        mask_uni = bspline_uni_err > 0

        if np.any(mask_f):
            ax.semilogy(nq_arr[mask_f], fourier_err[mask_f], 'bo-',
                        label='Fourier (spectral)', linewidth=2)
        if np.any(mask_uni):
            ax.semilogy(nq_arr[mask_uni], bspline_uni_err[mask_uni], 'r^--',
                        label='B-spline + uniform (trapezoidal)', linewidth=2)
        if np.any(mask_gl):
            ax.semilogy(nq_arr[mask_gl], bspline_gl_err[mask_gl], 'gs-',
                        label='B-spline + GL panels (new)', linewidth=2)

        nq_ref = np.array([32, 512], dtype=float)
        ax.semilogy(nq_ref, 0.5 * (nq_ref / 32) ** -2, 'r:', alpha=0.5,
                    label=r'$O(N^{-2})$ reference')

        ax.set_xlabel('Number of quadrature points', fontsize=13)
        ax.set_ylabel('Relative error in B', fontsize=13)
        ax.set_title('Biot-Savart quadrature convergence', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, 'convergence.png'), dpi=150)
        print(f"\nPlot saved to {OUT_DIR}/convergence.png")
    except ImportError:
        print("\nmatplotlib not available, skipping plot")

    # ---- Summary ----
    idx64 = nq_sweep.index(64)
    print("\n--- Summary ---")
    print(f"Fourier at nq=64:           err = {fourier_err[idx64]:.2e}")
    print(f"B-spline uniform at nq=64:  err = {bspline_uni_err[idx64]:.2e}")
    print(f"B-spline GL at nq=64:       err = {bspline_gl_err[idx64]:.2e}")

    gl_64 = bspline_gl_err[idx64]
    uni_64 = bspline_uni_err[idx64]
    if uni_64 > 0 and gl_64 > 0:
        print(f"GL improvement factor at nq=64: {uni_64 / gl_64:.1f}x")


if __name__ == '__main__':
    run()
