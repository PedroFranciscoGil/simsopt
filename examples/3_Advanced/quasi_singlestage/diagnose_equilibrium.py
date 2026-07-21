#!/usr/bin/env python3
"""
diagnose_equilibrium.py — Post-processing diagnostics for QSS QI optimization output.

Reads the VMEC equilibrium (wout file) produced by qss_script_jax_qi.py and
generates equilibrium metric reports and publication-quality plots.

Usage:
    python diagnose_equilibrium.py <output_dir>
    python diagnose_equilibrium.py --wout /path/to/wout_file.nc

The script auto-detects the wout file inside the given output directory.
If final_metrics.json is present, it is loaded and printed as well.

Produces:
    plot1_profiles.pdf   — iota, pressure, buco, bvco, jcuru, jcurv, DMerc
    plot2_poincare.pdf   — Poincaré cross-sections at several toroidal angles
    plot3_surface3d.pdf  — 3D boundary surface colored by |B|
    plot4_boozer.pdf     — Boozer |B| contour plots at several flux surfaces
    plot5_fieldlines.pdf — |B| along field lines at several flux surfaces
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from fractions import Fraction
from scipy.interpolate import UnivariateSpline

from simsopt.mhd.vmec import Vmec
from simsopt import make_optimizable

# Try importing Boozer — needed for Boozer plots and QI metrics
try:
    from simsopt.mhd.boozer import Boozer
    HAS_BOOZER = True
except ImportError:
    HAS_BOOZER = False

# Import QI targets from qi_goodman/Targets.py
_script_dir = os.path.dirname(os.path.abspath(__file__))
_qi_targets_dir = os.path.join(_script_dir, 'qi_goodman')
if _qi_targets_dir not in sys.path:
    sys.path.insert(0, _qi_targets_dir)
try:
    from Targets import QIBResidual2
    HAS_QI = True
except ImportError:
    HAS_QI = False

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Post-processing diagnostics for a VMEC equilibrium from QSS QI optimization.",
)
parser.add_argument(
    "output_dir", nargs="?", default=None,
    help="Output directory from qss_script_jax_qi.py (auto-detects wout file).",
)
parser.add_argument(
    "--wout", type=str, default=None,
    help="Explicit path to a wout_*.nc file. Overrides output_dir detection.",
)
parser.add_argument(
    "--no-plots", action="store_true", default=False,
    help="Skip plot generation; only print metrics.",
)
args = parser.parse_args()

# Resolve wout file and save path
if args.wout is not None:
    wout_file = os.path.abspath(args.wout)
    savepath = os.path.dirname(wout_file) + os.sep
elif args.output_dir is not None:
    outdir = os.path.abspath(args.output_dir)
    candidates = sorted(glob.glob(os.path.join(outdir, "wout_*.nc")))
    if len(candidates) == 0:
        print(f"ERROR: No wout_*.nc files found in {outdir}")
        sys.exit(1)
    wout_file = candidates[-1]  # take the last (most recent) one
    savepath = outdir + os.sep
    if len(candidates) > 1:
        print(f"Found {len(candidates)} wout files, using: {os.path.basename(wout_file)}")
else:
    parser.print_help()
    sys.exit(1)

print(f"Reading: {wout_file}")
print(f"Saving plots to: {savepath}")

# ---------------------------------------------------------------------------
# Load VMEC equilibrium
# ---------------------------------------------------------------------------
vmec = Vmec(wout_file, verbose=False)
vmec.run()

# ---------------------------------------------------------------------------
# Extract wout data
# ---------------------------------------------------------------------------
phi = vmec.wout.phi
iotaf = vmec.wout.iotaf
presf = vmec.wout.presf
iotas = vmec.wout.iotas
pres = vmec.wout.pres
ns = vmec.wout.ns
nfp = vmec.wout.nfp
xn = vmec.wout.xn
xm = vmec.wout.xm
xn_nyq = vmec.wout.xn_nyq
xm_nyq = vmec.wout.xm_nyq
rmnc = vmec.wout.rmnc.T
zmns = vmec.wout.zmns.T
bmnc = vmec.wout.bmnc.T
raxis_cc = vmec.wout.raxis_cc
zaxis_cs = vmec.wout.zaxis_cs
buco = vmec.wout.buco
bvco = vmec.wout.bvco
jcuru = vmec.wout.jcuru
jcurv = vmec.wout.jcurv
lasym = vmec.wout.lasym

mpol = vmec.wout.mpol
ntor = vmec.wout.ntor
Aminor_p = vmec.wout.Aminor_p
Rmajor_p = vmec.wout.Rmajor_p
aspect = vmec.wout.aspect
betatotal = vmec.wout.betatotal
betapol = vmec.wout.betapol
betator = vmec.wout.betator
betaxis = vmec.wout.betaxis
ctor = vmec.wout.ctor
DMerc = vmec.wout.DMerc

if lasym == 1:
    rmns = vmec.wout.rmns.T
    zmnc = vmec.wout.zmnc.T
    bmns = vmec.wout.bmns.T
    raxis_cs = vmec.wout.raxis_cs
    zaxis_cc = vmec.wout.zaxis_cc
else:
    rmns = 0 * rmnc
    zmnc = 0 * rmnc
    bmns = 0 * bmnc
    raxis_cs = 0 * raxis_cc
    zaxis_cc = 0 * raxis_cc

# Derived grids
nmodes = len(xn)
s = np.linspace(0, 1, ns)
s_half = [(i - 0.5) / (ns - 1) for i in range(1, ns)]
phiedge = phi[-1]

# ---------------------------------------------------------------------------
# Print scalar equilibrium metrics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  VMEC Equilibrium Metrics")
print("=" * 60)
print(f"  wout file:        {os.path.basename(wout_file)}")
print(f"  NFP:              {nfp}")
print(f"  ns (radial pts):  {ns}")
print(f"  mpol:             {mpol}")
print(f"  ntor:             {ntor}")
print(f"  R_major:          {Rmajor_p:.6f} m")
print(f"  a_minor:          {Aminor_p:.6f} m")
print(f"  Aspect ratio:     {aspect:.6f}")
print(f"  Phiedge:          {phiedge:.6e} Wb")
print(f"  Toroidal current: {ctor:.6e} A")
print(f"  Beta total:       {betatotal:.6e}")
print(f"  Beta poloidal:    {betapol:.6e}")
print(f"  Beta toroidal:    {betator:.6e}")
print(f"  Beta axis:        {betaxis:.6e}")
print(f"  Iota (axis):      {iotaf[0]:.6f}")
print(f"  Iota (edge):      {iotaf[-1]:.6f}")
print(f"  Mean iota:        {np.mean(iotaf):.6f}")
print(f"  Iota range:       [{np.min(iotaf):.6f}, {np.max(iotaf):.6f}]")

# Mirror ratio — replicates MirrorRatioPen from qi_goodman/Targets.py so the
# value reported here matches the one used inside the optimization. That uses
# the first off-axis half-grid surface (bmnc radial index 1), a 100x100 grid
# over (theta, phi in [0, 2*pi/nfp]), and m = (Bmax - Bmin)/(Bmax + Bmin).
xm_nyq_arr = vmec.wout.xm_nyq
xn_nyq_arr = vmec.wout.xn_nyq
bmnc_T = vmec.wout.bmnc.T            # same .T convention as Targets.py
bmns_T = (vmec.wout.bmns.T if lasym else np.zeros_like(bmnc_T))

ntheta_mirror = 100
nzeta_mirror = 100
theta_m = np.linspace(0, 2 * np.pi, ntheta_mirror)
zeta_m = np.linspace(0, 2 * np.pi / nfp, nzeta_mirror)
zeta_2d, theta_2d = np.meshgrid(zeta_m, theta_m)
b_surf = np.zeros_like(zeta_2d)
for imode in range(len(xn_nyq_arr)):
    angle = xm_nyq_arr[imode] * theta_2d - xn_nyq_arr[imode] * zeta_2d
    b_surf += bmnc_T[1, imode] * np.cos(angle) + bmns_T[1, imode] * np.sin(angle)
Bmax_surf = np.max(b_surf)
Bmin_surf = np.min(b_surf)
mirror_ratio = (Bmax_surf - Bmin_surf) / (Bmax_surf + Bmin_surf)
print(f"  Mirror ratio (MirrorRatioPen surface): {mirror_ratio:.6f}")
print(f"  |B| range (s_half[1] surface): [{Bmin_surf:.6f}, {Bmax_surf:.6f}] T")

# DMerc summary
dmerc_positive_frac = np.sum(DMerc[1:-1] > 0) / max(len(DMerc[1:-1]), 1)
print(f"  DMerc > 0 fraction: {dmerc_positive_frac:.2%}")
print(f"  DMerc range:        [{np.min(DMerc[1:-1]):.6e}, {np.max(DMerc[1:-1]):.6e}]")

# Elongation proxy from boundary shape
ntheta_elong = 500
nzeta_elong = 2 * nfp + 1
theta_e = np.linspace(0, 2 * np.pi, ntheta_elong)
zeta_e = np.linspace(0, 2 * np.pi / nfp, nzeta_elong, endpoint=True)
iradius = ns - 1
max_elongation = 0.0
for iz in range(nzeta_elong):
    R_cs = np.zeros(ntheta_elong)
    Z_cs = np.zeros(ntheta_elong)
    for imode in range(nmodes):
        angle = xm[imode] * theta_e - xn[imode] * zeta_e[iz]
        R_cs += rmnc[iradius, imode] * np.cos(angle) + rmns[iradius, imode] * np.sin(angle)
        Z_cs += zmns[iradius, imode] * np.sin(angle) + zmnc[iradius, imode] * np.cos(angle)
    R_range = np.max(R_cs) - np.min(R_cs)
    Z_range = np.max(Z_cs) - np.min(Z_cs)
    if R_range > 0:
        elong = max(Z_range / R_range, R_range / Z_range)
        max_elongation = max(max_elongation, elong)
print(f"  Max elongation:   {max_elongation:.6f}")

# QI residual (requires booz_xform via QIBResidual2)
if HAS_QI and HAS_BOOZER:
    print("\n" + "-" * 60)
    print("  QI Omnigenity Residual (QIBResidual2)")
    print("-" * 60)

    # Use same parameters as qss_script_jax_qi.py
    QI_NPHI    = 301
    QI_NALPHA  = 75
    QI_NBJ     = 51
    QI_MPOL    = 20
    QI_NTOR    = 20
    QI_SARR    = [1/51, 10/51, 20/51, 30/51, 40/51, 50/51]
    QI_WEIGHTS = 1e-6 / np.array([4.1e-8, 5.9e-7, 9.6e-7, 1.2e-6, 1.48e-6, 2.38e-6])

    # Override from final_metrics.json if available
    metrics_file_check = os.path.join(savepath, "final_metrics.json")
    if os.path.isfile(metrics_file_check):
        with open(metrics_file_check) as _f:
            _fm = json.load(_f)
        if "qi_parameters" in _fm:
            qp = _fm["qi_parameters"]
            QI_NPHI   = qp.get("nphi", QI_NPHI)
            QI_NALPHA = qp.get("nalpha", QI_NALPHA)
            QI_NBJ    = qp.get("nBj", QI_NBJ)
            QI_MPOL   = qp.get("mpol", QI_MPOL)
            QI_NTOR   = qp.get("ntor", QI_NTOR)
            QI_SARR   = qp.get("sarr", QI_SARR)
            if "weights" in qp:
                QI_WEIGHTS = np.array(qp["weights"])
            print("  (QI parameters loaded from final_metrics.json)")

    print(f"  nphi={QI_NPHI}, nalpha={QI_NALPHA}, nBj={QI_NBJ}, "
          f"mpol={QI_MPOL}, ntor={QI_NTOR}")
    print(f"  sarr={[round(x, 4) for x in QI_SARR]}")

    qi_opt = make_optimizable(
        QIBResidual2, vmec, QI_SARR,
        nphi=QI_NPHI, nalpha=QI_NALPHA, nBj=QI_NBJ,
        mpol=QI_MPOL, ntor=QI_NTOR, weights=QI_WEIGHTS,
    )
    qi_residuals = qi_opt.J()
    qi_total = float(np.sum(qi_residuals**2))
    print(f"  QI residual (sum of squares): {qi_total:.6e}")
    print(f"  QI residual (rms):            {float(np.sqrt(np.mean(qi_residuals**2))):.6e}")
    print(f"  QI residual (max |r_i|):      {float(np.max(np.abs(qi_residuals))):.6e}")
    print(f"  QI residual vector length:    {len(qi_residuals)}")
elif not HAS_BOOZER:
    print("\n  SKIPPING QI residual: booz_xform not installed.")
else:
    print("\n  SKIPPING QI residual: could not import QIBResidual2 from qi_goodman/Targets.py.")

# ---------------------------------------------------------------------------
# Load final_metrics.json if available
# ---------------------------------------------------------------------------
metrics_file = os.path.join(savepath, "final_metrics.json")
final_metrics = None
if os.path.isfile(metrics_file):
    with open(metrics_file) as f:
        final_metrics = json.load(f)
    print("\n" + "-" * 60)
    print("  Optimization metrics (from final_metrics.json)")
    print("-" * 60)
    for section_name, section in final_metrics.items():
        if isinstance(section, dict):
            print(f"\n  [{section_name}]")
            for k, v in section.items():
                if isinstance(v, list) and len(v) > 10:
                    print(f"    {k}: [{v[0]}, ..., {v[-1]}] ({len(v)} items)")
                else:
                    print(f"    {k}: {v}")

print("\n" + "=" * 60)

if args.no_plots:
    print("Skipping plots (--no-plots).")
    sys.exit(0)

# ===================================================================
# PLOT 1: Profiles (iota, pressure, buco, bvco, jcuru, jcurv, DMerc)
# ===================================================================
print("Generating plot 1: radial profiles...")

fig = plt.figure("VMEC Profiles", figsize=(14, 7))
fig.patch.set_facecolor("white")
xLabel = r"$s = \psi_N$"

numCols = 3
numRows = 3
plotNum = 1

plt.subplot(numRows, numCols, plotNum); plotNum += 1
plt.plot(s, iotaf, ".-", label="iotaf")
plt.plot(s_half, iotas[1:], ".-", label="iotas")
plt.legend(fontsize="x-small")
plt.xlabel(xLabel)
plt.title("Rotational transform")

plt.subplot(numRows, numCols, plotNum); plotNum += 1
plt.plot(s, presf, ".-", label="presf")
plt.plot(s_half, pres[1:], ".-", label="pres")
plt.legend(fontsize="x-small")
plt.xlabel(xLabel)
plt.title("Pressure")

plt.subplot(numRows, numCols, plotNum); plotNum += 1
plt.plot(s_half, buco[1:], ".-")
plt.title("buco")
plt.xlabel(xLabel)

plt.subplot(numRows, numCols, plotNum); plotNum += 1
plt.plot(s_half, bvco[1:], ".-")
plt.title("bvco")
plt.xlabel(xLabel)

plt.subplot(numRows, numCols, plotNum); plotNum += 1
plt.plot(s, jcuru, ".-")
plt.title("jcuru")
plt.xlabel(xLabel)

plt.subplot(numRows, numCols, plotNum); plotNum += 1
plt.plot(s, jcurv, ".-")
plt.title("jcurv")
plt.xlabel(xLabel)

plt.subplot(numRows, numCols, plotNum); plotNum += 1
plt.plot(s[:-2], np.sign(DMerc[:-2]), ".-")
plt.title("sgn(DMerc)")
plt.xlabel(xLabel)

plt.subplot(numRows, numCols, plotNum); plotNum += 1
plt.plot(s[1:-1], phiedge**2 * DMerc[1:-1] * s[1:-1], ".-")
plt.title(r"$\psi_{edge}^2 \, s \, D_{Merc}$")
plt.xlabel(xLabel)

plt.subplot(numRows, numCols, plotNum); plotNum += 1
plt.plot(s[1:-1], DMerc[1:-1], ".-")
plt.title(r"$D_{Merc}$")
plt.xlabel(xLabel)

plt.subplots_adjust(left=0.054, bottom=0.07, right=0.94, top=0.95, wspace=0.29, hspace=0.44)
plt.savefig(savepath + "plot1_profiles.pdf")
plt.close(fig)
print(f"  Saved {savepath}plot1_profiles.pdf")

# ===================================================================
# PLOT 2: Poincare cross-sections
# ===================================================================
print("Generating plot 2: Poincare cross-sections...")

ntheta = 500
nzeta = 5
nradius = 9
radii = np.floor(np.linspace(1, ns - 1, nradius)).astype(int)
theta_arr = np.linspace(0, 2 * np.pi, num=ntheta)
zeta_arr = np.linspace(0, 2 * np.pi / nfp / 2, num=nzeta, endpoint=True)

denoms = np.linspace(0, 1, num=nzeta)
titles = []
for d in denoms:
    frac = Fraction(d).limit_denominator()
    anglestr = str(frac.numerator) + r"$\pi/$" + str(frac.denominator)
    if anglestr[:7] == r"1$\pi/$":
        anglestr = r"$\pi/$" + anglestr[7:]
    titles.append(r"$\phi=$" + anglestr)
titles[0] = r"$\phi=0$"
titles[-1] = r"$\phi=\pi$"

R_poinc = np.zeros((ntheta, nzeta, nradius))
Z_poinc = np.zeros((ntheta, nzeta, nradius))
for itheta in range(ntheta):
    for izeta in range(nzeta):
        for iradius in range(nradius):
            rad = radii[iradius]
            angle = xm * theta_arr[itheta] - xn * zeta_arr[izeta]
            rb = np.sum(rmnc[rad, :] * np.cos(angle) + rmns[rad, :] * np.sin(angle))
            zb = np.sum(zmns[rad, :] * np.sin(angle) + zmnc[rad, :] * np.cos(angle))
            R_poinc[itheta, izeta, iradius] = rb
            Z_poinc[itheta, izeta, iradius] = zb

Raxis = np.zeros(nzeta)
Zaxis = np.zeros(nzeta)
for jn in range(len(raxis_cc)):
    n = jn * nfp
    Raxis += raxis_cc[jn] * np.cos(n * zeta_arr)
    Zaxis += zaxis_cs[jn] * np.sin(n * zeta_arr)

fig = plt.figure("Poincare Plots", figsize=(14, 7))
fig.patch.set_facecolor("white")
for izeta in range(nzeta):
    plt.subplot(2, 3, izeta + 1)
    for iradius in range(nradius):
        lw = 1.0 if iradius == nradius - 1 else 0.4
        plt.plot(R_poinc[:, izeta, iradius], Z_poinc[:, izeta, iradius], "k-", linewidth=lw)
    plt.plot(Raxis[izeta], Zaxis[izeta], "r+", markersize=6)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("R")
    plt.ylabel("Z")
    plt.title(titles[izeta])

plt.subplots_adjust(wspace=0.39, hspace=0.444)
plt.savefig(savepath + "plot2_poincare.pdf")
plt.close(fig)
print(f"  Saved {savepath}plot2_poincare.pdf")

# ===================================================================
# PLOT 3: 3D surface with |B| coloring
# ===================================================================
print("Generating plot 3: 3D boundary surface...")

ntheta_3d = 200
nzeta_3d = 300
theta1D = np.linspace(0, 2 * np.pi, num=ntheta_3d)
zeta1D = np.linspace(0, 2 * np.pi, num=nzeta_3d) + 2 * np.pi / nfp
zeta2D, theta2D = np.meshgrid(zeta1D, theta1D)
iradius = ns - 1

R_3d = np.zeros((ntheta_3d, nzeta_3d))
Z_3d = np.zeros((ntheta_3d, nzeta_3d))
B_3d = np.zeros((ntheta_3d, nzeta_3d))

for imode in range(nmodes):
    angle = xm[imode] * theta2D - xn[imode] * zeta2D
    R_3d += rmnc[iradius, imode] * np.cos(angle) + rmns[iradius, imode] * np.sin(angle)
    Z_3d += zmns[iradius, imode] * np.sin(angle) + zmnc[iradius, imode] * np.cos(angle)

for imode in range(len(xn_nyq)):
    angle = xm_nyq[imode] * theta2D - xn_nyq[imode] * zeta2D
    B_3d += bmnc[iradius, imode] * np.cos(angle) + bmns[iradius, imode] * np.sin(angle)

X_3d = R_3d * np.cos(zeta2D)
Y_3d = R_3d * np.sin(zeta2D)
B_rescaled = (B_3d - B_3d.min()) / (B_3d.max() - B_3d.min())

fig = plt.figure("3D Surface Plot")
fig.patch.set_facecolor("white")
ax = fig.add_subplot(projection="3d", azim=0, elev=90)
ax.dist = 13
ax._axis3don = False
ax.plot_surface(X_3d, Y_3d, Z_3d, facecolors=cm.jet(B_rescaled),
                rstride=1, cstride=1, antialiased=False, zorder=2)
ax.auto_scale_xyz([X_3d.min(), X_3d.max()], [X_3d.min(), X_3d.max()],
                  [X_3d.min(), X_3d.max()])
plt.savefig(savepath + "plot3_surface3d.pdf")
plt.close(fig)
print(f"  Saved {savepath}plot3_surface3d.pdf")

# ===================================================================
# PLOT 4: Boozer |B| contours
# ===================================================================
if HAS_BOOZER:
    print("Generating plot 4: Boozer |B| contours...")

    sarr_booz = [0.01, 0.5, 0.75, 0.996]
    nphi_booz = 100
    ntheta_booz = 100
    mpol_booz = 40
    ntor_booz = 40
    nconts = 20

    nfp_loc = vmec.wout.nfp

    if vmec.wout.bmnc[1, 1] < 0:
        phimin_booz = np.pi / nfp_loc
    else:
        phimin_booz = 0
    phimax_booz = phimin_booz + 2 * np.pi / nfp_loc

    if nfp_loc == 1:
        phimax_lab = r"2$\pi$"
    elif nfp_loc % 2 == 1:
        phimax_lab = r"2$\pi$/" + str(int(nfp_loc))
    else:
        if nfp_loc == 2:
            phimax_lab = r"$\pi$"
        else:
            phimax_lab = r"$\pi$/" + str(int(nfp_loc / 2))

    phis_booz = np.linspace(0, 2 * np.pi / nfp_loc, nphi_booz) + phimin_booz
    thetas_booz = np.linspace(0, 2 * np.pi, ntheta_booz)
    phis2D_b, thetas2D_b = np.meshgrid(phis_booz, thetas_booz)

    B_booz = np.zeros((len(sarr_booz), ntheta_booz, nphi_booz))
    iotas_booz = np.zeros(len(sarr_booz))

    fig = plt.figure(figsize=(10, 7))
    for js in range(len(sarr_booz)):
        s_val = sarr_booz[js]
        boozer = Boozer(vmec, mpol_booz, ntor_booz)
        boozer.bx.verbose = False
        boozer.register(s_val)
        boozer.run()

        xm_b = boozer.bx.xm_b
        xn_b = boozer.bx.xn_b
        bmnc_b = boozer.bx.bmnc_b
        bmns_b = boozer.bx.bmns_b

        for jmn in range(len(xm_b)):
            angle = xm_b[jmn] * thetas2D_b - xn_b[jmn] * phis2D_b
            B_booz[js, :, :] += bmnc_b[jmn] * np.cos(angle)
            if lasym:
                B_booz[js, :, :] += bmns_b[jmn] * np.sin(angle)
        iotas_booz[js] = boozer.bx.iota[js]

        plt.subplot(2, 2, js + 1)
        plt.contour(phis2D_b - phimin_booz, thetas2D_b, B_booz[js, :, :],
                     nconts, linewidths=1.0, cmap=cm.jet)
        plt.colorbar()
        plt.gca().tick_params(direction="in", length=0)
        color = (0.9, 0.9, 0.9)
        plt.text(0.05, 6.13, f"|B| @ s={s_val}", ha="left", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.1", ec=color, fc=color))
        plt.yticks([0, 2 * np.pi], ["0", r"$2\pi$"])
        plt.ylabel(r"$\theta$", labelpad=-12)
        plt.xticks([0, 2 * np.pi / nfp_loc], ["0", phimax_lab])
        plt.xlabel(r"$\varphi$", labelpad=-7)

    # Print mirror ratio from Boozer on innermost surface
    Bmax_b0 = np.max(B_booz[0, :, :])
    Bmin_b0 = np.min(B_booz[0, :, :])
    print(f"  Boozer mirror ratio (s={sarr_booz[0]}): "
          f"{(Bmax_b0 - Bmin_b0) / (Bmax_b0 + Bmin_b0):.6f}")

    plt.subplots_adjust(left=0.054, bottom=0.07, right=0.94, top=0.982,
                        wspace=0.39, hspace=0.244)
    plt.savefig(savepath + "plot4_boozer.pdf")
    plt.close(fig)
    print(f"  Saved {savepath}plot4_boozer.pdf")
else:
    print("SKIPPING plot 4: booz_xform not installed (pip install booz_xform).")

# ===================================================================
# PLOT 5: |B| along field lines
# ===================================================================
if HAS_BOOZER:
    print("Generating plot 5: |B| along field lines...")

    nfpinc = 10
    Nphi_fl = 301 * nfpinc
    sarr_fl = [2 / ns, 0.25, 0.5, 0.75, 1.0]
    phis_fl = np.linspace(0, 2 * nfpinc * np.pi / nfp, Nphi_fl)
    mpol_fl = 40
    ntor_fl = 40

    n_surfaces = len(sarr_fl)
    B_fl = np.zeros((n_surfaces, Nphi_fl))

    fig, axs = plt.subplots(n_surfaces, 1, figsize=(5, n_surfaces * 9 / 5),
                            sharex=True, sharey=True)
    if n_surfaces == 1:
        axs = [axs]

    for js in range(n_surfaces):
        ax = axs[js]
        s_val = sarr_fl[js]
        boozer = Boozer(vmec, mpol_fl, ntor_fl)
        boozer.bx.verbose = False
        boozer.register(s_val)
        boozer.run()

        xm_b = boozer.bx.xm_b
        xn_b = boozer.bx.xn_b
        bmnc_b = boozer.bx.bmnc_b

        iota = UnivariateSpline(vmec.s_half_grid, vmec.wout.iotas[1:], k=1, s=0)(s_val)

        for jmn in range(len(xm_b)):
            angle = xm_b[jmn] * iota * phis_fl - xn_b[jmn] * phis_fl
            B_fl[js, :] += bmnc_b[jmn] * np.cos(angle)

        if js == 0:
            B00 = np.mean(B_fl[js, :])
        B_fl[js] = B_fl[js] / B00

        ax.plot(phis_fl, B_fl[js, :])
        phimax_ax = ax.get_xlim()[1]
        color = (0.9, 0.9, 0.9)
        ax.text(0.99 * phimax_ax, 0.22, f"s={np.round(s_val, 2)}",
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.1", ec=color, fc=color))

    fig.text(0.5, 0.01, r"Boozer toroidal angle $\varphi$", ha="center")
    fig.text(0.01, 0.5, "Magnetic field strength", va="center", rotation="vertical")
    plt.subplots_adjust(left=0.14, bottom=0.05, right=0.94, top=0.99)
    plt.savefig(savepath + "plot5_fieldlines.pdf")
    plt.close(fig)
    print(f"  Saved {savepath}plot5_fieldlines.pdf")
else:
    print("SKIPPING plot 5: booz_xform not installed.")

# ===================================================================
# Done
# ===================================================================
print("\n" + "=" * 60)
print("  Diagnostics complete.")
print("=" * 60)
