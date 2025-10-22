#!/usr/bin/env python
r"""
auglag_w7x_Gaussian.py
===============
This script performs coil optimization for the W7-X coil design 
using the Augmented Lagrangian Method (ALM) with a Gaussian perturbation 
to generate the initial coil set.

In order to reproduce the coilsets of the paper the following thresholds/parameters should be set:
1) 4 coils solution (#1)
    ncoils_choice : 4
    LENGTH_TARGET = 38
    FLUX_THRESHOLD = 1e-15
    CC_THRESHOLD = 0.28
    CS_THRESHOLD = 0.3 
    CURVATURE_THRESHOLD = 2.5
    MSC_THRESHOLD = 1.5
    FORCE_THRESHOLD = 3.8 # units of MN/m

2) 5 coils solution (#2)
    ncoils_choice : 5
    LENGTH_TARGET = 45
    FLUX_THRESHOLD = 1e-15
    CC_THRESHOLD = 0.28
    CS_THRESHOLD = 0.3 
    CURVATURE_THRESHOLD = 2
    MSC_THRESHOLD = 1.5
    FORCE_THRESHOLD = 3.0 # units of MN/m

3) 5 coils solution (#3)
    ncoils_choice : 5
    LENGTH_TARGET = 43
    FLUX_THRESHOLD = 1e-15
    CC_THRESHOLD = 0.28
    CS_THRESHOLD = 0.3
    CURVATURE_THRESHOLD = 2
    MSC_THRESHOLD = 1.5
    FORCE_THRESHOLD = 3.5 # units of MN/m
"""

import os
import sys
from pathlib import Path
import time
import numpy as np
from numpy.random import PCG64DXSM, Generator
from simsopt.field import BiotSavart
from simsopt.field import coils_to_vtk
from simsopt.field import coils_via_symmetries, regularization_circ
from simsopt.solve.augmented_lagrangian_initial_weight import augmented_lagrangian_method
from simsopt.field.force import LpCurveForce, B2Energy, coil_force
from simsopt.util import calculate_modB_on_major_radius
from simsopt.geo import (
    CurveLength, CurveCurveDistance, 
    LpCurveCurvature, CurveSurfaceDistance, LinkingNumber,
    SurfaceRZFourier, MeanSquaredCurvature,
    GaussianSampler, CurvePerturbed, PerturbationSample
)
from simsopt.objectives import SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions

def safe_divide(numerator, denominator, epsilon=1e-12):
    """
    Safely divide numerator by denominator, avoiding division by zero.
    Returns 0.0 where denominator is too small.
    """
    return np.where(np.abs(denominator) > epsilon, numerator / denominator, 0.0)

# Set some parameters -- warning this is super low resolution!
if in_github_actions:
    nphi = 4
    ntheta = 4
    MAXITER = 10
    MAXITER_lag = 5
else:
    # Define the number of phi and theta points
    nphi = 32
    ntheta = 32
    MAXITER = 1000  # 1000 for high-resolution
    MAXITER_lag = 40  # 40 for high-resolution

CC_THRESHOLD = 0.28
CS_THRESHOLD = 0.3
LENGTH_TARGET = 38
FORCE_THRESHOLD = 3.8  # 3 for the 5 coil solution, 3.8 for the other solutions
FLUX_THRESHOLD = 1e-15
ncoils_choice = 4
CURVATURE_THRESHOLD = 2.5  # 2 for the 5 coil solutions
MSC_THRESHOLD = 1.5

# Parameters for Gaussian Perturbation
SIGMA = 0.05  # Standard deviation σ for coil errors
L = 0.25  # Length scale L for coil errors

t1 = time.time()

# File for the desired boundary magnetic surface:
TEST_DIR = Path(__file__).parent / '../' / '../' / '../' / 'tests/test_files'
input_name = 'input.W7-X_without_coil_ripple_beta0p05_d23p4_tm'
filename = TEST_DIR / input_name

# Initialize the boundary magnetic surface:
range_param = "half period"
s = SurfaceRZFourier.from_vmec_input(filename, range=range_param, nphi=nphi, ntheta=ntheta)

qphi = nphi * 2
qtheta = ntheta * 2
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)

# Make high resolution, full torus version of the plasma boundary for plotting
s_plot = SurfaceRZFourier.from_vmec_input(
    filename,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)

# Get R0 for reference
R0 = s.get_rc(0, 0)

# Get order from command line argument (SLURM array index + 1)
if len(sys.argv) > 1:
    order = int(sys.argv[1])
else:
    # Default to order 4 
    order = 4

print(f"Running with order = {order}")

# initialize the TF coils
def w7x_coils(s, ncoils=3, order=1):
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.field import Current

    # parameters for the TF coils, increase order for a better solution
    # Total current scaled to give B ~ 5.7 T on axis (actually averaged over the major radius)
    R0 = s.get_rc(0, 0) * 1
    R1 = s.get_rc(1, 0) * 3.5
    amperes = 15000.0
    # Non-planar coils have 108 turns. Planar coils have 36 turns.
    turns = 108
    total_current = amperes * turns * 5

    print('Total current = ', total_current)

    # Create the initial coils
    base_curves = create_equally_spaced_curves(
        ncoils, s.nfp, stellsym=True,
        R0=R0, R1=R1, order=order, numquadpoints=256,
    )
    base_currents = [(Current(total_current / ncoils * 1e-7) * 1e7) for _ in range(ncoils - 1)]
    total_current = Current(total_current)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    a = 0.15
    regularizations = [regularization_circ(a) for _ in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True, regularizations=regularizations)
    curves = [c.curve for c in coils]
    return base_curves, curves, coils, base_currents

results_raw = [[0 for i in range(4)] for j in range(30)]
results_pruned = [[] for _ in range(30)]

# Loop over different runs for each order
for run in range(4):
    print(f"\n{'-'*60}")
    print(f"ORDER {order}, RUN {run+1}")
    print(f"{'-'*60}")

    # Directory for output with Gaussian perturbation
    OUT_DIR = (f"./output_paper/w7x_order{order}_ncoils{ncoils_choice}_run{run+1}_curvature{CURVATURE_THRESHOLD}_" + \
               f"force{FORCE_THRESHOLD}_flux{FLUX_THRESHOLD}_length{LENGTH_TARGET}_" + \
               f"cc{CC_THRESHOLD}_cs{CS_THRESHOLD}_msc{MSC_THRESHOLD}_gaussian_initial_auglag/")
    os.makedirs(OUT_DIR, exist_ok=True)

    ncoils = ncoils_choice
    base_curves_TF, curves_TF, coils_TF, currents_TF = w7x_coils(s, ncoils=ncoils, order=order)
    ncoils = len(base_curves_TF)
    base_curves_TF = curves_TF[:ncoils]
    base_coils_TF = coils_TF[:ncoils]
    
    # Save unperturbed initial coils
    coils_to_vtk(coils_TF, OUT_DIR + f"coils_init_w7x_{order}")

    # Calculate initial magnetic field
    bs = BiotSavart(coils_TF)
    bs.set_points(s.gamma().reshape((-1, 3)))
    Bn = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
    absB = np.linalg.norm(bs.B().reshape(nphi, ntheta, 3), axis=-1)
    pointData = {"B_N": Bn[:, :, None],
                 "B_N / B": (Bn / absB)[:, :, None],
                }
    s.to_vtk(OUT_DIR + f"s_init_w7x_{order}", extra_data=pointData)

    calculate_modB_on_major_radius(bs, s)
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None],
                 "B_N / B": (np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2
                                    ) / np.linalg.norm(bs.B().reshape(qphi, qtheta, 3), axis=-1))[:, :, None]}
    s_plot.to_vtk(OUT_DIR + f"surf_init_w7x_{order}", extra_data=pointData)

    # Set up stochastic optimization with different seed for each run
    seed = order * 100 + run  # Unique seed for each order-run combination
    rg = Generator(PCG64DXSM(seed))

    # Create Gaussian sampler using the first curve's quadpoints
    sampler = GaussianSampler(curves_TF[0].quadpoints, SIGMA, L, n_derivs=2)

    # Generate a single set of perturbed coils
    base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves_TF]
    a = 0.15
    regularizations = [regularization_circ(a) for _ in range(len(base_curves_perturbed))]
    coils_perturbed = coils_via_symmetries(base_curves_perturbed, currents_TF, s.nfp, s.stellsym, regularizations=regularizations)

    # Use perturbed coils
    curves_pert = [c.curve for c in coils_perturbed]
    coils_pert = coils_perturbed
    bs_pert = BiotSavart(coils_perturbed)

    # Save perturbed coils
    coils_to_vtk(coils_pert, OUT_DIR + f"coils_init_pert_{order}")

    # Define the individual terms objective function using perturbed coils:
    bs_pert.set_points(s.gamma().reshape((-1, 3)))
    Jf = SquaredFlux(s, bs_pert, definition="normalized", threshold=FLUX_THRESHOLD)
    base_curves_pert = [c.curve for c in coils_perturbed[:ncoils]]
    curves_pert_all = [c.curve for c in coils_perturbed]
    Jls = [CurveLength(c) for c in base_curves_pert]
    Jl = sum(QuadraticPenalty(jj, LENGTH_TARGET, "max") for jj in Jls)
    Jccdist = CurveCurveDistance(curves_pert_all, CC_THRESHOLD, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves_pert_all, s, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves_pert]
    Jlink = LinkingNumber(curves_pert_all, downsample=2)
    base_coils_pert = coils_perturbed[:ncoils]
    Jforce = LpCurveForce(base_coils_pert, coils_pert, p=2.0, threshold=FORCE_THRESHOLD, downsample=2)
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves_pert]

    force = [np.max(np.linalg.norm(coil_force(c, coils_pert), axis=1)) for c in base_coils_pert]
    print("Forces (perturbed coils):")
    print(",".join(f"{f:.2e}" for f in force))

    # Print initial values using perturbed coils
    print('Initial normalized flux (perturbed coils):', Jf.J())
    print('Initial CS-Sep constraint (perturbed coils):', Jcsdist.J())
    print('Initial CS-sep minimum distance (perturbed coils):', Jcsdist.shortest_distance())
    print('Initial CC-Sep constraint (perturbed coils):', Jccdist.J())
    print('Initial CC-sep minimum distance (perturbed coils):', Jccdist.shortest_distance())
    print('Initial Len constraint (perturbed coils):', Jl.J())
    print('Initial Curv constraint (perturbed coils):', sum(Jcs).J())
    print('Initial Link constraint (perturbed coils):', Jlink.J())
    print('Initial Max Curvatures (perturbed coils):', [np.max(c.kappa()) for c in base_curves_pert])
    print('Initial Max MSC Curvatures (perturbed coils):', [float(J.J()) for J in Jmscs])
    print('Initial Lengths (perturbed coils):', [CurveLength(c).J() for c in base_curves_pert], sum(Jls).J())
    print('Initial Force constraint (perturbed coils):', Jforce.J())

    # Main optimization function
    f = None

    # Constraint list using perturbed coils
    c_list = [Jf, 
              Jccdist, 
              Jcsdist, 
              QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max"),
              sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs),
              sum(Jcs), 
              Jlink,
              Jforce
    ]

    start_time = time.time()
    x, fnc, lag_mul = augmented_lagrangian_method(
        f=f,
        equality_constraints=c_list,
        tau=5,
        MAXITER=MAXITER,
        MAXITER_lag=MAXITER_lag,
        grad_tol=1e-8,
        c_tol=1e-8,
        cs_weight_init=10
    )
    end_time = time.time()

    print(f"Time taken: {end_time - start_time} seconds")
    print('Final normalized flux (perturbed coils):', Jf.J())
    print('Final CS-Sep constraint (perturbed coils):', Jcsdist.J())
    print('Final CS-sep minimum distance (perturbed coils):', Jcsdist.shortest_distance())
    print('Final CC-Sep constraint (perturbed coils):', Jccdist.J())
    print('Final CC-sep minimum distance (perturbed coils):', Jccdist.shortest_distance())
    print('Final Len constraint (perturbed coils):', Jl.J())
    print('Final Curv constraint (perturbed coils):', sum(Jcs).J())
    print('Final Link constraint (perturbed coils):', Jlink.J())
    print('Final Max Curvatures (perturbed coils):', [np.max(c.kappa()) for c in base_curves_pert])
    print('Final Max MSC Curvatures (perturbed coils):', [float(J.J()) for J in Jmscs])
    print('Final Lengths (perturbed coils):', [CurveLength(c).J() for c in base_curves_pert], sum(Jls).J())
    print('Final Force constraint (perturbed coils):', Jforce.J())
    force = [np.max(np.linalg.norm(coil_force(c, coils_pert), axis=1)) for c in base_coils_pert]
    print("Forces (perturbed coils):")
    print(",".join(f"{f:.2e}" for f in force))

    # Save optimized perturbed coils
    coils_to_vtk(coils_pert, OUT_DIR + f"coils_optimized_gaussian_auglag_w7x_{order}")

    # Evaluate optimized coils (perturbed coils)
    bs_pert.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {"B_N": np.sum(bs_pert.B().reshape((qphi, qtheta, 3)) *
                            s_plot.unitnormal(), axis=2)[:, :, None],
            "B_N/|B|": np.sum(bs_pert.B().reshape((qphi, qtheta, 3)) *
                            s_plot.unitnormal(), axis=2)[:, :, None] /
            bs_pert.AbsB().reshape((qphi, qtheta, 1)),
            "modB": bs_pert.AbsB().reshape((qphi, qtheta, 1))}
    s_plot.to_vtk(OUT_DIR + f"surf_optimized_gaussian_auglag_w7x_{order}", extra_data=pointData)

    # Final calculations and output (perturbed coils)
    t2 = time.time()
    max_BdotN_overB = np.max(safe_divide(np.sum(bs_pert.B().reshape((qphi, qtheta, 3)) *
                            s_plot.unitnormal(), axis=2)[:, :, None],
            bs_pert.AbsB().reshape((qphi, qtheta, 1))))
    bs_pert.set_points(s.gamma().reshape((-1, 3)))
    BdotN = np.mean(np.abs(np.sum(bs_pert.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    avg_BdotN_over_B = BdotN / bs_pert.AbsB().mean()
    print("--------------------------------------------------------------------------------------------------------------------------------------------")
    print(f"<B_N>/<|B|> = {avg_BdotN_over_B:.2e}, Max BdotN/|B| = {max_BdotN_overB:.2e}")
    print("FINAL LAGRANGE MULTIPLIERS:", lag_mul)
    print('Total time = ', t2 - t1)
    print("--------------------------------------------------------------------------------------------------------------------------------------------")
    print("Final NORMALIZED SQUARED FLUX:", Jf.J())
    print(f'FINISHED OPTIMIZATION for order={order}, run={run+1}')
    print("Output directory:", OUT_DIR)

    # Save the optimized BiotSavart object directly
    #bs_pert.save(OUT_DIR + f"biot_savart_optimized_gaussian_auglag_w7x_{order}" + ".json")
    
    # Filter out if the average BN/B error is greater than or equal to 0.1 (e-01)
    if (avg_BdotN_over_B < 0.1):
        results_pruned[order - 1].append(f"{avg_BdotN_over_B:.2e}")
    results_raw[order - 1][run] = f"{avg_BdotN_over_B:.2e}"

print("\nRaw data:")
print(results_raw)
print("\nPruned data:")
print(results_pruned)