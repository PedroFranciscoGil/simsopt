#!/usr/bin/env python
"""
auglag_qa_reactorscale_Gaussian.py
===============
This script performs coil optimization for the reactor-scale Landreman-Paul 2021 precise QA configuration using the Augmented Lagrangian Method (ALM) with a Gaussian perturbation to generate the initial coil set.

In order to reproduce the coilsets of the paper the following thresholds/parameters should be set:
1) 3 coils solution
    ncoils : 3
    LENGTH_TARGET = 182.2
    FLUX_THRESHOLD = 1e-15
    CC_THRESHOLD = 1.0
    CS_THRESHOLD = 1.5
    CURVATURE_THRESHOLD = 0.5
    MSC_THRESHOLD = 0.05
    FORCE_THRESHOLD = 1.1e2

2) 4 coils solution
    ncoils : 4
    LENGTH_TARGET = 182.2
    FLUX_THRESHOLD = 1e-15
    CC_THRESHOLD = 1.0
    CS_THRESHOLD = 1.5
    CURVATURE_THRESHOLD = 0.5
    MSC_THRESHOLD = 0.05
    FORCE_THRESHOLD = 1.1e2
"""

import numpy as np
import os
import sys
from numpy.random import PCG64DXSM, Generator
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import SurfaceRZFourier
from simsopt.geo import LinkingNumber
from simsopt.geo import CurveLength, CurveCurveDistance, \
    LpCurveCurvature, CurveSurfaceDistance, MeanSquaredCurvature, \
    GaussianSampler, CurvePerturbed, PerturbationSample
from simsopt.solve.augmented_lagrangian_initial_weight import augmented_lagrangian_method
from simsopt.field import BiotSavart, coils_to_vtk
from simsopt.field.force import LpCurveForce, coil_force
from simsopt.field import coils_via_symmetries, regularization_circ
from pathlib import Path
from simsopt.util import in_github_actions, initialize_coils_simple
import time

def safe_divide(numerator, denominator, epsilon=1e-12):
    """
    Safely divide numerator by denominator, avoiding division by zero.
    Returns 0.0 where denominator is too small.
    """
    return np.where(np.abs(denominator) > epsilon, numerator / denominator, 0.0)

# Define the test directory
TEST_DIR = Path(__file__).parent / '../' / '../' / '../' / 'tests/test_files'

# Define the filename
filename = TEST_DIR / 'input.LandremanPaul2021_QA_reactorscale_lowres'

t1 = time.time()

# Set some parameters -- warning this is super low resolution!
if in_github_actions:
    nphi = 4
    ntheta = 4
    MAXITER = 5
    MAXITER_lag = 5
else:
    # Define the number of phi and theta points
    nphi = 32
    ntheta = 32
    MAXITER = 1000 # 1000 for high-resolution
    MAXITER_lag = 50  # 50 for high-resolution

# Define the surface
s = SurfaceRZFourier.from_vmec_input(
    filename,
    range="half period",
    nphi=nphi,
    ntheta=ntheta)

# Define a high-resolution, full-torus surface for plotting
qphi = 4 * nphi
qtheta = 4 * ntheta
quadpoints_phi = np.linspace(0, 1, qphi)
quadpoints_theta = np.linspace(0, 1, qtheta)
s_plot = SurfaceRZFourier.from_vmec_input(
    filename,
    range="full torus",
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta)

# Define the upper and lower bounds for the constraints
LENGTH_TARGET = 182.2  # 182.2 m is smallest of the Wechsung et al. 2021 coils
FLUX_THRESHOLD = 1e-15
CC_THRESHOLD = 1.0
CS_THRESHOLD = 1.5
MSC_THRESHOLD = 0.05
CURVATURE_THRESHOLD = 0.5
FORCE_THRESHOLD = 1.1e2  # Stay within ~ 0.57 MN/m assuming 200 turns of coil

# Parameters for Gaussian Perturbation
R0 = s.get_rc(0, 0)
SIGMA = 0.01  # Standard deviation σ for coil errors
L = 0.5  # Length scale L for coil errors

# Define the number of coils, rotation order, and non-planar base curves
ncoils = 3
a = 0.15  # radius of the coil

# Get order from command line argument (SLURM array index + 1)
if len(sys.argv) > 1:
    order = int(sys.argv[1])
else:
    # Default to order 1 
    order = 1

print(f"Running with order = {order}")

# Single order run (no nested loop)
# Loop over different runs for this order
for run in range(4):
    print(f"\n{'-'*60}")
    print(f"ORDER {order}, RUN {run+1}")
    print(f"{'-'*60}")
    
    # Define the output directory for this specific order and run
    OUT_DIR = (f"./output_paper/qa_ncoils{ncoils}_order{order}_run{run+1}_curvature{CURVATURE_THRESHOLD}_msc{MSC_THRESHOLD}_" + \
           f"force{FORCE_THRESHOLD}_flux{FLUX_THRESHOLD}_length{LENGTH_TARGET}_" + \
           f"cc{CC_THRESHOLD}_cs{CS_THRESHOLD}_gaussian_initial_auglag/")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Create initial coils using initialize_coils_simple (same as original QA file)
    coils = initialize_coils_simple(s, ncoils=ncoils, order=order, regularization=regularization_circ(a))
    base_coils = coils[:ncoils]
    curves = [c.curve for c in coils]
    base_curves = curves[:ncoils]
    currents = [c.current for c in coils]

    # Extract base currents for use in perturbed coils
    base_currents = [c.current for c in base_coils]
    print("Number of coils:", len(coils))

    # Save the biot-savart field data for unperturbed coils
    bs = BiotSavart(coils)
    curves = [c.curve for c in coils]
    coils_to_vtk(coils, OUT_DIR + "coils_init_qa_reactorscale")
    bs.set_points(s_plot.gamma().reshape((-1, 3))) 
    pointData = {"B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                       s_plot.unitnormal(), axis=2)[:, :, None] / bs.AbsB().reshape((qphi, qtheta, 1)),
             "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
    s_plot.to_vtk(OUT_DIR + "surf_init_qa_reactorscale", extra_data=pointData)

    # Set up stochastic optimization with different seed for each run
    seed = order * 100 + run  # Unique seed for each order-run combination
    rg = Generator(PCG64DXSM(seed))

    # Create Gaussian sampler using the first curve's quadpoints
    sampler = GaussianSampler(curves[0].quadpoints, SIGMA, L, n_derivs=2)

    # Generate a single set of perturbed coils  
    # First add the 'systematic' error. Repeated regularization to ensure saving force data.
    base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
    regularizations = [regularization_circ(a) for _ in range(len(base_curves_perturbed))]
    coils_perturbed = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, s.stellsym, regularizations=regularizations)

    # Use perturbed coils
    curves_pert = [c.curve for c in coils_perturbed]
    coils_pert = coils_perturbed
    bs_pert = BiotSavart(coils_perturbed)

    # Save perturbed coils
    coils_to_vtk(coils_pert, OUT_DIR + "coils_init_pert")

    # Define the individual terms of objective function using perturbed coils:
    bs_pert.set_points(s.gamma().reshape((-1, 3)))
    Jf = SquaredFlux(s, bs_pert, definition="normalized", threshold=FLUX_THRESHOLD)
    base_curves_pert = [c.curve for c in coils_perturbed[:ncoils]] 
    curves_pert_all = [c.curve for c in coils_perturbed]  
    Jls = [CurveLength(c) for c in base_curves_pert]
    Jl = sum(QuadraticPenalty(jj, LENGTH_TARGET, "max") for jj in Jls)
    Jccdist = CurveCurveDistance(curves_pert_all, CC_THRESHOLD, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves_pert_all, s, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves_pert]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves_pert]
    Jlink = LinkingNumber(curves_pert_all, downsample=2)
    base_coils_pert = coils_perturbed[:ncoils]
    Jforce = LpCurveForce(base_coils_pert, coils_pert, p=2.0, threshold=FORCE_THRESHOLD)

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
    c_list = [ Jf,
        Jccdist, 
        Jcsdist, 
        #    Jl,            # Length constraint commented out 
        sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs),
        QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max"), 
        sum(Jcs), 
        Jlink,
        Jforce
    ]

    start_time = time.time()

    # For the 25m Long coils Pareto front, the parameters for optimization were:
    #       tau: 2    
    #       MAXITER: 1500
    #       MAXITER_lag: 50
    #       grad_tol: 1e-8
    #       c_tol: 1e-8
    # For the 30m Long coils Pareto front, the parameters for optimization were:
    #       tau: 4, 5 and 6 (4 for 14 kN/m, 5 for 12, 11 and 9 kN/m, 6 for 9.5 kN/m)
    #       MAXITER: 1500 and 2000 (2000 for 9.5 kN/m)
    #       grad_tol: 1e-8
    #       c_tol: 1e-8
    ## The pareto front for 30m long coils seemed to be much more sensitive to the tau parameter than the 25. 
    # between 1500 and 2000 MAXITER doesn't actually affect the optimization. 

    x, fnc, lag_mul = augmented_lagrangian_method(f=f,
        equality_constraints=c_list,
        tau=10, #4 for 14, 5 for 12, 5 for 11, 5 for 10, 6 for 9.5, 5 for 9
        MAXITER=MAXITER, #1500 for all results except for 9.5 and 9
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
    coils_to_vtk(coils_pert, OUT_DIR + "coils_optimized_gaussian_auglag_qa_reactorscale")

    # Evaluate optimized coils (perturbed coils)
    bs_pert.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {"B_N": np.sum(bs_pert.B().reshape((qphi, qtheta, 3)) *
                s_plot.unitnormal(), axis=2)[:, :, None],
        "B_N/|B|": np.sum(bs_pert.B().reshape((qphi, qtheta, 3)) *
                s_plot.unitnormal(), axis=2)[:, :, None] /
        bs_pert.AbsB().reshape((qphi, qtheta, 1)),
        "modB": bs_pert.AbsB().reshape((qphi, qtheta, 1))}
    s_plot.to_vtk(OUT_DIR + "surf_optimized_gaussian_auglag_qa_reactorscale", extra_data=pointData)

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
    print("--------------------------------------------------------------------------------------------------------------------------------------------")
    print("Final NORMALIZED SQUARED FLUX:", Jf.J())
    print('FINISHED OPTIMIZATION')
    print("Output directory:", OUT_DIR)

    # Save the optimized BiotSavart object directly (CurvePerturbed objects are now serializable)
    #bs_pert.save(OUT_DIR + "biot_savart_optimized_gaussian_auglag_qa_reactorscale.json")
    
    # Log results for this run
    print(f"ORDER {order}, RUN {run+1} COMPLETED:")
    print(f"  Final avg BN/B error: {avg_BdotN_over_B:.2e}")
    if avg_BdotN_over_B < 0.1:
        print("  Result accepted (< 0.1)")
    else:
        print("  Result rejected (>= 0.1)")

print(f"All runs completed for order {order}")
    