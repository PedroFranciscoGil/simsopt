#!/usr/bin/env python
"""
auglag_hsx_Gaussian.py
===============
This script performs coil optimization for the HSX coil design sing the Augmented Lagrangian Method (ALM) with Gaussian perturbations to generate the initial coil set.

In order to reproduce the coilsets of the paper the following thresholds/parameters should be set:
1) 4 coils solution
    ncoils : 4
    LENGTH_TARGET = 14.5/6*ncoils_choice 
    FLUX_THRESHOLD = 1e-15
    CC_THRESHOLD = 0.1
    CS_THRESHOLD = 0.14 
    CURVATURE_THRESHOLD = 10
    MSC_THRESHOLD = 25
    (no force active)

2) 5 coils solution
    ncoils : 5
    LENGTH_TARGET = 14.5/6*ncoils_choice 
    FLUX_THRESHOLD = 1e-15
    CC_THRESHOLD = 0.1
    CS_THRESHOLD = 0.14 
    CURVATURE_THRESHOLD = 12
    MSC_THRESHOLD = 30
    FORCE_THRESHOLD = 0.12 # units of MN/m

3) 6 coils solution
    ncoils : 6
    LENGTH_TARGET = 14.5/6*ncoils_choice 
    FLUX_THRESHOLD = 1e-15
    CC_THRESHOLD = 0.1
    CS_THRESHOLD = 0.14 
    CURVATURE_THRESHOLD = 8
    MSC_THRESHOLD = 30
    (no force active)

"""

import numpy as np
import os
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import SurfaceRZFourier
from simsopt.geo import create_equally_spaced_curves
from simsopt.geo import LinkingNumber
from simsopt.geo import CurveLength, CurveCurveDistance, \
    LpCurveCurvature, CurveSurfaceDistance, MeanSquaredCurvature, \
    GaussianSampler, CurvePerturbed, PerturbationSample
from simsopt.solve import augmented_lagrangian_method
from simsopt.field import BiotSavart, coils_to_vtk
from simsopt.field.force import LpCurveForce, coil_force
from simsopt.field import Current, coils_via_symmetries, regularization_circ
from pathlib import Path
from simsopt.util import in_github_actions
from numpy.random import PCG64DXSM, Generator
import time

ncoils_choice = 5
# Define the upper and lower bounds for the constraints
LENGTH_TARGET = 14.5 / 6 * ncoils_choice 
FLUX_THRESHOLD = 1e-15
CC_THRESHOLD = 0.1
CS_THRESHOLD = 0.14
CURVATURE_THRESHOLD = 12
MSC_THRESHOLD = 30
FORCE_THRESHOLD = 0.12 # units of MN/m

# Parameters for Gaussian Perturbation
SIGMA = 1e-1  # Standard deviation σ for coil errors
L = 0.1  # Length scale L for coil errors

# Define the output directory   
OUT_DIR = (f"./output_paper/hsx_ncoils{ncoils_choice}_curvature{CURVATURE_THRESHOLD}_" + \
           f"force{FORCE_THRESHOLD}_flux{FLUX_THRESHOLD}_length{LENGTH_TARGET}_" + \
           f"cc{CC_THRESHOLD}_cs{CS_THRESHOLD}_gaussian_initial_auglag/")
os.makedirs(OUT_DIR, exist_ok=True)

# Define the test directory
TEST_DIR = Path(__file__).parent / '../' / '../' / '../' / 'tests/test_files'

# Define the filename
filename = TEST_DIR / 'input.QHS_mn1824_ns101' #'input.HSX_QHS_vacuum_ns201''input.hsxt'  

# Set some parameters -- warning this is super low resolution!
if in_github_actions:
    nphi = 4
    ntheta = 4
    MAXITER = 5
    MAXITER_lag = 5
else:
    # Define the number of phi and theta points
    nphi = 32  # 64 for high-resolution
    ntheta = 32  # 64 for high-resolution
    MAXITER = 5  # 1500 for high-resolution
    MAXITER_lag = 5  # 50 for high-resolution

# Define the surface
s = SurfaceRZFourier.from_vmec_input(
    filename,
    range="half period",
    nphi=nphi,
    ntheta=ntheta)

qphi = 4 * nphi
qtheta = 4 * ntheta
quadpoints_phi = np.linspace(0, 1, qphi)
quadpoints_theta = np.linspace(0, 1, qtheta)
s_plot = SurfaceRZFourier.from_vmec_input(
    filename,
    range="full torus",
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta)

# Define the number of coils, rotation order, and non-planar base curves
ncoils = ncoils_choice

def hsx_coils(s, ncoils=3, order=8):

    # parameters for the TF coils, increase order for a better solution
    # Total current scaled to give B ~ 5.7 T on axis (actually averaged over the major radius)
    R0 = s.get_rc(0, 0) * 1
    R1 = s.get_rc(1, 0) * 2.5
 
    total_current = sum([1.500725500000000e+05, 1.500725500000000e+05, 
                         1.500725500000000e+05, 1.500725500000000e+05, 
                         1.500725500000000e+05, 1.500725500000000e+05])
    print('Total current = ', total_current)

    # Create the initial coils
    base_curves = create_equally_spaced_curves(
        ncoils, s.nfp, stellsym=True,
        R0=R0, R1=R1, order=order, numquadpoints=256,
    ) 
    
    base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils - 1)]
    total_current = Current(total_current)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    regularizations = [regularization_circ(0.05) for _ in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True, regularizations=regularizations)
    curves = [c.curve for c in coils]
    return base_curves, curves, coils, base_currents

base_curves, curves, coils, base_currents = hsx_coils(s, ncoils=ncoils, order=7)
# Above, the factors of 1e-5 and 1e5 are included so the current
# degrees of freedom are O(1) rather than ~ MA.  The optimization
# algorithm may not perform well if the dofs are scaled badly.

base_coils = coils[:ncoils]
currents = [c.current for c in coils]
print("Number of coils:", len(coils))

# Save the biot-savart field data for unperturbed coils
bs = BiotSavart(coils)
curves = [c.curve for c in coils]
coils_to_vtk(coils, OUT_DIR + "coils_init_hsx")
bs.set_points(s_plot.gamma().reshape((-1, 3))) 
pointData = {"B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                               s_plot.unitnormal(), axis=2)[:, :, None] / bs.AbsB().reshape((qphi, qtheta, 1)),
             "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR + "surf_init_hsx", extra_data=pointData)

# Set up stochastic optimization
seed = 0
rg = Generator(PCG64DXSM(seed))

# Create Gaussian sampler using the first curve's quadpoints
sampler = GaussianSampler(curves[0].quadpoints, SIGMA, L, n_derivs=2)

# Generate perturbed coils by applying Gaussian perturbations to the base curves
base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
regularizations = [regularization_circ(0.05) for _ in range(len(base_curves_perturbed))]
coils_perturbed = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True, regularizations=regularizations)

# Use perturbed coils for optimization
curves_pert = [c.curve for c in coils_perturbed]
coils_pert = coils_perturbed
print("Number of perturbed coils:", len(coils_perturbed))

# Save perturbed initial coils
coils_to_vtk(coils_pert, OUT_DIR + "coils_init_pert")

# Calculate magnetic field using perturbed coils
bs_pert = BiotSavart(coils_pert)
bs_pert.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N/|B|": np.sum(bs_pert.B().reshape((qphi, qtheta, 3)) *
                               s_plot.unitnormal(), axis=2)[:, :, None] / bs_pert.AbsB().reshape((qphi, qtheta, 1)),
             "modB": bs_pert.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR + "surf_init_pert", extra_data=pointData)

# Define the individual terms objective function using perturbed coils:
bs_pert.set_points(s.gamma().reshape((-1, 3)))
Jf = SquaredFlux(s, bs_pert, definition="normalized", threshold=FLUX_THRESHOLD)
Jls = [CurveLength(c) for c in base_curves_perturbed]
Jl = sum(QuadraticPenalty(jj, LENGTH_TARGET, "max") for jj in Jls)
Jccdist = CurveCurveDistance(curves_pert, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves_pert, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves_perturbed]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves_perturbed]
Jlink = LinkingNumber(curves_pert, downsample=2)
Jforce = LpCurveForce(base_coils, coils_pert, p=2.0, threshold=FORCE_THRESHOLD)

force = [np.max(np.linalg.norm(coil_force(c, coils_pert), axis=1)) for c in base_coils]
print("Forces:")
print(",".join(f"{f:.2e}" for f in force))

# Main optimization function
f = None

# Constraint list using perturbed coils
c_list = [ Jf,
        Jccdist, 
        Jcsdist, 
        QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max"), 
        sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs),
        sum(Jcs), 
        Jlink,
        Jforce
]

print('Initial normalized flux (perturbed coils):', Jf.J())
print('Initial CS-Sep constraint (perturbed coils):', Jcsdist.J())
print('Initial CS-sep minimum distance (perturbed coils):', Jcsdist.shortest_distance())
print('Initial CC-Sep constraint (perturbed coils):', Jccdist.J())
print('Initial CC-sep minimum distance (perturbed coils):', Jccdist.shortest_distance())
print('Initial Len constraint (perturbed coils):', Jl.J())
print('Initial Curv constraint (perturbed coils):', sum(Jcs).J())
print('Initial Link constraint (perturbed coils):', Jlink.J())
print('Initial Max Curvatures (perturbed coils):', [np.max(c.kappa()) for c in base_curves_perturbed])
print('Initial Lengths (perturbed coils):', [CurveLength(c).J() for c in base_curves_perturbed], sum(Jls).J())

start_time = time.time()
x, fnc, lag_mul = augmented_lagrangian_method(f=f,
    equality_constraints=c_list,
    tau=6, #6 worked with cc 0.1 and cs 0.1 and l 15
    MAXITER=MAXITER,
    MAXITER_lag=MAXITER_lag,
    grad_tol=1e-8,
    c_tol=1e-8,
)

bs_pert.save(OUT_DIR + "biot_savart_optimized_gaussian_auglag_hsx.json")
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
print('Final Max Curvatures (perturbed coils):', [np.max(c.kappa()) for c in base_curves_perturbed])
print('Final Lengths (perturbed coils):', [CurveLength(c).J() for c in base_curves_perturbed], sum(Jls).J())
print('Final Mean Squared Curvature (perturbed coils):', [MeanSquaredCurvature(c).J() for c in base_curves_perturbed])
# print('Final Force constraint (perturbed coils):', Jforce.J())
force = [np.max(np.linalg.norm(coil_force(c, coils_pert), axis=1)) for c in base_coils]
print("Forces:")
print(",".join(f"{f:.2e}" for f in force))
coils_to_vtk(coils_pert, OUT_DIR + "coils_optimized_gaussian_auglag_hsx")
bs_pert.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs_pert.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None],
        "B_N/|B|": np.sum(bs_pert.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None] /
        bs_pert.AbsB().reshape((qphi, qtheta, 1)),
        "modB": bs_pert.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR + "surf_optimized_gaussian_auglag_hsx", extra_data=pointData)
max_BdotN_overB = np.max(np.sum(bs_pert.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None] /
        bs_pert.AbsB().reshape((qphi, qtheta, 1)))
bs_pert.set_points(s.gamma().reshape((-1, 3)))
BdotN = np.mean(np.abs(np.sum(bs_pert.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
avg_BdotN_over_B = BdotN / bs_pert.AbsB().mean()
print("--------------------------------------------------------------------------------------------------------------------------------------------")
print(f"<B_N>/<|B|> = {avg_BdotN_over_B:.2e}, Max BdotN/|B| = {max_BdotN_overB:.2e}")
print("FINAL LAGRANGE MULTIPLIERS:", lag_mul)
print("--------------------------------------------------------------------------------------------------------------------------------------------")
print("Final NORMALIZED SQUARED FLUX:", Jf.J())
print('FINISHED OPTIMIZATION')