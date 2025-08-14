#!/usr/bin/env python
"""

This script performs coil optimization for the reactor-scale Landreman-Paul 2021 precise QH configuration using the Augmented Lagrangian Method (ALM) with a Gaussian perturbation to generate the initial coil set.

In order to reproduce the coilsets of the paper the following thresholds/parameters should be set:
1) 4 coils solution (#1)
    ncoils : 4
    LENGTH_TARGET = 4*40 
    FLUX_THRESHOLD = 1e-15
    CC_THRESHOLD = 0.8 
    CS_THRESHOLD = 1
    CURVATURE_THRESHOLD = 1 
    MSC_THRESHOLD = 0.1
    FORCE_THRESHOLD = 10 # units of MN/m

2) 5 coils solution (#2)
    ncoils_choice : 6
    LENGTH_TARGET = 5*35.56 
    FLUX_THRESHOLD = 1e-15
    CC_THRESHOLD = 1.1 
    CS_THRESHOLD = 1.6
    CURVATURE_THRESHOLD = 0.88 
    MSC_THRESHOLD = 0.08
    FORCE_THRESHOLD = 10 # units of MN/m

"""

import numpy as np
import os
from numpy.random import PCG64DXSM, Generator
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import SurfaceRZFourier
from simsopt.geo import CurveXYZFourier
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
filename = TEST_DIR / 'input.LandremanPaul2021_QH_reactorScale_lowres'

t1 = time.time()

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
    MAXITER = 5  # 1500 for high-resolution
    MAXITER_lag = 5  # 30 for high-resolution

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

# Define the upper and bounds for the constraints
LENGTH_TARGET = 4 * 40  # 4 coils, 40 m per coil
FLUX_THRESHOLD = 1e-15
CC_THRESHOLD = 0.8
CS_THRESHOLD = 1 
CURVATURE_THRESHOLD = 1 
MSC_THRESHOLD = 0.1 
FORCE_THRESHOLD = 10 # units of MN/m

# Parameters for Gaussian Perturbation
SIGMA = 1e-3  # Standard deviation σ for coil errors
L = 0.5  # Length scale L for coil errors

# Define the number of coils, rotation order, and non-planar base curves
R0 = s.x[0]
R1 = 0.5 * s.x[0]
order = 7
ncoils = 4
curves = create_equally_spaced_curves(
    ncoils, s.nfp, stellsym=s.stellsym, R0=R0, R1=R1, order=order, numquadpoints=128)
total_current = 45642162
base_currents = [Current(total_current / ncoils * 1e-7) * 1e7 for _ in range(ncoils)]
base_currents[0].fix_all()

# Define the output directory   
OUT_DIR = (f"./output_paper/qh_ncoils{ncoils}_curvature{CURVATURE_THRESHOLD}_msc{MSC_THRESHOLD}_" + \
           f"force{FORCE_THRESHOLD}_flux{FLUX_THRESHOLD}_length{LENGTH_TARGET}_" + \
           f"cc{CC_THRESHOLD}_cs{CS_THRESHOLD}_gaussian_initial_auglag/")
os.makedirs(OUT_DIR, exist_ok=True)

base_curves = curves[:ncoils]
a = 0.15  # radius of the coil
regularizations = [regularization_circ(a) for _ in range(ncoils)]
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym, regularizations=regularizations)
base_coils = coils[:ncoils]
curves = [c.curve for c in coils]
currents = [c.current for c in coils]
print("Number of coils:", len(coils))

# Save the biot-savart field data for unperturbed coils
bs = BiotSavart(coils)
curves = [c.curve for c in coils]
coils_to_vtk(coils, OUT_DIR + "coils_init_qh_reactorscale")
bs.set_points(s_plot.gamma().reshape((-1, 3))) 
pointData = {"B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                               s_plot.unitnormal(), axis=2)[:, :, None] / bs.AbsB().reshape((qphi, qtheta, 1)),
             "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR + "surf_init_qh_reactorscale", extra_data=pointData)

# Set up stochastic optimization
seed = 0
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
print('Initial Lengths (perturbed coils):', [CurveLength(c).J() for c in base_curves_pert], sum(Jls).J())

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
        #   Jforce
]

start_time = time.time()
x, fnc, lag_mul = augmented_lagrangian_method(f=None,
    equality_constraints=c_list,
    tau=10,
    MAXITER=MAXITER,
    MAXITER_lag=MAXITER_lag,
    grad_tol=1e-8,
    c_tol=1e-8,
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
print('Final Lengths (perturbed coils):', [CurveLength(c).J() for c in base_curves_pert], sum(Jls).J())
print('Final Mean Squared Curvature (perturbed coils):', [MeanSquaredCurvature(c).J() for c in base_curves_pert])
# print('Final Force constraint (perturbed coils):', Jforce.J())
force = [np.max(np.linalg.norm(coil_force(c, coils_pert), axis=1)) for c in base_coils_pert]
print("Forces (perturbed coils):")
print(",".join(f"{f:.2e}" for f in force))

# Save optimized perturbed coils
coils_to_vtk(coils_pert, OUT_DIR + "coils_optimized_gaussian_auglag_qh_reactorscale")

# Evaluate optimized coils (perturbed coils)
bs_pert.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs_pert.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None],
        "B_N/|B|": np.sum(bs_pert.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None] /
        bs_pert.AbsB().reshape((qphi, qtheta, 1)),
        "modB": bs_pert.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR + "surf_optimized_gaussian_auglag_qh_reactorscale", extra_data=pointData)


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
print('Total time = ', t2 - t1)



# Was having issues saving JSON file with BiotSavart, so created completely new curves from the perturbed ones to avoid any random generator references 
# Extract the optimized parameters from perturbed curves and create cleaned curves
base_curves_cleaned = []
for curve_pert in base_curves_perturbed:
    # Get the curve properties
    underlying_curve = curve_pert.curve
    dofs = underlying_curve.get_dofs()
    quadpoints = underlying_curve.quadpoints
    order = underlying_curve.order
    
    # Create a new curve with the same parameters (no random generator references)
    curve_cleaned = CurveXYZFourier(quadpoints, order)
    curve_cleaned.set_dofs(dofs)
    
    base_curves_cleaned.append(curve_cleaned)

# Create cleaned coils and BiotSavart object
coils_cleaned = coils_via_symmetries(base_curves_cleaned, base_currents, s.nfp, s.stellsym, regularizations=regularizations)
bs_cleaned = BiotSavart(coils_cleaned)

# Now save the cleaned BiotSavart object
bs_cleaned.save(OUT_DIR + "biot_savart_optimized_gaussian_auglag_qh_reactorscale.json")