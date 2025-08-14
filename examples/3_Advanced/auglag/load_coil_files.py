import numpy as np
import os
from pathlib import Path
from simsopt.geo import SurfaceRZFourier, CurveSurfaceDistance
from simsopt import load
from simsopt.field import coils_to_vtk

OUTDIR = './output_paper/w7x_gaussian/'
# Create output directory if it doesn't exist
os.makedirs(OUTDIR, exist_ok=True)
nphi = 128 #256
ntheta = 128 #256
quadpoints_phi = np.linspace(0, 1, nphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
input_name = 'input.W7-X_without_coil_ripple_beta0p05_d23p4_tm'
coilname = './output_paper/w7x_ncoils4_curvature2.5_force3.8_flux1e-15_length38_cc0.28_cs0.3_gaussian_initial_auglag/biot_savart_optimized_gaussian_auglag_w7x.json'
filename = TEST_DIR / input_name 
Bfield = load(coilname)
coils = Bfield.coils
s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
curves = [c.curve for c in coils]
base_curves = curves[:len(curves) // (s.nfp * 2)]
base_coils = coils[:len(coils) // (s.nfp * 2)]
eval_points = s.gamma().reshape(-1, 3)
Bfield.set_points(s.gamma().reshape((-1, 3)))
BdotN = np.mean(np.abs(np.sum(Bfield.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
BdotN_over_B = np.mean(np.abs(np.sum(Bfield.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
                        ) / np.mean(Bfield.AbsB())
print(BdotN, BdotN_over_B)
min_dist = CurveSurfaceDistance(base_curves, s, 0.1).shortest_distance()
print(min_dist)
coils_to_vtk(coils, OUTDIR + "coils_loaded")
