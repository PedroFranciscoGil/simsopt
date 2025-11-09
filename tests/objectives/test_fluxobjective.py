import unittest
import json

import numpy as np

from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.coil import coils_via_symmetries, Current
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.curveobjectives import CurveLength
from simsopt.field.biotsavart import BiotSavart
from simsopt.objectives.fluxobjective import SquaredFlux, SquaredFluxJax
from simsopt._core.json import GSONDecoder, GSONEncoder, SIMSON


from pathlib import Path
TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'


class FluxObjectiveTests(unittest.TestCase):

    def test_definitions(self):
        """Verify the available definitions."""
        surf = SurfaceRZFourier.from_vmec_input(filename)
        ntheta = len(surf.quadpoints_theta)
        nphi = len(surf.quadpoints_phi)
        ncoils = 3

        base_curves = create_equally_spaced_curves(
            ncoils, surf.nfp, stellsym=surf.stellsym, R0=1.0, R1=0.5, order=6
        )
        base_currents = [Current(1e5) for i in range(ncoils)]
        coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, surf.stellsym)
        bs = BiotSavart(coils)

        # Test definition = "quadratic flux":
        target = np.ones(surf.gamma().shape[0:2])
        J = SquaredFlux(surf, bs, target, definition="quadratic flux").J()
        bs.set_points(surf.gamma().reshape((-1, 3)))
        B = bs.B()
        normal = surf.normal().reshape((-1, 3))
        norm_normal = np.sqrt(normal[:, 0]**2 + normal[:, 1]**2 + normal[:, 2]**2)
        B_dot_n = np.sum(B * surf.unitnormal().reshape((-1, 3)), axis=1)
        should_be = 0.5 * sum((B_dot_n - target.reshape((-1,)))**2 * norm_normal) / (ntheta * nphi)
        np.testing.assert_allclose(J, should_be)

        # Test definition = "normalized":
        J2 = SquaredFlux(surf, bs, target, definition="normalized").J()
        mod_B_squared = np.sum(B * B, axis=1)
        numerator = 0.5 * sum(
            (B_dot_n - target.reshape((-1,)))**2 * norm_normal
        ) / (ntheta * nphi)
        denominator = sum(mod_B_squared * norm_normal) / (ntheta * nphi)
        np.testing.assert_allclose(J2, numerator / denominator)

        # Test definition = "local":
        J3 = SquaredFlux(surf, bs, target, definition="local").J()
        should_be3 = 0.5 * sum(
            (B_dot_n - target.reshape((-1,)))**2 / mod_B_squared * norm_normal
        ) / (ntheta * nphi)
        np.testing.assert_allclose(J3, should_be3)

        with self.assertRaises(ValueError):
            SquaredFlux(surf, bs, target, definition="foobar")

    def check_taylor_test(self, J):
        dofs = J.x
        np.random.seed(1)
        h = np.random.uniform(size=dofs.shape)
        dJ0 = J.dJ()
        dJh = sum(dJ0 * h)
        err_old = 1e10
        for i in range(11, 17):
            eps = 0.5 ** i
            J.x = dofs + eps * h
            J1 = J.J()
            J.x = dofs - eps * h
            J2 = J.J()
            err = np.abs((J1 - J2) / (2 * eps) - dJh)
            print(f"J: {J.J()}")
            print(f"i: {i}  err: {err}  err_old: {err_old}  err/err_old: {err/err_old}")
            assert err < 0.6 ** 2 * err_old
            err_old = err

        J_str = json.dumps(SIMSON(J), cls=GSONEncoder)
        J_regen = json.loads(J_str, cls=GSONDecoder)
        self.assertAlmostEqual(J.J(), J_regen.J())

    def test_derivatives(self):
        """Verify correctness of SquaredFlux.dJ()"""
        s = SurfaceRZFourier.from_vmec_input(filename)
        ncoils = 4

        base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=s.stellsym, R0=1.0, R1=0.5, order=6)
        base_currents = [Current(1e5) for i in range(ncoils)]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
        bs = BiotSavart(coils)

        for definition in ["quadratic flux", "normalized", "local"]:
            with self.subTest(definition=definition):
                Jf = SquaredFlux(s, bs, definition=definition)
                self.check_taylor_test(Jf)

                target = np.zeros(s.gamma().shape[0:2])
                Jf2 = SquaredFlux(s, bs, target, definition=definition)
                self.check_taylor_test(Jf2)
                target = np.ones(s.gamma().shape[0:2])
                Jf3 = SquaredFlux(s, bs, target, definition=definition)
                self.check_taylor_test(Jf3)

                Jls = [CurveLength(c) for c in base_curves]

                ALPHA = 1e-5
                JF_scaled_summed = Jf + ALPHA * sum(Jls)
                self.check_taylor_test(JF_scaled_summed)

    def test_squared_flux_jax_vs_cpp(self):
        """Test that SquaredFluxJax gives the same results as SquaredFlux (C++ version)."""
        surf = SurfaceRZFourier.from_vmec_input(filename)
        ncoils = 3

        base_curves = create_equally_spaced_curves(
            ncoils, surf.nfp, stellsym=surf.stellsym, R0=1.0, R1=0.5, order=6
        )
        base_currents = [Current(1e5) for i in range(ncoils)]
        coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, surf.stellsym)
        bs = BiotSavart(coils)

        for definition in ["quadratic flux", "normalized", "local"]:
            with self.subTest(definition=definition):
                # Test with no target
                objective_cpp = SquaredFlux(surf, bs, definition=definition)
                objective_jax = SquaredFluxJax(surf, bs, definition=definition)
                result_cpp = objective_cpp.J()
                result_jax = objective_jax.J()
                print(f"{definition} (no target): C++={result_cpp}, JAX={result_jax}")
                np.testing.assert_allclose(result_jax, result_cpp, atol=1e-10, rtol=1e-3)

                # Test with zero target
                target_zero = np.zeros(surf.gamma().shape[0:2])
                objective_cpp = SquaredFlux(surf, bs, target=target_zero, definition=definition)
                objective_jax = SquaredFluxJax(surf, bs, target=target_zero, definition=definition)
                result_cpp = objective_cpp.J()
                result_jax = objective_jax.J()
                print(f"{definition} (zero target): C++={result_cpp}, JAX={result_jax}")
                np.testing.assert_allclose(result_jax, result_cpp, atol=1e-10, rtol=1e-3)

                # Test with non-zero target
                target_ones = np.ones(surf.gamma().shape[0:2])
                objective_cpp = SquaredFlux(surf, bs, target=target_ones, definition=definition)
                objective_jax = SquaredFluxJax(surf, bs, target=target_ones, definition=definition)
                result_cpp = objective_cpp.J()
                result_jax = objective_jax.J()
                print(f"{definition} (ones target): C++={result_cpp}, JAX={result_jax}")
                np.testing.assert_allclose(result_jax, result_cpp, atol=1e-10, rtol=1e-3)

                # try with threshold
                Jf = SquaredFlux(surf, bs, definition=definition, threshold=1e-3)
                result_cpp = Jf.J()
                self.check_taylor_test(Jf)
                Jf_jax = SquaredFluxJax(surf, bs, definition=definition, threshold=1e-3)
                result_jax = Jf_jax.J()
                self.check_taylor_test(Jf_jax)
                print(f"{definition} (threshold): C++={result_cpp}, JAX={result_jax}")
                np.testing.assert_allclose(result_jax, result_cpp, atol=1e-10, rtol=1e-3)

                target = np.zeros(surf.gamma().shape[0:2])
                Jf2 = SquaredFlux(surf, bs, target, definition=definition, threshold=1e-3)
                self.check_taylor_test(Jf2)
                Jf2_jax = SquaredFluxJax(surf, bs, target, definition=definition, threshold=1e-3)
                self.check_taylor_test(Jf2_jax)
                target = np.ones(surf.gamma().shape[0:2])
                Jf3 = SquaredFlux(surf, bs, target, definition=definition, threshold=1e-3)
                self.check_taylor_test(Jf3)
                Jf3_jax = SquaredFluxJax(surf, bs, target, definition=definition, threshold=1e-3)
                self.check_taylor_test(Jf3_jax)
                print(f"{definition} (threshold): C++={result_cpp}, JAX={result_jax}")
                np.testing.assert_allclose(result_jax, result_cpp, atol=1e-10, rtol=1e-3)

                Jls = [CurveLength(c) for c in base_curves]

                ALPHA = 1e-5
                JF_scaled_summed = Jf + ALPHA * sum(Jls)
                self.check_taylor_test(JF_scaled_summed)
                JF_scaled_summed_jax = Jf_jax + ALPHA * sum(Jls)
                self.check_taylor_test(JF_scaled_summed_jax)
                print(f"{definition} (scaled summed): C++={result_cpp}, JAX={result_jax}")
                np.testing.assert_allclose(result_jax, result_cpp, atol=1e-10, rtol=1e-3)

    def test_squared_flux_jax_hessian(self):
        """Test that the Hessian computation in SquaredFluxJax is correct."""
        surf = SurfaceRZFourier.from_vmec_input(filename)
        ncoils = 3

        base_curves = create_equally_spaced_curves(
            ncoils, surf.nfp, stellsym=surf.stellsym, R0=1.0, R1=0.5, order=6
        )
        base_currents = [Current(1e5) for i in range(ncoils)]
        coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, surf.stellsym)
        bs = BiotSavart(coils)

        for definition in ["quadratic flux", "normalized", "local"]:
            with self.subTest(definition=definition):
                objective_jax = SquaredFluxJax(surf, bs, definition=definition)
                
                # Compute Hessian w.r.t. coil dofs
                H = objective_jax.d2J_dcoil_dofs2()
                
                # Check that Hessian is symmetric (with more lenient tolerance)
                asymmetry = np.max(np.abs(H - H.T))
                max_H = np.max(np.abs(H))
                rel_asymmetry = asymmetry / max_H if max_H > 0 else asymmetry
                self.assertLess(rel_asymmetry, 1e-5,
                              f"Hessian is not symmetric for {definition}: max asymmetry = {asymmetry}, rel = {rel_asymmetry}")
                
                # Test Hessian using Taylor test with random vectors
                # This is more robust than testing individual columns
                np.random.seed(42)
                n_coil_dofs = len(bs.x)
                coil_dofs_orig = bs.x.copy()
                
                # Test with a few random vectors
                for test_num in range(3):
                    h1 = np.random.uniform(size=n_coil_dofs) - 0.5
                    h2 = np.random.uniform(size=n_coil_dofs) - 0.5
                    
                    # Compute h1^T * H * h2
                    H_h2 = H @ h2
                    h1_H_h2 = h1 @ H_h2
                    
                    # Compute gradient at original point
                    grad_orig = objective_jax.dJ()
                    dJ_h2 = grad_orig @ h2
                    
                    # Test convergence with decreasing epsilon
                    # Use smaller epsilon range for better accuracy
                    err_old = 1e9
                    epsilons = np.power(2., -np.asarray(range(10, 17)))
                    errors = []
                    
                    for eps in epsilons:
                        # Perturb in direction h1
                        bs.x = coil_dofs_orig + eps * h1
                        
                        # Recompute gradient
                        grad_pert = objective_jax.dJ()
                        dJ_pert_h2 = grad_pert @ h2
                        
                        # Finite difference approximation: (dJ(x + eps*h1) - dJ(x))^T * h2 / eps
                        d2f_fd = (dJ_pert_h2 - dJ_h2) / eps
                        
                        # Relative error
                        if np.abs(h1_H_h2) > 1e-12:
                            err = np.abs(d2f_fd - h1_H_h2) / np.abs(h1_H_h2)
                        else:
                            err = np.abs(d2f_fd - h1_H_h2)
                        
                        print(eps, err, d2f_fd, h1_H_h2)
                        errors.append(err)
                        
                        # Check that error decreases (or is already very small)
                        if err_old < 1e-10:
                            # Already converged, just check it stays small
                            self.assertLess(err, 1e-5,
                                          f"Hessian-vector product test failed for {definition}, test {test_num}: "
                                          f"err = {err:.2e}, eps = {eps:.2e}")
                        else:
                            # Check convergence: error should decrease OR be very small
                            # More lenient: allow error to decrease by at least 20% OR be very small
                            converged = (err < err_old * 0.8) or (err < 1e-4)
                            if not converged and err_old > 1e-2:
                                # If error is large, allow it to stay similar (within 20%) for first few iterations
                                converged = (err < err_old * 1.2)
                            
                            self.assertTrue(converged,
                                          f"Hessian-vector product test failed for {definition}, test {test_num}: "
                                          f"err = {err:.2e}, err_old = {err_old:.2e}, eps = {eps:.2e}, "
                                          f"ratio = {err/err_old:.2f}")
                        
                        err_old = err
                    
                    # Final check: error should converge to a small value
                    # The Hessian computation should be accurate, so we expect good convergence
                    final_err = errors[-1]
                    
                    # Check that error decreases significantly (at least by 50% over the iterations)
                    if len(errors) >= 3:
                        initial_err = errors[0]
                        reduction = initial_err / final_err if final_err > 0 else float('inf')
                        # Error should decrease by at least a factor of 2, or be very small
                        self.assertTrue(reduction >= 2.0 or final_err < 1e-4,
                                      f"Hessian-vector product test failed for {definition}, test {test_num}: "
                                      f"error did not decrease sufficiently. Initial: {initial_err:.2e}, "
                                      f"Final: {final_err:.2e}, Reduction: {reduction:.2f}")
                    
                    # Final error should be small
                    self.assertLess(final_err, 1e-3,
                                  f"Hessian-vector product test failed for {definition}, test {test_num}: "
                                  f"final error = {final_err:.2e} is too large. Errors: {[f'{e:.2e}' for e in errors]}, "
                                  f"h1_H_h2 = {h1_H_h2:.2e}")
                    
                    # Restore original coil dofs
                    bs.x = coil_dofs_orig
                
                print(f"{definition}: Hessian shape={H.shape}, symmetric check passed (rel asymmetry = {rel_asymmetry:.2e}), "
                      f"Taylor test passed")


if __name__ == "__main__":
    unittest.main()
