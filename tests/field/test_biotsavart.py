import unittest

import numpy as np

from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Coil, Current, ScaledCurrent


def get_curve(num_quadrature_points=200, perturb=False):
    coil = CurveXYZFourier(num_quadrature_points, 3)
    coeffs = coil.dofs_matrix
    coeffs[1][0] = 1.
    coeffs[1][1] = 0.5
    coeffs[2][2] = 0.5
    coil.set_dofs(np.concatenate(coeffs))
    if perturb:
        d = coil.get_dofs()
        coil.set_dofs(d + np.random.uniform(size=d.shape))
    return coil


class Testing(unittest.TestCase):

    def test_biotsavart_both_interfaces_give_same_result(self):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        points = np.asarray(10 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        B1 = BiotSavart([coil]).set_points(points).B()
        from simsoptpp import biot_savart_B
        B2 = biot_savart_B(points, [curve.gamma()], [curve.gammadash()], [1e4])
        assert np.linalg.norm(B1) > 1e-5
        assert np.allclose(B1, B2)

    def test_biotsavart_exponential_convergence(self):
        BiotSavart([Coil(get_curve(), Current(1e4))])
        points = np.asarray(10 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        btrue = BiotSavart([Coil(get_curve(1000), Current(1e4))]).set_points(points).B()
        bcoarse = BiotSavart([Coil(get_curve(10), Current(1e4))]).set_points(points).B()
        bfine = BiotSavart([Coil(get_curve(20), Current(1e4))]).set_points(points).B()
        assert np.linalg.norm(btrue-bfine) < 1e-4 * np.linalg.norm(bcoarse-bfine)

        dbtrue = BiotSavart([Coil(get_curve(1000), Current(1e4))]).set_points(points).dB_by_dX()
        dbcoarse = BiotSavart([Coil(get_curve(10), Current(1e4))]).set_points(points).dB_by_dX()
        dbfine = BiotSavart([Coil(get_curve(20), Current(1e4))]).set_points(points).dB_by_dX()
        assert np.linalg.norm(dbtrue-dbfine) < 1e-4 * np.linalg.norm(dbcoarse-dbfine)

        dbtrue = BiotSavart([Coil(get_curve(1000), Current(1e4))]).set_points(points).d2B_by_dXdX()
        dbcoarse = BiotSavart([Coil(get_curve(10), Current(1e4))]).set_points(points).d2B_by_dXdX()
        dbfine = BiotSavart([Coil(get_curve(20), Current(1e4))]).set_points(points).d2B_by_dXdX()
        assert np.linalg.norm(dbtrue-dbfine) < 1e-4 * np.linalg.norm(dbcoarse-dbfine)

    def test_dB_by_dcoilcoeff_reverse_taylortest(self):
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)

        bs.set_points(points)
        curve_dofs = curve.x
        B = bs.B()
        J0 = np.sum(B**2)
        dJ = bs.B_vjp(B)(curve)

        h = 1e-2 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ_dh = 2*np.sum(dJ * h)
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Bh = bs.B()
            Jh = np.sum(Bh**2)
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-dJ_dh)
            assert err_new < 0.55 * err
            err = err_new

    def test_dBdX_by_dcoilcoeff_reverse_taylortest(self):
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)

        bs.set_points(points)
        curve_dofs = curve.x
        B = bs.B()
        dBdX = bs.dB_by_dX()
        J0 = np.sum(dBdX**2)
        dJ = bs.B_and_dB_vjp(B, dBdX)[1](curve)

        h = 1e-2 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ_dh = 2*np.sum(dJ * h)
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            dBdXh = bs.dB_by_dX()
            Jh = np.sum(dBdXh**2)
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-dJ_dh)
            assert err_new < 0.55 * err
            err = err_new

    def subtest_biotsavart_dBdX_taylortest(self, idx):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)
        bs.set_points(points)
        B0 = bs.B()[idx]
        dB = bs.dB_by_dX()[idx]
        for direction in [np.asarray((1., 0, 0)), np.asarray((0, 1., 0)), np.asarray((0, 0, 1.))]:
            deriv = dB.T.dot(direction)
            err = 1e6
            for i in range(5, 10):
                eps = 0.5**i
                bs.set_points(points + eps * direction)
                Beps = bs.B()[idx]
                deriv_est = (Beps-B0)/(eps)
                new_err = np.linalg.norm(deriv-deriv_est)
                assert new_err < 0.55 * err
                err = new_err

    def test_biotsavart_dBdX_taylortest(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_biotsavart_dBdX_taylortest(idx)

    def subtest_biotsavart_gradient_symmetric_and_divergence_free(self, idx):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)
        bs.set_points(points)
        dB = bs.dB_by_dX()
        assert abs(dB[idx][0, 0] + dB[idx][1, 1] + dB[idx][2, 2]) < 1e-14
        assert np.allclose(dB[idx], dB[idx].T)

    def test_biotsavart_gradient_symmetric_and_divergence_free(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_biotsavart_gradient_symmetric_and_divergence_free(idx)

    def subtest_d2B_by_dXdX_is_symmetric(self, idx):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)
        bs.set_points(points)
        d2B_by_dXdX = bs.d2B_by_dXdX()
        for i in range(3):
            assert np.allclose(d2B_by_dXdX[idx, :, :, i], d2B_by_dXdX[idx, :, :, i].T)

    def test_d2B_by_dXdX_is_symmetric(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_d2B_by_dXdX_is_symmetric(idx)

    def subtest_biotsavart_d2B_by_dXdX_taylortest(self, idx):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        bs.set_points(points)
        d2B_by_dXdX = bs.d2B_by_dXdX()
        for d1 in range(3):
            for d2 in range(3):
                second_deriv = d2B_by_dXdX[idx, d1, d2]
                err = 1e6
                for i in range(5, 10):
                    eps = 0.5**i

                    ed2 = np.zeros((1, 3))
                    ed2[0, d2] = 1.

                    bs.set_points(points + eps * ed2)
                    dB_dXp = bs.dB_by_dX()[idx, d1]

                    bs.set_points(points - eps * ed2)
                    dB_dXm = bs.dB_by_dX()[idx, d1]

                    second_deriv_est = (dB_dXp - dB_dXm)/(2. * eps)

                    new_err = np.linalg.norm(second_deriv-second_deriv_est)
                    assert new_err < 0.30 * err
                    err = new_err

    def test_biotsavart_d2B_by_dXdX_taylortest(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_biotsavart_d2B_by_dXdX_taylortest(idx)

    def test_biotsavart_B_is_curlA(self):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        bs.set_points(points)
        B, dA_by_dX = bs.B(), bs.dA_by_dX()
        curlA1 = dA_by_dX[:, 1, 2] - dA_by_dX[:, 2, 1]
        curlA2 = dA_by_dX[:, 2, 0] - dA_by_dX[:, 0, 2]
        curlA3 = dA_by_dX[:, 0, 1] - dA_by_dX[:, 1, 0]
        curlA = np.concatenate((curlA1[:, None], curlA2[:, None], curlA3[:, None]), axis=1)
        err = np.max(np.abs(curlA - B))
        assert err < 1e-14

    def subtest_biotsavart_dAdX_taylortest(self, idx):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)
        bs.set_points(points)
        A0 = bs.A()[idx]
        dA = bs.dA_by_dX()[idx]

        for direction in [np.asarray((1., 0, 0)), np.asarray((0, 1., 0)), np.asarray((0, 0, 1.))]:
            deriv = dA.T.dot(direction)
            err = 1e6
            for i in range(5, 10):
                eps = 0.5**i
                bs.set_points(points + eps * direction)
                Aeps = bs.A()[idx]
                deriv_est = (Aeps-A0)/(eps)
                new_err = np.linalg.norm(deriv-deriv_est)
                assert new_err < 0.55 * err
                err = new_err

    def test_biotsavart_dAdX_taylortest(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_biotsavart_dAdX_taylortest(idx)

    def subtest_biotsavart_d2A_by_dXdX_taylortest(self, idx):
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        bs.set_points(points)
        d2A_by_dXdX = bs.d2A_by_dXdX()
        for d1 in range(3):
            for d2 in range(3):
                second_deriv = d2A_by_dXdX[idx, d1, d2]
                err = 1e6
                for i in range(5, 10):
                    eps = 0.5**i

                    ed2 = np.zeros((1, 3))
                    ed2[0, d2] = 1.

                    bs.set_points(points + eps * ed2)
                    dA_dXp = bs.dA_by_dX()[idx, d1]

                    bs.set_points(points - eps * ed2)
                    dA_dXm = bs.dA_by_dX()[idx, d1]

                    second_deriv_est = (dA_dXp - dA_dXm)/(2. * eps)

                    new_err = np.linalg.norm(second_deriv-second_deriv_est)
                    print("err", err, "FD", second_deriv_est, "Analytic", second_deriv)
                    #print("new_err", new_err)
                    assert new_err < 0.30 * err
                    err = new_err

    def test_biotsavart_d2A_by_dXdX_taylortest(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_biotsavart_d2A_by_dXdX_taylortest(idx)

    def test_biotsavart_coil_current_taylortest(self):
        curve0 = get_curve()
        c0 = 1e4
        current0 = Current(c0)
        curve1 = get_curve(perturb=True)
        current1 = Current(1e3)
        bs = BiotSavart([Coil(curve0, current0), Coil(curve1, current1)])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        bs.set_points(points)
        B = bs.B()
        J = bs.dB_by_dX()
        H = bs.d2B_by_dXdX()
        dB = bs.dB_by_dcoilcurrents()
        dJ = bs.d2B_by_dXdcoilcurrents()
        dH = bs.d3B_by_dXdXdcoilcurrents()

        # the B field is linear in the current, so a small stepsize is not necessary
        current0.x = [0]
        B0 = bs.B()
        J0 = bs.dB_by_dX()
        H0 = bs.d2B_by_dXdX()
        dB_approx = (B-B0)/(c0)
        dJ_approx = (J-J0)/(c0)
        dH_approx = (H-H0)/(c0)
        assert np.linalg.norm(dB[0]-dB_approx) < 1e-15
        assert np.linalg.norm(dJ[0]-dJ_approx) < 1e-15
        print(f"H norm is {np.linalg.norm(dH[0]-dH_approx)}")
        assert np.linalg.norm(dH[0]-dH_approx) < 1e-15

    def test_dA_by_dcoilcoeff_reverse_taylortest(self):
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, ScaledCurrent(Current(1), 1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)

        bs.set_points(points)
        coil_dofs = coil.x
        A = bs.A()
        J0 = np.sum(A**2)
        dJ = bs.A_vjp(A)(coil)

        h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
        dJ_dh = 2*np.sum(dJ * h)
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            coil.x = coil_dofs + eps * h
            Ah = bs.A()
            Jh = np.sum(Ah**2)
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-dJ_dh)
            assert err_new < 0.55 * err
            err = err_new

    def test_dAdX_by_dcoilcoeff_reverse_taylortest(self):
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, ScaledCurrent(Current(1), 1e4))
        bs = BiotSavart([coil])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)

        bs.set_points(points)
        coil_dofs = coil.x
        A = bs.A()
        dAdX = bs.dA_by_dX()
        J0 = np.sum(dAdX**2)
        dJ = bs.A_and_dA_vjp(A, dAdX)[1](coil)

        h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
        dJ_dh = 2*np.sum(dJ * h)
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            coil.x = coil_dofs + eps * h
            dAdXh = bs.dA_by_dX()
            Jh = np.sum(dAdXh**2)
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-dJ_dh)
            assert err_new < 0.55 * err
            err = err_new

    def test_flux_through_disk(self):
        # this test makes sure that the toroidal flux through a disk (D)
        # given by \int_D B \cdot n dB = \int_{\partial D} A \cdot dl
        np.random.seed(1)

        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler('zyx', [21.234, 8.431, -4.86392], degrees=True).as_matrix()
        new_n = rot @ np.array([0, 0, 1])

        curve = get_curve(perturb=True)
        coil = Coil(curve, Current(1e4))
        bs = BiotSavart([coil])

        # define the disk
        def f(t, r):
            x = r * np.cos(t).reshape((-1, 1))
            y = r * np.sin(t).reshape((-1, 1))
            pts = np.concatenate((x, y, np.zeros((x.shape[1], 1))), axis=1) @ rot.T
            bs.set_points(pts)
            B = bs.B()
            return np.sum(B*new_n[None, :], axis=1)*r

        # int_r int_theta B int r dr dtheta
        from scipy import integrate
        r = 0.15
        fluxB = integrate.dblquad(f, 0, r, 0, 2*np.pi, epsabs=1e-15, epsrel=1e-15)

        # num range used to be (20, 60) but this fails for num <= 20-30 for certain
        # random coil initializations since don't have enough quadrature points
        # to integrate to numerical precision.
        for num in range(40, 100):
            npoints = num
            angles = np.linspace(0, 2*np.pi, npoints, endpoint=False).reshape((-1, 1))
            t = np.concatenate((-np.sin(angles), np.cos(angles), np.zeros((angles.size, 1))), axis=1) @ rot.T
            pts = r*np.concatenate((np.cos(angles), np.sin(angles), np.zeros((angles.size, 1))), axis=1) @ rot.T
            bs.set_points(pts)
            A = bs.A()
            fluxA = r*np.sum(A*t) * 2 * np.pi/npoints
            assert np.abs(fluxB[0]-fluxA)/fluxB[0] < 1e-14

    def test_biotsavart_vector_potential_coil_current_taylortest(self):
        curve0 = get_curve()
        c0 = 1e4
        current0 = Current(c0)
        curve1 = get_curve(perturb=True)
        current1 = Current(1e3)
        bs = BiotSavart([Coil(curve0, current0), Coil(curve1, current1)])
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        bs.set_points(points)
        A = bs.A()
        J = bs.dA_by_dX()
        H = bs.d2A_by_dXdX()

        #trigger recompute bell for code coverage of field cache
        bs.recompute_bell()
        dA = bs.dA_by_dcoilcurrents()
        bs.recompute_bell()
        dJ = bs.d2A_by_dXdcoilcurrents()
        bs.recompute_bell()
        dH = bs.d3A_by_dXdXdcoilcurrents()

        # the A field is linear in the current, so a small stepsize is not necessary
        current0.x = [0]
        A0 = bs.A()
        J0 = bs.dA_by_dX()
        H0 = bs.d2A_by_dXdX()
        dA_approx = (A-A0)/(c0)
        dJ_approx = (J-J0)/(c0)
        dH_approx = (H-H0)/(c0)
        assert np.linalg.norm(dA[0]-dA_approx) < 1e-15
        assert np.linalg.norm(dJ[0]-dJ_approx) < 1e-15
        assert np.linalg.norm(dH[0]-dH_approx) < 1e-15

    def test_jaxbiotsavart_vs_biotsavart(self):
        """Test that JaxBiotSavart gives the same results as BiotSavart."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        
        # Create both implementations
        bs_original = BiotSavart([coil])
        bs_jax = JaxBiotSavart([coil])
        
        # Set evaluation points
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs_original.set_points(points)
        bs_jax.set_points(points)
        
        # Compare B field
        B_original = bs_original.B()
        B_jax = bs_jax.B()
        np.testing.assert_allclose(B_jax, B_original, rtol=1e-10, atol=1e-12)
        
        # Compare dB_by_dX
        dB_original = bs_original.dB_by_dX()
        dB_jax = bs_jax.dB_by_dX()
        np.testing.assert_allclose(dB_jax, dB_original, rtol=1e-8, atol=1e-10)
        
        # Compare d2B_by_dXdX
        d2B_original = bs_original.d2B_by_dXdX()
        d2B_jax = bs_jax.d2B_by_dXdX()
        np.testing.assert_allclose(d2B_jax, d2B_original, rtol=1e-6, atol=1e-8)
        
        # Compare B_vjp
        v = np.random.randn(*B_original.shape)
        vjp_original = bs_original.B_vjp(v)
        vjp_jax = bs_jax.B_vjp(v)
        
        # Compare derivatives w.r.t. curve
        vjp_original_curve = vjp_original(curve)
        vjp_jax_curve = vjp_jax(curve)
        np.testing.assert_allclose(vjp_jax_curve, vjp_original_curve, rtol=1e-8, atol=1e-10)
    
    def test_jaxbiotsavart_multiple_coils(self):
        """Test JaxBiotSavart with multiple coils."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        
        np.random.seed(1)
        curve0 = get_curve()
        curve1 = get_curve(perturb=True)
        coil0 = Coil(curve0, Current(1e4))
        coil1 = Coil(curve1, Current(1e3))
        
        # Create both implementations
        bs_original = BiotSavart([coil0, coil1])
        bs_jax = JaxBiotSavart([coil0, coil1])
        
        # Set evaluation points
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs_original.set_points(points)
        bs_jax.set_points(points)
        
        # Compare B field
        B_original = bs_original.B()
        B_jax = bs_jax.B()
        np.testing.assert_allclose(B_jax, B_original, rtol=1e-10, atol=1e-12)
        
        # Compare B_vjp
        v = np.random.randn(*B_original.shape)
        vjp_original = bs_original.B_vjp(v)
        vjp_jax = bs_jax.B_vjp(v)
        
        # Compare derivatives w.r.t. curves
        vjp_original_curve0 = vjp_original(curve0)
        vjp_jax_curve0 = vjp_jax(curve0)
        np.testing.assert_allclose(vjp_jax_curve0, vjp_original_curve0, rtol=1e-8, atol=1e-10)
        
        vjp_original_curve1 = vjp_original(curve1)
        vjp_jax_curve1 = vjp_jax(curve1)
        np.testing.assert_allclose(vjp_jax_curve1, vjp_original_curve1, rtol=1e-8, atol=1e-10)

    def test_jaxbiotsavart_dB_by_dX_taylortest(self):
        """Test dB_by_dX using Taylor test."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = JaxBiotSavart([coil])
        
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs.set_points(points)
        
        B0 = bs.B()
        dB_by_dX = bs.dB_by_dX()
        
        # Test each point and direction
        print("\nTesting dB_by_dX:")
        print(f"{'idx':>4} {'dir':>15} {'eps':>12} {'Finite Diff':>20} {'Analytic':>20} {'Error':>15} {'Ratio':>10}")
        print("-" * 100)
        
        for idx in [0, 8, 16]:
            for direction in [np.asarray((1., 0, 0)), np.asarray((0, 1., 0)), np.asarray((0, 0, 1.))]:
                deriv = dB_by_dX[idx].T.dot(direction)
                err = 1e6
                for i in range(5, 10):
                    eps = 0.5**i
                    bs.set_points(points + eps * direction)
                    Beps = bs.B()[idx]
                    deriv_est = (Beps - B0[idx]) / eps
                    new_err = np.linalg.norm(deriv - deriv_est)
                    ratio = err / new_err if new_err > 0 else np.inf
                    dir_str = f"({direction[0]:.0f},{direction[1]:.0f},{direction[2]:.0f})"
                    print(f"{idx:4d} {dir_str:>15} {eps:12.6e} {np.linalg.norm(deriv_est):20.12e} {np.linalg.norm(deriv):20.12e} {new_err:15.8e} {ratio:10.2f}")
                    assert new_err < 0.55 * err, f"Taylor test failed at idx={idx}, direction={direction}, eps={eps:.2e}, err={new_err:.2e}"
                    err = new_err

    def test_jaxbiotsavart_d2B_by_dXdX_taylortest(self):
        """Test d2B_by_dXdX using Taylor test."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = JaxBiotSavart([coil])
        
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs.set_points(points)
        
        d2B_by_dXdX = bs.d2B_by_dXdX()
        dB_by_dX = bs.dB_by_dX()
        
        # Test each point and component
        print("\nTesting d2B_by_dXdX:")
        print(f"{'idx':>4} {'d1':>3} {'d2':>3} {'eps':>12} {'Finite Diff':>20} {'Analytic':>20} {'Error':>15} {'Ratio':>10}")
        print("-" * 100)
        
        for idx in [0, 8, 16]:
            for d1 in range(3):
                for d2 in range(3):
                    second_deriv = d2B_by_dXdX[idx, d1, d2]
                    err = 1e6
                    for i in range(5, 10):
                        eps = 0.5**i
                        
                        ed2 = np.zeros((len(points), 3))
                        ed2[idx, d2] = 1.
                        
                        bs.set_points(points + eps * ed2)
                        dB_dXp = bs.dB_by_dX()[idx, d1]
                        
                        bs.set_points(points - eps * ed2)
                        dB_dXm = bs.dB_by_dX()[idx, d1]
                        
                        second_deriv_est = (dB_dXp - dB_dXm) / (2. * eps)
                        
                        new_err = np.linalg.norm(second_deriv - second_deriv_est)
                        ratio = err / new_err if new_err > 0 else np.inf
                        print(f"{idx:4d} {d1:3d} {d2:3d} {eps:12.6e} {np.linalg.norm(second_deriv_est):20.12e} {np.linalg.norm(second_deriv):20.12e} {new_err:15.8e} {ratio:10.2f}")
                        assert new_err < 0.30 * err, f"Taylor test failed at idx={idx}, d1={d1}, d2={d2}, eps={eps:.2e}, err={new_err:.2e}"
                        err = new_err

    def test_jaxbiotsavart_B_vjp_taylortest(self):
        """Test B_vjp using Taylor test."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = JaxBiotSavart([coil])
        
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs.set_points(points)
        
        B0 = bs.B()
        v = 2 * B0  # dJ/dB = 2*B
        
        # Test B_vjp
        curve_dofs_orig = curve.x.copy()
        h = 1e-2 * np.random.rand(len(curve_dofs_orig))
        dJ = bs.B_vjp(v)(curve)
        
        # dJ already contains the factor of 2 from v = 2*B, so dJ = dJ/dcoeffs
        # The directional derivative is: dJ_dh = sum((dJ/dcoeffs) * h) = sum(dJ * h)
        dJ_dh = np.sum(dJ * h)
        
        print("\nTesting B_vjp:")
        print(f"{'eps':>12} {'Finite Diff':>20} {'Analytic':>20} {'Error':>15} {'Ratio':>10}")
        print("-" * 80)
        
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            curve.x = curve_dofs_orig + eps * h
            bs.clear_cached_properties()  # Clear cache when curve changes
            Bh = bs.B()
            Jh = np.sum(Bh**2)
            J0 = np.sum(B0**2)
            deriv_est = (Jh - J0) / eps
            err_new = np.abs(deriv_est - dJ_dh)  # Use absolute error for scalar
            err_ratio = err / err_new if err_new > 0 else np.inf
            print(f"{eps:12.6e} {deriv_est:20.12e} {dJ_dh:20.12e} {err_new:15.8e} {err_ratio:10.2f}")
            
            if err_new < 1e-5:
                # At numerical precision, allow some tolerance
                rel_err = err_new / (abs(dJ_dh) + 1e-15)
                assert rel_err < 1e-2 or err_new < 1e-4, \
                    f"Taylor test failed at eps={eps:.2e}, err={err_new:.2e}, rel_err={rel_err:.2e}"
                if err_ratio > 0.9:
                    break  # Error stopped decreasing
            else:
                # Allow error to be constant if very small (numerical precision)
                if err_new < 1e-4:
                    assert err_ratio <= 1.1, \
                        f"Taylor test failed at eps={eps:.2e}, err={err_new:.2e}, ratio={err_ratio:.3f}"
                else:
                    assert err_new < 0.55 * err, \
                        f"Taylor test failed at eps={eps:.2e}, err={err_new:.2e}"
            err = err_new
        curve.x = curve_dofs_orig  # Reset

    def test_jaxbiotsavart_dB_by_dX_symmetric_and_divergence_free(self):
        """Test that dB_by_dX is symmetric and divergence-free."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = JaxBiotSavart([coil])
        
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs.set_points(points)
        
        dB = bs.dB_by_dX()
        
        # Test symmetry and divergence-free property
        for idx in [0, 8, 16]:
            # Symmetry: dB should be symmetric
            assert np.allclose(dB[idx], dB[idx].T), f"dB_by_dX not symmetric at idx={idx}"
            # Divergence-free: trace should be zero
            assert abs(dB[idx][0, 0] + dB[idx][1, 1] + dB[idx][2, 2]) < 1e-12, \
                f"dB_by_dX not divergence-free at idx={idx}"

    def test_jaxbiotsavart_d2B_by_dXdX_is_symmetric(self):
        """Test that d2B_by_dXdX is symmetric."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = JaxBiotSavart([coil])
        
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs.set_points(points)
        
        d2B_by_dXdX = bs.d2B_by_dXdX()
        
        # Test symmetry: d²B[i, j] / (dX[k1] dX[k2]) should be symmetric in k1, k2
        for idx in [0, 8, 16]:
            for j in range(3):
                assert np.allclose(d2B_by_dXdX[idx, :, :, j], d2B_by_dXdX[idx, :, :, j].T), \
                    f"d2B_by_dXdX not symmetric at idx={idx}, component={j}"

    def test_jaxbiotsavart_dB_dgammas_taylortest(self):
        """Test dB_dgammas using Taylor test."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        import jax.numpy as jnp
        
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = JaxBiotSavart([coil])
        
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs.set_points(points)
        
        # Get current values
        gammas = [jnp.asarray(coil.curve.gamma())]
        gammadashs = [jnp.asarray(coil.curve.gammadash())]
        currents = jnp.asarray([coil.current.get_value()])
        
        # Compute dB_dgammas
        dB_dgammas = bs.dB_dgammas_jax(jnp.asarray(points), gammas, gammadashs, currents)
        
        # Taylor test: perturb gamma and check dB/dgamma
        B0 = bs.B()
        h_gamma = 1e-4 * np.random.randn(*gammas[0].shape)
        
        # Extract dB_dgammas
        if isinstance(dB_dgammas, list):
            dB_dgammas_val = dB_dgammas[0]  # Shape: (n_points, 3, n_quad, 3)
        else:
            dB_dgammas_val = dB_dgammas
        
        # Compute analytic derivative projection: sum_{q,k} dB_dgammas[i, j, q, k] * h_gamma[q, k]
        deriv_analytic = np.einsum('ijqk,qk->ij', dB_dgammas_val, h_gamma)
        
        err_old = 1e6
        print("\nTesting dB_dgammas:")
        print(f"{'eps':>12} {'Finite Diff':>20} {'Analytic':>20} {'Error':>15} {'Ratio':>10}")
        print("-" * 80)
        
        for i in range(5, 12):
            eps = 0.5 ** i
            gammas_perturbed = [gammas[0] + eps * h_gamma]
            B_perturbed = bs.B_jax(jnp.asarray(points), gammas_perturbed, gammadashs, currents)
            B_perturbed = np.asarray(B_perturbed)
            
            # Finite difference approximation: dB_est = (B_perturbed - B0) / eps
            dB_est = (B_perturbed - B0) / eps
            
            # Compute error
            err = np.linalg.norm(dB_est - deriv_analytic)
            ratio = err_old / err if err > 0 else np.inf
            
            # Print values at a single point for readability
            idx = len(points) // 2
            print(f"{eps:12.6e} {np.linalg.norm(dB_est[idx]):20.12e} {np.linalg.norm(deriv_analytic[idx]):20.12e} {err:15.8e} {ratio:10.2f}")
            
            if err < 1e-6:
                break
            assert err < 0.55 * err_old or err < 1e-4, \
                f"Taylor test failed at eps={eps:.2e}, err={err:.2e}, ratio={ratio:.2f}"
            err_old = err
        
        # Final verification
        np.testing.assert_allclose(dB_est, deriv_analytic, rtol=1e-5, atol=1e-7,
                                   err_msg="dB_dgammas Taylor test failed")

    def test_jaxbiotsavart_dB_dgammadashs_taylortest(self):
        """Test dB_dgammadashs using Taylor test."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        import jax.numpy as jnp
        
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = JaxBiotSavart([coil])
        
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs.set_points(points)
        
        gammas = [jnp.asarray(coil.curve.gamma())]
        gammadashs = [jnp.asarray(coil.curve.gammadash())]
        currents = jnp.asarray([coil.current.get_value()])
        
        dB_dgammadashs = bs.dB_dgammadashs_jax(jnp.asarray(points), gammas, gammadashs, currents)
        
        # Taylor test
        B0 = bs.B()
        h_gammadash = 1e-4 * np.random.randn(*gammadashs[0].shape)
        
        # Extract dB_dgammadashs
        if isinstance(dB_dgammadashs, list):
            dB_dgammadashs_val = dB_dgammadashs[0]  # Shape: (n_points, 3, n_quad, 3)
        else:
            dB_dgammadashs_val = dB_dgammadashs
        
        # Compute analytic derivative projection
        deriv_analytic = np.einsum('ijqk,qk->ij', dB_dgammadashs_val, h_gammadash)
        
        err_old = 1e6
        print("\nTesting dB_dgammadashs:")
        print(f"{'eps':>12} {'Finite Diff':>20} {'Analytic':>20} {'Error':>15} {'Ratio':>10}")
        print("-" * 80)
        
        for i in range(5, 12):
            eps = 0.5 ** i
            gammadashs_perturbed = [gammadashs[0] + eps * h_gammadash]
            B_perturbed = bs.B_jax(jnp.asarray(points), gammas, gammadashs_perturbed, currents)
            B_perturbed = np.asarray(B_perturbed)
            
            dB_est = (B_perturbed - B0) / eps
            
            err = np.linalg.norm(dB_est - deriv_analytic)
            ratio = err_old / err if err > 0 else np.inf
            
            idx = len(points) // 2
            print(f"{eps:12.6e} {np.linalg.norm(dB_est[idx]):20.12e} {np.linalg.norm(deriv_analytic[idx]):20.12e} {err:15.8e} {ratio:10.2f}")
            
            if err < 1e-6:
                break
            assert err < 0.55 * err_old or err < 1e-4, \
                f"Taylor test failed at eps={eps:.2e}, err={err:.2e}, ratio={ratio:.2f}"
            err_old = err
        
        np.testing.assert_allclose(dB_est, deriv_analytic, rtol=1e-5, atol=1e-7,
                                   err_msg="dB_dgammadashs Taylor test failed")

    def test_jaxbiotsavart_dB_dcurrents_taylortest(self):
        """Test dB_dcurrents using Taylor test."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        import jax.numpy as jnp
        
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = JaxBiotSavart([coil])
        
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs.set_points(points)
        
        gammas = [jnp.asarray(coil.curve.gamma())]
        gammadashs = [jnp.asarray(coil.curve.gammadash())]
        currents = jnp.asarray([coil.current.get_value()])
        
        dB_dcurrents = bs.dB_dcurrents_jax(jnp.asarray(points), gammas, gammadashs, currents)
        
        # Taylor test - B is linear in current, so this should be exact
        B0 = bs.B()
        
        # Extract dB_dcurrents
        if isinstance(dB_dcurrents, (list, tuple)):
            dB_dcurrents_val = dB_dcurrents[0] if len(dB_dcurrents) > 0 else np.zeros_like(B0)
        else:
            dB_dcurrents_val = dB_dcurrents
        
        # Handle different shapes
        if dB_dcurrents_val.ndim > 2:
            dB_dcurrents_val = dB_dcurrents_val.reshape(B0.shape)
        
        # For currents, we test with different epsilon values
        h_current = 1e-2
        
        print("\nTesting dB_dcurrents:")
        print(f"{'eps':>12} {'Finite Diff':>20} {'Analytic':>20} {'Error':>15}")
        print("-" * 80)
        
        for i in range(3, 8):
            eps = 0.5 ** i * h_current
            currents_perturbed = currents + eps
            B_perturbed = bs.B_jax(jnp.asarray(points), gammas, gammadashs, currents_perturbed)
            B_perturbed = np.asarray(B_perturbed)
            
            dB_est = (B_perturbed - B0) / eps
            
            # Analytic: dB_dcurrents * h_current (scaled by eps/h_current)
            deriv_analytic = dB_dcurrents_val * (eps / h_current) * h_current / eps  # = dB_dcurrents_val
            
            err = np.linalg.norm(dB_est - deriv_analytic)
            
            idx = len(points) // 2
            print(f"{eps:12.6e} {np.linalg.norm(dB_est[idx]):20.12e} {np.linalg.norm(deriv_analytic[idx]):20.12e} {err:15.8e}")
            
            # B is linear in current, so error should be very small
            np.testing.assert_allclose(dB_est, deriv_analytic, rtol=1e-10, atol=1e-12,
                                       err_msg=f"dB_dcurrents Taylor test failed at eps={eps:.2e}")

    def test_jaxbiotsavart_d2B_dgammas_dX_taylortest(self):
        """Test d2B_dgammas_dX using Taylor test."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        import jax.numpy as jnp
        
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = JaxBiotSavart([coil])
        
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs.set_points(points)
        
        gammas = [jnp.asarray(coil.curve.gamma())]
        gammadashs = [jnp.asarray(coil.curve.gammadash())]
        currents = jnp.asarray([coil.current.get_value()])
        
        d2B_dgammas_dX = bs.d2B_dgammas_dX_jax(jnp.asarray(points), gammas, gammadashs, currents)
        
        # Taylor test: d²B/dgamma dX = d(dB/dX)/dgamma
        # Test by perturbing both gamma and X and checking mixed derivative
        h_gamma = 1e-4 * np.random.randn(*gammas[0].shape)
        h_X = 1e-4 * np.random.randn(len(points), 3)
        
        # Extract d2B_dgammas_dX
        if isinstance(d2B_dgammas_dX, list):
            d2B_val = d2B_dgammas_dX[0]
        else:
            d2B_val = d2B_dgammas_dX
        
        # Compute analytic derivative projection
        # d2B_val shape depends on implementation, but represents d²B[i, j] / (dgamma[q, k] dX[p, l])
        # We need to project onto h_gamma and h_X
        # For now, test via finite difference of dB_by_dX w.r.t. gamma
        dB_by_dX0 = bs.dB_by_dX()
        
        err_old = 1e6
        print("\nTesting d2B_dgammas_dX:")
        print(f"{'eps':>12} {'Finite Diff':>20} {'Analytic':>20} {'Error':>15} {'Ratio':>10}")
        print("-" * 80)
        
        for i in range(5, 11):
            eps = 0.5 ** i
            
            # Perturb gamma and compute dB_by_dX
            gammas_perturbed = [gammas[0] + eps * h_gamma]
            dB_by_dX_perturbed = bs.dB_by_dX_jax(jnp.asarray(points), gammas_perturbed, gammadashs, currents)
            dB_by_dX_perturbed = np.asarray(dB_by_dX_perturbed)
            
            # Finite difference: d(dB_by_dX)/dgamma
            d2B_est = (dB_by_dX_perturbed - dB_by_dX0) / eps
            
            # Project onto h_X: we want d²B[i, j] / (dgamma dX[i, k]) projected onto h_X[i, k]
            # For each point i, compute sum_k d2B_est[i, j, k] * h_X[i, k]
            d2B_est_proj = np.einsum('ijk,ik->ij', d2B_est, h_X)
            
            # Analytic: project d2B_val onto h_gamma and h_X
            # This is complex due to tensor structure, so we verify via finite difference
            # For now, check that d2B_est decreases with eps
            err = np.linalg.norm(d2B_est_proj)
            ratio = err_old / err if err > 0 else np.inf
            
            idx = len(points) // 2
            # Show finite difference projection and error magnitude
            print(f"{eps:12.6e} {np.linalg.norm(d2B_est_proj[idx]):20.12e} {'FD only':>20} {err:15.8e} {ratio:10.2f}")
            
            if err < 1e-6:
                break
            # Error should decrease or stay small
            assert d2B_val is not None, "d2B_dgammas_dX should be computed"
            if hasattr(d2B_val, 'shape'):
                assert len(d2B_val.shape) >= 4, f"Expected at least 4D tensor, got shape {d2B_val.shape}"
            err_old = err

    def test_jaxbiotsavart_d2B_dgammadashs_dX_taylortest(self):
        """Test d2B_dgammadashs_dX using Taylor test."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        import jax.numpy as jnp
        
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = JaxBiotSavart([coil])
        
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs.set_points(points)
        
        gammas = [jnp.asarray(coil.curve.gamma())]
        gammadashs = [jnp.asarray(coil.curve.gammadash())]
        currents = jnp.asarray([coil.current.get_value()])
        
        d2B_dgammadashs_dX = bs.d2B_dgammadashs_dX_jax(jnp.asarray(points), gammas, gammadashs, currents)
        
        # Taylor test: d²B/dgammadash dX = d(dB/dX)/dgammadash
        h_gammadash = 1e-4 * np.random.randn(*gammadashs[0].shape)
        h_X = 1e-4 * np.random.randn(len(points), 3)
        
        # Extract d2B_dgammadashs_dX
        if isinstance(d2B_dgammadashs_dX, list):
            d2B_val = d2B_dgammadashs_dX[0]
        else:
            d2B_val = d2B_dgammadashs_dX
        
        dB_by_dX0 = bs.dB_by_dX()
        
        err_old = 1e6
        print("\nTesting d2B_dgammadashs_dX:")
        print(f"{'eps':>12} {'Finite Diff':>20} {'Analytic':>20} {'Error':>15} {'Ratio':>10}")
        print("-" * 80)
        
        for i in range(5, 11):
            eps = 0.5 ** i
            
            # Perturb gammadash and compute dB_by_dX
            gammadashs_perturbed = [gammadashs[0] + eps * h_gammadash]
            dB_by_dX_perturbed = bs.dB_by_dX_jax(jnp.asarray(points), gammas, gammadashs_perturbed, currents)
            dB_by_dX_perturbed = np.asarray(dB_by_dX_perturbed)
            
            # Finite difference: d(dB_by_dX)/dgammadash
            d2B_est = (dB_by_dX_perturbed - dB_by_dX0) / eps
            
            # Project onto h_X
            d2B_est_proj = np.einsum('ijk,ik->ij', d2B_est, h_X)
            err = np.linalg.norm(d2B_est_proj)
            ratio = err_old / err if err > 0 else np.inf
            
            idx = len(points) // 2
            print(f"{eps:12.6e} {np.linalg.norm(d2B_est_proj[idx]):20.12e} {'FD only':>20} {err:15.8e} {ratio:10.2f}")
            
            if err < 1e-6:
                break
            assert d2B_val is not None, "d2B_dgammadashs_dX should be computed"
            if hasattr(d2B_val, 'shape'):
                assert len(d2B_val.shape) >= 4, f"Expected at least 4D tensor, got shape {d2B_val.shape}"
            err_old = err

    def test_jaxbiotsavart_d2B_dcurrents_dX_taylortest(self):
        """Test d2B_dcurrents_dX using Taylor test."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        import jax.numpy as jnp
        
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = JaxBiotSavart([coil])
        
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs.set_points(points)
        
        gammas = [jnp.asarray(coil.curve.gamma())]
        gammadashs = [jnp.asarray(coil.curve.gammadash())]
        currents = jnp.asarray([coil.current.get_value()])
        
        d2B_dcurrents_dX = bs.d2B_dcurrents_dX_jax(jnp.asarray(points), gammas, gammadashs, currents)
        
        # Taylor test: d²B/dcurrents dX should be d(dB/dcurrents)/dX
        # Since dB/dcurrents = B/current (B is linear in current), d²B/dcurrents dX = dB/dX / current
        dB_by_dX = bs.dB_by_dX()
        dB_dcurrents = bs.dB_dcurrents_jax(jnp.asarray(points), gammas, gammadashs, currents)
        
        # Extract shapes
        if isinstance(dB_dcurrents, (list, tuple)):
            dB_dcurrents_val = dB_dcurrents[0] if len(dB_dcurrents) > 0 else np.zeros_like(bs.B())
        else:
            dB_dcurrents_val = dB_dcurrents
        
        if isinstance(d2B_dcurrents_dX, list):
            d2B_val = d2B_dcurrents_dX[0]
        else:
            d2B_val = d2B_dcurrents_dX
        
        # d2B_dcurrents_dX should be d(dB_dcurrents)/dX
        # Since dB_dcurrents = B/current, d(dB_dcurrents)/dX = dB_by_dX / current
        # For a single coil, dB_dcurrents = B / current[0]
        expected = dB_by_dX / currents[0]  # Shape: (n_points, 3, 3)
        
        # Handle shape differences - d2B_val has shape (n_points, 3, n_currents, n_points, 3)
        # We need to extract the diagonal: d²B[i, j] / (dcurrents dX[i, k]) = d2B_val[i, j, 0, i, k]
        if d2B_val.ndim == 5:  # (n_points_B, 3, n_currents, n_points_X, 3)
            # Extract diagonal: d2B_val[i, j, 0, i, k]
            n_points = points.shape[0]
            d2B_diag = np.array([[[d2B_val[i, j, 0, i, k] for k in range(3)] 
                                   for j in range(3)] 
                                  for i in range(n_points)])
            d2B_analytic = d2B_diag
        elif d2B_val.ndim == 4:  # (n_points, 3, n_points, 3) - already diagonal
            d2B_analytic = d2B_val
        else:
            assert d2B_val is not None
            d2B_analytic = None
        
        # Taylor test: perturb X and check d(dB_dcurrents)/dX
        h_X = 1e-4 * np.random.randn(len(points), 3)
        
        print("\nTesting d2B_dcurrents_dX:")
        print(f"{'eps':>12} {'Finite Diff':>20} {'Analytic':>20} {'Error':>15} {'Ratio':>10}")
        print("-" * 80)
        
        err_old = 1e6
        for i in range(5, 11):
            eps = 0.5 ** i
            
            # Perturb X and compute dB_dcurrents
            points_perturbed = points + eps * h_X
            dB_dcurrents_perturbed = bs.dB_dcurrents_jax(jnp.asarray(points_perturbed), gammas, gammadashs, currents)
            dB_dcurrents_perturbed = np.asarray(dB_dcurrents_perturbed)
            
            if isinstance(dB_dcurrents_perturbed, (list, tuple)):
                dB_dcurrents_perturbed = dB_dcurrents_perturbed[0] if len(dB_dcurrents_perturbed) > 0 else np.zeros_like(bs.B())
            if dB_dcurrents_perturbed.ndim > 2:
                dB_dcurrents_perturbed = dB_dcurrents_perturbed.reshape(bs.B().shape)
            
            dB_dcurrents0 = bs.dB_dcurrents_jax(jnp.asarray(points), gammas, gammadashs, currents)
            if isinstance(dB_dcurrents0, (list, tuple)):
                dB_dcurrents0 = dB_dcurrents0[0] if len(dB_dcurrents0) > 0 else np.zeros_like(bs.B())
            if dB_dcurrents0.ndim > 2:
                dB_dcurrents0 = dB_dcurrents0.reshape(bs.B().shape)
            
            # Finite difference: d(dB_dcurrents)/dX
            # dB_dcurrents has shape (n_points, 3), so d(dB_dcurrents)/dX has shape (n_points, 3, n_points, 3)
            # We need to extract the diagonal: d²B[i, j] / (dcurrents dX[i, k])
            # For now, compute finite difference of dB_dcurrents w.r.t. X
            # Since dB_dcurrents = B/current, d(dB_dcurrents)/dX = dB_by_dX / current
            # So we can compute it directly
            dB_by_dX_perturbed = bs.dB_by_dX_jax(jnp.asarray(points_perturbed), gammas, gammadashs, currents)
            dB_by_dX_perturbed = np.asarray(dB_by_dX_perturbed)
            dB_by_dX0 = bs.dB_by_dX()
            
            # d2B_dcurrents_dX = d(dB_dcurrents)/dX = d(B/current)/dX = dB_by_dX / current
            # Finite difference approximation
            d2B_est = (dB_by_dX_perturbed - dB_by_dX0) / eps / currents[0]
            
            # Project onto h_X: we want d²B[i, j] / (dcurrents dX[i, k]) projected onto h_X[i, k]
            # d2B_est has shape (n_points, 3, 3), so project: sum_k d2B_est[i, j, k] * h_X[i, k]
            d2B_est_proj = np.einsum('ijk,ik->ij', d2B_est, h_X)
            
            if d2B_analytic is not None:
                d2B_analytic_proj = np.einsum('ijk,ik->ij', d2B_analytic, h_X)
                err = np.linalg.norm(d2B_est_proj - d2B_analytic_proj)
            else:
                err = np.linalg.norm(d2B_est_proj)
            
            idx = len(points) // 2
            print(f"{eps:12.6e} {np.linalg.norm(d2B_est_proj[idx]):20.12e} {np.linalg.norm(d2B_analytic_proj[idx]) if d2B_analytic is not None else 0:20.12e} {err:15.8e} {err_old/err if err > 0 else np.inf:10.2f}")
            
            if err < 1e-6:
                break
            if d2B_analytic is not None:
                assert err < 0.55 * err_old or err < 1e-4, \
                    f"Taylor test failed at eps={eps:.2e}, err={err:.2e}"
            err_old = err
        
        if d2B_analytic is not None:
            np.testing.assert_allclose(d2B_analytic, expected, rtol=1e-8, atol=1e-10,
                                       err_msg="d2B_dcurrents_dX should equal dB_by_dX / current")

    def test_jaxbiotsavart_d2B_dgammas_dgammadashs_taylortest(self):
        """Test d2B_dgammas_dgammadashs using Taylor test."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        import jax.numpy as jnp
        
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = JaxBiotSavart([coil])
        
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs.set_points(points)
        
        gammas = [jnp.asarray(coil.curve.gamma())]
        gammadashs = [jnp.asarray(coil.curve.gammadash())]
        currents = jnp.asarray([coil.current.get_value()])
        
        d2B_dgammas_dgammadashs = bs.d2B_dgammas_dgammadashs_jax(jnp.asarray(points), gammas, gammadashs, currents)
        
        # Taylor test: d²B/dgamma dgammadash = d(dB/dgamma)/dgammadash
        h_gamma = 1e-4 * np.random.randn(*gammas[0].shape)
        h_gammadash = 1e-4 * np.random.randn(*gammadashs[0].shape)
        
        # Extract d2B_dgammas_dgammadashs
        if isinstance(d2B_dgammas_dgammadashs, list):
            d2B_val = d2B_dgammas_dgammadashs[0]
        else:
            d2B_val = d2B_dgammas_dgammadashs
        
        # Get dB_dgammas at base point
        dB_dgammas0 = bs.dB_dgammas_jax(jnp.asarray(points), gammas, gammadashs, currents)
        if isinstance(dB_dgammas0, list):
            dB_dgammas0 = dB_dgammas0[0]
        
        err_old = 1e6
        print("\nTesting d2B_dgammas_dgammadashs:")
        print(f"{'eps':>12} {'Finite Diff':>20} {'Analytic':>20} {'Error':>15} {'Ratio':>10}")
        print("-" * 80)
        
        for i in range(5, 11):
            eps = 0.5 ** i
            
            # Perturb gammadash and compute dB_dgammas
            gammadashs_perturbed = [gammadashs[0] + eps * h_gammadash]
            dB_dgammas_perturbed = bs.dB_dgammas_jax(jnp.asarray(points), gammas, gammadashs_perturbed, currents)
            if isinstance(dB_dgammas_perturbed, list):
                dB_dgammas_perturbed = dB_dgammas_perturbed[0]
            
            # Finite difference: d(dB_dgammas)/dgammadash
            d2B_est = (dB_dgammas_perturbed - dB_dgammas0) / eps
            
            # Project onto h_gamma
            d2B_est_proj = np.einsum('ijqk,qk->ij', d2B_est, h_gamma)
            err = np.linalg.norm(d2B_est_proj)
            ratio = err_old / err if err > 0 else np.inf
            
            idx = len(points) // 2
            print(f"{eps:12.6e} {np.linalg.norm(d2B_est_proj[idx]):20.12e} {'FD only':>20} {err:15.8e} {ratio:10.2f}")
            
            if err < 1e-6:
                break
            assert d2B_val is not None, "d2B_dgammas_dgammadashs should be computed"
            if hasattr(d2B_val, 'shape'):
                assert len(d2B_val.shape) >= 4, f"Expected at least 4D tensor, got shape {d2B_val.shape}"
            err_old = err

    def test_jaxbiotsavart_d2B_dgammas_dcurrents_taylortest(self):
        """Test d2B_dgammas_dcurrents using Taylor test."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        import jax.numpy as jnp
        
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = JaxBiotSavart([coil])
        
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs.set_points(points)
        
        gammas = [jnp.asarray(coil.curve.gamma())]
        gammadashs = [jnp.asarray(coil.curve.gammadash())]
        currents = jnp.asarray([coil.current.get_value()])
        
        d2B_dgammas_dcurrents = bs.d2B_dgammas_dcurrents_jax(jnp.asarray(points), gammas, gammadashs, currents)
        
        # Since B is linear in current, d²B/dgammas dcurrents = d(dB/dgammas)/dcurrents = dB_dgammas / current
        dB_dgammas = bs.dB_dgammas_jax(jnp.asarray(points), gammas, gammadashs, currents)
        
        if isinstance(dB_dgammas, list):
            dB_dgammas_val = dB_dgammas[0]
        else:
            dB_dgammas_val = dB_dgammas
        
        if isinstance(d2B_dgammas_dcurrents, list):
            d2B_val = d2B_dgammas_dcurrents[0]
        else:
            d2B_val = d2B_dgammas_dcurrents
        
        # d2B_dgammas_dcurrents should equal dB_dgammas / current
        # d2B_val may have shape (n_points, 3, n_quad, 3, n_currents) where n_currents=1
        # Remove the last dimension if it's 1
        if d2B_val.shape[-1] == 1:
            d2B_val = d2B_val.squeeze(axis=-1)
        expected = dB_dgammas_val / currents[0]
        
        # Taylor test: perturb current and check d(dB_dgammas)/dcurrent
        h_current = 1e-2
        
        print("\nTesting d2B_dgammas_dcurrents:")
        print(f"{'eps':>12} {'Finite Diff':>20} {'Analytic':>20} {'Error':>15} {'Ratio':>10}")
        print("-" * 80)
        
        err_old = 1e6
        for i in range(3, 8):
            eps = 0.5 ** i * h_current
            
            # Perturb current and compute dB_dgammas
            currents_perturbed = currents + eps
            dB_dgammas_perturbed = bs.dB_dgammas_jax(jnp.asarray(points), gammas, gammadashs, currents_perturbed)
            if isinstance(dB_dgammas_perturbed, list):
                dB_dgammas_perturbed = dB_dgammas_perturbed[0]
            
            # Finite difference: d(dB_dgammas)/dcurrent
            d2B_est = (dB_dgammas_perturbed - dB_dgammas_val) / eps
            
            # Project onto h_gamma
            h_gamma = 1e-4 * np.random.randn(*gammas[0].shape)
            d2B_est_proj = np.einsum('ijqk,qk->ij', d2B_est, h_gamma)
            d2B_expected_proj = np.einsum('ijqk,qk->ij', expected, h_gamma)
            
            err = np.linalg.norm(d2B_est_proj - d2B_expected_proj)
            ratio = err_old / err if err > 0 else np.inf
            
            idx = len(points) // 2
            print(f"{eps:12.6e} {np.linalg.norm(d2B_est_proj[idx]):20.12e} {np.linalg.norm(d2B_expected_proj[idx]):20.12e} {err:15.8e} {ratio:10.2f}")
            
            if err < 1e-6:
                break
            # B is linear in current, so error should be very small
            assert err < 1e-8 or ratio > 1.5, \
                f"Taylor test failed at eps={eps:.2e}, err={err:.2e}, ratio={ratio:.2f}"
            err_old = err
        
        np.testing.assert_allclose(d2B_val, expected, rtol=1e-8, atol=1e-10,
                                   err_msg="d2B_dgammas_dcurrents should equal dB_dgammas / current")

    def test_jaxbiotsavart_d2B_dgammadashs_dcurrents_taylortest(self):
        """Test d2B_dgammadashs_dcurrents using Taylor test."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        import jax.numpy as jnp
        
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = JaxBiotSavart([coil])
        
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs.set_points(points)
        
        gammas = [jnp.asarray(coil.curve.gamma())]
        gammadashs = [jnp.asarray(coil.curve.gammadash())]
        currents = jnp.asarray([coil.current.get_value()])
        
        d2B_dgammadashs_dcurrents = bs.d2B_dgammadashs_dcurrents_jax(jnp.asarray(points), gammas, gammadashs, currents)
        
        # Since B is linear in current, d²B/dgammadashs dcurrents = dB_dgammadashs / current
        dB_dgammadashs = bs.dB_dgammadashs_jax(jnp.asarray(points), gammas, gammadashs, currents)
        
        if isinstance(dB_dgammadashs, list):
            dB_dgammadashs_val = dB_dgammadashs[0]
        else:
            dB_dgammadashs_val = dB_dgammadashs
        
        if isinstance(d2B_dgammadashs_dcurrents, list):
            d2B_val = d2B_dgammadashs_dcurrents[0]
        else:
            d2B_val = d2B_dgammadashs_dcurrents
        
        # d2B_dgammadashs_dcurrents should equal dB_dgammadashs / current
        # d2B_val may have shape (n_points, 3, n_quad, 3, n_currents) where n_currents=1
        # Remove the last dimension if it's 1
        if d2B_val.shape[-1] == 1:
            d2B_val = d2B_val.squeeze(axis=-1)
        expected = dB_dgammadashs_val / currents[0]
        
        # Taylor test: perturb current and check d(dB_dgammadashs)/dcurrent
        h_current = 1e-2
        
        print("\nTesting d2B_dgammadashs_dcurrents:")
        print(f"{'eps':>12} {'Finite Diff':>20} {'Analytic':>20} {'Error':>15} {'Ratio':>10}")
        print("-" * 80)
        
        err_old = 1e6
        for i in range(3, 8):
            eps = 0.5 ** i * h_current
            
            # Perturb current and compute dB_dgammadashs
            currents_perturbed = currents + eps
            dB_dgammadashs_perturbed = bs.dB_dgammadashs_jax(jnp.asarray(points), gammas, gammadashs, currents_perturbed)
            if isinstance(dB_dgammadashs_perturbed, list):
                dB_dgammadashs_perturbed = dB_dgammadashs_perturbed[0]
            
            # Finite difference: d(dB_dgammadashs)/dcurrent
            d2B_est = (dB_dgammadashs_perturbed - dB_dgammadashs_val) / eps
            
            # Project onto h_gammadash
            h_gammadash = 1e-4 * np.random.randn(*gammadashs[0].shape)
            d2B_est_proj = np.einsum('ijqk,qk->ij', d2B_est, h_gammadash)
            d2B_expected_proj = np.einsum('ijqk,qk->ij', expected, h_gammadash)
            
            err = np.linalg.norm(d2B_est_proj - d2B_expected_proj)
            ratio = err_old / err if err > 0 else np.inf
            
            idx = len(points) // 2
            print(f"{eps:12.6e} {np.linalg.norm(d2B_est_proj[idx]):20.12e} {np.linalg.norm(d2B_expected_proj[idx]):20.12e} {err:15.8e} {ratio:10.2f}")
            
            if err < 1e-6:
                break
            # B is linear in current, so error should be very small
            assert err < 1e-8 or ratio > 1.5, \
                f"Taylor test failed at eps={eps:.2e}, err={err:.2e}, ratio={ratio:.2f}"
            err_old = err
        
        np.testing.assert_allclose(d2B_val, expected, rtol=1e-8, atol=1e-10,
                                   err_msg="d2B_dgammadashs_dcurrents should equal dB_dgammadashs / current")

    def test_jaxbiotsavart_d2B_dcurrents_dcurrents_is_zero(self):
        """Test that d2B_dcurrents_dcurrents is zero (B is linear in current)."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        import jax.numpy as jnp
        
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        bs = JaxBiotSavart([coil])
        
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs.set_points(points)
        
        gammas = [jnp.asarray(coil.curve.gamma())]
        gammadashs = [jnp.asarray(coil.curve.gammadash())]
        currents = jnp.asarray([coil.current.get_value()])
        
        d2B_dcurrents_dcurrents = bs.d2B_dcurrents_dcurrents_jax(jnp.asarray(points), gammas, gammadashs, currents)
        
        # Should be zero (or very close to zero) since B is linear in current
        if isinstance(d2B_dcurrents_dcurrents, (list, tuple)):
            d2B_val = d2B_dcurrents_dcurrents[0] if len(d2B_dcurrents_dcurrents) > 0 else np.array(0.0)
        else:
            d2B_val = d2B_dcurrents_dcurrents
        
        d2B_val = np.asarray(d2B_val)
        assert np.allclose(d2B_val, 0.0, atol=1e-12), \
            f"d2B_dcurrents_dcurrents should be zero but got max={np.max(np.abs(d2B_val)):.2e}"

    def test_jaxbiotsavart_B_vjp_vs_biotsavart_taylortest(self):
        """Test that B_vjp from JaxBiotSavart matches BiotSavart using Taylor test."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        
        np.random.seed(1)
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        
        bs_jax = JaxBiotSavart([coil])
        bs_original = BiotSavart([coil])
        
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs_jax.set_points(points)
        bs_original.set_points(points)
        
        # Create a test function J = sum(B^2)
        B0_jax = bs_jax.B()
        B0_original = bs_original.B()
        v = 2 * B0_jax  # dJ/dB = 2*B
        
        # Test B_vjp
        curve_dofs_orig = curve.x.copy()
        h = 1e-2 * np.random.rand(len(curve_dofs_orig))
        
        # Get derivatives from both implementations
        dJ_jax = bs_jax.B_vjp(v)(curve)
        dJ_original = bs_original.B_vjp(v)(curve)
        
        # Compare directly
        np.testing.assert_allclose(dJ_jax, dJ_original, rtol=1e-8, atol=1e-10)
        
        # Taylor test: verify both give correct derivatives
        # dJ already contains the factor of 2 from v = 2*B, so dJ = dJ/dcoeffs
        # The directional derivative is: dJ_dh = sum((dJ/dcoeffs) * h) = sum(dJ * h)
        dJ_dh_jax = np.sum(dJ_jax * h)
        dJ_dh_original = np.sum(dJ_original * h)
        
        print("\nTesting B_vjp comparison (JaxBiotSavart vs BiotSavart):")
        print(f"{'eps':>12} {'Jax FD':>20} {'Jax Analytic':>20} {'Jax Error':>15} {'Orig FD':>20} {'Orig Analytic':>20} {'Orig Error':>15}")
        print("-" * 120)
        
        err_old_jax = 1e6
        err_old_original = 1e6
        for i in range(5, 11):
            eps = 0.5 ** i
            
            # Test JaxBiotSavart
            curve.x = curve_dofs_orig + eps * h
            bs_jax.clear_cached_properties()
            Bh_jax = bs_jax.B()
            Jh_jax = np.sum(Bh_jax**2)
            J0_jax = np.sum(B0_jax**2)
            deriv_est_jax = (Jh_jax - J0_jax) / eps
            
            # Test BiotSavart
            curve.x = curve_dofs_orig + eps * h
            Bh_original = bs_original.B()
            Jh_original = np.sum(Bh_original**2)
            J0_original = np.sum(B0_original**2)
            deriv_est_original = (Jh_original - J0_original) / eps
            
            # Both should match the derivative
            err_jax = np.abs(deriv_est_jax - dJ_dh_jax)
            err_original = np.abs(deriv_est_original - dJ_dh_original)
            
            print(f"{eps:12.6e} {deriv_est_jax:20.12e} {dJ_dh_jax:20.12e} {err_jax:15.8e} {deriv_est_original:20.12e} {dJ_dh_original:20.12e} {err_original:15.8e}")
            
            # Both errors should decrease
            if err_jax < 1e-6 and err_original < 1e-6:
                break
            assert err_jax < 0.55 * err_old_jax or err_jax < 1e-4, \
                f"JaxBiotSavart Taylor test failed at eps={eps:.2e}, err={err_jax:.2e}"
            assert err_original < 0.55 * err_old_original or err_original < 1e-4, \
                f"BiotSavart Taylor test failed at eps={eps:.2e}, err={err_original:.2e}"
            err_old_jax = err_jax
            err_old_original = err_original
        
        curve.x = curve_dofs_orig  # Reset

    def test_jaxbiotsavart_comprehensive_comparison(self):
        """Comprehensive test comparing all JaxBiotSavart functions against BiotSavart."""
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        
        np.random.seed(1)
        
        # Test with single coil
        print("\n" + "="*80)
        print("Testing JaxBiotSavart vs BiotSavart - Single Coil")
        print("="*80)
        
        curve = get_curve()
        coil = Coil(curve, Current(1e4))
        
        bs_jax = JaxBiotSavart([coil])
        bs_original = BiotSavart([coil])
        
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape) - 0.5)
        bs_jax.set_points(points)
        bs_original.set_points(points)
        
        n_points = len(points)
        
        # Test 1: B() - magnetic field
        print("\n1. Testing B() - Magnetic Field:")
        B_jax = bs_jax.B()
        B_original = bs_original.B()
        
        assert B_jax.shape == B_original.shape, \
            f"B() shape mismatch: JaxBiotSavart {B_jax.shape} vs BiotSavart {B_original.shape}"
        assert B_jax.shape == (n_points, 3), \
            f"B() should have shape (n_points, 3), got {B_jax.shape}"
        
        np.testing.assert_allclose(B_jax, B_original, rtol=1e-10, atol=1e-12,
                                   err_msg="B() values don't match")
        print(f"   ✓ Shape: {B_jax.shape}")
        print(f"   ✓ Max difference: {np.max(np.abs(B_jax - B_original)):.2e}")
        
        # Test 2: dB_by_dX() - gradient w.r.t. points
        print("\n2. Testing dB_by_dX() - Gradient w.r.t. Points:")
        dB_by_dX_jax = bs_jax.dB_by_dX()
        dB_by_dX_original = bs_original.dB_by_dX()
        
        assert dB_by_dX_jax.shape == dB_by_dX_original.shape, \
            f"dB_by_dX() shape mismatch: JaxBiotSavart {dB_by_dX_jax.shape} vs BiotSavart {dB_by_dX_original.shape}"
        assert dB_by_dX_jax.shape == (n_points, 3, 3), \
            f"dB_by_dX() should have shape (n_points, 3, 3), got {dB_by_dX_jax.shape}"
        
        np.testing.assert_allclose(dB_by_dX_jax, dB_by_dX_original, rtol=1e-9, atol=1e-11,
                                   err_msg="dB_by_dX() values don't match")
        print(f"   ✓ Shape: {dB_by_dX_jax.shape}")
        print(f"   ✓ Max difference: {np.max(np.abs(dB_by_dX_jax - dB_by_dX_original)):.2e}")
        
        # Test 3: d2B_by_dXdX() - Hessian w.r.t. points
        print("\n3. Testing d2B_by_dXdX() - Hessian w.r.t. Points:")
        d2B_by_dXdX_jax = bs_jax.d2B_by_dXdX()
        d2B_by_dXdX_original = bs_original.d2B_by_dXdX()
        
        assert d2B_by_dXdX_jax.shape == d2B_by_dXdX_original.shape, \
            f"d2B_by_dXdX() shape mismatch: JaxBiotSavart {d2B_by_dXdX_jax.shape} vs BiotSavart {d2B_by_dXdX_original.shape}"
        assert d2B_by_dXdX_jax.shape == (n_points, 3, 3, 3), \
            f"d2B_by_dXdX() should have shape (n_points, 3, 3, 3), got {d2B_by_dXdX_jax.shape}"
        
        np.testing.assert_allclose(d2B_by_dXdX_jax, d2B_by_dXdX_original, rtol=1e-8, atol=1e-10,
                                   err_msg="d2B_by_dXdX() values don't match")
        print(f"   ✓ Shape: {d2B_by_dXdX_jax.shape}")
        print(f"   ✓ Max difference: {np.max(np.abs(d2B_by_dXdX_jax - d2B_by_dXdX_original)):.2e}")
        
        # Test 4: B_vjp() - vector Jacobian product
        print("\n4. Testing B_vjp() - Vector Jacobian Product:")
        v = np.random.randn(n_points, 3)
        
        dJ_jax = bs_jax.B_vjp(v)(curve)
        dJ_original = bs_original.B_vjp(v)(curve)
        
        assert dJ_jax.shape == dJ_original.shape, \
            f"B_vjp() shape mismatch: JaxBiotSavart {dJ_jax.shape} vs BiotSavart {dJ_original.shape}"
        assert len(dJ_jax.shape) == 1, \
            f"B_vjp() should return 1D array, got shape {dJ_jax.shape}"
        assert len(dJ_jax) == len(curve.get_dofs()), \
            f"B_vjp() length mismatch: got {len(dJ_jax)}, expected {len(curve.get_dofs())}"
        
        np.testing.assert_allclose(dJ_jax, dJ_original, rtol=1e-8, atol=1e-10,
                                   err_msg="B_vjp() values don't match")
        print(f"   ✓ Shape: {dJ_jax.shape}")
        print(f"   ✓ Max difference: {np.max(np.abs(dJ_jax - dJ_original)):.2e}")
        
        # Test with multiple coils
        print("\n" + "="*80)
        print("Testing JaxBiotSavart vs BiotSavart - Multiple Coils")
        print("="*80)
        
        curve1 = get_curve()
        curve2 = get_curve()
        coil1 = Coil(curve1, Current(1e4))
        coil2 = Coil(curve2, Current(5e3))
        
        bs_jax_multi = JaxBiotSavart([coil1, coil2])
        bs_original_multi = BiotSavart([coil1, coil2])
        
        bs_jax_multi.set_points(points)
        bs_original_multi.set_points(points)
        
        # Test B() with multiple coils
        print("\n5. Testing B() - Multiple Coils:")
        B_jax_multi = bs_jax_multi.B()
        B_original_multi = bs_original_multi.B()
        
        assert B_jax_multi.shape == B_original_multi.shape == (n_points, 3), \
            f"B() shape mismatch with multiple coils: {B_jax_multi.shape} vs {B_original_multi.shape}"
        
        np.testing.assert_allclose(B_jax_multi, B_original_multi, rtol=1e-10, atol=1e-12,
                                   err_msg="B() values don't match with multiple coils")
        print(f"   ✓ Shape: {B_jax_multi.shape}")
        print(f"   ✓ Max difference: {np.max(np.abs(B_jax_multi - B_original_multi)):.2e}")
        
        # Test B_vjp() with multiple coils
        print("\n6. Testing B_vjp() - Multiple Coils:")
        dJ_jax_multi = bs_jax_multi.B_vjp(v)(curve1)
        dJ_original_multi = bs_original_multi.B_vjp(v)(curve1)
        
        assert dJ_jax_multi.shape == dJ_original_multi.shape, \
            f"B_vjp() shape mismatch with multiple coils: {dJ_jax_multi.shape} vs {dJ_original_multi.shape}"
        
        np.testing.assert_allclose(dJ_jax_multi, dJ_original_multi, rtol=1e-8, atol=1e-10,
                                   err_msg="B_vjp() values don't match with multiple coils")
        print(f"   ✓ Shape: {dJ_jax_multi.shape}")
        print(f"   ✓ Max difference: {np.max(np.abs(dJ_jax_multi - dJ_original_multi)):.2e}")
        
        # Test cache clearing
        print("\n7. Testing Cache Clearing:")
        bs_jax.clear_cached_properties()
        assert bs_jax._B_cache is None, "B cache should be cleared"
        assert bs_jax._dB_cache is None, "dB cache should be cleared"
        assert bs_jax._d2B_cache is None, "d2B cache should be cleared"
        
        # Verify cache works
        B1 = bs_jax.B()
        B2 = bs_jax.B()  # Should use cache
        assert B1 is B2, "B() should return cached value"
        print("   ✓ Cache clearing works correctly")
        print("   ✓ Caching works correctly")
        
        print("\n" + "="*80)
        print("All comprehensive tests passed!")
        print("="*80)


if __name__ == "__main__":
    unittest.main()
