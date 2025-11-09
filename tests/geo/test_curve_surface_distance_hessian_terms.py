"""
Test individual terms in CurveSurfaceDistance Hessian using finite differences.
"""
import numpy as np
from simsopt.geo.curveobjectives import CurveSurfaceDistance
from simsopt.geo.curvexyzfourier import JaxCurveXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier


def test_hessian_terms_individually():
    """Test each term in the Hessian individually using finite differences."""
    np.random.seed(42)
    
    # Create a simple surface
    ntor = 0
    surface = SurfaceRZFourier.from_nphi_ntheta(nfp=1, nphi=32, ntheta=32, ntor=ntor)
    surface.set(f'rc(0,{ntor})', 1.6)
    surface.set(f'rc(1,{ntor})', 0.2)
    surface.set(f'zs(1,{ntor})', 0.2)
    
    # Create curve close to surface
    curve = JaxCurveXYZFourier(50, 3)
    # Set curve to be close to surface (radius ~1.6)
    curve.set('xc(0)', 1.6)
    curve.set('yc(0)', 0.0)
    curve.set('zc(0)', 0.0)
    # Add small perturbation
    curve.x = curve.x + 0.05 * np.random.randn(len(curve.x))
    
    J = CurveSurfaceDistance([curve], surface, 0.5)
    J.compute_candidates()
    
    if len(J.candidates) == 0:
        print("No candidates found, skipping test")
        return
    
    # Get surface data
    gammas = surface.gamma().reshape((-1, 3))
    ns = surface.normal().reshape((-1, 3))
    
    # Get current state
    gammac = curve.gamma()
    lc = curve.gammadash()
    curve_dofs = curve.x.copy()
    
    # Get first-order derivatives
    dJ_dgammac = np.asarray(J.dJ_dgamma(gammac, lc, gammas, ns))
    dJ_dlc = np.asarray(J.dJ_dlc(gammac, lc, gammas, ns))
    
    # Get first-order derivatives w.r.t. curve dofs
    dgammac_dx = np.asarray(curve.dgamma_by_dcoeff())
    dlc_dx = np.asarray(curve.dgammadash_by_dcoeff())
    n_quad, n_components, n_dofs = dgammac_dx.shape
    
    # Get Hessian w.r.t. gammac and lc
    d2J_dgammac2_flat = np.asarray(J.d2J_dgamma2(gammac, lc, gammas, ns))
    d2J_dlc2_flat = np.asarray(J.d2J_dlc2(gammac, lc, gammas, ns))
    d2J_dgammac2 = d2J_dgammac2_flat.reshape((n_quad, 3, n_quad, 3))
    d2J_dlc2 = d2J_dlc2_flat.reshape((n_quad, 3, n_quad, 3))
    
    # Get cross terms
    # Note: jacfwd returns d(output)/d(input), so we need to transpose
    # d2J_dgamma_dlc from jacfwd: [k, c, l, m] = d(dJ/dlc[k, c]) / d(gammac[l, m])
    # But we want: [k, c, l, m] = d(dJ/dlc[l, m]) / d(gammac[k, c])
    # So we transpose: [l, m, k, c] -> [k, c, l, m]
    d2J_dgammac_dlc_raw = np.asarray(J.d2J_dgamma_dlc(gammac, lc, gammas, ns))
    d2J_dgammac_dlc = d2J_dgammac_dlc_raw.transpose(2, 3, 0, 1)  # Swap indices to get correct order
    
    # Similarly for d2J_dlc_dgamma
    # d2J_dlc_dgamma from jacfwd: [l, m, k, c] = d(dJ/dgammac[l, m]) / d(lc[k, c])
    # But we want: [l, m, k, c] = d(dJ/dgammac[k, c]) / d(lc[l, m])
    # So we transpose: [k, c, l, m] -> [l, m, k, c]
    d2J_dlc_dgammac_raw = np.asarray(J.d2J_dlc_dgamma(gammac, lc, gammas, ns))
    d2J_dlc_dgammac = d2J_dlc_dgammac_raw.transpose(2, 3, 0, 1)  # Swap indices to get correct order
    
    # Create random test vectors
    np.random.seed(42)
    h = np.random.randn(n_dofs) * 1e-3
    
    print(f"\n{'='*60}")
    print("Testing individual Hessian terms with finite differences")
    print(f"{'='*60}\n")
    
    # Term 1: dJ/dgammac * d²gammac/dx²
    print("Term 1: dJ/dgammac * d²gammac/dx²")
    try:
        if hasattr(curve, 'd2gamma_by_d2coeff_impl') and hasattr(curve, 'd2gamma_by_d2coeff_jax'):
            d2gammac_dx2 = np.zeros((n_quad, 3, n_dofs, n_dofs))
            curve.d2gamma_by_d2coeff_impl(d2gammac_dx2)
            term1_analytical = np.einsum('kc,kcij->ij', dJ_dgammac, d2gammac_dx2)
            term1_h = h @ term1_analytical @ h
            
            # Finite difference: d/dx (dJ/dgammac * dgammac/dx) @ h
            # = d/dx (dJ/dgammac) * dgammac/dx @ h + dJ/dgammac * d²gammac/dx² @ h
            # We want the second part: dJ/dgammac * d²gammac/dx² @ h
            # This is: dJ/dgammac * d²gammac/dx² @ h
            # We can compute it as: dJ/dgammac @ (d²gammac/dx² @ h)
            # d²gammac/dx² @ h has shape (n_quad, 3, n_dofs) -> (n_quad, 3)
            d2gammac_dx2_h = np.einsum('kcij,j->kci', d2gammac_dx2, h)
            # Sum over quadrature points and components
            term1_fd = np.einsum('kc,kci->', dJ_dgammac, d2gammac_dx2_h)
            
            # Alternative: finite difference on dJ/dgammac * dgammac/dx
            eps = 1e-5
            curve.x = curve_dofs + eps * h
            gammac_pert = curve.gamma()
            dgammac_dx_pert = np.asarray(curve.dgamma_by_dcoeff())
            dJ_dgammac_pert = np.asarray(J.dJ_dgamma(gammac_pert, lc, gammas, ns))
            
            # dJ/dgammac * dgammac/dx @ h at perturbed point
            dJ_dgammac_dgammac_dx_h_pert = np.sum(dJ_dgammac_pert * np.einsum('kci,i->kc', dgammac_dx_pert, h))
            
            curve.x = curve_dofs - eps * h
            gammac_neg = curve.gamma()
            dgammac_dx_neg = np.asarray(curve.dgamma_by_dcoeff())
            dJ_dgammac_neg = np.asarray(J.dJ_dgamma(gammac_neg, lc, gammas, ns))
            dJ_dgammac_dgammac_dx_h_neg = np.sum(dJ_dgammac_neg * np.einsum('kci,i->kc', dgammac_dx_neg, h))
            
            # Central difference
            term1_fd2 = (dJ_dgammac_dgammac_dx_h_pert - dJ_dgammac_dgammac_dx_h_neg) / (2 * eps)
            
            # At original point
            curve.x = curve_dofs

            print(f"  Analytical: {term1_h:.6e}")
            print(f"  FD (direct): {term1_fd:.6e}")
            print(f"  FD (alternative): {term1_fd2:.6e}")
            print(f"  Error (direct): {abs(term1_h - term1_fd):.6e}")
            print(f"  Error (alt): {abs(term1_h - term1_fd2):.6e}")
        else:
            print("  Skipped (d2gamma_by_d2coeff not available)")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Term 2: (dgammac/dx)^T @ d²J/dgammac² @ (dgammac/dx)
    print("\nTerm 2: (dgammac/dx)^T @ d²J/dgammac² @ (dgammac/dx)")
    term2_analytical = np.einsum('kci,kclm,lmj->ij', dgammac_dx, d2J_dgammac2, dgammac_dx)
    term2_h = h @ term2_analytical @ h
    
    # Finite difference: perturb gammac in direction dgammac_dx @ h
    eps = 1e-5
    dgammac_dx_h = np.einsum('kci,i->kc', dgammac_dx, h)
    
    # Perturb gammac
    gammac_pert = gammac + eps * dgammac_dx_h
    dJ_dgammac_pert = np.asarray(J.dJ_dgamma(gammac_pert, lc, gammas, ns))
    dJ_dgammac_dgammac_dx_h_pert = np.sum(dJ_dgammac_pert * dgammac_dx_h)
    
    gammac_neg = gammac - eps * dgammac_dx_h
    dJ_dgammac_neg = np.asarray(J.dJ_dgamma(gammac_neg, lc, gammas, ns))
    dJ_dgammac_dgammac_dx_h_neg = np.sum(dJ_dgammac_neg * dgammac_dx_h)
    
    # This gives us the change in dJ/dgammac @ dgammac/dx @ h
    # But we need to subtract the contribution from term 1
    # Actually, this gives us term 1 + term 2 combined
    # Let's compute the full change and subtract term 1
    
    term1_plus_term2_fd = (dJ_dgammac_dgammac_dx_h_pert - dJ_dgammac_dgammac_dx_h_neg) / (2 * eps)
    
    # Subtract term 1 contribution
    if hasattr(curve, 'd2gamma_by_d2coeff_impl'):
        try:
            d2gammac_dx2 = np.zeros((n_quad, 3, n_dofs, n_dofs))
            curve.d2gamma_by_d2coeff_impl(d2gammac_dx2)
            term1_contribution = np.einsum('kc,kci->', dJ_dgammac, np.einsum('kcij,j->kci', d2gammac_dx2, h))
            term2_fd = term1_plus_term2_fd - term1_contribution
        except:
            term2_fd = term1_plus_term2_fd
    else:
        term2_fd = term1_plus_term2_fd
    
    print(f"  Analytical: {term2_h:.6e}")
    print(f"  FD: {term2_fd:.6e}")
    print(f"  Error: {abs(term2_h - term2_fd):.6e}")
    print(f"  Rel error: {abs(term2_h - term2_fd) / (abs(term2_h) + 1e-15):.6e}")
    
    # Term 3: dJ/dlc * d²lc/dx²
    print("\nTerm 3: dJ/dlc * d²lc/dx²")
    try:
        if hasattr(curve, 'd2gammadash_by_d2coeff_impl') and hasattr(curve, 'd2gammadash_by_d2coeff_jax'):
            d2lc_dx2 = np.zeros((n_quad, 3, n_dofs, n_dofs))
            curve.d2gammadash_by_d2coeff_impl(d2lc_dx2)
            term3_analytical = np.einsum('kc,kcij->ij', dJ_dlc, d2lc_dx2)
            term3_h = h @ term3_analytical @ h
            
            d2lc_dx2_h = np.einsum('kcij,j->kci', d2lc_dx2, h)
            term3_fd = np.einsum('kc,kci->', dJ_dlc, d2lc_dx2_h)
            
            print(f"  Analytical: {term3_h:.6e}")
            print(f"  FD: {term3_fd:.6e}")
            print(f"  Error: {abs(term3_h - term3_fd):.6e}")
        else:
            print("  Skipped (d2gammadash_by_d2coeff not available)")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Term 4: (dlc/dx)^T @ d²J/dlc² @ (dlc/dx)
    print("\nTerm 4: (dlc/dx)^T @ d²J/dlc² @ (dlc/dx)")
    term4_analytical = np.einsum('kci,kclm,lmj->ij', dlc_dx, d2J_dlc2, dlc_dx)
    term4_h = h @ term4_analytical @ h
    
    dlc_dx_h = np.einsum('kci,i->kc', dlc_dx, h)
    lc_pert = lc + eps * dlc_dx_h
    dJ_dlc_pert = np.asarray(J.dJ_dlc(gammac, lc_pert, gammas, ns))
    dJ_dlc_dlc_dx_h_pert = np.sum(dJ_dlc_pert * dlc_dx_h)
    
    lc_neg = lc - eps * dlc_dx_h
    dJ_dlc_neg = np.asarray(J.dJ_dlc(gammac, lc_neg, gammas, ns))
    dJ_dlc_dlc_dx_h_neg = np.sum(dJ_dlc_neg * dlc_dx_h)
    
    term3_plus_term4_fd = (dJ_dlc_dlc_dx_h_pert - dJ_dlc_dlc_dx_h_neg) / (2 * eps)
    
    if hasattr(curve, 'd2gammadash_by_d2coeff_impl'):
        try:
            d2lc_dx2 = np.zeros((n_quad, 3, n_dofs, n_dofs))
            curve.d2gammadash_by_d2coeff_impl(d2lc_dx2)
            term3_contribution = np.einsum('kc,kci->', dJ_dlc, np.einsum('kcij,j->kci', d2lc_dx2, h))
            term4_fd = term3_plus_term4_fd - term3_contribution
        except:
            term4_fd = term3_plus_term4_fd
    else:
        term4_fd = term3_plus_term4_fd
    
    print(f"  Analytical: {term4_h:.6e}")
    print(f"  FD: {term4_fd:.6e}")
    print(f"  Error: {abs(term4_h - term4_fd):.6e}")
    print(f"  Rel error: {abs(term4_h - term4_fd) / (abs(term4_h) + 1e-15):.6e}")
    
    # Term 5: (dgammac/dx)^T @ d²J/(dgammac dlc) @ (dlc/dx)
    print("\nTerm 5: (dgammac/dx)^T @ d²J/(dgammac dlc) @ (dlc/dx)")
    term5_analytical = np.einsum('kci,kclm,lmj->ij', dgammac_dx, d2J_dgammac_dlc, dlc_dx)
    term5_h = h @ term5_analytical @ h
    
    # Finite difference: perturb only gammac, measure change in dJ/dlc @ dlc/dx @ h
    # This gives us the cross-term contribution
    gammac_pert = gammac + eps * dgammac_dx_h
    dJ_dlc_pert_gammac = np.asarray(J.dJ_dlc(gammac_pert, lc, gammas, ns))
    dJ_dlc_dlc_dx_h_pert_gammac = np.sum(dJ_dlc_pert_gammac * dlc_dx_h)
    
    gammac_neg = gammac - eps * dgammac_dx_h
    dJ_dlc_neg_gammac = np.asarray(J.dJ_dlc(gammac_neg, lc, gammas, ns))
    dJ_dlc_dlc_dx_h_neg_gammac = np.sum(dJ_dlc_neg_gammac * dlc_dx_h)
    
    # This gives us the change in dJ/dlc @ dlc/dx @ h when only gammac is perturbed
    # This is exactly term 5 (the cross-term)
    term5_fd = (dJ_dlc_dlc_dx_h_pert_gammac - dJ_dlc_dlc_dx_h_neg_gammac) / (2 * eps)
    
    print(f"  Analytical: {term5_h:.6e}")
    print(f"  FD: {term5_fd:.6e}")
    print(f"  Error: {abs(term5_h - term5_fd):.6e}")
    print(f"  Rel error: {abs(term5_h - term5_fd) / (abs(term5_h) + 1e-15):.6e}")
    
    # Term 6: (dlc/dx)^T @ d²J/(dlc dgammac) @ (dgammac/dx)
    print("\nTerm 6: (dlc/dx)^T @ d²J/(dlc dgammac) @ (dgammac/dx)")
    term6_analytical = np.einsum('kci,kclm,lmj->ij', dlc_dx, d2J_dlc_dgammac, dgammac_dx)
    term6_h = h @ term6_analytical @ h
    
    # Should be the same as term 5
    print(f"  Analytical: {term6_h:.6e}")
    print(f"  Term 5: {term5_h:.6e}")
    print(f"  Difference: {abs(term6_h - term5_h):.6e}")
    print(f"  Should be equal (symmetric): {abs(term6_h - term5_h) < 1e-10}")
    
    # Restore original state
    curve.x = curve_dofs
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print("Total Hessian (sum of all terms):")
    H_total = (term1_analytical if hasattr(curve, 'd2gamma_by_d2coeff_impl') else np.zeros((n_dofs, n_dofs))) + \
              term2_analytical + \
              (term3_analytical if hasattr(curve, 'd2gammadash_by_d2coeff_impl') else np.zeros((n_dofs, n_dofs))) + \
              term4_analytical + \
              term5_analytical + \
              term6_analytical
    H_total_h = h @ H_total @ h
    print(f"  h^T H h (analytical): {H_total_h:.6e}")


if __name__ == '__main__':
    test_hessian_terms_individually()

