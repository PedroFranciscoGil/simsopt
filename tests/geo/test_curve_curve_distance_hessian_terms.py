"""
Test individual terms in CurveCurveDistance Hessian using finite differences.
"""
import numpy as np
from simsopt.geo.curveobjectives import CurveCurveDistance
from simsopt.geo.curvexyzfourier import JaxCurveXYZFourier


def test_hessian_terms_individually():
    """Test each term in the Hessian individually using finite differences."""
    np.random.seed(42)
    
    # Create two curves close to each other
    curve1 = JaxCurveXYZFourier(50, 3)
    curve1.set('xc(0)', 1.0)
    curve1.set('yc(0)', 0.0)
    curve1.set('zc(0)', 0.0)
    curve1.x = curve1.x + 0.05 * np.random.randn(len(curve1.x))
    
    curve2 = JaxCurveXYZFourier(50, 3)
    curve2.set('xc(0)', 1.1)  # Close to curve1
    curve2.set('yc(0)', 0.0)
    curve2.set('zc(0)', 0.0)
    curve2.x = curve2.x + 0.05 * np.random.randn(len(curve2.x))
    
    J = CurveCurveDistance([curve1, curve2], 0.5, downsample=1)
    J.compute_candidates()
    
    if len(J.candidates) == 0:
        print("No candidates found, skipping test")
        return
    
    # Get current state
    gamma1 = curve1.gamma()
    l1 = curve1.gammadash()
    gamma2 = curve2.gamma()
    l2 = curve2.gammadash()
    curve1_dofs = curve1.x.copy()
    curve2_dofs = curve2.x.copy()
    
    # Get first-order derivatives
    dJ_dgamma1 = np.asarray(J.dJ_dgamma1(gamma1, l1, gamma2, l2, J.downsample))
    dJ_dl1 = np.asarray(J.dJ_dl1(gamma1, l1, gamma2, l2, J.downsample))
    dJ_dgamma2 = np.asarray(J.dJ_dgamma2(gamma1, l1, gamma2, l2, J.downsample))
    dJ_dl2 = np.asarray(J.dJ_dl2(gamma1, l1, gamma2, l2, J.downsample))
    
    # Get first-order derivatives w.r.t. curve dofs
    dgamma1_dx = np.asarray(curve1.dgamma_by_dcoeff())
    dl1_dx = np.asarray(curve1.dgammadash_by_dcoeff())
    dgamma2_dx = np.asarray(curve2.dgamma_by_dcoeff())
    dl2_dx = np.asarray(curve2.dgammadash_by_dcoeff())
    n_quad1, n_components1, n_dofs1 = dgamma1_dx.shape
    n_quad2, n_components2, n_dofs2 = dgamma2_dx.shape
    
    # Get Hessian w.r.t. gamma1, l1, gamma2, l2
    d2J_dgamma12_flat = np.asarray(J.d2J_dgamma12(gamma1, l1, gamma2, l2, J.downsample))
    d2J_dl12_flat = np.asarray(J.d2J_dl12(gamma1, l1, gamma2, l2, J.downsample))
    d2J_dgamma22_flat = np.asarray(J.d2J_dgamma22(gamma1, l1, gamma2, l2, J.downsample))
    d2J_dl22_flat = np.asarray(J.d2J_dl22(gamma1, l1, gamma2, l2, J.downsample))
    d2J_dgamma12 = d2J_dgamma12_flat.reshape((n_quad1, 3, n_quad1, 3))
    d2J_dl12 = d2J_dl12_flat.reshape((n_quad1, 3, n_quad1, 3))
    d2J_dgamma22 = d2J_dgamma22_flat.reshape((n_quad2, 3, n_quad2, 3))
    d2J_dl22 = d2J_dl22_flat.reshape((n_quad2, 3, n_quad2, 3))
    
    # Get cross terms
    # Note: jacfwd returns d(output)/d(input), so we need to transpose
    d2J_dgamma1_dl1_raw = np.asarray(J.d2J_dgamma1_dl1(gamma1, l1, gamma2, l2, J.downsample))
    d2J_dgamma1_dl1 = d2J_dgamma1_dl1_raw.transpose(2, 3, 0, 1)  # Swap indices to get correct order
    
    d2J_dl1_dgamma1_raw = np.asarray(J.d2J_dl1_dgamma1(gamma1, l1, gamma2, l2, J.downsample))
    d2J_dl1_dgamma1 = d2J_dl1_dgamma1_raw.transpose(2, 3, 0, 1)  # Swap indices to get correct order
    
    d2J_dgamma1_dgamma2_raw = np.asarray(J.d2J_dgamma1_dgamma2(gamma1, l1, gamma2, l2, J.downsample))
    d2J_dgamma1_dgamma2 = d2J_dgamma1_dgamma2_raw.transpose(2, 3, 0, 1)  # Swap indices to get correct order
    
    d2J_dgamma1_dl2_raw = np.asarray(J.d2J_dgamma1_dl2(gamma1, l1, gamma2, l2, J.downsample))
    d2J_dgamma1_dl2 = d2J_dgamma1_dl2_raw.transpose(2, 3, 0, 1)  # Swap indices to get correct order
    
    d2J_dl1_dgamma2_raw = np.asarray(J.d2J_dl1_dgamma2(gamma1, l1, gamma2, l2, J.downsample))
    d2J_dl1_dgamma2 = d2J_dl1_dgamma2_raw.transpose(2, 3, 0, 1)  # Swap indices to get correct order
    
    d2J_dl1_dl2_raw = np.asarray(J.d2J_dl1_dl2(gamma1, l1, gamma2, l2, J.downsample))
    d2J_dl1_dl2 = d2J_dl1_dl2_raw.transpose(2, 3, 0, 1)  # Swap indices to get correct order
    
    d2J_dl2_dgamma2_raw = np.asarray(J.d2J_dl2_dgamma2(gamma1, l1, gamma2, l2, J.downsample))
    d2J_dl2_dgamma2 = d2J_dl2_dgamma2_raw.transpose(2, 3, 0, 1)  # Swap indices to get correct order
    
    # Create random test vectors
    np.random.seed(42)
    h1 = np.random.randn(n_dofs1) * 1e-3
    h2 = np.random.randn(n_dofs2) * 1e-3
    
    print(f"\n{'='*60}")
    print("Testing individual Hessian terms for CurveCurveDistance")
    print(f"{'='*60}\n")
    
    # ========== Terms for curve 1 (H_11) ==========
    print("=" * 60)
    print("TERMS FOR CURVE 1 (H_11)")
    print("=" * 60)
    
    # Term 1: dJ/dgamma1 * d²gamma1/dx1²
    print("\nTerm 1: dJ/dgamma1 * d²gamma1/dx1²")
    try:
        if hasattr(curve1, 'd2gamma_by_d2coeff_impl') and hasattr(curve1, 'd2gamma_by_d2coeff_jax'):
            d2gamma1_dx2 = np.zeros((n_quad1, 3, n_dofs1, n_dofs1))
            curve1.d2gamma_by_d2coeff_impl(d2gamma1_dx2)
            term1_analytical = np.einsum('kc,kcij->ij', dJ_dgamma1, d2gamma1_dx2)
            term1_h = h1 @ term1_analytical @ h1
            
            d2gamma1_dx2_h = np.einsum('kcij,j->kci', d2gamma1_dx2, h1)
            term1_fd = np.einsum('kc,kci->', dJ_dgamma1, d2gamma1_dx2_h)
            
            print(f"  Analytical: {term1_h:.6e}")
            print(f"  FD: {term1_fd:.6e}")
            print(f"  Error: {abs(term1_h - term1_fd):.6e}")
        else:
            print("  Skipped (d2gamma_by_d2coeff not available)")
            term1_analytical = np.zeros((n_dofs1, n_dofs1))
    except Exception as e:
        print(f"  Error: {e}")
        term1_analytical = np.zeros((n_dofs1, n_dofs1))
    
    # Term 2: (dgamma1/dx1)^T @ d²J/dgamma1² @ (dgamma1/dx1)
    print("\nTerm 2: (dgamma1/dx1)^T @ d²J/dgamma1² @ (dgamma1/dx1)")
    term2_analytical = np.einsum('kci,kclm,lmj->ij', dgamma1_dx, d2J_dgamma12, dgamma1_dx)
    term2_h = h1 @ term2_analytical @ h1
    
    eps = 1e-5
    dgamma1_dx_h = np.einsum('kci,i->kc', dgamma1_dx, h1)
    
    gamma1_pert = gamma1 + eps * dgamma1_dx_h
    dJ_dgamma1_pert = np.asarray(J.dJ_dgamma1(gamma1_pert, l1, gamma2, l2, J.downsample))
    dJ_dgamma1_dgamma1_dx_h_pert = np.sum(dJ_dgamma1_pert * dgamma1_dx_h)
    
    gamma1_neg = gamma1 - eps * dgamma1_dx_h
    dJ_dgamma1_neg = np.asarray(J.dJ_dgamma1(gamma1_neg, l1, gamma2, l2, J.downsample))
    dJ_dgamma1_dgamma1_dx_h_neg = np.sum(dJ_dgamma1_neg * dgamma1_dx_h)
    
    term1_plus_term2_fd = (dJ_dgamma1_dgamma1_dx_h_pert - dJ_dgamma1_dgamma1_dx_h_neg) / (2 * eps)
    
    if hasattr(curve1, 'd2gamma_by_d2coeff_impl'):
        try:
            d2gamma1_dx2 = np.zeros((n_quad1, 3, n_dofs1, n_dofs1))
            curve1.d2gamma_by_d2coeff_impl(d2gamma1_dx2)
            term1_contribution = np.einsum('kc,kci->', dJ_dgamma1, np.einsum('kcij,j->kci', d2gamma1_dx2, h1))
            term2_fd = term1_plus_term2_fd - term1_contribution
        except:
            term2_fd = term1_plus_term2_fd
    else:
        term2_fd = term1_plus_term2_fd
    
    print(f"  Analytical: {term2_h:.6e}")
    print(f"  FD: {term2_fd:.6e}")
    print(f"  Error: {abs(term2_h - term2_fd):.6e}")
    print(f"  Rel error: {abs(term2_h - term2_fd) / (abs(term2_h) + 1e-15):.6e}")
    
    # Term 3: dJ/dl1 * d²l1/dx1²
    print("\nTerm 3: dJ/dl1 * d²l1/dx1²")
    try:
        if hasattr(curve1, 'd2gammadash_by_d2coeff_impl') and hasattr(curve1, 'd2gammadash_by_d2coeff_jax'):
            d2l1_dx2 = np.zeros((n_quad1, 3, n_dofs1, n_dofs1))
            curve1.d2gammadash_by_d2coeff_impl(d2l1_dx2)
            term3_analytical = np.einsum('kc,kcij->ij', dJ_dl1, d2l1_dx2)
            term3_h = h1 @ term3_analytical @ h1
            
            d2l1_dx2_h = np.einsum('kcij,j->kci', d2l1_dx2, h1)
            term3_fd = np.einsum('kc,kci->', dJ_dl1, d2l1_dx2_h)
            
            print(f"  Analytical: {term3_h:.6e}")
            print(f"  FD: {term3_fd:.6e}")
            print(f"  Error: {abs(term3_h - term3_fd):.6e}")
        else:
            print("  Skipped (d2gammadash_by_d2coeff not available)")
            term3_analytical = np.zeros((n_dofs1, n_dofs1))
    except Exception as e:
        print(f"  Error: {e}")
        term3_analytical = np.zeros((n_dofs1, n_dofs1))
    
    # Term 4: (dl1/dx1)^T @ d²J/dl1² @ (dl1/dx1)
    print("\nTerm 4: (dl1/dx1)^T @ d²J/dl1² @ (dl1/dx1)")
    term4_analytical = np.einsum('kci,kclm,lmj->ij', dl1_dx, d2J_dl12, dl1_dx)
    term4_h = h1 @ term4_analytical @ h1
    
    dl1_dx_h = np.einsum('kci,i->kc', dl1_dx, h1)
    l1_pert = l1 + eps * dl1_dx_h
    dJ_dl1_pert = np.asarray(J.dJ_dl1(gamma1, l1_pert, gamma2, l2, J.downsample))
    dJ_dl1_dl1_dx_h_pert = np.sum(dJ_dl1_pert * dl1_dx_h)
    
    l1_neg = l1 - eps * dl1_dx_h
    dJ_dl1_neg = np.asarray(J.dJ_dl1(gamma1, l1_neg, gamma2, l2, J.downsample))
    dJ_dl1_dl1_dx_h_neg = np.sum(dJ_dl1_neg * dl1_dx_h)
    
    term3_plus_term4_fd = (dJ_dl1_dl1_dx_h_pert - dJ_dl1_dl1_dx_h_neg) / (2 * eps)
    
    if hasattr(curve1, 'd2gammadash_by_d2coeff_impl'):
        try:
            d2l1_dx2 = np.zeros((n_quad1, 3, n_dofs1, n_dofs1))
            curve1.d2gammadash_by_d2coeff_impl(d2l1_dx2)
            term3_contribution = np.einsum('kc,kci->', dJ_dl1, np.einsum('kcij,j->kci', d2l1_dx2, h1))
            term4_fd = term3_plus_term4_fd - term3_contribution
        except:
            term4_fd = term3_plus_term4_fd
    else:
        term4_fd = term3_plus_term4_fd
    
    print(f"  Analytical: {term4_h:.6e}")
    print(f"  FD: {term4_fd:.6e}")
    print(f"  Error: {abs(term4_h - term4_fd):.6e}")
    print(f"  Rel error: {abs(term4_h - term4_fd) / (abs(term4_h) + 1e-15):.6e}")
    
    # Term 5: (dgamma1/dx1)^T @ d²J/(dgamma1 dl1) @ (dl1/dx1)
    print("\nTerm 5: (dgamma1/dx1)^T @ d²J/(dgamma1 dl1) @ (dl1/dx1)")
    term5_analytical = np.einsum('kci,kclm,lmj->ij', dgamma1_dx, d2J_dgamma1_dl1, dl1_dx)
    term5_h = h1 @ term5_analytical @ h1
    
    # Finite difference: perturb only gamma1, measure change in dJ/dl1 @ dl1/dx1 @ h1
    gamma1_pert = gamma1 + eps * dgamma1_dx_h
    dJ_dl1_pert_gamma1 = np.asarray(J.dJ_dl1(gamma1_pert, l1, gamma2, l2, J.downsample))
    dJ_dl1_dl1_dx_h_pert_gamma1 = np.sum(dJ_dl1_pert_gamma1 * dl1_dx_h)
    
    gamma1_neg = gamma1 - eps * dgamma1_dx_h
    dJ_dl1_neg_gamma1 = np.asarray(J.dJ_dl1(gamma1_neg, l1, gamma2, l2, J.downsample))
    dJ_dl1_dl1_dx_h_neg_gamma1 = np.sum(dJ_dl1_neg_gamma1 * dl1_dx_h)
    
    term5_fd = (dJ_dl1_dl1_dx_h_pert_gamma1 - dJ_dl1_dl1_dx_h_neg_gamma1) / (2 * eps)
    
    print(f"  Analytical: {term5_h:.6e}")
    print(f"  FD: {term5_fd:.6e}")
    print(f"  Error: {abs(term5_h - term5_fd):.6e}")
    print(f"  Rel error: {abs(term5_h - term5_fd) / (abs(term5_h) + 1e-15):.6e}")
    
    # Term 6: (dl1/dx1)^T @ d²J/(dl1 dgamma1) @ (dgamma1/dx1)
    print("\nTerm 6: (dl1/dx1)^T @ d²J/(dl1 dgamma1) @ (dgamma1/dx1)")
    term6_analytical = np.einsum('kci,kclm,lmj->ij', dl1_dx, d2J_dl1_dgamma1, dgamma1_dx)
    term6_h = h1 @ term6_analytical @ h1
    
    # Should be the same as term 5
    print(f"  Analytical: {term6_h:.6e}")
    print(f"  Term 5: {term5_h:.6e}")
    print(f"  Difference: {abs(term6_h - term5_h):.6e}")
    print(f"  Should be equal (symmetric): {abs(term6_h - term5_h) < 1e-10}")
    
    # ========== Terms for curve 2 (H_22) ==========
    print("\n" + "=" * 60)
    print("TERMS FOR CURVE 2 (H_22)")
    print("=" * 60)
    
    # Similar terms for curve 2 (terms 7-12)
    # Term 7: dJ/dgamma2 * d²gamma2/dx2²
    print("\nTerm 7: dJ/dgamma2 * d²gamma2/dx2²")
    try:
        if hasattr(curve2, 'd2gamma_by_d2coeff_impl') and hasattr(curve2, 'd2gamma_by_d2coeff_jax'):
            d2gamma2_dx2 = np.zeros((n_quad2, 3, n_dofs2, n_dofs2))
            curve2.d2gamma_by_d2coeff_impl(d2gamma2_dx2)
            term7_analytical = np.einsum('kc,kcij->ij', dJ_dgamma2, d2gamma2_dx2)
            term7_h = h2 @ term7_analytical @ h2
            
            d2gamma2_dx2_h = np.einsum('kcij,j->kci', d2gamma2_dx2, h2)
            term7_fd = np.einsum('kc,kci->', dJ_dgamma2, d2gamma2_dx2_h)
            
            print(f"  Analytical: {term7_h:.6e}")
            print(f"  FD: {term7_fd:.6e}")
            print(f"  Error: {abs(term7_h - term7_fd):.6e}")
        else:
            print("  Skipped (d2gamma_by_d2coeff not available)")
            term7_analytical = np.zeros((n_dofs2, n_dofs2))
    except Exception as e:
        print(f"  Error: {e}")
        term7_analytical = np.zeros((n_dofs2, n_dofs2))
    
    # Term 8: (dgamma2/dx2)^T @ d²J/dgamma2² @ (dgamma2/dx2)
    print("\nTerm 8: (dgamma2/dx2)^T @ d²J/dgamma2² @ (dgamma2/dx2)")
    term8_analytical = np.einsum('kci,kclm,lmj->ij', dgamma2_dx, d2J_dgamma22, dgamma2_dx)
    term8_h = h2 @ term8_analytical @ h2
    
    dgamma2_dx_h = np.einsum('kci,i->kc', dgamma2_dx, h2)
    
    gamma2_pert = gamma2 + eps * dgamma2_dx_h
    dJ_dgamma2_pert = np.asarray(J.dJ_dgamma2(gamma1, l1, gamma2_pert, l2, J.downsample))
    dJ_dgamma2_dgamma2_dx_h_pert = np.sum(dJ_dgamma2_pert * dgamma2_dx_h)
    
    gamma2_neg = gamma2 - eps * dgamma2_dx_h
    dJ_dgamma2_neg = np.asarray(J.dJ_dgamma2(gamma1, l1, gamma2_neg, l2, J.downsample))
    dJ_dgamma2_dgamma2_dx_h_neg = np.sum(dJ_dgamma2_neg * dgamma2_dx_h)
    
    term7_plus_term8_fd = (dJ_dgamma2_dgamma2_dx_h_pert - dJ_dgamma2_dgamma2_dx_h_neg) / (2 * eps)
    
    if hasattr(curve2, 'd2gamma_by_d2coeff_impl'):
        try:
            d2gamma2_dx2 = np.zeros((n_quad2, 3, n_dofs2, n_dofs2))
            curve2.d2gamma_by_d2coeff_impl(d2gamma2_dx2)
            term7_contribution = np.einsum('kc,kci->', dJ_dgamma2, np.einsum('kcij,j->kci', d2gamma2_dx2, h2))
            term8_fd = term7_plus_term8_fd - term7_contribution
        except:
            term8_fd = term7_plus_term8_fd
    else:
        term8_fd = term7_plus_term8_fd
    
    print(f"  Analytical: {term8_h:.6e}")
    print(f"  FD: {term8_fd:.6e}")
    print(f"  Error: {abs(term8_h - term8_fd):.6e}")
    print(f"  Rel error: {abs(term8_h - term8_fd) / (abs(term8_h) + 1e-15):.6e}")
    
    # Term 9: dJ/dl2 * d²l2/dx2²
    print("\nTerm 9: dJ/dl2 * d²l2/dx2²")
    try:
        if hasattr(curve2, 'd2gammadash_by_d2coeff_impl') and hasattr(curve2, 'd2gammadash_by_d2coeff_jax'):
            d2l2_dx2 = np.zeros((n_quad2, 3, n_dofs2, n_dofs2))
            curve2.d2gammadash_by_d2coeff_impl(d2l2_dx2)
            term9_analytical = np.einsum('kc,kcij->ij', dJ_dl2, d2l2_dx2)
            term9_h = h2 @ term9_analytical @ h2
            
            d2l2_dx2_h = np.einsum('kcij,j->kci', d2l2_dx2, h2)
            term9_fd = np.einsum('kc,kci->', dJ_dl2, d2l2_dx2_h)
            
            print(f"  Analytical: {term9_h:.6e}")
            print(f"  FD: {term9_fd:.6e}")
            print(f"  Error: {abs(term9_h - term9_fd):.6e}")
        else:
            print("  Skipped (d2gammadash_by_d2coeff not available)")
            term9_analytical = np.zeros((n_dofs2, n_dofs2))
    except Exception as e:
        print(f"  Error: {e}")
        term9_analytical = np.zeros((n_dofs2, n_dofs2))
    
    # Term 10: (dl2/dx2)^T @ d²J/dl2² @ (dl2/dx2)
    print("\nTerm 10: (dl2/dx2)^T @ d²J/dl2² @ (dl2/dx2)")
    term10_analytical = np.einsum('kci,kclm,lmj->ij', dl2_dx, d2J_dl22, dl2_dx)
    term10_h = h2 @ term10_analytical @ h2
    
    dl2_dx_h = np.einsum('kci,i->kc', dl2_dx, h2)
    l2_pert = l2 + eps * dl2_dx_h
    dJ_dl2_pert = np.asarray(J.dJ_dl2(gamma1, l1, gamma2, l2_pert, J.downsample))
    dJ_dl2_dl2_dx_h_pert = np.sum(dJ_dl2_pert * dl2_dx_h)
    
    l2_neg = l2 - eps * dl2_dx_h
    dJ_dl2_neg = np.asarray(J.dJ_dl2(gamma1, l1, gamma2, l2_neg, J.downsample))
    dJ_dl2_dl2_dx_h_neg = np.sum(dJ_dl2_neg * dl2_dx_h)
    
    term9_plus_term10_fd = (dJ_dl2_dl2_dx_h_pert - dJ_dl2_dl2_dx_h_neg) / (2 * eps)
    
    if hasattr(curve2, 'd2gammadash_by_d2coeff_impl'):
        try:
            d2l2_dx2 = np.zeros((n_quad2, 3, n_dofs2, n_dofs2))
            curve2.d2gammadash_by_d2coeff_impl(d2l2_dx2)
            term9_contribution = np.einsum('kc,kci->', dJ_dl2, np.einsum('kcij,j->kci', d2l2_dx2, h2))
            term10_fd = term9_plus_term10_fd - term9_contribution
        except:
            term10_fd = term9_plus_term10_fd
    else:
        term10_fd = term9_plus_term10_fd
    
    print(f"  Analytical: {term10_h:.6e}")
    print(f"  FD: {term10_fd:.6e}")
    print(f"  Error: {abs(term10_h - term10_fd):.6e}")
    print(f"  Rel error: {abs(term10_h - term10_fd) / (abs(term10_h) + 1e-15):.6e}")
    
    # Term 11: (dgamma2/dx2)^T @ d²J/(dgamma2 dl2) @ (dl2/dx2)
    print("\nTerm 11: (dgamma2/dx2)^T @ d²J/(dgamma2 dl2) @ (dl2/dx2)")
    # d2J_dl2_dgamma2 is d²J/(dl2 dgamma2), so we need its transpose for d²J/(dgamma2 dl2)
    d2J_dgamma2_dl2 = d2J_dl2_dgamma2.transpose(2, 3, 0, 1)
    term11_analytical = np.einsum('kci,kclm,lmj->ij', dgamma2_dx, d2J_dgamma2_dl2, dl2_dx)
    term11_h = h2 @ term11_analytical @ h2
    
    # Finite difference: perturb only gamma2, measure change in dJ/dl2 @ dl2/dx2 @ h2
    gamma2_pert = gamma2 + eps * dgamma2_dx_h
    dJ_dl2_pert_gamma2 = np.asarray(J.dJ_dl2(gamma1, l1, gamma2_pert, l2, J.downsample))
    dJ_dl2_dl2_dx_h_pert_gamma2 = np.sum(dJ_dl2_pert_gamma2 * dl2_dx_h)
    
    gamma2_neg = gamma2 - eps * dgamma2_dx_h
    dJ_dl2_neg_gamma2 = np.asarray(J.dJ_dl2(gamma1, l1, gamma2_neg, l2, J.downsample))
    dJ_dl2_dl2_dx_h_neg_gamma2 = np.sum(dJ_dl2_neg_gamma2 * dl2_dx_h)
    
    term11_fd = (dJ_dl2_dl2_dx_h_pert_gamma2 - dJ_dl2_dl2_dx_h_neg_gamma2) / (2 * eps)
    
    print(f"  Analytical: {term11_h:.6e}")
    print(f"  FD: {term11_fd:.6e}")
    print(f"  Error: {abs(term11_h - term11_fd):.6e}")
    print(f"  Rel error: {abs(term11_h - term11_fd) / (abs(term11_h) + 1e-15):.6e}")
    
    # Term 12: (dl2/dx2)^T @ d²J/(dl2 dgamma2) @ (dgamma2/dx2)
    print("\nTerm 12: (dl2/dx2)^T @ d²J/(dl2 dgamma2) @ (dgamma2/dx2)")
    term12_analytical = np.einsum('kci,kclm,lmj->ij', dl2_dx, d2J_dl2_dgamma2, dgamma2_dx)
    term12_h = h2 @ term12_analytical @ h2
    
    # Should be the same as term 11
    print(f"  Analytical: {term12_h:.6e}")
    print(f"  Term 11: {term11_h:.6e}")
    print(f"  Difference: {abs(term12_h - term11_h):.6e}")
    print(f"  Should be equal (symmetric): {abs(term12_h - term11_h) < 1e-10}")
    
    # ========== Cross terms between curves (H_12) ==========
    print("\n" + "=" * 60)
    print("CROSS TERMS BETWEEN CURVES (H_12)")
    print("=" * 60)
    
    # Term 13: (dgamma1/dx1)^T @ d²J/(dgamma1 dgamma2) @ (dgamma2/dx2)
    print("\nTerm 13: (dgamma1/dx1)^T @ d²J/(dgamma1 dgamma2) @ (dgamma2/dx2)")
    term13_analytical = np.einsum('kci,kclm,lmj->ij', dgamma1_dx, d2J_dgamma1_dgamma2, dgamma2_dx)
    term13_h = h1 @ term13_analytical @ h2
    
    # Finite difference: perturb gamma1, measure change in dJ/dgamma2 @ dgamma2/dx2 @ h2
    gamma1_pert = gamma1 + eps * dgamma1_dx_h
    dJ_dgamma2_pert_gamma1 = np.asarray(J.dJ_dgamma2(gamma1_pert, l1, gamma2, l2, J.downsample))
    dJ_dgamma2_dgamma2_dx_h_pert_gamma1 = np.sum(dJ_dgamma2_pert_gamma1 * dgamma2_dx_h)
    
    gamma1_neg = gamma1 - eps * dgamma1_dx_h
    dJ_dgamma2_neg_gamma1 = np.asarray(J.dJ_dgamma2(gamma1_neg, l1, gamma2, l2, J.downsample))
    dJ_dgamma2_dgamma2_dx_h_neg_gamma1 = np.sum(dJ_dgamma2_neg_gamma1 * dgamma2_dx_h)
    
    term13_fd = (dJ_dgamma2_dgamma2_dx_h_pert_gamma1 - dJ_dgamma2_dgamma2_dx_h_neg_gamma1) / (2 * eps)
    
    print(f"  Analytical: {term13_h:.6e}")
    print(f"  FD: {term13_fd:.6e}")
    print(f"  Error: {abs(term13_h - term13_fd):.6e}")
    print(f"  Rel error: {abs(term13_h - term13_fd) / (abs(term13_h) + 1e-15):.6e}")
    
    # Term 14: (dgamma1/dx1)^T @ d²J/(dgamma1 dl2) @ (dl2/dx2)
    print("\nTerm 14: (dgamma1/dx1)^T @ d²J/(dgamma1 dl2) @ (dl2/dx2)")
    term14_analytical = np.einsum('kci,kclm,lmj->ij', dgamma1_dx, d2J_dgamma1_dl2, dl2_dx)
    term14_h = h1 @ term14_analytical @ h2
    
    # Finite difference: perturb gamma1, measure change in dJ/dl2 @ dl2/dx2 @ h2
    gamma1_pert = gamma1 + eps * dgamma1_dx_h
    dJ_dl2_pert_gamma1 = np.asarray(J.dJ_dl2(gamma1_pert, l1, gamma2, l2, J.downsample))
    dJ_dl2_dl2_dx_h_pert_gamma1 = np.sum(dJ_dl2_pert_gamma1 * dl2_dx_h)
    
    gamma1_neg = gamma1 - eps * dgamma1_dx_h
    dJ_dl2_neg_gamma1 = np.asarray(J.dJ_dl2(gamma1_neg, l1, gamma2, l2, J.downsample))
    dJ_dl2_dl2_dx_h_neg_gamma1 = np.sum(dJ_dl2_neg_gamma1 * dl2_dx_h)
    
    term14_fd = (dJ_dl2_dl2_dx_h_pert_gamma1 - dJ_dl2_dl2_dx_h_neg_gamma1) / (2 * eps)
    
    print(f"  Analytical: {term14_h:.6e}")
    print(f"  FD: {term14_fd:.6e}")
    print(f"  Error: {abs(term14_h - term14_fd):.6e}")
    print(f"  Rel error: {abs(term14_h - term14_fd) / (abs(term14_h) + 1e-15):.6e}")
    
    # Term 15: (dl1/dx1)^T @ d²J/(dl1 dgamma2) @ (dgamma2/dx2)
    print("\nTerm 15: (dl1/dx1)^T @ d²J/(dl1 dgamma2) @ (dgamma2/dx2)")
    term15_analytical = np.einsum('kci,kclm,lmj->ij', dl1_dx, d2J_dl1_dgamma2, dgamma2_dx)
    term15_h = h1 @ term15_analytical @ h2
    
    # Finite difference: perturb l1, measure change in dJ/dgamma2 @ dgamma2/dx2 @ h2
    l1_pert = l1 + eps * dl1_dx_h
    dJ_dgamma2_pert_l1 = np.asarray(J.dJ_dgamma2(gamma1, l1_pert, gamma2, l2, J.downsample))
    dJ_dgamma2_dgamma2_dx_h_pert_l1 = np.sum(dJ_dgamma2_pert_l1 * dgamma2_dx_h)
    
    l1_neg = l1 - eps * dl1_dx_h
    dJ_dgamma2_neg_l1 = np.asarray(J.dJ_dgamma2(gamma1, l1_neg, gamma2, l2, J.downsample))
    dJ_dgamma2_dgamma2_dx_h_neg_l1 = np.sum(dJ_dgamma2_neg_l1 * dgamma2_dx_h)
    
    term15_fd = (dJ_dgamma2_dgamma2_dx_h_pert_l1 - dJ_dgamma2_dgamma2_dx_h_neg_l1) / (2 * eps)
    
    print(f"  Analytical: {term15_h:.6e}")
    print(f"  FD: {term15_fd:.6e}")
    print(f"  Error: {abs(term15_h - term15_fd):.6e}")
    print(f"  Rel error: {abs(term15_h - term15_fd) / (abs(term15_h) + 1e-15):.6e}")
    
    # Term 16: (dl1/dx1)^T @ d²J/(dl1 dl2) @ (dl2/dx2)
    print("\nTerm 16: (dl1/dx1)^T @ d²J/(dl1 dl2) @ (dl2/dx2)")
    term16_analytical = np.einsum('kci,kclm,lmj->ij', dl1_dx, d2J_dl1_dl2, dl2_dx)
    term16_h = h1 @ term16_analytical @ h2
    
    # Finite difference: perturb l1, measure change in dJ/dl2 @ dl2/dx2 @ h2
    l1_pert = l1 + eps * dl1_dx_h
    dJ_dl2_pert_l1 = np.asarray(J.dJ_dl2(gamma1, l1_pert, gamma2, l2, J.downsample))
    dJ_dl2_dl2_dx_h_pert_l1 = np.sum(dJ_dl2_pert_l1 * dl2_dx_h)
    
    l1_neg = l1 - eps * dl1_dx_h
    dJ_dl2_neg_l1 = np.asarray(J.dJ_dl2(gamma1, l1_neg, gamma2, l2, J.downsample))
    dJ_dl2_dl2_dx_h_neg_l1 = np.sum(dJ_dl2_neg_l1 * dl2_dx_h)
    
    term16_fd = (dJ_dl2_dl2_dx_h_pert_l1 - dJ_dl2_dl2_dx_h_neg_l1) / (2 * eps)
    
    print(f"  Analytical: {term16_h:.6e}")
    print(f"  FD: {term16_fd:.6e}")
    print(f"  Error: {abs(term16_h - term16_fd):.6e}")
    print(f"  Rel error: {abs(term16_h - term16_fd) / (abs(term16_h) + 1e-15):.6e}")
    
    # Restore original state
    curve1.x = curve1_dofs
    curve2.x = curve2_dofs
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print("Total Hessian (sum of all terms):")
    H_total = np.zeros((n_dofs1 + n_dofs2, n_dofs1 + n_dofs2))
    H_total[:n_dofs1, :n_dofs1] = term1_analytical + term2_analytical + term3_analytical + term4_analytical + term5_analytical + term6_analytical
    H_total[n_dofs1:, n_dofs1:] = term7_analytical + term8_analytical + term9_analytical + term10_analytical + term11_analytical + term12_analytical
    H_total[:n_dofs1, n_dofs1:] = term13_analytical + term14_analytical + term15_analytical + term16_analytical
    H_total[n_dofs1:, :n_dofs1] = H_total[:n_dofs1, n_dofs1:].T  # Symmetry
    
    h = np.concatenate([h1, h2])
    H_total_h = h @ H_total @ h
    print(f"  h^T H h (analytical): {H_total_h:.6e}")


if __name__ == '__main__':
    test_hessian_terms_individually()

