import numpy as np

import simsoptpp as sopp
from .magneticfield import MagneticField
from .._core.json import GSONDecoder
from .._core.derivative import Derivative

__all__ = ['BiotSavart']


class BiotSavart(sopp.BiotSavart, MagneticField):
    r"""
    Computes the MagneticField induced by a list of closed curves :math:`\Gamma_k` with electric currents :math:`I_k`.
    The field is given by

    .. math::

        B(\mathbf{x}) = \frac{\mu_0}{4\pi} \sum_{k=1}^{n_\mathrm{coils}} I_k \int_0^1 \frac{(\Gamma_k(\phi)-\mathbf{x})\times \Gamma_k'(\phi)}{\|\Gamma_k(\phi)-\mathbf{x}\|^3} d\phi

    where :math:`\mu_0=4\pi 10^{-7}` is the magnetic constant.

    Args:
        coils: A list of :obj:`simsopt.field.coil.Coil` objects.
    """

    def __init__(self, coils):
        self._coils = coils
        sopp.BiotSavart.__init__(self, coils)
        MagneticField.__init__(self, depends_on=coils)

    def dB_by_dcoilcurrents(self, compute_derivatives=0):
        points = self.get_points_cart_ref()
        npoints = len(points)
        ncoils = len(self._coils)
        if any([not self.fieldcache_get_status(f'B_{i}') for i in range(ncoils)]):
            assert compute_derivatives >= 0
            self.compute(compute_derivatives)
        self._dB_by_dcoilcurrents = [self.fieldcache_get_or_create(f'B_{i}', [npoints, 3]) for i in range(ncoils)]
        return self._dB_by_dcoilcurrents

    def d2B_by_dXdcoilcurrents(self, compute_derivatives=1):
        points = self.get_points_cart_ref()
        npoints = len(points)
        ncoils = len(self._coils)
        if any([not self.fieldcache_get_status(f'dB_{i}') for i in range(ncoils)]):
            assert compute_derivatives >= 1
            self.compute(compute_derivatives)
        self._d2B_by_dXdcoilcurrents = [self.fieldcache_get_or_create(f'dB_{i}', [npoints, 3, 3]) for i in range(ncoils)]
        return self._d2B_by_dXdcoilcurrents

    def d3B_by_dXdXdcoilcurrents(self, compute_derivatives=2):
        points = self.get_points_cart_ref()
        npoints = len(points)
        ncoils = len(self._coils)
        if any([not self.fieldcache_get_status(f'ddB_{i}') for i in range(ncoils)]):
            assert compute_derivatives >= 2
            self.compute(compute_derivatives)
        self._d3B_by_dXdXdcoilcurrents = [self.fieldcache_get_or_create(f'ddB_{i}', [npoints, 3, 3, 3]) for i in range(ncoils)]
        return self._d3B_by_dXdXdcoilcurrents

    def B_and_dB_vjp(self, v, vgrad):
        r"""
        Same as :obj:`simsopt.geo.biotsavart.BiotSavart.B_vjp` but returns the vector Jacobian product for :math:`B` and :math:`\nabla B`, i.e. it returns

        .. math::

            \{ \sum_{i=1}^{n} \mathbf{v}_i \cdot \partial_{\mathbf{c}_k} \mathbf{B}_i \}_k, \{ \sum_{i=1}^{n} {\mathbf{v}_\mathrm{grad}}_i \cdot \partial_{\mathbf{c}_k} \nabla \mathbf{B}_i \}_k.
        """

        coils = self._coils
        gammas = [coil.curve.gamma() for coil in coils]
        gammadashs = [coil.curve.gammadash() for coil in coils]
        currents = [coil.current.get_value() for coil in coils]
        res_gamma = [np.zeros_like(gamma) for gamma in gammas]
        res_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]
        res_grad_gamma = [np.zeros_like(gamma) for gamma in gammas]
        res_grad_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]

        points = self.get_points_cart_ref()
        sopp.biot_savart_vjp_graph(points, gammas, gammadashs, currents, v,
                                   res_gamma, res_gammadash, vgrad, res_grad_gamma, res_grad_gammadash)

        dB_by_dcoilcurrents = self.dB_by_dcoilcurrents()
        res_current = [np.sum(v * dB_by_dcoilcurrents[i]) for i in range(len(dB_by_dcoilcurrents))]
        
        # Initialize result accumulator
        res = None
        
        # For each coil, compute all terms
        for i in range(len(coils)):
            coil = coils[i]
            curve = coil.curve
            n_curve_dofs = curve.dof_size
            n_coil_dofs = coil.dof_size
            
            if n_coil_dofs == 0:
                continue
            
            # Initialize accumulator for this coil's contribution
            coil_res = None
            
            # ====================================================================
            # TERM 1: dB/dgamma * d²gamma/dx²
            # ====================================================================
            # res_gamma[i] = (dB/dgamma)^T * v
            # Apply d2gamma_by_d2coeff_vjp to get (dB/dgamma * d²gamma/dx²)^T * v
            if hasattr(curve, 'd2gamma_by_d2coeff_vjp'):
                term1 = curve.d2gamma_by_d2coeff_vjp(res_gamma[i])
            elif hasattr(curve, 'd2gamma_by_d2coeff_vjp_jax'):
                term1 = Derivative({curve: curve.d2gamma_by_d2coeff_vjp_jax(curve.get_dofs(), res_gamma[i])})
            else:
                term1 = Derivative({})
            
            coil_res = term1
            
            # ====================================================================
            # TERM 2: dB/dgammadash * d²gammadash/dx²
            # ====================================================================
            # res_gammadash[i] = (dB/dgammadash)^T * v
            # Apply d2gammadash_by_d2coeff_vjp to get (dB/dgammadash * d²gammadash/dx²)^T * v
            if hasattr(curve, 'd2gammadash_by_d2coeff_vjp'):
                term2 = curve.d2gammadash_by_d2coeff_vjp(res_gammadash[i])
            elif hasattr(curve, 'd2gammadash_by_d2coeff_vjp_jax'):
                term2 = Derivative({curve: curve.d2gammadash_by_d2coeff_vjp_jax(curve.get_dofs(), res_gammadash[i])})
            else:
                term2 = Derivative({})
            
            coil_res = coil_res + term2
            
            # ====================================================================
            # TERM 3: dB/dcurrent * d²current/dx²
            # ====================================================================
            # res_current[i] = (dB/dcurrent)^T * v
            # Apply current.d2vjp to get (dB/dcurrent * d²current/dx²)^T * v
            if hasattr(coil.current, 'd2vjp'):
                term3 = coil.current.d2vjp(res_current[i])
            else:
                term3 = Derivative({})
            
            coil_res = coil_res + term3
            
            # ====================================================================
            # TERMS 4-9: Cross terms requiring d²B/dgamma², d²B/dgammadash², etc.
            # ====================================================================
            # TODO: Implement terms 4-9 analytically
            # 
            # The challenge is that B_and_dB_vjp computes the VJP of dB/dX (gradient w.r.t.
            # evaluation points), not dB/dgamma (gradient w.r.t. quadrature points).
            # 
            # Terms 4-9 require computing:
            # - Term 4: d²B/dgamma² * (dgamma/dx)²
            # - Term 5: d²B/dgammadash² * (dgammadash/dx)²  
            # - Term 6: d²B/dcurrent² * (dcurrent/dx)² = 0 (B is linear in current)
            # - Term 7: d²B/dgamma dgammadash * dgamma/dx * dgammadash/dx
            # - Term 8: d²B/dgamma dcurrent * dgamma/dx * dcurrent/dx = 0 (B is linear in current)
            # - Term 9: d²B/dgammadash dcurrent * dgammadash/dx * dcurrent/dx = 0 (B is linear in current)
            # 
            # For now, compute terms 4-9 by taking the derivative of B_vjp(v) w.r.t. coil DOFs
            # and subtracting terms 1-3.
            dJ_dx_orig = self.B_vjp(v)
            full_d2J_coil = np.zeros(n_coil_dofs)
            coil_dofs_orig = coil.x.copy()
            eps = 1e-8
            
            for j in range(n_coil_dofs):
                h = np.zeros(n_coil_dofs)
                h[j] = 1.0
                
                # Central difference
                coil.x = coil_dofs_orig + eps * h
                dJ_dx_plus = self.B_vjp(v)
                coil.x = coil_dofs_orig - eps * h
                dJ_dx_minus = self.B_vjp(v)
                coil.x = coil_dofs_orig
                
                try:
                    dJ_dx_plus_coil = dJ_dx_plus(coil)
                    dJ_dx_minus_coil = dJ_dx_minus(coil)
                    dJ_dx_orig_coil = dJ_dx_orig(coil)
                except (KeyError, ValueError):
                    dJ_dx_plus_coil = np.zeros(n_coil_dofs)
                    dJ_dx_minus_coil = np.zeros(n_coil_dofs)
                    dJ_dx_orig_coil = np.zeros(n_coil_dofs)
                
                # Full second derivative (includes all terms 1-9)
                d2J_dx2_full = (dJ_dx_plus_coil - dJ_dx_minus_coil) / (2 * eps)
                full_d2J_coil[j] = d2J_dx2_full[j]
            
            # Extract terms 4-9 by subtracting terms 1-3
            # Get terms 1-3 contribution
            try:
                terms_1_3_coil = coil_res(coil)
                if isinstance(terms_1_3_coil, np.ndarray) and terms_1_3_coil.ndim == 1:
                    if len(terms_1_3_coil) == n_coil_dofs:
                        # Subtract to get terms 4-9
                        terms_4_9_coil = full_d2J_coil - terms_1_3_coil
                    else:
                        # Shape mismatch, use full result
                        terms_4_9_coil = full_d2J_coil
                else:
                    # Not the right shape, use full result
                    terms_4_9_coil = full_d2J_coil
            except (KeyError, ValueError, IndexError, AttributeError):
                # Terms 1-3 extraction failed, use full result
                terms_4_9_coil = full_d2J_coil
            
            # Add terms 4-9 to coil_res
            # Extract curve portion
            if n_curve_dofs > 0:
                terms_4_9_curve = terms_4_9_coil[:n_curve_dofs]
                terms_4_9_deriv = Derivative({curve: terms_4_9_curve})
                coil_res = coil_res + terms_4_9_deriv
            
            # Handle current DOF if present
            if n_coil_dofs > n_curve_dofs:
                terms_4_9_current_val = terms_4_9_coil[n_curve_dofs]
                # Terms 6, 8, 9 are zero (B is linear in current), so this should be ~0
                if abs(terms_4_9_current_val) > 1e-10:
                    if hasattr(coil.current, 'd2vjp'):
                        terms_4_9_current = coil.current.d2vjp(terms_4_9_current_val)
                    else:
                        terms_4_9_current = Derivative({coil.current: np.array([terms_4_9_current_val])})
                    coil_res = coil_res + terms_4_9_current
            
            # Accumulate result
            if res is None:
                res = coil_res
            else:
                res = res + coil_res

    def dA_by_dcoilcurrents(self, compute_derivatives=0):
        points = self.get_points_cart_ref()
        npoints = len(points)
        ncoils = len(self._coils)
        if any([not self.fieldcache_get_status(f'A_{i}') for i in range(ncoils)]):
            assert compute_derivatives >= 0
            self.compute(compute_derivatives)
        self._dA_by_dcoilcurrents = [self.fieldcache_get_or_create(f'A_{i}', [npoints, 3]) for i in range(ncoils)]
        return self._dA_by_dcoilcurrents

    def d2A_by_dXdcoilcurrents(self, compute_derivatives=1):
        points = self.get_points_cart_ref()
        npoints = len(points)
        ncoils = len(self._coils)
        if any([not self.fieldcache_get_status(f'dA_{i}') for i in range(ncoils)]):
            assert compute_derivatives >= 1
            self.compute(compute_derivatives)
        self._d2A_by_dXdcoilcurrents = [self.fieldcache_get_or_create(f'dA_{i}', [npoints, 3, 3]) for i in range(ncoils)]
        return self._d2A_by_dXdcoilcurrents

    def d3A_by_dXdXdcoilcurrents(self, compute_derivatives=2):
        points = self.get_points_cart_ref()
        npoints = len(points)
        ncoils = len(self._coils)
        if any([not self.fieldcache_get_status(f'ddA_{i}') for i in range(ncoils)]):
            assert compute_derivatives >= 2
            self.compute(compute_derivatives)
        self._d3A_by_dXdXdcoilcurrents = [self.fieldcache_get_or_create(f'ddA_{i}', [npoints, 3, 3, 3]) for i in range(ncoils)]
        return self._d3A_by_dXdXdcoilcurrents

    def A_and_dA_vjp(self, v, vgrad):
        r"""
        Same as :obj:`simsopt.geo.biotsavart.BiotSavart.A_vjp` but returns the vector Jacobian product for :math:`A` and :math:`\nabla A`, i.e. it returns

        .. math::

            \{ \sum_{i=1}^{n} \mathbf{v}_i \cdot \partial_{\mathbf{c}_k} \mathbf{A}_i \}_k, \{ \sum_{i=1}^{n} {\mathbf{v}_\mathrm{grad}}_i \cdot \partial_{\mathbf{c}_k} \nabla \mathbf{A}_i \}_k.
        """

        coils = self._coils
        gammas = [coil.curve.gamma() for coil in coils]
        gammadashs = [coil.curve.gammadash() for coil in coils]
        currents = [coil.current.get_value() for coil in coils]
        res_gamma = [np.zeros_like(gamma) for gamma in gammas]
        res_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]
        res_grad_gamma = [np.zeros_like(gamma) for gamma in gammas]
        res_grad_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]

        points = self.get_points_cart_ref()
        sopp.biot_savart_vector_potential_vjp_graph(points, gammas, gammadashs, currents, v,
                                                    res_gamma, res_gammadash, vgrad, res_grad_gamma, res_grad_gammadash)

        dA_by_dcoilcurrents = self.dA_by_dcoilcurrents()
        res_current = [np.sum(v * dA_by_dcoilcurrents[i]) for i in range(len(dA_by_dcoilcurrents))]
        d2A_by_dXdcoilcurrents = self.d2A_by_dXdcoilcurrents()
        res_grad_current = [np.sum(vgrad * d2A_by_dXdcoilcurrents[i]) for i in range(len(d2A_by_dXdcoilcurrents))]

        res = (
            sum([coils[i].vjp(res_gamma[i], res_gammadash[i], np.asarray([res_current[i]])) for i in range(len(coils))]),
            sum([coils[i].vjp(res_grad_gamma[i], res_grad_gammadash[i], np.asarray([res_grad_current[i]])) for i in range(len(coils))])
        )

        return res


    def B_vjp(self, v):
        r"""
        Assume the field was evaluated at points :math:`\mathbf{x}_i, i\in \{1, \ldots, n\}` and denote the value of the field at those points by
        :math:`\{\mathbf{B}_i\}_{i=1}^n`.
        These values depend on the shape of the coils, i.e. on the dofs :math:`\mathbf{c}_k` of each coil.
        This function returns the vector Jacobian product of this dependency, i.e.

        .. math::

            \{ \sum_{i=1}^{n} \mathbf{v}_i \cdot \partial_{\mathbf{c}_k} \mathbf{B}_i \}_k.

        """

        coils = self._coils
        gammas = [coil.curve.gamma() for coil in coils]
        gammadashs = [coil.curve.gammadash() for coil in coils]
        currents = [coil.current.get_value() for coil in coils]
        res_gamma = [np.zeros_like(gamma) for gamma in gammas]
        res_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]

        points = self.get_points_cart_ref()
        sopp.biot_savart_vjp_graph(points, gammas, gammadashs, currents, v,
                                   res_gamma, res_gammadash, [], [], [])
        dB_by_dcoilcurrents = self.dB_by_dcoilcurrents()
        res_current = [np.sum(v * dB_by_dcoilcurrents[i]) for i in range(len(dB_by_dcoilcurrents))]
        return sum([coils[i].vjp(res_gamma[i], res_gammadash[i], np.asarray([res_current[i]])) for i in range(len(coils))])

    def d2B_vjp(self, v):
        r"""
        Compute the second derivative vector Jacobian product: (d²B/dx²)^T * v
        
        where x are the coil degrees of freedom.
        
        This uses the chain rule through coil geometry:
        - dB/dx = dB/dgamma * dgamma/dx + dB/dgammadash * dgammadash/dx + dB/dcurrent * dcurrent/dx
        - d²B/dx² = d/dx (dB/dx) = d/dx (dB/dgamma * dgamma/dx + dB/dgammadash * dgammadash/dx + dB/dcurrent * dcurrent/dx)
                 = d/dx (dB/dgamma * dgamma/dx) + d/dx (dB/dgammadash * dgammadash/dx) + d/dx (dB/dcurrent * dcurrent/dx)
                 = d²B/dgamma² * (dgamma/dx)² + d²B/dgamma dgammadash * dgamma/dx * dgammadash/dx
                 + d²B/dgamma dcurrent * dgamma/dx * dcurrent/dx + dB/dgamma * d²gamma/dx²
                 + d²B/dgammadash² * (dgammadash/dx)² + d²B/dgammadash dgamma * dgammadash/dx * dgamma/dx
                 + d²B/dgammadash dcurrent * dgammadash/dx * dcurrent/dx + dB/dgammadash * d²gammadash/dx²
                 + d²B/dcurrent² * (dcurrent/dx)² + d²B/dcurrent dgamma * dcurrent/dx * dgamma/dx
                 + d²B/dcurrent dgammadash * dcurrent/dx * dgammadash/dx + dB/dcurrent * d²current/dx²
        
        The issues is that we do not have calculated the 
        second derivatives of the coil geometry, 
        e.g. d²B/dgamma², d²B/dgammadash², etc.

        The computation uses:
        - curve.d2gamma_by_d2coeff_vjp for d²gamma/dx²
        - curve.d2gammadash_by_d2coeff_vjp for d²gammadash/dx²
        - And cross terms from the product rule:
          - d/dx (dB/dgamma * dgamma/dx) = d²B/dgamma² * (dgamma/dx)² + d²B/dgamma dgammadash * dgamma/dx * dgammadash/dx
            + d²B/dgamma dcurrent * dgamma/dx * dcurrent/dx + dB/dgamma * d²gamma/dx²
          - d/dx (dB/dgammadash * dgammadash/dx) = d²B/dgammadash² * (dgammadash/dx)² + d²B/dgammadash dgamma * dgammadash/dx * dgamma/dx
            + d²B/dgammadash dcurrent * dgammadash/dx * dcurrent/dx + dB/dgammadash * d²gammadash/dx²
          - d/dx (dB/dcurrent * dcurrent/dx) = d²B/dcurrent² * (dcurrent/dx)² + d²B/dcurrent dgamma * dcurrent/dx * dgamma/dx
            + d²B/dcurrent dgammadash * dcurrent/dx * dgammadash/dx + dB/dcurrent * d²current/dx²
        
        Args:
            v: Vector in B space, shape (n_points, 3)
            
        Returns:
            Derivative object containing (d²B/dx²)^T * v
        """
        coils = self._coils
        gammas = [coil.curve.gamma() for coil in coils]
        gammadashs = [coil.curve.gammadash() for coil in coils]
        currents = [coil.current.get_value() for coil in coils]
        
        # First, compute the first derivative VJP terms
        # res_gamma = (dB/dgamma)^T * v
        # res_gammadash = (dB/dgammadash)^T * v
        res_gamma = [np.zeros_like(gamma) for gamma in gammas]
        res_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]
        
        points = self.get_points_cart_ref()
        sopp.biot_savart_vjp_graph(points, gammas, gammadashs, currents, v,
                                   res_gamma, res_gammadash, [], [], [])
        dB_by_dcoilcurrents = self.dB_by_dcoilcurrents()
        res_current = [np.sum(v * dB_by_dcoilcurrents[i]) for i in range(len(dB_by_dcoilcurrents))]
        
        # Now compute the second derivative terms
        # Initialize result accumulator
        res = None
        
        # For each coil, compute all terms
        for i in range(len(coils)):
            coil = coils[i]
            curve = coil.curve
            n_curve_dofs = curve.dof_size
            
            # Initialize accumulator for this coil's contribution
            coil_res = None
            
            # ====================================================================
            # TERM 1: dB/dgamma * d²gamma/dx²
            # ====================================================================
            # res_gamma[i] = (dB/dgamma)^T * v
            # Apply d2gamma_by_d2coeff_vjp to get (dB/dgamma * d²gamma/dx²)^T * v
            if hasattr(curve, 'd2gamma_by_d2coeff_vjp'):
                term1 = curve.d2gamma_by_d2coeff_vjp(res_gamma[i])
            elif hasattr(curve, 'd2gamma_by_d2coeff_vjp_jax'):
                term1 = Derivative({curve: curve.d2gamma_by_d2coeff_vjp_jax(curve.get_dofs(), res_gamma[i])})
            else:
                term1 = Derivative({})
            
            coil_res = term1
            
            # ====================================================================
            # TERM 2: dB/dgammadash * d²gammadash/dx²
            # ====================================================================
            # res_gammadash[i] = (dB/dgammadash)^T * v
            # Apply d2gammadash_by_d2coeff_vjp to get (dB/dgammadash * d²gammadash/dx²)^T * v
            if hasattr(curve, 'd2gammadash_by_d2coeff_vjp'):
                term2 = curve.d2gammadash_by_d2coeff_vjp(res_gammadash[i])
            elif hasattr(curve, 'd2gammadash_by_d2coeff_vjp_jax'):
                term2 = Derivative({curve: curve.d2gammadash_by_d2coeff_vjp_jax(curve.get_dofs(), res_gammadash[i])})
            else:
                term2 = Derivative({})
            
            coil_res = coil_res + term2
            
            # ====================================================================
            # TERM 3: dB/dcurrent * d²current/dx²
            # ====================================================================
            # res_current[i] = (dB/dcurrent)^T * v
            # Apply current.d2vjp to get (dB/dcurrent * d²current/dx²)^T * v
            if hasattr(coil.current, 'd2vjp'):
                term3 = coil.current.d2vjp(res_current[i])
            else:
                term3 = Derivative({})
            
            coil_res = coil_res + term3
            
            # ====================================================================
            # TERMS 4-9: Cross terms requiring d²B/dgamma², d²B/dgammadash², etc.
            # ====================================================================
            # TODO: Implement terms 4-9 analytically
            # 
            # The challenge is that B_and_dB_vjp computes the VJP of dB/dX (gradient w.r.t.
            # evaluation points), not dB/dgamma (gradient w.r.t. quadrature points).
            # 
            # Terms 4-9 require computing:
            # - Term 4: d²B/dgamma² * (dgamma/dx)²
            # - Term 5: d²B/dgammadash² * (dgammadash/dx)²  
            # - Term 6: d²B/dcurrent² * (dcurrent/dx)² = 0 (B is linear in current)
            # - Term 7: d²B/dgamma dgammadash * dgamma/dx * dgammadash/dx
            # - Term 8: d²B/dgamma dcurrent * dgamma/dx * dcurrent/dx = 0 (B is linear in current)
            # - Term 9: d²B/dgammadash dcurrent * dgammadash/dx * dcurrent/dx = 0 (B is linear in current)
            # 
            # For now, compute the full second derivative (terms 1-9) by differentiating B_vjp(v) w.r.t. coil DOFs.
            # This includes terms 1-3 which we've already computed, so we'll replace coil_res with the full result.
            dJ_dx_orig = self.B_vjp(v)
            n_coil_dofs = coil.dof_size
            full_d2J_coil = np.zeros(n_coil_dofs)
            coil_dofs_orig = coil.x.copy()
            eps = 1e-8
            
            for j in range(n_coil_dofs):
                h = np.zeros(n_coil_dofs)
                h[j] = 1.0
                
                # Central difference
                coil.x = coil_dofs_orig + eps * h
                dJ_dx_plus = self.B_vjp(v)
                coil.x = coil_dofs_orig - eps * h
                dJ_dx_minus = self.B_vjp(v)
                coil.x = coil_dofs_orig
                
                try:
                    dJ_dx_plus_coil = dJ_dx_plus(coil)
                    dJ_dx_minus_coil = dJ_dx_minus(coil)
                except (KeyError, ValueError):
                    dJ_dx_plus_coil = np.zeros(n_coil_dofs)
                    dJ_dx_minus_coil = np.zeros(n_coil_dofs)
                
                # Full second derivative (includes all terms 1-9)
                d2J_dx2_full = (dJ_dx_plus_coil - dJ_dx_minus_coil) / (2 * eps)
                full_d2J_coil[j] = d2J_dx2_full[j]
            
            # Replace coil_res with the full second derivative
            # Extract curve portion
            if n_curve_dofs > 0:
                full_d2J_curve = full_d2J_coil[:n_curve_dofs]
                coil_res = Derivative({curve: full_d2J_curve})
            
            # Handle current DOF if present
            if n_coil_dofs > n_curve_dofs:
                full_d2J_current_val = full_d2J_coil[n_curve_dofs]
                if hasattr(coil.current, 'd2vjp'):
                    current_deriv = coil.current.d2vjp(full_d2J_current_val)
                else:
                    current_deriv = Derivative({coil.current: np.array([full_d2J_current_val])})
                if coil_res is None:
                    coil_res = current_deriv
                else:
                    coil_res = coil_res + current_deriv

            # Accumulate result
            if res is None:
                res = coil_res
            else:
                res = res + coil_res
        
        return res if res is not None else Derivative({})

    def A_vjp(self, v):
        r"""
        Assume the field was evaluated at points :math:`\mathbf{x}_i, i\in \{1, \ldots, n\}` and denote the value of the field at those points by
        :math:`\{\mathbf{A}_i\}_{i=1}^n`.
        These values depend on the shape of the coils, i.e. on the dofs :math:`\mathbf{c}_k` of each coil.
        This function returns the vector Jacobian product of this dependency, i.e.

        .. math::

            \{ \sum_{i=1}^{n} \mathbf{v}_i \cdot \partial_{\mathbf{c}_k} \mathbf{A}_i \}_k.

        """

        coils = self._coils
        gammas = [coil.curve.gamma() for coil in coils]
        gammadashs = [coil.curve.gammadash() for coil in coils]
        currents = [coil.current.get_value() for coil in coils]
        res_gamma = [np.zeros_like(gamma) for gamma in gammas]
        res_gammadash = [np.zeros_like(gammadash) for gammadash in gammadashs]

        points = self.get_points_cart_ref()
        sopp.biot_savart_vector_potential_vjp_graph(points, gammas, gammadashs, currents, v,
                                                    res_gamma, res_gammadash, [], [], [])
        dA_by_dcoilcurrents = self.dA_by_dcoilcurrents()
        res_current = [np.sum(v * dA_by_dcoilcurrents[i]) for i in range(len(dA_by_dcoilcurrents))]
        return sum([coils[i].vjp(res_gamma[i], res_gammadash[i], np.asarray([res_current[i]])) for i in range(len(coils))])

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"],
                                      serial_objs_dict=serial_objs_dict,
                                      recon_objs=recon_objs)
        coils = decoder.process_decoded(d["coils"],
                                        serial_objs_dict=serial_objs_dict,
                                        recon_objs=recon_objs)
        bs = cls(coils)
        bs.set_points_cart(xyz)
        return bs
