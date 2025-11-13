import numpy as np
import jax.numpy as jnp
from jax import jit, vjp, jacfwd, vmap

from .magneticfield import MagneticField

__all__ = ['JaxBiotSavart']


def biot_savart_single_coil_pure(points, gamma, gammadash, current):
    r"""
    Pure JAX function for computing Biot-Savart magnetic field from a single coil.
    
    Args:
        points: Evaluation points, shape (n_points, 3)
        gamma: Curve positions, shape (n_quad, 3)
        gammadash: Curve derivatives, shape (n_quad, 3)
        current: Current (scalar)
    
    Returns:
        Magnetic field B at points, shape (n_points, 3)
    """
    mu0_over_4pi = 1e-7  # mu_0 / (4*pi) = 4*pi*1e-7 / (4*pi) = 1e-7
    n_quad = gamma.shape[0]
    n_points = points.shape[0]
    
    # Normalize by number of quadrature points (trapezoidal rule)
    dphi = 1.0 / n_quad
    
    # Vectorize over quadrature points
    # points: (n_points, 3)
    # gamma: (n_quad, 3)
    # gammadash: (n_quad, 3)
    
    # Expand dimensions for broadcasting
    # points_expanded: (n_points, 1, 3)
    # gamma_expanded: (1, n_quad, 3)
    points_expanded = points[:, None, :]  # (n_points, 1, 3)
    gamma_expanded = gamma[None, :, :]    # (1, n_quad, 3)
    gammadash_expanded = gammadash[None, :, :]  # (1, n_quad, 3)
    
    # diff: (n_points, n_quad, 3)
    diff = points_expanded - gamma_expanded
    
    # norm_diff: (n_points, n_quad)
    norm_diff_sq = jnp.sum(diff**2, axis=2)
    norm_diff = jnp.sqrt(norm_diff_sq + 1e-15)  # Add small epsilon for numerical stability
    norm_diff_3 = norm_diff**3
    
    # Cross product: gammadash × diff
    # For each (point, quad) pair: cross(gammadash[j], diff[i, j])
    # Use broadcasting: gammadash_expanded: (1, n_quad, 3), diff: (n_points, n_quad, 3)
    cross_prod = jnp.cross(gammadash_expanded, diff)  # (n_points, n_quad, 3)
    
    # B contribution: current * cross_prod / norm_diff^3 * dphi
    # Sum over quadrature points
    B = current * jnp.sum(cross_prod / norm_diff_3[:, :, None] * dphi, axis=1)  # (n_points, 3)
    
    return mu0_over_4pi * B


def biot_savart_pure(points, gammas, gammadashs, currents):
    r"""
    Pure JAX function for computing Biot-Savart magnetic field from multiple coils.
    
    Computes:
    
    .. math::
        B(\mathbf{x}) = \frac{\mu_0}{4\pi} \sum_{k=1}^{n_\mathrm{coils}} I_k \int_0^1 \frac{(\Gamma_k(\phi)-\mathbf{x})\times \Gamma_k'(\phi)}{\|\Gamma_k(\phi)-\mathbf{x}\|^3} d\phi
    
    where :math:`\mu_0=4\pi 10^{-7}` is the magnetic constant.
    
    Args:
        points: Evaluation points, shape (n_points, 3)
        gammas: List of curve positions, each shape (n_quad, 3)
        gammadashs: List of curve derivatives, each shape (n_quad, 3)
        currents: Array of currents, shape (n_coils,)
    
    Returns:
        Magnetic field B at points, shape (n_points, 3)
    """
    # Convert lists to arrays if needed for vmap
    if isinstance(gammas, list):
        # Stack into array: (n_coils, n_quad, 3)
        gammas_array = jnp.stack(gammas)
        gammadashs_array = jnp.stack(gammadashs)
    else:
        gammas_array = gammas
        gammadashs_array = gammadashs
    
    if isinstance(currents, (list, tuple)):
        currents_array = jnp.asarray(currents)
    else:
        currents_array = currents
    
    # Use vmap to vectorize over coils - this is faster than a Python loop
    # vmap applies biot_savart_single_coil_pure to each coil in parallel
    B_per_coil = vmap(biot_savart_single_coil_pure, in_axes=(None, 0, 0, 0))(
        points, gammas_array, gammadashs_array, currents_array
    )  # Shape: (n_coils, n_points, 3)
    
    # Sum contributions from all coils
    B = jnp.sum(B_per_coil, axis=0)  # Shape: (n_points, 3)
    
    return B

class JaxBiotSavart(MagneticField):
    r"""
    JAX-based implementation of Biot-Savart magnetic field computation.
    
    This class uses pure JAX functions with JIT compilation for computing the
    Biot-Savart magnetic field. It provides the same interface as :obj:`BiotSavart`
    but uses JAX for automatic differentiation.
    
    Computes the MagneticField induced by a list of closed curves :math:`\Gamma_k` 
    with electric currents :math:`I_k`. The field is given by
    
    .. math::
        B(\mathbf{x}) = \frac{\mu_0}{4\pi} \sum_{k=1}^{n_\mathrm{coils}} I_k \int_0^1 \frac{(\Gamma_k(\phi)-\mathbf{x})\times \Gamma_k'(\phi)}{\|\Gamma_k(\phi)-\mathbf{x}\|^3} d\phi
    
    where :math:`\mu_0=4\pi 10^{-7}` is the magnetic constant.
    
    Args:
        coils: A list of :obj:`simsopt.field.coil.Coil` objects.
    """
    
    def __init__(self, coils):
        self._coils = coils
        MagneticField.__init__(self, depends_on=coils)
        
        # Cache for computed values
        self._points = None
        self._B_cache = None
        self._dB_cache = None
        self._d2B_cache = None
        
        # Pre-jitted functions for performance
        # B function
        self.B_jax = jit(lambda points, gammas, gammadashs, currents: biot_savart_pure(
            points, gammas, gammadashs, currents))
        
        # dB_by_dX: gradient w.r.t. points (argnums=0)
        # Use jacfwd with argnums to differentiate w.r.t. first argument (points)
        def dB_by_dX_fn(points, gammas, gammadashs, currents):
            def B_fn(p):
                return biot_savart_pure(p, gammas, gammadashs, currents)
            dB_val = jacfwd(B_fn)(points)  # Shape: (n_points, 3, n_points, 3)
            # Extract diagonal: dB[i, j] / dX[i, :]
            n_points = points.shape[0]
            return jnp.array([dB_val[i, :, i, :] for i in range(n_points)])
        self.dB_by_dX_jax = jit(dB_by_dX_fn)
        
        # d2B_by_dXdX: Hessian w.r.t. points (argnums=0)
        def d2B_by_dXdX_fn(points, gammas, gammadashs, currents):
            def B_fn(p):
                return biot_savart_pure(p, gammas, gammadashs, currents)
            d2B_val = jacfwd(jacfwd(B_fn))(points)  # Shape: (n_points, 3, n_points, 3, n_points, 3)
            # Extract diagonal: d2B[i, j, k1, k2] = d²B[i, j] / (dX[i, k1] dX[i, k2])
            n_points = points.shape[0]
            d2B_diag = jnp.zeros((n_points, 3, 3, 3))
            for i in range(n_points):
                for j in range(3):
                    for k1 in range(3):
                        for k2 in range(3):
                            d2B_diag = d2B_diag.at[i, j, k1, k2].set(d2B_val[i, j, i, k1, i, k2])
            return d2B_diag
        self.d2B_by_dXdX_jax = jit(d2B_by_dXdX_fn)
        
        # dB_dgammas: gradient w.r.t. gammas (argnums=1)
        def dB_dgammas_fn(points, gammas, gammadashs, currents):
            def B_fn(gs):
                return biot_savart_pure(points, gs, gammadashs, currents)
            return jacfwd(B_fn)(gammas)
        self.dB_dgammas_jax = jit(dB_dgammas_fn)
        
        # dB_dgammadashs: gradient w.r.t. gammadashs (argnums=2)
        def dB_dgammadashs_fn(points, gammas, gammadashs, currents):
            def B_fn(gds):
                return biot_savart_pure(points, gammas, gds, currents)
            return jacfwd(B_fn)(gammadashs)
        self.dB_dgammadashs_jax = jit(dB_dgammadashs_fn)
        
        # dB_dcurrents: gradient w.r.t. currents (argnums=3)
        def dB_dcurrents_fn(points, gammas, gammadashs, currents):
            def B_fn(cs):
                return biot_savart_pure(points, gammas, gammadashs, cs)
            return jacfwd(B_fn)(currents)
        self.dB_dcurrents_jax = jit(dB_dcurrents_fn)
        
        # d2B_dgammas_dX: mixed derivative w.r.t. gammas and points
        def d2B_dgammas_dX_fn(points, gammas, gammadashs, currents):
            def B_fn_p(p):
                return biot_savart_pure(p, gammas, gammadashs, currents)
            def dB_dgammas_at_points(p):
                def B_fn_g(gs):
                    return biot_savart_pure(p, gs, gammadashs, currents)
                return jacfwd(B_fn_g)(gammas)
            return jacfwd(dB_dgammas_at_points)(points)
        self.d2B_dgammas_dX_jax = jit(d2B_dgammas_dX_fn)
        
        # d2B_dgammadashs_dX: mixed derivative w.r.t. gammadashs and points
        def d2B_dgammadashs_dX_fn(points, gammas, gammadashs, currents):
            def dB_dgammadashs_at_points(p):
                def B_fn_gd(gds):
                    return biot_savart_pure(p, gammas, gds, currents)
                return jacfwd(B_fn_gd)(gammadashs)
            return jacfwd(dB_dgammadashs_at_points)(points)
        self.d2B_dgammadashs_dX_jax = jit(d2B_dgammadashs_dX_fn)
        
        # d2B_dcurrents_dX: mixed derivative w.r.t. currents and points
        def d2B_dcurrents_dX_fn(points, gammas, gammadashs, currents):
            def dB_dcurrents_at_points(p):
                def B_fn_c(cs):
                    return biot_savart_pure(p, gammas, gammadashs, cs)
                return jacfwd(B_fn_c)(currents)
            return jacfwd(dB_dcurrents_at_points)(points)
        self.d2B_dcurrents_dX_jax = jit(d2B_dcurrents_dX_fn)
        
        # d2B_dgammas_dgammadashs: mixed derivative w.r.t. gammas and gammadashs
        def d2B_dgammas_dgammadashs_fn(points, gammas, gammadashs, currents):
            def dB_dgammas_at_gammadashs(gds):
                def B_fn_g(gs):
                    return biot_savart_pure(points, gs, gds, currents)
                return jacfwd(B_fn_g)(gammas)
            return jacfwd(dB_dgammas_at_gammadashs)(gammadashs)
        self.d2B_dgammas_dgammadashs_jax = jit(d2B_dgammas_dgammadashs_fn)
        
        # d2B_dgammas_dcurrents: mixed derivative w.r.t. gammas and currents
        def d2B_dgammas_dcurrents_fn(points, gammas, gammadashs, currents):
            def dB_dgammas_at_currents(cs):
                def B_fn_g(gs):
                    return biot_savart_pure(points, gs, gammadashs, cs)
                return jacfwd(B_fn_g)(gammas)
            return jacfwd(dB_dgammas_at_currents)(currents)
        self.d2B_dgammas_dcurrents_jax = jit(d2B_dgammas_dcurrents_fn)
        
        # d2B_dgammadashs_dcurrents: mixed derivative w.r.t. gammadashs and currents
        def d2B_dgammadashs_dcurrents_fn(points, gammas, gammadashs, currents):
            def dB_dgammadashs_at_currents(cs):
                def B_fn_gd(gds):
                    return biot_savart_pure(points, gammas, gds, cs)
                return jacfwd(B_fn_gd)(gammadashs)
            return jacfwd(dB_dgammadashs_at_currents)(currents)
        self.d2B_dgammadashs_dcurrents_jax = jit(d2B_dgammadashs_dcurrents_fn)
        
        # d2B_dcurrents_dcurrents: second derivative w.r.t. currents (should be zero since B is linear in currents)
        def d2B_dcurrents_dcurrents_fn(points, gammas, gammadashs, currents):
            def dB_dcurrents_at_currents(cs):
                def B_fn_c(cs2):
                    return biot_savart_pure(points, gammas, gammadashs, cs2)
                return jacfwd(B_fn_c)(cs)
            return jacfwd(dB_dcurrents_at_currents)(currents)
        self.d2B_dcurrents_dcurrents_jax = jit(d2B_dcurrents_dcurrents_fn)
        
        # d2B_dgammas_dgammas: second derivative w.r.t. gammas twice
        def d2B_dgammas_dgammas_fn(points, gammas, gammadashs, currents):
            def dB_dgammas_at_gammas(gs2):
                def B_fn_g(gs):
                    return biot_savart_pure(points, gs, gammadashs, currents)
                return jacfwd(B_fn_g)(gs2)
            return jacfwd(dB_dgammas_at_gammas)(gammas)
        self.d2B_dgammas_dgammas_jax = jit(d2B_dgammas_dgammas_fn)
        
        # d2B_dgammadashs_dgammadashs: second derivative w.r.t. gammadashs twice
        def d2B_dgammadashs_dgammadashs_fn(points, gammas, gammadashs, currents):
            def dB_dgammadashs_at_gammadashs(gds2):
                def B_fn_gd(gds):
                    return biot_savart_pure(points, gammas, gds, currents)
                return jacfwd(B_fn_gd)(gds2)
            return jacfwd(dB_dgammadashs_at_gammadashs)(gammadashs)
        self.d2B_dgammadashs_dgammadashs_jax = jit(d2B_dgammadashs_dgammadashs_fn)
    
    def set_points_cart(self, xyz):
        """Set evaluation points and clear cache."""
        if len(xyz.shape) != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz array should have shape (n, 3), but has shape {xyz.shape}")
        self._points = jnp.asarray(xyz)
        self.clear_cached_properties()
        return self
    
    def set_points(self, xyz):
        """Alias for set_points_cart."""
        return self.set_points_cart(xyz)
    
    def get_points_cart_ref(self):
        """Get reference to evaluation points."""
        if self._points is None:
            raise ValueError("Points not set. Call set_points_cart() first.")
        return np.asarray(self._points)
    
    def clear_cached_properties(self):
        """Clear cached computed values."""
        self._B_cache = None
        self._dB_cache = None
        self._d2B_cache = None
    
    def B(self):
        """Compute magnetic field B at evaluation points."""
        if self._B_cache is not None:
            return self._B_cache
        
        if self._points is None:
            raise ValueError("Points not set. Call set_points_cart() first.")
        
        # Get coil geometries
        gammas = [jnp.asarray(coil.curve.gamma()) for coil in self._coils]
        gammadashs = [jnp.asarray(coil.curve.gammadash()) for coil in self._coils]
        currents = jnp.asarray([coil.current.get_value() for coil in self._coils])
        
        # Compute B using the jitted function
        B_val = self.B_jax(self._points, gammas, gammadashs, currents)
        self._B_cache = np.asarray(B_val)
        return self._B_cache
    
    def dB_by_dX(self):
        """Compute gradient of B w.r.t. evaluation points X."""
        if self._dB_cache is not None:
            return self._dB_cache
        
        if self._points is None:
            raise ValueError("Points not set. Call set_points_cart() first.")
        
        # Get coil geometries
        gammas = [jnp.asarray(coil.curve.gamma()) for coil in self._coils]
        gammadashs = [jnp.asarray(coil.curve.gammadash()) for coil in self._coils]
        currents = jnp.asarray([coil.current.get_value() for coil in self._coils])
        
        # Compute dB_by_dX using jitted function
        dB_val = self.dB_by_dX_jax(self._points, gammas, gammadashs, currents)
        self._dB_cache = np.asarray(dB_val)
        return self._dB_cache
    
    def d2B_by_dXdX(self):
        """Compute Hessian of B w.r.t. evaluation points X."""
        if self._d2B_cache is not None:
            return self._d2B_cache
        
        if self._points is None:
            raise ValueError("Points not set. Call set_points_cart() first.")
        
        # Get coil geometries
        gammas = [jnp.asarray(coil.curve.gamma()) for coil in self._coils]
        gammadashs = [jnp.asarray(coil.curve.gammadash()) for coil in self._coils]
        currents = jnp.asarray([coil.current.get_value() for coil in self._coils])
        
        # Compute d2B_by_dXdX using jitted function
        d2B_val = self.d2B_by_dXdX_jax(self._points, gammas, gammadashs, currents)
        self._d2B_cache = np.asarray(d2B_val)
        return self._d2B_cache
    
    def B_vjp(self, v):
        r"""
        Compute vector Jacobian product: (dB/dcoeffs)^T * v
        
        Args:
            v: Vector in B space, shape (n_points, 3)
            
        Returns:
            Derivative object containing (dB/dcoeffs)^T * v
        """
        if self._points is None:
            raise ValueError("Points not set. Call set_points_cart() first.")
        
        v_jax = jnp.asarray(v)
        
        # Get coil geometries
        gammas = [jnp.asarray(coil.curve.gamma()) for coil in self._coils]
        gammadashs = [jnp.asarray(coil.curve.gammadash()) for coil in self._coils]
        currents = jnp.asarray([coil.current.get_value() for coil in self._coils])
        
        # Compute VJP w.r.t. gammas, gammadashs, and currents
        def B_fn(gammas_in, gammadashs_in, currents_in):
            return self.B_jax(self._points, gammas_in, gammadashs_in, currents_in)
        
        _, vjp_fn = vjp(B_fn, gammas, gammadashs, currents)
        vjp_gammas, vjp_gammadashs, vjp_currents = vjp_fn(v_jax)
        
        # Convert back to numpy arrays
        vjp_gammas_list = [np.asarray(vg) for vg in vjp_gammas]
        vjp_gammadashs_list = [np.asarray(vgd) for vgd in vjp_gammadashs]
        vjp_currents_list = [np.asarray(vc) for vc in vjp_currents]
        
        # Use coil.vjp to propagate through coil structure
        res = sum([
            self._coils[i].vjp(
                vjp_gammas_list[i], 
                vjp_gammadashs_list[i], 
                np.asarray([vjp_currents_list[i]])
            ) 
            for i in range(len(self._coils))
        ])
        
        return res
