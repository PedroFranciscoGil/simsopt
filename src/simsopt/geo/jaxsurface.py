"""
JAX-based surface implementations for automatic differentiation.

This module provides JAX-based implementations of surface classes that enable
automatic differentiation and JIT compilation. The main class is JaxSurface,
which is similar to JaxCurve but for surfaces.
"""

import numpy as np
from jax import vjp, jacfwd, jvp
from jax import jit as jax_jit
import jax.numpy as jnp
from .jit import jit
from .surface import Surface

__all__ = ['JaxSurface', 'JaxSurfaceRZFourier', 'jaxrzfouriersurface_pure']


def _parse_rzfourier_dofs(dofs, mpol, ntor, stellsym):
    """
    Parse dofs into full rc, zs, rs, zc arrays.
    This is a non-jitted helper function using NumPy.
    """
    shift = (mpol + 1) * (2 * ntor + 1)
    
    # Initialize full arrays
    rc_full = np.zeros((mpol + 1, 2 * ntor + 1))
    zs_full = np.zeros((mpol + 1, 2 * ntor + 1))
    rs_full = np.zeros((mpol + 1, 2 * ntor + 1))
    zc_full = np.zeros((mpol + 1, 2 * ntor + 1))
    
    counter = 0
    if stellsym:
        # rc: indices from ntor to shift-1
        for idx in range(ntor, shift):
            m = idx // (2 * ntor + 1)
            i = idx % (2 * ntor + 1)
            rc_full[m, i] = dofs[counter]
            counter += 1
        # zs: indices from ntor+1 to shift-1
        for idx in range(ntor + 1, shift):
            m = idx // (2 * ntor + 1)
            i = idx % (2 * ntor + 1)
            zs_full[m, i] = dofs[counter]
            counter += 1
    else:
        # rc: indices from ntor to shift-1
        for idx in range(ntor, shift):
            m = idx // (2 * ntor + 1)
            i = idx % (2 * ntor + 1)
            rc_full[m, i] = dofs[counter]
            counter += 1
        # rs: indices from ntor+1 to shift-1
        for idx in range(ntor + 1, shift):
            m = idx // (2 * ntor + 1)
            i = idx % (2 * ntor + 1)
            rs_full[m, i] = dofs[counter]
            counter += 1
        # zc: indices from ntor to shift-1
        for idx in range(ntor, shift):
            m = idx // (2 * ntor + 1)
            i = idx % (2 * ntor + 1)
            zc_full[m, i] = dofs[counter]
            counter += 1
        # zs: indices from ntor+1 to shift-1
        for idx in range(ntor + 1, shift):
            m = idx // (2 * ntor + 1)
            i = idx % (2 * ntor + 1)
            zs_full[m, i] = dofs[counter]
            counter += 1
    
    return rc_full, zs_full, rs_full, zc_full


def _parse_rzfourier_dofs_jax(dofs, mpol, ntor, stellsym):
    """
    Parse dofs into full rc, zs, rs, zc arrays using pure JAX.
    This is a jitted function that can be traced by JAX.
    """
    shift = (mpol + 1) * (2 * ntor + 1)
    
    # Initialize full arrays
    rc_full = jnp.zeros((mpol + 1, 2 * ntor + 1))
    zs_full = jnp.zeros((mpol + 1, 2 * ntor + 1))
    rs_full = jnp.zeros((mpol + 1, 2 * ntor + 1))
    zc_full = jnp.zeros((mpol + 1, 2 * ntor + 1))
    
    if stellsym:
        # Build mapping from (m, i) to dof index
        # rc: indices from ntor to shift-1
        counter = 0
        for idx in range(ntor, shift):
            m = idx // (2 * ntor + 1)
            i = idx % (2 * ntor + 1)
            rc_full = rc_full.at[m, i].set(dofs[counter])
            counter += 1
        # zs: indices from ntor+1 to shift-1
        for idx in range(ntor + 1, shift):
            m = idx // (2 * ntor + 1)
            i = idx % (2 * ntor + 1)
            zs_full = zs_full.at[m, i].set(dofs[counter])
            counter += 1
    else:
        counter = 0
        # rc: indices from ntor to shift-1
        for idx in range(ntor, shift):
            m = idx // (2 * ntor + 1)
            i = idx % (2 * ntor + 1)
            rc_full = rc_full.at[m, i].set(dofs[counter])
            counter += 1
        # rs: indices from ntor+1 to shift-1
        for idx in range(ntor + 1, shift):
            m = idx // (2 * ntor + 1)
            i = idx % (2 * ntor + 1)
            rs_full = rs_full.at[m, i].set(dofs[counter])
            counter += 1
        # zc: indices from ntor to shift-1
        for idx in range(ntor, shift):
            m = idx // (2 * ntor + 1)
            i = idx % (2 * ntor + 1)
            zc_full = zc_full.at[m, i].set(dofs[counter])
            counter += 1
        # zs: indices from ntor+1 to shift-1
        for idx in range(ntor + 1, shift):
            m = idx // (2 * ntor + 1)
            i = idx % (2 * ntor + 1)
            zs_full = zs_full.at[m, i].set(dofs[counter])
            counter += 1
    
    return rc_full, zs_full, rs_full, zc_full


# Cache jitted functions for different static argument combinations
_jitted_inner_cache = {}

def jaxrzfouriersurface_pure(dofs, quadpoints_phi, quadpoints_theta, mpol, ntor, nfp, stellsym):
    """
    Pure JAX function for SurfaceRZFourier that computes gamma, gammadash1, gammadash2, and normal.
    
    This function parses dofs and calls the inner jitted function.
    
    Args:
        dofs: Array of dofs (rc, zs, and optionally rs, zc coefficients)
        quadpoints_phi: Array of phi quadrature points
        quadpoints_theta: Array of theta quadrature points
        mpol: Maximum poloidal mode number
        ntor: Maximum toroidal mode number (divided by nfp)
        nfp: Number of field periods
        stellsym: Whether the surface is stellarator-symmetric
    
    Returns:
        Dictionary with keys 'gamma', 'gammadash1', 'gammadash2', 'normal'
    """
    # Parse dofs (non-jitted) - convert to numpy for parsing
    dofs_np = np.asarray(dofs)
    rc_full, zs_full, rs_full, zc_full = _parse_rzfourier_dofs(dofs_np, mpol, ntor, stellsym)
    
    # Convert to JAX arrays
    rc_full = jnp.asarray(rc_full)
    zs_full = jnp.asarray(zs_full)
    rs_full = jnp.asarray(rs_full)
    zc_full = jnp.asarray(zc_full)
    
    # Cache jitted function for this combination of static arguments
    cache_key = (mpol, ntor, nfp, stellsym)
    if cache_key not in _jitted_inner_cache:
        _jitted_inner_cache[cache_key] = jax_jit(_jaxrzfouriersurface_pure_inner, static_argnums=(6, 7, 8, 9))
    
    # Call inner jitted function (with static arguments)
    return _jitted_inner_cache[cache_key](rc_full, zs_full, rs_full, zc_full, quadpoints_phi, quadpoints_theta, mpol, ntor, nfp, stellsym)


def _jaxrzfouriersurface_pure_inner(rc_full, zs_full, rs_full, zc_full, quadpoints_phi, quadpoints_theta, mpol, ntor, nfp, stellsym):
    """
    Inner jitted function that computes gamma, gammadash1, gammadash2, and normal.
    
    This function takes already-parsed rc, zs, rs, zc arrays.
    """
    
    # Convert to radians
    phi = 2 * jnp.pi * quadpoints_phi
    theta = 2 * jnp.pi * quadpoints_theta
    
    # Create meshgrid
    phi_2d = phi[:, None]  # (n_phi, 1)
    theta_2d = theta[None, :]  # (1, n_theta)
    
    # Create mode arrays with correct broadcasting: (mpol+1, 2*ntor+1, n_phi, n_theta)
    m_array = jnp.arange(mpol + 1)[:, None, None, None]  # (mpol+1, 1, 1, 1)
    n_array = jnp.arange(-ntor, ntor + 1)[None, :, None, None]  # (1, 2*ntor+1, 1, 1)
    phi_4d = phi_2d[None, None, :, :]  # (1, 1, n_phi, n_theta)
    theta_4d = theta_2d[None, None, :, :]  # (1, 1, n_phi, n_theta)
    
    # Compute angles for all modes at once: (mpol+1, 2*ntor+1, n_phi, n_theta)
    angle = m_array * theta_4d - n_array * nfp * phi_4d
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    
    # Compute r and z using einsum
    # rc: (mpol+1, 2*ntor+1), cos_angle: (mpol+1, 2*ntor+1, n_phi, n_theta)
    # Result: (n_phi, n_theta)
    r = jnp.einsum('mn,mnij->ij', rc_full, cos_angle)
    z = jnp.einsum('mn,mnij->ij', zs_full, sin_angle)
    
    if not stellsym:
        r = r + jnp.einsum('mn,mnij->ij', rs_full, sin_angle)
        z = z + jnp.einsum('mn,mnij->ij', zc_full, cos_angle)
    
    # Compute derivatives
    # d/dphi: multiply by -n*nfp
    n_array_2d = jnp.arange(-ntor, ntor + 1)[None, :, None, None]  # (1, 2*ntor+1, 1, 1)
    r_dphi = jnp.einsum('mn,mnij->ij', rc_full, -n_array_2d * nfp * (-sin_angle))
    z_dphi = jnp.einsum('mn,mnij->ij', zs_full, -n_array_2d * nfp * cos_angle)
    
    if not stellsym:
        r_dphi = r_dphi + jnp.einsum('mn,mnij->ij', rs_full, -n_array_2d * nfp * cos_angle)
        z_dphi = z_dphi + jnp.einsum('mn,mnij->ij', zc_full, -n_array_2d * nfp * (-sin_angle))
    
    # d/dtheta: multiply by m
    m_array_2d = jnp.arange(mpol + 1)[:, None, None, None]  # (mpol+1, 1, 1, 1)
    r_dtheta = jnp.einsum('mn,mnij->ij', rc_full, m_array_2d * (-sin_angle))
    z_dtheta = jnp.einsum('mn,mnij->ij', zs_full, m_array_2d * cos_angle)
    
    if not stellsym:
        r_dtheta = r_dtheta + jnp.einsum('mn,mnij->ij', rs_full, m_array_2d * cos_angle)
        z_dtheta = z_dtheta + jnp.einsum('mn,mnij->ij', zc_full, m_array_2d * (-sin_angle))
    
    # Convert to Cartesian coordinates
    cos_phi = jnp.cos(phi_2d)
    sin_phi = jnp.sin(phi_2d)
    
    # Gamma in Cartesian
    gamma_x = r * cos_phi
    gamma_y = r * sin_phi
    gamma_z = z
    gamma = jnp.stack([gamma_x, gamma_y, gamma_z], axis=-1)
    
    # Gammadash1 in Cartesian (d/dphi)
    # Note: d/dphi includes the factor 2*pi from d/d(quadpoint_phi)
    gammadash1_x = r_dphi * cos_phi * (2 * jnp.pi) - r * sin_phi * (2 * jnp.pi)
    gammadash1_y = r_dphi * sin_phi * (2 * jnp.pi) + r * cos_phi * (2 * jnp.pi)
    gammadash1_z = z_dphi * (2 * jnp.pi)
    gammadash1 = jnp.stack([gammadash1_x, gammadash1_y, gammadash1_z], axis=-1)
    
    # Gammadash2 in Cartesian (d/dtheta)
    # Note: d/dtheta includes the factor 2*pi from d/d(quadpoint_theta)
    gammadash2_x = r_dtheta * cos_phi * (2 * jnp.pi)
    gammadash2_y = r_dtheta * sin_phi * (2 * jnp.pi)
    gammadash2_z = z_dtheta * (2 * jnp.pi)
    gammadash2 = jnp.stack([gammadash2_x, gammadash2_y, gammadash2_z], axis=-1)
    
    # Normal = gammadash1 × gammadash2
    normal = jnp.cross(gammadash1, gammadash2)
    
    return {
        'gamma': gamma,
        'gammadash1': gammadash1,
        'gammadash2': gammadash2,
        'normal': normal
    }


class JaxSurface(Surface):
    """
    A base class for surfaces defined by pure JAX functions.
    
    This class is similar to JaxCurve but for surfaces. It provides
    automatic differentiation and JIT compilation for surface operations.
    
    Args:
        quadpoints_phi: Array of phi quadrature points
        quadpoints_theta: Array of theta quadrature points
        gamma_pure: Pure function that computes gamma, gammadash1, gammadash2, normal
        mpol: Maximum poloidal mode number
        ntor: Maximum toroidal mode number (divided by nfp)
        nfp: Number of field periods
        stellsym: Whether the surface is stellarator-symmetric
        **kwargs: Additional keyword arguments
    """
    
    def __init__(self, quadpoints_phi, quadpoints_theta, gamma_pure, mpol, ntor, nfp, stellsym, **kwargs):
        if isinstance(quadpoints_phi, np.ndarray):
            quadpoints_phi = list(quadpoints_phi)
        if isinstance(quadpoints_theta, np.ndarray):
            quadpoints_theta = list(quadpoints_theta)
        
        # Initialize as a Surface (not sopp.Surface since we're not using C++)
        Surface.__init__(self, **kwargs)
        
        self.quadpoints_phi = quadpoints_phi
        self.quadpoints_theta = quadpoints_theta
        self.gamma_pure = gamma_pure
        self.mpol = mpol
        self.ntor = ntor
        self.nfp = nfp
        self.stellsym = stellsym
        
        # Convert to JAX arrays
        phi_points = jnp.asarray(quadpoints_phi)
        theta_points = jnp.asarray(quadpoints_theta)
        ones_phi = jnp.ones_like(phi_points)
        ones_theta = jnp.ones_like(theta_points)
        
        # Cache jitted function for this combination of static arguments (outside of traced functions)
        cache_key = (self.mpol, self.ntor, self.nfp, self.stellsym)
        if cache_key not in _jitted_inner_cache:
            _jitted_inner_cache[cache_key] = jax_jit(_jaxrzfouriersurface_pure_inner, static_argnums=(6, 7, 8, 9))
        inner_jitted = _jitted_inner_cache[cache_key]
        
        # Create wrapper functions that parse dofs and call the inner jitted function
        # These use pure JAX parsing so they can be traced by JAX
        def parse_and_call_inner(dofs, result_key):
            """Parse dofs and call the inner jitted function"""
            # Parse dofs using pure JAX (can be traced)
            rc_full, zs_full, rs_full, zc_full = _parse_rzfourier_dofs_jax(dofs, self.mpol, self.ntor, self.stellsym)
            
            # Call inner jitted function
            result = inner_jitted(rc_full, zs_full, rs_full, zc_full, phi_points, theta_points, self.mpol, self.ntor, self.nfp, self.stellsym)
            return result[result_key]
        
        # Create wrapper functions (can be jitted since they use pure JAX)
        def gamma_jax(dofs):
            return parse_and_call_inner(dofs, 'gamma')
        
        def gammadash1_jax(dofs):
            return parse_and_call_inner(dofs, 'gammadash1')
        
        def gammadash2_jax(dofs):
            return parse_and_call_inner(dofs, 'gammadash2')
        
        def normal_jax(dofs):
            return parse_and_call_inner(dofs, 'normal')
        
        # Jit the wrapper functions (they now use pure JAX)
        self.gamma_jax = jit(gamma_jax)
        self.gammadash1_jax = jit(gammadash1_jax)
        self.gammadash2_jax = jit(gammadash2_jax)
        self.normal_jax = jit(normal_jax)
        
        # Derivatives - jit these as well
        self.dgamma_by_dcoeff_jax = jit(jacfwd(self.gamma_jax))
        self.dgamma_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.gamma_jax, x)[1](v)[0])
        self.d2gamma_by_d2coeff_vjp_jax = jit(lambda x, v: vjp(lambda d: jvp(self.gamma_jax, (d,), (jnp.ones_like(d),))[1], x)[1](v)[0])
        self.d2gamma_by_d2coeff_jax = jit(jacfwd(self.dgamma_by_dcoeff_jax))
        
        self.dgammadash1_by_dcoeff_jax = jit(jacfwd(self.gammadash1_jax))
        self.dgammadash1_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.gammadash1_jax, x)[1](v)[0])
        
        self.dgammadash2_by_dcoeff_jax = jit(jacfwd(self.gammadash2_jax))
        self.dgammadash2_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.gammadash2_jax, x)[1](v)[0])
        
        self.dnormal_by_dcoeff_jax = jit(jacfwd(self.normal_jax))
        self.dnormal_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.normal_jax, x)[1](v)[0])
        self.d2normal_by_dcoeffdcoeff_jax = jit(jacfwd(self.dnormal_by_dcoeff_jax))
        self.d2normal_by_dcoeffdcoeff_vjp_jax = jit(lambda x, v: vjp(lambda d: jvp(self.normal_jax, (d,), (jnp.ones_like(d),))[1], x)[1](v)[0])
    
    def gamma(self):
        """Return gamma (surface positions)"""
        return np.asarray(self.gamma_jax(self.get_dofs()))
    
    def gammadash1(self):
        """Return gammadash1 (d/dphi)"""
        return np.asarray(self.gammadash1_jax(self.get_dofs()))
    
    def gammadash2(self):
        """Return gammadash2 (d/dtheta)"""
        return np.asarray(self.gammadash2_jax(self.get_dofs()))
    
    def normal(self):
        """Return normal (gammadash1 × gammadash2)"""
        return np.asarray(self.normal_jax(self.get_dofs()))
    
    def dgamma_by_dcoeff(self):
        """Return dgamma/dcoeff"""
        return np.asarray(self.dgamma_by_dcoeff_jax(self.get_dofs()))
    
    def dgamma_by_dcoeff_vjp(self, v):
        """Return VJP of dgamma/dcoeff"""
        return np.asarray(self.dgamma_by_dcoeff_vjp_jax(self.get_dofs(), v))
    
    def d2gamma_by_d2coeff_vjp(self, v):
        """Return VJP of d²gamma/d²coeff"""
        return np.asarray(self.d2gamma_by_d2coeff_vjp_jax(self.get_dofs(), v))
    
    def d2gamma_by_d2coeff(self):
        """Return d²gamma/d²coeff"""
        return np.asarray(self.d2gamma_by_d2coeff_jax(self.get_dofs()))
    
    def dgammadash1_by_dcoeff(self):
        """Return dgammadash1/dcoeff"""
        return np.asarray(self.dgammadash1_by_dcoeff_jax(self.get_dofs()))
    
    def dgammadash1_by_dcoeff_vjp(self, v):
        """Return VJP of dgammadash1/dcoeff"""
        return np.asarray(self.dgammadash1_by_dcoeff_vjp_jax(self.get_dofs(), v))
    
    def dgammadash2_by_dcoeff(self):
        """Return dgammadash2/dcoeff"""
        return np.asarray(self.dgammadash2_by_dcoeff_jax(self.get_dofs()))
    
    def dgammadash2_by_dcoeff_vjp(self, v):
        """Return VJP of dgammadash2/dcoeff"""
        return np.asarray(self.dgammadash2_by_dcoeff_vjp_jax(self.get_dofs(), v))
    
    def dnormal_by_dcoeff(self):
        """Return dnormal/dcoeff"""
        return np.asarray(self.dnormal_by_dcoeff_jax(self.get_dofs()))
    
    def dnormal_by_dcoeff_vjp(self, v):
        """Return VJP of dnormal/dcoeff"""
        return np.asarray(self.dnormal_by_dcoeff_vjp_jax(self.get_dofs(), v))

    def d2normal_by_dcoeffdcoeff(self):
        """Return d²normal/d²coeff"""
        return np.asarray(self.d2normal_by_dcoeffdcoeff_jax(self.get_dofs()))
    
    def d2normal_by_dcoeffdcoeff_vjp(self, v):
        """Return VJP of d²normal/d²coeff"""
        return np.asarray(self.d2normal_by_dcoeffdcoeff_vjp_jax(self.get_dofs(), v))

class JaxSurfaceRZFourier(JaxSurface):
    """
    A JAX-based implementation of SurfaceRZFourier.
    
    This class provides automatic differentiation and JIT compilation
    for SurfaceRZFourier operations.
    
    Args:
        quadpoints_phi: Array of phi quadrature points
        quadpoints_theta: Array of theta quadrature points
        mpol: Maximum poloidal mode number
        ntor: Maximum toroidal mode number (divided by nfp)
        nfp: Number of field periods
        stellsym: Whether the surface is stellarator-symmetric
        dofs: Optional array of dofs
        **kwargs: Additional keyword arguments
    """
    
    def __init__(self, quadpoints_phi, quadpoints_theta, mpol, ntor, nfp, stellsym, dofs=None, **kwargs):
        # Store parameters for _make_names (must be set before calling _make_names)
        self.mpol = mpol
        self.ntor = ntor
        self.stellsym = stellsym
        
        # Compute number of dofs
        shift = (mpol + 1) * (2 * ntor + 1)
        if stellsym:
            n_dofs = 2 * shift - ntor - (ntor + 1)
        else:
            n_dofs = 4 * shift - 2 * ntor - 2 * (ntor + 1)
        
        # Initialize dofs
        names = self._make_names()
        if dofs is None:
            x0 = np.zeros(n_dofs)
            # Set default values similar to SurfaceRZFourier
            # rc[0, ntor] = 1.0, rc[1, ntor] = 0.1, zs[1, ntor] = 0.1
            # This requires parsing, so we'll set defaults after initialization
            super().__init__(quadpoints_phi, quadpoints_theta, jaxrzfouriersurface_pure,
                            mpol, ntor, nfp, stellsym, x0=x0, names=names, **kwargs)
            # Set default values
            self.set('rc(0,{})'.format(ntor), 1.0)
            self.set('rc(1,{})'.format(ntor), 0.1)
            self.set('zs(1,{})'.format(ntor), 0.1)
        else:
            super().__init__(quadpoints_phi, quadpoints_theta, jaxrzfouriersurface_pure,
                            mpol, ntor, nfp, stellsym, x0=dofs, names=names, **kwargs)
    
    def num_dofs(self):
        """Return the number of dofs"""
        shift = (self.mpol + 1) * (2 * self.ntor + 1)
        if self.stellsym:
            return 2 * shift - self.ntor - (self.ntor + 1)
        else:
            return 4 * shift - 2 * self.ntor - 2 * (self.ntor + 1)
    
    def get_dofs(self):
        """Return the dofs"""
        return self.x
    
    def set_dofs(self, dofs):
        """Set the dofs"""
        self.x = dofs
    
    def _make_names(self):
        """Create names for dofs"""
        names = []
        shift = (self.mpol + 1) * (2 * self.ntor + 1)
        
        if self.stellsym:
            # rc: indices from ntor to shift-1
            for idx in range(self.ntor, shift):
                m = idx // (2 * self.ntor + 1)
                n = idx % (2 * self.ntor + 1) - self.ntor
                names.append(f'rc({m},{n})')
            # zs: indices from ntor+1 to shift-1
            for idx in range(self.ntor + 1, shift):
                m = idx // (2 * self.ntor + 1)
                n = idx % (2 * self.ntor + 1) - self.ntor
                names.append(f'zs({m},{n})')
        else:
            # rc: indices from ntor to shift-1
            for idx in range(self.ntor, shift):
                m = idx // (2 * self.ntor + 1)
                n = idx % (2 * self.ntor + 1) - self.ntor
                names.append(f'rc({m},{n})')
            # rs: indices from ntor+1 to shift-1
            for idx in range(self.ntor + 1, shift):
                m = idx // (2 * self.ntor + 1)
                n = idx % (2 * self.ntor + 1) - self.ntor
                names.append(f'rs({m},{n})')
            # zc: indices from ntor to shift-1
            for idx in range(self.ntor, shift):
                m = idx // (2 * self.ntor + 1)
                n = idx % (2 * self.ntor + 1) - self.ntor
                names.append(f'zc({m},{n})')
            # zs: indices from ntor+1 to shift-1
            for idx in range(self.ntor + 1, shift):
                m = idx // (2 * self.ntor + 1)
                n = idx % (2 * self.ntor + 1) - self.ntor
                names.append(f'zs({m},{n})')
        
        return names
    
    def to_RZFourier(self):
        """Convert to SurfaceRZFourier"""
        from .surfacerzfourier import SurfaceRZFourier
        surf = SurfaceRZFourier(
            quadpoints_phi=self.quadpoints_phi,
            quadpoints_theta=self.quadpoints_theta,
            mpol=self.mpol, ntor=self.ntor, nfp=self.nfp, stellsym=self.stellsym,
            dofs=self.get_dofs()
        )
        return surf
