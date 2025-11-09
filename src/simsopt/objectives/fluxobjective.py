import numpy as np
import jax.numpy as jnp
from jax import grad, hessian, jit as jaxjit
from ..geo.jit import jit
from ..geo.config import parameters
import simsoptpp as sopp
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec


__all__ = ['SquaredFlux', 'SquaredFluxJax']


class SquaredFlux(Optimizable):

    r"""
    Objective representing quadratic-flux-like quantities, useful for stage-2
    coil optimization. Several variations are available, which can be selected
    using the ``definition`` argument. For ``definition="quadratic flux"`` 
    (the default), the objective is defined as

    .. math::
        J = \frac12 \int_{S} (\mathbf{B}\cdot \mathbf{n} - B_T)^2 ds,

    where :math:`\mathbf{n}` is the surface unit normal vector and
    :math:`B_T` is an optional (zero by default) target value for the
    magnetic field. Also :math:`\int_{S} ds` indicates a surface integral.
    For ``definition="normalized"``, the objective is defined as

    .. math::
        J = \frac12 \frac{\int_{S} (\mathbf{B}\cdot \mathbf{n} - B_T)^2 ds}
                         {\int_{S} |\mathbf{B}|^2 ds}.

    For ``definition="local"``, the objective is defined as

    .. math::
        J = \frac12 \int_{S} \frac{(\mathbf{B}\cdot \mathbf{n} - B_T)^2}{|\mathbf{B}|^2} ds.

    The definition ``"quadratic flux"`` has the advantage of simplicity, and it
    is used in other contexts such as REGCOIL. However for stage-2 optimization,
    the optimizer can "cheat", lowering this objective by reducing the magnitude
    of the field. The definitions ``"normalized"`` and ``"local"`` close this loophole.

    Args:
        surface: A :obj:`simsopt.geo.surface.Surface` object on which to compute the flux
        field: A :obj:`simsopt.field.magneticfield.MagneticField` for which to compute the flux.
        target: A ``nphi x ntheta`` numpy array containing target values for the flux. Here 
          ``nphi`` and ``ntheta`` correspond to the number of quadrature points on `surface` 
          in ``phi`` and ``theta`` direction.
        definition: A string to select among the definitions above. The
          available options are ``"quadratic flux"``, ``"normalized"``, and ``"local"``.
    """

    def __init__(self, surface, field, target=None, definition="quadratic flux", threshold=0.0):
        self.surface = surface
        if target is not None:
            self.target = np.ascontiguousarray(target)
        else:
            self.target = np.zeros(self.surface.normal().shape[:2])
        self.field = field
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        if definition not in ["quadratic flux", "normalized", "local"]:
            raise ValueError("Unrecognized option for 'definition'.")
        self.definition = definition
        self.threshold = threshold
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    def J(self):
        n = self.surface.normal()
        Bcoil = self.field.B().reshape(n.shape)
        sq_flux = sopp.integral_BdotN(Bcoil, self.target, n, self.definition)
        if sq_flux < self.threshold:
            return 0.0
        else:
            return sq_flux

    @derivative_dec
    def dJ(self):
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1. / absn)[:, :, None]
        Bcoil = self.field.B().reshape(n.shape)
        Bcoil_n = np.sum(Bcoil * unitn, axis=2)
        if self.target is not None:
            B_n = (Bcoil_n - self.target)
        else:
            B_n = Bcoil_n

        if self.definition == "quadratic flux":
            dJdB = (B_n[..., None] * unitn * absn[..., None]) / absn.size
            dJdB = dJdB.reshape((-1, 3))

        elif self.definition == "local":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            dJdB = ((
                (B_n/mod_Bcoil)[..., None] * (
                    unitn / mod_Bcoil[..., None] - (B_n / mod_Bcoil**3)[..., None] * Bcoil
                )) * absn[..., None]) / absn.size

        elif self.definition == "normalized":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            num = np.mean(B_n**2 * absn)
            denom = np.mean(mod_Bcoil**2 * absn)

            dnum = 2 * (B_n[..., None] * unitn * absn[..., None]) / absn.size
            ddenom = 2 * (Bcoil * absn[..., None]) / absn.size
            dJdB = 0.5 * (dnum / denom - num * ddenom / denom**2)

        else:
            raise ValueError("Should never get here")

        dJdB = dJdB.reshape((-1, 3))
        if np.isclose(self.J(), 0.0, atol=1e-10, rtol=1e-10):
            return self.field.B_vjp(np.zeros_like(dJdB))
        else:
            return self.field.B_vjp(dJdB)   

def squared_flux_pure(Bcoil, target, normals, definition):
    r"""Pure function for computing squared flux objective using JAX arrays.
    
    Implements three definitions:
    
    1. "quadratic flux": J = (1/2) ∫ (B·n - B_T)² ds
    2. "normalized": J = (1/2) [∫ (B·n - B_T)² ds] / [∫ |B|² ds]
    3. "local": J = (1/2) ∫ [(B·n - B_T)² / |B|²] ds
    
    Args:
        Bcoil: (nphi, ntheta, 3) array of magnetic field vectors
        target: (nphi, ntheta) array of target flux values
        unitn: (nphi, ntheta, 3) array of unit normal vectors
        absn: (nphi, ntheta) array of surface element magnitudes
        definition: string selecting the definition ("quadratic flux", "normalized", or "local")
    
    Returns:
        Scalar value of the squared flux objective
    """
    # Compute B·n (dot product of B with unit normal)
    absn = jnp.linalg.norm(normals, axis=2)
    unitn = normals * (1. / absn)[:, :, None]
    Bcoil_n = jnp.sum(Bcoil * unitn, axis=2)  # Shape: (nphi, ntheta)
    
    # Compute B_n - target
    B_n = Bcoil_n - target  # Shape: (nphi, ntheta)
    
    nphi, ntheta = Bcoil.shape[0], Bcoil.shape[1]
    
    if definition == "quadratic flux":
        # J = (1/2) ∫ (B·n - B_T)² ds / (nphi * ntheta)
        integrand = B_n**2 * absn  # Shape: (nphi, ntheta)
        return 0.5 * jnp.sum(integrand) / (nphi * ntheta)
    
    elif definition == "normalized":
        # J = (1/2) [∫ (B·n - B_T)² ds] / [∫ |B|² ds]
        num = jnp.sum(B_n**2 * absn)  # ∫ (B·n - B_T)² ds
        mod_Bcoil = jnp.linalg.norm(Bcoil, axis=2)  # |B|, shape: (nphi, ntheta)
        denom = jnp.sum(mod_Bcoil**2 * absn)  # ∫ |B|² ds
        return 0.5 * num / denom
    
    elif definition == "local":
        # J = (1/2) ∫ [(B·n - B_T)² / |B|²] ds / (nphi * ntheta)
        mod_Bcoil = jnp.linalg.norm(Bcoil, axis=2)  # |B|, shape: (nphi, ntheta)
        integrand = (B_n**2 / mod_Bcoil**2) * absn  # Shape: (nphi, ntheta)
        return 0.5 * jnp.sum(integrand) / (nphi * ntheta)
    
    else:
        raise ValueError(f"Unrecognized definition: {definition}")

class SquaredFluxJax(Optimizable):

    r"""
    Objective representing quadratic-flux-like quantities, useful for stage-2
    coil optimization. Several variations are available, which can be selected
    using the ``definition`` argument. For ``definition="quadratic flux"`` 
    (the default), the objective is defined as

    .. math::
        J = \frac12 \int_{S} (\mathbf{B}\cdot \mathbf{n} - B_T)^2 ds,

    where :math:`\mathbf{n}` is the surface unit normal vector and
    :math:`B_T` is an optional (zero by default) target value for the
    magnetic field. Also :math:`\int_{S} ds` indicates a surface integral.
    For ``definition="normalized"``, the objective is defined as

    .. math::
        J = \frac12 \frac{\int_{S} (\mathbf{B}\cdot \mathbf{n} - B_T)^2 ds}
                         {\int_{S} |\mathbf{B}|^2 ds}.

    For ``definition="local"``, the objective is defined as

    .. math::
        J = \frac12 \int_{S} \frac{(\mathbf{B}\cdot \mathbf{n} - B_T)^2}{|\mathbf{B}|^2} ds.

    The definition ``"quadratic flux"`` has the advantage of simplicity, and it
    is used in other contexts such as REGCOIL. However for stage-2 optimization,
    the optimizer can "cheat", lowering this objective by reducing the magnitude
    of the field. The definitions ``"normalized"`` and ``"local"`` close this loophole.

    Args:
        surface: A :obj:`simsopt.geo.surface.Surface` object on which to compute the flux
        field: A :obj:`simsopt.field.magneticfield.MagneticField` for which to compute the flux.
        target: A ``nphi x ntheta`` numpy array containing target values for the flux. Here 
          ``nphi`` and ``ntheta`` correspond to the number of quadrature points on `surface` 
          in ``phi`` and ``theta`` direction.
        definition: A string to select among the definitions above. The
          available options are ``"quadratic flux"``, ``"normalized"``, and ``"local"``.
    """

    def __init__(self, surface, field, target=None, definition="quadratic flux", threshold=0.0):
        self.surface = surface
        if target is not None:
            self.target = np.ascontiguousarray(target)
        else:
            self.target = np.zeros(self.surface.normal().shape[:2])
        self.field = field
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        if definition not in ["quadratic flux", "normalized", "local"]:
            raise ValueError("Unrecognized option for 'definition'.")
        self.definition = definition
        self.threshold = threshold
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])
        # Use JAX's jit directly with static_argnums to mark definition as static (not traced)
        # This allows string arguments to be passed without JAX trying to trace them
        if parameters['jit']:
            self.J_jax = jaxjit(squared_flux_pure, static_argnums=(3,))
        else:
            self.J_jax = squared_flux_pure
        self.dJ_dBcoil = jit(lambda Bcoil, target, normals: grad(self.J_jax, argnums=0)(Bcoil, target, normals, self.definition))
        self.dJ_dtarget = jit(lambda Bcoil, target, normals: grad(self.J_jax, argnums=1)(Bcoil, target, normals, self.definition))
        self.dJ_dnormals = jit(lambda Bcoil, target, normals: grad(self.J_jax, argnums=2)(Bcoil, target, normals, self.definition))
        # Compute Hessian with respect to Bcoil
        self.d2J_dBcoil2_jit = jit(lambda Bcoil, target, normals: hessian(self.J_jax, argnums=0)(Bcoil, target, normals, self.definition))

    def J(self):
        n = self.surface.normal()
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        Bcoil = self.field.B().reshape(n.shape)
        sq_flux = self.J_jax(jnp.asarray(Bcoil), jnp.asarray(self.target), jnp.asarray(n), self.definition)
        sq_flux_val = float(sq_flux)  # Convert JAX array to Python float
        if sq_flux_val < self.threshold:
            return 0.0
        else:
            return sq_flux_val

    @derivative_dec
    def dJ(self):
        """Return Jacobian with respect to both coil and plasma dofs"""
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        n = self.surface.normal()
        Bcoil = self.field.B().reshape(n.shape)
        dJdB = self.dJ_dBcoil(jnp.asarray(Bcoil), jnp.asarray(self.target), jnp.asarray(n))
        # Reshape dJdB from (nphi, ntheta, 3) to (nphi * ntheta, 3) for B_vjp
        dJdB_flat = dJdB.reshape((-1, 3))
        if np.isclose(self.J(), 0.0, atol=1e-10, rtol=1e-10):
            return self.field.B_vjp(jnp.zeros_like(dJdB_flat))
        else:
            return self.field.B_vjp(dJdB_flat) #\
            # + Derivative({self.surface.x: self.dtarget_dcoefs(self.dJ_dtarget(Bcoil, self.target, n))}) \
            # + Derivative({self.surface.x: self.dnormals_dcoefs(self.dJ_dnormals(Bcoil, self.target, n))})
    
    def d2J_dcoil_dofs2(self):
        """Return Hessian with respect to coil degrees of freedom.
        
        This applies the chain rule through B_vjp to convert the Hessian
        w.r.t. Bcoil to the Hessian w.r.t. coil dofs.
        
        The Hessian w.r.t. coil dofs is: (dB/dx)^T * H_B * (dB/dx)
        where H_B is the Hessian w.r.t. Bcoil.
        
        We compute this by:
        1. For each coil dof i, compute (dB/dx) * e_i (change in B when dof i changes)
        2. Multiply by H_B to get H_B * (dB/dx) * e_i
        3. Apply B_vjp to get (dB/dx)^T * H_B * (dB/dx) * e_i (i-th column of Hessian)
        
        Returns:
            Hessian matrix of shape (n_coil_dofs, n_coil_dofs)
        """
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        n = self.surface.normal()
        Bcoil = self.field.B().reshape(n.shape)
        Bcoil_jax = jnp.asarray(Bcoil)
        target_jax = jnp.asarray(self.target)
        normals_jax = jnp.asarray(n)
        
        # Compute Hessian w.r.t. Bcoil: shape will be (nphi, ntheta, 3, nphi, ntheta, 3)
        H_B = self.d2J_dBcoil2_jit(Bcoil_jax, target_jax, normals_jax)
        
        # Reshape to (nphi * ntheta * 3, nphi * ntheta * 3)
        nphi, ntheta = Bcoil.shape[0], Bcoil.shape[1]
        n_B = nphi * ntheta * 3
        H_B_flat = H_B.reshape((n_B, n_B))
        
        # Get the number of coil dofs
        dJdB_flat = self.dJ_dBcoil(Bcoil_jax, target_jax, normals_jax).reshape((-1, 3))
        dJ_dcoil_dofs = self.field.B_vjp(dJdB_flat)
        n_coil_dofs = len(dJ_dcoil_dofs(self.field))
        
        # Initialize Hessian w.r.t. coil dofs
        H_coil = np.zeros((n_coil_dofs, n_coil_dofs))
        
        # Compute Hessian as matrix product: H_x = (dB/dx)^T * H_B * (dB/dx)
        # 
        # To compute dB/dx analytically:
        # B_vjp(v) gives us (dB/dx)^T * v, which is the derivative of (v^T * B) w.r.t. coil dofs
        # For each point j, B_vjp(e_j) gives us dB_j/dx for all coil dofs (j-th row of dB/dx)
        
        # Pre-compute dB/dx matrix: dB_dx[j, i] = dB_j/dx_i
        # B is evaluated at nphi * ntheta points, each with 3 components
        # So B has shape (nphi * ntheta, 3), and when flattened it's (n_B,) where n_B = nphi * ntheta * 3
        n_points = nphi * ntheta
        dB_dx = np.zeros((n_B, n_coil_dofs))
        
        # For each point and component where B is evaluated
        for point_idx in range(n_points):
            for comp_idx in range(3):
                j = point_idx * 3 + comp_idx
                
                # Create unit vector at position (point_idx, comp_idx)
                # B_vjp expects input of shape (n_points, 3)
                e_j_reshaped = np.zeros((n_points, 3))
                e_j_reshaped[point_idx, comp_idx] = 1.0
                
                # Compute (dB/dx)^T * e_j using B_vjp
                # This gives us dB_j/dx for all coil dofs (j-th row of dB/dx)
                # B_vjp computes the derivative of (e_j^T * B) w.r.t. coil dofs
                # which is (dB/dx)^T * e_j, i.e., dB_j/dx for all coil dofs
                dB_j_dx = self.field.B_vjp(jnp.asarray(e_j_reshaped))
                dB_j_dx_array = dB_j_dx(self.field)
                
                # Store in dB/dx matrix (j-th row is dB_j/dx)
                dB_dx[j, :] = dB_j_dx_array
        
        # Compute Hessian: H_x = (dB/dx)^T * H_B * (dB/dx) + (dJ/dB) @ (d²B/dx²)
        # 
        # First term: (dB/dx)^T * H_B * (dB/dx)
        # Step 1: Compute H_B * dB/dx (matrix multiplication)
        # H_B_flat is (n_B, n_B), dB_dx is (n_B, n_coil_dofs)
        # Result is (n_B, n_coil_dofs)
        H_B_times_dB_dx = jnp.dot(H_B_flat, dB_dx)
        
        # Step 2: Compute (dB/dx)^T * (H_B * dB/dx)
        # This is done by applying B_vjp to each column of H_B_times_dB_dx
        H_B_times_dB_dx_reshaped = H_B_times_dB_dx.reshape((-1, 3))  # Shape: (n_points, 3)
        
        # For each column of H_B_times_dB_dx, apply B_vjp to get (dB/dx)^T * column
        for i in range(n_coil_dofs):
            # Get i-th column of H_B_times_dB_dx
            col_i = H_B_times_dB_dx[:, i].reshape((-1, 3))  # Shape: (n_points, 3)
            
            # Apply B_vjp to get (dB/dx)^T * col_i
            # This gives us the i-th column of the first term of the Hessian
            dH_col_dcoil_dofs = self.field.B_vjp(jnp.asarray(col_i))
            H_coil[:, i] = dH_col_dcoil_dofs(self.field)
        
        # Second term: (dJ/dB) @ (d²B/dx²)
        # 
        # To compute d²B/dx² analytically, we use the chain rule:
        # 
        # First derivative: dB/dx = dB/dgamma * dgamma/dx + dB/dgammadash * dgammadash/dx + dB/dcurrent * dcurrent/dx
        # 
        # Second derivative (using product rule):
        # d²B/dx² = d/dx (dB/dx)
        #          = d/dx (dB/dgamma * dgamma/dx) + d/dx (dB/dgammadash * dgammadash/dx) + d/dx (dB/dcurrent * dcurrent/dx)
        # 
        # Using product rule for each term:
        # d/dx (dB/dgamma * dgamma/dx) = (d²B/dgamma² * dgamma/dx) * dgamma/dx + dB/dgamma * d²gamma/dx²
        #                              + d²B/dgamma dgammadash * dgamma/dx * dgammadash/dx
        #                              + d²B/dgamma dcurrent * dgamma/dx * dcurrent/dx
        # 
        # Similar for other terms.
        # 
        # The key insight: We can compute this using B_vjp and the second derivatives of gamma, gammadash, current
        # 
        # For each coil dof i, we need: sum_k (dJ/dB_k) * d²B_k/dx²[:, i]
        # This is equivalent to: (d²B/dx²)^T * dJ/dB evaluated at column i
        # 
        # We can compute this by:
        # 1. For each coil dof i, compute how dB/dx changes when coil dof i changes
        # 2. This requires computing d²B/dx²[:, :, i]
        # 3. Then contract with dJ/dB
        # 
        # To compute d²B/dx²[:, :, i] analytically:
        # - We need d²gamma/dx², d²gammadash/dx², d²current/dx² (from curve/coil)
        # - We need d²B/dgamma², d²B/dgammadash², d²B/dgamma dgammadash (from BiotSavart)
        # 
        # Actually, a more efficient approach: compute the second term by taking the derivative
        # of dJ/dx w.r.t. coil dofs, then subtract the first term
        # 
        # dJ/dx = (dB/dx)^T * dJ/dB
        # d/dx_i (dJ/dx) = d/dx_i [(dB/dx)^T * dJ/dB]
        #                = [d/dx_i (dB/dx)^T] * dJ/dB + (dB/dx)^T * [d/dx_i (dJ/dB)]
        #                = [d²B/dx²[:, :, i]^T] * dJ/dB + (dB/dx)^T * [d²J/dB² * dB/dx[:, i]]
        #                = [d²B/dx²[:, :, i]^T] * dJ/dB + (dB/dx)^T * H_B * dB/dx[:, i]
        # 
        # The first part is the second term, the second part is already in the first term.
        # 
        # So: second_term[:, i] = d/dx_i (dJ/dx) - first_term[:, i]
        # 
        # We can compute d/dx_i (dJ/dx) by taking the derivative of dJ/dx w.r.t. coil dof i
        # using finite differences or by computing it analytically
        
        # For now, let's compute the second term using finite differences on dJ/dx
        # This is more efficient than computing d²B/dx² directly
        dJ_dB = dJdB_flat.reshape((-1, 3))  # Shape: (n_points, 3)
        epsilon = 1e-6
        coil_dofs_orig = self.field.x.copy()
        second_term = np.zeros((n_coil_dofs, n_coil_dofs))
        
        # Compute dJ/dx at original point
        dJ_dx_orig = objective_jax.dJ()
        
        # For each coil dof i, compute d/dx_i (dJ/dx)
        for i in range(n_coil_dofs):
            # Create unit vector for coil dof i
            e_i = np.zeros(n_coil_dofs)
            e_i[i] = 1.0
            
            # Perturb coil dof i
            self.field.x = coil_dofs_orig + epsilon * e_i
            
            # Recompute dJ/dx at perturbed point
            dJ_dx_pert = objective_jax.dJ()
            
            # Compute d/dx_i (dJ/dx) using finite differences
            d2J_dx2_i = (dJ_dx_pert - dJ_dx_orig) / epsilon
            
            # The second term is: d²J/dx² - first_term
            # So: second_term[:, i] = d2J_dx2_i - H_coil[:, i]
            second_term[:, i] = d2J_dx2_i - H_coil[:, i]
        
        # Restore original coil dofs
        self.field.x = coil_dofs_orig
        self.field.set_points(xyz.reshape((-1, 3)))
        
        # Add the second term to the Hessian
        H_coil += second_term
        
        # Symmetrize the Hessian (should be symmetric)
        H_coil = 0.5 * (H_coil + H_coil.T)
        
        return H_coil
