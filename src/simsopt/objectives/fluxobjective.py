import numpy as np
import jax.numpy as jnp
from jax import grad, hessian, jit as jaxjit, jacfwd
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

    def __init__(self, surface, field, target=None, definition="quadratic flux", threshold=0.0, fixed_surface=True, fixed_coils=False):
        from simsopt.geo.jaxsurface import JaxSurfaceRZFourier
        from simsopt.field.jaxbiotsavart import JaxBiotSavart
        if not isinstance(field, JaxBiotSavart):
            raise ValueError("Field must be a JaxBiotSavart object")
        self.surface = surface
        self.fixed_surface = fixed_surface
        self.fixed_coils = fixed_coils
        # Compute target shape before fixing (normal() needs dofs)
        # Get normal before fixing/unfixing to ensure we have DOFs
        if target is not None:
            self.target = np.ascontiguousarray(target)
        else:
            # Call normal() before fixing - surface should have free DOFs at this point
            n_shape = self.surface.normal().shape[:2]
            self.target = np.zeros(n_shape)
        # Fix/unfix coils/curves if needed
        if fixed_coils:
            # Fix all coils/curves in the field
            if hasattr(field, '_coils'):
                for coil in field._coils:
                    if hasattr(coil, 'curve'):
                        coil.curve.fix_all()
                    if hasattr(coil, 'current'):
                        coil.current.fix_all()
        else:
            # Unfix coils/curves to ensure they're free for optimization
            if hasattr(field, '_coils'):
                for coil in field._coils:
                    if hasattr(coil, 'curve'):
                        if coil.curve.dof_size == 0:
                            coil.curve.unfix_all()
                        elif not np.any(coil.curve.dofs_free_status):
                            coil.curve.unfix_all()
                    if hasattr(coil, 'current'):
                        if coil.current.dof_size == 0:
                            coil.current.unfix_all()
                        elif not np.any(coil.current.dofs_free_status):
                            coil.current.unfix_all()
        # Now fix/unfix the surface after we've computed what we need
        xyz = self.surface.gamma()
        # if fixed_surface:
        #     self.surface.fix_all()
        # else:
        if not fixed_surface:
            if not isinstance(self.surface, JaxSurfaceRZFourier):
                raise ValueError("Surface must be a JaxSurfaceRZFourier if fixed_surface is False for Hessian computation")
            # Unfix the surface to ensure it's free for optimization
            # self.surface.unfix_all()
        self.field = field
        self.field.set_points(xyz.reshape((-1, 3)))
        if definition not in ["quadratic flux", "normalized", "local"]:
            raise ValueError("Unrecognized option for 'definition'.")
        self.definition = definition
        self.threshold = threshold
        if fixed_surface:
            Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])
        elif fixed_coils:
            Optimizable.__init__(self, x0=np.asarray([]), depends_on=[surface])
        else:
            Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field, surface])
        # Use JAX's jit directly with static_argnums to mark definition as static (not traced)
        # This allows string arguments to be passed without JAX trying to trace them
        if parameters['jit']:
            self.J_jax = jaxjit(squared_flux_pure, static_argnums=(3,))
        else:
            self.J_jax = squared_flux_pure
        # Create wrapper functions that take definition as a parameter for mixed derivatives
        def dJ_dnormals_wrapper(Bcoil, target, normals, definition):
            return grad(self.J_jax, argnums=2)(Bcoil, target, normals, definition)
        
        # First-order derivatives
        self.dJ_dBcoil = jit(lambda Bcoil, target, normals: grad(self.J_jax, argnums=0)(Bcoil, target, normals, self.definition))
        self.dJ_dtarget = jit(lambda Bcoil, target, normals: grad(self.J_jax, argnums=1)(Bcoil, target, normals, self.definition))
        self.dJ_dnormals = jit(lambda Bcoil, target, normals: grad(self.J_jax, argnums=2)(Bcoil, target, normals, self.definition))
        
        # Second-order derivatives - pure Hessians
        self.d2J_dBcoil2 = jit(lambda Bcoil, target, normals: hessian(self.J_jax, argnums=0)(Bcoil, target, normals, self.definition))
        self.d2J_dnormals2 = jit(lambda Bcoil, target, normals: hessian(self.J_jax, argnums=2)(Bcoil, target, normals, self.definition))
        
        # Mixed second-order derivatives - need to use wrappers with static_argnums
        def make_d2J_dBcoil_dnormals(definition):
            def mixed_deriv(Bcoil, target, normals):
                return jacfwd(dJ_dnormals_wrapper, argnums=0)(Bcoil, target, normals, definition)
            if parameters['jit']:
                return jaxjit(mixed_deriv)
            else:
                return mixed_deriv
        
        self.d2J_dBcoil_dnormals = make_d2J_dBcoil_dnormals(self.definition)
   
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
        """Return Jacobian with respect to both coil and surface dofs"""
        from .._core.derivative import Derivative
        
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        n = self.surface.normal()
        Bcoil = self.field.B().reshape(n.shape)
        
        # Convert to JAX arrays
        Bcoil_jax = jnp.asarray(Bcoil)
        target_jax = jnp.asarray(self.target)
        normals_jax = jnp.asarray(n)
        
        # Compute dJ/dB (derivative w.r.t. B field)
        dJdB = self.dJ_dBcoil(Bcoil_jax, target_jax, normals_jax)
        # Reshape dJdB from (nphi, ntheta, 3) to (nphi * ntheta, 3) for B_vjp
        # Convert JAX array to numpy array for B_vjp
        dJdB_flat = np.asarray(dJdB.reshape((-1, 3)))
        
        # Compute derivative w.r.t. coil dofs
        # Only use zeros if J() actually returns 0.0 (i.e., below threshold)
        J_val = self.J()
        if J_val == 0.0:
            coil_deriv = self.field.B_vjp(np.zeros_like(dJdB_flat))
        else:
            coil_deriv = self.field.B_vjp(dJdB_flat)
        
        # If surface is fixed, only return coil derivatives
        if self.fixed_surface:
            return coil_deriv
        
        # Compute derivatives w.r.t. surface dofs
        # dJ/ds = dJ/dn * dn/ds + dJ/dB * dB/dgamma * dgamma/ds
        # where dB/dgamma comes from the fact that B is evaluated at surface points
        
        # dJ/dn (derivative w.r.t. normals)
        dJdn = self.dJ_dnormals(Bcoil_jax, target_jax, normals_jax)
        # Convert JAX array to numpy array for dnormal_by_dcoeff_vjp
        # dnormal_by_dcoeff_vjp expects shape (nphi, ntheta, 3), not flattened
        dJdn_np = np.asarray(dJdn)
        
        # Compute dJ/dgamma (derivative w.r.t. surface position through B field)
        # This is dJ/dB * dB/dgamma, where dB/dgamma comes from evaluating B at different points
        # We can compute this using B_vjp with dJdB, but we need to account for the fact
        # that changing gamma changes where B is evaluated
        # Actually, B_vjp already accounts for this through the field's dependence on points
        
        # For now, we'll compute the surface derivative contributions:
        # 1. dJ/dn * dn/ds (normal contribution)
        dJdn_surf = self.surface.dnormal_by_dcoeff_vjp(dJdn_np)
        
        # 2. dJ/dB * dB/dX * dX/ds since the dJ/dB terms from the coils 
        # changing (at fixed surface) are already included in the coil derivatives, 
        # The only thing changing here is that Bcoil is being evaluated 
        # on different surface points, so this dJ/dB term needs to be added here.
        # dB_by_dX has shape (npoints, 3, 3) where dB_by_dX[i, j, k] = dB[i, j]/dX[i, k]
        # dgamma_by_dcoeff has shape (nphi, ntheta, 3, n_dofs)
        # dJdB has shape (nphi, ntheta, 3)
        # Ensure field has computed dB_by_dX
        # self.field.compute(1)  # Compute first derivatives including dB_by_dX
        dB_by_dX = self.field.dB_by_dX()  # Shape: (npoints, 3, 3)
        dgamma_by_dcoeff = self.surface.dgamma_by_dcoeff()  # Shape: (nphi, ntheta, 3, n_dofs)
        nphi, ntheta = dgamma_by_dcoeff.shape[0], dgamma_by_dcoeff.shape[1]
        
        # Reshape dB_by_dX to (nphi, ntheta, 3, 3) for easier contraction
        dB_by_dX_reshaped = dB_by_dX.reshape((nphi, ntheta, 3, 3))
        
        # Convert dJdB from JAX to numpy
        dJdB_np = np.asarray(dJdB)  # Shape: (nphi, ntheta, 3)
        
        # Contract: dJdB[i,j,k] * dB_by_dX[i,j,k,l] * dgamma_by_dcoeff[i,j,l,m]
        # Result shape: (n_dofs,)
        # Using einsum: 'ijk,ijkl,ijlm->m'
        dJdB_surf = np.einsum('ijk,ijkl,ijlm->m', dJdB_np, dB_by_dX_reshaped, dgamma_by_dcoeff, optimize=True)
        
        # Combine both surface derivative contributions: dJ/dn * dn/ds + dJ/dB * dB/dX * dX/ds
        if isinstance(dJdn_surf, np.ndarray):
            dJ_surf_total = dJdn_surf + dJdB_surf
        elif isinstance(dJdn_surf, Derivative):
            # If dJdn_surf is a Derivative, extract the array and add
            dJdn_surf_array = dJdn_surf(self.surface)
            dJ_surf_total = dJdn_surf_array + dJdB_surf
        else:
            dJ_surf_total = np.asarray(dJdn_surf) + dJdB_surf
        
        # Combine derivatives
        # B_vjp returns a Derivative object, so we can add to it
        if isinstance(coil_deriv, Derivative):
            # Add surface derivative to the existing Derivative
            surface_deriv = Derivative({self.surface: dJ_surf_total})
            return coil_deriv + surface_deriv
        else:
            # If coil_deriv is not a Derivative (shouldn't happen), create one
            derivs = {}
            if isinstance(coil_deriv, np.ndarray):
                # This shouldn't happen, but handle it
                pass
            derivs[self.surface] = dJ_surf_total
            return Derivative(derivs)
    
    def d2J(self):
        r"""
        Computes the Hessian of the squared flux penalty.
        
        The full Hessian should be computed using the chain rule. For the coil-coil block:
        
        .. math::
            \frac{\partial^2 J}{\partial c_i \partial c_j} = 
            \frac{\partial B_k}{\partial c_i} \frac{\partial^2 J}{\partial B_k \partial B_l} \frac{\partial B_l}{\partial c_j}
            + \frac{\partial J}{\partial B_k} \frac{\partial^2 B_k}{\partial c_i \partial c_j}
            + \frac{\partial n_k}{\partial c_i} \frac{\partial^2 J}{\partial n_k \partial n_l} \frac{\partial n_l}{\partial c_j}
            + \frac{\partial B_k}{\partial c_i} \frac{\partial^2 J}{\partial B_k \partial n_l} \frac{\partial n_l}{\partial c_j}
            + \frac{\partial n_k}{\partial c_i} \frac{\partial^2 J}{\partial n_k \partial B_l} \frac{\partial B_l}{\partial c_j}
            + \frac{\partial B_k}{\partial c_i} \frac{\partial^2 J}{\partial B_k \partial T_l} \frac{\partial T_l}{\partial c_j}
            + \frac{\partial n_k}{\partial c_i} \frac{\partial^2 J}{\partial n_k \partial T_l} \frac{\partial T_l}{\partial c_j}
        
        where :math:`c_i` are the coil dofs, :math:`s_j` are the surface dofs,
        :math:`B_k` are the magnetic field components, :math:`n_k` are the normal components,
        and :math:`T_l` are the target values.
        
        Note: Currently, this implementation only computes the first term
        :math:`(\partial B_k/\partial c_i) (\partial^2 J/\partial B_k \partial B_l) (\partial B_l/\partial c_j)`
        for the coil-coil block. The second term :math:`(\partial J/\partial B_k) (\partial^2 B_k/\partial c_i \partial c_j)`
        (second derivative of B w.r.t. coil DOFs) is not yet implemented, which may lead to
        inaccuracies in the Hessian computation, especially when the magnetic field
        has significant nonlinear dependence on the coil DOFs.
        
        For the surface-surface block:
        
        .. math::
            \frac{\partial^2 J}{\partial s_i \partial s_j} = 
            \frac{\partial n_k}{\partial s_i} \frac{\partial^2 J}{\partial n_k \partial n_l} \frac{\partial n_l}{\partial s_j}
            + \frac{\partial J}{\partial n_k} \frac{\partial^2 n_k}{\partial s_i \partial s_j}
            + \frac{\partial B_k}{\partial s_i} \frac{\partial^2 J}{\partial B_k \partial B_l} \frac{\partial B_l}{\partial s_j}
            + \frac{\partial B_k}{\partial s_i} \frac{\partial^2 J}{\partial B_k \partial n_l} \frac{\partial n_l}{\partial s_j}
            + \frac{\partial n_k}{\partial s_i} \frac{\partial^2 J}{\partial n_k \partial B_l} \frac{\partial B_l}{\partial s_j}
        
        where the second term :math:`(\partial J/\partial n_k) (\partial^2 n_k/\partial s_i \partial s_j)` and
        the third term involving :math:`\partial B_k/\partial s_i` (derivative of B w.r.t. surface position)
        are not yet fully implemented.
        
        Returns:
            Tuple of four matrices: (H_cc, H_cs, H_sc, H_ss) where:
            - H_cc: Hessian w.r.t. coil dofs (shape: (n_coil_dofs, n_coil_dofs))
            - H_cs: Mixed Hessian coil-surface (shape: (n_coil_dofs, n_surf_dofs))
            - H_sc: Mixed Hessian surface-coil (shape: (n_surf_dofs, n_coil_dofs))
            - H_ss: Hessian w.r.t. surface dofs (shape: (n_surf_dofs, n_surf_dofs))
        """
        n = self.surface.normal()
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        Bcoil = self.field.B().reshape(n.shape)
        
        nphi, ntheta = Bcoil.shape[0], Bcoil.shape[1]
        n_B = nphi * ntheta * 3
        
        # Convert to JAX arrays
        Bcoil_jax = jnp.asarray(Bcoil)
        target_jax = jnp.asarray(self.target)
        normals_jax = jnp.asarray(n)
        
        # Compute Hessians w.r.t. Bcoil, normals, and mixed terms
        # H_B: shape (nphi, ntheta, 3, nphi, ntheta, 3) -> reshape to (n_B, n_B)
        H_B = self.d2J_dBcoil2(Bcoil_jax, target_jax, normals_jax)
        H_B_flat = np.asarray(H_B.reshape((n_B, n_B)))
        
        # H_n: shape (nphi, ntheta, 3, nphi, ntheta, 3) -> reshape to (n_B, n_B)
        H_n = self.d2J_dnormals2(Bcoil_jax, target_jax, normals_jax)
        H_n_flat = np.asarray(H_n.reshape((n_B, n_B)))
        
        # H_Bn: shape (nphi, ntheta, 3, nphi, ntheta, 3) -> reshape to (n_B, n_B)
        H_Bn = self.d2J_dBcoil_dnormals(Bcoil_jax, target_jax, normals_jax)
        H_Bn_flat = np.asarray(H_Bn.reshape((n_B, n_B)))
        
        # Compute dJ/dB for the second derivative term
        dJdB = self.dJ_dBcoil(Bcoil_jax, target_jax, normals_jax)  # Shape: (nphi, ntheta, 3)
        
        # Get number of dofs
        n_coil_dofs = self.field.dof_size
        # Check if surface is fixed - use dof_size == 0 as indicator
        n_surf_dofs = self.surface.dof_size
        
        # Initialize output Hessian blocks
        H_cc = np.zeros((n_coil_dofs, n_coil_dofs))
        H_cs = np.zeros((n_coil_dofs, n_surf_dofs))
        H_sc = np.zeros((n_surf_dofs, n_coil_dofs))
        H_ss = np.zeros((n_surf_dofs, n_surf_dofs))
        
        # Compute dB/dc (derivative of B w.r.t. coil dofs)
        # Shape: (npoints, 3, n_coil_dofs)
        dB_dc = self._compute_B_derivs()
        dB_dc_flat = dB_dc.reshape((n_B, n_coil_dofs))
        
        # Compute dn/ds (derivative of normal w.r.t. surface dofs)
        # Shape: (nphi, ntheta, 3, n_surf_dofs)
        if n_surf_dofs > 0:
            dn_ds = self.surface.dnormal_by_dcoeff()  # Shape: (nphi, ntheta, 3, n_surf_dofs)
            dn_ds_flat = dn_ds.reshape((n_B, n_surf_dofs))
        else:
            dn_ds_flat = np.zeros((n_B, 0))
        
        # Compute Hessian blocks using chain rule:
        # Since target is fixed (dT/dc = 0, dT/ds = 0), we only need:
        # H_cc = (dB/dc)^T @ H_B @ (dB/dc) + (dJ/dB) @ (d²B/dc²)
        H_cc = dB_dc_flat.T @ H_B_flat @ dB_dc_flat
        
        # Add second term: (dJ/dB) @ (d²B/dc²)
        # This term accounts for the second derivative of B w.r.t. coil DOFs
        # d²B/dc² = d²B/dgamma² @ (dgamma/dc)² + dB/dgamma @ d²gamma/d²c + 
        #            d²B/dgammadash² @ (dgammadash/dc)² + dB/dgammadash @ d²gammadash/d²c +
        #            cross terms
        dJdB_flat = np.asarray(dJdB).reshape((-1,))  # Flatten to (n_B,)
        H_cc_d2B_term = self._compute_d2B_dc2_term(dJdB_flat)
        H_cc += H_cc_d2B_term
        
        if n_surf_dofs > 0:
            # H_ss = (dn/ds)^T @ H_n @ (dn/ds)
            H_ss = dn_ds_flat.T @ H_n_flat @ dn_ds_flat
            
            # H_cs = (dB/dc)^T @ H_Bn @ (dn/ds)
            H_cs = dB_dc_flat.T @ H_Bn_flat @ dn_ds_flat
            
            # H_sc = (dn/ds)^T @ H_Bn^T @ (dB/dc) = H_cs^T
            H_sc = dn_ds_flat.T @ H_Bn_flat.T @ dB_dc_flat
        
        return H_cc, H_cs, H_sc, H_ss
    
    def _compute_B_derivs(self):
        """
        Compute the derivative of B with respect to coil dofs.
        Returns a matrix of shape (npoints, 3, n_coil_dofs) where
        dB_derivs[i, j, k] = d(B[i, j])/d(coil_dof[k])
        """
        from .._core.derivative import Derivative
        
        points = self.field.get_points_cart_ref()
        npoints = len(points)
        n_coil_dofs = self.field.dof_size
        
        # Initialize output array
        dB_dc = np.zeros((npoints, 3, n_coil_dofs))
        
        # Compute derivative by calling B_vjp with unit vectors for each output component
        for i in range(npoints):
            for j in range(3):
                # Create unit vector for point i, component j
                v = np.zeros((npoints, 3))
                v[i, j] = 1.0
                # B_vjp returns a Derivative object containing gradients w.r.t. coil dofs
                grad_deriv = self.field.B_vjp(v)
                # Extract the full gradient array from the Derivative object
                if isinstance(grad_deriv, Derivative):
                    # Get the full gradient by calling the Derivative with the field object
                    # This returns the gradient w.r.t. all coil dofs concatenated
                    grad_array = grad_deriv(self.field)
                    if grad_array.shape[0] == n_coil_dofs:
                        dB_dc[i, j, :] = grad_array
                    else:
                        # If shape doesn't match, try to extract from individual coils
                        # This shouldn't happen, but handle it
                        raise ValueError(f"Gradient shape mismatch: expected {n_coil_dofs}, got {grad_array.shape[0]}")
                elif isinstance(grad_deriv, np.ndarray):
                    if grad_deriv.shape[0] == n_coil_dofs:
                        dB_dc[i, j, :] = grad_deriv
                    else:
                        raise ValueError(f"Gradient shape mismatch: expected {n_coil_dofs}, got {grad_deriv.shape[0]}")
                else:
                    # Try to convert to array
                    grad_array = np.asarray(grad_deriv)
                    if grad_array.shape[0] == n_coil_dofs:
                        dB_dc[i, j, :] = grad_array
                    else:
                        raise ValueError(f"Gradient shape mismatch: expected {n_coil_dofs}, got {grad_array.shape[0]}")
        
        return dB_dc
    
    def _compute_d2B_dc2_term(self, dJdB_flat):
        """
        Compute the term (dJ/dB) @ (d²B/dc²) for the Hessian.
        
        This uses the chain rule:
        d²B/dc² = d²B/dgamma² @ (dgamma/dc)² + dB/dgamma @ d²gamma/d²c + 
                   d²B/dgammadash² @ (dgammadash/dc)² + dB/dgammadash @ d²gammadash/d²c +
                   d²B/dgamma dgammadash @ (dgamma/dc) @ (dgammadash/dc)
        
        Args:
            dJdB_flat: Flattened dJ/dB, shape (n_B,) where n_B = n_points * 3
            
        Returns:
            Contribution to H_cc from d²B/dc² term, shape (n_coil_dofs, n_coil_dofs)
        """
        from .._core.derivative import Derivative
        import jax.numpy as jnp
        
        points = self.field.get_points_cart_ref()
        npoints = len(points)
        n_coil_dofs = self.field.dof_size
        
        # Initialize output
        H_cc_d2B = np.zeros((n_coil_dofs, n_coil_dofs))
        
        # Get coil geometries
        gammas = [jnp.asarray(coil.curve.gamma()) for coil in self.field._coils]
        gammadashs = [jnp.asarray(coil.curve.gammadash()) for coil in self.field._coils]
        currents = jnp.asarray([coil.current.get_value() for coil in self.field._coils])
        
        # Reshape dJdB_flat to (npoints, 3) for easier manipulation
        dJdB_reshaped = dJdB_flat.reshape((npoints, 3))
        
        # Track DOF offset for each coil
        dof_offset = 0
        
        for coil_idx, coil in enumerate(self.field._coils):
            curve = coil.curve
            
            n_curve_dofs = curve.dof_size
            n_current_dofs = coil.current.dof_size
            n_coil_dofs_local = n_curve_dofs + n_current_dofs
            
            # Get first derivatives: dgamma/dc, dgammadash/dc
            dgamma_dc = curve.dgamma_by_dcoeff()  # Shape: (n_quad, 3, n_curve_dofs)
            dgammadash_dc = curve.dgammadash_by_dcoeff()  # Shape: (n_quad, 3, n_curve_dofs)
            
            # Get first derivatives of B w.r.t. gamma and gammadash for this coil
            dB_dgammas_coil = self.field.dB_dgammas_jax(jnp.asarray(points), gammas, gammadashs, currents)
            dB_dgammadashs_coil = self.field.dB_dgammadashs_jax(jnp.asarray(points), gammas, gammadashs, currents)
            
            dB_dgamma = np.asarray(dB_dgammas_coil[coil_idx])  # Shape: (npoints, 3, n_quad, 3)
            dB_dgammadash = np.asarray(dB_dgammadashs_coil[coil_idx])  # Shape: (npoints, 3, n_quad, 3)
            
            # Term 1: dB/dgamma @ d²gamma/d²c
            dJdB_dgamma = np.einsum('ijqk,ij->qk', dB_dgamma, dJdB_reshaped)  # Shape: (n_quad, 3)
            d2gamma_vjp_result = curve.d2gamma_by_d2coeff_vjp(dJdB_dgamma)
            d2gamma_vjp_array = d2gamma_vjp_result(curve) if isinstance(d2gamma_vjp_result, Derivative) else np.asarray(d2gamma_vjp_result)
            if d2gamma_vjp_array.ndim == 1:
                H_cc_d2B[dof_offset:dof_offset+n_curve_dofs, dof_offset:dof_offset+n_curve_dofs] += np.diag(d2gamma_vjp_array)
            else:
                H_cc_d2B[dof_offset:dof_offset+n_curve_dofs, dof_offset:dof_offset+n_curve_dofs] += d2gamma_vjp_array
            
            # Term 2: dB/dgammadash @ d²gammadash/d²c
            dJdB_dgammadash = np.einsum('ijqk,ij->qk', dB_dgammadash, dJdB_reshaped)  # Shape: (n_quad, 3)
            d2gammadash_vjp_result = curve.d2gammadash_by_d2coeff_vjp(dJdB_dgammadash)
            d2gammadash_vjp_array = d2gammadash_vjp_result(curve) if isinstance(d2gammadash_vjp_result, Derivative) else np.asarray(d2gammadash_vjp_result)
            if d2gammadash_vjp_array.ndim == 1:
                H_cc_d2B[dof_offset:dof_offset+n_curve_dofs, dof_offset:dof_offset+n_curve_dofs] += np.diag(d2gammadash_vjp_array)
            else:
                H_cc_d2B[dof_offset:dof_offset+n_curve_dofs, dof_offset:dof_offset+n_curve_dofs] += d2gammadash_vjp_array
            
            # Term 3: d²B/dgamma² @ (dgamma/dc)²
            d2B_dgamma2 = self.field.d2B_dgammas_dgammas_jax(jnp.asarray(points), gammas, gammadashs, currents)
            d2B_dgamma2_coil = np.asarray(d2B_dgamma2[coil_idx][coil_idx])  # Shape: (npoints, 3, n_quad1, 3, n_quad2, 3)
            
            for j in range(n_curve_dofs):
                dgamma_dc_j = dgamma_dc[:, :, j]  # Shape: (n_quad, 3)
                temp = np.einsum('pbqkrl,rl->pbqk', d2B_dgamma2_coil, dgamma_dc_j, optimize=True)
                for l in range(n_curve_dofs):
                    H_cc_d2B[dof_offset+j, dof_offset+l] += np.einsum('pbqk,pb,qk->', temp, dJdB_reshaped, dgamma_dc[:, :, l], optimize=True)
            
            # Term 4: d²B/dgammadash² @ (dgammadash/dc)²
            d2B_dgammadash2 = self.field.d2B_dgammadashs_dgammadashs_jax(jnp.asarray(points), gammas, gammadashs, currents)
            d2B_dgammadash2_coil = np.asarray(d2B_dgammadash2[coil_idx][coil_idx])  # Shape: (npoints, 3, n_quad1, 3, n_quad2, 3)
            
            for j in range(n_curve_dofs):
                dgammadash_dc_j = dgammadash_dc[:, :, j]
                temp = np.einsum('pbqkrl,rl->pbqk', d2B_dgammadash2_coil, dgammadash_dc_j, optimize=True)
                for l in range(n_curve_dofs):
                    H_cc_d2B[dof_offset+j, dof_offset+l] += np.einsum('pbqk,pb,qk->', temp, dJdB_reshaped, dgammadash_dc[:, :, l], optimize=True)
            
            # Term 5: d²B/dgamma dgammadash @ (dgamma/dc) @ (dgammadash/dc)
            d2B_dgamma_dgammadash = self.field.d2B_dgammas_dgammadashs_jax(jnp.asarray(points), gammas, gammadashs, currents)
            d2B_dgamma_dgammadash_coil = np.asarray(d2B_dgamma_dgammadash[coil_idx][coil_idx])  # Shape: (npoints, 3, n_quad1, 3, n_quad2, 3)
            
            for j in range(n_curve_dofs):
                dgamma_dc_j = dgamma_dc[:, :, j]
                for l in range(n_curve_dofs):
                    dgammadash_dc_l = dgammadash_dc[:, :, l]
                    val = np.einsum('pbqkrl,pb,qk,rl->', d2B_dgamma_dgammadash_coil, dJdB_reshaped, dgamma_dc_j, dgammadash_dc_l, optimize=True)
                    H_cc_d2B[dof_offset+j, dof_offset+l] += val
                    H_cc_d2B[dof_offset+l, dof_offset+j] += val  # Symmetric term
            
            # Update DOF offset
            dof_offset += n_coil_dofs_local
        
        return H_cc_d2B
    
    return_fn_map = {'J': J, 'dJ': dJ, 'd2J': d2J}