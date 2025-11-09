from deprecated import deprecated

import numpy as np
from jax import grad, vjp, hessian, jacfwd
import jax.numpy as jnp
from .jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec, Derivative
import simsoptpp as sopp

__all__ = ['CurveLength', 'LpCurveCurvature', 'LpCurveTorsion',
           'CurveCurveDistance', 'CurveSurfaceDistance', 'ArclengthVariation',
           'MeanSquaredCurvature', 'LinkingNumber']


@jit
def curve_length_pure(l):
    """
    Compute the mean of the incremental arclengths along a curve (the curve length).

    Args:
        l (array-like): Array of incremental arclengths along the curve.

    Returns:
        float: The mean arclength (i.e., the curve length).
    """
    return jnp.mean(l)


class CurveLength(Optimizable):
    r"""
    CurveLength is a class that computes the length of a curve, i.e.

    .. math::
        J = \int_{\text{curve}}~dl.

    """

    def __init__(self, curve):
        self.curve = curve
        self.dJ_dl = jit(lambda l: grad(curve_length_pure)(l))
        self.d2J_dl2 = jit(lambda l: hessian(curve_length_pure)(l))
        self.d2J_dl2_vjp = jit(lambda l, v: vjp(lambda x: hessian(curve_length_pure)(x), l)[1](v)[0])
        super().__init__(depends_on=[curve])

    def J(self):
        """
        This returns the value of the quantity.
        """
        return curve_length_pure(self.curve.incremental_arclength())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """

        return self.curve.dincremental_arclength_by_dcoeff_vjp(
            self.dJ_dl(self.curve.incremental_arclength()))

    def d2J(self):
        """ 
        Hessian for x = coil_dofs:
        d2J / d2x = d/dx (dJ/dx)
         = d/dx (dJ/dl * dl/dx)
         = dJ/dl * d²l/dx² + (dl/dx)^T @ (d²J/dl²) @ (dl/dx)
        """
        dJ_dl = self.dJ_dl(self.curve.incremental_arclength())
        dl_dx = self.curve.dincremental_arclength_by_dcoeff() # dl/dx has shape (n_quad_points, n_dofs)
        n_quad_points, n_dofs = dl_dx.shape
        H = np.zeros((n_dofs, n_dofs))
        
        # First term: dJ/dl * d²l/dx²
        # For each dof pair (i, j): sum_k (dJ/dl)[k] * d²l[k]/(dx_i dx_j)
        # This can be computed using einsum: np.einsum('k,kij->ij', dJ_dl, d2l_dx2)
        if hasattr(self.curve, 'd2incremental_arclength_by_d2coeff_impl') and hasattr(self.curve, 'd2incremental_arclength_by_d2coeff_jax'):
            # Get the full Hessian tensor d²l/dx² with shape (n_quad_points, n_dofs, n_dofs)
            d2l_dx2 = np.zeros((n_quad_points, n_dofs, n_dofs))
            self.curve.d2incremental_arclength_by_d2coeff_impl(d2l_dx2)
            # Compute first term using einsum: sum over quad points
            H += np.einsum('k,kij->ij', dJ_dl, d2l_dx2)
        
        # Second term: (dl/dx)^T @ (d²J/dl²) @ (dl/dx)
        # This is computed as: dl_dx.T @ (d2J_dl2 @ dl_dx)
        # Using einsum: sum_k sum_l (d²J/dl²)[k, l] * (dl/dx)[k, i] * (dl/dx)[l, j]
        l = self.curve.incremental_arclength()
        d2J_dl2_matrix = self.d2J_dl2(l)  # Shape: (n_quad_points, n_quad_points)
        
        # Compute using einsum: (dl/dx)^T @ (d²J/dl²) @ (dl/dx)
        # This is: sum_k sum_l (d²J/dl²)[k, l] * (dl/dx)[k, i] * (dl/dx)[l, j]
        H += np.einsum('ki,kl,lj->ij', dl_dx, d2J_dl2_matrix, dl_dx)
        
        return H

    return_fn_map = {'J': J, 'dJ': dJ, 'd2J': d2J}


@jit
def Lp_curvature_pure(kappa, gammadash, p, desired_kappa):
    """
    Compute the Lp penalty for curvature exceeding a threshold along a curve.

    Args:
        kappa (array-like): Curvature values along the curve.
        gammadash (array-like): Tangent vectors along the curve.
        p (float): The Lp norm exponent.
        desired_kappa (float): The threshold curvature value.

    Returns:
        float: The Lp penalty value.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return (1./p)*jnp.mean(jnp.maximum(kappa-desired_kappa, 0)**p * arc_length)


class LpCurveCurvature(Optimizable):
    r"""
    This class computes a penalty term based on the :math:`L_p` norm
    of the curve's curvature, and penalizes where the local curve curvature exceeds a threshold

    .. math::
        J = \frac{1}{p} \int_{\text{curve}} \text{max}(\kappa - \kappa_0, 0)^p ~dl

    where :math:`\kappa_0` is a threshold curvature, given by the argument ``threshold``.
    """

    def __init__(self, curve, p, threshold=0.0):
        self.curve = curve
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda kappa, gammadash: Lp_curvature_pure(kappa, gammadash, p, threshold))
        self.dJ_dkappa = jit(lambda kappa, gammadash: grad(self.J_jax, argnums=0)(kappa, gammadash))
        self.dJ_dgammadash = jit(lambda kappa, gammadash: grad(self.J_jax, argnums=1)(kappa, gammadash))
        # Hessian w.r.t. kappa and gammadash
        # Flatten gammadash for Hessian computation to get consistent shapes
        self.d2J_dkappa2 = jit(lambda kappa, gammadash: hessian(self.J_jax, argnums=0)(kappa, gammadash))
        # For gammadash Hessian, flatten gammadash first, then reshape the result
        # Create a wrapper function that handles flattening/reshaping
        def J_jax_flat(kappa, gammadash_flat):
            n_quad = kappa.shape[0]
            return self.J_jax(kappa, gammadash_flat.reshape((n_quad, 3)))
        self.d2J_dgammadash2 = jit(lambda kappa, gammadash: hessian(J_jax_flat, argnums=1)(kappa, gammadash.flatten()))
        # Cross terms: d²J/(dkappa dgammadash) and d²J/(dgammadash dkappa)
        # Flatten gammadash for cross term computation
        def dJ_dgammadash_flat(kappa, gammadash_flat):
            n_quad = kappa.shape[0]
            return grad(self.J_jax, argnums=1)(kappa, gammadash_flat.reshape((n_quad, 3)))
        self.d2J_dkappa_dgammadash = jit(lambda kappa, gammadash: jacfwd(dJ_dgammadash_flat, argnums=0)(kappa, gammadash.flatten()))
        def dJ_dkappa_flat(kappa, gammadash_flat):
            n_quad = kappa.shape[0]
            return grad(self.J_jax, argnums=0)(kappa, gammadash_flat.reshape((n_quad, 3)))
        self.d2J_dgammadash_dkappa = jit(lambda kappa, gammadash: jacfwd(dJ_dkappa_flat, argnums=1)(kappa, gammadash.flatten()))

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.J_jax(self.curve.kappa(), self.curve.gammadash())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        grad0 = self.dJ_dkappa(self.curve.kappa(), self.curve.gammadash())
        grad1 = self.dJ_dgammadash(self.curve.kappa(), self.curve.gammadash())
        return self.curve.dkappa_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)

    def d2J(self):
        """
        Hessian for x = coil_dofs:
        d²J/dx² = d/dx (dJ/dx)
                = d/dx (dJ/dkappa * dkappa/dx + dJ/dgammadash * dgammadash/dx)
                = dJ/dkappa * d²kappa/dx² + (dkappa/dx)^T @ (d²J/dkappa²) @ (dkappa/dx)
                + dJ/dgammadash * d²gammadash/dx² + (dgammadash/dx)^T @ (d²J/dgammadash²) @ (dgammadash/dx)
                + (dkappa/dx)^T @ (d²J/(dkappa dgammadash)) @ (dgammadash/dx)
                + (dgammadash/dx)^T @ (d²J/(dgammadash dkappa)) @ (dkappa/dx)
        """
        kappa = self.curve.kappa()
        gammadash = self.curve.gammadash()
        
        # Get first-order derivatives
        dJ_dkappa = np.asarray(self.dJ_dkappa(kappa, gammadash))
        dJ_dgammadash = np.asarray(self.dJ_dgammadash(kappa, gammadash))
        
        # Get first-order derivatives w.r.t. curve dofs
        dkappa_dx = np.asarray(self.curve.dkappa_by_dcoeff())  # Shape: (n_quad_points, n_dofs)
        dgammadash_dx = np.asarray(self.curve.dgammadash_by_dcoeff())  # Shape: (n_quad_points, 3, n_dofs)
        
        n_quad_points, n_dofs = dkappa_dx.shape
        n_quad_points_g, n_components, n_dofs_g = dgammadash_dx.shape
        assert n_quad_points == n_quad_points_g, "Mismatch in number of quad points"
        assert n_dofs == n_dofs_g, "Mismatch in number of dofs"
        
        H = np.zeros((n_dofs, n_dofs))
        
        # Get Hessian w.r.t. kappa and gammadash
        d2J_dkappa2 = np.asarray(self.d2J_dkappa2(kappa, gammadash))  # Shape: (n_quad_points, n_quad_points)
        d2J_dgammadash2_flat = np.asarray(self.d2J_dgammadash2(kappa, gammadash))  # Shape: (n_quad_points*3, n_quad_points*3)
        d2J_dkappa_dgammadash = np.asarray(self.d2J_dkappa_dgammadash(kappa, gammadash))  # Shape: (n_quad_points, 3, n_quad_points) - already correct
        d2J_dgammadash_dkappa_flat = np.asarray(self.d2J_dgammadash_dkappa(kappa, gammadash))  # Shape: (n_quad_points, n_quad_points*3)
        
        # Reshape flattened Hessians to proper shapes
        d2J_dgammadash2 = d2J_dgammadash2_flat.reshape((n_quad_points, 3, n_quad_points, 3))  # Shape: (n_quad_points, 3, n_quad_points, 3)
        # d2J_dkappa_dgammadash already has shape (n_quad_points, 3, n_quad_points) - no reshape needed
        d2J_dgammadash_dkappa = d2J_dgammadash_dkappa_flat.reshape((n_quad_points, n_quad_points, 3))  # Shape: (n_quad_points, n_quad_points, 3)
        
        # First term: dJ/dkappa * d²kappa/dx²
        # Skip if not available (some curves don't have this implemented)
        try:
            if hasattr(self.curve, 'd2kappa_by_d2coeff_impl') and hasattr(self.curve, 'd2kappa_by_d2coeff_jax'):
                d2kappa_dx2 = np.zeros((n_quad_points, n_dofs, n_dofs))
                self.curve.d2kappa_by_d2coeff_impl(d2kappa_dx2)
                H += np.einsum('k,kij->ij', dJ_dkappa, d2kappa_dx2)
        except (AttributeError, TypeError):
            pass  # Skip this term if not available
        
        # Second term: (dkappa/dx)^T @ (d²J/dkappa²) @ (dkappa/dx)
        H += np.einsum('ki,kl,lj->ij', dkappa_dx, d2J_dkappa2, dkappa_dx)
        
        # Third term: dJ/dgammadash * d²gammadash/dx²
        # Skip if not available (some curves don't have this implemented)
        try:
            if hasattr(self.curve, 'd2gammadash_by_d2coeff_impl') and hasattr(self.curve, 'd2gammadash_by_d2coeff_jax'):
                d2gammadash_dx2 = np.zeros((n_quad_points, 3, n_dofs, n_dofs))
                self.curve.d2gammadash_by_d2coeff_impl(d2gammadash_dx2)
                # dJ_dgammadash has shape (n_quad_points, 3)
                # d2gammadash_dx2 has shape (n_quad_points, 3, n_dofs, n_dofs)
                # We need to contract over quad points and components
                H += np.einsum('kc,kcij->ij', dJ_dgammadash, d2gammadash_dx2)
        except (AttributeError, TypeError):
            pass  # Skip this term if not available
        
        # Fourth term: (dgammadash/dx)^T @ (d²J/dgammadash²) @ (dgammadash/dx)
        # dgammadash_dx has shape (n_quad_points, 3, n_dofs)
        # d2J_dgammadash2 has shape (n_quad_points, 3, n_quad_points, 3)
        # We need to contract over quad points and components
        H += np.einsum('kci,kclm,lmj->ij', dgammadash_dx, d2J_dgammadash2, dgammadash_dx)
        
        # Fifth term: (dkappa/dx)^T @ (d²J/(dkappa dgammadash)) @ (dgammadash/dx)
        # dkappa_dx has shape (n_quad_points, n_dofs)
        # d2J_dkappa_dgammadash has shape (n_quad_points, 3, n_quad_points)
        # dgammadash_dx has shape (n_quad_points, 3, n_dofs)
        H += np.einsum('ki,kcl,lcj->ij', dkappa_dx, d2J_dkappa_dgammadash, dgammadash_dx)
        
        # Sixth term: (dgammadash/dx)^T @ (d²J/(dgammadash dkappa)) @ (dkappa/dx)
        # dgammadash_dx has shape (n_quad_points, 3, n_dofs)
        # d2J_dgammadash_dkappa has shape (n_quad_points, n_quad_points, 3)
        # dkappa_dx has shape (n_quad_points, n_dofs)
        H += np.einsum('kci,klc,lj->ij', dgammadash_dx, d2J_dgammadash_dkappa, dkappa_dx)
        
        return H
        
    return_fn_map = {'J': J, 'dJ': dJ, 'd2J': d2J}


@jit
def Lp_torsion_pure(torsion, gammadash, p, threshold):
    """
    Compute the Lp penalty for torsion exceeding a threshold along a curve.

    Args:
        torsion (array-like): Torsion values along the curve.
        gammadash (array-like): Tangent vectors along the curve.
        p (float): The Lp norm exponent.
        threshold (float): The threshold torsion value.

    Returns:
        float: The Lp penalty value.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return (1./p)*jnp.mean(jnp.maximum(jnp.abs(torsion)-threshold, 0)**p * arc_length)


class LpCurveTorsion(Optimizable):
    r"""
    LpCurveTorsion is a class that computes a penalty term based on the :math:`L_p` norm
    of the curve's torsion:

    .. math::
        J = \frac{1}{p} \int_{\text{curve}} \max(|\tau|-\tau_0, 0)^p ~dl.

    """

    def __init__(self, curve, p, threshold=0.0):
        self.curve = curve
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda torsion, gammadash: Lp_torsion_pure(torsion, gammadash, p, threshold))
        self.dJ_dtorsion = jit(lambda torsion, gammadash: grad(self.J_jax, argnums=0)(torsion, gammadash))
        self.dJ_dgammadash = jit(lambda torsion, gammadash: grad(self.J_jax, argnums=1)(torsion, gammadash))
        # Hessian w.r.t. torsion and gammadash
        # Flatten gammadash for Hessian computation to get consistent shapes
        self.d2J_dtorsion2 = jit(lambda torsion, gammadash: hessian(self.J_jax, argnums=0)(torsion, gammadash))
        # For gammadash Hessian, flatten gammadash first, then reshape the result
        def J_jax_flat(torsion, gammadash_flat):
            n_quad = torsion.shape[0]
            return self.J_jax(torsion, gammadash_flat.reshape((n_quad, 3)))
        self.d2J_dgammadash2 = jit(lambda torsion, gammadash: hessian(J_jax_flat, argnums=1)(torsion, gammadash.flatten()))
        # Cross terms: d²J/(dtorsion dgammadash) and d²J/(dgammadash dtorsion)
        # Flatten gammadash for cross term computation
        def dJ_dgammadash_flat(torsion, gammadash_flat):
            n_quad = torsion.shape[0]
            return grad(self.J_jax, argnums=1)(torsion, gammadash_flat.reshape((n_quad, 3)))
        self.d2J_dtorsion_dgammadash = jit(lambda torsion, gammadash: jacfwd(dJ_dgammadash_flat, argnums=0)(torsion, gammadash.flatten()))
        def dJ_dtorsion_flat(torsion, gammadash_flat):
            n_quad = torsion.shape[0]
            return grad(self.J_jax, argnums=0)(torsion, gammadash_flat.reshape((n_quad, 3)))
        self.d2J_dgammadash_dtorsion = jit(lambda torsion, gammadash: jacfwd(dJ_dtorsion_flat, argnums=1)(torsion, gammadash.flatten()))

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.J_jax(self.curve.torsion(), self.curve.gammadash())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        grad0 = self.dJ_dtorsion(self.curve.torsion(), self.curve.gammadash())
        grad1 = self.dJ_dgammadash(self.curve.torsion(), self.curve.gammadash())
        return self.curve.dtorsion_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)

    def d2J(self):
        """
        Hessian for x = coil_dofs:
        d²J/dx² = d/dx (dJ/dx)
                = d/dx (dJ/dtorsion * dtorsion/dx + dJ/dgammadash * dgammadash/dx)
                = dJ/dtorsion * d²torsion/dx² + (dtorsion/dx)^T @ (d²J/dtorsion²) @ (dtorsion/dx)
                + dJ/dgammadash * d²gammadash/dx² + (dgammadash/dx)^T @ (d²J/dgammadash²) @ (dgammadash/dx)
                + (dtorsion/dx)^T @ (d²J/(dtorsion dgammadash)) @ (dgammadash/dx)
                + (dgammadash/dx)^T @ (d²J/(dgammadash dtorsion)) @ (dtorsion/dx)
        """
        torsion = self.curve.torsion()
        gammadash = self.curve.gammadash()
        
        # Get first-order derivatives
        dJ_dtorsion = np.asarray(self.dJ_dtorsion(torsion, gammadash))
        dJ_dgammadash = np.asarray(self.dJ_dgammadash(torsion, gammadash))
        
        # Get first-order derivatives w.r.t. curve dofs
        dtorsion_dx = np.asarray(self.curve.dtorsion_by_dcoeff())  # Shape: (n_quad_points, n_dofs)
        dgammadash_dx = np.asarray(self.curve.dgammadash_by_dcoeff())  # Shape: (n_quad_points, 3, n_dofs)
        
        n_quad_points, n_dofs = dtorsion_dx.shape
        n_quad_points_g, n_components, n_dofs_g = dgammadash_dx.shape
        assert n_quad_points == n_quad_points_g, "Mismatch in number of quad points"
        assert n_dofs == n_dofs_g, "Mismatch in number of dofs"
        
        H = np.zeros((n_dofs, n_dofs))
        
        # Get Hessian w.r.t. torsion and gammadash
        d2J_dtorsion2 = np.asarray(self.d2J_dtorsion2(torsion, gammadash))  # Shape: (n_quad_points, n_quad_points)
        d2J_dgammadash2_flat = np.asarray(self.d2J_dgammadash2(torsion, gammadash))  # Shape: (n_quad_points*3, n_quad_points*3)
        d2J_dtorsion_dgammadash = np.asarray(self.d2J_dtorsion_dgammadash(torsion, gammadash))  # Shape: (n_quad_points, 3, n_quad_points) - already correct
        d2J_dgammadash_dtorsion_flat = np.asarray(self.d2J_dgammadash_dtorsion(torsion, gammadash))  # Shape: (n_quad_points, n_quad_points*3)
        
        # Reshape flattened Hessians to proper shapes
        d2J_dgammadash2 = d2J_dgammadash2_flat.reshape((n_quad_points, 3, n_quad_points, 3))  # Shape: (n_quad_points, 3, n_quad_points, 3)
        # d2J_dtorsion_dgammadash already has shape (n_quad_points, 3, n_quad_points) - no reshape needed
        d2J_dgammadash_dtorsion = d2J_dgammadash_dtorsion_flat.reshape((n_quad_points, n_quad_points, 3))  # Shape: (n_quad_points, n_quad_points, 3)
        
        # First term: dJ/dtorsion * d²torsion/dx²
        # Skip if not available (some curves don't have this implemented)
        try:
            if hasattr(self.curve, 'd2torsion_by_d2coeff_impl') and hasattr(self.curve, 'd2torsion_by_d2coeff_jax'):
                d2torsion_dx2 = np.zeros((n_quad_points, n_dofs, n_dofs))
                self.curve.d2torsion_by_d2coeff_impl(d2torsion_dx2)
                H += np.einsum('k,kij->ij', dJ_dtorsion, d2torsion_dx2)
        except (AttributeError, TypeError):
            pass  # Skip this term if not available
        
        # Second term: (dtorsion/dx)^T @ (d²J/dtorsion²) @ (dtorsion/dx)
        H += np.einsum('ki,kl,lj->ij', dtorsion_dx, d2J_dtorsion2, dtorsion_dx)
        
        # Third term: dJ/dgammadash * d²gammadash/dx²
        # Skip if not available (some curves don't have this implemented)
        try:
            if hasattr(self.curve, 'd2gammadash_by_d2coeff_impl') and hasattr(self.curve, 'd2gammadash_by_d2coeff_jax'):
                d2gammadash_dx2 = np.zeros((n_quad_points, 3, n_dofs, n_dofs))
                self.curve.d2gammadash_by_d2coeff_impl(d2gammadash_dx2)
                # dJ_dgammadash has shape (n_quad_points, 3)
                # d2gammadash_dx2 has shape (n_quad_points, 3, n_dofs, n_dofs)
                # We need to contract over quad points and components
                H += np.einsum('kc,kcij->ij', dJ_dgammadash, d2gammadash_dx2)
        except (AttributeError, TypeError):
            pass  # Skip this term if not available
        
        # Fourth term: (dgammadash/dx)^T @ (d²J/dgammadash²) @ (dgammadash/dx)
        # dgammadash_dx has shape (n_quad_points, 3, n_dofs)
        # d2J_dgammadash2 has shape (n_quad_points, 3, n_quad_points, 3)
        # We need to contract over quad points and components
        H += np.einsum('kci,kclm,lmj->ij', dgammadash_dx, d2J_dgammadash2, dgammadash_dx)
        
        # Fifth term: (dtorsion/dx)^T @ (d²J/(dtorsion dgammadash)) @ (dgammadash/dx)
        # dtorsion_dx has shape (n_quad_points, n_dofs)
        # d2J_dtorsion_dgammadash has shape (n_quad_points, 3, n_quad_points)
        # dgammadash_dx has shape (n_quad_points, 3, n_dofs)
        H += np.einsum('ki,kcl,lcj->ij', dtorsion_dx, d2J_dtorsion_dgammadash, dgammadash_dx)
        
        # Sixth term: (dgammadash/dx)^T @ (d²J/(dgammadash dtorsion)) @ (dtorsion/dx)
        # dgammadash_dx has shape (n_quad_points, 3, n_dofs)
        # d2J_dgammadash_dtorsion has shape (n_quad_points, n_quad_points, 3)
        # dtorsion_dx has shape (n_quad_points, n_dofs)
        H += np.einsum('kci,klc,lj->ij', dgammadash_dx, d2J_dgammadash_dtorsion, dtorsion_dx)
        
        return H

    return_fn_map = {'J': J, 'dJ': dJ, 'd2J': d2J}

def cc_distance_pure(gamma1, l1, gamma2, l2, minimum_distance, downsample=1):
    """
    Compute the curve-curve distance penalty between two curves.

    Args:
        gamma1 (array-like): Points along the first curve.
        l1 (array-like): Tangent vectors along the first curve.
        gamma2 (array-like): Points along the second curve.
        l2 (array-like): Tangent vectors along the second curve.
        minimum_distance (float): The minimum allowed distance between curves.
        downsample (int, default=1): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 

    Returns:
        float: The curve-curve distance penalty value.
    """
    gamma1 = gamma1[::downsample, :]
    gamma2 = gamma2[::downsample, :]
    l1 = l1[::downsample, :]
    l2 = l2[::downsample, :]
    dists = jnp.sqrt(jnp.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
    alen = jnp.linalg.norm(l1, axis=1)[:, None] * jnp.linalg.norm(l2, axis=1)[None, :]
    return jnp.sum(alen * jnp.maximum(minimum_distance-dists, 0)**2)/(gamma1.shape[0]*gamma2.shape[0])


class CurveCurveDistance(Optimizable):
    r"""
    CurveCurveDistance is a class that computes

    .. math::
        J = \sum_{i = 1}^{\text{num_coils}} \sum_{j = 1}^{i-1} d_{i,j}

    where 

    .. math::
        d_{i,j} = \int_{\text{curve}_i} \int_{\text{curve}_j} \max(0, d_{\min} - \| \mathbf{r}_i - \mathbf{r}_j \|_2)^2 ~dl_j ~dl_i\\

    and :math:`\mathbf{r}_i`, :math:`\mathbf{r}_j` are points on coils :math:`i` and :math:`j`, respectively.
    :math:`d_\min` is a desired threshold minimum intercoil distance.  This penalty term is zero when the points on coil :math:`i` and 
    coil :math:`j` lie more than :math:`d_\min` away from one another, for :math:`i, j \in \{1, \cdots, \text{num_coils}\}`

    If num_basecurves is passed, then the code only computes the distance to
    the first `num_basecurves` many curves, which is useful when the coils
    satisfy symmetries that can be exploited.

    """

    def __init__(self, curves, minimum_distance, num_basecurves=None, downsample=1):
        self.curves = curves
        self.minimum_distance = minimum_distance
        self.downsample = downsample
        args = {"static_argnums": (4,)}
        self.J_jax = jit(lambda gamma1, l1, gamma2, l2, dsample: cc_distance_pure(gamma1, l1, gamma2, l2, minimum_distance, dsample), **args)
        self.dJ_dgamma1 = jit(lambda gamma1, l1, gamma2, l2, dsample: grad(self.J_jax, argnums=0)(gamma1, l1, gamma2, l2, dsample), **args)
        self.dJ_dl1 = jit(lambda gamma1, l1, gamma2, l2, dsample: grad(self.J_jax, argnums=1)(gamma1, l1, gamma2, l2, dsample), **args)
        self.dJ_dgamma2 = jit(lambda gamma1, l1, gamma2, l2, dsample: grad(self.J_jax, argnums=2)(gamma1, l1, gamma2, l2, dsample), **args)
        self.dJ_dl2 = jit(lambda gamma1, l1, gamma2, l2, dsample: grad(self.J_jax, argnums=3)(gamma1, l1, gamma2, l2, dsample), **args)
        # Hessian w.r.t. gamma1, l1, gamma2, l2
        # Flatten arrays for Hessian computation to get consistent shapes
        def J_jax_flat_gamma1(gamma1_flat, l1, gamma2, l2, dsample):
            n_quad = l1.shape[0]
            return self.J_jax(gamma1_flat.reshape((n_quad, 3)), l1, gamma2, l2, dsample)
        def J_jax_flat_l1(gamma1, l1_flat, gamma2, l2, dsample):
            n_quad = gamma1.shape[0]
            return self.J_jax(gamma1, l1_flat.reshape((n_quad, 3)), gamma2, l2, dsample)
        def J_jax_flat_gamma2(gamma1, l1, gamma2_flat, l2, dsample):
            n_quad = l2.shape[0]
            return self.J_jax(gamma1, l1, gamma2_flat.reshape((n_quad, 3)), l2, dsample)
        def J_jax_flat_l2(gamma1, l1, gamma2, l2_flat, dsample):
            n_quad = gamma2.shape[0]
            return self.J_jax(gamma1, l1, gamma2, l2_flat.reshape((n_quad, 3)), dsample)
        self.d2J_dgamma12 = jit(lambda gamma1, l1, gamma2, l2, dsample: hessian(J_jax_flat_gamma1, argnums=0)(gamma1.flatten(), l1, gamma2, l2, dsample), **args)
        self.d2J_dl12 = jit(lambda gamma1, l1, gamma2, l2, dsample: hessian(J_jax_flat_l1, argnums=1)(gamma1, l1.flatten(), gamma2, l2, dsample), **args)
        self.d2J_dgamma22 = jit(lambda gamma1, l1, gamma2, l2, dsample: hessian(J_jax_flat_gamma2, argnums=2)(gamma1, l1, gamma2.flatten(), l2, dsample), **args)
        self.d2J_dl22 = jit(lambda gamma1, l1, gamma2, l2, dsample: hessian(J_jax_flat_l2, argnums=3)(gamma1, l1, gamma2, l2.flatten(), dsample), **args)
        # Cross terms
        def dJ_dl1_flat(gamma1, l1_flat, gamma2, l2, dsample):
            n_quad = gamma1.shape[0]
            return grad(self.J_jax, argnums=1)(gamma1, l1_flat.reshape((n_quad, 3)), gamma2, l2, dsample)
        self.d2J_dgamma1_dl1 = jit(lambda gamma1, l1, gamma2, l2, dsample: jacfwd(dJ_dl1_flat, argnums=0)(gamma1, l1.flatten(), gamma2, l2, dsample), **args)
        def dJ_dgamma1_flat(gamma1_flat, l1, gamma2, l2, dsample):
            n_quad = l1.shape[0]
            return grad(self.J_jax, argnums=0)(gamma1_flat.reshape((n_quad, 3)), l1, gamma2, l2, dsample)
        self.d2J_dl1_dgamma1 = jit(lambda gamma1, l1, gamma2, l2, dsample: jacfwd(dJ_dgamma1_flat, argnums=1)(gamma1.flatten(), l1, gamma2, l2, dsample), **args)
        # Cross terms between curves
        def dJ_dgamma2_flat(gamma1, l1, gamma2_flat, l2, dsample):
            n_quad = l2.shape[0]
            return grad(self.J_jax, argnums=2)(gamma1, l1, gamma2_flat.reshape((n_quad, 3)), l2, dsample)
        self.d2J_dgamma1_dgamma2 = jit(lambda gamma1, l1, gamma2, l2, dsample: jacfwd(dJ_dgamma2_flat, argnums=0)(gamma1, l1, gamma2.flatten(), l2, dsample), **args)
        def dJ_dl2_flat(gamma1, l1, gamma2, l2_flat, dsample):
            n_quad = gamma2.shape[0]
            return grad(self.J_jax, argnums=3)(gamma1, l1, gamma2, l2_flat.reshape((n_quad, 3)), dsample)
        self.d2J_dgamma1_dl2 = jit(lambda gamma1, l1, gamma2, l2, dsample: jacfwd(dJ_dl2_flat, argnums=0)(gamma1, l1, gamma2, l2.flatten(), dsample), **args)
        self.d2J_dl1_dgamma2 = jit(lambda gamma1, l1, gamma2, l2, dsample: jacfwd(dJ_dgamma2_flat, argnums=1)(gamma1, l1, gamma2.flatten(), l2, dsample), **args)
        self.d2J_dl1_dl2 = jit(lambda gamma1, l1, gamma2, l2, dsample: jacfwd(dJ_dl2_flat, argnums=1)(gamma1, l1, gamma2, l2.flatten(), dsample), **args)
        # Note: d2J_dgamma2_dgamma1, d2J_dgamma2_dl1, d2J_dl2_dgamma1, d2J_dl2_dl1 are not needed
        # since we use H_ij.T for the symmetric part. They were removed to avoid unnecessary computations.
        def dJ_dgamma2_flat_for_l2(gamma1, l1, gamma2_flat, l2, dsample):
            n_quad = l2.shape[0]
            return grad(self.J_jax, argnums=2)(gamma1, l1, gamma2_flat.reshape((n_quad, 3)), l2, dsample)
        self.d2J_dl2_dgamma2 = jit(lambda gamma1, l1, gamma2, l2, dsample: jacfwd(dJ_dgamma2_flat_for_l2, argnums=3)(gamma1, l1, gamma2.flatten(), l2, dsample), **args)
        self.candidates = None
        self.num_basecurves = num_basecurves or len(curves)
        super().__init__(depends_on=curves)

    def recompute_bell(self, parent=None):
        self.candidates = None

    def compute_candidates(self):
        if self.candidates is None:
            candidates = sopp.get_pointclouds_closer_than_threshold_within_collection(
                [c.gamma()[::self.downsample, :] for c in self.curves], self.minimum_distance, self.num_basecurves)
            self.candidates = candidates

    def shortest_distance_among_candidates(self):
        self.compute_candidates()
        from scipy.spatial.distance import cdist
        return min([self.minimum_distance] + [np.min(cdist(self.curves[i].gamma()[::self.downsample, :],
                                                           self.curves[j].gamma()[::self.downsample, :])) for i, j in self.candidates])

    def shortest_distance(self):
        self.compute_candidates()
        if len(self.candidates) > 0:
            return self.shortest_distance_among_candidates()
        from scipy.spatial.distance import cdist
        return min([np.min(cdist(self.curves[i].gamma()[::self.downsample, :],
                                 self.curves[j].gamma()[::self.downsample, :])) for i in range(len(self.curves)) for j in range(i)])

    def J(self):
        """
        This returns the value of the quantity.
        """
        self.compute_candidates()
        res = 0
        for i, j in self.candidates:
            gamma1 = self.curves[i].gamma()
            l1 = self.curves[i].gammadash()
            gamma2 = self.curves[j].gamma()
            l2 = self.curves[j].gammadash()
            res += self.J_jax(gamma1, l1, gamma2, l2, self.downsample)

        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        self.compute_candidates()
        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in self.curves]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in self.curves]

        for i, j in self.candidates:
            gamma1 = self.curves[i].gamma()
            l1 = self.curves[i].gammadash()
            gamma2 = self.curves[j].gamma()
            l2 = self.curves[j].gammadash()
            dgamma_by_dcoeff_vjp_vecs[i] += self.dJ_dgamma1(gamma1, l1, gamma2, l2, self.downsample)
            dgammadash_by_dcoeff_vjp_vecs[i] += self.dJ_dl1(gamma1, l1, gamma2, l2, self.downsample)
            dgamma_by_dcoeff_vjp_vecs[j] += self.dJ_dgamma2(gamma1, l1, gamma2, l2, self.downsample)
            dgammadash_by_dcoeff_vjp_vecs[j] += self.dJ_dl2(gamma1, l1, gamma2, l2, self.downsample)

        res = [self.curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i]) + self.curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i]) for i in range(len(self.curves))]
        return sum(res)

    def d2J(self):
        """
        Hessian for x = all curve dofs.
        For each curve pair (i, j), we compute:
        d²J/dx² = d/dx (dJ/dx)
                = d/dx (dJ/dgamma1 * dgamma1/dx_i + dJ/dl1 * dl1/dx_i + dJ/dgamma2 * dgamma2/dx_j + dJ/dl2 * dl2/dx_j)
        
        This expands to many terms for each curve pair, and we need to accumulate
        contributions from all pairs.
        """
        self.compute_candidates()
        
        # Get total number of dofs across all curves
        dof_sizes = [c.dof_size for c in self.curves]
        total_dofs = sum(dof_sizes)
        dof_offsets = np.cumsum([0] + dof_sizes[:-1])
        
        # Initialize Hessian matrix
        H = np.zeros((total_dofs, total_dofs))
        
        # Process each candidate pair
        for i, j in self.candidates:
            gamma1 = self.curves[i].gamma()
            l1 = self.curves[i].gammadash()
            gamma2 = self.curves[j].gamma()
            l2 = self.curves[j].gammadash()
            
            # Get first-order derivatives
            dJ_dgamma1 = np.asarray(self.dJ_dgamma1(gamma1, l1, gamma2, l2, self.downsample))
            dJ_dl1 = np.asarray(self.dJ_dl1(gamma1, l1, gamma2, l2, self.downsample))
            dJ_dgamma2 = np.asarray(self.dJ_dgamma2(gamma1, l1, gamma2, l2, self.downsample))
            dJ_dl2 = np.asarray(self.dJ_dl2(gamma1, l1, gamma2, l2, self.downsample))
            
            # Get first-order derivatives w.r.t. curve dofs
            dgamma1_dx = np.asarray(self.curves[i].dgamma_by_dcoeff())  # Shape: (n_quad1, 3, n_dofs_i)
            dl1_dx = np.asarray(self.curves[i].dgammadash_by_dcoeff())  # Shape: (n_quad1, 3, n_dofs_i)
            dgamma2_dx = np.asarray(self.curves[j].dgamma_by_dcoeff())  # Shape: (n_quad2, 3, n_dofs_j)
            dl2_dx = np.asarray(self.curves[j].dgammadash_by_dcoeff())  # Shape: (n_quad2, 3, n_dofs_j)
            
            n_quad1, n_components1, n_dofs_i = dgamma1_dx.shape
            n_quad2, n_components2, n_dofs_j = dgamma2_dx.shape
            
            # Get indices for this curve pair in the global Hessian
            idx_i_start = dof_offsets[i]
            idx_i_end = idx_i_start + n_dofs_i
            idx_j_start = dof_offsets[j]
            idx_j_end = idx_j_start + n_dofs_j
            
            # Get Hessian w.r.t. gamma1, l1, gamma2, l2
            d2J_dgamma12_flat = np.asarray(self.d2J_dgamma12(gamma1, l1, gamma2, l2, self.downsample))  # Shape: (n_quad1*3, n_quad1*3)
            d2J_dl12_flat = np.asarray(self.d2J_dl12(gamma1, l1, gamma2, l2, self.downsample))  # Shape: (n_quad1*3, n_quad1*3)
            d2J_dgamma22_flat = np.asarray(self.d2J_dgamma22(gamma1, l1, gamma2, l2, self.downsample))  # Shape: (n_quad2*3, n_quad2*3)
            d2J_dl22_flat = np.asarray(self.d2J_dl22(gamma1, l1, gamma2, l2, self.downsample))  # Shape: (n_quad2*3, n_quad2*3)
            
            # Reshape flattened Hessians
            d2J_dgamma12 = d2J_dgamma12_flat.reshape((n_quad1, 3, n_quad1, 3))
            d2J_dl12 = d2J_dl12_flat.reshape((n_quad1, 3, n_quad1, 3))
            d2J_dgamma22 = d2J_dgamma22_flat.reshape((n_quad2, 3, n_quad2, 3))
            d2J_dl22 = d2J_dl22_flat.reshape((n_quad2, 3, n_quad2, 3))
            
            # Get cross terms
            # Note: jacfwd returns d(output)/d(input), so we need to transpose
            # d2J_dgamma1_dl1 from jacfwd: [k, c, l, m] = d(dJ/dlc[k, c]) / d(gammac[l, m])
            # But we want: [k, c, l, m] = d(dJ/dlc[l, m]) / d(gammac[k, c])
            # So we transpose: [l, m, k, c] -> [k, c, l, m]
            d2J_dgamma1_dl1_raw = np.asarray(self.d2J_dgamma1_dl1(gamma1, l1, gamma2, l2, self.downsample))  # Shape: (n_quad1, 3, n_quad1, 3)
            d2J_dgamma1_dl1 = d2J_dgamma1_dl1_raw.transpose(2, 3, 0, 1)  # Swap indices to get correct order
                        
            # For cross-curve terms: only compute one direction, use transpose for H_ji
            d2J_dgamma1_dgamma2_raw = np.asarray(self.d2J_dgamma1_dgamma2(gamma1, l1, gamma2, l2, self.downsample))  # Shape: (n_quad1, 3, n_quad2, 3)
            d2J_dgamma1_dgamma2 = d2J_dgamma1_dgamma2_raw.transpose(2, 3, 0, 1)  # Swap indices to get correct order
            
            d2J_dgamma1_dl2_raw = np.asarray(self.d2J_dgamma1_dl2(gamma1, l1, gamma2, l2, self.downsample))  # Shape: (n_quad1, 3, n_quad2, 3)
            d2J_dgamma1_dl2 = d2J_dgamma1_dl2_raw.transpose(2, 3, 0, 1)  # Swap indices to get correct order
            
            d2J_dl1_dgamma2_raw = np.asarray(self.d2J_dl1_dgamma2(gamma1, l1, gamma2, l2, self.downsample))  # Shape: (n_quad1, 3, n_quad2, 3)
            d2J_dl1_dgamma2 = d2J_dl1_dgamma2_raw.transpose(2, 3, 0, 1)  # Swap indices to get correct order
            
            d2J_dl1_dl2_raw = np.asarray(self.d2J_dl1_dl2(gamma1, l1, gamma2, l2, self.downsample))  # Shape: (n_quad1, 3, n_quad2, 3)
            d2J_dl1_dl2 = d2J_dl1_dl2_raw.transpose(2, 3, 0, 1)  # Swap indices to get correct order
            
            # For curve j same-curve cross-term: d2J_dl2_dgamma2 is transpose of d2J_dgamma2_dl2 (if it existed)
            # We compute d2J_dl2_dgamma2 directly since we need it
            d2J_dl2_dgamma2_raw = np.asarray(self.d2J_dl2_dgamma2(gamma1, l1, gamma2, l2, self.downsample))  # Shape: (n_quad2, 3, n_quad2, 3)
            d2J_dl2_dgamma2 = d2J_dl2_dgamma2_raw.transpose(2, 3, 0, 1)  # Swap indices to get correct order
            
            # Terms for curve i (d²J/dx_i²)
            H_ii = np.zeros((n_dofs_i, n_dofs_i))
            
            # First term: dJ/dgamma1 * d²gamma1/dx_i²
            try:
                if hasattr(self.curves[i], 'd2gamma_by_d2coeff_impl') and hasattr(self.curves[i], 'd2gamma_by_d2coeff_jax'):
                    d2gamma1_dx2 = np.zeros((n_quad1, 3, n_dofs_i, n_dofs_i))
                    self.curves[i].d2gamma_by_d2coeff_impl(d2gamma1_dx2)
                    H_ii += np.einsum('kc,kcij->ij', dJ_dgamma1, d2gamma1_dx2)
            except (AttributeError, TypeError):
                pass
            
            # Second term: (dgamma1/dx_i)^T @ (d²J/dgamma1²) @ (dgamma1/dx_i)
            H_ii += np.einsum('kci,kclm,lmj->ij', dgamma1_dx, d2J_dgamma12, dgamma1_dx)
            
            # Third term: dJ/dl1 * d²l1/dx_i²
            try:
                if hasattr(self.curves[i], 'd2gammadash_by_d2coeff_impl') and hasattr(self.curves[i], 'd2gammadash_by_d2coeff_jax'):
                    d2l1_dx2 = np.zeros((n_quad1, 3, n_dofs_i, n_dofs_i))
                    self.curves[i].d2gammadash_by_d2coeff_impl(d2l1_dx2)
                    H_ii += np.einsum('kc,kcij->ij', dJ_dl1, d2l1_dx2)
            except (AttributeError, TypeError):
                pass
            
            # Fourth term: (dl1/dx_i)^T @ (d²J/dl1²) @ (dl1/dx_i)
            H_ii += np.einsum('kci,kclm,lmj->ij', dl1_dx, d2J_dl12, dl1_dx)
            
            # Fifth term: (dgamma1/dx_i)^T @ (d²J/(dgamma1 dl1)) @ (dl1/dx_i)
            # Note: This term is not symmetric by itself, so we need to add its transpose
            # to ensure the Hessian is symmetric. Since d²J/(dl1 dgamma1) = (d²J/(dgamma1 dl1))^T,
            # term6 = term5.T, so we can compute term5 + term5.T directly.
            term5 = np.einsum('kci,kclm,lmj->ij', dgamma1_dx, d2J_dgamma1_dl1, dl1_dx)
            H_ii += term5 + term5.T
            
            # Terms for curve j (d²J/dx_j²)
            H_jj = np.zeros((n_dofs_j, n_dofs_j))
            
            # First term: dJ/dgamma2 * d²gamma2/dx_j²
            try:
                if hasattr(self.curves[j], 'd2gamma_by_d2coeff_impl') and hasattr(self.curves[j], 'd2gamma_by_d2coeff_jax'):
                    d2gamma2_dx2 = np.zeros((n_quad2, 3, n_dofs_j, n_dofs_j))
                    self.curves[j].d2gamma_by_d2coeff_impl(d2gamma2_dx2)
                    H_jj += np.einsum('kc,kcij->ij', dJ_dgamma2, d2gamma2_dx2)
            except (AttributeError, TypeError):
                pass
            
            # Second term: (dgamma2/dx_j)^T @ (d²J/dgamma2²) @ (dgamma2/dx_j)
            H_jj += np.einsum('kci,kclm,lmj->ij', dgamma2_dx, d2J_dgamma22, dgamma2_dx)
            
            # Third term: dJ/dl2 * d²l2/dx_j²
            try:
                if hasattr(self.curves[j], 'd2gammadash_by_d2coeff_impl') and hasattr(self.curves[j], 'd2gammadash_by_d2coeff_jax'):
                    d2l2_dx2 = np.zeros((n_quad2, 3, n_dofs_j, n_dofs_j))
                    self.curves[j].d2gammadash_by_d2coeff_impl(d2l2_dx2)
                    H_jj += np.einsum('kc,kcij->ij', dJ_dl2, d2l2_dx2)
            except (AttributeError, TypeError):
                pass
            
            # Fourth term: (dl2/dx_j)^T @ (d²J/dl2²) @ (dl2/dx_j)
            H_jj += np.einsum('kci,kclm,lmj->ij', dl2_dx, d2J_dl22, dl2_dx)
            
            # Fifth term: (dgamma2/dx_j)^T @ (d²J/(dgamma2 dl2)) @ (dl2/dx_j)
            # d2J_dl2_dgamma2 is d²J/(dl2 dgamma2), so we need its transpose for d²J/(dgamma2 dl2)
            # But d2J_dl2_dgamma2 is already transposed from jacfwd, so we just need to swap indices
            d2J_dgamma2_dl2 = d2J_dl2_dgamma2.transpose(2, 3, 0, 1)  # Transpose: (n_quad2, 3, n_quad2, 3)
            # Note: This term is not symmetric by itself, so we need to add its transpose
            # to ensure the Hessian is symmetric. Since d²J/(dl2 dgamma2) = (d²J/(dgamma2 dl2))^T,
            # term12 = term11.T, so we can compute term11 + term11.T directly.
            term11 = np.einsum('kci,kclm,lmj->ij', dgamma2_dx, d2J_dgamma2_dl2, dl2_dx)
            H_jj += term11 + term11.T
            
            # Cross terms between curves i and j (d²J/(dx_i dx_j))
            H_ij = np.zeros((n_dofs_i, n_dofs_j))
            
            # (dgamma1/dx_i)^T @ (d²J/(dgamma1 dgamma2)) @ (dgamma2/dx_j)
            H_ij += np.einsum('kci,kclm,lmj->ij', dgamma1_dx, d2J_dgamma1_dgamma2, dgamma2_dx)
            
            # (dgamma1/dx_i)^T @ (d²J/(dgamma1 dl2)) @ (dl2/dx_j)
            H_ij += np.einsum('kci,kclm,lmj->ij', dgamma1_dx, d2J_dgamma1_dl2, dl2_dx)
            
            # (dl1/dx_i)^T @ (d²J/(dl1 dgamma2)) @ (dgamma2/dx_j)
            H_ij += np.einsum('kci,kclm,lmj->ij', dl1_dx, d2J_dl1_dgamma2, dgamma2_dx)
            
            # (dl1/dx_i)^T @ (d²J/(dl1 dl2)) @ (dl2/dx_j)
            H_ij += np.einsum('kci,kclm,lmj->ij', dl1_dx, d2J_dl1_dl2, dl2_dx)
            
            # Add contributions to global Hessian
            H[idx_i_start:idx_i_end, idx_i_start:idx_i_end] += H_ii
            H[idx_j_start:idx_j_end, idx_j_start:idx_j_end] += H_jj
            H[idx_i_start:idx_i_end, idx_j_start:idx_j_end] += H_ij
            H[idx_j_start:idx_j_end, idx_i_start:idx_i_end] += H_ij.T  # Symmetry
        
        return H

    return_fn_map = {'J': J, 'dJ': dJ, 'd2J': d2J}

def cs_distance_pure(gammac, lc, gammas, ns, minimum_distance):
    """
    Compute the curve-surface distance penalty between a curve and a surface.

    Args:
        gammac (array-like): Points along the curve.
        lc (array-like): Tangent vectors along the curve.
        gammas (array-like): Points on the surface.
        ns (array-like): Surface normal vectors.
        minimum_distance (float): The minimum allowed distance between curve and surface.

    Returns:
        float: The curve-surface distance penalty value.
    """
    dists = jnp.sqrt(jnp.sum(
        (gammac[:, None, :] - gammas[None, :, :])**2, axis=2))
    integralweight = jnp.linalg.norm(lc, axis=1)[:, None] \
        * jnp.linalg.norm(ns, axis=1)[None, :]
    return jnp.mean(integralweight * jnp.maximum(minimum_distance-dists, 0)**2)


class CurveSurfaceDistance(Optimizable):
    r"""
    CurveSurfaceDistance is a class that computes

    .. math::
        J = \sum_{i = 1}^{\text{num_coils}} d_{i}

    where

    .. math::
        d_{i} = \int_{\text{curve}_i} \int_{surface} \max(0, d_{\min} - \| \mathbf{r}_i - \mathbf{s} \|_2)^2 ~dl_i ~ds\\

    and :math:`\mathbf{r}_i`, :math:`\mathbf{s}` are points on coil :math:`i`
    and the surface, respectively. :math:`d_\min` is a desired threshold
    minimum coil-to-surface distance.  This penalty term is zero when the
    points on all coils :math:`i` and on the surface lie more than
    :math:`d_\min` away from one another.

    """

    def __init__(self, curves, surface, minimum_distance):
        self.curves = curves
        self.surface = surface
        self.minimum_distance = minimum_distance

        self.J_jax = jit(lambda gammac, lc, gammas, ns: cs_distance_pure(gammac, lc, gammas, ns, minimum_distance))
        self.dJ_dgamma = jit(lambda gammac, lc, gammas, ns: grad(self.J_jax, argnums=0)(gammac, lc, gammas, ns))
        self.dJ_dlc = jit(lambda gammac, lc, gammas, ns: grad(self.J_jax, argnums=1)(gammac, lc, gammas, ns))
        
        # Hessian computation methods
        # Note: gammas and ns are arrays, so we can't use static_argnums for them
        # Hessian w.r.t. gammac
        def J_jax_flat_gammac(gammac_flat, lc, gammas, ns):
            n_quad = lc.shape[0]
            return self.J_jax(gammac_flat.reshape((n_quad, 3)), lc, gammas, ns)
        self.d2J_dgamma2 = jit(lambda gammac, lc, gammas, ns: hessian(J_jax_flat_gammac, argnums=0)(gammac.flatten(), lc, gammas, ns))
        # Hessian w.r.t. lc
        def J_jax_flat_lc(gammac, lc_flat, gammas, ns):
            n_quad = gammac.shape[0]
            return self.J_jax(gammac, lc_flat.reshape((n_quad, 3)), gammas, ns)
        self.d2J_dlc2 = jit(lambda gammac, lc, gammas, ns: hessian(J_jax_flat_lc, argnums=1)(gammac, lc.flatten(), gammas, ns))
        # Cross terms
        def dJ_dlc_flat(gammac, lc_flat, gammas, ns):
            n_quad = gammac.shape[0]
            return grad(self.J_jax, argnums=1)(gammac, lc_flat.reshape((n_quad, 3)), gammas, ns)
        self.d2J_dgamma_dlc = jit(lambda gammac, lc, gammas, ns: jacfwd(dJ_dlc_flat, argnums=0)(gammac, lc.flatten(), gammas, ns))
        def dJ_dgamma_flat(gammac_flat, lc, gammas, ns):
            n_quad = lc.shape[0]
            return grad(self.J_jax, argnums=0)(gammac_flat.reshape((n_quad, 3)), lc, gammas, ns)
        self.d2J_dlc_dgamma = jit(lambda gammac, lc, gammas, ns: jacfwd(dJ_dgamma_flat, argnums=1)(gammac.flatten(), lc, gammas, ns))
        
        self.candidates = None
        super().__init__(depends_on=curves)  # Bharat's comment: Shouldn't we add surface here

    def recompute_bell(self, parent=None):
        self.candidates = None

    def compute_candidates(self):
        if self.candidates is None:
            candidates = sopp.get_pointclouds_closer_than_threshold_between_two_collections(
                [c.gamma() for c in self.curves], [self.surface.gamma().reshape((-1, 3))], self.minimum_distance)
            self.candidates = candidates

    def shortest_distance_among_candidates(self):
        self.compute_candidates()
        from scipy.spatial.distance import cdist
        xyz_surf = self.surface.gamma().reshape((-1, 3))
        return min([self.minimum_distance] + [np.min(cdist(self.curves[i].gamma(), xyz_surf)) for i, _ in self.candidates])

    def shortest_distance(self):
        self.compute_candidates()
        if len(self.candidates) > 0:
            return self.shortest_distance_among_candidates()
        from scipy.spatial.distance import cdist
        xyz_surf = self.surface.gamma().reshape((-1, 3))
        return min([np.min(cdist(self.curves[i].gamma(), xyz_surf)) for i in range(len(self.curves))])

    def J(self):
        """
        This returns the value of the quantity.
        """
        self.compute_candidates()
        res = 0
        gammas = self.surface.gamma().reshape((-1, 3))
        ns = self.surface.normal().reshape((-1, 3))
        for i, _ in self.candidates:
            gammac = self.curves[i].gamma()
            lc = self.curves[i].gammadash()
            res += self.J_jax(gammac, lc, gammas, ns)
        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        self.compute_candidates()
        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in self.curves]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in self.curves]
        gammas = self.surface.gamma().reshape((-1, 3))

        gammas = self.surface.gamma().reshape((-1, 3))
        ns = self.surface.normal().reshape((-1, 3))
        for i, _ in self.candidates:
            gammac = self.curves[i].gamma()
            lc = self.curves[i].gammadash()
            dgamma_by_dcoeff_vjp_vecs[i] += self.dJ_dgamma(gammac, lc, gammas, ns)
            dgammadash_by_dcoeff_vjp_vecs[i] += self.dJ_dlc(gammac, lc, gammas, ns)
        res = [self.curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i]) + self.curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i]) for i in range(len(self.curves))]
        return sum(res)

    def d2J(self):
        """
        Hessian for x = all curve dofs.
        For each curve i, we compute:
        d²J/dx² = d/dx (dJ/dx)
                = d/dx (dJ/dgammac * dgammac/dx_i + dJ/dlc * dlc/dx_i)
        
        This expands to:
        = dJ/dgammac * d²gammac/dx_i² + (dgammac/dx_i)^T @ (d²J/dgammac²) @ (dgammac/dx_i)
        + dJ/dlc * d²lc/dx_i² + (dlc/dx_i)^T @ (d²J/dlc²) @ (dlc/dx_i)
        + (dgammac/dx_i)^T @ (d²J/(dgammac dlc)) @ (dlc/dx_i)
        + (dlc/dx_i)^T @ (d²J/(dlc dgammac)) @ (dgammac/dx_i)
        """
        self.compute_candidates()
        
        # Get total number of dofs across all curves
        dof_sizes = [c.dof_size for c in self.curves]
        total_dofs = sum(dof_sizes)
        dof_offsets = np.cumsum([0] + dof_sizes[:-1])
        
        # Initialize Hessian matrix
        H = np.zeros((total_dofs, total_dofs))
        
        # Get surface data (fixed)
        gammas = self.surface.gamma().reshape((-1, 3))
        ns = self.surface.normal().reshape((-1, 3))
        
        # Process each candidate curve
        for i, _ in self.candidates:
            gammac = self.curves[i].gamma()
            lc = self.curves[i].gammadash()
            
            # Get first-order derivatives
            dJ_dgammac = np.asarray(self.dJ_dgamma(gammac, lc, gammas, ns))
            dJ_dlc = np.asarray(self.dJ_dlc(gammac, lc, gammas, ns))
            
            # Get first-order derivatives w.r.t. curve dofs
            dgammac_dx = np.asarray(self.curves[i].dgamma_by_dcoeff())  # Shape: (n_quad, 3, n_dofs_i)
            dlc_dx = np.asarray(self.curves[i].dgammadash_by_dcoeff())  # Shape: (n_quad, 3, n_dofs_i)
            
            n_quad, n_components, n_dofs_i = dgammac_dx.shape
            
            # Get indices for this curve in the global Hessian
            idx_i_start = dof_offsets[i]
            idx_i_end = idx_i_start + n_dofs_i
            
            # Get Hessian w.r.t. gammac and lc
            d2J_dgammac2_flat = np.asarray(self.d2J_dgamma2(gammac, lc, gammas, ns))  # Shape: (n_quad*3, n_quad*3)
            d2J_dlc2_flat = np.asarray(self.d2J_dlc2(gammac, lc, gammas, ns))  # Shape: (n_quad*3, n_quad*3)
            
            # Reshape flattened Hessians
            d2J_dgammac2 = d2J_dgammac2_flat.reshape((n_quad, 3, n_quad, 3))
            d2J_dlc2 = d2J_dlc2_flat.reshape((n_quad, 3, n_quad, 3))
            
            # Get cross terms
            # Note: jacfwd returns d(output)/d(input), so we need to transpose
            # d2J_dgamma_dlc from jacfwd: [k, c, l, m] = d(dJ/dlc[k, c]) / d(gammac[l, m])
            # But we want: [k, c, l, m] = d(dJ/dlc[l, m]) / d(gammac[k, c])
            # So we transpose: [l, m, k, c] -> [k, c, l, m]
            d2J_dgammac_dlc_raw = np.asarray(self.d2J_dgamma_dlc(gammac, lc, gammas, ns))  # Shape: (n_quad, 3, n_quad, 3)
            d2J_dgammac_dlc = d2J_dgammac_dlc_raw.transpose(2, 3, 0, 1)  # Swap indices to get correct order
            
            # Terms for curve i (d²J/dx_i²)
            H_ii = np.zeros((n_dofs_i, n_dofs_i))
            
            # First term: dJ/dgammac * d²gammac/dx_i²
            try:
                if hasattr(self.curves[i], 'd2gamma_by_d2coeff_impl') and hasattr(self.curves[i], 'd2gamma_by_d2coeff_jax'):
                    d2gammac_dx2 = np.zeros((n_quad, 3, n_dofs_i, n_dofs_i))
                    self.curves[i].d2gamma_by_d2coeff_impl(d2gammac_dx2)
                    H_ii += np.einsum('kc,kcij->ij', dJ_dgammac, d2gammac_dx2)
            except (AttributeError, TypeError):
                pass
            
            # Second term: (dgammac/dx_i)^T @ (d²J/dgammac²) @ (dgammac/dx_i)
            H_ii += np.einsum('kci,kclm,lmj->ij', dgammac_dx, d2J_dgammac2, dgammac_dx)
            
            # Third term: dJ/dlc * d²lc/dx_i²
            try:
                if hasattr(self.curves[i], 'd2gammadash_by_d2coeff_impl') and hasattr(self.curves[i], 'd2gammadash_by_d2coeff_jax'):
                    d2lc_dx2 = np.zeros((n_quad, 3, n_dofs_i, n_dofs_i))
                    self.curves[i].d2gammadash_by_d2coeff_impl(d2lc_dx2)
                    H_ii += np.einsum('kc,kcij->ij', dJ_dlc, d2lc_dx2)
            except (AttributeError, TypeError):
                pass
            
            # Fourth term: (dlc/dx_i)^T @ (d²J/dlc²) @ (dlc/dx_i)
            H_ii += np.einsum('kci,kclm,lmj->ij', dlc_dx, d2J_dlc2, dlc_dx)
            
            # Fifth term: (dgammac/dx_i)^T @ (d²J/(dgammac dlc)) @ (dlc/dx_i)
            # Note: This term is not symmetric by itself, so we need to add its transpose
            # to ensure the Hessian is symmetric. Since d²J/(dlc dgammac) = (d²J/(dgammac dlc))^T,
            # term6 = term5.T, so we can compute term5 + term5.T directly.
            term5 = np.einsum('kci,kclm,lmj->ij', dgammac_dx, d2J_dgammac_dlc, dlc_dx)
            H_ii += term5 + term5.T
            
            # Add contributions to global Hessian
            H[idx_i_start:idx_i_end, idx_i_start:idx_i_end] += H_ii
        
        return H

    return_fn_map = {'J': J, 'dJ': dJ, 'd2J': d2J}


@jit
def curve_arclengthvariation_pure(l, mat):
    """
    Compute the variance of the average incremental arclengths over intervals.

    Args:
        l (array-like): Incremental arclengths along the curve.
        mat (array-like): Matrix mapping arclengths to intervals.

    Returns:
        float: The variance of the average arclengths over intervals.
    """
    return jnp.var(mat @ l)


class ArclengthVariation(Optimizable):

    def __init__(self, curve, nintervals="full"):
        r"""
        This class penalizes variation of the arclength along a curve.
        The idea of this class is to avoid ill-posedness of curve objectives due to
        non-uniqueness of the underlying parametrization. Essentially we want to
        achieve constant arclength along the curve. Since we can not expect
        perfectly constant arclength along the entire curve, this class has
        some support to relax this notion. Consider a partition of the :math:`[0, 1]`
        interval into intervals :math:`\{I_i\}_{i=1}^L`, and tenote the average incremental arclength
        on interval :math:`I_i` by :math:`\ell_i`. This objective then penalises the variance

        .. math::
            J = \mathrm{Var}(\ell_i)

        it remains to choose the number of intervals :math:`L` that :math:`[0, 1]` is split into.
        If ``nintervals="full"``, then the number of intervals :math:`L` is equal to the number of quadrature
        points of the curve. If ``nintervals="partial"``, then the argument is as follows:

        A curve in 3d space is defined uniquely by an initial point, an initial
        direction, and the arclength, curvature, and torsion along the curve. For a
        :mod:`simsopt.geo.curvexyzfourier.CurveXYZFourier`, the intuition is now as
        follows: assuming that the curve has order :math:`p`, that means we have
        :math:`3*(2p+1)` degrees of freedom in total. Assuming that three each are
        required for both the initial position and direction, :math:`6p-3` are left
        over for curvature, torsion, and arclength. We want to fix the arclength,
        so we can afford :math:`2p-1` constraints, which corresponds to
        :math:`L=2p`.

        Finally, the user can also provide an integer value for `nintervals`
        and thus specify the number of intervals directly.
        """
        super().__init__(depends_on=[curve])

        assert nintervals in ["full", "partial"] \
            or (isinstance(nintervals, int) and 0 < nintervals <= curve.gamma().shape[0])
        self.curve = curve
        nquadpoints = len(curve.quadpoints)
        if nintervals == "full":
            nintervals = curve.gamma().shape[0]
        elif nintervals == "partial":
            from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier
            if isinstance(curve, CurveXYZFourier) or isinstance(curve, JaxCurveXYZFourier):
                nintervals = 2*curve.order
            else:
                raise RuntimeError("Please provide a value other than `partial` for `nintervals`. We only have a default for `CurveXYZFourier` and `JaxCurveXYZFourier`.")

        self.nintervals = nintervals
        indices = np.floor(np.linspace(0, nquadpoints, nintervals+1, endpoint=True)).astype(int)
        mat = np.zeros((nintervals, nquadpoints))
        for i in range(nintervals):
            mat[i, indices[i]:indices[i+1]] = 1/(indices[i+1]-indices[i])
        self.mat = mat
        self.dJ_dl = jit(lambda l: grad(lambda x: curve_arclengthvariation_pure(x, mat))(l))
        self.d2J_dl2 = jit(lambda l: hessian(lambda x: curve_arclengthvariation_pure(x, mat))(l))

    def J(self):
        return float(curve_arclengthvariation_pure(self.curve.incremental_arclength(), self.mat))

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        return self.curve.dincremental_arclength_by_dcoeff_vjp(
            self.dJ_dl(self.curve.incremental_arclength()))

    def d2J(self):
        """
        Hessian for x = coil_dofs:
        d²J/dx² = d/dx (dJ/dx)
                = d/dx (dJ/dl * dl/dx)
                = dJ/dl * d²l/dx² + (dl/dx)^T @ (d²J/dl²) @ (dl/dx)
        """
        l = self.curve.incremental_arclength()
        dJ_dl = self.dJ_dl(l)
        dl_dx = self.curve.dincremental_arclength_by_dcoeff()  # dl/dx has shape (n_quad_points, n_dofs)
        n_quad_points, n_dofs = dl_dx.shape
        H = np.zeros((n_dofs, n_dofs))
        
        # First term: dJ/dl * d²l/dx²
        if hasattr(self.curve, 'd2incremental_arclength_by_d2coeff_impl') and hasattr(self.curve, 'd2incremental_arclength_by_d2coeff_jax'):
            # Get the full Hessian tensor d²l/dx² with shape (n_quad_points, n_dofs, n_dofs)
            d2l_dx2 = np.zeros((n_quad_points, n_dofs, n_dofs))
            self.curve.d2incremental_arclength_by_d2coeff_impl(d2l_dx2)
            # Compute first term using einsum: sum over quad points
            H += np.einsum('k,kij->ij', dJ_dl, d2l_dx2)
        
        # Second term: (dl/dx)^T @ (d²J/dl²) @ (dl/dx)
        d2J_dl2_matrix = self.d2J_dl2(l)  # Shape: (n_quad_points, n_quad_points)
        # Compute using einsum: (dl/dx)^T @ (d²J/dl²) @ (dl/dx)
        H += np.einsum('ki,kl,lj->ij', dl_dx, d2J_dl2_matrix, dl_dx)
        
        return H

    return_fn_map = {'J': J, 'dJ': dJ, 'd2J': d2J}


@jit
def curve_msc_pure(kappa, gammadash):
    """
    Compute the mean squared curvature objective for a curve.

    Args:
        kappa (array-like): Curvature values along the curve.
        gammadash (array-like): Tangent vectors along the curve.

    Returns:
        float: The mean squared curvature value.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return jnp.mean(kappa**2 * arc_length)/jnp.mean(arc_length)


class MeanSquaredCurvature(Optimizable):

    def __init__(self, curve):
        r"""
        Compute the mean of the squared curvature of a curve.

        .. math::
            J = (1/L) \int_{\text{curve}} \kappa^2 ~dl

        where :math:`L` is the curve length, :math:`\ell` is the incremental
        arclength, and :math:`\kappa` is the curvature.

        Args:
            curve: the curve of which the curvature should be computed.
        """
        super().__init__(depends_on=[curve])
        self.curve = curve
        self.dJ_dkappa = jit(lambda kappa, gammadash: grad(curve_msc_pure, argnums=0)(kappa, gammadash))
        self.dJ_dgammadash = jit(lambda kappa, gammadash: grad(curve_msc_pure, argnums=1)(kappa, gammadash))
        
        # Hessian computation methods
        # Hessian w.r.t. kappa
        def J_msc_flat_kappa(kappa_flat, gammadash, kappa_shape):
            n_quad = gammadash.shape[0]
            return curve_msc_pure(kappa_flat.reshape((n_quad,)), gammadash)
        self.d2J_dkappa2 = jit(lambda kappa, gammadash: hessian(lambda k: J_msc_flat_kappa(k, gammadash, kappa.shape))(kappa.flatten()))
        # Hessian w.r.t. gammadash
        def J_msc_flat_gammadash(kappa, gammadash_flat, gammadash_shape):
            n_quad = kappa.shape[0]
            return curve_msc_pure(kappa, gammadash_flat.reshape((n_quad, 3)))
        self.d2J_dgammadash2 = jit(lambda kappa, gammadash: hessian(lambda g: J_msc_flat_gammadash(kappa, g, gammadash.shape))(gammadash.flatten()))
        # Cross terms
        def dJ_dgammadash_flat(kappa, gammadash_flat, gammadash_shape):
            n_quad = kappa.shape[0]
            return grad(curve_msc_pure, argnums=1)(kappa, gammadash_flat.reshape((n_quad, 3)))
        self.d2J_dkappa_dgammadash = jit(lambda kappa, gammadash: jacfwd(dJ_dgammadash_flat, argnums=0)(kappa, gammadash.flatten(), gammadash.shape))
        def dJ_dkappa_flat(kappa_flat, gammadash, kappa_shape):
            n_quad = gammadash.shape[0]
            return grad(curve_msc_pure, argnums=0)(kappa_flat.reshape((n_quad,)), gammadash)
        self.d2J_dgammadash_dkappa = jit(lambda kappa, gammadash: jacfwd(dJ_dkappa_flat, argnums=1)(kappa.flatten(), gammadash, kappa.shape))

    def J(self):
        return float(curve_msc_pure(self.curve.kappa(), self.curve.gammadash()))

    @derivative_dec
    def dJ(self):
        grad0 = self.dJ_dkappa(self.curve.kappa(), self.curve.gammadash())
        grad1 = self.dJ_dgammadash(self.curve.kappa(), self.curve.gammadash())
        return self.curve.dkappa_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)

    def d2J(self):
        """
        Hessian for x = coil_dofs:
        d²J/dx² = d/dx (dJ/dx)
                = d/dx (dJ/dkappa * dkappa/dx + dJ/dgammadash * dgammadash/dx)
                = dJ/dkappa * d²kappa/dx² + (dkappa/dx)^T @ (d²J/dkappa²) @ (dkappa/dx)
                + dJ/dgammadash * d²gammadash/dx² + (dgammadash/dx)^T @ (d²J/dgammadash²) @ (dgammadash/dx)
                + (dkappa/dx)^T @ (d²J/(dkappa dgammadash)) @ (dgammadash/dx)
                + (dgammadash/dx)^T @ (d²J/(dgammadash dkappa)) @ (dkappa/dx)
        """
        kappa = self.curve.kappa()
        gammadash = self.curve.gammadash()
        
        # Get first-order derivatives
        dJ_dkappa = np.asarray(self.dJ_dkappa(kappa, gammadash))
        dJ_dgammadash = np.asarray(self.dJ_dgammadash(kappa, gammadash))
        
        # Get first-order derivatives w.r.t. curve dofs
        dkappa_dx = np.asarray(self.curve.dkappa_by_dcoeff())  # Shape: (n_quad_points, n_dofs)
        dgammadash_dx = np.asarray(self.curve.dgammadash_by_dcoeff())  # Shape: (n_quad_points, 3, n_dofs)
        
        n_quad_points, n_dofs = dkappa_dx.shape
        n_quad_points_g, n_components, n_dofs_g = dgammadash_dx.shape
        assert n_quad_points == n_quad_points_g, "Mismatch in number of quad points"
        assert n_dofs == n_dofs_g, "Mismatch in number of dofs"
        
        H = np.zeros((n_dofs, n_dofs))
        
        # Get Hessian w.r.t. kappa and gammadash
        d2J_dkappa2 = np.asarray(self.d2J_dkappa2(kappa, gammadash))  # Shape: (n_quad_points, n_quad_points)
        d2J_dgammadash2_flat = np.asarray(self.d2J_dgammadash2(kappa, gammadash))  # Shape: (n_quad_points*3, n_quad_points*3)
        d2J_dkappa_dgammadash = np.asarray(self.d2J_dkappa_dgammadash(kappa, gammadash))  # Shape: (n_quad_points, 3, n_quad_points)
        d2J_dgammadash_dkappa = np.asarray(self.d2J_dgammadash_dkappa(kappa, gammadash))  # Shape: (n_quad_points, n_quad_points, 3)
        
        # Reshape flattened Hessian to proper shape
        d2J_dgammadash2 = d2J_dgammadash2_flat.reshape((n_quad_points, 3, n_quad_points, 3))
        
        # First term: dJ/dkappa * d²kappa/dx²
        try:
            if hasattr(self.curve, 'd2kappa_by_d2coeff_impl') and hasattr(self.curve, 'd2kappa_by_d2coeff_jax'):
                d2kappa_dx2 = np.zeros((n_quad_points, n_dofs, n_dofs))
                self.curve.d2kappa_by_d2coeff_impl(d2kappa_dx2)
                H += np.einsum('k,kij->ij', dJ_dkappa, d2kappa_dx2)
        except (AttributeError, TypeError):
            pass
        
        # Second term: (dkappa/dx)^T @ (d²J/dkappa²) @ (dkappa/dx)
        H += np.einsum('ki,kl,lj->ij', dkappa_dx, d2J_dkappa2, dkappa_dx)
        
        # Third term: dJ/dgammadash * d²gammadash/dx²
        try:
            if hasattr(self.curve, 'd2gammadash_by_d2coeff_impl') and hasattr(self.curve, 'd2gammadash_by_d2coeff_jax'):
                d2gammadash_dx2 = np.zeros((n_quad_points, 3, n_dofs, n_dofs))
                self.curve.d2gammadash_by_d2coeff_impl(d2gammadash_dx2)
                H += np.einsum('kc,kcij->ij', dJ_dgammadash, d2gammadash_dx2)
        except (AttributeError, TypeError):
            pass
        
        # Fourth term: (dgammadash/dx)^T @ (d²J/dgammadash²) @ (dgammadash/dx)
        H += np.einsum('kci,kclm,lmj->ij', dgammadash_dx, d2J_dgammadash2, dgammadash_dx)
        
        # Fifth term: (dkappa/dx)^T @ (d²J/(dkappa dgammadash)) @ (dgammadash/dx)
        # d2J_dkappa_dgammadash has shape (n_quad_points, 3, n_quad_points) = d²J/(dkappa[k] dgammadash[l, c])
        # We need to contract: sum_k sum_c sum_l (dkappa/dx)[k, i] * (d²J/(dkappa dgammadash))[k, c, l] * (dgammadash/dx)[l, c, j]
        H += np.einsum('ki,kcl,lcj->ij', dkappa_dx, d2J_dkappa_dgammadash, dgammadash_dx)
        
        # Sixth term: (dgammadash/dx)^T @ (d²J/(dgammadash dkappa)) @ (dkappa/dx)
        # d2J_dgammadash_dkappa has shape (n_quad_points, n_quad_points, 3) = d²J/(dgammadash[k, c] dkappa[l])
        # We need to contract: sum_k sum_l sum_c (dgammadash/dx)[k, c, i] * (d²J/(dgammadash dkappa))[k, l, c] * (dkappa/dx)[l, j]
        H += np.einsum('kci,klc,lj->ij', dgammadash_dx, d2J_dgammadash_dkappa, dkappa_dx)
        
        return H

    return_fn_map = {'J': J, 'dJ': dJ, 'd2J': d2J}


@deprecated("`MinimumDistance` has been deprecated and will be removed. Please use `CurveCurveDistance` instead.")
class MinimumDistance(CurveCurveDistance):
    pass


class LinkingNumber(Optimizable):

    def __init__(self, curves, downsample=1):
        Optimizable.__init__(self, depends_on=curves)
        self.curves = curves
        for curve in curves:
            assert np.mod(len(curve.quadpoints), downsample) == 0, f"Downsample {downsample} does not divide the number of quadpoints {len(curve.quadpoints)}."

        self.downsample = downsample
        self.dphis = np.array([(c.quadpoints[1] - c.quadpoints[0]) * downsample for c in self.curves])

        r"""
        Compute the Gauss linking number of a set of curves, i.e. whether the curves
        are interlocked or not.

        The value is an integer, >= 1 if the curves are interlocked, 0 if not. For each pair
        of curves, the contribution to the linking number is
        
        .. math::
            Link(c_1, c_2) = \frac{1}{4\pi} \left| \oint_{c_1}\oint_{c_2}\frac{\textbf{r}_1 - \textbf{r}_2}{|\textbf{r}_1 - \textbf{r}_2|^3} (d\textbf{r}_1 \times d\textbf{r}_2) \right|
            
        where :math:`c_1` is the first curve, :math:`c_2` is the second curve,
        :math:`\textbf{r}_1` is the position vector along the first curve, and
        :math:`\textbf{r}_2` is the position vector along the second curve.

        Args:
            curves (list of Curve, shape (n_curves)): 
                The set of curves for which the linking number should be computed.
            downsample (int, default=1): 
                Factor by which to downsample the quadrature points 
                by skipping through the array by a factor of ``downsample``,
                e.g. curve.gamma()[::downsample, :]. 
                Setting this parameter to a value larger than 1 will speed up the calculation,
                which may be useful if the set of coils is large, though it may introduce
                inaccuracy if ``downsample`` is set too large, or not a multiple of the 
                total number of quadrature points (since this will produce a nonuniform set of points). 
                This parameter is used to speed up expensive calculations during optimization, 
                while retaining higher accuracy for the other objectives. 
        """

    def J(self):
        return sopp.compute_linking_number(
            [c.gamma() for c in self.curves],
            [c.gammadash() for c in self.curves],
            self.dphis,
            self.downsample,
        )

    @derivative_dec
    def dJ(self):
        return Derivative({})
    
    def d2J(self):
        """
        Hessian for linking number. Since the linking number is a topological invariant,
        its derivative and Hessian are zero for smooth deformations that don't cause
        curve crossings.
        
        Returns:
            Zero Hessian matrix with shape (total_dofs, total_dofs) where total_dofs
            is the sum of all curve degrees of freedom.
        """
        # Get total number of dofs across all curves
        total_dofs = sum(c.dof_size for c in self.curves)
        return np.zeros((total_dofs, total_dofs))

    return_fn_map = {'J': J, 'dJ': dJ, 'd2J': d2J}