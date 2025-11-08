import numpy as np
import jax.numpy as jnp
from jax import grad, jit as jaxjit
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

    def __init__(self, surface, field, target=None, definition="quadratic flux"):
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
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    def J(self):
        n = self.surface.normal()
        Bcoil = self.field.B().reshape(n.shape)
        return sopp.integral_BdotN(Bcoil, self.target, n, self.definition)

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

    def __init__(self, surface, field, target=None, definition="quadratic flux"):
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

    def J(self):
        n = self.surface.normal()
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        Bcoil = self.field.B().reshape(n.shape)
        return self.J_jax(jnp.asarray(Bcoil), jnp.asarray(self.target), jnp.asarray(n), self.definition)

    @derivative_dec
    def dJ(self):
        """Return Jacobian with respect to both coil and plasma dofs"""
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        n = self.surface.normal()
        Bcoil = self.field.B().reshape(n.shape)
        return self.field.B_vjp(self.dJ_dBcoil(jnp.asarray(Bcoil), jnp.asarray(self.target), jnp.asarray(n))) #\
            # + Derivative({self.surface.x: self.dtarget_dcoefs(self.dJ_dtarget(Bcoil, self.target, n))}) \
            # + Derivative({self.surface.x: self.dnormals_dcoefs(self.dJ_dnormals(Bcoil, self.target, n))})
