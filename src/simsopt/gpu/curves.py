"""Functional Fourier-curve geometry for device-resident objectives."""

from __future__ import annotations

from math import cos, pi, sin

import jax.numpy as jnp
import numpy as np


def fourier_basis(quadpoints, order: int, derivative: int = 0, dtype=None):
    r"""Build a Cartesian Fourier basis compatible with CurveXYZFourier.

    Columns follow SIMSOPT's per-coordinate degree-of-freedom order
    [c0, s1, c1, ..., s_order, c_order]. Derivatives are with respect
    to the normalized curve parameter in [0, 1).
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    if derivative < 0:
        raise ValueError("derivative must be non-negative")

    points = jnp.asarray(quadpoints, dtype=dtype)
    if points.ndim != 1:
        raise ValueError("quadpoints must be one-dimensional")

    constant = jnp.ones_like(points) if derivative == 0 else jnp.zeros_like(points)
    columns = [constant]
    for mode in range(1, order + 1):
        angular_frequency = 2 * pi * mode
        phase = angular_frequency * points
        scale = angular_frequency**derivative
        sine = jnp.sin(phase)
        cosine = jnp.cos(phase)
        sine_derivatives = (sine, cosine, -sine, -cosine)
        cosine_derivatives = (cosine, -sine, -cosine, sine)
        cycle_index = derivative % 4
        columns.extend(
            (
                scale * sine_derivatives[cycle_index],
                scale * cosine_derivatives[cycle_index],
            )
        )
    return jnp.stack(columns, axis=-1)


def fourier_basis_set(quadpoints, order: int, max_derivative: int = 3, dtype=None):
    """Return bases for derivatives zero through max_derivative."""
    if max_derivative < 0:
        raise ValueError("max_derivative must be non-negative")
    return jnp.stack(
        [
            fourier_basis(quadpoints, order, derivative, dtype=dtype)
            for derivative in range(max_derivative + 1)
        ],
        axis=0,
    )


def dofs_to_coefficients(dofs):
    """Reshape flattened curve dofs to (..., 3, 2*order+1)."""
    dofs = jnp.asarray(dofs)
    if dofs.ndim < 1 or dofs.shape[-1] % 3 != 0:
        raise ValueError("the final dof dimension must be divisible by three")
    coefficients_per_axis = dofs.shape[-1] // 3
    if coefficients_per_axis % 2 != 1:
        raise ValueError("each axis must have 2*order+1 Fourier coefficients")
    return dofs.reshape(dofs.shape[:-1] + (3, coefficients_per_axis))


def coefficients_to_dofs(coefficients):
    """Flatten (..., 3, 2*order+1) coefficients in SIMSOPT order."""
    coefficients = jnp.asarray(coefficients)
    if (
        coefficients.ndim < 2
        or coefficients.shape[-2] != 3
        or coefficients.shape[-1] % 2 != 1
    ):
        raise ValueError("coefficients must have shape (..., 3, 2*order+1)")
    return coefficients.reshape(coefficients.shape[:-2] + (-1,))


def evaluate_cartesian_fourier(coefficients, basis):
    """Evaluate one derivative of one or more Cartesian Fourier curves."""
    coefficients = jnp.asarray(coefficients)
    basis = jnp.asarray(basis)
    if coefficients.ndim < 2 or coefficients.shape[-2] != 3:
        raise ValueError("coefficients must have shape (..., 3, ncoeff)")
    if basis.ndim != 2:
        raise ValueError("basis must have shape (nquad, ncoeff)")
    if coefficients.shape[-1] != basis.shape[-1]:
        raise ValueError("coefficient and basis sizes do not match")
    return jnp.einsum("...ac,qc->...qa", coefficients, basis)


def evaluate_cartesian_fourier_derivatives(coefficients, bases):
    """Evaluate derivative bases, returning (nderiv, ..., nquad, 3)."""
    coefficients = jnp.asarray(coefficients)
    bases = jnp.asarray(bases)
    if coefficients.ndim < 2 or coefficients.shape[-2] != 3:
        raise ValueError("coefficients must have shape (..., 3, ncoeff)")
    if bases.ndim != 3:
        raise ValueError("bases must have shape (nderiv, nquad, ncoeff)")
    if coefficients.shape[-1] != bases.shape[-1]:
        raise ValueError("coefficient and basis sizes do not match")
    return jnp.einsum("...ac,rqc->r...qa", coefficients, bases)


def symmetry_transforms(nfp: int, stellsym: bool, dtype=None) -> tuple:
    """Return row-vector transforms and current signs in SIMSOPT coil order."""
    if nfp <= 0:
        raise ValueError("nfp must be positive")

    transforms = []
    current_signs = []
    flip_values = (False, True) if stellsym else (False,)
    reflection = np.diag([1.0, -1.0, -1.0])
    for field_period in range(nfp):
        angle = 2 * pi * field_period / nfp
        rotation = np.asarray(
            [
                [cos(angle), -sin(angle), 0.0],
                [sin(angle), cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ).T
        for flip in flip_values:
            transforms.append(rotation @ reflection if flip else rotation)
            current_signs.append(-1.0 if flip else 1.0)
    return jnp.asarray(transforms, dtype=dtype), jnp.asarray(current_signs, dtype=dtype)


def expand_by_symmetry(
    base_gamma, base_gammadash, base_currents, transforms, current_signs
):
    """Expand base coils in the same order as coils_via_symmetries."""
    base_gamma = jnp.asarray(base_gamma)
    base_gammadash = jnp.asarray(base_gammadash)
    base_currents = jnp.asarray(base_currents)
    transforms = jnp.asarray(transforms)
    current_signs = jnp.asarray(current_signs)

    if base_gamma.ndim != 3 or base_gamma.shape[-1] != 3:
        raise ValueError("base_gamma must have shape (nbase, nquad, 3)")
    if base_gammadash.shape != base_gamma.shape:
        raise ValueError("base_gammadash must match base_gamma")
    if base_currents.shape != (base_gamma.shape[0],):
        raise ValueError("base_currents must have shape (nbase,)")
    if transforms.ndim != 3 or transforms.shape[1:] != (3, 3):
        raise ValueError("transforms must have shape (ntransform, 3, 3)")
    if current_signs.shape != (transforms.shape[0],):
        raise ValueError("current_signs must have shape (ntransform,)")

    ntransforms = transforms.shape[0]
    nbase, nquad, _ = base_gamma.shape
    gamma = jnp.einsum("bqj,tjk->tbqk", base_gamma, transforms)
    gammadash = jnp.einsum("bqj,tjk->tbqk", base_gammadash, transforms)
    currents = current_signs[:, None] * base_currents[None, :]
    return (
        gamma.reshape((ntransforms * nbase, nquad, 3)),
        gammadash.reshape((ntransforms * nbase, nquad, 3)),
        currents.reshape((ntransforms * nbase,)),
    )


def curve_lengths(gammadash):
    """Return trapezoidal lengths for uniformly sampled closed curves."""
    gammadash = jnp.asarray(gammadash)
    if gammadash.ndim < 2 or gammadash.shape[-1] != 3:
        raise ValueError("gammadash must have shape (..., nquad, 3)")
    return jnp.mean(jnp.linalg.norm(gammadash, axis=-1), axis=-1)
