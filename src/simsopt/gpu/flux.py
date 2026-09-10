"""Flux objective terms implemented as device-resident array functions."""

import jax.numpy as jnp


def _flux_inputs(field, normal, target):
    field = jnp.asarray(field)
    normal = jnp.asarray(normal)
    target = jnp.asarray(target)
    if field.ndim != 2 or field.shape[-1] != 3:
        raise ValueError("field must have shape (npoint, 3)")
    if normal.shape != field.shape:
        raise ValueError("normal must match field")
    if target.ndim > 1 or (target.ndim == 1 and target.shape != field.shape[:1]):
        raise ValueError("target must be scalar or have shape (npoint,)")

    area_element = jnp.linalg.norm(normal, axis=-1)
    unit_normal = normal / area_element[:, None]
    residual = jnp.sum(field * unit_normal, axis=-1) - target
    return field, area_element, residual


def quadratic_flux(field, normal, target=0.0):
    r"""Return SIMSOPT's discretized quadratic-flux objective.

    normal is the unnormalized surface normal. Its norm supplies the
    quadrature area factor, matching SquaredFlux on a tensor-product grid.
    """
    _, area_element, residual = _flux_inputs(field, normal, target)
    return 0.5 * jnp.mean(residual * residual * area_element)


def normalized_flux(field, normal, target=0.0):
    """Return SIMSOPT's globally normalized quadratic-flux objective."""
    field, area_element, residual = _flux_inputs(field, normal, target)
    numerator = jnp.mean(residual * residual * area_element)
    denominator = jnp.mean(jnp.sum(field * field, axis=-1) * area_element)
    return 0.5 * numerator / denominator
