"""Composed device-resident objectives for coil optimization."""

import jax.numpy as jnp

from .biot_savart import biot_savart_field, biot_savart_field_custom_vjp
from .curves import (
    curve_lengths,
    dofs_to_coefficients,
    evaluate_cartesian_fourier_derivatives,
    expand_by_symmetry,
)
from .flux import normalized_flux, quadratic_flux


def minimal_coil_objective(
    curve_dofs,
    base_currents,
    bases,
    transforms,
    current_signs,
    surface_points,
    surface_normal,
    target_normal_field,
    *,
    length_target,
    length_weight,
    flux_definition: str = "quadratic flux",
    target_tile_size: int = 128,
    source_tile_size: int = 256,
    vjp_mode: str = "custom",
):
    """Minimal stage-two objective implemented as one differentiable program.

    Both curve_dofs and base_currents are differentiable inputs. Fixed or
    shared-variable semantics belong in the outer parameter adapter, rather
    than as host-side state in this numerical function.
    """
    coefficients = dofs_to_coefficients(curve_dofs)
    derivatives = evaluate_cartesian_fourier_derivatives(coefficients, bases[:2])
    base_gamma, base_gammadash = derivatives[0], derivatives[1]
    gamma, gammadash, currents = expand_by_symmetry(
        base_gamma, base_gammadash, base_currents, transforms, current_signs
    )
    if vjp_mode == "autodiff":
        field_function = biot_savart_field
    elif vjp_mode == "custom":
        field_function = biot_savart_field_custom_vjp
    else:
        raise ValueError("vjp_mode must be 'autodiff' or 'custom'")
    field = field_function(
        surface_points,
        gamma,
        gammadash,
        currents,
        target_tile_size=target_tile_size,
        source_tile_size=source_tile_size,
    )
    if flux_definition == "quadratic flux":
        flux = quadratic_flux(field, surface_normal, target_normal_field)
    elif flux_definition == "normalized":
        flux = normalized_flux(field, surface_normal, target_normal_field)
    else:
        raise ValueError("flux_definition must be 'quadratic flux' or 'normalized'")
    total_base_length = jnp.sum(curve_lengths(base_gammadash))
    length_excess = jnp.maximum(total_base_length - length_target, 0.0)
    return flux + 0.5 * length_weight * length_excess * length_excess
