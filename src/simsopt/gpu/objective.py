"""Composed device-resident objectives for coil optimization."""

import jax.numpy as jnp

from .biot_savart import biot_savart_field, biot_savart_field_custom_vjp
from .curves import (
    curve_lengths,
    dofs_to_coefficients,
    evaluate_cartesian_fourier_derivatives,
    expand_by_symmetry,
)
from .distances import coil_coil_distance, coil_surface_distance
from .flux import normalized_flux, quadratic_flux
from .regularizers import (
    arclength_variation,
    lp_curve_curvature_penalty,
    mean_squared_curvature,
)


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
    curvature_threshold: float = 5.0,
    curvature_weight: float = 0.0,
    mean_squared_curvature_threshold: float = 5.0,
    mean_squared_curvature_weight: float = 0.0,
    arclength_variation_weight: float = 0.0,
    coil_coil_pair_indices=None,
    coil_coil_distance_threshold: float = 0.1,
    coil_coil_distance_weight: float = 0.0,
    coil_surface_distance_threshold: float = 0.3,
    coil_surface_distance_weight: float = 0.0,
):
    """Minimal stage-two objective implemented as one differentiable program.

    Both curve_dofs and base_currents are differentiable inputs. Fixed or
    shared-variable semantics belong in the outer parameter adapter, rather
    than as host-side state in this numerical function.
    """
    coefficients = dofs_to_coefficients(curve_dofs)
    needs_second_derivative = curvature_weight or mean_squared_curvature_weight
    basis_count = 3 if needs_second_derivative else 2
    derivatives = evaluate_cartesian_fourier_derivatives(
        coefficients, bases[:basis_count]
    )
    base_gamma, base_gammadash = derivatives[0], derivatives[1]
    base_gammadashdash = derivatives[2] if needs_second_derivative else None
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
    if length_target is None:
        objective = flux + length_weight * total_base_length
    else:
        length_excess = jnp.maximum(total_base_length - length_target, 0.0)
        objective = flux + 0.5 * length_weight * length_excess * length_excess
    if curvature_weight:
        objective = objective + curvature_weight * jnp.sum(
            lp_curve_curvature_penalty(
                base_gammadash,
                base_gammadashdash,
                p=2.0,
                threshold=curvature_threshold,
            )
        )
    if mean_squared_curvature_weight:
        msc = mean_squared_curvature(base_gammadash, base_gammadashdash)
        msc_excess = jnp.maximum(msc - mean_squared_curvature_threshold, 0.0)
        objective = objective + 0.5 * mean_squared_curvature_weight * jnp.sum(
            msc_excess * msc_excess
        )
    if arclength_variation_weight:
        objective = objective + arclength_variation_weight * jnp.sum(
            arclength_variation(base_gammadash)
        )
    if coil_coil_distance_weight:
        if coil_coil_pair_indices is None:
            raise ValueError(
                "coil_coil_pair_indices is required when its weight is nonzero"
            )
        objective = objective + coil_coil_distance_weight * coil_coil_distance(
            gamma,
            gammadash,
            coil_coil_pair_indices,
            coil_coil_distance_threshold,
        )
    if coil_surface_distance_weight:
        objective = objective + coil_surface_distance_weight * coil_surface_distance(
            gamma,
            gammadash,
            surface_points,
            surface_normal,
            coil_surface_distance_threshold,
            target_tile_size=target_tile_size,
        )
    return objective
