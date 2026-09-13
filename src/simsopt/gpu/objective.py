"""Composed device-resident objectives for coil optimization."""

import jax.numpy as jnp

from .biot_savart import biot_savart_field, biot_savart_field_custom_vjp
from .curves import (
    curve_lengths,
    dofs_to_coefficients,
    evaluate_cartesian_fourier_derivatives,
    expand_by_symmetry,
)
from .distances import (
    coil_coil_distance,
    coil_coil_distance_residuals,
    coil_surface_distance,
    coil_surface_distance_residuals,
)
from .flux import normalized_flux, quadratic_flux
from .regularizers import (
    arclength_variation,
    curve_curvature_residuals,
    lp_curve_curvature_penalty,
    mean_squared_curvature,
    mean_squared_curvature_residuals,
)

LOCAL_RESIDUAL_FAMILIES = (
    "coil_coil_distance",
    "coil_surface_distance",
    "curvature",
    "mean_squared_curvature",
)


def local_engineering_residual_layout(
    base_curve_count: int,
    physical_curve_count: int,
    quadrature_count: int,
    pair_count: int,
):
    """Describe the fixed family slices in the local residual vector."""
    counts = (
        pair_count * quadrature_count,
        physical_curve_count * quadrature_count,
        base_curve_count * quadrature_count,
        base_curve_count,
    )
    if min(base_curve_count, physical_curve_count, quadrature_count, pair_count) < 0:
        raise ValueError("residual layout dimensions must be nonnegative")
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return {
        name: {"start": int(offsets[index]), "stop": int(offsets[index + 1])}
        for index, name in enumerate(LOCAL_RESIDUAL_FAMILIES)
    }


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


def minimal_coil_augmented_lagrangian_terms(
    curve_dofs,
    base_currents,
    bases,
    transforms,
    current_signs,
    surface_points,
    surface_normal,
    target_normal_field,
    *,
    length_weight: float = 1e-6,
    flux_definition: str = "quadratic flux",
    target_tile_size: int = 128,
    source_tile_size: int = 256,
    vjp_mode: str = "custom",
    curvature_threshold: float = 5.0,
    mean_squared_curvature_threshold: float = 5.0,
    coil_coil_pair_indices=None,
    coil_coil_distance_threshold: float = 0.1,
    coil_surface_distance_threshold: float = 0.3,
):
    """Return the base objective and zero-target engineering constraints.

    The constraint vector contains the same nonnegative hinge-penalty
    quantities used by SIMSOPT's engineering objective, in this fixed order:
    coil--coil distance, coil--surface distance, pointwise curvature, and
    mean-squared curvature.  Each entry is exactly zero when its corresponding
    inequality is satisfied.  This is the equality-to-zero construction used
    by the augmented-Lagrangian reference workflow.
    """
    if coil_coil_pair_indices is None:
        raise ValueError("coil_coil_pair_indices is required")

    coefficients = dofs_to_coefficients(curve_dofs)
    derivatives = evaluate_cartesian_fourier_derivatives(coefficients, bases[:3])
    base_gamma, base_gammadash, base_gammadashdash = derivatives
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

    base_objective = flux + length_weight * jnp.sum(curve_lengths(base_gammadash))
    mean_curvature = mean_squared_curvature(base_gammadash, base_gammadashdash)
    mean_curvature_excess = jnp.maximum(
        mean_curvature - mean_squared_curvature_threshold, 0.0
    )
    constraints = jnp.stack(
        (
            coil_coil_distance(
                gamma,
                gammadash,
                coil_coil_pair_indices,
                coil_coil_distance_threshold,
            ),
            coil_surface_distance(
                gamma,
                gammadash,
                surface_points,
                surface_normal,
                coil_surface_distance_threshold,
                target_tile_size=target_tile_size,
            ),
            jnp.sum(
                lp_curve_curvature_penalty(
                    base_gammadash,
                    base_gammadashdash,
                    p=2.0,
                    threshold=curvature_threshold,
                )
            ),
            0.5 * jnp.sum(mean_curvature_excess * mean_curvature_excess),
        )
    )
    return base_objective, constraints


def minimal_coil_local_residual_terms(
    curve_dofs,
    base_currents,
    bases,
    transforms,
    current_signs,
    surface_points,
    surface_normal,
    target_normal_field,
    *,
    length_weight: float = 1e-6,
    flux_definition: str = "quadratic flux",
    target_tile_size: int = 128,
    source_tile_size: int = 256,
    vjp_mode: str = "custom",
    curvature_threshold: float = 5.0,
    curvature_feasibility_tolerance: float = 1e-3,
    mean_squared_curvature_threshold: float = 5.0,
    mean_squared_curvature_feasibility_tolerance: float = 1e-3,
    coil_coil_pair_indices=None,
    coil_coil_distance_threshold: float = 0.1,
    coil_surface_distance_threshold: float = 0.3,
    distance_feasibility_tolerance: float = 1e-4,
):
    """Return the base objective and local, dimensionless hinge residuals.

    Unlike :func:`minimal_coil_augmented_lagrangian_terms`, this function does
    not aggregate and square engineering violations before the solver sees
    them.  Each residual is normalized by its physical feasibility allowance,
    and its zero set matches the benchmark's direct feasibility gate.  The
    fixed family order is given by :data:`LOCAL_RESIDUAL_FAMILIES`.
    """
    if coil_coil_pair_indices is None:
        raise ValueError("coil_coil_pair_indices is required")
    if distance_feasibility_tolerance <= 0:
        raise ValueError("distance_feasibility_tolerance must be positive")
    if curvature_feasibility_tolerance <= 0:
        raise ValueError("curvature_feasibility_tolerance must be positive")
    if mean_squared_curvature_feasibility_tolerance <= 0:
        raise ValueError(
            "mean_squared_curvature_feasibility_tolerance must be positive"
        )
    if distance_feasibility_tolerance >= min(
        coil_coil_distance_threshold, coil_surface_distance_threshold
    ):
        raise ValueError("distance feasibility tolerance must be below thresholds")

    coefficients = dofs_to_coefficients(curve_dofs)
    derivatives = evaluate_cartesian_fourier_derivatives(coefficients, bases[:3])
    base_gamma, base_gammadash, base_gammadashdash = derivatives
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
    base_objective = flux + length_weight * jnp.sum(curve_lengths(base_gammadash))

    distance_scale = distance_feasibility_tolerance
    coil_coil = coil_coil_distance_residuals(
        gamma,
        coil_coil_pair_indices,
        coil_coil_distance_threshold - distance_feasibility_tolerance,
        distance_scale,
    )
    coil_surface = coil_surface_distance_residuals(
        gamma,
        surface_points,
        coil_surface_distance_threshold - distance_feasibility_tolerance,
        distance_scale,
        target_tile_size=target_tile_size,
    )
    curvature = curve_curvature_residuals(
        base_gammadash,
        base_gammadashdash,
        curvature_threshold + curvature_feasibility_tolerance,
        curvature_feasibility_tolerance,
    )
    mean_curvature = mean_squared_curvature_residuals(
        base_gammadash,
        base_gammadashdash,
        mean_squared_curvature_threshold + mean_squared_curvature_feasibility_tolerance,
        mean_squared_curvature_feasibility_tolerance,
    )
    residuals = jnp.concatenate(
        (
            coil_coil.reshape((-1,)),
            coil_surface.reshape((-1,)),
            curvature.reshape((-1,)),
            mean_curvature.reshape((-1,)),
        )
    )
    return base_objective, residuals
