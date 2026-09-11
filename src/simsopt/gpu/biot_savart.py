"""Tiled, differentiable Biot--Savart field evaluation."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

MU0_OVER_4PI = 1.0e-7


def _as_common_dtype(points, gamma, gammadash, currents):
    dtype = jnp.result_type(points, gamma, gammadash, currents)
    return (
        jnp.asarray(points, dtype=dtype),
        jnp.asarray(gamma, dtype=dtype),
        jnp.asarray(gammadash, dtype=dtype),
        jnp.asarray(currents, dtype=dtype),
    )


def _validate_inputs(points, gamma, gammadash, currents):
    if points.ndim != 2 or points.shape[-1] != 3 or points.shape[0] == 0:
        raise ValueError("points must have non-empty shape (ntarget, 3)")
    if gamma.ndim != 3 or gamma.shape[-1] != 3:
        raise ValueError("gamma must have shape (ncoil, nquad, 3)")
    if gamma.shape[0] == 0 or gamma.shape[1] == 0:
        raise ValueError("gamma must contain at least one coil and quadrature point")
    if gammadash.shape != gamma.shape:
        raise ValueError("gammadash must match gamma")
    if currents.shape != (gamma.shape[0],):
        raise ValueError("currents must have shape (ncoil,)")


def biot_savart_field_reference(points, gamma, gammadash, currents):
    """Untiled correctness reference; its interaction tensor may be large."""
    points, gamma, gammadash, currents = _as_common_dtype(
        points, gamma, gammadash, currents
    )
    _validate_inputs(points, gamma, gammadash, currents)

    nquad = gamma.shape[1]
    difference = points[:, None, None, :] - gamma[None, :, :, :]
    inverse_distance_cubed = jnp.sum(difference * difference, axis=-1) ** -1.5
    integrand = jnp.cross(gammadash[None, :, :, :], difference)
    weighted = currents[None, :, None] * inverse_distance_cubed
    return MU0_OVER_4PI * jnp.sum(integrand * weighted[..., None], axis=(1, 2)) / nquad


@partial(jax.jit, static_argnames=("target_tile_size", "source_tile_size"))
def _biot_savart_field_tiled(
    points,
    gamma,
    gammadash,
    currents,
    *,
    target_tile_size: int,
    source_tile_size: int,
):
    ntarget = points.shape[0]
    ncoil, nquad, _ = gamma.shape
    nsource = ncoil * nquad

    target_padding = (-ntarget) % target_tile_size
    source_padding = (-nsource) % source_tile_size

    padded_points = jnp.pad(points, ((0, target_padding), (0, 0)), mode="edge")
    source_gamma = gamma.reshape((nsource, 3))
    source_gammadash = gammadash.reshape((nsource, 3))
    source_weights = jnp.repeat(currents / nquad, nquad)

    padded_gamma = jnp.pad(source_gamma, ((0, source_padding), (0, 0)))
    padded_gammadash = jnp.pad(source_gammadash, ((0, source_padding), (0, 0)))
    padded_weights = jnp.pad(source_weights, ((0, source_padding),))
    source_is_active = jnp.arange(nsource + source_padding) < nsource

    ntarget_blocks = padded_points.shape[0] // target_tile_size
    nsource_blocks = padded_gamma.shape[0] // source_tile_size
    field = jnp.zeros_like(padded_points)

    def target_body(target_block_index, full_field):
        target_start = target_block_index * target_tile_size
        target_block = lax.dynamic_slice(
            padded_points, (target_start, 0), (target_tile_size, 3)
        )

        def source_body(source_block_index, block_field):
            source_start = source_block_index * source_tile_size
            gamma_block = lax.dynamic_slice(
                padded_gamma, (source_start, 0), (source_tile_size, 3)
            )
            gammadash_block = lax.dynamic_slice(
                padded_gammadash, (source_start, 0), (source_tile_size, 3)
            )
            weight_block = lax.dynamic_slice(
                padded_weights, (source_start,), (source_tile_size,)
            )
            active_block = lax.dynamic_slice(
                source_is_active, (source_start,), (source_tile_size,)
            )

            difference = target_block[:, None, :] - gamma_block[None, :, :]
            distance_squared = jnp.sum(difference * difference, axis=-1)
            safe_distance_squared = jnp.where(
                active_block[None, :], distance_squared, 1.0
            )
            inverse_distance_cubed = safe_distance_squared**-1.5
            weighted_kernel = jnp.where(
                active_block[None, :],
                weight_block[None, :] * inverse_distance_cubed,
                0.0,
            )
            integrand = jnp.cross(gammadash_block[None, :, :], difference)
            return block_field + jnp.sum(integrand * weighted_kernel[..., None], axis=1)

        block_field = lax.fori_loop(
            0,
            nsource_blocks,
            source_body,
            jnp.zeros((target_tile_size, 3), dtype=points.dtype),
        )
        return lax.dynamic_update_slice(
            full_field, MU0_OVER_4PI * block_field, (target_start, 0)
        )

    field = lax.fori_loop(0, ntarget_blocks, target_body, field)
    return field[:ntarget]


@partial(jax.jit, static_argnames=("target_tile_size", "source_tile_size"))
def _biot_savart_field_tiled_vjp(
    points,
    gamma,
    gammadash,
    currents,
    field_cotangent,
    *,
    target_tile_size: int,
    source_tile_size: int,
):
    """Analytically contract a field cotangent without retaining a primal tape."""
    ntarget = points.shape[0]
    ncoil, nquad, _ = gamma.shape
    nsource = ncoil * nquad

    target_padding = (-ntarget) % target_tile_size
    source_padding = (-nsource) % source_tile_size
    padded_points = jnp.pad(points, ((0, target_padding), (0, 0)), mode="edge")
    padded_cotangent = jnp.pad(
        field_cotangent, ((0, target_padding), (0, 0))
    )
    source_gamma = gamma.reshape((nsource, 3))
    source_gammadash = gammadash.reshape((nsource, 3))
    source_weights = jnp.repeat(currents / nquad, nquad)
    padded_gamma = jnp.pad(source_gamma, ((0, source_padding), (0, 0)))
    padded_gammadash = jnp.pad(source_gammadash, ((0, source_padding), (0, 0)))
    padded_weights = jnp.pad(source_weights, ((0, source_padding),))
    source_is_active = jnp.arange(nsource + source_padding) < nsource

    ntarget_blocks = padded_points.shape[0] // target_tile_size
    nsource_blocks = padded_gamma.shape[0] // source_tile_size
    gradient_gamma = jnp.zeros_like(padded_gamma)
    gradient_gammadash = jnp.zeros_like(padded_gammadash)
    gradient_weights = jnp.zeros_like(padded_weights)
    gradient_points = jnp.zeros_like(padded_points)

    def source_body(source_block_index, full_gradients):
        full_gamma, full_gammadash, full_weights, full_points = full_gradients
        source_start = source_block_index * source_tile_size
        gamma_block = lax.dynamic_slice(
            padded_gamma, (source_start, 0), (source_tile_size, 3)
        )
        gammadash_block = lax.dynamic_slice(
            padded_gammadash, (source_start, 0), (source_tile_size, 3)
        )
        weight_block = lax.dynamic_slice(
            padded_weights, (source_start,), (source_tile_size,)
        )
        active_block = lax.dynamic_slice(
            source_is_active, (source_start,), (source_tile_size,)
        )

        def target_body(target_block_index, block_gradients):
            block_gamma, block_gammadash, block_weights, accumulated_points = (
                block_gradients
            )
            target_start = target_block_index * target_tile_size
            target_block = lax.dynamic_slice(
                padded_points, (target_start, 0), (target_tile_size, 3)
            )
            cotangent_block = lax.dynamic_slice(
                padded_cotangent, (target_start, 0), (target_tile_size, 3)
            )

            difference = target_block[:, None, :] - gamma_block[None, :, :]
            distance_squared = jnp.sum(difference * difference, axis=-1)
            safe_distance_squared = jnp.where(
                active_block[None, :], distance_squared, 1.0
            )
            inverse_distance_cubed = safe_distance_squared**-1.5
            inverse_distance_fifth = inverse_distance_cubed / safe_distance_squared
            cross_tangent_difference = jnp.cross(
                gammadash_block[None, :, :], difference
            )
            contracted_field = jnp.sum(
                cotangent_block[:, None, :] * cross_tangent_difference, axis=-1
            )
            weighted_scale = (
                MU0_OVER_4PI
                * weight_block[None, :]
                * active_block[None, :]
            )
            gradient_gamma_interaction = weighted_scale[..., None] * (
                jnp.cross(
                    gammadash_block[None, :, :], cotangent_block[:, None, :]
                )
                * inverse_distance_cubed[..., None]
                + 3.0
                * difference
                * contracted_field[..., None]
                * inverse_distance_fifth[..., None]
            )
            gradient_gammadash_interaction = weighted_scale[..., None] * (
                jnp.cross(difference, cotangent_block[:, None, :])
                * inverse_distance_cubed[..., None]
            )
            gradient_weight_interaction = (
                MU0_OVER_4PI
                * contracted_field
                * inverse_distance_cubed
                * active_block[None, :]
            )

            block_gamma = block_gamma + jnp.sum(
                gradient_gamma_interaction, axis=0
            )
            block_gammadash = block_gammadash + jnp.sum(
                gradient_gammadash_interaction, axis=0
            )
            block_weights = block_weights + jnp.sum(
                gradient_weight_interaction, axis=0
            )
            point_gradient = -jnp.sum(gradient_gamma_interaction, axis=1)
            previous_points = lax.dynamic_slice(
                accumulated_points,
                (target_start, 0),
                (target_tile_size, 3),
            )
            accumulated_points = lax.dynamic_update_slice(
                accumulated_points,
                previous_points + point_gradient,
                (target_start, 0),
            )
            return (
                block_gamma,
                block_gammadash,
                block_weights,
                accumulated_points,
            )

        block_gamma, block_gammadash, block_weights, full_points = lax.fori_loop(
            0,
            ntarget_blocks,
            target_body,
            (
                jnp.zeros_like(gamma_block),
                jnp.zeros_like(gammadash_block),
                jnp.zeros_like(weight_block),
                full_points,
            ),
        )
        full_gamma = lax.dynamic_update_slice(
            full_gamma, block_gamma, (source_start, 0)
        )
        full_gammadash = lax.dynamic_update_slice(
            full_gammadash, block_gammadash, (source_start, 0)
        )
        full_weights = lax.dynamic_update_slice(
            full_weights, block_weights, (source_start,)
        )
        return full_gamma, full_gammadash, full_weights, full_points

    gradient_gamma, gradient_gammadash, gradient_weights, gradient_points = (
        lax.fori_loop(
            0,
            nsource_blocks,
            source_body,
            (
                gradient_gamma,
                gradient_gammadash,
                gradient_weights,
                gradient_points,
            ),
        )
    )
    gradient_currents = jnp.sum(
        gradient_weights[:nsource].reshape((ncoil, nquad)), axis=1
    ) / nquad
    return (
        gradient_points[:ntarget],
        gradient_gamma[:nsource].reshape(gamma.shape),
        gradient_gammadash[:nsource].reshape(gammadash.shape),
        gradient_currents,
    )


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _biot_savart_field_with_custom_vjp(
    points,
    gamma,
    gammadash,
    currents,
    target_tile_size,
    source_tile_size,
):
    return _biot_savart_field_tiled(
        points,
        gamma,
        gammadash,
        currents,
        target_tile_size=target_tile_size,
        source_tile_size=source_tile_size,
    )


def _biot_savart_field_custom_fwd(
    points,
    gamma,
    gammadash,
    currents,
    target_tile_size,
    source_tile_size,
):
    field = _biot_savart_field_tiled(
        points,
        gamma,
        gammadash,
        currents,
        target_tile_size=target_tile_size,
        source_tile_size=source_tile_size,
    )
    return field, (points, gamma, gammadash, currents)


def _biot_savart_field_custom_bwd(
    target_tile_size,
    source_tile_size,
    residuals,
    field_cotangent,
):
    points, gamma, gammadash, currents = residuals
    return _biot_savart_field_tiled_vjp(
        points,
        gamma,
        gammadash,
        currents,
        field_cotangent,
        target_tile_size=target_tile_size,
        source_tile_size=source_tile_size,
    )


_biot_savart_field_with_custom_vjp.defvjp(
    _biot_savart_field_custom_fwd,
    _biot_savart_field_custom_bwd,
)


def biot_savart_field(
    points,
    gamma,
    gammadash,
    currents,
    *,
    target_tile_size: int = 128,
    source_tile_size: int = 256,
):
    """Evaluate the filamentary field without a full interaction tensor."""
    if target_tile_size <= 0 or source_tile_size <= 0:
        raise ValueError("tile sizes must be positive")
    points, gamma, gammadash, currents = _as_common_dtype(
        points, gamma, gammadash, currents
    )
    _validate_inputs(points, gamma, gammadash, currents)
    return _biot_savart_field_tiled(
        points,
        gamma,
        gammadash,
        currents,
        target_tile_size=target_tile_size,
        source_tile_size=source_tile_size,
    )


def biot_savart_field_custom_vjp(
    points,
    gamma,
    gammadash,
    currents,
    *,
    target_tile_size: int = 128,
    source_tile_size: int = 256,
):
    """Evaluate the tiled field with a memory-bounded analytic reverse pass.

    This variant is intended for reverse-mode objective gradients. Use
    :func:`biot_savart_field` when a forward-mode JVP is required.
    """
    if target_tile_size <= 0 or source_tile_size <= 0:
        raise ValueError("tile sizes must be positive")
    points, gamma, gammadash, currents = _as_common_dtype(
        points, gamma, gammadash, currents
    )
    _validate_inputs(points, gamma, gammadash, currents)
    return _biot_savart_field_with_custom_vjp(
        points,
        gamma,
        gammadash,
        currents,
        target_tile_size,
        source_tile_size,
    )
