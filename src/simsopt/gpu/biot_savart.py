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
