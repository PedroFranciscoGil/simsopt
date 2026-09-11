"""Device-resident engineering distance penalties for coil optimization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


def curve_pair_indices(ncurves: int, num_basecurves: int | None = None):
    """Return the static curve pairs used by SIMSOPT's symmetry reduction.

    ``CurveCurveDistance`` considers pairs ``(i, j)`` for which ``j < i``.
    When ``num_basecurves`` is supplied, only pairs whose lower index belongs
    to that first block are needed; symmetry makes the remaining pairs
    duplicates.
    """
    if ncurves < 0:
        raise ValueError("ncurves must be non-negative")
    if num_basecurves is None:
        num_basecurves = ncurves
    if not 0 <= num_basecurves <= ncurves:
        raise ValueError("num_basecurves must lie between zero and ncurves")
    return np.asarray(
        [(i, j) for i in range(ncurves) for j in range(i) if j < num_basecurves],
        dtype=np.int32,
    ).reshape((-1, 2))


def coil_coil_distance(
    gamma,
    gammadash,
    pair_indices,
    minimum_distance: float,
):
    r"""Evaluate the all-pairs coil--coil penalty used by SIMSOPT.

    The pair list is static and complete. SIMSOPT's CPU spatial filter merely
    removes pairs whose hinge penalty is exactly zero, so this formulation has
    identical objective semantics without data-dependent host work.
    """
    gamma = jnp.asarray(gamma)
    gammadash = jnp.asarray(gammadash)
    pairs = jnp.asarray(pair_indices, dtype=jnp.int32)
    if gamma.ndim != 3 or gamma.shape[-1] != 3:
        raise ValueError("gamma must have shape (ncurves, nquad, 3)")
    if gammadash.shape != gamma.shape:
        raise ValueError("gammadash must match gamma")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pair_indices must have shape (npairs, 2)")

    nquad = gamma.shape[1]

    def add_pair(pair_number, total):
        first = pairs[pair_number, 0]
        second = pairs[pair_number, 1]
        difference = gamma[first, :, None, :] - gamma[second, None, :, :]
        distances = jnp.sqrt(jnp.sum(difference * difference, axis=-1))
        weights = (
            jnp.linalg.norm(gammadash[first], axis=-1)[:, None]
            * jnp.linalg.norm(gammadash[second], axis=-1)[None, :]
        )
        hinge = jnp.maximum(minimum_distance - distances, 0.0)
        return total + jnp.sum(weights * hinge * hinge)

    total = lax.fori_loop(0, pairs.shape[0], add_pair, jnp.zeros((), gamma.dtype))
    return total / (nquad * nquad)


def _surface_tile_contribution(
    gamma,
    curve_speeds,
    surface_points,
    surface_weights,
    active,
    minimum_distance,
):
    difference = gamma[:, :, None, :] - surface_points[None, None, :, :]
    squared_distance = jnp.sum(difference * difference, axis=-1)
    # Padded targets are inactive. Giving them a finite, nonzero distance also
    # keeps the derivative of sqrt well-defined if a curve passes the origin.
    squared_distance = jnp.where(active[None, None, :], squared_distance, 1.0)
    distances = jnp.sqrt(squared_distance)
    hinge = jnp.maximum(minimum_distance - distances, 0.0)
    weights = curve_speeds[:, :, None] * surface_weights[None, None, :]
    return jnp.sum(weights * hinge * hinge)


def coil_surface_distance(
    gamma,
    gammadash,
    surface_points,
    surface_normal,
    minimum_distance: float,
    *,
    target_tile_size: int = 512,
):
    r"""Evaluate SIMSOPT's coil--surface distance penalty in target tiles."""
    gamma = jnp.asarray(gamma)
    gammadash = jnp.asarray(gammadash)
    surface_points = jnp.asarray(surface_points)
    surface_normal = jnp.asarray(surface_normal)
    if gamma.ndim != 3 or gamma.shape[-1] != 3:
        raise ValueError("gamma must have shape (ncurves, nquad, 3)")
    if gammadash.shape != gamma.shape:
        raise ValueError("gammadash must match gamma")
    if surface_points.ndim != 2 or surface_points.shape[1] != 3:
        raise ValueError("surface_points must have shape (ntargets, 3)")
    if surface_normal.shape != surface_points.shape:
        raise ValueError("surface_normal must match surface_points")
    if target_tile_size <= 0:
        raise ValueError("target_tile_size must be positive")

    ntargets = surface_points.shape[0]
    padding = (-ntargets) % target_tile_size
    padded_points = jnp.pad(surface_points, ((0, padding), (0, 0)))
    padded_weights = jnp.pad(jnp.linalg.norm(surface_normal, axis=-1), ((0, padding),))
    active = jnp.arange(padded_points.shape[0]) < ntargets
    curve_speeds = jnp.linalg.norm(gammadash, axis=-1)
    tile_contribution = jax.checkpoint(_surface_tile_contribution)

    def add_tile(tile_number, total):
        start = tile_number * target_tile_size
        points = lax.dynamic_slice_in_dim(
            padded_points, start, target_tile_size, axis=0
        )
        weights = lax.dynamic_slice_in_dim(
            padded_weights, start, target_tile_size, axis=0
        )
        tile_active = lax.dynamic_slice_in_dim(active, start, target_tile_size, axis=0)
        return total + tile_contribution(
            gamma,
            curve_speeds,
            points,
            weights,
            tile_active,
            minimum_distance,
        )

    ntiles = padded_points.shape[0] // target_tile_size
    total = lax.fori_loop(0, ntiles, add_tile, jnp.zeros((), gamma.dtype))
    return total / (gamma.shape[1] * ntargets)
