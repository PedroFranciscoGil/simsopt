"""Curve-local regularizers for device-resident coil objectives."""

import jax.numpy as jnp


def curve_curvatures(gammadash, gammadashdash):
    """Return curvature at uniformly sampled points on one or more curves."""
    gammadash = jnp.asarray(gammadash)
    gammadashdash = jnp.asarray(gammadashdash)
    if gammadash.ndim < 2 or gammadash.shape[-1] != 3:
        raise ValueError("gammadash must have shape (..., nquad, 3)")
    if gammadashdash.shape != gammadash.shape:
        raise ValueError("gammadashdash must match gammadash")
    cross = jnp.cross(gammadash, gammadashdash)
    speed = jnp.linalg.norm(gammadash, axis=-1)
    return jnp.linalg.norm(cross, axis=-1) / speed**3


def lp_curve_curvature_penalty(
    gammadash,
    gammadashdash,
    *,
    p: float,
    threshold: float = 0.0,
):
    r"""Return ``1/p int max(kappa-threshold, 0)^p dl`` per curve."""
    if p <= 0:
        raise ValueError("p must be positive")
    speed = jnp.linalg.norm(gammadash, axis=-1)
    curvature = curve_curvatures(gammadash, gammadashdash)
    excess = jnp.maximum(curvature - threshold, 0.0)
    return jnp.mean(excess**p * speed, axis=-1) / p


def mean_squared_curvature(gammadash, gammadashdash):
    r"""Return ``(1/L) int kappa^2 dl`` for each curve."""
    speed = jnp.linalg.norm(gammadash, axis=-1)
    curvature = curve_curvatures(gammadash, gammadashdash)
    return jnp.mean(curvature**2 * speed, axis=-1) / jnp.mean(speed, axis=-1)


def arclength_variation(gammadash):
    """Return full-resolution incremental-arclength variance per curve."""
    gammadash = jnp.asarray(gammadash)
    if gammadash.ndim < 2 or gammadash.shape[-1] != 3:
        raise ValueError("gammadash must have shape (..., nquad, 3)")
    speed = jnp.linalg.norm(gammadash, axis=-1)
    return jnp.var(speed, axis=-1)
