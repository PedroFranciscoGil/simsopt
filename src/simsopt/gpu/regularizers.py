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


def curve_curvature_residuals(
    gammadash,
    gammadashdash,
    allowed_maximum_curvature: float,
    curvature_scale: float,
):
    """Return dimensionless pointwise curvature-excess residuals."""
    if curvature_scale <= 0:
        raise ValueError("curvature_scale must be positive")
    curvature = curve_curvatures(gammadash, gammadashdash)
    return jnp.maximum(curvature - allowed_maximum_curvature, 0.0) / curvature_scale


def mean_squared_curvature_residuals(
    gammadash,
    gammadashdash,
    allowed_maximum_mean_squared_curvature: float,
    mean_squared_curvature_scale: float,
):
    """Return one dimensionless mean-squared-curvature residual per curve."""
    if mean_squared_curvature_scale <= 0:
        raise ValueError("mean_squared_curvature_scale must be positive")
    values = mean_squared_curvature(gammadash, gammadashdash)
    return (
        jnp.maximum(values - allowed_maximum_mean_squared_curvature, 0.0)
        / mean_squared_curvature_scale
    )


def arclength_variation(gammadash):
    """Return full-resolution incremental-arclength variance per curve."""
    gammadash = jnp.asarray(gammadash)
    if gammadash.ndim < 2 or gammadash.shape[-1] != 3:
        raise ValueError("gammadash must have shape (..., nquad, 3)")
    speed = jnp.linalg.norm(gammadash, axis=-1)
    return jnp.var(speed, axis=-1)
