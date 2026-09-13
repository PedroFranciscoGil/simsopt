"""Equality-to-zero augmented-Lagrangian support for GPU objectives.

This module follows the sign convention used by SIMSOPT's experimental
``augmented_lagrangian_method`` workflow::

    L_A(x, lambda, mu) = f(x) - lambda.T @ c(x)
                         + 0.5 * sum(mu * c(x)**2)

Consequently a successful multiplier step is ``lambda -= mu * c``.  The
constraints may be ordinary equalities or nonnegative hinge penalties that
are zero exactly when an engineering inequality is satisfied.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

CONSTRAINT_TRANSFORMS = ("identity", "smooth_sqrt")


def smooth_sqrt_constraints(constraints, epsilon: float):
    """Convert nonnegative squared penalties to smooth residual coordinates.

    ``sqrt(c + epsilon**2) - epsilon`` is zero at ``c = 0`` and approaches
    ``sqrt(c)`` away from the smoothing region.  Applying the AL quadratic to
    this coordinate avoids squaring an already squared hinge penalty twice.
    """
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    constraints = jnp.asarray(constraints)
    return jnp.sqrt(constraints + epsilon * epsilon) - epsilon


def equality_augmented_lagrangian(
    base_objective, constraints, lagrange_multipliers, penalties
):
    """Compose one equality augmented-Lagrangian scalar."""
    constraints = jnp.asarray(constraints)
    lagrange_multipliers = jnp.asarray(lagrange_multipliers)
    penalties = jnp.asarray(penalties)
    if constraints.ndim != 1:
        raise ValueError("constraints must be one-dimensional")
    if lagrange_multipliers.shape != constraints.shape:
        raise ValueError("lagrange_multipliers must match constraints")
    if penalties.shape != constraints.shape:
        raise ValueError("penalties must match constraints")
    return (
        base_objective
        - jnp.vdot(lagrange_multipliers, constraints)
        + 0.5 * jnp.vdot(penalties, constraints * constraints)
    )


class ScipyAugmentedLagrangianBridge:
    """Compile JAX base/constraint terms with dynamic AL state for SciPy.

    ``terms(x)`` must return ``(base_objective, constraint_vector)``.  The
    multipliers and penalties are executable inputs, so every outer update
    reuses one compiled GPU program instead of triggering recompilation.
    Optional positive ``constraint_scales`` and ``constraint_transform`` define
    the AL coordinates; diagnostics continue to return the raw ``c``.
    """

    def __init__(
        self,
        terms,
        initial_x,
        constraint_count: int,
        constraint_scales=None,
        *,
        constraint_transform: str = "identity",
        transform_epsilon: float = 1e-4,
    ):
        initial_x = np.asarray(initial_x)
        if initial_x.ndim != 1:
            raise ValueError("initial_x must be one-dimensional")
        if not np.issubdtype(initial_x.dtype, np.floating):
            raise TypeError("initial_x must have a floating-point dtype")
        if constraint_count < 1:
            raise ValueError("constraint_count must be positive")
        if constraint_transform not in CONSTRAINT_TRANSFORMS:
            raise ValueError("constraint_transform must be 'identity' or 'smooth_sqrt'")
        if not np.isfinite(transform_epsilon) or transform_epsilon <= 0:
            raise ValueError("transform_epsilon must be finite and positive")
        self._terms = terms
        self._shape = initial_x.shape
        self._dtype = initial_x.dtype
        self._constraint_count = int(constraint_count)
        self._constraint_transform = constraint_transform
        self._transform_epsilon = float(transform_epsilon)
        if constraint_scales is None:
            constraint_scales = np.ones(constraint_count, dtype=initial_x.dtype)
        self._constraint_scales = self._coerce_constraint_scales(constraint_scales)
        self._compiled = None
        self._compiled_terms = None
        self._lagrange_multipliers = np.zeros(constraint_count, dtype=initial_x.dtype)
        self._penalties = np.ones(constraint_count, dtype=initial_x.dtype)
        self.evaluations = 0

        def objective(x, lagrange_multipliers, penalties):
            base_objective, constraints = terms(x)
            constraints = self._map_constraints_jax(constraints)
            return equality_augmented_lagrangian(
                base_objective, constraints, lagrange_multipliers, penalties
            )

        self._value_and_grad = jax.jit(jax.value_and_grad(objective, argnums=0))
        self._jitted_terms = jax.jit(terms)

    def _coerce_constraint_scales(self, scales):
        scales = np.asarray(scales, dtype=self._dtype)
        expected = (self._constraint_count,)
        if scales.shape != expected:
            raise ValueError(
                f"constraint_scales must have shape {expected}, got {scales.shape}"
            )
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("constraint_scales must be finite and positive")
        return scales.copy()

    @property
    def constraint_scales(self):
        return self._constraint_scales.copy()

    @property
    def constraint_transform(self):
        return self._constraint_transform

    @property
    def transform_epsilon(self):
        return self._transform_epsilon

    def _map_constraints_jax(self, constraints):
        constraints = jnp.asarray(constraints)
        if self._constraint_transform == "smooth_sqrt":
            constraints = smooth_sqrt_constraints(constraints, self._transform_epsilon)
        return constraints / jnp.asarray(self._constraint_scales)

    def map_constraints(self, constraints):
        """Map raw nonnegative penalties to the coordinates used by the AL."""
        constraints = self._coerce_state(constraints, "constraints")
        if self._constraint_transform == "smooth_sqrt":
            if np.any(constraints < 0):
                raise ValueError("smooth_sqrt constraints must be nonnegative")
            constraints = (
                np.sqrt(constraints + self._transform_epsilon**2)
                - self._transform_epsilon
            )
        return constraints / self._constraint_scales

    def scale_constraints(self, constraints):
        """Backward-compatible alias for :meth:`map_constraints`."""
        return self.map_constraints(constraints)

    @property
    def is_compiled(self):
        return self._compiled is not None and self._compiled_terms is not None

    def _coerce_x(self, x):
        x = np.asarray(x, dtype=self._dtype)
        if x.shape != self._shape:
            raise ValueError(f"x must have shape {self._shape}, got {x.shape}")
        return x

    def _coerce_state(self, value, name):
        value = np.asarray(value, dtype=self._dtype)
        expected = (self._constraint_count,)
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite")
        return value

    def set_state(self, lagrange_multipliers, penalties):
        """Set dynamic multipliers and positive penalty parameters."""
        lagrange_multipliers = self._coerce_state(
            lagrange_multipliers, "lagrange_multipliers"
        )
        penalties = self._coerce_state(penalties, "penalties")
        if np.any(penalties <= 0):
            raise ValueError("penalties must be positive")
        self._lagrange_multipliers = lagrange_multipliers.copy()
        self._penalties = penalties.copy()
        return self

    def compile(self, example_x=None):
        """Compile both the AL value-gradient and diagnostic term programs."""
        if example_x is None:
            example_x = np.zeros(self._shape, dtype=self._dtype)
        x = jnp.asarray(self._coerce_x(example_x))
        lagrange_multipliers = jnp.asarray(self._lagrange_multipliers)
        penalties = jnp.asarray(self._penalties)
        self._compiled = self._value_and_grad.lower(
            x, lagrange_multipliers, penalties
        ).compile()
        value, gradient = self._compiled(x, lagrange_multipliers, penalties)
        value.block_until_ready()
        gradient.block_until_ready()
        self._compiled_terms = self._jitted_terms.lower(x).compile()
        base_objective, constraints = self._compiled_terms(x)
        base_objective.block_until_ready()
        constraints.block_until_ready()
        if constraints.shape != (self._constraint_count,):
            raise ValueError(
                "terms returned constraint shape "
                f"{constraints.shape}, expected {(self._constraint_count,)}"
            )
        return self

    def __call__(self, x):
        """Return the current AL value and gradient on the host."""
        x = self._coerce_x(x)
        if not self.is_compiled:
            self.compile(x)
        value, gradient = self._compiled(
            jnp.asarray(x),
            jnp.asarray(self._lagrange_multipliers),
            jnp.asarray(self._penalties),
        )
        value.block_until_ready()
        gradient.block_until_ready()
        self.evaluations += 1
        return float(value), np.asarray(gradient)

    def evaluate_terms(self, x):
        """Return the unaugmented base objective and constraint vector."""
        x = self._coerce_x(x)
        if not self.is_compiled:
            self.compile(x)
        base_objective, constraints = self._compiled_terms(jnp.asarray(x))
        base_objective.block_until_ready()
        constraints.block_until_ready()
        return float(base_objective), np.asarray(constraints)


@dataclass(frozen=True)
class AugmentedLagrangianResult:
    """Result and complete outer-loop history for one AL solve."""

    x: np.ndarray
    fun: float
    jac: np.ndarray
    lagrange_multipliers: np.ndarray
    penalties: np.ndarray
    constraints: np.ndarray
    scaled_constraints: np.ndarray
    base_objective: float
    success: bool
    message: str
    outer_iterations: int
    total_inner_iterations: int
    total_evaluations: int
    seconds: float
    history: tuple
    terminated_by_inner_safeguard: bool


def minimize_equality_augmented_lagrangian(
    bridge,
    initial_x,
    *,
    mu_init=10.0,
    lagrange_multiplier_init=None,
    tau: float = 10.0,
    max_outer_iterations: int = 8,
    max_inner_iterations: int = 50,
    gradient_tolerance: float = 1e-8,
    constraint_tolerance: float = 1e-8,
    argmin_tolerance: float = 1e-15,
    maxcor: int = 100,
    maxls: int = 20,
    mu_max: float = np.inf,
    require_inner_stationarity: bool = False,
    inner_stationarity_factor: float = 1.0,
    stage_callback=None,
):
    """Minimize a zero-equality augmented Lagrangian with L-BFGS-B.

    The outer update mirrors the attached SIMSOPT AL workflow: penalties grow
    componentwise for constraints outside ``constraint_tolerance``; when the
    scaled infinity norm is below the current ``eta`` threshold, multipliers
    are updated with the module's minus-sign convention.  Convergence and the
    componentwise violation mask always use raw constraints, so scaling cannot
    silently alter the requested feasibility tolerance. ``bridge`` must
    provide ``set_state()``, ``evaluate_terms()``, and SciPy's value-gradient
    protocol; it may provide ``map_constraints()`` or the legacy
    ``scale_constraints()``.  When ``require_inner_stationarity`` is true, an
    outer update is never applied unless the inner gradient infinity norm
    satisfies the requested L-BFGS-B tolerance.
    """
    x = np.asarray(initial_x, dtype=float).copy()
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("initial_x must be a finite one-dimensional vector")
    for name, value in (
        ("gradient_tolerance", gradient_tolerance),
        ("constraint_tolerance", constraint_tolerance),
        ("argmin_tolerance", argmin_tolerance),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(tau) or tau <= 1:
        raise ValueError("tau must be finite and greater than one")
    if np.isnan(mu_max) or mu_max <= 1:
        raise ValueError("mu_max must be greater than one")
    if max_outer_iterations < 1 or max_inner_iterations < 1:
        raise ValueError("iteration limits must be positive")
    if maxcor < 1 or maxls < 1:
        raise ValueError("maxcor and maxls must be positive")
    if not np.isfinite(inner_stationarity_factor) or inner_stationarity_factor < 1:
        raise ValueError("inner_stationarity_factor must be finite and at least one")

    base_objective, raw_constraints = bridge.evaluate_terms(x)
    raw_constraints = np.asarray(raw_constraints, dtype=float)
    if raw_constraints.ndim != 1 or raw_constraints.size == 0:
        raise ValueError("bridge constraints must be a nonempty vector")
    if not np.isfinite(base_objective) or not np.all(np.isfinite(raw_constraints)):
        raise ValueError("initial objective and constraints must be finite")
    constraint_count = raw_constraints.size
    if hasattr(bridge, "map_constraints"):
        scaled_constraints = np.asarray(
            bridge.map_constraints(raw_constraints), dtype=float
        )
    elif hasattr(bridge, "scale_constraints"):
        scaled_constraints = np.asarray(
            bridge.scale_constraints(raw_constraints), dtype=float
        )
    else:
        scaled_constraints = raw_constraints.copy()
    if scaled_constraints.shape != raw_constraints.shape or not np.all(
        np.isfinite(scaled_constraints)
    ):
        raise ValueError(
            "scaled constraints must be finite and match the raw constraint vector"
        )
    penalties = np.asarray(mu_init, dtype=float)
    if penalties.ndim == 0:
        penalties = np.full(constraint_count, float(penalties))
    else:
        penalties = penalties.copy()
    if (
        penalties.shape != raw_constraints.shape
        or not np.all(np.isfinite(penalties))
        or np.any(penalties <= 1)
        or np.any(penalties > mu_max)
    ):
        raise ValueError(
            "mu_init must be finite, greater than one, no larger than mu_max, "
            "and match the constraint vector"
        )
    if lagrange_multiplier_init is None:
        lagrange_multipliers = np.zeros_like(raw_constraints)
    else:
        lagrange_multipliers = np.asarray(lagrange_multiplier_init, dtype=float).copy()
        if lagrange_multipliers.shape != raw_constraints.shape:
            raise ValueError(
                "lagrange_multiplier_init must match the constraint vector"
            )
        if not np.all(np.isfinite(lagrange_multipliers)):
            raise ValueError("lagrange_multiplier_init must be finite")

    mean_penalty = float(np.mean(penalties))
    omega = 1.0 / mean_penalty
    eta = 1.0 / mean_penalty**0.1
    history = []
    total_inner_iterations = 0
    total_evaluations = 0
    final_gradient = np.full_like(x, np.nan)
    final_value = np.nan
    converged = False
    terminated_by_inner_safeguard = False
    start = time.perf_counter()

    for outer_iteration in range(1, max_outer_iterations + 1):
        multipliers_before = lagrange_multipliers.copy()
        penalties_before = penalties.copy()
        x_before = x.copy()
        bridge.set_state(lagrange_multipliers, penalties)
        evaluation_start = getattr(bridge, "evaluations", 0)
        inner_start = time.perf_counter()
        requested_inner_gradient_tolerance = max(omega, gradient_tolerance)
        inner_result = minimize(
            bridge,
            x,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": max_inner_iterations,
                "maxcor": maxcor,
                "maxls": maxls,
                "gtol": requested_inner_gradient_tolerance,
            },
            tol=argmin_tolerance,
        )
        inner_seconds = time.perf_counter() - inner_start
        x = np.asarray(inner_result.x, dtype=float)
        final_value = float(inner_result.fun)
        final_gradient = np.asarray(inner_result.jac, dtype=float)
        base_objective, raw_constraints = bridge.evaluate_terms(x)
        raw_constraints = np.asarray(raw_constraints, dtype=float)
        if hasattr(bridge, "map_constraints"):
            scaled_constraints = np.asarray(
                bridge.map_constraints(raw_constraints), dtype=float
            )
        elif hasattr(bridge, "scale_constraints"):
            scaled_constraints = np.asarray(
                bridge.scale_constraints(raw_constraints), dtype=float
            )
        else:
            scaled_constraints = raw_constraints.copy()
        if raw_constraints.shape != (constraint_count,):
            raise ValueError(
                "bridge constraint shape changed during the augmented-Lagrangian solve"
            )
        if scaled_constraints.shape != raw_constraints.shape:
            raise ValueError("scaled constraints must match the raw constraint vector")
        if (
            not np.isfinite(final_value)
            or not np.all(np.isfinite(final_gradient))
            or not np.isfinite(base_objective)
            or not np.all(np.isfinite(raw_constraints))
            or not np.all(np.isfinite(scaled_constraints))
        ):
            raise FloatingPointError(
                "inner solve produced a non-finite AL state; inspect scaling, "
                "penalty growth, and coil geometry"
            )
        constraint_norm = float(np.linalg.norm(raw_constraints, ord=np.inf))
        scaled_constraint_norm = float(np.linalg.norm(scaled_constraints, ord=np.inf))
        gradient_norm = float(np.linalg.norm(final_gradient))
        gradient_norm_infinity = float(np.linalg.norm(final_gradient, ord=np.inf))
        step_norm_infinity = float(np.linalg.norm(x - x_before, ord=np.inf))
        inner_stationary = gradient_norm_infinity <= (
            inner_stationarity_factor
            * requested_inner_gradient_tolerance
            * (1.0 + 10.0 * np.finfo(float).eps)
        )
        inner_stage_accepted = inner_stationary or not require_inner_stationarity
        total_inner_iterations += int(inner_result.nit)
        stage_evaluations = int(
            getattr(bridge, "evaluations", evaluation_start) - evaluation_start
        )
        total_evaluations += stage_evaluations

        progress_accepted = scaled_constraint_norm < eta and inner_stage_accepted
        converged = (
            gradient_norm <= gradient_tolerance
            and constraint_norm <= constraint_tolerance
        )
        # Keep a converged result internally consistent: its reported AL value
        # and gradient correspond to the returned multipliers and penalties.
        outer_update_applied = not converged and inner_stage_accepted
        if outer_update_applied:
            violation_mask = np.abs(raw_constraints) > constraint_tolerance
            penalties[violation_mask] = np.minimum(
                penalties[violation_mask] * tau, mu_max
            )
            mean_penalty = float(np.mean(penalties))
            if progress_accepted:
                lagrange_multipliers -= penalties * scaled_constraints
                omega = max(omega / mean_penalty, gradient_tolerance)
                eta = max(eta / mean_penalty, constraint_tolerance)
            else:
                omega = max(1.0 / mean_penalty, gradient_tolerance)
                eta = max(1.0 / mean_penalty**0.1, constraint_tolerance)

        record = {
            "outer_iteration": outer_iteration,
            "inner_success": bool(inner_result.success),
            "inner_status": int(inner_result.status),
            "inner_message": str(inner_result.message),
            "inner_iterations": int(inner_result.nit),
            "inner_evaluations": stage_evaluations,
            "inner_seconds": inner_seconds,
            "augmented_lagrangian": final_value,
            "base_objective": float(base_objective),
            "gradient_norm": gradient_norm,
            "gradient_norm_infinity": gradient_norm_infinity,
            "requested_inner_gradient_tolerance": requested_inner_gradient_tolerance,
            "step_norm_infinity": step_norm_infinity,
            "inner_stationary": bool(inner_stationary),
            "inner_stage_accepted": bool(inner_stage_accepted),
            "outer_update_applied": bool(outer_update_applied),
            "constraint_norm_infinity": constraint_norm,
            "scaled_constraint_norm_infinity": scaled_constraint_norm,
            "constraints": raw_constraints.tolist(),
            "scaled_constraints": scaled_constraints.tolist(),
            "multipliers_before": multipliers_before.tolist(),
            "multipliers_after": lagrange_multipliers.tolist(),
            "penalties_before": penalties_before.tolist(),
            "penalties_after": penalties.tolist(),
            "progress_accepted": bool(progress_accepted),
            "omega_after": omega,
            "eta_after": eta,
            "optimizer_variables": x.tolist(),
        }
        history.append(record)
        if stage_callback is not None:
            stage_callback(x.copy(), dict(record))
        if converged:
            break
        if require_inner_stationarity and not inner_stationary:
            terminated_by_inner_safeguard = True
            break

    seconds = time.perf_counter() - start
    if converged:
        message = "gradient and equality constraints satisfy tolerances"
    elif terminated_by_inner_safeguard:
        message = (
            "inner stationarity safeguard stopped outer updates before penalty "
            "or multiplier escalation"
        )
    else:
        message = "maximum augmented-Lagrangian outer iterations reached"
    return AugmentedLagrangianResult(
        x=x,
        fun=final_value,
        jac=final_gradient,
        lagrange_multipliers=lagrange_multipliers,
        penalties=penalties,
        constraints=raw_constraints,
        scaled_constraints=scaled_constraints,
        base_objective=float(base_objective),
        success=converged,
        message=message,
        outer_iterations=len(history),
        total_inner_iterations=total_inner_iterations,
        total_evaluations=total_evaluations,
        seconds=seconds,
        history=tuple(history),
        terminated_by_inner_safeguard=terminated_by_inner_safeguard,
    )
