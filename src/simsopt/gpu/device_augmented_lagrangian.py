"""Fully device-resident safeguarded augmented-Lagrangian optimization."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .augmented_lagrangian import (
    CONSTRAINT_TRANSFORMS,
    PENALTY_UPDATE_MODES,
    equality_augmented_lagrangian,
    smooth_abs_constraints,
    smooth_sqrt_constraints,
)


@dataclass(frozen=True)
class DeviceAugmentedLagrangianConfig:
    """Static controls for one compiled augmented-Lagrangian solve."""

    mu_init: float = 10.0
    tau: float = 2.0
    mu_max: float = 1e12
    max_outer_iterations: int = 8
    max_inner_iterations: int = 300
    history_size: int = 20
    max_line_search_iterations: int = 50
    gradient_tolerance: float = 1e-8
    constraint_tolerance: float = 1e-6
    armijo_coefficient: float = 1e-4
    backtracking_factor: float = 0.5
    minimum_step_size: float = 1e-12
    curvature_tolerance: float = 1e-12
    require_inner_stationarity: bool = True
    inner_stationarity_factor: float = 1.0
    inner_stationarity_relative_tolerance: float | None = 0.01
    penalty_update_mode: str = "global"
    constraint_transform: str = "smooth_abs"
    transform_epsilon: float = 0.1
    constraint_scale_reduction_factor: float = 0.5

    def __post_init__(self):
        for name in (
            "mu_init",
            "tau",
            "mu_max",
            "gradient_tolerance",
            "constraint_tolerance",
            "armijo_coefficient",
            "backtracking_factor",
            "minimum_step_size",
            "curvature_tolerance",
            "inner_stationarity_factor",
            "transform_epsilon",
            "constraint_scale_reduction_factor",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.mu_init <= 1 or self.mu_init > self.mu_max:
            raise ValueError("mu_init must be greater than one and no larger than mu_max")
        if self.tau <= 1:
            raise ValueError("tau must be greater than one")
        if self.max_outer_iterations < 1 or self.max_inner_iterations < 1:
            raise ValueError("iteration limits must be positive")
        if self.history_size < 1 or self.max_line_search_iterations < 1:
            raise ValueError("history and line-search limits must be positive")
        if self.armijo_coefficient >= 1 or self.backtracking_factor >= 1:
            raise ValueError("Armijo and backtracking coefficients must be below one")
        if self.inner_stationarity_factor < 1:
            raise ValueError("inner_stationarity_factor must be at least one")
        relative = self.inner_stationarity_relative_tolerance
        if relative is not None and not 0 < relative < 1:
            raise ValueError(
                "inner_stationarity_relative_tolerance must be between zero and one"
            )
        if self.penalty_update_mode not in PENALTY_UPDATE_MODES:
            raise ValueError("unsupported penalty_update_mode")
        if self.constraint_transform not in CONSTRAINT_TRANSFORMS:
            raise ValueError("unsupported constraint_transform")
        if self.constraint_scale_reduction_factor > 1:
            raise ValueError("constraint_scale_reduction_factor must not exceed one")


class _LineState(NamedTuple):
    step: jax.Array
    x: jax.Array
    value: jax.Array
    gradient: jax.Array
    attempts: jax.Array
    accepted: jax.Array


class _InnerState(NamedTuple):
    x: jax.Array
    value: jax.Array
    gradient: jax.Array
    s_history: jax.Array
    y_history: jax.Array
    rho_history: jax.Array
    history_count: jax.Array
    iterations: jax.Array
    evaluations: jax.Array
    reason: jax.Array


class _OuterState(NamedTuple):
    x: jax.Array
    lagrange_multipliers: jax.Array
    penalties: jax.Array
    constraint_scales: jax.Array
    omega: jax.Array
    eta: jax.Array
    outer_iterations: jax.Array
    total_inner_iterations: jax.Array
    total_evaluations: jax.Array
    reason: jax.Array
    final_value: jax.Array
    final_gradient: jax.Array
    base_objective: jax.Array
    raw_constraints: jax.Array
    mapped_constraints: jax.Array
    base_objective_history: jax.Array
    augmented_lagrangian_history: jax.Array
    constraint_norm_history: jax.Array
    mapped_constraint_norm_history: jax.Array
    gradient_norm_history: jax.Array
    initial_gradient_norm_history: jax.Array
    step_norm_history: jax.Array
    inner_iteration_history: jax.Array
    inner_evaluation_history: jax.Array
    inner_status_history: jax.Array
    inner_stationary_history: jax.Array
    progress_accepted_history: jax.Array
    outer_update_history: jax.Array
    penalty_max_history: jax.Array


@dataclass(frozen=True)
class DeviceAugmentedLagrangianResult:
    """Host representation of one fully device-resident AL solve."""

    x: np.ndarray
    fun: float
    jac: np.ndarray
    lagrange_multipliers: np.ndarray
    penalties: np.ndarray
    constraint_scales: np.ndarray
    constraints: np.ndarray
    scaled_constraints: np.ndarray
    base_objective: float
    success: bool
    status: int
    message: str
    outer_iterations: int
    total_inner_iterations: int
    total_evaluations: int
    seconds: float
    history: tuple
    terminated_by_inner_safeguard: bool
    device_resident: bool = True
    host_callbacks: int = 0


_STATUS_MESSAGES = {
    1: "gradient and equality constraints satisfy tolerances",
    2: "inner stationarity safeguard stopped outer updates",
    3: "maximum augmented-Lagrangian outer iterations reached",
}


class DeviceAugmentedLagrangian:
    """Compile the complete AL outer and inner iteration tree on one device.

    ``terms(x)`` returns ``(base_objective, raw_constraint_vector)``. Constraint
    mapping, L-BFGS inner solves, multiplier and penalty updates, scaling
    continuation, and safeguards all execute in one compiled call.
    """

    def __init__(
        self,
        terms,
        initial_x,
        constraint_count: int,
        constraint_scales=None,
        minimum_constraint_scales=None,
        *,
        config: DeviceAugmentedLagrangianConfig | None = None,
        platform: str | None = None,
    ):
        if config is None:
            config = DeviceAugmentedLagrangianConfig()
        initial_x = np.asarray(initial_x)
        if initial_x.ndim != 1:
            raise ValueError("initial_x must be one-dimensional")
        if not np.issubdtype(initial_x.dtype, np.floating):
            raise TypeError("initial_x must have a floating-point dtype")
        if constraint_count < 1:
            raise ValueError("constraint_count must be positive")
        self.config = config
        self._shape = initial_x.shape
        self._dtype = initial_x.dtype
        self._constraint_count = int(constraint_count)
        if platform is None:
            self._device = None
        else:
            try:
                self._device = jax.devices(platform)[0]
            except (RuntimeError, IndexError) as error:
                raise ValueError(f"JAX platform {platform!r} is unavailable") from error
        if constraint_scales is None:
            constraint_scales = np.ones(constraint_count, dtype=initial_x.dtype)
        self._constraint_scales = self._coerce_scales(
            constraint_scales, "constraint_scales"
        )
        if minimum_constraint_scales is None:
            minimum_constraint_scales = self._constraint_scales
        self._minimum_constraint_scales = self._coerce_scales(
            minimum_constraint_scales, "minimum_constraint_scales"
        )
        if np.any(self._minimum_constraint_scales > self._constraint_scales):
            raise ValueError("minimum constraint scales must not exceed initial scales")
        self._solver = jax.jit(self._build_solver(terms, config))
        self._compiled = None

    def _coerce_scales(self, value, name):
        value = np.asarray(value, dtype=self._dtype)
        expected = (self._constraint_count,)
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
        if not np.all(np.isfinite(value)) or np.any(value <= 0):
            raise ValueError(f"{name} must be finite and positive")
        return value.copy()

    @property
    def is_compiled(self):
        return self._compiled is not None

    @property
    def device_platform(self):
        if self._device is None:
            return jax.default_backend()
        return self._device.platform

    def _device_put(self, value):
        return jax.device_put(value, self._device)

    def _coerce_x(self, x):
        x = np.asarray(x, dtype=self._dtype)
        if x.shape != self._shape:
            raise ValueError(f"x must have shape {self._shape}, got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must be finite")
        return x

    @staticmethod
    def _build_solver(terms, config):
        history_size = config.history_size
        max_outer = config.max_outer_iterations
        max_inner = config.max_inner_iterations

        def map_constraints(raw, scales):
            if config.constraint_transform == "smooth_sqrt":
                mapped = smooth_sqrt_constraints(raw, config.transform_epsilon)
            elif config.constraint_transform == "smooth_abs":
                mapped = smooth_abs_constraints(raw, config.transform_epsilon)
            else:
                mapped = raw
            return mapped / scales

        def al_value(x, multipliers, penalties, scales):
            base, raw = terms(x)
            mapped = map_constraints(raw, scales)
            return equality_augmented_lagrangian(
                base, mapped, multipliers, penalties
            )

        value_and_grad = jax.value_and_grad(al_value, argnums=0)

        def direction(state):
            coefficients = jnp.zeros((history_size,), dtype=state.x.dtype)

            def first(index, carry):
                vector, alpha = carry
                active = index < state.history_count
                coefficient = jnp.where(
                    active,
                    state.rho_history[index]
                    * jnp.vdot(state.s_history[index], vector),
                    0.0,
                )
                return (
                    vector - coefficient * state.y_history[index],
                    alpha.at[index].set(coefficient),
                )

            vector, coefficients = jax.lax.fori_loop(
                0, history_size, first, (state.gradient, coefficients)
            )
            newest_scale = jnp.vdot(
                state.s_history[0], state.y_history[0]
            ) / jnp.maximum(jnp.vdot(state.y_history[0], state.y_history[0]), 1e-30)
            vector = jnp.where(
                state.history_count > 0,
                jnp.clip(newest_scale, 1e-8, 1e8) * vector,
                vector,
            )

            def second(offset, vector):
                index = history_size - 1 - offset
                active = index < state.history_count
                beta = jnp.where(
                    active,
                    state.rho_history[index]
                    * jnp.vdot(state.y_history[index], vector),
                    0.0,
                )
                return vector + state.s_history[index] * (
                    coefficients[index] - beta
                )

            vector = jax.lax.fori_loop(0, history_size, second, vector)
            candidate = -vector
            dot = jnp.vdot(state.gradient, candidate)
            floor = (
                1e-14
                * jnp.linalg.norm(state.gradient)
                * jnp.maximum(jnp.linalg.norm(candidate), 1e-30)
            )
            return jnp.where(dot < -floor, candidate, -state.gradient)

        def inner_solve(initial_x, multipliers, penalties, scales, tolerance):
            initial_value, initial_gradient = value_and_grad(
                initial_x, multipliers, penalties, scales
            )
            initial_gradient_norm = jnp.linalg.norm(
                initial_gradient, ord=jnp.inf
            )
            initial = _InnerState(
                x=initial_x,
                value=initial_value,
                gradient=initial_gradient,
                s_history=jnp.zeros(
                    (history_size, initial_x.size), dtype=initial_x.dtype
                ),
                y_history=jnp.zeros(
                    (history_size, initial_x.size), dtype=initial_x.dtype
                ),
                rho_history=jnp.zeros((history_size,), dtype=initial_x.dtype),
                history_count=jnp.asarray(0, dtype=jnp.int32),
                iterations=jnp.asarray(0, dtype=jnp.int32),
                evaluations=jnp.asarray(1, dtype=jnp.int32),
                reason=jnp.where(initial_gradient_norm <= tolerance, 1, 0).astype(
                    jnp.int32
                ),
            )

            def iteration(state):
                search_direction = direction(state)
                directional_derivative = jnp.vdot(
                    state.gradient, search_direction
                )
                first_step = jnp.minimum(
                    1.0,
                    1.0
                    / jnp.maximum(
                        jnp.linalg.norm(state.gradient, ord=jnp.inf), 1e-30
                    ),
                )
                line_initial = _LineState(
                    step=jnp.where(state.iterations == 0, first_step, 1.0).astype(
                        state.x.dtype
                    ),
                    x=state.x,
                    value=state.value,
                    gradient=state.gradient,
                    attempts=jnp.asarray(0, dtype=jnp.int32),
                    accepted=jnp.asarray(False),
                )

                def line_condition(line):
                    return (
                        (line.attempts < config.max_line_search_iterations)
                        & (~line.accepted)
                        & (line.step >= config.minimum_step_size)
                    )

                def line_iteration(line):
                    candidate_x = state.x + line.step * search_direction
                    candidate_value, candidate_gradient = value_and_grad(
                        candidate_x, multipliers, penalties, scales
                    )
                    finite = jnp.isfinite(candidate_value) & jnp.all(
                        jnp.isfinite(candidate_gradient)
                    )
                    armijo = candidate_value <= (
                        state.value
                        + config.armijo_coefficient
                        * line.step
                        * directional_derivative
                    )
                    accepted = finite & armijo
                    return _LineState(
                        step=jnp.where(
                            accepted,
                            line.step,
                            line.step * config.backtracking_factor,
                        ),
                        x=candidate_x,
                        value=candidate_value,
                        gradient=candidate_gradient,
                        attempts=line.attempts + 1,
                        accepted=accepted,
                    )

                line = jax.lax.while_loop(
                    line_condition, line_iteration, line_initial
                )

                def failed(_):
                    return state._replace(
                        evaluations=state.evaluations + line.attempts,
                        reason=jnp.asarray(4, dtype=jnp.int32),
                    )

                def accepted(_):
                    displacement = line.x - state.x
                    gradient_change = line.gradient - state.gradient
                    curvature = jnp.vdot(displacement, gradient_change)
                    curvature_floor = (
                        config.curvature_tolerance
                        * jnp.linalg.norm(displacement)
                        * jnp.linalg.norm(gradient_change)
                    )
                    update = curvature > curvature_floor
                    shifted_s = jnp.concatenate(
                        (displacement[None, :], state.s_history[:-1]), axis=0
                    )
                    shifted_y = jnp.concatenate(
                        (gradient_change[None, :], state.y_history[:-1]), axis=0
                    )
                    shifted_rho = jnp.concatenate(
                        (
                            jnp.asarray([1.0 / jnp.maximum(curvature, 1e-30)]),
                            state.rho_history[:-1],
                        )
                    )
                    iterations = state.iterations + 1
                    converged = (
                        jnp.linalg.norm(line.gradient, ord=jnp.inf) <= tolerance
                    )
                    return _InnerState(
                        x=line.x,
                        value=line.value,
                        gradient=line.gradient,
                        s_history=jnp.where(update, shifted_s, state.s_history),
                        y_history=jnp.where(update, shifted_y, state.y_history),
                        rho_history=jnp.where(
                            update, shifted_rho, state.rho_history
                        ),
                        history_count=jnp.where(
                            update,
                            jnp.minimum(state.history_count + 1, history_size),
                            state.history_count,
                        ),
                        iterations=iterations,
                        evaluations=state.evaluations + line.attempts,
                        reason=jnp.where(converged, 1, 0).astype(jnp.int32),
                    )

                return jax.lax.cond(line.accepted, accepted, failed, operand=None)

            def condition(state):
                return (state.iterations < max_inner) & (state.reason == 0)

            state = jax.lax.while_loop(condition, iteration, initial)
            state = state._replace(
                reason=jnp.where(
                    (state.reason == 0) & (state.iterations >= max_inner),
                    5,
                    state.reason,
                ).astype(jnp.int32)
            )
            return state, initial_gradient_norm

        def solve(initial_x, initial_scales, minimum_scales):
            base, raw = terms(initial_x)
            mapped = map_constraints(raw, initial_scales)
            penalties = jnp.full_like(raw, config.mu_init)
            multipliers = jnp.zeros_like(raw)
            mean_penalty = jnp.mean(penalties)
            outer_shape = (max_outer,)
            nan_history = jnp.full(outer_shape, jnp.nan, dtype=initial_x.dtype)
            int_history = jnp.zeros(outer_shape, dtype=jnp.int32)
            bool_history = jnp.zeros(outer_shape, dtype=jnp.bool_)
            initial = _OuterState(
                x=initial_x,
                lagrange_multipliers=multipliers,
                penalties=penalties,
                constraint_scales=initial_scales,
                omega=1.0 / mean_penalty,
                eta=1.0 / mean_penalty**0.1,
                outer_iterations=jnp.asarray(0, dtype=jnp.int32),
                total_inner_iterations=jnp.asarray(0, dtype=jnp.int32),
                total_evaluations=jnp.asarray(0, dtype=jnp.int32),
                reason=jnp.asarray(0, dtype=jnp.int32),
                final_value=jnp.asarray(jnp.nan, dtype=initial_x.dtype),
                final_gradient=jnp.full_like(initial_x, jnp.nan),
                base_objective=base,
                raw_constraints=raw,
                mapped_constraints=mapped,
                base_objective_history=nan_history,
                augmented_lagrangian_history=nan_history,
                constraint_norm_history=nan_history,
                mapped_constraint_norm_history=nan_history,
                gradient_norm_history=nan_history,
                initial_gradient_norm_history=nan_history,
                step_norm_history=nan_history,
                inner_iteration_history=int_history,
                inner_evaluation_history=int_history,
                inner_status_history=int_history,
                inner_stationary_history=bool_history,
                progress_accepted_history=bool_history,
                outer_update_history=bool_history,
                penalty_max_history=nan_history,
            )

            def outer_iteration(state):
                requested_tolerance = jnp.maximum(
                    state.omega, config.gradient_tolerance
                )
                inner, initial_gradient_norm = inner_solve(
                    state.x,
                    state.lagrange_multipliers,
                    state.penalties,
                    state.constraint_scales,
                    requested_tolerance,
                )
                base, raw = terms(inner.x)
                mapped = map_constraints(raw, state.constraint_scales)
                raw_norm = jnp.linalg.norm(raw, ord=jnp.inf)
                mapped_norm = jnp.linalg.norm(mapped, ord=jnp.inf)
                gradient_norm = jnp.linalg.norm(inner.gradient, ord=jnp.inf)
                absolute_threshold = (
                    config.inner_stationarity_factor
                    * requested_tolerance
                    * (1.0 + 10.0 * jnp.finfo(initial_x.dtype).eps)
                )
                if config.inner_stationarity_relative_tolerance is None:
                    relative_threshold = jnp.asarray(0.0, dtype=initial_x.dtype)
                else:
                    relative_threshold = (
                        config.inner_stationarity_relative_tolerance
                        * initial_gradient_norm
                    )
                stationarity_threshold = jnp.maximum(
                    absolute_threshold, relative_threshold
                )
                inner_stationary = gradient_norm <= stationarity_threshold
                inner_accepted = inner_stationary | (
                    not config.require_inner_stationarity
                )
                progress = (mapped_norm < state.eta) & inner_accepted
                if config.inner_stationarity_relative_tolerance is None:
                    stationary_convergence = (
                        jnp.linalg.norm(inner.gradient)
                        <= config.gradient_tolerance
                    )
                else:
                    stationary_convergence = inner_stationary
                converged = stationary_convergence & (
                    raw_norm <= config.constraint_tolerance
                )
                apply_update = (~converged) & inner_accepted
                violation = jnp.abs(raw) > config.constraint_tolerance
                if config.penalty_update_mode == "global":
                    violation = jnp.full_like(
                        violation, jnp.any(violation), dtype=jnp.bool_
                    )
                grown_penalties = jnp.where(
                    violation,
                    jnp.minimum(state.penalties * config.tau, config.mu_max),
                    state.penalties,
                )
                updated_penalties = jnp.where(
                    apply_update, grown_penalties, state.penalties
                )
                mean_penalty = jnp.mean(updated_penalties)
                multiplier_candidate = (
                    state.lagrange_multipliers - updated_penalties * mapped
                )
                scale_candidate = jnp.maximum(
                    minimum_scales,
                    state.constraint_scales
                    * config.constraint_scale_reduction_factor,
                )
                multiplier_candidate = multiplier_candidate * (
                    scale_candidate / state.constraint_scales
                )
                progress_update = apply_update & progress
                updated_multipliers = jnp.where(
                    progress_update,
                    multiplier_candidate,
                    state.lagrange_multipliers,
                )
                updated_scales = jnp.where(
                    progress_update, scale_candidate, state.constraint_scales
                )
                progress_omega = jnp.maximum(
                    state.omega / mean_penalty, config.gradient_tolerance
                )
                rejected_omega = jnp.maximum(
                    1.0 / mean_penalty, config.gradient_tolerance
                )
                progress_eta = jnp.maximum(
                    state.eta / mean_penalty, config.constraint_tolerance
                )
                rejected_eta = jnp.maximum(
                    1.0 / mean_penalty**0.1, config.constraint_tolerance
                )
                updated_omega = jnp.where(
                    apply_update,
                    jnp.where(progress, progress_omega, rejected_omega),
                    state.omega,
                )
                updated_eta = jnp.where(
                    apply_update,
                    jnp.where(progress, progress_eta, rejected_eta),
                    state.eta,
                )
                mapped_after = map_constraints(raw, updated_scales)
                stopped_by_safeguard = (
                    config.require_inner_stationarity & (~inner_stationary)
                )
                reason = jnp.where(
                    converged,
                    1,
                    jnp.where(stopped_by_safeguard, 2, 0),
                ).astype(jnp.int32)
                index = state.outer_iterations
                return _OuterState(
                    x=inner.x,
                    lagrange_multipliers=updated_multipliers,
                    penalties=updated_penalties,
                    constraint_scales=updated_scales,
                    omega=updated_omega,
                    eta=updated_eta,
                    outer_iterations=index + 1,
                    total_inner_iterations=(
                        state.total_inner_iterations + inner.iterations
                    ),
                    total_evaluations=state.total_evaluations + inner.evaluations,
                    reason=reason,
                    final_value=inner.value,
                    final_gradient=inner.gradient,
                    base_objective=base,
                    raw_constraints=raw,
                    mapped_constraints=mapped_after,
                    base_objective_history=state.base_objective_history.at[
                        index
                    ].set(base),
                    augmented_lagrangian_history=(
                        state.augmented_lagrangian_history.at[index].set(
                            inner.value
                        )
                    ),
                    constraint_norm_history=state.constraint_norm_history.at[
                        index
                    ].set(raw_norm),
                    mapped_constraint_norm_history=(
                        state.mapped_constraint_norm_history.at[index].set(
                            mapped_norm
                        )
                    ),
                    gradient_norm_history=state.gradient_norm_history.at[index].set(
                        gradient_norm
                    ),
                    initial_gradient_norm_history=(
                        state.initial_gradient_norm_history.at[index].set(
                            initial_gradient_norm
                        )
                    ),
                    step_norm_history=state.step_norm_history.at[index].set(
                        jnp.linalg.norm(inner.x - state.x, ord=jnp.inf)
                    ),
                    inner_iteration_history=state.inner_iteration_history.at[
                        index
                    ].set(inner.iterations),
                    inner_evaluation_history=state.inner_evaluation_history.at[
                        index
                    ].set(inner.evaluations),
                    inner_status_history=state.inner_status_history.at[index].set(
                        inner.reason
                    ),
                    inner_stationary_history=state.inner_stationary_history.at[
                        index
                    ].set(inner_stationary),
                    progress_accepted_history=state.progress_accepted_history.at[
                        index
                    ].set(progress),
                    outer_update_history=state.outer_update_history.at[index].set(
                        apply_update
                    ),
                    penalty_max_history=state.penalty_max_history.at[index].set(
                        jnp.max(updated_penalties)
                    ),
                )

            def condition(state):
                return (state.outer_iterations < max_outer) & (state.reason == 0)

            state = jax.lax.while_loop(condition, outer_iteration, initial)
            return state._replace(
                reason=jnp.where(
                    (state.reason == 0) & (state.outer_iterations >= max_outer),
                    3,
                    state.reason,
                ).astype(jnp.int32)
            )

        return solve

    def compile(self, example_x=None):
        """Compile the complete nested AL solve without executing it."""
        if example_x is None:
            example_x = np.zeros(self._shape, dtype=self._dtype)
        x = self._device_put(self._coerce_x(example_x))
        scales = self._device_put(self._constraint_scales)
        minimum_scales = self._device_put(self._minimum_constraint_scales)
        self._compiled = self._solver.lower(x, scales, minimum_scales).compile()
        return self

    def run(self, initial_x):
        """Execute one compiled AL solve and transfer only its result to host."""
        x = self._device_put(self._coerce_x(initial_x))
        if self._compiled is None:
            self.compile(initial_x)
        start = time.perf_counter()
        raw = self._compiled(
            x,
            self._device_put(self._constraint_scales),
            self._device_put(self._minimum_constraint_scales),
        )
        jax.block_until_ready(raw)
        seconds = time.perf_counter() - start
        outer_iterations = int(raw.outer_iterations)
        history = []
        for index in range(outer_iterations):
            history.append(
                {
                    "outer_iteration": index + 1,
                    "inner_iterations": int(raw.inner_iteration_history[index]),
                    "inner_evaluations": int(raw.inner_evaluation_history[index]),
                    "inner_status": int(raw.inner_status_history[index]),
                    "augmented_lagrangian": float(
                        raw.augmented_lagrangian_history[index]
                    ),
                    "base_objective": float(raw.base_objective_history[index]),
                    "gradient_norm_infinity": float(
                        raw.gradient_norm_history[index]
                    ),
                    "stage_initial_gradient_norm_infinity": float(
                        raw.initial_gradient_norm_history[index]
                    ),
                    "step_norm_infinity": float(raw.step_norm_history[index]),
                    "constraint_norm_infinity": float(
                        raw.constraint_norm_history[index]
                    ),
                    "scaled_constraint_norm_infinity": float(
                        raw.mapped_constraint_norm_history[index]
                    ),
                    "inner_stationary": bool(
                        raw.inner_stationary_history[index]
                    ),
                    "progress_accepted": bool(
                        raw.progress_accepted_history[index]
                    ),
                    "outer_update_applied": bool(raw.outer_update_history[index]),
                    "maximum_penalty_after": float(raw.penalty_max_history[index]),
                }
            )
        status = int(raw.reason)
        return DeviceAugmentedLagrangianResult(
            x=np.asarray(raw.x),
            fun=float(raw.final_value),
            jac=np.asarray(raw.final_gradient),
            lagrange_multipliers=np.asarray(raw.lagrange_multipliers),
            penalties=np.asarray(raw.penalties),
            constraint_scales=np.asarray(raw.constraint_scales),
            constraints=np.asarray(raw.raw_constraints),
            scaled_constraints=np.asarray(raw.mapped_constraints),
            base_objective=float(raw.base_objective),
            success=status == 1,
            status=status,
            message=_STATUS_MESSAGES.get(status, "unknown termination"),
            outer_iterations=outer_iterations,
            total_inner_iterations=int(raw.total_inner_iterations),
            total_evaluations=int(raw.total_evaluations),
            seconds=seconds,
            history=tuple(history),
            terminated_by_inner_safeguard=status == 2,
        )
