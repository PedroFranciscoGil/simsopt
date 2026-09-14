"""Device-resident limited-memory BFGS and target-aware checkpointing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class TargetAwareConfig:
    """Stopping and checkpoint policy shared by host and device solvers."""

    target_flux: float = 1.1e-5
    feasibility_tolerance: float = 1e-12
    checkpoint_patience: int = 25
    infeasible_patience: int = 15
    minimum_relative_improvement: float = 1e-3

    def __post_init__(self):
        if not math.isfinite(self.target_flux) or self.target_flux <= 0:
            raise ValueError("target_flux must be finite and positive")
        if (
            not math.isfinite(self.feasibility_tolerance)
            or self.feasibility_tolerance < 0
        ):
            raise ValueError("feasibility_tolerance must be finite and nonnegative")
        if self.checkpoint_patience < 1 or self.infeasible_patience < 1:
            raise ValueError("checkpoint patience values must be positive")
        if (
            not math.isfinite(self.minimum_relative_improvement)
            or not 0 <= self.minimum_relative_improvement < 1
        ):
            raise ValueError("minimum_relative_improvement must be in [0, 1)")


class TargetAwareCheckpointRecorder:
    """Stop SciPy after a qualified accepted checkpoint plateaus."""

    def __init__(self, objective, quality, config: TargetAwareConfig):
        self.objective = objective
        self.quality = quality
        self.config = config
        self.evaluations = 0
        self.iteration_states = []
        self.checkpoints = []
        self._initial_state = None
        self._target_found = False
        self._significant_flux = math.inf
        self._stagnation = 0
        self._infeasible = 0
        self.stop_reason = None

    def __call__(self, x):
        self.evaluations += 1
        return self.objective(x)

    def initialize(self, x):
        """Register the initial state before passing the callback to SciPy."""
        if self.checkpoints:
            raise RuntimeError("checkpoint recorder is already initialized")
        self._initial_state = np.asarray(x, dtype=float).copy()
        self._record(self._initial_state, accepted_iteration=0)
        return self

    def callback(self, x):
        state = np.asarray(x, dtype=float).copy()
        self.iteration_states.append(state)
        self._record(state, accepted_iteration=len(self.iteration_states))
        if self.stop_reason is not None:
            raise StopIteration

    def _record(self, x, accepted_iteration):
        flux, residuals = self.quality(x)
        residual_norm = float(np.linalg.norm(residuals, ord=np.inf))
        flux = float(flux)
        feasible = residual_norm <= self.config.feasibility_tolerance
        target_feasible = feasible and flux <= self.config.target_flux
        significant = target_feasible and (
            not self._target_found
            or flux
            <= self._significant_flux * (1.0 - self.config.minimum_relative_improvement)
        )
        self._target_found = self._target_found or target_feasible
        if significant:
            self._significant_flux = flux
            self._stagnation = 0
        elif self._target_found:
            self._stagnation += 1
        if self._target_found and not feasible:
            self._infeasible += 1
        else:
            self._infeasible = 0
        checkpoint = {
            "index": int(accepted_iteration),
            "quadratic_flux": flux,
            "target_envelope_residual_norm_infinity": residual_norm,
            "target_feasible": bool(feasible),
            "flux_target_satisfied": bool(target_feasible),
            "significant_target_improvement": bool(significant),
            "stagnation_count": self._stagnation,
            "consecutive_infeasible_count": self._infeasible,
        }
        self.checkpoints.append(checkpoint)
        if self._target_found and self._infeasible >= self.config.infeasible_patience:
            self.stop_reason = "target_reached_then_sustained_infeasibility"
        elif self._target_found and self._stagnation >= self.config.checkpoint_patience:
            self.stop_reason = "target_reached_then_feasible_stagnation"

    def selection(self):
        """Return the minimum-flux feasible accepted state and summary."""
        if self._initial_state is None:
            raise RuntimeError("checkpoint recorder has not been initialized")
        states = [self._initial_state, *self.iteration_states]
        feasible = [
            index
            for index, checkpoint in enumerate(self.checkpoints)
            if checkpoint["target_feasible"]
        ]
        if feasible:
            selected = min(
                feasible,
                key=lambda index: self.checkpoints[index]["quadratic_flux"],
            )
        else:
            selected = min(
                range(len(self.checkpoints)),
                key=lambda index: (
                    self.checkpoints[index]["target_envelope_residual_norm_infinity"],
                    self.checkpoints[index]["quadratic_flux"],
                ),
            )
        return np.asarray(states[selected], dtype=float).copy(), {
            "candidate_count": len(self.checkpoints),
            "feasible_candidate_count": len(feasible),
            "selected_index": selected,
            "selected": self.checkpoints[selected],
            "stop_reason": self.stop_reason,
            "checkpoints": list(self.checkpoints),
        }


@dataclass(frozen=True)
class DeviceLBFGSConfig:
    """Static controls for one compiled device-resident L-BFGS solve."""

    history_size: int = 20
    max_iterations: int = 600
    max_line_search_iterations: int = 30
    gradient_tolerance: float = 1e-8
    armijo_coefficient: float = 1e-4
    backtracking_factor: float = 0.5
    minimum_step_size: float = 1e-12
    curvature_tolerance: float = 1e-12
    target: TargetAwareConfig = TargetAwareConfig()

    def __post_init__(self):
        if self.history_size < 1 or self.max_iterations < 1:
            raise ValueError("history_size and max_iterations must be positive")
        if self.max_line_search_iterations < 1:
            raise ValueError("max_line_search_iterations must be positive")
        for name in (
            "gradient_tolerance",
            "armijo_coefficient",
            "backtracking_factor",
            "minimum_step_size",
            "curvature_tolerance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.armijo_coefficient >= 1:
            raise ValueError("armijo_coefficient must be below one")
        if self.backtracking_factor >= 1:
            raise ValueError("backtracking_factor must be below one")


class _LineSearchState(NamedTuple):
    step: jax.Array
    x: jax.Array
    value: jax.Array
    gradient: jax.Array
    attempts: jax.Array
    accepted: jax.Array


class _LBFGSState(NamedTuple):
    x: jax.Array
    value: jax.Array
    gradient: jax.Array
    s_history: jax.Array
    y_history: jax.Array
    rho_history: jax.Array
    history_count: jax.Array
    iteration: jax.Array
    evaluations: jax.Array
    reason: jax.Array
    best_x: jax.Array
    best_flux: jax.Array
    best_iteration: jax.Array
    fallback_x: jax.Array
    best_residual: jax.Array
    fallback_flux: jax.Array
    fallback_iteration: jax.Array
    found_feasible: jax.Array
    target_found: jax.Array
    significant_flux: jax.Array
    stagnation: jax.Array
    infeasible: jax.Array
    objective_history: jax.Array
    gradient_history: jax.Array
    flux_history: jax.Array
    residual_history: jax.Array
    feasible_history: jax.Array
    step_history: jax.Array


@dataclass(frozen=True)
class DeviceLBFGSResult:
    """Host representation of one device-resident solve."""

    x: np.ndarray
    terminal_x: np.ndarray
    fun: float
    jac: np.ndarray
    quadratic_flux: float
    target_envelope_residual_norm_infinity: float
    success: bool
    status: int
    message: str
    iterations: int
    evaluations: int
    found_feasible_checkpoint: bool
    target_found: bool
    selected_iteration: int
    objective_history: np.ndarray
    gradient_norm_infinity_history: np.ndarray
    quadratic_flux_history: np.ndarray
    target_residual_norm_infinity_history: np.ndarray
    target_feasible_history: np.ndarray
    step_size_history: np.ndarray


_STATUS_MESSAGES = {
    1: "gradient tolerance satisfied",
    2: "target reached; feasible checkpoint improvement plateaued",
    3: "target reached; sustained target-envelope infeasibility",
    4: "Armijo line search failed",
    5: "maximum iterations reached",
}


class DeviceLBFGS:
    """Compile and execute an entire L-BFGS solve on one JAX device."""

    def __init__(
        self,
        objective,
        quality,
        initial_x,
        *,
        config: DeviceLBFGSConfig | None = None,
        platform: str | None = None,
    ):
        if config is None:
            config = DeviceLBFGSConfig()
        initial_x = np.asarray(initial_x)
        if initial_x.ndim != 1:
            raise ValueError("initial_x must be one-dimensional")
        if not np.issubdtype(initial_x.dtype, np.floating):
            raise TypeError("initial_x must have a floating-point dtype")
        self.config = config
        self._shape = initial_x.shape
        self._dtype = initial_x.dtype
        if platform is None:
            self._device = None
        else:
            try:
                self._device = jax.devices(platform)[0]
            except (RuntimeError, IndexError) as error:
                raise ValueError(f"JAX platform {platform!r} is unavailable") from error
        value_and_grad = jax.value_and_grad(objective)
        self._solver = jax.jit(self._build_solver(value_and_grad, quality, config))
        self._compiled = None

    @staticmethod
    def _build_solver(value_and_grad, quality, config):
        history_size = config.history_size
        max_iterations = config.max_iterations
        target = config.target

        def quality_values(x):
            flux, residuals = quality(x)
            return flux, jnp.max(jnp.abs(residuals))

        def direction(state):
            alpha = jnp.zeros((history_size,), dtype=state.x.dtype)

            def first_loop(index, carry):
                vector, coefficients = carry
                active = index < state.history_count
                coefficient = jnp.where(
                    active,
                    state.rho_history[index] * jnp.vdot(state.s_history[index], vector),
                    0.0,
                )
                vector = vector - coefficient * state.y_history[index]
                coefficients = coefficients.at[index].set(coefficient)
                return vector, coefficients

            vector, alpha = jax.lax.fori_loop(
                0, history_size, first_loop, (state.gradient, alpha)
            )
            newest_scale = jnp.vdot(
                state.s_history[0], state.y_history[0]
            ) / jnp.maximum(jnp.vdot(state.y_history[0], state.y_history[0]), 1e-30)
            scale = jnp.where(state.history_count > 0, newest_scale, 1.0)
            scale = jnp.clip(scale, 1e-8, 1e8)
            vector = scale * vector

            def second_loop(offset, vector):
                index = history_size - 1 - offset
                active = index < state.history_count
                beta = jnp.where(
                    active,
                    state.rho_history[index] * jnp.vdot(state.y_history[index], vector),
                    0.0,
                )
                return vector + state.s_history[index] * (alpha[index] - beta)

            vector = jax.lax.fori_loop(0, history_size, second_loop, vector)
            candidate = -vector
            gradient_dot_direction = jnp.vdot(state.gradient, candidate)
            descent_scale = (
                1e-14
                * jnp.linalg.norm(state.gradient)
                * jnp.maximum(jnp.linalg.norm(candidate), 1e-30)
            )
            return jnp.where(
                gradient_dot_direction < -descent_scale,
                candidate,
                -state.gradient,
            )

        def iteration_body(state):
            search_direction = direction(state)
            directional_derivative = jnp.vdot(state.gradient, search_direction)
            first_step = jnp.minimum(
                1.0,
                1.0 / jnp.maximum(jnp.linalg.norm(state.gradient, ord=jnp.inf), 1e-30),
            )
            initial_line_state = _LineSearchState(
                step=jnp.where(state.iteration == 0, first_step, 1.0).astype(
                    state.x.dtype
                ),
                x=state.x,
                value=state.value,
                gradient=state.gradient,
                attempts=jnp.asarray(0, dtype=jnp.int32),
                accepted=jnp.asarray(False),
            )

            def line_condition(line_state):
                return (
                    (line_state.attempts < config.max_line_search_iterations)
                    & (~line_state.accepted)
                    & (line_state.step >= config.minimum_step_size)
                )

            def line_body(line_state):
                candidate_x = state.x + line_state.step * search_direction
                candidate_value, candidate_gradient = value_and_grad(candidate_x)
                finite = jnp.isfinite(candidate_value) & jnp.all(
                    jnp.isfinite(candidate_gradient)
                )
                armijo = candidate_value <= (
                    state.value
                    + config.armijo_coefficient
                    * line_state.step
                    * directional_derivative
                )
                accepted = finite & armijo
                return _LineSearchState(
                    step=jnp.where(
                        accepted,
                        line_state.step,
                        line_state.step * config.backtracking_factor,
                    ),
                    x=candidate_x,
                    value=candidate_value,
                    gradient=candidate_gradient,
                    attempts=line_state.attempts + 1,
                    accepted=accepted,
                )

            line = jax.lax.while_loop(line_condition, line_body, initial_line_state)

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
                update_history = curvature > curvature_floor
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
                s_history = jnp.where(update_history, shifted_s, state.s_history)
                y_history = jnp.where(update_history, shifted_y, state.y_history)
                rho_history = jnp.where(update_history, shifted_rho, state.rho_history)
                history_count = jnp.where(
                    update_history,
                    jnp.minimum(state.history_count + 1, history_size),
                    state.history_count,
                )
                flux, residual = quality_values(line.x)
                feasible = residual <= target.feasibility_tolerance
                target_feasible = feasible & (flux <= target.target_flux)
                lower_feasible_flux = feasible & (
                    (~state.found_feasible) | (flux < state.best_flux)
                )
                best_x = jnp.where(lower_feasible_flux, line.x, state.best_x)
                best_flux = jnp.where(lower_feasible_flux, flux, state.best_flux)
                best_iteration = jnp.where(
                    lower_feasible_flux, state.iteration + 1, state.best_iteration
                )
                lower_residual = (residual < state.best_residual) | (
                    (residual == state.best_residual) & (flux < state.fallback_flux)
                )
                fallback_x = jnp.where(lower_residual, line.x, state.fallback_x)
                best_residual = jnp.where(lower_residual, residual, state.best_residual)
                fallback_flux = jnp.where(lower_residual, flux, state.fallback_flux)
                fallback_iteration = jnp.where(
                    lower_residual,
                    state.iteration + 1,
                    state.fallback_iteration,
                )
                significant = target_feasible & (
                    (~state.target_found)
                    | (
                        flux
                        <= state.significant_flux
                        * (1.0 - target.minimum_relative_improvement)
                    )
                )
                target_found = state.target_found | target_feasible
                significant_flux = jnp.where(significant, flux, state.significant_flux)
                stagnation = jnp.where(
                    ~target_found,
                    0,
                    jnp.where(significant, 0, state.stagnation + 1),
                )
                infeasible = jnp.where(
                    target_found & (~feasible), state.infeasible + 1, 0
                )
                gradient_norm = jnp.linalg.norm(line.gradient, ord=jnp.inf)
                converged = gradient_norm <= config.gradient_tolerance
                target_infeasible_stop = target_found & (
                    infeasible >= target.infeasible_patience
                )
                target_stagnation_stop = target_found & (
                    stagnation >= target.checkpoint_patience
                )
                reason = jnp.where(
                    converged,
                    1,
                    jnp.where(
                        target_infeasible_stop,
                        3,
                        jnp.where(target_stagnation_stop, 2, 0),
                    ),
                ).astype(jnp.int32)
                iteration = state.iteration + 1
                return state._replace(
                    x=line.x,
                    value=line.value,
                    gradient=line.gradient,
                    s_history=s_history,
                    y_history=y_history,
                    rho_history=rho_history,
                    history_count=history_count,
                    iteration=iteration,
                    evaluations=state.evaluations + line.attempts,
                    reason=reason,
                    best_x=best_x,
                    best_flux=best_flux,
                    best_iteration=best_iteration,
                    fallback_x=fallback_x,
                    best_residual=best_residual,
                    fallback_flux=fallback_flux,
                    fallback_iteration=fallback_iteration,
                    found_feasible=state.found_feasible | feasible,
                    target_found=target_found,
                    significant_flux=significant_flux,
                    stagnation=stagnation,
                    infeasible=infeasible,
                    objective_history=state.objective_history.at[iteration].set(
                        line.value
                    ),
                    gradient_history=state.gradient_history.at[iteration].set(
                        gradient_norm
                    ),
                    flux_history=state.flux_history.at[iteration].set(flux),
                    residual_history=state.residual_history.at[iteration].set(residual),
                    feasible_history=state.feasible_history.at[iteration].set(feasible),
                    step_history=state.step_history.at[iteration].set(line.step),
                )

            return jax.lax.cond(line.accepted, accepted, failed, operand=None)

        def solve(initial_x):
            value, gradient = value_and_grad(initial_x)
            flux, residual = quality_values(initial_x)
            feasible = residual <= target.feasibility_tolerance
            target_found = feasible & (flux <= target.target_flux)
            history_shape = (max_iterations + 1,)
            objective_history = (
                jnp.full(history_shape, jnp.nan, dtype=initial_x.dtype).at[0].set(value)
            )
            gradient_history = (
                jnp.full(history_shape, jnp.nan, dtype=initial_x.dtype)
                .at[0]
                .set(jnp.linalg.norm(gradient, ord=jnp.inf))
            )
            flux_history = (
                jnp.full(history_shape, jnp.nan, dtype=initial_x.dtype).at[0].set(flux)
            )
            residual_history = (
                jnp.full(history_shape, jnp.nan, dtype=initial_x.dtype)
                .at[0]
                .set(residual)
            )
            feasible_history = (
                jnp.zeros(history_shape, dtype=jnp.bool_).at[0].set(feasible)
            )
            step_history = (
                jnp.full(history_shape, jnp.nan, dtype=initial_x.dtype).at[0].set(0.0)
            )
            initial = _LBFGSState(
                x=initial_x,
                value=value,
                gradient=gradient,
                s_history=jnp.zeros(
                    (history_size, initial_x.size), dtype=initial_x.dtype
                ),
                y_history=jnp.zeros(
                    (history_size, initial_x.size), dtype=initial_x.dtype
                ),
                rho_history=jnp.zeros((history_size,), dtype=initial_x.dtype),
                history_count=jnp.asarray(0, dtype=jnp.int32),
                iteration=jnp.asarray(0, dtype=jnp.int32),
                evaluations=jnp.asarray(1, dtype=jnp.int32),
                reason=jnp.where(
                    jnp.linalg.norm(gradient, ord=jnp.inf) <= config.gradient_tolerance,
                    1,
                    0,
                ).astype(jnp.int32),
                best_x=initial_x,
                best_flux=jnp.where(feasible, flux, jnp.inf),
                best_iteration=jnp.asarray(0, dtype=jnp.int32),
                fallback_x=initial_x,
                best_residual=residual,
                fallback_flux=flux,
                fallback_iteration=jnp.asarray(0, dtype=jnp.int32),
                found_feasible=feasible,
                target_found=target_found,
                significant_flux=jnp.where(target_found, flux, jnp.inf),
                stagnation=jnp.asarray(0, dtype=jnp.int32),
                infeasible=jnp.asarray(0, dtype=jnp.int32),
                objective_history=objective_history,
                gradient_history=gradient_history,
                flux_history=flux_history,
                residual_history=residual_history,
                feasible_history=feasible_history,
                step_history=step_history,
            )

            def condition(state):
                return (state.iteration < max_iterations) & (state.reason == 0)

            state = jax.lax.while_loop(condition, iteration_body, initial)
            reason = jnp.where(
                (state.reason == 0) & (state.iteration >= max_iterations),
                5,
                state.reason,
            )
            selected_x = jnp.where(state.found_feasible, state.best_x, state.fallback_x)
            selected_iteration = jnp.where(
                state.found_feasible,
                state.best_iteration,
                state.fallback_iteration,
            )
            selected_flux, selected_residual = quality_values(selected_x)
            return (
                state._replace(reason=reason),
                selected_x,
                selected_iteration,
                selected_flux,
                selected_residual,
            )

        return solve

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

    def _coerce(self, x):
        x = np.asarray(x, dtype=self._dtype)
        if x.shape != self._shape:
            raise ValueError(f"x must have shape {self._shape}, got {x.shape}")
        return x

    def compile(self, example_x):
        """Compile the whole optimizer without executing its iteration loop."""
        x = self._device_put(self._coerce(example_x))
        self._compiled = self._solver.lower(x).compile()
        return self

    def run(self, initial_x):
        """Execute one compiled optimizer call and transfer its result to host."""
        x = self._device_put(self._coerce(initial_x))
        if self._compiled is None:
            self.compile(initial_x)
        (
            raw_state,
            selected_x,
            selected_iteration,
            selected_flux,
            selected_residual,
        ) = self._compiled(x)
        jax.block_until_ready(
            (
                raw_state,
                selected_x,
                selected_iteration,
                selected_flux,
                selected_residual,
            )
        )
        iterations = int(raw_state.iteration)
        history_slice = slice(0, iterations + 1)
        reason = int(raw_state.reason)
        return DeviceLBFGSResult(
            x=np.asarray(selected_x),
            terminal_x=np.asarray(raw_state.x),
            fun=float(raw_state.value),
            jac=np.asarray(raw_state.gradient),
            quadratic_flux=float(selected_flux),
            target_envelope_residual_norm_infinity=float(selected_residual),
            success=reason == 1,
            status=reason,
            message=_STATUS_MESSAGES.get(reason, "unknown termination"),
            iterations=iterations,
            evaluations=int(raw_state.evaluations),
            found_feasible_checkpoint=bool(raw_state.found_feasible),
            target_found=bool(raw_state.target_found),
            selected_iteration=int(selected_iteration),
            objective_history=np.asarray(raw_state.objective_history)[history_slice],
            gradient_norm_infinity_history=np.asarray(raw_state.gradient_history)[
                history_slice
            ],
            quadratic_flux_history=np.asarray(raw_state.flux_history)[history_slice],
            target_residual_norm_infinity_history=np.asarray(
                raw_state.residual_history
            )[history_slice],
            target_feasible_history=np.asarray(raw_state.feasible_history)[
                history_slice
            ],
            step_size_history=np.asarray(raw_state.step_history)[history_slice],
        )
