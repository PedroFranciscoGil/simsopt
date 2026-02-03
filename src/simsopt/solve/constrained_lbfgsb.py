"""
Constrained L-BFGS-B optimizer with hard constraint support during line search.

This module provides a custom L-BFGS-B implementation that supports hard constraints
that must be satisfied at every iteration. Unlike penalty-based approaches, hard
constraints are NOT included in the objective function - they only determine whether
a proposed step is acceptable.

**Key feature**: Uses scipy's line_search_wolfe2 with the extra_condition parameter
to properly enforce hard constraints. The extra_condition is called only for steps
that already satisfy the strong Wolfe conditions, and if it returns False (constraint
violated), the line search continues to find another step. This is the correct way
to add additional acceptance criteria to a Wolfe line search.

This is particularly useful for topological constraints like LinkingNumber, where:
1. The constraint value is discrete (0, ±1, ±2, ...)
2. The gradient is zero (no useful direction information)
3. Violations should be avoided entirely, not penalized

Example usage:
    >>> optimizer = ConstrainedLBFGSB(
    ...     fun=objective_and_grad,
    ...     x0=initial_dofs,
    ...     hard_constraints=[linking_number_obj],
    ...     feasibility_check=lambda hcs: all(abs(hc.J()) < 0.5 for hc in hcs)
    ... )
    >>> result = optimizer.minimize()
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Any, Union
from scipy.optimize import line_search

__all__ = ['ConstrainedLBFGSB', 'ConstrainedLBFGSBResult', 'minimize_with_hard_constraints']


@dataclass
class ConstrainedLBFGSBResult:
    """Result object from ConstrainedLBFGSB optimization.
    
    Attributes:
        x: Final optimized parameters
        fun: Final objective value
        jac: Final gradient
        nit: Number of iterations
        nfev: Number of function evaluations
        njev: Number of Jacobian evaluations
        success: Whether optimization converged successfully
        message: Description of termination reason
        n_constraint_rejections: Number of steps rejected due to hard constraint violations
        constraint_rejection_history: List of (iteration, alpha) tuples where rejections occurred
    """
    x: np.ndarray
    fun: float
    jac: np.ndarray
    nit: int
    nfev: int
    njev: int
    success: bool
    message: str
    n_constraint_rejections: int
    constraint_rejection_history: List[Tuple[int, float]]


class ConstrainedLBFGSB:
    """
    L-BFGS-B optimizer with support for hard constraints during line search.
    
    Uses scipy's line_search_wolfe2 with the extra_condition parameter to properly
    enforce hard constraints. The extra_condition is called only for steps that
    already satisfy the strong Wolfe conditions, and if it returns False (constraint
    violated), the line search continues to find another step.
    
    Key differences from penalty-based constraint handling:
    1. Hard constraints are NOT in the objective - no gradient contribution
    2. L-BFGS Hessian approximation is only updated with feasible points
    3. The optimizer "sees" a well-conditioned problem
    4. Identical to scipy L-BFGS-B when no hard constraints are active
    
    Parameters
    ----------
    fun : callable
        Objective function. If jac=True, should return (f, g) tuple.
        Signature: fun(x) -> float or fun(x) -> (float, ndarray)
    x0 : ndarray
        Initial guess for parameters.
    jac : bool or callable, optional
        If True, fun returns (f, g). If callable, jac(x) returns gradient.
        Default is True.
    bounds : sequence of (min, max) pairs, optional
        Bounds for each parameter. None means unbounded.
    hard_constraints : list, optional
        List of constraint objects with .J() method. A point is feasible
        if all constraints evaluate to acceptable values.
    feasibility_check : callable, optional
        Function that takes the list of hard_constraints and returns True
        if the current point is feasible. Default checks if all constraints
        have J() values with absolute value < 0.5 (for integer constraints like LinkingNumber).
    maxiter : int, optional
        Maximum number of iterations. Default is 100.
        Note: scipy's L-BFGS-B default is 15000. We use a smaller default since
        this optimizer is typically used within outer loops (e.g., augmented Lagrangian).
    maxcor : int, optional
        Maximum number of variable metric corrections (L-BFGS memory).
        Default is 10. [Matches scipy L-BFGS-B default]
    ftol : float, optional
        Function tolerance for convergence. Iteration stops when
        ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
        Default is 2.220446049250313e-09. [Matches scipy L-BFGS-B default]
    gtol : float, optional
        Gradient tolerance for convergence. Iteration stops when
        ``max{|proj g_i|} <= gtol`` where proj g_i is the i-th component
        of the projected gradient.
        Default is 1e-5. [Matches scipy L-BFGS-B default]
    maxls : int, optional
        Maximum number of line search steps per iteration.
        Default is 20. [Matches scipy L-BFGS-B default]
    c1 : float, optional
        Armijo (sufficient decrease) condition parameter for line search.
        Default is 1e-4. [Standard value for quasi-Newton methods]
    c2 : float, optional
        Curvature condition parameter for line search.
        Default is 0.9. [Standard value for quasi-Newton methods]
    alpha_init : float, optional
        Initial step size for line search. Default is 1.0.
    alpha_min : float, optional
        Minimum step size before line search gives up. Default is 1e-12.
    verbose : int, optional
        Verbosity level (0=silent, 1=summary, 2=detailed). Default is 0.
    callback : callable, optional
        Called after each iteration: callback(xk).
    
    Notes
    -----
    The following parameters match scipy.optimize.minimize(method='L-BFGS-B') defaults:
    maxcor=10, ftol=2.220446049250313e-09, gtol=1e-5, maxls=20.
    
    The maxiter default (100) differs from scipy's default (15000) because this
    optimizer is designed for use within outer iteration loops where fewer
    inner iterations are typically needed.
    
    See Also
    --------
    scipy.optimize.minimize : General minimization interface
    minimize_with_hard_constraints : Convenience function with scipy-like interface
    """
    
    def __init__(
        self,
        fun: Callable,
        x0: np.ndarray,
        jac: Union[bool, Callable] = True,
        bounds: Optional[List[Tuple[float, float]]] = None,
        hard_constraints: Optional[List[Any]] = None,
        feasibility_check: Optional[Callable] = None,
        objective: Optional[Any] = None,
        maxiter: int = 100,
        maxcor: int = 10,
        ftol: float = 2.220446049250313e-09,  # Matches scipy L-BFGS-B default
        gtol: float = 1e-5,
        maxls: int = 20,
        c1: float = 1e-4,
        c2: float = 0.9,
        alpha_init: float = 1.0,
        alpha_min: float = 1e-12,
        verbose: int = 0,
        callback: Optional[Callable] = None,
    ):
        self.fun = fun
        self.x0 = np.asarray(x0, dtype=np.float64).copy()
        self.n = len(self.x0)
        self.jac = jac
        self.bounds = bounds
        self.hard_constraints = hard_constraints or []
        self.objective = objective  # Optional simsopt Optimizable for DOF updates
        self.maxiter = maxiter
        self.maxcor = maxcor
        self.ftol = ftol
        self.gtol = gtol
        self.maxls = maxls
        self.c1 = c1
        self.c2 = c2
        self.alpha_init = alpha_init
        self.alpha_min = alpha_min
        self.verbose = verbose
        self.callback = callback
        
        # Default feasibility check: all constraint values must be < 0.5 in absolute value
        # This works for integer constraints like LinkingNumber where 0 = feasible, ±1 = infeasible
        if feasibility_check is None:
            self.feasibility_check = lambda hcs: all(abs(hc.J()) < 0.5 for hc in hcs)
        else:
            self.feasibility_check = feasibility_check
        
        # Parse bounds
        if bounds is not None:
            self.lower = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
            self.upper = np.array([b[1] if b[1] is not None else np.inf for b in bounds])
        else:
            self.lower = np.full(self.n, -np.inf)
            self.upper = np.full(self.n, np.inf)
        
        # Counters and history
        self.nfev = 0
        self.njev = 0
        self.n_constraint_rejections = 0
        self.constraint_rejection_history = []
        
        # L-BFGS memory storage
        self.S = []  # s_k = x_{k+1} - x_k
        self.Y = []  # y_k = g_{k+1} - g_k
        self.rho = []  # 1 / (y_k^T s_k)
    
    def _evaluate(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluate objective and gradient at x (uses cache)."""
        # Use cached evaluation functions which handle caching
        f = self._eval_f(x)
        g = self._eval_g(x)
        return f, g
    
    def _numerical_gradient(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Compute gradient via finite differences."""
        g = np.zeros(self.n)
        f0 = self.fun(x)
        for i in range(self.n):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = self.fun(x_plus)
            g[i] = (f_plus - f0) / eps
            self.nfev += 1
        return g
    
    def _project(self, x: np.ndarray) -> np.ndarray:
        """Project x onto the box constraints."""
        return np.clip(x, self.lower, self.upper)
    
    def _is_feasible(self, x: np.ndarray) -> bool:
        """Check if x satisfies all hard constraints.
        
        NOTE: This should be called AFTER the objective function has been evaluated
        at x, which ensures DOFs are properly updated via simsopt's Optimizable graph.
        The constraint objects share the same underlying Optimizables (curves, 
        surfaces, etc.) as the objective.
        
        If an objective Optimizable was provided and DOFs haven't been set yet,
        this method will set them via objective.x = x.
        """
        if not self.hard_constraints:
            return True
        
        # If we have the objective Optimizable, ensure DOFs are set.
        # This is a safety measure in case the objective hasn't been evaluated yet.
        if self.objective is not None:
            self.objective.x = x
        
        return self.feasibility_check(self.hard_constraints)
    
    def _lbfgs_direction(self, g: np.ndarray) -> np.ndarray:
        """
        Compute L-BFGS search direction using two-loop recursion.
        
        Returns -H_k * g where H_k is the inverse Hessian approximation.
        """
        q = g.copy()
        m = len(self.S)
        
        if m == 0:
            # Initial iteration: use steepest descent
            return -g
        
        alpha = np.zeros(m)
        
        # First loop (backward)
        for i in range(m - 1, -1, -1):
            alpha[i] = self.rho[i] * np.dot(self.S[i], q)
            q = q - alpha[i] * self.Y[i]
        
        # Compute H_0 * q (initial Hessian approximation)
        # Use scaled identity: H_0 = (s^T y / y^T y) * I
        s_last = self.S[-1]
        y_last = self.Y[-1]
        gamma = np.dot(s_last, y_last) / (np.dot(y_last, y_last) + 1e-12)
        r = gamma * q
        
        # Second loop (forward)
        for i in range(m):
            beta = self.rho[i] * np.dot(self.Y[i], r)
            r = r + (alpha[i] - beta) * self.S[i]
        
        return -r
    
    def _update_lbfgs_memory(self, s: np.ndarray, y: np.ndarray):
        """Update L-BFGS memory with new s, y pair."""
        ys = np.dot(y, s)
        
        # Skip update if curvature condition not satisfied
        if ys <= 1e-12 * np.dot(y, y):
            if self.verbose >= 2:
                print(f"  Skipping L-BFGS update: y^T s = {ys:.2e}")
            return
        
        if len(self.S) >= self.maxcor:
            self.S.pop(0)
            self.Y.pop(0)
            self.rho.pop(0)
        
        self.S.append(s.copy())
        self.Y.append(y.copy())
        self.rho.append(1.0 / ys)
    
    def _line_search_with_feasibility(
        self, 
        x: np.ndarray, 
        f: float, 
        g: np.ndarray, 
        d: np.ndarray,
        iteration: int,
        old_old_fval: Optional[float] = None
    ) -> Tuple[float, np.ndarray, float, np.ndarray, bool, int, float]:
        """
        Wolfe line search with feasibility checking via extra_condition.
        
        Uses scipy's line_search with extra_condition callback to reject
        infeasible steps. The objective is evaluated at all trial points
        (including infeasible ones), but only feasible steps satisfying
        Wolfe conditions are accepted.
        
        Returns:
            alpha: Accepted step size
            x_new: New iterate
            f_new: Function value at x_new
            g_new: Gradient at x_new
            success: Whether a valid step was found
            n_rejected: Number of steps rejected due to constraint violation
            old_fval: Function value for use in next iteration
        """
        descent = np.dot(g, d)
        
        if descent >= 0:
            if self.verbose >= 2:
                print(f"  Warning: Not a descent direction (g^T d = {descent:.2e}), using -g")
            d = -g.copy()
            descent = -np.dot(g, g)
        
        n_rejected = 0
        
        # Feasibility check via extra_condition
        # Called only for steps satisfying Wolfe conditions
        def feasibility_condition(alpha_trial, x_trial, f_trial, g_trial):
            nonlocal n_rejected
            if not self.hard_constraints:
                return True
            
            # Check feasibility
            is_feasible = self._is_feasible(x_trial)
            
            if not is_feasible:
                n_rejected += 1
                self.constraint_rejection_history.append((iteration, alpha_trial))
                if self.verbose >= 2:
                    print(f"  Wolfe step alpha={alpha_trial:.2e} REJECTED (infeasible)")
            
            return is_feasible
        
        # Try scipy's Wolfe line search with feasibility via extra_condition
        try:
            result = line_search(
                f=lambda x_trial: self._eval_f(x_trial),
                myfprime=lambda x_trial: self._eval_g(x_trial),
                xk=x,
                pk=d,
                gfk=g,
                old_fval=f,
                old_old_fval=old_old_fval,
                c1=self.c1,
                c2=self.c2,
                maxiter=self.maxls,
                extra_condition=feasibility_condition if self.hard_constraints else None
            )
            alpha_wolfe, fc, gc, f_new, old_fval, g_new = result
            # Note: nfev/njev already updated by _eval_f/_eval_g
        except Exception as e:
            if self.verbose >= 1:
                print(f"  Line search exception: {e}")
            alpha_wolfe = None
        
        # If Wolfe line search succeeded
        if alpha_wolfe is not None and alpha_wolfe > 0:
            x_new = self._project(x + alpha_wolfe * d)
            
            # Re-evaluate to get correct f, g (scipy may have cached values)
            f_new, g_new = self._evaluate(x_new)
            
            if self.verbose >= 2:
                print(f"  Wolfe step alpha={alpha_wolfe:.2e}, f={f_new:.6e} ACCEPTED")
            return alpha_wolfe, x_new, f_new, g_new, True, n_rejected, old_fval if old_fval else f
        
        # Fallback to Armijo backtracking if Wolfe fails
        if self.verbose >= 2:
            print(f"  Wolfe line search failed, trying Armijo backtracking...")
        
        alpha, x_new, f_new, g_new, success, n_rej = self._armijo_backtracking(
            x, f, g, d, descent, iteration
        )
        n_rejected += n_rej
        
        return alpha, x_new, f_new, g_new, success, n_rejected, f
    
    def _armijo_backtracking(
        self,
        x: np.ndarray,
        f: float,
        g: np.ndarray,
        d: np.ndarray,
        descent: float,
        iteration: int
    ) -> Tuple[float, np.ndarray, float, np.ndarray, bool, int]:
        """
        Armijo backtracking line search with feasibility checking and interpolation.
        
        Uses quadratic interpolation to find better step sizes. Also tracks
        the best feasible point found in case Armijo can't be satisfied.
        
        Returns:
            alpha, x_new, f_new, g_new, success, n_rejected
        """
        n_rejected = 0
        
        # Track best feasible point
        best_alpha = None
        best_f = f
        best_g = None
        best_x = None
        
        # For interpolation
        alpha_prev = None
        f_prev = None
        
        alpha = self.alpha_init
        
        for ls_iter in range(self.maxls):
            if alpha < self.alpha_min:
                break
            
            x_trial = self._project(x + alpha * d)
            f_trial, g_trial = self._evaluate(x_trial)
            
            # Check feasibility
            if self.hard_constraints and not self._is_feasible(x_trial):
                n_rejected += 1
                self.constraint_rejection_history.append((iteration, alpha))
                if self.verbose >= 2:
                    print(f"  Armijo alpha={alpha:.2e}, f={f_trial:.6e} REJECTED (infeasible)")
                alpha *= 0.25  # Aggressive backtrack for infeasible
                continue
            
            # Track best feasible point
            if f_trial < best_f:
                best_alpha = alpha
                best_f = f_trial
                best_g = g_trial
                best_x = x_trial
            
            # Check Armijo sufficient decrease condition
            armijo_bound = f + self.c1 * alpha * descent
            if f_trial <= armijo_bound:
                if self.verbose >= 2:
                    print(f"  Armijo alpha={alpha:.2e}, f={f_trial:.6e} ACCEPTED")
                return alpha, x_trial, f_trial, g_trial, True, n_rejected
            
            if self.verbose >= 2:
                print(f"  Armijo alpha={alpha:.2e}, f={f_trial:.6e} > bound={armijo_bound:.6e}")
            
            # Try quadratic interpolation if we have two feasible points
            alpha_next = None
            if alpha_prev is not None and f_prev is not None:
                # Quadratic fit through (0,f), (alpha_prev, f_prev), (alpha, f_trial)
                # phi(a) = f + a*descent + 0.5*c*a^2
                # Solve for c using the two nonzero points
                denom = alpha_prev * alpha * (alpha_prev - alpha)
                if abs(denom) > 1e-12:
                    a_coef = (alpha * (f_prev - f - descent * alpha_prev) - 
                              alpha_prev * (f_trial - f - descent * alpha)) / denom
                    if a_coef > 1e-12:  # Convex
                        alpha_interp = -descent / (2 * a_coef)
                        # Safeguard
                        alpha_lo = 0.1 * min(alpha, alpha_prev)
                        alpha_hi = 0.9 * max(alpha, alpha_prev)
                        alpha_interp = max(alpha_lo, min(alpha_hi, alpha_interp))
                        if alpha_interp > self.alpha_min:
                            alpha_next = alpha_interp
            
            alpha_prev = alpha
            f_prev = f_trial
            
            if alpha_next is not None:
                alpha = alpha_next
            else:
                alpha *= 0.5  # Default backtracking
        
        # Accept best feasible decrease even without Armijo (avoid oscillation)
        if best_alpha is not None and best_f < f:
            if self.verbose >= 2:
                print(f"  Accepting best feasible: alpha={best_alpha:.2e}, f={best_f:.6e}")
            return best_alpha, best_x, best_f, best_g, True, n_rejected
        
        return 0.0, x, f, g, False, n_rejected
    
    def _eval_f(self, x: np.ndarray) -> float:
        """Evaluate objective function only (with caching to avoid duplicate calls)."""
        x_tuple = tuple(x)
        if not hasattr(self, '_eval_cache') or self._eval_cache_x != x_tuple:
            if self.jac is True:
                result = self.fun(x)
                self._eval_cache = (float(result[0]), np.asarray(result[1], dtype=np.float64))
            elif callable(self.jac):
                self._eval_cache = (float(self.fun(x)), None)  # Gradient computed separately
            else:
                self._eval_cache = (float(self.fun(x)), None)  # Numerical gradient
            self._eval_cache_x = x_tuple
            self.nfev += 1
        return self._eval_cache[0]
    
    def _eval_g(self, x: np.ndarray) -> np.ndarray:
        """Evaluate gradient only (with caching to avoid duplicate calls)."""
        x_tuple = tuple(x)
        
        # First ensure f is evaluated (this also ensures DOFs are set)
        _ = self._eval_f(x)
        
        if self._eval_cache[1] is None:
            # Need gradient - compute it based on jac type
            if callable(self.jac):
                g = np.asarray(self.jac(x), dtype=np.float64)
                self.njev += 1
            else:
                # Numerical gradient
                g = self._numerical_gradient(x)
            self._eval_cache = (self._eval_cache[0], g)
        
        return self._eval_cache[1]
    
    def minimize(self) -> ConstrainedLBFGSBResult:
        """
        Run the constrained L-BFGS-B optimization.
        
        Returns:
            ConstrainedLBFGSBResult with optimization results.
        """
        x = self._project(self.x0.copy())
        
        # Evaluate objective first to set DOFs via simsopt's Optimizable graph
        f, g = self._evaluate(x)
        
        # Check initial feasibility AFTER evaluating objective (DOFs now set)
        if not self._is_feasible(x):
            return ConstrainedLBFGSBResult(
                x=x,
                fun=np.inf,
                jac=np.zeros(self.n),
                nit=0,
                nfev=self.nfev,
                njev=self.njev,
                success=False,
                message="Initial point is infeasible",
                n_constraint_rejections=0,
                constraint_rejection_history=[]
            )
        g_norm = np.linalg.norm(g, ord=np.inf)
        
        if self.verbose >= 1:
            print(f"Initial: f={f:.6e}, ||g||_inf={g_norm:.2e}")
        
        # Check initial convergence
        if g_norm <= self.gtol:
            return ConstrainedLBFGSBResult(
                x=x,
                fun=f,
                jac=g,
                nit=0,
                nfev=self.nfev,
                njev=self.njev,
                success=True,
                message="Initial point satisfies gradient tolerance",
                n_constraint_rejections=0,
                constraint_rejection_history=[]
            )
        
        f_prev = f
        old_old_fval = None  # For scipy line search
        
        for iteration in range(1, self.maxiter + 1):
            # Compute search direction
            d = self._lbfgs_direction(g)
            
            # Line search with feasibility checking (using scipy's line search)
            alpha, x_new, f_new, g_new, ls_success, n_rejected, old_fval = \
                self._line_search_with_feasibility(x, f, g, d, iteration, old_old_fval)
            
            self.n_constraint_rejections += n_rejected
            
            if not ls_success:
                # Try steepest descent as fallback
                d = -g.copy()
                alpha, x_new, f_new, g_new, ls_success, n_rejected, old_fval = \
                    self._line_search_with_feasibility(x, f, g, d, iteration, old_old_fval)
                self.n_constraint_rejections += n_rejected
                
                if not ls_success:
                    return ConstrainedLBFGSBResult(
                        x=x,
                        fun=f,
                        jac=g,
                        nit=iteration,
                        nfev=self.nfev,
                        njev=self.njev,
                        success=False,
                        message=f"Line search failed at iteration {iteration}",
                        n_constraint_rejections=self.n_constraint_rejections,
                        constraint_rejection_history=self.constraint_rejection_history
                    )
            
            # Update L-BFGS memory (only with feasible points and meaningful steps)
            s = x_new - x
            y = g_new - g
            
            # Track objective history for oscillation detection
            if not hasattr(self, '_f_history'):
                self._f_history = []
            self._f_history.append(f_new)
            if len(self._f_history) > 10:
                self._f_history.pop(0)
            
            # Detect oscillation: objective bouncing between similar values
            if len(self._f_history) >= 4:
                recent = self._f_history[-4:]
                # Check if alternating: f1 ≈ f3, f2 ≈ f4, but f1 != f2
                f1, f2, f3, f4 = recent
                tol = 1e-6 * max(abs(f1), abs(f2), 1.0)
                oscillating = (abs(f1 - f3) < tol and abs(f2 - f4) < tol and 
                              abs(f1 - f2) > tol)
                if oscillating:
                    oscillation_count = getattr(self, '_oscillation_count', 0) + 1
                    self._oscillation_count = oscillation_count
                    if self.verbose >= 1:
                        print(f"  Warning: Oscillation detected (count={oscillation_count})")
                    
                    # Reset L-BFGS memory to try to escape
                    if oscillation_count >= 2:
                        if self.verbose >= 1:
                            print(f"  Resetting L-BFGS memory due to oscillation")
                        self.S = []
                        self.Y = []
                        self.rho = []
                        self._oscillation_count = 0
                        self._f_history = [f_new]  # Reset history too
                else:
                    self._oscillation_count = 0
            
            # Track if we're stuck at constraint boundary
            if alpha < 1e-6 and n_rejected > 0:
                small_step_count = getattr(self, '_small_step_count', 0) + 1
                self._small_step_count = small_step_count
                
                if self.verbose >= 2:
                    print(f"  Warning: Very small step (alpha={alpha:.2e}) with rejections. "
                          f"Likely at constraint boundary. Count: {small_step_count}")
                
                # Reset L-BFGS memory to try fresh directions
                if small_step_count >= 3:
                    if self.verbose >= 1:
                        print(f"  Resetting L-BFGS memory after {small_step_count} small steps")
                    self.S = []
                    self.Y = []
                    self.rho = []
                    self._small_step_count = 0
                
                # If still stuck after multiple resets, we're at a constrained local minimum
                if small_step_count >= 10:
                    return ConstrainedLBFGSBResult(
                        x=x_new,
                        fun=f_new,
                        jac=g_new,
                        nit=iteration,
                        nfev=self.nfev,
                        njev=self.njev,
                        success=True,
                        message="Converged at constraint boundary (no feasible descent)",
                        n_constraint_rejections=self.n_constraint_rejections,
                        constraint_rejection_history=self.constraint_rejection_history
                    )
            else:
                self._small_step_count = 0
            
            self._update_lbfgs_memory(s, y)
            
            # Update iterate
            x = x_new
            g = g_new
            old_old_fval = f_prev  # Store for next iteration's line search
            f_prev = f
            f = f_new
            
            g_norm = np.linalg.norm(g, ord=np.inf)
            
            if self.verbose >= 1:
                print(f"Iter {iteration}: f={f:.6e}, ||g||_inf={g_norm:.2e}, "
                      f"alpha={alpha:.2e}, rejections={n_rejected}")
            
            if self.callback is not None:
                self.callback(x)
            
            # Check convergence
            if g_norm <= self.gtol:
                return ConstrainedLBFGSBResult(
                    x=x,
                    fun=f,
                    jac=g,
                    nit=iteration,
                    nfev=self.nfev,
                    njev=self.njev,
                    success=True,
                    message="Gradient tolerance reached",
                    n_constraint_rejections=self.n_constraint_rejections,
                    constraint_rejection_history=self.constraint_rejection_history
                )
            
            if abs(f_prev - f) <= self.ftol * max(abs(f), abs(f_prev), 1.0):
                return ConstrainedLBFGSBResult(
                    x=x,
                    fun=f,
                    jac=g,
                    nit=iteration,
                    nfev=self.nfev,
                    njev=self.njev,
                    success=True,
                    message="Function tolerance reached",
                    n_constraint_rejections=self.n_constraint_rejections,
                    constraint_rejection_history=self.constraint_rejection_history
                )
        
        return ConstrainedLBFGSBResult(
            x=x,
            fun=f,
            jac=g,
            nit=self.maxiter,
            nfev=self.nfev,
            njev=self.njev,
            success=False,
            message="Maximum iterations reached",
            n_constraint_rejections=self.n_constraint_rejections,
            constraint_rejection_history=self.constraint_rejection_history
        )


def minimize_with_hard_constraints(
    fun: Callable,
    x0: np.ndarray,
    hard_constraints: Optional[List[Any]] = None,
    feasibility_check: Optional[Callable] = None,
    objective: Optional[Any] = None,
    jac: Union[bool, Callable] = True,
    bounds: Optional[List[Tuple[float, float]]] = None,
    options: Optional[dict] = None,
) -> ConstrainedLBFGSBResult:
    """
    Minimize a function subject to hard constraints using constrained L-BFGS-B.
    
    This is a convenience function that creates a ConstrainedLBFGSB optimizer
    and runs it. It provides an interface similar to scipy.optimize.minimize.
    
    Parameters
    ----------
    fun : callable
        Objective function. If jac=True, should return (f, g) tuple.
    x0 : ndarray
        Initial guess for parameters.
    hard_constraints : list, optional
        List of constraint objects with .J() method.
    feasibility_check : callable, optional
        Function that takes the list of hard_constraints and returns True if feasible.
    objective : Optimizable, optional
        The simsopt Optimizable object for the objective function. If provided,
        enables efficient feasibility pre-checking by updating DOFs via objective.x = x
        before checking constraints. This avoids calling the user's fun() for
        infeasible points, preventing unnecessary side effects (like printing).
    jac : bool or callable, optional
        If True, fun returns (f, g). If callable, jac(x) returns gradient. Default True.
    bounds : sequence of (min, max) pairs, optional
        Bounds for each parameter.
    options : dict, optional
        Additional options passed to ConstrainedLBFGSB. Supported keys:
        
        - maxiter (int): Maximum iterations. Default 100.
        - maxcor (int): L-BFGS memory size. Default 10. [scipy default]
        - ftol (float): Function tolerance. Default 2.220446049250313e-09. [scipy default]
        - gtol (float): Gradient tolerance. Default 1e-5. [scipy default]
        - maxls (int): Max line search steps. Default 20. [scipy default]
        - c1 (float): Armijo parameter. Default 1e-4.
        - c2 (float): Curvature parameter. Default 0.9.
        - alpha_init (float): Initial step size. Default 1.0.
        - alpha_min (float): Minimum step size. Default 1e-12.
        - verbose (int): Verbosity level. Default 0.
        - callback (callable): Called after each iteration.
    
    Returns
    -------
    ConstrainedLBFGSBResult
        Optimization result.
    
    Notes
    -----
    Default parameters (maxcor, ftol, gtol, maxls) match scipy.optimize.minimize
    with method='L-BFGS-B', except maxiter which defaults to 100 instead of 15000.
    
    Example
    -------
    >>> def rosenbrock_with_grad(x):
    ...     f = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    ...     g = np.array([
    ...         -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
    ...         200*(x[1] - x[0]**2)
    ...     ])
    ...     return f, g
    >>> result = minimize_with_hard_constraints(rosenbrock_with_grad, np.array([0.0, 0.0]))
    >>> print(result.x)  # Should be close to [1, 1]
    """
    opts = options or {}
    
    optimizer = ConstrainedLBFGSB(
        fun=fun,
        x0=x0,
        jac=jac,
        bounds=bounds,
        hard_constraints=hard_constraints,
        feasibility_check=feasibility_check,
        objective=objective,
        maxiter=opts.get('maxiter', 100),
        maxcor=opts.get('maxcor', 100),
        ftol=opts.get('ftol', 2.220446049250313e-09),  # scipy default
        gtol=opts.get('gtol', 1e-5),
        maxls=opts.get('maxls', 20),
        c1=opts.get('c1', 1e-4),
        c2=opts.get('c2', 0.9),
        alpha_init=opts.get('alpha_init', 1.0),
        alpha_min=opts.get('alpha_min', 1e-12),
        verbose=opts.get('verbose', 0),
        callback=opts.get('callback', None),
    )
    
    return optimizer.minimize()
