import numpy as np
from scipy.optimize import minimize
from simsopt.geo import curves_to_vtk
from simsopt.objectives import SquaredFlux
from threadpoolctl import threadpool_limits

__all__ = ['augmented_lagrangian_objective', 
           'grad_augmented_lagrangian', 'augmented_lagrangian_method',
]

class dummyObjective:
    def __init__(self, x):
        self.x = x
    def J(self):
        return 0.0
    def dJ(self):
        return np.zeros_like(self.x)


def _safe_call_J(obj, use_threadlimit=True):
    """Helper to safely call obj.J() with threadpool limits."""
    if use_threadlimit:
        try:
            with threadpool_limits(limits=1, user_api='openmp'):
                return float(obj.J())
        except Exception:
            return float(obj.J())
    else:
        return float(obj.J())


def _safe_call_dJ(obj, use_threadlimit=True):
    """Helper to safely call obj.dJ() with threadpool limits."""
    if use_threadlimit:
        try:
            with threadpool_limits(limits=1, user_api='openmp'):
                return np.asarray(obj.dJ(), dtype=np.float64)
        except Exception:
            return np.asarray(obj.dJ(), dtype=np.float64)
    else:
        return np.asarray(obj.dJ(), dtype=np.float64)


def _eval_constraints_and_jacobian(equality_constraints, dofs, use_threadlimit=True):
    """
    Evaluate constraint values and Jacobian matrix together to avoid redundant evaluations.
    Optimized with pre-computed offsets and in-place operations.
    
    Returns:
        c_vals: array of constraint values (m,)
        jac: Jacobian matrix (m, n)
    """
    m = len(equality_constraints)
    n = len(dofs)
    
    if m == 0:
        return np.empty(0, dtype=np.float64), np.zeros((0, n), dtype=np.float64)
    
    c_vals = np.empty(m, dtype=np.float64)
    jac = np.zeros((m, n), dtype=np.float64)
    
    # Pre-compute constraint x offsets to avoid repeated len() calls
    constraint_sizes = np.array([len(c_i.x) for c_i in equality_constraints], dtype=int)
    dof_offsets = n - constraint_sizes
    
    # Set constraint x values once
    for i, c_i in enumerate(equality_constraints):
        c_i.x = dofs[dof_offsets[i]:]
    
    # Evaluate constraints and gradients together
    for i, c_i in enumerate(equality_constraints):
        c_vals[i] = _safe_call_J(c_i, use_threadlimit)
        grad_ci = _safe_call_dJ(c_i, use_threadlimit)
        # Use pre-computed offset
        dof_dif = dof_offsets[i]
        grad_ci_flat = np.atleast_1d(grad_ci).flatten()
        grad_len = len(grad_ci_flat)
        jac[i, dof_dif:dof_dif + grad_len] = grad_ci_flat
    
    return c_vals, jac


def jac_constraint(constraint_list, dofs):
    """
    Compute the Jacobian matrix of a list of constraint functions with respect to the coil degrees of freedom.

    Args:
        constraint_list (list): List of constraint objects, each with .x and .dJ() methods.
        dofs (np.ndarray): The full vector of coil degrees of freedom.

    Returns:
        np.ndarray: Jacobian matrix of shape (len(constraint_list), len(dofs)), where each row is the gradient of a constraint.
    """
    _, jac = _eval_constraints_and_jacobian(constraint_list, dofs)
    return jac


def augmented_lagrangian_objective(dofs, f, equality_constraints, lag_mul, mu):
    """
    Compute the value of the augmented Lagrangian for the current optimization variables.

    The standard augmented Lagrangian is:

        .. math::

            L_A(x, \lambda, \mu) = f(x) - \lambda^T g(x) + \frac{\mu}{2} \|g(x)\|^2

    where:
        - :math:`f(x)`: objective function (e.g., squared flux)
        - :math:`g(x)`: vector of constraint functions
        - :math:`\lambda`: vector of Lagrange multipliers
        - :math:`\mu`: penalty parameter

    Args:
        f: Objective function object (with .J() and .x attributes).
        dofs (np.ndarray): Current degrees of freedom.
        equality_constraints (list): List of constraint objects (with .J() and .dJ() methods).
        lag_mul (np.ndarray): Lagrange multipliers.
        mu (float or np.ndarray): Penalty parameter (can be vector).

    Returns:
        float: Value of the augmented Lagrangian.
    """
    f.x = dofs
    if isinstance(f, dummyObjective):
        f_val = f.J()
    else:
        f_val = _safe_call_J(f)
    
    # Evaluate constraints
    c_vals, _ = _eval_constraints_and_jacobian(equality_constraints, dofs)
    
    # Build Lagrangian with mu (can be vector) - optimized with in-place operations
    c_vals_sq = c_vals * c_vals  # Reuse for both terms
    L = f_val - np.dot(lag_mul, c_vals) + 0.5 * np.dot(mu, c_vals_sq)
    
    return L


def grad_augmented_lagrangian(dofs, f, equality_constraints, lag_mul, mu):
    """
    Compute the gradient of the augmented Lagrangian with respect to the optimization variables.

    For the standard form:

        .. math::

            \nabla_x L_A(x, \lambda, \mu) = \nabla f(x) - \sum_i \lambda_i \nabla g_i(x) + \mu J_g^T g(x)

    where:
        - :math:`J_g`: Jacobian matrix of constraints (rows: constraints, columns: dofs)
        - :math:`g(x)`: vector of constraint values

    Args:
        dofs (np.ndarray): Current degrees of freedom.
        f: Objective function object (with .J() and .dJ() methods).
        equality_constraints (list): List of constraint objects.
        lag_mul (np.ndarray): Lagrange multipliers.
        mu (float or np.ndarray): Penalty parameter (can be vector).

    Returns:
        np.ndarray: Gradient of the augmented Lagrangian.
    """
    f.x = dofs
    if isinstance(f, dummyObjective):
        grad_f = f.dJ()
    else:
        grad_f = _safe_call_dJ(f)
    grad_f = np.asarray(grad_f, dtype=np.float64)
    
    # Evaluate constraints and Jacobian together
    c_vals, c_jac = _eval_constraints_and_jacobian(equality_constraints, dofs)
    
    # Build gradient of Lagrangian: grad_f - lag_mul^T @ jac + jac^T @ (mu * c_vals)
    # Use in-place operations where possible
    dL = grad_f.copy()
    dL -= np.dot(lag_mul, c_jac)
    # Pre-compute mu * c_vals to avoid recomputation
    mu_c_vals = mu * c_vals
    dL += np.dot(c_jac.T, mu_c_vals)
    
    return dL


def _augmented_lagrangian_fun_and_grad(dofs, f, equality_constraints, lag_mul, mu):
    """
    Combined evaluation of augmented Lagrangian objective and gradient.
    This avoids redundant constraint evaluations.
    Returns cached constraint values and Jacobian for reuse.
    
    Returns:
        (L, grad_L, c_vals, c_jac): tuple of (objective value, gradient, constraint values, Jacobian)
    """
    f.x = dofs
    if isinstance(f, dummyObjective):
        f_val = f.J()
        grad_f = f.dJ()
    else:
        f_val = _safe_call_J(f)
        grad_f = _safe_call_dJ(f)
    grad_f = np.asarray(grad_f, dtype=np.float64)
    
    # Evaluate constraints and Jacobian together (key optimization!)
    c_vals, c_jac = _eval_constraints_and_jacobian(equality_constraints, dofs)
    
    # Build Lagrangian - optimized
    c_vals_sq = c_vals * c_vals
    L = f_val - np.dot(lag_mul, c_vals) + 0.5 * np.dot(mu, c_vals_sq)
    
    # Build gradient - use in-place operations and pre-compute
    dL = grad_f.copy()
    dL -= np.dot(lag_mul, c_jac)
    mu_c_vals = mu * c_vals  # Pre-compute for reuse
    dL += np.dot(c_jac.T, mu_c_vals)
    
    return float(L), dL, c_vals, c_jac


def augmented_lagrangian_method(
        f=None,
        equality_constraints=[],
        mu_init=10.0,
        grad_tol=1e-15,
        c_tol=1e-15,
        tau=4,
        MAXITER=50,
        argmin_tol=1e-15,
        minimize_method='L-BFGS-B',
        MAXITER_lag=10,
        OUT_DIR='',
        verbose=False,
        penalty_type=None,
        save_vtk_every=None,  # New: only save VTK every N iterations (None = never, 0 = only at end)
        ):
    """
    Run the Augmented Lagrangian Method (ALM) for constrained optimization.
    Maximally optimized version with caching and reduced redundant evaluations.

    This method solves:

        .. math::

            \min_x f(x) \quad \text{subject to} \quad g_i(x) = 0

    by iteratively minimizing the augmented Lagrangian and updating multipliers and penalty parameters.

    The main loop alternates between minimizing the augmented Lagrangian and updating the multipliers/penalty:
        - Minimize :math:`L_A(x, \lambda, \mu)` with respect to :math:`x`
        - Update :math:`\lambda` and :math:`\mu` based on constraint satisfaction

    Args:
        f (Optimizable, optional): Main objective function (with .J(), .dJ(), .x attributes).
            If not provided, only constraints are used in the optimization.
        equality_constraints (list of Optimizable): List of constraint objects corresponding to equality constraints.
            These are equivalent to inequality constraints if the constraint is set up so that if the value is below
            some threshold, the value is set to zero.
        mu_init (float, optional, default=10.0): Initial penalty parameter (must be > 0).
        grad_tol (float, optional, default=1e-15): Tolerance for gradient norm of Lagrangian.
        c_tol (float, optional, default=1e-15): Tolerance for constraint norm.
        MAXITER (int, optional, default=50): Max iterations for inner minimization.
        argmin_tol (float, optional, default=1e-15): Tolerance for inner minimization.
        minimize_method (str, optional, default='L-BFGS-B'): Optimization method for inner loop.
        MAXITER_lag (int, optional, default=10): Max outer ALM iterations.
        OUT_DIR (str, optional, default=''): Directory for output files.
        verbose (bool, optional, default=False): Whether to print verbose output.
        save_vtk_every (int, optional, default=None): Save VTK files every N iterations. 
            None = never, 0 = only at end, positive = every N iterations.

    Returns:
        tuple: (x, final_Lagrangian_value, lagrange_multipliers)
            - x (np.ndarray): Optimized degrees of freedom
            - final_Lagrangian_value (float): Final value of the augmented Lagrangian
            - lagrange_multipliers (np.ndarray): Final Lagrange multipliers
    """
    from simsopt.geo import CurveLength
    np.random.seed(1)
    m_eq = len(equality_constraints)

    if mu_init <= 0 or grad_tol <= 0 or c_tol <= 0:
        raise ValueError(
            "eta_init, mu_init and omega_init  must be strictly positive")
    if not penalty_type:
        if np.isscalar(mu_init):
            mu_k = np.full(m_eq, mu_init, dtype=np.float64)  # More efficient than ones * mu_init
        else:
            mu_k = np.asarray(mu_init, dtype=np.float64)
            if mu_k.size != m_eq:
                raise ValueError("mu_init vector length must match number of constraints")
        if np.any(mu_k <= 1):
            raise ValueError("All components of mu_init must be > 1")
    elif penalty_type == 'scalar':
        mu_k = float(mu_init)
    else:
        raise ValueError("Invalid penalty type")
    
    k = 1

    # Picks the most dofs from the objective function or the first equality constraint
    try:
        x = np.asarray(f.x, dtype=np.float64).copy()
    except:
        x = np.asarray(equality_constraints[0].x, dtype=np.float64).copy()
    
    if verbose:
        print('----------------------------------------------------------------')
        print(f'METHOD {minimize_method} IS SELECTED FOR THE OPTIMIZATION')
        try:
            print(f'INITIAL SQUARED FLUX: {f.J():0.6f}')
        except:
            print(f'INITIAL SQUARED FLUX: {equality_constraints[0].J():0.6f}')
        print('----------------------------------------------------------------')

    if f is None:
        f = dummyObjective(x)

    # Initialize multipliers randomly - pre-allocate array
    if m_eq > 0:
        c_vals = np.empty(m_eq, dtype=np.float64)
        for i, J in enumerate(equality_constraints):
            c_vals[i] = _safe_call_J(J)
        lag_mul = -np.random.rand(m_eq) * np.sign(c_vals)
    else:
        c_vals = np.empty(0, dtype=np.float64)
        lag_mul = np.empty(0, dtype=np.float64)

    # Evaluate initial lagrangian
    c_norm = np.linalg.norm(c_vals) if m_eq > 0 else 0.0

    mu_k_scalar = np.mean(mu_k) if m_eq > 0 else float(mu_init)
    omega_k = 1.0 / mu_k_scalar
    eta_k = 1.0 / (mu_k_scalar ** 0.1)
    aug_lag, grad_vec, _, _ = _augmented_lagrangian_fun_and_grad(x, f, equality_constraints, lag_mul, mu_k)
    grad_aug_lag_norm = np.linalg.norm(grad_vec)

    if verbose:
        print("--------------------------------------------------------------------------------------------------------------------------------------------")
        # Handle mu_k formatting - it can be a scalar or array
        if np.isscalar(mu_k):
            mu_k_str = f"{mu_k:.2e}"
        else:
            mu_k_str = f"[{', '.join([f'{m:.2e}' for m in mu_k])}]"
        print(f"Iteration {0}, \u03BC_k={mu_k_str}, \u03C9_k={omega_k:.2e}, \u03B7_k={eta_k:.2e}, \u221A║∇L_A║ = {grad_aug_lag_norm:.2e}, \u221A║g║ = {c_norm:.2e}")
        print(f'L_A value: {aug_lag:0.5f}')
        print("--------------------------------------------------------------------------------------------------------------------------------------------")

    # Pre-allocate options dict
    options = {
        'disp': False,
        'maxiter': MAXITER,
    }
    if minimize_method == 'L-BFGS-B':
        options['maxcor'] = 100

    # Pre-compute VTK save condition
    save_vtk_condition = save_vtk_every is not None and save_vtk_every > 0

    # If the penalty parameter is too large, stop the optimization
    while (grad_aug_lag_norm > grad_tol or c_norm > c_tol) and k < MAXITER_lag:
        # Solve arg min of the augmented lagrangian
        dofs_before = x.copy()
        
        # Use combined function to avoid redundant constraint evaluations
        # Create a closure that only returns (L, grad_L) for scipy
        def fun(dofs):
            L, grad_L, _, _ = _augmented_lagrangian_fun_and_grad(dofs, f, equality_constraints, lag_mul, mu_k)
            return L, grad_L
        
        options['gtol'] = omega_k
        
        if k == 1:
            # Taylor test
            h = np.random.uniform(size=x.shape)
            J0, dJ0 = fun(x)
            dJh = np.sum(dJ0 * h)
            err = 1e100
            for eps in [1e-3, 1e-4, 1e-5]:
                J1, _ = fun(x + eps*h)
                J2, _ = fun(x - eps*h)
                err_new = np.abs((J1-J2)/(2*eps) - dJh)
                if not (err_new < err * 0.5) and err > 1e-10:
                    print("Taylor test failed, err_new = {:.2e}, err = {:.2e}".format(err_new, err))
                    raise ValueError("Taylor test failed, check your objective and constraint functions")
                err = err_new
            print("Taylor test passed")

        res = minimize(fun, x, method=minimize_method, options=options, 
                        jac=True, tol=argmin_tol)
        x = np.asarray(res.x, dtype=np.float64)
        
        if verbose:
            print('||Δx||:', np.linalg.norm(x - dofs_before))

        # Recompute gradient and constraints - get cached values
        aug_lag, grad_vec, c_vals, _ = _augmented_lagrangian_fun_and_grad(x, f, equality_constraints, lag_mul, mu_k)
        grad_aug_lag_norm = np.linalg.norm(grad_vec)
        c_norm = np.linalg.norm(c_vals, ord=np.inf) if m_eq > 0 else 0.0

        # Save VTK conditionally
        if save_vtk_condition and (k % save_vtk_every == 0):
            try:
                if isinstance(f, SquaredFlux):
                    curves = [c.curve for c in f.field.coils]
                    curves_to_vtk(curves, OUT_DIR + "last_optimized_coils_auglag" + str(k))
                else:
                    curves = [c.curve for c in equality_constraints[0].Jobj.field.coils]
                    curves_to_vtk(curves, OUT_DIR + "last_optimized_coils_auglag" + str(k))
            except:
                pass

        # Print detailed progress
        if verbose:
            print(f"Iteration {k}")
            try:
                print(f"  Objective f(x) = {f.J()}")
            except:
                pass
            print(f"  Constraints g(x) = {c_vals}")
            print(f"  Constraint norm = {c_norm}")
            print(f"  Gradient norm = {grad_aug_lag_norm}")
            print(f"  Lagrange multipliers = {lag_mul}")
            # Handle mu_k formatting - it can be a scalar or array
            if np.isscalar(mu_k):
                print(f"  Penalty parameter mu_k = {mu_k}")
            else:
                print(f"  Penalty parameter mu_k = [{', '.join([f'{m:.2e}' for m in mu_k])}]")
            print(f"  Penalty parameter omega_k = {omega_k}")
            print(f"  Penalty parameter eta_k = {eta_k}")
            print(f"  Change in x = {np.linalg.norm(x - dofs_before)}")
            print("--------------------------------------------------")

        if m_eq > 0:
            if verbose:
                try:
                    ncoils = len(equality_constraints[0].field.coils) // (equality_constraints[0].surface.stellsym + 1) // (equality_constraints[0].surface.nfp)
                    print(f"  Normalized flux: {equality_constraints[0].J():.2e}")
                    print(f"  CS separation: {equality_constraints[1].J():.2e} (min distance: {equality_constraints[1].shortest_distance():.3f})")
                    print(f"  CC separation: {equality_constraints[2].J():.2e} (min distance: {equality_constraints[2].shortest_distance():.3f})")
                    print(f"  Length constraint: {equality_constraints[3].J():.2e}")
                    print(f"  Curvature constraint: {equality_constraints[4].J():.2e}")
                    print(f"  MSC Curvature constraint: {equality_constraints[5].J():.2e}")
                    print(f"  Linking number: {equality_constraints[6].J():.2e}")
                    print(f"  Force constraint: {equality_constraints[7].J():.2e}")
                    print(f"  Max curvatures: {[np.max(c.curve.kappa()) for c in equality_constraints[0].field.coils[:ncoils]]}")
                    print(f"  Lengths: {[CurveLength(c.curve).J() for c in equality_constraints[0].field.coils[:ncoils]]}")
                    print(f"  Total length: {sum([CurveLength(c.curve).J() for c in equality_constraints[0].field.coils[:ncoils]]):.2e}")
                except:
                    pass
            # Optimize string building
            try:
                f_val_str = f'{f.J():.2e}'
            except:
                f_val_str = 'N/A'
            c_str_parts = [f"Iter {k}: Jf = {f_val_str}"]
            for i, c in enumerate(c_vals):
                c_str_parts.append(f'c{i} = {abs(c):.2e}')
            print(', '.join(c_str_parts))
        if verbose:
            try:
                print('Max curvatures:', [np.max(c.kappa()) for c in equality_constraints[1].Jobj.curves])
            except:
                pass
        
        # increase penalty only for violated constraints - use vectorized operations
        if not penalty_type:
            # Vectorized penalty update
            violation_mask = np.abs(c_vals) > c_tol
            mu_k[violation_mask] *= tau
        
        # Constraints are making good progress, update Lagrange multipliers
        if c_norm < eta_k:
            # constraints sufficiently small: update multipliers
            if verbose:
                print("*Constraints are satisfied*")
            lag_mul -= mu_k * c_vals  # In-place subtraction
            # tighten tolerances based on current worst‐case mu
            omega_k = max(omega_k / mu_k_scalar, grad_tol)
            eta_k = max(eta_k / mu_k_scalar, c_tol)
        else:
            if verbose:
                print("*Constraints are not satisfied*")
            if not penalty_type:
                mu_k_scalar = np.mean(mu_k)
            else:
                mu_k = tau * mu_k
            omega_k = max(1.0 / mu_k_scalar, grad_tol)
            eta_k = max(1.0 / (mu_k_scalar ** 0.1), c_tol)
        if verbose:
            print("LAGRANGE MULTIPLIERS:", lag_mul)
            print("--------------------------------------------------------------------------------------------------------------------------------------------")

        k += 1

    # Save final VTK if requested
    if save_vtk_every == 0:
        try:
            if isinstance(f, SquaredFlux):
                curves = [c.curve for c in f.field.coils]
                curves_to_vtk(curves, OUT_DIR + "last_optimized_coils_auglag_final")
            else:
                curves = [c.curve for c in equality_constraints[0].Jobj.field.coils]
                curves_to_vtk(curves, OUT_DIR + "last_optimized_coils_auglag_final")
        except:
            pass

    if verbose:
        print('While loop finished because something became false: \n',
            'grad_tol_check = ', grad_aug_lag_norm > grad_tol,
            ', c_tol_check = ', c_norm > c_tol,
            ', maxiter_check = ', k < MAXITER_lag,
        )
    try:
        return x, res.fun, lag_mul
    except:  # while loop did not run a single time
        return x, None, lag_mul

