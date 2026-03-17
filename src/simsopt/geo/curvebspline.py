import numpy as np
import jax.numpy as jnp
from .curve import JaxCurve

__all__ = ['JaxCurveBSpline']


def _periodic_bspline_basis(quadpoints, n, degree):
    r"""
    Evaluate periodic uniform B-spline basis functions via de Boor recursion.

    For a periodic B-spline with ``n`` control points and uniform knot spacing
    ``h = 1/n``, at most ``degree + 1`` basis functions are non-zero at any
    parameter value.

    Args:
        quadpoints: parameter values in [0, 1), shape ``(nquad,)``
        n: number of control points
        degree: polynomial degree of the B-spline

    Returns:
        Tuple ``(b, indices)`` where ``b`` has shape ``(nquad, degree+1)`` with
        the non-zero basis function values, and ``indices`` has shape
        ``(nquad, degree+1)`` with the corresponding control point indices
        (wrapped mod ``n``).
    """
    s = quadpoints * n
    i = jnp.floor(s).astype(jnp.int32)
    u = s - jnp.floor(s)

    b = jnp.ones((quadpoints.shape[0], 1))

    for k in range(1, degree + 1):
        b_left = jnp.concatenate([jnp.zeros((b.shape[0], 1)), b], axis=1)
        b_right = jnp.concatenate([b, jnp.zeros((b.shape[0], 1))], axis=1)
        j_vals = jnp.arange(k + 1)
        left_w = (u[:, None] + k - j_vals[None, :]) / k
        right_w = (j_vals[None, :] + 1 - u[:, None]) / k
        b = left_w * b_left + right_w * b_right

    offsets = jnp.arange(-degree, 1)
    indices = (i[:, None] + offsets[None, :]) % n

    return b, indices


def curve_bspline_pure(dofs, quadpoints, n, degree):
    r"""
    Pure function for a periodic B-spline curve in 3D.

    Args:
        dofs: flattened control points, shape ``(3*n,)``
        quadpoints: parameter values in [0, 1), shape ``(nquad,)``
        n: number of control points
        degree: polynomial degree

    Returns:
        Curve positions, shape ``(nquad, 3)``
    """
    P = dofs.reshape(n, 3)
    b, indices = _periodic_bspline_basis(quadpoints, n, degree)
    selected_P = P[indices]
    gamma = jnp.sum(b[:, :, None] * selected_P, axis=1)
    return gamma


def _gauss_legendre_panels(n_control_points, degree, numquadpoints):
    r"""
    Compute composite Gauss-Legendre quadrature nodes and weights for a
    periodic B-spline with ``n_control_points`` control points.

    The knot spans are ``[k/n, (k+1)/n]`` for ``k = 0, ..., n-1``.  On each
    span ``p`` GL points are placed (where ``p = ceil(numquadpoints / n)``),
    giving ``p * n`` total quadrature points.

    Args:
        n_control_points: number of B-spline control points (= number of
            knot spans for a periodic uniform B-spline)
        degree: polynomial degree (not directly used for node placement,
            but ``p >= degree + 1`` is recommended for exactness)
        numquadpoints: requested total number of quadrature points

    Returns:
        ``(quadpoints, quadweights)`` numpy arrays, each of length ``p * n``
    """
    n = n_control_points
    p = int(np.ceil(numquadpoints / n))
    p = max(p, degree + 1)

    xi, wi = np.polynomial.legendre.leggauss(p)
    h = 1.0 / n

    all_points = np.empty(p * n)
    all_weights = np.empty(p * n)

    for k in range(n):
        a = k * h
        mid = a + 0.5 * h
        half_h = 0.5 * h
        idx = slice(k * p, (k + 1) * p)
        all_points[idx] = mid + half_h * xi
        all_weights[idx] = wi * half_h

    return all_points, all_weights


class JaxCurveBSpline(JaxCurve):
    r"""
    Periodic B-spline curve with composite Gauss-Legendre panel quadrature.

    Unlike Fourier representations, each control point only influences
    ``degree + 1`` adjacent curve segments.  When used with simsopt's
    C++ ``BiotSavart`` kernel, the per-point ``quadweights`` array enables
    high-order integration that respects the knot structure, eliminating
    the O(N^-2) convergence floor inherent in the uniform trapezoidal rule
    applied to C^2 integrands.

    The DOFs are the flattened 3D control-point coordinates:

    .. math::
        [x_0, y_0, z_0, x_1, y_1, z_1, \ldots, x_{n-1}, y_{n-1}, z_{n-1}]

    Args:
        quadpoints: total number of quadrature points (int).
            The actual number may be rounded up to ``p * n`` where
            ``p = ceil(quadpoints / n)`` and ``p >= degree + 1``.
        n_control_points: number of control points (must be > degree)
        degree: polynomial degree (default 3)
        dofs: optional initial DOFs array of shape ``(3 * n_control_points,)``
    """

    def __init__(self, quadpoints, n_control_points, degree=3, dofs=None):
        assert n_control_points > degree, \
            f"n_control_points ({n_control_points}) must exceed degree ({degree})"

        self.n_control_points = n_control_points
        self.degree = degree

        if isinstance(quadpoints, int):
            gl_points, gl_weights = _gauss_legendre_panels(
                n_control_points, degree, quadpoints)
            quadpoints_arr = gl_points
        else:
            quadpoints_arr = np.asarray(quadpoints)
            gl_weights = np.full(len(quadpoints_arr),
                                 1.0 / len(quadpoints_arr))

        def pure(d, pts):
            return curve_bspline_pure(d, pts, n_control_points, degree)

        self.coefficients = np.zeros(3 * n_control_points)
        names = self._make_names(n_control_points)

        if dofs is None:
            super().__init__(quadpoints_arr, pure, x0=self.coefficients.copy(),
                             names=names,
                             external_dof_setter=JaxCurveBSpline.set_dofs_impl)
        else:
            super().__init__(quadpoints_arr, pure, dofs=dofs,
                             names=names,
                             external_dof_setter=JaxCurveBSpline.set_dofs_impl)

        if isinstance(gl_weights, np.ndarray) and gl_weights is not None:
            self.quadweights = gl_weights

    @staticmethod
    def _make_names(n):
        names = []
        for i in range(n):
            names.extend([f'x({i})', f'y({i})', f'z({i})'])
        return names

    def num_dofs(self):
        return 3 * self.n_control_points

    def get_dofs(self):
        return self.coefficients.copy()

    def set_dofs_impl(self, dofs):
        self.coefficients[:] = dofs
