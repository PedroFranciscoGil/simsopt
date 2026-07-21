"""
qss_helpers.py — Helper functions for qss_script_jax.py
========================================================

Extracted from main() for readability. All former closures over
main()-local variables now take those values as explicit parameters.
"""

from contextlib import contextmanager
from typing import List

import numpy as np
from simsopt.objectives import (
    LeastSquaresProblem, QuadraticPenalty,
    SquaredFluxJax, SquaredFluxJaxAnalytic,
)
from simsopt.geo import CurveSurfaceDistance
from simsopt.geo.curveobjectives_fast import (
    CurveLength_Fast, CurveCurveDistance_Fast, MeanSquaredCurvature_Fast,
    LpCurveCurvature_Fast, LinkingNumber_Fast,
)
from simsopt.util import proc0_print


# ---------------------------------------------------------------------------
# Data structures (re-exported for convenience)
# ---------------------------------------------------------------------------

from typing import NamedTuple, Callable


class Objective(NamedTuple):
    """A term that is minimised: min ||fn() - target||^2."""
    fn:     Callable
    target: float
    label:  str


class Constraint(NamedTuple):
    """An equality constraint enforced via the AL1 penalty: fn() = target."""
    fn:     Callable
    target: float
    label:  str


class IneqConstraint(NamedTuple):
    """An inequality constraint g(x) <= 0 handled via PHR ALM.

    fn returns the RAW SIGNED metric g (negative when satisfied, positive
    when violated), e.g. aspect - 9, not a clipped/squared penalty. fn must
    be a bound method of an Optimizable (e.g. make_optimizable(...).J) so
    the LeastSquaresProblem dependency graph can be built from it.
    """
    fn:    Callable
    label: str


# ---------------------------------------------------------------------------
# AL1 machinery
# ---------------------------------------------------------------------------

def build_al_subproblem(
    objectives:  List[Objective],
    constraints: List[Constraint],
    lam:         np.ndarray,
    rho:         float,
) -> LeastSquaresProblem:
    # LeastSquaresProblem terms are weight * (f - goal)^2 — the weight
    # multiplies the SQUARED residual. The AL term (rho/2)(c + lam/rho)^2
    # therefore needs weight rho/2, matching the dual update
    # lam <- lam + rho*c and the coil penalty J2 = (rho/2)*sum(...)^2.
    rho_half = rho / 2.0
    tuples = [(obj.fn, obj.target, 1.0) for obj in objectives]
    for i, con in enumerate(constraints):
        shifted_target = con.target - lam[i] / rho
        tuples.append((con.fn, shifted_target, rho_half))
    return LeastSquaresProblem.from_tuples(tuples)


def build_al_subproblem_phr(
    objectives:       List[Objective],
    ineq_constraints: List["IneqConstraint"],
    lam:              np.ndarray,
    rho:              np.ndarray,
) -> LeastSquaresProblem:
    """PHR (Powell-Hestenes-Rockafellar) AL subproblem for g_i(x) <= 0.

    Each constraint contributes (rho_i/2) * max(0, g_i + lam_i/rho_i)^2,
    encoded as a LeastSquares term with residual fn max(0, g_i + lam_i/rho_i),
    goal 0 and weight rho_i/2 (LeastSquaresProblem weights multiply the
    SQUARED residual). Unlike the equality form, the multiplier shift sits
    INSIDE the hinge, so a positive lam_i keeps the constraint gradient
    alive while g_i > -lam_i/rho_i — this is what lets PHR hold a constraint
    at its boundary.

    lam and rho are per-constraint arrays, frozen into this subproblem;
    the caller rebuilds it every outer iteration (fresh hinge wrappers are
    created on purpose: mutating lam/rho in place would not invalidate the
    Optimizable caches).
    """
    from simsopt import make_optimizable

    lam = np.asarray(lam, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    tuples = [(obj.fn, obj.target, 1.0) for obj in objectives]
    for i, con in enumerate(ineq_constraints):
        parent = getattr(con.fn, "__self__", None)
        if parent is None:
            raise TypeError(
                f"IneqConstraint '{con.label}': fn must be a bound method "
                f"of an Optimizable (e.g. make_optimizable(...).J).")
        shift = float(lam[i] / rho[i])

        def _hinge(_parent, _fn=con.fn, _shift=shift):
            return max(0.0, float(_fn()) + _shift)

        term = make_optimizable(_hinge, parent)
        tuples.append((term.J, 0.0, float(rho[i]) / 2.0))
    return LeastSquaresProblem.from_tuples(tuples)


def update_dual_phr(lam, rho, g_vals, lam_max):
    """PHR multiplier update for inequality constraints g_i <= 0:

        lam_i <- clip(lam_i + rho_i * g_i, 0, lam_max)

    g_vals are the RAW SIGNED constraint values. Unlike the equality-style
    update, a satisfied constraint (g_i < 0) pulls its multiplier DOWN
    (floored at 0), so constraints that go inactive release their
    multiplier instead of ratcheting.

    Returns (lam_new, violations) with violations = max(0, g).
    """
    g = np.asarray(g_vals, dtype=np.float64)
    lam_new = np.clip(lam + rho * g, 0.0, lam_max)
    return lam_new, np.maximum(0.0, g)


def update_rho_conditioned(rho, v, v_prev, theta=0.25, tau=2.0,
                           rho_max=1e4, v_tol=1e-4):
    """LANCELOT-style conditioned per-constraint penalty growth.

    Grow rho_i (factor tau, capped at rho_max) only where the violation
    v_i = max(0, g_i) is above v_tol AND failed to contract to theta times
    its value at the previous outer iteration. On the first call
    (v_prev is None) rho is returned unchanged.

    Returns (rho_new, grew) where grew is a boolean mask.
    """
    rho = np.asarray(rho, dtype=np.float64).copy()
    v = np.asarray(v, dtype=np.float64)
    if v_prev is None:
        return rho, np.zeros(rho.shape, dtype=bool)
    v_prev = np.asarray(v_prev, dtype=np.float64)
    grew = (v > v_tol) & (v > theta * v_prev)
    rho[grew] = np.minimum(rho[grew] * tau, rho_max)
    return rho, grew


def phr_diagnostics(al_iter, al_total, g_eq, eq_labels, lam_eq, rho_eq,
                    g_coil, coil_labels, lam_coil, rho_coil, objectives):
    """Per-constraint PHR table: raw g, violation v, lambda, rho, activity."""
    obj_parts = ", ".join(
        f"{obj.label} rms="
        f"{float(np.sqrt(np.mean((np.atleast_1d(np.asarray(obj.fn(), dtype=float)) - obj.target) ** 2))):.2e}"
        for obj in objectives
    )
    proc0_print(f"  PHR iter {al_iter+1}/{al_total}:  {obj_parts}")

    def _rows(labels, g, lam, rho):
        for i, label in enumerate(labels):
            gi = float(g[i])
            if not np.isfinite(gi):
                proc0_print(f"    {label:12s}  g=NaN (update skipped)")
                continue
            vi = max(0.0, gi)
            status = "VIOLATED" if vi > 0 else (
                "active*" if lam[i] > 0 else "inactive")
            proc0_print(
                f"    {label:12s}  g={gi:+.3e}  v={vi:.3e}  "
                f"lam={lam[i]:.3e}  rho={rho[i]:.4g}  {status}")

    _rows(eq_labels, g_eq, lam_eq, rho_eq)
    if len(g_coil) > 0:
        _rows(coil_labels, g_coil, lam_coil, rho_coil)


class CoilRidge:
    """B2 ridge regularizer for the AL2 (inner coil) objective.

    A weak Tikhonov term  0.5 * w * || y_coil - ref ||^2  on the coil DOFs,
    added to the AL2 objective f. Its Hessian is  w * I  on the coil-coil
    block, which LIFTS the near-null eigenvalues of the flat coil optimum
    (~1e-7) up to ~w, so that:
      (i)  y*(x) becomes a unique, smooth function of the surface (the FD
           reference for Eq.(18) is then well defined), and
      (ii) the AL2 solve is stabilized against mu-runaway under perturbation.

    Duck-typed exactly like augmented_lagrangian_fast.dummyObjective (it is
    NOT a graph Optimizable): the caller sets `.x`, and J/dJ/d2J act on it.
    `.x` may be coil-only (size n_coil, during the AL2 solve with the surface
    fixed) OR combined [coil, surf] (size n_dofs, in the Eq.18 Hessian). Since
    coils come first in the canonical ordering, we always penalize the first
    n_coil entries. `ref` is captured ONCE (the baseline coils) so it is
    x-independent — the ridge then adds w*I to the coil-coil Hessian and 0 to
    the cross-Hessian, exactly the eigenvalue lift wanted without perturbing
    the IFT cross term.
    """

    def __init__(self, n_coil, weight):
        self.n_coil = int(n_coil)
        self.weight = float(weight)
        self.x = np.zeros(self.n_coil, dtype=np.float64)
        self.ref = None

    def ensure_ref(self, x_coil):
        """Capture the (fixed) reference coil DOFs on first use."""
        if self.ref is None:
            self.ref = np.asarray(x_coil, dtype=np.float64)[:self.n_coil].copy()

    def _ref(self):
        return (np.zeros(self.n_coil) if self.ref is None else self.ref)

    def J(self):
        y = np.asarray(self.x, dtype=np.float64)[:self.n_coil]
        return float(0.5 * self.weight * np.sum((y - self._ref()) ** 2))

    def dJ(self):
        x = np.asarray(self.x, dtype=np.float64)
        g = np.zeros_like(x)
        m = min(self.n_coil, x.size)
        g[:m] = self.weight * (x[:m] - self._ref()[:m])
        return g

    def d2J(self):
        x = np.asarray(self.x, dtype=np.float64)
        n = x.size
        h = np.zeros((n, n), dtype=np.float64)
        m = min(self.n_coil, n)
        idx = np.arange(m)
        h[idx, idx] = self.weight
        return h


class ArclengthGaugeFix:
    """Targeted gauge-fix for the AL2 (inner coil) objective.

    f = w * sum_i ArclengthVariation(curve_i). The arclength-variation penalty
    fixes the CURVE REPARAMETRISATION null direction of CurveXYZFourier (a point
    sliding along the same geometric curve) — the dominant flat direction of the
    coil optimum. Crucially it is ORTHOGONAL to the squared-flux / geometry
    gradient by construction (reparametrisation changes neither the curve nor the
    field), so unlike a generic ridge it lifts the flat directions WITHOUT
    fighting the objective — letting the AL2 solve actually reach a stationary
    point (the prerequisite for Eq.(18)).

    Duck-typed like dummyObjective. `.x` may be coil-only (solve) or combined
    [coil, surf] (Eq.18 Hessian); curves come first, so we sync the base curves
    from the first n_curve entries and place each per-curve gradient/Hessian at
    its coil-DOF offset (currents and surface entries are zero — arclength
    depends on neither)."""

    def __init__(self, base_curves, reference_dof_names, weight,
                 nintervals="full"):
        from simsopt.geo.curveobjectives_fast import ArclengthVariation_Fast
        self.base_curves = list(base_curves)
        self.arcs = [ArclengthVariation_Fast(c, nintervals=nintervals)
                     for c in base_curves]
        self.weight = float(weight)
        # Map each curve's DOFs to their positions in the AL2 coil vector by
        # matching DOF NAMES against the reference (e.g. BiotSavart.dof_names).
        # The coil-vector order is NOT [curve0, curve1, ...]; in practice the
        # currents come FIRST, then the curves — so an assumed contiguous
        # layout silently scrambles the curves. Name-matching is layout-proof.
        pos = {name: i for i, name in enumerate(list(reference_dof_names))}
        self.idx = []
        for c in base_curves:
            try:
                ii = np.array([pos[n] for n in c.dof_names], dtype=int)
            except KeyError as e:
                raise RuntimeError(
                    f"ArclengthGaugeFix: curve DOF {e} absent from the "
                    f"reference dof_names; cannot map the coil vector.")
            self.idx.append(ii)
        self.n_ref = len(pos)
        self.x = np.zeros(self.n_ref, dtype=np.float64)

    def ensure_ref(self, x_coil):     # no reference point needed (it's a gauge)
        pass

    def _sync_curves(self):
        """Move the base curves to their DOFs in self.x (mapped by name) so the
        per-curve arclength evaluations reflect the current optimisation point
        (the AL driver calls f.J()/f.dJ() BEFORE re-evaluating the constraints).
        Indices are coil-block positions (< n_coil), valid whether self.x is
        coil-only (solve) or [coil, surf] (Eq.18 Hessian)."""
        x = np.asarray(self.x, dtype=np.float64)
        for ii, c in zip(self.idx, self.base_curves):
            if ii.size and int(ii.max()) < x.size:
                c.x = x[ii]

    def J(self):
        self._sync_curves()
        return float(self.weight * sum(a.J() for a in self.arcs))

    def dJ(self):
        self._sync_curves()
        x = np.asarray(self.x, dtype=np.float64)
        g = np.zeros_like(x)
        for ii, a in zip(self.idx, self.arcs):
            gi = np.asarray(a.dJ(), dtype=np.float64).ravel()
            g[ii] = self.weight * gi
        return g

    def d2J(self):
        self._sync_curves()
        x = np.asarray(self.x, dtype=np.float64)
        n_dof = x.size
        H = np.zeros((n_dof, n_dof), dtype=np.float64)
        for ii, a in zip(self.idx, self.arcs):
            hi = np.asarray(a.d2J(), dtype=np.float64)
            H[np.ix_(ii, ii)] = self.weight * hi
        return H


class CombinedF:
    """Duck-typed sum of AL2 objective terms (e.g. ridge + arclength gauge).
    Propagates `.x` to every term; sums J/dJ/d2J."""

    def __init__(self, terms):
        self.terms = list(terms)
        self._x = np.zeros(0, dtype=np.float64)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, v):
        self._x = np.asarray(v, dtype=np.float64)
        for t in self.terms:
            t.x = self._x

    def ensure_ref(self, x_coil):
        for t in self.terms:
            if hasattr(t, "ensure_ref"):
                t.ensure_ref(x_coil)

    def J(self):
        return float(sum(t.J() for t in self.terms))

    def dJ(self):
        return sum(np.asarray(t.dJ(), dtype=np.float64) for t in self.terms)

    def d2J(self):
        return sum(np.asarray(t.d2J(), dtype=np.float64) for t in self.terms)


def update_dual(lam, rho, violations, constraint_tol, lam_max):
    """Update dual variables given constraint violations.

    Parameters
    ----------
    lam : ndarray
        Current multipliers.
    rho : float
        Penalty parameter.
    violations : ndarray or list[Constraint]
        Pre-computed violation array, or a list of Constraint namedtuples
        (in which case violations are computed as fn() - target).
    constraint_tol : float
        Skip update if |violation| <= this.
    lam_max : float
        Clamp magnitude of multipliers.
    """
    if not isinstance(violations, np.ndarray):
        violations = np.array([con.fn() - con.target for con in violations])
    lam_new = lam + rho * violations
    mask = np.abs(violations) <= constraint_tol
    lam_new[mask] = lam[mask]
    return np.clip(lam_new, -lam_max, lam_max), violations


def diagnose_al2_multipliers(al2_lag_mul, al2_vals, constraint_names,
                              history, mult_tol=1e-3, value_tol=1e-4,
                              flip_window=5, flip_threshold=2):
    """Detect 'dropped-active' failure mode of the inner AL.

    Per Fu et al. (arXiv:2510.16243) App. B and Fig. 15-16, an augmented
    Lagrangian solver can converge with a constraint that should be active
    having multiplier ~ 0. When this happens, the constraint's contribution
    to the H1 (multiplier) and partially H2 (augmented) Hessian terms is
    lost, making Eq. (18) inaccurate.

    Two regimes are flagged:
      - DROPPED_ACTIVE: |lambda_i| / max|lambda| < mult_tol AND |c_i| > value_tol
        (the constraint is violated but its multiplier is dead).
      - SPORADIC: across the last `flip_window` AL2 solves, the multiplier
        of constraint i toggled in/out of the "tiny" regime at least
        `flip_threshold` times (the signature of Fig. 16 in the paper).

    The `history` dict is updated in place with one (lambda, value) entry
    per constraint per call, so it can be serialised at end-of-run.

    Returns:
        list of (name, flag, lambda, value) tuples in the order of
        constraint_names. `flag` is one of:
          {"OK", "DROPPED_ACTIVE", "INACTIVE_SATISFIED", "SPORADIC"}.
    """
    lag_mul = np.atleast_1d(np.asarray(al2_lag_mul, dtype=np.float64))
    vals = np.atleast_1d(np.asarray(al2_vals, dtype=np.float64))
    lam_scale = max(float(np.max(np.abs(lag_mul))), 1e-30)

    results = []
    for i, name in enumerate(constraint_names):
        lam_i = float(lag_mul[i])
        val_i = float(vals[i])
        history.setdefault(name, []).append({"lambda": lam_i, "value": val_i})

        tiny = abs(lam_i) / lam_scale < mult_tol
        violating = abs(val_i) > value_tol

        if tiny and violating:
            flag = "DROPPED_ACTIVE"
        elif tiny:
            flag = "INACTIVE_SATISFIED"
        else:
            flag = "OK"

        # Sporadic-flip detection across recent history. Recompute the
        # "tiny" classification using each iter's own per-iter scale, so
        # the detector measures relative collapse independent of absolute
        # magnitude drift.
        recent = history[name][-flip_window:]
        if len(recent) >= 3:
            tinies = []
            for entry in recent:
                l = abs(entry["lambda"])
                tinies.append(l < mult_tol * max(l, 1e-30) + mult_tol * lam_scale)
            flips = sum(a != b for a, b in zip(tinies[:-1], tinies[1:]))
            if flips >= flip_threshold and flag == "OK":
                flag = "SPORADIC"
        results.append((name, flag, lam_i, val_i))
    return results


def al_diagnostics(al_iter, al_total, c_eq, eq_constraints, c_coil, coil_names,
                   objectives, lam_eq, lam_coil, rho):
    eq_parts = ", ".join(
        f"{con.label} err={c_eq[i]:+.2e}" for i, con in enumerate(eq_constraints)
    )
    coil_parts = ", ".join(
        f"{coil_names[i]} err={c_coil[i]:+.2e}" for i in range(len(c_coil))
    )
    obj_parts = ", ".join(
        f"{obj.label} rms="
        f"{float(np.sqrt(np.mean((np.atleast_1d(np.asarray(obj.fn(), dtype=float)) - obj.target) ** 2))):.2e}"
        for obj in objectives
    )
    lam_all = np.concatenate([lam_eq, lam_coil])
    lam_str = "[" + ", ".join(f"{v:+.4g}" for v in lam_all) + "]"
    proc0_print(
        f"  AL1 iter {al_iter+1}/{al_total}:  "
        f"|c_eq| = {np.linalg.norm(c_eq):.2e}  ({eq_parts})  "
        f"|c_coil| = {np.linalg.norm(c_coil):.2e}  ({coil_parts})  "
        f"{obj_parts}  lam = {lam_str}  rho = {rho:.4g}"
    )


# ---------------------------------------------------------------------------
# Coil objective builder
# ---------------------------------------------------------------------------

def build_coil_objectives(surf, bs_obj, base_curves_full, base_coils_full,
                          curves, ncoils, length_target,
                          hessian_method, flux_threshold, cc_threshold,
                          cs_threshold, curvature_threshold, msc_threshold):
    """Create all coil objectives/constraints and return them as a dict.

    Parameters added (formerly closed over main()-locals):
        hessian_method, flux_threshold, cc_threshold,
        cs_threshold, curvature_threshold, msc_threshold
    """
    if hessian_method == "jax":
        Jf = SquaredFluxJaxAnalytic(
            surf, bs_obj,
            base_curves=base_curves_full,
            base_currents=[c.current for c in base_coils_full],
            nfp=surf.nfp, stellsym=surf.stellsym,
            definition="normalized", threshold=flux_threshold, fixed_surface=False,
        )
    else:
        Jf = SquaredFluxJax(
            surf, bs_obj,
            definition="normalized", threshold=flux_threshold, fixed_surface=False,
        )
    Jls = [CurveLength_Fast(c) for c in base_curves_full]
    Jccdist = CurveCurveDistance_Fast(curves, cc_threshold, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(base_curves_full, surf, cs_threshold, fix_surface=False)
    J_kappa = [LpCurveCurvature_Fast(c, 2, curvature_threshold) for c in base_curves_full]
    Jlink = LinkingNumber_Fast(curves, downsample=2)
    Jmscs = [MeanSquaredCurvature_Fast(c) for c in base_curves_full]
    c_list = [
        Jf, Jccdist, Jcsdist,
        sum(QuadraticPenalty(J, msc_threshold, "max") for J in Jmscs),
        sum(QuadraticPenalty(J, length_target, "max") for J in Jls),
        sum(J_kappa), Jlink,
    ]
    return {
        "Jf": Jf, "Jls": Jls, "Jccdist": Jccdist, "Jcsdist": Jcsdist,
        "J_kappa": J_kappa, "Jlink": Jlink, "Jmscs": Jmscs, "c_list": c_list,
    }


# ---------------------------------------------------------------------------
# Surface / VTK helpers
# ---------------------------------------------------------------------------

@contextmanager
def fixed_surface(surf, max_mode):
    """Temporarily fix all surface DOFs, restoring the active range on exit."""
    surf.fix_all()
    try:
        yield
    finally:
        surf.unfix_all()
        surf.fix_all()
        surf.fixed_range(mmin=0, mmax=max_mode,
                         nmin=-max_mode, nmax=max_mode, fixed=False)
        surf.fix("rc(0,0)")


def compute_bn_modb(surf, bs_obj):
    """Compute B_n/|B| and |B| on the surface grid. Returns (Bn_over_B, absB)."""
    pts = surf.gamma().reshape((-1, 3))
    bs_obj.set_points(pts)
    nphi, ntheta = surf.gamma().shape[0], surf.gamma().shape[1]
    B = np.asarray(bs_obj.B()).reshape(nphi, ntheta, 3)
    absB = np.sqrt(np.sum(B**2, axis=-1))
    n = surf.normal()
    n_mag = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-15
    unit_n = n / n_mag
    Bn_over_B = np.sum(B * unit_n, axis=-1) / (absB + 1e-15)
    return Bn_over_B, absB


_vtk_cache = {}  # keyed by (surf.x bytes) to avoid recomputation


def surface_vtk_point_data(surf, bs_obj):
    """VTK extra data dict for surface export, with caching.

    Keyed on the surface DOFs AND the full Biot-Savart DOFs (coil shapes
    + currents, fixed or free): B.n/|B| changes when the coils move even
    if the surface does not, e.g. before/after an AL2 solve.
    """
    cache_key = (surf.x.tobytes(),
                 np.asarray(bs_obj.full_x, dtype=np.float64).tobytes())
    cached = _vtk_cache.get(cache_key)
    if cached is not None:
        return cached
    Bn_over_B, absB = compute_bn_modb(surf, bs_obj)
    nphi, ntheta = surf.gamma().shape[0], surf.gamma().shape[1]
    contig = np.ascontiguousarray
    result = {
        "B_N/|B|": contig(Bn_over_B.reshape(1, nphi, ntheta)),
        "modB": contig(absB.reshape(1, nphi, ntheta)),
    }
    _vtk_cache.clear()
    _vtk_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Coil re-initialisation
# ---------------------------------------------------------------------------

def reinit_coils_to_surface_scaled_circular(surf, base_curves_full, base_currents,
                                            ncoils, order, total_current,
                                            coil_radius_factor, arc_length_samples):
    """Re-initialise coils along the magnetic axis of the current equilibrium.

    Parameters added (formerly closed over module-level constants):
        coil_radius_factor, arc_length_samples
    """
    from scipy.interpolate import CubicSpline

    nfp = surf.nfp
    stellsym = surf.stellsym
    a_minor = surf.minor_radius()
    R1_new = coil_radius_factor * a_minor

    gamma = surf.gamma()
    axis_coarse = np.mean(gamma, axis=1)

    phi_surf = np.asarray(surf.quadpoints_phi).ravel()
    if phi_surf[-1] <= 1.0 + 1e-10:
        phi_surf = phi_surf * 2 * np.pi
    axis_ext = np.concatenate([axis_coarse, axis_coarse[:1]], axis=0)
    phi_ext = np.concatenate([phi_surf, [phi_surf[0] + 2 * np.pi]])
    cs_x = CubicSpline(phi_ext, axis_ext[:, 0], bc_type='periodic')
    cs_y = CubicSpline(phi_ext, axis_ext[:, 1], bc_type='periodic')
    cs_z = CubicSpline(phi_ext, axis_ext[:, 2], bc_type='periodic')

    phi_period = 2 * np.pi / (nfp * (1 + int(stellsym)))
    n_hp = arc_length_samples
    phi_hp = np.linspace(0, phi_period, n_hp + 1)
    ax_hp = np.column_stack([cs_x(phi_hp), cs_y(phi_hp), cs_z(phi_hp)])
    seg_lengths = np.linalg.norm(np.diff(ax_hp, axis=0), axis=1)
    s_cum = np.concatenate([[0], np.cumsum(seg_lengths)])
    s_total = float(s_cum[-1])

    s_targets = np.array([(i + 0.5) * s_total / ncoils for i in range(ncoils)])
    phi_coils = np.interp(s_targets, s_cum, phi_hp)

    ref_curve = base_curves_full[0]
    nq = len(ref_curve.quadpoints)
    theta = 2 * np.pi * np.asarray(ref_curve.quadpoints)
    k = 2 * order + 1
    A = np.zeros((nq, k))
    A[:, 0] = 1.0
    for m in range(1, order + 1):
        A[:, 2 * m - 1] = np.sin(m * theta)
        A[:, 2 * m]     = np.cos(m * theta)
    A_pinv = np.linalg.pinv(A)

    from simsopt.field import Current

    for i_coil in range(ncoils):
        phi_c = phi_coils[i_coil]
        centre = np.array([cs_x(phi_c), cs_y(phi_c), cs_z(phi_c)])

        tangent = np.array([cs_x(phi_c, 1), cs_y(phi_c, 1), cs_z(phi_c, 1)])
        tangent /= np.linalg.norm(tangent) + 1e-30

        d2 = np.array([cs_x(phi_c, 2), cs_y(phi_c, 2), cs_z(phi_c, 2)])
        normal = d2 - np.dot(d2, tangent) * tangent
        nn = np.linalg.norm(normal)
        if nn < 1e-12:
            R_c = np.sqrt(centre[0]**2 + centre[1]**2) + 1e-30
            normal = np.array([centre[0] / R_c, centre[1] / R_c, 0.0])
            normal -= np.dot(normal, tangent) * tangent
            nn = np.linalg.norm(normal)
        normal /= nn + 1e-30

        binormal = np.cross(tangent, normal)
        binormal /= np.linalg.norm(binormal) + 1e-30

        t = 2 * np.pi * np.asarray(base_curves_full[i_coil].quadpoints)
        coil_pts = (centre[None, :]
                    + R1_new * (np.cos(t)[:, None] * normal[None, :]
                                - np.sin(t)[:, None] * binormal[None, :]))

        coeffs = A_pinv @ coil_pts
        new_dofs = np.concatenate([coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]])
        base_curves_full[i_coil].x = new_dofs

    new_base_currents = [Current(total_current / ncoils * 1e-5) * 1e5 for _ in range(ncoils - 1)]
    new_total_current_obj = Current(total_current)
    new_total_current_obj.fix_all()
    new_base_currents += [new_total_current_obj - sum(new_base_currents)]
    current_per_coil = total_current / ncoils
    for i in range(ncoils - 1):
        base_currents[i].x = new_base_currents[i].x

    proc0_print(f"  Re-initialised coils along magnetic axis: R1={coil_radius_factor}*a={R1_new:.4f} "
                f"(a_minor={a_minor:.4f}), I/base={current_per_coil:.2f}")


# ---------------------------------------------------------------------------
# Coil diagnostics
# ---------------------------------------------------------------------------

def print_coil_properties(surf, bs_obj, Jccdist, Jcsdist, base_curves_full, Jmscs, Jlink):
    Bn_over_B, _ = compute_bn_modb(surf, bs_obj)
    avg_Bn = float(np.mean(np.abs(Bn_over_B)))
    max_Bn = float(np.max(np.abs(Bn_over_B)))

    w = 52
    proc0_print("")
    proc0_print("  +" + "-" * w + "+")
    proc0_print("  |  Coil properties (after AL2 solve)" + " " * (w - 38) + "|")
    proc0_print("  +" + "-" * w + "+")
    proc0_print(f"  |  CC dist penalty:                {Jccdist.J():>12.6e}  |")
    proc0_print(f"  |  Min CC dist:                    {Jccdist.shortest_distance():>12.6f}  |")
    proc0_print(f"  |  CS dist penalty:                {Jcsdist.J():>12.6e}  |")
    proc0_print(f"  |  Min CS dist:                    {Jcsdist.shortest_distance():>12.6f}  |")
    proc0_print(f"  |  Max curvature:                  {max(float(np.max(c.kappa())) for c in base_curves_full):>12.6f}  |")
    proc0_print(f"  |  Mean sq curvature (avg):        {sum(J.J() for J in Jmscs)/len(Jmscs):>12.6f}  |")
    proc0_print(f"  |  Linking number:                 {Jlink.J():>12.6e}  |")
    proc0_print(f"  |  Avg |B.n/B|:                   {avg_Bn:>12.6e}  |")
    proc0_print(f"  |  Max |B.n/B|:                   {max_Bn:>12.6e}  |")
    proc0_print("  +" + "-" * w + "+")
    proc0_print("")


# ---------------------------------------------------------------------------
# ALM warm-start helpers
# ---------------------------------------------------------------------------

def alm_reset_decision_from_surface_change(surf, state, threshold):
    current_x = np.asarray(surf.x, dtype=np.float64).ravel().copy()
    prev_x = state.get("prev_surface_x")
    if prev_x is None or np.shape(prev_x) != np.shape(current_x):
        return True, current_x, None
    delta = float(np.max(np.abs(current_x - prev_x)))
    return delta > threshold, current_x, delta


def update_alm_warmstart_state(state, surface_x, lag_mul, mu):
    state["prev_surface_x"] = np.asarray(surface_x, dtype=np.float64).ravel().copy()
    state["prev_lag_mul"] = np.asarray(lag_mul, dtype=np.float64).copy()
    state["prev_mu"] = np.asarray(mu, dtype=np.float64).copy()


# ---------------------------------------------------------------------------
# Least-squares callbacks
# ---------------------------------------------------------------------------

SEP_THIN = "-" * 60


def make_ls_callbacks(is_qss, max_mode, surf=None, curves=None,
                      out_dir=None, bs=None, history=None,
                      step=None, al_iter=None, al_outer_iter=None,
                      n_continuation_steps=None,
                      vtk_exporter=None, is_proc0=True,
                      j2_holder=None):
    """Build (iteration_callback, jac_callback) for least_squares solvers.

    Parameters
    ----------
    is_qss : bool
        True for the QSS branch (J1 + J2), False for equilibrium-only.
    max_mode : int
        Current Fourier-mode resolution (for printouts).
    surf, curves, bs : optional
        Geometry handles used by the VTK exporter.
    out_dir : str, optional
        Output directory (kept for backwards compatibility, unused here).
    history : list, optional
        Mutable list of dicts; one entry appended per surface iteration with
        keys ``J1``, ``J2``, ``grad_J1``, ``grad_J2``, ``step``, ``al_iter``,
        ``max_mode``, ``nevals``.
    step, al_iter, al_outer_iter, n_continuation_steps : int, optional
        Context used in the printout banner ("AL1 iter X/Y, continuation
        step S/N, max_mode=M").
    vtk_exporter : OptimizationVTKExporter, optional
        If provided, ``export_frame`` is called on every surface iteration so
        the surface shape (and coils, if any) is dumped per frame and shows
        up in the final PVD animation. Only proc0 actually writes files.
    is_proc0 : bool
        Whether this rank is rank 0 (used to gate VTK writes).
    j2_holder : list, optional
        A length-1 mutable container that the AL1 outer code updates with the
        latest J2 value (single-stage coil penalty) so the iteration callback
        can record it alongside J1.
    """
    last_residuals = [None]

    def iteration_callback(nevals, x, residuals, objective):
        last_residuals[0] = np.asarray(residuals)
        J1 = float(objective)

        # Context banner
        ctx_bits = []
        if step is not None:
            if n_continuation_steps is not None:
                ctx_bits.append(f"continuation step {step+1}/{n_continuation_steps}")
            else:
                ctx_bits.append(f"continuation step {step+1}")
        if al_iter is not None:
            if al_outer_iter is not None:
                ctx_bits.append(f"AL1 iter {al_iter+1}/{al_outer_iter}")
            else:
                ctx_bits.append(f"AL1 iter {al_iter+1}")
        ctx_bits.append(f"max_mode={max_mode}")
        ctx_str = ", ".join(ctx_bits)

        proc0_print(SEP_THIN)
        proc0_print(f"  [{ctx_str}]")
        if is_qss:
            J2_latest = None
            if j2_holder is not None and len(j2_holder) > 0:
                J2_latest = j2_holder[0]
            if J2_latest is not None:
                proc0_print(
                    f"  Surface iter {nevals} (max_mode={max_mode})  "
                    f"J1 = {J1:.6e}  J2 = {J2_latest:.6e}")
            else:
                proc0_print(
                    f"  Surface iter {nevals} (max_mode={max_mode})  J1 = {J1:.6e}")
        else:
            proc0_print(f"  Surface iter {nevals} (max_mode={max_mode})  J = {J1:.6e}")

        if history is not None:
            J2_record = 0.0
            if is_qss and j2_holder is not None and len(j2_holder) > 0 and j2_holder[0] is not None:
                J2_record = float(j2_holder[0])
            history.append({
                "J1": J1,
                "J2": J2_record,
                "grad_J1": None,
                "grad_J2": None,
                "step": step,
                "al_iter": al_iter,
                "max_mode": max_mode,
                "nevals": int(nevals),
            })

        # ---- VTK frame export (every surface iteration) ----
        if vtk_exporter is not None and is_proc0 and surf is not None and curves is not None:
            extra = surface_vtk_point_data(surf, bs) if bs is not None else None
            frame_idx = vtk_exporter.next_frame()
            vtk_exporter.export_frame(
                surf, curves, frame_idx,
                surface_extra_data=extra,
                only_on_proc0=True, is_proc0=is_proc0,
            )

    def jac_callback(nevals, x, jac):
        r = last_residuals[0]
        if r is None:
            return
        jac = np.asarray(jac)
        r = np.atleast_1d(r)
        if jac.shape[0] == len(r):
            grad = 2.0 * (r @ jac)
        else:
            grad = 2.0 * (jac @ r)
        n1 = float(np.linalg.norm(grad))
        proc0_print(f"  ||grad J|| = {n1:.6e}")
        if history is not None and len(history) > 0:
            history[-1]["grad_J1"] = n1

    return iteration_callback, jac_callback
