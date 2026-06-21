"""Validation of MixedMembraneStrainShell — the Guo (2020) assumed-membrane-strain
Hellinger-Reissner treatment of membrane locking, re-homed from a Condition into a
faithful, global, uncondensed element that wraps any base displacement shell.

The element suppresses the base membrane block at the source and re-supplies it through
an independent strain field on coarser anisotropic bases, kept as global sparse auxiliary
DOFs (no condensation). It removes membrane locking, recovers a *smooth* membrane force
from the field — exact on a flat patch, oscillation-free where the pure displacement
recovery locks — and stays faithful through the standard ``Function`` recovery and a
Nitsche boundary condition (the membrane consistency flux couples to the field DOFs).
"""
import numpy as np
from scipy.sparse import csr_matrix

import pyck as ck


def _mat(E=4.32e8, nu=0.0, t=0.25):
    return ck.PlaneStress2d(E, nu, t, 1.0)


def _roof_patch(n_cp, radius=25.0, length=50.0, half_angle=40.0, deg=2):
    base = ck.SurfacePatch.quarter_cylinder(nu=n_cp, nv=n_cp, deg=deg, radius=radius,
                                            height=length, angle=2 * half_angle, name="roof")
    f = np.asarray(base.control_points)
    c, s = np.cos(np.radians(half_angle)), np.sin(np.radians(half_angle))
    new = np.column_stack([-f[:, 0] * s + f[:, 1] * c, f[:, 2], f[:, 0] * c + f[:, 1] * s])
    return ck.SurfacePatch(base.basis[0], base.basis[1], new, name="roof")


def _flat_problem(mixed):
    """Flat plate under uniaxial strain (u_x = 0.1 on the x=L edge, u_y=u_z=0).

    ``mixed`` selects the MixedMembraneStrainShell wrapper vs a pure base shell.
    """
    L, E, t, n = 10.0, 1000.0, 0.1, 6
    patch = ck.SurfacePatch.rectangle(L, L, nu=n, nv=n, deg=2, name="plate")
    base = ck.ShellReissnerMindlinHier5p(ck.PlaneStress2d(E, 0.0, t, 1.0))
    quad = ck.GaussLegendre.from_patch(patch)
    element = ck.MixedMembraneStrainShell(patch, base, quad) if mixed else base
    prob = ck.LinearElasticProblem([patch], element, quad)
    nd, ncp = 5, patch.num_control_pts
    x0 = [int(c) for c in patch.boundary(0, True).displacement_dofs]
    xL = [int(c) for c in patch.boundary(0, False).displacement_dofs]
    zero = [cp * nd + 1 for cp in range(ncp)] + [cp * nd + 2 for cp in range(ncp)] \
           + [cp * nd + 0 for cp in x0]
    prob.add_constraint(ck.DirectConstraint(sorted(set(zero)), value=0.0))
    prob.add_constraint(ck.DirectConstraint([cp * nd + 0 for cp in xL], value=0.1))
    return prob, element, patch


def _curved_problem(mixed, n_cp=9, nitsche=False, penalty=1.0e8):
    """Scordelis-Lo roof under self-weight (rigid diaphragms on the arc edges)."""
    patch = _roof_patch(n_cp)
    base = ck.ShellReissnerMindlinHier5p(_mat())
    quad = ck.GaussLegendre.from_patch(patch)
    element = ck.MixedMembraneStrainShell(patch, base, quad) if mixed else base
    prob = ck.LinearElasticProblem([patch], element, quad)
    prob.add_domain_load(np.array([0.0, 0.0, -90.0]), patch="roof")
    if nitsche:
        for at_start in (True, False):
            nit = ck.NitscheBoundaryCondition(patch.boundary(1, at_start),
                                              ck.GaussLegendre(4, dim=1))
            nit.add(ck.Field.U_X, penalty).add(ck.Field.U_Z, penalty)
            prob.add_condition(nit, patch="roof")
    else:
        diaphragm = set()
        for at_start in (True, False):
            diaphragm.update(int(c) for c in patch.boundary(1, at_start).displacement_dofs)
        prob.add_constraint(ck.DirectConstraint(
            [cp * 5 + c for cp in diaphragm for c in (0, 2)], value=0.0))
    return prob, element, patch


def _crown_path(n=60):
    return np.column_stack([np.linspace(0.0, 1.0, n), np.full(n, 0.5)])


def _wz(prob, element, patch):
    u = ck.solve(prob)
    d = ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)(np.array([[0.0, 0.5]]))
    return abs(np.asarray(d).reshape(-1, 3)[0, 2])


# === Tests ==========================================================================

def test_scordelis_unlocks():
    """The element removes membrane locking: near the 0.3024 reference at a coarse mesh,
    far above the locked pure shell."""
    tt, tb, tp = _curved_problem(True)
    treated = _wz(tt, tb, tp)
    pt, pb, pp = _curved_problem(False)
    pure = _wz(pt, pb, pp)
    assert 0.29 < treated < 0.305      # locking-free, near 0.3024
    assert treated > 1.3 * pure        # clearly unlocks the pure (locked) shell


def test_recovery_flat_is_exact():
    """On a flat plate (no locking) the field-recovered membrane force is constant and
    equals the pure-shell displacement recovery (which is itself exact there)."""
    params = np.column_stack([np.linspace(0.2, 0.8, 12), np.full(12, 0.5)])

    prob_t, mixed, _ = _flat_problem(True)
    u_full = ck.solve(prob_t, full=True)
    n_field = mixed.recover_membrane_force(u_full, params)[:, 0]

    prob_p, base_p, patch_p = _flat_problem(False)
    u_p = ck.solve(prob_p)
    n_pure = np.asarray(ck.Function(u_p, base_p, patch_p, ck.FieldType.TRACTION)(params))[:, 0]

    assert n_field.std() < 1e-12                       # constant recovered stress (patch test)
    assert n_pure.std() < 1e-12                        # pure recovery is also exact on flat
    assert abs(n_field.mean() - n_pure.mean()) <= 1e-9 * abs(n_pure.mean())


def test_recovery_smooth_on_curved():
    """On the locking-prone curved roof, the field-recovered membrane force is far smoother
    than the locked pure-shell displacement-gradient recovery."""
    params = _crown_path()

    prob_t, mixed, _ = _curved_problem(True)
    u_full = ck.solve(prob_t, full=True)
    n_field = mixed.recover_membrane_force(u_full, params)[:, 0]

    prob_p, base_p, patch_p = _curved_problem(False)
    u_p = ck.solve(prob_p)
    n_pure = np.asarray(ck.Function(u_p, base_p, patch_p, ck.FieldType.TRACTION)(params))[:, 0]

    rough = lambda y: np.abs(np.diff(y, 2)).sum()
    assert rough(n_field) < 0.05 * rough(n_pure)       # field is dramatically smoother


def test_function_traction_matches_recovery():
    """The faithful membrane force flows through the standard ``Function`` machinery:
    ``Function(u_full, mixed, patch, TRACTION)`` reproduces ``recover_membrane_force``."""
    params = _crown_path(30)
    prob, mixed, patch = _curved_problem(True)
    u_full = ck.solve(prob, full=True)

    n_method = mixed.recover_membrane_force(u_full, params)
    n_func = np.asarray(ck.Function(u_full, mixed, patch, ck.FieldType.TRACTION)(params))
    assert n_func.shape[1] == 5                         # [n11, n22, n12, q1, q2]
    assert np.max(np.abs(n_method - n_func[:, :3])) < 1e-9


def test_function_traction_needs_full_solution():
    """Querying the membrane force with only the physical (truncated) solution raises."""
    params = _crown_path(5)
    prob, mixed, patch = _curved_problem(True)
    u_disp = ck.solve(prob)            # truncated: no auxiliary ε̃ blocks
    try:
        ck.Function(u_disp, mixed, patch, ck.FieldType.TRACTION)(params)
    except Exception:
        return
    raise AssertionError("expected an error when the full solution is not supplied")


def test_nitsche_on_mixed():
    """A Nitsche boundary condition on the mixed element is faithful: the membrane
    consistency flux couples to the ε̃ field, the augmented system stays symmetric, and
    weak Dirichlet reproduces the strong-constraint solution."""
    prob_n, mixed, patch = _curved_problem(True, nitsche=True)
    K, _ = prob_n.assemble()
    K = csr_matrix(K)
    asym = abs(K - K.T)
    scale = abs(K).max()
    assert (asym.max() if asym.nnz else 0.0) <= 1e-9 * scale    # symmetric augmented system

    w_weak = _wz(prob_n, mixed, patch)
    prob_s, ms, ps = _curved_problem(True, nitsche=False)
    w_strong = _wz(prob_s, ms, ps)
    assert abs(w_weak - w_strong) <= 1e-3 * w_strong            # weak ≈ strong constraint
