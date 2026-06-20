"""Validation of MembraneAssumedStrainCondition — the Guo (2020) assumed-membrane-strain
Hellinger-Reissner treatment of membrane locking.

Paired with a base displacement shell, it suppresses the displacement membrane block and
re-supplies the membrane through an independent strain field on coarser anisotropic bases,
kept as global sparse auxiliary DOFs (no condensation). It removes membrane locking and
recovers a *smooth* membrane force from the strain field (``recover_membrane_force``) — exact
on a flat patch, and oscillation-free where the (untreated) displacement recovery locks.

Because the treated element reports zero membrane stress (it no longer owns the membrane),
the recovery is compared against a *pure* shell with no treatment: on a flat patch the pure
displacement recovery is itself exact (no locking), while on a curved shell it oscillates.
"""
import numpy as np

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


def _flat_problem(use_condition):
    """Flat plate under uniaxial strain (u_x = 0.1 on the x=L edge, u_y=u_z=0)."""
    L, E, t, n = 10.0, 1000.0, 0.1, 6
    patch = ck.SurfacePatch.rectangle(L, L, nu=n, nv=n, deg=2, name="plate")
    base = ck.ShellReissnerMindlinHier5p(ck.PlaneStress2d(E, 0.0, t, 1.0))
    quad = ck.GaussLegendre.from_patch(patch)
    prob = ck.LinearElasticProblem([patch], base, quad)
    cond = None
    if use_condition:
        cond = ck.MembraneAssumedStrainCondition(patch, base, quad)
        prob.add_condition(cond, patch="plate")
    nd, ncp = 5, patch.num_control_pts
    x0 = [int(c) for c in patch.boundary(0, True).displacement_dofs]
    xL = [int(c) for c in patch.boundary(0, False).displacement_dofs]
    zero = [cp * nd + 1 for cp in range(ncp)] + [cp * nd + 2 for cp in range(ncp)] \
           + [cp * nd + 0 for cp in x0]
    prob.add_constraint(ck.DirectConstraint(sorted(set(zero)), value=0.0))
    prob.add_constraint(ck.DirectConstraint([cp * nd + 0 for cp in xL], value=0.1))
    return prob, base, cond, patch


def _curved_problem(use_condition, n_cp=9):
    """Scordelis-Lo roof under self-weight (rigid diaphragms on the arc edges)."""
    patch = _roof_patch(n_cp)
    base = ck.ShellReissnerMindlinHier5p(_mat())
    quad = ck.GaussLegendre.from_patch(patch)
    prob = ck.LinearElasticProblem([patch], base, quad)
    cond = None
    if use_condition:
        cond = ck.MembraneAssumedStrainCondition(patch, base, quad)
        prob.add_condition(cond, patch="roof")
    prob.add_domain_load(np.array([0.0, 0.0, -90.0]), patch="roof")
    diaphragm = set()
    for at_start in (True, False):
        diaphragm.update(int(c) for c in patch.boundary(1, at_start).displacement_dofs)
    prob.add_constraint(ck.DirectConstraint([cp * 5 + c for cp in diaphragm for c in (0, 2)], value=0.0))
    return prob, base, cond, patch


def _crown_path(n=60):
    return np.column_stack([np.linspace(0.0, 1.0, n), np.full(n, 0.5)])


# === Tests ==========================================================================

def test_scordelis_unlocks():
    """The condition removes membrane locking: near the 0.3024 reference at a coarse mesh,
    far above the locked pure shell."""
    def wz(prob, base, patch):
        u = ck.solve(prob)
        d = ck.Function(u, base, patch, ck.FieldType.DISPLACEMENT)(np.array([[0.0, 0.5]]))
        return abs(np.asarray(d).reshape(-1, 3)[0, 2])

    tt, tb, _, tp = _curved_problem(True)
    treated = wz(tt, tb, tp)
    pt, pb, _, pp = _curved_problem(False)
    pure = wz(pt, pb, pp)
    assert 0.29 < treated < 0.305      # locking-free, near 0.3024
    assert treated > 1.3 * pure        # clearly unlocks the pure (locked) shell


def test_recovery_flat_is_exact():
    """On a flat plate (no locking) the field-recovered membrane force is constant and equals
    the pure-shell displacement recovery (which is itself exact there)."""
    params = np.column_stack([np.linspace(0.2, 0.8, 12), np.full(12, 0.5)])

    prob_t, _, cond, _ = _flat_problem(True)
    u_full = ck.solve(prob_t, full=True)
    n_field = np.asarray(cond.recover_membrane_force(u_full, params))[:, 0]

    prob_p, base_p, _, patch_p = _flat_problem(False)
    u_p = ck.solve(prob_p)
    n_pure = np.asarray(ck.Function(u_p, base_p, patch_p, ck.FieldType.TRACTION)(params))[:, 0]

    assert n_field.std() < 1e-12                       # constant recovered stress (patch test)
    assert n_pure.std() < 1e-12                        # pure recovery is also exact on flat
    assert abs(n_field.mean() - n_pure.mean()) <= 1e-9 * abs(n_pure.mean())


def test_recovery_smooth_on_curved():
    """On the locking-prone curved roof, the field-recovered membrane force is far smoother
    than the locked pure-shell displacement-gradient recovery."""
    params = _crown_path()

    prob_t, _, cond, _ = _curved_problem(True)
    u_full = ck.solve(prob_t, full=True)
    n_field = np.asarray(cond.recover_membrane_force(u_full, params))[:, 0]

    prob_p, base_p, _, patch_p = _curved_problem(False)
    u_p = ck.solve(prob_p)
    n_pure = np.asarray(ck.Function(u_p, base_p, patch_p, ck.FieldType.TRACTION)(params))[:, 0]

    rough = lambda y: np.abs(np.diff(y, 2)).sum()
    assert rough(n_field) < 0.05 * rough(n_pure)       # field is dramatically smoother
