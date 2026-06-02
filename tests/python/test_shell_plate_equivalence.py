"""Shell <-> plate equivalence: same flat BVP, same transverse load, same answer.

Underwrites deprecating the dedicated plate elements in favour of the complete
shell formulations. For each shell/plate pair we assemble a LinearElasticProblem
on the SAME flat square with the SAME clamped boundary and the SAME uniform
transverse load, solve, and require the recovered transverse-deflection fields to
agree to floating-point precision. Pairings:

    ShellKirchhoffLove3p         <-> PlateKirchhoffLove1p        (Kirchhoff-Love)
    ShellReissnerMindlin5p       <-> PlateReissnerMindlin3p      (standard Reissner-Mindlin)
    ShellReissnerMindlin4p       <-> PlateReissnerMindlinDispl2p (rotation-free, w_b + psi)
    ShellReissnerMindlin4p|psi=0 <-> PlateReissnerMindlin1p      (rotation-free, w_b only)

The shells carry extra DOFs that a transverse plate problem must not excite, so
they are pinned to zero: the in-plane membrane (u_x,u_y / u_1,u_2) decouples on a
flat patch, and for the 1p comparison the 4p twist potential psi is set to zero
with a direct constraint (as if it did not exist), reducing it to the single
bending potential w_b.
"""

import numpy as np
import pyck as ck

E, NU, T = 1.0e7, 0.3, 0.1
DEG, N = 3, 6                  # cubic (C^2) satisfies every formulation's continuity
Q = -1.0                       # uniform transverse pressure
SAMPLE = np.array([[a, b] for a in np.linspace(0.15, 0.85, 6)
                          for b in np.linspace(0.15, 0.85, 6)])   # interior, parametric


def _flat_square():
    return ck.SurfacePatch.rectangle(1.0, 1.0, nu=N, nv=N, deg=DEG, name="flat")


def _boundary_cps(surf, n_rows):
    """Control-point indices of the outer `n_rows` rings (grid is u-inner:
    flat = i_u + i_v*nu, so the row adjacent to a boundary is a fixed stride off)."""
    nu = surf.basis[0].num_basis
    stride = {(0, True): 1, (0, False): -1, (1, True): nu, (1, False): -nu}
    cps = set()
    for pd in (0, 1):
        for st in (True, False):
            row0 = np.asarray(surf.boundary(pdim=pd, at_start=st).displacement_dofs, dtype=int)
            off = stride[(pd, st)]
            for r in range(n_rows):
                cps.update((row0 + r * off).tolist())
    return cps


def _solve_transverse(element, surf, ndof, *, clamp_slots, clamp_rows,
                      zero_everywhere=()):
    """Clamp `clamp_slots` on the outer `clamp_rows` rings, zero `zero_everywhere`
    slots on every node, apply a uniform transverse load, solve, and return the
    recovered transverse deflection at the interior sample points."""
    prob = ck.LinearElasticProblem([surf], element, ck.GaussLegendre.from_patch(surf))
    prob.add_domain_load(np.array([0.0, 0.0, Q]))

    fixed = set()
    for cp in _boundary_cps(surf, clamp_rows):
        for s in clamp_slots:
            fixed.add(cp * ndof + s)
    for cp in range(surf.num_control_pts):
        for s in zero_everywhere:
            fixed.add(cp * ndof + s)
    prob.add_constraint(ck.DirectConstraint(sorted(fixed), 0.0))

    K, f = prob.assemble()
    u = ck.solve(K, f, physical_dofs=ndof * surf.num_control_pts)
    return np.asarray(ck.Function(u, element, surf, ck.FieldType.DISPLACEMENT)(SAMPLE))[:, 2]


def _assert_match(w_shell, w_plate, tol):
    scale = np.abs(w_plate).max()
    assert scale > 1e-30                                  # the load actually deflects the plate
    assert np.abs(w_shell - w_plate).max() / scale < tol


def test_kl_shell_matches_kl_plate():
    """Clamped flat plate, transverse load: ShellKirchhoffLove3p (u_z, membrane
    pinned) gives the same deflection as PlateKirchhoffLove1p (w)."""
    mat = ck.PlaneStress2d(E, NU, T)
    surf = _flat_square()
    w_plate = _solve_transverse(ck.PlateKirchhoffLove1p(mat), surf, 1,
                                clamp_slots=[0], clamp_rows=2)
    w_shell = _solve_transverse(ck.ShellKirchhoffLove3p(mat), surf, 3,
                                clamp_slots=[2], clamp_rows=2, zero_everywhere=[0, 1])
    _assert_match(w_shell, w_plate, 1e-11)


def test_5p_shell_matches_rm3p_plate():
    """Clamped flat plate, transverse load: ShellReissnerMindlin5p (u_z + rotations,
    membrane pinned) gives the same deflection as PlateReissnerMindlin3p (w + rotations)."""
    mat = ck.PlaneStress2d(E, NU, T)
    surf = _flat_square()
    w_plate = _solve_transverse(ck.PlateReissnerMindlin3p(mat), surf, 3,
                                clamp_slots=[0, 1, 2], clamp_rows=1)
    w_shell = _solve_transverse(ck.ShellReissnerMindlin5p(mat), surf, 5,
                                clamp_slots=[2, 3, 4], clamp_rows=1, zero_everywhere=[0, 1])
    _assert_match(w_shell, w_plate, 1e-11)


def test_4p_shell_matches_rm2p_plate():
    """Clamped flat plate, transverse load: ShellReissnerMindlin4p (w_b, psi;
    membrane pinned) gives the same deflection as PlateReissnerMindlinDispl2p (w_b, psi)."""
    mat = ck.PlaneStress2d(E, NU, T)
    surf = _flat_square()
    w_plate = _solve_transverse(ck.PlateReissnerMindlinDispl2p(mat), surf, 2,
                                clamp_slots=[0, 1], clamp_rows=2)
    w_shell = _solve_transverse(ck.ShellReissnerMindlin4p(mat), surf, 4,
                                clamp_slots=[2, 3], clamp_rows=2, zero_everywhere=[0, 1])
    _assert_match(w_shell, w_plate, 1e-11)


def test_4p_shell_with_psi_zero_matches_rm1p_plate():
    """Clamped flat plate, transverse load: ShellReissnerMindlin4p with the twist
    potential psi set to zero (direct constraint, as if it did not exist) reduces
    to the single bending potential w_b and matches PlateReissnerMindlin1p (w_b)."""
    mat = ck.PlaneStress2d(E, NU, T)
    surf = _flat_square()
    w_plate = _solve_transverse(ck.PlateReissnerMindlin1p(mat), surf, 1,
                                clamp_slots=[0], clamp_rows=2)
    w_shell = _solve_transverse(ck.ShellReissnerMindlin4p(mat), surf, 4,
                                clamp_slots=[2], clamp_rows=2, zero_everywhere=[0, 1, 3])
    _assert_match(w_shell, w_plate, 1e-11)
