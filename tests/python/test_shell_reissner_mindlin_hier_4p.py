"""Tests for the hierarchic four-parameter Reissner-Mindlin shell.

The element is built on the Kirchhoff-Love Cartesian displacement field plus a
twist potential, so its defining checks are:

* on a flat patch it must reproduce ``ShellReissnerMindlin4p`` exactly (both
  reduce to the same two-parameter plate), and
* it must converge to ``ShellKirchhoffLove3p`` as the shell thins (the
  shear-deflection correction is O(t^2)).

Both are verified by solving a real cantilever BVP and comparing recovered
displacement fields, not stiffness matrices.
"""

import numpy as np
import pyck as ck


L, W = 8.0, 4.0
E, NU = 1.0e7, 0.3
FZ = -1.0
AD, AR = 1.0e12, 1.0e10


def _cantilever_tip_w(element, nd, thickness, *, pin_psi):
    """Tip transverse deflection of a flat cantilever plate clamped at x=0 and
    loaded by a transverse tip traction at x=L."""
    patch = ck.SurfacePatch.rectangle(L, W, nu=12, nv=6, deg=3,
                                      cx=L / 2, cy=W / 2, name="m")
    prob = ck.LinearElasticProblem([patch], element,
                                   ck.GaussLegendre.from_patch(patch))

    clamp = ck.PenaltyBoundaryCondition(patch.boundary(0, True),
                                        ck.GaussLegendre(4, dim=1))
    clamp.add(ck.Field.U_X, AD).add(ck.Field.U_Y, AD).add(ck.Field.U_Z, AD)
    clamp.add(ck.Field.ROT_N, AR)
    prob.add_condition(clamp, patch="m")

    load = ck.LoadBoundaryCondition(patch.boundary(0, False),
                                    ck.GaussLegendre(4, dim=1))
    load.add(ck.Field.U_Z, FZ)
    prob.add_condition(load, patch="m")

    n = patch.num_control_pts
    if pin_psi:
        # Suppress the twist field (zero by symmetry for this bending problem;
        # also removes the constant-psi zero-energy mode).
        prob.add_constraint(ck.DirectConstraint([cp * nd + 3 for cp in range(n)],
                                                value=0.0))

    u = ck.solve(prob)
    w = np.asarray(ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)(
        np.array([[1.0, 0.5]]))).reshape(-1, 3)[0, 2]
    return w


def test_hier_flat_matches_rm4p():
    """On a flat patch the hierarchic shell and the covariant
    ``ShellReissnerMindlin4p`` are the same element (the transverse field lives in
    the same B-spline space, with identical bending and shear operators); their
    tip deflections must agree to solver precision."""
    th = 0.1
    rm4 = _cantilever_tip_w(ck.ShellReissnerMindlin4p(ck.PlaneStress2d(E, NU, th)),
                            4, th, pin_psi=True)
    hier = _cantilever_tip_w(ck.ShellReissnerMindlinHier4p(ck.PlaneStress2d(E, NU, th)),
                             4, th, pin_psi=True)
    assert abs(hier - rm4) <= 1e-9 * abs(rm4)


def test_hier_converges_to_kirchhoff_love_when_thin():
    """As the shell thins the shear-deflection correction vanishes and the
    hierarchic shell must approach the Kirchhoff-Love result; at finite thickness
    it is strictly more compliant (shear adds flexibility)."""
    kl_thin = _cantilever_tip_w(ck.ShellKirchhoffLove3p(ck.PlaneStress2d(E, NU, 0.02)),
                                3, 0.02, pin_psi=False)
    hier_thin = _cantilever_tip_w(ck.ShellReissnerMindlinHier4p(ck.PlaneStress2d(E, NU, 0.02)),
                                  4, 0.02, pin_psi=True)
    assert abs(hier_thin - kl_thin) <= 1e-3 * abs(kl_thin)

    kl = _cantilever_tip_w(ck.ShellKirchhoffLove3p(ck.PlaneStress2d(E, NU, 0.1)),
                           3, 0.1, pin_psi=False)
    hier = _cantilever_tip_w(ck.ShellReissnerMindlinHier4p(ck.PlaneStress2d(E, NU, 0.1)),
                             4, 0.1, pin_psi=True)
    assert abs(hier) > abs(kl)                 # shear softens the response
    assert abs(hier - kl) <= 1e-2 * abs(kl)    # but only slightly for a thin plate


def test_hier_flat_rigid_body_is_strain_free():
    """A rigid Cartesian translation produces no strain on a flat patch (A_3 is
    constant, so the recovered shear deflection w_s = -(Kb/Ks) Laplacian(v.A_3)
    vanishes)."""
    el = ck.ShellReissnerMindlinHier4p(ck.PlaneStress2d(E, NU, 0.1))
    patch = ck.SurfacePatch.rectangle(L, W, nu=8, nv=6, deg=3, name="m")
    params = np.array([[0.41, 0.63]])
    B = el.strain_matrix(patch, params)        # (8, 4*ncp)

    n = patch.num_control_pts
    u = np.zeros(4 * n)
    for cp in range(n):                        # translation (1, 2, 3), psi = 0
        u[4 * cp + 0], u[4 * cp + 1], u[4 * cp + 2] = 1.0, 2.0, 3.0

    strain = B @ u
    assert np.abs(strain).max() <= 1e-9 * np.abs(B).max()


def test_hier_twist_does_no_membrane_work():
    """The twist potential psi enters only the bending and shear measures, never
    the membrane strain: the psi columns of the membrane rows are exactly zero."""
    el = ck.ShellReissnerMindlinHier4p(ck.PlaneStress2d(E, NU, 0.1))
    patch = ck.SurfacePatch.rectangle(L, W, nu=8, nv=6, deg=3, name="m")
    B = el.strain_matrix(patch, np.array([[0.41, 0.63]]))
    membrane_psi = B[0:3, 3::4]                # rows eps_11, eps_22, 2eps_12; psi cols
    assert np.abs(membrane_psi).max() == 0.0
