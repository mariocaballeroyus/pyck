"""Thick doubly-curved shell: RM-Hier-4p must reproduce the true RM-5p shell.

A *thick* (t/L = 0.2) partly-clamped hyperbolic paraboloid (saddle, Gaussian curvature
K < 0) under self-weight, half-modelled with a symmetry plane. At this thickness the response
is far from any thin-shell locking regime, so the rotation-free hierarchic RM-Hier-4p and the
standard 5-parameter Reissner-Mindlin shell must converge to the same answer.

This is the test that exposed the missing Weingarten term in the 4p transverse-shear strain:
the DOF contribution to the shear deflection is w_s = -(K_b/K_s)(Delta N_i) A_3(k), and the
shear strain gamma_alpha = d_alpha(w_s) must differentiate the space-varying normal A_3(k)
too. That term is O(curvature) and vanishes on a flat plate, so plate tests never caught it;
without it the 4p answer drifts *away* from the 5p reference as the mesh is refined.
"""
import numpy as np
import pytest

import pyck as ck

L = 1.0
E = 2.0e11
NU = 0.3
RHO = 8000.0
C_NIT = 300.0
W_POINT = np.array([[1.0, 0.0]])     # tip X=L/2, Y=0 on the symmetry plane


def _hypar_patch(deg: int, nel: int) -> ck.SurfacePatch:
    """Exact bi-quadratic Bezier saddle Z = X^2 - Y^2 on the half domain
    X in [-L/2, L/2], Y in [0, L/2], degree-elevated to `deg`, uniformly refined to
    nel x (nel//2) elements (square cells)."""
    h = L / 2.0
    xs = np.array([-h, 0.0, h])
    cx = np.array([h * h, -h * h, h * h])
    ys = np.array([0.0, h / 2.0, h])
    cy = np.array([0.0, 0.0, -h * h])
    cps = np.array([[xs[i], ys[j], cx[i] + cy[j]]
                    for j in range(3) for i in range(3)], dtype=float)
    b = ck.BSpline.clamped_uniform(2, 3)
    patch = ck.SurfacePatch(b, b, cps, name="hypar")
    patch = patch.elevate_degree(0, deg - 2).elevate_degree(1, deg - 2)
    nel_v = max(1, nel // 2)
    for k in range(1, nel):
        patch = patch.insert_knot(0, k / nel)
    for k in range(1, nel_v):
        patch = patch.insert_knot(1, k / nel_v)
    return patch


def _solve(deg: int, nel: int, t: float, element_cls):
    """Return (strain energy U, tip deflection magnitude |w|) for the saddle."""
    patch = _hypar_patch(deg, nel)
    element = element_cls(ck.PlaneStress2d(E, NU, t))
    gauss2 = ck.GaussLegendre(deg + 1, dim=2)
    prob = ck.LinearElasticProblem([patch], element, gauss2)
    prob.add_domain_load(np.array([0.0, 0.0, -RHO * t]))

    gauss1 = ck.GaussLegendre(deg + 1, dim=1)
    w_u, w_rot = C_NIT * E * t * nel, C_NIT * E * t**3 * nel
    clamp = ck.NitscheBoundaryCondition(patch.boundary(0, True), gauss1)
    clamp.add(ck.Field.U_X, w_u).add(ck.Field.U_Y, w_u).add(ck.Field.U_Z, w_u)
    clamp.add(ck.Field.ROT_N, w_rot).add(ck.Field.ROT_S, w_rot)
    prob.add_condition(clamp, patch="hypar")
    sym = ck.NitscheBoundaryCondition(patch.boundary(1, True), gauss1)
    sym.add(ck.Field.U_Y, w_u).add(ck.Field.ROT_N, w_rot)
    prob.add_condition(sym, patch="hypar")
    if element.num_node_dofs == 4:                       # pin the constant-psi null mode
        prob.add_constraint(ck.DirectConstraint([3], value=0.0))

    u_full = ck.solve(prob, full=True)
    K_energy, _ = ck.LinearElasticProblem([patch], element, gauss2).assemble()
    U = 2.0 * 0.5 * float(u_full @ (K_energy @ u_full))
    u = u_full[:prob.num_physical_dofs]
    w = abs(float(ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)(W_POINT)[0, 2]))
    return U, w


def test_thick_hypar_hier4p_matches_rm5p():
    """On a refined thick saddle the 4p element agrees with the 5p reference to <0.3%."""
    t, deg, nel = 0.2 * L, 4, 32
    U5, w5 = _solve(deg, nel, t, ck.ShellReissnerMindlin5p)
    U4, w4 = _solve(deg, nel, t, ck.ShellReissnerMindlinHier4p)
    assert abs(U4 - U5) <= 3.0e-3 * abs(U5)
    assert abs(w4 - w5) <= 3.0e-3 * abs(w5)


def test_thick_hypar_4p_5p_gap_shrinks_under_refinement():
    """The 4p->5p deflection gap must *decrease* with refinement (consistency): the missing
    Weingarten shear term made it grow instead."""
    t, deg = 0.2 * L, 4
    gaps = []
    for nel in (8, 16, 32):
        U5, w5 = _solve(deg, nel, t, ck.ShellReissnerMindlin5p)
        U4, w4 = _solve(deg, nel, t, ck.ShellReissnerMindlinHier4p)
        gaps.append(abs(w4 - w5) / abs(w5))
    assert gaps[1] < gaps[0] and gaps[2] < gaps[1]
