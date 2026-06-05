"""Tests for force and moment boundary loads on ``LoadBoundaryCondition``.

A force conjugates to the full 3D displacement, so its three global Cartesian
components ``Field.U_X/U_Y/U_Z`` cover both in-plane (membrane / axial) and
out-of-plane (transverse shear) loading; the element's internal membrane/shear
split is irrelevant to the load. A moment conjugates to the rotation:
``Field.ROT_N`` (bending) and ``Field.ROT_S`` (twisting).
"""

import numpy as np
import pyck as ck


def _assemble_load(patch, element, build_condition):
    """Assemble (K, f) for a problem whose only condition is the boundary load
    produced by ``build_condition(LoadBoundaryCondition)``."""
    prob = ck.LinearElasticProblem([patch], element, ck.GaussLegendre.from_patch(patch))
    lc = ck.LoadBoundaryCondition(patch.boundary(0, False), ck.GaussLegendre(4, dim=1))
    build_condition(lc)
    prob.add_condition(lc)
    return prob.assemble()


def test_force_traction_global_resultant():
    """A global force per unit length integrates to (force x edge_length) in each
    Cartesian component, scattered onto the matching displacement DOF slots."""
    Lx, Ly = 2.0, 3.0
    ND = 3  # ShellKirchhoffLove3p: slots (u_x, u_y, u_z)
    patch = ck.SurfacePatch.rectangle(Lx, Ly, nu=5, nv=5, deg=3, name="plate")
    el = ck.ShellKirchhoffLove3p(ck.PlaneStress2d(1.0e7, 0.3, 0.1))
    N = patch.num_control_pts

    t = np.array([4.0, -7.0, 11.0])
    K, f = _assemble_load(patch, el, lambda lc: (lc.add(ck.Field.U_X, t[0])
                                                   .add(ck.Field.U_Y, t[1])
                                                   .add(ck.Field.U_Z, t[2])))
    f = np.asarray(f).ravel()[: ND * N]

    edge_len = Ly  # boundary(0, False) is the x = +Lx/2 edge, spanning y in [0, Ly]
    for slot in range(3):
        assert abs(f[slot::ND].sum() - t[slot] * edge_len) < 1e-9 * edge_len * abs(t).max()


def test_force_traction_membrane_uniaxial():
    """An in-plane (membrane / axial) edge force produces the closed-form
    uniaxial-tension displacement u_x = f_x L / (E t)."""
    Lx, Ly, E, nu, h, fx = 4.0, 1.0, 1.0e7, 0.3, 0.1, 200.0
    ND = 3
    patch = ck.SurfacePatch.rectangle(Lx, Ly, nu=10, nv=10, deg=3, name="strip")
    el = ck.ShellKirchhoffLove3p(ck.PlaneStress2d(E, nu, h))
    prob = ck.LinearElasticProblem([patch], el, ck.GaussLegendre.from_patch(patch))
    N = patch.num_control_pts
    cps = np.asarray(patch.control_points)

    # Suppress all transverse DOFs (decoupled from membrane on a flat patch) and
    # clamp the in-plane DOFs on the u = 0 edge; load the u = 1 edge in +x.
    x0 = cps[:, 0].min()
    left = {cp for cp in range(N) if abs(cps[cp, 0] - x0) < 1e-9}
    fixed = []
    for cp in range(N):
        fixed.append(cp * ND + 2)
        if cp in left:
            fixed += [cp * ND + 0, cp * ND + 1]
    prob.add_constraint(ck.DirectConstraint(sorted(set(fixed)), value=0.0))

    lc = ck.LoadBoundaryCondition(patch.boundary(0, False), ck.GaussLegendre(4, dim=1))
    lc.add(ck.Field.U_X, fx)
    prob.add_condition(lc)

    K, f = prob.assemble()
    u = ck.solve(K, f, physical_dofs=ND * N)

    disp = np.asarray(
        ck.Function(u, el, patch, ck.FieldType.DISPLACEMENT)(
            np.array([[1.0, v] for v in np.linspace(0.2, 0.8, 5)])  # parametric: loaded edge
        )
    ).reshape(-1, 3)
    ux_exact = fx * Lx / (E * h)
    assert abs(disp[:, 0].mean() - ux_exact) / ux_exact < 1e-2
    assert abs(disp[:, 2]).max() < 1e-12  # no out-of-plane motion


def test_moment_load_assembles_and_superposes():
    """A moment load is conjugate to the rotation (Field.ROT_N bending, Field.ROT_S
    twisting). Prescribing it on a Reissner-Mindlin edge assembles a finite,
    nonzero load vector, and the combined load is the sum of the components
    (the external-work integral is linear in the prescribed moment)."""
    patch = ck.SurfacePatch.quarter_cylinder(
        nu=6, nv=6, deg=3, radius=1.0, height=0.5, angle=90.0, name="cyl")
    el = ck.ShellReissnerMindlin5p(ck.PlaneStress2d(1.0e7, 0.3, 0.02))

    _, f_nn = _assemble_load(patch, el, lambda lc: lc.add(ck.Field.ROT_N, 3.0))
    _, f_ns = _assemble_load(patch, el, lambda lc: lc.add(ck.Field.ROT_S, -2.0))
    _, f_both = _assemble_load(
        patch, el, lambda lc: lc.add(ck.Field.ROT_N, 3.0).add(ck.Field.ROT_S, -2.0))

    f_both = np.asarray(f_both).ravel()
    assert np.isfinite(f_both).all()
    assert np.linalg.norm(f_both) > 0.0
    assert np.allclose(f_both, np.asarray(f_nn).ravel() + np.asarray(f_ns).ravel(),
                       atol=1e-14, rtol=0.0)
