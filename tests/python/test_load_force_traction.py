"""Tests for the unified force-traction boundary load (``add_force``) and the
``add_moment`` convenience on ``LoadBoundaryCondition``.

A force conjugates to the full 3D displacement field, so a single 3-vector
covers both in-plane (membrane / axial) and out-of-plane (transverse shear)
loading; the element's internal membrane/shear split is irrelevant to the load.
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
    K, f = _assemble_load(patch, el, lambda lc: lc.add_force(t))
    f = np.asarray(f).ravel()[: ND * N]

    edge_len = Ly  # boundary(0, False) is the x = +Lx/2 edge, spanning y in [0, Ly]
    for slot in range(3):
        assert abs(f[slot::ND].sum() - t[slot] * edge_len) < 1e-9 * edge_len * abs(t).max()


def test_force_traction_local_equals_transverse_displacement():
    """On a curved surface, the local-frame a_3 component of a force reproduces
    the transverse-displacement ('w' = n.u) load to machine precision, while the
    global z-component does not (the director is not aligned with global z)."""
    patch = ck.SurfacePatch.quarter_cylinder(
        nu=6, nv=6, deg=3, radius=1.0, height=0.5, angle=90.0, name="cyl")
    el = ck.ShellReissnerMindlin5p(ck.PlaneStress2d(1.0e7, 0.3, 0.02))
    Q = 7.0

    _, f_w = _assemble_load(patch, el, lambda lc: lc.add("w", Q))
    _, f_local = _assemble_load(patch, el, lambda lc: lc.add_force([0, 0, Q], frame="local"))
    _, f_global = _assemble_load(patch, el, lambda lc: lc.add_force([0, 0, Q]))
    f_w = np.asarray(f_w).ravel()
    f_local = np.asarray(f_local).ravel()
    f_global = np.asarray(f_global).ravel()

    assert np.allclose(f_local, f_w, atol=1e-12, rtol=0.0)
    assert np.linalg.norm(f_global - f_w) > 1e-3 * np.linalg.norm(f_w)


def test_force_traction_membrane_uniaxial():
    """An in-plane (membrane / axial) edge force is now applicable and produces
    the closed-form uniaxial-tension displacement u_x = f_x L / (E t). This path
    was previously unreachable: the only displacement field was the out-of-plane
    'w', with no in-plane conjugate."""
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
    lc.add_force([fx, 0.0, 0.0])
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


def test_add_moment_matches_rotation_conjugate():
    """add_moment(m_n, m_s) is exactly the rot_n / rot_s scalar conjugates."""
    patch = ck.SurfacePatch.quarter_cylinder(
        nu=6, nv=6, deg=3, radius=1.0, height=0.5, angle=90.0, name="cyl")
    el = ck.ShellReissnerMindlin5p(ck.PlaneStress2d(1.0e7, 0.3, 0.02))

    _, f_pair = _assemble_load(
        patch, el, lambda lc: lc.add("rot_n", 3.0).add("rot_s", -2.0))
    _, f_moment = _assemble_load(
        patch, el, lambda lc: lc.add_moment(m_n=3.0, m_s=-2.0))
    assert np.allclose(np.asarray(f_pair).ravel(), np.asarray(f_moment).ravel(),
                       atol=1e-14, rtol=0.0)
