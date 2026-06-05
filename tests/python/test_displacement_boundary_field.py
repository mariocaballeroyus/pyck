"""Tests for the Cartesian displacement-component boundary values
(``Field.U_X/U_Y/U_Z``).

Each is a scalar essential value giving one global Cartesian component of the
recovered displacement ``u_C`` (via ``Element::displacement_shape_matrix``), so
they compose: clamp the components you want and leave the rest free.
"""

import numpy as np
import pyck as ck


def _displacement(u, el, patch, params):
    return np.asarray(
        ck.Function(u, el, patch, ck.FieldType.DISPLACEMENT)(params)
    ).reshape(-1, 3)


def test_displacement_components_clamp_all():
    """Penalty-clamping U_X, U_Y, U_Z on an edge holds all three components to
    zero there, while an axial force on the opposite edge produces the
    closed-form uniaxial-tension displacement."""
    Lx, Ly, E, nu, h, fx = 4.0, 1.0, 1.0e7, 0.3, 0.1, 200.0
    ND = 3
    patch = ck.SurfacePatch.rectangle(Lx, Ly, nu=10, nv=10, deg=3, name="strip")
    el = ck.ShellKirchhoffLove3p(ck.PlaneStress2d(E, nu, h))
    prob = ck.LinearElasticProblem([patch], el, ck.GaussLegendre.from_patch(patch))
    N = patch.num_control_pts

    # Suppress transverse bending DOFs (decoupled on a flat membrane problem).
    cps = np.asarray(patch.control_points)
    prob.add_constraint(ck.DirectConstraint([cp * ND + 2 for cp in range(N)], value=0.0))

    # Clamp all three displacement components on the u = 0 edge (penalty).
    pen = ck.PenaltyBoundaryCondition(patch.boundary(0, True), ck.GaussLegendre(4, dim=1))
    for comp in (ck.Field.U_X, ck.Field.U_Y, ck.Field.U_Z):
        pen.add(comp, 1.0e12)
    prob.add_condition(pen)

    # Axial tension on the u = 1 edge.
    lc = ck.LoadBoundaryCondition(patch.boundary(0, False), ck.GaussLegendre(4, dim=1))
    lc.add(ck.Field.U_X, fx)
    prob.add_condition(lc)

    K, f = prob.assemble()
    u = ck.solve(K, f, physical_dofs=ND * N)

    clamped = _displacement(u, el, patch, np.array([[0.0, v] for v in np.linspace(0.2, 0.8, 5)]))
    loaded = _displacement(u, el, patch, np.array([[1.0, v] for v in np.linspace(0.2, 0.8, 5)]))
    assert np.abs(clamped).max() < 1e-6 * (fx * Lx / (E * h))      # all components held
    assert abs(loaded[:, 0].mean() - fx * Lx / (E * h)) / (fx * Lx / (E * h)) < 1e-2


def test_partial_clamp_leaves_other_components_free():
    """Clamping only U_Z on an edge holds the out-of-plane component there but
    leaves the in-plane components free to respond to an in-plane edge force."""
    Lx, Ly, E, nu, h, fx = 4.0, 1.0, 1.0e7, 0.3, 0.1, 200.0
    ND = 3
    patch = ck.SurfacePatch.rectangle(Lx, Ly, nu=10, nv=10, deg=3, name="strip")
    el = ck.ShellKirchhoffLove3p(ck.PlaneStress2d(E, nu, h))
    prob = ck.LinearElasticProblem([patch], el, ck.GaussLegendre.from_patch(patch))
    N = patch.num_control_pts
    prob.add_constraint(ck.DirectConstraint([cp * ND + 2 for cp in range(N)], value=0.0))

    # Clamp the full in-plane motion on the u = 0 edge (so the system is well-posed)
    # but only uz on the u = 1 edge; pull u = 1 in +x.
    left = ck.PenaltyBoundaryCondition(patch.boundary(0, True), ck.GaussLegendre(4, dim=1))
    left.add(ck.Field.U_X, 1.0e12).add(ck.Field.U_Y, 1.0e12)
    prob.add_condition(left)
    right_clamp = ck.PenaltyBoundaryCondition(patch.boundary(0, False), ck.GaussLegendre(4, dim=1))
    right_clamp.add(ck.Field.U_Z, 1.0e12)
    prob.add_condition(right_clamp)
    lc = ck.LoadBoundaryCondition(patch.boundary(0, False), ck.GaussLegendre(4, dim=1))
    lc.add(ck.Field.U_X, fx)
    prob.add_condition(lc)

    K, f = prob.assemble()
    u = ck.solve(K, f, physical_dofs=ND * N)

    loaded = _displacement(u, el, patch, np.array([[1.0, v] for v in np.linspace(0.2, 0.8, 5)]))
    assert np.abs(loaded[:, 2]).max() < 1e-6 * (fx * Lx / (E * h))   # uz held at the loaded edge
    assert loaded[:, 0].mean() > 0.5 * (fx * Lx / (E * h))          # ux free and stretched


def test_displacement_component_is_cartesian_on_covariant_element():
    """U_X/U_Y/U_Z read the recovered Cartesian displacement, not a raw DOF slot,
    so they give true u_x/u_y/u_z even on ShellReissnerMindlin4p whose first two
    slots are covariant in-plane components. The clamp must therefore drive the
    physical Cartesian displacement (not slot values) to zero."""
    R, Lz, t, E, nu = 1.0, 0.5, 0.02, 1.0e7, 0.3
    ND = 4
    patch = ck.SurfacePatch.quarter_cylinder(
        nu=8, nv=8, deg=4, radius=R, height=Lz, angle=90.0, name="cyl")
    el = ck.ShellReissnerMindlin4p(ck.PlaneStress2d(E, nu, t))
    prob = ck.LinearElasticProblem([patch], el, ck.GaussLegendre.from_patch(patch))
    N = patch.num_control_pts

    # Pin the twist potential everywhere (kills the zero-energy psi mode) and clamp
    # all three Cartesian displacement components on one edge via penalty.
    prob.add_constraint(ck.DirectConstraint([cp * ND + 3 for cp in range(N)], value=0.0))
    pen = ck.PenaltyBoundaryCondition(patch.boundary(0, True), ck.GaussLegendre(5, dim=1))
    for comp in (ck.Field.U_X, ck.Field.U_Y, ck.Field.U_Z):
        pen.add(comp, 1.0e13)
    prob.add_condition(pen)
    lc = ck.LoadBoundaryCondition(patch.boundary(0, False), ck.GaussLegendre(5, dim=1))
    lc.add(ck.Field.U_Z, 1.0)
    prob.add_condition(lc)

    K, f = prob.assemble()
    u = ck.solve(K, f, physical_dofs=ND * N)

    clamped = _displacement(u, el, patch, np.array([[0.0, v] for v in np.linspace(0.2, 0.8, 5)]))
    loaded = _displacement(u, el, patch, np.array([[1.0, v] for v in np.linspace(0.2, 0.8, 5)]))
    ref = np.abs(loaded).max()
    assert ref > 0                                  # the load actually moves the free edge
    assert np.abs(clamped).max() < 1e-4 * ref        # physical Cartesian u ~ 0 at the clamp
