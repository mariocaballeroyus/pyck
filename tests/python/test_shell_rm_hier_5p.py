"""Validation of `ck.ShellReissnerMindlinHier5p`.

References: 
1. R. Echter, B. Oesterle, M. Bischoff, "A hierarchic family of
   isogeometric shell finite elements", CMAME 254 (2013) 170-180.
"""
from __future__ import annotations

import numpy as np
import pytest

import pyck as ck


# =============================================================================
# Case 1 - Simply supported (Navier) square plate (paper sec. 5.1, Table 2)
# =============================================================================

# 5p-hier. row of Table 2: center deflection w_z,max vs slenderness L/t.
NAVIER_PLATE_REF = {
    5: 0.5839,
    10: 0.4938,
    100: 0.4431,
    1000: 0.4423,
    10000: 0.4423,
}


def _solve_navier_plate(slenderness: int, num_cp: int = 12, deg: int = 2) -> float:
    """Center deflection of a simply supported square plate. Boundary conditions via 
    strong constraints.
    """
    L, E, nu = 10.0, 1000.0, 0.3
    t = L / slenderness

    material = ck.PlaneStress2d(E, nu, t, 1.0)
    element = ck.ShellReissnerMindlinHier5p(material)
    patch = ck.SurfacePatch.rectangle(L, L, nu=num_cp, nv=num_cp, deg=deg, name="plate")

    problem = ck.LinearElasticProblem([patch], element, ck.GaussLegendre.from_patch(patch))
    problem.add_domain_load(np.array([0.0, 0.0, t**3]), patch="plate")

    edge_cps: set[int] = set()
    for parametric_dir in (0, 1):
        for side in (True, False):
            edge_cps.update(int(c) for c in patch.boundary(parametric_dir, side).displacement_dofs)
    pinned = [cp * 5 + comp for cp in edge_cps for comp in (0, 1, 2)]
    problem.add_constraint(ck.DirectConstraint(sorted(set(pinned)), value=0.0))

    u = ck.solve(problem)
    center = ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)(np.array([[0.5, 0.5]]))
    return abs(np.asarray(center).reshape(-1, 3)[0, 2])


@pytest.mark.parametrize("slenderness", list(NAVIER_PLATE_REF))
def test_navier_plate(slenderness: int) -> None:
    w = _solve_navier_plate(slenderness)
    assert w == pytest.approx(NAVIER_PLATE_REF[slenderness], abs=1e-4)


# =============================================================================
# Case 2 - Cylindrical shell strip (paper sec. 5.2, Table 3)
# =============================================================================

# 5p-hier. row of Table 3: radial tip displacement u_x vs slenderness R/t.
CYLINDER_STRIP_REF = {
    5: 0.9302,
    10: 0.9342,
    100: 0.6635,
    1000: 0.0225,
    10000: 0.0002,
}


def _solve_cylinder_strip(slenderness: int, num_arc_cp: int = 12, deg: int = 2) -> float:
    """Radial tip displacement of a clamped cylindrical shell strip. Boundary conditions 
    via strong constraints.
    """
    R, width, E, nu = 10.0, 1.0, 1000.0, 0.0
    t = R / slenderness

    material = ck.PlaneStress2d(E, nu, t, 1.0)
    element = ck.ShellReissnerMindlinHier5p(material)
    patch = ck.SurfacePatch.quarter_cylinder(
        nu=num_arc_cp, nv=deg + 1, deg=deg, radius=R, height=width, angle=90.0, name="strip",
    )

    quadrature = ck.GaussLegendre(deg + 1, dim=2)
    problem = ck.LinearElasticProblem([patch], element, quadrature)

    num_cp = np.asarray(patch.control_points).shape[0]
    clamped = [i for i in range(num_cp) if i % num_arc_cp >= num_arc_cp - 2]
    pinned = [cp * 5 + comp for cp in clamped for comp in range(5)]
    problem.add_constraint(ck.DirectConstraint(sorted(set(pinned)), value=0.0))

    load = ck.LoadBoundaryCondition(patch.boundary(0, True), ck.GaussLegendre(deg + 1, dim=1))
    load.add(ck.Field.U_X, 0.1 * t**3)
    problem.add_condition(load, patch="strip")

    u = ck.solve(problem)
    tip = ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)(np.array([[0.0, 0.5]]))
    return abs(np.asarray(tip).reshape(-1, 3)[0, 0])


@pytest.mark.parametrize("slenderness", list(CYLINDER_STRIP_REF))
def test_cylinder_strip(slenderness: int) -> None:
    # Looser tolerance than the flat plate: pyck and the paper differ in the
    # clamp discretisation and exact-normal construction, and the reference
    # values are themselves locking-contaminated (mesh-dependent). The tolerance
    # still pins the locking collapse - a locking-free result would fail at the
    # thin slendernesses by a wide margin.
    u_x = _solve_cylinder_strip(slenderness)
    assert u_x == pytest.approx(CYLINDER_STRIP_REF[slenderness], rel=0.2, abs=1e-4)
