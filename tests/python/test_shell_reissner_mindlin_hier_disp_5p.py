"""Validation of ShellReissnerMindlinHierDisp5p — the rotation-free, shear-deformable
hierarchic five-parameter shell of

    B. Oesterle, E. Ramm, M. Bischoff, "A shear deformable, rotation-free isogeometric
    shell formulation", Comput. Methods Appl. Mech. Engrg. 307 (2016) 235-255, §4

("5p-hier.-disp."). The Kirchhoff-Love bending displacement is enriched by two
displacement-like shear parameters v^{sα} A₃, so the transverse shear strain is a
*derivative* of a displacement (2ε₁₃ = v^{s1}_{,1}, 2ε₂₃ = v^{s2}_{,2}) — one order
below the deflection, which removes the transverse-shear-force oscillations of the
earlier rotation-based hierarchic element (Echter et al. 2013).

Three properties are checked:

1. **Kirchhoff-Love limit** — setting v^{s1} = v^{s2} = 0 must reproduce
   ShellKirchhoffLove3p to machine precision (the v_b kinematics are untouched).

2. **No transverse-shear locking** — on a flat plate the thin-limit deflection is
   thickness-independent (the defining feature of the formulation): the same value at
   L/t = 100, 1000, 10000, converging to the Kirchhoff series solution 0.442892
   (Timoshenko-Woinowsky-Krieger via Oesterle Eq. 39).

3. **Scordelis-Lo roof** — the displacement-based and rotation-based hierarchic
   elements "practically yield identical results in terms of the displacement"
   (Oesterle 2016, §5.2). With a mixed/assumed-strain membrane treatment removed
   (bare element), the curved shell membrane-locks and converges slowly from below,
   reproducing the Echter-dissertation "5p-hier." table mesh by mesh.

The rotation-free split carries a spurious near-zero-energy boundary mode in the shear
displacements (their constant part); per Oesterle §4.5 it is removed by pinning the
shear slots (3, 4) on the boundary.
"""

import numpy as np
import pytest

import pyck as ck

ND = 5
REL_TOL = 1.5e-2  # the displacement-/rotation-based hierarchic elements coincide to ~1 %


# === Helpers ========================================================================

def _pin_vs_on_boundary(patch, dofs: list[int]) -> None:
    """Append the shear-displacement slots (3, 4) on every boundary control point to
    @p dofs — removes the constant-v_s zero-energy mode (Oesterle §4.5)."""
    edge: set[int] = set()
    for pdim in (0, 1):
        for at_start in (True, False):
            edge.update(int(c) for c in patch.boundary(pdim, at_start).displacement_dofs)
    dofs.extend(cp * ND + 3 for cp in edge)
    dofs.extend(cp * ND + 4 for cp in edge)


def _simply_supported_plate(slenderness: float, *, length: float = 10.0,
                            E: float = 1000.0, nu: float = 0.3, n: int = 12) -> float:
    """Center deflection of a simply supported square plate, UDL q_z = t³ (scaled so the
    thin-limit deflection is slenderness-independent). Oesterle Fig. 10: L=10, E=1000,
    nu=0.3, biquadratic NURBS."""
    t = length / slenderness
    patch = ck.SurfacePatch.rectangle(length, length, nu=n, nv=n, deg=2, name="plate")
    element = ck.ShellReissnerMindlinHierDisp5p(ck.PlaneStress2d(E, nu, t, 1.0))
    prob = ck.LinearElasticProblem([patch], element, ck.GaussLegendre.from_patch(patch))
    prob.add_domain_load(np.array([0.0, 0.0, t ** 3]), patch="plate")

    edge: set[int] = set()
    for pdim in (0, 1):
        for at_start in (True, False):
            edge.update(int(c) for c in patch.boundary(pdim, at_start).displacement_dofs)
    dofs = [cp * ND + c for cp in edge for c in (0, 1, 2)]   # soft support u = 0
    _pin_vs_on_boundary(patch, dofs)
    prob.add_constraint(ck.DirectConstraint(sorted(set(dofs)), value=0.0))

    u = ck.solve(prob)
    w = ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)(np.array([[0.5, 0.5]]))
    return abs(np.asarray(w).reshape(-1, 3)[0, 2])


def _scordelis_roof_patch(n_cp: int, *, radius: float = 25.0, length: float = 50.0,
                          half_angle: float = 40.0, deg: int = 2) -> ck.SurfacePatch:
    base = ck.SurfacePatch.quarter_cylinder(
        nu=n_cp, nv=n_cp, deg=deg, radius=radius, height=length,
        angle=2 * half_angle, name="roof")
    f = np.asarray(base.control_points)
    c, s = np.cos(np.radians(half_angle)), np.sin(np.radians(half_angle))
    new = np.column_stack([-f[:, 0] * s + f[:, 1] * c, f[:, 2], f[:, 0] * c + f[:, 1] * s])
    return ck.SurfacePatch(base.basis[0], base.basis[1], new, name="roof")


def _scordelis_lo(n_cp: int, *, E: float = 4.32e8, t: float = 0.25,
                  load: float = 90.0) -> float:
    """Vertical displacement w_z at the free-edge midpoint A of the Scordelis-Lo roof."""
    patch = _scordelis_roof_patch(n_cp)
    element = ck.ShellReissnerMindlinHierDisp5p(ck.PlaneStress2d(E, 0.0, t, 1.0))
    prob = ck.LinearElasticProblem([patch], element, ck.GaussLegendre.from_patch(patch))
    prob.add_domain_load(np.array([0.0, 0.0, -load]), patch="roof")

    diaphragm: set[int] = set()
    for at_start in (True, False):
        diaphragm.update(int(c) for c in patch.boundary(1, at_start).displacement_dofs)
    dofs = [cp * ND + c for cp in diaphragm for c in (0, 2)]
    _pin_vs_on_boundary(patch, dofs)
    prob.add_constraint(ck.DirectConstraint(sorted(set(dofs)), value=0.0))

    u = ck.solve(prob)
    d = ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)(np.array([[0.0, 0.5]]))
    return abs(np.asarray(d).reshape(-1, 3)[0, 2])


# === Tests ==========================================================================

def test_kirchhoff_love_limit():
    """v^{s1} = v^{s2} = 0 reproduces ShellKirchhoffLove3p to machine precision."""
    L, W, E, t = 10.0, 2.0, 1.0e6, 0.05
    patch = ck.SurfacePatch.rectangle(L, W, nu=8, nv=4, deg=3, cx=L / 2, cy=W / 2, name="m")
    quad = ck.GaussLegendre.from_patch(patch)
    ncp = patch.num_control_pts

    def tip(element, nd, pin_vs):
        prob = ck.LinearElasticProblem([patch], element, quad)
        x0 = [int(c) for c in patch.boundary(0, True).displacement_dofs]
        fix = [cp * nd + c for cp in x0 for c in range(nd)]
        if pin_vs:
            fix += [cp * nd + 3 for cp in range(ncp)] + [cp * nd + 4 for cp in range(ncp)]
        prob.add_constraint(ck.DirectConstraint(sorted(set(fix)), value=0.0))
        load = ck.LoadBoundaryCondition(patch.boundary(0, False), ck.GaussLegendre(4, dim=1))
        load.add(ck.Field.U_Z, -1.0)
        prob.add_condition(load, patch="m")
        u = ck.solve(prob)
        d = ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)(np.array([[1.0, 0.5]]))
        return abs(np.asarray(d).reshape(-1, 3)[0, 2])

    kl = tip(ck.ShellKirchhoffLove3p(ck.PlaneStress2d(1.0e6, 0.0, t, 1.0)), 3, False)
    h5 = tip(ck.ShellReissnerMindlinHierDisp5p(ck.PlaneStress2d(1.0e6, 0.0, t, 1.0)), 5, True)
    assert h5 == pytest.approx(kl, rel=1e-12)


@pytest.mark.parametrize("slenderness", [100, 1000, 10000])
def test_no_transverse_shear_locking(slenderness):
    """Thin-limit plate deflection is thickness-independent and equals the Kirchhoff
    series solution 0.442892 — the formulation is locking-free by construction."""
    assert _simply_supported_plate(slenderness) == pytest.approx(0.442892, rel=REL_TOL)


# Echter dissertation, Scordelis-Lo roof, quadratic NURBS, "5p-hier." column. The
# displacement-based element coincides with that rotation-based one to ~1 %.
SCORDELIS = {5: 0.03998, 9: 0.20774, 13: 0.28030, 20: 0.29781, 25: 0.29978, 30: 0.30049}


@pytest.mark.parametrize("n_cp, expected", list(SCORDELIS.items()))
def test_scordelis_lo_membrane_locking(n_cp, expected):
    """Bare element (no membrane treatment): slow membrane-locking convergence from
    below toward 0.3024, matching the hierarchic reference table mesh by mesh."""
    assert _scordelis_lo(n_cp) == pytest.approx(expected, rel=REL_TOL)
