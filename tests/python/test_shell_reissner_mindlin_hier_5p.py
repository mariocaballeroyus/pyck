"""Validation of ShellReissnerMindlinHier5p against published reference values.

The hierarchic five-parameter shell implemented here is the "5p-hier." element of

    R. Echter, B. Oesterle, M. Bischoff, "A hierarchic family of isogeometric
    shell finite elements", Comput. Methods Appl. Mech. Engrg. 254 (2013) 170-180,

i.e. the *pure*, displacement-based formulation with a hierarchic difference
vector and NO membrane-locking treatment (unlike the mixed-displacement variant
ShellReissnerMindlinHier5pMD). It is
validated against the two benchmarks for which that exact element has tabulated
reference values:

1. Simply supported square plate, maximum vertical displacement w_z,max.
   Source: Echter et al. (2013), sec. 5.1, Table 2, the "5p-hier." row,
   for slenderness L/t = 5, 10, 100, 1000, 10000.
   The plate is flat, so membrane and bending decouple and there is no membrane
   locking: this isolates the shear-deformable kinematics, and the element must
   reproduce the table at every slenderness (locking-free shear behaviour).

2. Scordelis-Lo roof, vertical displacement w_z at the free-edge midpoint A.
   Source: R. Echter, doctoral dissertation, Institut fuer Baustatik und
   Baudynamik, Universitaet Stuttgart (2013); Scordelis-Lo roof with 2nd-order
   (quadratic) NURBS, the "5p-hier." column, for 5, 9, 13, 20, 25, 30 control
   points per edge. This is a curved shell that membrane-locks: the pure element
   converges slowly *from below* to the MacNeal-Harder reference 0.3024, matching
   the dissertation table mesh by mesh.

Both reference setups use Young's modulus / Poisson ratio / geometry exactly as in
the cited sources. Echter derives the transverse-shear stiffness directly from the
3D constitutive law (no engineering 5/6 correction), so the shear factor is k = 1.

The paper's third benchmark - the cylindrical shell strip of Table 3 - is
deliberately NOT used as a regression target: its published value is a
membrane-locking residual (the element locks down to the 3p Kirchhoff-Love shell
there) and depends on the edge-load type, which differs between the journal (a
radial line load) and the dissertation (an edge moment). It is therefore not a
stable point-for-point reference. The element's correctness on curved,
membrane-locking shells is instead covered exactly by the Scordelis-Lo benchmark.
"""

import numpy as np
import pytest

import pyck as ck

ND = 5  # ShellReissnerMindlinHier5p degrees of freedom per control point

# Relative tolerance for matching the published tables. Both benchmarks reproduce
# the references to ~0.01-0.1 %; 0.5 % comfortably absorbs the 4-5 significant
# figures of the tabulated values while still catching any real regression.
REL_TOL = 5.0e-3


# --- Echter et al. (2013), Table 2: simply supported plate, "5p-hier." row -------
#     slenderness L/t -> max. vertical displacement w_z,max
PLATE_TABLE2 = {5: 0.5839, 10: 0.4938, 100: 0.4431, 1000: 0.4423, 10000: 0.4423}

# --- Echter dissertation: Scordelis-Lo roof, quadratic NURBS, "5p-hier." column --
#     control points per edge -> vertical displacement w_z at point A
SCORDELIS_DISSERTATION = {5: 0.03998, 9: 0.20774, 13: 0.28030,
                          20: 0.29781, 25: 0.29978, 30: 0.30049}


# === Benchmark drivers ==============================================================

def _simply_supported_plate(slenderness: float, *, length: float = 10.0,
                            E: float = 1000.0, nu: float = 0.3,
                            n_el: int = 10, deg: int = 2) -> float:
    """Max. vertical displacement of a simply supported square plate under a uniform
    transverse load q_z = t^3 (scaled with t^3 so the thin-limit deflection is
    slenderness-independent). Echter et al. (2013), sec. 5.1: L = 10, E = 1000,
    nu = 0.3, 10 biquadratic elements per side."""
    t = length / slenderness
    n = n_el + deg  # control points per side (n_el elements, C^1)

    patch = ck.SurfacePatch.rectangle(length, length, nu=n, nv=n, deg=deg, name="plate")
    element = ck.ShellReissnerMindlinHier5p(ck.PlaneStress2d(E, nu, t, 1.0))
    prob = ck.LinearElasticProblem([patch], element, ck.GaussLegendre.from_patch(patch))

    prob.add_domain_load(np.array([0.0, 0.0, t ** 3]), patch="plate")

    # Soft (Navier) simple support: u = 0 on all four edges, rotations free. On a
    # flat plate membrane and bending decouple, so pinning the in-plane components
    # too (to remove rigid-body / membrane modes) leaves the bending answer intact.
    edge_cps: set[int] = set()
    for pdim in (0, 1):
        for at_start in (True, False):
            edge_cps.update(int(c) for c in patch.boundary(pdim, at_start).displacement_dofs)
    dofs = [cp * ND + c for cp in edge_cps for c in (0, 1, 2)]
    prob.add_constraint(ck.DirectConstraint(dofs, value=0.0))

    u = ck.solve(prob)
    w = ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)(np.array([[0.5, 0.5]]))
    return abs(np.asarray(w).reshape(-1, 3)[0, 2])


def _scordelis_roof_patch(n_cp: int, *, radius: float = 25.0, length: float = 50.0,
                          half_angle: float = 40.0, deg: int = 2) -> ck.SurfacePatch:
    """Scordelis-Lo roof geometry: a 2*half_angle circular arc, symmetric about the
    vertical (crown at the top), extruded along the length. Built by affine-
    transforming the exact NURBS arc from ``quarter_cylinder`` (weights preserved)."""
    base = ck.SurfacePatch.quarter_cylinder(
        nu=n_cp, nv=n_cp, deg=deg, radius=radius, height=length,
        angle=2 * half_angle, name="roof")
    f = np.asarray(base.control_points)
    c, s = np.cos(np.radians(half_angle)), np.sin(np.radians(half_angle))
    # rotate the arc by -half_angle in its own plane, then relabel axes so the crown
    # is at +Z (up), the width spans X, and the length runs along Y.
    new = np.column_stack([-f[:, 0] * s + f[:, 1] * c, f[:, 2], f[:, 0] * c + f[:, 1] * s])
    return ck.SurfacePatch(base.basis[0], base.basis[1], new, name="roof")


def _scordelis_lo(n_cp: int, *, E: float = 4.32e8, t: float = 0.25,
                  load: float = 90.0) -> float:
    """Vertical displacement w_z at the free-edge midpoint A of the Scordelis-Lo
    roof under self-weight. Echter dissertation / MacNeal-Harder: R = 25, L = 50,
    half-angle = 40 deg, E = 4.32e8, nu = 0, t = 0.25, dead load 90 per unit area."""
    patch = _scordelis_roof_patch(n_cp)
    element = ck.ShellReissnerMindlinHier5p(ck.PlaneStress2d(E, 0.0, t, 1.0))
    prob = ck.LinearElasticProblem([patch], element, ck.GaussLegendre.from_patch(patch))

    # Self-weight: dead load per unit mid-surface area, vertically down (-Z).
    prob.add_domain_load(np.array([0.0, 0.0, -load]), patch="roof")

    # Rigid diaphragms on the two arc edges (v = 0, v = 1): u_x = u_z = 0, u_y free,
    # imposed directly on the edge control points (the source's recipe).
    diaphragm: set[int] = set()
    for at_start in (True, False):
        diaphragm.update(int(c) for c in patch.boundary(1, at_start).displacement_dofs)
    dofs = [cp * ND + c for cp in diaphragm for c in (0, 2)]
    prob.add_constraint(ck.DirectConstraint(dofs, value=0.0))

    u = ck.solve(prob)
    d = ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)(np.array([[0.0, 0.5]]))
    return abs(np.asarray(d).reshape(-1, 3)[0, 2])


# === Tests ==========================================================================

@pytest.mark.parametrize("slenderness, expected", list(PLATE_TABLE2.items()))
def test_simply_supported_plate_matches_echter_table2(slenderness, expected):
    """Reproduce Echter et al. (2013), Table 2 ("5p-hier." row): the locking-free
    shear-deformable plate deflection at every slenderness."""
    assert _simply_supported_plate(slenderness) == pytest.approx(expected, rel=REL_TOL)


@pytest.mark.parametrize("n_cp, expected", list(SCORDELIS_DISSERTATION.items()))
def test_scordelis_lo_matches_echter_dissertation(n_cp, expected):
    """Reproduce the Echter dissertation Scordelis-Lo table (quadratic NURBS,
    "5p-hier." column): slow membrane-locking convergence from below to 0.3024."""
    assert _scordelis_lo(n_cp) == pytest.approx(expected, rel=REL_TOL)
