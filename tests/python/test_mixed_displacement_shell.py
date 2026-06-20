"""Validation of MixedDisplacementShell — the Mixed-Displacement membrane-locking remedy
applied as a composition decorator over any base shell (Bieber, Oesterle, Ramm, Bischoff,
IJNME 114 (2018) 801-827).

This single decorator replaces the deleted per-shell MD elements
(ShellReissnerMindlinHier{4,5}pMD). It is verified on the Scordelis-Lo roof: the pure
ShellReissnerMindlinHier5p membrane-locks (converges slowly from below), whereas
MixedDisplacementShell(Hier5p) removes the locking and approaches the MacNeal-Harder
reference 0.3024 already at coarse meshes. The expected values below were captured from the
original ShellReissnerMindlinHier5pMD element before its deletion; the decorator reproduces
it to solver round-off (the assembled global stiffness is bit-identical).
"""
import numpy as np
import pytest

import pyck as ck

ND5 = 8          # MixedDisplacementShell over the 5-parameter base (5 + 3)
REL_TOL = 5.0e-3

# MixedDisplacementShell(ShellReissnerMindlinHier5p), Scordelis-Lo w_z at the free-edge
# midpoint A, control points per edge -> deflection (captured from the old 5pMD element).
MD_SCORDELIS = {9: 0.2984869632, 20: 0.3006757605}


def _mat(E: float = 4.32e8, t: float = 0.25) -> "ck.PlaneStress2d":
    return ck.PlaneStress2d(E, 0.0, t, 1.0)


def _roof_patch(n_cp: int, *, radius=25.0, length=50.0, half_angle=40.0, deg=2):
    base = ck.SurfacePatch.quarter_cylinder(
        nu=n_cp, nv=n_cp, deg=deg, radius=radius, height=length,
        angle=2 * half_angle, name="roof")
    f = np.asarray(base.control_points)
    c, s = np.cos(np.radians(half_angle)), np.sin(np.radians(half_angle))
    new = np.column_stack([-f[:, 0] * s + f[:, 1] * c, f[:, 2], f[:, 0] * c + f[:, 1] * s])
    return ck.SurfacePatch(base.basis[0], base.basis[1], new, name="roof")


def _scordelis(element, nd: int, n_cp: int, *, load: float = 90.0, pin_md: bool = False) -> float:
    patch = _roof_patch(n_cp)
    prob = ck.LinearElasticProblem([patch], element, ck.GaussLegendre.from_patch(patch))
    prob.add_domain_load(np.array([0.0, 0.0, -load]), patch="roof")
    diaphragm: set[int] = set()
    for at_start in (True, False):
        diaphragm.update(int(c) for c in patch.boundary(1, at_start).displacement_dofs)
    dofs = [cp * nd + c for cp in diaphragm for c in (0, 2)]
    if pin_md:
        dofs += ck.membrane_md_boundary_dofs(patch, element)   # remove u~ zero-energy modes
    prob.add_constraint(ck.DirectConstraint(dofs, value=0.0))
    u = ck.solve(prob)
    d = ck.Function(u, element, patch, ck.FieldType.DISPLACEMENT)(np.array([[0.0, 0.5]]))
    return abs(np.asarray(d).reshape(-1, 3)[0, 2])


@pytest.mark.parametrize("n_cp, expected", list(MD_SCORDELIS.items()))
def test_scordelis_unlocked(n_cp, expected):
    """MixedDisplacementShell(Hier5p) removes membrane locking on the Scordelis-Lo roof,
    reproducing the original 5pMD element."""
    el = ck.MixedDisplacementShell(ck.ShellReissnerMindlinHier5p(_mat()))
    assert _scordelis(el, ND5, n_cp, pin_md=True) == pytest.approx(expected, rel=REL_TOL)


def test_unlocks_relative_to_pure_shell():
    """At a coarse mesh the pure displacement shell locks far below the remedy."""
    n = 9
    pure = _scordelis(ck.ShellReissnerMindlinHier5p(_mat()), 5, n)
    md = _scordelis(ck.MixedDisplacementShell(ck.ShellReissnerMindlinHier5p(_mat())),
                    ND5, n, pin_md=True)
    assert pure < 0.25          # locked, well below the 0.3024 reference
    assert md > 1.3 * pure      # the remedy clearly unlocks


def test_dof_count_and_material():
    """One decorator wraps any base: DOFs = base + 3, and the material is the base's."""
    for base_factory, ndof_base in [(ck.ShellReissnerMindlinHier4p, 4),
                                    (ck.ShellReissnerMindlinHier5p, 5)]:
        base = base_factory(_mat())
        md = ck.MixedDisplacementShell(base)
        assert md.num_node_dofs == ndof_base + 3
        assert md._material is base._material
        assert md.base is base


def test_boundary_dofs_target_last_three_slots():
    """membrane_md_boundary_dofs pins only the u~ slots (the last three per node)."""
    patch = _roof_patch(5)
    md = ck.MixedDisplacementShell(ck.ShellReissnerMindlinHier5p(_mat()))
    dofs = ck.membrane_md_boundary_dofs(patch, md)
    assert len(dofs) > 0
    assert all((d % md.num_node_dofs) in (5, 6, 7) for d in dofs)
