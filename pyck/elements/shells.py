"""Shell element formulations."""

from __future__ import annotations

import pyck._pyck as _pyck
from pyck.elements.element import Element
from pyck.materials import PlaneStress2d


class ShellReissnerMindlin5p(Element):
    """Reissner-Mindlin 5-parameter shell element.

    DOFs per control point:
        slot 0..2 : Cartesian displacements (u_x, u_y, u_z)
        slot 3..4 : director rotation amplitudes (theta_1, theta_2)

    See ``docs/shell_reissner_mindlin_5p.md`` for the formulation.

    Parameters
    ----------
    material : PlaneStress2d
        Shell material model.
    """
    num_node_dofs: int = 5

    def __init__(self, material: PlaneStress2d) -> None:
        self._material = material
        self._cpp_object = _pyck.ShellReissnerMindlin5p(self._material._cpp_object)

    @property
    def material(self) -> PlaneStress2d:
        return self._material

    def __repr__(self) -> str:
        return f"ShellReissnerMindlin5p(material={self._material})"


class ShellReissnerMindlin4p(Element):
    """Four-parameter rotation-free Reissner-Mindlin shell element.

    The director tilt is given a surface Helmholtz split and the transverse
    displacement a bending-shear split, eliminating the normal displacement
    and the irrotational tilt onto the bending potential ``w_b``. This leaves
    four DOFs per control point against five for the standard
    displacement-rotation shell:

        slot 0..1 : covariant in-plane displacement components (u_1, u_2)
        slot 2    : bending potential w_b
        slot 3    : twist potential psi

    The recovered fields are

        w     = w_b - (Kb/Ks) Laplacian(w_b)
        gamma = -(Kb/Ks) grad(Laplacian(w_b)) + curl(psi) + B . u

    The shear strain carries third derivatives of ``w_b``, so the basis must be
    C^2 (degree >= 3). The constant-``psi`` mode is a zero-energy mode and should
    be suppressed with a zero-mean Lagrange condition on slot 3.

    See ``report/sections/extra.tex`` for the formulation.

    Parameters
    ----------
    material : PlaneStress2d
        Shell material model.
    """
    num_node_dofs: int = 4

    def __init__(self, material: PlaneStress2d) -> None:
        self._material = material
        self._cpp_object = _pyck.ShellReissnerMindlin4p(self._material._cpp_object)

    @property
    def material(self) -> PlaneStress2d:
        return self._material

    def __repr__(self) -> str:
        return f"ShellReissnerMindlin4p(material={self._material})"


class ShellReissnerMindlinHier4p(Element):
    """Hierarchic four-parameter Reissner-Mindlin shell element.

    A shear-deformable shell built hierarchically on the Kirchhoff-Love
    displacement field: the primary unknowns are the three Cartesian mid-surface
    displacement components (as in ``ShellKirchhoffLove3p``) plus a single twist
    potential ``psi``:

        slot 0..2 : Cartesian displacements (u_x, u_y, u_z)
        slot 3    : twist potential psi

    Unlike ``ShellReissnerMindlin4p`` (covariant ``u_1, u_2, w_b, psi``), the
    bending potential is not an independent field: ``w_b = v . A_3``. The whole
    formulation is written in the base vectors, the director ``A_3`` and its
    derivatives, and the metric. The recovered fields are

        w_psi = grad(psi) x A_3
        w_s   = -(Kb/Ks) Laplacian(w_b),     w_b = v . A_3
        gamma = grad(w_s) + w_psi . A_alpha

    The shear carries third derivatives of ``w_b``, so the basis must be C^2
    (degree >= 3). The constant-``psi`` mode is a zero-energy mode and should be
    suppressed with a zero-mean Lagrange condition on slot 3.

    See ``report/sections/extra.tex`` for the formulation.

    Parameters
    ----------
    material : PlaneStress2d
        Shell material model.
    """
    num_node_dofs: int = 4

    def __init__(self, material: PlaneStress2d) -> None:
        self._material = material
        self._cpp_object = _pyck.ShellReissnerMindlinHier4p(self._material._cpp_object)

    @property
    def material(self) -> PlaneStress2d:
        return self._material

    def __repr__(self) -> str:
        return f"ShellReissnerMindlinHier4p(material={self._material})"


class ShellReissnerMindlinHier5p(Element):
    """Hierarchic five-parameter Reissner-Mindlin shell element.

    A shear-deformable shell built hierarchically on the Kirchhoff-Love
    displacement field (Echter, Oesterle & Bischoff 2013, sec. 2.2): the three
    Cartesian mid-surface displacements (as in ``ShellKirchhoffLove3p``) are
    enriched by a hierarchic difference vector ``w = w^1 A_1 + w^2 A_2`` added to
    the *rotated* director ``a_3 = (A_3 + Phi x A_3) + w``:

        slot 0..2 : Cartesian displacements (v_x, v_y, v_z)
        slot 3..4 : hierarchic difference vector (w^1, w^2)

    The hierarchic split of bending and shear makes the element free of
    transverse-shear locking by construction; setting ``w = 0`` recovers
    ``ShellKirchhoffLove3p`` exactly. The Kirchhoff-Love bending of ``v`` uses
    the covariant Hessian, so the basis must be C^1 (degree >= 2).

    Reference: R. Echter, B. Oesterle, M. Bischoff, "A hierarchic family of
    isogeometric shell finite elements", CMAME 254 (2013) 170-180, sec. 2.2.

    Parameters
    ----------
    material : PlaneStress2d
        Shell material model.
    """
    num_node_dofs: int = 5

    def __init__(self, material: PlaneStress2d) -> None:
        self._material = material
        self._cpp_object = _pyck.ShellReissnerMindlinHier5p(self._material._cpp_object)

    @property
    def material(self) -> PlaneStress2d:
        return self._material

    def __repr__(self) -> str:
        return f"ShellReissnerMindlinHier5p(material={self._material})"


class ShellReissnerMindlinHier4pMD(ShellReissnerMindlinHier4p):
    """Mixed-Displacement (MD) hierarchic four-parameter Reissner-Mindlin shell.

    Identical kinematics to :class:`ShellReissnerMindlinHier4p`, but membrane locking is
    removed by the Mixed-Displacement method (Bieber, Oesterle, Ramm, Bischoff, IJNME 2018):
    a 3-component, equal-order membrane strain-displacement field is added (slots 4..6),
    whose derivative supplies the lower-order membrane strain. The element builds the mixed
    (symmetric-indefinite) stiffness directly — no condensation, element-local assembly.

        slot 0..2 : Cartesian displacements (u_x, u_y, u_z)
        slot 3    : twist potential psi
        slot 4..6 : membrane strain-displacements (v~_11, v~_12, v~_22)

    The strain-displacement field has integration-constant zero-energy modes; pin its
    boundary DOFs with a Dirichlet constraint (see ``membrane_md_boundary_dofs``).

    Parameters
    ----------
    material : PlaneStress2d
        Shell material model.
    """
    num_node_dofs: int = 7

    def __init__(self, material: PlaneStress2d) -> None:
        self._material = material
        self._cpp_object = _pyck.ShellReissnerMindlinHier4pMD(self._material._cpp_object)

    def __repr__(self) -> str:
        return f"ShellReissnerMindlinHier4pMD(material={self._material})"


class ShellReissnerMindlinHier5pMD(ShellReissnerMindlinHier5p):
    """Mixed-Displacement (MD) hierarchic five-parameter Reissner-Mindlin shell.

    Identical kinematics to :class:`ShellReissnerMindlinHier5p`, but membrane locking is
    removed by the Mixed-Displacement method (Bieber, Oesterle, Ramm, Bischoff, IJNME 2018):
    a 3-component, equal-order membrane strain-displacement field is added (slots 5..7),
    whose derivative supplies the lower-order membrane strain. The element builds the mixed
    (symmetric-indefinite) stiffness directly — no condensation, element-local assembly.

        slot 0..2 : Cartesian displacements (v_x, v_y, v_z)
        slot 3..4 : hierarchic difference vector (w^1, w^2)
        slot 5..7 : membrane strain-displacements (v~_11, v~_12, v~_22)

    The strain-displacement field has integration-constant zero-energy modes; pin its
    boundary DOFs with a Dirichlet constraint (see ``membrane_md_boundary_dofs``).

    Parameters
    ----------
    material : PlaneStress2d
        Shell material model.
    """
    num_node_dofs: int = 8

    def __init__(self, material: PlaneStress2d) -> None:
        self._material = material
        self._cpp_object = _pyck.ShellReissnerMindlinHier5pMD(self._material._cpp_object)

    def __repr__(self) -> str:
        return f"ShellReissnerMindlinHier5pMD(material={self._material})"


def membrane_md_boundary_dofs(patch, element) -> list[int]:
    """Boundary strain-displacement DOFs to pin for a Mixed-Displacement shell.

    The MD membrane field has integration-constant zero-energy modes
    (``v~_11 = F(xi2)``, ``v~_22 = G(xi1)``, ``v~_12 = F(xi1) + G(xi2)``). They carry no
    stiffness and no influence on the solution, but make the system singular. Pinning
    these boundary DOFs to zero removes them: ``v~_11`` on a xi1-edge, ``v~_22`` on a
    xi2-edge, ``v~_12`` on one edge of each direction.

    Parameters
    ----------
    patch : SurfacePatch
        The patch carrying the MD element.
    element : ShellReissnerMindlinHier4pMD or ShellReissnerMindlinHier5pMD
        The MD element (used for ``num_node_dofs``).

    Returns
    -------
    list[int]
        Global DOF indices to constrain to zero (e.g. with ``DirectConstraint``).
    """
    nd = int(element.num_node_dofs)
    s11, s12, s22 = nd - 3, nd - 2, nd - 1  # u~ slots are the last three

    def edge_cps(pdim: int) -> list[int]:
        return [int(c) for c in patch.boundary(pdim, True).displacement_dofs]

    dofs: set[int] = set()
    for cp in edge_cps(0):  # xi1-edge
        dofs.add(cp * nd + s11)
        dofs.add(cp * nd + s12)
    for cp in edge_cps(1):  # xi2-edge
        dofs.add(cp * nd + s22)
        dofs.add(cp * nd + s12)
    return sorted(dofs)


class ShellKirchhoffLove3p(Element):
    """Kirchhoff-Love thin-shell element (rotation-free).

    Transverse shear is neglected, the director follows the deformed normal, and
    bending is the linearized change of the second fundamental form. The only
    unknowns are the three Cartesian displacement components per control point:

        slot 0..2 : Cartesian displacements (u_x, u_y, u_z)

    Being rotation-free, the basis must be C^1 (degree >= 2).

    Parameters
    ----------
    material : PlaneStress2d
        Shell material model.
    """
    num_node_dofs: int = 3

    def __init__(self, material: PlaneStress2d) -> None:
        self._material = material
        self._cpp_object = _pyck.ShellKirchhoffLove3p(self._material._cpp_object)

    @property
    def material(self) -> PlaneStress2d:
        return self._material

    def __repr__(self) -> str:
        return f"ShellKirchhoffLove3p(material={self._material})"
