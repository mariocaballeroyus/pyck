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


class ShellReissnerMindlinHier5pHelmholtz(Element):
    """Hierarchic five-parameter Reissner-Mindlin shell with an independent shear potential.

    Identical kinematics to ``ShellReissnerMindlinHier4p``, except the shear
    deflection ``w_s`` is carried as its own nodal field rather than condensed out
    through ``w_s = -(Kb/Ks)(Laplacian + 2 nu K) w_b``. The transverse shear is then
    the genuine Helmholtz decomposition ``gamma = grad(w_s) + curl(psi)`` of an
    independent gradient potential ``w_s`` and curl potential ``psi``:

        slot 0..2 : Cartesian mid-surface displacements v_b (v_x, v_y, v_z)
        slot 3    : twist (curl) potential psi
        slot 4    : shear (gradient) potential w_s

    The recovered mid-surface displacement is ``u = v_b + w_s A_3``. Freeing ``w_s``
    removes the third derivative of ``w_b`` that the 4p shear carried, so the basis
    need only be C^1 (degree >= 2). As in the 4p element, the constant-``psi`` mode is
    a zero-energy mode and should be suppressed with a zero-mean Lagrange condition on
    slot 3.

    Parameters
    ----------
    material : PlaneStress2d
        Shell material model.
    """
    num_node_dofs: int = 5

    def __init__(self, material: PlaneStress2d) -> None:
        self._material = material
        self._cpp_object = _pyck.ShellReissnerMindlinHier5pHelmholtz(self._material._cpp_object)

    @property
    def material(self) -> PlaneStress2d:
        return self._material

    def __repr__(self) -> str:
        return f"ShellReissnerMindlinHier5pHelmholtz(material={self._material})"


class ShellReissnerMindlinHier5pDispl(Element):
    """Rotation-free, shear-deformable hierarchic-*displacement* five-parameter shell.

    Oesterle, Ramm & Bischoff, "A shear deformable, rotation-free isogeometric
    shell formulation", CMAME 307 (2016) 235-255, sec. 4. The Kirchhoff-Love
    bending displacement ``v_b`` (three Cartesian mid-surface components) is
    enriched by two **displacement-like** shear parameters: the mid-surface
    displacement splits additively ``v = v_b + (v^s1 + v^s2) A_3``, with the
    ``v^sa`` shear deflections along the director:

        slot 0..2 : Cartesian bending displacements v_b (v_x, v_y, v_z)
        slot 3..4 : shear displacements (v^s1, v^s2) along A_3

    Unlike the rotation-like difference vector of ``ShellReissnerMindlinHier5p``
    (Echter 2013), the transverse shear strain is a *derivative* of a
    displacement (2e_13 = v^s1,1, 2e_23 = v^s2,2), one order below the deflection,
    which removes the transverse-shear-force oscillations. Setting
    ``v^s1 = v^s2 = 0`` recovers ``ShellKirchhoffLove3p`` exactly. The basis must
    be C^1 (degree >= 2). The constant part of each shear displacement is a
    spurious zero-energy mode; anchor it with a ``DirectConstraint`` on slots
    (3, 4) over (part of) the boundary, per Oesterle sec. 4.5 / Table 1.

    Parameters
    ----------
    material : PlaneStress2d
        Shell material model.
    """
    num_node_dofs: int = 5

    def __init__(self, material: PlaneStress2d) -> None:
        self._material = material
        self._cpp_object = _pyck.ShellReissnerMindlinHier5pDispl(self._material._cpp_object)

    @property
    def material(self) -> PlaneStress2d:
        return self._material

    def __repr__(self) -> str:
        return f"ShellReissnerMindlinHier5pDispl(material={self._material})"


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
