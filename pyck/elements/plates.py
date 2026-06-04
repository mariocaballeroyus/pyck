"""Shell element formulations."""

from __future__ import annotations

import pyck._pyck as _pyck
from pyck.elements.element import Element
from pyck.materials import PlaneStress2d, PlaneStressShell


class ShellReissnerMindlin5p(Element):
    """Reissner-Mindlin 5-parameter shell element.

    DOFs per control point:
        slot 0..2 : Cartesian displacements (u_x, u_y, u_z)
        slot 3..4 : director rotation amplitudes (theta_1, theta_2)

    See ``docs/shell_reissner_mindlin_5p.md`` for the formulation.

    Parameters
    ----------
    material : PlaneStressShell
        Shell material model.
    """
    num_node_dofs: int = 5

    def __init__(self, material: PlaneStressShell) -> None:
        self._material = material
        self._cpp_object = _pyck.ShellReissnerMindlin5p(self._material._cpp_object)

    @property
    def material(self) -> PlaneStressShell:
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


class ShellKirchhoffLove3p(Element):
    """Kirchhoff-Love thin-shell element (rotation-free).

    Transverse shear is neglected, the director follows the deformed normal, and
    bending is the linearized change of the second fundamental form. The only
    unknowns are the three Cartesian displacement components per control point:

        slot 0..2 : Cartesian displacements (u_x, u_y, u_z)

    Being rotation-free, the basis must be C^1 (degree >= 2).

    Parameters
    ----------
    material : PlaneStressShell
        Shell material model.
    """
    num_node_dofs: int = 3

    def __init__(self, material: PlaneStressShell) -> None:
        self._material = material
        self._cpp_object = _pyck.ShellKirchhoffLove3p(self._material._cpp_object)

    @property
    def material(self) -> PlaneStressShell:
        return self._material

    def __repr__(self) -> str:
        return f"ShellKirchhoffLove3p(material={self._material})"
