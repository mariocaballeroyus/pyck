"""Plate element formulations."""

from __future__ import annotations

import pyck._pyck as _pyck
from pyck.elements.element import Element
from pyck.materials import PlaneStress2d, PlaneStressShell


class PlateKirchhoffLove1p(Element):
    """Kirchhoff-Love thin plate element.

    Uses the transverse displacement w as the sole primary variable, 
    under the assumption that shear deformation can be neglected, and 
    therefore the rotation and displacement are coupled under the 
    relation: 
        phi = - nabla(w).

    Parameters
    ----------
    material : PlaneStress2d
        Plate material model.
    """
    _cpp_object: _pyck.PlateKirchhoffLove1p
    num_node_dofs: int = 1

    def __init__(self, material: PlaneStress2d) -> None:
        self._material = material
        self._cpp_object = _pyck.PlateKirchhoffLove1p(self._material._cpp_object)

    @property
    def material(self) -> PlaneStress2d:
        return self._material

    def __repr__(self) -> str:
        return f"PlateKirchhoffLove1p(material={self._material})"


class PlateReissnerMindlin1p(Element):
    """Rotation-free Reissner-Mindlin plate element.

    Uses a bending potential wb as the sole primary variable.
    The total deflection is recovered via:
        w = wb - (Kb / Ks) * nabla^2(wb)

    Requires at least C^2 continuity (cubic B-splines or higher).

    Parameters
    ----------
    material : PlaneStress2d
        Plate material model.
    """
    num_node_dofs: int = 1

    def __init__(self, material: PlaneStress2d) -> None:
        self._material = material
        self._cpp_object = _pyck.PlateReissnerMindlin1p(self._material._cpp_object)

    @property
    def material(self) -> PlaneStress2d:
        return self._material

    def __repr__(self) -> str:
        return f"PlateReissnerMindlin1p(material={self._material})"


class PlateReissnerMindlin3p(Element):
    """Standard Reissner-Mindlin plate element.

    Uses the transverse displacement w, and the rotations phi_x and phi_y 
    as the primary variables. Accounts for both bending and shear 
    deformation.

    Parameters
    ----------
    material : PlaneStress2d
        Plate material model.
    k_s : float, optional
        Shear correction factor (default 5/6).
    """
    num_node_dofs: int = 3

    def __init__(self, material: PlaneStress2d, k_s: float = 5.0 / 6.0) -> None:
        self._material = material
        self._k_s = float(k_s)
        self._cpp_object = _pyck.PlateReissnerMindlin3p(self._material._cpp_object)

    @property
    def material(self) -> PlaneStress2d:
        return self._material

    @property
    def shear_coefficient(self) -> float:
        return self._k_s

    def __repr__(self) -> str:
        return f"PlateReissnerMindlin3p(material={self._material}, k_s={self._k_s})"


class PlateReissnerMindlinDispl2p(Element):
    """Two-parameter rotation-free Reissner-Mindlin plate element.

    Uses a bending displacement field ``w_b`` and a Helmholtz scalar
    potential ``psi`` for the curl part of the rotation field. The
    recovered physical fields are

        w     = w_b - (Kb / Ks) * Laplacian(w_b)
        phi_x = -w_b,x + psi,y
        phi_y = -w_b,y - psi,x
        gamma = -(Kb/Ks) grad(Laplacian(w_b)) + curl(psi)
        kappa = L phi

    Requires at least C^2 continuity (cubic B-splines or higher).

    Parameters
    ----------
    material : PlaneStress2d
        Plate material model.
    """
    num_node_dofs: int = 2

    def __init__(self, material: PlaneStress2d) -> None:
        self._material = material
        self._cpp_object = _pyck.PlateReissnerMindlinDispl2p(
            self._material._cpp_object
        )

    @property
    def material(self) -> PlaneStress2d:
        return self._material

    def __repr__(self) -> str:
        return f"PlateReissnerMindlinDispl2p(material={self._material})"


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

    The curved-surface counterpart of :class:`PlateReissnerMindlinDispl2p`.
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

    Like the two-parameter plate the shear strain carries third derivatives of
    ``w_b``, so the basis must be C^2 (degree >= 3). The constant-``psi`` mode
    is a zero-energy mode (as for the plate) and should be suppressed with a
    zero-mean Lagrange condition on slot 3.

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

    The curved-surface counterpart of :class:`PlateKirchhoffLove1p`: transverse
    shear is neglected, the director follows the deformed normal, and bending is
    the linearized change of the second fundamental form. The only unknowns are
    the three Cartesian displacement components per control point:

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


