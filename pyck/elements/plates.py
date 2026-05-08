"""Plate element formulations."""

from __future__ import annotations

import pyck._pyck as _pyck
from pyck.materials import PlaneStress2d, PlaneStressShell


class PlateKirchhoffLove1p:
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

    def __init__(self, material: PlaneStress2d) -> None:
        self._material = material
        self._cpp_object = _pyck.PlateKirchhoffLove1p(self._material._cpp_object)

    @property
    def material(self) -> PlaneStress2d:
        return self._material

    def __repr__(self) -> str:
        return f"PlateKirchhoffLove1p(material={self._material})"


class PlateReissnerMindlin1p:
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

    def __init__(self, material: PlaneStress2d) -> None:
        self._material = material
        self._cpp_object = _pyck.PlateReissnerMindlin1p(self._material._cpp_object)

    @property
    def material(self) -> PlaneStress2d:
        return self._material

    def __repr__(self) -> str:
        return f"PlateReissnerMindlin1p(material={self._material})"


class PlateReissnerMindlin3p:
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


class PlateReissnerMindlinDispl3p:
    """Split-displacement Reissner-Mindlin plate element.

    Uses one bending displacement field ``w_b`` and two shear displacement
    fields ``w_s1`` and ``w_s2``. The recovered physical fields are

        w = w_b + w_s1 + w_s2
        phi_x = -w_b,x - w_s2,x
        phi_y = -w_b,y - w_s1,y
        gamma = [w_s1,x, w_s2,y]
        kappa = L phi

    Parameters
    ----------
    material : PlaneStress2d
        Plate material model.
    """

    def __init__(self, material: PlaneStress2d) -> None:
        self._material = material
        self._cpp_object = _pyck.PlateReissnerMindlinDispl3p(
            self._material._cpp_object
        )

    @property
    def material(self) -> PlaneStress2d:
        return self._material

    def __repr__(self) -> str:
        return f"PlateReissnerMindlinDispl3p(material={self._material})"


class PlateReissnerMindlinDispl2p:
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


def create_kirchhoff_love_plate(
    material: PlaneStress2d
) -> PlateKirchhoffLove1p:
    """Create a :class:`PlateKirchhoffLove1p` element."""
    return PlateKirchhoffLove1p(material)


def create_reissner_mindlin_plate(
    material: PlaneStress2d, k_s: float = 5.0 / 6.0
) -> PlateReissnerMindlin3p:
    """Create a :class:`PlateReissnerMindlin3p` element."""
    return PlateReissnerMindlin3p(material, k_s)


def create_reissner_mindlin_displ_plate(
    material: PlaneStress2d,
) -> PlateReissnerMindlinDispl3p:
    """Create a :class:`PlateReissnerMindlinDispl3p` element."""
    return PlateReissnerMindlinDispl3p(material)


def create_reissner_mindlin_displ_2p_plate(
    material: PlaneStress2d,
) -> PlateReissnerMindlinDispl2p:
    """Create a :class:`PlateReissnerMindlinDispl2p` element."""
    return PlateReissnerMindlinDispl2p(material)


class ShellReissnerMindlin5p:
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

    def __init__(self, material: PlaneStressShell) -> None:
        self._material = material
        self._cpp_object = _pyck.ShellReissnerMindlin5p(self._material._cpp_object)

    @property
    def material(self) -> PlaneStressShell:
        return self._material

    def __repr__(self) -> str:
        return f"ShellReissnerMindlin5p(material={self._material})"


def create_reissner_mindlin_shell(
    material: PlaneStressShell,
) -> ShellReissnerMindlin5p:
    """Create a :class:`ShellReissnerMindlin5p` element."""
    return ShellReissnerMindlin5p(material)
