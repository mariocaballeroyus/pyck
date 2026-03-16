"""Reissner-Mindlin plate element."""

from __future__ import annotations

import pyck._pyck as _pyck
from pyck.materials import PlaneStress2d


class ReissnerMindlinPlate:
    """Reissner-Mindlin plate element (3 DOFs per node: w, theta_x, theta_y).

    Wraps :class:`pyck::ReissnerMindlinPlate<double>`.

    Parameters
    ----------
    material : PlaneStress2d
        Plate material model.
    k_s : float, optional
        Shear correction factor (default 5/6).
    """

    def __init__(self, material: PlaneStress2d, k_s: float = 5.0 / 6.0) -> None:
        if not isinstance(material, PlaneStress2d):
            raise TypeError("material must be an instance of PlaneStress2d")
            
        self._material = material
        self._k_s = float(k_s)
        self._cpp_object = _pyck.ReissnerMindlinPlate(self._material._cpp_object, self._k_s)

    @property
    def material(self) -> PlaneStress2d:
        return self._material

    @property
    def shear_coefficient(self) -> float:
        return self._k_s

    @property
    def youngs_modulus(self) -> float:
        return self._material.youngs_modulus

    @property
    def poisson_ratio(self) -> float:
        return self._material.poisson_ratio

    @property
    def thickness(self) -> float:
        return self._material.thickness

    def bending_stiffness(self) -> float:
        return self._material.bending_stiffness()

    def shear_stiffness(self) -> float:
        return self._material.shear_stiffness()

    def __repr__(self) -> str:
        return f"ReissnerMindlinPlate(material={self._material}, k_s={self._k_s})"
