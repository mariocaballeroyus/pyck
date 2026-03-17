"""Single-variable Reissner-Mindlin plate element."""

from __future__ import annotations

import pyck._pyck as _pyck
from pyck.materials import PlaneStress2d


class PlateReissnerMindlin1p:
    """Single-variable Reissner-Mindlin plate element (1 DOF per node: w_b).

    Uses a bending potential w_b as the sole primary variable.
    The total deflection is recovered via a Laplacian correction:
      w = w_b - (K_b / K_s) * Laplacian(w_b)

    Requires at least C^2 continuity (cubic B-splines or higher).

    Parameters
    ----------
    material : PlaneStress2d
        Plate material model.
    """

    def __init__(self, material: PlaneStress2d) -> None:
        if not isinstance(material, PlaneStress2d):
            raise TypeError("material must be an instance of PlaneStress2d")
        self._material = material
        self._cpp_object = _pyck.PlateReissnerMindlin1p(self._material._cpp_object)

    @property
    def material(self) -> PlaneStress2d:
        return self._material

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
        return f"PlateReissnerMindlin1p(material={self._material})"
