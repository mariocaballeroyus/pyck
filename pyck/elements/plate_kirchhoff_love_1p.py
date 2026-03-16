"""Kirchhoff-Love thin plate element formulation."""

from __future__ import annotations

import pyck._pyck as _pyck

from pyck.materials import PlaneStress2d


class PlateKirchhoffLove1p:
    """Kirchhoff-Love thin plate element.

    Uses second covariant derivatives of the tensor-product B-spline
    basis to compute bending stiffness. Single DOF per control point
    (transverse displacement *w*).

    Parameters
    ----------
    material : PlaneStress2d
        Plate material model.
    """

    def __init__(self, material: PlaneStress2d) -> None:
        if not isinstance(material, PlaneStress2d):
            raise TypeError("material must be an instance of PlaneStress2d")
        self._material = material
        self._cpp_object = _pyck.PlateKirchhoffLove1p(self._material._cpp_object)

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

    @property
    def flexural_rigidity(self) -> float:
        """D = E h^3 / (12 (1 - nu^2))."""
        return self.youngs_modulus * self.thickness ** 3 / (12.0 * (1.0 - self.poisson_ratio ** 2))

    def __repr__(self) -> str:
        return f"PlateKirchhoffLove1p(material={self._material})"
