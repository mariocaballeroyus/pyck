"""1-parameter Timoshenko beam element."""

from __future__ import annotations

import pyck._pyck as _pyck
from pyck.materials import SlenderBeam1d


class BeamTimoshenko1p:
    """1-parameter Timoshenko beam element.

    Wraps :class:`pyck::TimoshenkoBeam1p<double>`.

    Parameters
    ----------
    material : SlenderBeam1d
        Beam material model.
    """

    def __init__(self, material: SlenderBeam1d) -> None:
        if not isinstance(material, SlenderBeam1d):
            raise TypeError("material must be an instance of SlenderBeam1d")
            
        self._material = material
        self._cpp_object = _pyck.BeamTimoshenko1p(self._material._cpp_object)

    @property
    def material(self) -> SlenderBeam1d:
        return self._material

    @property
    def youngs_modulus(self) -> float:
        return self._material.youngs_modulus

    @property
    def shear_modulus(self) -> float:
        return self._material.shear_modulus

    @property
    def section_area(self) -> float:
        return self._material.section_area

    @property
    def moment_inertia(self) -> float:
        return self._material.moment_inertia

    @property
    def shear_coefficient(self) -> float:
        return self._material.shear_coefficient

    def __repr__(self) -> str:
        return f"BeamTimoshenko1p(material={self._material})"
