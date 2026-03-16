"""Euler-Bernoulli beam element formulation."""

from __future__ import annotations

import pyck._pyck as _pyck
from pyck.materials import SlenderBeam1d


class BeamEulerBernoulli1p:
    """Euler-Bernoulli beam element.

    Parameters
    ----------
    material : SlenderBeam1d
        Beam material model.
    """

    def __init__(self, material: SlenderBeam1d) -> None:
        if not isinstance(material, SlenderBeam1d):
            raise TypeError("material must be an instance of SlenderBeam1d")
        self._material = material
        self._cpp_object = _pyck.BeamEulerBernoulli1p(self._material._cpp_object)

    @property
    def material(self) -> SlenderBeam1d:
        return self._material

    @property
    def youngs_modulus(self) -> float:
        return self._Youngs_modulus

    @youngs_modulus.getter
    def youngs_modulus(self) -> float:
        return self._material.youngs_modulus

    @property
    def section_area(self) -> float:
        return self._material.section_area

    @property
    def moment_inertia(self) -> float:
        return self._material.moment_inertia

    def __repr__(self) -> str:
        return f"BeamEulerBernoulli1p(material={self._material})"


def create_euler_bernoulli_beam(material: SlenderBeam1d) -> BeamEulerBernoulli1p:
    """Create a :class:`BeamEulerBernoulli1p` element."""
    return BeamEulerBernoulli1p(material)
