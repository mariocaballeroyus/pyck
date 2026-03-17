"""Beam element formulations."""

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

    def __repr__(self) -> str:
        return f"BeamEulerBernoulli1p(material={self._material})"


class BeamTimoshenko1p:
    """1-parameter Timoshenko beam element (rotation-free formulation).

    Wraps :class:`pyck::BeamTimoshenko1p<double>`.

    This formulation uses only the transverse displacement as the primary variable.
    The rotation is derived from the displacement gradient.

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

    def __repr__(self) -> str:
        return f"BeamTimoshenko1p(material={self._material})"


class BeamTimoshenko2p:
    r"""2-parameter Timoshenko beam element.

    Wraps :class:`pyck::BeamTimoshenko2p<double>`.

    This formulation treats the displacement `w` and cross-section
    rotation `theta` as independent degrees of freedom.

    Parameters
    ----------
    material : SlenderBeam1d
        Beam material model.
    """

    def __init__(self, material: SlenderBeam1d) -> None:
        if not isinstance(material, SlenderBeam1d):
            raise TypeError("material must be an instance of SlenderBeam1d")
        self._material = material
        self._cpp_object = _pyck.BeamTimoshenko2p(self._material._cpp_object)

    @property
    def material(self) -> SlenderBeam1d:
        return self._material

    def __repr__(self) -> str:
        return f"BeamTimoshenko2p(material={self._material})"


# Aliases for backward compatibility
EulerBernoulliBeam = BeamEulerBernoulli1p
TimoshenkoBeam1p = BeamTimoshenko1p
TimoshenkoBeam2p = BeamTimoshenko2p


# Factory functions
def create_euler_bernoulli_beam(material: SlenderBeam1d) -> BeamEulerBernoulli1p:
    """Create a :class:`BeamEulerBernoulli1p` element."""
    return BeamEulerBernoulli1p(material)


def create_timoshenko_beam(material: SlenderBeam1d) -> BeamTimoshenko2p:
    """Create a :class:`BeamTimoshenko2p` element."""
    return BeamTimoshenko2p(material)
