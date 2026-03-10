"""Timoshenko beam element formulation handling."""

from __future__ import annotations

from pyck.elements.element import Element
from pyck import _pyck


class TimoshenkoBeam2P:
    r"""Timoshenko beam finite element.

    Wraps :class:`pyck::TimoshenkoBeam2P<double>`.

    This formulation treats the displacement `w` and cross-section
    rotation `theta` as independent degrees of freedom.

    Parameters
    ----------
    youngs_modulus : float
        Young's modulus ($E$).
    section_area : float
        Cross-sectional area ($A$).
    moment_inertia : float
        Second moment of area ($I$).
    shear_modulus : float
        Shear modulus ($G$).
    shear_coefficient : float
        Timoshenko shear coefficient ($k$, default 5/6).
    """

    def __init__(
        self,
        youngs_modulus: float,
        section_area: float,
        moment_inertia: float,
        shear_modulus: float,
        shear_coefficient: float = 5.0 / 6.0,
    ) -> None:
        self._cpp_object = _pyck.TimoshenkoBeam2P(
            float(youngs_modulus), float(section_area), float(moment_inertia), float(shear_modulus), float(shear_coefficient)
        )
        self._E = youngs_modulus
        self._A = section_area
        self._I = moment_inertia
        self._G = shear_modulus
        self._k = shear_coefficient
        self._kGA = shear_coefficient * shear_modulus * section_area

    @property
    def youngs_modulus(self) -> float:
        return self._E

    @property
    def section_area(self) -> float:
        return self._A

    @property
    def moment_inertia(self) -> float:
        return self._I

    @property
    def shear_rigidity(self) -> float:
        return self._kGA

    @property
    def shear_modulus(self) -> float:
        return self._G

    @property
    def shear_coefficient(self) -> float:
        return self._k

    def __repr__(self) -> str:
        return f"TimoshenkoBeam2P(E={self._E}, A={self._A}, I={self._I}, G={self._G}, k={self._k})"

def create_timoshenko_beam(
    youngs_modulus: float,
    section_area: float,
    moment_inertia: float,
    shear_modulus: float,
    shear_coefficient: float = 5.0 / 6.0,
) -> TimoshenkoBeam2P:
    """Create a :class:`TimoshenkoBeam2P` element.

    Parameters
    ----------
    youngs_modulus : float
        Young's modulus ($E$).
    section_area : float
        Cross-sectional area ($A$).
    moment_inertia : float
        Second moment of area ($I$).
    shear_modulus : float
        Shear modulus ($G$).
    shear_coefficient : float
        Timoshenko shear coefficient ($k$, default 5/6).
        
    Returns
    -------
    TimoshenkoBeam2P
    """
    return TimoshenkoBeam2P(
        youngs_modulus, section_area, moment_inertia, shear_modulus, shear_coefficient
    )

