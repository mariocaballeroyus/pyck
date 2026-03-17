"""Material model for 1D beam elements."""

from __future__ import annotations
import pyck._pyck as _pyck


class SlenderBeam1d:
    """Material and section properties for 1D beam elements.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    A : float
        Cross-sectional area.
    I : float
        Second moment of area.
    k : float, optional
        Shear correction factor (default 5/6).
    """

    def __init__(self, E: float, nu: float, A: float, I: float, k: float = 5.0 / 6.0) -> None:
        self._E = float(E)
        self._nu = float(nu)
        self._A = float(A)
        self._I = float(I)
        self._k = float(k)

        self._cpp_object = _pyck.SlenderBeam1d(self._E, self._nu, self._A, self._I, self._k)

    @property
    def youngs_modulus(self) -> float:
        return self._E

    @property
    def poisson_ratio(self) -> float:
        return self._nu

    @property
    def section_area(self) -> float:
        return self._A

    @property
    def moment_inertia(self) -> float:
        return self._I

    @property
    def shear_modulus(self) -> float:
        return self._cpp_object.shear_modulus()

    @property
    def shear_coefficient(self) -> float:
        return self._k

    def bending_stiffness(self) -> float:
        return self._cpp_object.bending_stiffness()

    def shear_stiffness(self) -> float:
        return self._cpp_object.shear_stiffness()

    def __repr__(self) -> str:
        return f"SlenderBeam1d(E={self._E}, nu={self._nu}, A={self._A}, I={self._I}, G={self.shear_modulus:.2e}, k={self._k})"
