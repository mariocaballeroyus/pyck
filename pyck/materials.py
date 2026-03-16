"""Material models for structural elements."""

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


class PlaneStress2d:
    """Linear elastic isotropic material model for plane stress.
    
    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float, optional
        Poisson's ratio (default 0.3).
    h : float, optional
        Thickness (default 1.0).
    k : float, optional
        Shear correction factor (default 5/6).
    """
    def __init__(self, E: float, nu: float = 0.3, h: float = 1.0, k: float = 5.0 / 6.0) -> None:
        self._E = float(E)
        self._nu = float(nu)
        self._h = float(h)
        self._k = float(k)
        self._cpp_object = _pyck.PlaneStress2d(self._E, self._nu, self._h, self._k)

    @property
    def youngs_modulus(self) -> float:
        return self._E

    @property
    def poisson_ratio(self) -> float:
        return self._nu

    @property
    def thickness(self) -> float:
        return self._h

    @property
    def shear_correction_factor(self) -> float:
        return self._k

    @property
    def shear_modulus(self) -> float:
        # G = E / (2 * (1 + nu))
        return self._E / (2.0 * (1.0 + self._nu))

    def bending_matrix(self):
        return self._cpp_object.bending_matrix()

    def shear_matrix(self):
        return self._cpp_object.shear_matrix()

    def bending_stiffness(self) -> float:
        return self._cpp_object.bending_stiffness()

    def shear_stiffness(self) -> float:
        return self._cpp_object.shear_stiffness()

    def __repr__(self) -> str:
        return f"PlaneStress2d(E={self._E}, nu={self._nu}, h={self._h}, k={self._k})"
