"""Material model for 2D plane stress problems."""

from __future__ import annotations
import pyck._pyck as _pyck


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
