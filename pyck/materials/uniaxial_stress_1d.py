"""Material model for 1D beam and 3D frame elements."""

from __future__ import annotations

import pyck._pyck as _pyck
from pyck.materials import Material


class UniaxialStress1d(Material):
    """Isotropic linear-elastic uniaxial-stress material for beams and frames.

    Section axes: 1 = beam axis, 2 = strong (primary) bending axis,
                  3 = weak (secondary) bending axis.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    A : float
        Cross-sectional area.
    I22 : float
        Second moment of area about axis 2.
    k22 : float, optional
        Shear correction factor along axis 2. (default: 5/6)
    rho : float, optional
        Mass density. (default: 0)
    I33 : float, optional
        Second moment of area about axis 3. (default: 0)
    k33 : float, optional
        Shear correction factor along axis 3. (default: 0)
    J : float, optional
        Torsion constant (St-Venant). (default: 0)
    """

    def __init__(
        self, E: float, nu: float, A: float, I22: float,
        k22: float = 5.0 / 6.0, rho: float = 0.0,
        I33: float = 0.0, k33: float = 0.0, J: float = 0.0,
    ) -> None:
        self._E, self._nu, self._A, self._I22, self._k22 = E, nu, A, I22, k22
        self._rho, self._I33, self._k33, self._J = rho, I33, k33, J

        self._cpp_object = _pyck.UniaxialStress1d(
            self._E, self._nu, self._A, self._I22,
            self._k22, self._rho, self._I33, self._k33, self._J,
        )

    @property
    def youngs_modulus(self) -> float: return self._E
    @property
    def poisson_ratio(self) -> float: return self._nu
    @property
    def section_area(self) -> float: return self._A
    @property
    def moment_inertia_22(self) -> float: return self._I22
    @property
    def moment_inertia_33(self) -> float: return self._I33
    @property
    def shear_coefficient_22(self) -> float: return self._k22
    @property
    def shear_coefficient_33(self) -> float: return self._k33
    @property
    def torsion_constant(self) -> float: return self._J
    @property
    def density(self) -> float: return self._rho
    
    def bending_stiffness(self) -> float: return self._cpp_object.bending_stiffness()
    def shear_stiffness(self) -> float:   return self._cpp_object.shear_stiffness()

    def __repr__(self) -> str:
        return (f"UniaxialStress1d(E={self._E}, nu={self._nu}, A={self._A}, "
                f"I22={self._I22}, k22={self._k22}, rho={self._rho}, "
                f"I33={self._I33}, k33={self._k33}, J={self._J})")
