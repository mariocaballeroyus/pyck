"""Kirchhoff-Love thin plate element formulation."""

from __future__ import annotations

import pyck._pyck as _pyck


class KirchhoffLovePlate:
    """Kirchhoff-Love thin plate element.

    Uses second covariant derivatives of the tensor-product B-spline
    basis to compute bending stiffness. Single DOF per control point
    (transverse displacement *w*).

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio (must be in [0, 0.5)).
    h : float
        Plate thickness.
    """

    def __init__(self, E: float, nu: float, h: float) -> None:
        self._cpp_object = _pyck.KirchhoffLovePlate(float(E), float(nu), float(h))
        self._E, self._nu, self._h = E, nu, h

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
    def flexural_rigidity(self) -> float:
        """D = E h^3 / (12 (1 - nu^2))."""
        return self._E * self._h ** 3 / (12.0 * (1.0 - self._nu ** 2))

    def __repr__(self) -> str:
        return f"KirchhoffLovePlate(E={self._E}, nu={self._nu}, h={self._h})"
