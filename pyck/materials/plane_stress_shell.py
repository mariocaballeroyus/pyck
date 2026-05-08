"""Plane-stress shell material (per-quadrature-point surface-tensor form)."""

from __future__ import annotations

import pyck._pyck as _pyck


class PlaneStressShell:
    """Isotropic linear-elastic plane-stress material for shell elements.

    Exposes per-quadrature-point Voigt constitutive matrices contracted with
    the surface inverse metric ``g^{αβ}`` — see ``docs/shell_reissner_mindlin_5p.md``.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float, optional
        Poisson's ratio (default 0.3).
    t : float, optional
        Shell thickness (default 1.0).
    k_s : float, optional
        Transverse-shear correction factor (default 5/6).
    """

    def __init__(self, E: float, nu: float = 0.3, t: float = 1.0,
                 k_s: float = 5.0 / 6.0) -> None:
        self._E = float(E)
        self._nu = float(nu)
        self._t = float(t)
        self._k_s = float(k_s)
        self._cpp_object = _pyck.PlaneStressShell(self._E, self._nu, self._t, self._k_s)

    @property
    def youngs_modulus(self) -> float:
        return self._E

    @property
    def poisson_ratio(self) -> float:
        return self._nu

    @property
    def thickness(self) -> float:
        return self._t

    @property
    def shear_correction_factor(self) -> float:
        return self._k_s

    def __repr__(self) -> str:
        return (
            f"PlaneStressShell(E={self._E}, nu={self._nu}, "
            f"t={self._t}, k_s={self._k_s})"
        )
