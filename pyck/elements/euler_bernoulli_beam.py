"""Euler-Bernoulli beam element formulation."""

from __future__ import annotations

import pyck._pyck as _pyck


class EulerBernoulliBeam:
    """Euler-Bernoulli beam element.

    Parameters
    ----------
    E : float
        Young's modulus.
    A : float
        Cross-section area.
    I : float
        Second moment of area (moment of inertia).
    """

    def __init__(self, E: float, A: float, I: float) -> None:
        self._cpp_object = _pyck.EulerBernoulliBeam1P(float(E), float(A), float(I))
        self._E, self._A, self._I = E, A, I

    @property
    def youngs_modulus(self) -> float:
        """Young's modulus."""
        return self._E

    @property
    def section_area(self) -> float:
        """Cross-section area."""
        return self._A

    @property
    def moment_inertia(self) -> float:
        """Second moment of area."""
        return self._I

    def __repr__(self) -> str:
        return f"EulerBernoulliBeam(E={self._E}, A={self._A}, I={self._I})"


def create_euler_bernoulli_beam(E: float, A: float, I: float) -> EulerBernoulliBeam:  # noqa: E741
    """Create an :class:`EulerBernoulliBeam` element.

    Parameters
    ----------
    E : float
        Young's modulus.
    A : float
        Cross-section area.
    I : float
        Second moment of area (moment of inertia).

    Returns
    -------
    EulerBernoulliBeam
    """
    return EulerBernoulliBeam(E, A, I)
