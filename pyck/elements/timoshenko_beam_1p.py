"""Timoshenko beam 1-parameter element."""

from __future__ import annotations

import pyck._pyck as _pyck
from pyck.elements.element import Element


class TimoshenkoBeam1P:
    """1-parameter Timoshenko beam element.

    Wraps :class:`pyck::TimoshenkoBeam1P<double>`.

    Parameters
    ----------
    E : float
        Young's modulus.
    A : float
        Cross-sectional area.
    I : float
        Moment of inertia.
    G : float
        Shear modulus.
    k : float, default=5/6
        Shear correction factor.
    """

    def __init__(self, E: float, A: float, I: float, G: float, k: float = 5.0 / 6.0) -> None:
        self._cpp_object = _pyck.TimoshenkoBeam1P(
            float(E), float(A), float(I), float(G), float(k)
        )
        self.E = float(E)
        self.A = float(A)
        self.I = float(I)
        self.G = float(G)
        self.k = float(k)

    def __repr__(self) -> str:
        return f"TimoshenkoBeam1P({self.E=}, {self.A=}, {self.I=}, {self.G=}, {self.k=})"
