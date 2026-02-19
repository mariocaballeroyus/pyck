"""B-spline basis functions on a one-dimensional parametric space."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck

from pyck.basis.basis import Basis


class BSpline(Basis):
    """B-spline basis functions on a one-dimensional parametric space."""

    cpp_object: _pyck.BSpline

    def __init__(self, degree: int, knots: list[float]) -> None:
        if not isinstance(degree, (int, np.integer)):
            raise TypeError(f"degree must be an integer, got {type(degree).__name__}")
        if degree < 0:
            raise ValueError(f"degree must be non-negative, got {degree}")

        if not hasattr(knots, '__len__'):
            raise TypeError("knots must be a sequence of floats")
        knots = [float(k) for k in knots]

        n_basis = len(knots) - degree - 1
        if n_basis < 1:
            raise ValueError(
                f"knot vector length ({len(knots)}) too short for degree {degree}; "
                f"need at least {degree + 2} knots"
            )

        for i in range(len(knots) - 1):
            if knots[i] > knots[i + 1]:
                raise ValueError(
                    f"knot vector must be non-decreasing; "
                    f"knots[{i}]={knots[i]} > knots[{i+1}]={knots[i+1]}"
                )

        self.cpp_object = _pyck.BSpline(degree, knots)

    @property
    def degree(self) -> int:
        return self.cpp_object.degree()

    @property
    def num_basis(self) -> int:
        return self.cpp_object.num_basis()

    @property
    def knots(self) -> list[float]:
        return self.cpp_object.knots()

    def eval(self, u: np.ndarray, order: int = 0) -> list[np.ndarray]:
        u = np.asarray(u, dtype=np.float64)
        if u.ndim != 1:
            raise ValueError(f"u must be a 1-D array, got shape {u.shape}")
        if u.size == 0:
            raise ValueError("u must not be empty")

        if not isinstance(order, (int, np.integer)):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError(f"order must be non-negative, got {order}")

        lo, hi = self.knots[0], self.knots[-1]
        if np.any(u < lo) or np.any(u > hi):
            raise ValueError(
                f"all values in u must be in [{lo}, {hi}]; "
                f"got range [{u.min()}, {u.max()}]"
            )

        return self.cpp_object.eval(u, order)

    def __repr__(self) -> str:
        return f"BSpline(degree={self.degree}, num_basis={self.num_basis})"
