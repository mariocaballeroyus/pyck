"""Abstract base class for one-dimensional basis function families.

Defines the shared interface and implementation for B-splines and other
polynomial bases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck


class Basis(ABC):
    """Abstract base class for one-dimensional basis function families."""

    _cpp_object: _pyck.Basis

    @property
    def knots(self) -> npt.NDArray[np.float64]:
        """Read-only access to the raw knot values."""
        return np.asarray(self._cpp_object.knots())

    @property
    def degree(self) -> int:
        """Polynomial degree of the basis functions."""
        return self._cpp_object.degree()

    @property
    def num_basis(self) -> int:
        """Number of basis functions."""
        return self._cpp_object.num_basis()

    @property
    def num_intervals(self) -> int:
        """Number of knot intervals (including zero-width clamped spans)."""
        return self._cpp_object.num_intervals()

    @abstractmethod
    def insert_knot(
        self, u: float, count: int = 1
    ) -> tuple[Basis, npt.NDArray[np.float64]]:
        """Return a refined basis and the (n_new, n_old) control-point transform."""
        ...

    @abstractmethod
    def elevate_degree(
        self, count: int = 1
    ) -> tuple[Basis, npt.NDArray[np.float64]]:
        """Return a degree-elevated basis and the (n_new, n_old) CP transform."""
        ...

    def eval_all(
        self, pts: npt.ArrayLike, order: int = 0
    ) -> list[npt.NDArray[np.float64]]:
        """Evaluate all basis functions and their derivatives.

        Returns
        -------
        list[npt.NDArray]
            A list of (order+1) matrices, where results[k] contains the k-th
            order derivatives. Each matrix has shape (m, n) with m evaluation
            points and n basis functions.
        """
        if order < 0:
            raise ValueError(f"order must be non-negative, got {order}")

        pts = np.atleast_1d(np.asarray(pts, dtype=np.float64))
        return self._cpp_object.eval_all(pts, order)
