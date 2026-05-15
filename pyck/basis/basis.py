"""Abstract base class for one-dimensional basis function families.

Defines the shared interface and implementation for B-splines and other
polynomial bases, including knot vector access and derivative evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck.basis.knot_vector import KnotVector


class Basis(ABC):
    """Abstract base class for one-dimensional basis function families.
    
    Parameters
    ----------
    kv : KnotVector
        The knot vector for the basis.
    """
    
    _cpp_object: _pyck.Basis
    _knot_vector: KnotVector

    def __init__(self, kv: KnotVector) -> None:
        self._knot_vector = kv

    @property
    def knot_vector(self) -> KnotVector:
        """The Python :class:`KnotVector` associated with this basis."""
        return self._knot_vector

    @property
    def knots(self) -> npt.NDArray[np.float64]:
        """Read-only access to the raw knot values."""
        return self._knot_vector.knots

    @property
    def degree(self) -> int:
        """Polynomial degree of the basis functions."""
        return self._cpp_object.degree()

    @property
    def num_basis(self) -> int:
        """Number of basis functions."""
        return len(self.knots) - self.degree - 1

    @property
    def num_intervals(self) -> int:
        """Number of knot intervals (including zero-width clamped spans)."""
        return len(self.knots) - 1

    def eval_all(
        self, pts: npt.ArrayLike, order: int = 0
    ) -> list[npt.NDArray[np.float64]]:
        """Evaluate all basis functions and their derivatives.

        Parameters
        ----------
        pts : ArrayLike
            Values in the parameter space at which to evaluate the basis
            functions and their derivatives.
        order : int
            Highest order of derivatives to compute (default 0).

        Returns
        -------
        list[npt.NDArray]
            A list of (order+1) matrices, where results[k] contains the k-th
            order derivatives (k=0, 1, ..., order). 
            Each matrix has shape (m, n) with m evaluation points and 
            n basis functions.
        """
        if order < 0:
            raise ValueError(f"order must be non-negative, got {order}")

        pts = np.asarray(np.atleast_1d(pts), dtype=np.float64)
        return self._cpp_object.eval_all(pts, order)
