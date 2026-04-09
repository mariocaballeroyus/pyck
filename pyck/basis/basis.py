"""Abstract base class for one-dimensional basis function families.

Defines the shared interface and implementation for B-splines and other
polynomial bases, including knot vector access and derivative evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np

import pyck._pyck as _pyck

from pyck.basis.knot_vector import KnotVector


class Basis(ABC):
    """Abstract base class for one-dimensional basis function families."""
    
    _cpp_object: _pyck.Basis
    _knot_vector: KnotVector

    def __init__(self, kv: KnotVector) -> None:
        self._knot_vector = kv

    @classmethod
    @abstractmethod
    def _from_cpp(cls, cpp_obj: _pyck.Basis) -> Basis: 
        """Wrap an existing C++ `Basis` object."""

    @property
    def knot_vector(self) -> KnotVector:
        """The :class:`KnotVector` associated with this basis."""
        return self._knot_vector

    @property
    def knots(self) -> np.ndarray:
        """Read-only access to the raw knot values."""
        return self._knot_vector.knots

    @cached_property
    def degree(self) -> int:
        """Polynomial degree of the basis functions."""
        return self._cpp_object.degree()

    @cached_property
    def num_basis(self) -> int:
        """Number of basis functions."""
        return self._cpp_object.num_basis()

    @cached_property
    def num_intervals(self) -> int:
        """Number of knot intervals."""
        return self._cpp_object.num_intervals()

    def find_span(self, u: float) -> int:
        """Find the index of the knot span containing *u*."""
        return self._cpp_object.find_span(float(u))

    def eval_all(
        self, u: np.ndarray, order: int = 0
    ) -> list[np.ndarray]:
        """Evaluate all basis functions and their derivatives.

        Parameters
        ----------
        u : array_like
            Parameter values at which to evaluate.
        order : int, optional
            Highest order of derivatives to compute (default 0).

        Returns
        -------
        list of ndarray
            A list of (order+1) matrices, where results[k] contains the k-th
            order derivatives (k=0, 1, ..., order). Each matrix has shape
            (m, n) with m evaluation points and n basis functions.
            When order=0, returns a single-element list.
        """
        if order < 0:
            raise ValueError(f"order must be non-negative, got {order}")

        pts = np.asarray(np.atleast_1d(u), dtype=np.float64)
        return self._cpp_object.eval_all(pts, order)
