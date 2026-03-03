"""Open knot vector for basis functions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck


class KnotVector:
    """Non-decreasing knot sequence for B-spline evaluation.

    Parameters
    ----------
    knots : sequence of float
        Non-decreasing knot values.
    """
    _cpp_object: _pyck.KnotVector
    _knots_view: np.ndarray

    def __init__(self, knots: npt.ArrayLike) -> None:
        self._cpp_object = _pyck.KnotVector(knots)
        self._knots_view = np.asarray(self._cpp_object.data())

    @classmethod
    def _from_cpp(cls, cpp_object: _pyck.KnotVector) -> KnotVector:
        """Wrap an existing C++ KnotVector (internal use)."""
        obj = object.__new__(cls)
        obj._cpp_object = cpp_object
        obj._knots_view = np.asarray(cpp_object.data())
        return obj

    def __len__(self) -> int:
        """Number of knots."""
        return len(self._knots_view)

    def __getitem__(self, i: int) -> float:
        """Knot value at index *i*."""
        return float(self._knots_view[i])

    def __iter__(self):
        yield from self._knots_view

    @property
    def data(self) -> np.ndarray:
        """Zero-copy view of the raw knot values (read-only NumPy array)."""
        return self._knots_view

    @property
    def num_spans(self) -> int:
        """Total number of knot spans (including zero-length clamped ones)."""
        return self._cpp_object.num_spans()

    def __repr__(self) -> str:
        return f"KnotVector({self.data})"


def create_knot_vector(knots: Sequence[float]) -> KnotVector:
    """Create a knot vector from a sequence of knot values.

    Parameters
    ----------
    knots : sequence of float
        Knot values.

    Returns
    -------
    A :class:`KnotVector` instance.
    """
    return KnotVector(knots)


def create_clamped_uniform(degree: int, num_basis: int) -> KnotVector:
    """Create a clamped, uniformly-spaced knot vector.
    
    Parameters
    ----------
    degree : int
        Polynomial degree.
    num_basis : int
        Number of basis functions.
    
    Returns
    -------
    A :class:`KnotVector` instance.
    """
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"degree must be an integer, got {type(degree).__name__}")
    
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    
    if not isinstance(num_basis, (int, np.integer)):
        raise TypeError(
            f"num_basis must be an integer, got {type(num_basis).__name__}"
        )
    
    if num_basis < degree + 1:
        raise ValueError(
            f"num_basis ({num_basis}) must be >= degree + 1 ({degree + 1})"
        )
    
    cpp = _pyck.KnotVector.clamped_uniform(int(degree), int(num_basis))
    return KnotVector._from_cpp(cpp)
