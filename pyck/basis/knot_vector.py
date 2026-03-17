"""Knot vector for basis functions.

Knot data is owned by the underlying C++ object. The :attr:`knots` property 
exposes it as a NumPy array without copying. This array is marked read-only 
to prevent accidental modification.
"""

from __future__ import annotations

from functools import cached_property
from collections.abc import Iterator, Sequence

import numpy as np

import pyck._pyck as _pyck


class KnotVector:
    """Knot sequence for basis function evaluation.

    Parameters
    ----------
    knots : Sequence[float]
        Knot values in non-decreasing order.
    """

    _cpp_object: _pyck.KnotVector

    def __init__(self, knots: Sequence[float]) -> None:
        self._cpp_object = _pyck.KnotVector(knots)

    @classmethod
    def _from_cpp(cls, cpp_obj: _pyck.KnotVector) -> KnotVector:
        """Wrap an existing C++ `KnotVector`."""
        obj = object.__new__(cls)
        obj._cpp_object = cpp_obj
        return obj

    @cached_property
    def knots(self) -> np.ndarray:
        """Read-only access to the raw knot values."""
        knots_arr = np.asarray(self._cpp_object.data())
        knots_arr.flags.writeable = False
        return knots_arr
    
    @property
    def num_spans(self) -> int:
        """Number of knot intervals."""
        return len(self.knots) - 1

    def num_basis(self, degree: int) -> int:
        """Number of basis functions for a given polynomial degree."""
        return len(self.knots) - degree - 1

    def __len__(self) -> int:
        return len(self.knots)

    def __getitem__(self, i: int | slice) -> float | np.ndarray:
        return self.knots[i]
    
    def __iter__(self) -> Iterator[float]:
        yield from self.knots

    def __repr__(self) -> str:
        return f"KnotVector(n={len(self)}, knots={self.knots})"


def create_knot_vector(knots: Sequence[float]) -> KnotVector:
    """Create a knot vector from a sequence of knot values.

    Parameters
    ----------
    knots : Sequence[float]
        Knot values in non-decreasing order.

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
        Polynomial degree (must be non-negative).
    num_basis : int
        Number of basis functions (must be >= degree + 1).

    Returns
    -------
    A :class:`KnotVector` instance.
    """
    cpp_object = _pyck.KnotVector.clamped_uniform(int(degree), int(num_basis))
    return KnotVector._from_cpp(cpp_object)
