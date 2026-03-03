"""B-spline basis functions on a one-dimensional parametric domain."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck

from pyck.basis.basis import Basis
from pyck.basis.knot_vector import KnotVector


class BSpline(Basis):
    """B-spline basis function."""

    def __init__(self, degree: int, knots: KnotVector) -> None:
        """Initialize a B-spline basis.
        
        Parameters
        ----------
        degree : int
            The polynomial degree.
        knots : KnotVector
            A :class:`KnotVector` instance defining the knot sequence.
        """
        if not isinstance(knots, KnotVector):
            raise TypeError(
                f"knots must be a KnotVector, got {type(knots).__name__}"
            )

        self._cpp_object: _pyck.BSpline = _pyck.BSpline(int(degree), knots._cpp_object)
        self._knots: KnotVector         = knots

    @classmethod
    def _from_cpp(cls, cpp_obj: _pyck.BSpline) -> BSpline:
        """Wrap an existing C++ BSpline (internal use)."""
        obj = object.__new__(cls)
        obj._cpp_object = cpp_obj
        knots_cpp = _pyck.KnotVector(list(cpp_obj.knots()))
        obj._knots = KnotVector._from_cpp(knots_cpp)
        return obj

    @property
    def degree(self) -> int:
        """Polynomial degree."""
        return self._cpp_object.degree()

    @property
    def num_basis(self) -> int:
        """Number of basis functions."""
        return self._cpp_object.num_basis()

    @property
    def knot_vector(self) -> KnotVector:
        """The :class:`KnotVector` associated with this basis."""
        return self._knots

    @property
    def knots(self) -> npt.NDArray[np.float64]:
        """Zero-copy view of the raw knot values (read-only NumPy array)."""
        return self._knots.data

    def __repr__(self) -> str:
        """String representation of the basis."""
        return (
            f"BSpline(degree={self.degree}, num_basis={self.num_basis}, "
            f"num_knots={len(self._knots)})"
        )


def create_bspline(degree: int, knots: KnotVector) -> BSpline:
    """Create a :class:`BSpline` basis.

    Parameters
    ----------
    degree : int
        The polynomial degree.
    knots : KnotVector
        A :class:`KnotVector` instance defining the knot sequence.

    Returns
    -------
    The :class:`BSpline` basis instance.
    """
    return BSpline(degree, knots)
