"""B-spline basis functions on a one-dimensional parametric domain."""

from __future__ import annotations

from typing import Sequence

import pyck._pyck as _pyck

from pyck.basis.basis import Basis
from pyck.basis.knot_vector import KnotVector


class BSpline(Basis):
    """B-Spline (Basis-Spline) basis function.

    Parameters
    ----------
    degree : int
        The polynomial degree.
    kv : KnotVector
        A :class:`KnotVector` instance defining the knot sequence.
    """

    _cpp_object: _pyck.BSpline

    def __init__(self, deg: int, kv: KnotVector) -> None:
        super().__init__(kv)
        self._cpp_object = _pyck.BSpline(int(deg), kv._cpp_object)

    @classmethod
    def _from_cpp(cls, cpp_obj: _pyck.BSpline) -> BSpline:
        """Wrap an existing C++ `BSpline`."""
        obj = object.__new__(cls)
        obj._cpp_object = cpp_obj
        obj._knot_vector = KnotVector._from_cpp(cpp_obj.knot_vector())
        return obj

    def __repr__(self) -> str:
        return (
            f"BSpline(degree={self.degree}, num_basis={self.num_basis}, "
            f"knot_vector={self.knot_vector})"
        )


def create_bspline(degree: int, knots: KnotVector | Sequence[float]) -> BSpline:
    """Create a :class:`BSpline` basis.

    Parameters
    ----------
    degree : int
        The polynomial degree.
    knots : KnotVector or ArrayLike
        Either a :class:`KnotVector` instance or a sequence of knot values.

    Returns
    -------
    The :class:`BSpline` basis instance.
    """
    if not isinstance(knots, KnotVector):
        knots = KnotVector(knots)
    return BSpline(degree, knots)
