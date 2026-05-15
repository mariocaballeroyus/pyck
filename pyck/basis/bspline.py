"""B-spline basis functions on a one-dimensional parametric domain."""

from __future__ import annotations

import pyck._pyck as _pyck

from pyck.basis.basis import Basis
from pyck.basis.knot_vector import KnotVector


class BSpline(Basis):
    """B-Spline basis function.

    Parameters
    ----------
    deg : int
        The polynomial degree.
    kv : KnotVector
        A :class:`KnotVector` instance defining the knot sequence.
    """

    _cpp_object: _pyck.BSpline

    def __init__(self, deg: int, kv: KnotVector) -> None:
        super().__init__(kv=kv)
        self._cpp_object = _pyck.BSpline(deg, kv._cpp_object)

    def __repr__(self) -> str:
        return (
            f"BSpline(degree={self.degree}, num_basis={self.num_basis}, "
            f"knot_vector={self.knot_vector})"
        )

    # === Class Methods ===============================================================

    @classmethod
    def clamped_uniform(cls, deg: int, num_basis: int) -> BSpline:
        """Create a B-spline basis with a clamped uniform knot vector."""
        return cls(deg, KnotVector.clamped_uniform(deg, num_basis))
