"""B-spline basis functions on a one-dimensional parametric domain."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

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

    @classmethod
    def _from_cpp(cls, cpp_obj: _pyck.BSpline, kv: KnotVector) -> BSpline:
        instance = cls.__new__(cls)
        instance._cpp_object = cpp_obj
        instance._knot_vector = kv
        return instance

    # === Properties ==================================================================

    def __repr__(self) -> str:
        return (
            f"BSpline(degree={self.degree}, num_basis={self.num_basis}, "
            f"knot_vector={self.knot_vector})"
        )

    # === Refinement ==================================================================

    def insert_knot(
        self, u: float, count: int = 1
    ) -> tuple[BSpline, npt.NDArray[np.float64]]:
        """Refine the basis by inserting knot value `u` `count` times.

        Returns the refined basis and the (n_new, n_old) control-point transform.
        """
        result = self._cpp_object.insert_knot(float(u), int(count))
        new_cpp = result.basis
        assert isinstance(new_cpp, _pyck.BSpline)
        new_basis = BSpline._from_cpp(new_cpp, self._knot_vector.insert(u, count))
        return new_basis, np.asarray(result.transform)

    def elevate_degree(
        self, count: int = 1
    ) -> tuple[BSpline, npt.NDArray[np.float64]]:
        """Refine the basis by elevating the polynomial degree `count` times.

        Continuity at every existing internal knot is preserved (p-refinement).
        For maximum smoothness at new knots, elevate first then `insert_knot`
        (k-refinement).
        """
        result = self._cpp_object.elevate_degree(int(count))
        new_cpp = result.basis
        assert isinstance(new_cpp, _pyck.BSpline)
        new_basis = BSpline._from_cpp(new_cpp, self._knot_vector.elevate(count))
        return new_basis, np.asarray(result.transform)

    # === Class Methods ===============================================================

    @classmethod
    def clamped_uniform(cls, deg: int, num_basis: int) -> BSpline:
        """Create a B-spline basis with a clamped uniform knot vector."""
        return cls(deg, KnotVector.clamped_uniform(deg, num_basis))
