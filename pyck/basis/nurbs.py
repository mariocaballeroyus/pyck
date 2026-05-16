"""NURBS (Non-Uniform Rational B-Spline) basis functions on a 1D parametric domain."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck.basis.knot_vector import KnotVector
from pyck.basis.basis import Basis
from pyck.basis.bspline import BSpline


class NURBS(Basis):
    """NURBS (Non-Uniform Rational B-Spline) basis function.

    Parameters
    ----------
    deg : int
        The polynomial degree of the underlying B-spline.
    kv : KnotVector
        A :class:`KnotVector` instance defining the knot sequence.
    weights : npt.ArrayLike
        Per-basis-function weights (length `num_basis`), each strictly positive.
    """

    _cpp_object: _pyck.NURBS

    def __init__(self, deg: int, kv: KnotVector, weights: npt.ArrayLike) -> None:
        super().__init__(kv=kv)
        self._cpp_object = _pyck.NURBS(
            deg, kv._cpp_object, np.asarray(weights, dtype=np.float64)
        )

    @classmethod
    def _from_cpp(cls, cpp_obj: _pyck.NURBS, kv: KnotVector) -> NURBS:
        instance = cls.__new__(cls)
        instance._cpp_object = cpp_obj
        instance._knot_vector = kv
        return instance

    # === Properties ==================================================================

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Read-only access to the basis weights."""
        weights_arr = np.asarray(self._cpp_object.weights())
        weights_arr.flags.writeable = False
        return weights_arr

    def __repr__(self) -> str:
        return (
            f"NURBS(degree={self.degree}, num_basis={self.num_basis}, "
            f"knot_vector={self.knot_vector}, weights={self.weights})"
        )

    # === Refinement ==================================================================

    def insert_knot(
        self, u: float, count: int = 1
    ) -> tuple[NURBS, npt.NDArray[np.float64]]:
        """Refine the basis by inserting knot value `u` `count` times.

        The transform applies directly to unweighted control points; the
        refined weights are baked into the returned basis.
        """
        result = self._cpp_object.insert_knot(float(u), int(count))
        new_cpp = result.basis
        assert isinstance(new_cpp, _pyck.NURBS)
        new_basis = NURBS._from_cpp(new_cpp, self._knot_vector.insert(u, count))
        return new_basis, np.asarray(result.transform)

    def elevate_degree(
        self, count: int = 1
    ) -> tuple[NURBS, npt.NDArray[np.float64]]:
        """Refine the basis by elevating the polynomial degree `count` times.

        The transform applies directly to unweighted control points; the
        refined weights are baked into the returned basis.
        """
        result = self._cpp_object.elevate_degree(int(count))
        new_cpp = result.basis
        assert isinstance(new_cpp, _pyck.NURBS)
        new_basis = NURBS._from_cpp(new_cpp, self._knot_vector.elevate(count))
        return new_basis, np.asarray(result.transform)

    # === Class Methods ===============================================================

    @classmethod
    def clamped_uniform(cls, deg: int, num_basis: int, weights: npt.ArrayLike) -> NURBS:
        """Create a clamped uniform B-spline basis with given weights."""
        kv = KnotVector.clamped_uniform(deg, num_basis)
        return cls(deg, kv, weights)
