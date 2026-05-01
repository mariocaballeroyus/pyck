"""NURBS (Non-Uniform Rational B-Spline) basis functions on a 1D parametric domain."""

from __future__ import annotations

from typing import Sequence

import numpy as np

import pyck._pyck as _pyck

from pyck.basis.basis import Basis
from pyck.basis.knot_vector import KnotVector


class NURBS(Basis):
    """NURBS (Non-Uniform Rational B-Spline) basis function.

    The rational shape functions are

    .. math::

        R_{i,p}(u) = \\frac{w_i\\, N_{i,p}(u)}{\\sum_j w_j\\, N_{j,p}(u)}

    where :math:`N_{i,p}` are the underlying B-spline functions of degree
    :math:`p` on the given knot vector and :math:`w_i > 0` are user-supplied
    weights (one per basis function). NURBS can represent conics (circles,
    ellipses, parabolas, hyperbolas) exactly.

    Parameters
    ----------
    deg : int
        The polynomial degree of the underlying B-spline.
    kv : KnotVector
        A :class:`KnotVector` instance defining the knot sequence.
    weights : Sequence[float]
        Per-basis-function weights (length ``num_basis``), each strictly
        positive.
    """

    _cpp_object: _pyck.NURBS

    def __init__(self, deg: int, kv: KnotVector, weights: Sequence[float]) -> None:
        super().__init__(kv)
        w = list(float(x) for x in weights)
        self._cpp_object = _pyck.NURBS(int(deg), kv._cpp_object, w)

    @classmethod
    def _from_cpp(cls, cpp_obj: _pyck.NURBS) -> NURBS:
        """Wrap an existing C++ `NURBS`."""
        obj = object.__new__(cls)
        obj._cpp_object = cpp_obj
        obj._knot_vector = KnotVector._from_cpp(cpp_obj.knot_vector())
        return obj

    @property
    def weights(self) -> np.ndarray:
        """Read-only access to the basis weights."""
        return np.asarray(self._cpp_object.weights())

    def __repr__(self) -> str:
        return (
            f"NURBS(degree={self.degree}, num_basis={self.num_basis}, "
            f"knot_vector={self.knot_vector}, weights={self.weights})"
        )


def create_nurbs(
    degree: int,
    knots: KnotVector | Sequence[float],
    weights: Sequence[float],
) -> NURBS:
    """Create a :class:`NURBS` basis.

    Parameters
    ----------
    degree : int
        The polynomial degree.
    knots : KnotVector or ArrayLike
        Either a :class:`KnotVector` instance or a sequence of knot values.
    weights : Sequence[float]
        Per-basis-function weights, all strictly positive.

    Returns
    -------
    The :class:`NURBS` basis instance.
    """
    if not isinstance(knots, KnotVector):
        knots = KnotVector(knots)
    return NURBS(degree, knots, weights)
