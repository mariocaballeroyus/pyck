"""Gauss-Legendre quadrature rules."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck.assembly.quadrature import QuadratureRule


class GaussLegendre:
    """Gauss-Legendre quadrature on [-1, 1] (strictly one-dimensional).

    Tensor-product methods are provided to extrapolate the 1D rule to
    surface or volume geometries without changing the stored rule itself.

    Parameters
    ----------
    num_points : int
        Number of integration points.  An *n*-point rule integrates
        polynomials up to degree *2n − 1* exactly.
    """

    _cpp_object: _pyck.GaussLegendre
    _points_view: np.ndarray
    _weights_view: np.ndarray

    def __init__(self, num_points: int) -> None:
        self._cpp_object = _pyck.GaussLegendre(int(num_points))
        self._points_view = np.asarray(self._cpp_object.points())
        self._weights_view = np.asarray(self._cpp_object.weights())

    @classmethod
    def _from_cpp(cls, cpp_obj: _pyck.GaussLegendre) -> GaussLegendre:
        """Wrap an existing C++ GaussLegendre object."""
        obj = object.__new__(cls)
        obj._cpp_object = cpp_obj
        obj._points_view = np.asarray(obj._cpp_object.points())
        obj._weights_view = np.asarray(obj._cpp_object.weights())
        return obj

    @property
    def points(self) -> npt.NDArray[np.float64]:
        """1D quadrature points on the reference element, shape `(Q,)`."""
        return self._points_view

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """1D quadrature weights, shape `(Q,)`."""
        return self._weights_view

    @property
    def num_points(self) -> int:
        """Number of 1D quadrature points."""
        return self._cpp_object.num_points()

    # === 1D Domain Mapping ===========================================================

    def map_to_domain(
        self, lo: float, hi: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Map the 1D reference points/weights from [-1, 1] to [lo, hi].

        Parameters
        ----------
        lo : float
            Lower bound of the target interval.
        hi : float
            Upper bound of the target interval.

        Returns
        -------
        points : ndarray, shape `(Q,)`
            Mapped quadrature points.
        weights : ndarray, shape `(Q,)`
            Mapped quadrature weights.
        """
        pts, wts = self._cpp_object.map_to_domain(lo, hi)
        return np.asarray(pts), np.asarray(wts)

    # === Tensor-Product Methods ======================================================

    def tensor_product(
        self,
        *others: QuadratureRule,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Form the tensor product of this rule with zero or more additional 1D rules.

        When called with no arguments, the result is identical to the 1D rule
        itself (points reshaped to `(Q, 1)`).

        Parameters
        ----------
        *others : QuadratureRule
            Additional 1D quadrature rules (one per extra parametric direction).

        Returns
        -------
        points : ndarray, shape `(Q_total, d)`
            Tensor-product quadrature points.
        weights : ndarray, shape `(Q_total,)`
            Tensor-product quadrature weights.

        Examples
        --------
        >>> gl = GaussLegendre(3)
        >>> pts_2d, wts_2d = gl.tensor_product(gl)   # 3×3 = 9-point 2D rule
        >>> pts_3d, wts_3d = gl.tensor_product(gl, gl)  # 27-point 3D rule
        """
        all_rules = [self] + list(others)
        d = len(all_rules)

        if d == 1:
            pts, wts = _pyck.tensor_product_1d(self._cpp_object)
        elif d == 2:
            pts, wts = _pyck.tensor_product_2d(
                all_rules[0]._cpp_object, all_rules[1]._cpp_object
            )
        elif d == 3:
            pts, wts = _pyck.tensor_product_3d(
                all_rules[0]._cpp_object,
                all_rules[1]._cpp_object,
                all_rules[2]._cpp_object,
            )
        else:
            raise ValueError(f"Tensor product not supported for d={d}")

        return np.asarray(pts), np.asarray(wts)

    # === Dunder Methods ==============================================================

    def __len__(self) -> int:
        """Number of 1D quadrature points."""
        return self._cpp_object.num_points()

    def __repr__(self) -> str:
        return f"GaussLegendre(num_points={self.num_points})"


def create_gauss_legendre(num_points: int) -> GaussLegendre:
    """Create a :class:`GaussLegendre` quadrature rule.

    Parameters
    ----------
    num_points : int
        Number of integration points.

    Returns
    -------
    GaussLegendre
    """
    return GaussLegendre(num_points)
