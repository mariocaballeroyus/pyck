"""NURBS (Non-Uniform Rational B-Spline) basis functions on a 1D parametric domain."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck.basis.basis import Basis


class NURBS(Basis):
    """NURBS basis function.

    Use :meth:`clamped_uniform` to construct.
    """

    _cpp_object: _pyck.NURBS

    @classmethod
    def _from_cpp(cls, cpp_obj: _pyck.NURBS) -> NURBS:
        instance = cls.__new__(cls)
        instance._cpp_object = cpp_obj
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
            f"weights={self.weights})"
        )

    # === Refinement ==================================================================

    def insert_knot(
        self, u: float, count: int = 1
    ) -> tuple[NURBS, npt.NDArray[np.float64]]:
        """Refine the basis by inserting knot value `u` `count` times.

        The transform applies directly to unweighted control points; the
        refined weights are baked into the returned basis.
        """
        cpp = self._cpp_object
        transform: np.ndarray | None = None
        for _ in range(int(count)):
            new_cpp, step_transform = cpp.insert_knot(float(u))
            T_step = np.asarray(step_transform)
            transform = T_step if transform is None else T_step @ transform
            cpp = new_cpp
        assert isinstance(cpp, _pyck.NURBS)
        return NURBS._from_cpp(cpp), (transform if transform is not None
                                      else np.eye(self.num_basis))

    def elevate_degree(
        self, count: int = 1
    ) -> tuple[NURBS, npt.NDArray[np.float64]]:
        """Refine the basis by elevating the polynomial degree `count` times."""
        cpp = self._cpp_object
        transform: np.ndarray | None = None
        for _ in range(int(count)):
            new_cpp, step_transform = cpp.elevate_degree()
            T_step = np.asarray(step_transform)
            transform = T_step if transform is None else T_step @ transform
            cpp = new_cpp
        assert isinstance(cpp, _pyck.NURBS)
        return NURBS._from_cpp(cpp), (transform if transform is not None
                                      else np.eye(self.num_basis))

    # === Class Methods ===============================================================

    @classmethod
    def clamped_uniform(
        cls, deg: int, num_basis: int, weights: npt.ArrayLike
    ) -> NURBS:
        """Create a NURBS basis on a clamped uniform knot vector."""
        weights_arr = np.asarray(weights, dtype=np.float64)
        return cls._from_cpp(
            _pyck.NURBS.clamped_uniform(int(deg), int(num_basis), weights_arr)
        )
