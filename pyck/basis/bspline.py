"""B-spline basis functions on a one-dimensional parametric domain."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck

from pyck.basis.basis import Basis


class BSpline(Basis):
    """B-spline basis function.

    Use :meth:`clamped_uniform` to construct.
    """

    _cpp_object: _pyck.BSpline

    @classmethod
    def _from_cpp(cls, cpp_obj: _pyck.BSpline) -> BSpline:
        instance = cls.__new__(cls)
        instance._cpp_object = cpp_obj
        return instance

    # === Properties ==================================================================

    def __repr__(self) -> str:
        return f"BSpline(degree={self.degree}, num_basis={self.num_basis})"

    # === Refinement ==================================================================

    def insert_knot(
        self, u: float, count: int = 1
    ) -> tuple[BSpline, npt.NDArray[np.float64]]:
        """Refine the basis by inserting knot value `u` `count` times.

        Returns the refined basis and the (n_new, n_old) control-point transform.
        """
        cpp = self._cpp_object
        transform: np.ndarray | None = None
        for _ in range(int(count)):
            new_cpp, step_transform = cpp.insert_knot(float(u))
            T_step = np.asarray(step_transform)
            transform = T_step if transform is None else T_step @ transform
            cpp = new_cpp
        assert isinstance(cpp, _pyck.BSpline)
        return BSpline._from_cpp(cpp), (transform if transform is not None
                                        else np.eye(self.num_basis))

    def elevate_degree(
        self, count: int = 1
    ) -> tuple[BSpline, npt.NDArray[np.float64]]:
        """Refine the basis by elevating the polynomial degree `count` times."""
        cpp = self._cpp_object
        transform: np.ndarray | None = None
        for _ in range(int(count)):
            new_cpp, step_transform = cpp.elevate_degree()
            T_step = np.asarray(step_transform)
            transform = T_step if transform is None else T_step @ transform
            cpp = new_cpp
        assert isinstance(cpp, _pyck.BSpline)
        return BSpline._from_cpp(cpp), (transform if transform is not None
                                        else np.eye(self.num_basis))

    # === Class Methods ===============================================================

    @classmethod
    def clamped_uniform(cls, deg: int, num_basis: int) -> BSpline:
        """Create a B-spline basis with a clamped uniform knot vector."""
        return cls._from_cpp(_pyck.BSpline.clamped_uniform(int(deg), int(num_basis)))
