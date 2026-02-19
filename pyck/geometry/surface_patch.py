"""Surface patch class definition."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck

from pyck.basis.bspline import BSpline
from pyck.geometry.patch import Patch


class SurfacePatch(Patch):
    """Tensor-product surface patch in parametric space (u, v)."""

    cpp_object: _pyck.SurfacePatch

    def __init__(
        self,
        gdim: int,
        basis_u: BSpline,
        basis_v: BSpline,
        control_points: npt.NDArray[np.float64],
    ) -> None:
        # --- gdim ---
        if not isinstance(gdim, (int, np.integer)):
            raise TypeError(f"gdim must be an integer, got {type(gdim).__name__}")
        if gdim < 1:
            raise ValueError(f"gdim must be >= 1, got {gdim}")

        # --- bases ---
        if not isinstance(basis_u, BSpline):
            raise TypeError(
                f"basis_u must be a BSpline instance, got {type(basis_u).__name__}"
            )
        if not isinstance(basis_v, BSpline):
            raise TypeError(
                f"basis_v must be a BSpline instance, got {type(basis_v).__name__}"
            )

        # --- control points ---
        control_points = np.asarray(control_points, dtype=np.float64)
        if control_points.ndim != 2:
            raise ValueError(
                f"control_points must be a 2-D array, got shape {control_points.shape}"
            )

        expected_rows = basis_u.num_basis * basis_v.num_basis
        if control_points.shape[0] != expected_rows:
            raise ValueError(
                f"control_points has {control_points.shape[0]} rows, "
                f"expected n_u * n_v = {basis_u.num_basis} * {basis_v.num_basis} "
                f"= {expected_rows}"
            )
        if control_points.shape[1] != gdim:
            raise ValueError(
                f"control_points has {control_points.shape[1]} columns, "
                f"expected gdim = {gdim}"
            )

        self._basis_u = basis_u
        self._basis_v = basis_v
        self.cpp_object = _pyck.SurfacePatch(
            gdim,
            basis_u.cpp_object,
            basis_v.cpp_object,
            control_points,
        )

    @property
    def gdim(self) -> int:
        return self.cpp_object.gdim()

    @property
    def tdim(self) -> int:
        return self.cpp_object.tdim()

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        return self.cpp_object.control_points()

    @property
    def basis_u(self) -> BSpline:
        return self._basis_u

    @property
    def basis_v(self) -> BSpline:
        return self._basis_v

    def eval(
        self,
        u: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        order: int = 0,
    ) -> list[list[npt.NDArray[np.float64]]]:
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

        if u.ndim != 1:
            raise ValueError(f"u must be a 1-D array, got shape {u.shape}")
        if v.ndim != 1:
            raise ValueError(f"v must be a 1-D array, got shape {v.shape}")
        if u.size == 0:
            raise ValueError("u must not be empty")
        if v.size == 0:
            raise ValueError("v must not be empty")

        if not isinstance(order, (int, np.integer)):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError(f"order must be non-negative, got {order}")

        knots_u = self._basis_u.knots
        knots_v = self._basis_v.knots
        lo_u, hi_u = knots_u[0], knots_u[-1]
        lo_v, hi_v = knots_v[0], knots_v[-1]

        if np.any(u < lo_u) or np.any(u > hi_u):
            raise ValueError(
                f"all values in u must be in [{lo_u}, {hi_u}]; "
                f"got range [{u.min()}, {u.max()}]"
            )
        if np.any(v < lo_v) or np.any(v > hi_v):
            raise ValueError(
                f"all values in v must be in [{lo_v}, {hi_v}]; "
                f"got range [{v.min()}, {v.max()}]"
            )

        return self.cpp_object.eval(u, v, order)

    def __repr__(self) -> str:
        n_u = self._basis_u.num_basis
        n_v = self._basis_v.num_basis
        return (
            f"SurfacePatch(gdim={self.gdim}, "
            f"basis_u={self._basis_u!r}, basis_v={self._basis_v!r}, "
            f"control_points={n_u * n_v}×{self.gdim})"
        )
