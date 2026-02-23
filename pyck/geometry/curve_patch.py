"""Curve patch class definition."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck

from pyck.basis.bspline import BSpline
from pyck.geometry.patch import Patch


class CurvePatch(Patch):
    """B-spline curve patch in parametric space (u)."""

    cpp_object: _pyck.CurvePatch

    def __init__(
        self,
        gdim: int,
        basis: BSpline,
        control_points: npt.NDArray[np.float64],
    ) -> None:
        """Create a B-spline curve patch.

        Args:
            gdim: Geometric (embedding) dimension.
            basis: A B-spline basis for the u direction.
            control_points: Control-point matrix of shape
                ``(n, gdim)``.

        Raises:
            TypeError: If ``gdim`` is not an integer or the basis
                is not a ``BSpline`` instance.
            ValueError: If ``gdim < 1`` or ``control_points`` has an
                incompatible shape.
        """
        # --- gdim ---
        if not isinstance(gdim, (int, np.integer)):
            raise TypeError(f"gdim must be an integer, got {type(gdim).__name__}")
        if gdim < 1:
            raise ValueError(f"gdim must be >= 1, got {gdim}")

        # --- basis ---
        if not isinstance(basis, BSpline):
            raise TypeError(
                f"basis must be a BSpline instance, got {type(basis).__name__}"
            )

        # --- control points ---
        control_points = np.asarray(control_points, dtype=np.float64)
        if control_points.ndim != 2:
            raise ValueError(
                f"control_points must be a 2-D array, got shape {control_points.shape}"
            )

        expected_rows = basis.num_basis
        if control_points.shape[0] != expected_rows:
            raise ValueError(
                f"control_points has {control_points.shape[0]} rows, "
                f"expected n = {expected_rows}"
            )
        if control_points.shape[1] != gdim:
            raise ValueError(
                f"control_points has {control_points.shape[1]} columns, "
                f"expected gdim = {gdim}"
            )

        self._basis = basis
        self.cpp_object = _pyck.CurvePatch(
            gdim,
            basis.cpp_object,
            control_points,
        )

    @property
    def gdim(self) -> int:
        """Geometric (embedding) dimension."""
        return self.cpp_object.gdim()

    @property
    def tdim(self) -> int:
        """Topological (parametric) dimension."""
        return self.cpp_object.tdim()

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        """Control-point matrix of shape ``(n_cp, gdim)``."""
        return self.cpp_object.control_points()

    @property
    def basis(self) -> BSpline:
        """B-spline basis in the u parametric direction."""
        return self._basis

    def eval(
        self,
        params: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluate the curve at the given parametric points.

        Args:
            params: Evaluation points, shape ``(Q, 1)``. Each row is ``(u,)``.

        Returns:
            Position array of shape ``(Q, gdim)``.

        Raises:
            ValueError: If *params* has invalid shape.
        """
        params = self._validate_params(params)
        return self.cpp_object.eval(params)

    def eval_derivs(
        self,
        params: npt.NDArray[np.float64],
        order: int = 1,
    ) -> list[npt.NDArray[np.float64]]:
        """Evaluate the curve and its parametric derivatives.

        Args:
            params: Evaluation points, shape ``(Q, 1)``. Each row is ``(u,)``.
            order: Maximum derivative order (default 1).

        Returns:
            A list indexed as ``result[k]`` for
            :math:`d^k C / du^k`, each of shape ``(Q, gdim)``.

        Raises:
            TypeError: If *order* is not an integer.
            ValueError: If *params* has invalid shape or *order* is negative.
        """
        params = self._validate_params(params)

        if not isinstance(order, (int, np.integer)):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError(f"order must be non-negative, got {order}")

        # Use the tensor product eval_derivs through the C++ side
        # For now, just do position + first-order via jacobian
        # We expose the eval and jacobian; higher-order parametric derivs
        # can be built from the basis directly.
        raise NotImplementedError("Use eval_physical_derivs for derivative evaluation")

    def jacobian(
        self,
        params: npt.NDArray[np.float64],
    ) -> list[npt.NDArray[np.float64]]:
        """Compute the Jacobian (tangent vector) at each evaluation point.

        Args:
            params: Evaluation points, shape ``(Q, 1)``. Each row is ``(u,)``.

        Returns:
            A list of Q arrays, each of shape ``(gdim, 1)``.
            The single column is dC/du.

        Raises:
            ValueError: If *params* has invalid shape.
        """
        params = self._validate_params(params)
        return self.cpp_object.jacobian(params)

    def jacobian_det(
        self,
        params: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the Jacobian determinant at each evaluation point.

        Args:
            params: Evaluation points, shape ``(Q, 1)``. Each row is ``(u,)``.

        Returns:
            A 1-D array of length Q containing ``||dC/du||`` at each point.

        Raises:
            ValueError: If *params* has invalid shape.
        """
        params = self._validate_params(params)
        return self.cpp_object.jacobian_det(params)

    def _validate_params(
        self, params: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Coerce and validate the (Q, 1) parameter array."""
        params = np.asarray(params, dtype=np.float64)
        if params.ndim != 2 or params.shape[1] != 1:
            raise ValueError(
                f"params must have shape (Q, 1), got {params.shape}"
            )
        return params

    def __repr__(self) -> str:
        n = self._basis.num_basis
        return (
            f"CurvePatch(gdim={self.gdim}, "
            f"basis={self._basis!r}, "
            f"control_points={n}×{self.gdim})"
        )
