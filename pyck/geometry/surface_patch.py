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
        bases: tuple[BSpline, BSpline] | list[BSpline],
        control_points: npt.NDArray[np.float64],
    ) -> None:
        """Create a tensor-product surface patch.

        Args:
            gdim: Geometric (embedding) dimension.
            bases: A pair of B-spline bases for the u and v directions.
            control_points: Control-point matrix of shape
                ``(n_u * n_v, gdim)``, ordered with v running fastest.

        Raises:
            TypeError: If ``gdim`` is not an integer or a basis argument
                is not a ``BSpline`` instance.
            ValueError: If ``gdim < 1`` or ``control_points`` has an
                incompatible shape.
        """
        # --- gdim ---
        if not isinstance(gdim, (int, np.integer)):
            raise TypeError(f"gdim must be an integer, got {type(gdim).__name__}")
        if gdim < 1:
            raise ValueError(f"gdim must be >= 1, got {gdim}")

        # --- bases ---
        if len(bases) != 2:
            raise ValueError(f"bases must have exactly 2 elements, got {len(bases)}")
        for i, b in enumerate(bases):
            if not isinstance(b, BSpline):
                raise TypeError(
                    f"bases[{i}] must be a BSpline instance, got {type(b).__name__}"
                )

        # --- control points ---
        control_points = np.asarray(control_points, dtype=np.float64)
        if control_points.ndim != 2:
            raise ValueError(
                f"control_points must be a 2-D array, got shape {control_points.shape}"
            )

        expected_rows = bases[0].num_basis * bases[1].num_basis
        if control_points.shape[0] != expected_rows:
            raise ValueError(
                f"control_points has {control_points.shape[0]} rows, "
                f"expected n_u * n_v = {bases[0].num_basis} * {bases[1].num_basis} "
                f"= {expected_rows}"
            )
        if control_points.shape[1] != gdim:
            raise ValueError(
                f"control_points has {control_points.shape[1]} columns, "
                f"expected gdim = {gdim}"
            )

        self._bases = list(bases)
        self.cpp_object = _pyck.SurfacePatch(
            gdim,
            [b.cpp_object for b in bases],
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
    def basis_u(self) -> BSpline:
        """B-spline basis in the u parametric direction."""
        return self._bases[0]

    @property
    def basis_v(self) -> BSpline:
        """B-spline basis in the v parametric direction."""
        return self._bases[1]

    def eval(
        self,
        params: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluate the surface at the given parametric points.

        Args:
            params: Evaluation points, shape ``(Q, 2)``. Each row is ``(u, v)``.

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
    ) -> list[list[npt.NDArray[np.float64]]]:
        """Evaluate the surface and its partial derivatives.

        Args:
            params: Evaluation points, shape ``(Q, 2)``. Each row is ``(u, v)``.
            order: Maximum total derivative order (default 1).

        Returns:
            A nested list indexed as ``result[i][j]`` for
            :math:`\\partial^{i+j} S / \\partial u^i \\partial v^j`,
            with ``0 <= i + j <= order``.  Each entry is an array of
            shape ``(Q, gdim)``.

        Raises:
            TypeError: If *order* is not an integer.
            ValueError: If *params* has invalid shape or *order* is negative.
        """
        params = self._validate_params(params)

        if not isinstance(order, (int, np.integer)):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError(f"order must be non-negative, got {order}")

        return self.cpp_object.eval_derivs(params, order)

    def jacobian(
        self,
        params: npt.NDArray[np.float64],
    ) -> list[npt.NDArray[np.float64]]:
        """Compute the Jacobian matrices at each evaluation point.

        Args:
            params: Evaluation points, shape ``(Q, 2)``. Each row is ``(u, v)``.

        Returns:
            A list of Q arrays, each of shape ``(gdim, 2)``.
            Column 0 is ∂S/∂u, column 1 is ∂S/∂v.

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
            params: Evaluation points, shape ``(Q, 2)``. Each row is ``(u, v)``.

        Returns:
            A 1-D array of length Q containing :math:`\\sqrt{\\det(J^T J)}`
            at each point.

        Raises:
            ValueError: If *params* has invalid shape.
        """
        params = self._validate_params(params)
        return self.cpp_object.jacobian_det(params)

    def _validate_params(
        self, params: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Coerce and validate the (Q, 2) parameter array."""
        params = np.asarray(params, dtype=np.float64)
        if params.ndim != 2 or params.shape[1] != 2:
            raise ValueError(
                f"params must have shape (Q, 2), got {params.shape}"
            )
        return params

    def __repr__(self) -> str:
        n_u = self._bases[0].num_basis
        n_v = self._bases[1].num_basis
        return (
            f"SurfacePatch(gdim={self.gdim}, "
            f"bases=[{self._bases[0]!r}, {self._bases[1]!r}], "
            f"control_points={n_u * n_v}×{self.gdim})"
        )
