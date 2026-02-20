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
        """Create a tensor-product surface patch.

        Args:
            gdim: Geometric (embedding) dimension.
            basis_u: B-spline basis in the u parametric direction.
            basis_v: B-spline basis in the v parametric direction.
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
        return self._basis_u

    @property
    def basis_v(self) -> BSpline:
        """B-spline basis in the v parametric direction."""
        return self._basis_v

    def _validate_uv(
        self,
        u: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Coerce and validate parameter arrays.

        Args:
            u: Parameter values in the u direction.
            v: Parameter values in the v direction.

        Returns:
            The validated ``(u, v)`` arrays cast to float64.

        Raises:
            ValueError: If an array is not 1-D, empty, or out of range.
        """
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

        return u, v

    def eval(
        self,
        u: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        order: int = 0,
    ) -> list[list[npt.NDArray[np.float64]]]:
        """Evaluate the surface and its partial derivatives.

        Given vectors *u* (size M) and *v* (size N), the evaluation points
        form an M x N tensor-product grid.

        Args:
            u: Parameter values in the u direction (size M).
            v: Parameter values in the v direction (size N).
            order: Maximum total derivative order (default 0).

        Returns:
            A nested list indexed as ``result[i][j]`` for
            :math:`\\partial^{i+j} S / \\partial u^i \\partial v^j`,
            with ``0 <= i + j <= order``.  Each entry is an array of
            shape ``(M*N, gdim)``.

        Raises:
            TypeError: If *order* is not an integer.
            ValueError: If *u* or *v* are invalid or out of the knot range.
        """
        u, v = self._validate_uv(u, v)

        if not isinstance(order, (int, np.integer)):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError(f"order must be non-negative, got {order}")

        return self.cpp_object.eval(u, v, order)

    def jacobian(
        self,
        u: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
    ) -> tuple[list[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Compute the Jacobian matrix and determinant at each grid point.

        Args:
            u: Parameter values in the u direction (size M).
            v: Parameter values in the v direction (size N).

        Returns:
            A tuple ``(jacobians, det_jacobians)`` where *jacobians* is a
            list of ``(gdim, 2)`` arrays (columns are dS/du and dS/dv) and
            *det_jacobians* is a 1-D array of length ``M*N`` containing
            :math:`\\sqrt{\\det(J^T J)}` at each point.

        Raises:
            ValueError: If *u* or *v* are invalid or out of the knot range.
        """
        u, v = self._validate_uv(u, v)
        return self.cpp_object.jacobian(u, v)

    def tensor_basis(
        self,
        u: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        order: int = 0,
    ) -> list[list[npt.NDArray[np.float64]]]:
        """Build tensor-product basis matrices and parametric derivatives.

        This calls into C++ and avoids the Python/NumPy Kronecker product
        loop that would otherwise be needed.

        Args:
            u: Parameter values in the u direction (size M).
            v: Parameter values in the v direction (size N).
            order: Maximum total derivative order (default 0).

        Returns:
            A nested list indexed as ``result[i][j]`` for the
            ``(i, j)``-derivative of the tensor-product basis.  Each entry
            is an array of shape ``(M*N, n_u * n_v)``.

        Raises:
            TypeError: If *order* is not an integer.
            ValueError: If *u* or *v* are invalid or out of the knot range.
        """
        u, v = self._validate_uv(u, v)

        if not isinstance(order, (int, np.integer)):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError(f"order must be non-negative, got {order}")

        return self.cpp_object.tensor_basis(u, v, order)

    def __repr__(self) -> str:
        n_u = self._basis_u.num_basis
        n_v = self._basis_v.num_basis
        return (
            f"SurfacePatch(gdim={self.gdim}, "
            f"basis_u={self._basis_u!r}, basis_v={self._basis_v!r}, "
            f"control_points={n_u * n_v}×{self.gdim})"
        )
