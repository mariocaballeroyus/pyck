"""Gauss-Legendre quadrature rules for different topological dimensions."""
 
from __future__ import annotations
 
from typing import Any, Sequence, overload
 
import numpy as np
import numpy.typing as npt
 
import pyck._pyck as _pyck
from pyck.assembly.quadrature import QuadratureRule


class GaussLegendre(QuadratureRule):
    """Gauss-Legendre tensor-product quadrature rule.
 
    This class provides a unified interface for 1D, 2D, and 3D Gauss-Legendre
    quadrature rules. It wraps the underlying C++ implementations and ensures
    consistent array shapes across dimensions.
 
    Parameters
    ----------
    num_points : int or Sequence[int]
        Number of integration points per direction. For 2D and 3D, if an
        integer is provided, the same number of points is used in each
        parametric direction (uniform rule). If a sequence is provided, its
        length must match ``dim``.
    dim : int, optional
        Topological dimension (1, 2, or 3). If not provided, it is inferred
        from ``num_points`` if it's a sequence, otherwise defaults to 1.
    """
 
    _cpp_object: _pyck.GaussLegendre | _pyck.GaussLegendre2d | _pyck.GaussLegendre3d
    _dim: int
 
    @overload
    def __init__(self, num_points: int, dim: int = 1) -> None: ...
 
    @overload
    def __init__(self, num_points: Sequence[int], dim: int | None = None) -> None: ...
 
    def __init__(
        self, num_points: int | Sequence[int], dim: int | None = None
    ) -> None:
        if isinstance(num_points, Sequence):
            if dim is not None and len(num_points) != dim:
                raise ValueError(
                    f"Length of num_points ({len(num_points)}) must match dim ({dim})"
                )
            dim = len(num_points)
            if not all(n == num_points[0] for n in num_points):
                raise NotImplementedError(
                    "Non-uniform number of points per direction is not yet supported "
                    "by the C++ backend for tensor-product rules."
                )
            num_points = num_points[0]
 
        if dim is None:
            dim = 1
 
        if dim == 1:
            self._cpp_object = _pyck.GaussLegendre(int(num_points))
        elif dim == 2:
            self._cpp_object = _pyck.GaussLegendre2d(int(num_points))
        elif dim == 3:
            self._cpp_object = _pyck.GaussLegendre3d(int(num_points))
        else:
            raise ValueError(f"Unsupported dimension: {dim}")
 
        self._dim = dim
 
    @classmethod
    def _from_cpp(
        cls, cpp_obj: _pyck.GaussLegendre | _pyck.GaussLegendre2d | _pyck.GaussLegendre3d
    ) -> GaussLegendre:
        """Wrap an existing C++ Gauss-Legendre object.
 
        Parameters
        ----------
        cpp_obj : C++ object
            Underlying C++ quadrature rule from :module:`_pyck`.
 
        Returns
        -------
        GaussLegendre
            Python wrapper for the C++ rule.
        """
        obj = object.__new__(cls)
        obj._cpp_object = cpp_obj
        if isinstance(cpp_obj, _pyck.GaussLegendre):
            obj._dim = 1
        elif isinstance(cpp_obj, _pyck.GaussLegendre2d):
            obj._dim = 2
        else:
            obj._dim = 3
        return obj
 
    @property
    def points(self) -> npt.NDArray[np.float64]:
        """Quadrature points on the reference element.
 
        Returns
        -------
        ndarray, shape `(Q, dim)`
            Array of quadrature points where `Q` is the total number of points.
        """
        pts = np.asarray(self._cpp_object.points())
        if self._dim == 1 and pts.ndim == 1:
            return pts[:, np.newaxis]
        return pts
 
    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Quadrature weights.
 
        Returns
        -------
        ndarray, shape `(Q,)`
            Array of quadrature weights.
        """
        wts = np.asarray(self._cpp_object.weights())
        if wts.ndim > 1:
            return wts.ravel()
        return wts
 
    @property
    def num_points(self) -> int:
        """Total number of integration points."""
        return self._cpp_object.num_points()
 
    @property
    def dim(self) -> int:
        """Topological dimension of the quadrature rule."""
        return self._dim
 
    def __len__(self) -> int:
        """Total number of quadrature points."""
        return self.num_points
 
    def __repr__(self) -> str:
        return f"GaussLegendre(num_points={self.num_points}, dim={self._dim})"
 
 
def create_gauss_legendre(num_points: int, dim: int = 1) -> GaussLegendre:
    """Create a :class:`GaussLegendre` quadrature rule.
 
    Parameters
    ----------
    num_points : int
        Number of integration points per direction.
    dim : int, optional
        Topological dimension (1, 2, or 3). Defaults to 1.
 
    Returns
    -------
    GaussLegendre
    """
    return GaussLegendre(num_points, dim=dim)
 
 
def create_gauss_legendre_2d(num_points: int) -> GaussLegendre:
    """Create a 2D tensor-product Gauss-Legendre quadrature rule.
 
    Parameters
    ----------
    num_points : int
        Number of integration points per direction.
 
    Returns
    -------
    GaussLegendre
    """
    return GaussLegendre(num_points, dim=2)


def create_gauss_from_patch(patch: Any) -> GaussLegendre:
    """Create a Gauss-Legendre quadrature rule suitable for a patch.
 
    The number of integration points per direction is set to degree + 1,
    which is sufficient to integrate the mass matrix of the patch exactly.
 
    Parameters
    ----------
    patch : Patch
        The patch to create the quadrature rule for.
 
    Returns
    -------
    GaussLegendre
    """
    # Find max degree across all parametric directions
    max_degree = 0
    for i in range(patch.tdim):
        max_degree = max(max_degree, patch.basis(i).degree)
    
    return GaussLegendre(max_degree + 1, dim=patch.tdim)


# Alias for backward compatibility
GaussLegendre2d = GaussLegendre
