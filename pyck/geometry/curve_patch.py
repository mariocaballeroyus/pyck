"""B-spline curve patch."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck

from pyck.basis import Basis, BSpline
from pyck.geometry.boundary_patch import BoundaryPatch


class CurvePatch:
    """Isogeometric curve patch embedded in the 3D physical space."""

    def __init__(self, basis: Basis, control_pts: npt.ArrayLike, *, name: str = "patch") -> None:
        """Initialize a curve patch.

        Parameters
        ----------
        basis : Basis
            Univariate B-spline basis for the parametric direction.
        control_pts : ndarray, shape (n, 3)
            Control-point coordinates. Rows must equal ``basis.num_basis``.
        name : str, optional
            Human-readable label for the patch (default ``"patch"``).
        """
        if not isinstance(basis, Basis):
            raise TypeError(
                f"basis must be a Basis instance, got {type(basis).__name__}"
            )
        
        npts, dim = control_pts.shape
        if dim < 3:
            # Ensure 3D embedding by padding zeros if necessary
            pts_3d = np.zeros((npts, 3), dtype=np.float64)
            pts_3d[:, :dim] = control_pts
            control_pts = pts_3d
        
        self._basis: Basis                       = basis
        self._name: str                           = name
        self._cpp_object: _pyck.CurvePatch       = _pyck.CurvePatch(basis._cpp_object, control_pts)
        self._cpts_view: npt.NDArray[np.float64] = np.asarray(self._cpp_object.control_pts())

    @classmethod
    def _from_cpp(cls, cpp_obj: _pyck.CurvePatch, basis: Basis, *, name: str = "patch") -> CurvePatch:
        """Wrap an existing C++ CurvePatch (internal use)."""
        obj = object.__new__(cls)
        obj._cpp_object = cpp_obj
        obj._basis = basis
        obj._name = name
        obj._cpts_view = np.asarray(cpp_obj.control_pts())
        return obj

    @property
    def name(self) -> str:
        """Human-readable label for the patch."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def tdim(self) -> int:
        """Topological (parametric) dimension — always 1 for a curve."""
        return 1

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        """Control-point matrix, shape ``(n, gdim)`` — zero-copy view into C++."""
        return self._cpts_view

    @property
    def num_control_pts(self) -> int:
        """Number of control points."""
        return self._cpp_object.num_control_pts()

    @property
    def basis(self) -> Basis:
        """B-spline basis in the u parametric direction."""
        return self._basis

    def __repr__(self) -> str:
        n = self.num_control_pts
        return f"CurvePatch(name='{self._name}', basis={self._basis!r}, control_points={n})"

    def boundary(self, side: str) -> BoundaryPatch:
        """Extract a boundary from this patch.

        Parameters
        ----------
        side : ``"start"`` or ``"end"``
            Which end of the parametric domain.

        Returns
        -------
        BoundaryPatch
            Object carrying the DOF indices on this boundary.
        """
        if side not in ("start", "end"):
            raise ValueError(f"side must be 'start' or 'end', got '{side}'")
        at_start = side == "start"
        cpp_bp = self._cpp_object.boundary(0, at_start)
        return BoundaryPatch(cpp_bp, parent=self)


def create_curve_patch(basis: BSpline, control_pts: npt.ArrayLike, *, name: str = "patch") -> CurvePatch:
    """Create a curve patch from a basis and control points.

    Parameters
    ----------
    basis : Basis
        Univariate B-spline basis.
    control_pts : ndarray, shape (n, 3)
        Control-point coordinates.
    name : str, optional
        Human-readable label for the patch (default ``"patch"``).

    Returns
    -------
    The :class:`CurvePatch` instance.
    """
    return CurvePatch(basis, control_pts, name=name)


def create_line_segment(basis: BSpline, length: float, *, name: str = "patch") -> CurvePatch:
    """Create a straight line-segment patch along the x-direction.

    Control points are placed at the Greville abscissae so the linear
    mapping is exact for any degree.

    Parameters
    ----------
    basis : BSpline
        B-Spline basis for the parametric direction.
    length : float
        Physical length of the segment.
    name : str, optional
        Human-readable label for the patch (default ``"patch"``).

    Returns
    -------
    CurvePatch
        The :class:`CurvePatch` instance representing the line segment.
    """
    if not isinstance(basis, BSpline):
        raise TypeError(
            f"basis must be a BSpline instance, got {type(basis).__name__}"
        )
        
    cpp = _pyck.line_segment(basis._cpp_object, float(length))
    return CurvePatch._from_cpp(cpp, basis, name=name)
