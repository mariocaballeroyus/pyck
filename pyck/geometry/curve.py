"""Curve patch.

A curve is defined by a univariate basis defined in the parametric space 
and a set of control points in the physical space. The mapping from 
parametric to physical space is: C(u) = sum_i N_{i,p}(u) * P_i
"""

from __future__ import annotations

import numpy as np

import pyck._pyck as _pyck

from pyck.basis import Basis
from pyck.geometry.patch import Patch
from pyck.geometry.patch_boundary import PatchBoundary


class CurvePatch(Patch):
    """Isogeometric curve patch embedded in the 3D physical space.

    Parameters
    ----------
    basis : Basis
        Univariate B-spline basis for the parametric direction.
    control_pts : np.ndarray
        Control-point coordinates, shape (n, 3). Rows must equal `basis.num_basis`.
    name : str, optional
        Human-readable label for the patch (default "patch").
    """

    _cpp_object: _pyck.Patch1d
    _basis: Basis

    def __init__(
        self, basis: Basis, control_pts: np.ndarray, *, name: str = "patch"
    ) -> None:
        control_pts = np.asarray(control_pts, dtype=np.float64)
        npts, dim = control_pts.shape
        if dim < 3:
            # Ensure 3D embedding by padding zeros if necessary
            pts_3d = np.zeros((npts, 3), dtype=np.float64)
            pts_3d[:, :dim] = control_pts
            control_pts = pts_3d

        self._basis = basis
        self._name = name
        self._cpp_object = _pyck.Patch1d(basis._cpp_object, control_pts)

    @classmethod
    def _from_cpp(
        cls, cpp_obj: _pyck.Patch1d, basis: Basis, *, name: str = "patch"
    ) -> CurvePatch:
        """Wrap an existing C++ `Patch1d` object."""
        obj = object.__new__(cls)
        obj._cpp_object = cpp_obj
        obj._basis = basis
        obj._name = name
        return obj

    @property
    def tdim(self) -> int:
        """Topological (parametric) dimension — always 1 for a curve."""
        return 1

    @property
    def basis(self) -> Basis:
        """B-spline basis in the u parametric direction."""
        return self._basis

    def boundary(self, side: str) -> PatchBoundary:
        """Extract a boundary from this patch.

        Parameters
        ----------
        side : "start" or "end"
            Which end of the parametric domain.

        Returns
        -------
        PatchBoundary
            Object carrying the DOF indices on this boundary.
        """
        if side not in ("start", "end"):
            raise ValueError(f"side must be 'start' or 'end', got '{side}'")

        at_start = side == "start"
        cpp_bp = self._cpp_object.boundary(0, at_start)
        return PatchBoundary(cpp_bp, parent=self)

    def eval_geometry(self, u: np.ndarray) -> np.ndarray:
        """Evaluate curve coordinates at given parametric values.

        Composed in numpy: for each point, find the containing span, evaluate
        the basis values N (order 0), and matrix-multiply by the active
        control points.

        Parameters
        ----------
        u : np.ndarray, shape (Q,)
            Parametric values at which to evaluate.

        Returns
        -------
        np.ndarray, shape (Q, 3)
            Physical coordinates.
        """
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        degree = self._basis.degree
        kv_cpp = self._basis.knot_vector._cpp_object
        coords = np.empty((len(u), 3), dtype=np.float64)

        for i, ui in enumerate(u):
            span = kv_cpp.find_span(degree, float(ui))
            pt = np.array([[ui]], dtype=np.float64)
            basis = self._cpp_object.eval_basis(pt, span, 0)
            act_pts = self._cpp_object.active_control_pts(span)
            coords[i, :] = (np.asarray(basis.N) @ np.asarray(act_pts))[0, :]

        return coords


    def eval_shape_functions(
        self, params: np.ndarray, span: int, order: int = 0
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Evaluate shape functions and their derivatives on a given span."""
        params = np.atleast_2d(np.asarray(params, dtype=np.float64))
        return self._cpp_object.eval_shape_functions(params, span, order)

    def evaluate(self, n_points: int = 200) -> np.ndarray:
        """Evaluate the curve at uniformly spaced parametric points.

        Parameters
        ----------
        n_points : int, optional
            Number of evaluation points (default 200).

        Returns
        -------
        np.ndarray, shape (n_points, 3)
            Physical coordinates along the curve.
        """
        knots = self._basis.knots
        u = np.linspace(knots[0], knots[-1], n_points)
        return self.eval_geometry(u)

    def __repr__(self) -> str:
        return (
            f"CurvePatch(name='{self.name}', "
            f"basis={self.basis!r}, "
            f"num_control_pts={self.num_control_pts})"
        )


def create_curve_patch(
    basis: Basis, control_pts: np.ndarray, *, name: str = "patch"
) -> CurvePatch:
    """Create a curve patch from a basis and control points.

    Parameters
    ----------
    basis : Basis
        Univariate basis function.
    control_pts : np.ndarray
        Control-point coordinates.
    name : str, optional
        Human-readable label for the patch (default "patch").

    Returns
    -------
    CurvePatch
        The newly created curve patch.
    """
    return CurvePatch(basis, control_pts, name=name)


def create_line_segment(
    basis: Basis, length: float, *, name: str = "patch"
) -> CurvePatch:
    """Create a straight line-segment patch along the x-direction.

    Control points are placed at the Greville abscissae such that the linear
    mapping is exact for any degree.

    Parameters
    ----------
    basis : Basis
        Univariate basis function.
    length : float
        Physical length of the segment.
    name : str, optional
        Human-readable label for the patch (default "patch").

    Returns
    -------
    CurvePatch
        The line-segment patch.
    """
    cpp = _pyck.line_segment(basis._cpp_object, float(length))
    return CurvePatch._from_cpp(cpp, basis, name=name)
