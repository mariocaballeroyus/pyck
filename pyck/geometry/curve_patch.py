"""Curve patch."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck.basis import Basis, BSpline
from pyck.geometry.patch import Patch


class CurvePatch(Patch):
    """Isogeometric curve patch embedded in the 3D physical space.

    Parameters
    ----------
    basis : Basis
        Univariate B-spline basis for the parametric direction.
    cps : npt.ArrayLike
        Control-point coordinates, shape (n, 3).
    name : str, optional
        Human-readable label for the patch (default "patch").
    """

    _cpp_object: _pyck.Patch1d
    _basis: Basis

    def __init__(
        self, basis: Basis, cps: npt.ArrayLike, *, name: str = "patch"
    ) -> None:
        cps = np.asarray(cps, dtype=np.float64)
        num_control_pts, dim = cps.shape

        if basis.num_basis != num_control_pts:
            raise ValueError(
                f"Number of control points ({num_control_pts}) must equal "
                f"number of basis functions ({basis.num_basis})."
            )

        if dim > 3:
            raise ValueError(f"Control-point dimension must be 3 or less, got {dim}.")

        if dim < 3:  
            # Ensure 3d embedding by padding zeros if necessary
            pts_3d = np.zeros((num_control_pts, 3), dtype=np.float64)
            pts_3d[:, :dim] = cps
            cps = pts_3d

        super().__init__(name=name)
        self._basis = basis
        self._cpp_object = _pyck.Patch1d(basis._cpp_object, cps)

    @classmethod
    def _from_cpp(
        cls, cpp_obj: _pyck.Patch1d, basis: Basis, *, name: str = "patch",
    ) -> CurvePatch:
        instance = cls.__new__(cls)
        instance._cpp_object = cpp_obj
        instance._basis = basis
        instance._name = name
        return instance

    # === Properties ==================================================================

    @property
    def tdim(self) -> int: return 1

    @property
    def basis(self) -> list[Basis]: return [self._basis]

    def __repr__(self) -> str:
        return (
            f"CurvePatch(degree={self._basis.degree}, "
            f"num_basis={self._basis.num_basis}, name={self._name!r})"
        )

    # === Refinement ==================================================================

    def insert_knot(self, u: float, count: int = 1) -> CurvePatch:
        """Refine the patch by inserting knot value `u` `count` times.

        Returns a new patch with the refined basis and updated control points.
        The original patch is unmodified; any derived objects (assembler entries,
        boundary conditions) must be rebuilt against the returned patch.
        """
        new_basis, _ = self._basis.insert_knot(u, count)
        new_cpp = self._cpp_object.insert_knot(float(u), int(count))
        return CurvePatch._from_cpp(new_cpp, new_basis, name=self._name)

    def elevate_degree(self, count: int = 1) -> CurvePatch:
        """Refine the patch by elevating its polynomial degree `count` times."""
        new_basis, _ = self._basis.elevate_degree(count)
        new_cpp = self._cpp_object.elevate_degree(int(count))
        return CurvePatch._from_cpp(new_cpp, new_basis, name=self._name)

    # === Class Methods =============================================================

    @classmethod
    def line_segment(
        cls, nu: int, deg: int, l: float, *, name: str = "patch"
    ) -> CurvePatch:
        """Create a straight line-segment patch along the x-direction. Control points
        placed at the Greville abscissae.

        Parameters
        ----------
        l : float
            Physical length of the segment.
        nu : int
            Number of basis functions.
        deg : int
            Polynomial degree.
        name : str, optional
            Human-readable label for the patch (default "patch").

        Returns
        -------
        CurvePatch
            The line-segment patch.
        """
        basis = BSpline.clamped_uniform(deg, nu)
        xi = np.asarray(basis._cpp_object.greville_abscissae(), dtype=np.float64)

        u_min, u_max = xi[0], xi[-1]
        u_range = u_max - u_min if abs(u_max - u_min) > 1e-14 else 1.0

        control_pts = np.zeros((xi.size, 3), dtype=np.float64)
        control_pts[:, 0] = float(l) * (xi - u_min) / u_range

        return cls(basis, control_pts, name=name)
