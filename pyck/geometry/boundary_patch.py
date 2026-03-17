"""Boundary patch representing a face or edge of a parent patch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyck._pyck as _pyck

if TYPE_CHECKING:
    from pyck.geometry.curve_patch import CurvePatch
    from pyck.geometry.surface_patch import SurfacePatch


class BoundaryPatch:
    """A boundary face of a parametric patch.

    Typically obtained via :func:`CurvePatch.boundary` or
    :func:`SurfacePatch.boundary` rather than constructed directly.

    Parameters
    ----------
    cpp_object : _pyck.BoundaryPatch1D | _pyck.BoundaryPatch2D
        The underlying C++ boundary patch object.
    parent : CurvePatch | SurfacePatch
        The Python-side parent patch.
    """

    _cpp_object: _pyck.BoundaryPatch1D | _pyck.BoundaryPatch2D
    _parent: CurvePatch | SurfacePatch

    def __init__(
        self,
        cpp_object: _pyck.BoundaryPatch1D | _pyck.BoundaryPatch2D,
        parent: CurvePatch | SurfacePatch,
    ) -> None:
        self._cpp_object = cpp_object
        self._parent = parent

    @property
    def displacement_dofs(self) -> list[int]:
        """Parent DOFs on the outermost boundary layer (displacement)."""
        return self._cpp_object.displacement_dofs()

    @property
    def rotation_dofs(self) -> list[int]:
        """Parent DOFs on the adjacent boundary layer (rotation / slope)."""
        return self._cpp_object.rotation_dofs()

    @property
    def boundary_dofs(self) -> list[int]:
        """All parent DOFs on both boundary layers."""
        return self._cpp_object.boundary_dofs()

    @property
    def param_dim(self) -> int:
        """Parametric direction normal to this boundary (0=u, 1=v, ...)."""
        return self._cpp_object.param_dim()

    @property
    def at_start(self) -> bool:
        """True if this boundary is at the start (lower bound) of the domain."""
        return self._cpp_object.at_start()

    @property
    def side(self) -> str:
        """Side identifier: 'start' or 'end'."""
        return "start" if self.at_start else "end"

    @property
    def parent(self) -> CurvePatch | SurfacePatch:
        """The parent patch this boundary was extracted from."""
        return self._parent

    def __repr__(self) -> str:
        return (
            f"BoundaryPatch(side='{self.side}', "
            f"param_dim={self.param_dim}, "
            f"disp_dofs={self.displacement_dofs}, "
            f"rot_dofs={self.rotation_dofs})"
        )
