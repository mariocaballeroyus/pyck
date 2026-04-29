"""Boundary patch representing a face or edge of a parent patch."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import pyck._pyck as _pyck

if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.geometry.curve import CurvePatch
    from pyck.geometry.surface import SurfacePatch


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

    @cached_property
    def quadrature(self) -> QuadratureRule:
        """Default Gauss-Legendre rule for integrating along this boundary.

        For a 2-D surface boundary this returns a 1-D rule with
        ``deg_tangent + 1`` points, where the tangent direction is the
        parametric direction *not* normal to the boundary. Used as the
        fallback rule by :class:`PenaltyCondition`,
        :class:`LagrangeMultiplierCondition`, and their factories when no
        explicit quadrature is provided.
        """
        from pyck.assembly.gauss import GaussLegendre

        parent_tdim = self._parent.tdim
        if parent_tdim != 2:
            raise NotImplementedError(
                f"BoundaryPatch.quadrature is only defined for boundaries "
                f"of 2-D patches (got parent tdim={parent_tdim}). Pass an "
                f"explicit quadrature rule instead."
            )
        tangent_dim = 1 - self.param_dim
        deg = self._parent.basis(tangent_dim).degree
        return GaussLegendre(deg + 1, dim=1)

    def __repr__(self) -> str:
        return (
            f"BoundaryPatch(side='{self.side}', "
            f"param_dim={self.param_dim}, "
            f"disp_dofs={self.displacement_dofs}, "
            f"rot_dofs={self.rotation_dofs})"
        )
