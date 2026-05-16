"""Boundary patch representing a face or edge of a parent patch."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck.geometry.patch import Patch


class PatchBoundary:
    """A boundary face of a parametric patch.

    Parameters
    ----------
    parent : Patch
        The parent patch.
    param_dim : int
        Parametric direction normal to the edge (0 = u, 1 = v).
    at_start : bool
        True for the edge at the lower bound of that direction, False for
        the upper bound.
    """

    _cpp_object: _pyck.PatchBoundary2d
    _parent: Patch

    def __init__(
        self, parent: Patch, param_dim: int, at_start: bool,
    ) -> None:
        self._parent = parent
        cpp = parent._cpp_object

        if isinstance(cpp, _pyck.Patch2d):
            self._cpp_object = _pyck.PatchBoundary2d(cpp, param_dim, at_start)
        else:
            raise ValueError(
                f"Patch boundaries are not supported for {parent.tdim}-dimensional patches."
            )

    # === Properties ==================================================================

    @property
    def param_dim(self) -> int:
        """Parametric direction normal to this boundary (0 = u, 1 = v)."""
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
    def parent(self) -> Patch:
        """The parent patch this boundary was extracted from."""
        return self._parent

    def __repr__(self) -> str:
        return (
            f"PatchBoundary(side='{self.side}', "
            f"param_dim={self.param_dim}, "
            f"disp_dofs={self.displacement_dofs}, "
            f"rot_dofs={self.rotation_dofs})"
        )

    # === DOF Accesors ================================================================

    @property
    def displacement_dofs(self) -> npt.NDArray[np.int64]:
        """Parent DOFs on the outermost boundary layer (displacement)."""
        return np.asarray(self._cpp_object.displacement_dofs())

    @property
    def rotation_dofs(self) -> npt.NDArray[np.int64]:
        """Parent DOFs on the adjacent boundary layer (rotation / slope)."""
        return np.asarray(self._cpp_object.rotation_dofs())
