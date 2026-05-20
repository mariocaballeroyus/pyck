"""Abstract base class for parametric patches.

Patches are the building blocks of isogeometric geometry. All patches in pyck
are embedded in 3D physical space; this dimension is fixed and cannot be reduced.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck.basis.basis import Basis


class Patch(ABC):
    """Abstract base class for tensor-product parametric patches."""

    _cpp_object: _pyck.Patch1d | _pyck.Patch2d
    _name: str

    def __init__(self, *, name: str = "patch") -> None:
        self._name = name

    # === Properties ==================================================================

    @property
    def name(self) -> str:
        """Patch label."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def gdim(self) -> int:
        """Geometric (physical) dimension."""
        return 3

    @property
    @abstractmethod
    def tdim(self) -> int:
        """Topological (parametric) dimension."""
        ...

    @property
    @abstractmethod
    def basis(self) -> list[Basis]:
        """List of basis functions for each parametric dimension."""
        ...

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        """Read-only view of the control-point matrix, shape (n, 3)."""
        pts = np.asarray(self._cpp_object.control_pts())
        pts.flags.writeable = False
        return pts

    @property
    def num_control_pts(self) -> int:
        """Total number of control points."""
        return self.control_points.shape[0]
