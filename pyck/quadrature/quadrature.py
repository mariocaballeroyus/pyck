"""Abstract base class for numerical quadrature rules."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck.geometry.patch import Patch


class QuadratureRule(ABC):
    """Abstract base class for numerical quadrature rules."""

    _cpp_object: _pyck.QuadratureRule1d | _pyck.QuadratureRule2d
    _dim: int

    def __init__(self, dim: int) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def points(self) -> npt.NDArray[np.float64]:
        """Read-only access to the quadrature points."""
        arr = np.asarray(self._cpp_object.points())
        arr.flags.writeable = False
        return arr

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Read-only access to the quadrature weights."""
        arr = np.asarray(self._cpp_object.weights())
        arr.flags.writeable = False
        return arr

    def __len__(self) -> int:
        return len(self.points)

    @abstractmethod
    def __repr__(self) -> str:
        ... 

    # === Class Methods =============================================================

    @classmethod
    @abstractmethod
    def from_patch(cls, patch: Patch) -> QuadratureRule:
        """Create a quadrature rule suitable for the higherst-order basis function on 
        the given patch.

        Parameters
        ----------
        patch : Patch
            The patch to create the quadrature rule for.
        """
        ... 
