"""Protocol for univariate basis functions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from pyck.basis.knot_vector import KnotVector


@runtime_checkable
class Basis(Protocol):
    """Interface for one-dimensional basis function families."""

    @property
    def degree(self) -> int:
        """Polynomial degree."""
        ...

    @property
    def num_basis(self) -> int:
        """Number of basis functions."""
        ...

    @property
    def knot_vector(self) -> KnotVector:
        """The :class:`KnotVector` associated with this basis."""
        ...

    @property
    def knots(self) -> npt.NDArray[np.float64]:
        """Zero-copy view of the raw knot values."""
        ...

    def __repr__(self) -> str:
        """String representation of the basis."""
        ...
        