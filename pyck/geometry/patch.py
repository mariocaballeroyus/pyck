"""Protocol for parametric patches."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class Patch(Protocol):
    """Interface for tensor-product parametric patches."""

    @property
    def name(self) -> str:
        """Human-readable label for the patch."""
        ...

    @property
    def tdim(self) -> int:
        """Topological (parametric) dimension."""
        ...

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        """Control-point matrix, shape `(n, gdim)`."""
        ...

    @property
    def num_control_pts(self) -> int:
        """Number of control points."""
        ...
    