"""Abstract base class for parametric patches."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Patch(ABC):
    """Abstract base class for tensor-product parametric patches."""

    @property
    @abstractmethod
    def gdim(self) -> int:
        """Geometric (embedding) dimension."""
        ...

    @property
    @abstractmethod
    def tdim(self) -> int:
        """Topological (parametric) dimension."""
        ...

    @property
    @abstractmethod
    def control_points(self) -> npt.NDArray[np.float64]:
        """Control-point matrix, shape ``(n_cp, gdim)``."""
        ...
