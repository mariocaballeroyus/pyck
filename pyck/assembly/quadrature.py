from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class QuadratureRule(ABC):
    """Abstract base class for numerical quadrature rules.

    A :class:`QuadratureRule` stores reference points and weights on [-1, 1]^d.
    """

    @property
    @abstractmethod
    def points(self) -> npt.NDArray[np.float64]:
        """Quadrature points on the reference element, shape `(Q, dim)`."""
        ...

    @property
    @abstractmethod
    def weights(self) -> npt.NDArray[np.float64]:
        """Quadrature weights, shape `(Q,)`."""
        ...

    @property
    @abstractmethod
    def num_points(self) -> int:
        """Total number of integration points."""
        ...

    @abstractmethod
    def __repr__(self) -> str:
        """String representation."""
        ...
