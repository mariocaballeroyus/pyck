"""Quadrature rules for numerical integration."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class QuadratureRule(Protocol):
    """Protocol for strictly one-dimensional quadrature rules.

    A :class:`QuadratureRule` stores 1D reference points and weights on [-1, 1].
    Tensor-product methods are provided to extrapolate to surface or volume
    geometries.
    """

    @property
    def points(self) -> npt.NDArray[np.float64]:
        """1D quadrature points on the reference element, shape `(Q,)`."""
        ...

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """1D quadrature weights, shape `(Q,)`."""
        ...

    @property
    def num_points(self) -> int:
        """Number of 1D quadrature points."""
        ...

    @property
    def _cpp_object(self) -> Any:
        """The underlying C++ quadrature object."""
        ...

    def map_to_domain(
        self, lo: float, hi: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Map the 1D reference points/weights from [-1, 1] to [lo, hi]."""
        ...

    def __repr__(self) -> str:
        """String representation."""
        ...
