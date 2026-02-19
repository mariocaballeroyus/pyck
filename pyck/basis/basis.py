"""Abstract base class for basis functions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Basis(ABC):
    """Abstract base class for 1-D basis functions."""

    @property
    @abstractmethod
    def degree(self) -> int:
        """Polynomial degree of the basis functions."""
        ...

    @property
    @abstractmethod
    def num_basis(self) -> int:
        """Number of basis functions."""
        ...

    @abstractmethod
    def eval(
        self,
        u: npt.NDArray[np.float64],
        order: int = 0,
    ) -> list[npt.NDArray[np.float64]]:
        """Evaluate basis functions and derivatives up to *order*.

        Parameters
        ----------
        u : (M,) array
            Parameter values.
        order : int
            Maximum derivative order (0 = values only).

        Returns
        -------
        list of (M, num_basis) arrays
            ``result[k]`` is the *k*-th derivative for *k = 0 … order*.
        """
        ...
