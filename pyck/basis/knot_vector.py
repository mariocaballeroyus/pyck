"""Knot vector for basis functions.

Knot data is owned by the underlying C++ object. The :attr:`knots` property 
exposes it as a NumPy array without copying. This array is marked read-only 
to prevent accidental modification.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck


class KnotVector:
    """Knot sequence for basis function evaluation.

    Parameters
    ----------
    knots : ArrayLike
        Knot values in non-decreasing order.
    """

    _cpp_object: _pyck.KnotVector

    def __init__(self, knots: npt.ArrayLike) -> None:
        self._cpp_object = _pyck.KnotVector(
            np.asarray(knots, dtype=np.float64).ravel()
        )

    @classmethod
    def _from_cpp(cls, cpp_obj: _pyck.KnotVector) -> KnotVector:
        instance = cls.__new__(cls)
        instance._cpp_object = cpp_obj
        return instance

    def insert(self, u: float, count: int = 1) -> KnotVector:
        """Return a new knot vector with `u` inserted `count` times."""
        return KnotVector._from_cpp(self._cpp_object.insert(float(u), int(count)))

    def elevate(self, count: int = 1) -> KnotVector:
        """Return the knot vector for a `count`-fold degree elevation.

        Each unique knot's multiplicity is incremented by `count`.
        """
        return KnotVector._from_cpp(self._cpp_object.elevate(int(count)))

    @property
    def knots(self) -> npt.NDArray[np.float64]:
        """Read-only access to the raw knot values."""
        knots_arr = np.asarray(self._cpp_object.knots())
        knots_arr.flags.writeable = False
        return knots_arr
    
    @property
    def num_spans(self) -> int:
        """Number of knot intervals."""
        return len(self.knots) - 1

    def __len__(self) -> int:
        return len(self.knots)

    def __getitem__(self, i: int | slice) -> float | npt.NDArray[np.float64]:
        return self.knots[i]
    
    def __iter__(self) -> Iterator[float]:
        yield from self.knots

    def __repr__(self) -> str:
        return f"KnotVector(n={len(self)}, knots={self.knots})"

    @classmethod
    def clamped_uniform(cls, deg: int, num_basis: int) -> KnotVector:
        """Create a clamped, uniformly-spaced knot vector.
        
        Parameters
        ----------
        deg : int
            Polynomial degree of the basis functions.
        num_basis : int
            Number of basis functions.

        Returns
        -------
        A :class:`KnotVector` instance.
        """
        if deg < 0:
            raise ValueError("degree must be non-negative.")
        
        if num_basis < deg + 1:
            raise ValueError("num_basis must be at least degree + 1.")

        num_spans = num_basis - deg
        interior = np.linspace(0.0, 1.0, num_spans + 1)[1:-1]

        knots = np.concatenate([
            np.zeros(deg + 1),
            interior,
            np.ones(deg + 1),
        ])
        return cls(knots)
