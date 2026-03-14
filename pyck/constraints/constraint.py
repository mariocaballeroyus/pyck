"""Python wrapper classes for boundary constraints."""

from __future__ import annotations

from typing import Sequence
import numpy as np

import pyck._pyck as _pyck


class Constraint:
    """Base class for boundary constraints.

    Subclasses wrap concrete C++ constraint types and expose a
    `_cpp_object` attribute used by the assembler.
    """

    _cpp_object: _pyck.Constraint

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class LinearConstraint(Constraint):
    """Generic linear multipoint constraint coupling DOFs.

    Enforces u_slave[k] = sum(weights[i] * u_masters[k * n_masters + i]) + constant

    Wraps :class:`pyck::LinearConstraint<double>`.

    Parameters
    ----------
    pairs : sequence of (int, int), optional
        `(master, slave)` DOF index pairs for 1-to-1 relationships.
    slaves : sequence of int, optional
        List of slave DOF indices.
    masters : 2D array-like of int, optional
        Matrix of master DOFs where row `k` corresponds to `slaves[k]`.
    weights : sequence of float, optional
        Shared coefficient weights.
    constant : float, optional
        Shared non-homogeneous offset offset.
    """

    def __init__(
        self, 
        pairs: Sequence[tuple[int, int]] | None = None,
        slaves: Sequence[int] | None = None,
        masters: Sequence[Sequence[int]] | None = None,
        weights: Sequence[float] | None = None,
        constant: float = 0.0
    ) -> None:
        if pairs is not None:
            self._slaves = [int(s) for m, s in pairs]
            # Internal C++ constructor for pairs handles matrix creation
            self._cpp_object = _pyck.LinearConstraint(list(zip([int(m) for m, s in pairs], self._slaves)))
            self._masters = self._cpp_object.masters()
            self._weights = self._cpp_object.weights()
            self._constant = self._cpp_object.constant()
        else:
            self._slaves = [int(s) for s in slaves] if slaves is not None else []
            # masters should be a 2D array (N_slaves, N_masters)
            self._masters = np.array(masters, dtype=np.uintp) if masters is not None else np.zeros((0, 0), dtype=np.uintp)
            self._weights = [float(w) for w in weights] if weights is not None else []
            self._constant = float(constant)
            self._cpp_object = _pyck.LinearConstraint(
                self._slaves, self._masters, self._weights, self._constant
            )

    @property
    def slaves(self) -> list[int]:
        return self._slaves

    @property
    def masters(self) -> np.ndarray:
        return self._masters

    @property
    def weights(self) -> list[float]:
        return self._weights

    @property
    def constant(self) -> float:
        return self._constant

    def __repr__(self) -> str:
        return f"LinearConstraint(n_slaves={len(self._slaves)}, n_weights={len(self._weights)})"
