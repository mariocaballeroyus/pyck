"""Multi-point constraint class."""

from __future__ import annotations

from typing import Sequence

import pyck._pyck as _pyck
from pyck.constraints.constraint import Constraint


class MultipointConstraint(Constraint):
    """Multi-point constraint enforcing an algebraic relation.

    Enforces: u_slave = sum(alpha_i * u_m_i) + c

    Parameters
    ----------
    slave : int
        The dependent DOF index.
    masters_weights : sequence of (int, float)
        List of ``(master_dof, coefficient)`` pairs.
    constant : float, optional
        The constant offset `c`, defaults to 0.0.
    """

    def __init__(self, slave: int, masters_weights: Sequence[tuple[int, float]], constant: float = 0.0) -> None:
        self._slave = int(slave)
        self._masters_weights = [(int(m), float(w)) for m, w in masters_weights]
        self._constant = float(constant)
        self._cpp_object = _pyck.MultipointConstraint(self._slave, self._masters_weights, self._constant)

    @property
    def slave(self) -> int:
        """The dependent DOF index."""
        return self._slave

    @property
    def masters_weights(self) -> list[tuple[int, float]]:
        """List of (master_dof, coefficient) pairs."""
        return self._masters_weights

    @property
    def constant(self) -> float:
        """The constant offset."""
        return self._constant

    def __repr__(self) -> str:
        return f"MultipointConstraint(slave={self._slave}, n_masters={len(self._masters_weights)}, c={self._constant})"
