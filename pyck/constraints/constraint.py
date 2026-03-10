"""Python wrapper classes for boundary constraints."""

from __future__ import annotations

from typing import Sequence

import pyck._pyck as _pyck


class Constraint:
    """Base class for boundary constraints.

    Subclasses wrap concrete C++ constraint types and expose a
    `_cpp_object` attribute used by the assembler.
    """

    _cpp_object: _pyck.Constraint

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class MasterSlaveConstraint(Constraint):
    """Constraint coupling master and slave DOFs.

    Wraps :class:`pyck::MasterSlaveConstraint<double>`.

    Parameters
    ----------
    pairs : sequence of (int, int)
        `(master, slave)` DOF index pairs.
    """

    def __init__(self, pairs: Sequence[tuple[int, int]]) -> None:
        self._pairs = [(int(m), int(s)) for m, s in pairs]
        self._cpp_object = _pyck.MasterSlaveConstraint(self._pairs)

    @property
    def pairs(self) -> list[tuple[int, int]]:
        """Master-slave DOF pairs."""
        return self._pairs

    def __repr__(self) -> str:
        return f"MasterSlaveConstraint(n_pairs={len(self._pairs)})"
