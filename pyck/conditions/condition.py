"""Abstract base for assembly conditions (loads, BCs, …)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pyck.assembly.quadrature import QuadratureRule


@runtime_checkable
class Condition(Protocol):
    """Base class for conditions constributing to the global system."""

    def bind(self, quadrature: QuadratureRule, element=None) -> None:
        """Bind the condition to a quadrature rule and physical geometry.

        This is called during assembly (e.g., by `problem.add_condition()`)
        to finalize any lazy-initialized components like load values.
        """
        pass
