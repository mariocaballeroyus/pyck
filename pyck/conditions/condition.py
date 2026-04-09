from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pyck.assembly.quadrature import QuadratureRule

if TYPE_CHECKING:
    from pyck.elements.element import Element


@runtime_checkable
class Condition(Protocol):
    """Base class for conditions contributing to the global system."""

    def bind(self, quadrature: QuadratureRule, element: Element) -> None:
        """Bind the condition to a quadrature rule and physical geometry.

        This is called during assembly (e.g., by `problem.add_condition()`)
        to finalize any lazy-initialized components like load values.
        """
        pass
