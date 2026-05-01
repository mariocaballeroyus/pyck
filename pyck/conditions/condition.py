from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pyck.assembly.quadrature import QuadratureRule

if TYPE_CHECKING:
    from pyck.elements.element import Element


@runtime_checkable
class Condition(Protocol):
    """Protocol for conditions contributing to the global system.

    A condition may either expose a ready-to-use C++ object immediately
    (for example ``PenaltyBoundaryCondition``), or build it lazily during assembly
    (for example ``LoadCondition``).
    """

    @property
    def _cpp_object(self) -> Any | None:
        """The underlying C++ condition object, if already available."""
        ...


@runtime_checkable
class BindableCondition(Condition, Protocol):
    """Condition that can lazily build its underlying C++ object."""

    def bind(self, quadrature: QuadratureRule, element: Element) -> None:
        """Bind the condition to a quadrature rule and physical geometry.

        This is called during assembly (e.g., by `problem.add_condition()`)
        to finalize any lazy-initialized components like load values.
        """
        ...
