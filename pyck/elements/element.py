"""Abstract base for finite elements."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from pyck.materials import Material

@runtime_checkable
class Element(Protocol):
    """Base class for isogeometric finite element formulations."""

    @property
    def _cpp_object(self) -> Any:
        """The underlying C++ element object."""
        ...

    @property
    def material(self) -> "Material":
        """The material model for this element."""
        ...
