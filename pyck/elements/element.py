"""Abstract base for finite elements."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class Element(Protocol):
    """Base class for isogeometric finite element formulations."""

    @property
    def _cpp_object(self) -> Any:
        """The underlying C++ element object."""
        ...
