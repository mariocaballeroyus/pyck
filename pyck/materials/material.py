"""Abstract base for material models."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Material(Protocol):
    """Base protocol for material models."""

    @property
    def _cpp_object(self) -> Any:
        """The underlying C++ material object."""
        ...
