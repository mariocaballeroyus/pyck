"""Shell element formulations."""

from __future__ import annotations

import pyck._pyck as _pyck
from pyck.elements.element import Element
from pyck.materials import Material, PlaneStress2d


class ShellReissnerMindlin5p(Element):
    """Standard Reissner-Mindlin shell element.

    Parameters
    ----------
    material : Material
        Material model for the shell.
    """
    num_node_dofs: int = 5

    def __init__(self, material: Material) -> None:
        if not isinstance(material, PlaneStress2d):
            raise TypeError(
                f"ShellReissnerMindlin5p requires PlaneStress2d, "
                f"got {type(material).__name__}."
            )
        self._material = material
        self._cpp_object = _pyck.ShellReissnerMindlin5p(material._cpp_object)
