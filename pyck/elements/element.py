"""Abstract base for finite elements."""

from __future__ import annotations

from abc import ABC
from typing import Any, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pyck.geometry.patch import Patch
    from pyck.materials import Material


class Element(ABC):
    """Abstract base class for isogeometric finite element formulations.

    Subclasses must set the ``num_node_dofs`` class attribute and assign
    ``_material`` and ``_cpp_object`` in ``__init__``.
    """

    num_node_dofs: int
    _cpp_object: Any
    _material: Any

    @property
    def material(self) -> "Material":
        """The material model for this element."""
        return self._material

    def displacement_shape_matrix(
        self, patch: "Patch", params: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Global displacement shape matrix evaluated at parametric points.

        Shape: ``(num_disp_components * Q, num_global_dofs)``.
        """
        return np.asarray(
            self._cpp_object.displacement_shape_matrix(patch._cpp_object, params)
        )

    def rotation_shape_matrix(
        self, patch: "Patch", params: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Global rotation shape matrix evaluated at parametric points.

        Shape: ``(num_rot_components * Q, num_global_dofs)``.
        """
        return np.asarray(
            self._cpp_object.rotation_shape_matrix(patch._cpp_object, params)
        )

    def strain_matrix(
        self, patch: "Patch", params: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Global strain (B) matrix evaluated at parametric points.

        Shape: ``(num_strain_components * Q, num_global_dofs)``.
        """
        return np.asarray(
            self._cpp_object.strain_matrix(patch._cpp_object, params)
        )

    def stress_shape_matrix(
        self, patch: "Patch", params: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Global stress (D B) matrix evaluated at parametric points.

        Shape: ``(num_stress_components * Q, num_global_dofs)``.
        """
        return np.asarray(
            self._cpp_object.stress_shape_matrix(patch._cpp_object, params)
        )
