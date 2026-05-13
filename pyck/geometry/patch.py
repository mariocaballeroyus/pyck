"""Abstract base class for parametric patches.

Patches are the building blocks of isogeometric geometry. All patches in pyck
are embedded in 3D physical space; this dimension is fixed and cannot be reduced.
"""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np

import pyck._pyck as _pyck

if typing.TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.geometry.patch_boundary import PatchBoundary


class Patch(ABC):
    """Abstract base class for tensor-product parametric patches."""

    _cpp_object: _pyck.Patch1d | _pyck.Patch2d
    _name: str

    @property
    def name(self) -> str:
        """Patch label."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def gdim(self) -> int:
        """Geometric (physical) dimension."""
        return 3

    @property
    @abstractmethod
    def tdim(self) -> int:
        """Topological (parametric) dimension."""

    @cached_property
    def control_points(self) -> np.ndarray:
        """Read-only view of the control-point matrix, shape (n, 3)."""
        pts = np.asarray(self._cpp_object.control_pts())
        pts.flags.writeable = False
        return pts

    @property
    def num_control_pts(self) -> int:
        """Total number of control points."""
        return self._cpp_object.num_control_pts()

    def dof_mapper(self) -> _pyck.DofMapper1d | _pyck.DofMapper2d:
        """Return the DOF mapper for this patch."""
        return self._cpp_object.dof_mapper()

    def layer_dofs(self, param_dim: int, at_start: bool, layer_idx: int = 0) -> list[int]:
        """Return the DOF indices for a given boundary layer."""
        return self._cpp_object.layer_dofs(param_dim, at_start, layer_idx)

    @abstractmethod
    def boundary(self, side: str) -> PatchBoundary:
        """Extract a boundary patch from this patch."""

    @abstractmethod
    def basis(self, direction: int) -> typing.Any:
        """Return the basis in the given direction."""

    @cached_property
    def quadrature(self) -> QuadratureRule:
        """Default tensor-product Gauss-Legendre rule for this patch.

        Number of points per direction is ``max_degree + 1`` (sufficient to
        integrate the mass matrix exactly). Used as the fallback rule by
        :class:`LinearElasticProblem` and the condition factories when no
        explicit quadrature is provided.
        """
        from pyck.assembly.gauss import GaussLegendre

        # `basis` is a method on SurfacePatch and a property on CurvePatch;
        # handle both transparently.
        basis_attr = getattr(type(self), "basis", None)
        is_property = isinstance(basis_attr, property)
        max_deg = 0
        for i in range(self.tdim):
            b = self.basis if is_property else self.basis(i)
            max_deg = max(max_deg, b.degree)
        return GaussLegendre(max_deg + 1, dim=self.tdim)

    @abstractmethod
    def eval_shape_functions(
        self, params: np.ndarray, span: int, order: int = 0
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Evaluate shape functions and their derivatives on a given span.
        
        Returns
        -------
        tuple[list[np.ndarray], np.ndarray]
            (Shape function derivatives, Jacobian determinants)
        """

