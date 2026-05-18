"""Gauss-Legendre quadrature rules for different topological dimensions."""

from __future__ import annotations

from typing import Any

import pyck._pyck as _pyck
from pyck.geometry.patch import Patch
from pyck.quadrature.quadrature import QuadratureRule


class GaussLegendre(QuadratureRule):
    """Gauss-Legendre tensor-product quadrature rule.

    Parameters
    ----------
    num_points : int
        Number of integration points per parametric direction.
    dim : int, optional
        Topological dimension (default: 1).
    """

    _cpp_object: _pyck.GaussLegendre1d | _pyck.GaussLegendre2d

    def __init__(self, num_points: int, dim: int = 1) -> None:
        super().__init__(dim=dim)

        if dim == 1:
            self._cpp_object = _pyck.GaussLegendre1d(int(num_points))
        elif dim == 2:
            self._cpp_object = _pyck.GaussLegendre2d(int(num_points))
        else:
            raise ValueError(f"Unsupported dimension: {dim}")

    def __repr__(self) -> str:
        return f"GaussLegendre(num_points={len(self)}, dim={self.dim})"

    # === Class Methods ===============================================================

    @classmethod
    def from_patch(cls, patch: Patch) -> GaussLegendre:
        max_degree = max(b.degree for b in patch.basis)
        return cls(max_degree + 1, dim=patch.tdim)
