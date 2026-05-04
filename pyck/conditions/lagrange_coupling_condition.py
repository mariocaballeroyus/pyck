"""Lagrange-multiplier coupling between two patches at a shared boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyck._pyck as _pyck

if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.elements.element import Element
    from pyck.geometry.patch_boundary import PatchBoundary


class LagrangeCouplingCondition:
    """Lagrange-multiplier coupling on a shared interface between two patches.

    Construct with Python objects (``PatchBoundary``, ``Element``,
    ``QuadratureRule``); register coupled fields with :meth:`add`; attach
    to a problem with :meth:`LinearElasticProblem.add_coupling`.

    Each registered field-pair adds an independent multiplier block (one
    multiplier per side-A boundary control point) and contributes the
    symmetric saddle-point block enforcing
    ``∫_Γ μ · (C_A − sign_b · C_B − value) dΓ = 0``  ∀μ
    in the discrete multiplier space (no penalty parameter required).

    Parameters
    ----------
    side_a, side_b : PatchBoundary
        The two boundaries sharing the interface.
    patch_a_idx, patch_b_idx : int
        Indices of patches A and B inside the parent
        :class:`LinearElasticProblem`.
    element : Element
        Element formulation (assumed shared by both patches).
    quadrature : QuadratureRule
        1-D quadrature rule along the interface curve.
    reverse : bool, optional
        Whether ``side_b``'s parametrisation is reversed relative to
        ``side_a`` (default ``False``).
    """

    _cpp_object: _pyck.LagrangeCouplingCondition2d

    def __init__(
        self,
        side_a: "PatchBoundary",
        side_b: "PatchBoundary",
        *,
        patch_a_idx: int,
        patch_b_idx: int,
        element: "Element",
        quadrature: "QuadratureRule",
        reverse: bool = False,
    ) -> None:
        self._side_a = side_a
        self._side_b = side_b
        self._patch_a_idx = int(patch_a_idx)
        self._patch_b_idx = int(patch_b_idx)
        self._element = element
        self._quadrature = quadrature
        self._reverse = bool(reverse)
        self._cpp_object = _pyck.LagrangeCouplingCondition2d(
            side_a._cpp_object,
            side_b._cpp_object,
            patch_a_idx=self._patch_a_idx,
            patch_b_idx=self._patch_b_idx,
            element_a=element._cpp_object,
            element_b=element._cpp_object,
            quadrature=quadrature._cpp_object,
            reverse=self._reverse,
        )

    def add(
        self,
        field_a: _pyck.BoundaryField,
        field_b: _pyck.BoundaryField,
        sign_b: float = 1.0,
        value: float = 0.0,
    ) -> "LagrangeCouplingCondition":
        """Add a coupled field-pair to the interface.

        Parameters
        ----------
        field_a, field_b : BoundaryField
            Field projections on each side (e.g. :class:`BasisValue`,
            :class:`BasisNormalSlope`, :class:`BasisNormalCurvature`).
        sign_b : float, optional
            Sign of the side-B contribution. Use ``(−1)^k`` for the
            ``k``-th-order normal-derivative field with opposing outward
            normals; default ``+1.0``.
        value : float, optional
            Prescribed jump (default ``0.0``).
        """
        self._cpp_object.add(
            field_a, field_b,
            sign_b=float(sign_b),
            value=float(value),
        )
        return self

    @property
    def patch_a_idx(self) -> int:
        return self._patch_a_idx

    @property
    def patch_b_idx(self) -> int:
        return self._patch_b_idx

    def __repr__(self) -> str:
        return (
            f"LagrangeCouplingCondition("
            f"patch_a_idx={self._patch_a_idx}, "
            f"patch_b_idx={self._patch_b_idx}, "
            f"reverse={self._reverse})"
        )
