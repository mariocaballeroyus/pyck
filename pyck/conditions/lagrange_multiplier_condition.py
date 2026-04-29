"""Lagrange-multiplier boundary condition enforcement."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyck._pyck as _pyck

if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.elements.element import Element
    from pyck.geometry.boundary_patch import BoundaryPatch


class LagrangeMultiplierCondition:
    """Boundary trace constraints enforced with Lagrange multipliers.

    The condition appends additional multiplier DOFs to the assembled system,
    so the solved vector contains the physical DOFs first and the multiplier
    unknowns at the end.

    The element formulation is injected automatically when this condition is
    registered via ``problem.add_condition()``.

    Parameters
    ----------
    boundary : BoundaryPatch
        1-D boundary extracted from a 2-D surface patch.
    quadrature : QuadratureRule, optional
        1-D quadrature rule for the boundary integral. Defaults to the
        boundary patch's own quadrature rule.
    w_bar : float | None, optional
        Prescribed displacement. If ``None``, the displacement trace is not
        enforced. Default is ``0.0``.
    phi_n_bar : float | None, optional
        Prescribed normal rotation. If ``None``, it is not enforced.
    phi_s_bar : float | None, optional
        Prescribed tangential rotation. If ``None``, it is not enforced.
    """

    _cpp_object: _pyck.LagrangeMultiplierCondition2d | None

    def __init__(
        self,
        boundary: BoundaryPatch,
        quadrature: QuadratureRule | None = None,
        w_bar: float | None = 0.0,
        phi_n_bar: float | None = None,
        phi_s_bar: float | None = None,
    ) -> None:
        if not isinstance(boundary._cpp_object, _pyck.BoundaryPatch2d):
            raise TypeError(
                f"LagrangeMultiplierCondition requires a 2-D boundary patch, "
                f"got {type(boundary._cpp_object).__name__}."
            )
        if w_bar is None and phi_n_bar is None and phi_s_bar is None:
            raise ValueError(
                "LagrangeMultiplierCondition requires at least one enforced field."
            )

        self._boundary = boundary
        self._quadrature = quadrature
        self._w_bar = w_bar
        self._phi_n_bar = phi_n_bar
        self._phi_s_bar = phi_s_bar
        self._cpp_object = None

    def bind(self, _: QuadratureRule, element: Element) -> None:
        """Build the C++ object using the element from the parent problem."""
        rule = self._quadrature if self._quadrature is not None else self._boundary.quadrature
        enforce_w = self._w_bar is not None
        enforce_phi_n = self._phi_n_bar is not None
        enforce_phi_s = self._phi_s_bar is not None
        self._cpp_object = _pyck.LagrangeMultiplierCondition2d(
            self._boundary._cpp_object,
            element._cpp_object,
            rule._cpp_object,
            enforce_w,
            0.0 if self._w_bar is None else float(self._w_bar),
            enforce_phi_n,
            0.0 if self._phi_n_bar is None else float(self._phi_n_bar),
            enforce_phi_s,
            0.0 if self._phi_s_bar is None else float(self._phi_s_bar),
        )

    def __repr__(self) -> str:
        return "LagrangeMultiplierCondition()"


def create_displacement_lagrange(
    boundary: "BoundaryPatch",
    quadrature: "QuadratureRule | None" = None,
    w_bar: float = 0.0,
) -> LagrangeMultiplierCondition:
    """Create a displacement-only Lagrange multiplier condition."""
    return LagrangeMultiplierCondition(boundary, quadrature, w_bar=w_bar)


def create_simply_supported_lagrange(
    boundary: "BoundaryPatch",
    quadrature: "QuadratureRule | None" = None,
) -> LagrangeMultiplierCondition:
    """Create a simply-supported Lagrange multiplier condition (w = 0, φ_s = 0)."""
    return LagrangeMultiplierCondition(boundary, quadrature, w_bar=0.0, phi_s_bar=0.0)


def create_clamped_lagrange(
    boundary: "BoundaryPatch",
    quadrature: "QuadratureRule | None" = None,
) -> LagrangeMultiplierCondition:
    """Create a clamped Lagrange multiplier condition (w = 0, φ_n = 0, φ_s = 0)."""
    return LagrangeMultiplierCondition(
        boundary, quadrature, w_bar=0.0, phi_n_bar=0.0, phi_s_bar=0.0
    )
