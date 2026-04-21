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

    Parameters
    ----------
    boundary : BoundaryPatch
        1-D boundary extracted from a 2-D surface patch.
    element : Element
        2-D element formulation defining the trace operators.
    quadrature : QuadratureRule
        1-D quadrature rule for the boundary integral.
    w_bar : float | None, optional
        Prescribed displacement. If ``None``, the displacement trace is not
        enforced. Default is ``0.0``.
    phi_n_bar : float | None, optional
        Prescribed normal rotation. If ``None``, it is not enforced.
    phi_s_bar : float | None, optional
        Prescribed tangential rotation. If ``None``, it is not enforced.
    """

    def __init__(
        self,
        boundary: "BoundaryPatch",
        element: "Element",
        quadrature: "QuadratureRule",
        w_bar: float | None = 0.0,
        phi_n_bar: float | None = None,
        phi_s_bar: float | None = None,
    ) -> None:
        boundary_cpp = boundary._cpp_object
        element_cpp = element._cpp_object
        quadrature_cpp = quadrature._cpp_object

        if not isinstance(boundary_cpp, _pyck.BoundaryPatch2d):
            raise TypeError(
                f"LagrangeMultiplierCondition requires a 2-D boundary patch, "
                f"got {type(boundary_cpp).__name__}."
            )

        enforce_w = w_bar is not None
        enforce_phi_n = phi_n_bar is not None
        enforce_phi_s = phi_s_bar is not None
        if not (enforce_w or enforce_phi_n or enforce_phi_s):
            raise ValueError(
                "LagrangeMultiplierCondition requires at least one enforced field."
            )

        self._cpp_object = _pyck.LagrangeMultiplierCondition2d(
            boundary_cpp,
            element_cpp,
            quadrature_cpp,
            enforce_w,
            0.0 if w_bar is None else float(w_bar),
            enforce_phi_n,
            0.0 if phi_n_bar is None else float(phi_n_bar),
            enforce_phi_s,
            0.0 if phi_s_bar is None else float(phi_s_bar),
        )

    def __repr__(self) -> str:
        return "LagrangeMultiplierCondition()"


def create_displacement_lagrange(
    boundary: "BoundaryPatch",
    element: "Element",
    quadrature: "QuadratureRule",
    w_bar: float = 0.0,
) -> LagrangeMultiplierCondition:
    """Create a displacement-only Lagrange multiplier condition."""
    return LagrangeMultiplierCondition(
        boundary,
        element,
        quadrature,
        w_bar=w_bar,
    )


def create_simply_supported_lagrange(
    boundary: "BoundaryPatch",
    element: "Element",
    quadrature: "QuadratureRule",
) -> LagrangeMultiplierCondition:
    """Create a simply-supported Lagrange multiplier condition."""
    return LagrangeMultiplierCondition(
        boundary,
        element,
        quadrature,
        w_bar=0.0,
        phi_s_bar=0.0,
    )


def create_clamped_lagrange(
    boundary: "BoundaryPatch",
    element: "Element",
    quadrature: "QuadratureRule",
) -> LagrangeMultiplierCondition:
    """Create a clamped Lagrange multiplier condition."""
    return LagrangeMultiplierCondition(
        boundary,
        element,
        quadrature,
        w_bar=0.0,
        phi_n_bar=0.0,
        phi_s_bar=0.0,
    )
