"""Penalty-based boundary condition enforcement."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyck._pyck as _pyck

if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.geometry.boundary_patch import BoundaryPatch
    from pyck.elements.element import Element


class PenaltyCondition:
    """Penalty method for weakly enforcing Dirichlet boundary conditions.

    Adds penalty contributions to the global stiffness matrix and load vector:

        K_pen += α_w  ∫_Γ N_w^T  N_w  dΓ
        K_pen += α_φn ∫_Γ N_φn^T N_φn dΓ   (RM-3p only)
        K_pen += α_φs ∫_Γ N_φs^T N_φs dΓ   (RM-3p only)

        f_pen += α_w  ∫_Γ N_w^T  w̄    dΓ
        f_pen += α_φn ∫_Γ N_φn^T φ̄_n  dΓ   (RM-3p only)
        f_pen += α_φs ∫_Γ N_φs^T φ̄_s  dΓ   (RM-3p only)

    For single-DOF elements (KL-1p, RM-1p) only the displacement penalty is
    assembled; the rotation terms are silently ignored.

    The element formulation is injected automatically when this condition is
    registered via ``problem.add_condition()``.

    Parameters
    ----------
    boundary : BoundaryPatch
        1-D boundary extracted from a 2-D surface patch.
    quadrature : QuadratureRule, optional
        1-D Gauss–Legendre rule for the boundary integral. Defaults to the
        boundary patch's own quadrature rule.
    alpha_w : float
        Penalty factor for the transverse displacement constraint.
    w_bar : float, optional
        Prescribed displacement value (default 0.0).
    alpha_phi_n : float, optional
        Penalty factor for the normal rotation (RM-3p only, default 0.0).
    phi_n_bar : float, optional
        Prescribed normal rotation (default 0.0).
    alpha_phi_s : float, optional
        Penalty factor for the tangential rotation (RM-3p only, default 0.0).
    phi_s_bar : float, optional
        Prescribed tangential rotation (default 0.0).
    """

    _cpp_object: _pyck.PenaltyCondition2d | None

    def __init__(
        self,
        boundary: "BoundaryPatch",
        quadrature: "QuadratureRule | None" = None,
        alpha_w: float = 0.0,
        w_bar: float = 0.0,
        alpha_phi_n: float = 0.0,
        phi_n_bar: float = 0.0,
        alpha_phi_s: float = 0.0,
        phi_s_bar: float = 0.0,
    ):
        if not isinstance(boundary._cpp_object, _pyck.BoundaryPatch2d):
            raise TypeError(
                f"PenaltyCondition requires a 2-D boundary patch, "
                f"got {type(boundary._cpp_object).__name__}."
            )

        self._boundary = boundary
        self._quadrature = quadrature
        self._alpha_w = float(alpha_w)
        self._w_bar = float(w_bar)
        self._alpha_phi_n = float(alpha_phi_n)
        self._phi_n_bar = float(phi_n_bar)
        self._alpha_phi_s = float(alpha_phi_s)
        self._phi_s_bar = float(phi_s_bar)
        self._cpp_object = None

    def bind(self, _: "QuadratureRule", element: "Element") -> None:
        """Build the C++ object using the element from the parent problem."""
        rule = self._quadrature if self._quadrature is not None else self._boundary.quadrature
        self._cpp_object = _pyck.PenaltyCondition2d(
            self._boundary._cpp_object,
            element._cpp_object,
            rule._cpp_object,
            self._alpha_w,
            self._w_bar,
            self._alpha_phi_n,
            self._phi_n_bar,
            self._alpha_phi_s,
            self._phi_s_bar,
        )

    def __repr__(self) -> str:
        return "PenaltyCondition()"


def create_displacement_penalty(
    boundary: "BoundaryPatch",
    quadrature: "QuadratureRule | None" = None,
    alpha: float = 1e12,
    w_bar: float = 0.0,
) -> PenaltyCondition:
    """Create a displacement penalty condition  (w = w̄)."""
    return PenaltyCondition(boundary, quadrature, alpha_w=alpha, w_bar=w_bar)


def create_simply_supported_penalty(
    boundary: "BoundaryPatch",
    quadrature: "QuadratureRule | None" = None,
    alpha: float = 1e12,
) -> PenaltyCondition:
    """Create a simply-supported penalty condition (w = 0, θ_s = 0)."""
    return PenaltyCondition(
        boundary,
        quadrature,
        alpha_w=alpha,
        w_bar=0.0,
        alpha_phi_n=0.0,
        phi_n_bar=0.0,
        alpha_phi_s=alpha,
        phi_s_bar=0.0,
    )


def create_clamped_penalty(
    boundary: "BoundaryPatch",
    quadrature: "QuadratureRule | None" = None,
    alpha: float = 1e12,
) -> PenaltyCondition:
    """Create a clamped penalty condition (w = 0, θ_n = 0, θ_s = 0)."""
    return PenaltyCondition(
        boundary,
        quadrature,
        alpha_w=alpha,
        w_bar=0.0,
        alpha_phi_n=alpha,
        phi_n_bar=0.0,
        alpha_phi_s=alpha,
        phi_s_bar=0.0,
    )
