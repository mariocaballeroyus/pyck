"""Penalty-based boundary condition enforcement."""

from typing import TYPE_CHECKING

import pyck._pyck as _pyck

if TYPE_CHECKING:
    from pyck.geometry.boundary_patch import BoundaryPatch
    from pyck.elements.element import Element
    from pyck.assembly.quadrature import QuadratureRule


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

    Parameters
    ----------
    boundary : BoundaryPatch
        1-D boundary extracted from a 2-D surface patch.
    element : Element
        2-D element formulation (determines the DOF layout).
    quadrature : QuadratureRule
        1-D Gauss–Legendre rule for the boundary integral.
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

    def __init__(
        self,
        boundary: "BoundaryPatch",
        element: "Element",
        quadrature: "QuadratureRule",
        alpha_w: float,
        w_bar: float = 0.0,
        alpha_phi_n: float = 0.0,
        phi_n_bar: float = 0.0,
        alpha_phi_s: float = 0.0,
        phi_s_bar: float = 0.0,
    ):
        boundary_cpp = boundary._cpp_object
        element_cpp = element._cpp_object
        quadrature_cpp = quadrature._cpp_object

        if not isinstance(boundary_cpp, _pyck.BoundaryPatch2d):
            raise TypeError(
                f"PenaltyCondition requires a 2-D boundary patch, "
                f"got {type(boundary_cpp).__name__}."
            )

        self._cpp_object = _pyck.PenaltyCondition2d(
            boundary_cpp,
            element_cpp,
            quadrature_cpp,
            float(alpha_w),
            float(w_bar),
            float(alpha_phi_n),
            float(phi_n_bar),
            float(alpha_phi_s),
            float(phi_s_bar),
        )

    def __repr__(self) -> str:
        return f"PenaltyCondition()"


def create_displacement_penalty(
    boundary: "BoundaryPatch",
    element: "Element",
    quadrature: "QuadratureRule",
    alpha: float = 1e12,
    w_bar: float = 0.0,
) -> PenaltyCondition:
    """Create a displacement penalty condition  (w = w̄).

    Parameters
    ----------
    boundary : BoundaryPatch
        1-D boundary extracted from a 2-D surface patch.
    element : Element
        2-D element formulation.
    quadrature : QuadratureRule
        1-D quadrature rule for the boundary integral.
    alpha : float
        Penalty factor (default 1e12).
    w_bar : float
        Prescribed displacement (default 0.0).

    Returns
    -------
    PenaltyCondition
    """
    return PenaltyCondition(boundary, element, quadrature, alpha_w=alpha, w_bar=w_bar)


def create_simply_supported_penalty(
    boundary: "BoundaryPatch",
    element: "Element",
    quadrature: "QuadratureRule",
    alpha: float = 1e12,
) -> PenaltyCondition:
    """Create a simply-supported penalty condition (w = 0, θ_s = 0).

    Enforces the standard simply-supported Kirchhoff–Love / Reissner–Mindlin
    boundary conditions: zero transverse displacement and zero tangential
    rotation.  For single-DOF elements (KL-1p, RM-1p) only the displacement
    penalty is assembled.

    Parameters
    ----------
    boundary : BoundaryPatch
        1-D boundary extracted from a 2-D surface patch.
    element : Element
        2-D element formulation.
    quadrature : QuadratureRule
        1-D quadrature rule for the boundary integral.
    alpha : float
        Penalty factor for both displacement and rotation (default 1e12).

    Returns
    -------
    PenaltyCondition
    """
    return PenaltyCondition(
        boundary,
        element,
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
    element: "Element",
    quadrature: "QuadratureRule",
    alpha: float = 1e12,
) -> PenaltyCondition:
    """Create a clamped penalty condition (w = 0, θ_n = 0, θ_s = 0).

    Parameters
    ----------
    boundary : BoundaryPatch
        1-D boundary extracted from a 2-D surface patch.
    element : Element
        2-D element formulation.
    quadrature : QuadratureRule
        1-D quadrature rule for the boundary integral.
    alpha : float
        Penalty factor for displacement and all rotations (default 1e12).

    Returns
    -------
    PenaltyCondition
    """
    return PenaltyCondition(
        boundary,
        element,
        quadrature,
        alpha_w=alpha,
        w_bar=0.0,
        alpha_phi_n=alpha,
        phi_n_bar=0.0,
        alpha_phi_s=alpha,
        phi_s_bar=0.0,
    )
