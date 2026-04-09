"""Penalty-based boundary condition enforcement."""

from typing import TYPE_CHECKING

import pyck._pyck as _pyck

if TYPE_CHECKING:
    from pyck.geometry.patch import Patch
    from pyck.geometry.boundary_patch import BoundaryPatch
    from pyck.elements.element import Element
    from pyck.assembly.quadrature import QuadratureRule


class PenaltyCondition:
    """Penalty-based boundary condition.

    Weakly enforces boundary conditions using a penalty method by adding
    penalty terms to the stiffness matrix and load vector:

        K_penalty = ∫_Γ α * B^T * B dΓ
        F_penalty = ∫_Γ α * B^T * prescribed_value dΓ

    where B is either the shape functions (displacement penalty) or their
    derivatives (derivative penalty).

    Parameters
    ----------
    boundary : BoundaryPatch
        The boundary patch where the penalty is applied.
    element : Element
        The element formulation (provides shape functions).
    quadrature : QuadratureRule
        Quadrature rule for boundary integration.
    penalty_type : PenaltyType
        Type of penalty (Displacement or Derivative).
    penalty_parameter : float
        Penalty parameter α (large positive value, e.g., 1e10-1e12).
        Larger values enforce the constraint more strongly but may
        worsen conditioning.
    derivative_order : int, optional
        Order of derivative to penalize (0=value, 1=gradient, 2=curvature).
        Default is 0.
    prescribed_value : float, optional
        Prescribed value to enforce. Default is 0.0.

    Examples
    --------
    >>> # Penalty for displacement w = 0 on boundary
    >>> boundary = surface.boundary(0, True)
    >>> quad = create_gauss_from_patch(boundary)
    >>> penalty = PenaltyCondition(
    ...     boundary, element, quad,
    ...     _pyck.PenaltyType.Displacement,
    ...     penalty_parameter=1e12
    ... )
    >>> problem.add_condition(penalty)

    >>> # Penalty for curvature ∂²w/∂n² = 0 on boundary
    >>> penalty = PenaltyCondition(
    ...     boundary, element, quad,
    ...     _pyck.PenaltyType.Derivative,
    ...     penalty_parameter=1e10,
    ...     derivative_order=2
    ... )
    >>> problem.add_condition(penalty)
    """

    def __init__(
        self,
        boundary: "BoundaryPatch",
        element: "Element",
        quadrature: "QuadratureRule",
        penalty_type: _pyck.PenaltyType,
        penalty_parameter: float,
        derivative_order: int = 0,
        prescribed_value: float = 0.0,
    ):
        # Determine the dimension from the boundary patch
        boundary_cpp = boundary._cpp_object
        element_cpp = element._cpp_object
        quadrature_cpp = quadrature._cpp_object

        # Create the C++ penalty condition (only 2D supported currently)
        if isinstance(boundary_cpp, _pyck.BoundaryPatch2d):
            self._cpp_object = _pyck.PenaltyCondition2d(
                boundary_cpp,
                element_cpp,
                quadrature_cpp,
                penalty_type,
                penalty_parameter,
                derivative_order,
                prescribed_value,
            )
        else:
            raise TypeError(
                f"Unsupported boundary patch type: {type(boundary_cpp)}. "
                f"Only 2D surfaces are currently supported for penalty conditions."
            )

    def bind(self, quadrature: "QuadratureRule", element: "Element") -> None:
        """Bind the condition to a quadrature rule and element.

        For PenaltyCondition, this is a no-op since the penalty terms are
        computed during construction.
        """
        pass

    def __repr__(self) -> str:
        return f"PenaltyCondition({self._cpp_object})"


def create_displacement_penalty(
    boundary: "BoundaryPatch",
    element: "Element",
    quadrature: "QuadratureRule",
    alpha: float = 1e12,
) -> PenaltyCondition:
    """Create a penalty condition for displacement constraint (w = 0).

    Parameters
    ----------
    boundary : BoundaryPatch
        The boundary where the penalty is applied.
    element : Element
        The element formulation.
    quadrature : QuadratureRule
        Quadrature rule for boundary integration.
    alpha : float, optional
        Penalty parameter (default: 1e12).

    Returns
    -------
    PenaltyCondition
        Penalty condition enforcing displacement = 0.

    Examples
    --------
    >>> boundary = surface.boundary(0, True)
    >>> quad = create_gauss_from_patch(boundary)
    >>> penalty = create_displacement_penalty(boundary, element, quad)
    >>> problem.add_condition(penalty)
    """
    return PenaltyCondition(
        boundary,
        element,
        quadrature,
        _pyck.PenaltyType.Displacement,
        alpha,
        derivative_order=0,
        prescribed_value=0.0,
    )


def create_curvature_penalty(
    boundary: "BoundaryPatch",
    element: "Element",
    quadrature: "QuadratureRule",
    alpha: float = 1e10,
) -> PenaltyCondition:
    """Create a penalty condition for curvature constraint (∂²w/∂n² = 0).

    Parameters
    ----------
    boundary : BoundaryPatch
        The boundary where the penalty is applied.
    element : Element
        The element formulation.
    quadrature : QuadratureRule
        Quadrature rule for boundary integration.
    alpha : float, optional
        Penalty parameter (default: 1e10).

    Returns
    -------
    PenaltyCondition
        Penalty condition enforcing curvature = 0.

    Examples
    --------
    >>> boundary = surface.boundary(0, True)
    >>> quad = create_gauss_from_patch(boundary)
    >>> penalty = create_curvature_penalty(boundary, element, quad)
    >>> problem.add_condition(penalty)
    """
    return PenaltyCondition(
        boundary,
        element,
        quadrature,
        _pyck.PenaltyType.Derivative,
        alpha,
        derivative_order=2,
        prescribed_value=0.0,
    )
