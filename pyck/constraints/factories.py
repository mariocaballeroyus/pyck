"""Python factory functions for boundary constraints.

These functions inspect the element formulation and return the appropriate
low-level :class:`Constraint` object (:class:`DirectConstraint` or
:class:`LinearConstraint`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyck.constraints.direct_constraint import DirectConstraint
from pyck.constraints.constraint import LinearConstraint
from pyck.elements.beams import BeamEulerBernoulli1p, BeamTimoshenko1p, BeamTimoshenko2p

if TYPE_CHECKING:
    from pyck.elements.element import Element
    from pyck.geometry.curve import CurvePatch
    from pyck.geometry.surface import SurfacePatch


def _resolve_side(at: str) -> str:
    side = {"left": "start", "right": "end"}.get(at, at)
    if side not in ("start", "end"):
        raise ValueError(f"'at' must be 'left'/'start' or 'right'/'end', got {at!r}")
    return side


def create_direct_displacement_constraint(
    patch: CurvePatch,
    *,
    at: str,
    value: float = 0.0,
    element: Element | None = None,
) -> DirectConstraint:
    """Create a constraint prescribing displacement at a patch boundary.

    Always returns a :class:`DirectConstraint` regardless of element
    formulation — displacement DOFs are treated identically for all beam
    theories.

    Parameters
    ----------
    patch : CurvePatch
        The geometry patch.
    at : `"left"` / `"start"` or `"right"` / `"end"`
        Which boundary.
    value : float
        Prescribed displacement (default 0).
    element : Element | None
        The element formulation, used to scale DOFs if `num_node_dofs > 1`.
    """
    bnd = patch.boundary(_resolve_side(at))
    dofs = np.asarray(bnd.displacement_dofs, dtype=int)
    if element is not None:
        ndof = element._cpp_object.num_node_dofs()
        dofs = dofs * ndof
    return DirectConstraint(dofs.tolist(), value)


def create_direct_rotation_constraint(
    patch: CurvePatch,
    *,
    element: Element,
    at: str,
    value: float = 0.0,
):
    """Create a constraint prescribing rotation at a patch boundary.

    The concrete constraint type depends on the element formulation:

    * **Euler–Bernoulli** → :class:`LinearConstraint` coupling the
      boundary and adjacent control-point DOFs.
    * **Timoshenko** (or any other) → :class:`DirectConstraint`
      directly prescribing the independent rotation DOFs.

    Parameters
    ----------
    patch : CurvePatch
        The geometry patch.
    element : Element
        The element formulation (used to choose the constraint type).
    at : `"left"` / `"start"` or `"right"` / `"end"`
        Which boundary.
    value : float
        Prescribed rotation (default 0).
    """
    bnd = patch.boundary(_resolve_side(at))
    ndof = element._cpp_object.num_node_dofs()
    if isinstance(element, (BeamEulerBernoulli1p, BeamTimoshenko1p)):
        pairs = list(zip(bnd.displacement_dofs, bnd.rotation_dofs))
        return LinearConstraint(pairs)
    else:
        # For Timoshenko, rotation is an independent DOF at index 1 of the node
        dofs = np.asarray(bnd.displacement_dofs, dtype=int) * ndof + 1
        return DirectConstraint(dofs.tolist(), value)


# ======================================================================
# Surface-patch constraint factories (Kirchhoff-Love plate, etc.)
# ======================================================================


def create_plate_displacement_constraint(
    patch: SurfacePatch,
    *,
    at: str,
    value: float = 0.0,
    element: Element | None = None,
) -> DirectConstraint:
    """Constrain displacement DOFs on a surface patch boundary edge.

    Parameters
    ----------
    patch : SurfacePatch
        The surface geometry patch.
    at : ``"u0"``, ``"u1"``, ``"v0"``, ``"v1"``
        Which boundary edge.
    value : float
        Prescribed displacement (default 0).
    element : Element | None
        The element formulation (to scale DOFs if ndof > 1).
    """
    bnd = patch.boundary(at)
    dofs = np.asarray(bnd.displacement_dofs, dtype=int)
    if element is not None:
        ndof = element._cpp_object.num_node_dofs()
        dofs = dofs * ndof
    return DirectConstraint(dofs.tolist(), value)


def create_plate_clamped_constraint(
    patch: SurfacePatch,
    *,
    at: str,
    value: float = 0.0,
    element: Element | None = None,
) -> DirectConstraint:
    """Constrain both displacement and rotation DOFs (clamped BC).

    Parameters
    ----------
    patch : SurfacePatch
        The surface geometry patch.
    at : ``"u0"``, ``"u1"``, ``"v0"``, ``"v1"``
        Which boundary edge.
    value : float
        Prescribed displacement/rotation (default 0).
    element : Element | None
        The element formulation (to scale DOFs if ndof > 1).
    """
    bnd = patch.boundary(at)
    nodal_indices = np.asarray(bnd.boundary_dofs, dtype=int)
    if element is not None:
        ndof = element._cpp_object.num_node_dofs()
        if ndof == 1:
            # Kirchhoff–Love: boundary_dofs includes both the outermost layer
            # and the adjacent layer — both are needed to enforce C1 (slope = 0).
            return DirectConstraint(nodal_indices.tolist(), value)
        else:
            # Reissner–Mindlin (ndof=3): constrain all DOFs [w, θ_x, θ_y]
            # only at the outermost boundary nodes (displacement_dofs).
            disp_nodes = np.asarray(bnd.displacement_dofs, dtype=int)
            all_dofs = [int(i * ndof + d) for i in disp_nodes for d in range(ndof)]
            return DirectConstraint(all_dofs, value)
    return DirectConstraint(nodal_indices.tolist(), value)
