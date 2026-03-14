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
from pyck.elements.euler_bernoulli_beam import EulerBernoulliBeam
from pyck.elements.timoshenko_beam_2p import TimoshenkoBeam2P

if TYPE_CHECKING:
    from pyck.elements.element import Element
    from pyck.geometry.curve_patch import CurvePatch


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
        The element formulation, used to scale DOFs if `num_dofs_per_node > 1`.
    """
    bnd = patch.boundary(_resolve_side(at))
    dofs = np.asarray(bnd.displacement_dofs, dtype=int)
    if element is not None:
        ndof = element._cpp_object.num_dofs_per_node()
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
    ndof = element._cpp_object.num_dofs_per_node()
    if isinstance(element, EulerBernoulliBeam):
        pairs = list(zip(bnd.displacement_dofs, bnd.rotation_dofs))
        return LinearConstraint(pairs)
    else:
        # For Timoshenko, rotation is an independent DOF at index 1 of the node
        dofs = np.asarray(bnd.displacement_dofs, dtype=int) * ndof + 1
        return DirectConstraint(dofs.tolist(), value)
