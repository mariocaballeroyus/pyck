"""Python factory functions for boundary constraints.

These functions inspect the element formulation and return the appropriate
low-level :class:`Constraint` object (:class:`DirichletBC` or
:class:`MasterSlaveConstraint`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyck.constraints.dirichlet_bc import DirichletBC
from pyck.constraints.constraint import MasterSlaveConstraint
from pyck.elements.euler_bernoulli_beam import EulerBernoulliBeam

if TYPE_CHECKING:
    from pyck.elements.element import Element
    from pyck.geometry.curve_patch import CurvePatch


def _resolve_side(at: str) -> str:
    side = {"left": "start", "right": "end"}.get(at, at)
    if side not in ("start", "end"):
        raise ValueError(f"'at' must be 'left'/'start' or 'right'/'end', got {at!r}")
    return side


def create_displacement_constraint(
    patch: CurvePatch,
    *,
    at: str,
    value: float = 0.0,
) -> DirichletBC:
    """Create a constraint prescribing displacement at a patch boundary.

    Always returns a :class:`DirichletBC` regardless of element
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
    """
    bnd = patch.boundary(_resolve_side(at))
    return DirichletBC(bnd.displacement_dofs, value)


def create_rotation_constraint(
    patch: CurvePatch,
    *,
    element: Element,
    at: str,
    value: float = 0.0,
):
    """Create a constraint prescribing rotation at a patch boundary.

    The concrete constraint type depends on the element formulation:

    * **Euler–Bernoulli** → :class:`MasterSlaveConstraint` coupling the
      boundary and adjacent control-point DOFs.
    * **Timoshenko** (or any other) → :class:`DirichletBC`
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
    if isinstance(element, EulerBernoulliBeam):
        pairs = list(zip(bnd.displacement_dofs, bnd.rotation_dofs))
        return MasterSlaveConstraint(pairs)
    else:
        return DirichletBC(bnd.rotation_dofs, value)
