"""Penalty-based multipatch coupling condition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyck._pyck as _pyck
from pyck.conditions.condition import Field

if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.elements.element import Element
    from pyck.geometry.patch_boundary import PatchBoundary


class PenaltyCouplingCondition:
    """Penalty coupling of displacement continuity across a shared interface.

    Ties two patch boundaries that meet at the same physical curve. The two
    sides may be **non-conforming** — different knot vectors, element counts and
    degree — as long as the seam is straight (or, more generally, affinely
    parametrised, the case produced by Greville-placed control nets such as
    :meth:`SurfacePatch.rectangle`). The constructor builds the common refinement
    of both sides' breakpoints into integration segments, so the coupling
    integrates exactly. A curved or mismatched-parametrisation seam (where the
    affine map is invalid) is detected and rejected pending a point-inversion
    driver; a conforming interface is just the degenerate case.

    Couples the three displacement components (``U_X/U_Y/U_Z``) and the normal
    (bending) rotation ``ROT_N``. Displacement keeps the surface connected;
    ``ROT_N`` keeps it kink-free (G1), so a bending plate transfers moment across
    the seam instead of hinging. Use :meth:`couple_displacement` for a
    membrane-only (C0) tie or :meth:`couple_kinematics` for the full G1 tie.
    """

    _cpp_object: _pyck.PenaltyCouplingCondition2d | None

    def __init__(
        self,
        boundary_a: "PatchBoundary",
        boundary_b: "PatchBoundary",
        quadrature: "QuadratureRule | None" = None,
    ) -> None:
        for name, boundary in (("boundary_a", boundary_a), ("boundary_b", boundary_b)):
            if not isinstance(boundary._cpp_object, _pyck.PatchBoundary2d):
                raise TypeError(
                    f"PenaltyCouplingCondition requires 2-D boundary patches; "
                    f"{name} is {type(boundary._cpp_object).__name__}."
                )

        self._boundary_a = boundary_a
        self._boundary_b = boundary_b
        self._quadrature = quadrature
        self._terms: list[tuple[_pyck.BoundaryValue, float]] = []
        self._cpp_object = None

    def add(self, field: Field, penalty: float) -> "PenaltyCouplingCondition":
        """Tie one displacement component across the interface.

        Parameters
        ----------
        field : Field
            Component to tie (only ``Field.U_X``, ``U_Y``, ``U_Z``).
        penalty : float
            Penalty factor for this component.

        Returns
        -------
        PenaltyCouplingCondition
            Self, to allow chaining.
        """
        if self._cpp_object is not None:
            raise RuntimeError(
                "Cannot add fields after the condition has been bound to a problem."
            )
        self._terms.append((_pyck.BoundaryValue(field), float(penalty)))
        return self

    def couple_displacement(self, penalty: float) -> "PenaltyCouplingCondition":
        """Tie all three displacement components with a single penalty.

        Parameters
        ----------
        penalty : float
            Penalty factor applied to ``U_X``, ``U_Y`` and ``U_Z``.

        Returns
        -------
        PenaltyCouplingCondition
            Self, to allow chaining.
        """
        self.add(Field.U_X, penalty)
        self.add(Field.U_Y, penalty)
        self.add(Field.U_Z, penalty)
        return self

    def couple_kinematics(
        self, penalty_disp: float, penalty_rot: float
    ) -> "PenaltyCouplingCondition":
        """Tie displacement and the normal (bending) rotation across the seam.

        Adds ``U_X/U_Y/U_Z`` at ``penalty_disp`` and ``ROT_N`` at ``penalty_rot``.
        This is the full G1 coupling: displacement keeps the surface connected and
        ``ROT_N`` keeps it kink-free, so a bending plate transfers moment across the
        interface instead of hinging. Translation and rotation scale differently,
        hence the two penalties.

        Parameters
        ----------
        penalty_disp : float
            Penalty factor for the displacement components.
        penalty_rot : float
            Penalty factor for the normal/bending rotation.

        Returns
        -------
        PenaltyCouplingCondition
            Self, to allow chaining.
        """
        self.add(Field.U_X, penalty_disp)
        self.add(Field.U_Y, penalty_disp)
        self.add(Field.U_Z, penalty_disp)
        self.add(Field.ROT_N, penalty_rot)
        return self

    def bind(self, _: "QuadratureRule", element: "Element") -> None:
        """Build the C++ object using the element from the parent problem."""
        rule = (
            self._quadrature
            if self._quadrature is not None
            else self._boundary_a.quadrature
        )
        cpp = _pyck.PenaltyCouplingCondition2d(
            self._boundary_a._cpp_object,
            self._boundary_b._cpp_object,
            element._cpp_object,
            rule._cpp_object,
        )
        for field, penalty in self._terms:
            cpp.add(field, penalty)
        self._cpp_object = cpp

    def __repr__(self) -> str:
        return f"PenaltyCouplingCondition(num_fields={len(self._terms)})"
