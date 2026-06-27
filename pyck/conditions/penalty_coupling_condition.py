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
        """Tie one field component across the interface.

        Parameters
        ----------
        field : Field
            Field to tie: the bending-surface displacement (``VB_X/VB_Y/VB_Z``) or
            total displacement (``U_X/U_Y/U_Z``), the normal/bending rotation
            (``ROT_N``), or the hierarchic shear potential and its normal slope
            (``PSI``, ``PSI_N``).
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
        """Tie the three bending-surface displacement components with a single penalty.

        Ties ``VB_X/VB_Y/VB_Z`` — the Cartesian bending surface ``v_b`` (primal
        translation slots), not the recovered total displacement (which for a
        hierarchic shell carries a shear correction). For Kirchhoff–Love the two
        coincide.

        Parameters
        ----------
        penalty : float
            Penalty factor applied to ``VB_X``, ``VB_Y`` and ``VB_Z``.

        Returns
        -------
        PenaltyCouplingCondition
            Self, to allow chaining.
        """
        self.add(Field.VB_X, penalty)
        self.add(Field.VB_Y, penalty)
        self.add(Field.VB_Z, penalty)
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
        self.add(Field.VB_X, penalty_disp)
        self.add(Field.VB_Y, penalty_disp)
        self.add(Field.VB_Z, penalty_disp)
        self.add(Field.ROT_N, penalty_rot)
        return self

    def couple_director_continuity(self, penalty: float) -> "PenaltyCouplingCondition":
        """Penalise G1 (slope) continuity across the seam, in director form.

        Adds ``DIR_N`` — the director-based normal rotation ``δa_3·n`` computed per patch
        and combined as a sum across the seam (it is odd in the co-normal, which flips
        ``n_A = −n_B``). For the displacement-based shells this is the ``ROT_N`` slope tie
        in director variables; for Reissner–Mindlin shells it ties the *surface* normal
        rather than the director field. Pair with :meth:`couple_displacement` for the full
        G1 coupling. Requires an element supplying the director variation.

        Parameters
        ----------
        penalty : float
            Penalty factor for the director-continuity (C1) term.

        Returns
        -------
        PenaltyCouplingCondition
            Self, to allow chaining.
        """
        self.add(Field.DIR_N, penalty)
        return self

    def couple_curvature_continuity(self, penalty: float) -> "PenaltyCouplingCondition":
        """Penalise C2 (curvature) continuity across the seam.

        Adds ``KAPPA_NN`` — the normal curvature ``n^α n^β κ_αβ`` (the bending strain
        contracted twice with the co-normal) computed per patch and combined as a
        difference across the seam (it is even in the co-normal). On top of the C0
        (:meth:`couple_displacement`) and C1 (:meth:`couple_director_continuity`) ties it
        adds the cross-seam curvature, completing C2 of the bending surface. Requires an
        element supplying the normal-curvature trace (any shell).

        Parameters
        ----------
        penalty : float
            Penalty factor for the curvature-continuity (C2) term.

        Returns
        -------
        PenaltyCouplingCondition
            Self, to allow chaining.
        """
        self.add(Field.KAPPA_NN, penalty)
        return self

    def couple_psi_continuity(
        self, penalty_value: float, penalty_slope: float
    ) -> "PenaltyCouplingCondition":
        """Tie the hierarchic field ψ to C1 across the seam (four-parameter shells).

        Adds ``PSI`` (value, C0) at ``penalty_value`` and ``PSI_N`` (the boundary-normal
        slope ψ_,n) at ``penalty_slope``. The C0 tie makes the along-seam derivative
        continuous; the normal slope completes a continuous ∇ψ (C1 of ψ), which is what
        renders the ψ-sourced shear rotation continuous across the seam.

        Parameters
        ----------
        penalty_value : float
            Penalty factor for the ψ value (C0).
        penalty_slope : float
            Penalty factor for the ψ normal slope ψ_,n (the C1 part).

        Returns
        -------
        PenaltyCouplingCondition
            Self, to allow chaining.
        """
        self.add(Field.PSI, penalty_value)
        self.add(Field.PSI_N, penalty_slope)
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
