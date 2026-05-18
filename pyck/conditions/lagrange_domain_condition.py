"""Lagrange-multiplier domain-integral condition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyck._pyck as _pyck

if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.elements.element import Element
    from pyck.geometry.patch import Patch


class LagrangeDomainCondition:
    """Domain-integral constraints enforced with Lagrange multipliers.

    Each :meth:`add` registers one DOF slot to constrain via a single scalar
    multiplier λ enforcing

        ∫_Ω N_slot · u dΩ = value · |Ω|

    Use case: pin the constant mode of an otherwise un-constrained scalar
    DOF (e.g. ψ in :class:`PlateReissnerMindlinDispl2p`) by registering its
    slot with ``value = 0``.
    """

    _cpp_object: _pyck.LagrangeDomainCondition2d | None

    def __init__(
        self,
        patch: "Patch",
        quadrature: "QuadratureRule | None" = None,
    ) -> None:
        if not isinstance(patch._cpp_object, _pyck.Patch2d):
            raise TypeError(
                f"LagrangeDomainCondition requires a 2-D patch, "
                f"got {type(patch._cpp_object).__name__}."
            )

        self._patch = patch
        self._quadrature = quadrature
        self._terms: list[tuple[int, float]] = []
        self._cpp_object = None

    def add(
        self,
        dof_index: int,
        value: float = 0.0,
    ) -> "LagrangeDomainCondition":
        """Register a DOF slot to constrain.

        Parameters
        ----------
        dof_index : int
            Per-node DOF index whose mean is constrained.
        value : float, optional
            Prescribed mean value over Ω (default 0.0, i.e. zero-mean).

        Returns
        -------
        LagrangeDomainCondition
            Self, to allow chaining.
        """
        if self._cpp_object is not None:
            raise RuntimeError(
                "Cannot add fields after the condition has been bound to a problem."
            )
        self._terms.append((int(dof_index), float(value)))
        return self

    def bind(self, quadrature: "QuadratureRule", element: "Element") -> None:
        """Build the C++ object using the element from the parent problem."""
        rule = self._quadrature if self._quadrature is not None else quadrature
        cpp = _pyck.LagrangeDomainCondition2d(
            self._patch._cpp_object,
            element._cpp_object,
            rule._cpp_object,
        )
        for dof_index, value in self._terms:
            cpp.add(dof_index, value)
        self._cpp_object = cpp

    def __repr__(self) -> str:
        return f"LagrangeDomainCondition(num_terms={len(self._terms)})"


