"""Distributed load condition for assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck

if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.elements.element import Element
    from pyck.geometry.patch import Patch


class LoadCondition:
    """Distributed load condition applied to a patch.

    The load values are evaluated at quadrature points and passed to the
    C++ :class:`LoadCondition` which handles assembly.

    Parameters
    ----------
    patch : Patch
        Geometry patch on which the load is applied.
    load_values : ndarray
        Pre-evaluated load values at quadrature points. Can be either:
        - Shape (n_points,) for scalar load (broadcast to first DOF)
        - Shape (n_points * ndof,) for full DOF array

    Notes
    -----
    Use factory methods such as :func:`create_load_condition` for easier
    construction.
    """

    _cpp_object: _pyck.LoadCondition1d | _pyck.LoadCondition2d | None

    def __init__(
        self,
        patch: Patch,
        load_values: npt.NDArray[np.floating],
        quadrature: QuadratureRule | None = None,
    ) -> None:
        self._patch = patch
        self._load_values = np.asarray(load_values, dtype=float).ravel()
        self._quadrature = quadrature
        self._cpp_object = None

    def bind(self, quadrature: QuadratureRule, element: Element) -> None:
        """Build the C++ LoadCondition, binding it to an element.

        If this condition was created with an explicit quadrature rule (via
        a factory), that rule is used and the ``quadrature`` argument is
        ignored — the load values were evaluated at *that* rule's points and
        must be integrated with the same rule.

        Parameters
        ----------
        quadrature : QuadratureRule
            Fallback quadrature rule (used only if none was stored at
            construction time).
        element : Element
            Element formulation (determines DOFs per node).
        """
        rule = self._quadrature if self._quadrature is not None else quadrature
        if isinstance(self._patch._cpp_object, _pyck.Patch1d):
            self._cpp_object = _pyck.LoadCondition1d(
                self._patch._cpp_object,
                element._cpp_object,
                rule._cpp_object,
                self._load_values,
            )
        elif isinstance(self._patch._cpp_object, _pyck.Patch2d):
            self._cpp_object = _pyck.LoadCondition2d(
                self._patch._cpp_object,
                element._cpp_object,
                rule._cpp_object,
                self._load_values,
            )
        else:
            raise ValueError(f"Unsupported patch dimension: {self._patch.tdim}")

    def __repr__(self) -> str:
        status = "pending" if self._cpp_object is None else "bound"
        return f"LoadCondition({status})"


