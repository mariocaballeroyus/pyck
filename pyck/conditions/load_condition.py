"""Distributed (transverse) domain load condition for assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck

if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.elements.element import Element
    from pyck.geometry.patch import Patch


class LoadCondition:
    """Distributed body load integrated over a patch into the load vector.

    Mirrors :class:`LoadBoundaryCondition`: register one or more loads via
    :meth:`add`, either a uniform constant or per-quadrature-point values.

    Parameters
    ----------
    patch : Patch
        Geometry patch on which the load is applied.
    quadrature : QuadratureRule, optional
        Quadrature rule. If omitted, the parent problem's rule is used.
    """

    _cpp_object: _pyck.LoadCondition1d | _pyck.LoadCondition2d | None

    def __init__(
        self,
        patch: Patch,
        quadrature: QuadratureRule | None = None,
    ) -> None:
        self._patch = patch
        self._quadrature = quadrature
        # Each term is a float (uniform) or an ndarray (per quadrature point).
        self._terms: list[float | npt.NDArray[np.floating]] = []
        self._cpp_object = None

    def add(self, value: float | npt.NDArray[np.floating] = 0.0) -> LoadCondition:
        """Add a load term.

        Parameters
        ----------
        value : float or ndarray
            Constant uniform load, or an array sized to the patch's active
            quadrature points (one value per Gauss point).
        """
        if self._cpp_object is not None:
            raise RuntimeError(
                "Cannot add loads after the condition has been bound to a problem."
            )
        if isinstance(value, (int, float, np.floating)):
            self._terms.append(float(value))
        else:
            self._terms.append(np.asarray(value, dtype=float).ravel())
        return self

    def bind(self, quadrature: QuadratureRule, element: Element) -> None:
        """Build the C++ object, binding it to the parent problem's element."""
        rule = self._quadrature if self._quadrature is not None else quadrature
        if isinstance(self._patch._cpp_object, _pyck.Patch1d):
            cpp = _pyck.LoadCondition1d(
                self._patch._cpp_object, element._cpp_object, rule._cpp_object
            )
        elif isinstance(self._patch._cpp_object, _pyck.Patch2d):
            cpp = _pyck.LoadCondition2d(
                self._patch._cpp_object, element._cpp_object, rule._cpp_object
            )
        else:
            raise ValueError(f"Unsupported patch dimension: {self._patch.tdim}")
        for value in self._terms:
            cpp.add(value)
        self._cpp_object = cpp

    def __repr__(self) -> str:
        status = "pending" if self._cpp_object is None else "bound"
        return f"LoadCondition({status}, num_terms={len(self._terms)})"
