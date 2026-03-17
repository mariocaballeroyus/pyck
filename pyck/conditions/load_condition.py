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

    The load function is evaluated at quadrature points and passed to the
    C++ :class:`LoadCondition` which handles assembly.

    Parameters
    ----------
    patch : Patch
        Geometry patch on which the load is applied.
    load_func : callable
        Function that returns load values. Called with physical coordinates
        of quadrature points:
        - For 1D & 2D: ``f(x_phys: ndarray) -> array_like``
    """

    _cpp_object: _pyck.LoadCondition1d | _pyck.LoadCondition2d | None

    def __init__(self, patch: Patch, load_func: Callable) -> None:
        self._patch = patch
        self._load_func = load_func
        self._cpp_object = None

    def bind(self, quadrature: QuadratureRule, element: Element) -> None:
        """Build the C++ LoadCondition by evaluating the load function, 
        bounding it to a specific quadrature rule and element.

        Parameters
        ----------
        quadrature : QuadratureRule
            Quadrature rule for integration.
        element : Element
            Element formulation (determines DOFs per node).
        """
        # Recover physical coordinates of quadrature points
        x_phys = np.asarray(
            self._patch._cpp_object.eval_physical_points(quadrature._cpp_object)
        )

        # Evaluate load function at quadrature points
        load_vals = self._load_func(x_phys)
        load_vals = np.asarray(load_vals, dtype=np.float64).ravel()

        # Build the C++ object
        if isinstance(self._patch._cpp_object, _pyck.Patch2d):
            self._cpp_object = _pyck.LoadCondition2d(
                self._patch._cpp_object,
                element._cpp_object,
                quadrature._cpp_object,
                load_vals,
            )
        elif isinstance(self._patch._cpp_object, _pyck.Patch1d):
            self._cpp_object = _pyck.LoadCondition1d(
                self._patch._cpp_object,
                element._cpp_object,
                quadrature._cpp_object,
                load_vals,
            )
        else:
            raise ValueError("Introduce a valid Patch type.")

    def __repr__(self) -> str:
        status = "pending" if self._cpp_object is None else "bound"
        return f"LoadCondition({status})"


def create_load_condition(
    patch: Patch, load_func: Callable
) -> LoadCondition:
    """Create a distributed :class:`LoadCondition`.

    Parameters
    ----------
    patch : Patch
        Geometry patch on which the load acts.
    load_func : callable
        Function returning load values at physical coordinates.

    Returns
    -------
    LoadCondition
        Load condition that will be bound when added to a problem.
    """
    return LoadCondition(patch, load_func)
