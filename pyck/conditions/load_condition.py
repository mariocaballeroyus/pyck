"""Distributed load condition for assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck.conditions.condition import Condition

if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.geometry.curve_patch import CurvePatch


class LoadCondition:
    """Distributed load condition applied to a patch.

    Wraps ``pyck::LoadCondition<double, 1>``.

    The condition is *lazy*: the C++ object is built only when the condition
    is added to a :class:`~pyck.assembly.LinearElasticProblem` via
    ``problem.add_condition()``.  At that point the problem supplies its
    quadrature rule; the physical quadrature-point x-coordinates are computed
    entirely in C++ (``pyck::eval_physical_quadrature_points``), the Python
    callable is evaluated on the resulting array, and the values are forwarded
    to the C++ ``LoadCondition``.

    Parameters
    ----------
    patch : CurvePatch
        Geometry patch on which the load is applied.
    load_func : callable
        ``f(x: ndarray) -> array_like``.  Receives the physical
        x-coordinates of all active quadrature points (computed in C++) and
        must return the corresponding load values.

    Examples
    --------
    .. code-block:: python

        def q_load(x):
            return q0 * np.sin(np.pi * x / L)

        load = pyck.create_load_condition(patch, q_load)
        problem.add_condition(load)  # triggers C++ evaluation
    """

    def __init__(
        self,
        patch: CurvePatch,
        load_func,
    ) -> None:
        self._patch = patch
        self._load_func = load_func
        self._cpp_object = None  # built lazily in bind

    def bind(self, quadrature: QuadratureRule) -> None:
        """Evaluate load and build the C++ LoadCondition.

        Called by :meth:`~pyck.assembly.LinearElasticProblem.add_condition`.

        The physical quadrature points are obtained from C++
        (``_pyck.eval_physical_quadrature_points``); only the load function
        evaluation and the final scalar-to-vector conversion happen in Python.
        """
        x = np.asarray(
            _pyck.eval_physical_quadrature_points(self._patch._cpp_object, quadrature._cpp_object)
        )

        vals = np.asarray(self._load_func(x), dtype=np.float64).ravel()

        self._cpp_object = _pyck.LoadCondition1D(
            self._patch._cpp_object, quadrature._cpp_object, vals
        )

    def __repr__(self) -> str:
        status = "pending" if self._cpp_object is None else "active"
        return f"LoadCondition({status})"


def create_load_condition(
    patch: CurvePatch,
    load_func,
) -> LoadCondition:
    """Create a lazy distributed :class:`LoadCondition`.

    Parameters
    ----------
    patch : CurvePatch
        Geometry patch on which the load acts.
    load_func : callable
        ``f(x: ndarray) -> array_like``.  Called with the physical
        x-coordinates of the active quadrature points (a numpy array computed
        in C++) and must return the load values at those points.

    Returns
    -------
    LoadCondition
        A pending condition.  Finalised when passed to
        ``problem.add_condition()``.

    Examples
    --------
    .. code-block:: python

        def q_load(x):
            return q0 * np.sin(np.pi * x / L)

        load = pyck.create_load_condition(patch, q_load)
        problem.add_condition(load)
    """
    return LoadCondition(patch, load_func)
