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

    Wraps :class:`pyck::LoadCondition<double, 1>`.

    The condition is *lazy*: the C++ object is built only when the condition
    is added to a :class:`~pyck.assembly.LinearElasticProblem` via
    `problem.add_condition()`.  At that point the problem supplies its
    quadrature rule; the physical quadrature-point x-coordinates are computed
    entirely in C++ (`Patch.eval_physical_points`), the Python
    callable is evaluated on the resulting array, and the values are forwarded
    to the C++ :class:`LoadCondition`.

    Parameters
    ----------
    patch : CurvePatch
        Geometry patch on which the load is applied.
    load_func : callable
        `f(x: ndarray) -> array_like`.  Receives the physical
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

    def bind(self, quadrature: QuadratureRule, element=None) -> None:
        """Evaluate load and build the C++ LoadCondition.

        Called by :meth:`~pyck.assembly.LinearElasticProblem.add_condition`.

        The physical quadrature points are obtained from C++
        (`Patch.eval_physical_points`); only the load function
        evaluation and the final scalar-to-vector conversion happen in Python.
        """
        x_3d = np.asarray(
            self._patch._cpp_object.eval_physical_points(quadrature._cpp_object)
        )

        # Pass the first physical coordinate (x) to the load function,
        # not the full 3D embedding coordinates.
        x = x_3d[:, 0]

        try:
            raw = self._load_func(x)
        except (TypeError, ValueError, AttributeError):
            # Fallback for functions that don't support array inputs
            # e.g. math.sin (TypeError) or if/else blocks (ValueError)
            raw = np.array([self._load_func(xi) for xi in x])

        vals = np.asarray(raw, dtype=np.float64)
        ndof = element._cpp_object.num_dofs_per_node()
        num_pts = x.shape[0]

        # 1. Handle projection from 3D vectors (either single vector or per-point)
        #    We assume the Y-component is the transversal force.
        if (vals.ndim == 1 and vals.size == 3) or (vals.ndim == 2 and vals.shape[1] == 3):
            # Extract Y component (index 1)
            y_comp = vals[1] if vals.ndim == 1 else vals[:, 1]
            
            # Pack into element DOFs (w, [theta, ...])
            new_vals = np.zeros((num_pts, ndof), dtype=np.float64)
            new_vals[:, 0] = y_comp # Force in Y maps to w
            vals = new_vals
            
        # 2. Handle scalar or single-value returns (broadcast to first DOF)
        elif vals.size == 1:
            val = float(vals.item())
            vals = np.zeros((num_pts, ndof), dtype=np.float64)
            vals[:, 0] = val
            
        # 3. Validation for custom-shaped arrays
        elif vals.size != num_pts * ndof:
             # If it's already (num_pts, ndof), ravel() will fix it.
             # If it's something else, we might need to broadcast or error.
             if vals.ndim == 1 and vals.size == num_pts:
                 # Assume user provided one value per point, map to first DOF
                 new_vals = np.zeros((num_pts, ndof), dtype=np.float64)
                 new_vals[:, 0] = vals
                 vals = new_vals
        
        # Ensure it is a flat 1D array for pybind11 Eigen::VectorXd
        vals = np.ascontiguousarray(vals.ravel(), dtype=np.float64)

        cpp_element = element._cpp_object if element is not None else None
        self._cpp_object = _pyck.LoadCondition1D(
            self._patch._cpp_object, cpp_element, quadrature._cpp_object, vals
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
        `f(x: ndarray) -> array_like`.  Called with the physical
        x-coordinates of the active quadrature points (a numpy array computed
        in C++) and must return the load values at those points.

    Returns
    -------
    LoadCondition
        A pending condition.  Finalised when passed to
        `problem.add_condition()`.

    Examples
    --------
    .. code-block:: python

        def q_load(x):
            return q0 * np.sin(np.pi * x / L)

        load = pyck.create_load_condition(patch, q_load)
        problem.add_condition(load)
    """
    return LoadCondition(patch, load_func)
