"""Generic per-element integration on a patch.

Uses the same per-element pattern as the assembly conditions: only the
``(p+1)^d`` active basis functions are materialised per quadrature batch,
so memory stays bounded regardless of mesh size.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

import pyck._pyck as _pyck
from pyck.assembly.quadrature import QuadratureRule
from pyck.geometry.patch import Patch


_Integrand = Callable[
    [np.ndarray, object, object, Optional[np.ndarray]],
    np.ndarray,
]


def integrate_on_patch(
    patch: Patch,
    quadrature: QuadratureRule,
    integrand: _Integrand,
    *,
    cp_values: Optional[np.ndarray] = None,
    ndof: Optional[int] = None,
    order: int = 0,
) -> np.ndarray:
    """Integrate a user-defined per-quadrature-point functional over a patch.

    The integrator loops over elements; per element it builds the active
    ``BasisDerivs`` and ``LocalFrame`` at the mapped Gauss points and the
    physical coordinates; optionally gathers the active DOF block from
    ``cp_values``; calls the user ``integrand`` to obtain a
    ``(Q, n_components)`` matrix of per-quadrature-point values;
    multiplies by ``dV = w * |J|`` and accumulates each component.

    Parameters
    ----------
    patch
        2D surface patch or 1D curve patch.
    quadrature
        Reference quadrature rule of matching dimension; mapped per element
        internally.
    integrand : callable
        ``integrand(phys_pts, basis, local, u_local) -> values`` where

          - ``phys_pts`` is ``(Q, 3)`` physical coordinates,
          - ``basis`` is the ``BasisDerivs`` struct for the element,
          - ``local`` is the ``LocalFrame`` (tangents, metric, jac) struct,
          - ``u_local`` is a ``(K_active * ndof,)`` array of gathered DOFs,
            or ``None`` if ``cp_values`` was not provided,
          - ``values`` is a ``(Q,)`` vector or ``(Q, n_components)`` matrix
            of per-quadrature-point quantities.
    cp_values : np.ndarray, optional
        Per-control-point DOFs, length ``n_cp * ndof`` (node-major). Pass
        ``None`` to skip the per-element DOF gather.
    ndof : int, optional
        Number of DOFs per control point in ``cp_values``. Required when
        ``cp_values`` is provided.
    order : int, optional
        Shape-derivative order to request when building the per-element
        ``BasisDerivs``. Defaults to 0.

    Returns
    -------
    np.ndarray
        1-D array of length ``n_components`` — one integral per component
        returned by the integrand.
    """
    if cp_values is None:
        cp_arr = np.zeros(0, dtype=np.float64)
        ndof_int = 0
    else:
        if ndof is None:
            raise TypeError("integrate_on_patch: ndof is required when cp_values is given.")
        cp_arr = np.ascontiguousarray(np.asarray(cp_values, dtype=np.float64).ravel())
        ndof_int = int(ndof)

    def _wrap(phys_pts, basis, local, u_local):
        u = None if u_local.size == 0 else u_local
        out = integrand(phys_pts, basis, local, u)
        arr = np.asarray(out, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError(
                "integrate_on_patch: integrand must return a 1-D (Q,) or 2-D "
                f"(Q, n) array, got shape {arr.shape}."
            )
        return np.ascontiguousarray(arr)

    result = _pyck.integrate_on_patch(
        patch._cpp_object,
        quadrature._cpp_object,
        int(order),
        cp_arr,
        ndof_int,
        _wrap,
    )
    return np.asarray(result)
