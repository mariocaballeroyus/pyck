"""Standalone functions for evaluating parametric patches.

These functions handle span searching internally, providing a convenient 
but slightly slower alternative to the direct span-based methods.
"""

from __future__ import annotations

import numpy as np
import pyck._pyck as _pyck
from pyck.geometry.patch import Patch


def eval_shape_at(
    patch: Patch, params: np.ndarray, order: int = 0
) -> list[np.ndarray]:
    """Evaluate shape functions at given parametric points (auto span finding).

    Parameters
    ----------
    patch : Patch
        The patch to evaluate.
    params : np.ndarray, shape (Q, dim)
        Parametric coordinates for each evaluation point.
    order : int, optional
        Maximum derivative order (default 0).

    Returns
    -------
    list[np.ndarray]
        List of shape function values and derivatives.
    """
    params = np.atleast_2d(np.asarray(params, dtype=np.float64))
    return _pyck.eval_shape_at(patch._cpp_object, params, order)


def eval_geometry_at(patch: Patch, params: np.ndarray) -> np.ndarray:
    """Evaluate physical coordinates at given parametric points (auto span finding).

    Parameters
    ----------
    patch : Patch
        The patch to evaluate.
    params : np.ndarray, shape (Q, dim)
        Parametric coordinates for each evaluation point.

    Returns
    -------
    np.ndarray, shape (Q, 3)
        Physical coordinates in 3D space.
    """
    params = np.atleast_2d(np.asarray(params, dtype=np.float64))
    return _pyck.eval_geometry_at(patch._cpp_object, params)
