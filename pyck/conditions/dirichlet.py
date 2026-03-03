"""Dirichlet (essential) boundary condition helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck


def assign_scalar(
    dofs: Sequence[int],
    values: Sequence[float],
    K: npt.NDArray[np.float64],
    F: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Enforce Dirichlet conditions with prescribed values.

    Modifies the stiffness matrix *K* and load vector *F* in place so
    that ``u[dof] == value`` for each ``(dof, value)`` pair, using the
    row-zeroing / column-rearrangement method.

    Parameters
    ----------
    dofs : sequence of int
        Global DOF indices.
    values : sequence of float
        Prescribed values, same length as *dofs*.
    K : ndarray, shape (n, n)
        Global stiffness matrix.
    F : ndarray, shape (n,)
        Global load vector.

    Returns
    -------
    K : ndarray
        Modified stiffness matrix.
    F : ndarray
        Modified load vector.
    """
    dofs_list = [int(d) for d in dofs]
    vals_list = [float(v) for v in values]
    K = np.asarray(K, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64).ravel()
    K_new, F_new = _pyck.assign_scalar(dofs_list, vals_list, K, F)
    return np.asarray(K_new), np.asarray(F_new).ravel()


def assign_zeros(
    dofs: Sequence[int],
    K: npt.NDArray[np.float64],
    F: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Enforce homogeneous Dirichlet conditions (u = 0).

    Convenience wrapper around :func:`assign_scalar` with all values
    set to zero.

    Parameters
    ----------
    dofs : sequence of int
        Global DOF indices to constrain.
    K : ndarray, shape (n, n)
        Global stiffness matrix.
    F : ndarray, shape (n,)
        Global load vector.

    Returns
    -------
    K : ndarray
        Modified stiffness matrix.
    F : ndarray
        Modified load vector.
    """
    dofs_list = [int(d) for d in dofs]
    K = np.asarray(K, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64).ravel()
    K_new, F_new = _pyck.assign_zeros(dofs_list, K, F)
    return np.asarray(K_new), np.asarray(F_new).ravel()
