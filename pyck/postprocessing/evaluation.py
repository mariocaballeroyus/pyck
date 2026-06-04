"""Sampling helpers for splines at arbitrary parametric points.

The composable C++ kernel API (``Patch.eval_basis(params, span, order)``) is
span-local: it returns basis values relative to the active control points
of a single knot span. Postprocessing typically needs the opposite —
shape values at scattered evaluation points, indexed by *global* control
point. These helpers wrap the per-span calls and scatter back to global
indices.
"""

from __future__ import annotations

import numpy as np

from pyck.geometry.patch import Patch


_DERIV_NAMES_1D = ("N", "N_u", "N_uu", "N_uuu")
_DERIV_NAMES_2D = ("N", "N_u", "N_v", "N_uu", "N_uv", "N_vv",
                   "N_uuu", "N_uuv", "N_uvv", "N_vvv")
_N_ENTRIES_2D = (1, 3, 6, 10)
_N_ENTRIES_1D = (1, 2, 3, 4)


# Rows-per-quadrature-point for each element shape matrix.
_ROWS_PER_QP = {
    "displacement": 1,
    "rotation":     2,
    "bending_strain": 3,
    "shear_strain":   2,
}


def eval_shape_at(
    patch: Patch, params: np.ndarray, order: int = 0
) -> list[np.ndarray]:
    """Evaluate parametric basis derivatives at scattered points (auto span).

    Returns a list of dense ``(Q, n_global)`` matrices, one per parametric
    derivative component, in the canonical order:

      * 1D: ``[N, N_u, N_uu, N_uuu]`` truncated to ``order + 1`` entries.
      * 2D: ``[N, N_u, N_v, N_uu, N_uv, N_vv, N_uuu, N_uuv, N_uvv, N_vvv]``
        truncated to the number of entries up to ``order``.

    Each row holds the basis values/derivatives at one evaluation point,
    with global column indexing (zeros on inactive control points).

    Parameters
    ----------
    patch : Patch
        Curve or surface patch.
    params : np.ndarray, shape (Q, tdim)
        Parametric coordinates per evaluation point. A 1D array is
        promoted to ``(Q, 1)`` for curve patches.
    order : int
        Maximum derivative order (0..3).

    Returns
    -------
    list[np.ndarray]
        Global-indexed basis derivative matrices.
    """
    tdim = patch.tdim
    if tdim == 2:
        return _eval_shape_at_2d(patch, params, order)
    if tdim == 1:
        return _eval_shape_at_1d(patch, params, order)
    raise ValueError(f"Unsupported patch tdim={tdim}")


def eval_geometry_at(patch: Patch, params: np.ndarray) -> np.ndarray:
    """Physical coordinates at scattered parametric points.

    Thin wrapper that forwards to the patch's ``eval_geometry`` method.
    """
    return patch.eval_geometry(params)


def evaluate_field(
    element,
    patch: Patch,
    params: np.ndarray,
    u: np.ndarray,
    ndof: int,
    *,
    kind: str = "displacement",
    order: int = 3,
) -> np.ndarray:
    """Sample an element-defined field at scattered parametric points.

    Loops over the unique knot spans of ``params``, builds the per-span
    ``BasisDerivs`` and ``LocalFrame``, calls
    ``element.<kind>_shape_matrix(patch, basis, local)``, gathers the
    active DOFs from ``u`` and multiplies to obtain the per-quadrature-
    point field. Results from all spans are merged back to the original
    point order.

    Parameters
    ----------
    element
        2D shell element wrapper (e.g. ``ShellReissnerMindlin4p``).
    patch : Patch
        Surface patch the field lives on.
    params : np.ndarray, shape (Q, 2)
        Parametric coordinates per evaluation point.
    u : np.ndarray, shape (n_global * ndof,)
        Per-patch DOF vector (node-major).
    ndof : int
        Number of DOFs per control point.
    kind : str
        One of ``"displacement"``, ``"rotation"``, ``"bending_strain"``,
        ``"shear_strain"``.
    order : int
        Maximum basis-derivative order to build (default 3, covers every
        formulation including RM-D2's third-order shear term).

    Returns
    -------
    np.ndarray, shape (Q, k)
        Sampled field. ``k`` is 1 for displacement, 2 for rotation/shear,
        3 for bending strain (Voigt).
    """
    if kind not in _ROWS_PER_QP:
        raise ValueError(f"Unknown kind '{kind}'. Expected one of {list(_ROWS_PER_QP)}.")
    if patch.tdim != 2:
        raise ValueError("evaluate_field is currently only implemented for 2D patches.")

    params = np.atleast_2d(np.asarray(params, dtype=np.float64))
    Q = params.shape[0]
    k = _ROWS_PER_QP[kind]
    u = np.asarray(u, dtype=np.float64).ravel()

    cpp = patch._cpp_object
    bu_cpp = patch._basis_u._cpp_object
    bv_cpp = patch._basis_v._cpp_object
    intervals_u = patch._basis_u.num_intervals
    mapper = cpp.dof_mapper()
    elem_cpp = element._cpp_object
    matrix_fn = getattr(elem_cpp, f"{kind}_shape_matrix")

    spans = np.empty(Q, dtype=np.int64)
    for i, (u_p, v_p) in enumerate(params):
        span_u = bu_cpp.find_span(float(u_p))
        span_v = bv_cpp.find_span(float(v_p))
        spans[i] = span_u + span_v * intervals_u

    out = np.zeros((Q, k), dtype=np.float64)
    for span in np.unique(spans):
        idx = np.flatnonzero(spans == span)
        basis = cpp.eval_basis(params[idx], int(span), order)
        active_pts = cpp.active_control_pts(int(span))
        local = cpp.eval_local_frame(basis, active_pts)
        mat = np.asarray(matrix_fn(cpp, basis, local))
        dofs = np.asarray(mapper.get_element_dofs(int(span)), dtype=np.int64)
        u_local = u[(dofs[:, None] * ndof + np.arange(ndof)[None, :]).ravel()]
        vals = mat @ u_local  # shape (k * len(idx),)
        out[idx, :] = vals.reshape(len(idx), k)
    return out if k > 1 else out[:, 0]


def _eval_shape_at_1d(
    patch: Patch, params: np.ndarray, order: int
) -> list[np.ndarray]:
    params = np.asarray(params, dtype=np.float64)
    if params.ndim == 1:
        params = params[:, None]
    elif params.ndim == 2 and params.shape[1] != 1:
        params = params[:, :1]

    Q = params.shape[0]
    n_global = patch.num_control_pts
    cpp = patch._cpp_object
    basis_cpp = patch._basis._cpp_object
    mapper = cpp.dof_mapper()

    n_entries = _N_ENTRIES_1D[min(order, 3)]
    out = [np.zeros((Q, n_global), dtype=np.float64) for _ in range(n_entries)]

    spans = np.empty(Q, dtype=np.int64)
    for i, u in enumerate(params[:, 0]):
        spans[i] = basis_cpp.find_span(float(u))

    for span in np.unique(spans):
        idx = np.flatnonzero(spans == span)
        basis = cpp.eval_basis(params[idx], int(span), order)
        cols = np.asarray(mapper.get_element_dofs(int(span)), dtype=np.int64)
        for k, name in enumerate(_DERIV_NAMES_1D[:n_entries]):
            out[k][np.ix_(idx, cols)] = np.asarray(getattr(basis, name))
    return out


def _eval_shape_at_2d(
    patch: Patch, params: np.ndarray, order: int
) -> list[np.ndarray]:
    params = np.atleast_2d(np.asarray(params, dtype=np.float64))
    Q = params.shape[0]
    n_global = patch.num_control_pts
    cpp = patch._cpp_object
    bu_cpp = patch._basis_u._cpp_object
    bv_cpp = patch._basis_v._cpp_object
    intervals_u = patch._basis_u.num_intervals
    mapper = cpp.dof_mapper()

    n_entries = _N_ENTRIES_2D[min(order, 3)]
    out = [np.zeros((Q, n_global), dtype=np.float64) for _ in range(n_entries)]

    spans = np.empty(Q, dtype=np.int64)
    for i, (u, v) in enumerate(params):
        span_u = bu_cpp.find_span(float(u))
        span_v = bv_cpp.find_span(float(v))
        spans[i] = span_u + span_v * intervals_u

    for span in np.unique(spans):
        idx = np.flatnonzero(spans == span)
        basis = cpp.eval_basis(params[idx], int(span), order)
        cols = np.asarray(mapper.get_element_dofs(int(span)), dtype=np.int64)
        for k, name in enumerate(_DERIV_NAMES_2D[:n_entries]):
            out[k][np.ix_(idx, cols)] = np.asarray(getattr(basis, name))
    return out
