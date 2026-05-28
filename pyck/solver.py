"""Simple linear solver for assembled systems.

The assembled stiffness matrix is sparse and solved with
``scipy.sparse.linalg.spsolve``. For augmented saddle-point systems (for
example when Lagrange-multiplier boundary conditions are active), the matrix
can become numerically singular even when the physical problem is still
well-posed. In that case we fall back to a dense least-squares solve to
recover a consistent solution.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

if TYPE_CHECKING:
    from pyck.assembly.assembler import LinearElasticProblem


@overload
def solve(
    K: sparse.spmatrix | npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
    *,
    physical_dofs: int | None = None,
) -> npt.NDArray[np.float64]:
    ...


@overload
def solve(
    K: "LinearElasticProblem",
    f: None = None,
    *,
    physical_dofs: int | None = None,
) -> npt.NDArray[np.float64]:
    ...


def solve(
    K: sparse.spmatrix | npt.NDArray[np.float64] | "LinearElasticProblem",
    f: npt.NDArray[np.float64] | None = None,
    *,
    physical_dofs: int | None = None,
) -> npt.NDArray[np.float64]:
    """Solve a linear system or a linear-elastic problem.

    When passed a :class:`LinearElasticProblem`, this function assembles the
    system and returns only the physical DOFs, excluding appended auxiliary
    unknowns such as Lagrange multipliers.

    Args:
        K: Stiffness matrix `(n, n)` with BCs applied (sparse or dense), or a
            `LinearElasticProblem`.
        f: Load vector `(n,)` with BCs applied. Omit when solving a problem.
        physical_dofs: Optional number of leading physical DOFs to return.
            This is inferred automatically when `K` is a problem.

    Returns:
        Solution vector.
    """
    if _looks_like_problem(K):
        if f is not None:
            raise TypeError("f must be omitted when solving a LinearElasticProblem")
        problem = K
        K, f = problem.assemble()
        physical_dofs = problem.num_physical_dofs
    elif f is None:
        raise TypeError("solve() missing required load vector 'f'")

    if sparse.issparse(K):
        solution = _solve_sparse(K, f)
    else:
        solution = np.linalg.solve(K, f)

    solution = np.asarray(solution, dtype=np.float64).ravel()
    if physical_dofs is not None:
        return solution[: int(physical_dofs)]
    return solution


def _solve_sparse(
    K: sparse.spmatrix, f: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Sparse direct solve, falling back to a dense least-squares solve for
    singular (e.g. saddle-point) systems."""
    solution = sparse_linalg.spsolve(K.tocsc(), f)
    if np.all(np.isfinite(solution)):
        return solution

    solution, _, rank, _ = np.linalg.lstsq(K.toarray(), f, rcond=None)
    print(
        f"[pyck.solve] spsolve returned a non-finite solution; "
        f"falling back to lstsq. K shape={K.shape}, rank={rank} "
        f"(deficit={K.shape[0] - rank}).",
        file=sys.stderr,
    )
    return solution


def _looks_like_problem(value: Any) -> bool:
    return callable(getattr(value, "assemble", None))
