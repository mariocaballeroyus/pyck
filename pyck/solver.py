"""Linear solver for assembled systems.

The assembled stiffness matrix is sparse and solved with a sparse direct solver.
When the optional ``pypardiso`` package is installed, Intel MKL PARDISO is used:
it runs **multithreaded across all CPU cores** and handles both symmetric
positive-definite supports (essential / penalty / Nitsche) and symmetric-
indefinite saddle-points (Lagrange multipliers) in one robust path. Without it
the solver falls back to single-threaded SuperLU (``scipy.sparse.linalg.spsolve``).

For a rank-deficient system (e.g. an unconstrained rigid-body mode) the direct
backends fail or return a polluted result; a dense least-squares solve then
recovers a consistent solution.

Thread count follows the standard ``OMP_NUM_THREADS`` environment variable, or
set it explicitly with :func:`set_num_threads`. PARDISO uses every core by
default.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

if TYPE_CHECKING:
    from pyck.assembly.assembler import LinearElasticProblem


# --- Optional multithreaded backend (Intel MKL PARDISO via pypardiso) ---------

try:
    from pypardiso import PyPardisoSolver  # type: ignore[import-not-found]

    HAVE_PARDISO = True
except ImportError:  # pragma: no cover - optional acceleration
    PyPardisoSolver = None  # type: ignore[assignment,misc]
    HAVE_PARDISO = False

# Managed PARDISO instance, created lazily. ``set_num_threads`` rebuilds it so a
# new thread count takes effect (``PyPardisoSolver`` reads ``OMP_NUM_THREADS`` at
# construction).
_pardiso_solver = None


def backend() -> str:
    """Name of the active sparse direct backend (``"pardiso"`` or ``"superlu"``)."""
    return "pardiso" if HAVE_PARDISO else "superlu"


def set_num_threads(n: int) -> None:
    """Set the number of CPU threads the PARDISO backend uses.

    Sets ``OMP_NUM_THREADS`` and rebuilds the PARDISO solver so the change takes
    effect. No-op (beyond the env var) when ``pypardiso`` is not installed.
    """
    os.environ["OMP_NUM_THREADS"] = str(int(n))
    global _pardiso_solver
    if PyPardisoSolver is not None:
        _pardiso_solver = PyPardisoSolver()


def _get_pardiso():
    """The shared PARDISO solver, or ``None`` if ``pypardiso`` is unavailable."""
    global _pardiso_solver
    if _pardiso_solver is None and PyPardisoSolver is not None:
        _pardiso_solver = PyPardisoSolver()
    return _pardiso_solver


# --- Public API ---------------------------------------------------------------

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


# --- Internal helpers ---------------------------------------------------------

def _solve_sparse(
    K: sparse.spmatrix, f: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Sparse direct solve (PARDISO if available, else SuperLU), falling back to
    a dense least-squares solve for singular (e.g. saddle-point) systems."""
    solution = _backend_solve(K.tocsr(), f)
    if np.all(np.isfinite(solution)):
        return solution

    solution, _, rank, _ = np.linalg.lstsq(np.asarray(K.todense()), f, rcond=None)
    print(
        f"[pyck.solve] direct solve returned a non-finite solution; "
        f"falling back to lstsq. K shape={K.shape}, rank={rank} "
        f"(deficit={K.shape[0] - rank}).",
        file=sys.stderr,
    )
    return solution


def _backend_solve(
    K: sparse.csr_matrix, f: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Solve through the active direct backend: multithreaded PARDISO when
    available, otherwise SuperLU. PARDISO raises on a singular / structurally
    degenerate matrix; that is caught so SuperLU (and then the least-squares
    fallback in :func:`_solve_sparse`) can handle it."""
    pardiso = _get_pardiso()
    if pardiso is not None:
        try:
            return pardiso.solve(K, f)
        except Exception:
            pass
    return sparse_linalg.spsolve(K.tocsc(), f)


def _looks_like_problem(value: Any) -> bool:
    return callable(getattr(value, "assemble", None))
