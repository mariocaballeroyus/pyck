"""Simple linear solver for assembled systems.

The assembled systems are usually solved with ``numpy.linalg.solve``. For
augmented saddle-point systems (for example when Lagrange-multiplier boundary
conditions are active), the matrix can become numerically singular even when
the physical problem is still well-posed. In that case we fall back to a
least-squares solve to recover a consistent solution.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pyck.assembly.assembler import LinearElasticProblem


@overload
def solve(
    K: npt.NDArray[np.float64],
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
    K: npt.NDArray[np.float64] | "LinearElasticProblem",
    f: npt.NDArray[np.float64] | None = None,
    *,
    physical_dofs: int | None = None,
) -> npt.NDArray[np.float64]:
    """Solve a linear system or a linear-elastic problem.

    When passed a :class:`LinearElasticProblem`, this function assembles the
    system and returns only the physical DOFs, excluding appended auxiliary
    unknowns such as Lagrange multipliers.

    Args:
        K: Stiffness matrix `(n, n)` with BCs applied, or a
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

    try:
        solution = np.linalg.solve(K, f)
    except np.linalg.LinAlgError as exc:
        solution, residuals, rank, _ = np.linalg.lstsq(K, f, rcond=None)
        print(
            f"[pyck.solve] np.linalg.solve failed ({exc}); "
            f"falling back to lstsq. K shape={K.shape}, rank={rank} "
            f"(deficit={K.shape[0] - rank}).",
            file=sys.stderr,
        )

    solution = np.asarray(solution, dtype=np.float64).ravel()
    if physical_dofs is not None:
        return solution[: int(physical_dofs)]
    return solution


def _looks_like_problem(value: Any) -> bool:
    return callable(getattr(value, "assemble", None))
