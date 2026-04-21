"""Simple linear solver for assembled systems.

The assembled systems are usually solved with ``numpy.linalg.solve``. For
augmented saddle-point systems (for example when Lagrange-multiplier boundary
conditions are active), the matrix can become numerically singular even when
the physical problem is still well-posed. In that case we fall back to a
least-squares solve to recover a consistent solution.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def solve(
    K: npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solve the linear system K d = f.

    BCs are expected to be already enforced in K and f (the element
    handles this).

    Args:
        K: Stiffness matrix `(n, n)` with BCs applied.
        f: Load vector `(n,)` with BCs applied.

    Returns:
        Solution vector `(n,)`.
    """
    try:
        return np.linalg.solve(K, f)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(K, f, rcond=None)
        return sol
