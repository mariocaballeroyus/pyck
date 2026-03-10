"""Simple linear solver for assembled systems.

Since elements now return stiffness and load with BCs already applied,
the solver is just a thin wrapper around `numpy.linalg.solve`.
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
    handles this). This function simply calls `numpy.linalg.solve`.

    Args:
        K: Stiffness matrix `(n, n)` with BCs applied.
        f: Load vector `(n,)` with BCs applied.

    Returns:
        Solution vector `(n,)`.
    """
    return np.linalg.solve(K, f)
