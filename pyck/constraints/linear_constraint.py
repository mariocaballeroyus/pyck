"""Linear multipoint constraint."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck.constraints.constraint import Constraint


class LinearConstraint(Constraint):
    """Generic linear multipoint constraint coupling DOFs.

    Parameters
    ----------
    slaves : ArrayLike[int]
        Slave DOF indices.
    masters : ArrayLike[int]
        Matrix of master DOFs; row k corresponds to slaves[k].
    weights : ArrayLike[float]
        Shared coefficient weights.
    constant : float, optional
        Non-homogeneous offset (default: 0.0).
    """

    def __init__(
        self,
        slaves: npt.ArrayLike,
        masters: npt.ArrayLike,
        weights: npt.ArrayLike,
        constant: float = 0.0,
    ) -> None:
        self._cpp_object = _pyck.LinearConstraint(
            np.asarray(slaves, dtype=np.uintp).ravel(),
            np.asarray(masters, dtype=np.uintp),
            np.asarray(weights, dtype=np.float64).ravel(),
            float(constant),
        )

    @property
    def slaves(self) -> npt.NDArray[np.uintp]:
        """Read-only access to slave DOF indices."""
        arr = np.asarray(self._cpp_object.slaves())
        arr.flags.writeable = False
        return arr

    @property
    def masters(self) -> npt.NDArray[np.uintp]:
        """Read-only access to master DOF indices."""
        arr = np.asarray(self._cpp_object.masters())
        arr.flags.writeable = False
        return arr

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Read-only access to coefficient weights."""
        arr = np.asarray(self._cpp_object.weights())
        arr.flags.writeable = False
        return arr

    @property
    def constant(self) -> float:
        """Non-homogeneous offset."""
        return self._cpp_object.constant()

    def __repr__(self) -> str:
        return f"LinearConstraint(n_slaves={len(self.slaves)}, n_weights={len(self.weights)})"
