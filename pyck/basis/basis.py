from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from pyck.basis.knot_vector import KnotVector


class Basis(ABC):
    """Abstract base class for one-dimensional basis function families."""

    _cpp_object: Any

    @property
    @abstractmethod
    def degree(self) -> int:
        """Polynomial degree."""
        ...

    @property
    @abstractmethod
    def num_basis(self) -> int:
        """Number of basis functions."""
        ...

    @property
    @abstractmethod
    def knot_vector(self) -> KnotVector:
        """The :class:`KnotVector` associated with this basis."""
        ...

    @property
    @abstractmethod
    def knots(self) -> npt.NDArray[np.float64]:
        """Zero-copy view of the raw knot values."""
        ...

    def eval_all(
        self, u: npt.ArrayLike, order: int = 0
    ) -> npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]:
        """Evaluate all basis functions and their derivatives.

        Parameters
        ----------
        u : array_like
            Parameter values at which to evaluate.
        order : int, optional
            Highest order of derivatives to compute (default 0).

        Returns
        -------
        matrix or list of matrices
            If order=0, returns an (m, n) matrix.
            If order > 0, returns a list of (order+1) (m, n) matrices.
        """
        pts = np.atleast_1d(u).astype(np.float64)
        deriv_list = self._cpp_object.eval_all(pts, int(order))
        
        results = [np.asarray(m) for m in deriv_list]
        return results[0] if order == 0 else results

    def plot(self, order: int = 0, n_points: int = 200, ax: Optional[Any] = None, **kwargs) -> Any:
        """Plot the basis functions and the knot vector.

        Parameters
        ----------
        order : int, optional
            Derivative order to plot (default 0).
        n_points : int, optional
            Number of points for the plot (default 200).
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot onto.
        **kwargs : dict
            Additional arguments passed to ``ax.plot()``.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        u_min, u_max = self.knots[0], self.knots[-1]
        u = np.linspace(u_min, u_max, n_points)
        
        # Evaluate
        vals = self.eval_all(u, order=order)
        if order > 0:
            vals = vals[order]
            ylabel = f"$d^{{{order}}}N / du^{{{order}}}$"
        else:
            ylabel = "$N(u)$"

        # Plot functions
        ax.plot(u, vals, **kwargs)
        
        # Plot knots on the axis
        unique_knots = np.unique(self.knots)
        ax.vlines(unique_knots, 0, ax.get_ylim()[1] if order > 0 else 1.1, 
                  colors='gray', linestyles='--', alpha=0.3, label='Knots' if order == 0 else None)
        
        ax.set_xlabel("$u$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Basis degree={self.degree} (order={order})")
        ax.grid(True, alpha=0.2)
        
        return ax

    def __repr__(self) -> str:
        """String representation of the basis."""
        return f"Basis(degree={self.degree}, num_basis={self.num_basis})"
        