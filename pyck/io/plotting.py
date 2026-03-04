"""Visualization utilities for isogeometric analysis."""

from __future__ import annotations

from typing import Any, Iterator, Optional

import numpy as np

from pyck.basis.basis import Basis


class basis_plot:
    """Class-based context manager for setting up and tearing down a basis plot.

    Parameters
    ----------
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis (default "$u$").
    ylabel : str, optional
        Label for the y-axis.
    figsize : tuple, optional
        Size of the figure (default (8, 4)).
    ax : matplotlib.axes.Axes, optional
        Existing axes to use. If None, a new figure and axes are created.
    """

    def __init__(
        self,
        title: Optional[str] = None,
        xlabel: str = "$u$",
        ylabel: Optional[str] = None,
        figsize: tuple[float, float] = (8, 4),
        ax: Optional[Any] = None,
    ):
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._figsize = figsize
        self._ax = ax
        self._grid = True
        self._legend: Optional[bool] = None
        self._created_fig = False

    def title(self, text: str) -> basis_plot:
        """Set the plot title."""
        self._title = text
        return self

    def xlabel(self, text: str) -> basis_plot:
        """Set the x-axis label."""
        self._xlabel = text
        return self

    def ylabel(self, text: str) -> basis_plot:
        """Set the y-axis label."""
        self._ylabel = text
        return self

    def grid(self, show: bool = True) -> basis_plot:
        """Toggle the grid."""
        self._grid = show
        return self

    def legend(self, show: bool = True) -> basis_plot:
        """Toggle the legend."""
        self._legend = show
        return self

    def __enter__(self) -> Any:
        import matplotlib.pyplot as plt

        if self._ax is None:
            self.fig, self.ax = plt.subplots(figsize=self._figsize)
            self._created_fig = True
        else:
            self.ax = self._ax
        return self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):
        import matplotlib.pyplot as plt

        if self._title:
            self.ax.set_title(self._title)
        if self._xlabel:
            self.ax.set_xlabel(self._xlabel)
        if self._ylabel:
            self.ax.set_ylabel(self._ylabel)

        if self._grid:
            self.ax.grid(True, alpha=0.2)

        # Only show legend if the user added labeled plots or explicitly requested it
        show_legend = self._legend
        if show_legend is None:
            show_legend = bool(self.ax.get_legend_handles_labels()[0])

        if show_legend:
            self.ax.legend()

        if self._created_fig:
            plt.tight_layout()


def plot_basis(
    basis: Basis,
    order: int = 0,
    n_points: int = 200,
    ax: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Plot all basis functions and their derivatives.

    Parameters
    ----------
    basis : Basis
        The basis function family to plot.
    order : int, optional
        Derivative order to plot (default 0).
    n_points : int, optional
        Number of points for plotting (default 200).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    **kwargs : dict
        Arguments passed to ``ax.plot()``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object used for plotting.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    u_min, u_max = basis.knots[0], basis.knots[-1]
    u = np.linspace(u_min, u_max, n_points)

    # Evaluate using the consolidated eval_all method
    vals = basis.eval_all(u, order=order)
    if order > 0:
        vals = vals[order]
        ylabel = f"$d^{{{order}}}N / du^{{{order}}}$"
    else:
        ylabel = "$N(u)$"

    # Plot functions
    ax.plot(u, vals, **kwargs)

    # Plot knots on the axis
    unique_knots = np.unique(basis.knots)
    ax.vlines(
        unique_knots,
        0,
        ax.get_ylim()[1] if order > 0 else 1.1,
        colors="gray",
        linestyles="--",
        alpha=0.3,
        label="Knots" if order == 0 else None,
    )

    ax.set_xlabel("$u$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Basis degree={basis.degree} (order={order})")
    ax.grid(True, alpha=0.2)

    return ax
