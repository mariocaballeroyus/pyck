"""Visualization utilities for B-spline bases."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

import numpy as np

from matplotlib.axes import Axes

from pyck.basis.basis import Basis
from pyck.io.plotting import basis_plot


def plot_basis_functions(
    basis: Basis,
    order: int = 0,
    n_points: int = 200,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Evaluates and plots the continuous basis functions.

    Parameters
    ----------
    basis : Basis
        The basis function family to plot.
    order : int, optional
        Derivative order to plot (default 0).
    n_points : int, optional
        Number of points for plotting (default 200).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    **kwargs : dict
        Arguments passed to ``ax.plot()``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    u_min, u_max = basis.knots[0], basis.knots[-1]
    u = np.linspace(u_min, u_max, n_points)

    vals = basis.eval_all(u, order=order)
    if order > 0:
        vals = vals[order]

    ax.plot(u, vals, **kwargs)
    return ax


def plot_knots(
    basis: Basis,
    ymin: float = 0.0,
    ymax: float = 1.1,
    ax: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Plots vertical lines representing the knot vector.

    Parameters
    ----------
    basis : Basis
        The basis function family providing the knots.
    ymin : float, optional
        Bottom y-coordinate for the lines (default 0.0).
    ymax : float, optional
        Top y-coordinate for the lines (default 1.1).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    **kwargs : dict
        Arguments passed to ``ax.vlines()``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    unique_knots = np.unique(basis.knots)
    
    # Set defaults if not provided
    kwargs.setdefault("colors", "gray")
    kwargs.setdefault("linestyles", "--")
    kwargs.setdefault("alpha", 0.3)
    kwargs.setdefault("label", "Knots")

    ax.vlines(unique_knots, ymin, ymax, **kwargs)
    return ax


def plot_partition_of_unity(
    basis: Basis,
    n_points: int = 200,
    ax: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Plots the sum of all basis functions to verify partition of unity.

    Parameters
    ----------
    basis : Basis
        The basis function family to plot.
    n_points : int, optional
        Number of points for plotting (default 200).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    **kwargs : dict
        Arguments passed to ``ax.plot()``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    u_min, u_max = basis.knots[0], basis.knots[-1]
    u = np.linspace(u_min, u_max, n_points)

    vals = basis.eval_all(u, order=0)
    # Sum along the basis functions axis (cols) for each point (row).
    # Since vals is (n_points, n_basis), we sum along axis 1.
    total_sum = np.sum(vals, axis=1)

    kwargs.setdefault("label", "Sum of Basis")
    kwargs.setdefault("color", "black")
    kwargs.setdefault("linestyle", "--")
    kwargs.setdefault("linewidth", 2)

    ax.plot(u, total_sum, **kwargs)
    return ax
