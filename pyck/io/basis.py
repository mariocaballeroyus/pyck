"""Visualization utilities for B-spline bases."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pyck.basis.basis import Basis

__all__ = [
    "basis_plot",
    "plot_basis",
    "plot_basis_functions",
    "plot_basis_function",
    "plot_basis_derivatives",
    "plot_knots",
    "plot_partition_of_unity",
    "highlight_incomplete_support",
    "plot_knot_markers",
]


def plot_basis(
    basis: Basis,
    order: int = 0,
    n_points: int = 200,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
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


def plot_basis_functions(
    basis: Basis,
    order: Optional[int] = None,
    n_points: int = 200,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Evaluates and plots the continuous basis functions.

    Parameters
    ----------
    basis : Basis
        The basis function family to plot.
    order : int, optional
        Polynomial degree to plot. If None, uses basis.degree.
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
    if ax is None:
        ax = plt.gca()

    p = order if order is not None else basis.degree

    # If degree is different, we need a new basis object with same knots
    if p != basis.degree:
        from pyck.basis.bspline import BSpline

        plot_basis_obj = BSpline(p, basis.knot_vector)
    else:
        plot_basis_obj = basis

    u_min, u_max = basis.knots[0], basis.knots[-1]
    u = np.linspace(u_min, u_max, n_points)

    vals = plot_basis_obj.eval_all(u, order=0)
    ax.plot(u, vals, **kwargs)
    return ax


def plot_basis_function(
    basis: Basis,
    index: int,
    order: Optional[int] = None,
    n_points: int = 200,
    span: Optional[int] = None,
    ax: Optional[Axes] = None,
    tol: float = 1e-5,
    fill_alpha: Optional[float] = None,
    **kwargs: Any,
) -> Axes:
    """Plots a single basis function, optionally restricted to a specific knot span.

    Parameters
    ----------
    basis : Basis
        The basis function family to plot.
    index : int
        The index of the basis function to plot.
    order : int, optional
        Polynomial degree to plot. If None, uses basis.degree.
    n_points : int, optional
        Number of points for plotting (default 200).
    span : int, optional
        If provided, only plots the function within the specified knot span [u_span, u_{span+1}].
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    tol : float, optional
        Tolerance below which the function is considered zero and not plotted (default 1e-5).
    fill_alpha : float, optional
        Transparency for filling the area under the curve. If None, no fill is applied.
    **kwargs : dict
        Arguments passed to ``ax.plot()``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    p = order if order is not None else basis.degree

    if p != basis.degree:
        from pyck.basis.bspline import BSpline

        eval_basis = BSpline(p, basis.knot_vector)
    else:
        eval_basis = basis

    if span is not None:
        u_min = eval_basis.knots[span]
        u_max = eval_basis.knots[span + 1]
        if abs(u_max - u_min) < 1e-12:
            return ax  # Empty span
        # Omit the endpoint to prevent drawing vertical "walls" at discontinuities
        u = np.linspace(u_min, u_max, n_points, endpoint=False)
    else:
        u_min, u_max = eval_basis.knots[0], eval_basis.knots[-1]
        u = np.linspace(u_min, u_max, n_points)

    vals = eval_basis.eval_all(u, order=0)
    y = vals[:, index]

    # Only plot if there's any non-zero value
    if np.max(np.abs(y)) > tol:
        line, = ax.plot(u, y, **kwargs)
        if fill_alpha is not None:
            color = kwargs.get("color", line.get_color())
            ax.fill_between(u, y, alpha=fill_alpha, color=color)

    return ax



def plot_basis_derivatives(
    basis: Basis,
    order: int = 1,
    basis_order: Optional[int] = None,
    n_points: int = 200,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Evaluates and plots the derivatives of basis functions.

    Parameters
    ----------
    basis : Basis
        The basis function family to plot.
    order : int, optional
        Derivative order to plot (default 1).
    basis_order : int, optional
        Polynomial degree to derive from. If None, uses basis.degree.
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
    if ax is None:
        ax = plt.gca()

    p = basis_order if basis_order is not None else basis.degree

    if p != basis.degree:
        from pyck.basis.bspline import BSpline

        eval_basis = BSpline(p, basis.knot_vector)
    else:
        eval_basis = basis

    u_min, u_max = basis.knots[0], basis.knots[-1]
    u = np.linspace(u_min, u_max, n_points)

    derivs = eval_basis.eval_all(u, order=order)
    # derivs is a list of arrays, derivs[order] is the one we want
    if order == 0:
        vals = derivs
    else:
        vals = derivs[order]

    ax.plot(u, vals, **kwargs)
    return ax


def plot_knots(
    basis: Basis,
    ymin: float = 0.0,
    ymax: float = 1.1,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
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
    order: Optional[int] = None,
    n_points: int = 200,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Plots the sum of all basis functions to verify partition of unity.

    Parameters
    ----------
    basis : Basis
        The basis function family to plot.
    order : int, optional
        Polynomial degree to check. If None, uses basis.degree.
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
    if ax is None:
        ax = plt.gca()

    p = order if order is not None else basis.degree
    if p != basis.degree:
        from pyck.basis.bspline import BSpline

        eval_basis = BSpline(p, basis.knot_vector)
    else:
        eval_basis = basis

    u_min, u_max = eval_basis.knots[0], eval_basis.knots[-1]
    u = np.linspace(u_min, u_max, n_points)

    vals = eval_basis.eval_all(u, order=0)
    # Sum along the basis functions axis (cols) for each point (row).
    # Since vals is (n_points, n_basis), we sum along axis 1.
    total_sum = np.sum(vals, axis=1)

    kwargs.setdefault("label", f"Sum of Basis (p={p})")
    kwargs.setdefault("color", "black")
    kwargs.setdefault("linestyle", "--")
    kwargs.setdefault("linewidth", 2)

    ax.plot(u, total_sum, **kwargs)
    return ax


def highlight_incomplete_support(
    basis: Basis,
    order: Optional[int] = None,
    ax: Optional[Axes] = None,
    color: str = "red",
    alpha: float = 0.1,
    **kwargs: Any,
) -> Axes:
    """Visually highlights parametric regions where partition of unity is not 1.

    These are the regions outside the full support [u_p, u_n] where the
    basis is incomplete.

    Parameters
    ----------
    basis : Basis
        The basis function family to analyze.
    order : int, optional
        Polynomial degree to analyze. If None, uses basis.degree.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    color : str, optional
        Color of the highlighting (default 'red').
    alpha : float, optional
        Transparency of the highlighting (default 0.1).
    **kwargs : dict
        Arguments passed to ``ax.axvspan()``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    p = order if order is not None else basis.degree
    knots = basis.knots
    # Number of basis functions for degree p is m - p - 1
    n = len(knots) - p - 1

    # The region where partition of unity is 1 is [u_p, u_n]
    u_start_valid = knots[p]
    u_end_valid = knots[n]

    u_min, u_max = knots[0], knots[-1]

    # Handle potentially overlapping regions if the valid range is empty
    if u_start_valid >= u_end_valid:
        # The entire domain is incomplete
        ax.axvspan(u_min, u_max, color=color, alpha=alpha, **kwargs)
    else:
        # Highlight [u_min, u_start_valid] if u_min < u_start_valid
        if u_min < u_start_valid:
            ax.axvspan(u_min, u_start_valid, color=color, alpha=alpha, **kwargs)
            # Avoid duplicate label if both regions are highlighted
            kwargs.pop("label", None)

        # Highlight [u_end_valid, u_max] if u_end_valid < u_max
        if u_end_valid < u_max:
            ax.axvspan(u_end_valid, u_max, color=color, alpha=alpha, **kwargs)

    return ax


def plot_knot_markers(
    basis: Basis,
    ax: Optional[Axes] = None,
    size: float = 0.05,
    color: str = "#005682",
    **kwargs: Any,
) -> Axes:
    """Plots small vertical marks along the axis showing knot multiplicity.

    Multiple knots at the same location are arranged symmetrically.

    Parameters
    ----------
    basis : Basis
        The basis function family providing the knots.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    size : float, optional
        Vertical size of the marks in axis units (default 0.05).
    color : str, optional
        Color of the marks (default '#005682' - teal blue).
    **kwargs : dict
        Arguments passed to ``ax.vlines()``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    knots = basis.knots
    unique_knots, counts = np.unique(knots, return_counts=True)

    # Small horizontal offset to distinguish multiple knots
    domain_width = knots[-1] - knots[0]
    # If domain_width is 0 (all knots same), use a default
    dx = (domain_width if domain_width > 0 else 1.0) * 0.01

    x_coords = []
    for val, count in zip(unique_knots, counts):
        if count == 1:
            x_coords.append(val)
        else:
            # Symmetrically arranged around the knot value
            offsets = np.linspace(-(count - 1) / 2, (count - 1) / 2, count) * dx
            x_coords.extend(val + offsets)

    kwargs.setdefault("alpha", 0.8)
    kwargs.setdefault("linewidth", 2.5)
    # Don't include in legend by default as per user request
    kwargs.setdefault("label", None)

    # Draw marks centered at the axis (y=0)
    ax.vlines(x_coords, -size / 2, size / 2, colors=color, **kwargs)

    return ax
