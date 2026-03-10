"""Visualization utilities for B-spline curves."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

from pyck.geometry.curve_patch import CurvePatch


def evaluate_curve(
    curve: CurvePatch,
    n_points: int = 200,
) -> npt.NDArray[np.float64]:
    """Evaluate a curve patch at uniformly spaced parametric points.

    Uses De Boor's algorithm (via `eval_geometry`) to compute physical
    coordinates across the full parametric domain.

    Parameters
    ----------
    curve : CurvePatch
        The curve to evaluate.
    n_points : int, optional
        Total number of evaluation points (default 200).

    Returns
    -------
    ndarray, shape (n_points, 3)
        Physical coordinates along the curve.
    """
    return curve.evaluate(n_points)


def plot_curve(
    curve: CurvePatch,
    n_points: int = 200,
    ax: Optional[Axes] = None,
    show_control_points: bool = True,
    show_control_polygon: bool = True,
    show_knots: bool = False,
    curve_kwargs: Optional[dict[str, Any]] = None,
    cp_kwargs: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> Axes:
    """Plot a B-spline curve in 2D (x vs y projection).

    Parameters
    ----------
    curve : CurvePatch
        The curve patch to plot.
    n_points : int, optional
        Number of evaluation points (default 200).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    show_control_points : bool, optional
        Whether to show control points (default True).
    show_control_polygon : bool, optional
        Whether to show the control polygon (default True).
    show_knots : bool, optional
        Whether to mark knot locations on the curve (default False).
    curve_kwargs : dict, optional
        Extra keyword arguments for the curve line plot.
    cp_kwargs : dict, optional
        Extra keyword arguments for the control point scatter plot.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    coords = evaluate_curve(curve, n_points)
    cpts = curve.control_points

    # Curve
    ckw = {"color": "#2176ae", "linewidth": 2, "label": curve.name}
    if curve_kwargs:
        ckw.update(curve_kwargs)
    ax.plot(coords[:, 0], coords[:, 1], **ckw)

    # Control polygon
    if show_control_polygon:
        ax.plot(cpts[:, 0], cpts[:, 1],
                "--", color="gray", alpha=0.5, linewidth=1)

    # Control points
    if show_control_points:
        cpkw: dict[str, Any] = {
            "color": "#e84855", "s": 40, "zorder": 5, "label": "Control points",
        }
        if cp_kwargs:
            cpkw.update(cp_kwargs)
        ax.scatter(cpts[:, 0], cpts[:, 1], **cpkw)

    # Knot locations on the curve
    if show_knots:
        _plot_knot_points(curve, ax)

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    ax.legend()

    return ax


def plot_curve_3d(
    curve: CurvePatch,
    n_points: int = 200,
    ax: Optional[Axes3D] = None,
    show_control_points: bool = True,
    show_control_polygon: bool = True,
    curve_kwargs: Optional[dict[str, Any]] = None,
    cp_kwargs: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> Axes3D:
    """Plot a B-spline curve in 3D.

    Parameters
    ----------
    curve : CurvePatch
        The curve patch to plot.
    n_points : int, optional
        Number of evaluation points (default 200).
    ax : Axes3D, optional
        3D axes to plot on. If None, a new figure is created.
    show_control_points : bool, optional
        Whether to show control points (default True).
    show_control_polygon : bool, optional
        Whether to show the control polygon (default True).
    curve_kwargs : dict, optional
        Extra keyword arguments for the curve line plot.
    cp_kwargs : dict, optional
        Extra keyword arguments for the control point scatter plot.

    Returns
    -------
    Axes3D
    """
    if ax is None:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")

    coords = evaluate_curve(curve, n_points)
    cpts = curve.control_points

    # Curve
    ckw = {"color": "#2176ae", "linewidth": 2, "label": curve.name}
    if curve_kwargs:
        ckw.update(curve_kwargs)
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], **ckw)

    # Control polygon
    if show_control_polygon:
        ax.plot(cpts[:, 0], cpts[:, 1], cpts[:, 2],
                "--", color="gray", alpha=0.5, linewidth=1)

    # Control points
    if show_control_points:
        cpkw: dict[str, Any] = {
            "color": "#e84855", "s": 40, "zorder": 5, "label": "Control points",
        }
        if cp_kwargs:
            cpkw.update(cp_kwargs)
        ax.scatter(cpts[:, 0], cpts[:, 1], cpts[:, 2], **cpkw)

    ax.legend()
    return ax


def plot_control_polygon(
    curve: CurvePatch,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Plot only the control polygon (2D projection).

    Parameters
    ----------
    curve : CurvePatch
        The curve patch whose control polygon to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    cpts = curve.control_points
    kwargs.setdefault("color", "gray")
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("markersize", 5)
    kwargs.setdefault("linewidth", 1)
    kwargs.setdefault("linestyle", "--")
    kwargs.setdefault("alpha", 0.6)
    kwargs.setdefault("label", "Control polygon")

    ax.plot(cpts[:, 0], cpts[:, 1], **kwargs)
    return ax


def _plot_knot_points(
    curve: CurvePatch,
    ax: Axes,
    **kwargs: Any,
) -> None:
    """Mark the images of unique internal knots on the curve."""
    knots = curve.basis.knots
    unique_knots = np.unique(knots)
    pts = curve.eval_geometry(unique_knots)

    kwargs.setdefault("color", "#57a773")
    kwargs.setdefault("s", 25)
    kwargs.setdefault("zorder", 6)
    kwargs.setdefault("marker", "D")
    kwargs.setdefault("label", "Knots")

    ax.scatter(pts[:, 0], pts[:, 1], **kwargs)


def plot_knot_marks(
    curve: CurvePatch,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Plot markers at the physical locations of unique knots on a curve.

    Parameters
    ----------
    curve : CurvePatch
        The curve patch to mark.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    **kwargs : dict
        Arguments passed to `ax.scatter()`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    knots = curve.basis.knots
    unique_knots = np.unique(knots)
    pts = curve.eval_geometry(unique_knots)

    kwargs.setdefault("color", "#57a773")
    kwargs.setdefault("s", 30)
    kwargs.setdefault("zorder", 6)
    kwargs.setdefault("marker", "D")
    kwargs.setdefault("label", "Knots")

    ax.scatter(pts[:, 0], pts[:, 1], **kwargs)
    return ax

    ax.scatter(pts[:, 0], pts[:, 1], **kwargs)
