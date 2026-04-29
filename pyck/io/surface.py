"""Visualization utilities for B-spline surfaces."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from pyck.geometry.surface import SurfacePatch


def evaluate_surface(
    surface: SurfacePatch,
    n_u: int = 40,
    n_v: int = 40,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Evaluate a surface patch on a uniform parametric grid.

    Parameters
    ----------
    surface : SurfacePatch
        The surface to evaluate.
    n_u, n_v : int
        Number of evaluation points in each parametric direction.

    Returns
    -------
    X, Y, Z : ndarray, each shape (n_u, n_v)
        Coordinate arrays suitable for ``plot_surface``.
    """
    coords = surface.evaluate(n_u, n_v)  # (n_u * n_v, 3), u-fastest
    X = coords[:, 0].reshape(n_u, n_v)
    Y = coords[:, 1].reshape(n_u, n_v)
    Z = coords[:, 2].reshape(n_u, n_v)
    return X, Y, Z


def plot_surface(
    surface: SurfacePatch,
    n_u: int = 40,
    n_v: int = 40,
    ax: Optional[Axes3D] = None,
    show_control_points: bool = True,
    show_control_net: bool = True,
    show_knot_lines: bool = False,
    surface_kwargs: Optional[dict[str, Any]] = None,
    cp_kwargs: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> Axes3D:
    """Plot a B-spline surface in 3D.

    Parameters
    ----------
    surface : SurfacePatch
        The surface patch to plot.
    n_u, n_v : int
        Number of evaluation points per parametric direction.
    ax : Axes3D, optional
        3D axes to plot on.  If *None*, a new figure is created.
    show_control_points : bool
        Whether to scatter-plot the control points (default *True*).
    show_control_net : bool
        Whether to draw the control net (default *True*).
    show_knot_lines : bool
        Whether to draw iso-parametric lines at unique internal knots.
    surface_kwargs : dict, optional
        Extra keyword arguments passed to ``ax.plot_surface``.
    cp_kwargs : dict, optional
        Extra keyword arguments passed to ``ax.scatter``.

    Returns
    -------
    Axes3D
    """
    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

    X, Y, Z = evaluate_surface(surface, n_u, n_v)

    # Surface
    skw: dict[str, Any] = {
        "alpha": 0.55,
        "cmap": "viridis",
        "edgecolor": "none",
        "label": surface.name,
    }
    if surface_kwargs:
        skw.update(surface_kwargs)
    ax.plot_surface(X, Y, Z, **skw)

    # Control net
    cpts = surface.control_points
    nu = surface.basis_u.num_basis
    nv = surface.basis_v.num_basis

    if show_control_net:
        _plot_control_net(ax, cpts, nu, nv)

    # Control points
    if show_control_points:
        cpkw: dict[str, Any] = {
            "color": "#e84855",
            "s": 30,
            "zorder": 10,
            "depthshade": False,
            "label": "Control points",
        }
        if cp_kwargs:
            cpkw.update(cp_kwargs)
        ax.scatter(cpts[:, 0], cpts[:, 1], cpts[:, 2], **cpkw)

    # Knot iso-lines
    if show_knot_lines:
        _plot_knot_lines(ax, surface, n_u, n_v)

    ax.legend()
    return ax


def plot_surface_2d(
    surface: SurfacePatch,
    n_u: int = 40,
    n_v: int = 40,
    ax: Optional[Axes] = None,
    show_control_points: bool = True,
    show_control_net: bool = True,
    show_knot_lines: bool = False,
    surface_kwargs: Optional[dict[str, Any]] = None,
    cp_kwargs: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> Axes:
    """Plot a flat (z ≈ 0) surface patch as a 2D x–y view.

    Parameters
    ----------
    surface : SurfacePatch
        The surface patch to plot.
    n_u, n_v : int
        Evaluation grid density.
    ax : Axes, optional
        Matplotlib axes.  If *None*, a new figure is created.
    show_control_points : bool
        Whether to show control points.
    show_control_net : bool
        Whether to draw the control net.
    show_knot_lines : bool
        Whether to draw iso-parametric knot lines.
    surface_kwargs : dict, optional
        Extra kwargs for ``ax.pcolormesh`` (not used if *None*).
    cp_kwargs : dict, optional
        Extra kwargs for ``ax.scatter``.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    X, Y, Z = evaluate_surface(surface, n_u, n_v)

    # Draw mesh lines from the evaluation grid
    for i in range(X.shape[0]):
        ax.plot(X[i, :], Y[i, :], color="#2176ae", linewidth=0.5, alpha=0.4)
    for j in range(X.shape[1]):
        ax.plot(X[:, j], Y[:, j], color="#2176ae", linewidth=0.5, alpha=0.4)

    cpts = surface.control_points
    nu = surface.basis_u.num_basis
    nv = surface.basis_v.num_basis

    if show_control_net:
        _plot_control_net_2d(ax, cpts, nu, nv)

    if show_control_points:
        cpkw: dict[str, Any] = {
            "color": "#e84855",
            "s": 30,
            "zorder": 10,
            "label": "Control points",
        }
        if cp_kwargs:
            cpkw.update(cp_kwargs)
        ax.scatter(cpts[:, 0], cpts[:, 1], **cpkw)

    if show_knot_lines:
        _plot_knot_lines_2d(ax, surface, n_u, n_v)

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    ax.legend()
    return ax


def plot_control_net(
    surface: SurfacePatch,
    ax: Optional[Axes3D] = None,
    **kwargs: Any,
) -> Axes3D:
    """Plot the control net of a surface patch in 3D."""
    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

    cpts = surface.control_points
    nu = surface.basis_u.num_basis
    nv = surface.basis_v.num_basis
    _plot_control_net(ax, cpts, nu, nv, **kwargs)
    ax.scatter(cpts[:, 0], cpts[:, 1], cpts[:, 2],
               color="#e84855", s=30, zorder=10, depthshade=False,
               label="Control points")
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _plot_control_net(
    ax: Axes3D,
    cpts: npt.NDArray[np.float64],
    nu: int,
    nv: int,
    **kwargs: Any,
) -> None:
    """Draw control net edges in 3D (u-fastest ordering)."""
    kwargs.setdefault("color", "gray")
    kwargs.setdefault("linewidth", 0.8)
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("linestyle", "--")

    # u-lines (fixed v)
    for iv in range(nv):
        idx = [iu + iv * nu for iu in range(nu)]
        ax.plot(cpts[idx, 0], cpts[idx, 1], cpts[idx, 2], **kwargs)

    # v-lines (fixed u)
    for iu in range(nu):
        idx = [iu + iv * nu for iv in range(nv)]
        ax.plot(cpts[idx, 0], cpts[idx, 1], cpts[idx, 2], **kwargs)


def _plot_control_net_2d(
    ax: Axes,
    cpts: npt.NDArray[np.float64],
    nu: int,
    nv: int,
    **kwargs: Any,
) -> None:
    """Draw control net edges in 2D (x-y projection)."""
    kwargs.setdefault("color", "gray")
    kwargs.setdefault("linewidth", 0.8)
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("linestyle", "--")

    for iv in range(nv):
        idx = [iu + iv * nu for iu in range(nu)]
        ax.plot(cpts[idx, 0], cpts[idx, 1], **kwargs)

    for iu in range(nu):
        idx = [iu + iv * nu for iv in range(nv)]
        ax.plot(cpts[idx, 0], cpts[idx, 1], **kwargs)


def _plot_knot_lines(
    ax: Axes3D,
    surface: SurfacePatch,
    n_eval: int = 60,
    _: int = 60,
) -> None:
    """Draw iso-parametric lines at unique internal knots (3D)."""
    knots_u = np.unique(surface.basis_u.knots)
    knots_v = np.unique(surface.basis_v.knots)
    v_range = np.linspace(float(knots_v[0]), float(knots_v[-1]), n_eval)
    u_range = np.linspace(float(knots_u[0]), float(knots_u[-1]), n_eval)

    # u = const lines
    for u in knots_u:
        uv = np.column_stack([np.full(n_eval, u), v_range])
        pts = surface.eval_geometry(uv)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color="#57a773", linewidth=1.0, alpha=0.7)

    # v = const lines
    for v in knots_v:
        uv = np.column_stack([u_range, np.full(n_eval, v)])
        pts = surface.eval_geometry(uv)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color="#57a773", linewidth=1.0, alpha=0.7)


def _plot_knot_lines_2d(
    ax: Axes,
    surface: SurfacePatch,
    n_eval: int = 60,
    _: int = 60,
) -> None:
    """Draw iso-parametric lines at unique internal knots (2D)."""
    knots_u = np.unique(surface.basis_u.knots)
    knots_v = np.unique(surface.basis_v.knots)
    v_range = np.linspace(float(knots_v[0]), float(knots_v[-1]), n_eval)
    u_range = np.linspace(float(knots_u[0]), float(knots_u[-1]), n_eval)

    label_set = False
    for u in knots_u:
        uv = np.column_stack([np.full(n_eval, u), v_range])
        pts = surface.eval_geometry(uv)
        lbl = "Knot lines" if not label_set else None
        ax.plot(pts[:, 0], pts[:, 1],
                color="#57a773", linewidth=1.2, alpha=0.8, label=lbl)
        label_set = True

    for v in knots_v:
        uv = np.column_stack([u_range, np.full(n_eval, v)])
        pts = surface.eval_geometry(uv)
        ax.plot(pts[:, 0], pts[:, 1],
                color="#57a773", linewidth=1.2, alpha=0.8)
