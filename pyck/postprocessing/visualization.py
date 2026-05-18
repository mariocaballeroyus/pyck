"""Visualization utilities for patches and solution fields."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyck.geometry import SurfacePatch


def visualize_patch(
    patch: SurfacePatch,
    nu: int = 30,
    nv: int = 30,
    **kwargs,
) -> None:
    """Visualize a surface patch using matplotlib 3D plotting.

    Parameters
    ----------
    patch : SurfacePatch
        The surface patch to visualize.
    nu : int, optional
        Number of sample points in the u direction (default 30).
    nv : int, optional
        Number of sample points in the v direction (default 30).
    **kwargs
        Additional keyword arguments forwarded to matplotlib's plot_surface.
    """
    import matplotlib.pyplot as plt

    u = np.linspace(0.0, 1.0, nu)
    v = np.linspace(0.0, 1.0, nv)
    uu, vv = np.meshgrid(u, v, indexing='ij')
    pts = np.column_stack([uu.ravel(), vv.ravel()])

    xyz = patch.eval_physical(pts).reshape(nu, nv, 3)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2], edgecolor='none', **kwargs)

    # Element lines along unique (non-repeated) knot values in each parametric direction.
    n_line = max(nu, nv)
    s = np.linspace(0.0, 1.0, n_line)
    for dim, basis in enumerate(patch.basis):
        for kval in np.unique(basis.knots):
            if dim == 0:
                pts_line = np.column_stack([np.full(n_line, kval), s])
            else:
                pts_line = np.column_stack([s, np.full(n_line, kval)])
            line = patch.eval_physical(pts_line)
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color='k', linewidth=0.6)

    cps = patch.control_points
    ax.scatter(cps[:, 0], cps[:, 1], cps[:, 2], color='red', s=30, zorder=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(patch.name)

    # Equal axes: set limits symmetrically around the centroid using the max data range.
    half = np.array([np.ptp(xyz[:, :, i]) for i in range(3)]).max() / 2
    mid = np.array([(xyz[:, :, i].max() + xyz[:, :, i].min()) / 2 for i in range(3)])
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

    plt.show()
