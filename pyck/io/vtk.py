"""VTK export of solution fields on isogeometric surface patches.

Each Bezier element (knot span) is sampled on a tensor-product parametric
grid and emitted as a block of linear ``VTK_QUAD`` cells. The export is
formulation-agnostic: it relies on each element's
``displacement_shape_matrix`` / ``rotation_shape_matrix``, so the same
function works for KL-1p, RM-3p, RM-displ-3p, RM-1p, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.typing as npt

from pyck.geometry.evaluation import eval_geometry_at, eval_shape_at
from pyck.geometry.surface_patch import SurfacePatch


_VTK_QUAD = 9


def export_field_vtk(
    path: str | Path,
    surf: SurfacePatch,
    element,
    solution: npt.NDArray[np.float64],
    *,
    samples_per_span: int = 8,
    fields: Iterable[str] = ("displacement",),
    title: str = "pyck export",
) -> Path:
    """Export solution fields on a surface patch as a legacy ASCII VTK file.

    Parameters
    ----------
    path : str or Path
        Output filename (``.vtk`` recommended).
    surf : SurfacePatch
        The geometry patch the solution lives on.
    element : plate element (e.g. PlateKirchhoffLove1p,
        PlateReissnerMindlin3p, PlateReissnerMindlinDispl3p,
        PlateReissnerMindlin1p)
        Used only via its ``_cpp_object.displacement_shape_matrix`` and
        ``rotation_shape_matrix`` methods.
    solution : ndarray, shape (n_dof,)
        Global solution vector returned by the solver.
    samples_per_span : int, optional
        Number of sample points per knot span in each parametric direction
        (default 8). The sampled grid is then split into ``(s-1)**2``
        linear quad cells per element.
    fields : iterable of str, optional
        Which fields to write. Recognised names: ``"displacement"`` (alias
        ``"w"``) and ``"rotation"`` (alias ``"phi"``). Default
        ``("displacement",)``.
    title : str, optional
        Header line written to the VTK file (default ``"pyck export"``).

    Returns
    -------
    Path
        The path that was written.
    """
    requested = {_canonical(name) for name in fields}
    unknown = requested - {"w", "phi"}
    if unknown:
        raise ValueError(
            f"Unknown field name(s): {sorted(unknown)}. "
            f"Supported: 'displacement'/'w', 'rotation'/'phi'."
        )

    s = int(samples_per_span)
    if s < 2:
        raise ValueError(f"samples_per_span must be >= 2, got {s}.")

    knots_u = np.unique(np.asarray(surf.basis_u.knots, dtype=np.float64))
    knots_v = np.unique(np.asarray(surf.basis_v.knots, dtype=np.float64))

    # Per-element sample grid (parametric).
    xi = np.linspace(0.0, 1.0, s)
    eta = np.linspace(0.0, 1.0, s)
    xx, yy = np.meshgrid(xi, eta, indexing="ij")
    xi_flat = xx.ravel()
    eta_flat = yy.ravel()

    # Quad connectivity within one s x s sample block (CCW).
    block_quads = _quad_connectivity(s)

    pts_blocks: list[npt.NDArray[np.float64]] = []
    quads_blocks: list[npt.NDArray[np.int64]] = []
    w_blocks: list[npt.NDArray[np.float64]] = []
    phi_blocks: list[npt.NDArray[np.float64]] = []

    solution = np.ascontiguousarray(solution, dtype=np.float64).ravel()
    point_offset = 0

    for vi in range(len(knots_v) - 1):
        v0, v1 = knots_v[vi], knots_v[vi + 1]
        for ui in range(len(knots_u) - 1):
            u0, u1 = knots_u[ui], knots_u[ui + 1]

            u_pts = u0 + xi_flat * (u1 - u0)
            v_pts = v0 + eta_flat * (v1 - v0)
            params = np.column_stack([u_pts, v_pts])

            coords = eval_geometry_at(surf, params)
            shape = eval_shape_at(surf, params, order=2 if "w" in requested else 0)

            pts_blocks.append(coords)
            quads_blocks.append(block_quads + point_offset)
            point_offset += s * s

            if "w" in requested:
                Nw = element._cpp_object.displacement_shape_matrix(shape)
                w_blocks.append(np.asarray(Nw @ solution).ravel())

            if "phi" in requested:
                # rotation_shape_matrix needs first derivatives for KL/RM-1p
                # and third derivatives are unused for displacement output, so
                # request order=3 once if we need rotation alongside w.
                shape_phi = (
                    eval_shape_at(surf, params, order=3)
                    if _needs_higher_order_for_phi(element)
                    else shape
                )
                Nphi = element._cpp_object.rotation_shape_matrix(shape_phi)
                phi_flat = np.asarray(Nphi @ solution).ravel()
                phi_blocks.append(phi_flat.reshape(-1, 2))

    points = np.vstack(pts_blocks)
    quads = np.vstack(quads_blocks)

    point_data: dict[str, npt.NDArray[np.float64]] = {}
    if "w" in requested:
        point_data["w"] = np.concatenate(w_blocks)
    if "phi" in requested:
        point_data["rotation"] = np.vstack(phi_blocks)

    out = Path(path)
    _write_legacy_vtk(out, title, points, quads, point_data)
    return out


def _canonical(name: str) -> str:
    n = name.strip().lower()
    if n in ("w", "displacement", "deflection"):
        return "w"
    if n in ("phi", "rotation", "rotations", "theta"):
        return "phi"
    return n


def _needs_higher_order_for_phi(element) -> bool:
    """Formulations with derived rotations need shape derivatives for phi."""
    cls = type(element).__name__
    return cls in (
        "PlateKirchhoffLove1p",
        "PlateReissnerMindlin1p",
        "PlateReissnerMindlinDispl3p",
    )


def _quad_connectivity(s: int) -> npt.NDArray[np.int64]:
    """Connectivity for an s x s point grid (xi-fastest), as (s-1)^2 quads."""
    quads = np.empty(((s - 1) * (s - 1), 4), dtype=np.int64)
    k = 0
    for j in range(s - 1):
        for i in range(s - 1):
            n0 = i + j * s
            n1 = (i + 1) + j * s
            n2 = (i + 1) + (j + 1) * s
            n3 = i + (j + 1) * s
            quads[k] = (n0, n1, n2, n3)
            k += 1
    return quads


def _write_legacy_vtk(
    path: Path,
    title: str,
    points: npt.NDArray[np.float64],
    quads: npt.NDArray[np.int64],
    point_data: dict[str, npt.NDArray[np.float64]],
) -> None:
    n_pts = points.shape[0]
    n_cells = quads.shape[0]

    with path.open("w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{title}\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        f.write(f"POINTS {n_pts} float\n")
        np.savetxt(f, points, fmt="%.8e")

        f.write(f"\nCELLS {n_cells} {5 * n_cells}\n")
        cells_aug = np.column_stack([np.full(n_cells, 4, dtype=np.int64), quads])
        np.savetxt(f, cells_aug, fmt="%d")

        f.write(f"\nCELL_TYPES {n_cells}\n")
        np.savetxt(f, np.full(n_cells, _VTK_QUAD, dtype=np.int64), fmt="%d")

        if point_data:
            f.write(f"\nPOINT_DATA {n_pts}\n")
            for name, arr in point_data.items():
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    f.write(f"SCALARS {name} float 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    np.savetxt(f, arr, fmt="%.8e")
                elif arr.ndim == 2 and arr.shape[1] == 2:
                    f.write(f"VECTORS {name} float\n")
                    padded = np.column_stack([arr, np.zeros(arr.shape[0])])
                    np.savetxt(f, padded, fmt="%.8e")
                elif arr.ndim == 2 and arr.shape[1] == 3:
                    f.write(f"VECTORS {name} float\n")
                    np.savetxt(f, arr, fmt="%.8e")
                else:
                    raise ValueError(
                        f"Unsupported field shape for '{name}': {arr.shape}"
                    )
