"""VTK export of solution fields on isogeometric surface patches.

Each Bezier element (knot span) is sampled on a tensor-product parametric
grid and emitted as a block of linear ``VTK_QUAD`` cells. The export is
formulation-agnostic: it relies on each element's
``displacement_shape_matrix`` / ``rotation_shape_matrix``, so the same
function works for KL-1p, RM-3p, RM-displ-3p, RM-1p, etc.  The
RM-displ-2p formulation additionally exposes its primary ``w_b`` and
``psi`` fields and their recovered contributions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.typing as npt

from pyck.geometry.evaluation import eval_geometry_at, eval_shape_at
from pyck.geometry.surface import SurfacePatch


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
        ``"w"``), ``"rotation"`` (alias ``"phi"``), ``"kappa"``,
        ``"gamma"``, and for RM-displ-2p also ``"wb"``, ``"psi"``,
        ``"shear_displacement"``, ``"curl_psi"`` as the scalar magnitude
        ``|curl(psi)|``, and ``"curl_psi_vector"``. Pass ``"all"`` to
        write every field supported by the element. Default
        ``("displacement",)``.
    title : str, optional
        Header line written to the VTK file (default ``"pyck export"``).

    Returns
    -------
    Path
        The path that was written.
    """
    requested = _expand_fields({_canonical(name) for name in fields}, element)
    supported = {
        "w",
        "phi",
        "kappa",
        "gamma",
        "wb",
        "ws",
        "psi",
        "curl_psi",
        "curl_psi_vector",
    }
    unknown = requested - supported
    if unknown:
        raise ValueError(
            f"Unknown field name(s): {sorted(unknown)}. "
            f"Supported: 'all', 'displacement'/'w', 'rotation'/'phi', "
            f"'kappa', 'gamma', 'wb', 'shear_displacement', 'psi', 'curl_psi'."
        )
    if requested & {"wb", "ws", "psi", "curl_psi", "curl_psi_vector"} and not _is_rm_displ_2p(element):
        raise ValueError(
            "Fields 'wb', 'shear_displacement', 'psi', 'curl_psi', and "
            "'curl_psi_vector' are only available for PlateReissnerMindlinDispl2p."
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
    field_blocks: dict[str, list[npt.NDArray[np.float64]]] = {
        name: [] for name in requested
    }

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
            shape = eval_shape_at(
                surf, params, order=_shape_order_for_fields(element, requested)
            )

            pts_blocks.append(coords)
            quads_blocks.append(block_quads + point_offset)
            point_offset += s * s

            if "w" in requested:
                Nw = element._cpp_object.displacement_shape_matrix(shape)
                field_blocks["w"].append(np.asarray(Nw @ solution).ravel())

            if "phi" in requested:
                Nphi = element._cpp_object.rotation_shape_matrix(shape)
                phi_flat = np.asarray(Nphi @ solution).ravel()
                field_blocks["phi"].append(phi_flat.reshape(-1, 2))

            if requested & {"kappa", "gamma"}:
                B = element._cpp_object.strain_displacement_matrix(shape)
                strain = np.asarray(B @ solution).reshape(-1, 5)
                if "kappa" in requested:
                    field_blocks["kappa"].append(strain[:, :3])
                if "gamma" in requested:
                    field_blocks["gamma"].append(strain[:, 3:5])

            if requested & {"wb", "ws", "psi", "curl_psi", "curl_psi_vector"}:
                wb_coeff = solution[0::2]
                psi_coeff = solution[1::2]
                wb = np.asarray(shape[0] @ wb_coeff).ravel()
                psi = np.asarray(shape[0] @ psi_coeff).ravel()

                if "wb" in requested:
                    field_blocks["wb"].append(wb)
                if "psi" in requested:
                    field_blocks["psi"].append(psi)
                if "ws" in requested:
                    if "w" in requested:
                        w = field_blocks["w"][-1]
                    else:
                        Nw = element._cpp_object.displacement_shape_matrix(shape)
                        w = np.asarray(Nw @ solution).ravel()
                    field_blocks["ws"].append(w - wb)
                if requested & {"curl_psi", "curl_psi_vector"}:
                    psi_x = np.asarray(shape[1] @ psi_coeff).ravel()
                    psi_y = np.asarray(shape[2] @ psi_coeff).ravel()
                    curl_psi = np.column_stack([psi_y, -psi_x])
                    if "curl_psi" in requested:
                        field_blocks["curl_psi"].append(
                            np.linalg.norm(curl_psi, axis=1)
                        )
                    if "curl_psi_vector" in requested:
                        field_blocks["curl_psi_vector"].append(curl_psi)

    points = np.vstack(pts_blocks)
    quads = np.vstack(quads_blocks)

    point_data: dict[str, npt.NDArray[np.float64]] = {}
    names = {
        "w": "w",
        "phi": "rotation",
        "kappa": "kappa",
        "gamma": "gamma",
        "wb": "wb",
        "ws": "shear_displacement",
        "psi": "psi",
        "curl_psi": "curl_psi",
        "curl_psi_vector": "curl_psi_vector",
    }
    for key in sorted(requested, key=_field_sort_key):
        blocks = field_blocks[key]
        if not blocks:
            continue
        arr = np.concatenate(blocks) if blocks[0].ndim == 1 else np.vstack(blocks)
        point_data[names[key]] = arr

    out = Path(path)
    _write_legacy_vtk(out, title, points, quads, point_data)
    return out


def _canonical(name: str) -> str:
    n = name.strip().lower()
    if n == "all":
        return "all"
    if n in ("w", "displacement", "deflection"):
        return "w"
    if n in ("phi", "rotation", "rotations", "theta"):
        return "phi"
    if n in ("kappa", "bending_strain", "bending-strain"):
        return "kappa"
    if n in ("gamma", "shear_strain", "shear-strain"):
        return "gamma"
    if n in ("wb", "w_b", "bending_displacement", "bending-displacement"):
        return "wb"
    if n in ("ws", "w_s", "shear_displacement", "shear-displacement"):
        return "ws"
    if n in ("psi", "potential"):
        return "psi"
    if n in ("curl_psi", "curl-psi", "curl_psi_magnitude", "curl-psi-magnitude"):
        return "curl_psi"
    if n in ("curl_psi_vector", "curl-psi-vector"):
        return "curl_psi_vector"
    return n


def _expand_fields(requested: set[str], element) -> set[str]:
    if "all" not in requested:
        return requested
    requested = set(requested)
    requested.remove("all")
    requested.update({"w", "phi", "kappa", "gamma"})
    if _is_rm_displ_2p(element):
        requested.update({"wb", "ws", "psi", "curl_psi", "curl_psi_vector"})
    return requested


def _is_rm_displ_2p(element) -> bool:
    return type(element).__name__ == "PlateReissnerMindlinDispl2p"


def _needs_higher_order_for_phi(element) -> bool:
    """Formulations with derived rotations need shape derivatives for phi."""
    cls = type(element).__name__
    return cls in (
        "PlateKirchhoffLove1p",
        "PlateReissnerMindlin1p",
        "PlateReissnerMindlinDispl3p",
    )


def _shape_order_for_fields(element, requested: set[str]) -> int:
    if _is_rm_displ_2p(element):
        return 3
    if requested & {"kappa", "gamma"}:
        return 3
    if "phi" in requested and _needs_higher_order_for_phi(element):
        return 3
    if "w" in requested:
        return 2
    return 0


def _field_sort_key(name: str) -> int:
    order = [
        "w",
        "wb",
        "ws",
        "psi",
        "curl_psi",
        "curl_psi_vector",
        "phi",
        "kappa",
        "gamma",
    ]
    return order.index(name) if name in order else len(order)


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
