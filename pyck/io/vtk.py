"""VTK export of solution fields on isogeometric surface patches.

Two exporters live here:

* ``export_field_vtk`` — formulation-aware sampler that turns each knot
  span into a block of linear ``VTK_QUAD`` cells via per-element
  ``displacement_shape_matrix`` / ``rotation_shape_matrix`` calls. Works
  for KL-1p, RM-3p, RM-1p, RM-displ-2p, etc.
* ``export_bezier_vtu`` — rational-Bezier exporter. Bezier-extracts the
  patch (one ``VTK_BEZIER_QUADRILATERAL`` cell per knot span) and writes
  any user fields on the same extracted basis, so ParaView 5.9+ renders
  the geometry and fields exactly (no sampling artefact).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from pyck.basis import Basis
from pyck.postprocessing import eval_geometry_at, eval_shape_at
from pyck.geometry.surface_patch import SurfacePatch


_VTK_QUAD = 9
_VTK_BEZIER_QUADRILATERAL = 77


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
        PlateReissnerMindlin3p, PlateReissnerMindlin1p)
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


# === Bezier-element export =======================================================


def export_bezier_vtu(
    path: str | Path,
    surf: SurfacePatch | list[SurfacePatch],
    *,
    point_data: Mapping[str, npt.ArrayLike]
              | Sequence[Mapping[str, npt.ArrayLike] | None]
              | None = None,
    point_data_extracted: Mapping[str, npt.ArrayLike]
              | Sequence[Mapping[str, npt.ArrayLike] | None]
              | None = None,
    title: str = "pyck bezier",
) -> Path:
    """Export one or more surface patches as rational-Bezier elements (.vtu).

    Each non-empty knot span becomes one ``VTK_BEZIER_QUADRILATERAL`` cell of
    degree (p, q). The geometry and every ``point_data`` field are pushed
    onto the per-element Bezier basis via repeated knot insertion to full
    multiplicity (the standard Bezier-extraction operator), so the surface
    and fields are represented exactly — ParaView 5.9+ evaluates them with
    Bernstein polynomials and ``RationalWeights`` for the NURBS denominator.

    Parameters
    ----------
    path : str or Path
        Output filename (``.vtu`` recommended).
    surf : SurfacePatch or list of SurfacePatch
        Patch(es) to export. Multiple patches are concatenated as independent
        Bezier blocks in the same file — useful for symmetry-mirrored copies.
        All patches must share the same set of ``point_data`` field names
        (and trailing dimensions) so ParaView sees a single array per name.
    point_data : dict or list of dicts, optional
        Per-CP field arrays defined on the same basis as the patch. For a
        single patch, a single ``{name: array}`` dict. For multiple patches,
        a parallel list (one dict per patch, same keys). Each array has
        shape ``(num_control_pts,)`` for scalars or ``(num_control_pts, k)``
        for tensors. Bezier-extracted with the same operator as the geometry
        so the rendered field is exact.
    point_data_extracted : dict or list of dicts, optional
        Per-CP field arrays already evaluated on the **extracted** (post-
        knot-insertion) basis — for derived fields like strain or stress
        that don't live on the displacement basis. Each array has shape
        ``(n_ext_cps,)`` or ``(n_ext_cps, k)`` where ``n_ext_cps`` matches
        ``bezier_anchor_params(patch).shape[0]``. The standard recipe is to
        evaluate the field at ``bezier_anchor_params(patch)`` (Greville
        abscissae of the extracted basis); ParaView then interpolates via
        the same rational Bernstein form as the geometry. The result is a
        quasi-interpolant — exact for smooth fields up to ``O(h^{p+1})``,
        with ~1 element of smoothing near singularities.
    title : str, optional
        XML header comment (default ``"pyck bezier"``).

    Returns
    -------
    Path
        The written file path.
    """
    patches, pd_list = _normalise_per_patch(surf, point_data, "point_data")
    _, pde_list = _normalise_per_patch(surf, point_data_extracted,
                                       "point_data_extracted")

    cell_pts_list: list[np.ndarray] = []
    cell_w_list: list[np.ndarray] = []
    cell_field_lists: dict[str, list[np.ndarray]] = {}
    cell_degree_list: list[np.ndarray] = []
    total_cells = 0

    field_dims: dict[str, int] | None = None
    for patch, pd, pde in zip(patches, pd_list, pde_list):
        pts, w, fields, p, q, n_cells = _extract_bezier_cells(patch, pd, pde)

        dims_here = {name: arr.shape[-1] for name, arr in fields.items()}
        if field_dims is None:
            field_dims = dims_here
            for name in field_dims:
                cell_field_lists[name] = []
        elif dims_here != field_dims:
            raise ValueError(
                f"point_data field set / dims mismatch across patches: "
                f"first patch has {field_dims}, this one has {dims_here}."
            )

        cell_pts_list.append(pts)
        cell_w_list.append(w)
        for name, arr in fields.items():
            cell_field_lists[name].append(arr)
        cell_degree_list.append(np.tile(np.asarray([p, q, 0], dtype=np.int32),
                                        (n_cells, 1)))
        total_cells += n_cells

    points = np.vstack(cell_pts_list).reshape(-1, 3)
    weights = np.concatenate([w.reshape(-1) for w in cell_w_list])
    npts_total = points.shape[0]

    pd_out: dict[str, np.ndarray] = {}
    for name, blocks in cell_field_lists.items():
        arr = np.vstack(blocks)
        arr = arr.reshape(-1, arr.shape[-1])
        pd_out[name] = arr.squeeze(-1) if arr.shape[1] == 1 else arr
    pd_out["RationalWeights"] = weights

    connectivity = np.arange(npts_total, dtype=np.int64)
    cell_sizes = np.concatenate(
        [np.full(b.shape[0], b.shape[1], dtype=np.int64) for b in cell_pts_list]
    )
    offsets = np.cumsum(cell_sizes)
    cell_types = np.full(total_cells, _VTK_BEZIER_QUADRILATERAL, dtype=np.uint8)
    degrees = np.vstack(cell_degree_list)

    out = Path(path)
    _write_xml_vtu(
        out, title, points, connectivity, offsets, cell_types,
        point_data=pd_out, cell_data={"HigherOrderDegrees": degrees},
    )
    return out


def _extract_bezier_cells(
    surf: SurfacePatch,
    point_data: Mapping[str, npt.ArrayLike] | None,
    point_data_extracted: Mapping[str, npt.ArrayLike] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], int, int, int]:
    """Bezier-extract one patch and pack per-cell points/weights/fields.

    Returns ``(pts, weights, fields, p, q, n_cells)`` where ``pts`` has shape
    ``(n_cells, (p+1)*(q+1), 3)`` in VTK Bezier-quad ordering, ``weights``
    matches the first two dims, and ``fields`` is a name -> array map of the
    same per-cell layout with trailing field dimension.
    """
    bu, bv = surf.basis[0], surf.basis[1]
    bu_bez, Cu = _bezier_extraction(bu)
    bv_bez, Cv = _bezier_extraction(bv)

    p, q = bu.degree, bv.degree
    nu_old, nv_old = bu.num_basis, bv.num_basis
    nu_new, nv_new = bu_bez.num_basis, bv_bez.num_basis

    cps_old = np.asarray(surf.control_points, dtype=np.float64).reshape(nv_old, nu_old, 3)
    cps_new = _kron_apply(Cu, Cv, cps_old)

    wu_new = np.asarray(getattr(bu_bez, "weights", np.ones(nu_new)), dtype=np.float64)
    wv_new = np.asarray(getattr(bv_bez, "weights", np.ones(nv_new)), dtype=np.float64)
    weights_new = np.outer(wv_new, wu_new)

    extracted: dict[str, np.ndarray] = {}
    if point_data:
        for name, arr in point_data.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.shape[0] != nu_old * nv_old:
                raise ValueError(
                    f"point_data['{name}']: first dim must be num_control_pts "
                    f"({nu_old * nv_old}), got {a.shape[0]}."
                )
            if a.ndim == 1:
                a3 = a.reshape(nv_old, nu_old, 1)
            elif a.ndim == 2:
                a3 = a.reshape(nv_old, nu_old, a.shape[1])
            else:
                raise ValueError(
                    f"point_data['{name}']: must be 1D (scalar) or 2D (tensor), "
                    f"got shape {a.shape}."
                )
            extracted[name] = _kron_apply(Cu, Cv, a3)

    if point_data_extracted:
        for name, arr in point_data_extracted.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.shape[0] != nu_new * nv_new:
                raise ValueError(
                    f"point_data_extracted['{name}']: first dim must be the "
                    f"number of extracted CPs ({nu_new * nv_new}, matches "
                    f"bezier_anchor_params(patch).shape[0]), got {a.shape[0]}."
                )
            if name in extracted:
                raise ValueError(
                    f"Field '{name}' appears in both point_data and "
                    f"point_data_extracted; choose one source."
                )
            if a.ndim == 1:
                extracted[name] = a.reshape(nv_new, nu_new, 1)
            elif a.ndim == 2:
                extracted[name] = a.reshape(nv_new, nu_new, a.shape[1])
            else:
                raise ValueError(
                    f"point_data_extracted['{name}']: must be 1D (scalar) or "
                    f"2D (tensor), got shape {a.shape}."
                )

    n_eu = (nu_new - 1) // p
    n_ev = (nv_new - 1) // q
    if n_eu * p + 1 != nu_new or n_ev * q + 1 != nv_new:
        raise RuntimeError(
            "Bezier extraction did not yield N_e * p + 1 basis functions — "
            "the basis is not fully C^0-extracted; this is a bug."
        )

    perm = _bezier_quad_ordering(p, q)
    npts_per_cell = (p + 1) * (q + 1)
    n_cells = n_eu * n_ev

    pts = np.empty((n_cells, npts_per_cell, 3), dtype=np.float64)
    w = np.empty((n_cells, npts_per_cell), dtype=np.float64)
    fields = {
        name: np.empty((n_cells, npts_per_cell, arr.shape[-1]), dtype=np.float64)
        for name, arr in extracted.items()
    }

    cell = 0
    for ev in range(n_ev):
        j0 = ev * q
        for eu in range(n_eu):
            i0 = eu * p
            lc = cps_new[j0:j0 + q + 1, i0:i0 + p + 1, :].reshape(-1, 3)
            lw = weights_new[j0:j0 + q + 1, i0:i0 + p + 1].reshape(-1)
            pts[cell] = lc[perm]
            w[cell] = lw[perm]
            for name, arr in extracted.items():
                local = arr[j0:j0 + q + 1, i0:i0 + p + 1, :].reshape(-1, arr.shape[-1])
                fields[name][cell] = local[perm]
            cell += 1

    return pts, w, fields, p, q, n_cells


def _normalise_per_patch(
    surf: SurfacePatch | Sequence[SurfacePatch],
    data: Mapping[str, npt.ArrayLike]
        | Sequence[Mapping[str, npt.ArrayLike] | None]
        | None,
    kw_name: str,
) -> tuple[list[SurfacePatch], list[Mapping[str, npt.ArrayLike] | None]]:
    """Coerce `surf` to a list and `data` to a parallel list of dicts."""
    if isinstance(surf, SurfacePatch):
        patches = [surf]
        out = [data if isinstance(data, Mapping) else None]
    else:
        patches = list(surf)
        if data is None:
            out = [None] * len(patches)
        elif isinstance(data, Mapping):
            raise TypeError(
                f"When passing a list of patches, {kw_name} must be a parallel "
                f"list of dicts (one per patch), not a single dict."
            )
        else:
            out = list(data)
            if len(out) != len(patches):
                raise ValueError(
                    f"{kw_name} list length ({len(out)}) does not match number "
                    f"of patches ({len(patches)})."
                )
    return patches, out


def bezier_anchor_params(surf: SurfacePatch) -> npt.NDArray[np.float64]:
    """Greville parametric coordinates of the Bezier-extracted basis CPs.

    Returns an ``(n_ext_cps, 2)`` array of ``(u, v)`` anchor points, in
    u-fastest order matching the extracted CP layout. Sampling a derived
    field (strain, stress, …) at these points and passing the values to
    :func:`export_bezier_vtu` via ``point_data_extracted`` is the standard
    Greville quasi-interpolation recipe for visualising fields that do not
    live on the displacement basis.
    """
    bu, bv = surf.basis[0], surf.basis[1]
    bu_bez, _ = _bezier_extraction(bu)
    bv_bez, _ = _bezier_extraction(bv)
    g_u = _greville(bu_bez)
    g_v = _greville(bv_bez)
    uu, vv = np.meshgrid(g_u, g_v, indexing="xy")  # v outer, u inner
    return np.column_stack([uu.ravel(), vv.ravel()])


def _greville(basis: Basis) -> npt.NDArray[np.float64]:
    """Greville abscissae G_i = mean(knots[i+1 : i+p+1]) for i in 0..n-1."""
    knots = np.asarray(basis.knots, dtype=np.float64)
    p = basis.degree
    n = basis.num_basis
    return np.asarray([knots[i + 1 : i + p + 1].mean() for i in range(n)])


def _bezier_extraction(basis: Basis) -> tuple[Basis, npt.NDArray[np.float64]]:
    """Refine `basis` so every interior knot has multiplicity p; return (basis, T)."""
    p = basis.degree
    knots = np.asarray(basis.knots, dtype=np.float64)
    interior = knots[p + 1:-(p + 1)]
    new_basis = basis
    transform: np.ndarray | None = None
    if interior.size > 0:
        for u in np.unique(interior):
            current_mult = int(np.sum(np.isclose(np.asarray(new_basis.knots), u)))
            count = p - current_mult
            if count > 0:
                new_basis, T_step = new_basis.insert_knot(float(u), count=count)
                transform = T_step if transform is None else T_step @ transform
    if transform is None:
        transform = np.eye(basis.num_basis)
    return new_basis, transform


def _kron_apply(Cu: np.ndarray, Cv: np.ndarray,
                arr: np.ndarray) -> np.ndarray:
    """Apply Cu along the u (axis 1) and Cv along the v (axis 0) direction.

    arr has shape (nv_old, nu_old, k); returns (nv_new, nu_new, k).
    """
    tmp = np.einsum("ij,vjk->vik", Cu, arr)
    return np.einsum("Jv,vik->Jik", Cv, tmp)


def _bezier_quad_ordering(p: int, q: int) -> npt.NDArray[np.int64]:
    """Permutation mapping VTK Bezier-quad ordinal -> lex index ``i + j*(p+1)``.

    VTK Lagrange/Bezier quad layout (vtkHigherOrderQuadrilateral): 4 corners,
    then per-edge interior CPs ordered along *increasing* parametric coords —
    so both i-axis edges run with i=1..p-1 and both j-axis edges with
    j=1..q-1 (NOT a CCW corner walk; edges 2 and 3 are not reversed). Then
    the interior face block in lex order with i fastest.
    """
    def lex(i: int, j: int) -> int:
        return i + j * (p + 1)

    perm: list[int] = [lex(0, 0), lex(p, 0), lex(p, q), lex(0, q)]
    perm.extend(lex(i, 0) for i in range(1, p))    # edge 0: j=0, i ascending
    perm.extend(lex(p, j) for j in range(1, q))    # edge 1: i=p, j ascending
    perm.extend(lex(i, q) for i in range(1, p))    # edge 2: j=q, i ascending
    perm.extend(lex(0, j) for j in range(1, q))    # edge 3: i=0, j ascending
    for j in range(1, q):
        for i in range(1, p):
            perm.append(lex(i, j))                  # interior face, lex
    return np.asarray(perm, dtype=np.int64)


def _write_xml_vtu(
    path: Path,
    title: str,
    points: npt.NDArray[np.float64],
    connectivity: npt.NDArray[np.int64],
    offsets: npt.NDArray[np.int64],
    cell_types: npt.NDArray[np.uint8],
    point_data: Mapping[str, npt.NDArray],
    cell_data: Mapping[str, npt.NDArray],
) -> None:
    n_pts = points.shape[0]
    n_cells = offsets.shape[0]

    def write_array(f, name: str, arr: np.ndarray, dtype: str) -> None:
        ncomp = 1 if arr.ndim == 1 else arr.shape[1]
        name_attr = f' Name="{name}"' if name else ""
        ncomp_attr = f' NumberOfComponents="{ncomp}"' if ncomp > 1 else ""
        f.write(
            f'        <DataArray type="{dtype}"{name_attr}{ncomp_attr} '
            f'format="ascii">\n'
        )
        fmt = "%.8e" if dtype.startswith("Float") else "%d"
        data = arr.reshape(-1, ncomp) if arr.ndim > 1 else arr[:, None]
        np.savetxt(f, data, fmt=fmt)
        f.write("        </DataArray>\n")

    with path.open("w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write(f"<!-- {title} -->\n")
        f.write(
            '<VTKFile type="UnstructuredGrid" version="2.2" '
            'byte_order="LittleEndian">\n'
        )
        f.write("  <UnstructuredGrid>\n")
        f.write(
            f'    <Piece NumberOfPoints="{n_pts}" NumberOfCells="{n_cells}">\n'
        )

        f.write("      <PointData>\n")
        for name, arr in point_data.items():
            write_array(f, name, np.asarray(arr), "Float64")
        f.write("      </PointData>\n")

        f.write("      <CellData>\n")
        for name, arr in cell_data.items():
            write_array(f, name, np.asarray(arr), "Int32")
        f.write("      </CellData>\n")

        f.write("      <Points>\n")
        write_array(f, "", points, "Float64")
        f.write("      </Points>\n")

        f.write("      <Cells>\n")
        write_array(f, "connectivity", connectivity, "Int64")
        write_array(f, "offsets", offsets, "Int64")
        write_array(f, "types", cell_types, "UInt8")
        f.write("      </Cells>\n")

        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")
