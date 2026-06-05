"""VTK export of solution fields on isogeometric surface patches.

``export_bezier_vtu`` — rational-Bezier exporter. Bezier-extracts the patch (one
``VTK_BEZIER_QUADRILATERAL`` cell per knot span) and writes any user fields on the
same extracted basis, so ParaView 5.9+ renders the geometry and fields exactly
(no sampling artefact). Extraction, packing and serialization are done in C++
(:mod:`pyck._pyck`); this module is the ergonomic front door that normalizes the
Python field-dict API and marshals to the native writer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck.geometry.surface_patch import SurfacePatch


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
    binary: bool = True,
) -> Path:
    """Export one or more surface patches as rational-Bezier elements (.vtu).

    Each non-empty knot span becomes one ``VTK_BEZIER_QUADRILATERAL`` cell of
    degree (p, q). The geometry and every ``point_data`` field are pushed
    onto the per-element Bezier basis via the (rational, for NURBS) Bezier
    extraction operator, so the surface and fields are represented exactly —
    ParaView 5.9+ evaluates them with Bernstein polynomials and
    ``RationalWeights`` for the NURBS denominator.

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
        the same rational Bernstein form as the geometry.
    title : str, optional
        XML header comment (default ``"pyck bezier"``).
    binary : bool, optional
        Write array data as raw appended binary (default *True*) — compact and
        fast to load in ParaView. Set *False* for human-readable inline ASCII.

    Returns
    -------
    Path
        The written file path.
    """
    patches, pd_list = _normalise_per_patch(surf, point_data, "point_data")
    _, pde_list = _normalise_per_patch(surf, point_data_extracted,
                                       "point_data_extracted")

    cpp_patches = [p._cpp_object for p in patches]
    pd_maps = [_to_field_map(d) for d in pd_list]
    pde_maps = [_to_field_map(d) for d in pde_list]

    out = Path(path)
    _pyck.export_bezier_vtu(str(out), cpp_patches, pd_maps, pde_maps,
                            title, bool(binary))
    return out


def bezier_anchor_params(surf: SurfacePatch) -> npt.NDArray[np.float64]:
    """Greville parametric coordinates of the Bezier-extracted basis CPs.

    Returns an ``(n_ext_cps, 2)`` array of ``(u, v)`` anchor points, in
    u-fastest order matching the extracted CP layout. Sampling a derived
    field (strain, stress, …) at these points and passing the values to
    :func:`export_bezier_vtu` via ``point_data_extracted`` is the standard
    Greville quasi-interpolation recipe for visualising fields that do not
    live on the displacement basis.
    """
    return np.asarray(_pyck.bezier_anchor_params(surf._cpp_object))


# === Internal helpers ============================================================


def _to_field_map(
    data: Mapping[str, npt.ArrayLike] | None,
) -> dict[str, np.ndarray]:
    """Coerce a field dict to ``name -> (n, k)`` float64 arrays (k≥1).

    Scalars given as 1-D ``(n,)`` are reshaped to ``(n, 1)`` so the native
    writer always sees a 2-D array and reads the component count from its
    columns.
    """
    if not data:
        return {}
    out: dict[str, np.ndarray] = {}
    for name, arr in data.items():
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        elif a.ndim != 2:
            raise ValueError(
                f"field '{name}': must be 1-D (scalar) or 2-D (tensor), "
                f"got shape {a.shape}."
            )
        out[name] = a
    return out


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
