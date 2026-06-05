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
from typing import Callable, Mapping, Sequence, TypeVar

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck.geometry.surface_patch import SurfacePatch

_V = TypeVar("_V")


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
    functions: Mapping[str, Callable[[npt.ArrayLike], npt.ArrayLike]]
              | Sequence[Mapping[str, Callable[[npt.ArrayLike], npt.ArrayLike]] | None]
              | None = None,
    title: str = "pyck bezier",
    binary: bool = True,
    multiblock: bool = False,
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
    functions : dict or list of dicts, optional
        Derived fields given as callables ``field(params) -> (Q, k)`` — e.g. a
        :class:`pyck.Function` (or any composite field expression). Each is
        sampled internally at ``bezier_anchor_params(patch)`` and added to
        ``point_data_extracted``, so you can pass strain/stress evaluators
        directly without sampling them yourself. Names must not collide with
        ``point_data_extracted``.
    title : str, optional
        XML header comment (default ``"pyck bezier"``).
    binary : bool, optional
        Write array data as raw appended binary (default *True*) — compact and
        fast to load in ParaView. Set *False* for human-readable inline ASCII.
    multiblock : bool, optional
        Write a VTK multiblock dataset (``.vtm`` + one ``.vtu`` per patch in a
        sibling directory) instead of merging all patches into one ``.vtu``
        piece. Keeps patches individually selectable in ParaView (default
        *False*).

    Returns
    -------
    Path
        The written file path.
    """
    patches, pd_list = _normalise_per_patch(surf, point_data, "point_data")
    _, pde_list = _normalise_per_patch(surf, point_data_extracted,
                                       "point_data_extracted")
    _, fn_list = _normalise_per_patch(surf, functions, "functions")

    cpp_patches = [p._cpp_object for p in patches]
    pd_maps = [_to_field_map(d) for d in pd_list]
    pde_maps = [_to_field_map(d) for d in pde_list]

    # Sample callable (Function) fields at each patch's extracted anchors and
    # fold them into the extracted point data.
    for patch, fns, pde in zip(patches, fn_list, pde_maps):
        if not fns:
            continue
        anchors = bezier_anchor_params(patch)
        for name, field in fns.items():
            if name in pde:
                raise ValueError(
                    f"field '{name}' is given in both 'functions' and "
                    f"'point_data_extracted'; choose one source."
                )
            vals = np.ascontiguousarray(np.asarray(field(anchors), dtype=np.float64))
            pde[name] = vals.reshape(-1, 1) if vals.ndim == 1 else vals

    out = Path(path)
    if multiblock:
        names = [p.name for p in patches]
        _pyck.export_bezier_vtm(str(out), cpp_patches, pd_maps, pde_maps,
                                names, title, bool(binary))
    else:
        _pyck.export_bezier_vtu(str(out), cpp_patches, pd_maps, pde_maps,
                                title, bool(binary))
    return out


class BezierVtuWriter:
    """Context manager that accumulates patches + fields and writes on exit.

    A thin, Pythonic front for :func:`export_bezier_vtu`: add patches and their
    fields incrementally, and the file is written when the ``with`` block exits
    cleanly (or via an explicit :meth:`write`).

    Examples
    --------
    Single patch, one derived field::

        with BezierVtuWriter("beam.vtu", patch) as w:
            w.add_field("displacement", disp_fn)   # a pyck.Function

    Multiple patches into a multiblock ``.vtm``::

        with BezierVtuWriter("model.vtm", multiblock=True) as w:
            w.add(patch_a, functions={"disp": disp_a})
            w.add(patch_b, functions={"disp": disp_b})

    Parameters
    ----------
    path : str or Path
        Output filename (``.vtu``, or ``.vtm`` when ``multiblock``).
    surf : SurfacePatch, optional
        If given, the first patch is added immediately so :meth:`add_field`
        can target it without an explicit :meth:`add`.
    title : str, optional
        XML header comment.
    binary : bool, optional
        Raw appended binary (default *True*) vs inline ASCII.
    multiblock : bool, optional
        Write a ``.vtm`` multiblock dataset (one selectable block per patch)
        instead of merging patches into one ``.vtu`` piece.
    """

    def __init__(
        self,
        path: str | Path,
        surf: SurfacePatch | None = None,
        *,
        title: str = "pyck bezier",
        binary: bool = True,
        multiblock: bool = False,
    ) -> None:
        self.path = Path(path)
        self.title = title
        self.binary = binary
        self.multiblock = multiblock
        self._patches: list[SurfacePatch] = []
        self._point_data: list[dict[str, npt.ArrayLike]] = []
        self._extracted: list[dict[str, npt.ArrayLike]] = []
        self._functions: list[dict[str, Callable[[npt.ArrayLike], npt.ArrayLike]]] = []
        if surf is not None:
            self.add(surf)

    def add(
        self,
        surf: SurfacePatch,
        *,
        point_data: Mapping[str, npt.ArrayLike] | None = None,
        point_data_extracted: Mapping[str, npt.ArrayLike] | None = None,
        functions: Mapping[str, Callable[[npt.ArrayLike], npt.ArrayLike]] | None = None,
    ) -> BezierVtuWriter:
        """Add a patch and (optionally) its fields. Returns *self* for chaining."""
        self._patches.append(surf)
        self._point_data.append(dict(point_data) if point_data else {})
        self._extracted.append(dict(point_data_extracted) if point_data_extracted else {})
        self._functions.append(dict(functions) if functions else {})
        return self

    def add_field(
        self,
        name: str,
        field: npt.ArrayLike | Callable[[npt.ArrayLike], npt.ArrayLike],
        *,
        extracted: bool = False,
    ) -> BezierVtuWriter:
        """Add one field to the most recently added patch. Returns *self*.

        A callable (e.g. a :class:`pyck.Function`) is sampled at the extracted
        anchors; an array on the control-point basis is taken as ``point_data``,
        or as ``point_data_extracted`` when ``extracted=True``.
        """
        if not self._patches:
            raise RuntimeError(
                "no patch to attach the field to; pass surf= to the constructor "
                "or call add(patch) first."
            )
        if callable(field):
            self._functions[-1][name] = field
        elif extracted:
            self._extracted[-1][name] = field
        else:
            self._point_data[-1][name] = field
        return self

    def write(self) -> Path:
        """Write the accumulated patches/fields to disk and return the path."""
        if not self._patches:
            raise RuntimeError("nothing to write: no patches were added.")
        single = len(self._patches) == 1
        surf = self._patches[0] if single else self._patches
        return export_bezier_vtu(
            self.path, surf,
            point_data=(self._point_data[0] if single else self._point_data),
            point_data_extracted=(self._extracted[0] if single else self._extracted),
            functions=(self._functions[0] if single else self._functions),
            title=self.title, binary=self.binary, multiblock=self.multiblock,
        )

    def __enter__(self) -> BezierVtuWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.write()
        return False


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
    data: Mapping[str, _V] | Sequence[Mapping[str, _V] | None] | None,
    kw_name: str,
) -> tuple[list[SurfacePatch], list[Mapping[str, _V] | None]]:
    """Coerce `surf` to a list and `data` to a parallel list of dicts."""
    out: list[Mapping[str, _V] | None]
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
