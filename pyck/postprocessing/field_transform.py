"""Map covariant surface-field components to physical Cartesian components.

A surface field expressed by its covariant components ``v_alpha = v . A_alpha``
(rotation ``theta_alpha``, transverse shear ``gamma_alpha``, the curl-of-psi
shear, ...) is *basis dependent*: across a non-conforming or slanted multipatch
seam the local tangent basis ``A_alpha`` differs in length and direction, so a
covariant component looks discontinuous in ParaView even when the physical
vector is continuous. :func:`as_cartesian_vector` raises the index and
re-expresses the field in the global Cartesian frame
(``v = v_alpha A^alpha``), giving frame-consistent components that are
continuous across the seam — the form to export for visualization.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck.geometry.patch import Patch


def as_cartesian_vector(
    field: Callable[[npt.ArrayLike], npt.ArrayLike],
    patch: Patch,
) -> Callable[[npt.ArrayLike], npt.NDArray[np.float64]]:
    """Wrap a covariant-component vector field so it returns Cartesian components.

    Use it on any field whose ``(Q, 2)`` output is the covariant components
    ``v_alpha`` of a surface vector — e.g. ``Function(u, el, patch,
    FieldType.ROTATION)`` or a custom shear/curl evaluator — to get a callable
    producing the physical ``(Q, 3)`` Cartesian vector, continuous across
    multipatch seams. Drop the wrapped callable straight into the ``functions``
    dict of :func:`pyck.io.export_bezier_vtu`.

    Parameters
    ----------
    field : callable
        ``field(params) -> (Q, 2)`` covariant components at the parametric points.
    patch : Patch
        Surface patch supplying the geometry (its tangent basis and metric).

    Returns
    -------
    callable
        ``wrapped(params) -> (Q, 3)`` physical Cartesian components.
    """
    cpp_patch = patch._cpp_object

    def wrapped(params: npt.ArrayLike) -> npt.NDArray[np.float64]:
        p = np.ascontiguousarray(np.asarray(params, dtype=np.float64))
        comps = np.ascontiguousarray(np.asarray(field(p), dtype=np.float64))
        return _pyck.covariant_to_cartesian(cpp_patch, p, comps)

    return wrapped
