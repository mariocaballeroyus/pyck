"""Callable wrapper around a solution DOF vector and an element formulation.

A :class:`Function` bundles ``(u, element, patch, field)`` into an object that,
when called on a `(Q, d)` array of parametric coordinates, returns the
corresponding ``(Q, k)`` field values. The field is selected at construction
time from :class:`FieldType` and dispatches to the matching shape matrix on
the element (displacement, rotation, strain, stress).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck._pyck import FieldType
from pyck.elements.element import Element
from pyck.geometry.patch import Patch


class Function:
    """Callable field defined by ``(u, element, patch, field)``."""

    def __init__(
        self,
        u: npt.ArrayLike,
        element: Element,
        patch: Patch,
        field: FieldType,
    ) -> None:
        u_arr = np.ascontiguousarray(np.asarray(u, dtype=np.float64).ravel())
        cls = _pyck.Function1d if patch.tdim == 1 else _pyck.Function2d
        self._cpp_object = cls(u_arr, element._cpp_object, patch._cpp_object, field)
        self._patch = patch
        self._element = element
        self._field = field

    def __call__(self, params: npt.ArrayLike) -> np.ndarray:
        """Evaluate the field at parametric points.

        Parameters
        ----------
        params : array_like, shape (Q, d)
            Parametric coordinates. A 1D array is promoted to ``(Q, 1)`` for
            curve patches.

        Returns
        -------
        np.ndarray, shape (Q, k)
            Field values per point. ``k`` is the number of components
            produced by the corresponding shape matrix on the element.
        """
        arr = np.asarray(params, dtype=np.float64)
        if self._patch.tdim == 1 and arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim == 1:
            arr = arr[None, :]
        return np.asarray(self._cpp_object(arr))

    @property
    def field(self) -> FieldType:
        return self._field

    @property
    def patch(self) -> Patch:
        return self._patch

    @property
    def element(self) -> Element:
        return self._element
