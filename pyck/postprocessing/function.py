"""Callable wrapper around a solution DOF vector and an element formulation.

A :class:`Function` bundles ``(u, element, patch, field)`` into an object that,
when called on a ``(Q, d)`` array of parametric coordinates, returns the
corresponding ``(Q, k)`` field values. The field is selected at construction
time from :class:`FieldType` and dispatches to the matching shape matrix on
the element (displacement, rotation, strain, stress).

Functions support pointwise arithmetic with other Functions, scalars, and bare
Python callables. A bare callable is interpreted as a physical-space field
``f(x, y, z) -> values`` and is composed with a Function by first mapping
parametric coordinates to physical ones via the Function's patch. This lets
you write::

    err = u_numerical - u_analytical   # u_analytical defined in (x, y, z)
    err(params)                        # pointwise error, (Q, k)
    (err ** 2)(params)                 # squared error, ready for integration
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck._pyck import FieldType
from pyck.elements.element import Element
from pyck.geometry.patch import Patch


class _Field:
    """Base for composable parametric-space fields.

    Subclasses implement ``__call__(params) -> (Q, k) array``. The arithmetic
    dunders return lazy composite fields so expressions stay reusable.
    """

    _patch: Patch

    @property
    def patch(self) -> Patch:
        return self._patch

    def __call__(self, params: npt.ArrayLike) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def __neg__(self) -> "_Field":
        return _UnaryOp(self, np.negative)

    def __add__(self, other) -> "_Field":
        return _BinOp(self, _coerce(other, self._patch), np.add)

    def __radd__(self, other) -> "_Field":
        return _BinOp(_coerce(other, self._patch), self, np.add)

    def __sub__(self, other) -> "_Field":
        return _BinOp(self, _coerce(other, self._patch), np.subtract)

    def __rsub__(self, other) -> "_Field":
        return _BinOp(_coerce(other, self._patch), self, np.subtract)

    def __mul__(self, other) -> "_Field":
        return _BinOp(self, _coerce(other, self._patch), np.multiply)

    def __rmul__(self, other) -> "_Field":
        return _BinOp(_coerce(other, self._patch), self, np.multiply)

    def __truediv__(self, other) -> "_Field":
        return _BinOp(self, _coerce(other, self._patch), np.divide)

    def __rtruediv__(self, other) -> "_Field":
        return _BinOp(_coerce(other, self._patch), self, np.divide)

    def __pow__(self, exponent) -> "_Field":
        return _UnaryOp(self, lambda a, e=exponent: np.power(a, e))

    def __getitem__(self, key: int | slice | tuple[int, ...]) -> "_Field":
        """Select component(s) by column index — sugar for :meth:`component`.

        ``f[2]`` is a scalar ``(Q, 1)`` field; ``f[:3]`` and ``f[0, 1, 2]``
        keep their columns as a ``(Q, m)`` field.
        """
        if isinstance(key, tuple):
            key = list(key)
        return _Component(self, key)

    def component(self, *idx: int) -> "_Field":
        """Project onto one or more components, returning a lazy field.

        A single index yields a scalar ``(Q, 1)`` field; several indices keep
        the selected columns as a ``(Q, len(idx))`` field. This is the generic,
        physics-agnostic way to pull individual resultants out of a packed
        field (stresses, tractions, moments) — the caller supplies the meaning;
        the field only knows its column count, so it works unchanged whether a
        shell packs 5 + 3 resultants or a 3-D solid packs 6 stresses.
        """
        return _Component(self, idx[0] if len(idx) == 1 else list(idx))


def _coerce(obj, patch: Patch) -> "_Field":
    """Lift a Function, scalar, or bare physical-space callable into a _Field."""
    if isinstance(obj, _Field):
        return obj
    if callable(obj):
        return _PhysicalCallable(obj, patch)
    if isinstance(obj, (int, float, np.number)):
        return _Constant(float(obj), patch)
    raise TypeError(
        f"Cannot use object of type {type(obj).__name__} in Function arithmetic; "
        "expected a Function, scalar, or callable f(x, y, z)."
    )


class _PhysicalCallable(_Field):
    """Wraps ``f(x, y, z) -> values`` defined in physical coordinates.

    On call with parametric ``params``, evaluates the patch geometry to get
    physical ``(x, y, z)`` and dispatches to ``f``. Scalar or ``(Q,)`` returns
    are promoted to ``(Q, 1)`` so shapes match :class:`Function`.
    """

    def __init__(self, fn: Callable, patch: Patch) -> None:
        self._fn = fn
        self._patch = patch

    def __call__(self, params: npt.ArrayLike) -> np.ndarray:
        xyz = self._patch.eval_geometry(params)
        out = self._fn(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        out = np.asarray(out, dtype=np.float64)
        if out.ndim == 0:
            out = np.full(xyz.shape[0], float(out))
        if out.ndim == 1:
            out = out[:, None]
        return out


class _Constant(_Field):
    """Scalar broadcast to ``(Q, 1)`` across all parametric points."""

    def __init__(self, value: float, patch: Patch) -> None:
        self._value = value
        self._patch = patch

    def __call__(self, params: npt.ArrayLike) -> np.ndarray:
        arr = np.atleast_2d(np.asarray(params, dtype=np.float64))
        return np.full((arr.shape[0], 1), self._value)


class _BinOp(_Field):
    def __init__(self, lhs: _Field, rhs: _Field, op: Callable) -> None:
        self._lhs = lhs
        self._rhs = rhs
        self._op = op
        self._patch = lhs._patch

    def __call__(self, params: npt.ArrayLike) -> np.ndarray:
        return self._op(self._lhs(params), self._rhs(params))


class _UnaryOp(_Field):
    def __init__(self, operand: _Field, op: Callable) -> None:
        self._operand = operand
        self._op = op
        self._patch = operand._patch

    def __call__(self, params: npt.ArrayLike) -> np.ndarray:
        return self._op(self._operand(params))


class _Component(_Field):
    """Lazy projection onto a subset of a field's components.

    ``key`` is anything valid as a column index into the operand's ``(Q, k)``
    output (an ``int``, a ``slice``, or a list of ints). An ``int`` collapses
    to a single column but is promoted back to ``(Q, 1)`` so the result is
    always 2-D and composes with the rest of the field algebra.
    """

    def __init__(self, operand: _Field, key) -> None:
        self._operand = operand
        self._key = key
        self._patch = operand._patch

    def __call__(self, params: npt.ArrayLike) -> np.ndarray:
        vals = np.asarray(self._operand(params))
        if vals.ndim == 1:
            vals = vals[:, None]
        out = vals[:, self._key]
        return out[:, None] if out.ndim == 1 else out


class Function(_Field):
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
    def element(self) -> Element:
        return self._element