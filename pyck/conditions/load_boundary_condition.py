"""Natural (Neumann) boundary condition: F += integral_Gamma N_v^T t_bar dGamma."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck.conditions.condition import Field

if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.elements.element import Element
    from pyck.geometry.patch_boundary import PatchBoundary


ScalarOrArray = Union[float, npt.NDArray[np.floating]]


class LoadBoundaryCondition:
    """Natural (Neumann) boundary condition.

    Integrates the variational external-work term

        deltaW_ext|Gamma_N = integral_{Gamma_N}  t_bar . delta v  dGamma

    into the global load vector. ``v`` is a primal kinematic variable and
    ``t_bar`` is its work-conjugate prescribed traction. Register work-conjugate
    pairs via :meth:`add` (e.g. ``Field.U_X`` for an x-force, ``Field.ROT_N`` for
    a bending moment); per-span shape evaluation is shared across terms.

    For Kirchhoff-Love elements, prescribing the twisting moment (``Field.ROT_S``)
    as a single edge integral is incomplete: the user must combine it into the
    Kirchhoff effective shear Q_eff_bar = Q_bar + d(M_ns_bar)/ds and add corner
    forces externally. No runtime check is performed.
    """

    _cpp_object: _pyck.LoadBoundaryCondition2d | None

    def __init__(
        self,
        boundary: "PatchBoundary",
        quadrature: "QuadratureRule | None" = None,
    ) -> None:
        if not isinstance(boundary._cpp_object, _pyck.PatchBoundary2d):
            raise TypeError(
                f"LoadBoundaryCondition requires a 2-D boundary patch, "
                f"got {type(boundary._cpp_object).__name__}."
            )
        self._boundary = boundary
        self._quadrature = quadrature
        # Each term is (field, value) where value is float or ndarray.
        self._terms: list[tuple[_pyck.BoundaryValue, ScalarOrArray]] = []
        self._cpp_object = None

    def add(
        self,
        field: Field,
        value: ScalarOrArray = 0.0,
    ) -> "LoadBoundaryCondition":
        """Add a prescribed traction conjugate to ``field``.

        Parameters
        ----------
        field : Field
            Work-conjugate value selector (e.g. ``Field.U_X`` for an x-force,
            ``Field.ROT_N`` for a bending moment).
        value : float or ndarray
            Constant scalar, or an array sized to the boundary's active
            quadrature points (one value per Gauss point).
        """
        if self._cpp_object is not None:
            raise RuntimeError(
                "Cannot add fields after the condition has been bound to a problem."
            )
        if isinstance(value, (int, float, np.floating)):
            self._terms.append((_pyck.BoundaryValue(field), float(value)))
        else:
            self._terms.append((_pyck.BoundaryValue(field),
                                np.asarray(value, dtype=float).ravel()))
        return self

    def bind(self, _: "QuadratureRule", element: "Element") -> None:
        """Build the C++ object using the element from the parent problem."""
        rule = self._quadrature if self._quadrature is not None else self._boundary.quadrature
        cpp = _pyck.LoadBoundaryCondition2d(
            self._boundary._cpp_object,
            element._cpp_object,
            rule._cpp_object,
        )
        for field, value in self._terms:
            cpp.add(field, value)
        self._cpp_object = cpp

    def __repr__(self) -> str:
        return f"LoadBoundaryCondition(num_fields={len(self._terms)})"
