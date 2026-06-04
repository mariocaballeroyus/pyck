"""Natural (Neumann) boundary condition: F += integral_Gamma N_v^T t_bar dGamma."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck.conditions.boundary_field import FieldName, resolve_field

if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.elements.element import Element
    from pyck.geometry.patch_boundary import PatchBoundary


ScalarOrArray = Union[float, npt.NDArray[np.floating]]


class LoadBoundaryCondition:
    """Natural (Neumann) boundary condition.

    Integrates the variational external-work term

        deltaW_ext|Gamma_N = integral_{Gamma_N}  t_bar . delta v  dGamma

    into the global load vector. ``v`` is a primal kinematic variable
    (transverse displacement w, normal rotation theta_n, tangential
    rotation theta_s, KL normal slope dw/dn) and ``t_bar`` is its
    work-conjugate prescribed traction. Multiple work-conjugate pairs
    can be added via :meth:`add`; per-span shape evaluation is shared.

    Field selection:
        ``"w"``                              -> shear Q_bar
        ``"rot_n"`` (RM)                     -> bending moment M_n_bar
        ``"rot_s"`` (RM)                     -> twisting moment M_ns_bar
        ``BasisNormalSlope(0)`` (KL-1p)      -> bending moment M_n_bar

    For Kirchhoff-Love elements, prescribing M_ns_bar as a single edge
    integral is incomplete: the user must combine it into the Kirchhoff
    effective shear Q_eff_bar = Q_bar + d(M_ns_bar)/ds and add corner
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
        self._terms: list[tuple[_pyck.BoundaryField, ScalarOrArray]] = []
        self._cpp_object = None

    def add(
        self,
        field: "FieldName | _pyck.BoundaryField",
        value: ScalarOrArray = 0.0,
    ) -> "LoadBoundaryCondition":
        """Add a prescribed traction conjugate to ``field``.

        Parameters
        ----------
        field : str or BoundaryField
            Work-conjugate field selector (e.g. ``"w"`` for shear).
        value : float or ndarray
            Constant scalar, or an array sized to the boundary's active
            quadrature points (one value per Gauss point).
        """
        if self._cpp_object is not None:
            raise RuntimeError(
                "Cannot add fields after the condition has been bound to a problem."
            )
        if isinstance(value, (int, float, np.floating)):
            self._terms.append((resolve_field(field), float(value)))
        else:
            self._terms.append((resolve_field(field),
                                np.asarray(value, dtype=float).ravel()))
        return self

    def add_force(
        self,
        traction: "npt.ArrayLike",
        frame: str = "global",
    ) -> "LoadBoundaryCondition":
        """Add a distributed force per unit boundary length.

        A force conjugates to the full 3D displacement, so a single 3-vector
        covers both in-plane (membrane / axial) and out-of-plane (transverse
        shear) loading; the element's internal membrane/shear split is
        irrelevant to an applied load.

        Parameters
        ----------
        traction : array_like, shape (3,)
            Force per unit length. In the ``"global"`` frame these are Cartesian
            components ``(t_x, t_y, t_z)``; in the ``"local"`` frame they are
            ``(t_n, t_s, t_3)`` along the outward in-surface normal, the
            in-surface tangent ``s = a_3 x n``, and the surface director ``a_3``.
        frame : {"global", "local"}, optional
            Coordinate frame the components are given in (default ``"global"``).
        """
        if self._cpp_object is not None:
            raise RuntimeError(
                "Cannot add fields after the condition has been bound to a problem."
            )
        try:
            local = {"global": False, "local": True}[frame]
        except KeyError:
            raise ValueError(
                f"Unknown frame {frame!r}. Expected 'global' or 'local'."
            ) from None
        vec = np.asarray(traction, dtype=float).ravel()
        if vec.size != 3:
            raise ValueError("traction must have 3 components.")
        self._terms.append((_pyck.ForceTraction(vec, local), 1.0))
        return self

    def add_moment(
        self,
        m_n: float = 0.0,
        m_s: float = 0.0,
    ) -> "LoadBoundaryCondition":
        """Add a distributed edge moment per unit boundary length.

        Moments conjugate to the rotation field and are given in the local
        boundary frame: ``m_n`` (bending) acts about the outward normal and
        pairs with the normal rotation; ``m_s`` (twist) acts about the tangent
        and pairs with the tangential rotation. Zero components are skipped.

        Parameters
        ----------
        m_n : float, optional
            Bending moment per unit length (conjugate to the normal rotation).
        m_s : float, optional
            Twisting moment per unit length (conjugate to the tangential rotation).
        """
        if m_n:
            self.add("rot_n", m_n)
        if m_s:
            self.add("rot_s", m_s)
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


