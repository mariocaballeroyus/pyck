"""Mixed assumed-membrane-strain shell element (Guo, Zou & Ruess, IJNME 2020)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck.elements.element import Element

if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.geometry.surface_patch import SurfacePatch


class MixedMembraneStrainShell(Element):
    """Faithful, global Hellinger-Reissner treatment of membrane locking.

    Wraps any displacement shell on the same patch and *is* the element handed to
    the problem. The base element's membrane strain block is suppressed at the
    source, and the membrane is re-supplied by an independent strain field
    interpolated on three coarser anisotropic B-spline spaces (degrees ``(p-1, q)``,
    ``(p, q-1)``, ``(p-1, q-1)``; Echter Table 6.2) kept as **global sparse**
    auxiliary DOFs — no static condensation. Bending, transverse shear, constitutive
    and shape matrices forward to the base shell, so displacement / rotation / moment
    recovery is unchanged; the faithful, oscillation-free membrane strain / force are
    recovered from the field (see :meth:`recover_membrane_force` and
    ``ck.Function(u_full, shell, patch, ck.FieldType.TRACTION)``).

    Parameters
    ----------
    patch : SurfacePatch
        The patch carrying the displacement shell and the assumed-strain field.
    base : Element
        The base displacement shell element (any shell on ``patch``).
    quadrature : QuadratureRule
        Quadrature rule for the mixed coupling; use the problem's.
    degree_drop : int, default 1
        Degree reduction of the assumed-strain space.
    """

    def __init__(self, patch: SurfacePatch, base: Element,
                 quadrature: QuadratureRule, degree_drop: int = 1) -> None:
        self._patch = patch
        self._base = base
        self._quadrature = quadrature
        self._degree_drop = degree_drop
        self.num_node_dofs = base.num_node_dofs
        self._material = base.material
        self._cpp_object = _pyck.MixedMembraneStrainShell(
            patch._cpp_object, base._cpp_object, quadrature._cpp_object, degree_drop)

    @property
    def base(self) -> Element:
        """The wrapped base displacement element."""
        return self._base

    def recover_membrane_force(
        self, full_u: npt.NDArray[np.float64], params: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Smooth membrane force ``[n11, n22, n12]`` from the assumed-strain field at
        parametric ``params`` (Q x 2). ``full_u`` is the untruncated solution
        (``ck.solve(problem, full=True)``); free of the membrane-locking oscillation.
        Equivalent to ``ck.Function(full_u, shell, patch, ck.FieldType.TRACTION)``."""
        return np.asarray(self._cpp_object.recover_membrane_force(full_u, params))

    def __repr__(self) -> str:
        return (f"MixedMembraneStrainShell(base={self._base!r}, "
                f"degree_drop={self._degree_drop})")
