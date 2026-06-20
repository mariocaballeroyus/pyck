"""Guo (2020) assumed-membrane-strain Hellinger-Reissner membrane-locking treatment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyck._pyck as _pyck

if TYPE_CHECKING:
    from pyck.assembly.quadrature import QuadratureRule
    from pyck.elements.element import Element
    from pyck.geometry.surface_patch import SurfacePatch


class MembraneAssumedStrainCondition:
    """Assumed-membrane-strain (Guo, Zou & Ruess, IJNME 2020) membrane-locking treatment.

    A Hellinger-Reissner mixed treatment: an independent membrane strain field is
    interpolated on three coarser anisotropic B-spline spaces (degrees ``(p-1, q)``,
    ``(p, q-1)``, ``(p-1, q-1)``; Echter Table 6.2) and kept as **global sparse**
    unknowns — no static condensation. Pair with a normal hierarchic shell element on the
    same patch: the displacement element suppresses its membrane block at the source and
    this condition supplies the membrane through the mixed coupling instead.

    Parameters
    ----------
    patch : SurfacePatch
        The patch carrying the displacement shell.
    element : Element
        The displacement shell element (the same instance passed to the problem).
    quadrature : QuadratureRule
        Quadrature rule for the mixed coupling; use the problem's. The ``element`` must be
        the same instance passed to the problem, so its membrane suppression takes effect.
    degree_drop : int, default 1
        Degree reduction of the assumed-strain space.
    """

    _cpp_object: _pyck.MembraneAssumedStrainCondition2d

    def __init__(self, patch: SurfacePatch, element: Element,
                 quadrature: QuadratureRule, degree_drop: int = 1) -> None:
        self._patch = patch
        self._element = element
        self._quadrature = quadrature
        self._degree_drop = degree_drop
        self._cpp_object = _pyck.MembraneAssumedStrainCondition2d(
            patch._cpp_object, element._cpp_object, quadrature._cpp_object, degree_drop)

    def recover_membrane_force(self, u_full, params):
        """Smooth membrane force ``[n11, n22, n12]`` from the assumed-strain field at
        parametric ``params`` (Q x 2). ``u_full`` is the untruncated solution
        (``ck.solve(problem, full=True)``); free of the membrane-locking oscillation."""
        return self._cpp_object.recover_membrane_force(u_full, params)

    def __repr__(self) -> str:
        return f"MembraneAssumedStrainCondition(degree_drop={self._degree_drop})"
