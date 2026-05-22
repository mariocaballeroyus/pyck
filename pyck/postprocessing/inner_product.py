"""L² inner product of two parametric-space fields over a patch."""

from __future__ import annotations

import numpy as np

import pyck._pyck as _pyck
from pyck.assembly.quadrature import QuadratureRule
from pyck.postprocessing.function import _Field


def inner_product(f: _Field, g: _Field, quadrature: QuadratureRule) -> float:
    """L² inner product ⟨f, g⟩ = ∫_Ω (f · g) dΩ on the patch ``f.patch``.

    Both arguments must be :class:`Function`-like objects (a Function, a bare
    callable already wrapped via arithmetic, or any composite). The integration
    domain is taken from ``f.patch``; ``g.patch`` must match.

    Parameters
    ----------
    f, g : Function or composite
        Fields with shape ``(Q, k)`` per parametric query point.
    quadrature : QuadratureRule
        Reference quadrature rule, mapped per element.

    Returns
    -------
    float
        Scalar inner product. For an L² norm use ``sqrt(inner_product(f, f, q))``.
    """
    patch = f.patch
    if g.patch is not patch:
        raise ValueError("inner_product: f and g must live on the same patch.")

    params, _dV = _pyck.eval_quadrature_data(patch._cpp_object, quadrature._cpp_object)
    f_vals = np.ascontiguousarray(f(params), dtype=np.float64)
    g_vals = np.ascontiguousarray(g(params), dtype=np.float64)
    return float(_pyck.inner_product(
        patch._cpp_object, quadrature._cpp_object, f_vals, g_vals))
