"""Postprocessing: spline sampling and per-element integration.

Helpers for evaluating spline fields at arbitrary parametric points
(``eval_shape_at``, ``eval_geometry_at``) and integrating user-defined
functionals over a patch (``integrate_on_patch``).
"""

from pyck.postprocessing.evaluation import (
    eval_shape_at,
    eval_geometry_at,
    evaluate_field,
)
from pyck.postprocessing.integrate import integrate_on_patch

__all__ = [
    "eval_shape_at",
    "eval_geometry_at",
    "evaluate_field",
    "integrate_on_patch",
]
