"""Single and multi-patch geometric description."""

from pyck.geometry.boundary_patch import BoundaryPatch
from pyck.geometry.patch import Patch
from pyck.geometry.curve_patch import CurvePatch, create_curve_patch, create_line_segment


__all__ = [
    "CurvePatch",
    "create_curve_patch",
    "create_line_segment",
]
