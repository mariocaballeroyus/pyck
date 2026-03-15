"""Single and multi-patch geometric description."""

from pyck.geometry.boundary_patch import BoundaryPatch
from pyck.geometry.patch import Patch
from pyck.geometry.curve_patch import CurvePatch, create_curve_patch, create_line_segment
from pyck.geometry.surface_patch import SurfacePatch, create_surface_patch, create_rectangle


__all__ = [
    "CurvePatch",
    "create_curve_patch",
    "create_line_segment",
    "SurfacePatch",
    "create_surface_patch",
    "create_rectangle",
]
