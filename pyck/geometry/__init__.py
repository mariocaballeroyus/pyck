"""Single and multi-patch geometric description."""

from pyck.geometry.patch_boundary import PatchBoundary
from pyck.geometry.patch import Patch
from pyck.geometry.curve import CurvePatch, create_curve_patch, create_line_segment
from pyck.geometry.surface import SurfacePatch, create_surface_patch, create_rectangle
from pyck.geometry.evaluation import eval_shape_at, eval_geometry_at


__all__ = [
    "Patch",
    "PatchBoundary",
    "CurvePatch",
    "create_curve_patch",
    "create_line_segment",
    "SurfacePatch",
    "create_surface_patch",
    "create_rectangle",
    "eval_shape_at",
    "eval_geometry_at",
]
