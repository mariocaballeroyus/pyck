"""Python factory functions for boundary constraints.

These functions inspect the element formulation and return the appropriate
low-level :class:`Constraint` object (:class:`DirectConstraint` or
:class:`LinearConstraint`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyck.constraints.direct_constraint import DirectConstraint
from pyck.constraints.constraint import LinearConstraint

if TYPE_CHECKING:
    from pyck.elements.element import Element
    from pyck.geometry.curve import CurvePatch
    from pyck.geometry.surface import SurfacePatch


def _resolve_side(at: str) -> str:
    side = {"left": "start", "right": "end"}.get(at, at)
    if side not in ("start", "end"):
        raise ValueError(f"'at' must be 'left'/'start' or 'right'/'end', got {at!r}")
    return side


# ======================================================================
# Surface-patch constraint factories (Kirchhoff-Love plate, etc.)
# ======================================================================


