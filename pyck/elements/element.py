"""Abstract base class for finite element formulations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pyck.geometry.patch import Patch
from pyck.conditions.neumann import NeumannBC


class Element:
    """Abstract base class for finite element formulations."""

    @staticmethod
    def assemble_local_stiffness(
        patch: Patch, 
        quadrature: tuple | None = None
    ) -> npt.NDArray[np.float64]:
        """Assemble the local stiffness matrix."""
        # For now ignore the quadrature as an argument
        # Just build a legendre guadrature based on the patch basis degree
        pass

    @staticmethod
    def assemble_local_load(
        patch: Patch, 
        neumann_bcs: list[NeumannBC], 
        quadrature: tuple | None = None
    ) -> npt.NDArray[np.float64]:
        """Assemble the local load vector."""
        # For now ignore the quadrature as an argument
        # Ignore the Neumann boundary conditions for now too
        # Just build a legendre guadrature based on the patch basis degree
        pass

