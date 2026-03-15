"""B-spline surface patch."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck

from pyck.basis import Basis, BSpline
from pyck.geometry.boundary_patch import BoundaryPatch


class SurfacePatch:
    """Isogeometric surface patch embedded in the 3D physical space.

    A surface is parameterised by two directions (u, v) using a tensor-product
    B-spline basis.
    """

    def __init__(
        self,
        basis_u: Basis,
        basis_v: Basis,
        control_pts: npt.ArrayLike,
        *,
        name: str = "patch",
    ) -> None:
        """Initialise a surface patch.

        Parameters
        ----------
        basis_u : Basis
            B-spline basis in the u direction.
        basis_v : Basis
            B-spline basis in the v direction.
        control_pts : ndarray, shape (n_u * n_v, 3)
            Control-point matrix in *u-fastest* order.
        name : str, optional
            Human-readable label (default ``"patch"``).
        """
        if not isinstance(basis_u, Basis) or not isinstance(basis_v, Basis):
            raise TypeError("basis_u and basis_v must be Basis instances")

        control_pts = np.asarray(control_pts, dtype=np.float64)
        npts, dim = control_pts.shape
        if dim < 3:
            pts_3d = np.zeros((npts, 3), dtype=np.float64)
            pts_3d[:, :dim] = control_pts
            control_pts = pts_3d

        self._basis_u: Basis = basis_u
        self._basis_v: Basis = basis_v
        self._name: str = name
        self._cpp_object: _pyck.SurfacePatch = _pyck.SurfacePatch(
            basis_u._cpp_object, basis_v._cpp_object, control_pts
        )
        self._cpts_view: npt.NDArray[np.float64] = np.asarray(
            self._cpp_object.control_pts()
        )

    @classmethod
    def _from_cpp(
        cls,
        cpp_obj: _pyck.SurfacePatch,
        basis_u: Basis,
        basis_v: Basis,
        *,
        name: str = "patch",
    ) -> SurfacePatch:
        """Wrap an existing C++ SurfacePatch (internal use)."""
        obj = object.__new__(cls)
        obj._cpp_object = cpp_obj
        obj._basis_u = basis_u
        obj._basis_v = basis_v
        obj._name = name
        obj._cpts_view = np.asarray(cpp_obj.control_pts())
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def tdim(self) -> int:
        """Topological (parametric) dimension — always 2 for a surface."""
        return 2

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        """Control-point matrix, shape ``(n, 3)`` — zero-copy view into C++."""
        return self._cpts_view

    @property
    def num_control_pts(self) -> int:
        return self._cpp_object.num_control_pts()

    @property
    def basis_u(self) -> Basis:
        """B-spline basis in the u direction."""
        return self._basis_u

    @property
    def basis_v(self) -> Basis:
        """B-spline basis in the v direction."""
        return self._basis_v

    def basis(self, direction: int = 0) -> Basis:
        """Return the basis for the given parametric direction (0=u, 1=v)."""
        return self._basis_u if direction == 0 else self._basis_v

    # ------------------------------------------------------------------
    # Boundary helpers
    # ------------------------------------------------------------------

    def boundary(self, side: str) -> BoundaryPatch:
        """Extract a boundary from this patch.

        Parameters
        ----------
        side : ``"u0"``, ``"u1"``, ``"v0"``, ``"v1"``
            Which boundary edge of the parametric domain.
        """
        mapping = {
            "u0": (0, True),
            "u1": (0, False),
            "v0": (1, True),
            "v1": (1, False),
        }
        if side not in mapping:
            raise ValueError(
                f"side must be one of {list(mapping)}, got '{side}'"
            )
        param_dim, at_start = mapping[side]
        cpp_bp = self._cpp_object.boundary(param_dim, at_start)
        return BoundaryPatch(cpp_bp, parent=self)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def eval_geometry(self, uv: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Evaluate surface coordinates at parametric values.

        Parameters
        ----------
        uv : array_like, shape (Q, 2)
            Parametric coordinates.

        Returns
        -------
        ndarray, shape (Q, 3)
        """
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        # Find which element each point belongs to and evaluate
        results = []
        for row in uv:
            u, v = row
            span_u = self._basis_u.knot_vector._cpp_object.find_span(
                self._basis_u.degree, float(u)
            )
            span_v = self._basis_v.knot_vector._cpp_object.find_span(
                self._basis_v.degree, float(v)
            )
            intervals_v = len(self._basis_v.knots) - 1
            flat = span_u * intervals_v + span_v
            pt = np.array([[u, v]], dtype=np.float64)
            results.append(np.asarray(self._cpp_object.eval_geometry(pt, flat)))
        return np.vstack(results)

    def evaluate(
        self, n_u: int = 30, n_v: int = 30
    ) -> npt.NDArray[np.float64]:
        """Evaluate the surface on a uniform parametric grid.

        Parameters
        ----------
        n_u, n_v : int
            Number of evaluation points in each direction.

        Returns
        -------
        ndarray, shape (n_u * n_v, 3)
            Physical coordinates, ordered u-fastest.
        """
        knots_u = self._basis_u.knots
        knots_v = self._basis_v.knots
        u = np.linspace(float(knots_u[0]), float(knots_u[-1]), n_u)
        v = np.linspace(float(knots_v[0]), float(knots_v[-1]), n_v)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        uv = np.column_stack([uu.ravel(), vv.ravel()])
        return self.eval_geometry(uv)

    def __repr__(self) -> str:
        n = self.num_control_pts
        return (
            f"SurfacePatch(name='{self._name}', "
            f"basis_u={self._basis_u!r}, basis_v={self._basis_v!r}, "
            f"control_points={n})"
        )


# === Factory functions ===========================================================


def create_surface_patch(
    basis_u: BSpline,
    basis_v: BSpline,
    control_pts: npt.ArrayLike,
    *,
    name: str = "patch",
) -> SurfacePatch:
    """Create a surface patch from two bases and control points.

    Parameters
    ----------
    basis_u, basis_v : BSpline
        Univariate B-spline bases for the u and v directions.
    control_pts : ndarray, shape (n_u * n_v, 3)
        Control-point coordinates in u-fastest order.
    name : str, optional
        Human-readable label.

    Returns
    -------
    SurfacePatch
    """
    return SurfacePatch(basis_u, basis_v, control_pts, name=name)


def create_rectangle(
    basis_u: BSpline,
    basis_v: BSpline,
    width: float,
    height: float,
    *,
    name: str = "patch",
) -> SurfacePatch:
    """Create a flat rectangular surface patch in the xy-plane.

    Parameters
    ----------
    basis_u, basis_v : BSpline
        B-spline bases for u and v directions.
    width : float
        Physical width (x-extent).
    height : float
        Physical height (y-extent).
    name : str, optional
        Human-readable label.

    Returns
    -------
    SurfacePatch
    """
    if not isinstance(basis_u, BSpline) or not isinstance(basis_v, BSpline):
        raise TypeError("basis_u and basis_v must be BSpline instances")

    cpp = _pyck.rectangle(
        basis_u._cpp_object, basis_v._cpp_object, float(width), float(height)
    )
    return SurfacePatch._from_cpp(cpp, basis_u, basis_v, name=name)
