"""B-spline surface patch.

A surface is parameterised by two directions (u, v) using a tensor-product
B-spline basis. The mapping from parametric to physical space is:
S(u, v) = sum_i sum_j N_{i,p}(u) * M_{j,q}(v) * P_{i,j}
"""

from __future__ import annotations

import numpy as np

import pyck._pyck as _pyck
from pyck.basis import Basis
from pyck.geometry.patch import Patch
from pyck.geometry.boundary_patch import BoundaryPatch


class SurfacePatch(Patch):
    """Isogeometric surface patch embedded in the 3D physical space.

    Parameters
    ----------
    basis_u : Basis
        B-spline basis in the u direction.
    basis_v : Basis
        B-spline basis in the v direction.
    control_pts : np.ndarray
        Control-point matrix in *u-fastest* order, shape (n_u * n_v, 3).
    name : str, optional
        Human-readable label (default "patch").
    """

    _cpp_object: _pyck.SurfacePatch
    _basis_u: Basis
    _basis_v: Basis

    def __init__(
        self,
        basis_u: Basis,
        basis_v: Basis,
        control_pts: np.ndarray,
        *,
        name: str = "patch",
    ) -> None:
        control_pts = np.asarray(control_pts, dtype=np.float64)
        npts, dim = control_pts.shape
        if dim < 3:
            pts_3d = np.zeros((npts, 3), dtype=np.float64)
            pts_3d[:, :dim] = control_pts
            control_pts = pts_3d

        self._basis_u = basis_u
        self._basis_v = basis_v
        self._name = name
        self._cpp_object = _pyck.SurfacePatch(
            basis_u._cpp_object, basis_v._cpp_object, control_pts
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
        """Wrap an existing C++ `SurfacePatch`."""
        obj = object.__new__(cls)
        obj._cpp_object = cpp_obj
        obj._basis_u = basis_u
        obj._basis_v = basis_v
        obj._name = name
        return obj

    @property
    def tdim(self) -> int:
        """Topological (parametric) dimension — always 2 for a surface."""
        return 2

    @property
    def basis_u(self) -> Basis:
        """B-spline basis in the u direction."""
        return self._basis_u

    @property
    def basis_v(self) -> Basis:
        """B-spline basis in the v direction."""
        return self._basis_v

    def basis(self, direction: int = 0) -> Basis:
        """Return the basis for the given parametric direction (0=u, 1=v). Gordito."""
        return self._basis_u if direction == 0 else self._basis_v

    def boundary(self, side: str) -> BoundaryPatch:
        """Extract a boundary from this patch.

        Parameters
        ----------
        side : "u0", "u1", "v0", "v1"
            Which boundary edge of the parametric domain.

        Returns
        -------
        BoundaryPatch
            The extracted boundary patch.
        """
        mapping = {
            "u0": (0, True),
            "u1": (0, False),
            "v0": (1, True),
            "v1": (1, False),
        }
        if side not in mapping:
            raise ValueError(
                f"side must be one of {list(mapping.keys())}, got '{side}'"
            )
        param_dim, at_start = mapping[side]
        cpp_bp = self._cpp_object.boundary(param_dim, at_start)
        return BoundaryPatch(cpp_bp, parent=self)

    def eval_geometry(self, uv: np.ndarray) -> np.ndarray:
        """Evaluate surface coordinates at parametric values.

        Parameters
        ----------
        uv : np.ndarray, shape (Q, 2)
            Parametric coordinates (u, v) for each evaluation point.

        Returns
        -------
        np.ndarray, shape (Q, 3)
            Physical coordinates.
        """
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        results = np.empty((len(uv), 3), dtype=np.float64)

        kv_u_cpp = self._basis_u.knot_vector._cpp_object
        kv_v_cpp = self._basis_v.knot_vector._cpp_object
        deg_u = self._basis_u.degree
        deg_v = self._basis_v.degree
        num_intervals_v = len(self._basis_v.knots) - 1

        for i, row in enumerate(uv):
            u, v = row
            span_u = kv_u_cpp.find_span(deg_u, float(u))
            span_v = kv_v_cpp.find_span(deg_v, float(v))

            # Flat element index (lexicographical u-fastest)
            element_idx = span_u * num_intervals_v + span_v

            pt = np.array([[u, v]], dtype=np.float64)
            results[i, :] = self._cpp_object.eval_geometry(pt, element_idx)[0, :]

        return results

    def evaluate(
        self, n_u: int = 30, n_v: int = 30
    ) -> np.ndarray:
        """Evaluate the surface on a uniform parametric grid.

        Parameters
        ----------
        n_u, n_v : int
            Number of evaluation points in each direction.

        Returns
        -------
        np.ndarray, shape (n_u * n_v, 3)
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
        return (
            f"SurfacePatch(name='{self.name}', "
            f"basis_u={self._basis_u!r}, basis_v={self._basis_v!r}, "
            f"num_control_pts={self.num_control_pts})"
        )


def create_surface_patch(
    basis_u: Basis,
    basis_v: Basis,
    control_pts: np.ndarray,
    *,
    name: str = "patch",
) -> SurfacePatch:
    """Create a surface patch from two bases and control points.

    Parameters
    ----------
    basis_u, basis_v : Basis
        Univariate B-spline bases for the u and v directions.
    control_pts : np.ndarray
        Control-point coordinates in u-fastest order.
    name : str, optional
        Human-readable label for the patch (default "patch").

    Returns
    -------
    SurfacePatch
        The newly created surface patch.
    """
    return SurfacePatch(basis_u, basis_v, control_pts, name=name)


def create_rectangle(
    basis_u: Basis,
    basis_v: Basis,
    width: float,
    height: float,
    *,
    name: str = "patch",
) -> SurfacePatch:
    """Create a flat rectangular surface patch in the xy-plane.

    Parameters
    ----------
    basis_u, basis_v : Basis
        B-spline bases for u and v directions.
    width : float
        Physical width (x-extent).
    height : float
        Physical height (y-extent).
    name : str, optional
        Human-readable label for the patch (default "patch").

    Returns
    -------
    SurfacePatch
        The rectangular patch.
    """
    cpp = _pyck.rectangle(
        basis_u._cpp_object, basis_v._cpp_object, float(width), float(height)
    )
    return SurfacePatch._from_cpp(cpp, basis_u, basis_v, name=name)
