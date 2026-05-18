"""Surface patch."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import pyck._pyck as _pyck
from pyck.basis import Basis, BSpline, NURBS
from pyck.geometry.patch import Patch
from pyck.geometry.patch_boundary import PatchBoundary


class SurfacePatch(Patch):
    """Isogeometric surface patch embedded in the 3D physical space.

    Parameters
    ----------
    basis_u : Basis
        B-spline basis in the `u` parametric direction.
    basis_v : Basis
        B-spline basis in the `v` parametric direction.
    cps : npt.ArrayLike
        Control-point matrix in u-fastest order, shape (n_u * n_v, 3).
    name : str, optional
        Human-readable label for the patch (default "patch").
    """

    _cpp_object: _pyck.Patch2d
    _basis_u: Basis
    _basis_v: Basis

    def __init__(
        self, basis_u: Basis, basis_v: Basis, cps: npt.ArrayLike, *,
        name: str = "patch",
    ) -> None:
        cps = np.asarray(cps, dtype=np.float64)
        npts, dim = cps.shape

        if basis_u.num_basis * basis_v.num_basis != npts:
            raise ValueError(
                f"Number of control points ({npts}) must equal "
                f"basis_u.num_basis * basis_v.num_basis ({basis_u.num_basis * basis_v.num_basis})."
            )

        if dim < 3:
            # Ensure 3D embedding by padding zeros if necessary
            pts_3d = np.zeros((npts, 3), dtype=np.float64)
            pts_3d[:, :dim] = cps
            cps = pts_3d

        super().__init__(name=name)
        self._basis_u = basis_u
        self._basis_v = basis_v
        self._cpp_object = _pyck.Patch2d(basis_u._cpp_object, basis_v._cpp_object, cps)

    @classmethod
    def _from_cpp(
        cls, cpp_obj: _pyck.Patch2d, basis_u: Basis, basis_v: Basis, *,
        name: str = "patch",
    ) -> SurfacePatch:
        instance = cls.__new__(cls)
        instance._cpp_object = cpp_obj
        instance._basis_u = basis_u
        instance._basis_v = basis_v
        instance._name = name
        return instance

    # === Properties ==================================================================

    @property
    def tdim(self) -> int: return 2

    @property
    def basis(self) -> list[Basis]: return [self._basis_u, self._basis_v]

    def __repr__(self) -> str:
        return (
            f"SurfacePatch(name='{self.name}', "
            f"basis_u={self._basis_u!r}, basis_v={self._basis_v!r}, "
            f"num_control_pts={self.num_control_pts})"
        )

    # === Refinement ==================================================================

    def insert_knot(self, dir: int, u: float, count: int = 1) -> SurfacePatch:
        """Refine the patch by inserting knot value `u` in parametric direction `dir`."""
        if dir not in (0, 1):
            raise ValueError(f"dir must be 0 or 1, got {dir}")

        old_basis = self._basis_u if dir == 0 else self._basis_v
        new_basis, _ = old_basis.insert_knot(u, count)

        new_cpp = self._cpp_object
        for _ in range(int(count)):
            new_cpp = new_cpp.insert_knot(int(dir), float(u))

        basis_u = new_basis if dir == 0 else self._basis_u
        basis_v = new_basis if dir == 1 else self._basis_v
        return SurfacePatch._from_cpp(new_cpp, basis_u, basis_v, name=self._name)

    def elevate_degree(self, dir: int, count: int = 1) -> SurfacePatch:
        """Elevate the polynomial degree in parametric direction `dir` by `count`."""
        if dir not in (0, 1):
            raise ValueError(f"dir must be 0 or 1, got {dir}")

        old_basis = self._basis_u if dir == 0 else self._basis_v
        new_basis, _ = old_basis.elevate_degree(count)

        new_cpp = self._cpp_object
        for _ in range(int(count)):
            new_cpp = new_cpp.elevate_degree(int(dir))

        basis_u = new_basis if dir == 0 else self._basis_u
        basis_v = new_basis if dir == 1 else self._basis_v
        return SurfacePatch._from_cpp(new_cpp, basis_u, basis_v, name=self._name)

    # === Boundaries ==================================================================

    def boundary(self, pdim: int, at_start: bool) -> PatchBoundary:
        """Return the boundary patch on the given side.

        Parameters
        ----------
        pdim : int
            Parametric direction normal to the edge (0=u, 1=v).
        at_start : bool
            True for the edge at the lower end of that direction, False
            for the upper end.

        Returns
        -------
        PatchBoundary
            The boundary patch.
        """
        return PatchBoundary(self, pdim, at_start)

    # === Class Methods ===============================================================

    @classmethod
    def rectangle(
        cls, l: float, w: float, nu: int, nv: int, deg: int, *,
        cx: float = 0.0, cy: float = 0.0, name: str = "patch",
    ) -> SurfacePatch:
        """Create a flat rectangular patch in the xy-plane.

        Parameters
        ----------
        l : float
            Length in the x direction.
        w : float
            Width in the y direction.
        nu : int
            Number of basis functions in the u (x) direction.
        nv : int
            Number of basis functions in the v (y) direction.
        deg : int
            Polynomial degree for both directions.
        cx : float, optional
            Center x-coordinate (default 0.0).
        cy : float, optional
            Center y-coordinate (default 0.0).
        name : str, optional
            Human-readable label for the patch (default "patch").

        Returns
        -------
        SurfacePatch
            The rectangular patch.
        """
        basis_u = BSpline.clamped_uniform(deg, nu)
        basis_v = BSpline.clamped_uniform(deg, nv)

        x = np.asarray(basis_u._cpp_object.greville_abscissae()) * l - l / 2 + cx
        y = np.asarray(basis_v._cpp_object.greville_abscissae()) * w - w / 2 + cy
        xx, yy = np.meshgrid(x, y)

        cps = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nu * nv)])
        return cls(basis_u, basis_v, cps, name=name)

    @classmethod
    def quarter_cylinder(
        cls, nu: int, nv: int, deg: int, radius: float, height: float, *,
        angle: float = 90.0, name: str = "cylinder",
    ) -> SurfacePatch:
        """Create an exact cylinder arc surface patch using NURBS.

        Parameters
        ----------
        nu : int
            Number of basis functions in the arc direction. Must be >= deg + 1.
        nv : int
            Number of basis functions in the axial direction. Must be >= deg + 1.
        deg : int
            Polynomial degree for both directions. Must be >= 2.
        radius : float
            Radius of the cylinder.
        height : float
            Height of the cylinder.
        angle : float, optional
            Quarter-cylinder angle in degrees (default 90). Must be in (0, 90].
        name : str, optional
            Human-readable label for the patch (default "cylinder").

        Returns
        -------
        SurfacePatch
            The exact cylinder arc surface patch.
        """
        if deg < 2:
            raise ValueError(f"deg must be at least 2, got {deg}.")
        if nu < deg + 1:
            raise ValueError(f"nu must be at least deg + 1 = {deg + 1}, got {nu}.")
        if not (0.0 < angle <= 90.0):
            raise ValueError(f"Angle must be in (0, 90], got {angle}")

        alpha = np.radians(angle)
        basis_v = BSpline.clamped_uniform(deg, nv)
        z = np.asarray(basis_v._cpp_object.greville_abscissae()) * height
        c = radius

        # Build the base p=2 NURBS arc.
        w_base = [1.0, np.cos(alpha / 2), 1.0]
        basis_u_base = NURBS.clamped_uniform(2, 3, w_base)
        ctrl_u_base = np.array([
            [c,                  0.0                  ],
            [c,                  c * np.tan(alpha / 2)],
            [c * np.cos(alpha),  c * np.sin(alpha)    ],
        ])

        cps = np.column_stack([
            np.tile(ctrl_u_base[:, 0], len(z)),
            np.tile(ctrl_u_base[:, 1], len(z)),
            np.repeat(z, len(ctrl_u_base)),
        ])
        patch = cls(basis_u_base, basis_v, cps, name=name)

        if deg > 2:
            patch = patch.elevate_degree(0, deg - 2)
        for u_new in np.arange(1, nu - deg) / (nu - deg):
            patch = patch.insert_knot(0, float(u_new))

        return patch

    @classmethod
    def quarter_sphere(
        cls, nu: int, nv: int, deg: int, radius: float, *,
        name: str = "sphere",
    ) -> SurfacePatch:
        """Create an exact quarter-sphere surface NURBS patch.

        Parameters
        ----------
        nu : int
            Number of basis functions in the u arc direction. Must be >= deg + 1.
        nv : int
            Number of basis functions in the v arc direction. Must be >= deg + 1.
        deg : int
            Polynomial degree for both directions. Must be >= 2.
        radius : float
            Sphere radius.
        name : str, optional
            Human-readable label for the patch (default "sphere").

        Returns
        -------
        SurfacePatch
            The exact quarter-sphere surface patch.
        """
        if deg < 2:
            raise ValueError(f"deg must be at least 2, got {deg}.")
        if nu < deg + 1 or nv < deg + 1:
            raise ValueError(
                f"nu and nv must each be at least deg + 1 = {deg + 1}, "
                f"got nu={nu}, nv={nv}."
            )

        w_base = [1.0, 1.0 / np.sqrt(2.0), 1.0]
        basis_uv_base = NURBS.clamped_uniform(2, 3, w_base)

        R = radius
        cps_flat = np.array([
            [R, 0, 0], [R, 0, R], [0, 0, R],
            [R, R, 0], [R, R, R], [0, 0, R],
            [0, R, 0], [0, R, R], [0, 0, R],
        ], dtype=np.float64)

        patch = cls(basis_uv_base, basis_uv_base, cps_flat, name=name)

        if deg > 2:
            patch = patch.elevate_degree(0, deg - 2).elevate_degree(1, deg - 2)
        for u_new in np.arange(1, nu - deg) / (nu - deg):
            patch = patch.insert_knot(0, float(u_new))
        for v_new in np.arange(1, nv - deg) / (nv - deg):
            patch = patch.insert_knot(1, float(v_new))

        return patch
