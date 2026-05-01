from __future__ import annotations
import collections.abc
import enum
import numpy
import numpy.typing
import typing

__all__: list[str] = ['BSpline', 'Basis', 'BeamEulerBernoulli1p', 'BeamTimoshenko1p', 'BeamTimoshenko2p', 'BoundaryField', 'PatchBoundary1d', 'PatchBoundary2d', 'Condition', 'Constraint', 'CurvePatch', 'DirectConstraint', 'DofMapper1d', 'DofMapper2d', 'Element1d', 'Element2d', 'GaussLegendre', 'GaussLegendre2d', 'GaussLegendre3d', 'KnotVector', 'LagrangeMultiplierCondition2d', 'LinearConstraint', 'LinearElasticProblem1d', 'LinearElasticProblem2d', 'LoadCondition1d', 'LoadCondition2d', 'Material1d', 'Material2d', 'NitscheCondition2d', 'NormalBendingMoment', 'NormalRotation', 'NormalTransverseShear', 'Patch1d', 'Patch2d', 'PenaltyCondition2d', 'PlaneStress2d', 'PlateKirchhoffLove1p', 'PlateReissnerMindlin1p', 'PlateReissnerMindlin3p', 'PlateReissnerMindlinDispl3p', 'PlateReissnerMindlinDispl2p', 'QuadratureRule1d', 'QuadratureRule2d', 'QuadratureRule3d', 'SlenderBeam1d', 'SurfacePatch', 'TangentialRotation', 'TransverseDisplacement', 'TwistingMoment', 'line_segment', 'rectangle', 'tensor_product', 'tensor_product_1d', 'tensor_product_2d', 'eval_shape_at', 'eval_geometry_at']

class BoundaryField:
    pass

class TransverseDisplacement(BoundaryField):
    def __init__(self) -> None:
        ...

class NormalRotation(BoundaryField):
    def __init__(self) -> None:
        ...

class TangentialRotation(BoundaryField):
    def __init__(self) -> None:
        ...

class NormalTransverseShear(BoundaryField):
    def __init__(self) -> None:
        ...

class NormalBendingMoment(BoundaryField):
    def __init__(self) -> None:
        ...

class TwistingMoment(BoundaryField):
    def __init__(self) -> None:
        ...

class BSpline(Basis):
    def __init__(self, degree: typing.SupportsInt | typing.SupportsIndex, knot_vector: KnotVector) -> None:
        ...
    def knot_vector(self) -> KnotVector:
        ...

class Basis:
    def degree(self) -> int:
        ...
    def eval(self, u: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], span: typing.SupportsInt | typing.SupportsIndex, order: typing.SupportsInt | typing.SupportsIndex = 0) -> list[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]]:
        ...
    def eval_all(self, u: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], order: typing.SupportsInt | typing.SupportsIndex = 0) -> list[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]]:
        """
        Evaluate all basis functions and derivatives (vector of matrices).
        """
    def find_span(self, u: typing.SupportsFloat | typing.SupportsIndex) -> int:
        ...
    def num_basis(self) -> int:
        ...
    def num_intervals(self) -> int:
        ...

class BeamEulerBernoulli1p(Element1d):
    def __init__(self, material: SlenderBeam1d) -> None:
        ...

class BeamTimoshenko1p(Element1d):
    def __init__(self, material: SlenderBeam1d) -> None:
        ...

class BeamTimoshenko2p(Element1d):
    def __init__(self, material: SlenderBeam1d) -> None:
        ...

class PatchBoundary1d(Patch1d):
    def at_start(self) -> bool:
        ...
    def param_dim(self) -> int:
        ...

class PatchBoundary2d(Patch2d):
    def at_start(self) -> bool:
        ...
    def param_dim(self) -> int:
        ...

class Condition:
    pass

class Constraint:
    pass

class CurvePatch(Patch1d):
    def __init__(self, basis: Basis, control_points: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 3]"]) -> None:
        ...
    def eval_basis_functions(self, params: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], span: typing.SupportsInt | typing.SupportsIndex, order: typing.SupportsInt | typing.SupportsIndex = 0) -> list[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]]:
        ...
    def eval_geometry(self, params: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], span: typing.SupportsInt | typing.SupportsIndex) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 3]"]:
        ...
    def eval_shape_functions(self, params: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], span: typing.SupportsInt | typing.SupportsIndex, order: typing.SupportsInt | typing.SupportsIndex = 0) -> tuple[list[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]]:
        ...

class DirectConstraint(Constraint):
    @typing.overload
    def __init__(self, dofs: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], values: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    @typing.overload
    def __init__(self, dofs: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], value: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        ...
    def apply(self, stiffness: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], load: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> None:
        ...
    def dofs(self) -> list[int]:
        ...
    def values(self) -> list[float]:
        ...

class DofMapper1d:
    def degree(self) -> typing.Annotated[list[int], "FixedSize(1)"]:
        ...
    def get_layer_dofs(self, param_dim: typing.SupportsInt | typing.SupportsIndex, at_start: bool, layer_idx: typing.SupportsInt | typing.SupportsIndex) -> list[int]:
        ...
    def get_element_dofs(self, elem_idx: typing.SupportsInt | typing.SupportsIndex) -> list[int]:
        ...
    def num_basis(self) -> typing.Annotated[list[int], "FixedSize(1)"]:
        ...

class DofMapper2d:
    def degree(self) -> typing.Annotated[list[int], "FixedSize(2)"]:
        ...
    def get_layer_dofs(self, param_dim: typing.SupportsInt | typing.SupportsIndex, at_start: bool, layer_idx: typing.SupportsInt | typing.SupportsIndex) -> list[int]:
        ...
    def get_element_dofs(self, elem_idx: typing.SupportsInt | typing.SupportsIndex) -> list[int]:
        ...
    def num_basis(self) -> typing.Annotated[list[int], "FixedSize(2)"]:
        ...

class Element1d:
    def num_node_dofs(self) -> int:
        ...
    def min_order(self) -> int:
        ...
    def displacement_shape_matrix(self, shape_derivs: list[numpy.typing.NDArray[numpy.float64]]) -> numpy.typing.NDArray[numpy.float64]:
        ...
    def rotation_shape_matrix(self, shape_derivs: list[numpy.typing.NDArray[numpy.float64]]) -> numpy.typing.NDArray[numpy.float64]:
        ...
    def displacement_dof_index(self) -> int:
        ...
    def rotation_dof_index(self) -> int:
        ...
    def strain_displacement_matrix(self, shape_derivs: list[numpy.typing.NDArray[numpy.float64]]) -> numpy.typing.NDArray[numpy.float64]:
        ...

class Element2d:
    def num_node_dofs(self) -> int:
        ...
    def min_order(self) -> int:
        ...
    def displacement_shape_matrix(self, shape_derivs: list[numpy.typing.NDArray[numpy.float64]]) -> numpy.typing.NDArray[numpy.float64]:
        ...
    def rotation_shape_matrix(self, shape_derivs: list[numpy.typing.NDArray[numpy.float64]]) -> numpy.typing.NDArray[numpy.float64]:
        ...
    def displacement_dof_index(self) -> int:
        ...
    def rotation_dof_indices(self) -> tuple[int, int]:
        ...
    def strain_displacement_matrix(self, shape_derivs: list[numpy.typing.NDArray[numpy.float64]]) -> numpy.typing.NDArray[numpy.float64]:
        ...

class GaussLegendre(QuadratureRule1d):
    def __init__(self, num_pts: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

class GaussLegendre2d(QuadratureRule2d):
    def __init__(self, num_pts: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

class GaussLegendre3d(QuadratureRule3d):
    def __init__(self, num_pts: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...

class KnotVector:
    @staticmethod
    def clamped_uniform(degree: typing.SupportsInt | typing.SupportsIndex, num_basis: typing.SupportsInt | typing.SupportsIndex) -> KnotVector:
        ...
    def __init__(self, knots: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    def data(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    def find_span(self, degree: typing.SupportsInt | typing.SupportsIndex, u: typing.SupportsFloat | typing.SupportsIndex) -> int:
        ...
    def span_bounds(self, span: typing.SupportsInt | typing.SupportsIndex) -> tuple[float, float]:
        ...

class LinearConstraint(Constraint):
    def __init__(self, slaves: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], masters: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, n]"], weights: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], constant: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        ...
    def apply(self, stiffness: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], load: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> None:
        ...
    def constant(self) -> float:
        ...
    def masters(self) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, n]"]:
        ...
    def slaves(self) -> list[int]:
        ...
    def weights(self) -> list[float]:
        ...

class LinearElasticProblem1d:
    def __init__(self, patch: Patch1d, element: Element1d, quadrature: QuadratureRule1d) -> None:
        ...
    def add_condition(self, condition: Condition) -> None:
        ...
    @typing.overload
    def add_constraint(self, constraint: Constraint) -> None:
        ...
    @typing.overload
    def add_constraint(self, direct_constraint: DirectConstraint) -> None:
        ...
    def add_direct_constraint(self, constraint: DirectConstraint) -> None:
        ...
    def assemble(self) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]]:
        """
        Assemble global stiffness matrix and load vector. Returns (K, F).
        """

class LinearElasticProblem2d:
    def __init__(self, patch: Patch2d, element: Element2d, quadrature: QuadratureRule2d) -> None:
        ...
    def add_condition(self, condition: Condition) -> None:
        ...
    @typing.overload
    def add_constraint(self, constraint: Constraint) -> None:
        ...
    @typing.overload
    def add_constraint(self, direct_constraint: DirectConstraint) -> None:
        ...
    def add_direct_constraint(self, constraint: DirectConstraint) -> None:
        ...
    def assemble(self) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]]:
        """
        Assemble global stiffness matrix and load vector. Returns (K, F).
        """

class LoadCondition1d(Condition):
    def __init__(self, patch: Patch1d, element: Element1d, quadrature: QuadratureRule1d, load_values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> None:
        ...

class LoadCondition2d(Condition):
    def __init__(self, patch: Patch2d, element: Element2d, quadrature: QuadratureRule2d, load_values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> None:
        ...

class PenaltyCondition2d(Condition):
    def __init__(
        self,
        boundary: PatchBoundary2d,
        element: Element2d,
        quadrature: QuadratureRule1d,
    ) -> None:
        ...
    def add(
        self,
        field: BoundaryField,
        penalty: typing.SupportsFloat | typing.SupportsIndex,
        value: typing.SupportsFloat | typing.SupportsIndex = 0.0,
    ) -> PenaltyCondition2d:
        ...

class LagrangeMultiplierCondition2d(Condition):
    def __init__(
        self,
        boundary: PatchBoundary2d,
        element: Element2d,
        quadrature: QuadratureRule1d,
    ) -> None:
        ...
    def add(
        self,
        field: BoundaryField,
        value: typing.SupportsFloat | typing.SupportsIndex = 0.0,
    ) -> LagrangeMultiplierCondition2d:
        ...

class NitscheCondition2d(Condition):
    def __init__(
        self,
        boundary: PatchBoundary2d,
        element: Element2d,
        quadrature: QuadratureRule1d,
        thickness: typing.SupportsFloat | typing.SupportsIndex,
    ) -> None:
        ...
    def add(
        self,
        displacement_field: BoundaryField,
        traction_field: BoundaryField,
        penalty: typing.SupportsFloat | typing.SupportsIndex,
        value: typing.SupportsFloat | typing.SupportsIndex = 0.0,
    ) -> NitscheCondition2d:
        ...

class Material1d:
    def bending_stiffness(self) -> float:
        ...
    def name(self) -> str:
        ...
    def poisson_ratio(self) -> float:
        ...
    def shear_modulus(self) -> float:
        ...
    def shear_stiffness(self) -> float:
        ...
    def youngs_modulus(self) -> float:
        ...

class Material2d:
    def bending_stiffness(self) -> float:
        ...
    def name(self) -> str:
        ...
    def poisson_ratio(self) -> float:
        ...
    def shear_modulus(self) -> float:
        ...
    def shear_stiffness(self) -> float:
        ...
    def youngs_modulus(self) -> float:
        ...

class Patch1d:
    @staticmethod
    def eval_physical_points(*args, **kwargs) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 3]"]:
        """
        Coordinate mapping of all quadrature points in active elements.
        """
    def active_control_pts(self, spans: typing.Annotated[collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], "FixedSize(1)"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 3]"]:
        ...
    def boundary(self, param_dim: typing.SupportsInt | typing.SupportsIndex, at_start: bool) -> PatchBoundary1d:
        """
        Extract a boundary face of this patch.
        """
    def layer_dofs(self, param_dim: typing.SupportsInt | typing.SupportsIndex, at_start: bool, layer_idx: int = 0) -> list[int]:
        ...
    def control_pts(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 3]"]:
        ...
    def dof_mapper(self) -> DofMapper1d:
        """
        Return the DofMapper of this patch.
        """
    def gdim(self) -> int:
        ...
    def num_control_pts(self) -> int:
        ...
    def tdim(self) -> int:
        ...

class Patch2d:
    @staticmethod
    def eval_physical_points(*args, **kwargs) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 3]"]:
        """
        Coordinate mapping of all quadrature points in active elements.
        """
    def active_control_pts(self, spans: typing.Annotated[collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], "FixedSize(2)"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 3]"]:
        ...
    def boundary(self, param_dim: typing.SupportsInt | typing.SupportsIndex, at_start: bool) -> PatchBoundary2d:
        """
        Extract a boundary face of this patch.
        """
    def layer_dofs(self, param_dim: typing.SupportsInt | typing.SupportsIndex, at_start: bool, layer_idx: int = 0) -> list[int]:
        ...
    def control_pts(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 3]"]:
        ...
    def dof_mapper(self) -> DofMapper2d:
        """
        Return the DofMapper of this patch.
        """
    def gdim(self) -> int:
        ...
    def num_control_pts(self) -> int:
        ...
    def tdim(self) -> int:
        ...

class PlaneStress2d(Material2d):
    def __init__(self, E: typing.SupportsFloat | typing.SupportsIndex, nu: typing.SupportsFloat | typing.SupportsIndex = 0.3, h: typing.SupportsFloat | typing.SupportsIndex = 1.0, k: typing.SupportsFloat | typing.SupportsIndex = 0.8333333333333334) -> None:
        ...
    def shear_correction_factor(self) -> float:
        ...
    def thickness(self) -> float:
        ...

class PlateKirchhoffLove1p(Element2d):
    def __init__(self, material: PlaneStress2d) -> None:
        ...

class PlateReissnerMindlin1p(Element2d):
    def __init__(self, material: PlaneStress2d) -> None:
        ...

class PlateReissnerMindlin3p(Element2d):
    def __init__(self, material: PlaneStress2d) -> None:
        ...
    def rotation_shape_matrix(self, shape_derivs: list[numpy.typing.NDArray[numpy.float64]]) -> numpy.typing.NDArray[numpy.float64]:
        ...

class PlateReissnerMindlinDispl3p(Element2d):
    def __init__(self, material: PlaneStress2d) -> None:
        ...

class PlateReissnerMindlinDispl2p(Element2d):
    def __init__(self, material: PlaneStress2d) -> None:
        ...

class QuadratureRule1d:
    def num_points(self) -> int:
        ...
    def points(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def weights(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...

class QuadratureRule2d:
    @typing.overload
    def __init__(self, rule: QuadratureRule1d) -> None:
        ...
    @typing.overload
    def __init__(self, rules: typing.Annotated[collections.abc.Sequence[QuadratureRule1d], "FixedSize(2)"]) -> None:
        ...
    def num_points(self) -> int:
        ...
    def points(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 2]"]:
        ...
    def weights(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...

class QuadratureRule3d:
    @typing.overload
    def __init__(self, rule: QuadratureRule1d) -> None:
        ...
    @typing.overload
    def __init__(self, rules: typing.Annotated[collections.abc.Sequence[QuadratureRule1d], "FixedSize(3)"]) -> None:
        ...
    def num_points(self) -> int:
        ...
    def points(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 3]"]:
        ...
    def weights(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...

class SlenderBeam1d(Material1d):
    def __init__(self, E: typing.SupportsFloat | typing.SupportsIndex, nu: typing.SupportsFloat | typing.SupportsIndex, A: typing.SupportsFloat | typing.SupportsIndex, I: typing.SupportsFloat | typing.SupportsIndex, k: typing.SupportsFloat | typing.SupportsIndex = 0.8333333333333334) -> None:
        ...
    def moment_inertia(self) -> float:
        ...
    def section_area(self) -> float:
        ...
    def shear_coefficient(self) -> float:
        ...

class SurfacePatch(Patch2d):
    def __init__(self, basis_u: Basis, basis_v: Basis, control_points: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 3]"]) -> None:
        ...
    def eval_basis_functions(self, params: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 2]"], span: typing.SupportsInt | typing.SupportsIndex, order: typing.SupportsInt | typing.SupportsIndex = 0) -> list[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]]:
        ...
    def eval_geometry(self, params: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 2]"], span: typing.SupportsInt | typing.SupportsIndex) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 3]"]:
        ...
    def eval_shape_functions(self, params: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 2]"], span: typing.SupportsInt | typing.SupportsIndex, order: typing.SupportsInt | typing.SupportsIndex = 0) -> tuple[list[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]]:
        ...

def line_segment(basis: Basis, length: typing.SupportsFloat | typing.SupportsIndex) -> CurvePatch:
    """
    Create a straight line segment C(u) = (L*u, 0, 0) along the x-axis.
    """

def rectangle(basis_u: Basis, basis_v: Basis, width: typing.SupportsFloat | typing.SupportsIndex, height: typing.SupportsFloat | typing.SupportsIndex) -> SurfacePatch:
    """
    Create a flat rectangular surface patch in the xy-plane.
    """

@typing.overload
def tensor_product(rules: typing.Annotated[collections.abc.Sequence[QuadratureRule1d], "FixedSize(2)"]) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 2]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]]:
    """
    Form the tensor product of 2 one-dimensional quadrature rules.
    """

@typing.overload
def tensor_product(rules: typing.Annotated[collections.abc.Sequence[QuadratureRule1d], "FixedSize(3)"]) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 3]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]]:
    """
    Form the tensor product of 3 one-dimensional quadrature rules.
    """

def tensor_product_1d(rule: QuadratureRule1d) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]]:
    """
    Return (points, weights) for a 1D tensor product (identity).
    """

def tensor_product_2d(rule_u: QuadratureRule1d, rule_v: QuadratureRule1d) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 2]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]]:
    """
    Form the 2D tensor product of two 1D quadrature rules.
    """

def eval_shape_at(patch: Patch1d | Patch2d, params: numpy.typing.NDArray[numpy.float64], order: int = 0) -> list[numpy.typing.NDArray[numpy.float64]]:
    ...

def eval_geometry_at(patch: Patch1d | Patch2d, params: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
    ...
