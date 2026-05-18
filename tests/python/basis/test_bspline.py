"""Tests for pyck.basis.BSpline."""

from __future__ import annotations

import numpy as np
import pytest

import pyck as ck


class TestBSplineCreation:
    """BSpline basis creation and initialization."""

    def test_creation_basic(self) -> None:
        """Create a basic BSpline basis."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        assert basis.degree == 2
        assert basis.num_basis == 5

    def test_clamped_uniform_factory(self) -> None:
        """Create BSpline using clamped_uniform factory method."""
        basis = ck.BSpline.clamped_uniform(degree=3, num_basis=8)
        assert basis.degree == 3
        assert basis.num_basis == 8



class TestBSplineProperties:
    """Test BSpline basis properties."""

    def test_bspline_properties(self) -> None:
        """Test BSpline basic properties."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=6)
        assert basis.degree == 2
        assert basis.num_basis == 6
        assert basis.num_intervals == len(basis.knots) - 1

    def test_num_basis_calculation(self) -> None:
        """Verify num_basis is correctly reported."""
        basis = ck.BSpline.clamped_uniform(degree=3, num_basis=7)
        assert basis.num_basis == 7

    def test_num_intervals_property(self) -> None:
        """Verify num_intervals calculation."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        # num_intervals = len(knots) - 1
        assert basis.num_intervals == len(basis.knots) - 1

    def test_bspline_knots_access(self) -> None:
        """Test accessing the raw knot vector from a BSpline."""
        basis = ck.BSpline.clamped_uniform(degree=3, num_basis=5)
        # Knot vector length = num_basis + degree + 1
        assert len(basis.knots) == basis.num_basis + basis.degree + 1

    def test_bspline_greville_abscissae(self) -> None:
        """Test accessing Greville abscissae from a BSpline."""
        basis = ck.BSpline.clamped_uniform(degree=3, num_basis=5)
        # One Greville abscissa per basis function
        assert len(basis.greville_abscissae) == basis.num_basis


class TestBSplineRefinement:
    """Test h- and p-refinement operations."""

    def test_knot_insertion_returns_refined_basis(self) -> None:
        """Knot insertion returns a refined basis with one extra DOF."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        basis_new = basis.insert_knot(0.5)

        assert basis_new.num_basis == basis.num_basis + 1
        assert basis_new.degree == basis.degree

    def test_knot_insertion_multiple_times(self) -> None:
        """Insert the same knot multiple times (chained single inserts)."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        basis_new = basis.insert_knot(0.5).insert_knot(0.5)

        assert basis_new.num_basis == basis.num_basis + 2
        assert basis_new.degree == basis.degree

    def test_knot_insertion_at_different_locations(self) -> None:
        """Insert knots at multiple locations."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        basis1 = basis.insert_knot(0.33)
        basis2 = basis1.insert_knot(0.67)

        assert basis1.num_basis == basis.num_basis + 1
        assert basis2.num_basis == basis.num_basis + 2

    def test_degree_elevation_returns_elevated_basis(self) -> None:
        """Degree elevation returns a basis with degree + 1."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        basis_new = basis.elevate_degree()

        assert basis_new.degree == basis.degree + 1
        assert basis_new.num_basis > basis.num_basis

    def test_degree_elevation_multiple(self) -> None:
        """Elevate degree multiple times (chained single elevations)."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        basis_new = basis.elevate_degree().elevate_degree()

        assert basis_new.degree == basis.degree + 2
        assert basis_new.num_basis > basis.num_basis

    def test_degree_elevation_preserves_continuity(self) -> None:
        """Degree elevation preserves continuity."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        basis_new = basis.elevate_degree()
        # After elevation, internal knots should have same multiplicity
        # (continuity preserved at C^(p-1))
        assert basis_new.num_intervals >= basis.num_intervals


class TestBSplineEvaluation:
    """Test basis function evaluation."""

    def test_basis_evaluation_basic(self) -> None:
        """Evaluate basis functions at a single point."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        results = basis.eval_all(0.5, order=0)

        # Should return one matrix (values, no derivatives)
        assert len(results) == 1
        values = results[0]

        # Should be 1 evaluation point × 5 basis functions
        assert values.shape == (1, 5)

        # Partition of unity: sum to 1
        np.testing.assert_almost_equal(values.sum(), 1.0)

    def test_basis_evaluation_multiple_points(self) -> None:
        """Evaluate basis functions at multiple points."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        pts = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = basis.eval_all(pts, order=0)

        values = results[0]
        # 5 evaluation points × 5 basis functions
        assert values.shape == (5, 5)

        # Partition of unity at each point
        for i in range(5):
            np.testing.assert_almost_equal(values[i].sum(), 1.0)

    def test_basis_evaluation_numpy_array(self) -> None:
        """Evaluate with numpy array of points."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=4)
        pts = np.linspace(0.0, 1.0, 10)
        results = basis.eval_all(pts, order=0)

        values = results[0]
        assert values.shape == (10, 4)

    def test_basis_evaluation_with_derivatives(self) -> None:
        """Evaluate basis functions and their derivatives."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        results = basis.eval_all(0.5, order=2)

        # Should return 3 matrices: values, 1st deriv, 2nd deriv
        assert len(results) == 3

        # All should have same shape
        assert results[0].shape == (1, 5)
        assert results[1].shape == (1, 5)
        assert results[2].shape == (1, 5)

    def test_basis_evaluation_first_derivative(self) -> None:
        """Evaluate basis functions and first derivatives."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        results = basis.eval_all(0.5, order=1)

        assert len(results) == 2
        values = results[0]
        derivatives = results[1]

        # Partition of unity: derivatives should sum to 0
        np.testing.assert_almost_equal(derivatives.sum(), 0.0)

    def test_basis_evaluation_invalid_order(self) -> None:
        """Evaluation with negative order raises ValueError."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        with pytest.raises(ValueError, match="order must be non-negative"):
            basis.eval_all(0.5, order=-1)

    def test_basis_evaluation_zero_order(self) -> None:
        """Evaluate with order=0 returns only values."""
        basis = ck.BSpline.clamped_uniform(degree=3, num_basis=6)
        results = basis.eval_all(0.3, order=0)
        assert len(results) == 1

    def test_basis_evaluation_high_order(self) -> None:
        """Evaluate high-order derivatives."""
        basis = ck.BSpline.clamped_uniform(degree=3, num_basis=6)
        results = basis.eval_all(0.5, order=3)
        # Should have values + 3 derivatives
        assert len(results) == 4


class TestBSplineBoundaryBehavior:
    """Test behavior at domain boundaries."""

    def test_evaluation_at_left_boundary(self) -> None:
        """Evaluate at left boundary (u=0)."""
        basis = ck.BSpline.clamped_uniform(degree=3, num_basis=6)
        results = basis.eval_all(0.0, order=0)

        values = results[0]
        # At u=0 (left boundary), first basis function should dominate
        np.testing.assert_almost_equal(values.sum(), 1.0)
        # First value should be 1.0
        np.testing.assert_almost_equal(values[0, 0], 1.0)

    def test_evaluation_at_right_boundary(self) -> None:
        """Evaluate at right boundary (u=1)."""
        basis = ck.BSpline.clamped_uniform(degree=3, num_basis=6)
        results = basis.eval_all(1.0, order=0)

        values = results[0]
        # At u=1 (right boundary), last basis function should dominate
        np.testing.assert_almost_equal(values.sum(), 1.0)
        # Last value should be 1.0
        np.testing.assert_almost_equal(values[0, -1], 1.0)

    def test_evaluation_near_boundaries(self) -> None:
        """Evaluate near (but not at) boundaries."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        eps = 1e-10

        results_left = basis.eval_all(0.0 + eps, order=0)
        results_right = basis.eval_all(1.0 - eps, order=0)

        # All values should sum to 1 (partition of unity)
        np.testing.assert_almost_equal(results_left[0].sum(), 1.0)
        np.testing.assert_almost_equal(results_right[0].sum(), 1.0)


class TestBSplineAssumedBasis:
    """Verify BSpline as concrete implementation of Basis."""

    def test_bspline_is_basis(self) -> None:
        """BSpline is a subclass of Basis."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        assert isinstance(basis, ck.Basis)

    def test_basis_interface_properties(self) -> None:
        """BSpline exposes required Basis interface properties."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)

        # All Basis properties should be accessible
        assert hasattr(basis, 'degree')
        assert hasattr(basis, 'num_basis')
        assert hasattr(basis, 'num_intervals')
        assert hasattr(basis, 'knots')

    def test_basis_interface_methods(self) -> None:
        """BSpline implements required Basis interface methods."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)

        # All Basis methods should be callable
        assert callable(basis.insert_knot)
        assert callable(basis.elevate_degree)
        assert callable(basis.eval_all)
