"""Tests for refinement workflows: h- and p-refinement."""

from __future__ import annotations

import numpy as np
import pytest

import pyck as ck


class TestHRefinement:
    """Test h-refinement (knot insertion) workflows."""

    def test_single_h_refinement(self) -> None:
        """Perform single knot insertion."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=4)
        basis_h = basis.insert_knot(0.5)

        assert basis_h.num_basis == basis.num_basis + 1
        assert basis_h.degree == basis.degree

    def test_sequential_h_refinement(self) -> None:
        """Perform successive h-refinement steps."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=4)
        initial_basis = basis.num_basis

        # Insert knot three times at different locations
        basis1 = basis.insert_knot(0.25)
        basis2 = basis1.insert_knot(0.5)
        basis3 = basis2.insert_knot(0.75)

        assert basis1.num_basis == initial_basis + 1
        assert basis2.num_basis == initial_basis + 2
        assert basis3.num_basis == initial_basis + 3
        assert basis3.degree == basis.degree

    def test_h_refinement_preserves_degree(self) -> None:
        """h-refinement does not change polynomial degree."""
        basis = ck.BSpline.clamped_uniform(degree=3, num_basis=6)
        original_degree = basis.degree

        for _ in range(5):
            basis = basis.insert_knot(0.5)

        assert basis.degree == original_degree

    def test_h_refinement_at_knot_location(self) -> None:
        """Insert knot at location where it already exists."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        # Insert at 0.5 (might already be a knot)
        basis1 = basis.insert_knot(0.5)
        basis2 = basis1.insert_knot(0.5)

        # Both operations should increase num_basis
        assert basis1.num_basis >= basis.num_basis
        assert basis2.num_basis >= basis1.num_basis

    def test_uniform_h_refinement_sequence(self) -> None:
        """Uniform h-refinement by repeated midpoint insertion."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=3)
        num_refinements = 3

        for _ in range(num_refinements):
            # Insert at each interval midpoint
            # This is a simplified version - in reality you'd need to track intervals
            basis = basis.insert_knot(0.5)

        assert basis.num_basis >= 3 + num_refinements


class TestPRefinement:
    """Test p-refinement (degree elevation) workflows."""

    def test_single_p_refinement(self) -> None:
        """Perform single degree elevation."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        basis_p = basis.elevate_degree()

        assert basis_p.degree == basis.degree + 1
        assert basis_p.num_basis > basis.num_basis

    def test_sequential_p_refinement(self) -> None:
        """Perform successive degree elevations."""
        basis = ck.BSpline.clamped_uniform(degree=1, num_basis=4)
        initial_degree = basis.degree

        # Elevate degree three times
        basis1 = basis.elevate_degree()
        basis2 = basis1.elevate_degree()
        basis3 = basis2.elevate_degree()

        assert basis1.degree == initial_degree + 1
        assert basis2.degree == initial_degree + 2
        assert basis3.degree == initial_degree + 3

    def test_p_refinement_increases_basis_functions(self) -> None:
        """Degree elevation increases the number of basis functions."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        basis_p = basis.elevate_degree()

        assert basis_p.num_basis > basis.num_basis

    def test_p_refinement_monotonic_increase(self) -> None:
        """Each p-refinement increases degree and num_basis."""
        basis = ck.BSpline.clamped_uniform(degree=1, num_basis=6)

        prev_degree = basis.degree
        prev_basis = basis.num_basis

        for _ in range(4):
            basis = basis.elevate_degree()
            assert basis.degree == prev_degree + 1
            assert basis.num_basis >= prev_basis
            prev_degree = basis.degree
            prev_basis = basis.num_basis

    def test_p_refinement_from_linear(self) -> None:
        """Elevate from linear to cubic."""
        basis = ck.BSpline.clamped_uniform(degree=1, num_basis=4)
        basis_quad = basis.elevate_degree()
        basis_cubic = basis_quad.elevate_degree()

        assert basis.degree == 1
        assert basis_quad.degree == 2
        assert basis_cubic.degree == 3


class TestHPRefinementCombined:
    """Test combined h- and p-refinement workflows."""

    def test_hp_refinement_h_then_p(self) -> None:
        """Perform h-refinement followed by p-refinement."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        initial_degree = basis.degree
        initial_basis = basis.num_basis

        # h-refine
        basis_h = basis.insert_knot(0.5)
        assert basis_h.num_basis == initial_basis + 1
        assert basis_h.degree == initial_degree

        # p-refine
        basis_hp = basis_h.elevate_degree()
        assert basis_hp.degree == initial_degree + 1
        assert basis_hp.num_basis > basis_h.num_basis

    def test_hp_refinement_p_then_h(self) -> None:
        """Perform p-refinement followed by h-refinement."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)
        initial_degree = basis.degree
        initial_basis = basis.num_basis

        # p-refine
        basis_p = basis.elevate_degree()
        assert basis_p.degree == initial_degree + 1

        # h-refine
        basis_ph = basis_p.insert_knot(0.5)
        assert basis_ph.degree == initial_degree + 1
        assert basis_ph.num_basis > basis_p.num_basis

    def test_interleaved_hp_refinement(self) -> None:
        """Alternate between h- and p-refinement."""
        basis = ck.BSpline.clamped_uniform(degree=1, num_basis=3)

        # h-refine
        basis = basis.insert_knot(0.5)
        h1_basis = basis.num_basis

        # p-refine
        basis = basis.elevate_degree()
        assert basis.degree == 2

        # h-refine again
        basis = basis.insert_knot(0.25)
        assert basis.num_basis > h1_basis

        # p-refine again
        basis = basis.elevate_degree()
        assert basis.degree == 3

    def test_hp_refinement_evaluation_consistency(self) -> None:
        """After hp-refinement, basis still satisfies partition of unity."""
        basis = ck.BSpline.clamped_uniform(degree=2, num_basis=5)

        # h-refine
        basis = basis.insert_knot(0.5)

        # p-refine
        basis = basis.elevate_degree()

        # Evaluate at multiple points
        pts = np.linspace(0.0, 1.0, 10)
        results = basis.eval_all(pts, order=0)
        values = results[0]

        # Partition of unity should still hold
        sums = values.sum(axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(10))


