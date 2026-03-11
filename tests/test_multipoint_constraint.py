import pytest
import numpy as np
import pyck

def test_multipoint_constraint():
    patch = pyck.create_line_segment(pyck.create_bspline(2, pyck.create_clamped_uniform(2, 5)), 1.0)
    element = pyck.create_euler_bernoulli_beam(E=1e5, A=1.0, I=1/12)
    le = pyck.create_linear_elastic_problem(patch, element)
    
    # Simple setup, we don't care about the physics, just the assembled matrix constraint
    le.set_quadrature(pyck.create_gauss_legendre(3))
    
    # Fix the first node's displacement and rotation so the system is well-posed (non-singular)
    le.add_constraint(pyck.DirichletBC([0], 0.0)) # displacement at x=0
    le.add_constraint(pyck.DirichletBC([1], 0.0)) # rotation at x=0
    
    # We add a multipoint constraint: DOF 4 = 2.0 * DOF 2 - 0.5 * DOF 3 + 1.5
    mpc = pyck.MultipointConstraint(slave=4, masters_weights=[(2, 2.0), (3, -0.5)], constant=1.5)
    le.add_constraint(mpc)
    
    K, F = le.assemble()
    
    # Solve system exactly
    # We must eliminate the zero rows/cols explicitly, or just use a solver that can handle them?
    # K contains 1s on diagonal and 0s elsewhere for the Dirichlet and slave DOFs.
    # We just solve K u = F
    u = np.linalg.solve(K, F)
    
    np.testing.assert_allclose(u[4], 2.0 * u[2] - 0.5 * u[3] + 1.5)
    
    # Check Symmetry
    np.testing.assert_allclose(K, K.T, atol=1e-12)
