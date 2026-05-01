import numpy as np
import pyck as ck


def test_lagrange_multiplier_condition_augments_system():
    basis = ck.create_clamped_uniform(2, 5)
    surface = ck.create_rectangle(basis, basis, 1.0, 1.0)
    gauss2d = ck.create_gauss_from_patch(surface)
    gauss1d = ck.GaussLegendre(3, dim=1)

    material = ck.PlaneStress2d(1.0e7, 0.3, 0.1)
    element = ck.PlateReissnerMindlin3p(material)

    boundary = surface.boundary("u0")
    problem = ck.create_linear_elastic_problem([surface], element, gauss2d)
    cond = ck.LagrangeBoundaryCondition(boundary, gauss1d)
    cond.add("w", 0.0)
    cond.add("rot_n", 0.0)
    problem.add_condition(cond)

    K, f = problem.assemble()

    n_phys = surface.num_control_pts * element._cpp_object.num_node_dofs()
    n_lambda = boundary._cpp_object.num_control_pts() * 2

    assert K.shape == (n_phys + n_lambda, n_phys + n_lambda)
    assert f.shape == (n_phys + n_lambda,)
    assert np.allclose(K, K.T)
