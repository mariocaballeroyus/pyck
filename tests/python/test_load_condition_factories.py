import numpy as np
import pyck as ck


def test_create_load_condition_is_available_from_conditions_namespace():
    knots = ck.create_clamped_uniform_knots(2, 3)
    basis = ck.BSpline(2, knots)
    patch = ck.create_line_segment(basis, 1.0)
    quadrature = ck.GaussLegendre(3)

    condition = ck.conditions.create_load_condition(
        patch=patch,
        load_fn=lambda pts: np.ones(pts.shape[0]),
        quadrature=quadrature,
    )

    assert isinstance(condition, ck.LoadCondition)


def test_create_load_condition_is_not_exported_at_top_level():
    assert not hasattr(ck, "create_load_condition")
    assert not hasattr(ck, "create_function_load_condition")


def test_create_shear_load_assembles_to_expected_F_norm():
    basis = ck.create_clamped_uniform(2, 5)
    surface = ck.create_rectangle(basis, basis, 1.0, 1.0)
    gauss2d = ck.create_gauss_from_patch(surface)
    gauss1d = ck.GaussLegendre(3, dim=1)
    material = ck.PlaneStress2d(1.0e7, 0.3, 0.1)
    element = ck.PlateReissnerMindlin3p(material)

    boundary = surface.boundary("u1")  # right edge, length = W = 1
    Q = 5.0
    cond = ck.conditions.create_shear_load(boundary, Q, gauss1d)
    assert isinstance(cond, ck.LoadBoundaryCondition)

    problem = ck.create_linear_elastic_problem([surface], element, gauss2d)
    problem.add_condition(cond)
    K, f = problem.assemble()

    n_node_dofs = element._cpp_object.num_node_dofs()
    f_w = f[0::n_node_dofs]
    # Total integrated transverse load = Q * edge_length.
    assert np.isclose(f_w.sum(), Q * 1.0, atol=1e-12)
    # Rotation slots (1, 2) should remain at zero.
    f_rot_x = f[1::n_node_dofs]
    f_rot_y = f[2::n_node_dofs]
    assert np.allclose(f_rot_x, 0.0, atol=1e-12)
    assert np.allclose(f_rot_y, 0.0, atol=1e-12)


def test_load_boundary_condition_chained_add_returns_self():
    basis = ck.create_clamped_uniform(2, 4)
    surface = ck.create_rectangle(basis, basis, 1.0, 1.0)
    boundary = surface.boundary("v0")
    cond = ck.LoadBoundaryCondition(boundary).add("w", 1.0).add("rot_n", 2.0)
    assert isinstance(cond, ck.LoadBoundaryCondition)
