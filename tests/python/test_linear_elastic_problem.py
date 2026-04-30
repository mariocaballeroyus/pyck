import pyck as ck


def test_problem_exposes_patches_read_only():
    knots = ck.create_clamped_uniform_knots(2, 3)
    basis = ck.BSpline(2, knots)
    patch = ck.create_line_segment(basis, 1.0)
    material = ck.material.create_slender_beam(210.0e9, 0.3, 0.02, 1.0e-6)
    element = ck.BeamEulerBernoulli1p(material)
    quadrature = ck.GaussLegendre(3)

    problem = ck.create_linear_elastic_problem(
        patches=[patch], element=element, quadrature=quadrature
    )

    assert problem.patches[0] is patch
    assert problem.patches == (patch,)
    assert problem.patch_names == [patch.name]
    assert problem.num_physical_dofs == (
        patch.num_control_pts * element._cpp_object.num_node_dofs()
    )
