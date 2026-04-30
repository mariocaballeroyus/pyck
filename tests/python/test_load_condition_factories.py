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
