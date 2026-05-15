import numpy as np
import pyck as ck


def _solve_rm_displ_3p_case(L, W, h, f0, p, n, bc_method):
    basis_u = ck.create_clamped_uniform(p, n)
    basis_v = ck.create_clamped_uniform(p, 2 * n)
    surface = ck.create_rectangle(basis_u, basis_v, L, W)
    gauss2d = ck.create_gauss_from_patch(surface)
    gauss1d = ck.GaussLegendre(p + 1, dim=1)

    material = ck.PlaneStress2d(1.0e7, 0.3, h)

    def load_func(phys_pts):
        x, y = phys_pts[:, 0], phys_pts[:, 1]
        return f0 * np.sin(np.pi * x / L) * np.sin(np.pi * y / W)

    element = ck.PlateReissnerMindlinDispl3p(material)
    problem = ck.LinearElasticProblem([surface], element, gauss2d)
    problem.add_condition(ck.conditions.create_load_condition(surface, load_func, gauss2d))

    for side in ("u0", "u1", "v0", "v1"):
        boundary = surface.boundary(side)
        if bc_method == "lagrange":
            problem.add_condition(
                ck.create_simply_supported_lagrange(boundary, gauss1d)
            )
        elif bc_method == "penalty":
            cond = ck.PenaltyBoundaryCondition(boundary, gauss1d)
            cond.add("w", 1.0e7, 0.0)
            cond.add("rot_s", 1.0e7, 0.0)
            problem.add_condition(cond)
        else:
            raise ValueError(f"Unknown bc_method: {bc_method}")

    K, F = problem.assemble()
    u_full = ck.solve(K, F)
    n_phys = surface.num_control_pts * element.num_node_dofs
    u = np.asarray(u_full)[:n_phys]

    return {
        "surface": surface,
        "element": element,
        "K": K,
        "u": u,
    }


def test_rm_displ_3p_smoke_penalty_and_lagrange():
    """Both boundary-enforcement paths assemble and solve with finite results."""
    cases = [
        _solve_rm_displ_3p_case(2.0, 1.0, 0.1, -10.0, 3, 8, "lagrange"),
        _solve_rm_displ_3p_case(1.0, 1.0, 0.05, -10.0, 3, 8, "penalty"),
    ]

    for result in cases:
        u = result["u"]
        assert np.all(np.isfinite(u))
        assert np.max(np.abs(u)) > 0.0
