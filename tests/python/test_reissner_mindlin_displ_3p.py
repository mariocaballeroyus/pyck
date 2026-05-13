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
    n_phys = surface.num_control_pts * element._cpp_object.num_node_dofs()
    u = np.asarray(u_full)[:n_phys]

    return {
        "surface": surface,
        "element": element,
        "K": K,
        "u": u,
    }


def test_rm_displ_3p_kinematic_consistency():
    """Split-displacement RMD3 satisfies the recovered RM kinematics exactly."""
    basis = ck.create_clamped_uniform(3, 6)
    surface = ck.create_rectangle(basis, basis, 1.0, 1.0)
    material = ck.PlaneStress2d(1.0e7, 0.3, 0.1)
    element = ck.PlateReissnerMindlinDispl3p(material)

    pts = np.array([
        [0.2, 0.3],
        [0.5, 0.4],
        [0.8, 0.7],
    ])
    shape = ck.eval_shape_at(surface, pts, order=2)
    Nw = np.asarray(element._cpp_object.displacement_shape_matrix(shape))
    Nphi = np.asarray(element._cpp_object.rotation_shape_matrix(shape))
    B = np.asarray(element._cpp_object.strain_matrix(shape))

    q_count = pts.shape[0]
    n_basis = shape[0].shape[1]
    dw_dx = np.zeros((q_count, 3 * n_basis))
    dw_dy = np.zeros((q_count, 3 * n_basis))

    for i in range(n_basis):
        dw_dx[:, 3 * i] = shape[1][:, i]
        dw_dx[:, 3 * i + 1] = shape[1][:, i]
        dw_dx[:, 3 * i + 2] = shape[1][:, i]
        dw_dy[:, 3 * i] = shape[2][:, i]
        dw_dy[:, 3 * i + 1] = shape[2][:, i]
        dw_dy[:, 3 * i + 2] = shape[2][:, i]

    gamma_x_from_kin = dw_dx + Nphi[0::2, :]
    gamma_y_from_kin = dw_dy + Nphi[1::2, :]
    gamma_x_from_b = B[3::5, :]
    gamma_y_from_b = B[4::5, :]

    assert np.allclose(gamma_x_from_kin, gamma_x_from_b)
    assert np.allclose(gamma_y_from_kin, gamma_y_from_b)

    u = np.arange(1, 3 * n_basis + 1, dtype=float)
    w_total = Nw @ u
    phi = (Nphi @ u).reshape(-1, 2)
    gamma = np.column_stack([gamma_x_from_b @ u, gamma_y_from_b @ u])
    kappa = np.column_stack([B[0::5, :] @ u, B[1::5, :] @ u, B[2::5, :] @ u])

    dphi_x_dx = np.zeros((q_count, 3 * n_basis))
    dphi_y_dy = np.zeros((q_count, 3 * n_basis))
    dphi_x_dy = np.zeros((q_count, 3 * n_basis))
    dphi_y_dx = np.zeros((q_count, 3 * n_basis))

    for i in range(n_basis):
        dphi_x_dx[:, 3 * i] = -shape[3][:, i]
        dphi_x_dx[:, 3 * i + 2] = -shape[3][:, i]

        dphi_y_dy[:, 3 * i] = -shape[5][:, i]
        dphi_y_dy[:, 3 * i + 1] = -shape[5][:, i]

        dphi_x_dy[:, 3 * i] = -shape[4][:, i]
        dphi_x_dy[:, 3 * i + 2] = -shape[4][:, i]

        dphi_y_dx[:, 3 * i] = -shape[4][:, i]
        dphi_y_dx[:, 3 * i + 1] = -shape[4][:, i]

    assert np.allclose(shape[1] @ (u[0::3] + u[1::3] + u[2::3]) + phi[:, 0], gamma[:, 0])
    assert np.allclose(shape[2] @ (u[0::3] + u[1::3] + u[2::3]) + phi[:, 1], gamma[:, 1])
    assert np.allclose(dphi_x_dx @ u, kappa[:, 0])
    assert np.allclose(dphi_y_dy @ u, kappa[:, 1])
    assert np.allclose((dphi_x_dy + dphi_y_dx) @ u, kappa[:, 2])
    assert np.all(np.isfinite(w_total))


def test_rm_displ_3p_smoke_penalty_and_lagrange():
    """Both boundary-enforcement paths assemble and solve with finite results."""
    cases = [
        _solve_rm_displ_3p_case(2.0, 1.0, 0.1, -10.0, 3, 8, "lagrange"),
        _solve_rm_displ_3p_case(1.0, 1.0, 0.05, -10.0, 3, 8, "penalty"),
    ]

    for result in cases:
        surface = result["surface"]
        element = result["element"]
        u = result["u"]
        assert np.all(np.isfinite(u))

        grid = np.linspace(0.0, 1.0, 21)
        uu, vv = np.meshgrid(grid, grid, indexing="xy")
        pts_param = np.column_stack([uu.ravel(), vv.ravel()])
        Nw = element._cpp_object.displacement_shape_matrix(
            ck.eval_shape_at(surface, pts_param, order=2)
        )
        w_vals = np.asarray(Nw @ u).ravel()
        assert np.all(np.isfinite(w_vals))
        assert np.max(np.abs(w_vals)) > 0.0
