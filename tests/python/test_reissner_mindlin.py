import numpy as np
import pyck as ck
import pytest


def test_navier_plate_bisinusoidal_load_rm3p():
    """Test: Reissner-Mindlin-3p Navier plate under bi-sinusoidal load."""
    E = 1.0e7
    nu = 0.3
    h = 0.1     # Thicker plate to see RM effects
    L = 2.0
    W = 1.0
    f0 = -10.0
    p = 3
    n = 12

    # Define geometry
    basis_u = ck.create_clamped_uniform(p, n)
    basis_v = ck.create_clamped_uniform(p, 2*n)
    surface = ck.create_rectangle(basis_u, basis_v, L, W)
    gauss = ck.create_gauss_from_patch(surface)

    material = ck.PlaneStress2d(E, nu, h)
    Kb = material.bending_stiffness()
    Ks = material.shear_stiffness()

    def expected_displ(x, y):
        """Exact RM3P solution."""
        term = (np.pi/L)**2 + (np.pi/W)**2
        w0 = f0 * (1.0 / (Kb * term**2) + 1.0 / (Ks * term))
        return w0 * np.sin(np.pi * x / L) * np.sin(np.pi * y / W)

    def expected_strain():
        """Exact RM3P strain energy."""
        term = (np.pi/L)**2 + (np.pi/W)**2
        return (f0**2 * L * W / 8.0) * (1.0 / (Kb * term**2) + 1.0 / (Ks * term))

    def load_func(phys_pts):
        x, y = phys_pts[:, 0], phys_pts[:, 1]
        return f0 * np.sin(np.pi * x / L) * np.sin(np.pi * y / W)

    load_cond = ck.create_function_load_condition(surface, load_func, gauss)

    element = ck.PlateReissnerMindlin3p(material)
    problem = ck.LinearElasticProblem([surface], element, gauss)
    problem.add_condition(load_cond)

    # BCs: Simply Supported (w=0, theta_tangent=0)
    for dim in (0, 1):
        for start in (True, False):
            idx = surface.layer_dofs(param_dim=dim, at_start=start, layer_idx=0)
            # w = 0 (offset 0)
            problem.add_constraint(ck.DirectConstraint([i * 3 for i in idx], 0.0))
            # theta_tangent = 0 (dim 0 -> offset 2, dim 1 -> offset 1)
            offset = 2 if dim == 0 else 1
            problem.add_constraint(ck.DirectConstraint([i * 3 + offset for i in idx], 0.0))

    K, F = problem.assemble()
    u = ck.solve(K, F)

    # 1. Strain Energy Verification
    U_num = 0.5 * np.dot(u, K @ u)
    U_exact = expected_strain()
    rel_err_energy = abs(U_num - U_exact) / abs(U_exact)
    print(f"\n  Strain Energy:    Numerical={U_num:.6e}, Exact={U_exact:.6e}, Error={rel_err_energy:.2%}")

    # 2. Maximum Displacement Verification (at center)
    max_w_exact = expected_displ(L/2, W/2)
    # w is at indices 0, 3, 6, ...
    u_w = u[0::3]
    max_w_num = np.max(np.abs(u_w))
    rel_err_w = abs(max_w_num - abs(max_w_exact)) / abs(max_w_exact)
    print(f"  Max Displacement: Numerical={max_w_num:.6e}, Exact={max_w_exact:.6e}, Error={rel_err_w:.2%}")

    assert rel_err_energy < 1e-2, f"Strain energy error too high: {rel_err_energy:.2%}"
    assert rel_err_w < 5e-2, f"Max displacement error too high: {rel_err_w:.2%}"

    # 3. Multi-point Displacement Verification
    print("\nVerifying displacement at internal points:")
    n_pts = 3
    u_eval = np.linspace(0.2, 0.8, n_pts)
    v_eval = np.linspace(0.2, 0.8, n_pts)
    for ui in u_eval:
        for vi in v_eval:
            x, y = ui * L, vi * W
            params = np.array([[ui, vi]])
            N = ck.eval_shape_at(surface, params)[0]
            w_num = (N @ u_w)[0]
            w_exact = expected_displ(x, y)
            error = abs(w_num - w_exact) / abs(w_exact)
            print(f"  Point ({x:.2f}, {y:.2f}): Numerical={w_num:.6e}, Exact={w_exact:.6e}, Error={error:.2%}")
            assert error < 5e-2, f"Displacement error at ({x}, {y}) too high: {error:.2%}"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
