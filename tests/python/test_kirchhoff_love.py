import numpy as np
import pyck as ck
import pytest


def test_navier_plate_bisinusoidal_load():
    """Test 1: Navier plate under bi-sinusoidal load."""
    E = 1.0e7   # Young's modulus
    nu = 0.3    # Poisson's ratio
    h = 0.01    # thickness
    L = 2.0     # length
    W = 1.0     # width
    f0 = -10.0  # peak intensity

    # Discretization parameters
    p = 3
    n = 10    

    # Define geometry 
    basis_u = ck.create_clamped_uniform(p, n)
    basis_v = ck.create_clamped_uniform(p, 2*n)
    surface = ck.create_rectangle(basis_u, basis_v, L, W)

    gauss = ck.create_gauss_from_patch(surface)

    def load_func(phys_pts):
        """Bi-sinusoidal load.

        f = f0 * sin(pi*x/L) * sin(pi*y/W)
        """
        x = phys_pts[:, 0]
        y = phys_pts[:, 1]
        return f0 * np.sin(np.pi * x / L) * np.sin(np.pi * y / W)

    load_cond = ck.conditions.create_load_condition(surface, load_func, gauss)

    def expected_displ(x, y):
        """Exact solution for Navier plate under bi-sinusoidal load."""
        Kb = E * h**3 / (12 * (1 - nu**2))
        term = (np.pi/L)**2 + (np.pi/W)**2
        w0 = f0 / (Kb * term**2)
        return w0 * np.sin(np.pi * x / L) * np.sin(np.pi * y / W)

    def expected_strain():
        """Exact strain for Navier plate under bi-sinusoidal load."""
        Kb = E * h**3 / (12 * (1 - nu**2))
        term = (np.pi/L)**2 + (np.pi/W)**2
        return (1/8) * (f0**2 * L * W) / (Kb * term**2)

    # Define problem and assemble
    material = ck.PlaneStress2d(E, nu, h)
    element = ck.PlateKirchhoffLove1p(material)
    problem = ck.LinearElasticProblem([surface], element, gauss)
    problem.add_condition(load_cond)

    # Apply simply supported boundary conditions (w = 0) on all edges
    for dim in (0, 1):
        for start in (True, False):
            dofs = surface.layer_dofs(param_dim=dim, at_start=start, layer_idx=0)
            bc = ck.DirectConstraint(dofs, 0.0)
            problem.add_constraint(bc)

    K, F = problem.assemble()
    
    # Solve system
    u = ck.solve(K, F)

    # 1. Strain Energy Verification
    U_num = 0.5 * np.dot(u, K @ u)
    U_exact = expected_strain()
    rel_err_energy = abs(U_num - U_exact) / abs(U_exact)
    
    print(f"  Strain Energy:    Numerical={U_num:.6e}, Exact={U_exact:.6e}, Error={rel_err_energy:.2%}")
    
    # 2. Maximum Displacement Verification
    max_w_exact = expected_displ(L/2, W/2)
    max_w_num = np.max(np.abs(u))
    rel_err_w = abs(max_w_num - abs(max_w_exact)) / abs(max_w_exact)

    print(f"  Max Displacement: Numerical={max_w_num:.6e}, Exact={max_w_exact:.6e}, Error={rel_err_w:.2%}")

    assert rel_err_energy < 1e-2, f"Strain energy error too high: {rel_err_energy:.2e}"
    assert rel_err_w < 5e-2, f"Max displacement error too high: {rel_err_w:.2e}"

    # 3. Multi-point Displacement Verification
    print("\nVerifying displacement at internal points:")
    n_pts = 3
    u_eval = np.linspace(0.2, 0.8, n_pts)
    v_eval = np.linspace(0.2, 0.8, n_pts)
    
    for ui in u_eval:
        for vi in v_eval:
            # Parametric to physical
            x, y = ui * L, vi * W
            
            # Numerical evaluation
            params = np.array([[ui, vi]])
            N = ck.eval_shape_at(surface, params)[0]
            w_num = (N @ u)[0]
            
            # Analytical evaluation
            w_exact = expected_displ(x, y)
            
            error = abs(w_num - w_exact) / abs(w_exact)
            print(f"  Point ({x:.2f}, {y:.2f}): Numerical={w_num:.6e}, Exact={w_exact:.6e}, Error={error:.2%}")
            
            assert error < 5e-2, f"Displacement error at ({x}, {y}) too high: {error:.2%}"


if __name__ == "__main__":
    pytest.main([__file__])
