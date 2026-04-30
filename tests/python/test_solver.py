import numpy as np
import pyck as ck


class DummyProblem:
    num_physical_dofs = 2

    def assemble(self):
        K = np.eye(4)
        f = np.array([1.0, 2.0, 10.0, 20.0])
        return K, f


def test_solve_problem_returns_physical_dofs_only():
    u = ck.solve(DummyProblem())

    np.testing.assert_allclose(u, [1.0, 2.0])


def test_solve_matrix_returns_full_solution_by_default():
    K = np.eye(4)
    f = np.array([1.0, 2.0, 10.0, 20.0])

    u = ck.solve(K, f)

    np.testing.assert_allclose(u, f)


def test_solve_matrix_can_trim_to_physical_dofs():
    K = np.eye(4)
    f = np.array([1.0, 2.0, 10.0, 20.0])

    u = ck.solve(K, f, physical_dofs=2)

    np.testing.assert_allclose(u, [1.0, 2.0])
