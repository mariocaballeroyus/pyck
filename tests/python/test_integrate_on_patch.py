"""Tests for ck.integrate_on_patch."""

from __future__ import annotations

import numpy as np
import pyck as ck


# --------------------------------------------------------------------------- #
# 1. Area sanity check                                                        #
# --------------------------------------------------------------------------- #
def test_integrate_on_patch_recovers_rectangle_area():
    L, W = 2.0, 1.5
    p = 2
    n = 5
    bu = ck.create_clamped_uniform(p, n)
    bv = ck.create_clamped_uniform(p, n)
    patch = ck.create_rectangle(bu, bv, L, W)
    gauss = ck.GaussLegendre(p + 1, dim=2)

    def integrand(phys_pts, shape_derivs, u_local):
        assert u_local is None
        return np.ones(phys_pts.shape[0])

    result = ck.integrate_on_patch(patch, gauss, integrand, order=0)
    assert result.shape == (1,)
    assert abs(result[0] - L * W) < 1e-12


# --------------------------------------------------------------------------- #
# 2. L^2 norm of the bi-sinusoidal exact field                                #
# --------------------------------------------------------------------------- #
def test_integrate_on_patch_l2_norm_of_bisinusoidal_field():
    L, W = 2.0, 1.5
    p = 3
    n = 8
    bu = ck.create_clamped_uniform(p, n)
    bv = ck.create_clamped_uniform(p, n)
    patch = ck.create_rectangle(bu, bv, L, W)
    gauss = ck.GaussLegendre(p + 2, dim=2)

    def integrand(phys_pts, shape_derivs, u_local):
        x, y = phys_pts[:, 0], phys_pts[:, 1]
        s = np.sin(np.pi * x / L) * np.sin(np.pi * y / W)
        return s * s

    result = ck.integrate_on_patch(patch, gauss, integrand, order=0)
    expected = L * W / 4.0
    assert abs(result[0] - expected) < 1e-10


# --------------------------------------------------------------------------- #
# 3. Four-field Navier plate L^2 error vs chunked-Python reference            #
# --------------------------------------------------------------------------- #
def _solve_navier_plate():
    L, W = 1.0, 1.0
    h = 0.001
    E = 1.0e7
    nu = 0.3
    kappa = 5.0 / 6.0
    f0 = -1.0
    p = 4
    n_elem = 8
    n = n_elem + p

    Kb = E * h**3 / (12.0 * (1.0 - nu**2))
    Ks = kappa * (E / (2.0 * (1.0 + nu))) * h
    a, b = np.pi / L, np.pi / W
    alpha2 = a**2 + b**2
    w_amp = f0 / (Kb * alpha2**2) + f0 / (Ks * alpha2)
    phi_x_amp = -a * f0 / (Kb * alpha2**2)
    phi_y_amp = -b * f0 / (Kb * alpha2**2)
    Mxx_amp = (a**2 + nu * b**2) * f0 / alpha2**2
    Myy_amp = (b**2 + nu * a**2) * f0 / alpha2**2
    Mxy_amp = -(1.0 - nu) * a * b * f0 / alpha2**2
    Qx_amp = a * f0 / alpha2
    Qy_amp = b * f0 / alpha2

    mat = ck.PlaneStress2d(E, nu, h, kappa)
    element = ck.PlateReissnerMindlinDispl2p(mat)
    bu = ck.create_clamped_uniform(p, n)
    bv = ck.create_clamped_uniform(p, n)
    patch = ck.create_rectangle(bu, bv, L, W)
    gauss2d = ck.GaussLegendre(8, dim=2)
    gauss1d = ck.GaussLegendre(8, dim=1)

    def load_fn(pts):
        x, y = pts[:, 0], pts[:, 1]
        return f0 * np.sin(np.pi * x / L) * np.sin(np.pi * y / W)

    prob = ck.create_linear_elastic_problem([patch], element, gauss2d)
    prob.add_condition(ck.conditions.create_load_condition(patch, load_fn, gauss2d))
    for side in ("u0", "u1", "v0", "v1"):
        prob.add_condition(ck.create_simply_supported_lagrange(patch.boundary(side), gauss1d))
    prob.add_condition(ck.create_zero_mean_lagrange(patch, 1, gauss2d))

    K, f = prob.assemble()
    u_full = ck.solve(K, f)
    n_phys = patch.num_control_pts * element._cpp_object.num_node_dofs()
    u_phys = np.asarray(u_full)[:n_phys]

    amps = (w_amp, phi_x_amp, phi_y_amp,
            Mxx_amp, Myy_amp, Mxy_amp, Qx_amp, Qy_amp)
    geom = (L, W, Kb, Ks, nu)
    return patch, element, u_phys, gauss2d, amps, geom


def _make_navier_integrand(element, amps):
    L = 1.0
    W = 1.0
    w_amp, phi_x_amp, phi_y_amp, Mxx_amp, Myy_amp, Mxy_amp, Qx_amp, Qy_amp = amps
    cpp = element._cpp_object

    def integrand(phys_pts, shape_derivs, u_local):
        Nw = np.asarray(cpp.displacement_shape_matrix(shape_derivs))
        Nphi = np.asarray(cpp.rotation_shape_matrix(shape_derivs))
        Nm = np.asarray(cpp.moment_matrix(shape_derivs))
        Nq = np.asarray(cpp.transverse_shear_matrix(shape_derivs))

        w_h = (Nw @ u_local).ravel()
        phi_h = (Nphi @ u_local).reshape(-1, 2)
        M_h = (Nm @ u_local).reshape(-1, 3)
        Q_h = (Nq @ u_local).reshape(-1, 2)

        x, y = phys_pts[:, 0], phys_pts[:, 1]
        sx, sy = np.sin(np.pi * x / L), np.sin(np.pi * y / W)
        cx, cy = np.cos(np.pi * x / L), np.cos(np.pi * y / W)
        w_ex = w_amp * sx * sy
        phi_x_ex = phi_x_amp * cx * sy
        phi_y_ex = phi_y_amp * sx * cy
        Mxx_ex = Mxx_amp * sx * sy
        Myy_ex = Myy_amp * sx * sy
        Mxy_ex = Mxy_amp * cx * cy
        Qx_ex = Qx_amp * cx * sy
        Qy_ex = Qy_amp * sx * cy

        out = np.empty((phys_pts.shape[0], 4))
        out[:, 0] = (w_h - w_ex) ** 2
        out[:, 1] = (phi_h[:, 0] - phi_x_ex) ** 2 + (phi_h[:, 1] - phi_y_ex) ** 2
        out[:, 2] = ((M_h[:, 0] - Mxx_ex) ** 2
                     + (M_h[:, 1] - Myy_ex) ** 2
                     + 2 * (M_h[:, 2] - Mxy_ex) ** 2)
        out[:, 3] = (Q_h[:, 0] - Qx_ex) ** 2 + (Q_h[:, 1] - Qy_ex) ** 2
        return out
    return integrand


def _python_reference_squared_errors(patch, element, u_phys, gauss, geom, amps):
    L, W, Kb, Ks, nu = geom
    w_amp, phi_x_amp, phi_y_amp, Mxx_amp, Myy_amp, Mxy_amp, Qx_amp, Qy_amp = amps

    Db = Kb * np.array([[1.0, nu, 0.0],
                        [nu, 1.0, 0.0],
                        [0.0, 0.0, (1.0 - nu) / 2.0]])
    Ds = Ks * np.eye(2)

    ref_pts = np.asarray(gauss.points)
    ref_wts = np.asarray(gauss.weights)
    n_u = patch.basis(0).num_intervals
    n_v = patch.basis(1).num_intervals
    kv_u = patch.basis(0).knot_vector
    kv_v = patch.basis(1).knot_vector
    cpp = element._cpp_object
    u = np.asarray(u_phys)

    spans = []
    for j in range(n_v):
        v_a, v_b = kv_v.span_bounds(j)
        for i in range(n_u):
            u_a, u_b = kv_u.span_bounds(i)
            spans.append((u_a, u_b, v_a, v_b))

    err = np.zeros(4)
    chunk = 32
    for start in range(0, len(spans), chunk):
        block = spans[start:start + chunk]
        pts_blocks, wts_blocks = [], []
        for u_a, u_b, v_a, v_b in block:
            half_du = 0.5 * (u_b - u_a)
            half_dv = 0.5 * (v_b - v_a)
            pts_blocks.append(np.column_stack([
                u_a + half_du * (ref_pts[:, 0] + 1.0),
                v_a + half_dv * (ref_pts[:, 1] + 1.0),
            ]))
            wts_blocks.append(ref_wts * half_du * half_dv)
        pts_param = np.vstack(pts_blocks)
        wts = np.concatenate(wts_blocks) * (L * W)

        shape = ck.eval_shape_at(patch, pts_param, order=3)
        Nw = np.asarray(cpp.displacement_shape_matrix(shape))
        Nphi = np.asarray(cpp.rotation_shape_matrix(shape))
        B = np.asarray(cpp.strain_displacement_matrix(shape))
        w_num = (Nw @ u).ravel()
        phi_num = (Nphi @ u).reshape(-1, 2)
        strain = (B @ u).reshape(-1, 5)
        M_num = strain[:, :3] @ Db.T
        Q_num = strain[:, 3:5] @ Ds.T

        x = L * pts_param[:, 0]
        y = W * pts_param[:, 1]
        sx, sy = np.sin(np.pi * x / L), np.sin(np.pi * y / W)
        cx, cy = np.cos(np.pi * x / L), np.cos(np.pi * y / W)
        w_ex = w_amp * sx * sy
        phi_x_ex = phi_x_amp * cx * sy
        phi_y_ex = phi_y_amp * sx * cy
        Mxx_ex = Mxx_amp * sx * sy
        Myy_ex = Myy_amp * sx * sy
        Mxy_ex = Mxy_amp * cx * cy
        Qx_ex = Qx_amp * cx * sy
        Qy_ex = Qy_amp * sx * cy

        err[0] += np.sum((w_num - w_ex) ** 2 * wts)
        err[1] += np.sum(((phi_num[:, 0] - phi_x_ex) ** 2
                          + (phi_num[:, 1] - phi_y_ex) ** 2) * wts)
        err[2] += np.sum(((M_num[:, 0] - Mxx_ex) ** 2
                          + (M_num[:, 1] - Myy_ex) ** 2
                          + 2 * (M_num[:, 2] - Mxy_ex) ** 2) * wts)
        err[3] += np.sum(((Q_num[:, 0] - Qx_ex) ** 2
                          + (Q_num[:, 1] - Qy_ex) ** 2) * wts)
    return err


def test_integrate_on_patch_navier_plate_matches_python_reference():
    patch, element, u_phys, gauss, amps, geom = _solve_navier_plate()

    integrand = _make_navier_integrand(element, amps)
    new = ck.integrate_on_patch(
        patch, gauss, integrand,
        cp_values=u_phys, ndof=2, order=3,
    )

    ref = _python_reference_squared_errors(patch, element, u_phys, gauss, geom, amps)

    assert new.shape == (4,)
    rtol = 1.0e-10
    for i in range(4):
        assert abs(new[i] - ref[i]) / ref[i] < rtol, (
            f"component {i}: new={new[i]}, ref={ref[i]}, "
            f"rel_diff={abs(new[i] - ref[i]) / ref[i]}"
        )
