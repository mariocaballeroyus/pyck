import numpy as np
import pyck as ck


def _dense(K):
    return np.asarray(K.todense()) if hasattr(K, "todense") else np.asarray(K)


def _assemble_stiffness(surface, element):
    problem = ck.LinearElasticProblem([surface], element, ck.GaussLegendre.from_patch(surface))
    K, _f = problem.assemble()
    return _dense(K)


def _hypar_patch(deg=3, n=6, a=1.5, b=-1.0, c=2.0, L=2.0):
    """Doubly-curved patch z = a x^2 + b y^2 + c x y over [0,L]^2. The cross
    term makes the parametrisation non-principal, so g_12 != 0 and B_12 != 0
    and the mixed shape operator is strongly non-symmetric (B^1_2 != B^2_1)."""
    flat = ck.SurfacePatch.rectangle(L, L, nu=n, nv=n, deg=deg, name="flat")
    bu, bv = flat.basis
    cps = np.asarray(flat.control_points).copy()
    x, y = cps[:, 0], cps[:, 1]
    cps[:, 2] = a * x * x + b * y * y + c * x * y
    return ck.SurfacePatch(bu, bv, cps, name="hypar")


def _frame_and_shape_operator(patch, uv, h=1e-4):
    """Surface frame, metric and mixed shape operator B^mu_beta at one
    parametric point via finite differences of the patch geometry."""
    def x(du, dv):
        return ck.eval_geometry_at(patch, np.array([[uv[0] + du, uv[1] + dv]]))[0]
    A1 = (x(h, 0) - x(-h, 0)) / (2 * h)
    A2 = (x(0, h) - x(0, -h)) / (2 * h)
    A11 = (x(h, 0) - 2 * x(0, 0) + x(-h, 0)) / h**2
    A22 = (x(0, h) - 2 * x(0, 0) + x(0, -h)) / h**2
    A12 = (x(h, h) - x(h, -h) - x(-h, h) + x(-h, -h)) / (4 * h**2)
    A3 = np.cross(A1, A2); A3 /= np.linalg.norm(A3)
    g = np.array([[A1 @ A1, A1 @ A2], [A1 @ A2, A2 @ A2]])
    Bcov = np.array([[A11 @ A3, A12 @ A3], [A12 @ A3, A22 @ A3]])
    Bmix = np.linalg.inv(g) @ Bcov          # Bmix[mu, beta] = B^mu_beta
    return g, Bcov, Bmix


def test_4p_shell_shear_has_no_membrane_coupling():
    """The transverse shear must NOT couple to the in-plane displacement DOFs
    (u_1, u_2). The full Reissner-Mindlin shear carries a membrane-curvature term
    B^g_a u_g, which the rotation-free kinematics cannot relax: retaining it in the
    shear locks it and over-stiffens membrane-bending problems (Scordelis-Lo) by
    ~4x, converging in both h and p to the wrong answer. The deformed-director
    referencing cancels it from the shear by construction (relocating it, with its
    grad-B partner, into bending). This regression guards against it reappearing in
    the shear: on a strongly non-principal doubly-curved patch (B^1_2 != B^2_1, so
    the bug-prone index would be plainly visible) the shear rows gamma_1, gamma_2
    must have zero u_1, u_2 columns."""
    el = ck.ShellReissnerMindlin4p(ck.PlaneStress2d(1.0e7, 0.3, 0.05))
    patch = _hypar_patch()
    uv = (0.37, 0.58)                       # generic interior, off-diagonal point
    params = np.array([uv])

    g, Bcov, Bmix = _frame_and_shape_operator(patch, uv)
    # The patch genuinely exercises curvature coupling: non-diagonal metric,
    # non-diagonal curvature, and a non-symmetric mixed shape operator. Were the
    # B^g_a u_g term present, these would make its (large) entries unmistakable.
    assert abs(g[0, 1]) > 1e-3 * abs(g[0, 0])
    assert abs(Bcov[0, 1]) > 1e-3 * abs(Bcov[0, 0])
    assert abs(Bmix[0, 1] - Bmix[1, 0]) > 1e-3 * abs(Bmix[0, 0])

    B = el.strain_matrix(patch, params)     # (8, 4*ncp), single qp
    shear_u = np.concatenate([B[6, 0::4], B[6, 1::4],     # gamma_1 vs u_1, u_2
                              B[7, 0::4], B[7, 1::4]])     # gamma_2 vs u_1, u_2
    # Reference scale: the curvature coupling, if present, would be O(B^g_a * N_i),
    # comparable to the (retained) bending-shear entries in the w_b column.
    scale = np.abs(B[6:8, 2::4]).max()
    assert scale > 0
    assert np.abs(shear_u).max() / scale < 1e-12


def test_4p_shell_reduces_to_2p_plate_on_flat_patch():
    """On a flat square patch the 4-parameter shell must reduce to the
    two-parameter plate: the (w_b, psi) sub-block of its stiffness equals the
    plate stiffness exactly (the bending block negates and cancels in B^T D B),
    and the in-plane membrane DOFs decouple from (w_b, psi) since B_{ab} = 0."""
    E, nu, t = 1.0e7, 0.3, 0.02
    deg, n = 3, 6
    mat = ck.PlaneStress2d(E, nu, t)

    # Square -> orthonormal parametric frame, where the 4p shell's curvilinear
    # twist permutation coincides with the plate's.
    surface = ck.SurfacePatch.rectangle(1.0, 1.0, nu=n, nv=n, deg=deg, name="flat")
    ncp = surface.num_control_pts

    K4 = _assemble_stiffness(surface, ck.ShellReissnerMindlin4p(mat))
    K2 = _assemble_stiffness(surface, ck.PlateReissnerMindlinDispl2p(mat))

    # (w_b, psi) live in slots 2,3 of the 4p element and 0,1 of the 2p element.
    idx_wp = np.array([[4 * i + 2, 4 * i + 3] for i in range(ncp)]).ravel()
    idx_u = np.array([[4 * i + 0, 4 * i + 1] for i in range(ncp)]).ravel()

    scale = np.abs(K2).max()
    assert np.abs(K4[np.ix_(idx_wp, idx_wp)] - K2).max() / scale < 1e-10
    # Membrane DOFs decouple from (w_b, psi) on a flat patch.
    assert np.abs(K4[np.ix_(idx_u, idx_wp)]).max() / scale < 1e-10


def test_4p_shell_pressurized_cylinder_membrane():
    """Thin cylinder under uniform outward pressure with pure radial motion
    (circumferential + axial displacement and twist suppressed): a plane-strain
    ring whose closed-form radial displacement is w = p R^2 (1-nu^2) / (E t).
    Exercises the shell-only membrane curvature coupling -B_{ab} (zero on a
    flat patch)."""
    R, Lz, t, E, nu, p = 1.0, 0.5, 0.02, 1.0e7, 0.3, 1.0
    ND, deg, ncp = 4, 4, 18
    w_exact = p * R**2 * (1.0 - nu**2) / (E * t)

    patch = ck.SurfacePatch.quarter_cylinder(
        nu=ncp, nv=ncp, deg=deg, radius=R, height=Lz, angle=90.0, name="cyl")
    el = ck.ShellReissnerMindlin4p(ck.PlaneStress2d(E, nu, t))
    prob = ck.LinearElasticProblem([patch], el, ck.GaussLegendre.from_patch(patch))
    N = patch.num_control_pts

    # Pure radial expansion: only the bending potential w_b is free.
    dofs = []
    for cp in range(N):
        dofs += [cp * ND + 0, cp * ND + 1, cp * ND + 3]  # u1, u2, psi = 0
    prob.add_constraint(ck.DirectConstraint(sorted(set(dofs)), value=0.0))

    # Outward pressure p along the radial (normal) direction (x, y, 0)/R.
    def radial_pressure(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts)
        out = np.zeros((pts.shape[0], 3))
        r = np.hypot(pts[:, 0], pts[:, 1])
        out[:, 0] = p * pts[:, 0] / r
        out[:, 1] = p * pts[:, 1] / r
        return out
    prob.add_domain_load(radial_pressure)

    K, f = prob.assemble()
    u = ck.solve(K, f, physical_dofs=ND * N)

    pts = np.array([[a, b] for a in np.linspace(0.3, 0.7, 5)
                           for b in np.linspace(0.3, 0.7, 5)])  # interior
    disp = np.asarray(ck.Function(u, el, patch, ck.FieldType.DISPLACEMENT)(pts)).reshape(-1, 3)
    # Radial component: project the in-plane displacement onto the outward normal.
    geo = np.asarray(ck.eval_geometry_at(patch, pts))[:, :2]
    normal = geo / np.linalg.norm(geo, axis=1, keepdims=True)
    w = np.einsum("ij,ij->i", disp[:, :2], normal)

    assert abs(w.mean() - w_exact) / w_exact < 2e-3
    assert (w.max() - w.min()) / abs(w.mean()) < 2e-3   # uniform field
